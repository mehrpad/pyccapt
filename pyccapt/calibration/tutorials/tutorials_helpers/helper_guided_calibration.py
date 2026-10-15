"""Optional notebook controls for an explicitly configured guided session."""
from __future__ import annotations

import json
from copy import copy, deepcopy

import ipywidgets as widgets
import numpy as np
from IPython.display import display

from pyccapt.calibration.core.calibration_advanced import auto_detect_reference_peaks
from pyccapt.calibration.core.exceptions import CalibrationInputError


def propose_reference_windows(variables, *, mode="mc", event_mask=None, **peak_options):
    """Propose peak windows, without assigning chemical identities or changing state.

    Review each window and supply its independently known mass before using its
    event_mask in GuidedCalibrationSession. Narrow windows can truncate a peak;
    review tails, overlap, background and acquisition-time coverage as well.
    """
    values = np.asarray(variables.get_calibration_array(mode), dtype=float)
    trusted = np.ones(values.size, dtype=bool) if event_mask is None else np.asarray(event_mask)
    if trusted.dtype != bool or trusted.shape != values.shape:
        raise CalibrationInputError("event_mask must be an aligned boolean array")
    peaks = auto_detect_reference_peaks(values, event_mask=trusted, **peak_options)
    return [dict(peak, event_mask=trusted & (values >= peak["x1"]) & (values < peak["x2"]),
                 identity="unconfirmed", axis_units="Da" if mode == "mc" else "ns") for peak in peaks]


def show_guided_calibration(session, *, receipt_path="guided_calibration_receipt.json", background_fit=False,
                            display_panel=True):
    """Display review steps; no fit or mutation is triggered by opening the panel."""
    output = widgets.Output()
    from pyccapt.calibration.tutorials.tutorials_helpers.guided_jobs import NotebookJob
    job = NotebookJob()
    cancel = widgets.Button(description="Cancel fitting")
    progress = widgets.FloatProgress(min=0, max=1, description="Fit:")
    active_mode = widgets.Dropdown(options=[("Mass (Da)", "mc"), ("Normalized ToF (ns)", "tof")],
                                   value=session.mode, description="Apply mode:")
    path = widgets.Text(value=str(receipt_path), description="Receipt:", layout=widgets.Layout(width="650px"))
    operations = {
        "Preview": session.preview, "Fit training": session.fit,
        "Validate selection": session.validate_selection, "Confirm once": session.confirm,
        "Accept model": session.accept,
        "Apply accepted": lambda: dict(applied_events=len(session.apply(active_mode=active_mode.value))),
        "Convert ToF to mass": lambda: dict(mass_events=len(session.convert_applied_tof_to_mass())),
        "Save receipt": lambda: str(session.save_receipt(path.value)),
    }
    allowed = {
        "Preview": None, "Fit training": {"preview", "fitted", "selection_failed", "selection_passed"},
        "Validate selection": {"fitted", "selection_failed", "selection_passed"},
        "Confirm once": {"selection_passed"}, "Accept model": {"confirmed"},
        "Apply accepted": {"accepted"}, "Convert ToF to mass": {"applied"}, "Save receipt": None,
    }
    buttons = {name: widgets.Button(description=name, layout=widgets.Layout(width="165px")) for name in operations}

    def refresh():
        for name, button in buttons.items():
            button.disabled = (job.running or session.status == "invalidated" or
                               (allowed[name] is not None and session.status not in allowed[name]) or
                               (name == "Convert ToF to mass" and session.mode != "tof"))
        cancel.disabled = not job.running

    def show_result(result):
        with output:
            output.clear_output(wait=True)
            print(json.dumps(result, indent=2, default=str))
            print(f"Session: {session.status}")

    def fit_background():
        # Fit a private model. Cancelling a non-interruptible solver discards its
        # result without consuming confirmation or changing the accepted session.
        working = copy(session)
        working.model, working.diagnostics = deepcopy(session.model), deepcopy(session.diagnostics)
        status_before = session.status
        def operation(cancel_event, report):
            report(0., "Fitting training data; cancellation discards the result when the solver returns")
            result = working.fit()
            report(1., "Fit ready")
            return result
        def accept_result(result):
            session._check_current()
            if session.status != status_before:
                raise CalibrationInputError("Session changed during fitting; repeat the fit")
            for name in ("model", "diagnostics", "status", "_model_hash"):
                setattr(session, name, getattr(working, name))
            show_result(result)
        job.start(operation, on_result=accept_result, on_error=lambda error: show_result(str(error)),
                  on_progress=lambda fraction, message: setattr(progress, "value", fraction), on_finished=refresh)
        refresh()

    def run(name):
        def callback(_):
            if background_fit and name == "Fit training":
                fit_background()
                return
            for button in buttons.values():
                button.disabled = True
            with output:
                output.clear_output(wait=True)
                try:
                    result = operations[name]()
                    print(json.dumps(result, indent=2, default=str))
                except (ValueError, RuntimeError, OSError) as error:
                    print(f"{type(error).__name__}: {error}")
                finally:
                    print(f"Session: {session.status}")
                    refresh()
        return callback

    for name, button in buttons.items():
        button.on_click(run(name))
    cancel.on_click(lambda _: job.cancel())
    refresh()
    notice = widgets.HTML(value=("<b>Guided multi-reference calibration</b><br>"
        "Reference identities, event masks and validation policy are fixed when creating the session. "
        "Selection and confirmation do not fit the model. Confirmation is used once. "
        "Excluded events become NaN in derived output; raw data remains available.<br>"
        "Normalized ToF uses the receipt's reference voltage/flight length and zero t0. "
        "Use <b>Convert ToF to mass</b> here; do not reapply per-event voltage/geometry or the old t0."))
    fit_controls = [widgets.HBox([progress, cancel])] if background_fit else []
    panel = widgets.VBox([notice, active_mode, *fit_controls, widgets.HBox(list(buttons.values())[:4]),
                          widgets.HBox(list(buttons.values())[4:]), path, output])
    panel.controls = dict(buttons, job=job, cancel=cancel)
    if display_panel:
        display(panel)
    return panel
