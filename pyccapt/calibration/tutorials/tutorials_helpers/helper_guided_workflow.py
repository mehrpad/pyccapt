"""A separate end-to-end notebook panel, leaving the legacy notebook unchanged."""
from __future__ import annotations

from html import escape
import json
from pathlib import Path
from types import SimpleNamespace

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from pyccapt.calibration.core.exceptions import CalibrationInputError, CalibrationStateError
from pyccapt.calibration.core.guided_workflow import GuidedWorkflow, STAGES
from pyccapt.calibration.tutorials.tutorials_helpers.guided_jobs import NotebookJob


GUIDED_UI_API_VERSION = 7


def show_guided_workflow(workflow=None, *, display_panel=True):
    """Render controls only. Loading, fitting, acceptance and file writes require clicks."""
    controller = {"workflow": workflow, "calibration_workspace": None, "calibration_session": None,
                  "reconstruction_workspace": None, "calibration_panel": None, "range_panel": None,
                  "crop_drawing": None, "restoring_crop": False,
                  "crop_dirty": not bool(workflow and workflow.accepted.get("crop")), "crop_error": None,
                  "reconstruction_generation": 0, "restoring_reconstruction": False,
                  "physical_generation": 0, "restoring_instrument": False, "standard_workspace": None, "standard_panel": None,
                  "crop_seeded": bool(workflow and (workflow.accepted.get("crop") or workflow.preview.get("crop")))}
    job = NotebookJob()
    status, progress_text = widgets.HTML(), widgets.Label()
    progress = widgets.FloatProgress(min=0, max=1, description="Progress:")
    output = widgets.Output()
    cancel, undo = widgets.Button(description="Cancel operation"), widgets.Button(description="Undo accepted change")
    path = widgets.Text(description="Dataset path:", layout=widgets.Layout(width="850px"))
    browse = widgets.Button(description="Browse dataset", icon="folder-open")
    format_hint = widgets.HTML()
    kind = widgets.Dropdown(options=[("Auto detect", "auto"), ("PyCCAPT HDF5", "pyccapt"), ("PyCCAPT NeXus mapping", "nexus"), ("LEAP epos", "leap_epos"),
        ("LEAP pos", "leap_pos"), ("LEAP apt", "leap_apt"), ("ATO v6", "ato_v6")], description="Format:")
    mode = widgets.Dropdown(options=[("Auto detect", "auto"), ("Processed /df", "processed"), ("Raw /dld", "raw")], description="HDF5 mode:")
    load = widgets.Button(description="Import dataset")
    session_path = widgets.Text(value="analysis_session.h5", description="Session path:", layout=widgets.Layout(width="850px"))
    resume = widgets.Button(description="Resume saved session")
    session_directory = widgets.Text(value=".", description="Session folder:", layout=widgets.Layout(width="650px"))
    session_query = widgets.Text(description="Search:")
    find_sessions = widgets.Button(description="Find sessions")
    session_list = widgets.Dropdown(options=[], description="Saved sessions:", layout=widgets.Layout(width="850px"))
    inspect = widgets.Button(description="Inspect source")
    crop_start = widgets.IntText(value=0, description="Start event:")
    crop_stop = widgets.IntText(value=1, description="Stop event:")
    crop_tof = widgets.Checkbox(description="Limit raw ToF (ns)", indent=False)
    crop_tof_min, crop_tof_max = widgets.FloatText(value=0., description="ToF min:"), widgets.FloatText(value=5000., description="ToF max:")
    crop_voltage = widgets.Checkbox(description="Limit DC voltage (V)", indent=False)
    crop_voltage_min, crop_voltage_max = widgets.FloatText(value=0., description="Voltage min:"), widgets.FloatText(value=20000., description="Voltage max:")
    crop_detector = widgets.Checkbox(description="Keep detector circle (cm)", indent=False)
    crop_x, crop_y, crop_radius = (widgets.FloatText(value=value, description=label) for value, label in ((0., "Center x:"), (0., "Center y:"), (4., "Radius:")))
    preview_crop = widgets.Button(description="Preview crop", button_style="primary")
    accept_crop = widgets.Button(description="Accept crop", button_style="success")
    crop_state = widgets.HTML()
    reset_crop = widgets.Button(description="Preview all events")
    draw_crop = widgets.Button(description="Draw crop regions")
    update_fdm = widgets.Button(description="Update detector map")
    suggest_crop = widgets.Button(description="Suggest crop from data")
    crop_suggestion = widgets.HTML()
    drawing_box = widgets.VBox()
    crop_output = widgets.Output()
    flight = widgets.FloatText(value=100., description="Flight mm:")
    t0 = widgets.FloatText(value=0., description="Time zero ns:")
    pulse_mode = widgets.Dropdown(options=["laser", "voltage"], description="Pulse mode:")
    instrument_confirmed = widgets.Checkbox(description="I checked flight length, time zero and pulse mode for this run", indent=False)
    calibration_help = widgets.HTML()
    calibration_state = widgets.HTML()
    start_cal = widgets.Button(description="Start calibration", button_style="primary")
    open_corrections = widgets.Button(description="Open voltage/bowl tools")
    capture_corrections = widgets.Button(description="Preview corrected calibration", button_style="primary")
    detector_diameter = widgets.FloatText(value=80., description="Detector mm:")
    correction_box = widgets.VBox()
    calibration_tabs = widgets.Tab()
    imported = widgets.Button(description="Preview imported mass")
    physical = widgets.Button(description="Preview physical mass")
    accept_cal = widgets.Button(description="Accept calibration")
    compare_cal = widgets.Button(description="Compare calibration")
    cal_output = widgets.Output()
    mass_output = widgets.Output()
    mass_view_min = widgets.FloatText(value=0., description="View mass min:")
    mass_view_max = widgets.FloatText(value=100., description="View mass max:")
    show_mass = widgets.Button(description="Show mass comparison")
    reference_windows = widgets.Textarea(value="[]", description="References:", layout=widgets.Layout(width="850px", height="150px"))
    propose_refs = widgets.Button(description="Propose reference windows")
    reference_domain = widgets.Dropdown(options=[("Accepted initial mass (helps when voltage changes)", "mass"),
        ("Raw flight time", "tof")], description="Window basis:", layout=widgets.Layout(width="650px"))
    reference_confirmed = widgets.Checkbox(description="I independently confirmed each reference mass identity", indent=False)
    roles = widgets.Dropdown(options=[("Interleaved 60/20/20 (correlated holdout)", "interleaved"),
        ("Contiguous acquisition 60/20/20", "blocked")], description="Validation:", layout=widgets.Layout(width="650px"))
    configure = widgets.Button(description="Create reference session")
    capture = widgets.Button(description="Capture applied model")
    open_ranges = widgets.Button(description="Open peak-review queue")
    range_box = widgets.VBox()
    recon_kind = widgets.Dropdown(options=["Geiser", "Bas"], description="Projection:")
    icf = widgets.FloatText(value=1.4, description="ICF:")
    kf = widgets.FloatText(value=3.3, description="kf:")
    efficiency = widgets.BoundedFloatText(value=.5, min=.001, max=1., description="Efficiency:")
    field = widgets.BoundedFloatText(value=30., min=.01, max=1000., description="Field V/nm:")
    density = widgets.BoundedFloatText(value=85., min=.01, max=1000., description="Atoms/nm³:")
    preview_recon = widgets.Button(description="Preview reconstruction")
    accept_recon = widgets.Button(description="Accept reconstruction")
    compare_recon = widgets.Button(description="Compare reconstruction")
    ratio = widgets.Button(description="Review ICF/kf ratio")
    capture_ratio = widgets.Button(description="Capture ratio geometry")
    recon_output = widgets.Output()
    ratio_box = widgets.VBox()
    plot_mode = widgets.Dropdown(options=[("In this cell", "inline"), ("Separate browser HTML", "browser")], description="Show plot:")
    plot_points = widgets.BoundedIntText(value=50000, min=100, max=1000000, description="Max shown:")
    reconstruction_tabs = widgets.Tab()
    analyze = widgets.Button(description="Analyze accepted ranges")
    save = widgets.Button(description="Save resumable session")
    export_path = widgets.Text(value="guided_result.h5", description="Result path:", layout=widgets.Layout(width="850px"))
    export = widgets.Button(description="Export accepted HDF5")
    export_nexus = widgets.Button(description="Export NeXus mapping")
    buttons = [browse, load, resume, find_sessions, inspect, draw_crop, update_fdm, suggest_crop, preview_crop, accept_crop, reset_crop, imported, physical, start_cal, open_corrections, capture_corrections, accept_cal, compare_cal, show_mass, propose_refs, configure,
               capture, open_ranges, preview_recon, accept_recon, compare_recon, ratio, capture_ratio, analyze, save, export, export_nexus, undo]
    for button in (*buttons, cancel):
        button.layout.width = "auto"
        button.layout.min_width = "170px"
        button.layout.flex = "0 0 auto"
    for checkbox in (instrument_confirmed, reference_confirmed, crop_tof, crop_voltage, crop_detector):
        checkbox.layout.width = "auto"
        checkbox.style.description_width = "initial"

    def row(children):
        return widgets.Box(children, layout=widgets.Layout(display="flex", flex_flow="row wrap", grid_gap="6px"))

    def update_format_hint(_=None):
        from pyccapt.calibration.data_tools.dataset_formats import FORMAT_LABELS, HDF_FORMATS, format_from_path
        inferred = format_from_path(path.value)
        chosen = inferred if kind.value == "auto" else kind.value
        mode.disabled = job.running or chosen not in HDF_FORMATS
        if kind.value != "auto":
            format_hint.value = f"Manual format: {escape(FORMAT_LABELS[kind.value])}. Choose Auto detect to infer it from the file."
        elif not path.value.strip():
            format_hint.value = "Browse for a dataset or paste its path. Format is detected when you import."
        elif inferred is None:
            format_hint.value = "Unknown extension: choose Format manually for a renamed supported dataset."
        elif inferred in HDF_FORMATS:
            format_hint.value = "Automatic: HDF5 container; raw /dld, processed /df or PyCCAPT NeXus layout is checked on import."
        else:
            format_hint.value = f"Automatic: {escape(FORMAT_LABELS[inferred])}."
        session = controller["workflow"]
        if kind.value == "auto" and session is not None and session.source_path and path.value.strip():
            recorded = session.source.attrs.get('pyccapt_import_format', {})
            if (Path(path.value).expanduser() == Path(session.source_path).expanduser() and
                    recorded.get('data_type') in FORMAT_LABELS):
                label = FORMAT_LABELS[recorded['data_type']]
                detail = 'raw /dld' if recorded.get('mode') == 'raw' else 'processed'
                format_hint.value = f"Automatic: <b>{escape(label)}</b> ({detail}) — imported."

    path.observe(update_format_hint, names="value")
    kind.observe(update_format_hint, names="value")

    def active():
        if controller["workflow"] is None:
            raise CalibrationStateError("Import a dataset or resume a saved session first")
        return controller["workflow"]

    def message(value):
        with output:
            output.clear_output(wait=True)
            print(json.dumps(value, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))

    def refresh():
        session = controller["workflow"]
        status.value = "<b>Import → Inspect → Crop → Calibrate → Range → Reconstruct → Analyze → Export</b><br>" + (
            " · ".join(f"{escape(stage)}: <b>{escape(value)}</b>" for stage, value in session.status().items()) if session else "No dataset loaded")
        for button in buttons:
            button.disabled = job.running or (session is None and button not in (browse, load, resume, find_sessions))
        path.disabled = kind.disabled = job.running
        update_format_hint()
        cancel.disabled = not job.running
        drawing = controller["crop_drawing"]
        if drawing is not None:
            drawing.set_enabled(not job.running)
        update_fdm.disabled = job.running or drawing is None
        if session is None:
            crop_state.value = "Import a dataset before cropping."
            accept_crop.tooltip = "Import a dataset, then preview a crop before accepting it."
        if session is not None:
            usable_mass = any(name in session.source and name not in session.source.attrs.get("pyccapt_unavailable_columns", [])
                and np.any(np.isfinite(session.source[name]) & (session.source[name] > 0)) for name in ("mc (Da)", "mc_uc (Da)"))
            imported.disabled = job.running or not usable_mass
            calibration_help.value = ("<b>1. Start.</b> Check the instrument values and confirmation box, then click <b>Start calibration</b>. "
                "This calculates an initial mass from raw ToF and opens the original notebook's voltage/bowl and Auto calibration tools. "
                "Use <b>Preview imported mass</b> to retain a saved mass instead. Opening tools does not run a fit or accept a result.")
            accept_crop.disabled = job.running or "crop" not in session.preview
            pending_crop = session.preview.get("crop")
            if job.running:
                crop_state.value = "Wait for the current operation to finish, then preview the selection."
                accept_crop.tooltip = "An operation is running."
            elif controller["crop_error"]:
                crop_state.value = "<b>Preview failed:</b> " + escape(controller["crop_error"]) + " Adjust the selection and click <b>Preview crop</b> again."
                accept_crop.tooltip = "Fix the preview error before accepting a crop."
            elif pending_crop is not None:
                count = int(pending_crop["mask"].sum())
                crop_state.value = f"<b>Preview ready:</b> {count:,} of {session.n:,} events will stay. Review the plots below, then click <b>Accept crop</b>."
                accept_crop.tooltip = "Apply the current reviewed preview. Earlier accepted calibration and reconstruction will be cleared."
            elif controller["crop_dirty"] or session.accepted["crop"] is None:
                crop_state.value = "<b>Draft selection — not applied.</b> Click <b>Preview crop</b> to check what stays and enable <b>Accept crop</b>. Moving a shape or editing a filter requires a new preview."
                accept_crop.tooltip = "Click Preview crop first. Changing a shape or filter invalidates its previous preview."
            else:
                crop_state.value = "<b>Crop accepted.</b> Continue to <b>Calibrate</b>, or edit the regions and preview another crop."
                accept_crop.tooltip = "This crop is already accepted. Preview a new selection to change it."
            if session.review is None:
                range_box.children = ()
            open_ranges.disabled = job.running or session.accepted["calibration"] is None
            for button in (preview_recon, ratio, analyze):
                button.disabled = job.running or session.accepted["ranges"] is None
            accept_cal.disabled = job.running or "calibration" not in session.preview
            pending = session.preview.get("calibration", {})
            has_mass = bool(pending) or session.accepted["calibration"] is not None
            compare_cal.disabled = job.running or not pending
            show_mass.disabled = job.running or not has_mass
            start_cal.disabled = job.running or not instrument_confirmed.value
            open_corrections.disabled = job.running or not has_mass or not instrument_confirmed.value
            capture_corrections.disabled = job.running or controller["standard_workspace"] is None or not instrument_confirmed.value
            if pending.get("receipt", {}).get("method") in ("tof2mc physical conversion", "original voltage/bowl correction tools") and not instrument_confirmed.value:
                accept_cal.disabled = True
            if pending:
                calibration_state.value = "<b>Preview ready.</b> Inspect the mass spectrum, then <b>Accept calibration</b> to continue to Range. Correction tools edit a separate workspace; click <b>Preview corrected calibration</b> after each trial to refresh this preview."
            elif session.accepted["calibration"] is not None:
                calibration_state.value = "<b>Calibration accepted.</b> Continue to Range, or open correction tools for a new trial."
            else:
                calibration_state.value = "<b>No calibration yet.</b> Click <b>Start calibration</b> after confirming the instrument settings. Compare and Accept become available after a successful preview."
            accept_recon.disabled = job.running or "reconstruction" not in session.preview
            undo.disabled = job.running or not session.history

    def act(operation):
        def callback(_):
            try:
                result = operation()
                refresh()
                if result is not None:
                    message(result)
            except Exception as error:
                message(f"{type(error).__name__}: {error}")
                refresh()
        return callback

    def launch(operation, consume):
        outcome = {"failed": False}

        def failed(error):
            outcome["failed"] = True
            message(f"{type(error).__name__}: {error}")

        def finished():
            progress.bar_style = "warning" if outcome["failed"] else "success"
            progress_text.value = "Operation stopped; see the message below." if outcome["failed"] else "Operation complete; review the result below."
            refresh()
            if tabs.selected_index == STAGES.index("Crop"):
                open_crop_tab(None)

        progress.value, progress.bar_style = 0., ""
        progress_text.value = "Operation running…"
        job.start(operation, on_result=consume, on_error=failed,
            on_progress=lambda fraction, text: (setattr(progress, "value", fraction), setattr(progress_text, "value", text)),
            on_finished=finished)
        refresh()

    def replace_workflow(session):
        close_drawing()
        clear_corrections()
        controller["physical_generation"] += 1
        controller["crop_seeded"] = bool(session.accepted.get("crop") or session.preview.get("crop"))
        controller.update(crop_dirty=not bool(session.accepted.get("crop")), crop_error=None)
        controller.update(workflow=session, calibration_workspace=None, calibration_session=None, reconstruction_workspace=None,
                          calibration_panel=None, range_panel=None)
        reference_windows.value = "[]"
        reference_confirmed.value = False
        controller["restoring_instrument"] = True
        try:
            instrument_confirmed.value = False
        finally:
            controller["restoring_instrument"] = False
        restore_controls(session)
        range_box.children = ()
        cal_output.clear_output(wait=False)
        recon_output.clear_output(wait=False)
        ratio_box.children = ()
        crop_output.clear_output(wait=False)
        mass_output.clear_output(wait=False)
        message(session.inspect())

    def restore_controls(session):
        from pyccapt.instrument_metadata import extract_instrument_settings
        known = extract_instrument_settings(session.source)
        cal = session.accepted["calibration"]
        spatial = session.preview.get("reconstruction", session.accepted["reconstruction"])
        receipt = cal["receipt"] if cal else {}
        pending = session.preview.get("calibration", {}).get("receipt", {})
        physical_pending = pending.get("method") in ("tof2mc physical conversion", "original voltage/bowl correction tools")
        if physical_pending:
            receipt = pending
        parameters = spatial["parameters"] if spatial else {}
        mode_value = receipt["pulse_mode"] if physical_pending else parameters.get("pulse_mode", receipt.get("pulse_mode", known.get("pulse_mode")))
        controller["restoring_instrument"] = True
        try:
            if mode_value in ("laser", "voltage"):
                pulse_mode.value = mode_value
            flight.value = receipt["flight_length_mm"] if physical_pending else parameters.get("flight_path_length", receipt.get("flight_length_mm", known.get("flight_path_length_mm", flight.value)))
            t0.value = receipt.get("model", {}).get("t0_ns", receipt.get("t0_ns", known.get("t0_ns", t0.value)))
            detector_diameter.value = receipt.get("detector_diameter_mm", detector_diameter.value)
        finally:
            controller["restoring_instrument"] = False
        selected_crop = session.preview.get("crop", session.accepted.get("crop"))
        criteria = selected_crop["criteria"] if selected_crop else {}
        restore_crop_controls(criteria, session.n)
        controller["restoring_reconstruction"] = True
        try:
            for widget, key in ((icf, "icf"), (kf, "kf"), (efficiency, "det_eff"), (field, "field_evap"), (density, "avg_dens")):
                if key in parameters:
                    widget.value = parameters[key]
            projection = parameters.get("projection", parameters.get("mode"))
            if projection in ("Geiser", "Bas"):
                recon_kind.value = projection
        finally:
            controller["restoring_reconstruction"] = False

    def restore_crop_controls(criteria, n):
        controller["restoring_crop"] = True
        try:
            crop_start.value, crop_stop.value = criteria.get("event_start", 0), criteria.get("event_stop", n)
            crop_tof.value, crop_voltage.value, crop_detector.value = (key in criteria for key in ("tof_bounds", "voltage_bounds", "detector_circle"))
            if crop_tof.value:
                crop_tof_min.value, crop_tof_max.value = criteria["tof_bounds"]
            if crop_voltage.value:
                crop_voltage_min.value, crop_voltage_max.value = criteria["voltage_bounds"]
            if crop_detector.value:
                crop_x.value, crop_y.value, crop_radius.value = criteria["detector_circle"]
        finally:
            controller["restoring_crop"] = False

    def import_data():
        selected = (path.value, kind.value, mode.value)
        launch(lambda cancel_event, report: GuidedWorkflow.import_file(selected[0], data_type=selected[1], mode=selected[2]), replace_workflow)

    def browse_dataset():
        from pyccapt.calibration.data_tools import file_dialog
        if job.running:
            raise CalibrationStateError("Wait for the current operation before choosing another dataset")
        browse.disabled = True
        try:
            selected = file_dialog.choose_file_path(path.value or None, file_kind="guided_dataset")
        finally:
            refresh()
        if selected:
            kind.value = "auto"
            mode.value = "auto"
            path.value = selected
            return "Dataset selected. Click Import dataset; the file format will be detected automatically."

    def resume_data():
        selected = session_path.value
        launch(lambda cancel_event, report: GuidedWorkflow.load(selected), replace_workflow)

    def search_sessions():
        from pyccapt.calibration.data_tools.analysis_session import list_analysis_sessions
        selected = (session_directory.value, session_query.value)
        def receive(rows):
            session_list.options = [(f"{Path(row['path']).name} · {row.get('created_utc', '')}", row["path"])
                                    for row in rows if row.get("integrity") != "invalid"]
            message([{key: value for key, value in row.items() if key != "metadata"} for row in rows])
        launch(lambda cancel_event, report: list_analysis_sessions(*selected), receive)

    def selected_session(change):
        if change["new"]:
            session_path.value = change["new"]
    session_list.observe(selected_session, names="value")
    reference_windows.observe(lambda _: setattr(reference_confirmed, "value", False), names="value")

    def discard_physical_preview():
        session = controller["workflow"]
        pending = session.preview.get("calibration", {}) if session else {}
        if pending.get("receipt", {}).get("method") in ("tof2mc physical conversion", "original voltage/bowl correction tools"):
            session.preview.pop("calibration")
            mass_output.clear_output(wait=False)

    def instrument_edited(_):
        if controller["restoring_instrument"]:
            return
        controller["physical_generation"] += 1
        instrument_confirmed.value = False
        clear_corrections()
        discard_physical_preview()
        refresh()

    def confirmation_edited(change):
        if controller["restoring_instrument"]:
            return
        controller["physical_generation"] += 1
        if not change["new"]:
            discard_physical_preview()
            clear_corrections()
        refresh()

    for setting in (flight, t0, pulse_mode):
        setting.observe(instrument_edited, names="value")
    instrument_confirmed.observe(confirmation_edited, names="value")

    def reconstruction_edited(_):
        if controller["restoring_instrument"] or controller["restoring_reconstruction"]:
            return
        controller["reconstruction_generation"] += 1
        session = controller["workflow"]
        if session is not None and session.preview.pop("reconstruction", None) is not None:
            recon_output.clear_output(wait=False)
        refresh()

    for setting in (icf, kf, efficiency, field, density, recon_kind, flight, pulse_mode):
        setting.observe(reconstruction_edited, names="value")

    def sync_drawing():
        drawing = controller["crop_drawing"]
        if drawing is not None:
            drawing.set_draft((crop_start.value, crop_stop.value),
                              (crop_x.value, crop_y.value, crop_radius.value) if crop_detector.value else None)

    def crop_edited(_):
        if controller["restoring_crop"]:
            return
        controller.update(crop_dirty=True, crop_error=None)
        session = controller["workflow"]
        if session is not None and session.preview.pop("crop", None) is not None:
            crop_output.clear_output(wait=False)
        sync_drawing()
        refresh()

    for setting in (crop_start, crop_stop, crop_tof, crop_tof_min, crop_tof_max,
                    crop_voltage, crop_voltage_min, crop_voltage_max, crop_detector,
                    crop_x, crop_y, crop_radius):
        setting.observe(crop_edited, names="value")

    def close_drawing():
        drawing = controller["crop_drawing"]
        controller["crop_drawing"] = None
        if drawing is not None:
            drawing.close()
        drawing_box.children = ()

    def open_crop_drawing():
        from pyccapt.calibration.tutorials.tutorials_helpers.helper_guided_crop import create_crop_drawing
        session = active()
        close_drawing()

        def current():
            return not job.running and controller["workflow"] is session and controller["crop_drawing"] is drawing

        def interval_selected(start, stop):
            if current():
                crop_start.value, crop_stop.value = start, stop
                try:
                    drawing.refresh_detector()
                except CalibrationInputError as error:
                    drawing.status.value = str(error)
                sync_drawing()

        def circle_selected(x, y, radius):
            if current():
                crop_x.value, crop_y.value, crop_radius.value = x, y, radius
                crop_detector.value = True

        drawing = create_crop_drawing(session, on_interval=interval_selected, on_circle=circle_selected,
                                      interval_provider=lambda: (crop_start.value, crop_stop.value))
        controller["crop_drawing"] = drawing
        if not controller["crop_seeded"]:
            controller["crop_seeded"] = True
            circle = drawing.initial_circle()
            if circle is not None and not crop_detector.value:
                crop_x.value, crop_y.value, crop_radius.value = circle
                crop_detector.value = True
            suggest_crop_regions()
        sync_drawing()
        drawing_box.children = (drawing.widget,)

    def open_crop_tab(_):
        if controller["workflow"] is not None and controller["crop_drawing"] is None and not job.running:
            act(open_crop_drawing)(None)

    def refresh_fdm():
        drawing = controller["crop_drawing"]
        if drawing is not None:
            drawing.refresh_detector()
            sync_drawing()

    def suggest_crop_regions():
        from pyccapt.calibration.core.crop_suggestions import suggest_temporal_crop, suggest_detector_circle
        session = active()
        temporal = suggest_temporal_crop(session.source)
        spatial = suggest_detector_circle(session.source, event_start=temporal['event_start'], event_stop=temporal['event_stop'])
        controller['restoring_crop'] = True
        try:
            crop_start.value, crop_stop.value = temporal['event_start'], temporal['event_stop']
            if spatial['supported']:
                crop_x.value, crop_y.value, crop_radius.value = spatial['circle_cm']
                crop_detector.value = True
        finally:
            controller['restoring_crop'] = False
        crop_edited(None)
        crop_suggestion.value = '<b>Editable heuristic proposal:</b> '+escape(temporal['reason'])+'<br>'+escape(spatial['reason'])
        from pyccapt.calibration.core.calibration_metadata import record_step
        record_step(session.variables, 'crop_suggestion', parameters={'temporal': temporal, 'detector': spatial}, status='proposal')
        if controller['crop_drawing'] is not None:
            refresh_fdm()
        return {'temporal': temporal, 'detector': spatial}

    def undo_accepted():
        result = active().undo()
        replace_workflow(active())
        return result

    def inspect_source():
        from pyccapt.calibration.tutorials.tutorials_helpers.helper_guided_crop import source_overview
        session = active()
        report = session.inspect()
        with output:
            output.clear_output(wait=True)
            print(json.dumps(report, indent=2))
            fig = source_overview(session)
            display(fig)
            plt.close(fig)

    def crop_preview(all_events=False):
        from pyccapt.calibration.tutorials.tutorials_helpers.helper_guided_crop import source_overview
        session = active()
        if all_events:
            restore_crop_controls({}, session.n)
            close_drawing()
        criteria = {} if all_events else dict(event_start=crop_start.value, event_stop=crop_stop.value,
            tof_bounds=(crop_tof_min.value, crop_tof_max.value) if crop_tof.value else None,
            voltage_bounds=(crop_voltage_min.value, crop_voltage_max.value) if crop_voltage.value else None,
            detector_circle=(crop_x.value, crop_y.value, crop_radius.value) if crop_detector.value else None)
        session.preview.pop("crop", None)
        controller["crop_error"] = None
        try:
            result = session.preview_crop(**criteria)
            with crop_output:
                crop_output.clear_output(wait=True)
                fig = source_overview(session, session.preview["crop"]["mask"])
                try:
                    display(fig)
                finally:
                    plt.close(fig)
        except Exception as error:
            session.preview.pop("crop", None)
            controller["crop_error"] = str(error)
            raise
        return result

    def accept_selection():
        result = active().accept("crop")
        clear_corrections()
        controller.update(crop_dirty=False, crop_error=None)
        controller.update(calibration_workspace=None, calibration_session=None, reconstruction_workspace=None,
                          calibration_panel=None, range_panel=None)
        reference_windows.value = "[]"
        reference_confirmed.value = False
        cal_output.clear_output(wait=False)
        recon_output.clear_output(wait=False)
        mass_output.clear_output(wait=False)
        restore_controls(active())
        return result

    def show_mass_comparison():
        session = active()
        if not session.preview.get("calibration") and session.accepted["calibration"] is None:
            return "No mass preview yet. Confirm the instrument settings and click Start calibration first."
        lo, hi = mass_view_min.value, mass_view_max.value
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise CalibrationInputError("Mass view limits must be finite and increasing")
        with mass_output:
            mass_output.clear_output(wait=True)
            fig, axis = plt.subplots(figsize=(11, 3))
            for label, payload in (("Accepted", session.accepted["calibration"]), ("Preview", session.preview.get("calibration"))):
                if payload is not None:
                    values = payload["mass"]
                    mask = session.active_mask & np.isfinite(values) & (values >= lo) & (values <= hi)
                    axis.hist(values[mask], bins=2000, range=(lo, hi), histtype="step", label=f"{label}: {int(mask.sum()):,} in view")
            axis.set(xlabel="Mass-to-charge (Da)", ylabel="Events", yscale="log",
                     xlim=(lo, hi),
                     title="Display window only; source crop and accepted values are unchanged")
            if axis.get_legend_handles_labels()[0]:
                axis.legend()
            display(fig)
            plt.close(fig)

    def imported_preview():
        clear_corrections()
        result = active().preview_imported_mass()
        show_mass_comparison()
        return result

    def physical_preview(*, with_tools=False):
        from pyccapt.calibration.mc.mc_tools import tof2mc
        if not instrument_confirmed.value:
            raise CalibrationStateError("Check the flight length, time zero and pulse mode, then tick their confirmation box")
        if not np.isfinite([flight.value, t0.value]).all() or flight.value <= 0:
            raise CalibrationInputError("Flight length must be finite and positive in mm, and time zero finite in ns")
        session, revision = active(), active().revision
        clear_corrections()
        generation = controller["physical_generation"]
        working = session.working_variables()
        working.pulse_mode = pulse_mode.value
        geometry = (flight.value, t0.value, pulse_mode.value)
        selected_mask = session.active_mask.copy()
        def operation(cancel_event, report):
            values = np.full(session.n, np.nan)
            for start in range(0, session.n, 250000):
                if cancel_event.is_set():
                    raise CalibrationStateError("Physical conversion cancelled")
                sl = slice(start, start+250000)
                valid = (selected_mask[sl] & np.isfinite(working.dld_t[sl]) & (working.dld_t[sl] > geometry[1]) &
                    np.isfinite(working.dld_high_voltage[sl]) & (working.dld_high_voltage[sl] > 0) &
                    np.isfinite(working.dld_x_det[sl]) & np.isfinite(working.dld_y_det[sl]))
                if geometry[2] == "voltage":
                    valid &= np.isfinite(working.dld_pulse_v[sl])
                values[sl][valid] = tof2mc(working.dld_t[sl][valid], geometry[1], working.dld_high_voltage[sl][valid],
                    working.dld_x_det[sl][valid], working.dld_y_det[sl][valid], geometry[0], working.dld_pulse_v[sl][valid], geometry[2])
                report(min(1., (start+250000)/session.n), "Converting raw ToF with reviewed physical parameters")
            return values
        def receive(masses):
            if (controller["workflow"] is not session or generation != controller["physical_generation"] or
                    not instrument_confirmed.value):
                raise CalibrationStateError("Instrument settings or their confirmation changed during conversion; preview again")
            result = session.preview_calibration(masses, expected_revision=revision,
                receipt={"method": "tof2mc physical conversion", "flight_length_mm": geometry[0], "t0_ns": geometry[1], "pulse_mode": geometry[2]})
            show_mass_comparison()
            if with_tools:
                open_standard_tools()
            message(result)
        launch(operation, receive)

    def accept_calibration():
        pending = active().preview.get("calibration", {})
        if pending.get("receipt", {}).get("method") in ("tof2mc physical conversion", "original voltage/bowl correction tools") and not instrument_confirmed.value:
            raise CalibrationStateError("Confirm the physical conversion settings before accepting calibration")
        result = active().accept("calibration")
        clear_corrections()
        return result

    def clear_corrections():
        tools = controller["standard_panel"]
        if tools is not None:
            tools.dispose()
        controller["standard_panel"] = None
        controller["standard_workspace"] = None
        correction_box.children = ()

    def open_standard_tools():
        from pyccapt.calibration.core.guided_corrections import CorrectionWorkspace
        from pyccapt.calibration.tutorials.tutorials_helpers.helper_calibration import call_voltage_bowl_calibration
        from pyccapt.calibration.tutorials.tutorials_helpers.helper_guided_startup import enable_calibration_widgets
        if not instrument_confirmed.value:
            raise CalibrationStateError("Confirm instrument settings before opening correction tools")
        if not np.isfinite(detector_diameter.value) or detector_diameter.value <= 0:
            raise CalibrationInputError("Detector diameter must be finite and positive in mm")
        clear_corrections()
        enable_calibration_widgets()
        workspace = CorrectionWorkspace(active(), flight_length_mm=flight.value, t0_ns=t0.value,
            pulse_mode=pulse_mode.value, detector_diameter_mm=detector_diameter.value)
        tools = call_voltage_bowl_calibration(workspace.variables, detector_diameter.value,
            flight.value, pulse_mode.value, t0.value, display_panel=False)
        from pyccapt.calibration.core.calibration_metadata import record_step
        record_step(workspace.variables, "open_original_correction_tools", status="working",
            parameters={"detector_diameter_mm": detector_diameter.value, "flight_length_mm": flight.value,
                        "t0_ns": t0.value, "pulse_mode": pulse_mode.value})
        tools.controls["mode_tabs"].selected_index = 1  # Ranging uses corrected mass, not independently normalized ToF.
        controller["standard_workspace"] = workspace
        controller["standard_panel"] = tools
        correction_box.children = (widgets.HTML(f"Private correction workspace: {len(workspace.indices):,} events. "
            f"{workspace.summary['selected_events_not_corrected']:,} selected events lack correction inputs and keep their initial values."), tools)
        calibration_tabs.selected_index = 1
        return "Use Plot hist and select a peak, then Auto calibration or the individual voltage/bowl controls. Preview corrected calibration and accept separately."

    def capture_standard_tools():
        workspace = controller["standard_workspace"]
        if workspace is None or not instrument_confirmed.value:
            raise CalibrationStateError("Start calibration or open the voltage/bowl tools first")
        result = workspace.preview(active(), (flight.value, t0.value, pulse_mode.value))
        show_mass_comparison()
        return result

    def detector_edited(_):
        if controller["restoring_instrument"]:
            return
        clear_corrections()
        pending = active().preview.get("calibration", {}) if controller["workflow"] is not None else {}
        if pending.get("receipt", {}).get("method") == "original voltage/bowl correction tools":
            active().preview.pop("calibration")
            mass_output.clear_output(wait=False)
        refresh()

    detector_diameter.observe(detector_edited, names="value")

    def propose_references():
        from pyccapt.calibration.core.calibration_advanced import auto_detect_reference_peaks
        session = active()
        if reference_domain.value == "mass":
            if session.accepted["calibration"] is None:
                raise CalibrationStateError("Create and accept an initial mass estimate first, or choose raw flight-time windows")
            values = session.accepted["calibration"]["mass"][session.active_mask]
            lower, upper, bin_size = "mass_min_Da", "mass_max_Da", .02
        else:
            values = session.source["t (ns)"].to_numpy(dtype=float)[session.active_mask]
            lower, upper, bin_size = "tof_min_ns", "tof_max_ns", .1
        found = auto_detect_reference_peaks(values, n_peaks=8, prominence=20, distance=3, hist_bin_size=bin_size)
        reference_windows.value = json.dumps([{"mass_Da": None, lower: r["x1"], upper: r["x2"]} for r in found], indent=2)
        return "Windows only: fill independently known mass_Da identities and inspect overlap/tails. Mass-selected windows depend on the initial calibration and are not independent labels."

    def configure_references():
        from pyccapt.calibration.core.guided_calibration import GuidedCalibrationSession
        from pyccapt.calibration.core.multireference_tof import physical_tof_baseline
        from pyccapt.calibration.tutorials.tutorials_helpers.helper_guided_calibration import show_guided_calibration
        if not reference_confirmed.value:
            raise CalibrationStateError("Confirm the independently known reference identities first")
        if not instrument_confirmed.value:
            raise CalibrationStateError("Confirm the flight length and pulse mode before fitting; the model will fit its own time zero")
        session = active()
        rows = json.loads(reference_windows.value)
        if not 2 <= len(rows) <= 12:
            raise CalibrationInputError("Provide between 2 and 12 nonoverlapping known reference windows")
        working = session.working_variables()
        working.pulse_mode = pulse_mode.value
        baseline = physical_tof_baseline(working.dld_high_voltage, working.dld_x_det, working.dld_y_det,
                                         flight.value, working.dld_pulse_v, pulse_mode.value)
        valid = (session.active_mask & np.isfinite(working.dld_t) & (working.dld_t > 0) & np.isfinite(baseline) & (baseline > 0) &
                 np.isfinite(working.dld_high_voltage) & (working.dld_high_voltage > 0))
        references = {}
        for row in rows:
            mass = float(row["mass_Da"])
            mass_keys = {"mass_min_Da", "mass_max_Da"} & row.keys()
            tof_keys = {"tof_min_ns", "tof_max_ns"} & row.keys()
            if (mass_keys and len(mass_keys) != 2) or (tof_keys and len(tof_keys) != 2):
                raise CalibrationInputError("Supply both bounds for a reference window")
            if mass_keys:
                if session.accepted["calibration"] is None or tof_keys:
                    raise CalibrationInputError("Mass windows require an accepted initial mass and must not also specify ToF bounds")
                window_values = session.accepted["calibration"]["mass"]
                lo, hi = float(row["mass_min_Da"]), float(row["mass_max_Da"])
            else:
                window_values = working.dld_t
                lo, hi = float(row["tof_min_ns"]), float(row["tof_max_ns"])
            if not np.isfinite([mass, lo, hi]).all() or mass <= 0 or mass in references or not lo < hi:
                raise CalibrationInputError("Reference masses must be finite, positive and unique; window bounds must be finite and increasing")
            references[mass] = valid & (window_values >= lo) & (window_values < hi)
        selected_indices = np.flatnonzero(valid)
        sequence = np.arange(len(selected_indices))
        if roles.value == "interleaved":
            train_local, selection_local = sequence % 5 < 3, sequence % 5 == 3
        else:
            train_local, selection_local = sequence < .6*len(sequence), (sequence >= .6*len(sequence)) & (sequence < .8*len(sequence))
        role_masks = {}
        for name, local_mask in (("train", train_local), ("selection", selection_local), ("confirmation", ~(train_local | selection_local))):
            role_masks[name] = np.zeros(session.n, dtype=bool)
            role_masks[name][selected_indices[local_mask]] = True
        model_session = GuidedCalibrationSession(working, references, role_masks, mode="mc", flight_length_mm=flight.value,
            pulse_mode=pulse_mode.value, apply_mask=valid,
            baseline_mass_da=working.mc_calib if session.accepted["calibration"] else None)
        controller.update(calibration_workspace=working, calibration_session=model_session, calibration_revision=session.revision,
                          calibration_reference_windows=rows)
        with cal_output:
            cal_output.clear_output(wait=True)
            controller["calibration_panel"] = show_guided_calibration(model_session, background_fit=True)
        return {"next": "Fit and validate in the independent workspace below. Apply there, then Capture applied model and Accept calibration here.",
                "excluded_invalid_source_events": int((~valid).sum())}

    def capture_calibration():
        from pyccapt.calibration.data_tools.model_state import export_model_state
        fitted = controller["calibration_session"]
        if fitted is None or fitted.status != "applied":
            raise CalibrationStateError("Complete and apply the reference session before capturing it")
        model_state = {"learned_model": export_model_state(fitted.model),
            "roles": dict(fitted.roles), "reference_labels": fitted.labels, "apply_mask": fitted.apply_mask}
        receipt = dict(fitted.receipt(), reference_windows=controller["calibration_reference_windows"])
        result = active().preview_calibration(controller["calibration_workspace"].mc_calib, receipt=receipt,
            model_state=model_state, expected_revision=controller["calibration_revision"])
        show_mass_comparison()
        return result

    def open_review():
        from pyccapt.calibration.tutorials.tutorials_helpers.helper_peak_review import show_peak_review
        controller["range_panel"] = show_peak_review(active(), display_panel=False, on_change=refresh)
        range_box.children = (controller["range_panel"],)

    def compute_reconstruction_preview(working, parameters, revision, ratio_result=None):
        from pyccapt.calibration.reconstructions.sdm_calibration import reconstruct_xyz
        session = active()
        generation = controller["reconstruction_generation"]
        parameters = dict(parameters, pulse_mode=getattr(working, "pulse_mode", "laser"))
        def operation(cancel_event, report):
            report(.05, "Computing geometry; cancellation discards this preview when the kernel returns")
            xyz = reconstruct_xyz(working, parameters["kf"], parameters["icf"], parameters["det_eff"],
                                  parameters["field_evap"], parameters["avg_dens"], parameters["flight_path_length"],
                                  mode=parameters.get("projection", parameters.get("mode")))
            report(1., "Geometry preview ready")
            return np.column_stack(xyz)
        def receive(xyz):
            if controller["workflow"] is not session or generation != controller["reconstruction_generation"]:
                raise CalibrationStateError("Reconstruction settings changed during calculation; preview again")
            result = session.preview_reconstruction(xyz, parameters, ratio_result=ratio_result, expected_revision=revision)
            from pyccapt.calibration.reconstructions.reconstruction import reconstruction_plot
            working.x, working.y, working.z = xyz.T
            with recon_output:
                recon_output.clear_output(wait=True)
                reconstruction_plot(working, [1.] * len(working.range_data), opacity=.6, rotary_fig_save=False,
                    figname="guided_preview", save=False,
                    unranged_fraction=.01, display_mode=plot_mode.value, max_display_points=plot_points.value)
            message(result)
        launch(operation, receive)

    def reconstruction_preview():
        session = active()
        parameters = dict(projection=recon_kind.value, flight_path_length=flight.value, kf=kf.value, det_eff=efficiency.value,
                          icf=icf.value, field_evap=field.value, avg_dens=density.value)
        working = session.working_variables()
        working.pulse_mode = pulse_mode.value
        workspace = controller["reconstruction_workspace"]
        ratio_result = None
        if workspace is not None and controller["reconstruction_revision"] == session.revision:
            accepted = dict(workspace.reconstruction_parameters)
            if "mode" in accepted:
                accepted["projection"] = accepted.pop("mode")
            if accepted == parameters:
                ratio_result = getattr(workspace, "reconstruction_ratio_result", {})
        compute_reconstruction_preview(working, parameters, session.revision, ratio_result)

    def open_ratio():
        from pyccapt.calibration.tutorials.tutorials_helpers.helper_sdm_calibration import call_sdm_icf_kf_calibration
        session = active()
        working = session.working_variables()
        working.pulse_mode = pulse_mode.value
        controller.update(reconstruction_workspace=working, reconstruction_revision=session.revision)
        ratio_box.children = (call_sdm_icf_kf_calibration(working, flight,
            SimpleNamespace(value=(None, density.value, field.value)), display_panel=False,
            on_parameters=lambda _: capture_geometry(), parameter_widgets=dict(
                icf=icf, kf=kf, det_eff=efficiency, field_evap=field, avg_dens=density, mode=recon_kind)),)
        reconstruction_tabs.selected_index = 1
        return "Tune in this tab, then accept reviewed parameters to transfer them to the reconstruction tab."

    def capture_geometry():
        working = controller["reconstruction_workspace"]
        if working is None or not getattr(working, "reconstruction_parameters", {}):
            raise CalibrationStateError("Apply a reviewed reconstruction in the ratio workspace first")
        if active().revision != controller["reconstruction_revision"]:
            raise CalibrationStateError("The workflow changed; reopen ICF/kf tuning for the current selection")
        parameters = working.reconstruction_parameters
        for name, widget in (("icf", icf), ("kf", kf), ("det_eff", efficiency), ("field_evap", field), ("avg_dens", density)):
            widget.value = parameters[name]
        recon_kind.value = parameters.get("projection", parameters.get("mode", "Geiser"))
        flight.value = parameters["flight_path_length"]
        reconstruction_tabs.selected_index = 0
        return "Reviewed parameters loaded. Click Preview reconstruction, inspect the cell plot, then Accept reconstruction."

    def export_accepted(nexus=False):
        session = active()
        session._check()
        if session.accepted["calibration"] is None:
            raise CalibrationStateError("Accept mass calibration before exporting derived results")
        session._sync_outputs()
        target = Path(export_path.value).expanduser().resolve()
        metadata = {"workflow_source_hash": session.source_hash, "workflow_revision": session.revision,
                    "accepted_stages": session.status()}
        frame = session.export_frame()
        metadata["selection"] = frame.attrs["pyccapt_selection"]
        if nexus:
            from pyccapt.calibration.data_tools.nexus_export import export_nexus
            return export_nexus(target, frame, metadata=metadata, ranges=session.variables.range_data,
                                reconstruction=session.variables.reconstruction_parameters)
        from pyccapt.calibration.data_tools.data_tools import save_data
        session.variables.set_result_directory(target.parent)
        session.variables.set_result_data_directory(target.parent)
        session.variables.result_data_name = target.stem
        save_data(frame, session.variables, name=target.stem, save_range=not session.variables.range_data.empty,
                  analysis_session=session.to_state(), nexus_metadata=metadata)
        return str(target.parent / (target.stem+".h5"))

    actions = {browse: browse_dataset, load: import_data, resume: resume_data, find_sessions: search_sessions, inspect: inspect_source,
        draw_crop: open_crop_drawing, update_fdm: refresh_fdm, suggest_crop: suggest_crop_regions,
        preview_crop: crop_preview, accept_crop: accept_selection, reset_crop: lambda: crop_preview(True),
        # Imported mass has unknown historical instrument settings unless the
        # source supplies them; do not stamp current widget defaults as provenance.
        imported: imported_preview, show_mass: show_mass_comparison,
        physical: physical_preview, start_cal: lambda: physical_preview(with_tools=True),
        open_corrections: open_standard_tools, capture_corrections: capture_standard_tools,
        accept_cal: accept_calibration, compare_cal: lambda: active().compare("calibration"),
        propose_refs: propose_references, configure: configure_references, capture: capture_calibration, open_ranges: open_review,
        preview_recon: reconstruction_preview, accept_recon: lambda: active().accept("reconstruction"),
        compare_recon: lambda: active().compare("reconstruction"), ratio: open_ratio, capture_ratio: capture_geometry,
        analyze: lambda: active().analyze(), save: lambda: active().save(session_path.value),
        export: export_accepted, export_nexus: lambda: export_accepted(True), undo: undo_accepted, cancel: job.cancel}
    for button, operation in actions.items():
        button.on_click(act(operation))
    exact_crop = widgets.Accordion(children=[widgets.VBox([
        row([crop_start, crop_stop]), row([crop_tof, crop_tof_min, crop_tof_max]),
        row([crop_voltage, crop_voltage_min, crop_voltage_max]),
        row([crop_detector, crop_x, crop_y, crop_radius])])], selected_index=None)
    exact_crop.set_title(0, "Exact values and optional filters")
    reconstruction_tabs.children = (
        widgets.VBox([widgets.HTML("<b>3D reconstruction.</b> Enter reviewed parameters, preview the coordinates, "
            "inspect the plot, then accept. The optional ICF/kf calibration tab shares these parameter values. "
            "After tuning, preview again to update the 3D result."), row([recon_kind, icf, kf]),
            row([efficiency, field, density]), row([plot_mode, plot_points]),
            row([preview_recon, compare_recon, accept_recon]), recon_output]),
        widgets.VBox([widgets.HTML("<b>Optional ICF/kf calibration.</b> Open the tuning controls below. "
            "Review the ROI, indexed spacing and an independent ICF anchor. Parameters are shared with the first tab; "
            "accepted suggestions return there. Click Preview reconstruction and accept separately to update geometry."),
            row([ratio, capture_ratio]), ratio_box]))
    reconstruction_tabs.set_title(0, "3D reconstruction")
    reconstruction_tabs.set_title(1, "ICF/kf calibration")
    calibration_tabs.children = (
        widgets.VBox([calibration_help, row([flight, t0, pulse_mode, detector_diameter]), instrument_confirmed,
            row([start_cal, imported, physical])]),
        widgets.VBox([widgets.HTML("<b>2. Correct voltage and detector effects.</b> These are the original notebook's calibration tools. "
            "Start in <b>mc calibration</b>: Plot hist, select a reference peak, then run Auto calibration or individual corrections. "
            "Auto tools can choose their own working peak; review their diagnostics. No peak identity or successful calibration is guaranteed. "
            "The ToF tab corrects a separate array and does not automatically update mass. <b>Save correction</b> is only a working backup. "
            "After a trial, click <b>Preview corrected calibration</b> below and inspect it before accepting."),
            open_corrections, correction_box, capture_corrections]),
        widgets.VBox([widgets.HTML("<b>Optional known-reference refinement.</b> This alternative requires independently identified masses. "
            "Accept an initial mass before proposing mass windows. Window choice and correlated holdout do not prove chemistry."),
            reference_domain, propose_refs, reference_windows, reference_confirmed, roles, row([configure, capture]), cal_output]))
    for index, title in enumerate(("1. Start calibration", "2. Voltage / bowl correction", "Optional reference fitting")):
        calibration_tabs.set_title(index, title)
    tabs = widgets.Tab(children=[
        widgets.VBox([row([browse, path]), row([kind, mode, load]), format_hint, row([session_directory, session_query, find_sessions]),
                      session_list, session_path, resume]),
        widgets.VBox([widgets.HTML("Inspect units, finite capabilities and source identity before calibration."), inspect]),
        widgets.VBox([widgets.HTML("<b>Crop with the mouse.</b> A rectangle and circle appear automatically. "
            "Drag the orange rectangle's left/right handles to select the useful voltage interval. Drag the cyan circle's center to move it and its handles to resize it. "
            "Only the rectangle's horizontal span filters events. The initial shapes suggest a smooth voltage interval and the dense detector interior; inspect and adjust them. "
            "Numeric controls are under <b>Exact values and optional filters</b>. After typing an interval, click <b>Update detector map</b>. "
            "Start is included; stop is excluded. Preview shows what stays. Accepting a crop clears later results; Undo restores them. "
            "The original source remains in the session so reconstruction retains the earlier evaporation history."),
            row([draw_crop, update_fdm, suggest_crop]), crop_suggestion, drawing_box,
            crop_state, row([preview_crop, accept_crop, reset_crop]), exact_crop, crop_output]),
        widgets.VBox([widgets.HTML("<b>Start → correct → preview → accept.</b> Set actual instrument values; defaults are not instrument measurements."),
            calibration_tabs, calibration_state, row([compare_cal, accept_cal]),
            row([mass_view_min, mass_view_max, show_mass]), mass_output]),
        widgets.VBox([open_ranges, range_box]),
        reconstruction_tabs,
        widgets.VBox([widgets.HTML("Composition uses accepted integer interval assignments; ambiguous mixtures require review."), analyze]),
        widgets.VBox([session_path, save, export_path, row([export, export_nexus]),
            widgets.HTML("Session files retain source, accepted outputs, masks, model snapshots, review history and stable event IDs. NeXus output reports its partial mapping and missing context.")])])
    for index, title in enumerate(STAGES):
        tabs.set_title(index, title)
    tabs.observe(lambda change: open_crop_tab(change) if change["new"] == STAGES.index("Crop") else None,
                 names="selected_index")
    panel = widgets.VBox([status, row([undo, cancel]), progress, progress_text, tabs, output])
    panel.controller = controller
    panel.controls = {button.description: button for button in (*buttons, cancel)}
    panel.controls.update(path=path, kind=kind, mode=mode, session_path=session_path, export_path=export_path,
                          format_hint=format_hint, crop_state=crop_state,
                          tabs=tabs, job=job, refresh=refresh, pulse_mode=pulse_mode, flight=flight, t0=t0,
                          progress=progress, progress_text=progress_text,
                          instrument_confirmed=instrument_confirmed, calibration_help=calibration_help,
                          calibration_tabs=calibration_tabs, calibration_state=calibration_state, detector_diameter=detector_diameter,
                          crop_start=crop_start, crop_stop=crop_stop, crop_tof=crop_tof, crop_tof_min=crop_tof_min,
                          crop_tof_max=crop_tof_max, crop_voltage=crop_voltage, crop_voltage_min=crop_voltage_min,
                          crop_voltage_max=crop_voltage_max, crop_detector=crop_detector, crop_x=crop_x, crop_y=crop_y, crop_radius=crop_radius,
                          icf=icf, kf=kf, efficiency=efficiency, field=field, density=density,
                          reconstruction_tabs=reconstruction_tabs, plot_mode=plot_mode, plot_points=plot_points,
                          reference_windows=reference_windows, reference_confirmed=reference_confirmed, roles=roles)
    panel.controls["Propose ToF windows"] = propose_refs  # Compatibility for existing scripted click paths.
    panel.controls.update(reference_domain=reference_domain, mass_view_min=mass_view_min, mass_view_max=mass_view_max)
    if workflow is not None:
        restore_controls(workflow)
    refresh()
    if display_panel:
        display(panel)
    return panel
