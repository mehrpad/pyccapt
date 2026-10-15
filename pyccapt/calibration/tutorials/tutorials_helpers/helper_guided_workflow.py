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


def show_guided_workflow(workflow=None, *, display_panel=True):
    """Render controls only. Loading, fitting, acceptance and file writes require clicks."""
    controller = {"workflow": workflow, "calibration_workspace": None, "calibration_session": None,
                  "reconstruction_workspace": None, "calibration_panel": None, "range_panel": None}
    job = NotebookJob()
    status, progress_text = widgets.HTML(), widgets.Label()
    progress = widgets.FloatProgress(min=0, max=1, description="Progress:")
    output = widgets.Output()
    cancel, undo = widgets.Button(description="Cancel operation"), widgets.Button(description="Undo accepted change")
    path = widgets.Text(description="Dataset path:", layout=widgets.Layout(width="850px"))
    kind = widgets.Dropdown(options=[("PyCCAPT HDF5", "pyccapt"), ("LEAP epos", "leap_epos"),
        ("LEAP pos", "leap_pos"), ("LEAP apt", "leap_apt"), ("ATO v6", "ato_v6")], description="Format:")
    mode = widgets.Dropdown(options=["processed", "raw"], description="HDF5 mode:")
    load = widgets.Button(description="Import dataset")
    session_path = widgets.Text(value="analysis_session.h5", description="Session path:", layout=widgets.Layout(width="850px"))
    resume = widgets.Button(description="Resume saved session")
    session_directory = widgets.Text(value=".", description="Session folder:", layout=widgets.Layout(width="650px"))
    session_query = widgets.Text(description="Search:")
    find_sessions = widgets.Button(description="Find sessions")
    session_list = widgets.Dropdown(options=[], description="Saved sessions:", layout=widgets.Layout(width="850px"))
    inspect = widgets.Button(description="Inspect source")
    flight = widgets.BoundedFloatText(value=100., min=.001, max=100000., description="Flight mm:")
    t0 = widgets.FloatText(value=0., description="Time zero ns:")
    pulse_mode = widgets.Dropdown(options=["laser", "voltage"], description="Pulse mode:")
    imported = widgets.Button(description="Preview imported mass")
    physical = widgets.Button(description="Preview physical mass")
    accept_cal = widgets.Button(description="Accept calibration")
    compare_cal = widgets.Button(description="Compare calibration")
    cal_output = widgets.Output()
    reference_windows = widgets.Textarea(value="[]", description="References:", layout=widgets.Layout(width="850px", height="150px"))
    propose_refs = widgets.Button(description="Propose ToF windows")
    reference_confirmed = widgets.Checkbox(description="I independently confirmed each reference mass identity", indent=False)
    roles = widgets.Dropdown(options=[("Interleaved 60/20/20 (correlated holdout)", "interleaved"),
        ("Contiguous acquisition 60/20/20", "blocked")], description="Validation:", layout=widgets.Layout(width="650px"))
    configure = widgets.Button(description="Create reference session")
    capture = widgets.Button(description="Capture applied model")
    open_ranges = widgets.Button(description="Open peak-review queue")
    range_box = widgets.VBox()
    recon_kind = widgets.Dropdown(options=["Geiser", "Bas"], description="Projection:")
    icf = widgets.BoundedFloatText(value=1.4, min=.1, max=10., description="ICF:")
    kf = widgets.BoundedFloatText(value=3.3, min=.1, max=20., description="kf:")
    efficiency = widgets.BoundedFloatText(value=.5, min=.001, max=1., description="Efficiency:")
    field = widgets.BoundedFloatText(value=30., min=.01, max=1000., description="Field V/nm:")
    density = widgets.BoundedFloatText(value=85., min=.01, max=1000., description="Atoms/nm³:")
    preview_recon = widgets.Button(description="Preview reconstruction")
    accept_recon = widgets.Button(description="Accept reconstruction")
    compare_recon = widgets.Button(description="Compare reconstruction")
    ratio = widgets.Button(description="Review ICF/kf ratio")
    capture_ratio = widgets.Button(description="Capture ratio geometry")
    recon_output = widgets.Output()
    analyze = widgets.Button(description="Analyze accepted ranges")
    save = widgets.Button(description="Save resumable session")
    export_path = widgets.Text(value="guided_result.h5", description="Result path:", layout=widgets.Layout(width="850px"))
    export = widgets.Button(description="Export accepted HDF5")
    export_nexus = widgets.Button(description="Export NeXus mapping")
    buttons = [load, resume, find_sessions, inspect, imported, physical, accept_cal, compare_cal, propose_refs, configure,
               capture, open_ranges, preview_recon, accept_recon, compare_recon, ratio, capture_ratio, analyze, save, export, export_nexus, undo]

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
        status.value = "<b>Import → Inspect → Calibrate → Range → Reconstruct → Analyze → Export</b><br>" + (
            " · ".join(f"{escape(stage)}: <b>{escape(value)}</b>" for stage, value in session.status().items()) if session else "No dataset loaded")
        for button in buttons:
            button.disabled = job.running or (session is None and button not in (load, resume, find_sessions))
        cancel.disabled = not job.running
        if session is not None:
            if session.review is None:
                range_box.children = ()
            open_ranges.disabled = job.running or session.accepted["calibration"] is None
            for button in (preview_recon, ratio, analyze):
                button.disabled = job.running or session.accepted["ranges"] is None
            accept_cal.disabled = job.running or "calibration" not in session.preview
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
        job.start(operation, on_result=consume, on_error=lambda error: message(f"{type(error).__name__}: {error}"),
            on_progress=lambda fraction, text: (setattr(progress, "value", fraction), setattr(progress_text, "value", text)),
            on_finished=refresh)
        refresh()

    def replace_workflow(session):
        controller.update(workflow=session, calibration_workspace=None, calibration_session=None, reconstruction_workspace=None,
                          calibration_panel=None, range_panel=None)
        reference_windows.value = "[]"
        reference_confirmed.value = False
        restore_controls(session)
        range_box.children = ()
        cal_output.clear_output(wait=False)
        recon_output.clear_output(wait=False)
        message(session.inspect())

    def restore_controls(session):
        cal = session.accepted["calibration"]
        spatial = session.accepted["reconstruction"]
        receipt = cal["receipt"] if cal else {}
        parameters = spatial["parameters"] if spatial else {}
        mode_value = parameters.get("pulse_mode", receipt.get("pulse_mode"))
        if mode_value in ("laser", "voltage"):
            pulse_mode.value = mode_value
        flight.value = parameters.get("flight_path_length", receipt.get("flight_length_mm", flight.value))
        t0.value = receipt.get("t0_ns", t0.value)
        for widget, key in ((icf, "icf"), (kf, "kf"), (efficiency, "det_eff"), (field, "field_evap"), (density, "avg_dens")):
            if key in parameters:
                widget.value = parameters[key]
        projection = parameters.get("projection", parameters.get("mode"))
        if projection in ("Geiser", "Bas"):
            recon_kind.value = projection

    def import_data():
        selected = (path.value, kind.value, mode.value)
        launch(lambda cancel_event, report: GuidedWorkflow.import_file(selected[0], data_type=selected[1], mode=selected[2]), replace_workflow)

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

    def inspect_source():
        session = active()
        report = session.inspect()
        with output:
            output.clear_output(wait=True)
            print(json.dumps(report, indent=2))
            column = "mc (Da)" if "mc (Da)" in session.source else "t (ns)"
            values = session.source[column].to_numpy(dtype=float)
            fig, axis = plt.subplots(figsize=(11, 3))
            axis.hist(values[np.isfinite(values)], bins=500, histtype="step")
            axis.set(xlabel=column, ylabel="Events", yscale="log", title="Imported source (accepted derived state is shown in the header)")
            display(fig)
            plt.close(fig)

    def physical_preview():
        from pyccapt.calibration.mc.mc_tools import tof2mc
        session, revision = active(), active().revision
        working = session.working_variables()
        working.pulse_mode = pulse_mode.value
        geometry = (flight.value, t0.value, pulse_mode.value)
        def operation(cancel_event, report):
            values = np.full(session.n, np.nan)
            for start in range(0, session.n, 250000):
                if cancel_event.is_set():
                    raise CalibrationStateError("Physical conversion cancelled")
                sl = slice(start, start+250000)
                valid = (np.isfinite(working.dld_t[sl]) & (working.dld_t[sl] > geometry[1]) &
                    np.isfinite(working.dld_high_voltage[sl]) & (working.dld_high_voltage[sl] > 0) &
                    np.isfinite(working.dld_x_det[sl]) & np.isfinite(working.dld_y_det[sl]))
                if geometry[2] == "voltage":
                    valid &= np.isfinite(working.dld_pulse_v[sl])
                values[sl][valid] = tof2mc(working.dld_t[sl][valid], geometry[1], working.dld_high_voltage[sl][valid],
                    working.dld_x_det[sl][valid], working.dld_y_det[sl][valid], geometry[0], working.dld_pulse_v[sl][valid], geometry[2])
                report(min(1., (start+250000)/session.n), "Converting raw ToF with reviewed physical parameters")
            return values
        launch(operation, lambda masses: message(session.preview_calibration(masses, expected_revision=revision,
            receipt={"method": "tof2mc physical conversion", "flight_length_mm": geometry[0], "t0_ns": geometry[1], "pulse_mode": geometry[2]})))

    def propose_references():
        from pyccapt.calibration.core.calibration_advanced import auto_detect_reference_peaks
        values = active().source["t (ns)"].to_numpy(dtype=float)
        found = auto_detect_reference_peaks(values, n_peaks=8, prominence=20, distance=3, hist_bin_size=.1)
        reference_windows.value = json.dumps([{"mass_Da": None, "tof_min_ns": r["x1"], "tof_max_ns": r["x2"]} for r in found], indent=2)
        return "Proposed ToF windows only. Fill independently known masses; inspect tails/overlap before creating a reference session."

    def configure_references():
        from pyccapt.calibration.core.guided_calibration import GuidedCalibrationSession
        from pyccapt.calibration.core.multireference_tof import physical_tof_baseline
        from pyccapt.calibration.tutorials.tutorials_helpers.helper_guided_calibration import show_guided_calibration
        if not reference_confirmed.value:
            raise CalibrationStateError("Confirm the independently known reference identities first")
        session = active()
        rows = json.loads(reference_windows.value)
        if not 2 <= len(rows) <= 12:
            raise CalibrationInputError("Provide between 2 and 12 nonoverlapping known reference windows")
        working = session.working_variables()
        working.pulse_mode = pulse_mode.value
        baseline = physical_tof_baseline(working.dld_high_voltage, working.dld_x_det, working.dld_y_det,
                                         flight.value, working.dld_pulse_v, pulse_mode.value)
        valid = (np.isfinite(working.dld_t) & (working.dld_t > 0) & np.isfinite(baseline) & (baseline > 0) &
                 np.isfinite(working.dld_high_voltage) & (working.dld_high_voltage > 0))
        references = {}
        for row in rows:
            mass, lo, hi = float(row["mass_Da"]), float(row["tof_min_ns"]), float(row["tof_max_ns"])
            if mass in references or not lo < hi:
                raise CalibrationInputError("Reference masses must be unique and windows increasing")
            references[mass] = valid & (working.dld_t >= lo) & (working.dld_t < hi)
        sequence = np.arange(session.n)
        if roles.value == "interleaved":
            train, selection = sequence % 5 < 3, sequence % 5 == 3
        else:
            train, selection = sequence < .6*session.n, (sequence >= .6*session.n) & (sequence < .8*session.n)
        role_masks = {"train": train, "selection": selection, "confirmation": ~(train | selection)}
        model_session = GuidedCalibrationSession(working, references, role_masks, mode="mc", flight_length_mm=flight.value, pulse_mode=pulse_mode.value)
        controller.update(calibration_workspace=working, calibration_session=model_session, calibration_revision=session.revision)
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
        return active().preview_calibration(controller["calibration_workspace"].mc_calib, receipt=fitted.receipt(),
            model_state=model_state, expected_revision=controller["calibration_revision"])

    def open_review():
        from pyccapt.calibration.tutorials.tutorials_helpers.helper_peak_review import show_peak_review
        controller["range_panel"] = show_peak_review(active(), display_panel=False, on_change=refresh)
        range_box.children = (controller["range_panel"],)

    def compute_reconstruction_preview(working, parameters, revision, ratio_result=None):
        from pyccapt.calibration.reconstructions.reconstruction import atom_probe_recons_from_detector_Geiser_et_al, atom_probe_recons_Bas_et_al
        from pyccapt.calibration.reconstructions.sdm_calibration import build_hv
        session = active()
        parameters = dict(parameters, pulse_mode=getattr(working, "pulse_mode", "laser"))
        def operation(cancel_event, report):
            report(.05, "Computing geometry; cancellation discards this preview when the kernel returns")
            algorithm = atom_probe_recons_from_detector_Geiser_et_al if parameters.get("projection", parameters.get("mode")) == "Geiser" else atom_probe_recons_Bas_et_al
            xyz = algorithm(working.dld_x_det, working.dld_y_det, build_hv(working), parameters["flight_path_length"],
                            parameters["kf"], parameters["det_eff"], parameters["icf"], parameters["field_evap"], parameters["avg_dens"])
            report(1., "Geometry preview ready")
            return np.column_stack(xyz)
        launch(operation, lambda xyz: message(session.preview_reconstruction(xyz, parameters,
            ratio_result=ratio_result, expected_revision=revision)))

    def reconstruction_preview():
        session = active()
        parameters = dict(projection=recon_kind.value, flight_path_length=flight.value, kf=kf.value, det_eff=efficiency.value,
                          icf=icf.value, field_evap=field.value, avg_dens=density.value)
        working = session.working_variables()
        working.pulse_mode = pulse_mode.value
        compute_reconstruction_preview(working, parameters, session.revision)

    def open_ratio():
        from pyccapt.calibration.tutorials.tutorials_helpers.helper_sdm_calibration import call_sdm_icf_kf_calibration
        session = active()
        working = session.working_variables()
        working.pulse_mode = pulse_mode.value
        controller.update(reconstruction_workspace=working, reconstruction_revision=session.revision)
        with recon_output:
            recon_output.clear_output(wait=True)
            call_sdm_icf_kf_calibration(working, flight, SimpleNamespace(value=(None, density.value, field.value)))
        return "Ratio profiling runs in an independent workspace. Review compatibility/anchor evidence before capturing its geometry."

    def capture_geometry():
        working = controller["reconstruction_workspace"]
        if working is None or not getattr(working, "reconstruction_parameters", {}):
            raise CalibrationStateError("Apply a reviewed reconstruction in the ratio workspace first")
        compute_reconstruction_preview(working, dict(working.reconstruction_parameters), controller["reconstruction_revision"],
                                       getattr(working, "reconstruction_ratio_result", {}))

    def export_accepted(nexus=False):
        session = active()
        session._check()
        if session.accepted["calibration"] is None:
            raise CalibrationStateError("Accept mass calibration before exporting derived results")
        session._sync_outputs()
        target = Path(export_path.value).expanduser().resolve()
        metadata = {"workflow_source_hash": session.source_hash, "workflow_revision": session.revision,
                    "acquisition_completeness": "unverified", "accepted_stages": session.status()}
        if nexus:
            from pyccapt.calibration.data_tools.nexus_export import export_nexus
            return export_nexus(target, session.variables.data, metadata=metadata, ranges=session.variables.range_data,
                                reconstruction=session.variables.reconstruction_parameters)
        from pyccapt.calibration.data_tools.data_tools import save_data
        session.variables.set_result_directory(target.parent)
        session.variables.set_result_data_directory(target.parent)
        session.variables.result_data_name = target.stem
        save_data(session.variables.data, session.variables, name=target.stem, save_range=True,
                  analysis_session=session.to_state(), nexus_metadata=metadata)
        return str(target.parent / (target.stem+".h5"))

    actions = {load: import_data, resume: resume_data, find_sessions: search_sessions, inspect: inspect_source,
        # Imported mass has unknown historical instrument settings unless the
        # source supplies them; do not stamp current widget defaults as provenance.
        imported: lambda: active().preview_imported_mass(),
        physical: physical_preview, accept_cal: lambda: active().accept("calibration"), compare_cal: lambda: active().compare("calibration"),
        propose_refs: propose_references, configure: configure_references, capture: capture_calibration, open_ranges: open_review,
        preview_recon: reconstruction_preview, accept_recon: lambda: active().accept("reconstruction"),
        compare_recon: lambda: active().compare("reconstruction"), ratio: open_ratio, capture_ratio: capture_geometry,
        analyze: lambda: active().analyze(), save: lambda: active().save(session_path.value),
        export: export_accepted, export_nexus: lambda: export_accepted(True), undo: lambda: active().undo(), cancel: job.cancel}
    for button, operation in actions.items():
        button.on_click(act(operation))
    tabs = widgets.Tab(children=[
        widgets.VBox([path, widgets.HBox([kind, mode, load]), widgets.HBox([session_directory, session_query, find_sessions]),
                      session_list, session_path, resume]),
        widgets.VBox([widgets.HTML("Inspect units, finite capabilities and source identity before calibration."), inspect]),
        widgets.VBox([widgets.HTML("<b>Preview first; accept separately.</b> Set actual instrument values; control defaults are not instrument measurements."),
            widgets.HBox([flight, t0, pulse_mode]), widgets.HBox([imported, physical, compare_cal, accept_cal]),
            widgets.HTML("Optional multi-reference model: fill mass_Da identities in proposed raw-ToF windows. Interleaved validation is correlated and does not replace an independent acquisition."),
            propose_refs, reference_windows, reference_confirmed, roles, widgets.HBox([configure, capture]), cal_output]),
        widgets.VBox([open_ranges, range_box]),
        widgets.VBox([widgets.HBox([recon_kind, icf, kf]), widgets.HBox([efficiency, field, density]),
            widgets.HBox([preview_recon, compare_recon, accept_recon]), widgets.HBox([ratio, capture_ratio]), recon_output]),
        widgets.VBox([widgets.HTML("Composition uses accepted integer interval assignments; ambiguous mixtures require review."), analyze]),
        widgets.VBox([session_path, save, export_path, widgets.HBox([export, export_nexus]),
            widgets.HTML("Session files retain source, accepted outputs, masks, model snapshots, review history and stable event IDs. NeXus output reports its partial mapping and missing context.")])])
    for index, title in enumerate(STAGES):
        tabs.set_title(index, title)
    panel = widgets.VBox([status, widgets.HBox([undo, cancel]), progress, progress_text, tabs, output])
    panel.controller = controller
    panel.controls = {button.description: button for button in (*buttons, cancel)}
    panel.controls.update(path=path, kind=kind, mode=mode, session_path=session_path, export_path=export_path,
                          tabs=tabs, job=job, refresh=refresh, pulse_mode=pulse_mode, flight=flight,
                          icf=icf, kf=kf, efficiency=efficiency, field=field, density=density,
                          reference_windows=reference_windows, reference_confirmed=reference_confirmed)
    if workflow is not None:
        restore_controls(workflow)
    refresh()
    if display_panel:
        display(panel)
    return panel
