"""Synthetic click-path integration for the new notebook; no file or device auto-actions."""
import ast
from pathlib import Path
from threading import Event

import nbformat
import numpy as np
import pandas as pd

from tests.calibration.test_guided_workflow_review import accepted_mass, accept_one_range, catalogue, workflow
from pyccapt.calibration.core.guided_workflow import GuidedWorkflow
from pyccapt.calibration.tutorials.tutorials_helpers import helper_guided_workflow as helper
from pyccapt.calibration.tutorials.tutorials_helpers import helper_peak_review as peak_helper


class ImmediateJob:
    """Deterministic widget callback delivery; production threading is tested separately."""
    running = False

    def start(self, operation, *, on_result, on_error, on_progress=lambda *_: None, on_finished=lambda: None):
        self.running = True
        try:
            on_result(operation(Event(), on_progress))
        except Exception as error:
            on_error(error)
        finally:
            self.running = False
            on_finished()

    def cancel(self):
        pass


def test_panel_requires_explicit_preview_and_acceptance(monkeypatch):
    monkeypatch.setattr(helper, "display", lambda *_: None)
    session = workflow()
    panel = helper.show_guided_workflow(session, display_panel=False)
    assert session.accepted["calibration"] is None
    assert panel.controls["Open peak-review queue"].disabled
    panel.controls["Preview imported mass"].click()
    assert session.accepted["calibration"] is None
    assert not panel.controls["Accept calibration"].disabled
    panel.controls["Accept calibration"].click()
    assert session.accepted["calibration"] is not None
    assert "flight_length_mm" not in session.accepted["calibration"]["receipt"]
    assert "pulse_mode" not in session.accepted["calibration"]["receipt"]
    assert not panel.controls["Open peak-review queue"].disabled


def test_reference_confirmation_resets_when_json_changes(monkeypatch):
    monkeypatch.setattr(helper, "display", lambda *_: None)
    panel = helper.show_guided_workflow(workflow(), display_panel=False)
    panel.controls["reference_confirmed"].value = True
    panel.controls["reference_windows"].value = '[{"mass_Da": 27}]'
    assert not panel.controls["reference_confirmed"].value


def test_voltage_mode_reconstruction_matches_ratio_kernel_and_resume_controls(monkeypatch):
    from pyccapt.calibration.reconstructions.sdm_calibration import reconstruct_xyz
    monkeypatch.setattr(helper, "NotebookJob", ImmediateJob)
    monkeypatch.setattr(helper, "display", lambda *_: None)
    session = workflow()
    # A pulse column belongs to the imported source; prepare it before creating the workflow.
    frame = session.source.copy()
    frame["pulse_v (V)"] = 1000.
    session.variables.sync_from_data(frame)
    session = GuidedWorkflow(session.variables)
    session.preview_imported_mass(pulse_mode="voltage", flight_length_mm=110.)
    session.accept("calibration")
    accept_one_range(session)
    panel = helper.show_guided_workflow(session, display_panel=False)
    panel.controls["Preview reconstruction"].click()
    preview = session.preview["reconstruction"]
    working = session.working_variables()
    expected = reconstruct_xyz(working, 3.3, 1.4, .5, 30., 85., 110., mode="Geiser")
    np.testing.assert_allclose(preview["xyz"], np.column_stack(expected))
    assert preview["parameters"]["pulse_mode"] == "voltage"
    panel.controls["Accept reconstruction"].click()
    resumed = GuidedWorkflow.from_state(*session.to_state())
    restored_panel = helper.show_guided_workflow(resumed, display_panel=False)
    assert restored_panel.controls["pulse_mode"].value == "voltage"
    assert restored_panel.controls["flight"].value == 110.
    assert restored_panel.controls["icf"].value == 1.4


def test_export_writes_accepted_outputs_and_resumable_state(tmp_path, monkeypatch):
    monkeypatch.setattr(helper, "display", lambda *_: None)
    session = workflow()
    mass = session.source["mc (Da)"].to_numpy()*1.01
    tof = session.source["t (ns)"].to_numpy()-3.
    session.preview_calibration(mass, tof=tof)
    session.accept("calibration")
    # Accepted identities remain independent of the deliberately shifted mass fixture.
    queue = session.begin_review()
    queue.propose(catalogue(), bin_width=.01, prominence=20, tolerance=.5)
    row = queue.records[0]
    queue.choose(row["id"], row["candidates"][0]["key"])
    queue.decide(row["id"], "accept", ambiguity_reviewed=True)
    session.preview_ranges()
    session.accept("ranges")
    xyz = np.column_stack((np.arange(session.n), np.zeros(session.n), np.arange(session.n)*.2))
    session.preview_reconstruction(xyz, {"icf": 1.4, "kf": 3.3})
    session.accept("reconstruction")
    target = tmp_path / "accepted.h5"
    panel = helper.show_guided_workflow(session, display_panel=False)
    panel.controls["export_path"].value = str(target)
    panel.controls["Export accepted HDF5"].click()
    assert target.is_file()
    frame = pd.read_hdf(target, "df")
    np.testing.assert_allclose(frame["mc (Da)"], mass)
    np.testing.assert_allclose(frame["t_c (ns)"], tof)
    np.testing.assert_allclose(frame["z (nm)"], xyz[:, 2])
    np.testing.assert_array_equal(frame["event_id"], session.event_ids)
    resumed = GuidedWorkflow.load(target)
    np.testing.assert_allclose(resumed.variables.mc, mass)
    np.testing.assert_allclose(resumed.variables.z, xyz[:, 2])


def test_peak_click_path_never_accepts_detection_implicitly(monkeypatch):
    monkeypatch.setattr(peak_helper, "NotebookJob", ImmediateJob)
    monkeypatch.setattr(peak_helper, "display", lambda *_: None)
    monkeypatch.setattr(peak_helper, "get_isotope_catalogue", lambda **_: catalogue())
    session = accepted_mass()
    panel = peak_helper.show_peak_review(session, display_panel=False)
    panel.controls["Propose peaks"].click()
    assert session.review.records and session.review.ranges().empty
    panel.controls["reviewed"].value = True
    panel.controls["Accept identity"].click()
    assert len(session.review.ranges()) == 1
    assert session.accepted["ranges"] is None
    panel.controls["Preview ranges"].click()
    panel.controls["Accept ranges"].click()
    assert len(session.accepted["ranges"]["rows"]) == 1


def test_proposal_result_cannot_overwrite_concurrent_review_edits(monkeypatch):
    class DeferredJob(ImmediateJob):
        def start(self, operation, **callbacks):
            self.running = True
            self.operation, self.callbacks = operation, callbacks

        def finish(self):
            self.running = False
            super().start(self.operation, **self.callbacks)
    monkeypatch.setattr(peak_helper, "NotebookJob", DeferredJob)
    monkeypatch.setattr(peak_helper, "display", lambda *_: None)
    monkeypatch.setattr(peak_helper, "get_isotope_catalogue", lambda **_: catalogue())
    session = accepted_mass()
    queue = session.begin_review()
    queue.propose(catalogue(), prominence=20, bin_width=.01)
    panel = peak_helper.show_peak_review(session, display_panel=False)
    panel.controls["Propose peaks"].click()
    row = queue.records[0]
    queue.edit_bounds(row["id"], 26.8, 27.2)  # edit from a second panel/notebook cell
    panel.controls["job"].finish()
    assert session.review is queue
    assert queue._get(row["id"])["lower"] == 26.8


def test_new_notebook_is_valid_and_initial_cells_only_render_ui(monkeypatch):
    path = Path(__file__).parents[2] / "pyccapt/calibration/tutorials/jupyter_files/data_processing_guided.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    monkeypatch.setattr(helper, "display", lambda *_: None)
    namespace = {}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            # Saved outputs may belong to a user's local notebook run. Exercise
            # current source cells rather than treating those views as code.
            exec(compile(cell.source, str(path), "exec"), namespace)
    assert namespace["guided_panel"].controller["workflow"] is None


def test_background_reference_fit_can_be_cancelled_without_model_mutation(monkeypatch):
    from tests.calibration.test_guided_calibration import known_data, session_for
    from pyccapt.calibration.tutorials.tutorials_helpers import helper_guided_calibration, guided_jobs
    original_job = guided_jobs.NotebookJob
    deliveries = []
    monkeypatch.setattr(guided_jobs, "NotebookJob", lambda: original_job(dispatch=deliveries.append))
    variables, refs, roles, _ = known_data()
    session = session_for(variables, refs, roles)
    original_model, mass = session.model, variables.mc_calib.copy()
    panel = helper_guided_calibration.show_guided_calibration(session, background_fit=True, display_panel=False)
    panel.controls["Fit training"].click()
    panel.controls["job"].thread.join(timeout=10)
    assert not panel.controls["job"].thread.is_alive()
    panel.controls["cancel"].click()
    for delivery in deliveries:
        delivery()
    assert session.status == "preview" and session.model is original_model
    np.testing.assert_array_equal(variables.mc_calib, mass)


def test_ratio_panel_receives_flight_widget_and_capture_uses_accepted_parameters(monkeypatch):
    from pyccapt.calibration.tutorials.tutorials_helpers import helper_sdm_calibration
    from pyccapt.calibration.reconstructions.sdm_calibration import reconstruct_xyz
    monkeypatch.setattr(helper, "NotebookJob", ImmediateJob)
    monkeypatch.setattr(helper, "display", lambda *_: None)
    captured = {}
    def fake_ratio(variables, flight, selected, colab=False, **kwargs):
        import ipywidgets as widgets
        captured.update(variables=variables, flight=flight.value, density=selected.value[1])
        variables.reconstruction_parameters = dict(projection="Geiser", flight_path_length=125.,
            icf=1.6, kf=4., det_eff=.6, field_evap=31., avg_dens=88.)
        variables.reconstruction_ratio_result = {"q": .4, "method": "synthetic independently anchored ratio"}
        variables.x = variables.y = variables.z = np.full(len(variables.mc), 999.)  # stale sandbox geometry
        return widgets.VBox()
    monkeypatch.setattr(helper_sdm_calibration, "call_sdm_icf_kf_calibration", fake_ratio)
    session = accepted_mass()
    accept_one_range(session)
    panel = helper.show_guided_workflow(session, display_panel=False)
    panel.controls["Review ICF/kf ratio"].click()
    assert captured["flight"] == 100. and captured["density"] == 85.
    panel.controls["Capture ratio geometry"].click()
    assert "reconstruction" not in session.preview
    assert panel.controls['reconstruction_tabs'].selected_index == 0
    panel.controls["Preview reconstruction"].click()
    result = session.preview["reconstruction"]
    expected = reconstruct_xyz(captured["variables"], 4., 1.6, .6, 31., 88., 125.)
    np.testing.assert_allclose(result["xyz"], np.column_stack(expected))
    assert result["ratio_result"]["q"] == .4
    assert session.accepted["reconstruction"] is None


def test_guided_reconstruction_first_and_optional_calibration_share_controls(monkeypatch):
    from pyccapt.calibration.reconstructions import reconstruction
    monkeypatch.setattr(helper, 'NotebookJob', ImmediateJob)
    monkeypatch.setattr(reconstruction, 'reconstruction_plot', lambda *a, **k: None)
    session = accepted_mass()
    accept_one_range(session)
    panel = helper.show_guided_workflow(session, display_panel=False)
    tabs = panel.controls['reconstruction_tabs']
    assert tabs.get_title(0) == '3D reconstruction' and tabs.get_title(1) == 'ICF/kf calibration'
    assert tabs.selected_index == 0
    panel.controls['Review ICF/kf ratio'].click()
    assert tabs.selected_index == 1
    calibration = tabs.children[1].children[-1].children[0]
    for key, control_key in [('icf', 'icf'), ('kf', 'kf'), ('det_eff', 'efficiency'), ('field_evap', 'field'), ('avg_dens', 'density')]:
        assert calibration.parameter_widgets[key] is panel.controls[control_key]
    panel.controls['Preview reconstruction'].click()
    assert 'reconstruction' in session.preview and not panel.controls['Accept reconstruction'].disabled
    calibration.parameter_widgets['kf'].value = 30.  # A reviewed value must not silently clamp to the former UI max of 20.
    assert panel.controls['kf'].value == 30.
    assert 'reconstruction' not in session.preview and panel.controls['Accept reconstruction'].disabled
    assert session.accepted['reconstruction'] is None
    panel.controls['Preview reconstruction'].click()
    assert session.preview['reconstruction']['parameters']['kf'] == 30.
    assert not panel.controls['Accept reconstruction'].disabled


def test_geometry_settings_changed_during_calculation_discard_pending_result(monkeypatch):
    from pyccapt.calibration.reconstructions import reconstruction, sdm_calibration
    original = sdm_calibration.reconstruct_xyz
    monkeypatch.setattr(helper, 'NotebookJob', ImmediateJob)
    monkeypatch.setattr(reconstruction, 'reconstruction_plot', lambda *a, **k: None)
    session = accepted_mass()
    accept_one_range(session)
    panel = helper.show_guided_workflow(session, display_panel=False)
    def compute(*args, **kwargs):
        panel.controls['icf'].value = 1.8  # New notebook input before the queued result is delivered.
        return original(*args, **kwargs)
    monkeypatch.setattr(sdm_calibration, 'reconstruct_xyz', compute)
    panel.controls['Preview reconstruction'].click()
    assert 'reconstruction' not in session.preview and session.accepted['reconstruction'] is None
    assert panel.controls['Accept reconstruction'].disabled
    assert panel.controls['progress'].bar_style == 'warning'


def test_reopening_reconstruction_preview_restores_its_parameters_without_discarding_it(monkeypatch):
    from pyccapt.calibration.reconstructions import reconstruction
    monkeypatch.setattr(helper, 'NotebookJob', ImmediateJob)
    monkeypatch.setattr(reconstruction, 'reconstruction_plot', lambda *a, **k: None)
    session = accepted_mass()
    accept_one_range(session)
    panel = helper.show_guided_workflow(session, display_panel=False)
    panel.controls['kf'].value = 4.7
    panel.controls['icf'].value = 1.8
    panel.controls['Preview reconstruction'].click()
    restored = GuidedWorkflow.from_state(*session.to_state())
    reopened = helper.show_guided_workflow(restored, display_panel=False)
    assert reopened.controls['kf'].value == 4.7 and reopened.controls['icf'].value == 1.8
    assert 'reconstruction' in restored.preview
    assert not reopened.controls['Accept reconstruction'].disabled


def test_unbounded_geometry_fields_still_require_physically_valid_positive_inputs(monkeypatch):
    monkeypatch.setattr(helper, 'NotebookJob', ImmediateJob)
    session = accepted_mass()
    accept_one_range(session)
    panel = helper.show_guided_workflow(session, display_panel=False)
    for key, invalid in [('kf', 0.), ('icf', -1.)]:
        panel.controls['kf'].value, panel.controls['icf'].value = 3.3, 1.4
        panel.controls[key].value = invalid
        panel.controls['Preview reconstruction'].click()
        assert 'reconstruction' not in session.preview and session.accepted['reconstruction'] is None
        assert panel.controls['Accept reconstruction'].disabled
        assert panel.controls['progress'].bar_style == 'warning'


def test_physical_preview_refuses_zero_filled_unavailable_detector_measurement(monkeypatch):
    monkeypatch.setattr(helper, "NotebookJob", ImmediateJob)
    monkeypatch.setattr(helper, "display", lambda *_: None)
    session = workflow()
    frame = session.source.copy()
    frame["x_det (cm)"] = 0.
    frame.attrs["pyccapt_unavailable_columns"] = ["x_det (cm)"]
    session.variables.sync_from_data(frame)
    session = GuidedWorkflow(session.variables)
    working = session.working_variables()
    assert session.inspect()['unknown_positions'] == session.n
    assert np.isnan(working.dld_x_det).all()
    assert np.all(working.dld_high_voltage > 0)
    panel = helper.show_guided_workflow(session, display_panel=False)
    panel.controls["Preview physical mass"].click()
    assert "calibration" not in session.preview
    assert session.accepted["calibration"] is None
    np.testing.assert_array_equal(session.source["x_det (cm)"], 0.)


def test_voltage_physical_preview_requires_available_pulse_voltage(monkeypatch):
    monkeypatch.setattr(helper, "NotebookJob", ImmediateJob)
    monkeypatch.setattr(helper, "display", lambda *_: None)
    session = workflow()  # no measured pulse_v column
    panel = helper.show_guided_workflow(session, display_panel=False)
    panel.controls["pulse_mode"].value = "voltage"
    panel.controls["instrument_confirmed"].value = True
    panel.controls["Preview physical mass"].click()
    assert "calibration" not in session.preview
    panel.controls["pulse_mode"].value = "laser"
    panel.controls["instrument_confirmed"].value = True
    panel.controls["Preview physical mass"].click()
    assert np.isfinite(session.preview["calibration"]["mass"]).all()


def test_reference_window_configuration_excludes_unknown_detector_rows(monkeypatch):
    import json
    from tests.calibration.test_guided_calibration import known_data
    from pyccapt.calibration.tutorials.tutorials_helpers import helper_guided_calibration
    monkeypatch.setattr(helper, "display", lambda *_: None)
    monkeypatch.setattr(helper_guided_calibration, "display", lambda *_: None)
    variables, refs, _, _ = known_data(missing=True)
    session = GuidedWorkflow(variables)
    panel = helper.show_guided_workflow(session, display_panel=False)
    panel.controls["flight"].value = 110.
    rows = [{"mass_Da": mass, "tof_min_ns": float(variables.dld_t[mask].min()-1),
             "tof_max_ns": float(variables.dld_t[mask].max()+1)} for mass, mask in refs.items()]
    panel.controls["reference_windows"].value = json.dumps(rows)
    panel.controls["reference_confirmed"].value = True
    panel.controls["instrument_confirmed"].value = True
    panel.controls["Create reference session"].click()
    configured = panel.controller["calibration_session"]
    assert configured is not None
    assert not configured.apply_mask[0] and np.isnan(configured.labels[0])
    assert configured.apply_mask[1:].all()
