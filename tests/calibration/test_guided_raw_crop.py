"""Raw no-mass import, reviewed selection and retained reconstruction chronology."""
import h5py
import numpy as np
import pandas as pd
import pytest

from pyccapt.calibration.core.exceptions import CalibrationInputError, CalibrationStateError
from pyccapt.calibration.core.guided_crop import crop_mask
from pyccapt.calibration.core.guided_workflow import GuidedWorkflow
from tests.calibration.test_guided_notebook_ui import ImmediateJob
from tests.calibration.test_guided_workflow_review import accepted_mass, accept_one_range, workflow


def test_crop_half_open_intersections_preserve_source_and_large_ids():
    session = workflow()
    before = session.source.copy(deep=True)
    info = session.preview_crop(event_start=100, event_stop=150, tof_bounds=(100., 200.), detector_circle=(0., 0., 2.))
    assert info["preview_selected"] == 50 and session.active_mask.all()
    session.accept("crop")
    assert session.active_mask.sum() == 50 and session.n == len(before)
    session.preview_imported_mass()
    session.accept("calibration")
    assert np.isnan(session.variables.mc[:100]).all()
    assert np.isnan(session.variables.mc[150:]).all()
    pd.testing.assert_frame_equal(session.source, before)
    exported = session.export_frame()
    np.testing.assert_array_equal(exported.event_id, before.event_id.iloc[100:150])
    assert exported.attrs["pyccapt_selection"]["preserves_full_source"] is False
    assert not session.active_mask.flags.writeable


@pytest.mark.parametrize("criteria", [dict(event_start=-1), dict(event_start=10, event_stop=10),
    dict(event_stop=5000), dict(tof_bounds=(1., 1.)), dict(detector_circle=(0., 0., -1.)),
    dict(voltage_bounds=(6000., 7000.))])
def test_invalid_or_empty_crop_is_atomic(criteria):
    session = accepted_mass()
    before = session.variables.mc.copy()
    with pytest.raises(CalibrationInputError):
        session.preview_crop(**criteria)
    np.testing.assert_array_equal(session.variables.mc, before)
    assert session.active_mask.all() and "crop" not in session.preview


def test_crop_on_unavailable_field_fails_but_index_crop_is_valid():
    frame = pd.DataFrame({"t (ns)": [10., 20., 30.]})
    mask, _ = crop_mask(frame, event_start=1, event_stop=3)
    np.testing.assert_array_equal(mask, [False, True, True])
    with pytest.raises(CalibrationInputError, match="unavailable"):
        crop_mask(frame, detector_circle=(0., 0., 1.))


def test_crop_invalidates_downstream_and_undo_restores_review_and_mass():
    session = accepted_mass()
    accept_one_range(session)
    previous = session.accepted["ranges"]
    session.preview_crop(event_start=100, event_stop=900)
    session.accept("crop")
    assert session.accepted["calibration"] is None and session.review is None
    assert session.accepted["ranges"] is None
    session.undo()
    assert session.active_mask.all()
    assert session.accepted["ranges"] is previous and session.review is not None


def test_reexport_cannot_erase_prior_missing_acquisition_events():
    session = workflow()
    frame = session.source.copy()
    frame.attrs["pyccapt_selection"] = {"source_events": 10000, "selected_events": len(frame), "preserves_full_source": False}
    session.variables.sync_from_data(frame)
    reimported = GuidedWorkflow(session.variables)
    reimported.preview_imported_mass()
    reimported.accept("calibration")
    metadata = reimported.export_frame().attrs["pyccapt_selection"]
    assert not metadata["preserves_full_source"] and metadata["source_events"] == 10000


def test_imported_mass_fallback_uses_selected_events_only():
    from pyccapt.calibration.core.share_variables import Variables
    frame = workflow().source.copy()
    frame.loc[100:899, "mc (Da)"] = np.nan
    frame["t_c (ns)"] = 100.
    variables = Variables()
    variables.sync_from_data(frame)
    session = GuidedWorkflow(variables)
    session.preview_crop(event_start=100, event_stop=900)
    session.accept("crop")
    report = session.preview_imported_mass()
    assert report["imported_mass_column"] == "mc_uc (Da)"
    assert session.preview["calibration"]["tof"] is None
    np.testing.assert_array_equal(session.preview["calibration"]["mass"][session.active_mask], frame["mc_uc (Da)"].iloc[100:900])


def test_crop_session_resume_and_legacy_schema_compatibility(tmp_path):
    session = accepted_mass()
    legacy_metadata, arrays = session.to_state()
    legacy_metadata["workflow_schema"] = "1.0"
    legacy_metadata["accepted"].pop("crop")
    for snapshot in legacy_metadata["history"]:
        snapshot["accepted"].pop("crop")
    assert GuidedWorkflow.from_state(legacy_metadata, arrays).active_mask.all()
    session.preview_crop(event_start=99, event_stop=999)
    session.accept("crop")
    session.preview_imported_mass()
    session.accept("calibration")
    path = tmp_path / "cropped_session.h5"
    session.save(path)
    restored = GuidedWorkflow.load(path)
    np.testing.assert_array_equal(restored.active_mask, session.active_mask)
    np.testing.assert_array_equal(restored.event_ids, session.event_ids)
    np.testing.assert_array_equal(restored.variables.mc, session.variables.mc)
    metadata, arrays = session.to_state()
    metadata["accepted"]["crop"]["mask"] = [True]
    with pytest.raises(CalibrationStateError, match="crop"):
        GuidedWorkflow.from_state(metadata, arrays)


@pytest.mark.parametrize("corruption", ["criteria", "preview_outside", "preview_dependency"])
def test_crop_receipt_and_preview_consistency_checked_before_resume(corruption):
    session = workflow()
    session.preview_crop(event_start=100, event_stop=900)
    session.accept("crop")
    session.preview_imported_mass()
    session.accept("calibration")
    accept_one_range(session)
    session.preview_reconstruction(np.ones((session.n, 3)), {"icf": 1.4, "kf": 3.3})
    metadata, arrays = session.to_state()
    if corruption == "criteria":
        metadata["accepted"]["crop"]["criteria"]["event_start"] = 0
    elif corruption == "preview_outside":
        metadata["preview"]["reconstruction"]["xyz"] = np.ones((session.n, 3)).tolist()
    else:
        metadata["accepted"]["ranges"] = None
    with pytest.raises(CalibrationStateError):
        GuidedWorkflow.from_state(metadata, arrays)


def test_mass_reference_windows_and_blocked_roles_follow_selected_order(monkeypatch):
    import json
    from pyccapt.calibration.core.share_variables import Variables
    from pyccapt.calibration.tutorials.tutorials_helpers import helper_guided_workflow as helper
    monkeypatch.setattr(helper, "display", lambda *_: None)
    source = workflow().source.copy()
    source["mc (Da)"] = np.tile([27., 28.], len(source)//2)
    source["mc_uc (Da)"] = source["mc (Da)"]
    variables = Variables()
    variables.sync_from_data(source)
    session = GuidedWorkflow(variables)
    session.preview_crop(event_start=100, event_stop=1100)
    session.accept("crop")
    session.preview_imported_mass()
    session.accept("calibration")
    panel = helper.show_guided_workflow(session, display_panel=False)
    panel.controls["roles"].value = "blocked"
    panel.controls["reference_windows"].value = json.dumps([
        {"mass_Da": mass, "mass_min_Da": mass-.1, "mass_max_Da": mass+.1} for mass in (27., 28.)])
    panel.controls["reference_confirmed"].value = True
    panel.controls["Create reference session"].click()
    assert panel.controller["calibration_session"] is None
    panel.controls["instrument_confirmed"].value = True
    panel.controls["Create reference session"].click()
    fitted = panel.controller["calibration_session"]
    assert fitted is not None
    assert [int(fitted.roles[role].sum()) for role in ("train", "selection", "confirmation")] == [600, 200, 200]
    np.testing.assert_array_equal(np.flatnonzero(fitted.roles["selection"]), np.arange(700, 900))
    assert np.isnan(fitted.labels[~session.active_mask]).all()
    np.testing.assert_array_equal(fitted.apply_mask, session.active_mask)


def test_cropped_reconstruction_preserves_full_voltage_prefix(monkeypatch):
    from pyccapt.calibration.reconstructions.sdm_calibration import reconstruct_xyz
    from pyccapt.calibration.tutorials.tutorials_helpers import helper_guided_workflow as helper
    monkeypatch.setattr(helper, "NotebookJob", ImmediateJob)
    monkeypatch.setattr(helper, "display", lambda *_: None)
    session = workflow()
    session.preview_crop(event_start=100, event_stop=900)
    session.accept("crop")
    session.preview_imported_mass()
    session.accept("calibration")
    accept_one_range(session)
    panel = helper.show_guided_workflow(session, display_panel=False)
    panel.controls["Preview reconstruction"].click()
    expected = np.column_stack(reconstruct_xyz(session.working_variables(), 3.3, 1.4, .5, 30., 85., 100., mode="Geiser"))
    actual = session.preview["reconstruction"]["xyz"]
    np.testing.assert_array_equal(actual[session.active_mask], expected[session.active_mask])
    assert np.isnan(actual[~session.active_mask]).all()
    assert len(session.working_variables().dld_high_voltage) == session.n


@pytest.mark.parametrize("bad_row", [
    {"mass_Da": 27., "tof_min_ns": -np.inf, "tof_max_ns": np.inf},
    {"mass_Da": np.inf, "tof_min_ns": 0., "tof_max_ns": 1000.},
    {"mass_Da": 0., "tof_min_ns": 0., "tof_max_ns": 1000.},
    {"mass_Da": 27., "mass_min_Da": 26., "tof_min_ns": 0., "tof_max_ns": 1000.},
    {"mass_Da": 27., "mass_min_Da": 26., "mass_max_Da": 28., "tof_min_ns": 0.},
])
def test_reference_window_inputs_are_complete_and_serializable(monkeypatch, bad_row):
    import json
    from pyccapt.calibration.tutorials.tutorials_helpers import helper_guided_workflow as helper
    monkeypatch.setattr(helper, "display", lambda *_: None)
    panel = helper.show_guided_workflow(accepted_mass(), display_panel=False)
    panel.controls["reference_windows"].value = json.dumps([
        bad_row, {"mass_Da": 28., "tof_min_ns": 1000., "tof_max_ns": 2000.}])
    panel.controls["reference_confirmed"].value = True
    panel.controls["instrument_confirmed"].value = True
    panel.controls["Create reference session"].click()
    assert panel.controller["calibration_session"] is None


@pytest.mark.parametrize("setting,value", [("flight", 0.), ("flight", -110.), ("flight", np.inf),
                                          ("t0", np.nan), ("t0", np.inf)])
def test_confirmed_physical_inputs_still_require_valid_geometry(monkeypatch, setting, value):
    from pyccapt.calibration.tutorials.tutorials_helpers import helper_guided_workflow as helper
    monkeypatch.setattr(helper, "NotebookJob", ImmediateJob)
    monkeypatch.setattr(helper, "display", lambda *_: None)
    session = workflow()
    panel = helper.show_guided_workflow(session, display_panel=False)
    panel.controls[setting].value = value
    panel.controls["instrument_confirmed"].value = True
    panel.controls["Preview physical mass"].click()
    assert "calibration" not in session.preview


def test_raw_hdf_no_mass_crop_physical_conversion_export_resume(tmp_path, monkeypatch):
    from pyccapt.calibration.tutorials.tutorials_helpers import helper_guided_workflow as helper
    monkeypatch.setattr(helper, "NotebookJob", ImmediateJob)
    monkeypatch.setattr(helper, "display", lambda *_: None)
    n = 1000
    rng = np.random.default_rng(80)
    x, y = rng.uniform(-1., 1., (2, n))
    voltage = rng.uniform(4000., 6000., n)
    mass = np.full(n, 27.)
    time = 38.+np.sqrt(mass*1.66e-27/(2*1.6e-19*voltage))*np.sqrt(.110**2+(.01*x)**2+(.01*y)**2)*1e9
    raw = tmp_path / "raw.h5"
    with h5py.File(raw, "w") as store:
        for name, values in dict(x=x, y=y, t=time, high_voltage=voltage, voltage_pulse=np.zeros(n),
                                 laser_pulse=np.ones(n), start_counter=np.arange(n, dtype=np.uint64)).items():
            store.create_dataset("dld/"+name, data=values)
    before = raw.read_bytes()
    panel = helper.show_guided_workflow(display_panel=False)
    panel.controls["path"].value = str(raw)
    panel.controls["Import dataset"].click()
    session = panel.controller["workflow"]
    assert session is not None and "mc (Da)" not in session.source
    assert panel.controls["Preview imported mass"].disabled
    panel.controls["Inspect source"].click()
    panel.controls["crop_start"].value = 100
    panel.controls["crop_stop"].value = 800
    panel.controls["Preview crop"].click()
    assert session.active_mask.all()
    panel.controls["Accept crop"].click()
    assert session.active_mask.sum() == 700
    panel.controls["flight"].value = 110.
    panel.controls["t0"].value = 38.
    panel.controls["Preview physical mass"].click()
    assert "calibration" not in session.preview
    panel.controls["instrument_confirmed"].value = True
    panel.controls["Preview physical mass"].click()
    np.testing.assert_allclose(session.preview["calibration"]["mass"][session.active_mask], mass[100:800], rtol=1e-13)
    panel.controls["Accept calibration"].click()
    target = tmp_path / "selected.h5"
    panel.controls["export_path"].value = str(target)
    panel.controls["Export accepted HDF5"].click()
    frame = pd.read_hdf(target, "df")
    assert len(frame) == 700
    np.testing.assert_array_equal(frame.event_id, np.arange(100, 800, dtype=np.uint64))
    restored = GuidedWorkflow.load(target)
    assert restored.n == n and restored.active_mask.sum() == 700
    np.testing.assert_allclose(restored.variables.mc[restored.active_mask], 27.)
    assert raw.read_bytes() == before
    panel.controls["t0"].value = 39.
    assert not panel.controls["instrument_confirmed"].value
