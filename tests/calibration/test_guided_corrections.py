"""Original correction tools embedded in the guided preview/accept boundary."""
import numpy as np
import pytest

from pyccapt.calibration.core.calibration_metadata import record_step
from pyccapt.calibration.core.exceptions import CalibrationInputError, CalibrationStateError
from pyccapt.calibration.core.guided_corrections import CorrectionWorkspace, METHOD
from pyccapt.calibration.core.guided_workflow import GuidedWorkflow
from pyccapt.calibration.core.share_variables import Variables
from pyccapt.calibration.tutorials.tutorials_helpers import helper_calibration, helper_guided_workflow as helper
from tests.calibration.test_guided_notebook_ui import ImmediateJob
from tests.calibration.test_guided_workflow_review import workflow


def raw_session():
    frame = workflow().source.drop(columns=["mc (Da)", "mc_uc (Da)"]).copy()
    frame["pulse_v (V)"] = 600.
    variables = Variables()
    variables.sync_from_data(frame)
    session = GuidedWorkflow(variables)
    session.preview_crop(event_start=13, event_stop=3600, detector_circle=(0., 0., .18))
    session.accept("crop")
    return session


def workspace(session):
    session.preview_calibration(np.full(session.n, 27.), receipt={"method": "test initial mass"})
    return CorrectionWorkspace(session, flight_length_mm=110., t0_ns=38., pulse_mode="voltage")


def test_correction_scatter_preserves_crop_ids_and_acceptance(tmp_path):
    session = raw_session()
    source = session.source.copy(deep=True)
    trial = workspace(session)
    assert len(trial.indices) < session.n
    np.testing.assert_array_equal(trial.variables.event_ids, session.event_ids[session.active_mask])
    expected = np.linspace(26.9, 27.1, len(trial.indices))
    trial.variables.mc_calib = expected.copy()
    record_step(trial.variables, "test_correction", parameters={"factor": 1.01}, status="working")
    trial.preview(session, (110., 38., "voltage"))
    assert session.accepted["calibration"] is None
    np.testing.assert_array_equal(session.source.to_numpy(), source.to_numpy())
    preview = session.preview["calibration"]
    np.testing.assert_allclose(preview["mass"][trial.indices], expected)
    assert np.isnan(preview["mass"][~session.active_mask]).all()
    assert preview["tof"] is None
    assert preview["receipt"]["working_history"]["steps"][-1]["operation"] == "test_correction"
    session.accept("calibration")
    target = tmp_path / "correction.h5"
    session.save(target)
    restored = GuidedWorkflow.load(target)
    np.testing.assert_allclose(restored.accepted["calibration"]["mass"], preview["mass"], equal_nan=True)
    assert restored.accepted["calibration"]["receipt"]["method"] == METHOD
    with pytest.raises(CalibrationStateError, match="changed"):
        trial.preview(session, (110., 38., "voltage"))


def test_tof_is_independent_and_invalid_measurements_keep_initial_mass():
    session = raw_session()
    # Build a fresh immutable source with one missing voltage measurement.
    frame = session.source.copy()
    index = np.flatnonzero(session.active_mask)[10]
    frame.loc[index, "high_voltage (V)"] = np.nan
    variables = Variables()
    variables.sync_from_data(frame)
    session = GuidedWorkflow(variables)
    trial = workspace(session)
    trial.variables.dld_t_calib *= .99
    trial.preview(session, (110., 38., "voltage"))
    preview = session.preview["calibration"]
    np.testing.assert_allclose(preview["mass"], 27.)
    assert np.isnan(preview["tof"][index])
    assert preview["receipt"]["selected_events_not_corrected"] == 1
    assert not preview["receipt"]["mass_changed"] and preview["receipt"]["tof_changed"]


@pytest.mark.parametrize("failure", ["settings", "order", "float_ids", "negative", "shape", "infinity"])
def test_wrong_context_or_bad_outputs_cannot_be_captured(failure):
    session = raw_session()
    trial = workspace(session)
    settings = (110., 38., "voltage")
    if failure == "settings":
        settings = (100., 38., "voltage")
    elif failure == "order":
        trial.variables.event_ids = trial.variables.event_ids[::-1]
    elif failure == "float_ids":
        trial.variables.data["event_id"] = trial.variables.data["event_id"].to_numpy(dtype=float)
    elif failure == "negative":
        trial.variables.mc_calib[0] = -1
    elif failure == "infinity":
        trial.variables.dld_t_calib[0] = np.inf
    else:
        trial.variables.mc_calib = trial.variables.mc_calib[:-1]
    before = session.preview["calibration"]
    with pytest.raises((CalibrationInputError, CalibrationStateError)):
        trial.preview(session, settings)
    assert session.preview["calibration"] is before


def test_start_button_opens_real_original_tools_without_fitting(monkeypatch):
    monkeypatch.setattr(helper, "NotebookJob", ImmediateJob)
    monkeypatch.setattr(helper, "display", lambda *_: None)
    monkeypatch.setattr(helper_calibration.calibration, "bowl_correction_main", lambda *_a, **_k: pytest.fail("No fit on opening"))
    session = raw_session()
    panel = helper.show_guided_workflow(session, display_panel=False)
    controls = panel.controls
    assert controls["Start calibration"].disabled
    assert controls["Compare calibration"].disabled and controls["Show mass comparison"].disabled
    controls["flight"].value, controls["t0"].value = 110., 38.
    controls["pulse_mode"].value = "voltage"
    controls["instrument_confirmed"].value = True
    controls["Start calibration"].click()
    trial = panel.controller["standard_workspace"]
    assert trial is not None
    assert controls["calibration_tabs"].selected_index == 1
    assert not controls["Compare calibration"].disabled
    assert not controls["Preview corrected calibration"].disabled
    assert session.accepted["calibration"] is None
    initial = session.preview["calibration"]["mass"].copy()
    trial.variables.mc_calib *= 1.001
    controls["Preview corrected calibration"].click()
    np.testing.assert_allclose(session.preview["calibration"]["mass"][trial.indices], initial[trial.indices]*1.001)
    controls["Accept calibration"].click()
    assert session.accepted["calibration"]["receipt"]["method"] == METHOD
    assert not controls["Open peak-review queue"].disabled


def test_setting_edits_invalidate_tools_and_correction_preview(monkeypatch):
    monkeypatch.setattr(helper, "NotebookJob", ImmediateJob)
    monkeypatch.setattr(helper, "display", lambda *_: None)
    panel = helper.show_guided_workflow(raw_session(), display_panel=False)
    c = panel.controls
    c["instrument_confirmed"].value = True
    c["Start calibration"].click()
    c["Preview corrected calibration"].click()
    assert panel.controller["workflow"].preview["calibration"]["receipt"]["method"] == METHOD
    tools = panel.controller["standard_panel"]
    c["detector_diameter"].value = 75.
    assert panel.controller["standard_workspace"] is None
    assert tools.comm is None and tools.controls["auto"].comm is None
    assert "calibration" not in panel.controller["workflow"].preview
    assert c["Accept calibration"].disabled and c["Compare calibration"].disabled


def test_embedded_original_tools_clear_only_their_own_outputs(monkeypatch):
    calls = []
    monkeypatch.setattr(helper_calibration, "clear_plot_on_click", lambda *_a, **kw: calls.append(kw))
    monkeypatch.setattr(helper_calibration.mc_plot, "hist_plot", lambda *_a, **_kw: pytest.fail("No automatic plot while embedding"))
    trial = workspace(raw_session())
    panel = helper_calibration.call_voltage_bowl_calibration(trial.variables, 80., 110., "voltage", 38., display_panel=False)
    panel.controls["clear"].click()
    assert calls == [{"clear_cell": False}]
    assert len(panel.children) == 2


def test_embedded_plot_keeps_parent_cell_and_workspace_arrays(monkeypatch):
    calls = []
    monkeypatch.setattr(helper_calibration, "clear_output", lambda **_: calls.append("parent cleared"))
    monkeypatch.setattr(helper_calibration.mc_plot, "hist_plot", lambda *_a, **_kw: calls.append("histogram"))
    trial = workspace(raw_session())
    initial = trial.variables.mc_calib.copy()
    panel = helper_calibration.call_voltage_bowl_calibration(trial.variables, 80., 110., "voltage", display_panel=False)
    assert not calls
    panel.controls["plot"].click()
    assert calls == ["histogram"]
    np.testing.assert_array_equal(trial.variables.mc_calib, initial)


def test_resumed_corrected_preview_restores_settings_and_requires_confirmation(monkeypatch):
    monkeypatch.setattr(helper, "display", lambda *_: None)
    session = raw_session()
    session.preview_calibration(np.full(session.n, 27.))
    trial = CorrectionWorkspace(session, flight_length_mm=110., t0_ns=38., pulse_mode="voltage", detector_diameter_mm=75.)
    trial.preview(session, (110., 38., "voltage"))
    resumed = GuidedWorkflow.from_state(*session.to_state())
    panel = helper.show_guided_workflow(resumed, display_panel=False)
    c = panel.controls
    assert (c["flight"].value, c["t0"].value, c["pulse_mode"].value) == (110., 38., "voltage")
    assert c["detector_diameter"].value == 75.
    assert not c["instrument_confirmed"].value and c["Accept calibration"].disabled
    c["instrument_confirmed"].value = True
    c["Accept calibration"].click()
    assert resumed.accepted["calibration"]["receipt"]["method"] == METHOD


def test_interactive_spectra_enable_notebook_backend_only_on_explicit_open(monkeypatch):
    import IPython
    import matplotlib
    from types import SimpleNamespace
    from pyccapt.calibration.tutorials.tutorials_helpers.helper_guided_startup import enable_calibration_widgets
    calls = []
    shell = SimpleNamespace(kernel=object(), run_line_magic=lambda *args: calls.append(args))
    monkeypatch.setattr(IPython, "get_ipython", lambda: shell)
    monkeypatch.setattr(matplotlib, "get_backend", lambda: "inline")
    enable_calibration_widgets()
    assert calls == [("matplotlib", "ipympl")]
    monkeypatch.setattr(matplotlib, "get_backend", lambda: "module://ipympl.backend_nbagg")
    enable_calibration_widgets()
    assert len(calls) == 1
    monkeypatch.setattr(IPython, "get_ipython", lambda: None)
    monkeypatch.setattr(matplotlib, "get_backend", lambda: "Agg")
    enable_calibration_widgets()
    assert len(calls) == 1


def test_unavailable_interactive_backend_explains_recovery(monkeypatch):
    import IPython
    import matplotlib
    from types import SimpleNamespace
    from pyccapt.calibration.tutorials.tutorials_helpers.helper_guided_startup import enable_calibration_widgets
    def missing(*_):
        raise ImportError("ipympl")
    monkeypatch.setattr(IPython, "get_ipython", lambda: SimpleNamespace(kernel=object(), run_line_magic=missing))
    monkeypatch.setattr(matplotlib, "get_backend", lambda: "inline")
    with pytest.raises(CalibrationStateError, match="ipympl"):
        enable_calibration_widgets()
