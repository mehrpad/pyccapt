"""State contract and integration tests, using no instrument hardware."""
from __future__ import annotations

import json
import logging
import multiprocessing
import pickle
import time
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from pyccapt.control.core.control_state import (
    STATE_SPECS, CommandStatus, Connection, ControlState, Evidence,
    commanded, complete, evolve, initial_states, observe, publish,
)
from pyccapt.control.core.share_variables import Variables


def make_variables(namespace=None):
    conf = {key: "test" for key in Variables._REQUIRED_CONFIG_KEYS}
    conf.update(save_meta_interval_camera=5, save_meta_interval_visualization=5,
                pulse_amp_per_supply_voltage=1, max_laser_power=5,
                laser="on", stage="on", camera="on", cryo="on", gates="on", tdc="on")
    return Variables(conf, namespace if namespace is not None else SimpleNamespace())


@pytest.fixture
def shared():
    return make_variables()


def transition(state, event, now, **changes):
    return evolve(state, state.owner, event, now=now, **changes)


@pytest.mark.parametrize("name", STATE_SPECS)
def test_every_resource_has_known_owner_and_safe_initial_state(name):
    spec = STATE_SPECS[name]
    state = initial_states({})[name]
    assert state.owner == spec.owner
    assert state.connection == Connection.UNKNOWN
    assert not state.valid and not state.is_fresh(0)
    assert state.observed == "unknown" and state.evidence == Evidence.NONE
    assert f"control_state_{name}" in Variables._OWNERSHIP
    assert pickle.loads(pickle.dumps(state)) == state
    json.dumps(state.to_dict(0), allow_nan=False)


@pytest.mark.parametrize("setting", ["off", "disabled", False])
def test_disabled_configuration_never_claims_a_hardware_state(setting):
    state = initial_states({"laser": setting})["laser"]
    assert state.connection == Connection.DISABLED
    assert not state.valid and state.evidence == Evidence.NONE


def test_observation_and_request_authority_are_separate(shared):
    shared.update_control_state("pump_load_lock", "main", "request", requested="pumping")
    with pytest.raises(ValueError, match="owner pump"):
        shared.update_control_state("pump_load_lock", "main", "observe", observed="pumping", evidence=Evidence.READBACK)
    with pytest.raises(ValueError, match="cannot publish"):
        shared.update_control_state("dc_supply", "main", "request", requested="on")
    with pytest.raises(KeyError):
        shared.control_state("missing_device")
    with pytest.raises(AttributeError, match="update_control_state"):
        shared.control_state_laser = None


def test_request_and_sent_ack_do_not_overwrite_confirmed_observation():
    state = transition(ControlState("laser", "main"), "observe", 10,
                       observed="Listen", evidence=Evidence.READBACK, connection=Connection.CONNECTED)
    state = transition(state, "request", 11, requested="Standby", command_id="a", deadline=20)
    state = transition(state, "result", 12, command_id="a", command_status=CommandStatus.SENT)
    assert state.requested == "Standby" and state.observed == "Listen"
    assert state.observed_at == 10 and state.command_status == CommandStatus.SENT
    assert state.command_overdue(21)
    assert not state.is_fresh(21)
    assert state.command_status == CommandStatus.SENT  # Reading never sends/cancels commands.


def test_command_write_preserves_measured_observation_and_separates_request_metadata(shared):
    observe(shared, "sample_stage", "main", "position", evidence=Evidence.READBACK,
            details={"xyz_m": (1, 2, 3)})
    previous = shared.control_state("sample_stage")
    with commanded(shared, "sample_stage", "main", "home", details={"target_m": (0, 0, 0)}):
        pass
    state = shared.control_state("sample_stage")
    assert state.command_status == CommandStatus.SENT and state.requested == "home"
    assert state.observed == "position" and state.observed_at == previous.observed_at
    assert state.evidence == Evidence.READBACK and state.details["xyz_m"] == (1, 2, 3)
    assert state.request_details["target_m"] == (0, 0, 0)


def test_equal_clock_timestamp_still_requires_an_observation_after_the_request():
    state = transition(ControlState("laser", "main"), "observe", 1, observed="Listen", evidence=Evidence.READBACK)
    state = transition(state, "request", 1, requested="Listen", command_id="a")
    with pytest.raises(ValueError, match="fresh evidence"):
        transition(state, "result", 1, command_id="a", command_status=CommandStatus.CONFIRMED)


def test_reference_callback_is_correlated_to_the_original_command(shared):
    with commanded(shared, "sample_stage", "main", "referencing", confirmation="referenced") as old:
        pass
    with commanded(shared, "sample_stage", "main", "referencing", confirmation="referenced") as current:
        pass
    before = shared.control_state("sample_stage")
    complete(shared, "sample_stage", "main", "referenced", evidence=Evidence.READBACK,
             expected="referencing", command_id=old.command_id)
    assert shared.control_state("sample_stage") == before
    complete(shared, "sample_stage", "main", "referenced", evidence=Evidence.READBACK,
             expected="referencing", command_id=current.command_id)
    assert shared.control_state("sample_stage").command_status == CommandStatus.CONFIRMED


def test_snapshot_checks_clock_after_receiving_concurrent_updates(shared):
    from pyccapt.control.core.state_diagnostics import collect_control_states
    class UpdatingReader:
        def control_states(self):
            shared.stage_position_snapshot = (0., 0., 0., time.monotonic())
            return shared.control_states()
    states = collect_control_states(UpdatingReader())
    assert states["sample_stage"]["fresh"]


def test_superseded_command_completion_or_failure_cannot_change_new_request(shared):
    with commanded(shared, "gate_main", "main", "open"):
        with commanded(shared, "gate_main", "main", "closed"):
            pass
    current = shared.control_state("gate_main")
    assert current.requested == current.observed == "closed"
    with pytest.raises(RuntimeError, match="older operation"):
        with commanded(shared, "gate_main", "main", "open"):
            with commanded(shared, "gate_main", "main", "closed"):
                pass
            raise RuntimeError("older operation failed")
    current = shared.control_state("gate_main")
    assert current.requested == current.observed == "closed"
    assert current.command_status == CommandStatus.SENT and not current.fault


def test_confirmation_requires_matching_fresh_readback_after_request():
    state = transition(ControlState("laser", "main"), "observe", 1, observed="On", evidence=Evidence.READBACK)
    state = transition(state, "request", 2, requested="On", command_id="a")
    with pytest.raises(ValueError, match="fresh evidence"):
        transition(state, "result", 3, command_id="a", command_status=CommandStatus.CONFIRMED)
    state = transition(state, "observe", 3, observed="Listen", evidence=Evidence.READBACK)
    with pytest.raises(ValueError, match="fresh evidence"):
        transition(state, "result", 4, command_id="a", command_status=CommandStatus.CONFIRMED)
    state = transition(state, "observe", 4, observed="On", evidence=Evidence.COMMAND)
    with pytest.raises(ValueError, match="fresh evidence"):
        transition(state, "result", 5, command_id="a", command_status=CommandStatus.CONFIRMED)
    state = transition(state, "observe", 5, observed="On", evidence=Evidence.READBACK)
    with pytest.raises(ValueError, match="fresh evidence"):
        transition(state, "result", 20, command_id="a", command_status=CommandStatus.CONFIRMED)
    state = transition(state, "result", 6, command_id="a", command_status=CommandStatus.CONFIRMED)
    assert state.command_status == CommandStatus.CONFIRMED
    with pytest.raises(ValueError, match="Invalid command transition"):
        transition(state, "result", 7, command_id="a", command_status=CommandStatus.SENT)


def test_late_reply_and_measurement_do_not_complete_new_command_or_clear_fault():
    state = transition(ControlState("laser", "main"), "request", 1, requested="On", command_id="old")
    state = transition(state, "request", 2, requested="Listen", command_id="new")
    assert transition(state, "result", 3, command_id="old", command_status=CommandStatus.CONFIRMED) is state
    state = transition(state, "fault", 4, fault="port lost", fail_command=True)
    assert state.command_status == CommandStatus.FAILED
    assert transition(state, "observe", 3, observed="On", evidence=Evidence.READBACK, fault="") is state
    restored = transition(state, "observe", 5, observed="Listen", evidence=Evidence.READBACK, fault="")
    assert restored.fault == "" and restored.valid
    assert restored.command_status == CommandStatus.FAILED  # Recovery cannot replay a failed command.


@pytest.mark.parametrize("event,changes", [
    ("request", {"requested": ""}),
    ("request", {"requested": "on", "deadline": float("nan")}),
    ("request", {"requested": "on", "deadline": 0}),
    ("observe", {"observed": "on", "evidence": Evidence.NONE}),
    ("observe", {"observed": "on", "evidence": "invented"}),
    ("connection", {"connection": "invented"}),
    ("bogus", {}),
    ("fault", {"fault": "error", "unknown_field": 1}),
])
def test_invalid_events_are_rejected_before_publishing(event, changes):
    with pytest.raises(ValueError):
        transition(ControlState("laser", "main"), event, 1, **changes)


def test_disconnect_invalidates_readback_without_claiming_output_off(shared):
    observe(shared, "laser", "main", "On", evidence=Evidence.READBACK, connection=Connection.CONNECTED)
    shared.flag_laser_connected = False
    state = shared.control_state("laser")
    assert state.observed == "On" and not state.valid and state.connection == Connection.DISCONNECTED


def test_legacy_sensor_error_and_recovery_preserve_original_values(shared):
    shared.vacuum_load_lock = 1e-8
    state = shared.control_state("gauge_load_lock")
    assert state.evidence == Evidence.READBACK and state.valid and state.is_fresh()
    assert not state.is_fresh(state.observed_at+7)
    shared.vacuum_load_lock = -1
    assert shared.vacuum_load_lock == -1
    assert shared.control_state("gauge_load_lock").fault
    shared.vacuum_load_lock = 2e-8
    assert shared.control_state("gauge_load_lock").fault == ""
    shared.temperature = 30
    assert shared.control_state("cryo").details == {"value": 30, "units": "K"}


@pytest.mark.parametrize("prefix,resource", [("alignment", "sample_motion"), ("laser_alignment", "laser_motion")])
def test_native_motion_ids_and_settling_are_preserved(shared, prefix, resource):
    setattr(shared, prefix+"_move_request", {"id": "one", "target_m": (0, 0, 0)})
    setattr(shared, prefix+"_move_status", {"id": "one", "state": "moving"})
    state = shared.control_state(resource)
    assert state.command_status == CommandStatus.SENT
    assert state.deadline is not None
    setattr(shared, prefix+"_move_request", {"id": "two", "target_m": (1, 0, 0)})
    setattr(shared, prefix+"_move_status", {"id": "one", "state": "done"})
    assert shared.control_state(resource).command_status == CommandStatus.REQUESTED
    setattr(shared, prefix+"_move_status", {"id": "two", "state": "done"})
    state = shared.control_state(resource)
    assert state.command_status == CommandStatus.CONFIRMED and state.evidence == Evidence.READBACK


def test_state_details_and_returned_snapshots_are_detached(shared):
    details = {"nested": {"value": 1}}
    state = shared.update_control_state("laser", "main", "observe", observed="Listen",
                                        evidence=Evidence.READBACK, details=details)
    details["nested"]["value"] = 2
    state.details["nested"]["value"] = 3
    shared.control_state_laser.details["nested"]["value"] = 4
    shared.control_states()["laser"].details["nested"]["value"] = 5
    assert shared.control_state("laser").details["nested"]["value"] == 1
    with pytest.raises(FrozenInstanceError):
        state.valid = False


def test_unchanged_numeric_polling_does_not_spam_transition_log(shared, caplog):
    with caplog.at_level(logging.INFO, logger="pyccapt.state"):
        shared.vacuum_main = 1e-8
        shared.vacuum_main = 2e-8
        shared.vacuum_main = 3e-8
    assert len(caplog.records) == 1
    assert shared.control_state("gauge_main").details["value"] == 3e-8


def test_command_wrapper_preserves_calls_return_behavior_and_exact_exception(shared):
    calls = []
    with commanded(shared, "dc_supply", "exp", "output_on"):
        calls.append("F1")
    assert calls == ["F1"]
    state = shared.control_state("dc_supply")
    assert state.command_status == CommandStatus.SENT and state.evidence == Evidence.COMMAND
    error = RuntimeError("serial write failed")
    with pytest.raises(RuntimeError) as caught:
        with commanded(shared, "dc_supply", "exp", "output_off"):
            raise error
    assert caught.value is error
    state = shared.control_state("dc_supply")
    assert state.command_status == CommandStatus.FAILED and state.fault == str(error)
    with commanded(SimpleNamespace(), "dc_supply", "exp", "output_on"):
        calls.append("legacy script still works")


def test_state_publication_failure_cannot_interrupt_existing_operation(shared, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("manager unavailable")
    monkeypatch.setattr(Variables, "update_control_state", fail)
    calls = []
    with commanded(shared, "dc_supply", "exp", "output_off"):
        calls.append("F0")
    shared.start_flag = True
    assert shared.start_flag is True and calls == ["F0"]
    assert publish(shared, "laser", "main", "fault", fault="failure") is None


def _publish_child(shared, device, owner, count):
    for index in range(count):
        shared.update_control_state(device, owner, "observe", observed="active",
                                    evidence=Evidence.SOFTWARE, details={"index": index})


def test_spawned_processes_share_atomic_registry_and_do_not_lose_updates():
    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager:
        shared = make_variables(manager.Namespace())
        # Same-record concurrent updates test atomic read/reduce/write; separate
        # records test that owner publications cannot replace another owner map.
        processes = [context.Process(target=_publish_child, args=(shared, device, owner, 20))
                     for device, owner in [("camera_0", "cam"), ("camera_0", "cam"), ("laser", "main")]]
        try:
            for process in processes:
                process.start()
            for process in processes:
                process.join(15)
                assert process.exitcode == 0
            assert shared.control_state("camera_0").revision == 40
            assert shared.control_state("laser").revision == 20
            assert shared.control_state("dc_supply").revision == 0
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(5)


def test_laser_only_real_readbacks_refresh_state_and_lower_request_supersedes(shared):
    from pyccapt.control.gui.gui_laser_control import Ui_Laser_Control
    from pyccapt.control.nkt_photonics.state import LaserState
    ui = Ui_Laser_Control.__new__(Ui_Laser_Control)
    ui.variables = shared
    ui.laser_device = object()
    ui._apply_button_locks_for_status = lambda status: None
    ui._publish_laser_state("ly_oxp2_dev_status 9")
    before = shared.control_state("laser")
    ui.on_mode = ui.enable_ouput_mode = False
    ui.laser_standby_clicked()
    first = shared.control_state("laser")
    assert first.observed_at == before.observed_at
    assert first.requested == LaserState.STANDBY.value
    ui.laser_listen_clicked()
    second = shared.control_state("laser")
    assert first.command_id != second.command_id and second.requested == LaserState.LISTEN.value
    shared.update_control_state("laser", "main", "result", command_id=second.command_id,
                                command_status=CommandStatus.SENT)
    ui._publish_laser_state("ly_oxp2_dev_status 17")
    assert shared.control_state("laser").command_status == CommandStatus.SENT
    ui._publish_laser_state("ly_oxp2_dev_status 9")
    assert shared.control_state("laser").command_status == CommandStatus.CONFIRMED


def test_pump_commands_and_rpm_readback_keep_the_original_sequence(shared, monkeypatch):
    from pyccapt.control.devices import initialize_devices
    calls = []
    def comm(command):
        calls.append(command)
        return "V911 95" if command == "?V911" else "unchanged pressure payload"
    monkeypatch.setattr(initialize_devices.time, "sleep", lambda value: None)
    shared.flag_pump_load_lock = True
    shared.flag_pump_load_lock_click = True
    reply = initialize_devices.command_edwards(
        {"pump_ll": "on", "COM_PORT_gauge_ll": "COM1", "COM_PORT_gauge_cll": "off"},
        shared, "pressure", SimpleNamespace(comm=comm), status="load_lock")
    assert calls == ["!C910 0", "!C904 0", "?V911", "?V940"]
    assert reply == "unchanged pressure payload" and not shared.flag_pump_load_lock
    state = shared.control_state("pump_load_lock")
    assert state.requested == "stopped" and state.observed == "at_speed"
    assert state.evidence == Evidence.READBACK and state.command_status == CommandStatus.SENT


def test_safe_off_flag_is_not_independent_physical_confirmation(shared):
    shared.hardware_safe = True
    state = shared.control_state("experiment_outputs")
    assert state.observed == "safe_off_reported" and state.evidence == Evidence.SOFTWARE
    shared.physical_estop_ok = True
    assert shared.control_state("safety_interlock").evidence == Evidence.NONE


def test_busy_or_abandoned_state_lock_cannot_trap_existing_stop_commands(shared):
    shared.lock_state.acquire()
    calls = []
    started = time.monotonic()
    try:
        with commanded(shared, "dc_supply", "exp", "output_off"):
            calls.append("F0")
        shared.stop_flag = True
        with pytest.raises(TimeoutError):
            shared.control_states()
        with pytest.raises(TimeoutError):
            shared.control_state_laser
    finally:
        shared.lock_state.release()
    assert calls == ["F0"] and shared.stop_flag is True
    assert time.monotonic()-started < 0.5


def test_final_diagnostics_are_strict_json_and_do_not_create_another_dataset(shared, tmp_path):
    from pyccapt.control.core.state_diagnostics import save_control_states
    shared.experiment_state = "complete"
    shared.laser_telemetry = {"valid": False, "error": "unsupported DUV monitor", "output_power_mw": float("nan")}
    assert save_control_states(shared, None) is None
    assert save_control_states(shared, tmp_path / "missing") is None
    assert not (tmp_path / "missing").exists()
    target = save_control_states(shared, tmp_path)
    document = json.loads(target.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert document["states"]["experiment"]["observed"] == "complete"
    optics = document["states"]["laser_optics"]
    assert optics["details"]["output_power_mw"] is None and not optics["valid"]
    assert optics["fault"] == "unsupported DUV monitor"
    assert list(tmp_path.glob("*.tmp")) == []


def test_readback_recovery_preserves_failed_command_diagnostics(shared):
    with pytest.raises(RuntimeError):
        with commanded(shared, "laser_stage", "main", "referencing", confirmation="referenced"):
            raise RuntimeError("reference sensor absent")
    shared.laser_stage_snapshot = (0., 0., 0., time.monotonic())
    state = shared.control_state("laser_stage")
    assert state.valid and state.fault == ""
    assert state.command_status == CommandStatus.FAILED and state.command_error == "reference sensor absent"


def test_health_consumes_common_states_without_changing_existing_summary(shared):
    from pyccapt.control.core.health import build_health_snapshot
    shared.experiment_state = "running"
    shared.hardware_safe = False
    shared.vacuum_buffer = -1
    health = build_health_snapshot(shared, SimpleNamespace(backend=None), [], heartbeat_monotonic=0)
    assert "UNSAFE" in health.summary()
    states = health.metrics()["control_states"]
    assert states["experiment"]["observed"] == "running"
    assert states["gauge_buffer"]["fault"] and not states["gauge_buffer"]["fresh"]


def test_experiment_and_alignment_requests_track_existing_software_phases(shared):
    shared.start_flag = True
    shared.experiment_state = "initializing"
    assert shared.control_state("experiment").command_status == CommandStatus.REQUESTED
    shared.experiment_state = "running"
    assert shared.control_state("experiment").command_status == CommandStatus.CONFIRMED
    shared.stop_flag = True
    shared.experiment_state = "stopping"
    shared.experiment_state = "complete"
    assert shared.control_state("experiment").command_status == CommandStatus.CONFIRMED
    shared.automatic_alignment_enabled = True
    shared.alignment_status = {"phase": "aligned", "sample": 1}
    assert shared.control_state("sample_alignment").command_status == CommandStatus.CONFIRMED
    shared.automatic_alignment_enabled = True
    shared.alignment_outcome = "attempt_limit"
    shared.alignment_status = {"phase": "finished", "reason": "No sample after retries"}
    state = shared.control_state("sample_alignment")
    assert state.command_status == CommandStatus.FAILED and state.fault == "No sample after retries"


def test_laser_alignment_gui_queue_status_does_not_claim_worker_progress(shared):
    shared.laser_alignment_status = {"phase": "pending", "active": True}
    assert shared.control_state("laser_alignment").observed == "unknown"
    shared.laser_alignment_command = {"id": "command", "mode": "coarse"}
    assert shared.control_state("laser_alignment").command_status == CommandStatus.REQUESTED
    shared.laser_alignment_status = {"phase": "coarse", "session": "actual-session", "active": True}
    assert shared.control_state("laser_alignment").command_status == CommandStatus.CONFIRMED
    shared.laser_alignment_status = {"phase": "cancelled", "active": False}
    assert shared.control_state("laser_alignment").observed == "coarse"
    shared.laser_alignment_command = {"id": "stop", "mode": "stop"}
    shared.laser_alignment_status = {"phase": "stopped", "session": "actual-session", "active": False}
    assert shared.control_state("laser_alignment").command_status == CommandStatus.CONFIRMED
