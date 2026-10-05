"""Compatibility adapters for existing shared-field publication sites.

Only named status/request fields are adapted; acquisition arrays and per-event
counters never enter the registry. No adapter changes the legacy field value.
"""
from __future__ import annotations

import math
import time
from functools import partial

from pyccapt.control.core.control_state import CommandStatus, Connection, Evidence, observe_sensor


STATUS_FIELDS = {
    "experiment_state", "experiment_error", "alignment_status", "laser_alignment_status",
    "alignment_move_request", "alignment_move_status", "laser_alignment_move_request",
    "laser_alignment_move_status", "stage_position_snapshot", "laser_stage_snapshot",
    "laser_telemetry", "flag_laser_connected", "start_flag", "stop_flag",
    "set_temperature_flag_cryo", "set_temperature_flag_ll", "temperature",
    "flag_pump_load_lock_click", "flag_pump_cryo_load_lock_click",
    "flag_vent_cryo_load_lock_partial",
    "laser_alignment_command", "automatic_alignment_enabled", "alignment_outcome", "hardware_safe",
    *{f"vacuum_{name}" for name in ("main", "buffer", "buffer_backing", "load_lock",
                                    "load_lock_backing", "cryo_load_lock", "cryo_load_lock_backing")},
}


def adapt(variables, field, value):
    """Called after releasing the legacy field lock, using the state lock only."""
    update = partial(variables.update_control_state, lock_timeout_s=0.01)

    def observed(device, owner, label, *, evidence=Evidence.SOFTWARE, valid=True,
                 fault="", **details):
        return update(device, owner, "observe", observed=label, evidence=evidence,
                      valid=valid, fault=fault, **details)

    if field in {"start_flag", "stop_flag"}:
        if value:
            update("experiment", "main", "request", requested="running" if field == "start_flag" else "stopped",
                   confirmation="running" if field == "start_flag" else "complete")
    elif field == "experiment_state":
        state = observed("experiment", "exp", str(value), valid=value != "failed",
                         fault=str(getattr(variables, "experiment_error", "")) if value == "failed" else "")
        if state.command_status in {CommandStatus.REQUESTED, CommandStatus.SENT}:
            if value == state.confirmation:
                update("experiment", "exp", "result", command_id=state.command_id,
                       command_status=CommandStatus.CONFIRMED)
            elif value == "failed":
                update("experiment", "exp", "result", command_id=state.command_id,
                       command_status=CommandStatus.FAILED)
    elif field == "experiment_error":
        update("experiment", "exp", "fault", fault=str(value)) if value else None
    elif field in {"alignment_status", "laser_alignment_status"}:
        if value:
            device = "sample_alignment" if field == "alignment_status" else "laser_alignment"
            phase = str(value.get("phase", "unknown"))
            if device == "laser_alignment" and phase in {"pending", "cancelled"} and "session" not in value:
                return  # GUI queue indicators are requests, not worker observations.
            outcome = str(getattr(variables, "alignment_outcome", "")) if device == "sample_alignment" else ""
            failed = phase in {"error", "failed"} or phase == "finished" and outcome not in {"", "aligned"}
            state = observed(device, "exp", phase, valid=not failed,
                             fault=str(value.get("error", value.get("reason", outcome))) if failed else "",
                             details=value)
            if state.command_status in {CommandStatus.REQUESTED, CommandStatus.SENT}:
                if phase == state.confirmation and not failed:
                    update(device, "exp", "result", command_id=state.command_id,
                           command_status=CommandStatus.CONFIRMED)
                elif failed:
                    update(device, "exp", "result", command_id=state.command_id,
                           command_status=CommandStatus.FAILED)
    elif field in {"alignment_move_request", "laser_alignment_move_request"}:
        if value and value.get("id"):
            device = "sample_motion" if field == "alignment_move_request" else "laser_motion"
            settings = (getattr(variables, "alignment_settings", {}) if device == "sample_motion"
                        else getattr(variables, "laser_alignment_run", {}).get("settings", {}))
            source = "main" if value.get("kind") == "transfer" else "exp"
            duration = float(settings.get("move_timeout_s", 120 if device == "sample_motion" else 60))
            update(device, source, "request", requested="done", command_id=str(value["id"]),
                   deadline=time.monotonic()+duration+float(settings.get("settle_s", 0))+2, details=value)
    elif field in {"alignment_move_status", "laser_alignment_move_status"}:
        if not value or not value.get("id"):
            return
        device = "sample_motion" if field == "alignment_move_status" else "laser_motion"
        state = variables.control_state(device)
        if state.command_id != str(value["id"]):
            return  # A delayed reply belongs to an older motion request.
        label = str(value.get("state", "unknown"))
        error = str(value.get("error", ""))
        observed(device, "main", label, evidence=Evidence.READBACK if label == "done" else Evidence.SOFTWARE,
                 valid=not error, fault=error, details=value, command_id=str(value["id"]))
        status = {"moving": CommandStatus.SENT, "done": CommandStatus.CONFIRMED,
                  "error": CommandStatus.FAILED}.get(label)
        if status is not None:
            update(device, "main", "result", command_id=str(value["id"]), command_status=status)
    elif field in {"stage_position_snapshot", "laser_stage_snapshot"}:
        device = "sample_stage" if field == "stage_position_snapshot" else "laser_stage"
        if len(value) == 4 and all(math.isfinite(float(x)) for x in value):
            observed(device, "main", "position", evidence=Evidence.READBACK,
                     valid=float(value[3]) > 0, now=float(value[3]),
                     connection=Connection.CONNECTED if float(value[3]) > 0 else Connection.UNKNOWN,
                     details={"xyz_m": tuple(value[:3])})
        else:
            update(device, "main", "fault", fault="Invalid XYZ position snapshot")
    elif field == "flag_laser_connected":
        update("laser", "main", "connection",
               connection=Connection.CONNECTED if value else Connection.DISCONNECTED)
    elif field == "laser_telemetry":
        if value:
            observed("laser_optics", "main", "available" if value.get("valid") else "unavailable",
                     evidence=Evidence.READBACK, valid=bool(value.get("valid")),
                     fault=str(value.get("error", "")), now=float(value.get("monotonic", time.monotonic())),
                     details=value)
    elif field.startswith("vacuum_") or field == "temperature":
        device = "cryo" if field == "temperature" else "gauge_"+field.removeprefix("vacuum_")
        observe_sensor(variables, device, "pump", value, "K" if field == "temperature" else "mbar")
    elif field in {"set_temperature_flag_cryo", "set_temperature_flag_ll"}:
        if value is not None:
            device = "heater_cryo" if field.endswith("cryo") else "heater_load_lock"
            update(device, "main", "request", requested="regulating" if value else "off")
    elif field.endswith("_click") and value:
        device = "pump_cryo_load_lock" if "cryo" in field else "pump_load_lock"
        update(device, "main", "request", requested="toggle")
    elif field == "flag_vent_cryo_load_lock_partial":
        observed("vent_cryo", "main", "vent_requested" if value else "pumping_requested", evidence=Evidence.COMMAND)
    elif field == "laser_alignment_command" and value:
        update("laser_alignment", "main", "request", requested=str(value.get("mode", "alignment")),
               command_id=str(value.get("id", "")),
               confirmation="stopped" if value.get("mode") == "stop" else str(value.get("mode", "alignment")),
               details=value)
    elif field == "automatic_alignment_enabled":
        if value:
            update("sample_alignment", "main", "request", requested="align", confirmation="aligned")
    elif field == "alignment_outcome" and value:
        current = variables.control_state("sample_alignment")
        if value not in {"aligned", ""}:
            update("sample_alignment", "exp", "fault", fault=str(value))
            if current.command_status in {CommandStatus.REQUESTED, CommandStatus.SENT}:
                update("sample_alignment", "exp", "result", command_id=current.command_id,
                       command_status=CommandStatus.CANCELLED if value == "cancelled" else CommandStatus.FAILED)
    elif field == "hardware_safe":
        observed("experiment_outputs", "exp", "safe_off_reported" if value else "outputs_not_safe_off",
                 evidence=Evidence.SOFTWARE, details={"hardware_safe_flag": bool(value)})
