"""Common state contract. This module performs no I/O or hardware actions.

Owners publish observations; clients publish requests. A command being sent is
never evidence that the physical device reached the requested state. Legacy
interlocks and scheduling remain in their existing owners.
"""
from __future__ import annotations

import logging
import math
import time
import uuid
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any


class Connection(str, Enum):
    UNKNOWN = "unknown"
    DISABLED = "disabled"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"


class Evidence(str, Enum):
    NONE = "none"
    COMMAND = "commanded"
    READBACK = "readback"
    SOFTWARE = "software"


class CommandStatus(str, Enum):
    NONE = "none"
    REQUESTED = "requested"
    SENT = "sent"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True)
class StateSpec:
    owner: str
    requesters: tuple[str, ...] = ("main",)
    config_key: str = ""
    max_age_s: float | None = None


# Exactly one observation owner per resource; requests may have other writers.
# "pump" is the existing vacuum polling owner (a thread in the main process).
STATE_SPECS = {
    "experiment": StateSpec("exp", ("main", "exp")),
    "experiment_outputs": StateSpec("exp", ("exp",)),
    "detector": StateSpec("exp", ("exp",), "tdc"),
    "sample_alignment": StateSpec("exp", ("main", "exp")),
    "laser_alignment": StateSpec("exp", ("main", "exp")),
    "sample_motion": StateSpec("main", ("main", "exp"), max_age_s=3),
    "laser_motion": StateSpec("main", ("main", "exp"), max_age_s=3),
    "sample_stage": StateSpec("main", config_key="stage", max_age_s=3),
    "laser_stage": StateSpec("main", config_key="stage", max_age_s=3),
    "laser": StateSpec("main", config_key="laser", max_age_s=6),
    "laser_optics": StateSpec("main", config_key="laser", max_age_s=6),
    "gate_main": StateSpec("main", config_key="gates"),
    "gate_load": StateSpec("main", config_key="gates"),
    "gate_cryo": StateSpec("main", config_key="gates"),
    "pump_load_lock": StateSpec("pump", config_key="pump_ll", max_age_s=6),
    "pump_cryo_load_lock": StateSpec("pump", config_key="pump_cll", max_age_s=6),
    "cryo": StateSpec("pump", config_key="cryo", max_age_s=6),
    "heater_cryo": StateSpec("pump", config_key="cryo"),
    "heater_load_lock": StateSpec("pump", config_key="cryo"),
    "camera_0": StateSpec("cam", config_key="camera", max_age_s=6),
    "camera_1": StateSpec("cam", config_key="camera", max_age_s=6),
    "camera_2": StateSpec("cam", config_key="camera", max_age_s=6),
    "illumination": StateSpec("cam", config_key="camera_illumination"),
    "illumination_settings": StateSpec("cam", config_key="camera_illumination"),
    "vent_cryo": StateSpec("main", config_key="pump_cll"),
    "valve_cll_backing": StateSpec("main", config_key="pump_cll"),
    "valve_cll_turbo": StateSpec("main", config_key="pump_cll"),
    "valve_cll_vent": StateSpec("main", config_key="pump_cll"),
    "dc_supply": StateSpec("exp", ("exp",), "v_dc"),
    "pulse_supply": StateSpec("exp", ("exp",), "v_p"),
    "signal_generator": StateSpec("exp", ("exp",), "signal_generator"),
    "signal_generator_settings": StateSpec("exp", ("exp",), "signal_generator"),
    "safety_interlock": StateSpec("exp", ("exp",), max_age_s=3),
    "baking": StateSpec("main", config_key="baking"),
    "baking_temperature_daq": StateSpec("main", config_key="baking", max_age_s=3),
    "load_lock_baking": StateSpec("main", config_key="cryo"),
    "visualization": StateSpec("viz"),
    **{f"gauge_{name}": StateSpec("pump", config_key="gauges", max_age_s=6)
       for name in ("main", "buffer", "buffer_backing", "load_lock", "load_lock_backing",
                    "cryo_load_lock", "cryo_load_lock_backing")},
}


@dataclass(frozen=True)
class ControlState:
    device: str
    owner: str
    connection: Connection = Connection.UNKNOWN
    requested: str = ""
    observed: str = "unknown"
    evidence: Evidence = Evidence.NONE
    valid: bool = False
    observed_at: float = 0.0
    updated_at: float = 0.0
    revision: int = 0
    fault: str = ""
    command_id: str = ""
    command_status: CommandStatus = CommandStatus.NONE
    command_error: str = ""
    deadline: float | None = None
    requested_at: float = 0.0
    requested_revision: int = 0
    observed_revision: int = 0
    confirmation: str = ""
    details: Any = None
    request_details: Any = None

    def is_fresh(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        age = now - self.observed_at
        limit = STATE_SPECS[self.device].max_age_s
        return self.valid and age >= 0 and (limit is None or age <= limit)

    def command_overdue(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return (self.command_status in {CommandStatus.REQUESTED, CommandStatus.SENT}
                and self.deadline is not None and now > self.deadline)

    def to_dict(self, now: float | None = None) -> dict:
        result = asdict(self)
        result.update(fresh=self.is_fresh(now), command_overdue=self.command_overdue(now))
        return result


def initial_states(conf: dict) -> dict[str, ControlState]:
    states = {}
    for device, spec in STATE_SPECS.items():
        value = str(conf.get(spec.config_key, "")).strip().lower()
        disabled = bool(spec.config_key) and value in {"off", "disabled", "false"}
        states[device] = ControlState(device, spec.owner,
                                     connection=Connection.DISABLED if disabled else Connection.UNKNOWN)
    return states


def evolve(state: ControlState, source: str, event: str, *, now: float | None = None,
           **changes) -> ControlState:
    """Pure, validated reducer. Late replies cannot finish a newer command."""
    now = time.monotonic() if now is None else float(now)
    if not math.isfinite(now) or now < 0:
        raise ValueError("State timestamps must be finite monotonic seconds")
    spec = STATE_SPECS[state.device]
    if source != spec.owner and not (event == "request" and source in spec.requesters):
        raise ValueError(f"{source} cannot publish {event} for {state.device} (owner {spec.owner})")
    if event in {"observe", "fault"}:
        command_id = changes.pop("command_id", None)
        if command_id is not None and command_id != state.command_id:
            return state
    updates = {"updated_at": max(now, state.updated_at), "revision": state.revision + 1}
    if event == "request":
        requested = str(changes.pop("requested"))
        if not requested:
            raise ValueError("A command request needs a named target/action")
        command_id = str(changes.pop("command_id", "") or uuid.uuid4().hex)
        deadline = changes.pop("deadline", None)
        if deadline is not None and (not math.isfinite(deadline) or deadline < now):
            raise ValueError("Command deadline must be finite and not already expired")
        updates.update(requested=requested, command_id=command_id,
                       command_status=CommandStatus.REQUESTED, deadline=deadline, requested_at=now,
                       requested_revision=updates["revision"],
                       confirmation=str(changes.pop("confirmation", requested)), command_error="")
        updates["request_details"] = deepcopy(changes.pop("details", None))
    elif event == "result":
        if changes.pop("command_id") != state.command_id:
            return state
        status = CommandStatus(changes.pop("command_status"))
        allowed = {
            CommandStatus.REQUESTED: {CommandStatus.SENT, CommandStatus.CONFIRMED, CommandStatus.FAILED,
                                      CommandStatus.CANCELLED, CommandStatus.TIMED_OUT},
            CommandStatus.SENT: {CommandStatus.CONFIRMED, CommandStatus.FAILED,
                                 CommandStatus.CANCELLED, CommandStatus.TIMED_OUT},
        }
        if status != state.command_status and status not in allowed.get(state.command_status, set()):
            raise ValueError(f"Invalid command transition {state.command_status.value} -> {status.value}")
        updates["command_status"] = status
        if status in {CommandStatus.FAILED, CommandStatus.CANCELLED, CommandStatus.TIMED_OUT}:
            updates["command_error"] = str(changes.pop("command_error", state.fault or status.value))
        # Confirmation is explicitly a software outcome or hardware readback.
        if status == CommandStatus.CONFIRMED and (
                state.evidence not in {Evidence.READBACK, Evidence.SOFTWARE}
                or not state.is_fresh(now) or state.observed_at < state.requested_at or state.fault
                or state.observed != state.confirmation or state.observed_revision <= state.requested_revision):
            raise ValueError("Confirmation requires fresh evidence following the request")
    elif event == "observe":
        if now < state.updated_at:
            return state  # Do not let a delayed measurement clear a newer fault/request.
        if (Evidence(changes.get("evidence")) == Evidence.COMMAND
                and state.evidence in {Evidence.READBACK, Evidence.SOFTWARE}):
            return state  # Writes do not overwrite the last measured/software observation.
        updates.update(observed=str(changes.pop("observed")),
                       evidence=Evidence(changes.pop("evidence")),
                       valid=bool(changes.pop("valid", True)), observed_at=now,
                       observed_revision=updates["revision"])
        if updates["evidence"] == Evidence.NONE and updates["valid"]:
            raise ValueError("A valid observation requires evidence")
    elif event == "connection":
        updates["connection"] = Connection(changes.pop("connection"))
        if updates["connection"] != Connection.CONNECTED:
            updates["valid"] = False
    elif event == "fault":
        updates.update(fault=str(changes.pop("fault")), valid=False)
        if changes.pop("fail_command", False) and state.command_status in {CommandStatus.REQUESTED, CommandStatus.SENT}:
            updates["command_status"] = CommandStatus.FAILED
            updates["command_error"] = updates["fault"]
    else:
        raise ValueError(f"Unknown state event {event!r}")
    for key in ("fault", "details", "connection"):
        if key in changes:
            updates[key] = Connection(changes.pop(key)) if key == "connection" else changes.pop(key)
    if changes:
        raise ValueError(f"Unexpected state fields: {', '.join(changes)}")
    if "details" in updates:
        updates["details"] = deepcopy(updates["details"])
    return replace(state, **updates)


def publish(variables, device: str, source: str, event: str, **changes) -> ControlState | None:
    """Best-effort telemetry boundary: never disrupt a legacy hardware operation.

    The Variables implementation locks the shared transaction. Test doubles and
    standalone device scripts without that API simply omit publication.
    """
    updater = getattr(variables, "update_control_state", None)
    if not callable(updater):
        return None
    try:
        return updater(device, source, event, lock_timeout_s=0.01, **changes)
    except Exception:
        logging.getLogger("pyccapt.state").exception("Could not publish %s state event %s", device, event)
        return None


def observe(variables, device, source, observed, *, evidence=Evidence.SOFTWARE,
            valid=True, fault="", **changes):
    return publish(variables, device, source, "observe", observed=observed,
                   evidence=evidence, valid=valid, fault=fault, **changes)


def request(variables, device, source, requested, **changes):
    return publish(variables, device, source, "request", requested=requested, **changes)


def observe_sensor(variables, device, source, value, units):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = math.nan
    valid = math.isfinite(numeric) and numeric > 0
    return observe(variables, device, source, "measuring" if valid else "unavailable",
                   evidence=Evidence.READBACK, valid=valid,
                   fault="" if valid else "No valid sensor readback",
                   connection=Connection.CONNECTED if valid else Connection.UNKNOWN,
                   details={"value": numeric, "units": units})


def result(variables, device, source, command_id, status, **changes):
    return publish(variables, device, source, "result", command_id=command_id,
                   command_status=status, **changes)


def complete(variables, device, source, observed, *, evidence=Evidence.SOFTWARE, expected=None, command_id=None):
    state = observe(variables, device, source, observed, evidence=evidence, command_id=command_id)
    if (state is not None and state.command_status in {CommandStatus.REQUESTED, CommandStatus.SENT}
            and command_id is not None and state.command_id == command_id
            and (expected is None or state.requested == expected)):
        result(variables, device, source, state.command_id, CommandStatus.CONFIRMED)


@contextmanager
def commanded(variables, device, source, action, *, details=None, confirmation=None):
    """Record an existing synchronous command without changing its execution.

    Successful writes remain SENT/commanded, never hardware-confirmed. The
    original exception propagates unchanged; state publication is best effort.
    """
    state = request(variables, device, source, action, details=details,
                    confirmation=action if confirmation is None else confirmation)
    try:
        yield state
    except BaseException as exc:
        publish(variables, device, source, "fault", fault=str(exc),
                command_id=state.command_id if state is not None else None)
        if state is not None:
            result(variables, device, source, state.command_id, CommandStatus.FAILED)
        raise
    else:
        observe(variables, device, source, action, evidence=Evidence.COMMAND, details=details,
                command_id=state.command_id if state is not None else None)
        if state is not None:
            result(variables, device, source, state.command_id, CommandStatus.SENT)
