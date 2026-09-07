"""Typed cross-process contracts for the control runtime.

These objects are immutable and pickle-friendly. High-rate samples use
shared-memory rings/chunk streams; queues carry only low-rate control data.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class CommandKind(str, Enum):
    STOP = "stop"
    EMERGENCY_STOP = "emergency_stop"
    HEARTBEAT = "heartbeat"


class StatusKind(str, Enum):
    STATE = "state"
    HEALTH = "health"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class RunConfig:
    """Validated snapshot consumed by one experiment worker."""

    ex_freq: float
    ex_time: float
    max_ions: int
    vdc_min: float
    vdc_max: float
    v_p_min: float
    v_p_max: float
    pulse_fraction: float
    pulse_frequency: float
    detection_rate: float
    pulse_amp_per_supply_voltage: float
    counter_source: str
    pulse_mode: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControlCommand:
    kind: CommandKind
    reason: str = ""
    issued_monotonic: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class WorkerStatus:
    worker: str
    kind: StatusKind
    state: str
    message: str = ""
    metrics: Mapping[str, float | int | bool | str] = field(default_factory=dict)
    emitted_monotonic: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class CompletionAck:
    state: str
    hardware_safe: bool
    error: str = ""
    emitted_monotonic: float = field(default_factory=time.monotonic)
