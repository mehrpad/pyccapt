"""Physical E-stop/watchdog integration and override audit trail."""

from __future__ import annotations

import datetime as dt
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable


class SafetyInterlock(ABC):
    """Physical safety boundary. Software overrides must never bypass it."""

    @abstractmethod
    def is_safe(self) -> bool:
        """Return whether the physical interlock permits energized output."""

    @abstractmethod
    def kick_watchdog(self) -> None:
        """Signal that the control process remains responsive."""

    @abstractmethod
    def close(self) -> None:
        """Release hardware and leave watchdog output inactive."""


class NullSafetyInterlock(SafetyInterlock):
    """Explicit no-hardware implementation for simulation and legacy rigs."""

    def is_safe(self) -> bool:
        return True

    def kick_watchdog(self) -> None:
        return None

    def close(self) -> None:
        return None


class NIDaqSafetyInterlock(SafetyInterlock):
    """NI-DAQ digital input E-stop with optional heartbeat output."""

    def __init__(self, input_channel: str, output_channel: str | None = None, *, safe_when_high: bool = True):
        try:
            import nidaqmx
        except ImportError as exc:
            raise RuntimeError("NI-DAQ safety interlock requires the nidaqmx package") from exc
        self._safe_when_high = bool(safe_when_high)
        self._heartbeat = False
        self._input_task = nidaqmx.Task()
        self._input_task.di_channels.add_di_chan(input_channel)
        self._output_task = None
        if output_channel:
            self._output_task = nidaqmx.Task()
            self._output_task.do_channels.add_do_chan(output_channel)
            self._output_task.write(False)

    def is_safe(self) -> bool:
        value = bool(self._input_task.read())
        return value if self._safe_when_high else not value

    def kick_watchdog(self) -> None:
        if self._output_task is not None:
            self._heartbeat = not self._heartbeat
            self._output_task.write(self._heartbeat)

    def close(self) -> None:
        if self._output_task is not None:
            try:
                self._output_task.write(False)
            finally:
                self._output_task.close()
        self._input_task.close()


def build_safety_interlock(conf: dict[str, Any]) -> SafetyInterlock:
    backend = str(conf.get("safety_interlock_backend", "none")).strip().lower()
    if backend in {"", "none", "disabled", "simulator"}:
        return NullSafetyInterlock()
    if backend == "nidaq":
        channel = str(conf.get("safety_estop_input_channel", "")).strip()
        if not channel:
            raise ValueError("safety_estop_input_channel is required for the nidaq safety backend")
        output = str(conf.get("safety_watchdog_output_channel", "")).strip() or None
        return NIDaqSafetyInterlock(
            channel,
            output,
            safe_when_high=bool(conf.get("safety_estop_safe_when_high", True)),
        )
    raise ValueError(f"Unsupported safety_interlock_backend: {backend!r}")


def audit_safety_override(
    metadata_directory: str | Path,
    *,
    operator: str,
    disabled_devices: Iterable[str],
    reason: str,
) -> Path:
    """Append a durable audit record for every software safety override."""
    directory = Path(metadata_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    audit_path = directory / "safety_overrides.jsonl"
    record = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "operator": operator,
        "disabled_devices": sorted(str(item) for item in disabled_devices),
        "reason": reason,
        "pid": os.getpid(),
    }
    descriptor = os.open(audit_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, (json.dumps(record, sort_keys=True) + "\n").encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return audit_path
