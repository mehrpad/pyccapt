"""Low-rate operational health snapshots for the control dashboard."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable
from pathlib import Path

from pyccapt.control.core.chunk_store import latest_manifest_record


@dataclass(frozen=True)
class HealthSnapshot:
    emitted_monotonic: float
    experiment_state: str
    hardware_safe: bool
    physical_estop_ok: bool
    detector_running: bool
    detector_message: str
    queue_depth: int
    dropped_records: int
    write_latency_ms: float
    worker_heartbeat_age_s: float

    def metrics(self) -> dict[str, float | int | bool | str]:
        return asdict(self)

    def summary(self) -> str:
        safety = "SAFE" if self.hardware_safe and self.physical_estop_ok else "UNSAFE"
        detector = "up" if self.detector_running else self.detector_message
        return (
            f"Health: {safety} | detector {detector} | queued {self.queue_depth} | "
            f"dropped {self.dropped_records} | write {self.write_latency_ms:.1f} ms"
        )


def build_health_snapshot(
    variables: Any,
    detector_runtime: Any,
    buffers: Iterable[Any],
    *,
    heartbeat_monotonic: float,
) -> HealthSnapshot:
    backend = getattr(detector_runtime, "backend", None)
    health = backend.health() if backend is not None else None
    queue_depth = 0
    dropped = 0
    for buffer in buffers:
        if buffer is None:
            continue
        queue_depth += int(buffer.pending())
        dropped += int(getattr(buffer, "dropped", 0))
    now = time.monotonic()
    write_latency = float(getattr(variables, "last_chunk_write_latency_ms", 0.0))
    run_path = str(getattr(variables, "path", ""))
    if run_path:
        latest = latest_manifest_record(Path(run_path) / "temp_data" / "chunks")
        if latest is not None:
            write_latency = float(latest.get("write_latency_ms", write_latency))
    return HealthSnapshot(
        emitted_monotonic=now,
        experiment_state=str(getattr(variables, "experiment_state", "unknown")),
        hardware_safe=bool(getattr(variables, "hardware_safe", False)),
        physical_estop_ok=bool(getattr(variables, "physical_estop_ok", False)),
        detector_running=bool(health.running) if health else False,
        detector_message=health.message if health else "disabled",
        queue_depth=queue_depth,
        dropped_records=dropped,
        write_latency_ms=write_latency,
        worker_heartbeat_age_s=max(0.0, now - heartbeat_monotonic),
    )
