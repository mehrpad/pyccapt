from __future__ import annotations

from types import SimpleNamespace

from pyccapt.control.core.health import build_health_snapshot


class _Buffer:
    def __init__(self, pending, dropped):
        self._pending = pending
        self.dropped = dropped

    def pending(self):
        return self._pending


class _Backend:
    def health(self):
        return SimpleNamespace(running=True, message="running")


def test_health_snapshot_reports_queue_drop_and_safe_state():
    variables = SimpleNamespace(
        experiment_state="running",
        hardware_safe=False,
        physical_estop_ok=True,
        last_chunk_write_latency_ms=2.5,
    )
    snapshot = build_health_snapshot(
        variables,
        SimpleNamespace(backend=_Backend()),
        [_Buffer(3, 4), _Buffer(5, 6)],
        heartbeat_monotonic=0.0,
    )
    assert snapshot.queue_depth == 8
    assert snapshot.dropped_records == 10
    assert snapshot.write_latency_ms == 2.5
    assert "UNSAFE" in snapshot.summary()
