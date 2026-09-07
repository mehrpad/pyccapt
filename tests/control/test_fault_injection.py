from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pyccapt.control.apt import simulator
from pyccapt.control.apt.detector_runtime import ProcessDetectorBackend
from pyccapt.control.core import chunk_store


class _Ring:
    def __init__(self):
        self.parts = []

    def write(self, values):
        self.parts.append(np.asarray(values).copy())


class _StopEvent:
    def is_set(self):
        return False


@pytest.mark.fault_injection
def test_simulator_worker_failure_is_reported_and_finishes():
    variables = SimpleNamespace(
        stop_flag=False,
        flag_stop_tdc=False,
        flag_tdc_failure=False,
        flag_finished_tdc=False,
        specimen_voltage=2500.0,
        pulse_frequency=100.0,
        simulator_seed=7,
        simulator_batch_size=4,
        simulator_interval_s=0.0,
        simulator_fail_after_batches=2,
    )
    rings = [_Ring() for _ in range(4)]
    simulator.experiment_measure(variables, *rings, _StopEvent())
    assert variables.flag_tdc_failure is True
    assert variables.flag_finished_tdc is True
    assert "injected simulator failure" in variables.detector_error
    assert variables.total_ions == 8


@pytest.mark.fault_injection
def test_atomic_chunk_disk_failure_publishes_no_manifest(tmp_path, monkeypatch):
    real_replace = chunk_store.os.replace

    def fail_replace(source, target):
        if str(target).endswith("x_chunk_1.npy"):
            raise OSError("injected disk full")
        return real_replace(source, target)

    monkeypatch.setattr(chunk_store.os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk full"):
        chunk_store.atomic_write_chunk_group(
            tmp_path, stream_name="dld", chunk_id=1, arrays={"x": np.arange(4.0)}
        )
    assert not list(tmp_path.glob("manifest*.jsonl"))
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.fault_injection
def test_corrupt_manifest_line_does_not_hide_valid_chunk(tmp_path):
    chunk_store.atomic_write_chunk_group(
        tmp_path, stream_name="dld", chunk_id=1, arrays={"x": np.arange(3.0)}
    )
    manifest = tmp_path / "manifest.dld.jsonl"
    with manifest.open("a", encoding="utf-8") as stream:
        stream.write('{"truncated":')
    valid, invalid = chunk_store.validate_manifest_records(tmp_path)
    assert len(valid) == 1
    assert len(invalid) == 1
    assert "invalid JSON" in invalid[0]["error"]


@pytest.mark.fault_injection
def test_forced_close_escalates_a_hung_detector_worker():
    class HungProcess:
        exitcode = None

        def __init__(self):
            self.alive = True
            self.terminated = False

        def start(self):
            return None

        def join(self, timeout):
            return None

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True
            self.alive = False
            self.exitcode = -15

    process = HungProcess()
    backend = ProcessDetectorBackend(
        name="fault-test",
        counter_source="TDC",
        target=lambda: None,
        args=(),
        variables=SimpleNamespace(path=""),
        process_factory=lambda **_: process,
        event_factory=lambda: SimpleNamespace(set=lambda: None, is_set=lambda: False),
    )

    backend.join(timeout=0.01)

    assert process.terminated is True
    assert process.alive is False
