"""Regression tests for the 2026-10 acquisition/data-path audit fixes."""
from __future__ import annotations

import json
import math
import os
import time
import uuid
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import h5py
import numpy as np
import pytest
from simple_pid import PID

from pyccapt.control.apt import apt_exp_control as module
from pyccapt.control.apt import simulator
from pyccapt.control.apt.apt_exp_control import APT_Exp_Control
from pyccapt.control.core import chunk_store
from pyccapt.control.core.data_integrity import recover_chunks, validate_hdf5
from pyccapt.control.core.hdf5_creator import hdf_creator
from pyccapt.control.core.shared_ring_buffer import SharedEventBuffer
from pyccapt.control.nkt_photonics.readback import TELEMETRY_UNITS

_ROW_LISTS = ("main_counter", "main_raw_counter", "main_temperature", "main_chamber_vacuum",
              "main_laser_x", "main_laser_y", "main_laser_z", "main_stage_x", "main_stage_y", "main_stage_z")


# --------------------------------------------------------------------------- 1: apt metadata

def _control_with_rows(tmp_path, rows):
    control = APT_Exp_Control(SimpleNamespace(ex_freq=1, path=str(tmp_path)), {}, None, None, None, None, None)
    control.log_apt = Mock()
    for name in _ROW_LISTS:
        setattr(control, name, [float(i + 1) for i in range(rows)])
    control.main_laser_telemetry = {key: [float(i) for i in range(rows)] for key in TELEMETRY_UNITS}
    return control, list(range(rows)), [1700000000. + i for i in range(rows)]


def _add_row(control, time_counter, time_ex):
    index = len(time_counter)
    for name in _ROW_LISTS:
        getattr(control, name).append(float(index + 1))
    for values in control.main_laser_telemetry.values():
        values.append(float(index))
    time_counter.append(index)
    time_ex.append(1700000000. + index)


def test_failed_apt_flush_is_recorded_and_retry_keeps_chunk_ids_contiguous(tmp_path, monkeypatch):
    control, time_counter, time_ex = _control_with_rows(tmp_path, 2)
    real_write = module.atomic_write_chunk_group
    failures = [OSError("transient disk error")]

    def flaky_write(*args, **kwargs):
        if failures:
            raise failures.pop()
        return real_write(*args, **kwargs)

    monkeypatch.setattr(module, "atomic_write_chunk_group", flaky_write)
    assert control._flush_apt_meta_chunks(time_counter, time_ex) is False
    assert control._apt_meta_chunk_id == 0 and control._apt_meta_flush_offset == 0
    assert "transient disk error" in control._apt_meta_flush_errors[0]

    assert control._flush_apt_meta_chunks(time_counter, time_ex) is True
    chunk_dir = tmp_path / "temp_data" / "chunks"
    assert (chunk_dir / "apt_id_chunk_1.npy").is_file()
    report = chunk_store.validated_files_by_stem(chunk_dir).integrity_report()
    assert report["status"] == "complete"
    assert report["streams"]["apt"]["missing_chunk_id_ranges"] == []


def test_final_save_uses_complete_memory_when_final_apt_flush_fails(tmp_path, monkeypatch):
    control, time_counter, time_ex = _control_with_rows(tmp_path, 2)
    assert control._flush_apt_meta_chunks(time_counter, time_ex)
    _add_row(control, time_counter, time_ex)
    monkeypatch.setattr(module, "atomic_write_chunk_group", Mock(side_effect=OSError("disk full")))
    assert control._flush_apt_meta_chunks(time_counter, time_ex) is False

    variables = SimpleNamespace(path=str(tmp_path), exp_name="run")
    hdf_creator(variables, {"tdc": "off"}, time_counter, time_ex,
                apt_meta=control._apt_meta_columns(time_counter, time_ex),
                apt_meta_flush_errors=control._apt_meta_flush_errors)
    with h5py.File(tmp_path / "run.h5") as handle:
        np.testing.assert_array_equal(handle["apt/id"], [0, 1, 2])
        np.testing.assert_array_equal(handle["apt/stage_x"], [1., 2., 3.])
        np.testing.assert_array_equal(handle["apt/laser_output_power_mw"], [0., 1., 2.])
        attrs = handle["provenance"].attrs
        apt_report = json.loads(attrs["chunk_integrity_json"])["apt_metadata"]
        assert apt_report["source"] == "memory" and apt_report["rows"] == 3
        assert "disk full" in apt_report["flush_errors"][0]
        assert attrs["acquisition_completeness"] == "complete"
    assert validate_hdf5(tmp_path / "run.h5")["valid"]


def test_chunk_fallback_reports_unloadable_apt_chunk_as_incomplete(tmp_path):
    chunk_dir = tmp_path / "temp_data" / "chunks"
    chunk_dir.mkdir(parents=True)
    for chunk_id in (1, 2):
        np.save(chunk_dir / f"apt_id_chunk_{chunk_id}.npy", np.array([chunk_id], dtype=np.uint64))
        np.save(chunk_dir / f"apt_temperature_chunk_{chunk_id}.npy", np.array([40.], dtype=np.float64))
    (chunk_dir / "apt_temperature_chunk_2.npy").write_bytes(b"corrupt")
    variables = SimpleNamespace(path=str(tmp_path), exp_name="run", main_counter=[1, 1], main_raw_counter=[1, 1],
                                main_temperature=[40., 40.], main_chamber_vacuum=[1e-8, 1e-8])
    hdf_creator(variables, {"tdc": "off"}, [1, 2], [0., 1.])
    with h5py.File(tmp_path / "run.h5") as handle:
        attrs = handle["provenance"].attrs
        apt_report = json.loads(attrs["chunk_integrity_json"])["apt_metadata"]
        assert apt_report["source"] == "chunks"
        assert apt_report["unloadable_chunks"][0]["file"] == "apt_temperature_chunk_2.npy"
        assert apt_report["length_mismatch"]["apt/temperature"] == 1
        assert attrs["acquisition_completeness"] == "incomplete"


def test_chunkless_save_writes_stage_laser_and_telemetry_datasets(tmp_path):
    control, time_counter, time_ex = _control_with_rows(tmp_path, 2)
    hdf_creator(SimpleNamespace(path=str(tmp_path), exp_name="run"), {"tdc": "off"}, time_counter, time_ex,
                apt_meta=control._apt_meta_columns(time_counter, time_ex))
    with h5py.File(tmp_path / "run.h5") as handle:
        for name in ("laser_x", "laser_y", "laser_z", "stage_x", "stage_y", "stage_z",
                     *("laser_" + key for key in TELEMETRY_UNITS)):
            assert len(handle["apt/" + name]) == 2


# --------------------------------------------------------------------------- 2/3/4: run harness

class _Interlock:
    def __init__(self, safe_calls=300):
        self.safe_calls = safe_calls
        self.kick_watchdog = Mock()
        self.close = Mock()

    def is_safe(self):
        # Bounded: an unbounded loop ends as an E-stop failure instead of hanging.
        self.safe_calls -= 1
        return self.safe_calls > 0


class _RunVariables(SimpleNamespace):
    def extend_to(self, name, values):
        getattr(self, name).extend(values)


def _run_harness(monkeypatch, tmp_path, conf, **overrides):
    data_path = tmp_path / "data" / "run"
    variables = _RunVariables(
        electrode_out=False, pulse_mode="Voltage", control_algorithm="Proportional",
        access_override_enabled=False, override_disabled_devices=[], experiment_plan_snapshot={},
        automatic_alignment_enabled=False, vdc_min=100., vdc_max=200., v_p_min=10., v_p_max=100.,
        ex_freq=100, detection_rate=1., detection_rate_current=1., pulse_fraction=20.,
        pulse_amp_per_supply_voltage=20., pulse_frequency=200., total_ions=0, total_raw_signals=0,
        stop_flag=False, flat_test_active=False, email_interval_events=0, criteria_email=False,
        email="operator@example.org", vdc_hold=False, flag_tdc_failure=False, criteria_ions=False,
        max_ions=0, criteria_vdc=False, criteria_time=True, ex_time=0.2, elapsed_time=0.,
        temperature=40., vacuum_main=1e-8, laser_pos_x=0., laser_pos_y=0., laser_pos_z=0.,
        stage_pos_x=0., stage_pos_y=0., stage_pos_z=0., laser_telemetry={}, vdc_step_up=1.,
        vdc_step_down=1., flag_new_min_voltage=False, counter_source="TDC", counter=1,
        start_flag=True, flag_finished_tdc=True, main_counter=[], main_raw_counter=[],
        main_temperature=[], main_chamber_vacuum=[], x=[], detector_warning="", detector_error="",
        experiment_state="initializing", experiment_error="", hardware_safe=True,
        physical_estop_ok=True, specimen_voltage=0., pulse_voltage=0., COM_PORT_V_dc="COM1",
        COM_PORT_V_p="COM2", exp_name="run",
    )
    for key, value in overrides.items():
        setattr(variables, key, value)

    def prepare(v):
        v.path, v.path_meta, v.log_path = str(data_path), str(data_path / "meta_data"), str(data_path / "meta_data")
        return data_path, data_path / "meta_data"

    import pyccapt.control.apt.laser_alignment_runtime as laser_runtime
    import pyccapt.control.core.alignment_diagnostics as alignment_diagnostics
    import pyccapt.control.core.experiment_plan as experiment_plan
    monkeypatch.setattr(module, "prepare_experiment_output_paths", prepare)
    monkeypatch.setattr(experiment_plan, "write_plan_snapshot", lambda *_: None)
    monkeypatch.setattr(alignment_diagnostics, "copy_transfer_journal", lambda *_: None)
    monkeypatch.setattr(laser_runtime, "LaserAlignmentRuntime", lambda _control: None)
    monkeypatch.setattr(module, "acquisition_metadata", lambda *_a, **_k: {})
    interlock = _Interlock()
    monkeypatch.setattr(module, "build_safety_interlock", lambda _conf: interlock)
    monkeypatch.setattr(module.loggi, "logger_creator", lambda *_a, **_k: Mock())
    monkeypatch.setattr(module.loggi, "log_configuration_snapshot", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "reset_runtime_variables", lambda *_a, **_k: None)
    finalize = SimpleNamespace(hdf=Mock(), stats=Mock(), email=Mock())
    monkeypatch.setattr(module.hdf5_creator, "hdf_creator", finalize.hdf)
    monkeypatch.setattr(module.experiment_statistics, "save_statistics_apt", finalize.stats)
    monkeypatch.setattr(module.apt_exp_control_func, "send_info_email", finalize.email)
    counter_file = tmp_path / "counter.txt"
    monkeypatch.setattr(module.runtime, "ensure_counter_file", lambda: counter_file)
    control = APT_Exp_Control(variables, conf, None, None, None, None, None)
    control._publish_health_if_due = lambda: None
    return control, variables, finalize, counter_file, interlock


def test_time_criterion_uses_experiment_process_clock_without_visualization(monkeypatch, tmp_path):
    control, variables, finalize, _, _ = _run_harness(monkeypatch, tmp_path, {})
    started = time.monotonic()
    control._run_experiment_impl()

    assert control._run_failure == ""  # Not the bounded E-stop guard: the time criterion fired.
    assert time.monotonic() - started < 3.
    assert 0.2 <= variables.elapsed_time < 3.
    finalize.hdf.assert_called_once()
    kwargs = finalize.hdf.call_args.kwargs
    assert len(kwargs["apt_meta"]["apt_stage_x"]) == len(finalize.hdf.call_args.args[2])


def test_stale_detector_finish_flag_is_reset_before_detector_start(monkeypatch, tmp_path):
    control, variables, _, _, _ = _run_harness(monkeypatch, tmp_path, {"tdc": "on"})
    seen = []
    process = SimpleNamespace(join=lambda *_: None, is_alive=lambda: False, exitcode=0)

    def start(*_args, **_kwargs):
        seen.append(variables.flag_finished_tdc)
        variables.flag_finished_tdc = True  # A detector that finishes promptly on Stop.
        return module.DetectorRuntime(tdc_process=process)

    monkeypatch.setattr(module, "start_detector_processes", start)
    control._run_experiment_impl()
    assert seen == [False]


def test_device_init_failure_skips_dataset_email_counter_and_closes_ports(monkeypatch, tmp_path):
    control, variables, finalize, counter_file, _ = _run_harness(
        monkeypatch, tmp_path, {"v_dc": "on", "v_p": "on"})
    ports = []

    def serial_factory(*args, **kwargs):
        port = Mock(is_open=True)
        ports.append((kwargs.get("port", args[0] if args else None), port))
        return port

    monkeypatch.setattr(module.serial, "Serial", serial_factory)
    monkeypatch.setattr(module.apt_exp_control_func, "initialization_v_dc", lambda *_: True)

    control._run_experiment_impl()
    assert control._run_failure == "Experiment device initialization failed."
    assert [name for name, _ in ports] == ["COM1"]  # V_p is not opened after V_dc failed.
    finalize.hdf.assert_not_called()
    finalize.stats.assert_not_called()
    finalize.email.assert_not_called()
    assert not counter_file.exists() and variables.counter == 1

    control.clear_up()
    ports[0][1].close.assert_called_once()


# --------------------------------------------------------------------------- 5/6/7: detectors

def test_simulator_rate_accounts_for_batch_interval(monkeypatch):
    variables = SimpleNamespace(stop_flag=False, flag_stop_tdc=False, specimen_voltage=1000.,
                                pulse_frequency=100., simulator_seed=1, simulator_batch_size=10,
                                simulator_interval_s=0.02, simulator_fail_after_batches=0)
    calls = deque([False, True])
    stop = SimpleNamespace(is_set=lambda: calls.popleft() if calls else True)
    monkeypatch.setattr(simulator.time, "sleep", lambda _s: None)
    simulator.experiment_measure(variables, None, None, None, None, stop)
    # 10 events per 20 ms at 100 kHz = 500 ions/s / 1e5 pulses/s = 0.5 %.
    assert variables.detection_rate_current == pytest.approx(0.5)


def test_finite_rate_frequency_never_returns_nan():
    from pyccapt.control.apt.detector_chunks import finite_rate_frequency_hz
    assert finite_rate_frequency_hz(float("nan"), 2500.) == 2500.
    assert finite_rate_frequency_hz(float("nan")) == 1.
    assert finite_rate_frequency_hz(0.2) == 1.
    assert finite_rate_frequency_hz(40000.) == 40000.


class _MemoryWriter:
    def check(self):
        pass

    def submit(self, *_args):
        pass

    def close(self):
        pass


def _roentdek_groups(n, first_ts=100):
    """n TDC groups: every channel has one hit, ch7 (MCP trigger) at 0, start (ch6) before it."""
    ts = np.arange(first_ts, first_ts + n, dtype=np.uint64)
    cnt = np.ones((n, 8), np.uint32)
    tdc = np.zeros((n, 8, 10), np.int32)
    tdc[:, :6, 0] = (np.arange(n)[:, None] + 1) * np.arange(1, 7)[None, :]
    tdc[:, 6, 0] = -1000 - np.arange(n)
    return ts, cnt, tdc


def _roentdek_fake_sdk(read):
    sdk = Mock()
    sdk.init_tdc.return_value = 0
    sdk.run_tdc.return_value = 0
    sdk.bin_size_ns.return_value = .025
    sdk.diagnostics.return_value = {}
    sdk.last_error.return_value = ''
    sdk.read.side_effect = read
    return sdk


def _roentdek_fake_reco(*_args, **_kwargs):
    """one ion per group: x = first u1 hit [mm], y = first v1 hit [mm], t = MCP - start [bins]"""
    def reconstruct(cnt, tdc, time_stamp=None):
        n = len(cnt)
        return {"event": np.arange(n), "hit": np.zeros(n, np.int8), "x": tdc[:, 0, 0].astype(float),
                "y": tdc[:, 2, 0].astype(float), "t": (tdc[:, 7, 0] - tdc[:, 6, 0]).astype(float),
                "method": np.zeros(n, np.int8), "time_stamp": time_stamp}
    return SimpleNamespace(bin_ns=.025, reconstruct=reconstruct, stats_text=lambda: "", signal_warnings=lambda: [])


def test_roentdek_nan_laser_frequency_keeps_rate_finite_and_passes_alignment_conf(monkeypatch, tmp_path):
    from pyccapt.control.apt import alignment_vision
    from pyccapt.control.tdc_roentdek import tdc_roentdek as roentdek
    variables = SimpleNamespace(path=str(tmp_path), ex_freq=10., specimen_voltage=1500., pulse_voltage=50.,
                                pulse_mode="Laser", flag_stop_tdc=False, stop_flag=False, total_ions=0,
                                total_raw_signals=0, automatic_alignment_enabled=False, laser_alignment_epoch="")
    clock = [0.]
    packets = deque([(.5, _roentdek_groups(5)), (1., None)])

    def packet():
        clock[0], buffer = packets.popleft()
        return buffer

    sdk = _roentdek_fake_sdk(packet)
    monkeypatch.setattr(roentdek, "make_reconstructor", _roentdek_fake_reco)
    seen_conf = []

    class Publisher:
        def __init__(self, _variables, conf=None):
            seen_conf.append(conf)

        def append(self, *_args):
            pass

    monkeypatch.setattr(alignment_vision, "AlignmentEventPublisher", Publisher)
    monkeypatch.setattr(roentdek.ctypes, "CDLL", lambda _: object())
    monkeypatch.setattr(roentdek, "TDC", lambda *_a, **_k: sdk)
    monkeypatch.setattr(roentdek, "ChunkWriter", lambda *_a, **_k: _MemoryWriter())
    monkeypatch.setattr(roentdek, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda _: None))
    monkeypatch.setattr(roentdek, "experiment_frequency_hz", lambda _: float("nan"))
    monkeypatch.setattr(roentdek, "pulse_energy_pj", lambda _: 12.)
    conf = {"alignment_window_ions": 321}
    assert roentdek.experiment_measure(variables, *[Mock() for _ in range(4)],
                                       SimpleNamespace(is_set=lambda: not packets), conf=conf) == 0
    assert math.isfinite(variables.detection_rate_current)
    assert seen_conf == [conf]


def test_roentdek_backend_receives_config():
    from pyccapt.control.apt import detector_runtime
    captured = {}

    def process_factory(target, args):
        captured["target"] = target
        return SimpleNamespace(start=lambda: None)

    variables = SimpleNamespace(counter_source="TDC")
    detector_runtime._build_backend({"tdc_model": "RoentDek", "alignment_window_ions": 7}, variables,
                                    None, None, None, None, process_factory, lambda: SimpleNamespace())
    assert captured["target"].keywords["conf"]["alignment_window_ions"] == 7


# --------------------------------------------------------------------------- 8: chunk cleanup

def test_failed_chunk_write_removes_only_its_own_temporaries(tmp_path, monkeypatch):
    other_stream = tmp_path / ".apt_id_chunk_1.npy.4242.tmp"  # e.g. the experiment's apt writer
    other_stream.write_bytes(b"in flight")
    real_replace = chunk_store.os.replace

    def fail_replace(source, target):
        if str(target).endswith("y_chunk_1.npy"):
            raise OSError("injected disk full")
        return real_replace(source, target)

    monkeypatch.setattr(chunk_store.os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk full"):
        chunk_store.atomic_write_chunk_group(tmp_path, stream_name="dld", chunk_id=1,
                                             arrays={"x": np.arange(3.), "y": np.arange(3.)})
    assert other_stream.is_file()
    assert not list(tmp_path.glob(f".y_chunk_1.npy.{os.getpid()}.tmp"))


# --------------------------------------------------------------------------- 9: recovery

def test_recover_run_includes_apt_metadata_and_laser_telemetry(tmp_path):
    from pyccapt.control.core import recover_chunks_to_hdf5
    control, time_counter, time_ex = _control_with_rows(tmp_path, 3)
    assert control._flush_apt_meta_chunks(time_counter, time_ex)
    chunk_dir = tmp_path / "temp_data" / "chunks"
    chunk_store.atomic_write_chunk_group(chunk_dir, stream_name="dld", chunk_id=1, arrays={
        "x": np.array([1.]), "y": np.array([2.]), "t": np.array([3.]), "voltage": np.array([500.]),
        "voltage_pulse": np.array([50.]), "laser_pulse": np.array([0.]),
        "start_counter": np.array([1], dtype=np.uint64)})
    result = recover_chunks(chunk_dir, tmp_path / "recovered.h5")
    assert result["datasets_written"]["apt/stage_x"] == 3
    with h5py.File(tmp_path / "recovered.h5") as handle:
        np.testing.assert_array_equal(handle["apt/laser_output_power_mw"], [0., 1., 2.])
        np.testing.assert_array_equal(handle["apt/id"], [0, 1, 2])
        assert len(handle["dld/x"]) == 1
    stems = {stem for _, stem, _ in recover_chunks_to_hdf5.APT_MAPPING}
    assert {"apt_laser_" + key for key in TELEMETRY_UNITS} <= stems


# --------------------------------------------------------------------------- 10: plot transport

class _ScriptedLock:
    def __init__(self, results):
        self.results = deque(results)
        self.release = Mock()

    def acquire(self, block=True, timeout=None):
        return self.results.popleft() if self.results else True


@pytest.fixture
def event_buffer():
    buffer = SharedEventBuffer.create("audit_" + uuid.uuid4().hex, 16)
    try:
        yield buffer
    finally:
        buffer.unlink()


def test_single_plot_lock_timeout_does_not_latch_transport_error(event_buffer):
    event_buffer._lock = _ScriptedLock([False])
    assert event_buffer.write_events([1], [2], [3], [4]) is False
    assert event_buffer.transport_error == ""
    assert event_buffer.write_events([1], [2], [3], [4]) is True


def test_latched_plot_transport_recovers_when_lock_is_free(event_buffer):
    event_buffer._lock = _ScriptedLock([False] * event_buffer._MAX_CONSECUTIVE_TIMEOUTS)
    for _ in range(event_buffer._MAX_CONSECUTIVE_TIMEOUTS):
        event_buffer.write_events([1], [2], [3], [4])
    assert event_buffer.transport_error
    assert event_buffer.write_events([1], [2], [3], [4]) is False  # Probe is rate-limited.
    event_buffer._next_probe = 0.
    assert event_buffer.write_events([1], [2], [3], [4]) is True
    assert event_buffer.transport_error == ""


# --------------------------------------------------------------------------- 11/12: control loop

def _loop_control(monkeypatch, algorithm="Proportional", **variables):
    v = SimpleNamespace(temperature=40., vacuum_main=1e-8, laser_pos_x=0, laser_pos_y=0, laser_pos_z=0,
                        stage_pos_x=0, stage_pos_y=0, stage_pos_z=0, control_algorithm=algorithm,
                        detection_rate_current=1., vdc_step_down=4, vdc_step_up=4, stop_flag=False,
                        vdc_hold=False)
    for key, value in variables.items():
        setattr(v, key, value)
    control = SimpleNamespace(variables=v, total_ions=100, count_last=100, _tdc_first_event_seen=True,
                              total_raw_signals=100, count_raw_signals_last=100, main_laser_telemetry={},
                              control_algorithm=algorithm, detection_rate=1, laser_alignment_runtime=None,
                              alignment=None, specimen_voltage=1000., vdc_min=500., vdc_max=5000.,
                              _vdc_active=lambda: True, com_port_v_dc=object(), pulse_mode="Laser",
                              _pid_max_step=40.)
    for key in _ROW_LISTS:
        setattr(control, key, [])
    monkeypatch.setattr(module, "fresh_snapshot", lambda _: {})
    command = Mock()
    monkeypatch.setattr(module.apt_exp_control_func, "command_v_dc", command)
    return control, command


def test_sensor_error_sentinels_are_stored_as_nan(monkeypatch):
    control, _ = _loop_control(monkeypatch, temperature=-1, vacuum_main=-1.0)
    module.APT_Exp_Control.main_ex_loop(control)
    assert math.isnan(control.main_temperature[0]) and math.isnan(control.main_chamber_vacuum[0])
    control.variables.temperature, control.variables.vacuum_main = 45., 2e-9
    module.APT_Exp_Control.main_ex_loop(control)
    assert control.main_temperature[1] == 45. and control.main_chamber_vacuum[1] == 2e-9


def test_skipped_pid_evaluation_applies_no_repeated_voltage_step(monkeypatch):
    control, command = _loop_control(monkeypatch, "PID", detection_rate_current=0.5)
    clock = [0.]
    control.pid = PID(1.0, 0.5, 0., setpoint=1., sample_time=0.2, output_limits=(-40., 40.),
                      proportional_on_measurement=True, time_fn=lambda: clock[0])
    module.APT_Exp_Control.main_ex_loop(control)  # First evaluation.
    clock[0] = 0.2
    module.APT_Exp_Control.main_ex_loop(control)  # Second evaluation: integral term > 0.
    after_evaluation = control.specimen_voltage
    assert after_evaluation > 1000.
    calls = command.call_count
    clock[0] = 0.3875  # dt below sample_time, as with a 15.6 ms Windows monotonic tick.
    module.APT_Exp_Control.main_ex_loop(control)
    assert control.specimen_voltage == after_evaluation
    assert command.call_count == calls


def test_pid_nan_output_does_not_step_to_the_cap(monkeypatch):
    control, command = _loop_control(monkeypatch, "PID", detection_rate_current=float("nan"))
    control.pid = PID(1.0, 0.5, 0., setpoint=1., sample_time=None, output_limits=(-40., 40.))
    module.APT_Exp_Control.main_ex_loop(control)
    assert control.specimen_voltage == 1000.
    command.assert_not_called()


# --------------------------------------------------------------------------- 13: min-voltage ramp

def _ramp_control(monkeypatch, safe=True):
    variables = SimpleNamespace(ex_freq=1, stop_flag=False, flag_new_min_voltage=True,
                                specimen_voltage=5000., specimen_voltage_plot=5000.)
    control = APT_Exp_Control(variables, {"v_dc": "on"}, None, None, None, None, None)
    control.com_port_v_dc, control.initialization_v_dc = object(), True
    control.specimen_voltage, control.vdc_min, control.vdc_max, control.pulse_mode = 5000., 4000., 6000., "Laser"
    control.safety_interlock = SimpleNamespace(kick_watchdog=Mock(), is_safe=lambda: safe)
    command = Mock()
    monkeypatch.setattr(module.apt_exp_control_func, "command_v_dc", command)
    monkeypatch.setattr(module, "time", SimpleNamespace(sleep=lambda _s: None, monotonic=time.monotonic))
    return control, variables, command


def test_min_voltage_ramp_kicks_watchdog_and_honours_stop(monkeypatch):
    control, variables, command = _ramp_control(monkeypatch)
    command.side_effect = lambda *_: setattr(variables, "stop_flag", True)
    control._ramp_to_new_min_voltage()
    assert command.call_count == 1
    control.safety_interlock.kick_watchdog.assert_called()
    assert variables.specimen_voltage == 4900. and variables.flag_new_min_voltage is False


def test_min_voltage_ramp_aborts_when_interlock_opens(monkeypatch):
    control, variables, command = _ramp_control(monkeypatch, safe=False)
    control._ramp_to_new_min_voltage()
    command.assert_not_called()
    assert variables.specimen_voltage == 5000.


def test_min_voltage_ramp_completes_normally(monkeypatch):
    control, variables, command = _ramp_control(monkeypatch)
    control._ramp_to_new_min_voltage()
    assert command.call_count == 10
    assert control.safety_interlock.kick_watchdog.call_count == 10
    assert variables.specimen_voltage == pytest.approx(4000.)

