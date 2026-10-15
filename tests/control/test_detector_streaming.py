"""Replay real acquisition loops with fake SDKs; no hardware is opened."""
from collections import deque
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from pyccapt.control.apt.detector_chunks import ChunkStream, ChunkWriter
from pyccapt.control.core.chunk_store import validate_manifest_records


class MemoryWriter:
    def __init__(self, *_args, **_kwargs):
        self.records = []

    def submit(self, stream, chunk_id, arrays):
        self.records.append((stream, chunk_id, {key: value.copy() for key, value in arrays.items()}))

    def check(self):
        pass

    def close(self):
        pass

    def column(self, key):
        return np.concatenate([record[2][key] for record in self.records if key in record[2]])


def test_typed_blocks_drain_every_chunk_and_preserve_order():
    writer = MemoryWriter()
    stream = ChunkStream(writer, "test", {"id": np.uint64, "value": np.float64}, chunk_size=3)
    stream.append({"id": np.arange(11), "value": np.arange(11)*.25})
    assert len(writer.records) == 3
    assert stream.rows == 2
    stream.append({"id": [11, 12], "value": [2.75, 3.]})
    stream.finish()
    np.testing.assert_array_equal(writer.column("id"), np.arange(13))
    np.testing.assert_array_equal(writer.column("value"), np.arange(13)*.25)
    assert [len(record[2]["id"]) for record in writer.records] == [3, 3, 3, 3, 1]
    assert writer.column("id").dtype == np.uint64
    assert stream.rows == 0


def test_failed_submission_retains_unsubmitted_buffer():
    writer = MemoryWriter()
    writer.submit = Mock(side_effect=RuntimeError("writer failed"))
    stream = ChunkStream(writer, "test", {"id": np.uint64}, chunk_size=3)
    with pytest.raises(RuntimeError, match="writer failed"):
        stream.append({"id": np.arange(5)})
    assert stream.rows == 5 and stream.chunk_id == 0


def test_stream_owns_inputs_before_async_writer_or_sdk_reuse():
    captured = []
    writer = SimpleNamespace(submit=lambda name, chunk_id, arrays: captured.append(arrays))
    stream = ChunkStream(writer, "test", {"id": np.uint64}, chunk_size=3)
    first = np.arange(5, dtype=np.uint64)
    stream.append({"id": first})
    first[:] = 999  # SDK reuses its batch before the queue feeder copies it.
    stream.finish()
    np.testing.assert_array_equal(np.concatenate([item["id"] for item in captured]), np.arange(5))


def test_full_writer_queue_times_out_and_shutdown_reaps_worker(monkeypatch):
    from pyccapt.control.apt import detector_chunks as module
    clock = [0.]
    process = Mock(exitcode=None)
    process.is_alive.return_value = True
    process.terminate.side_effect = lambda: setattr(process.is_alive, "return_value", False)
    queue = Mock()
    def full(*_args, **_kwargs):
        clock[0] += .1
        raise Full
    queue.put.side_effect = full
    writer = ChunkWriter.__new__(ChunkWriter)
    writer.timeout_s = .2
    writer.queue, writer.process, writer.closed = queue, process, False
    writer.errors = Mock(poll=Mock(return_value=False))
    writer.stop_event = SimpleNamespace(is_set=lambda: True)
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    with pytest.raises(TimeoutError, match="during shutdown"):
        writer.close()
    process.terminate.assert_called_once()
    process.join.assert_called_once_with(2.)
    queue.cancel_join_thread.assert_called_once()


def test_orphan_writer_exits_after_owner_dies_during_idle_poll(monkeypatch):
    from pyccapt.control.apt import detector_chunks as module
    owner = Mock()
    owner.is_alive.side_effect = [True, False]
    queue, errors = Mock(), Mock()
    queue.get.side_effect = Empty
    monkeypatch.setattr(module.multiprocessing, 'parent_process', lambda: owner)
    module._write_chunks('unused', queue, errors)
    queue.get.assert_called_once_with(timeout=.1)
    errors.close.assert_called_once()


def test_writer_finishes_committed_chunks_even_after_stop(tmp_path):
    writer = ChunkWriter(tmp_path, timeout_s=5, stop_event=SimpleNamespace(is_set=lambda: True))
    stream = ChunkStream(writer, "test", {"id": np.uint64}, chunk_size=3)
    stream.append({"id": np.arange(8)})
    stream.finish()
    writer.close()
    valid, invalid = validate_manifest_records(tmp_path, quarantine=False)
    assert not invalid and [row["rows"] for row in valid] == [3, 3, 2]
    assert not writer.process.is_alive()


def test_writer_io_failure_is_reported_and_worker_reaped(tmp_path):
    invalid_directory = tmp_path/"file"
    invalid_directory.write_text("not a directory")
    writer = ChunkWriter(invalid_directory, timeout_s=5)
    writer.submit("test", 1, {"id": np.arange(2)})
    with pytest.raises(RuntimeError, match="writer"):
        writer.close()
    assert not writer.process.is_alive()


@pytest.mark.parametrize("current,measured,expected", [(510., 2., 500.), (540., 2., 500.), (990., 0., 1000.)])
def test_control_voltage_clamps_to_both_inclusive_bounds(monkeypatch, current, measured, expected):
    from pyccapt.control.apt import apt_exp_control as module
    v = SimpleNamespace(temperature=0, vacuum_main=0, laser_pos_x=0, laser_pos_y=0,
                        laser_pos_z=0, stage_pos_x=0, stage_pos_y=0, stage_pos_z=0,
                        control_algorithm="Proportional", detection_rate_current=measured,
                        vdc_step_down=4, vdc_step_up=4, stop_flag=False, vdc_hold=False)
    control = SimpleNamespace(variables=v, total_ions=100, count_last=100, _tdc_first_event_seen=True,
                              total_raw_signals=100, count_raw_signals_last=100, main_laser_telemetry={},
                              control_algorithm="Proportional", detection_rate=1, laser_alignment_runtime=None,
                              alignment=None, specimen_voltage=current, vdc_min=500., vdc_max=1000.,
                              _vdc_active=lambda: True, com_port_v_dc=object(), pulse_mode="Laser")
    for key in ("main_counter", "main_raw_counter", "main_temperature", "main_chamber_vacuum",
                "main_laser_x", "main_laser_y", "main_laser_z", "main_stage_x", "main_stage_y", "main_stage_z"):
        setattr(control, key, [])
    monkeypatch.setattr(module, "fresh_snapshot", lambda _: {})
    command = Mock()
    monkeypatch.setattr(module.apt_exp_control_func, "command_v_dc", command)
    module.APT_Exp_Control.main_ex_loop(control)
    assert control.specimen_voltage == expected
    command.assert_called_once_with(control.com_port_v_dc, f">S0 {expected}")


def _variables(tmp_path):
    return SimpleNamespace(path=str(tmp_path), ex_freq=10., specimen_voltage=1500., pulse_voltage=50.,
                           pulse_mode="Voltage", flag_stop_tdc=False, stop_flag=False, total_ions=0,
                           total_raw_signals=0, automatic_alignment_enabled=False, laser_alignment_epoch="",
                           extend_to=Mock(side_effect=AssertionError("No bulk Manager transfers allowed")))


def test_roentdek_empty_packets_publish_zero_and_preserve_columns(monkeypatch, tmp_path):
    from pyccapt.control.tdc_roentdek import tdc_roentdek as module
    v = _variables(tmp_path)
    writer = MemoryWriter()
    clock = [0.]
    data = np.arange(60, dtype=float).reshape(5, 12)
    packets = deque([(.5, np.concatenate(([5], data.ravel()))), (1., np.zeros(1)), (2., np.zeros(1))])
    sdk = Mock()
    sdk.init_tdc.return_value = 0
    def packet():
        clock[0], result = packets.popleft()
        return result
    sdk.get_data_tdc_buf.side_effect = packet
    monkeypatch.setattr(module.ctypes, "CDLL", lambda _: object())
    monkeypatch.setattr(module, "TDC", lambda *_args, **_kwargs: sdk)
    monkeypatch.setattr(module, "ChunkWriter", lambda *_args, **_kwargs: writer)
    monkeypatch.setattr(module, "ChunkStream", lambda *args: ChunkStream(*args, chunk_size=3))
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda _: None))
    monkeypatch.setattr(module, "experiment_frequency_hz", lambda _: 1000.)
    monkeypatch.setattr(module, "pulse_energy_pj", lambda _: 12.)
    assert module.experiment_measure(v, *[Mock() for _ in range(4)],
                                     SimpleNamespace(is_set=lambda: not packets)) == 0
    assert v.detection_rate_current == v.detection_rate_current_plot == 0.
    assert v.total_ions == 5 and v.flag_finished_tdc
    for i in range(8):
        np.testing.assert_array_equal(writer.column(f"ch{i}"), data[:, i])
    for name, i in (("x", 8), ("y", 9), ("t", 10), ("time_stamp", 11)):
        np.testing.assert_array_equal(writer.column(name), data[:, i])
    np.testing.assert_array_equal(writer.column("voltage"), np.full(5, 1500.))
    np.testing.assert_array_equal(writer.column("laser_pulse"), np.full(5, 12.))
    v.extend_to.assert_not_called()
    sdk.stop_tdc.assert_called_once()


def _surface_replay(monkeypatch, tmp_path, dld_packets, raw_packets):
    from pyccapt.control.tdc_surface_concept import tdc_surface_concept as module
    v = _variables(tmp_path)
    writer = MemoryWriter()
    clock = [0.]
    queues = [deque(dld_packets), deque(raw_packets)]
    callbacks = []
    def callback(*_args, dld_events):
        index = 0 if dld_events else 1
        def packet(**_kwargs):
            if not queues[index]:
                raise Empty
            clock[0] += .5
            return module.QUEUE_DATA, queues[index].popleft()
        obj = SimpleNamespace(queue=Mock(get=Mock(side_effect=packet)),
                              start_measurement=Mock(return_value=0), close=Mock())
        callbacks.append(obj)
        return obj
    device = Mock(lib=object(), dev_desc=0)
    device.initialize.return_value = (0, "")
    monkeypatch.setattr(module.scTDC, "Device", lambda **_: device)
    monkeypatch.setattr(module, "BufDataCB4", callback)
    monkeypatch.setattr(module, "ChunkWriter", lambda *_args, **_kwargs: writer)
    monkeypatch.setattr(module, "CHUNK_SIZE", 3)
    monkeypatch.setattr(module._runtime, "load_project_config", lambda **_: ({}, tmp_path))
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(module, "experiment_frequency_hz", lambda _: 1000.)
    monkeypatch.setattr(module, "pulse_energy_pj", lambda _: 12.)
    stop = SimpleNamespace(is_set=lambda: not any(queues))
    return module, v, writer, device, callbacks, stop


def test_surface_drains_large_independent_packets_to_bounded_chunks(monkeypatch, tmp_path):
    dld = {name: np.arange(11, dtype=np.uint64) for name in ("dif1", "dif2", "time", "start_counter")}
    raw = {"channel": np.arange(17, dtype=np.uint32) % 8, "time": np.arange(17, dtype=np.uint64),
           "start_counter": np.arange(17, dtype=np.uint64)}
    module, v, writer, device, callbacks, stop = _surface_replay(monkeypatch, tmp_path, [dld], [raw])
    assert module.run_experiment_measure(v, *[Mock() for _ in range(4)], stop) == 0
    assert v.total_ions == 11 and v.total_raw_signals == 17
    assert all(len(next(iter(record[2].values()))) <= 3 for record in writer.records)
    np.testing.assert_array_equal(writer.column("start_counter"), dld["start_counter"])
    np.testing.assert_array_equal(writer.column("time"), raw["time"])
    np.testing.assert_array_equal(writer.column("t"), dld["time"]*(27.432/(1000*4)))
    np.testing.assert_array_equal(writer.column("x"), ((dld["dif1"]-1225.)*(80/4900*2))*.1)
    assert writer.column("time").dtype == np.uint64
    assert callbacks[0].queue.get.call_args.kwargs == {"block": False}
    assert callbacks[1].queue.get.call_args.kwargs == {"block": False}
    v.extend_to.assert_not_called()
    device.deinitialize.assert_called_once()


def test_surface_metadata_failure_is_explicit_and_retains_packet(monkeypatch, tmp_path):
    data = {name: np.arange(5, dtype=np.uint64) for name in ("dif1", "dif2", "time", "start_counter")}
    module, v, writer, device, callbacks, stop = _surface_replay(monkeypatch, tmp_path, [data], [])
    monkeypatch.setattr(module, "read_event_metadata", Mock(side_effect=RuntimeError("metadata unavailable")))
    assert module.run_experiment_measure(v, *[Mock() for _ in range(4)], stop) == -1
    assert v.flag_tdc_failure and v.flag_finished_tdc
    assert "metadata unavailable" in v.detector_error
    evidence = list((tmp_path/"temp_data"/"unattributed").glob("*.npz"))
    assert len(evidence) == 1
    with np.load(evidence[0]) as saved:
        np.testing.assert_array_equal(saved["start_counter"], data["start_counter"])
    assert not writer.records


def test_surface_stop_preserves_queued_tail_without_inventing_voltage(monkeypatch, tmp_path):
    dld = {name: np.arange(5, dtype=np.uint64) for name in ("dif1", "dif2", "time", "start_counter")}
    raw = {"channel": np.arange(7, dtype=np.uint32), "time": np.arange(7, dtype=np.uint64),
           "start_counter": np.arange(7, dtype=np.uint64)}
    module, v, writer, device, callbacks, _ = _surface_replay(monkeypatch, tmp_path, [dld], [raw])
    v.specimen_voltage = 0.  # The experiment owner has already initiated safe-off.
    assert module.run_experiment_measure(v, *[Mock() for _ in range(4)], SimpleNamespace(is_set=lambda: True)) == -1
    assert v.flag_tdc_failure and v.flag_finished_tdc
    assert '2 detector packet(s)' in v.detector_error
    assert v.total_ions == v.total_raw_signals == 0 and not writer.records
    paths = list((tmp_path/'temp_data'/'unattributed').glob('*.npz'))
    assert len(paths) == 2
    for label, expected in (('dld', dld), ('tdc', raw)):
        with np.load(next(path for path in paths if path.name.startswith(label))) as evidence:
            assert 'voltage' not in evidence.files
            for key, values in expected.items():
                np.testing.assert_array_equal(evidence[key], values)


def test_surface_replay_to_real_writer_and_hdf_preserves_committed_events_and_exposes_tail(monkeypatch, tmp_path):
    import h5py
    import json
    from pyccapt.control.core.hdf5_creator import hdf_creator
    from pyccapt.control.core.data_integrity import validate_hdf5
    def dld(ids):
        return {name: np.asarray(ids, dtype=np.uint64) for name in ('dif1', 'dif2', 'time', 'start_counter')}
    def raw(ids):
        return {'channel': np.asarray(ids, dtype=np.uint32) % 8,
                'time': np.asarray(ids, dtype=np.uint64), 'start_counter': np.asarray(ids, dtype=np.uint64)}
    module, v, _, _, _, _ = _surface_replay(monkeypatch, tmp_path, [dld([1, 2, 3]), dld([4, 5])],
                                            [raw([1, 2, 3, 4, 5]), raw([6, 7, 8, 9])])
    monkeypatch.setattr(module, 'ChunkWriter', ChunkWriter)
    assert module.run_experiment_measure(v, *[Mock() for _ in range(4)],
                                         SimpleNamespace(is_set=lambda: v.total_ions >= 3)) == -1
    v.exp_name, v.counter_source = 'replay', 'TDC'
    v.main_counter, v.main_raw_counter = [3], [5]
    v.main_temperature, v.main_chamber_vacuum = [40.], [1e-8]
    hdf_creator(v, {'tdc': 'on', 'tdc_model': 'Surface_Concept'}, [1], [1.])
    result = validate_hdf5(tmp_path/'replay.h5')
    assert result['structurally_valid'] and result['acquisition_completeness'] == 'incomplete'
    with h5py.File(tmp_path/'replay.h5') as handle:
        np.testing.assert_array_equal(handle['dld/start_counter'], [1, 2, 3])
        np.testing.assert_array_equal(handle['tdc/start_counter'], [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(handle['dld/high_voltage'], [1500., 1500., 1500.])
        evidence = json.loads(handle['provenance'].attrs['chunk_integrity_json'])['unattributed_packets']
        assert len(evidence) == 2
    directory = tmp_path/'temp_data'/'unattributed'
    with np.load(next(directory.glob('dld_shutdown*.npz'))) as tail:
        np.testing.assert_array_equal(tail['start_counter'], [4, 5])
    with np.load(next(directory.glob('tdc_shutdown*.npz'))) as tail:
        np.testing.assert_array_equal(tail['start_counter'], [6, 7, 8, 9])


def test_surface_blocked_callback_preserves_inflight_packet_on_stop(tmp_path):
    from pyccapt.control.tdc_surface_concept import tdc_surface_concept as module
    callback = module.BufDataCB4.__new__(module.BufDataCB4)
    callback.queue = Queue(maxsize=1)
    callback.queue.put(('existing', None))
    callback._closing, callback._publication_lock = Event(), Lock()
    callback.unattributed_directory, callback.stream_name = tmp_path, 'dld'
    callback.shutdown_packets, callback.publication_error, callback.end_of_meas = 0, '', False
    entered = Event()
    original_put = callback.queue.put
    def blocked_put(*args, **kwargs):
        entered.set()
        return original_put(*args, **kwargs)
    callback.queue.put = blocked_put
    data = {'time': np.arange(5, dtype=np.uint64)}
    thread = Thread(target=callback.on_data, args=(data,))
    thread.start()
    assert entered.wait(2.)
    callback.stop_publication()
    thread.join(2.)
    assert not thread.is_alive() and callback.shutdown_packets == 1 and not callback.publication_error
    assert callback.queue.get_nowait() == ('existing', None) and callback.queue.empty()
    with np.load(next((tmp_path/'unattributed').glob('*.npz'))) as evidence:
        np.testing.assert_array_equal(evidence['time'], data['time'])


@pytest.mark.parametrize("mode,test_value", [(None, 0), ("normal", 0), ("internal_calibration", 1)])
def test_drs_mode_is_explicit_and_waveform_layout_preserved(monkeypatch, tmp_path, mode, test_value):
    from pyccapt.control.drs import drs as module
    v = _variables(tmp_path)
    writer = MemoryWriter()
    data = (np.arange(8*1024, dtype=np.float32)-1000).reshape(8, 1024)/4
    sdk = Mock()
    sdk.reader.return_value = data.ravel()
    constructor = Mock(return_value=sdk)
    monkeypatch.setattr(module, "DRS", constructor)
    monkeypatch.setattr(module, "ChunkWriter", lambda *_args, **_kwargs: writer)
    conf = {} if mode is None else {"drs_acquisition_mode": mode}
    assert module.experiment_measure(v, *[Mock() for _ in range(4)],
                                     SimpleNamespace(is_set=lambda: v.total_ions >= 2), conf=conf) == 0
    assert constructor.call_args.kwargs["test"] == test_value
    for i in range(4):
        for j, suffix in enumerate(("time", "wave")):
            np.testing.assert_array_equal(writer.column(f"ch{i}_{suffix}"), np.tile(data[2*i+j], 2))
    np.testing.assert_array_equal(writer.column("voltage_drs"), [1500., 1500.])
    assert writer.column("ch0_wave").dtype == np.float32
    v.extend_to.assert_not_called()
    sdk.delete_drs_ox.assert_called_once()


def test_drs_rejects_unknown_mode_before_opening_hardware(monkeypatch, tmp_path):
    from pyccapt.control.drs import drs as module
    constructor = Mock()
    monkeypatch.setattr(module, "DRS", constructor)
    v = _variables(tmp_path)
    assert module.experiment_measure(v, conf={"drs_acquisition_mode": "typo"}) == -1
    assert v.flag_tdc_failure and v.flag_finished_tdc
    constructor.assert_not_called()
