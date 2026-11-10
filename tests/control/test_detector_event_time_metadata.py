"""Voltage/laser per detector event at the event's own time, and rate on event time.

Run 283 (RoentDek): a burst of ~135k groups in 3 s was read late in five 30000-group packets.
Each packet got the single voltage of its read moment, and the backlog flush looked like a huge
rate, so the controller dropped the voltage that the late packets were then labelled with.
"""
from collections import deque
from queue import Empty
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from pyccapt.control.apt.detector_chunks import (
    ChunkStream, DetectionRateWindow, DetectorClock, MetadataHistory,
)


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


def test_history_interpolates_and_clamps():
    history = MetadataHistory()
    history.append(0., (1000., 10., 1.))
    history.append(1., (2000., 20., 2.))
    dc, pulse, laser = history.at([-1., .25, 1., 5.])
    np.testing.assert_allclose(dc, [1000., 1250., 2000., 2000.])
    np.testing.assert_allclose(pulse, [10., 12.5, 20., 20.])
    np.testing.assert_allclose(laser, [1., 1.25, 2., 2.])


def test_history_keeps_unknown_laser_unknown():
    history = MetadataHistory()
    history.append(0., (1000., 0., 5.))
    history.append(1., (1000., 0., np.nan))
    history.append(2., (1000., 0., 7.))
    _, _, laser = history.at([-.5, .5, 1.5, 3.])
    assert laser[0] == 5. and np.isnan(laser[1]) and np.isnan(laser[2]) and laser[3] == 7.


def test_history_ignores_out_of_order_samples_and_needs_one_sample():
    history = MetadataHistory()
    with pytest.raises(RuntimeError):
        history.at([0.])
    history.append(2., (1., 0., 0.))
    history.append(1., (99., 0., 0.))
    assert len(history) == 1 and history.at([1.5])[0][0] == 1.


def test_detector_clock_offset_is_the_smallest_read_delay():
    clock = DetectorClock()
    clock.observe(10.05, 0.)    # read 50 ms after the last event
    clock.observe(12.00, 1.)    # backlogged read: 1 s late
    clock.observe(11.52, 1.5)   # 20 ms after
    assert clock.offset == pytest.approx(10.02)
    np.testing.assert_allclose(clock.host_time([0., 1.]), [10.02, 11.02])


def test_rate_window_flushed_backlog_is_not_a_spike():
    window = DetectionRateWindow()
    window.add(100, 1.0)
    assert window.rate(1.0, 1000.) == pytest.approx(10.)       # first window: host time
    # 1000 ions that happened over 10 s of event time, all read in one 0.5 s host window
    window.add(1000, 11.0)
    assert window.rate(.5, 1000.) == pytest.approx(10.)        # not 200 %
    assert window.rate(.5, 1000.) == 0.                        # silent detector
    window.add(50, 30.)
    assert window.rate(.5, 1000.) == pytest.approx(10.)        # after silence: host time again


def _roentdek_groups(n, first_ts):
    ts = np.arange(first_ts, first_ts + n, dtype=np.uint64)
    cnt = np.ones((n, 8), np.uint32)
    tdc = np.zeros((n, 8, 10), np.int32)
    tdc[:, :6, 0] = (np.arange(n)[:, None] + 1) * np.arange(1, 7)[None, :]
    tdc[:, 6, 0] = -1000 - np.arange(n)
    return ts, cnt, tdc


def _fake_reco(*_args, **_kwargs):
    def reconstruct(cnt, tdc, time_stamp=None):
        n = len(cnt)
        return {"event": np.arange(n), "hit": np.zeros(n, np.int8), "x": tdc[:, 0, 0].astype(float),
                "y": tdc[:, 2, 0].astype(float), "t": (tdc[:, 7, 0] - tdc[:, 6, 0]).astype(float),
                "method": np.zeros(n, np.int8), "time_stamp": time_stamp}
    return SimpleNamespace(bin_ns=.025, reconstruct=reconstruct, stats_text=lambda: "", signal_warnings=lambda: [])


def test_roentdek_late_packet_keeps_the_voltage_of_its_tdc_time(monkeypatch, tmp_path):
    from pyccapt.control.tdc_roentdek import tdc_roentdek as module
    bins_per_s = int(1 / .025e-9)
    v = SimpleNamespace(path=str(tmp_path), ex_freq=10., specimen_voltage=1500., pulse_voltage=0.,
                        pulse_mode="Laser", flag_stop_tdc=False, stop_flag=False, total_ions=0,
                        total_raw_signals=0, automatic_alignment_enabled=False, laser_alignment_epoch="")
    clock = [0.]
    on_time = _roentdek_groups(5, 1 * bins_per_s)                     # happened at TDC 1.0 s
    burst = _roentdek_groups(5, int(1.2 * bins_per_s))                # happened at TDC 1.2 s ...
    packets = deque([(1.0, 1500., on_time), (3.0, 800., burst), (3.6, 800., None)])   # ... read at 3.0 s

    def read():
        clock[0], v.specimen_voltage, packet = packets.popleft()
        return packet

    class Sampler:   # the controller lowered the voltage after the burst; the sampler thread saw it
        def __init__(self, history, *_args, **_kwargs):
            self.history = history

        def start(self):
            for t, dc in ((0., 1500.), (1.2, 1500.), (2.0, 1000.), (2.5, 900.)):
                self.history.append(t, (dc, 0., 12.))
            return self

        def stop(self):
            pass

    sdk = Mock()
    sdk.init_tdc.return_value = sdk.run_tdc.return_value = 0
    sdk.bin_size_ns.return_value = .025
    sdk.diagnostics.return_value, sdk.last_error.return_value = {}, ''
    sdk.read.side_effect = read
    writer = MemoryWriter()
    monkeypatch.setattr(module.ctypes, "CDLL", lambda _: object())
    monkeypatch.setattr(module, "TDC", lambda *_a, **_k: sdk)
    monkeypatch.setattr(module, "make_reconstructor", _fake_reco)
    monkeypatch.setattr(module, "MetadataSampler", Sampler)
    monkeypatch.setattr(module, "ChunkWriter", lambda *_a, **_k: writer)
    monkeypatch.setattr(module, "ChunkStream", lambda *args: ChunkStream(*args, chunk_size=3))
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda _: None))
    monkeypatch.setattr(module, "experiment_frequency_hz", lambda _: 1000.)
    monkeypatch.setattr(module, "pulse_energy_pj", lambda _: 12.)
    assert module.experiment_measure(v, *[Mock() for _ in range(4)],
                                     SimpleNamespace(is_set=lambda: not packets)) == 0
    # the old code labelled the burst with the voltage at its read time (800 V)
    np.testing.assert_allclose(writer.column("voltage"), np.full(10, 1500.))
    np.testing.assert_allclose(writer.column("ro_raw_voltage"), np.full(80, 1500.))
    np.testing.assert_allclose(writer.column("laser_pulse"), np.full(10, 12.))
    assert v.total_ions == 10 and v.detection_rate_current == 0.


def test_surface_packet_keeps_the_voltage_of_its_arrival(monkeypatch, tmp_path):
    from pyccapt.control.tdc_surface_concept import tdc_surface_concept as module
    v = SimpleNamespace(path=str(tmp_path), ex_freq=10., specimen_voltage=800., pulse_voltage=50.,
                        pulse_mode="Voltage", flag_stop_tdc=False, stop_flag=False, total_ions=0,
                        total_raw_signals=0, automatic_alignment_enabled=False, laser_alignment_epoch="")
    dld = {name: np.arange(4, dtype=np.uint64) for name in ("dif1", "dif2", "time", "start_counter")}
    raw = {"channel": np.arange(6, dtype=np.uint32), "time": np.arange(6, dtype=np.uint64),
           "start_counter": np.arange(6, dtype=np.uint64)}
    # stamped by the callback at arrival (1500 V), consumed after the controller went to 800 V
    queues = [deque([(module.QUEUE_DATA, dld, (1., (1500., 40., 0.)))]),
              deque([(module.QUEUE_DATA, raw, (1., (1500., 40., 0.)))])]
    writer = MemoryWriter()
    clock = [5.]

    def callback(*_args, dld_events):
        index = 0 if dld_events else 1

        def get(**_kwargs):
            if not queues[index]:
                raise Empty
            return queues[index].popleft()
        return SimpleNamespace(queue=Mock(get=Mock(side_effect=get)), start_measurement=Mock(return_value=0),
                               close=Mock())

    device = Mock(lib=object(), dev_desc=0)
    device.initialize.return_value = (0, "")
    monkeypatch.setattr(module.scTDC, "Device", lambda **_: device)
    monkeypatch.setattr(module, "BufDataCB4", callback)
    monkeypatch.setattr(module, "ChunkWriter", lambda *_a, **_k: writer)
    monkeypatch.setattr(module, "CHUNK_SIZE", 3)
    monkeypatch.setattr(module._runtime, "load_project_config", lambda **_: ({}, tmp_path))
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(module, "experiment_frequency_hz", lambda _: 1000.)
    monkeypatch.setattr(module, "pulse_energy_pj", lambda _: 0.)
    assert module.run_experiment_measure(v, *[Mock() for _ in range(4)],
                                         SimpleNamespace(is_set=lambda: not any(queues))) == 0
    np.testing.assert_allclose(writer.column("voltage"), np.full(4, 1500.))
    np.testing.assert_allclose(writer.column("voltage_pulse"), np.full(4, 40.))
    np.testing.assert_allclose(writer.column("voltage_tdc"), np.full(6, 1500.))


def test_surface_callback_stamps_arrival_metadata():
    from threading import Event, Lock
    from queue import Queue
    from pyccapt.control.tdc_surface_concept import tdc_surface_concept as module
    callback = module.BufDataCB4.__new__(module.BufDataCB4)
    callback.queue = Queue()
    callback._closing, callback._publication_lock = Event(), Lock()
    callback.end_of_meas = False
    callback.metadata_reader = lambda: (1234., 5., 6.)
    callback.on_data({"time": np.arange(3, dtype=np.uint64)})
    kind, data, stamp = callback.queue.get_nowait()
    assert kind == module.QUEUE_DATA and stamp[1] == (1234., 5., 6.)
    np.testing.assert_array_equal(data["time"], [0, 1, 2])
    # a failing reader leaves the packet for the consumer (fallback or unattributed), never drops it
    callback.metadata_reader = Mock(side_effect=RuntimeError("proxy gone"))
    callback.on_data({"time": np.arange(2, dtype=np.uint64)})
    assert callback.queue.get_nowait()[2][1] is None
