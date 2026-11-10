"""Run each real detector loop with fake SDK batches and a virtual clock."""
from types import SimpleNamespace
from unittest.mock import Mock
from queue import Empty

import numpy as np
import pytest


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


@pytest.mark.parametrize('backend', ['surface', 'roentdek'])
@pytest.mark.parametrize('elapsed_s', [.5, 1., 2.5])
def test_detector_rate_uses_actual_elapsed_time(monkeypatch, tmp_path, backend, elapsed_s):
    count = 100
    clock = [0.]
    v = SimpleNamespace(path=str(tmp_path), ex_freq=10., specimen_voltage=1500., pulse_voltage=0.,
                        pulse_mode='Voltage', flag_stop_tdc=False, total_ions=0,
                        automatic_alignment_enabled=False, laser_alignment_epoch='', extend_to=Mock())
    stop = SimpleNamespace(is_set=lambda: v.total_ions >= count)
    rings = [Mock() for _ in range(4)]

    if backend == 'roentdek':
        from pyccapt.control.tdc_roentdek import tdc_roentdek as module

        def packet():
            clock[0] = elapsed_s
            return _roentdek_groups(count)
        sdk = _roentdek_fake_sdk(packet)
        monkeypatch.setattr(module.ctypes, 'CDLL', lambda *_: object())
        monkeypatch.setattr(module, 'TDC', lambda *_args, **_kwargs: sdk)
        monkeypatch.setattr(module, 'make_reconstructor', _roentdek_fake_reco)
        run = module.experiment_measure
    else:
        from pyccapt.control.tdc_surface_concept import tdc_surface_concept as module
        device = Mock(lib=object(), dev_desc=0)
        device.initialize.return_value = (0, '')
        monkeypatch.setattr(module.scTDC, 'Device', lambda **_kwargs: device)
        def callback(*_args, dld_events):
            delivered = False
            def packet(**_kwargs):
                nonlocal delivered
                if dld_events and not delivered:
                    delivered = True
                    clock[0] = elapsed_s
                    data = {key: np.ones(count) for key in ('dif1', 'dif2', 'time', 'start_counter')}
                    return module.QUEUE_DATA, data
                raise Empty
            return SimpleNamespace(queue=Mock(get=Mock(side_effect=packet)),
                                   start_measurement=Mock(return_value=0), close=Mock())
        monkeypatch.setattr(module, 'BufDataCB4', callback)
        save_worker = Mock(exitcode=0)
        monkeypatch.setattr(module, 'mp', SimpleNamespace(Queue=Mock(return_value=Mock()),
                                                        Process=Mock(return_value=save_worker)))
        monkeypatch.setattr(module._runtime, 'load_project_config', lambda **_kwargs: ({}, tmp_path))
        run = module.run_experiment_measure

    # Replace only this module's clock; detector publishers retain their own clocks.
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: clock[0],
                                                      time=lambda: clock[0], sleep=lambda *_: None))
    monkeypatch.setattr(module, 'experiment_frequency_hz', lambda *_: 1000.)
    monkeypatch.setattr(module, 'pulse_energy_pj', lambda *_: 0.)
    assert run(v, *rings, stop) == 0
    assert v.total_ions == count and v.flag_finished_tdc
    assert v.detection_rate_current == pytest.approx(count/elapsed_s/1000.*100.)
    assert v.detection_rate_current_plot == v.detection_rate_current
    assert len(rings[0].write.call_args.args[0]) == count
