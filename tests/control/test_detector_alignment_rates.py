"""Run each real detector loop with fake SDK batches and a virtual clock."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


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
        sdk = Mock()
        sdk.init_tdc.return_value = 0
        def packet():
            clock[0] = elapsed_s
            return np.concatenate(([count], np.ones((count, 12)).ravel()))
        sdk.get_data_tdc_buf.side_effect = packet
        monkeypatch.setattr(module.ctypes, 'CDLL', lambda *_: object())
        monkeypatch.setattr(module, 'TDC', lambda *_args, **_kwargs: sdk)
        run = module.experiment_measure
    else:
        from pyccapt.control.tdc_surface_concept import tdc_surface_concept as module
        device = Mock(lib=object(), dev_desc=0)
        device.initialize.return_value = (0, '')
        monkeypatch.setattr(module.scTDC, 'Device', lambda **_kwargs: device)
        def callback(*_args, dld_events):
            def packet(**_kwargs):
                if dld_events:
                    clock[0] = elapsed_s
                    data = {key: np.ones(count) for key in ('dif1', 'dif2', 'time', 'start_counter')}
                    return module.QUEUE_DATA, data
                return None, None
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
