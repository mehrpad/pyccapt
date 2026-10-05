"""Simulated detector and motor tests. No physical hardware is opened."""
from types import SimpleNamespace
from unittest.mock import Mock
import math
import time

import numpy as np
import pytest

from pyccapt.control.apt.laser_alignment import LaserAlignment
from pyccapt.control.apt.laser_alignment_config import LaserAlignmentConfig
from pyccapt.control.apt.laser_alignment_data import LaserAlignmentPublisher, peak_quality
from pyccapt.control.devices.laser_alignment_stage import LaserAlignmentStageService
from pyccapt.control.core.share_variables import Variables


def rig(tmp_path):
    v = Variables({key: 1 for key in Variables._REQUIRED_CONFIG_KEYS}, SimpleNamespace())
    now = time.monotonic()
    for key, value in dict(start_flag=True, stop_flag=False, electrode_out=False,
            pulse_mode='Laser', vdc_min=1000., vdc_max=5000., specimen_voltage=1500.,
            experiment_state='running', path_meta=str(tmp_path), counter_source='TDC',
            laser_stage_snapshot=(0., 0., 0., now), laser_alignment_heartbeat=now,
            experiment_heartbeat_monotonic=now,
            laser_telemetry=dict(valid=True, monotonic=now, status_code=129,
                                 aom_percent=20., output_frequency_hz=1000., wavelength_index=3)).items():
        setattr(v, key, value)
    conf = dict(max_vdc=10000, tdc_model='Surface_Concept', laser_alignment_calibrated=True,
                laser_alignment_bounds_mm=[-1, 1, -1, 1, -1, 1], laser_alignment_travel_um=[30, 30, 10],
                laser_alignment_coarse_range_um=1., laser_alignment_coarse_step_um=1.,
                laser_alignment_fine_range_um=.2, laser_alignment_fine_step_um=.2,
                laser_alignment_focus_range_um=.2, laser_alignment_focus_step_um=.2,
                laser_alignment_settle_s=.2, laser_alignment_dwell_s=.2,
                laser_alignment_min_events=5, laser_alignment_min_rate_percent=.01)
    return v, conf, now


class Simulation:
    def __init__(self, tmp_path, mode='coarse'):
        self.v, self.conf, self.now = rig(tmp_path)
        self.engine = LaserAlignment(self.v, self.conf, mode, self.now)
        self.seen = []

    def tick(self, rate=1.0):
        self.now += .25
        self.v.laser_telemetry = {**self.v.laser_telemetry, 'monotonic': self.now}
        self.v.laser_alignment_heartbeat = self.now
        self.v.laser_stage_snapshot = (*self.v.laser_stage_snapshot[:3], self.now)
        e = self.engine
        self.seen.append(e.phase)
        if e.pending:
            request = self.v.laser_alignment_move_request
            self.v.laser_stage_snapshot = (*request['target_m'], self.now)
            self.v.laser_alignment_move_status = {'id': request['id'], 'state': 'done'}
        elif e.phase in ('coarse', 'fine', 'focus', 'tracking', 'verify'):
            # Large counts reduce uncertainty independently of simulated dwell.
            count = round(rate*1000)
            duration = 100.0
            self.v.laser_alignment_observation = dict(epoch=e.epoch, time=self.now,
                start=self.now-duration, count=count, tof_ns=[])
            # Use an explicit settled measurement rather than spoof a pre-epoch packet.
            e._measurement = lambda: e._record(rate, math.sqrt(count)/1000, count, {})
        elif e.phase == 'ramp':
            self.v.specimen_voltage = e.target_voltage
        e.tick(self.now)


def test_uncalibrated_or_nonfinite_motion_is_rejected(tmp_path):
    v, conf, now = rig(tmp_path)
    conf['laser_alignment_calibrated'] = False
    with pytest.raises(ValueError, match='Calibrate'):
        LaserAlignment(v, conf, now=now)
    cfg = LaserAlignmentConfig.from_config({**conf, 'laser_alignment_calibrated': True})
    with pytest.raises(ValueError, match='finite'):
        cfg.check_position((float('nan'), 0, 0), (0, 0, 0))


def test_full_sequence_and_fixed_dc(tmp_path):
    sim = Simulation(tmp_path)
    for _ in range(200):
        assert sim.engine.voltage_step(1500, .1) in (0., None)
        sim.tick()
        if sim.engine.phase == 'tracking_wait': break
    assert sim.engine.phase == 'tracking_wait', sim.engine.reason
    assert all(phase in sim.seen for phase in ('coarse', 'fine', 'focus'))
    assert sim.seen.index('focus') > sim.seen.index('fine')
    assert 'fine' in sim.seen[sim.seen.index('focus')+1:]
    assert sim.engine.best_position == (0., 0., 0.)


def test_unsuccessful_scan_returns_before_editable_increment_and_clamps(tmp_path):
    sim = Simulation(tmp_path)
    sim.engine.cfg.voltage_increment = 75
    for _ in range(100):
        sim.tick(rate=0)
        if sim.engine.phase == 'ramp': break
    assert sim.v.laser_stage_snapshot[:3] == sim.engine.centre
    assert sim.engine.target_voltage == 1575
    assert sim.engine.voltage_step(1500, .1) == 2
    sim.engine.target_voltage = sim.engine.limit-10
    sim.engine._retry()
    assert sim.engine.target_voltage == sim.engine.limit
    sim.engine._retry()
    assert sim.engine.finished and 'ceiling' in sim.engine.reason


def test_three_fine_failures_trigger_coarse_recovery(tmp_path):
    sim = Simulation(tmp_path, 'fine')
    for _ in range(150):
        sim.tick(rate=0)
        if sim.engine.recoveries: break
    assert sim.engine.recoveries == 1
    assert sim.engine.kind == 'coarse'


@pytest.mark.parametrize('field,value,reason', [
    ('electrode_out', True, 'interlock'), ('flag_tdc_failure', True, 'interlock'),
    ('pulse_mode', 'Voltage', 'changed'), ('laser_alignment_heartbeat', 0, 'unresponsive'),
    ('specimen_voltage', 1600., 'DC changed'),
])
def test_interlocks_stop_search(tmp_path, field, value, reason):
    v, conf, now = rig(tmp_path)
    e = LaserAlignment(v, conf, now=now)
    setattr(v, field, value)
    e.tick(now+.1)
    assert e.phase == 'failed' and reason in e.reason
    assert v.laser_alignment_cancel


def test_cancel_and_dc_hold_release_voltage(tmp_path):
    v, conf, now = rig(tmp_path)
    e = LaserAlignment(v, conf, now=now)
    v.vdc_hold = True
    e.tick(now+.1)
    assert e.finished and not e.owns_voltage
    assert e.voltage_step(1500, .1) is None


def test_epoch_discards_old_packet_and_preserves_rate_denominator():
    v = SimpleNamespace(laser_alignment_epoch='one', laser_alignment_observation={})
    publisher = LaserAlignmentPublisher(v)
    publisher.append(np.ones(100), now=10.)
    assert not v.laser_alignment_observation
    publisher.append(np.ones(20), now=10.5)
    assert v.laser_alignment_observation['count'] == 20
    v.laser_alignment_epoch = 'two'
    publisher.append(np.ones(999), now=11.)
    publisher.append(np.ones(3), now=11.5)
    assert v.laser_alignment_observation['epoch'] == 'two'
    assert v.laser_alignment_observation['count'] == 3


def test_old_epoch_or_pre_settle_measurements_do_not_score(tmp_path):
    v, conf, now = rig(tmp_path)
    e = LaserAlignment(v, conf, now=now)
    e._observe()
    v.laser_alignment_observation = dict(epoch='old', time=now+1, start=now, count=100)
    assert e._measurement() is None
    v.laser_alignment_observation = dict(epoch=e.epoch, time=now+1, start=now-1, count=100)
    assert e._measurement() is None
    e.now = now+1
    v.laser_alignment_observation = dict(epoch=e.epoch, time=now+1, start=now, count=10)
    result = e._measurement()
    assert result['rate_percent'] == 1.0


def test_quality_is_unavailable_without_peak_and_measured_with_enough_events():
    cfg = LaserAlignmentConfig()
    points = np.linspace(95, 105, 1000)
    assert peak_quality(points, cfg) == {}
    cfg.quality_peak_ns = (90, 110)
    assert peak_quality(points[:3], cfg) == {}
    assert peak_quality(points, cfg)['width_ns'] == pytest.approx(8)


class Stage:
    def __init__(self):
        self.position = dict(x=0., y=0., z=0.)
        self.stop = Mock()
        self.moves = []
    def validate_alignment_state(self): pass
    def get_position(self): return self.position
    def is_moving(self): return False
    def move_absolute(self, **kw):
        self.moves.append(kw)
        for axis in 'xyz':
            if kw[axis+'_m'] is not None: self.position[axis] = kw[axis+'_m']


def test_device_acknowledges_settled_position_and_cancels_on_experiment_stop(tmp_path):
    v, conf, now = rig(tmp_path)
    e = LaserAlignment(v, conf, now=now)
    stage = Stage()
    service = LaserAlignmentStageService(v, conf, lambda: stage)
    service.tick(now)
    assert v.laser_alignment_move_status['state'] == 'moving'
    service.tick(now+.3)
    assert v.laser_alignment_move_status['state'] == 'done'
    e.now = now+.4
    e._move((1e-6, 0, 0), 'observe')
    service.tick(now+.4)
    v.start_flag = False
    service.tick(now+.5)
    assert stage.stop.called and v.laser_alignment_move_status['state'] == 'error'


@pytest.mark.parametrize('changes', [
    {'target_m': (2e-3, 0, 0)}, {'target_m': (1e-6, 0, 1e-6)},
    {'speed_um_s': 10000}, {'issued': 0}, {'session': 'other'},
])
def test_device_rejects_bad_motion_requests(tmp_path, changes):
    v, conf, now = rig(tmp_path)
    LaserAlignment(v, conf, now=now)
    v.laser_alignment_move_request = {**v.laser_alignment_move_request, **changes}
    stage = Stage()
    service = LaserAlignmentStageService(v, conf, lambda: stage)
    service.tick(now)
    assert v.laser_alignment_move_status['state'] == 'error'
    assert not stage.moves


def test_motion_fault_ends_experiment_instead_of_becoming_operator_cancel(tmp_path):
    v, conf, now = rig(tmp_path)
    e = LaserAlignment(v, conf, now=now)
    v.laser_alignment_move_status = dict(id=e.pending[0], state='error', fault=True, error='Motor fault')
    v.laser_alignment_cancel = True
    e.tick(now+.1)
    assert e.phase == 'failed' and e.reason == 'Motor fault'


def test_dead_experiment_stops_stage(tmp_path):
    v, conf, now = rig(tmp_path)
    LaserAlignment(v, conf, now=now)
    v.experiment_heartbeat_monotonic = now-10
    stage = Stage()
    service = LaserAlignmentStageService(v, conf, lambda: stage)
    service.tick(now)
    assert not stage.moves and stage.stop.called
    assert v.laser_alignment_move_status['fault']


def test_periodic_tracking_keeps_unimproved_position(tmp_path):
    sim = Simulation(tmp_path)
    for _ in range(200):
        sim.tick()
        if sim.engine.phase == 'tracking_wait': break
    original = sim.engine.best_position
    sim.v.laser_alignment_tracking = True
    sim.now += sim.engine.cfg.tracking_interval_s
    sim.tick()
    assert sim.engine.kind == 'tracking'
    for _ in range(30):
        sim.tick()
        if sim.engine.phase == 'tracking_wait': break
    assert sim.engine.phase == 'tracking_wait'
    assert sim.engine.best_position == original


def test_runtime_stop_cancels_pending_auto_request_and_resets_pid(tmp_path):
    from pyccapt.control.apt.laser_alignment_runtime import LaserAlignmentRuntime
    v, conf, now = rig(tmp_path)
    owner = SimpleNamespace(variables=v, conf=conf, log_apt=Mock(), _switch_control_algorithm=Mock())
    runtime = LaserAlignmentRuntime(owner)
    runtime.engine = LaserAlignment(v, conf, now=now)
    v.laser_alignment_command = dict(id='stop-1', mode='stop', issued=now)
    assert runtime.tick()
    assert runtime.engine.finished
    assert not runtime.auto_pending
    owner._switch_control_algorithm.assert_called_once()


@pytest.mark.parametrize('laser_step,hold,expected', [(0., False, 1500), (2., False, 1502), (2., True, 1500)])
def test_experiment_owns_laser_scan_voltage_and_suspends_pid(monkeypatch, laser_step, hold, expected):
    from pyccapt.control.apt import apt_exp_control
    v = SimpleNamespace(**Variables._DEFAULTS)
    v.ex_freq = 10
    v.control_algorithm = 'PID'
    v.vdc_hold = hold
    controller = apt_exp_control.APT_Exp_Control(v, {}, None, None, None, None, None)
    controller.control_algorithm = 'PID'
    controller.pid = Mock(side_effect=AssertionError('PID may not integrate during a scan'))
    controller.laser_alignment_runtime = SimpleNamespace(voltage_step=lambda *_: laser_step)
    controller._tdc_first_event_seen = False
    controller.specimen_voltage = 1500
    controller.vdc_min, controller.vdc_max = 500, 9000
    controller.pulse_mode = 'Laser'
    controller._vdc_active = lambda: True
    controller._vp_active = lambda: False
    monkeypatch.setattr(apt_exp_control.apt_exp_control_func, 'command_v_dc', lambda *args: None)
    controller.main_ex_loop()
    assert controller.specimen_voltage == expected
    controller.pid.assert_not_called()


def test_aligned_experiment_can_exceed_search_ceiling_and_tracking_waits(tmp_path):
    sim = Simulation(tmp_path)
    sim.engine.pending = None
    sim.engine.phase = 'tracking_wait'
    sim.engine.wait_started = sim.now-100
    sim.v.laser_alignment_tracking = True
    sim.v.specimen_voltage = 7000
    sim.tick()
    assert sim.engine.phase == 'tracking_wait'
    assert 'paused' in sim.engine.reason
    assert not sim.engine.owns_voltage


def test_metadata_failure_finishes_and_cancels_motion(tmp_path):
    v, conf, now = rig(tmp_path)
    e = LaserAlignment(v, conf, now=now)
    e._event = Mock(side_effect=OSError('disk full'))
    e.finish('stop')
    assert v.laser_alignment_cancel and e.finished
    assert e.phase == 'failed' and 'disk full' in e.reason


@pytest.mark.parametrize('verified_rate,retained', [(.86, False), (1.04, False), (1.08, True)])
def test_tracking_verification_must_preserve_baseline_improvement(tmp_path, verified_rate, retained):
    v, conf, now = rig(tmp_path)
    e = LaserAlignment(v, conf, now=now)
    e.pending = None
    e.kind = 'tracking'
    baseline = e._record(1., .003, 100000, {})
    candidate = e._record(1.06, .003, 106000, {})
    candidate['position_m'] = (1e-6, 0., 0.)
    e.records = [baseline, candidate, baseline.copy()]
    e._complete_scan()
    v.laser_stage_snapshot = (1e-6, 0., 0., now)
    e.pending = None
    e._accepted(e._record(verified_rate, .003, int(verified_rate*100000), {}))
    if retained:
        assert e.phase == 'tracking_wait' and e.best_position == (1e-6, 0., 0.)
    else:
        assert e.pending[2] == 'retry'
        assert v.laser_alignment_move_request['target_m'] == e.centre


@pytest.mark.parametrize('kind,axis', [('fine', 0), ('fine', 1), ('focus', 2), ('tracking', 0)])
@pytest.mark.parametrize('sign', [-1, 1])
@pytest.mark.parametrize('absolute_bound', [False, True])
def test_scans_clip_to_absolute_and_session_travel_boundaries(tmp_path, kind, axis, sign, absolute_bound):
    v, conf, now = rig(tmp_path)
    conf['laser_alignment_travel_um'] = [1., 1., 1.]
    if absolute_bound:
        conf['laser_alignment_bounds_mm'] = [-.001, .001]*3
    e = LaserAlignment(v, conf, now=now)
    centre = [0., 0., 0.]
    centre[axis] = sign*1e-6
    v.laser_stage_snapshot = (*centre, now)
    e._begin_scan(kind, tuple(centre))
    points = [v.laser_alignment_move_request['target_m'], *e.points]
    assert points[0] == points[-1] == tuple(centre)
    assert any(point[axis] != centre[axis] for point in points)
    for point in points:
        e.cfg.check_position(point, e.origin)
        assert all(-1e-6 <= coordinate <= 1e-6 for coordinate in point)


def test_auto_laser_waits_for_stage_alignment_then_takes_voltage_ownership(tmp_path):
    from pyccapt.control.apt.laser_alignment_runtime import LaserAlignmentRuntime
    v, conf, now = rig(tmp_path)
    v.automatic_alignment_enabled = True
    v.alignment_status = {'phase': 'coarse'}
    v.laser_alignment_enabled = True
    owner = SimpleNamespace(variables=v, conf=conf, log_apt=Mock(), _switch_control_algorithm=Mock())
    runtime = LaserAlignmentRuntime(owner)
    assert runtime.tick() and runtime.engine is None
    assert runtime.voltage_step(1500., .1) is None
    assert not v.laser_alignment_move_request
    v.alignment_status = {'phase': 'aligned'}
    assert runtime.tick()
    assert runtime.engine is not None and runtime.engine.owns_voltage
    assert runtime.voltage_step(1500., .1) == 0.
    assert v.laser_alignment_move_request
