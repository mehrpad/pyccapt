"""Hardware-free alignment tests, including motion and detector-fault boundaries."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from pyccapt.control.apt.alignment_config import AlignmentConfig, coarse_positions
from pyccapt.control.apt.alignment_vision import AlignmentEventPublisher, Footprint, estimate_footprint
from pyccapt.control.apt.automatic_alignment import AutomaticAlignment
from pyccapt.control.devices.alignment_stage import AlignmentStageService, transfer_waypoints


def settings(**changes):
    cfg = AlignmentConfig(motion_calibrated=True, bounds_mm=((-1, 1), (-1, 1), (-1, 1)),
                          xy_range_um=(1., 1.), xy_jacobian_mm_per_um=((1., 0.), (0., 1.)),
                          z_direction=1, transfer_z_mm=-0.5)
    return replace(cfg, **changes)


def test_plot_publisher_is_throttled_and_keeps_percent_units(tmp_path):
    cfg = settings()
    v = state(cfg, tmp_path)
    engine = AutomaticAlignment(v, {'max_vdc': 10000}, now=0)
    engine.phase = 'aligned'  # Monitor keeps following the experiment after alignment.
    engine.tick(1500., .3, now=0)
    reading = v.alignment_plot_snapshot
    assert reading['rate_percent'] == .3
    assert reading['target_percent'] == 1.
    assert reading['position_m'] == (0., 0., 0.)
    engine.tick(1500., .4, now=.1)
    assert v.alignment_plot_snapshot == reading
    engine.tick(1500., .4, now=.25)
    assert v.alignment_plot_snapshot['rate_percent'] == .4
    engine.tick(1500., .5, now=3.)  # Stale stage position must not be paired with a new rate.
    assert v.alignment_plot_snapshot['time'] == .25


def state(cfg, tmp_path):
    return SimpleNamespace(alignment_settings=cfg.snapshot(), alignment_sample_position=(0., 0., 0.),
                           alignment_sample=1, path_meta=str(tmp_path), vdc_min=500, vdc_max=9000,
                           detection_rate=1., specimen_voltage=cfg.start_voltage,
                           detection_rate_current=0., stop_flag=False, start_flag=True,
                           alignment_cancel_motion=False, automatic_alignment_enabled=True,
                           alignment_stage_heartbeat=0., stage_position_snapshot=(0., 0., 0., 0.),
                           alignment_events={}, alignment_window_epoch='', alignment_outcome='',
                           alignment_move_request={}, alignment_move_status={}, hardware_safe=False)


class Rig:
    """Virtual clock and instant motor acknowledgements; no real device imports."""
    def __init__(self, tmp_path, cfg=None, footprint=None):
        self.cfg = cfg or settings()
        self.v = state(self.cfg, tmp_path)
        self.fit = footprint or Footprint(False, 'background')
        self.engine = AutomaticAlignment(self.v, {'max_vdc': 10000}, now=0,
                                        analyser=lambda *args: self.fit)
        self.time = 0
        self.sequence = 0
        self.moves = []

    def step(self, rate=0., fresh=True):
        self.time += 0.5
        self.v.detection_rate_current = rate
        self.v.alignment_stage_heartbeat = self.time
        position = self.v.stage_position_snapshot[:3]
        req = self.v.alignment_move_request
        if req and req['id'] != self.v.alignment_move_status.get('id'):
            position = req['target_m']
            self.moves.append(req)
            self.v.alignment_move_status = {'id': req['id'], 'state': 'done'}
        self.v.stage_position_snapshot = (*position, self.time)
        if fresh:
            self.sequence += 500
            self.v.alignment_events = {'epoch': self.v.alignment_window_epoch, 'sequence': self.sequence,
                                       'time': self.time, 'first_time': self.time,
                                       'points_mm': []}
        self.engine.tick(self.v.specimen_voltage, rate, now=self.time)
        self.v.specimen_voltage += self.engine.voltage_step(self.v.specimen_voltage, 0.5)

    def until(self, predicate, rate=0., limit=2000):
        for _ in range(limit):
            if predicate():
                return
            self.step(rate)
        raise AssertionError(f'Condition not reached: {self.engine.phase}, {self.engine.reason}')


def test_uncalibrated_motion_is_blocked():
    with pytest.raises(ValueError, match='not calibrated'):
        AlignmentConfig().validate_motion([(0, 0, 0)])


def test_spiral_is_bounded_and_never_changes_z():
    cfg = settings(xy_range_um=(2.5, 1.5))
    origin = (1e-4, 2e-4, 3e-4)
    points = list(coarse_positions(origin, cfg))
    assert points[0] == origin
    assert len(points) == len(set(points))
    assert all(p[2] == origin[2] for p in points)
    assert max(abs(p[0]-origin[0]) for p in points) == pytest.approx(2.5e-6)
    assert max(abs(p[1]-origin[1]) for p in points) == pytest.approx(1.5e-6)


def test_requested_stage_ranges_accept_full_coarse_grid():
    from pathlib import Path
    from pyccapt.control.core.read_files import read_toml_file
    conf = read_toml_file(Path(__file__).parents[2]/'pyccapt/config.toml')
    cfg = AlignmentConfig.from_mapping(conf)
    assert cfg.xy_range_um == (50., 50.)
    assert cfg.fine_xy_range_um == 15.
    calibrated = settings(xy_range_um=cfg.xy_range_um, fine_xy_range_um=cfg.fine_xy_range_um)
    calibrated.validate_motion([(0., 0., 0.)])
    assert len(list(coarse_positions((0., 0., 0.), calibrated))) == 10201


def test_fine_range_is_relative_to_coarse_result():
    cfg = settings(xy_range_um=(50., 50.))
    centre = (20e-6, -10e-6, 0.)
    cfg.check_fine_position((35e-6, 5e-6, 0.), centre)
    with pytest.raises(ValueError, match='fine|Fine'):
        cfg.check_fine_position((35.1e-6, 5e-6, 0.), centre)


def test_fine_range_limit_recovers_before_excess_move(tmp_path):
    rig = Rig(tmp_path, settings(xy_range_um=(50., 50.), fine_xy_range_um=.15),
              Footprint(True, 'signal', centre_mm=(5., 0.), radius_mm=10.))
    rig.until(lambda: rig.engine.attempts == 1, rate=.4)
    centre = rig.engine.fine_origin
    rig.until(lambda: any(r['event'] == 'fine_range_reached' for r in
                         [json.loads(line) for line in (tmp_path/'alignment.jsonl').read_text().splitlines()]), rate=.4)
    for move in rig.moves:
        if 'fine_origin_m' in move:
            assert abs(move['target_m'][0]-centre[0])*1e6 <= .15+1e-8


def test_coarse_holds_voltage_and_returns_before_increment(tmp_path):
    rig = Rig(tmp_path)
    rig.until(lambda: rig.engine.target_voltage > 1500)
    assert rig.engine.target_voltage == 1700
    assert rig.moves[-1]['target_m'] == (0., 0., 0.)
    assert all(move['target_m'][2] == 0 for move in rig.moves)
    records = [json.loads(line) for line in (tmp_path/'alignment.jsonl').read_text().splitlines()]
    observations = [r for r in records if r['event'] == 'observation']
    assert observations and all(r['voltage'] == 1500 for r in observations)


def test_voltage_limit_is_capped_and_skips_only_after_final_search(tmp_path):
    rig = Rig(tmp_path, settings(start_voltage=5900))
    rig.until(lambda: bool(rig.engine.outcome))
    assert rig.engine.target_voltage == 6000
    assert rig.v.specimen_voltage == 6000
    assert rig.engine.outcome == 'voltage_limit'
    assert rig.moves[-1]['target_m'] == (0., 0., 0.)


def test_stable_centred_signal_locks_stage_at_80_percent(tmp_path):
    fit = Footprint(True, 'sample', (0., 0.), 12., 0.09, 0.9, 0.02, 1.)
    rig = Rig(tmp_path, footprint=fit)
    rig.until(lambda: rig.engine.phase == 'fine', rate=0.4)
    assert rig.engine.attempts == 1
    rig.until(lambda: rig.engine.phase == 'aligned', rate=0.8)
    n_moves = len(rig.moves)
    for _ in range(20):
        rig.step(0.2)
    assert len(rig.moves) == n_moves
    assert rig.v.alignment_outcome == 'aligned'


def test_replayed_event_window_cannot_establish_stability(tmp_path):
    rig = Rig(tmp_path, footprint=Footprint(True, 'sample', (0., 0.), 10., .06, .9, .01, 1.))
    rig.until(lambda: rig.engine.phase == 'coarse')
    rig.step(.4)
    for _ in range(8):
        rig.step(.4, fresh=False)
    assert rig.engine.attempts == 0


def test_five_fine_attempts_abort_sequence(tmp_path):
    rig = Rig(tmp_path, footprint=Footprint(True, 'sample', (0., 0.), 10., .06, .9, .01, 1.))
    for attempt in range(1, 6):
        rig.until(lambda: rig.engine.phase == 'fine', rate=.4)
        assert rig.engine.attempts == attempt
        rig.until(lambda: rig.engine.phase != 'fine', rate=0.)
        if attempt < 5:
            rig.until(lambda: rig.engine.phase == 'coarse', rate=0.)
    assert rig.engine.outcome == 'attempt_limit'


def test_fine_correction_uses_calibration_and_bounded_step(tmp_path):
    rig = Rig(tmp_path, footprint=Footprint(True, 'sample', (2., -3.), 10., .06, .9, .01, 1.))
    rig.until(lambda: rig.engine.phase == 'fine', rate=.4)
    rig.step(.4)
    request = rig.v.alignment_move_request
    assert request['target_m'] == pytest.approx((-0.1e-6, 0.1e-6, 0.))


def test_loss_after_approach_retracts_before_xy_search(tmp_path):
    rig = Rig(tmp_path, settings(approach_enabled=True, z_max_advance_um=1),
              footprint=Footprint(True, 'sample', (0., 0.), 10., .06, .9, .01, 1.))
    rig.until(lambda: rig.engine.phase == 'fine', rate=.4)
    rig.step(.4)  # approach
    rig.until(lambda: rig.engine.phase == 'fine', rate=.4)
    assert rig.v.stage_position_snapshot[2] > 0
    rig.until(lambda: rig.engine.phase != 'fine', rate=0.)
    assert rig.v.alignment_move_request['target_m'][2] == 0
    rig.until(lambda: rig.engine.phase == 'coarse')
    assert rig.v.stage_position_snapshot[2] == 0


def test_approach_never_exceeds_measured_advance(tmp_path):
    rig = Rig(tmp_path, settings(approach_enabled=True, z_max_advance_um=.05),
              footprint=Footprint(True, 'sample', (0., 0.), 10., .06, .9, .01, 1.))
    rig.until(lambda: bool(rig.engine.outcome), rate=.4)
    assert rig.engine.outcome == 'clearance_limit'
    assert max(m['target_m'][2] for m in rig.moves) <= .05e-6


def test_stale_stage_position_aborts_without_motion(tmp_path):
    rig = Rig(tmp_path)
    rig.engine.tick(1500, .4, now=3)
    assert rig.engine.outcome == 'fault'
    assert not rig.v.alignment_move_request


def disk(rng, n, radius, centre=(0., 0.)):
    angle = rng.uniform(0, 2*np.pi, n)
    r = radius*np.sqrt(rng.uniform(0, 1, n))
    return np.column_stack((r*np.cos(angle), r*np.sin(angle))) + centre


def test_detector_background_and_hotspots_do_not_form_sample():
    rng = np.random.default_rng(42)
    assert not estimate_footprint(disk(rng, 2000, 40), 40).valid
    points = np.concatenate((disk(rng, 1700, 40), np.tile((5., 6.), (300, 1))))
    assert not estimate_footprint(points, 40).valid


def test_footprint_is_found_with_background_and_hotspot():
    rng = np.random.default_rng(12)
    points = np.concatenate((disk(rng, 1550, 12, (5, -4)), disk(rng, 400, 40), np.tile((-15., 15.), (50, 1))))
    fit = estimate_footprint(points, 40)
    assert fit.valid, fit
    assert fit.centre_mm == pytest.approx((5, -4), abs=1.0)
    assert fit.radius_mm == pytest.approx(12, abs=2)
    json.dumps(fit.snapshot(), allow_nan=False)


def test_large_footprint_is_detected_and_radius_uncertainty_prevents_over_approach():
    rng = np.random.default_rng(10)
    fit = estimate_footprint(np.concatenate((disk(rng, 1800, 37.95), disk(rng, 200, 40))), 40)
    assert fit.valid, fit
    assert ((fit.radius_mm + fit.radius_uncertainty_mm)/40)**2 >= .9


def test_clipped_footprint_cannot_authorize_approach():
    rng = np.random.default_rng(2)
    points = disk(rng, 7000, 25, (25, 0))
    points = points[(points**2).sum(axis=1) <= 40**2][:2000]
    assert not estimate_footprint(points, 40).valid


@pytest.mark.parametrize('events_seen,phase,expected', [(False, 'ramp', 1500), (True, 'coarse', 1500),
                                                      (True, 'moving', 1500), (True, 'ramp', 1510)])
def test_experiment_voltage_controller_obeys_alignment_and_first_event_gate(events_seen, phase, expected, monkeypatch):
    from unittest.mock import Mock
    from pyccapt.control.apt import apt_exp_control
    from pyccapt.control.core.share_variables import Variables
    v = SimpleNamespace(**Variables._DEFAULTS)
    v.ex_freq = 10
    v.control_algorithm = 'PID'
    control = apt_exp_control.APT_Exp_Control(v, {}, None, None, None, None, None)
    control.control_algorithm = 'PID'
    control.pid = Mock(side_effect=AssertionError('PID must be suspended during alignment'))
    control.alignment = SimpleNamespace(phase=phase, voltage_step=lambda *_: 10 if phase == 'ramp' else 0)
    control._tdc_first_event_seen = events_seen
    control.specimen_voltage = 1500
    control.vdc_min, control.vdc_max = 500, 9000
    control.pulse_mode = 'Voltage'
    control.pulse_amp_per_supply_voltage = 21.875
    control._vdc_active = lambda: True
    control._vp_active = lambda: False
    monkeypatch.setattr(apt_exp_control.apt_exp_control_func, 'command_v_dc', lambda *args: None)
    control.main_ex_loop()
    assert control.specimen_voltage == expected
    control.pid.assert_not_called()


def test_publisher_excludes_old_epoch_and_keeps_paired_mm_coordinates():
    v = SimpleNamespace(automatic_alignment_enabled=True, alignment_settings={'window_ions': 200},
                        alignment_window_epoch='one')
    publisher = AlignmentEventPublisher(v)
    publisher.append([1]*200, [2]*200, now=0)
    publisher.append([1]*200, [2]*200, now=1)
    assert v.alignment_events['points_mm'][0] == [10., 20.]
    v.alignment_window_epoch = 'two'
    publisher.append([3]*200, [4]*200, now=2)
    publisher.append([3]*200, [4]*200, now=3)
    assert v.alignment_events['epoch'] == 'two'
    assert v.alignment_events['points_mm'] == [[30., 40.]]*200


class Motor:
    def __init__(self):
        self.position = dict(x=0., y=0., z=0.)
        self.moves = []
        self.stopped = False

    def get_position(self):
        return self.position.copy()

    def is_moving(self):
        return False

    def validate_alignment_state(self):
        pass

    def move_absolute(self, **kwargs):
        self.moves.append(kwargs)
        for axis in 'xyz':
            if kwargs[axis+'_m'] is not None:
                self.position[axis] = kwargs[axis+'_m']

    def stop(self):
        self.stopped = True


@pytest.mark.parametrize('target,speed,kind', [((0., 0., 2e-3), 1, 'alignment'),
                                              ((1e-6, 0., 1e-6), 1, 'alignment'),
                                              ((1e-6, 0., 0.), 50, 'alignment'),
                                              ((1e-6, 0., 0.), 1, 'transfer')])
def test_stage_service_rejects_unsafe_requests(tmp_path, target, speed, kind):
    v = state(settings(), tmp_path)
    v.alignment_move_request = dict(id='one', target_m=target, speed_um_s=speed, kind=kind, issued=0)
    motor = Motor()
    service = AlignmentStageService(v, lambda: motor)
    service.tick(now=0)
    assert v.alignment_move_status['state'] == 'error'
    assert not motor.moves
    assert v.alignment_cancel_motion


def test_stage_service_rejects_fine_motion_beyond_15_um(tmp_path):
    v = state(settings(xy_range_um=(50., 50.)), tmp_path)
    v.alignment_move_request = dict(id='fine', target_m=(16e-6, 0., 0.),
                                   fine_origin_m=(0., 0., 0.), speed_um_s=1.,
                                   kind='alignment', issued=0.)
    motor = Motor()
    AlignmentStageService(v, lambda: motor).tick(now=0.)
    assert v.alignment_move_status['state'] == 'error'
    assert not motor.moves


def test_transfer_retracts_traverses_then_approaches_saved_position():
    target = (2e-4, 3e-4, .1e-3)
    points = transfer_waypoints((0, 0, 0), target, settings())
    assert points == [(0, 0, -.5e-3), (target[0], target[1], -.5e-3), target]
