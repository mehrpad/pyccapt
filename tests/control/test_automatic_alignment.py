"""Hardware-free alignment tests, including motion and detector-fault boundaries."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from pyccapt.control.apt.alignment_config import AlignmentConfig, coarse_positions
from pyccapt.control.apt.alignment_vision import AlignmentEventPublisher, Footprint, estimate_dense_region, estimate_footprint
from pyccapt.control.apt.automatic_alignment import AutomaticAlignment
from pyccapt.control.devices.alignment_stage import AlignmentStageService, transfer_waypoints


def settings(**changes):
    cfg = AlignmentConfig(motion_calibrated=True, bounds_mm=((-1, 1), (-1, 1), (-1, 1)),
                          xy_range_um=(1., 1.), xy_step_um=1., xy_jacobian_mm_per_um=((1., 0.), (0., 1.)),
                          approach_enabled=False, z_max_advance_um=0.,
                          z_direction=1, transfer_z_mm=-0.5)
    return replace(cfg, **changes)


def legacy_snapshot(cfg):
    """Settings published by a GUI started before the relative-search update."""
    values = cfg.snapshot()
    for name in ('coarse_dwell_s', 'coarse_max_dwell_s', 'search_min_events',
                 'jump_ratio', 'jump_sigma', 'fine_loss_ratio', 'fine_probe_step_um'):
        values.pop(name)
    values.update(entry_fraction=.3, loss_fraction=.1)
    return values


def test_legacy_settings_start_engine_and_preserve_explicit_limits(tmp_path):
    cfg = settings(voltage_increment=200., max_voltage=3000.)
    v = state(cfg, tmp_path)
    v.alignment_settings = legacy_snapshot(cfg)
    engine = AutomaticAlignment(v, {'max_vdc': 10000}, now=0)
    assert engine.cfg == cfg
    assert 'entry_fraction' not in engine.cfg.snapshot()
    assert 'loss_fraction' not in engine.cfg.snapshot()
    # Parsing must not mutate the settings shared with the running GUI.
    assert v.alignment_settings['entry_fraction'] == .3


@pytest.mark.parametrize('changes, error, message', [
    ({'z_max_advance_um': -1.}, ValueError, 'tolerance'),
    ({'approach_enabled': 'false'}, ValueError, 'booleans'),
    ({'z_max_advnce_um': 20.}, TypeError, 'z_max_advnce_um'),
])
def test_legacy_settings_still_reject_invalid_and_unknown_values(tmp_path, changes, error, message):
    v = state(settings(), tmp_path)
    v.alignment_settings = {**legacy_snapshot(settings()), **changes}
    with pytest.raises(error, match=message):
        AutomaticAlignment(v, {'max_vdc': 10000}, now=0)


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
                           alignment_move_request={}, alignment_move_status={}, hardware_safe=False,
                           physical_estop_ok=True, experiment_state='running',
                           experiment_heartbeat_monotonic=0., electrode_out=False, flag_tdc_failure=False)


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
        position = self.v.stage_position_snapshot[:3]
        if self.engine.phase in ('coarse', 'candidate', 'neighbour') and np.linalg.norm(position[:2]) > 1e-10:
            rate *= .25  # Nearby background; the saved origin is a repeatable spatial maximum.
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
                                       'points_mm': [(0., 0.)]*self.cfg.window_ions}
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
    assert cfg.motion_calibrated
    assert cfg.z_direction == 1
    assert cfg.transfer_z_mm == -4.
    assert cfg.transfer_speed_um_s == 300.
    assert cfg.xy_speed_um_s == 100.
    assert cfg.fine_xy_speed_um_s == 16.
    assert cfg.xy_step_um == 10.
    cfg.validate_motion([(0., 0., -3e-3)])
    calibrated = settings(xy_range_um=cfg.xy_range_um, fine_xy_range_um=cfg.fine_xy_range_um,
                          xy_step_um=cfg.xy_step_um)
    calibrated.validate_motion([(0., 0., 0.)])
    assert len(list(coarse_positions((0., 0., 0.), calibrated))) == 33
    assert cfg.voltage_increment == 100.
    assert cfg.approach_enabled and cfg.z_max_advance_um == 20.


def test_fine_range_is_relative_to_coarse_result():
    cfg = settings(xy_range_um=(50., 50.), xy_step_um=10.)
    centre = (20e-6, -10e-6, 0.)
    cfg.check_fine_position((35e-6, 5e-6, 0.), centre)
    with pytest.raises(ValueError, match='fine|Fine'):
        cfg.check_fine_position((35.1e-6, 5e-6, 0.), centre)


def test_fine_range_limit_recovers_before_excess_move(tmp_path):
    rig = Rig(tmp_path, settings(xy_range_um=(50., 50.), xy_step_um=10., fine_xy_range_um=.15, fine_probe_step_um=.1),
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
    assert rig.engine.target_voltage == 1600
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


def test_initial_ramp_holds_at_target_rate_and_enters_fine_at_that_voltage(tmp_path):
    fit = Footprint(True, 'sample', (2., 0.), 10., .06, .9, .01, 1.)
    rig = Rig(tmp_path, footprint=fit)
    rig.v.specimen_voltage = 1000.
    rig.step(rate=1.)
    assert rig.engine.phase == 'confirming'
    assert rig.engine.target_voltage == 1000.
    assert rig.v.specimen_voltage == 1000.
    assert rig.engine.voltage_step(1000., .2) == 0.
    rig.until(lambda: rig.engine.phase == 'fine', rate=1.)
    assert rig.v.specimen_voltage == 1000.
    assert rig.engine.target_voltage == 1000.
    assert rig.engine.attempts == 1
    assert not rig.moves  # No coarse scan before the successful early entry.
    records = [json.loads(line) for line in (tmp_path/'alignment.jsonl').read_text().splitlines()]
    assert any(r['event'] == 'initial_ramp_target_reached' and r['voltage'] == 1000.
               and r['configured_start_voltage'] == 1500. for r in records)
    assert any(r['event'] == 'fine_started' and r['source'] == 'initial_ramp' for r in records)


def test_initial_ramp_does_not_stop_at_a_fraction_of_target_rate(tmp_path):
    rig = Rig(tmp_path, footprint=Footprint(True, 'sample', (0., 0.), 10.))
    rig.v.specimen_voltage = 1000.
    for _ in range(4):
        rig.step(rate=.8)
    assert rig.engine.phase == 'ramp'
    assert rig.v.specimen_voltage == 1200.


def test_unconfirmed_target_rate_resumes_initial_ramp_without_fine_motion(tmp_path):
    rig = Rig(tmp_path)  # Background hits cannot authorize fine movement.
    rig.v.specimen_voltage = 1000.
    rig.step(rate=1.)
    assert rig.engine.phase == 'confirming'
    rig.until(lambda: rig.engine.phase == 'ramp', rate=1.)
    assert rig.engine.target_voltage == 1500.
    assert not rig.moves
    assert rig.engine.attempts == 0
    rig.step(rate=1.)
    assert rig.engine.phase == 'ramp'  # Do not repeatedly hold on the same invalid signal.


def test_early_fine_entry_requires_fresh_independent_event_windows(tmp_path):
    rig = Rig(tmp_path, footprint=Footprint(True, 'sample', (0., 0.), 10.))
    rig.v.specimen_voltage = 1000.
    rig.step(rate=1.)
    rig.step(rate=1.)
    for _ in range(8):
        rig.step(rate=1., fresh=False)
    assert rig.engine.phase == 'confirming'
    assert rig.engine.attempts == 0
    assert rig.v.specimen_voltage == 1000.


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


def test_detector_feedback_fine_alignment_works_without_fixed_jacobian(tmp_path):
    cfg = settings(xy_jacobian_mm_per_um=(), xy_range_um=(50., 50.), xy_step_um=10.)
    rig = Rig(tmp_path, cfg)
    def live_fit(*_):
        x, y, _z = rig.v.stage_position_snapshot[:3]
        return Footprint(True, 'sample', (2.-x*1e6, -2.+y*1e6), 10., .06, .9, .01, 1.)
    rig.engine.analyser = live_fit
    rig.until(lambda: rig.engine.phase == 'fine', rate=.4)
    rig.until(lambda: abs(rig.v.stage_position_snapshot[0]-2e-6) < 1e-9
              and abs(rig.v.stage_position_snapshot[1]-2e-6) < 1e-9, rate=.4)
    fine_moves = [move for move in rig.moves if 'fine_origin_m' in move]
    assert fine_moves
    assert all(move['speed_um_s'] == cfg.fine_xy_speed_um_s for move in fine_moves)
    assert all(abs(move['target_m'][axis]-rig.engine.fine_origin[axis])*1e6 <= 15
               for move in fine_moves for axis in (0, 1))
    assert any(json.loads(line)['event'] == 'fine_probe_accepted'
               for line in (tmp_path/'alignment.jsonl').read_text().splitlines())


def test_transfer_z_must_retract_from_saved_sample():
    cfg = settings(xy_jacobian_mm_per_um=(), transfer_z_mm=-4.,
                   bounds_mm=((-2., 6.), (-3., 7.), (-10., 7.)))
    cfg.validate_motion([(0., 0., -3e-3)])
    with pytest.raises(ValueError, match='retract'):
        cfg.validate_motion([(0., 0., -5e-3)])


def test_loss_after_approach_retracts_before_xy_search(tmp_path):
    rig = Rig(tmp_path, settings(approach_enabled=True, z_max_advance_um=1),
              footprint=Footprint(True, 'sample', (0., 0.), 10., .06, .9, .01, 1.))
    rig.until(lambda: rig.engine.phase == 'fine', rate=.4)
    rig.until(lambda: rig.v.stage_position_snapshot[2] > 0 and rig.engine.phase == 'fine', rate=.4)
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


@pytest.mark.parametrize('updated_field', ['stage_position_snapshot', 'alignment_stage_heartbeat'])
def test_live_alignment_accepts_updates_published_after_tick_started(tmp_path, monkeypatch, updated_field):
    clock = [10.]
    class UpdatingState(SimpleNamespace):
        def __getattribute__(self, name):
            if name == updated_field:
                clock[0] += .005  # GUI update arrives during the Manager read.
                return ((0., 0., 0., clock[0]) if name == 'stage_position_snapshot' else clock[0])
            return super().__getattribute__(name)
    cfg = settings()
    v = UpdatingState(**vars(state(cfg, tmp_path)))
    v.stage_position_snapshot = (0., 0., 0., 10.)
    v.alignment_stage_heartbeat = 10.
    engine = AutomaticAlignment(v, {'max_vdc': 10000}, now=0)
    monkeypatch.setattr('time.monotonic', lambda: clock[0])
    engine.tick(1500., 0.)
    assert engine.phase == 'moving'
    v.alignment_move_status = {'id': v.alignment_move_request['id'], 'state': 'done'}
    clock[0] += .5
    engine.tick(1500., 0.)
    assert engine.phase == 'coarse'
    clock[0] += .5
    engine.tick(1500., 0.)
    assert engine.phase == 'coarse'
    assert engine.outcome == ''


def test_future_stage_timestamp_still_faults_and_records_age(tmp_path):
    rig = Rig(tmp_path)
    rig.v.stage_position_snapshot = (0., 0., 0., 11.)
    rig.engine.tick(1500., .4, now=10.)
    assert rig.engine.outcome == 'fault'
    assert 'age -1.000 s' in rig.engine.reason
    records = [json.loads(line) for line in (tmp_path/'alignment.jsonl').read_text().splitlines()]
    assert any(r['event'] == 'position_timestamp_fault' and r['age_s'] == -1. for r in records)


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
                                                      (True, 'moving', 1500), (True, 'confirming', 1500),
                                                      (True, 'ramp', 1510)])
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
                                              ((1e-6, 0., 0.), 150, 'alignment'),
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
    v = state(settings(xy_range_um=(50., 50.), xy_step_um=10.), tmp_path)
    v.alignment_move_request = dict(id='fine', target_m=(16e-6, 0., 0.),
                                   fine_origin_m=(0., 0., 0.), speed_um_s=1.,
                                   kind='alignment', issued=0.)
    motor = Motor()
    AlignmentStageService(v, lambda: motor).tick(now=0.)
    assert v.alignment_move_status['state'] == 'error'
    assert not motor.moves


def test_stage_service_rejects_fine_speed_above_limit(tmp_path):
    v = state(settings(xy_range_um=(50., 50.), xy_step_um=10.), tmp_path)
    v.alignment_move_request = dict(id='fast-fine', target_m=(1e-6, 0., 0.),
                                    fine_origin_m=(0., 0., 0.), speed_um_s=17.,
                                    kind='alignment', issued=0.)
    motor = Motor()
    AlignmentStageService(v, lambda: motor).tick(now=0.)
    assert v.alignment_move_status['state'] == 'error'
    assert not motor.moves


def test_stage_service_keeps_stationary_alignment_position_fresh(tmp_path):
    v = state(settings(), tmp_path)
    motor = Motor()
    service = AlignmentStageService(v, lambda: motor)
    service.tick(now=0.)
    assert v.stage_position_snapshot == (0., 0., 0., 0.)
    motor.position['z'] = 1e-6
    service.tick(now=.25)
    assert v.stage_position_snapshot[3] == 0.
    service.tick(now=.5)
    assert v.stage_position_snapshot == (0., 0., 1e-6, .5)
    service.tick(now=1.)
    assert v.stage_position_snapshot[3] == 1.


@pytest.mark.parametrize('legacy', [False, True])
def test_z_transfer_ignores_held_xy_drift_but_requires_z_target_and_settling(tmp_path, legacy):
    cfg = settings()
    v = state(cfg, tmp_path)
    if legacy:
        v.alignment_settings = legacy_snapshot(cfg)
    v.hardware_safe = True
    v.alignment_move_request = dict(id='retract', target_m=(0., 0., -.5e-3),
                                   speed_um_s=1., kind='transfer', issued=0.)
    motor = Motor()
    service = AlignmentStageService(v, lambda: motor)
    service.tick(now=0.)
    # The held axes fluctuate more than 20 nm. The commanded axis still
    # has to reach its own 20 nm tolerance for the full settling interval.
    motor.position.update(x=50e-9, y=-50e-9, z=-.5e-3+50e-9)
    for now in (.1, .6, 1.):
        service.tick(now=now)
    assert v.alignment_move_status['state'] == 'moving'
    assert v.alignment_move_status['commanded_axes'] == ('z',)
    assert v.alignment_move_status['wait_reason'] == 'waiting for target position'
    assert v.alignment_move_status['error_um'] == pytest.approx((.05, -.05, .05))
    motor.position['z'] = -.5e-3
    motor.is_moving = lambda: True
    service.tick(now=1.1)
    assert v.alignment_move_status['wait_reason'] == 'stage moving'
    motor.is_moving = lambda: False
    service.tick(now=1.2)
    assert v.alignment_move_status['wait_reason'] == 'settling'
    service.tick(now=1.6)
    assert v.alignment_move_status['state'] == 'moving'
    service.tick(now=1.8)
    assert v.alignment_move_status['state'] == 'done'
    assert not motor.stopped


def test_transfer_still_faults_if_held_axis_leaves_calibrated_bounds(tmp_path):
    v = state(settings(), tmp_path)
    v.hardware_safe = True
    v.alignment_move_request = dict(id='retract', target_m=(0., 0., -.5e-3),
                                   speed_um_s=1., kind='transfer', issued=0.)
    motor = Motor()
    service = AlignmentStageService(v, lambda: motor)
    service.tick(now=0.)
    motor.position['x'] = 2e-3
    service.tick(now=.5)
    assert v.alignment_move_status['state'] == 'error'
    assert 'calibrated stage limits' in v.alignment_move_status['error']
    assert motor.stopped and v.alignment_cancel_motion


def test_transfer_timeout_reports_commanded_axes_and_position_error(tmp_path, caplog):
    cfg = settings(move_timeout_s=1.)
    v = state(cfg, tmp_path)
    v.hardware_safe = True
    v.alignment_move_request = dict(id='stuck', target_m=(0., 0., -.5e-3),
                                   speed_um_s=1., kind='transfer', issued=0.)
    motor = Motor()
    service = AlignmentStageService(v, lambda: motor)
    service.tick(now=0.)
    motor.position['z'] += 50e-9
    service.tick(now=.1)
    service.tick(now=1.1)
    error = v.alignment_move_status['error']
    assert v.alignment_move_status['state'] == 'error'
    assert 'waiting for target position' in error
    assert "commanded axes ('z',)" in error
    assert 'XYZ error' in error and 'tolerance 0.02 µm' in error
    assert 'Automatic stage command stuck failed' in caplog.text
    assert motor.stopped and v.alignment_cancel_motion


def test_stage_request_published_during_read_is_not_expired(tmp_path, monkeypatch):
    clock = [10.]
    class UpdatingState(SimpleNamespace):
        def __getattribute__(self, name):
            if name == 'alignment_move_request':
                clock[0] += .005
                return dict(id='new-request', target_m=(1e-6, 0., 0.),
                            speed_um_s=1., kind='alignment', issued=clock[0])
            return super().__getattribute__(name)
    v = UpdatingState(**vars(state(settings(), tmp_path)))
    v.experiment_heartbeat_monotonic = clock[0]
    motor = Motor()
    service = AlignmentStageService(v, lambda: motor)
    monkeypatch.setattr('time.monotonic', lambda: clock[0])
    service.tick()
    assert v.alignment_move_status['state'] == 'moving'
    assert len(motor.moves) == 1
    assert not v.alignment_cancel_motion


def test_transfer_retracts_traverses_then_approaches_saved_position():
    target = (2e-4, 3e-4, .1e-3)
    points = transfer_waypoints((0, 0, 0), target, settings())
    assert points == [(0, 0, -.5e-3), (target[0], target[1], -.5e-3), target]


def test_all_rejected_feedback_directions_recover_instead_of_repeating(tmp_path):
    rig = Rig(tmp_path, settings(xy_jacobian_mm_per_um=()),
              Footprint(True, 'unchanged centre', (2., 2.), 10.))
    rig.until(lambda: rig.engine.phase == 'fine', rate=.4)
    rig.until(lambda: rig.engine.pending and rig.engine.pending[2] == 'recover_xy', rate=.4)
    events = [json.loads(line) for line in (tmp_path/'alignment.jsonl').read_text().splitlines()]
    probes = [event for event in events if event['event'] == 'fine_probe']
    assert len(probes) == 4
    assert sum(event['event'] == 'fine_probe_rejected' for event in events) == 4
    assert rig.engine.attempts == 1
    rig.until(lambda: rig.engine.phase == 'coarse', rate=0.)


@pytest.mark.parametrize('field,value', [
    ('physical_estop_ok', False), ('experiment_state', 'failed'),
    ('experiment_heartbeat_monotonic', -10.), ('electrode_out', True),
    ('flag_tdc_failure', True), ('start_flag', False),
])
@pytest.mark.parametrize('during_move', [False, True])
def test_stage_interlocks_prevent_or_stop_motion(tmp_path, field, value, during_move):
    v = state(settings(), tmp_path)
    v.alignment_move_request = dict(id='guarded', target_m=(1e-6, 0., 0.),
                                   speed_um_s=1., kind='alignment', issued=0.)
    motor = Motor()
    service = AlignmentStageService(v, lambda: motor)
    if during_move:
        service.tick(now=0.)
        assert service.active is not None
    setattr(v, field, value)
    service.tick(now=.1)
    assert v.alignment_cancel_motion and motor.stopped
    assert v.alignment_move_status['id'] == 'guarded'
    assert v.alignment_move_status['state'] == 'error'
    assert len(motor.moves) == int(during_move)


def test_transfer_does_not_require_live_experiment_but_checks_physical_interlock(tmp_path):
    v = state(settings(), tmp_path)
    v.start_flag = False
    v.experiment_state = 'complete'
    v.experiment_heartbeat_monotonic = -100.
    v.hardware_safe = True
    v.alignment_move_request = dict(id='transfer', target_m=(0., 0., -.5e-3),
                                   speed_um_s=1., kind='transfer', issued=0.)
    motor = Motor()
    service = AlignmentStageService(v, lambda: motor)
    service.tick(now=0.)
    assert motor.moves and service.active is not None
    v.physical_estop_ok = False
    service.tick(now=.1)
    assert v.alignment_cancel_motion and motor.stopped


def test_motion_fault_is_reported_even_when_stop_raises(tmp_path):
    v = state(settings(), tmp_path)
    v.alignment_move_request = dict(id='broken', target_m=(1e-6, 0., 0.),
                                   speed_um_s=1., kind='alignment', issued=0.)
    motor = Motor()
    motor.stop = lambda: (_ for _ in ()).throw(OSError('controller disconnected'))
    v.physical_estop_ok = False
    AlignmentStageService(v, lambda: motor).tick(now=0.)
    assert v.alignment_move_status['state'] == 'error' and v.alignment_cancel_motion


def test_impossible_sweep_budget_is_rejected_before_movement():
    cfg = settings(xy_range_um=(50., 50.), timeout_s=10.)
    with pytest.raises(ValueError, match='sweep.*timeout'):
        cfg.validate_motion([(0., 0., 0.)])
    cfg = replace(cfg, timeout_s=120000.)
    cfg.validate_motion([(0., 0., 0.)])


def test_default_full_sweep_reaches_voltage_retry_before_timeout(tmp_path):
    cfg = settings(xy_range_um=(50., 50.), xy_step_um=10.)
    rig = Rig(tmp_path, cfg)
    rig.until(lambda: rig.engine.target_voltage > cfg.start_voltage, limit=4000)
    assert rig.engine.phase == 'ramp'
    assert rig.time < cfg.timeout_s
    assert rig.v.stage_position_snapshot[:3] == rig.engine.origin
    assert cfg.coarse_scan_duration_s() < cfg.timeout_s


def test_relative_jump_enters_fine_far_below_old_absolute_threshold(tmp_path):
    fit = Footprint(True, 'sample', (2., 0.), 10., .06, .9, .01, 1.)
    rig = Rig(tmp_path, footprint=fit)
    rig.until(lambda: rig.engine.phase == 'fine', rate=.03)
    assert rig.engine.attempts == 1 and rig.engine.fine_reference_rate == pytest.approx(.03)
    assert rig.engine.target_voltage == 1500.
    records = [json.loads(line) for line in (tmp_path/'alignment.jsonl').read_text().splitlines()]
    events = [record['event'] for record in records]
    assert 'relative_jump_neighbour_rechecked' in events
    assert 'relative_jump_confirmed' in events
    assert next(record for record in records if record['event'] == 'fine_started')['source'] == 'relative_jump'
    # A lower-than-requested rate remains useful; loss is relative to the found signal.
    for _ in range(8):
        rig.step(.02)
    assert rig.engine.attempts == 1 and rig.engine.outcome == ''


def test_uniform_high_rate_does_not_replace_spatial_contrast(tmp_path):
    from pyccapt.control.apt.alignment_search import significant_jump
    fit = Footprint(True, 'sample', (2., 0.), 10.)
    rig = Rig(tmp_path, footprint=fit)
    # Counter the fake rig's spatial model to give every coarse point the same rate.
    for _ in range(160):
        away = (rig.engine.phase in ('coarse', 'candidate', 'neighbour') and
                np.linalg.norm(rig.v.stage_position_snapshot[:2]) > 1e-10)
        rig.step(.4 if away else .1)
        if rig.engine.target_voltage > 1500:
            break
    assert rig.engine.attempts == 0
    baseline = dict(rate=.1, error=.005, coherent=True, voltage=1500.)
    assert not significant_jump(dict(baseline, rate=.12), baseline, 1.5, 3.)
    assert not significant_jump(dict(baseline, rate=.4, voltage=1600.), baseline, 1.5, 3.)


def test_uniform_temporal_rate_increase_is_rejected_on_neighbour_recheck(tmp_path):
    rig = Rig(tmp_path, footprint=Footprint(True, 'sample', (2., 0.), 10.))
    rig.until(lambda: rig.engine.phase == 'neighbour', rate=.03)
    # Now all positions have the same elevated rate. Rechecking the low point
    # must expose the drift rather than certify a spatial evaporation peak.
    for _ in range(60):
        away = (rig.engine.phase in ('coarse', 'candidate', 'neighbour') and
                np.linalg.norm(rig.v.stage_position_snapshot[:2]) > 1e-10)
        rig.step(.12 if away else .03)
        records = [json.loads(line) for line in (tmp_path/'alignment.jsonl').read_text().splitlines()]
        if any(item['event'] == 'relative_jump_rejected' for item in records):
            break
    assert any(item['event'] == 'relative_jump_rejected' for item in records)
    assert rig.engine.attempts == 0


def test_density_centroid_guides_xy_but_single_pixel_and_background_are_rejected():
    rng = np.random.default_rng(35)
    points = disk(rng, 350, 8)+np.array((5., -4.))
    fit = estimate_dense_region(points, 40, 200)
    assert fit.valid and fit.model == 'density'
    assert fit.centre_mm == pytest.approx((5., -4.), abs=1.2)
    assert not estimate_dense_region(np.zeros((2000, 2)), 40, 200).valid
    assert not estimate_dense_region(disk(rng, 2000, 40), 40, 200).valid
    assert not estimate_dense_region(points[:40], 40, 200).valid


def test_sparse_search_starts_near_saved_position_then_expands_without_z():
    cfg = settings(xy_range_um=(50., 50.), xy_step_um=10.)
    points = np.array(list(coarse_positions((1e-3, 2e-3, 3e-3), cfg)))
    offsets = np.round((points-np.array((1e-3, 2e-3, 3e-3)))*1e6, 8)
    assert len(points) == 33
    assert set(np.max(abs(offsets[:, :2]), axis=1)) == {0., 10., 20., 40., 50.}
    assert np.max(abs(offsets[1:9, :2])) == 10.
    assert np.all(offsets[:, 2] == 0.)
    assert len(set(map(tuple, points))) == len(points)
    assert cfg.coarse_scan_duration_s() < 250.


def test_low_rate_publisher_drops_old_hits_without_starving_new_density_windows():
    v = SimpleNamespace(automatic_alignment_enabled=True,
                        alignment_settings={'window_ions': 2000, 'window_max_age_s': 10.},
                        alignment_window_epoch='new')
    publisher = AlignmentEventPublisher(v)
    publisher.append([0]*250, [0]*250, now=0.)
    publisher.append([0]*250, [0]*250, now=1.)
    publisher.append([1]*250, [1]*250, now=12.)
    assert len(v.alignment_events['points_mm']) == 250
    assert v.alignment_events['first_time'] == 12.
    assert v.alignment_events['sequence'] == 500


def test_z_requires_stable_centred_circle_and_cannot_use_dense_region_only(tmp_path):
    cfg = settings(approach_enabled=True, z_max_advance_um=20.)
    fit = Footprint(True, 'sample', (0., 0.), 10., .06, .9, .01, 1.)
    rig = Rig(tmp_path, cfg, fit)
    rig.until(lambda: rig.engine.phase == 'fine', rate=.04)
    for _ in range(4):
        rig.step(.04)
    assert rig.v.stage_position_snapshot[2] == 0.
    rig.until(lambda: rig.v.stage_position_snapshot[2] > 0., rate=.04)
    assert rig.v.stage_position_snapshot[2] <= cfg.z_step_um*1e-6
    dense = Footprint(True, 'density', (0., 0.), 10., .06, .9, model='density')
    rig.fit = dense
    before = sum(bool(move['target_m'][2]) for move in rig.moves)
    for _ in range(12):
        rig.step(.04)
    assert sum(bool(move['target_m'][2]) for move in rig.moves) == before


def test_z_motion_masks_held_xy_drift_and_enforces_fine_marker_and_20_um_limit(tmp_path):
    cfg = settings(approach_enabled=True, z_max_advance_um=20.)
    v = state(cfg, tmp_path)
    motor = Motor()
    motor.position.update(x=50e-9, y=-50e-9)
    v.alignment_move_request = dict(id='fine-z', target_m=(0., 0., .05e-6), axes=('z',),
                                   fine_origin_m=(0., 0., 0.), speed_um_s=.1,
                                   kind='alignment', issued=0.)
    service = AlignmentStageService(v, lambda: motor)
    service.tick(now=0.)
    assert motor.moves[-1]['x_m'] is None and motor.moves[-1]['y_m'] is None
    assert motor.moves[-1]['z_m'] == .05e-6
    assert v.alignment_move_status['commanded_axes'] == ('z',)
    for target, marker in [(20.05e-6, True), (.1e-6, False)]:
        other = state(cfg, tmp_path)
        other.alignment_move_request = dict(id='bad-z', target_m=(0., 0., target), axes=('z',),
                                           speed_um_s=.1, kind='alignment', issued=0.)
        if marker:
            other.alignment_move_request['fine_origin_m'] = (0., 0., 0.)
        other_motor = Motor()
        AlignmentStageService(other, lambda: other_motor).tick(now=0.)
        assert other.alignment_move_status['state'] == 'error'
        assert not other_motor.moves


def test_invalid_observation_resets_xy_stability_before_z_can_advance(tmp_path):
    fit = Footprint(True, 'sample', (0., 0.), 10., .06, .9, .01, 1.)
    rig = Rig(tmp_path, settings(approach_enabled=True, z_max_advance_um=20.), fit)
    rig.until(lambda: rig.engine.phase == 'fine', rate=.04)
    for _ in range(4):
        rig.step(.04)
    rig.fit = Footprint(False, 'temporary invalid region')
    rig.step(.04)
    rig.fit = fit
    for _ in range(4):
        rig.step(.04)
    assert rig.v.stage_position_snapshot[2] == 0.
    rig.until(lambda: rig.v.stage_position_snapshot[2] > 0., rate=.04)


def test_measured_z_overshoot_stops_even_when_command_target_was_within_20_um(tmp_path):
    v = state(settings(approach_enabled=True, z_max_advance_um=20.), tmp_path)
    v.alignment_move_request = dict(id='last-z', target_m=(0., 0., 20e-6), axes=('z',),
                                   fine_origin_m=(0., 0., 0.), speed_um_s=.1,
                                   kind='alignment', issued=0.)
    motor = Motor()
    move = motor.move_absolute
    def overshooting_move(**kwargs):
        move(**kwargs)
        motor.position['z'] += .05e-6
    motor.move_absolute = overshooting_move
    AlignmentStageService(v, lambda: motor).tick(now=0.)
    assert v.alignment_move_status['state'] == 'error'
    assert 'Measured sample Z' in v.alignment_move_status['error']
    assert motor.stopped and v.alignment_cancel_motion
