"""Known-response replay; no detector, stage, camera or laser is connected."""
import json
import math
from dataclasses import replace

import numpy as np
import pytest

from pyccapt.control.apt.alignment_config import coarse_positions
from pyccapt.control.apt.alignment_response import coverage_probe, fit_local_response, informative_cross
from pyccapt.control.apt.alignment_replay import summarize_replay
from pyccapt.control.apt.alignment_vision import Footprint, estimate_footprint
from pyccapt.control.apt.laser_alignment import LaserAlignment
from pyccapt.control.devices.alignment_stage import AlignmentStageService
from tests.control.test_laser_alignment import rig
from tests.control.test_automatic_alignment import Rig, Motor, disk, settings, state


def response_records(function):
    points = informative_cross((0., 0., 0.), [-1e-6]*3, [1e-6]*3, 1., 1.)
    return [dict(position_m=p, rate_percent=function(np.asarray(p[:2])*1e6),
                 error_percent=.001, count=100000, voltage=1500.) for p in points]


def test_local_response_recovers_bounded_peak_and_rejects_flat_or_multimodal():
    records = response_records(lambda xy: float(np.exp(-np.sum((xy-[.3, -.2])**2))))
    result = fit_local_response(records, (0, 0, 0), 1., [-1e-6]*3, [1e-6]*3)
    assert result['usable']
    assert np.asarray(result['position_m'])*1e6 == pytest.approx([.3, -.2, 0.], abs=1e-8)
    flat = response_records(lambda xy: 1.)
    assert not fit_local_response(flat, (0, 0, 0), 1., [-1e-6]*3, [1e-6]*3)['usable']
    records[4]['rate_percent'] *= 2
    assert not fit_local_response(records, (0, 0, 0), 1., [-1e-6]*3, [1e-6]*3)['usable']


def test_response_never_extrapolates_or_uses_changed_voltage():
    records = response_records(lambda xy: float(np.exp(-.1*np.sum((xy-[2., 0.])**2))))
    assert not fit_local_response(records, (0, 0, 0), 1., [-1e-6]*3, [1e-6]*3)['usable']
    records[0]['voltage'] += 1.
    assert 'voltages' in fit_local_response(records, (0, 0, 0), 1., [-1e-6]*3, [1e-6]*3)['reason']


def _laser_replay(path, strategy, response, *, full=False, noise_seed=None):
    path.mkdir()
    v, conf, now = rig(path)
    conf.update(laser_alignment_coarse_step_um=.5, laser_alignment_search_strategy=strategy)
    v.laser_telemetry = {**v.laser_telemetry, 'output_frequency_hz': 1e6}
    engine = LaserAlignment(v, conf, now=now)
    signature = dict(v.laser_telemetry)
    rng = np.random.default_rng(noise_seed)
    for _ in range(2000):
        if ((engine.kind == 'fine' and not full) or engine.phase in ('ramp', 'tracking_wait') or engine.finished):
            break
        if engine.pending:
            request = v.laser_alignment_move_request
            distance = np.max(np.abs(np.subtract(request['target_m'], v.laser_stage_snapshot[:3])))*1e6
            now += distance/request['speed_um_s']+engine.cfg.settle_s
            v.laser_stage_snapshot = (*request['target_m'], now)
            v.laser_alignment_move_status = {'id': request['id'], 'state': 'done'}
        else:
            now += engine.cfg.dwell_s
            rate = response(np.asarray(v.laser_stage_snapshot[:2])*1e6)
            count = round(rate*1e6*engine.cfg.dwell_s/100)
            if noise_seed is not None:
                count = int(rng.poisson(count))
            rate = count/(1e6*engine.cfg.dwell_s)*100
            error = math.sqrt(max(count, 1))/(1e6*engine.cfg.dwell_s)*100
            engine._measurement = lambda: engine._record(rate, error, count, {})
        v.laser_stage_snapshot = (*v.laser_stage_snapshot[:3], now)
        v.laser_alignment_heartbeat = now
        v.laser_telemetry = {**v.laser_telemetry, 'monotonic': now}
        engine.tick(now)
    assert v.laser_telemetry['aom_percent'] == signature['aom_percent']
    assert v.specimen_voltage == 1500.
    records = [json.loads(line) for line in (path/'laser_alignment.jsonl').read_text().splitlines()]
    return engine, summarize_replay(records), records


def test_hybrid_replay_measures_proposal_before_accepting_and_uses_fewer_windows(tmp_path):
    response = lambda xy: float(np.exp(-np.sum((xy-[.4, .3])**2)/(1.2**2)))
    raster, old, _ = _laser_replay(tmp_path/'raster', 'raster', response)
    hybrid, new, records = _laser_replay(tmp_path/'hybrid', 'hybrid', response)
    assert raster.kind == hybrid.kind == 'fine'
    assert np.asarray(hybrid.best_position)*1e6 == pytest.approx([.4, .3, 0.], abs=.01)
    assert new['measurement_windows'] < old['measurement_windows']
    assert new['elapsed_s'] < old['elapsed_s']
    assert new['observed_ions'] < old['observed_ions']
    proposal = next(r for r in records if r['event'] == 'response_surface')
    assert proposal['usable']
    accepted = next(i for i, r in enumerate(records) if r['event'] == 'accepted')
    assert records[accepted-1]['event'] == 'observation'
    assert 'probability' not in proposal


def test_flat_hybrid_replay_falls_back_to_grid_without_displacing_centre(tmp_path):
    engine, metrics, records = _laser_replay(tmp_path/'flat', 'hybrid', lambda xy: 1.)
    assert engine.kind == 'fine' and engine.best_position == (0., 0., 0.)
    assert metrics['surface_fallbacks'] == 1
    observed = {tuple(r['position_m']) for r in records if r['event'] == 'observation'}
    assert len(observed) == 25


def test_adaptive_dwell_obeys_minimum_maximum_and_only_early_rejects(tmp_path):
    v, conf, now = rig(tmp_path)
    conf.update(laser_alignment_adaptive_dwell=True, laser_alignment_min_dwell_s=.5,
                laser_alignment_max_dwell_s=2., laser_alignment_min_events=30)
    engine = LaserAlignment(v, conf, now=now)
    engine._observe()
    def measured(duration, count):
        engine.now = now+duration
        v.laser_stage_snapshot = (0., 0., 0., engine.now)
        v.laser_alignment_observation = dict(epoch=engine.epoch, start=now, time=engine.now, count=count)
        return engine._measurement()
    assert measured(.25, 1000) is None
    assert measured(.5, 100)['dwell_decision'] == 'count_precision'
    assert measured(.5, 5) is None
    assert measured(2., 5)['dwell_decision'] == 'maximum_dwell'
    engine.records = [engine._record(20., .1, 10000, {})]
    assert measured(.5, 5)['dwell_decision'] == 'clearly_poor'
    engine.phase = 'verify'
    assert measured(.5, 5) is None


def test_legacy_spectral_roi_cannot_influence_motor_scoring(tmp_path):
    v, conf, now = rig(tmp_path)
    conf['laser_alignment_quality_peak_ns'] = [100., 101.]
    engine = LaserAlignment(v, conf, now=now)
    engine.kind = 'focus'
    assert engine._quality_ok({'quality': {}}, {'quality': {'width_ns': 1.}})


def test_reuse_requires_matching_explicit_conditions_and_fresh_verification(tmp_path):
    v, conf, now = rig(tmp_path)
    conf.update(laser_alignment_reuse_previous=True, laser_alignment_reuse_context='tip-A',
                laser_alignment_coordinate_calibration_id='reference-2026-10')
    first = LaserAlignment(v, conf, now=now)
    v.laser_alignment_solution_cache = dict(position_m=[.4e-6, .3e-6, 0.], time=now,
        voltage=1500., conditions=first._conditions(), surface={})
    reused = LaserAlignment(v, conf, now=now+.1)
    assert reused.reuse_in_progress and reused.pending[2] == 'observe'
    assert list(reused.points)[0] == (.4e-6, .3e-6, 0.)
    assert not reused.records and reused.phase != 'tracking_wait'
    conf['laser_alignment_reuse_context'] = 'different-tip'
    other = LaserAlignment(v, conf, now=now+.2)
    assert not other.reuse_in_progress


@pytest.mark.parametrize('bad_cache', [
    4, 'invalid', {'surface': 'invalid'}, {'surface': {'curvature_per_um2': [['a', 'b'], ['b', 'c']]}},
    {'surface': {'curvature_per_um2': [[-1., float('nan')], [0., -1.]]}},
    {'surface': {'curvature_per_um2': [[-1., 2.], [0., -1.]]}},
    {'position_m': [1., 0., 0.]}, {'time': -10000.}, {'voltage': 1700.},
    {'position_m': [.4e-6, .3e-6, .1e-6]},
])
def test_invalid_or_incompatible_cache_falls_back_before_any_cached_movement(tmp_path, bad_cache):
    v, conf, now = rig(tmp_path)
    conf.update(laser_alignment_reuse_previous=True, laser_alignment_reuse_context='tip-A',
                laser_alignment_coordinate_calibration_id='reference-2026-10')
    first = LaserAlignment(v, conf, now=now)
    valid = dict(position_m=[.4e-6, .3e-6, 0.], time=now, voltage=1500.,
                 conditions=first._conditions(), surface={})
    v.laser_alignment_solution_cache = {**valid, **bad_cache} if isinstance(bad_cache, dict) else bad_cache
    engine = LaserAlignment(v, conf, now=now+.1)
    assert not engine.reuse_in_progress
    assert v.laser_alignment_move_request['target_m'] == engine.origin
    assert len(engine.points) > 2


def test_reuse_that_loses_fresh_comparison_returns_to_centre_and_full_scan_at_same_dc(tmp_path):
    v, conf, now = rig(tmp_path)
    conf.update(laser_alignment_reuse_previous=True, laser_alignment_reuse_context='tip-A',
                laser_alignment_coordinate_calibration_id='reference-2026-10')
    first = LaserAlignment(v, conf, now=now)
    v.laser_alignment_solution_cache = dict(position_m=[.4e-6, .3e-6, 0.], time=now, voltage=1500.,
        conditions=first._conditions(), surface={'curvature_per_um2': [[-4., 0.], [0., -4.]]})
    engine = LaserAlignment(v, conf, now=now+.1)
    assert engine.reuse_in_progress and engine.reuse_radius == pytest.approx(.2)
    engine.records = [dict(engine._record(rate, .001, 1000, {}), position_m=position)
                      for rate, position in [(1., engine.origin), (.2, engine.reuse_target), (1., engine.origin)]]
    engine._complete_scan()
    assert engine.pending[2] == 'retry' and v.laser_alignment_move_request['target_m'] == engine.origin
    v.laser_alignment_move_status = {'id': engine.pending[0], 'state': 'done'}
    engine.tick(now+.2)
    assert engine.kind == 'coarse' and not engine.reuse_in_progress
    assert len(engine.points) > 2 and engine.target_voltage == v.specimen_voltage == 1500.


def test_space_filling_fallback_covers_a_narrow_island_missed_by_sparse_probes():
    cfg = settings(xy_range_um=(50., 50.))
    origin = (0., 0., 0.)
    low, high = cfg.search_xy_limits(origin)
    visited = list(coarse_positions(origin, cfg))
    island = np.array((12.45e-6, 37.35e-6))
    assert min(np.linalg.norm(np.asarray(p[:2])-island) for p in visited) > 10e-6
    for _ in range(64):
        point = coverage_probe(origin, low, high, visited)
        if point is None:
            break
        assert point[2] == 0. and np.all(np.asarray(point[:2]) >= low) and np.all(np.asarray(point[:2]) <= high)
        visited.append(point)
    assert min(np.linalg.norm(np.asarray(p[:2])-island) for p in visited) < 1e-12


def test_optional_coverage_engine_moves_within_original_limits(tmp_path):
    cfg = settings(xy_range_um=(50., 50.), coverage_max_probes=4)
    replay = Rig(tmp_path, cfg)
    replay.until(lambda: replay.engine.coverage_count == 4)
    assert replay.engine.search_stage == 'coverage_fallback'
    replay.until(lambda: replay.engine.target_voltage > cfg.start_voltage)
    assert replay.v.stage_position_snapshot[:3] == replay.engine.origin
    records = [json.loads(line) for line in (tmp_path/'alignment.jsonl').read_text().splitlines()]
    assert summarize_replay(records)['coverage_probes'] == 4
    assert all(r['target_m'][2] == 0. for r in records if r['event'] == 'coverage_probe')


def test_z_abstains_without_independent_response_calibration_at_engine_and_service(tmp_path):
    cfg = settings(approach_enabled=True, z_max_advance_um=20., z_response_calibrated=False,
                   z_response_calibration_id='')
    replay = Rig(tmp_path, cfg, Footprint(True, 'circle', (0., 0.), 10., .06, .9, .01, 1.))
    replay.until(lambda: replay.engine.phase == 'fine', rate=.04)
    replay.until(lambda: replay.engine.target_voltage > cfg.start_voltage, rate=.04)
    assert all(move['target_m'][2] == 0. for move in replay.moves)
    v = state(cfg, tmp_path)
    v.alignment_move_request = dict(id='z', target_m=(0., 0., .05e-6), axes=('z',),
        fine_origin_m=(0., 0., 0.), speed_um_s=.1, kind='alignment', issued=0.)
    motor = Motor()
    AlignmentStageService(v, lambda: motor).tick(now=0.)
    assert not motor.moves and 'calibrated response' in v.alignment_move_status['error']


def test_bright_pole_does_not_replace_geometric_footprint_centre():
    rng = np.random.default_rng(11)
    points = np.vstack((disk(rng, 6000, 12.), rng.normal((7., 3.), .5, (4000, 2))))
    fit = estimate_footprint(points, 40., 2000)
    assert fit.valid
    assert np.linalg.norm(fit.centre_mm) < 1.
    assert np.linalg.norm(fit.intensity_centre_mm) > 2.
    assert fit.geometry_intensity_offset_mm > 1.5


def test_replay_false_alignment_requires_independent_truth():
    records = [dict(event='aligned', elapsed_s=10., position_m=[2e-6, 0., 0.])]
    assert summarize_replay(records)['false_aligned_count'] is None
    assert summarize_replay(records, expected_position_m=[0., 0., 0.], tolerance_um=1.)['false_aligned_count'] == 1


@pytest.mark.parametrize('seed', [12, 37, 91])
def test_full_noisy_laser_replay_reports_success_and_false_alignment_metric(tmp_path, seed):
    response = lambda xy: float(np.exp(-np.sum((xy-[.4, .3])**2)/(1.2**2)))
    engine, _, records = _laser_replay(tmp_path/'full', 'hybrid', response, full=True, noise_seed=seed)
    assert engine.phase == 'tracking_wait', engine.reason
    metrics = summarize_replay(records, expected_position_m=[.4e-6, .3e-6, 0.], tolerance_um=.3)
    assert metrics['aligned'] and metrics['false_aligned_count'] == 0
    assert metrics['elapsed_s'] > 0 and metrics['observed_ions'] > 0


def test_sample_alignment_holds_user_laser_settings_and_rejects_changes(tmp_path):
    replay = Rig(tmp_path)
    replay.v.pulse_mode = 'Laser'
    replay.v.laser_telemetry = dict(valid=True, status_code=129, monotonic=0., aom_percent=20.,
                                     output_frequency_hz=1000., wavelength_index=3.)
    replay.engine = type(replay.engine)(replay.v, {'max_vdc': 10000}, now=0.)
    replay.step()
    assert replay.engine.fixed_laser_signature == (20., 1000., 3.)
    replay.v.laser_telemetry = {**replay.v.laser_telemetry, 'aom_percent': 21., 'monotonic': .5}
    replay.step()
    assert replay.engine.outcome == 'fault' and replay.v.alignment_cancel_motion
    assert 'power' in replay.engine.reason


@pytest.mark.parametrize('initial,changed', [('Voltage', 'Laser'), ('Laser', 'Voltage')])
def test_sample_alignment_rejects_any_midrun_pulse_mode_change(tmp_path, initial, changed):
    replay = Rig(tmp_path)
    replay.v.pulse_mode = initial
    replay.engine = type(replay.engine)(replay.v, {'max_vdc': 10000}, now=0.)
    replay.v.pulse_mode = changed
    replay.step()
    assert replay.engine.outcome == 'fault' and replay.v.alignment_cancel_motion
    assert 'mode changed' in replay.engine.reason


def test_low_count_surface_abstains_and_replay_truth_rejects_nan_tolerance():
    records = response_records(lambda xy: float(np.exp(-np.sum(xy**2))))
    for record in records:
        record['error_percent'] = record['rate_percent']
    assert not fit_local_response(records, (0, 0, 0), 1., [-1e-6]*3, [1e-6]*3)['usable']
    with pytest.raises(ValueError, match='positive tolerance'):
        summarize_replay([], expected_position_m=[0., 0., 0.], tolerance_um=float('nan'))


def test_new_settings_reject_unbounded_or_unidentified_options(tmp_path):
    v, conf, now = rig(tmp_path)
    with pytest.raises(ValueError, match='explicit specimen'):
        LaserAlignment(v, {**conf, 'laser_alignment_reuse_previous': True}, now=now)
    with pytest.raises(ValueError, match='Coverage'):
        replace(settings(), coverage_max_probes=257).validate()
