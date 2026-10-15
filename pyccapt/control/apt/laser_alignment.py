"""Laser-stage search and tracking. Device I/O stays with existing device owners."""
import json
import math
import time
import uuid
from collections import deque
from pathlib import Path

import numpy as np

from pyccapt.control.apt.laser_alignment_config import LaserAlignmentConfig
from pyccapt.control.apt.alignment_response import fit_local_response, informative_cross


class LaserAlignment:
    def __init__(self, variables, conf, mode='coarse', now=None):
        self.v = variables
        self.conf = conf
        self.now = time.monotonic() if now is None else now
        self.started = self.now
        self.cfg = LaserAlignmentConfig.from_config(conf, variables.laser_alignment_settings)
        self.origin = self._position()
        self.cfg.validate(self.origin)
        self.limit = min(self.cfg.max_voltage, float(variables.vdc_max), float(conf['max_vdc']))
        self.target_voltage = float(variables.specimen_voltage)
        if not 0 < self.target_voltage <= self.limit:
            raise ValueError('Laser alignment DC voltage exceeds its configured ceiling.')
        if variables.pulse_mode != 'Laser' or variables.flat_test_active:
            raise ValueError('Automatic laser alignment requires a Laser-mode experiment.')
        if variables.automatic_alignment_enabled and variables.alignment_status.get('phase') != 'aligned':
            raise ValueError('Wait for sample alignment to finish before moving the laser.')
        if mode not in ('coarse', 'fine', 'focus'):
            raise ValueError('Unknown laser alignment command.')
        self.session = uuid.uuid4().hex
        self.log_path = Path(variables.path_meta)/'laser_alignment.jsonl'
        self.pending = None
        self.phase = 'starting'
        self.reason = ''
        self.finished = False
        self.recoveries = 0
        self.fine_failures = 0
        self.fine_round = 0
        self.final_xy = False
        self.best_position = self.origin
        self.measurements = 0
        self.ions_observed = 0
        self.surface = {}
        self.last_usable_surface = {}
        self.reuse_in_progress = False
        self.reuse_radius = None
        self.progress = 'Preparing fixed-power detector-response scan'
        self.v.laser_alignment_cancel = False
        self.v.laser_alignment_run = {'id': self.session, 'origin_m': self.origin, 'settings': self.cfg.snapshot()}
        self.laser_signature = self._signature()
        self._event('start', settings=self.cfg.snapshot(), origin_m=self.origin, mode=mode,
                    voltage_limit=self.limit, metric='detection_rate_at_fixed_dc')
        self._publish()
        cached = self._compatible_solution() if mode == 'coarse' else None
        self._begin_scan(mode, self.origin)
        if cached is not None:
            # The first pending command is still the centre baseline. Replace
            # only subsequent targets; cached coordinates never skip settling,
            # fresh epochs or the usual candidate verification.
            self.points = deque([tuple(cached['position_m']), self.origin])
            self.model_pending = False
            self.reuse_in_progress = True
            self.reuse_target = tuple(cached['position_m'])
            curvature = np.asarray(cached.get('surface', {}).get('curvature_per_um2', []), dtype=float)
            if curvature.shape == (2, 2) and np.isfinite(curvature).all():
                eigenvalues = np.linalg.eigvalsh(curvature)
                if eigenvalues.max() < 0:
                    self.reuse_radius = float(np.clip(1/np.sqrt(-eigenvalues.min()),
                                                      self.cfg.fine_range_um/2, self.cfg.fine_range_um))
            self.progress = 'Verifying previous solution against a fresh centre baseline'
            self._event('reuse_proposed', solution=cached)

    @property
    def owns_voltage(self):
        return not self.finished and self.phase != 'tracking_wait'

    def _signature(self):
        # A GUI publication can race this tick's monotonic timestamp by a few ms.
        data = dict(self.v.laser_telemetry)
        if not -0.1 <= self.now-data.get('monotonic', -math.inf) <= 10:
            data = {}
        if not data.get('valid') or data.get('status_code') != 129:
            raise ValueError('Laser alignment requires fresh readback with laser output enabled.')
        result = tuple(float(data[k]) for k in ('aom_percent', 'output_frequency_hz', 'wavelength_index'))
        if not all(math.isfinite(x) for x in result) or result[1] <= 0:
            raise ValueError('Laser settings are unavailable.')
        return result

    def _position(self):
        data = self.v.laser_stage_snapshot
        if len(data) != 4 or not -0.1 <= self.now-float(data[3]) <= 3:
            raise ValueError('Laser stage position is stale.')
        return tuple(float(x) for x in data[:3])

    def _event(self, event, **values):
        record = {'event': event, 'session': self.session, 'time': self.now,
                  'elapsed_s': self.now-self.started, 'phase': self.phase, **values}
        with self.log_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False)+'\n')

    def _publish(self):
        self.v.laser_alignment_status = {
            'session': self.session, 'phase': self.phase, 'reason': self.reason,
            'active': not self.finished, 'time': self.now, 'target_voltage': self.target_voltage,
            'fine_failures': self.fine_failures, 'recoveries': self.recoveries,
            'owns_voltage': self.owns_voltage,
            'progress': self.progress, 'measurements': self.measurements,
            'ions_observed': self.ions_observed, 'search_strategy': self.cfg.search_strategy,
            'evidence': 'detector count/rate at fixed user laser settings',
        }

    def _conditions(self):
        return dict(context=self.cfg.reuse_context, coordinate_calibration_id=self.cfg.coordinate_calibration_id,
                    laser_settings=list(self.laser_signature),
                    sample=getattr(self.v, 'alignment_sample', None),
                    sample_position_m=list(getattr(self.v, 'alignment_sample_position', ())),
                    bounds_mm=list(self.cfg.bounds_mm), travel_um=list(self.cfg.travel_um))

    def _compatible_solution(self):
        if not self.cfg.reuse_previous:
            return None
        try:
            cache = dict(getattr(self.v, 'laser_alignment_solution_cache', {}) or {})
            position = tuple(float(x) for x in cache['position_m'])
            self.cfg.check_position(position, self.origin)
            surface = cache.get('surface', {})
            if not isinstance(surface, dict):
                raise ValueError('Invalid cached response surface')
            curvature = np.asarray(surface.get('curvature_per_um2', []), dtype=float)
            if curvature.size and (curvature.shape != (2, 2) or not np.isfinite(curvature).all() or
                                   not np.allclose(curvature, curvature.T)):
                raise ValueError('Invalid cached response curvature')
            compatible = (cache.get('conditions') == self._conditions() and
                          0 <= self.now-float(cache['time']) <= self.cfg.reuse_max_age_s and
                          abs(float(cache['voltage'])-self.target_voltage) <= self.cfg.reuse_voltage_tolerance_v and
                          abs(position[2]-self.origin[2])*1e6 <= self.cfg.tolerance_um)
        except (KeyError, TypeError, ValueError, IndexError):
            compatible = False
        if compatible:
            # Only copy validated values into the movement plan and journal.
            return dict(position_m=position, time=float(cache['time']), voltage=float(cache['voltage']),
                        conditions=self._conditions(),
                        surface={'curvature_per_um2': curvature.tolist()} if curvature.size else {})
        self._event('reuse_unavailable', reason='No compatible recent specimen, coordinate, laser, voltage and Z conditions')
        return None

    def _save_solution(self):
        if not self.cfg.reuse_previous:
            return
        self.v.laser_alignment_solution_cache = dict(position_m=list(self.best_position), time=self.now,
            voltage=self.target_voltage, conditions=self._conditions(), surface=self.last_usable_surface,
            session=self.session, evidence='verified detector response; always remeasure before reuse')

    def finish(self, reason, failed=False):
        self.reason = reason
        self.finished = True
        self.phase = 'failed' if failed else 'stopped'
        self.v.laser_alignment_cancel = True
        self.v.laser_alignment_epoch = ''
        try:
            self._event('finish', reason=reason)
        except OSError as exc:
            self.phase = 'failed'
            self.reason += f' Alignment metadata could not be written: {exc}'
        self._publish()

    def _move(self, target, after):
        self.cfg.check_position(target, self.origin)
        current = self._position()
        z_move = abs(target[2]-current[2]) > self.cfg.tolerance_um*1e-6
        if z_move and any(abs(target[i]-current[i]) > self.cfg.tolerance_um*1e-6 for i in (0, 1)):
            raise ValueError('Laser Z and XY must move separately.')
        identifier = uuid.uuid4().hex
        self.pending = (identifier, self.now, after)
        self.phase = 'moving'
        self.v.laser_alignment_epoch = ''
        self.v.laser_alignment_move_request = {
            'id': identifier, 'session': self.session, 'target_m': tuple(target),
            'speed_um_s': self.cfg.z_speed_um_s if z_move else self.cfg.xy_speed_um_s,
            'issued': self.now,
        }
        self._event('move', target_m=tuple(target), after=after)

    def _begin_scan(self, kind, centre, radius=None):
        self.kind = kind
        self.centre = tuple(centre)
        self.cfg.check_position(self.centre, self.origin)
        self.records = []
        self.surface = {}
        self.model_pending = False
        self.reuse_in_progress = False
        lower = [max(self.cfg.bounds_mm[2*i]*1e-3,
                     self.origin[i]-self.cfg.travel_um[i]*1e-6) for i in range(3)]
        upper = [min(self.cfg.bounds_mm[2*i+1]*1e-3,
                     self.origin[i]+self.cfg.travel_um[i]*1e-6) for i in range(3)]
        if kind == 'tracking':
            radius = self.cfg.tracking_step_um
            offsets = [(0, 0), (-radius, 0), (radius, 0), (0, -radius), (0, radius), (0, 0)]
            points = [(float(np.clip(centre[0]+x*1e-6, lower[0], upper[0])),
                       float(np.clip(centre[1]+y*1e-6, lower[1], upper[1])), centre[2])
                      for x, y in offsets]
        else:
            radius = radius or getattr(self.cfg, kind+'_range_um')
            step = getattr(self.cfg, kind+'_step_um')
            def axis(i):
                lo = max(lower[i], centre[i]-radius*1e-6)
                hi = min(upper[i], centre[i]+radius*1e-6)
                return np.linspace(lo, hi, min(21, int(math.ceil((hi-lo)*1e6/step-1e-9))+1))
            points = [tuple(centre)]
            if kind == 'focus':
                points.extend((centre[0], centre[1], float(z)) for z in axis(2))
            else:
                xs = axis(0)
                for row, y in enumerate(axis(1)):
                    points.extend((float(x), float(y), centre[2])
                                  for x in (xs if row % 2 == 0 else xs[::-1]))
            points.append(tuple(centre))  # repeated baseline detects time drift
        points = list(dict.fromkeys(points[:-1])) + [self.centre]
        # Validate the whole path before issuing its first command.
        for point in points:
            self.cfg.check_position(point, self.origin)
        self.scan_radius = radius
        self.scan_lower, self.scan_upper = lower, upper
        self.raster_points = points
        if self.cfg.search_strategy == 'hybrid' and kind in ('coarse', 'fine'):
            points = informative_cross(self.centre, lower, upper, radius, self.cfg.xy_speed_um_s)
            self.model_pending = True
            self.progress = 'Measuring centre, axes and diagonal response probes'
        else:
            self.progress = 'Measuring bounded '+kind+' scan'
        self.points = deque(points)
        self._event('scan_start', kind=kind, centre_m=centre, radius_um=radius,
                    voltage=self.target_voltage)
        self._next_point()

    def _next_point(self):
        point = self.points.popleft() if self.points else None
        if point is None and self.model_pending:
            self.model_pending = False
            self.surface = fit_local_response(self.records, self.centre, self.scan_radius,
                                              self.scan_lower, self.scan_upper)
            self._event('response_surface', **self.surface)
            visited = [r['position_m'] for r in self.records]
            if self.surface['usable']:
                proposal = tuple(self.surface['position_m'])
                self.cfg.check_position(proposal, self.origin)
                self.last_usable_surface = self.surface
                if not any(np.linalg.norm(np.subtract(proposal, p))*1e6 <= self.cfg.tolerance_um for p in visited):
                    self.points.append(proposal)
                self.progress = 'Remeasuring bounded model proposal and centre drift check'
            else:
                self.points.extend(p for p in self.raster_points[:-1] if not any(
                    np.linalg.norm(np.subtract(p, q))*1e6 <= self.cfg.tolerance_um for q in visited))
                self.progress = 'Raster fallback: '+self.surface['reason']
            self.points.append(self.centre)
            point = self.points.popleft()
        if point is None:
            self._complete_scan()
        else:
            self._move(point, 'observe')

    def _observe(self, validation=False):
        self.phase = 'verify' if validation else self.kind
        self.observation_start = self.now
        self.epoch = uuid.uuid4().hex
        self.v.laser_alignment_epoch = self.epoch

    def _measurement(self):
        data = self.v.laser_alignment_observation
        if (not data or data.get('epoch') != self.epoch or
                data.get('start', 0) < self.observation_start or
                not 0 <= self.now-data.get('time', 0) <= 2):
            return None
        duration = data['time']-data['start']
        minimum = self.cfg.min_dwell_s if self.cfg.adaptive_dwell else self.cfg.dwell_s
        if duration < minimum:
            return None
        count = int(data['count'])
        frequency = self.laser_signature[1]
        rate = count/(duration*frequency)*100
        error = math.sqrt(max(count, 1))/(duration*frequency)*100
        if self.cfg.adaptive_dwell:
            # Early rejection can only shorten a comparison, never certify an
            # improvement. Final verification still requires the signal test.
            reference = max(self.records, key=lambda r: r['rate_percent']) if self.records else None
            poor = (self.phase != 'verify' and reference is not None and self._signal(reference) and
                    rate+self.cfg.early_reject_sigma*error <
                    reference['rate_percent']-self.cfg.early_reject_sigma*reference['error_percent'])
            precise = count >= self.cfg.min_events and 1/math.sqrt(max(count, 1)) <= self.cfg.target_relative_error
            if not (poor or precise or duration >= self.cfg.max_dwell_s):
                self.progress = f'Collecting detector evidence: {count} ions, {duration:.1f}/{self.cfg.max_dwell_s:g} s'
                return None
            decision = 'clearly_poor' if poor else 'count_precision' if precise else 'maximum_dwell'
        else:
            decision = 'fixed_dwell'
        record = self._record(rate, error, count, {})
        record.update(duration_s=float(duration), dwell_decision=decision)
        return record

    def _record(self, rate, error, count, quality):
        return {'rate_percent': rate, 'error_percent': error, 'count': count,
                'quality': quality, 'position_m': self._position(),
                'voltage': float(self.v.specimen_voltage), 'time': self.now,
                'phase': self.kind, 'session': self.session, 'origin_m': self.origin}

    def _signal(self, record):
        return (record['count'] >= self.cfg.min_events and record['rate_percent']-3*record['error_percent'] >
                max(self.cfg.min_rate_percent, self.cfg.background_rate_percent))

    def _quality_ok(self, candidate, baseline):
        # Detector-only operation: legacy spectral ROI settings never influence
        # a motor decision. Focus means best measured response, not best MRP.
        return True

    def _complete_scan(self):
        baseline = self.records[0]
        candidates = [r for r in self.records if self._signal(r) and self._quality_ok(r, baseline)]
        if not candidates:
            self._move(self.centre, 'retry')
            return
        best = max(candidates, key=lambda r: r['rate_percent'])
        # A small apparent increase or a drifted baseline is not evidence for a move.
        self.reference = max((self.records[0], self.records[-1]), key=lambda r: r['rate_percent'])
        reference = self.reference['rate_percent']
        threshold = max(reference*self.cfg.improvement_fraction,
                        3*math.hypot(best['error_percent'], self.reference['error_percent']))
        if self._signal(baseline) and best['rate_percent']-reference <= threshold:
            best = baseline
        if (self.reuse_in_progress and
                np.linalg.norm(np.subtract(best['position_m'], self.reuse_target))*1e6 > self.cfg.tolerance_um):
            self._event('reuse_rejected', reason='Cached position did not improve on the fresh centre')
            self._move(self.centre, 'retry')
            return
        self.candidate = best
        self._move(best['position_m'], 'verify')

    def _accepted(self, measurement):
        if not self._signal(measurement) or not self._quality_ok(measurement, self.candidate):
            self._move(self.centre, 'retry')
            return
        # Repeatability check with Poisson uncertainty and a bounded relative tolerance.
        discrepancy = self.candidate['rate_percent']-measurement['rate_percent']
        tolerance = max(self.candidate['rate_percent']*0.2,
                        3*math.hypot(measurement['error_percent'], self.candidate['error_percent']))
        if discrepancy > tolerance:
            self._move(self.centre, 'retry')
            return
        displaced = any(abs(self.candidate['position_m'][i]-self.centre[i]) >
                        self.cfg.tolerance_um*1e-6 for i in range(3))
        if displaced and self._signal(self.reference):
            threshold = max(self.reference['rate_percent']*self.cfg.improvement_fraction,
                            3*math.hypot(measurement['error_percent'], self.reference['error_percent']))
            if (measurement['rate_percent']-self.reference['rate_percent'] <= threshold
                    or not self._quality_ok(measurement, self.records[0])):
                self._event('verification_rejected', reason='Verified improvement over the scan centre was insufficient')
                self._move(self.centre, 'retry')
                return
        self.best_position = measurement['position_m']
        reused = self.reuse_in_progress
        if reused:
            self._event('reuse_verified', position_m=self.best_position)
            self.reuse_in_progress = False
        self._event('accepted', measurement=measurement)
        self.fine_failures = 0
        if self.kind == 'coarse':
            self.final_xy = False
            self.fine_round = 0
            self._begin_scan('fine', self.best_position, self.reuse_radius if reused else None)
        elif self.kind == 'fine' and self.fine_round < 1:
            self.fine_round += 1
            self._begin_scan('fine', self.best_position, self.cfg.fine_range_um/2)
        elif self.kind == 'fine' and not self.final_xy:
            self._begin_scan('focus', self.best_position)
        elif self.kind == 'focus':
            self.final_xy = True
            self.fine_round = 0
            self._begin_scan('fine', self.best_position)
        else:
            self.phase = 'tracking_wait'
            self.wait_started = self.now
            self.search_started = None
            self.v.laser_alignment_epoch = ''
            self._event('aligned', position_m=self.best_position)
            self.progress = 'Aligned from repeated detector evidence; user laser settings unchanged'
            self._save_solution()

    def _retry(self):
        self._event('scan_unsuccessful', kind=self.kind)
        if self.reuse_in_progress:
            self.reuse_in_progress = False
            self._event('reuse_rejected', reason='Fresh verification failed; bounded coarse search at unchanged DC')
            self._begin_scan('coarse', self.centre)
            return
        if self.kind == 'coarse':
            if self.target_voltage >= self.limit:
                self.finish('No laser alignment before the DC voltage ceiling.', failed=True)
                return
            # Already returned to scan centre; only now may the experiment owner ramp DC.
            self.target_voltage = min(self.limit, self.target_voltage+self.cfg.voltage_increment)
            self.phase = 'ramp'
            self._event('voltage_increment', target_voltage=self.target_voltage,
                        increment=self.cfg.voltage_increment)
            return
        if self.kind == 'tracking':
            self.kind = 'fine'
            self.final_xy = True
        self.fine_failures += 1
        if self.fine_failures < 3:
            self._begin_scan(self.kind, self.centre)
        else:
            self.recoveries += 1
            if self.recoveries > self.cfg.max_recoveries:
                self.finish('Laser alignment recovery limit reached.', failed=True)
            else:
                self.fine_failures = 0
                self._begin_scan('coarse', self.centre)

    def voltage_step(self, voltage, dt):
        if not self.owns_voltage:
            return None
        if self.phase != 'ramp':
            return 0.0
        return max(0., min(self.target_voltage-voltage, self.cfg.ramp_v_s*dt))

    def tick(self, now=None):
        self.now = time.monotonic() if now is None else now
        if self.finished:
            return
        try:
            reply = self.v.laser_alignment_move_status
            if self.pending and reply.get('id') == self.pending[0] and reply.get('fault'):
                raise ValueError(reply.get('error', 'Laser stage service fault.'))
            if (self.v.stop_flag or not self.v.start_flag or self.v.laser_alignment_cancel):
                self.finish('Laser alignment stopped by operator / experiment.')
                return
            if self.v.electrode_out or self.v.flag_tdc_failure:
                raise ValueError('Electrode / detector interlock lost during laser alignment.')
            if self.v.vdc_hold:
                self.finish('Laser alignment stopped because DC Hold was selected.')
                return
            if self.v.pulse_mode != 'Laser' or self._signature() != self.laser_signature:
                raise ValueError('Laser mode, power, wavelength or frequency changed during alignment.')
            if not -0.1 <= self.now-self.v.laser_alignment_heartbeat <= 3:
                raise ValueError('Laser stage service is unresponsive.')
            self.cfg.check_position(self._position(), self.origin)
            if self.owns_voltage and float(self.v.specimen_voltage) > self.limit+0.1:
                raise ValueError('Laser alignment voltage ceiling exceeded.')
            if (self.owns_voltage and float(self.v.detection_rate) > 0 and
                    float(self.v.detection_rate_current) > self.cfg.max_rate_multiple*float(self.v.detection_rate)):
                raise ValueError('Detection rate exceeded the laser alignment overshoot limit.')
            if self.phase not in ('ramp', 'tracking_wait') and abs(float(self.v.specimen_voltage)-self.target_voltage) > 0.1:
                raise ValueError('DC changed during a fixed-voltage laser scan.')
            if self.phase != 'tracking_wait':
                search_started = getattr(self, 'search_started', self.started)
                if self.now-search_started > self.cfg.max_duration_s:
                    raise ValueError('Laser alignment search timed out.')
            if self.pending:
                identifier, issued, after = self.pending
                reply = self.v.laser_alignment_move_status
                if self.now-issued > self.cfg.move_timeout_s:
                    raise ValueError('Laser stage movement timed out.')
                if reply.get('id') != identifier:
                    return
                if reply.get('state') == 'error':
                    raise ValueError(reply.get('error', 'Laser motion failed.'))
                if reply.get('state') != 'done':
                    return
                self.pending = None
                if after == 'observe': self._observe()
                elif after == 'verify': self._observe(validation=True)
                elif after == 'retry': self._retry()
            elif self.phase == 'ramp':
                if abs(float(self.v.specimen_voltage)-self.target_voltage) <= 0.1:
                    self._begin_scan('coarse', self.centre)
            elif self.phase == 'tracking_wait':
                if self.v.laser_alignment_tracking and self.now-self.wait_started >= self.cfg.tracking_interval_s:
                    if float(self.v.specimen_voltage) > self.limit:
                        self.reason = 'Tracking paused above the laser alignment voltage ceiling.'
                        return
                    self.target_voltage = float(self.v.specimen_voltage)
                    self.reason = ''
                    self.search_started = self.now
                    self._begin_scan('tracking', self.best_position)
            else:
                measured = self._measurement()
                deadline = self.cfg.max_dwell_s if self.cfg.adaptive_dwell else self.cfg.dwell_s
                if measured is None and self.now-self.observation_start >= deadline+2:
                    measured = self._record(0., 0., 0, {})
                if measured is not None:
                    self.measurements += 1
                    self.ions_observed += measured['count']
                    self._event('observation', **measured)
                    self.v.laser_alignment_plot = measured
                    if self.phase == 'verify': self._accepted(measured)
                    else:
                        self.records.append(measured)
                        self._next_point()
        except Exception as exc:
            self.finish(str(exc), failed=True)
        finally:
            self._publish()
