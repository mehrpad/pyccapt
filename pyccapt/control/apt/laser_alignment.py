"""Laser-stage search and tracking. Device I/O stays with existing device owners."""
import json
import math
import time
import uuid
from pathlib import Path

import numpy as np

from pyccapt.control.apt.laser_alignment_config import LaserAlignmentConfig
from pyccapt.control.apt.laser_alignment_data import peak_quality


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
        self.v.laser_alignment_cancel = False
        self.v.laser_alignment_run = {'id': self.session, 'origin_m': self.origin, 'settings': self.cfg.snapshot()}
        self.laser_signature = self._signature()
        self._event('start', settings=self.cfg.snapshot(), origin_m=self.origin, mode=mode,
                    voltage_limit=self.limit, metric='detection_rate_at_fixed_dc')
        self._publish()
        self._begin_scan(mode, self.origin)

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
        }

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
        self.records = []
        if kind == 'tracking':
            radius = self.cfg.tracking_step_um
            offsets = [(0, 0), (-radius, 0), (radius, 0), (0, -radius), (0, radius), (0, 0)]
            points = [(centre[0]+x*1e-6, centre[1]+y*1e-6, centre[2]) for x, y in offsets]
        else:
            radius = radius or getattr(self.cfg, kind+'_range_um')
            step = getattr(self.cfg, kind+'_step_um')
            axis = np.linspace(-radius, radius, min(21, int(math.ceil(2*radius/step))+1))
            points = [tuple(centre)]
            if kind == 'focus':
                points.extend((centre[0], centre[1], centre[2]+float(z)*1e-6) for z in axis)
            else:
                for row, y in enumerate(axis):
                    points.extend((centre[0]+float(x)*1e-6, centre[1]+float(y)*1e-6, centre[2])
                                  for x in (axis if row % 2 == 0 else axis[::-1]))
            points.append(tuple(centre))  # repeated baseline detects time drift
        # Validate the whole path before issuing its first command.
        for point in points:
            self.cfg.check_position(point, self.origin)
        self.points = iter(points)
        self._event('scan_start', kind=kind, centre_m=centre, radius_um=radius,
                    voltage=self.target_voltage)
        self._next_point()

    def _next_point(self):
        point = next(self.points, None)
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
        if duration < self.cfg.dwell_s:
            return None
        count = int(data['count'])
        frequency = self.laser_signature[1]
        rate = count/(duration*frequency)*100
        error = math.sqrt(max(count, 1))/(duration*frequency)*100
        quality = peak_quality(data.get('tof_ns', []), self.cfg)
        return self._record(rate, error, count, quality)

    def _record(self, rate, error, count, quality):
        return {'rate_percent': rate, 'error_percent': error, 'count': count,
                'quality': quality, 'position_m': self._position(),
                'voltage': float(self.v.specimen_voltage), 'time': self.now,
                'phase': self.kind, 'session': self.session, 'origin_m': self.origin}

    def _signal(self, record):
        return (record['count'] >= self.cfg.min_events and record['rate_percent']-3*record['error_percent'] >
                max(self.cfg.min_rate_percent, self.cfg.background_rate_percent))

    def _quality_ok(self, candidate, baseline):
        if self.kind != 'focus' or not self.cfg.quality_peak_ns:
            return True
        a, b = candidate['quality'], baseline['quality']
        if not a or not b:
            return False
        return (a['width_ns'] <= b['width_ns']*(1+self.cfg.quality_max_degradation) and
                a['tail_fraction'] <= b['tail_fraction']+self.cfg.quality_max_degradation)

    def _complete_scan(self):
        baseline = self.records[0]
        candidates = [r for r in self.records if self._signal(r) and self._quality_ok(r, baseline)]
        if not candidates:
            self._move(self.centre, 'retry')
            return
        best = max(candidates, key=lambda r: r['rate_percent'])
        # A small apparent increase or a drifted baseline is not evidence for a move.
        reference = max(self.records[0]['rate_percent'], self.records[-1]['rate_percent'])
        threshold = max(reference*self.cfg.improvement_fraction,
                        3*math.hypot(best['error_percent'], baseline['error_percent']))
        if self._signal(baseline) and best['rate_percent']-reference <= threshold:
            best = baseline
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
        self.best_position = measurement['position_m']
        self._event('accepted', measurement=measurement)
        self.fine_failures = 0
        if self.kind == 'coarse':
            self.final_xy = False
            self.fine_round = 0
            self._begin_scan('fine', self.best_position)
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

    def _retry(self):
        self._event('scan_unsuccessful', kind=self.kind)
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
                if measured is None and self.now-self.observation_start >= self.cfg.dwell_s+2:
                    measured = self._record(0., 0., 0, {})
                if measured is not None:
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
