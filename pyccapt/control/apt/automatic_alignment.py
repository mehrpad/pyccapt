"""Non-blocking per-sample alignment state machine, driven by the experiment.

Voltage commands remain owned by APT_Exp_Control. Stage commands are acknowledged
by the single stage connection in the main process. This module does no device I/O.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import numpy as np

from pyccapt.control.apt.alignment_config import AlignmentConfig, coarse_positions
from pyccapt.control.apt.alignment_vision import Footprint, estimate_footprint


class AutomaticAlignment:
    def __init__(self, variables, conf, now=None, analyser=estimate_footprint):
        self.v = variables
        self.cfg = AlignmentConfig(**dict(variables.alignment_settings))
        self.cfg.validate()
        self.origin = tuple(variables.alignment_sample_position)
        self.fine_origin = None
        self.cfg.validate_motion([self.origin])
        self.limit = min(self.cfg.max_voltage, float(variables.vdc_max), float(conf['max_vdc']))
        if not float(variables.vdc_min) <= self.cfg.start_voltage <= self.limit:
            raise ValueError('Alignment start voltage must be within this experiment voltage range.')
        self.target_rate = float(variables.detection_rate)
        if self.target_rate <= 0:
            raise ValueError('Automatic alignment requires a positive target detection rate.')
        self.started = time.monotonic() if now is None else now
        self.now = self.started
        self.target_voltage = self.cfg.start_voltage
        self.phase = 'ramp'
        self.ramp_destination = 'coarse'
        self.attempts = 0
        self.outcome = ''
        self.reason = ''
        self.pending = None
        self.last_seq = -1
        self.last_plot_time = float('-inf')
        self.fit = Footprint(False, 'Waiting for fresh events')
        self.analyser = analyser
        self.stable_since = None
        self.stable_sequence = None
        self.loss_since = None
        self.events_path = Path(variables.path_meta) / 'alignment.jsonl'
        self._event('start', settings=self.cfg.snapshot(), sample=variables.alignment_sample,
                    saved_position_m=self.origin, effective_voltage_limit=self.limit,
                    sequence_id=getattr(variables, 'alignment_sequence_id', ''))
        self.v.alignment_outcome = ''
        self._publish()

    @property
    def active(self):
        return self.phase not in ('aligned', 'finished')

    def _event(self, event, **details):
        record = {'elapsed_s': self.now-self.started, 'phase': self.phase,
                  'event': event, 'attempt': self.attempts, **details}
        with self.events_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')

    def _publish(self):
        self.v.alignment_status = {'phase': self.phase, 'attempt': self.attempts,
                                   'target_voltage': self.target_voltage, 'reason': self.reason,
                                   'footprint': self.fit.snapshot(), 'sample': self.v.alignment_sample}

    def finish(self, outcome, reason):
        self.outcome, self.reason = outcome, reason
        self.phase = 'finished'
        self.v.alignment_outcome = outcome
        self.v.alignment_cancel_motion = True
        self._event('finish', outcome=outcome, reason=reason)
        self._publish()

    def interrupt(self, reason):
        if self.phase != 'finished':
            self.finish('cancelled', reason)

    def _position(self):
        snapshot = self.v.stage_position_snapshot
        if (len(snapshot) != 4 or not np.isfinite(float(snapshot[3]))
                or self.now-float(snapshot[3]) > 2 or self.now < float(snapshot[3])):
            raise ValueError('Stage position is stale; automatic alignment stopped.')
        position = tuple(float(x) for x in snapshot[:3])
        self.cfg.check_position(position)
        return position

    def _move(self, target, after, *, z=False):
        self.cfg.check_position(target)
        identifier = uuid.uuid4().hex
        self.pending = (identifier, self.now, after)
        self.phase = 'moving'
        request = {'id': identifier, 'target_m': tuple(target), 'kind': 'alignment',
                   'speed_um_s': self.cfg.z_speed_um_s if z else self.cfg.xy_speed_um_s,
                   'issued': self.now}
        if after == 'observe_fine':
            self.cfg.check_fine_position(target, self.fine_origin)
            request['fine_origin_m'] = self.fine_origin
        self.v.alignment_move_request = request
        self._event('move_requested', **request)
        self._publish()

    def _observe(self, fine=False):
        self.phase = 'fine' if fine else 'coarse'
        self.observation_start = self.now
        self.v.alignment_window_epoch = uuid.uuid4().hex
        self.last_seq = -1
        self.fit = Footprint(False, 'Collecting events after settling')
        self.stable_since = self.stable_sequence = self.loss_since = None
        self._publish()

    def _start_coarse(self):
        self.grid = iter(coarse_positions(self.origin, self.cfg))
        self._next_coarse()

    def _next_coarse(self):
        target = next(self.grid, None)
        if target is None:
            self._move(self.origin, 'increase_voltage')
        else:
            self._move(target, 'observe_coarse')

    def _after_move(self, action):
        if action == 'observe_coarse':
            self._observe()
        elif action == 'observe_fine':
            self._observe(True)
        elif action == 'increase_voltage':
            if self.target_voltage >= self.limit:
                self.finish('voltage_limit', 'No alignment at the configured voltage limit; skipping sample.')
            else:
                self.target_voltage = min(self.limit, self.target_voltage+self.cfg.voltage_increment)
                self.phase = 'ramp'
                self.ramp_destination = 'coarse'
                self._event('voltage_step', voltage=self.target_voltage, returned_to_origin=True)
        elif action == 'recover_xy':
            self._move(self.origin, 'restart_coarse')
        elif action == 'restart_coarse':
            self._start_coarse()

    def _recover(self):
        self._event('signal_lost')
        if self.attempts >= self.cfg.max_attempts:
            self.finish('attempt_limit', 'Five coarse/fine attempts failed.' if self.cfg.max_attempts == 5
                        else f'{self.cfg.max_attempts} coarse/fine attempts failed.')
            return
        # Retraction is a separate recovery phase. Coarse searching always uses
        # the saved Z, never the closer Z reached by a fine approach.
        position = self._position()
        self._move((position[0], position[1], self.origin[2]), 'recover_xy', z=True)

    def _fresh_fit(self):
        data = self.v.alignment_events
        if (not data or data.get('epoch') != self.v.alignment_window_epoch
                or not 0 <= self.now-data['time'] <= 1.0
                or data['first_time'] < self.observation_start
                or self.now-data['first_time'] > self.cfg.window_max_age_s):
            return Footprint(False, 'Waiting for fresh event window'), -1
        sequence = data['sequence']
        if sequence != self.last_seq:
            self.last_seq = sequence
            self.fit = self.analyser(data['points_mm'], self.cfg.detector_radius_mm, self.cfg.window_ions)
            self._event('observation', voltage=float(self.v.specimen_voltage),
                        detection_rate=float(self.v.detection_rate_current), sequence=sequence,
                        position_m=self._position(), footprint=self.fit.snapshot())
            self._publish()
        return self.fit, sequence

    def _stable(self, condition, sequence):
        if not condition:
            self.stable_since = self.stable_sequence = None
            return False
        if self.stable_since is None:
            self.stable_since, self.stable_sequence = self.now, sequence
        return (self.now-self.stable_since >= self.cfg.stable_s
                and sequence-self.stable_sequence >= self.cfg.window_ions)

    def tick(self, voltage, rate, now=None):
        self.now = time.monotonic() if now is None else now
        self._publish_plot(voltage, rate)
        if not self.active:
            return
        try:
            if not np.isfinite(voltage) or not np.isfinite(rate):
                raise ValueError('Invalid voltage or detection-rate reading.')
            if self.v.stop_flag:
                self.finish('cancelled', 'Alignment cancelled by Stop.')
                return
            if self.v.alignment_cancel_motion:
                reply = self.v.alignment_move_status
                if reply.get('state') == 'error':
                    self.finish('fault', reply.get('error', 'Stage movement failed.'))
                else:
                    self.finish('cancelled', 'Alignment cancelled.')
                return
            self._position()
            if not 0 <= self.now-self.v.alignment_stage_heartbeat <= 2:
                raise ValueError('Stage controller heartbeat lost.')
            if self.now-self.started > self.cfg.timeout_s:
                self.finish('timeout', 'Maximum automatic alignment duration reached.')
                return
            if self.phase == 'moving':
                identifier, started, after = self.pending
                reply = self.v.alignment_move_status
                if self.now-started > self.cfg.move_timeout_s + self.cfg.settle_s + 2:
                    raise ValueError('Alignment stage command timed out.')
                if reply.get('id') == identifier:
                    if reply.get('state') == 'error':
                        raise ValueError(reply.get('error', 'Stage movement failed.'))
                    if reply.get('state') == 'done':
                        self._event('move_complete', position_m=self._position())
                        self._after_move(after)
                return
            if self.phase == 'ramp':
                if abs(voltage-self.target_voltage) <= 0.5:
                    if self.ramp_destination == 'coarse':
                        self._start_coarse()
                    else:
                        self._observe(True)
                return
            fit, sequence = self._fresh_fit()
            fraction = rate/self.target_rate
            if self.phase == 'coarse':
                if self._stable(fit.valid and fraction >= self.cfg.entry_fraction, sequence):
                    self.attempts += 1
                    self.fine_origin = self._position()
                    self._event('fine_started', centre_m=self.fine_origin,
                                range_um=self.cfg.fine_xy_range_um)
                    self._observe(True)
                elif (self.now-self.observation_start >= self.cfg.dwell_s
                      and self.stable_since is None):
                    self._next_coarse()
                return
            if self.phase == 'fine':
                if not fit.valid or fraction < self.cfg.loss_fraction:
                    if self.now-self.observation_start < self.cfg.dwell_s:
                        return
                    if self.loss_since is None:
                        self.loss_since = self.now
                    if self.now-self.loss_since >= self.cfg.loss_s:
                        self._recover()
                    return
                self.loss_since = None
                centred = np.linalg.norm(fit.centre_mm) <= self.cfg.centre_tolerance*self.cfg.detector_radius_mm
                if self._stable(centred and fraction >= self.cfg.finish_fraction, sequence):
                    self.phase = 'aligned'
                    self.v.alignment_outcome = 'aligned'
                    self._event('aligned', footprint=fit.snapshot(), rate=rate)
                    self._publish()
                    return
                if centred and fraction >= self.cfg.finish_fraction:
                    return  # Wait for independent windows; never creep forward.
                position = np.asarray(self._position())
                if not centred:
                    correction = -np.linalg.solve(np.asarray(self.cfg.xy_jacobian_mm_per_um), fit.centre_mm)
                    correction = np.clip(correction, -self.cfg.fine_xy_step_um, self.cfg.fine_xy_step_um)
                    target = position.copy()
                    target[:2] += correction*1e-6
                    if np.any(np.abs((target[:2]-np.asarray(self.origin[:2]))*1e6) > np.asarray(self.cfg.xy_range_um)+1e-8):
                        self._recover()
                        return
                    try:
                        self.cfg.check_fine_position(target, self.fine_origin)
                    except ValueError:
                        self._event('fine_range_reached', centre_m=self.fine_origin)
                        self._recover()
                        return
                    self._move(tuple(target), 'observe_fine')
                elif (((fit.radius_mm+fit.radius_uncertainty_mm)/self.cfg.detector_radius_mm)**2
                      < self.cfg.area_target and self.cfg.approach_enabled
                      and fraction < self.cfg.finish_fraction):
                    advance = self.cfg.z_direction*(position[2]-self.origin[2])*1e6
                    if advance+self.cfg.z_step_um > self.cfg.z_max_advance_um + 1e-9:
                        self.finish('clearance_limit', 'Calibrated approach limit reached before alignment.')
                        return
                    position[2] += self.cfg.z_direction*self.cfg.z_step_um*1e-6
                    self._move(tuple(position), 'observe_fine', z=True)
                elif self.target_voltage < self.limit:
                    self.target_voltage = min(self.limit, self.target_voltage+self.cfg.voltage_increment)
                    self.phase, self.ramp_destination = 'ramp', 'fine'
                    self._event('stationary_fine_voltage_step', voltage=self.target_voltage)
                else:
                    self.finish('voltage_limit', 'Target rate not reached at alignment voltage limit; skipping sample.')
        except (ValueError, TypeError, KeyError) as exc:
            self.finish('fault', str(exc))

    def _publish_plot(self, voltage, rate):
        """Small independent monitor snapshot; never consume detector/viz rings."""
        if self.phase == 'finished' or self.now-self.last_plot_time < 0.2:
            return
        try:
            position = self._position()
            if not np.isfinite(voltage) or not np.isfinite(rate) or rate < 0:
                return
        except (ValueError, TypeError):
            return
        self.v.alignment_plot_snapshot = {
            'time': self.now, 'sequence_id': getattr(self.v, 'alignment_sequence_id', ''),
            'sample': self.v.alignment_sample, 'origin_m': self.origin,
            'position_m': position, 'rate_percent': float(rate),
            'target_percent': self.target_rate, 'voltage': float(voltage), 'phase': self.phase,
        }
        self.last_plot_time = self.now

    def voltage_step(self, voltage, dt):
        """Zero throughout scans/moves; bounded ramp only at stationary phases."""
        if self.phase != 'ramp':
            return 0.0
        return float(np.clip(self.target_voltage-voltage, -self.cfg.ramp_v_s*dt, self.cfg.ramp_v_s*dt))
