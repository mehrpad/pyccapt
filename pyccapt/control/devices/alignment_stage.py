"""Acknowledged automatic moves using the Stage GUI's existing device handle."""
from __future__ import annotations

import time

import numpy as np

from pyccapt.control.apt.alignment_config import AlignmentConfig


class AlignmentStageService:
    def __init__(self, variables, device_getter):
        self.v = variables
        self.device_getter = device_getter
        self.active = None
        self.last_id = None
        self.settled_since = None
        self.last_snapshot_at = float('-inf')

    def _reply(self, state, error=''):
        self.v.alignment_move_status = {'id': self.last_id, 'state': state, 'error': error}

    def _publish_position(self, position_map, now):
        position = tuple(float(position_map[axis]) for axis in 'xyz')
        cfg = AlignmentConfig(**dict(self.v.alignment_settings))
        cfg.check_position(position)
        self.v.stage_position_snapshot = (*position, now)
        self.v.stage_pos_x, self.v.stage_pos_y, self.v.stage_pos_z = position
        self.v.stage_pos_updated_at = now
        self.last_snapshot_at = now
        return position

    def cancel(self, reason='Automatic movement cancelled'):
        device = self.device_getter()
        if device is not None:
            device.stop()
        if self.active is not None:
            self._reply('error', reason)
        self.active = None
        self.settled_since = None

    def _check_interlocks(self, kind, now):
        if not self.v.physical_estop_ok:
            raise ValueError('Physical interlock opened during automatic stage movement.')
        if kind == 'transfer':
            if not self.v.hardware_safe:
                raise ValueError('Sample transfer requires completed output shutdown.')
        elif kind == 'alignment':
            if not self.v.start_flag or self.v.experiment_state != 'running':
                raise ValueError('Alignment movement requested outside a running experiment.')
            if self.v.electrode_out or self.v.flag_tdc_failure:
                raise ValueError('Electrode / detector interlock lost during sample alignment.')
            if not -0.1 <= now-float(self.v.experiment_heartbeat_monotonic) <= 3:
                raise ValueError('Experiment heartbeat lost; stopping sample stage movement.')
        else:
            raise ValueError('Unknown automatic stage movement kind.')

    def tick(self, now=None):
        live_clock = now is None
        now = time.monotonic() if now is None else now
        self.v.alignment_stage_heartbeat = now
        if not self.v.automatic_alignment_enabled:
            if self.active is not None:
                self.cancel()
            return
        if self.v.alignment_cancel_motion or self.v.stop_flag:
            if self.active is not None:
                self.cancel()
            return
        device = self.device_getter()
        request = self.v.alignment_move_request
        try:
            if request and request.get('id') != self.last_id:
                if self.active is not None:
                    raise ValueError('Concurrent automatic stage commands are not permitted.')
                self.last_id = request['id']
                if device is None:
                    raise ValueError('Sample stage is not connected.')
                device.validate_alignment_state()
                cfg = AlignmentConfig(**dict(self.v.alignment_settings))
                cfg.validate_motion([self.v.alignment_sample_position])
                checked_at = time.monotonic() if live_clock else now
                if not 0 <= checked_at-float(request['issued']) <= 2:
                    raise ValueError('Expired automatic stage command.')
                self._check_interlocks(request['kind'], checked_at)
                target = np.asarray(request['target_m'], dtype=float)
                cfg.check_position(target)
                current = device.get_position()
                position = np.array([current[axis] for axis in 'xyz'])
                changed = np.abs(target-position) > cfg.position_tolerance_um*1e-6
                if changed[2] and changed[:2].any():
                    raise ValueError('Automatic Z and lateral movement must be separate.')
                origin = np.asarray(self.v.alignment_sample_position, dtype=float)
                if request['kind'] == 'transfer':
                    if changed[:2].any() and abs(position[2]-cfg.transfer_z_mm*1e-3) > cfg.position_tolerance_um*1e-6:
                        raise ValueError('Lateral sample transfer requires the calibrated clearance Z.')
                    if changed[2]:
                        retract_target = abs(target[2]-cfg.transfer_z_mm*1e-3) <= cfg.position_tolerance_um*1e-6
                        sample_target = (abs(target[2]-origin[2]) <= cfg.position_tolerance_um*1e-6
                                         and np.all(np.abs(target[:2]-origin[:2]) <= cfg.position_tolerance_um*1e-6))
                        if not retract_target and not sample_target:
                            raise ValueError('Transfer Z must follow the calibrated clearance/sample path.')
                else:
                    if 'fine_origin_m' in request:
                        cfg.check_position(request['fine_origin_m'])
                        cfg.check_fine_position(target, request['fine_origin_m'])
                    if np.any(np.abs((target[:2]-origin[:2])*1e6) > np.asarray(cfg.xy_range_um)+1e-8):
                        raise ValueError('Alignment request exceeds the sample XY search range.')
                    advance = cfg.z_direction*(target[2]-origin[2])*1e6
                    allowed = cfg.z_max_advance_um if cfg.approach_enabled else 0.0
                    if not -cfg.position_tolerance_um <= advance <= allowed+1e-8:
                        raise ValueError('Alignment request exceeds the permitted sample approach.')
                    if changed[2] and not cfg.approach_enabled:
                        raise ValueError('Automatic approach is disabled.')
                speed = float(request['speed_um_s'])
                ceiling = (cfg.transfer_speed_um_s if request['kind'] == 'transfer'
                           else cfg.z_speed_um_s if changed[2]
                           else cfg.fine_xy_speed_um_s if 'fine_origin_m' in request
                           else cfg.xy_speed_um_s)
                if not np.isfinite(speed) or not 0 < speed <= ceiling:
                    raise ValueError('Automatic stage velocity exceeds calibrated limit.')
                if device.is_moving():
                    raise ValueError('Stage is already moving before an automatic command.')
                self.active = (target, now, cfg, request['kind'])
                self.settled_since = None
                self._reply('moving')
                device.move_absolute(**{axis+'_m': float(value) if change else None
                                        for axis, value, change in zip('xyz', target, changed)},
                                     velocity_m_s=speed*1e-6, wait=False)
            if self.active is None:
                # The experiment needs a fresh position even while voltage is
                # ramping or the detector is observing a stationary stage.
                # Do not depend on the Stage window's separate display timer.
                if now-self.last_snapshot_at >= 0.5:
                    if device is None:
                        raise ValueError('Sample stage disconnected during automatic alignment.')
                    self._publish_position(device.get_position(), time.monotonic() if live_clock else now)
                return
            target, started, cfg, kind = self.active
            self._check_interlocks(kind, time.monotonic() if live_clock else now)
            device.validate_alignment_state()
            if now-started > cfg.move_timeout_s:
                raise ValueError('Automatic stage movement timed out.')
            position = self._publish_position(device.get_position(), time.monotonic() if live_clock else now)
            reached = not device.is_moving() and np.all(np.abs(np.asarray(position)-target) <= cfg.position_tolerance_um*1e-6)
            if not reached:
                self.settled_since = None
            elif self.settled_since is None:
                self.settled_since = now
            elif now-self.settled_since >= cfg.settle_s:
                self._reply('done')
                self.active = None
        except Exception as exc:
            if device is not None:
                try:
                    device.stop()
                except Exception:
                    pass  # Publish the fault even if controller communication failed.
            self._reply('error', str(exc))
            self.active = None
            self.v.alignment_cancel_motion = True


def transfer_waypoints(current, target, cfg):
    """Retract Z, traverse XY at clearance Z, then approach saved Z."""
    cfg.check_position(current)
    cfg.check_position(target)
    z = cfg.transfer_z_mm*1e-3
    if cfg.z_direction*(z-current[2]) > 0 or cfg.z_direction*(z-target[2]) > 0:
        raise ValueError('Configured transfer Z does not retract from both positions.')
    points = [(current[0], current[1], z), (target[0], target[1], z), tuple(target)]
    for point in points:
        cfg.check_position(point)
    return points
