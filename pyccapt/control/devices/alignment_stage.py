"""Acknowledged automatic moves using the Stage GUI's existing device handle."""
from __future__ import annotations

import logging
import time

import numpy as np

from pyccapt.control.apt.alignment_config import AlignmentConfig
from pyccapt.control.core.alignment_diagnostics import transfer_event


class AlignmentStageService:
    def __init__(self, variables, device_getter):
        self.v = variables
        self.device_getter = device_getter
        self.active = None
        self.last_id = None
        self.settled_since = None
        self.last_snapshot_at = float('-inf')
        self.last_transfer_detail = None
        self.last_transfer_log_at = float('-inf')

    def _reply(self, state, error='', **details):
        self.v.alignment_move_status = {'id': self.last_id, 'state': state, 'error': error, **details}

    def _publish_position(self, position_map, now):
        position = tuple(float(position_map[axis]) for axis in 'xyz')
        cfg = AlignmentConfig(**dict(self.v.alignment_settings))
        cfg.check_position(position)
        request = self.v.alignment_move_request
        if request and request.get('kind') == 'alignment' and self.v.start_flag:
            origin = np.asarray(self.v.alignment_sample_position, dtype=float)
            advance = cfg.z_direction*(position[2]-origin[2])*1e6
            allowed = cfg.z_max_advance_um if cfg.approach_enabled else 0.
            if advance > allowed+cfg.position_tolerance_um:
                raise ValueError('Measured sample Z exceeds the maximum permitted approach.')
            if np.any(np.abs((np.asarray(position[:2])-origin[:2])*1e6)
                      > np.asarray(cfg.xy_range_um)+cfg.position_tolerance_um):
                raise ValueError('Measured sample XY exceeds the coarse search envelope.')
            if 'fine_origin_m' in request:
                centre = np.asarray(request['fine_origin_m'], dtype=float)
                if np.any(np.abs((np.asarray(position[:2])-centre[:2])*1e6)
                          > cfg.fine_xy_range_um+cfg.position_tolerance_um):
                    raise ValueError('Measured sample XY exceeds the fine search envelope.')
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
            if self.active[3] == 'transfer':
                try:
                    transfer_event(self.v, 'motion_cancelled', id=self.last_id, reason=reason,
                                   status=self.v.alignment_move_status)
                except Exception:
                    logging.getLogger('pyccapt.gui').exception('Could not record transfer cancellation')
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
                position = np.asarray(self._publish_position(current, checked_at))
                changed = np.abs(target-position) > cfg.position_tolerance_um*1e-6
                if 'axes' in request:
                    axes = tuple(request['axes'])
                    if axes not in (('x',), ('y',), ('x', 'y'), ('z',)):
                        raise ValueError('Invalid automatic stage command axes.')
                    changed &= np.array([axis in axes for axis in 'xyz'])
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
                    if (changed[2] and cfg.z_direction*(target[2]-position[2])*1e6 > cfg.position_tolerance_um
                            and 'fine_origin_m' not in request):
                        raise ValueError('Z approach is permitted only during fine alignment.')
                speed = float(request['speed_um_s'])
                ceiling = (cfg.transfer_speed_um_s if request['kind'] == 'transfer'
                           else cfg.z_speed_um_s if changed[2]
                           else cfg.fine_xy_speed_um_s if 'fine_origin_m' in request
                           else cfg.xy_speed_um_s)
                if not np.isfinite(speed) or not 0 < speed <= ceiling:
                    raise ValueError('Automatic stage velocity exceeds calibrated limit.')
                if device.is_moving():
                    raise ValueError('Stage is already moving before an automatic command.')
                self.active = (target, now, cfg, request['kind'], changed.copy())
                self.settled_since = None
                self._reply('moving')
                if request['kind'] == 'transfer':
                    self.last_transfer_detail = None
                    transfer_event(self.v, 'motion_command', request=request,
                                   position_m=tuple(position), commanded_axes=tuple(
                                       axis for axis, change in zip('xyz', changed) if change),
                                   tolerance_um=cfg.position_tolerance_um, settle_s=cfg.settle_s)
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
            target, started, cfg, kind, commanded = self.active
            self._check_interlocks(kind, time.monotonic() if live_clock else now)
            device.validate_alignment_state()
            if now-started > cfg.move_timeout_s:
                status = self.v.alignment_move_status
                raise ValueError('Automatic stage movement timed out: '
                                 f"{status.get('wait_reason', 'waiting for controller')}; "
                                 f"commanded axes {status.get('commanded_axes', ())}, "
                                 f"XYZ error {status.get('error_um', ())} µm "
                                 f'(tolerance {cfg.position_tolerance_um:g} µm).')
            position = self._publish_position(device.get_position(), time.monotonic() if live_clock else now)
            moving = bool(device.is_moving())
            error_um = (np.asarray(position)-target)*1e6
            # A Z-only command never corrects X/Y sensor drift (and vice
            # versa). Waiting for those held axes to return within 20 nm
            # can block a completed move indefinitely. Match the same axes
            # that move_absolute was asked to move; all axes must be stopped
            # and every measured position still passes the bounds check.
            reached = not moving and np.all(np.abs(error_um[commanded]) <= cfg.position_tolerance_um)
            details = dict(commanded_axes=tuple(axis for axis, changed in zip('xyz', commanded) if changed),
                           error_um=tuple(float(error) for error in error_um),
                           wait_reason='stage moving' if moving else 'settling' if reached else 'waiting for target position')
            self._reply('moving', **details)
            if kind == 'transfer' and (details['wait_reason'] != self.last_transfer_detail
                                      or now-self.last_transfer_log_at >= 1.):
                transfer_event(self.v, 'motion_progress', id=self.last_id, position_m=position, **details)
                self.last_transfer_detail = details['wait_reason']
                self.last_transfer_log_at = now
            if not reached:
                self.settled_since = None
            elif self.settled_since is None:
                self.settled_since = now
            elif now-self.settled_since >= cfg.settle_s:
                self._reply('done', **details)
                if kind == 'transfer':
                    transfer_event(self.v, 'motion_settled', id=self.last_id, position_m=position,
                                   settled_s=now-self.settled_since, **details)
                self.active = None
        except Exception as exc:
            logging.getLogger('pyccapt.gui').error('Automatic stage command %s failed: %s', self.last_id, exc)
            if request and request.get('kind') == 'transfer':
                try:
                    transfer_event(self.v, 'motion_error', id=self.last_id, error=str(exc), request=request,
                                   status=self.v.alignment_move_status,
                                   position_snapshot=self.v.stage_position_snapshot)
                except Exception:
                    logging.getLogger('pyccapt.gui').exception('Could not record transfer motion error')
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
