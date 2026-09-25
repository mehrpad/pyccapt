"""Execute acknowledged laser moves on the laser GUI's existing stage handle."""
import time
import numpy as np
from pyccapt.control.apt.laser_alignment_config import LaserAlignmentConfig


class LaserAlignmentStageService:
    def __init__(self, variables, conf, device_getter):
        self.v, self.conf, self.device_getter = variables, conf, device_getter
        self.active = None
        self.last_id = None
        self.settled = None

    def cancel(self, reason='Laser motion cancelled'):
        if self.active is not None:
            try:
                device = self.device_getter()
                if device is not None:
                    device.stop()
            finally:
                self.v.laser_alignment_move_status = {'id': self.last_id, 'state': 'error', 'error': reason}
                self.active = None
        self.settled = None

    def tick(self, now=None):
        now = time.monotonic() if now is None else now
        self.v.laser_alignment_heartbeat = now
        permitted = (self.v.start_flag and not self.v.stop_flag and not self.v.laser_alignment_cancel
                     and not self.v.electrode_out and self.v.physical_estop_ok
                     and self.v.experiment_state == 'running')
        if not permitted:
            self.cancel()
            return
        request = self.v.laser_alignment_move_request
        device = self.device_getter()
        try:
            if (self.active is not None or request and request.get('id') != self.last_id) and not (
                    -0.1 <= now-self.v.experiment_heartbeat_monotonic <= 3):
                raise ValueError('Experiment heartbeat lost; stopping laser movement.')
            if request and request.get('id') != self.last_id:
                self.last_id = request['id']
                if self.active is not None or device is None:
                    raise ValueError('Laser stage unavailable or already moving.')
                run = self.v.laser_alignment_run
                if request.get('session') != run.get('id') or not 0 <= now-request['issued'] <= 3:
                    raise ValueError('Expired laser stage request.')
                if self.v.pulse_mode != 'Laser' or self.v.flag_tdc_failure:
                    raise ValueError('Laser alignment acquisition interlock lost.')
                if self.v.automatic_alignment_enabled and self.v.alignment_status.get('phase') != 'aligned':
                    raise ValueError('Sample stage alignment must finish first.')
                # Calibrated limits always come from config, not an editable GUI payload.
                cfg = LaserAlignmentConfig.from_config(self.conf, run['settings'])
                origin = tuple(run['origin_m'])
                cfg.validate(origin)
                target = np.asarray(request['target_m'], dtype=float)
                cfg.check_position(target, origin)
                device.validate_alignment_state()
                current = device.get_position()
                position = np.array([current[k] for k in 'xyz'])
                cfg.check_position(position, origin)
                changed = abs(target-position) > cfg.tolerance_um*1e-6
                if changed[2] and changed[:2].any():
                    raise ValueError('Laser XY and Z moves must be separate.')
                speed = float(request['speed_um_s'])
                ceiling = cfg.z_speed_um_s if changed[2] else cfg.xy_speed_um_s
                if not np.isfinite(speed) or not 0 < speed <= ceiling:
                    raise ValueError('Laser speed exceeds calibrated limit.')
                if device.is_moving():
                    raise ValueError('Laser stage is already moving.')
                self.active = (target, now, cfg, origin)
                self.settled = None
                self.v.laser_alignment_move_status = {'id': self.last_id, 'state': 'moving'}
                if changed.any():
                    device.move_absolute(**{k+'_m': float(x) if c else None for k, x, c in zip('xyz', target, changed)},
                                         velocity_m_s=speed*1e-6, wait=False)
            if self.active is None:
                return
            target, started, cfg, origin = self.active
            if self.v.flag_tdc_failure or self.v.pulse_mode != 'Laser':
                raise ValueError('Laser acquisition interlock lost during movement.')
            if now-started > cfg.move_timeout_s:
                raise ValueError('Laser stage movement timed out.')
            device.validate_alignment_state()
            pos = device.get_position()
            position = tuple(float(pos[k]) for k in 'xyz')
            cfg.check_position(position, origin)
            self.v.laser_stage_snapshot = (*position, now)
            reached = not device.is_moving() and np.all(abs(np.array(position)-target) <= cfg.tolerance_um*1e-6)
            if not reached:
                self.settled = None
            elif self.settled is None:
                self.settled = now
            elif now-self.settled >= cfg.settle_s:
                self.v.laser_alignment_move_status = {'id': self.last_id, 'state': 'done'}
                self.active = None
        except Exception as exc:
            if device is not None:
                try:
                    device.stop()
                except Exception:
                    pass  # Still publish failure so the experiment shuts its outputs down.
            self.v.laser_alignment_move_status = {'id': request.get('id', self.last_id), 'state': 'error',
                                                  'error': str(exc), 'fault': True}
            self.v.laser_alignment_cancel = True
            self.active = None
