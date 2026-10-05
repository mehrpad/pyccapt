"""Experiment-side adapter: the experiment remains the sole DC supply owner."""
import time
from pyccapt.control.apt.laser_alignment import LaserAlignment
from pyccapt.control.apt.laser_alignment_config import LaserAlignmentConfig


def validate_laser_alignment(variables, conf):
    if variables.pulse_mode != 'Laser' or variables.flat_test_active:
        raise ValueError('Laser alignment requires Laser pulse mode and a normal experiment.')
    if conf.get('tdc_model') not in ('Surface_Concept', 'Surface_Consept', 'RoentDek') or variables.counter_source != 'TDC':
        raise ValueError('Laser alignment requires a Surface Concept or RoentDek TDC.')
    cfg = LaserAlignmentConfig.from_config(conf, variables.laser_alignment_settings)
    snapshot = variables.laser_stage_snapshot
    if len(snapshot) != 4 or not 0 <= time.monotonic()-snapshot[3] <= 3:
        raise ValueError('Laser stage position is unavailable or stale.')
    cfg.validate(snapshot[:3])
    if float(variables.vdc_min) > min(cfg.max_voltage, float(variables.vdc_max), float(conf['max_vdc'])):
        raise ValueError('Initial DC voltage exceeds the laser alignment voltage ceiling.')


class LaserAlignmentRuntime:
    def __init__(self, owner):
        self.owner = owner
        self.v = owner.variables
        self.engine = None
        self.last_command = self.v.laser_alignment_command.get('id')
        self.auto_pending = bool(self.v.laser_alignment_enabled)
        self.v.laser_alignment_status = {}
        self.v.laser_alignment_run = {}
        self.v.laser_alignment_move_request = {}
        self.v.laser_alignment_epoch = ''
        self.v.laser_alignment_cancel = True
        if self.auto_pending:
            validate_laser_alignment(self.v, owner.conf)

    def tick(self):
        before = self.engine.owns_voltage if self.engine else False
        command = self.v.laser_alignment_command
        requested = command.get('id') != self.last_command and bool(command)
        if requested and command.get('mode') == 'stop':
            self.last_command = command['id']
            self.auto_pending = False
            self.stop('Laser alignment stopped by operator.')
            if before:
                self.owner._switch_control_algorithm(self.v.control_algorithm)
            return True
        sample_ready = not self.v.automatic_alignment_enabled or self.v.alignment_status.get('phase') == 'aligned'
        if self.auto_pending and self.v.laser_alignment_cancel and self.v.laser_alignment_status.get('phase') == 'cancelled':
            self.auto_pending = False
        auto = self.auto_pending and sample_ready
        if requested or auto:
            self.last_command = command.get('id')
            if auto:
                self.auto_pending = False
            if self.engine is None or self.engine.finished:
                try:
                    if requested and not 0 <= time.monotonic()-command.get('issued', 0) <= 3:
                        raise ValueError('Laser alignment command expired.')
                    validate_laser_alignment(self.v, self.owner.conf)
                    self.engine = LaserAlignment(self.v, self.owner.conf, command['mode'] if requested else 'coarse')
                except Exception as exc:
                    self.v.laser_alignment_status = {'phase': 'failed', 'active': False, 'reason': str(exc)}
                    self.owner.log_apt.error('Laser alignment could not start: %s', exc)
                    if auto:
                        self.owner._run_failure = str(exc)
                        return False
        if self.engine and not self.engine.finished:
            self.engine.tick()
            if self.engine.phase == 'failed':
                self.owner._run_failure = self.engine.reason
                self.owner.log_apt.error('Laser alignment failed: %s', self.engine.reason)
                return False
        after = self.engine.owns_voltage if self.engine else False
        if before and not after:
            # Reset the integrator before ordinary rate control resumes.
            self.owner._switch_control_algorithm(self.v.control_algorithm)
        return True

    def voltage_step(self, voltage, dt):
        return self.engine.voltage_step(voltage, dt) if self.engine else None

    def stop(self, reason='Experiment ended.'):
        self.v.laser_alignment_cancel = True
        self.v.laser_alignment_epoch = ''
        if self.engine and not self.engine.finished:
            self.engine.finish(reason)
