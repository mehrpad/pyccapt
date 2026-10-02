"""Validated automatic-alignment settings; distances are explicit in each key."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class AlignmentConfig:
    start_voltage: float = 1500.0
    voltage_increment: float = 100.0
    coarse_dwell_s: float = 1.0
    coarse_max_dwell_s: float = 3.0
    search_min_events: int = 200
    jump_ratio: float = 1.5
    jump_sigma: float = 3.0
    fine_loss_ratio: float = 0.25
    max_voltage: float = 6000.0
    ramp_v_s: float = 100.0
    finish_fraction: float = 0.80
    stable_s: float = 3.0
    loss_s: float = 3.0
    dwell_s: float = 10.0
    timeout_s: float = 1800.0
    max_attempts: int = 1
    window_ions: int = 2000
    window_max_age_s: float = 10.0
    detector_radius_mm: float = 40.0
    centre_tolerance: float = 0.02
    area_target: float = 0.90
    xy_range_um: tuple = (50.0, 50.0)
    semi_xy_range_um: float = 15.0
    fine_xy_range_um: float = 5.0
    xy_boundary_margin_um: float = 0.2
    xy_step_um: float = 10.0
    fine_xy_step_um: float = 0.1
    fine_probe_step_um: float = 1.0
    xy_speed_um_s: float = 100.0
    fine_xy_speed_um_s: float = 16.0
    z_step_um: float = 0.05
    z_speed_um_s: float = 0.1
    z_direction: int = 0
    z_max_advance_um: float = 20.0
    approach_enabled: bool = True
    transfer_z_mm: float = 0.0
    transfer_speed_um_s: float = 300.0
    position_tolerance_um: float = 0.02
    settle_s: float = 0.5
    move_timeout_s: float = 120.0
    bounds_mm: tuple = ()
    xy_jacobian_mm_per_um: tuple = ()
    motion_calibrated: bool = False

    @classmethod
    def from_snapshot(cls, snapshot):
        """Read cross-process settings, including snapshots from older running GUIs.

        Only the retired absolute-rate thresholds are discarded. Unknown keys
        still fail construction, and explicit motion limits remain unchanged.
        """
        values = dict(snapshot)
        values.pop('entry_fraction', None)
        values.pop('loss_fraction', None)
        result = cls(**values)
        result.validate()
        return result

    @classmethod
    def from_mapping(cls, conf, start_voltage=None, voltage_increment=None):
        values = {}
        for name in cls.__dataclass_fields__:
            if 'alignment_' + name in conf:
                values[name] = conf['alignment_' + name]
        values['detector_radius_mm'] = float(conf.get('detector_diameter', 80)) / 2
        if start_voltage is not None:
            values['start_voltage'] = float(start_voltage)
        if voltage_increment is not None:
            values['voltage_increment'] = float(voltage_increment)
        for key in ('xy_range_um', 'bounds_mm', 'xy_jacobian_mm_per_um'):
            if key in values:
                values[key] = tuple(values[key])
        result = cls(**values)
        result.validate()
        return result

    def validate(self):
        if type(self.motion_calibrated) is not bool or type(self.approach_enabled) is not bool:
            raise ValueError('Alignment motion flags must be TOML booleans, not quoted strings.')
        for key, value in asdict(self).items():
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError(f'Alignment {key} must be finite.')
        positive = ('start_voltage', 'voltage_increment', 'max_voltage', 'ramp_v_s',
                    'coarse_dwell_s', 'coarse_max_dwell_s', 'jump_sigma',
                    'stable_s', 'loss_s', 'dwell_s', 'timeout_s', 'window_max_age_s',
                    'detector_radius_mm', 'xy_step_um', 'fine_xy_step_um', 'fine_probe_step_um',
                    'semi_xy_range_um', 'fine_xy_range_um', 'xy_boundary_margin_um',
                    'xy_speed_um_s', 'fine_xy_speed_um_s',
                    'z_step_um', 'z_speed_um_s', 'transfer_speed_um_s',
                    'position_tolerance_um', 'settle_s', 'move_timeout_s')
        if any(getattr(self, name) <= 0 for name in positive):
            raise ValueError('Alignment voltages, durations, steps and speeds must be positive.')
        if not 0 < self.finish_fraction <= 1:
            raise ValueError('Alignment completion rate fraction must be between zero and one.')
        if not 0 < self.area_target < 1 or not 0 < self.centre_tolerance < 1:
            raise ValueError('Alignment area and centring fractions must be between 0 and 1.')
        if self.start_voltage > self.max_voltage or self.dwell_s < self.stable_s:
            raise ValueError('Alignment start voltage or observation duration is invalid.')
        if self.z_max_advance_um < 0 or self.position_tolerance_um >= self.fine_xy_step_um:
            raise ValueError('Alignment position tolerance must be smaller than the fine XY step.')
        if self.fine_probe_step_um > self.fine_xy_range_um:
            raise ValueError('Fine probe step must fit inside the fine XY range.')
        if self.xy_boundary_margin_um < self.position_tolerance_um:
            raise ValueError('XY boundary margin must cover the positioning tolerance.')
        if self.approach_enabled and self.position_tolerance_um >= self.z_step_um:
            raise ValueError('Alignment position tolerance must be smaller than the approach step.')
        if int(self.window_ions) != self.window_ions or self.window_ions < 200:
            raise ValueError('Alignment window must contain at least 200 ions.')
        if int(self.max_attempts) != self.max_attempts or not 1 <= self.max_attempts <= 100:
            raise ValueError('Alignment attempts must be a positive integer (maximum 100).')
        if (not 1 < self.jump_ratio or not 0 < self.fine_loss_ratio < 1
                or self.coarse_max_dwell_s < self.coarse_dwell_s
                or int(self.search_min_events) != self.search_min_events
                or not 50 <= self.search_min_events <= self.window_ions):
            raise ValueError('Invalid relative-search ratio, dwell or minimum event count.')

    def validate_motion(self, positions):
        if not self.motion_calibrated:
            raise ValueError('Automatic alignment is not calibrated. Set measured XYZ bounds, XY search ranges, '
                             'approach direction and transfer Z in config.toml; '
                             'then set alignment_motion_calibrated = true. See control/AUTOMATIC_ALIGNMENT.md.')
        bounds = np.asarray(self.bounds_mm, dtype=float)
        jac = np.asarray(self.xy_jacobian_mm_per_um, dtype=float)
        ranges = np.asarray(self.xy_range_um, dtype=float).copy()
        if bounds.shape != (3, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError('Configure alignment_bounds_mm as three calibrated [minimum, maximum] pairs.')
        if jac.size and (jac.shape != (2, 2) or not np.isfinite(jac).all() or np.linalg.cond(jac) > 100):
            raise ValueError('Alignment XY Jacobian must be an invertible measured 2 by 2 matrix or empty for detector feedback.')
        if ranges.shape != (2,) or not np.isfinite(ranges).all() or np.any(ranges <= 0):
            raise ValueError('Configure positive calibrated alignment_xy_range_um values for X and Y.')
        if sum(1 for _ in coarse_positions((0., 0., 0.), self)) > 20000:
            raise ValueError('Alignment search exceeds 20000 positions.')
        if self.coarse_scan_duration_s() >= self.timeout_s:
            raise ValueError('A complete coarse alignment sweep and voltage retry exceed the alignment timeout. '
                             'Increase timeout, or reduce XY range / coarse dwell.')
        if self.z_direction not in (-1, 1):
            raise ValueError('Set alignment_z_direction to +1 or -1, toward the electrode.')
        if self.approach_enabled and self.z_max_advance_um <= 0:
            raise ValueError('Automatic approach needs a positive calibrated maximum advance.')
        self.check_position((bounds[0, 0] * 1e-3, bounds[1, 0] * 1e-3, self.transfer_z_mm * 1e-3))
        for position in positions:
            self.check_position(position)
            p = np.asarray(position, dtype=float)
            if self.z_direction * (self.transfer_z_mm * 1e-3 - p[2]) > 0:
                raise ValueError('Transfer Z must retract away from every saved sample position.')
            for sign in (-1, 1):
                self.check_position((p[0] + sign*ranges[0]*1e-6, p[1] + sign*ranges[1]*1e-6, p[2]))
            if self.approach_enabled:
                self.check_position((p[0], p[1], p[2] + self.z_direction*self.z_max_advance_um*1e-6))

    def check_position(self, position):
        point = np.asarray(position, dtype=float)
        bounds = np.asarray(self.bounds_mm, dtype=float) * 1e-3
        if point.shape != (3,) or not np.isfinite(point).all() or bounds.shape != (3, 2):
            raise ValueError('Invalid alignment position or movement bounds.')
        if np.any(point < bounds[:, 0]) or np.any(point > bounds[:, 1]):
            raise ValueError('Alignment movement would exceed calibrated stage limits.')

    def snapshot(self):
        return asdict(self)

    def coarse_scan_duration_s(self):
        """Nominal full sweep, origin return and one DC increment, including polling allowance."""
        previous = np.zeros(3)
        duration = self.voltage_increment/self.ramp_v_s
        for point in coarse_positions((0., 0., 0.), self):
            point = np.asarray(point)
            duration += (self.coarse_max_dwell_s + self.settle_s + 0.5
                         + np.max(np.abs(point-previous))*1e6/self.xy_speed_um_s)
            previous = point
        return float(duration + self.settle_s + 0.5
                     + np.max(np.abs(previous))*1e6/self.xy_speed_um_s)

    def check_fine_position(self, target, centre):
        target = np.asarray(target, dtype=float)
        centre = np.asarray(centre, dtype=float)
        if (target.shape != (3,) or centre.shape != (3,) or not np.isfinite(target).all()
                or not np.isfinite(centre).all()):
            raise ValueError('Invalid fine alignment position or centre.')
        if np.any(np.abs(target[:2]-centre[:2])*1e6 > self.fine_xy_range_um+1e-8):
            raise ValueError('Fine alignment exceeds its configured XY range.')

    def search_xy_limits(self, origin):
        """Inset commanded XY targets; measured guards retain the full envelope."""
        ranges = np.asarray(self.xy_range_um, dtype=float).copy()
        ranges -= np.minimum(self.xy_boundary_margin_um, ranges*.1)
        return (np.asarray(origin[:2])-ranges*1e-6,
                np.asarray(origin[:2])+ranges*1e-6)

    def check_search_xy(self, target, origin):
        lower, upper = self.search_xy_limits(origin)
        point = np.asarray(target[:2])
        if np.any(point < lower-1e-14) or np.any(point > upper+1e-14):
            raise ValueError('Alignment target exceeds the inset XY search envelope.')


def coarse_positions(origin, cfg):
    """Saved position plus 16 broad probes derived from the calibrated ranges."""
    from pyccapt.control.apt.alignment_search import range_positions
    lower, upper = cfg.search_xy_limits(origin)
    ranges = (upper-lower)*.5e6
    yield from range_positions(origin, ranges, lower, upper)
