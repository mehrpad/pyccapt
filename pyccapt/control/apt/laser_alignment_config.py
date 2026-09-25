"""Calibrated laser optics motion limits; distances are stage travel, not beam size."""
from dataclasses import asdict, dataclass
import math


@dataclass
class LaserAlignmentConfig:
    calibrated: bool = False
    bounds_mm: tuple = ()
    travel_um: tuple = ()
    coarse_range_um: float = 5.0
    coarse_step_um: float = 2.5
    fine_range_um: float = 1.0
    fine_step_um: float = 0.5
    focus_range_um: float = 1.0
    focus_step_um: float = 0.5
    xy_speed_um_s: float = 1.0
    z_speed_um_s: float = 0.25
    tolerance_um: float = 0.02
    settle_s: float = 1.0
    dwell_s: float = 1.5
    move_timeout_s: float = 60.0
    voltage_increment: float = 50.0
    max_voltage: float = 6000.0
    ramp_v_s: float = 20.0
    min_rate_percent: float = 0.01
    background_rate_percent: float = 0.0
    min_events: int = 30
    improvement_fraction: float = 0.05
    max_recoveries: int = 2
    max_duration_s: float = 1800.0
    tracking_interval_s: float = 60.0
    tracking_step_um: float = 0.1
    quality_peak_ns: tuple = ()
    quality_min_events: int = 200
    quality_max_degradation: float = 0.15
    max_rate_multiple: float = 2.0

    @classmethod
    def from_config(cls, conf, overrides=None):
        values = {key: conf['laser_alignment_'+key] for key in cls.__dataclass_fields__
                  if 'laser_alignment_'+key in conf}
        editable = ('coarse_range_um', 'coarse_step_um', 'fine_range_um', 'fine_step_um',
                    'focus_range_um', 'focus_step_um', 'voltage_increment')
        values.update({k: v for k, v in (overrides or {}).items() if k in editable})
        return cls(**values)

    def snapshot(self):
        return asdict(self)

    def validate(self, origin):
        if self.calibrated is not True:
            raise ValueError('Calibrate laser motion limits in config.toml before automatic laser alignment.')
        for key, value in self.snapshot().items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(value) or value < 0:
                    raise ValueError('Invalid laser alignment setting: '+key)
        positive = ('coarse_range_um', 'coarse_step_um', 'fine_range_um', 'fine_step_um',
                    'focus_range_um', 'focus_step_um', 'xy_speed_um_s', 'z_speed_um_s',
                    'tolerance_um', 'dwell_s', 'move_timeout_s', 'voltage_increment',
                    'max_voltage', 'ramp_v_s', 'min_events', 'min_rate_percent',
                    'max_duration_s', 'tracking_interval_s', 'tracking_step_um', 'max_rate_multiple')
        if any(getattr(self, k) <= 0 for k in positive):
            raise ValueError('Laser alignment ranges, steps, timing and limits must be positive.')
        if len(self.bounds_mm) != 6 or len(self.travel_um) != 3:
            raise ValueError('Set laser_alignment_bounds_mm [xmin,xmax,ymin,ymax,zmin,zmax] and travel_um [x,y,z].')
        if not all(math.isfinite(x) for x in (*self.bounds_mm, *self.travel_um)):
            raise ValueError('Laser motion limits must be finite.')
        if any(self.bounds_mm[i] >= self.bounds_mm[i+1] for i in (0, 2, 4)) or min(self.travel_um) <= 0:
            raise ValueError('Invalid calibrated laser travel limits.')
        if max(self.coarse_range_um, self.fine_range_um, self.tracking_step_um) > min(self.travel_um[:2]):
            raise ValueError('XY scan range exceeds calibrated laser travel.')
        if self.focus_range_um > self.travel_um[2]:
            raise ValueError('Focus scan range exceeds calibrated laser travel.')
        for prefix in ('coarse', 'fine', 'focus'):
            ratio = 2*getattr(self, prefix+'_range_um')/getattr(self, prefix+'_step_um')
            if not 1 <= ratio <= 20:
                raise ValueError('Each scan requires 2–21 positions per axis; adjust range or step.')
        if self.settle_s < 0.2 or self.dwell_s < 0.2:
            raise ValueError('Laser settling and measurement windows must be at least 0.2 seconds.')
        if self.quality_peak_ns and (len(self.quality_peak_ns) != 2 or
                not all(math.isfinite(x) for x in self.quality_peak_ns) or
                not 0 < self.quality_peak_ns[0] < self.quality_peak_ns[1]):
            raise ValueError('quality_peak_ns must be empty or a positive TOF peak window [low,high].')
        self.check_position(origin, origin)

    def check_position(self, target, origin):
        if len(target) != 3 or len(origin) != 3:
            raise ValueError('Laser stage XYZ position is missing.')
        for i, value in enumerate(target):
            if not math.isfinite(value) or not math.isfinite(origin[i]):
                raise ValueError('Laser position is not finite.')
            if not self.bounds_mm[2*i]*1e-3 <= value <= self.bounds_mm[2*i+1]*1e-3:
                raise ValueError('Laser target exceeds calibrated absolute travel.')
            if abs(value-origin[i])*1e6 > self.travel_um[i]+1e-7:
                raise ValueError('Laser target exceeds travel allowed around the starting position.')
