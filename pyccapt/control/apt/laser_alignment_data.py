"""Small, epoch-tagged detector windows separate from visualization consumers."""
import time
import numpy as np


class LaserAlignmentPublisher:
    def __init__(self, variables):
        self.v = variables
        self.epoch = None
        self.times = []
        self.count = 0
        self.last_publish = 0.0

    def append(self, tof_ns, now=None):
        now = time.monotonic() if now is None else now
        epoch = self.v.laser_alignment_epoch
        if not epoch:
            return
        if epoch != self.epoch:
            self.epoch = epoch
            self.times = []
            self.count = 0
            self.started = now
            self.last_publish = now
            # Discard the first packet: it can contain pre-settle events.
            return
        values = np.asarray(tof_ns, dtype=float).ravel()
        values = values[np.isfinite(values)]
        self.count += len(values)
        self.times.extend(values[-20000:].tolist())
        self.times = self.times[-20000:]
        if now-self.last_publish >= 0.1:
            self.v.laser_alignment_observation = {
                'epoch': epoch, 'time': now, 'start': self.started,
                'count': self.count, 'tof_ns': self.times,
            }
            self.last_publish = now


def peak_quality(tof_ns, cfg):
    """Robust TOF width and late-tail fraction in a user-calibrated isolated peak.

    This is a relative focus diagnostic at fixed voltage, not calibrated MRP.
    Empty/insufficient windows do not manufacture a quality measurement.
    """
    if not cfg.quality_peak_ns:
        return {}
    data = np.asarray(tof_ns, dtype=float)
    lo, hi = cfg.quality_peak_ns
    data = data[np.isfinite(data) & (data >= lo) & (data <= hi)]
    if len(data) < cfg.quality_min_events:
        return {}
    q10, q50, q90 = np.quantile(data, [.1, .5, .9])
    width = float(q90-q10)
    return {'width_ns': width, 'tail_fraction': float(np.mean(data > q50+max(q50-q10, 1e-12))),
            'peak_events': len(data)}
