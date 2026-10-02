"""Independent, bounded alignment observations from paired detector events.

Raw event coordinates enter in cm (the acquisition convention); analysis uses mm.
The plot rings retain their single visualization reader.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np


class AlignmentEventPublisher:
    def __init__(self, variables, conf=None):
        conf = conf or {}
        self.variables = variables
        self.enabled = bool(getattr(variables, 'automatic_alignment_enabled', False))
        settings = getattr(variables, 'alignment_settings', {})
        self.size = int(settings.get('window_ions', conf.get('alignment_window_ions', 2000)))
        self.points = deque(maxlen=self.size)
        self.epoch = None
        self.guard_until = 0.0
        self.last_publish = 0.0
        self.sequence = 0
        self.guard_s = float(conf.get('alignment_acquisition_guard_s', 0.5))
        self.max_age_s = float(settings.get('window_max_age_s', conf.get('alignment_window_max_age_s', 10.)))

    def append(self, x_cm, y_cm, now=None):
        if not self.enabled:
            return
        now = time.monotonic() if now is None else now
        epoch = self.variables.alignment_window_epoch
        if epoch != self.epoch:
            self.epoch = epoch
            self.points.clear()
            self.guard_until = now + self.guard_s
            # Discard the queued batch straddling a movement/voltage transition.
            return
        if now < self.guard_until:
            return
        points = np.column_stack((x_cm, y_cm)).astype(float) * 10.0
        points = points[np.isfinite(points).all(axis=1)]
        self.sequence += len(points)
        self.points.extend((float(x), float(y), now) for x, y in points[-self.size:])
        while self.points and now-self.points[0][2] > self.max_age_s:
            self.points.popleft()
        if now - self.last_publish < 0.2 or not self.points:
            return
        self.last_publish = now
        data = np.asarray(self.points)
        self.variables.alignment_events = {
            'epoch': epoch, 'sequence': self.sequence, 'time': now,
            'first_time': float(data[0, 2]), 'points_mm': data[:, :2].tolist(),
        }


@dataclass(frozen=True)
class Footprint:
    valid: bool
    reason: str
    centre_mm: tuple = (0., 0.)
    radius_mm: float = 0.
    area_fraction: float = 0.
    signal_fraction: float = 0.
    residual_fraction: float = 1.
    angular_coverage: float = 0.
    radius_uncertainty_mm: float = 0.
    model: str = 'circle'

    def snapshot(self):
        return asdict(self)


def estimate_dense_region(points_mm, detector_radius_mm, minimum_events=200):
    """Locate a coherent high-density region for XY; never authorize Z from it."""
    from scipy import ndimage
    points = np.asarray(points_mm, dtype=float)
    invalid = Footprint(False, 'No coherent dense detector region', model='density')
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum_events:
        return invalid
    radius = float(detector_radius_mm)
    points = points[np.isfinite(points).all(axis=1)]
    points = points[(points**2).sum(axis=1) <= radius**2]
    if len(points) < .8*minimum_events:
        return invalid
    bins = 32
    pixel = 2*radius/bins
    axis = np.linspace(-radius+pixel/2, radius-pixel/2, bins)
    gx, gy = np.meshgrid(axis, axis, indexing='ij')
    detector = gx*gx+gy*gy < radius*radius
    counts = np.histogram2d(points[:, 0], points[:, 1], bins=bins, range=[[-radius, radius]]*2)[0]
    nonzero = counts[counts > 0]
    cap = max(4., float(np.quantile(nonzero, .95)))
    # Cap to the diffuse occupancy as well: a single saturated pixel must
    # not manufacture a smoothed sample-shaped region.
    cap = min(cap, max(4., len(points)/max(1, detector.sum())*8))
    image = ndimage.gaussian_filter(np.minimum(counts, cap), 1.)
    background = float(np.quantile(image[detector], .05))
    noise = np.sqrt(max(background, .02)/(4*np.pi))
    mask = detector & (image > background+max(.25, 3*noise))
    labels, count = ndimage.label(mask)
    if not count:
        return invalid
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    region = labels == int(np.argmax(sizes))
    if region.sum() < 8 or np.count_nonzero(counts*region) < 8:
        return invalid
    indices = np.clip(((points+radius)/pixel).astype(int), 0, bins-1)
    inside = region[indices[:, 0], indices[:, 1]]
    n_signal = int(inside.sum())
    outside_area = max(1, int(detector.sum()-region.sum()))
    expected = (len(points)-n_signal)*region.sum()/outside_area
    if n_signal < max(50, .35*len(points)) or n_signal-expected < 3*np.sqrt(max(1., expected)):
        return invalid
    # Capped intensity reduces the influence of a bright pixel inside the
    # genuine region without discarding that region's useful centroid.
    weights = np.minimum(counts, cap)*region
    centre = (float((weights*gx).sum()/weights.sum()), float((weights*gy).sum()/weights.sum()))
    extent = float(np.quantile(np.linalg.norm(points[inside]-centre, axis=1), .95))
    if np.linalg.norm(centre)+extent >= radius*.995:
        return invalid
    return Footprint(True, 'Dense detector region accepted for XY', centre, extent,
                     float(region.sum()/detector.sum()), float((n_signal-expected)/len(points)),
                     radius_uncertainty_mm=pixel, model='density')


def estimate_footprint(points_mm, detector_radius_mm, minimum_events=2000):
    """Fit a robust circle to a background-subtracted region's boundary.

    Bright single pixels are clipped before smoothing. Dark counts, partial
    arcs, small hot spots and detector-edge clipping must not authorize motion.
    Thresholds are intentionally conservative and require instrument replay
    validation before the motion calibration flag is enabled.
    """
    from scipy import ndimage
    points = np.asarray(points_mm, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum_events:
        return Footprint(False, 'Waiting for fresh detector events')
    radius = float(detector_radius_mm)
    points = points[np.isfinite(points).all(axis=1)]
    points = points[(points**2).sum(axis=1) <= radius**2]
    if len(points) < 0.8 * minimum_events:
        return Footprint(False, 'Too many events outside the detector')
    bins = 64
    pixel = 2*radius/bins
    axis = np.linspace(-radius + pixel/2, radius - pixel/2, bins)
    gx, gy = np.meshgrid(axis, axis, indexing='ij')
    detector = gx**2 + gy**2 < radius**2
    counts = np.histogram2d(points[:, 0], points[:, 1], bins=bins,
                            range=[[-radius, radius]]*2)[0]
    nonzero = counts[counts > 0]
    cap = max(4., float(np.quantile(nonzero, 0.95)))
    image = ndimage.gaussian_filter(np.minimum(counts, cap), 1.2)
    # An outer/background quantile avoids treating a broad sample as noise.
    background = float(np.quantile(image[detector], 0.05))
    # Shot-noise variance after Gaussian smoothing; using a middle quantile
    # as 'noise' would erase a genuine footprint covering most of the detector.
    noise = np.sqrt(max(background, 0.02)/(4*np.pi*1.2**2))
    threshold = background + max(0.20, 3.0*noise)
    mask = detector & (image > threshold)
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(mask)) & detector
    labels, n = ndimage.label(mask)
    if not n:
        return Footprint(False, 'No coherent footprint above background')
    areas = np.bincount(labels.ravel()); areas[0] = 0
    region = labels == int(np.argmax(areas))
    if region.sum() < 0.015*detector.sum():
        return Footprint(False, 'Footprint too small; possible hot spot')
    # Locate the half-contrast edge, rather than the low detection threshold:
    # Gaussian tails otherwise inflate the apparent radius by several pixels.
    foreground = float(np.quantile(image[region], 0.65))
    refined = detector & (image > max(background + 0.5*(foreground-background), threshold))
    labels, _ = ndimage.label(ndimage.binary_fill_holes(ndimage.binary_closing(refined)))
    overlap = np.bincount(labels[region]); overlap[0] = 0
    if not overlap.any():
        return Footprint(False, 'No stable footprint edge')
    region = labels == int(np.argmax(overlap))
    edge = region & ~ndimage.binary_erosion(region)
    boundary = np.column_stack((gx[edge], gy[edge]))
    if len(boundary) < 16:
        return Footprint(False, 'Insufficient footprint boundary')
    # RANSAC candidates fit boundary points, never the raw interior ion cloud.
    rng = np.random.default_rng(0)
    best = None
    best_count = 0
    for _ in range(100):
        p = boundary[rng.choice(len(boundary), 3, replace=False)]
        matrix = 2*(p[1:] - p[0])
        if abs(np.linalg.det(matrix)) < 1e-8:
            continue
        centre = np.linalg.solve(matrix, (p[1:]**2).sum(axis=1) - (p[0]**2).sum())
        r = np.linalg.norm(p[0] - centre)
        if not pixel*2 < r < radius:
            continue
        inliers = np.abs(np.linalg.norm(boundary-centre, axis=1)-r) < 1.5*pixel
        if inliers.sum() > best_count:
            best_count = int(inliers.sum()); best = inliers
    if best is None or best_count < 0.7*len(boundary):
        return Footprint(False, 'Boundary is not consistently circular')
    p = boundary[best]
    solution = np.linalg.lstsq(np.column_stack((2*p, np.ones(len(p)))), (p**2).sum(axis=1), rcond=None)[0]
    centre = solution[:2]
    r = float(np.sqrt(max(0., solution[2] + (centre**2).sum())))
    if r <= 0:
        return Footprint(False, 'Invalid fitted radius')
    residual = float(np.sqrt(np.mean((np.linalg.norm(p-centre, axis=1)-r)**2))/r)
    r += pixel/2  # Boundary pixels represent their centres, not outer edges.
    angles = np.arctan2(p[:, 1]-centre[1], p[:, 0]-centre[0])
    coverage = np.count_nonzero(np.histogram(angles, bins=16, range=(-np.pi, np.pi))[0])/16
    in_circle = np.linalg.norm(points-centre, axis=1) <= r
    outside_area = max(1., np.pi*(radius**2-r**2))
    background_density = np.count_nonzero(~in_circle)/outside_area
    signal = max(0., (np.count_nonzero(in_circle) - background_density*np.pi*r*r)/len(points))
    valid = bool(residual <= 0.12 and coverage >= 0.75 and signal >= 0.25
                 and np.linalg.norm(centre)+r < radius*0.995)
    return Footprint(valid, 'Footprint accepted' if valid else 'Noisy, incomplete or clipped footprint',
                     tuple(float(x) for x in centre), r, (r/radius)**2, float(signal), residual, float(coverage),
                     max(pixel, residual*r))
