"""Deterministic, reviewable crop heuristics; never delete or accept events."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


VERSION = "pyccapt-crop-heuristic-1"


def suggest_temporal_crop(frame, *, max_windows=512, minimum_events=1000):
    """Find the longest positive-voltage run without large drops/jumps/roughness.

    Smooth curvature is allowed. This tests voltage stability, not chemistry,
    detection efficiency, or experimental completeness. Stop is exclusive.
    """
    n = len(frame)
    result = dict(algorithm=VERSION, kind="temporal", supported=False, event_start=0,
                  event_stop=n, reason="Insufficient voltage evidence; review the full run manually.")
    if (n < minimum_events or "high_voltage (V)" not in frame or
            "high_voltage (V)" in frame.attrs.get("pyccapt_unavailable_columns", [])):
        return result
    voltage = frame["high_voltage (V)"].to_numpy(dtype=float)
    edges = np.linspace(0, n, min(max_windows, max(12, n//100))+1, dtype=int)
    medians, noise, fraction, jumps = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        values = voltage[lo:hi]
        valid = np.isfinite(values) & (values > 0)
        fraction.append(float(valid.mean()))
        if valid.sum() < 3:
            medians.append(np.nan); noise.append(np.inf); jumps.append(np.inf)
            continue
        positions = np.flatnonzero(valid)
        values = values[valid]
        width = max(1, len(values)//10)
        trend = np.interp(positions, [positions[:width].mean(), positions[-width:].mean()],
                          [np.median(values[:width]), np.median(values[-width:])])
        residual = values-trend
        medians.append(float(np.median(values)))
        noise.append(float(1.4826*np.median(np.abs(residual-np.median(residual)))))
        delta = np.diff(values)
        jumps.append(float(np.max(np.abs(delta))) if delta.size else 0.)
    medians, noise, fraction, jumps = map(np.asarray, (medians, noise, fraction, jumps))
    good_values = medians[np.isfinite(medians)]
    if good_values.size < 8:
        return result
    level = float(np.median(good_values))
    step = np.diff(medians)
    positive = step[np.isfinite(step) & (step > 0)]
    jump_limit = max(.02*level, 10*float(np.median(positive)) if positive.size else 0., 2.)
    rough_limit = max(.003*level, 6*float(np.median(noise[np.isfinite(noise)])), 1.)
    bad = (fraction < .99) | (noise > rough_limit) | (jumps > jump_limit)
    transitions = np.flatnonzero((step < -max(.005*level, 2.)) | (step > jump_limit))
    bad[transitions] = True
    bad[transitions+1] = True
    if bad.any():
        bad = ndimage.binary_dilation(bad, iterations=1)
    runs, count = ndimage.label(~bad)
    choices = []
    for label in range(1, count+1):
        indices = np.flatnonzero(runs == label)
        lo, hi = int(edges[indices[0]]), int(edges[indices[-1]+1])
        choices.append((hi-lo, lo, hi))
    choices.sort(reverse=True)
    if not choices or choices[0][0] < max(minimum_events//2, .2*n):
        result["reason"] = "No sufficiently long stable voltage interval; choose the region manually."
        return result
    _, lo, hi = choices[0]
    ambiguous = len(choices) > 1 and choices[1][0] >= .9*choices[0][0]
    result.update(supported=True, event_start=lo, event_stop=hi,
        reason="Several similarly long stable intervals; review this longest candidate." if ambiguous else
               "Longest interval passing positive-voltage, jump, drop and local-roughness checks.",
        diagnostics=dict(windows=len(medians), bad_windows=int(bad.sum()), jump_limit_V=jump_limit,
                         roughness_limit_V=rough_limit, ambiguous=ambiguous))
    return result


def suggest_detector_circle(frame, *, event_start=0, event_stop=None, bins=128, minimum_events=1000):
    """Largest interior circle in the main supported density component.

    Log-density Otsu separation is used for a bright core plus weak halo. For
    nearly uniform intensity, use the supported detector footprint instead.
    Fill enclosed density holes, retaining edge-connected mouse bites.
    All finite hits contribute to the histogram in bounded chunks.
    """
    stop = len(frame) if event_stop is None else int(event_stop)
    result = dict(algorithm=VERSION, kind="detector", supported=False, circle_cm=None,
                  reason="Insufficient detector evidence; adjust the initial circle manually.")
    names = ("x_det (cm)", "y_det (cm)")
    unavailable = frame.attrs.get("pyccapt_unavailable_columns", [])
    if (not 0 <= event_start < stop <= len(frame) or stop-event_start < minimum_events or
            any(name not in frame or name in unavailable for name in names)):
        return result
    sample_indices = np.linspace(event_start, stop-1, min(200000, stop-event_start), dtype=int)
    sample = frame.iloc[sample_indices].loc[:, list(names)].to_numpy(dtype=float)
    sample = sample[np.isfinite(sample).all(axis=1)]
    if len(sample) < minimum_events:
        return result
    limits = np.quantile(sample, [.002, .998], axis=0)
    if np.any(limits[1] <= limits[0]):
        return result
    edges = [np.linspace(limits[0, i], limits[1, i], bins+1) for i in range(2)]
    counts = np.zeros((bins, bins))
    finite_count = 0
    for start in range(event_start, stop, 250000):
        xy = frame.iloc[start:min(start+250000, stop)].loc[:, list(names)].to_numpy(dtype=float)
        xy = xy[np.isfinite(xy).all(axis=1)]
        finite_count += len(xy)
        counts += np.histogram2d(xy[:, 0], xy[:, 1], bins=edges)[0]
    smooth = ndimage.gaussian_filter(counts, 1.2)
    occupied = smooth[counts > 0]
    if occupied.size < 30 or occupied.max() < 5:
        return result
    contrast = float(np.quantile(occupied, .9)/max(np.quantile(occupied, .1), 1e-9))
    if contrast >= 3:
        values = np.log1p(occupied)
        histogram, boundaries = np.histogram(values, bins=128)
        centers = (boundaries[:-1]+boundaries[1:])/2
        weights = np.cumsum(histogram)
        means = np.cumsum(histogram*centers)
        denominator = weights*(weights[-1]-weights)
        score = np.divide((means[-1]*weights-means*weights[-1])**2, denominator,
                          out=np.zeros(128), where=denominator > 0)
        threshold = float(np.expm1(centers[np.argmax(score)]))
        method = "log-density separation"
    else:
        threshold = max(1., .08*float(np.quantile(occupied, .9)))
        method = "supported footprint"
    support = ndimage.binary_fill_holes(smooth >= threshold)
    labels, number = ndimage.label(support)
    if not number:
        return result
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    support = labels == int(sizes.argmax())
    spacing = [edge[1]-edge[0] for edge in edges]
    # Padding supplies an outside boundary even if the component reaches an edge.
    distance = ndimage.distance_transform_edt(np.pad(support, 1), sampling=spacing)[1:-1, 1:-1]
    index = np.unravel_index(np.argmax(distance), distance.shape)
    radius = float(distance[index]-.75*np.hypot(*spacing))
    center = [float((edges[i][index[i]]+edges[i][index[i]+1])/2) for i in range(2)]
    if radius <= 2*max(spacing) or support.sum() < .03*bins*bins:
        result["reason"] = "The density component is too fragmented for a reliable circular proposal."
        return result
    result.update(supported=True, circle_cm=[*center, radius],
        reason="Largest interior circle in the main density region; inspect edge artifacts and composition bias.",
        diagnostics=dict(method=method, bins=bins, density_threshold=threshold, intensity_contrast=contrast,
                         finite_events=finite_count, histogram_events=int(counts.sum()),
                         bounds_sample_events=len(sample), bounds_quantiles=[.002, .998]))
    return result
