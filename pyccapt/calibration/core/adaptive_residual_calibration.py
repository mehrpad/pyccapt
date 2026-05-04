"""Adaptive residual calibration with learned peak templates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator, UnivariateSpline
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from pyccapt.calibration.core.exceptions import CalibrationInputError
from pyccapt.calibration.core.mc_plot_peak_helpers import gaussian_mrp_report


def _mode_aware_defaults(calibration_array, calibration_mode, user_template_bin_size, user_hist_bin_size=None):
    """Return template_bin_size and hist_bin_size scaled for tof vs mc domains.

    mc values live in ~0-400 Da where 0.01 Da bins are appropriate.
    tof values live in ~200-800 ns where peaks are relatively narrower,
    so we estimate a sensible bin size from the data range and typical
    peak count rather than using a fixed constant.
    """
    if calibration_mode == "tof":
        arr = np.asarray(calibration_array, dtype=float)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if arr.size > 100:
            data_range = float(np.percentile(arr, 99.5) - np.percentile(arr, 0.5))
        else:
            data_range = float(np.ptp(arr)) if arr.size > 0 else 1.0
        # For tof, a good template bin is ~0.1% of range (e.g. 0.5 ns for 500 ns span)
        auto_template_bin = max(0.05, data_range * 0.001)
        auto_hist_bin = max(0.1, data_range * 0.002)
        template_bin_size = user_template_bin_size if user_template_bin_size > auto_template_bin * 0.5 else auto_template_bin
        hist_bin_size = user_hist_bin_size if user_hist_bin_size is not None and user_hist_bin_size > auto_hist_bin * 0.5 else auto_hist_bin
    else:
        template_bin_size = max(user_template_bin_size, 1e-4)
        hist_bin_size = user_hist_bin_size if user_hist_bin_size is not None else max(user_template_bin_size, 1e-4)
    return float(template_bin_size), float(hist_bin_size)


@dataclass
class PeakTemplate:
    center: float
    x1: float
    x2: float
    bin_size: float
    offsets: np.ndarray
    logpdf: np.ndarray
    log_floor: float
    search_half_range: float
    n_ions: int


def _expanded_peak_mask(values, template, scale=1.0):
    half_range = max(
        template.search_half_range * max(1.0, float(scale)),
        (template.x2 - template.x1) * 0.75 * max(1.0, float(scale)),
        template.bin_size * 8.0,
    )
    return np.abs(np.asarray(values, dtype=float) - template.center) <= half_range


def _score_improved(candidate, best):
    if not np.isfinite(candidate):
        return False
    if not np.isfinite(best):
        return True
    return candidate > best + max(0.1, abs(best) * 0.002)


def _score_not_worse(candidate, best, tolerance_ratio=0.01):
    if not np.isfinite(best):
        return True
    if not np.isfinite(candidate):
        return False
    return candidate >= best - max(0.1, abs(best) * tolerance_ratio)


def _combined_quality_score(quality):
    train_score = quality.get("train_score", float("nan"))
    holdout_score = quality.get("holdout_score", float("nan"))
    if np.isfinite(train_score) and np.isfinite(holdout_score):
        return 0.75 * float(train_score) + 0.25 * float(holdout_score)
    if np.isfinite(train_score):
        return float(train_score)
    if np.isfinite(holdout_score):
        return float(holdout_score)
    return float("nan")


def _peak_quality_score(calibration_array, peak):
    local_bin_size = max(1e-4, min(0.02, (peak["x2"] - peak["x1"]) / 80.0))
    report = gaussian_mrp_report(calibration_array, peak["x1"], peak["x2"], bin_size=local_bin_size)
    if report is None:
        return float("nan")
    # Use recommended_mrp which applies the physical ceiling and cross-checks
    # Voigt/Gaussian/histogram values, preventing absurd scores.
    mrp_values = report["recommended_mrp"]
    weights = (0.6, 0.3, 0.1)
    score = 0.0
    weight_sum = 0.0
    for weight, value in zip(weights, mrp_values):
        if np.isfinite(value):
            score += weight * float(max(0.0, value))
            weight_sum += weight
    return float("nan") if weight_sum == 0 else score / weight_sum


def _evaluate_peak_group(calibration_array, peaks):
    weighted_scores = []
    for peak in peaks:
        score = _peak_quality_score(calibration_array, peak)
        if np.isfinite(score):
            weight = float(peak.get("weight", 1.0))
            weighted_scores.append((score, weight))
    if not weighted_scores:
        return float("nan")
    total_weight = sum(weight for _, weight in weighted_scores)
    return float(sum(score * weight for score, weight in weighted_scores) / total_weight)


def _split_reference_peaks(calibration_array, peaks, holdout_count=2):
    enriched = []
    for peak in peaks:
        mask = (calibration_array > peak["x1"]) & (calibration_array < peak["x2"])
        n_ions = int(np.count_nonzero(mask))
        if n_ions < 25:
            continue
        candidate = dict(peak)
        candidate["n_ions"] = n_ions
        candidate["weight"] = max(1.0, float(np.sqrt(n_ions)))
        enriched.append(candidate)
    enriched.sort(key=lambda item: (-item["n_ions"], item["position"]))
    holdout_n = min(holdout_count, max(0, len(enriched) // 3))
    if holdout_n == 0:
        return {"train": enriched, "holdout": []}
    return {"train": enriched[:-holdout_n], "holdout": enriched[-holdout_n:]}


def _evaluate_quality(calibration_array, reference_peaks):
    return {
        "train_score": _evaluate_peak_group(calibration_array, reference_peaks["train"]),
        "holdout_score": _evaluate_peak_group(calibration_array, reference_peaks["holdout"]),
    }


def _build_peak_template(calibration_array, peak, template_bin_size):
    mask = (calibration_array > peak["x1"]) & (calibration_array < peak["x2"])
    values = np.asarray(calibration_array[mask], dtype=float)
    # Aligned with the 25-ion floor in _split_reference_peaks. Returning
    # None lets the caller skip this peak instead of aborting the whole
    # calibration when one peak window is sparse (e.g. after a correction
    # step shifts ions outside the original auto-detected bounds).
    if values.size < 25:
        return None
    center = float(peak["position"])
    offsets = values - center
    search_half_range = max((peak["x2"] - peak["x1"]) * 0.75, template_bin_size * 6.0)
    n_bins = max(25, int(np.ceil((2.0 * search_half_range) / max(template_bin_size, 1e-6))))
    edges = np.linspace(-search_half_range, search_half_range, n_bins + 1)
    counts = np.histogram(offsets, bins=edges)[0].astype(float)
    counts = gaussian_filter1d(counts, sigma=1.0, mode="nearest")
    counts += max(1e-9, counts.max() * 1e-6)
    pdf = counts / np.sum(counts)
    centers = (edges[:-1] + edges[1:]) * 0.5
    logpdf = np.log(pdf)
    return PeakTemplate(
        center=center,
        x1=float(peak["x1"]),
        x2=float(peak["x2"]),
        bin_size=float(template_bin_size),
        offsets=centers,
        logpdf=logpdf,
        log_floor=float(np.min(logpdf) - 5.0),
        search_half_range=float(search_half_range),
        n_ions=int(values.size),
    )


def _estimate_shift_from_template(local_values, template):
    values = np.asarray(local_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 20:
        return np.nan, 0.0
    step = max(template.bin_size * 0.5, template.search_half_range / 120.0)
    shift_grid = np.arange(-template.search_half_range, template.search_half_range + step, step)
    centered = values[:, None] - template.center - shift_grid[None, :]
    logp = np.interp(
        centered.ravel(),
        template.offsets,
        template.logpdf,
        left=template.log_floor,
        right=template.log_floor,
    ).reshape(values.size, shift_grid.size)
    scores = np.sum(logp, axis=0)
    best_index = int(np.argmax(scores))
    return float(shift_grid[best_index]), float(scores[best_index])


def _window_slices(length, n_windows, overlap):
    n_windows = max(4, int(n_windows))
    overlap = float(np.clip(overlap, 0.0, 0.9))
    window_size = max(200, int(np.ceil(length / max(2, n_windows - overlap * (n_windows - 1)))))
    step = max(50, int(window_size * (1.0 - overlap)))
    starts = list(range(0, max(1, length - window_size + 1), step))
    if not starts or starts[-1] != max(0, length - window_size):
        starts.append(max(0, length - window_size))
    for start in starts:
        stop = min(length, start + window_size)
        if stop - start >= 50:
            yield int(start), int(stop)


def _collect_temporal_observations(calibration_array, template, n_windows, overlap, min_window_ions):
    thresholds = [max(10, int(min_window_ions)), max(10, int(min_window_ions * 0.6)), max(8, int(min_window_ions * 0.35))]
    scales = [1.0, 1.5, 2.0, 3.0]
    best_attempt = None
    for threshold in thresholds:
        for scale in scales:
            x_obs, y_obs, w_obs = [], [], []
            for start, stop in _window_slices(len(calibration_array), n_windows=n_windows, overlap=overlap):
                window_values = calibration_array[start:stop]
                local_values = window_values[_expanded_peak_mask(window_values, template, scale=scale)]
                if local_values.size < threshold:
                    continue
                shift, score = _estimate_shift_from_template(local_values, template)
                if not np.isfinite(shift):
                    continue
                x_obs.append(0.5 * (start + stop - 1))
                y_obs.append(shift)
                w_obs.append(max(1.0, np.sqrt(local_values.size)) * max(1.0, score / local_values.size))
            if best_attempt is None or len(x_obs) > best_attempt["n_observations"]:
                best_attempt = {"x_obs": np.asarray(x_obs, dtype=float), "y_obs": np.asarray(y_obs, dtype=float), "w_obs": np.asarray(w_obs, dtype=float), "n_observations": len(x_obs), "scale": scale, "threshold": threshold}
            if len(x_obs) >= 4:
                x_obs = np.asarray(x_obs, dtype=float)
                y_obs = np.asarray(y_obs, dtype=float)
                w_obs = np.asarray(w_obs, dtype=float)
                order = np.argsort(x_obs)
                return {
                    "x_obs": x_obs[order],
                    "y_obs": y_obs[order],
                    "w_obs": np.clip(w_obs[order], 1.0, None),
                    "n_observations": len(x_obs),
                    "scale": scale,
                    "threshold": threshold,
                }
    return best_attempt


def _fit_temporal_residual(calibration_array, template, n_windows, overlap, min_window_ions, smoothing):
    observations = _collect_temporal_observations(calibration_array, template, n_windows, overlap, min_window_ions)
    if observations is None or observations.get("n_observations", 0) < 4:
        return np.zeros_like(calibration_array), {"n_observations": 0}
    x_obs = observations["x_obs"]
    y_obs = observations["y_obs"]
    w_obs = observations["w_obs"]
    y_centered = y_obs - np.average(y_obs, weights=w_obs)
    unique_x = np.unique(x_obs)
    if unique_x.size < 4:
        correction = np.interp(np.arange(len(calibration_array)), x_obs, y_centered)
        return correction, {"n_observations": len(x_obs), "peak_center": template.center, "scale": observations.get("scale", 1.0), "threshold": observations.get("threshold", min_window_ions)}
    spline = UnivariateSpline(
        x_obs,
        y_centered,
        w=np.sqrt(w_obs),
        s=max(1.0, float(smoothing)) * len(x_obs),
        k=min(3, unique_x.size - 1),
    )
    correction = spline(np.arange(len(calibration_array), dtype=float))
    max_abs = max(np.nanstd(y_centered) * 4.0, np.nanpercentile(np.abs(y_centered), 95) * 1.5, 1e-6)
    return np.clip(correction, -max_abs, max_abs), {"n_observations": len(x_obs), "peak_center": template.center, "scale": observations.get("scale", 1.0), "threshold": observations.get("threshold", min_window_ions)}


def _fit_spatial_residual(calibration_array, template, x_det_cm, y_det_cm, grid_size, min_cell_ions):
    x_mm = np.asarray(x_det_cm, dtype=float) * 10.0
    y_mm = np.asarray(y_det_cm, dtype=float) * 10.0
    if x_mm.size != calibration_array.size or y_mm.size != calibration_array.size:
        return np.zeros_like(calibration_array), {"n_cells": 0}
    x_edges = np.linspace(np.nanmin(x_mm), np.nanmax(x_mm), max(4, int(grid_size)) + 1)
    y_edges = np.linspace(np.nanmin(y_mm), np.nanmax(y_mm), max(4, int(grid_size)) + 1)
    sum_grid = np.zeros((len(x_edges) - 1, len(y_edges) - 1), dtype=float)
    weight_grid = np.zeros_like(sum_grid)
    cell_count = 0
    best_payload = None
    for threshold in [max(8, int(min_cell_ions)), max(8, int(min_cell_ions * 0.6)), max(6, int(min_cell_ions * 0.35))]:
        for scale in [1.0, 1.5, 2.0, 3.0]:
            sum_grid.fill(0.0)
            weight_grid.fill(0.0)
            cell_count = 0
            for ix in range(sum_grid.shape[0]):
                x_mask = (x_mm >= x_edges[ix]) & (x_mm < x_edges[ix + 1])
                for iy in range(sum_grid.shape[1]):
                    cell_mask = x_mask & (y_mm >= y_edges[iy]) & (y_mm < y_edges[iy + 1])
                    if np.count_nonzero(cell_mask) < threshold:
                        continue
                    cell_values = calibration_array[cell_mask]
                    local_values = cell_values[_expanded_peak_mask(cell_values, template, scale=scale)]
                    if local_values.size < threshold:
                        continue
                    shift, _ = _estimate_shift_from_template(local_values, template)
                    if not np.isfinite(shift):
                        continue
                    shift_value = float(shift)
                    total_weight = max(1.0, float(np.sqrt(local_values.size)))
                    sum_grid[ix, iy] = shift_value * total_weight
                    weight_grid[ix, iy] = total_weight
                    cell_count += 1
            if best_payload is None or cell_count > best_payload["n_cells"]:
                best_payload = {"sum_grid": sum_grid.copy(), "weight_grid": weight_grid.copy(), "n_cells": cell_count, "scale": scale, "threshold": threshold}
            if cell_count >= 4:
                break
        if best_payload is not None and best_payload["n_cells"] >= 4:
            break
    if best_payload is None or best_payload["n_cells"] < 4:
        return np.zeros_like(calibration_array), {"n_cells": 0}
    sum_grid = best_payload["sum_grid"]
    weight_grid = best_payload["weight_grid"]
    mean_shift = np.divide(
        np.sum(sum_grid),
        np.sum(weight_grid),
        out=np.array(0.0, dtype=float),
        where=np.sum(weight_grid) > 0,
    )
    sum_grid = sum_grid - (weight_grid * float(mean_shift))
    smooth_sum = gaussian_filter(sum_grid, sigma=1.0, mode="nearest")
    smooth_weight = gaussian_filter(weight_grid, sigma=1.0, mode="nearest")
    with np.errstate(divide="ignore", invalid="ignore"):
        map_values = np.divide(smooth_sum, smooth_weight, out=np.zeros_like(smooth_sum), where=smooth_weight > 0)
    interp = RegularGridInterpolator(
        (((x_edges[:-1] + x_edges[1:]) * 0.5), ((y_edges[:-1] + y_edges[1:]) * 0.5)),
        map_values,
        bounds_error=False,
        fill_value=0.0,
    )
    correction = interp(np.column_stack([x_mm, y_mm]))
    max_abs = max(np.nanpercentile(np.abs(correction), 95) * 1.5, 1e-6)
    return np.clip(correction, -max_abs, max_abs), {"n_cells": best_payload["n_cells"], "peak_center": template.center, "scale": best_payload["scale"], "threshold": best_payload["threshold"]}


def _best_candidate(candidates, baseline_quality):
    accepted = []
    for candidate in candidates:
        quality = candidate["quality"]
        if _score_improved(quality["train_score"], baseline_quality["train_score"]) and _score_not_worse(
            quality["holdout_score"], baseline_quality["holdout_score"], tolerance_ratio=0.02
        ):
            accepted.append(candidate)
    if not accepted:
        return None
    return max(accepted, key=lambda item: _combined_quality_score(item["quality"]))


def adaptive_residual_calibration(
    variables,
    calibration_mode="mc",
    n_peaks=6,
    prominence=25,
    distance=10,
    n_windows=24,
    overlap=0.5,
    template_bin_size=0.01,
    temporal_smoothing=0.5,
    apply_spatial=True,
    spatial_grid=12,
    min_window_ions=40,
    min_cell_ions=35,
    max_rounds=8,
    verbose=True,
):
    """Apply adaptive residual calibration using learned peak templates."""
    from pyccapt.calibration.core import calibration as calibration_core

    calibration_key = "tof" if calibration_mode == "tof" else "mc"
    initial_array = np.asarray(variables.get_calibration_array(calibration_key), dtype=float)

    # Scale bin sizes for tof vs mc domains
    template_bin_size, hist_bin_size = _mode_aware_defaults(
        initial_array, calibration_mode, template_bin_size,
    )

    peaks = calibration_core.auto_detect_reference_peaks(
        initial_array, n_peaks=max(3, int(n_peaks)), prominence=prominence, distance=distance, hist_bin_size=hist_bin_size
    )
    reference_peaks = _split_reference_peaks(initial_array, peaks)
    if len(reference_peaks["train"]) < 2:
        raise CalibrationInputError("Adaptive residual calibration needs at least two training peaks")
    baseline_quality = _evaluate_quality(initial_array, reference_peaks)
    current_array = initial_array.copy()
    temporal_info = {"n_observations": 0}
    spatial_info = {"n_cells": 0}
    spatial_quality = None
    accepted_steps = []
    iteration_history = []
    stop_reason = "max_rounds"

    for round_index in range(max(1, int(max_rounds))):
        round_peaks = calibration_core.auto_detect_reference_peaks(
            current_array, n_peaks=max(3, int(n_peaks)), prominence=prominence, distance=distance, hist_bin_size=hist_bin_size
        )
        round_reference_peaks = _split_reference_peaks(current_array, round_peaks)
        if len(round_reference_peaks["train"]) < 2:
            stop_reason = "insufficient_peaks"
            break

        round_baseline_quality = _evaluate_quality(current_array, round_reference_peaks)
        round_quality = dict(round_baseline_quality)
        round_steps = []
        templates = [_build_peak_template(current_array, peak, template_bin_size) for peak in round_reference_peaks["train"]]
        peak_template_pairs = [(p, t) for p, t in zip(round_reference_peaks["train"], templates) if t is not None]
        if not peak_template_pairs:
            stop_reason = "insufficient_template_ions"
            break
        temporal_candidates = []
        for peak, template in peak_template_pairs:
            temporal_correction, candidate_info = _fit_temporal_residual(
                current_array, template, n_windows=n_windows, overlap=overlap, min_window_ions=min_window_ions, smoothing=temporal_smoothing
            )
            if candidate_info.get("n_observations", 0) < 4:
                continue
            temporal_candidate = current_array - temporal_correction
            temporal_candidates.append(
                {"label": f"temporal@{peak['position']:.4f}", "correction": temporal_correction, "quality": _evaluate_quality(temporal_candidate, round_reference_peaks), "info": candidate_info}
            )
        best_temporal = _best_candidate(temporal_candidates, round_quality)
        if best_temporal is not None:
            current_array = current_array - best_temporal["correction"]
            round_quality = dict(best_temporal["quality"])
            temporal_info = best_temporal["info"]
            round_steps.append(best_temporal["label"])

        spatial_quality = None
        if apply_spatial:
            templates = [_build_peak_template(current_array, peak, template_bin_size) for peak in round_reference_peaks["train"]]
            peak_template_pairs = [(p, t) for p, t in zip(round_reference_peaks["train"], templates) if t is not None]
            spatial_candidates = []
            for peak, template in peak_template_pairs:
                spatial_correction, candidate_info = _fit_spatial_residual(
                    current_array, template, variables.dld_x_det, variables.dld_y_det, grid_size=spatial_grid, min_cell_ions=min_cell_ions
                )
                if candidate_info.get("n_cells", 0) < 4:
                    continue
                spatial_candidate = current_array - spatial_correction
                spatial_candidates.append(
                    {"label": f"spatial@{peak['position']:.4f}", "correction": spatial_correction, "quality": _evaluate_quality(spatial_candidate, round_reference_peaks), "info": candidate_info}
                )
            best_spatial = _best_candidate(spatial_candidates, round_quality)
            if best_spatial is not None:
                current_array = current_array - best_spatial["correction"]
                round_quality = dict(best_spatial["quality"])
                spatial_quality = best_spatial["quality"]
                spatial_info = best_spatial["info"]
                round_steps.append(best_spatial["label"])

        iteration_history.append({"round": round_index + 1, "baseline_quality": round_baseline_quality, "final_quality": round_quality, "accepted_steps": list(round_steps)})
        if not round_steps:
            stop_reason = "no_improvement"
            break
        accepted_steps.extend(round_steps)

    if calibration_key == "tof":
        variables.dld_t_calib = current_array
    else:
        variables.mc_calib = current_array

    final_quality = _evaluate_quality(current_array, reference_peaks)
    result = {
        "reference_peaks": reference_peaks,
        "baseline_quality": baseline_quality,
        "final_quality": final_quality,
        "accepted_steps": accepted_steps,
        "temporal_info": temporal_info,
        "spatial_info": spatial_info,
        "used_spatial": bool(apply_spatial),
        "iteration_history": iteration_history,
        "n_iterations": len(iteration_history),
        "stop_reason": stop_reason,
    }
    if spatial_quality is not None:
        result["spatial_quality"] = spatial_quality
    if verbose:
        print(
            f"[Adaptive residual] train {baseline_quality['train_score']:.2f} -> "
            f"{final_quality['train_score']:.2f}; holdout {baseline_quality['holdout_score']:.2f} -> "
            f"{final_quality['holdout_score']:.2f}"
        )
        print(f"[Adaptive residual] accepted steps: {accepted_steps or ['none']}")
        print(
            f"[Adaptive residual] temporal observations: {temporal_info.get('n_observations', 0)}, "
            f"spatial cells: {spatial_info.get('n_cells', 0)}"
        )
        print(f"[Adaptive residual] rounds: {len(iteration_history)}, stop reason: {stop_reason}")
    return result
