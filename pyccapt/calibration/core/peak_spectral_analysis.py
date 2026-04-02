"""Peak background, ppm estimation, and overlap deconvolution helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import chi2

_FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))


@dataclass
class DeconvolutionComponent:
    """Single spectral component used in local overlap deconvolution."""

    label: str
    center: float
    row_index: int = -1


def build_histogram(values, bin_width, x_min=None, x_max=None):
    """Return histogram centers, counts, and edges for finite input values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite values are available for histogramming")
    bin_width = float(bin_width)
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")
    lo = float(np.min(values) if x_min is None else x_min)
    hi = float(np.max(values) if x_max is None else x_max)
    if hi <= lo:
        hi = lo + bin_width
    n_bins = max(10, int(np.ceil((hi - lo) / bin_width)))
    edges = lo + np.arange(n_bins + 1, dtype=float) * bin_width
    if edges[-1] < hi:
        edges = np.append(edges, edges[-1] + bin_width)
    counts, edges = np.histogram(values, bins=edges)
    centers = (edges[:-1] + edges[1:]) * 0.5
    return centers, counts.astype(float), edges


def suggest_sidebands(peak_left, peak_right, gap, width, min_x=None, max_x=None):
    """Return suggested left/right sideband windows around a peak window."""
    peak_left = float(peak_left)
    peak_right = float(peak_right)
    gap = max(0.0, float(gap))
    width = max(1e-6, float(width))
    left = [peak_left - gap - width, peak_left - gap]
    right = [peak_right + gap, peak_right + gap + width]
    if min_x is not None:
        left[0] = max(float(min_x), left[0])
        left[1] = max(float(min_x), left[1])
    if max_x is not None:
        right[0] = min(float(max_x), right[0])
        right[1] = min(float(max_x), right[1])
    return {
        "left": tuple(sorted(left)),
        "right": tuple(sorted(right)),
    }


def fit_local_linear_background(values, bin_width, peak_range, bg_left_range, bg_right_range, total_ions=None):
    """Fit a local linear background from left/right sidebands and quantify the peak."""
    peak_left, peak_right = map(float, peak_range)
    left0, left1 = map(float, bg_left_range)
    right0, right1 = map(float, bg_right_range)
    x_min = min(left0, peak_left, right0)
    x_max = max(left1, peak_right, right1)
    x, y, edges = build_histogram(values, bin_width, x_min=x_min, x_max=x_max)
    bg_mask = ((x >= left0) & (x <= left1)) | ((x >= right0) & (x <= right1))
    peak_mask = (x >= peak_left) & (x <= peak_right)
    if np.count_nonzero(bg_mask) < 2:
        raise ValueError("At least two background bins are required for linear background fitting")
    if np.count_nonzero(peak_mask) == 0:
        raise ValueError("Peak window does not contain any histogram bins")

    slope, intercept = np.polyfit(x[bg_mask], y[bg_mask], 1)
    fitted = np.maximum(0.0, slope * x + intercept)
    observed_peak_counts = float(np.sum(y[peak_mask]))
    background_peak_counts = float(np.sum(fitted[peak_mask]))
    corrected_counts = float(observed_peak_counts - background_peak_counts)
    peak_position = float(x[peak_mask][np.argmax(y[peak_mask])])
    total_ions = float(len(values) if total_ions is None else total_ions)
    peak_width = max(float(peak_right - peak_left), float(bin_width))
    return {
        "x": x,
        "y": y,
        "edges": edges,
        "background_y": fitted,
        "peak_position": peak_position,
        "peak_range": (peak_left, peak_right),
        "bg_left_range": (left0, left1),
        "bg_right_range": (right0, right1),
        "slope": float(slope),
        "intercept": float(intercept),
        "observed_peak_counts": observed_peak_counts,
        "background_peak_counts": background_peak_counts,
        "corrected_counts": corrected_counts,
        "corrected_ppm": corrected_counts / total_ions * 1.0e6,
        "background_fraction": background_peak_counts / max(observed_peak_counts, 1.0),
        "background_ppm_per_da": background_peak_counts / max(total_ions * peak_width, 1.0e-12) * 1.0e6,
    }


def estimate_background_ppm(values, background_range, target_range=None, confidence=0.95, total_ions=None):
    """Estimate ppm/Da background density from a background-only window."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    total_ions = float(values.size if total_ions is None else total_ions)
    bg_left, bg_right = map(float, background_range)
    if bg_right <= bg_left:
        raise ValueError("Background range must have positive width")
    target_left, target_right = map(float, target_range if target_range is not None else background_range)
    if target_right <= target_left:
        raise ValueError("Target range must have positive width")
    bg_width = float(bg_right - bg_left)
    target_width = float(target_right - target_left)
    bg_count = int(np.count_nonzero((values >= bg_left) & (values <= bg_right)))
    alpha = max(1e-12, 1.0 - float(confidence))
    lower_count = 0.0 if bg_count == 0 else 0.5 * chi2.ppf(alpha / 2.0, 2 * bg_count)
    upper_count = 0.5 * chi2.ppf(1.0 - alpha / 2.0, 2 * (bg_count + 1))
    density = bg_count / bg_width
    density_low = lower_count / bg_width
    density_high = upper_count / bg_width
    ppm_per_da = density / max(total_ions, 1.0) * 1.0e6
    ppm_per_da_low = density_low / max(total_ions, 1.0) * 1.0e6
    ppm_per_da_high = density_high / max(total_ions, 1.0) * 1.0e6
    expected_background = density * target_width
    expected_background_low = density_low * target_width
    expected_background_high = density_high * target_width
    return {
        "background_range": (bg_left, bg_right),
        "target_range": (target_left, target_right),
        "background_width": bg_width,
        "target_width": target_width,
        "background_counts": bg_count,
        "ppm_per_da": ppm_per_da,
        "ppm_per_da_ci": (ppm_per_da_low, ppm_per_da_high),
        "expected_background_counts": expected_background,
        "expected_background_counts_ci": (expected_background_low, expected_background_high),
        "confidence": confidence,
    }


def _profile(x, center, fwhm, shape="gaussian"):
    sigma = max(float(fwhm) / _FWHM_FACTOR, 1e-12)
    x = np.asarray(x, dtype=float)
    if shape == "pseudo_voigt":
        gamma = sigma
        gaussian = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        lorentz = gamma**2 / (((x - center) ** 2) + gamma**2)
        raw = 0.65 * gaussian + 0.35 * lorentz
    else:
        raw = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    total = np.sum(raw)
    return raw / total if total > 0 else raw


def _tokenize_candidate_text(candidate_text):
    if not candidate_text:
        return []
    return [token.strip() for token in re.split(r"[,\n;]+", str(candidate_text)) if token.strip()]


def resolve_deconvolution_components(range_data, window_range, candidate_text=""):
    """Resolve local deconvolution components from a range table and/or manual text."""
    left, right = map(float, window_range)
    requested = _tokenize_candidate_text(candidate_text)
    manual_components = []
    label_requests = []
    for token in requested:
        if "@" in token:
            label, center_text = token.rsplit("@", 1)
            manual_components.append(DeconvolutionComponent(label=label.strip() or token, center=float(center_text)))
            continue
        try:
            manual_components.append(DeconvolutionComponent(label=token, center=float(token)))
        except ValueError:
            label_requests.append(token.lower())

    rows = []
    include_range_rows = not manual_components or bool(label_requests)
    if include_range_rows and range_data is not None and not range_data.empty:
        data = range_data.copy()
        center_col = "mc" if "mc" in data.columns else "mass"
        low_col = "mc_low" if "mc_low" in data.columns else center_col
        high_col = "mc_up" if "mc_up" in data.columns else center_col

        def _label_for_row(row):
            for column in ("ion_name", "name", "ion"):
                value = row.get(column, "")
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return f"{row.get(center_col, 0.0):.4f}"

        overlap = (data[low_col].astype(float) <= right) & (data[high_col].astype(float) >= left)
        for idx, row in data.loc[overlap].iterrows():
            label = _label_for_row(row)
            if label_requests and label.lower() not in label_requests:
                continue
            rows.append(DeconvolutionComponent(label=label, center=float(row[center_col]), row_index=int(idx)))

        if label_requests and not rows:
            for idx, row in data.iterrows():
                label = _label_for_row(row)
                if label.lower() in label_requests:
                    rows.append(DeconvolutionComponent(label=label, center=float(row[center_col]), row_index=int(idx)))

    components = manual_components + rows
    if not components:
        raise ValueError("No deconvolution candidates were found. Load a range table or enter manual candidates like Cr@52.0, Mo@52.9")

    dedup = {}
    for comp in components:
        key = (comp.label, round(float(comp.center), 6))
        dedup[key] = comp
    resolved = sorted(dedup.values(), key=lambda comp: comp.center)
    return pd.DataFrame(
        {
            "label": [comp.label for comp in resolved],
            "center": [float(comp.center) for comp in resolved],
            "row_index": [int(comp.row_index) for comp in resolved],
        }
    )


def fit_peak_deconvolution(
    values,
    bin_width,
    window_range,
    components,
    shape="gaussian",
    background_mode="linear",
    initial_fwhm=None,
):
    """Fit a constrained local mixture model to an overlapping peak window."""
    left, right = map(float, window_range)
    if right <= left:
        raise ValueError("Window range must have positive width")
    component_df = pd.DataFrame(components).copy()
    if component_df.empty:
        raise ValueError("At least one component is required for deconvolution")
    x, y, edges = build_histogram(values, bin_width, x_min=left, x_max=right)
    centers = component_df["center"].to_numpy(dtype=float)
    window_width = right - left
    if initial_fwhm is None or initial_fwhm <= 0:
        initial_fwhm = max(float(bin_width) * 4.0, window_width / max(6.0, len(centers) * 2.5))
    x_zero = x - np.mean(x)
    signal_guess = max(np.sum(y) - np.min(y) * len(y), 1.0)
    n_comp = len(component_df)
    if background_mode == "linear":
        p0 = np.concatenate([[initial_fwhm, max(np.min(y), 0.0), 0.0], np.full(n_comp, signal_guess / n_comp)])
        lower = np.concatenate([[max(bin_width * 0.5, 1e-4), 0.0, -np.inf], np.zeros(n_comp)])
        upper = np.concatenate([[window_width, np.inf, np.inf], np.full(n_comp, max(np.sum(y) * 2.0, 1.0e9))])
        amp_offset = 3
    elif background_mode == "constant":
        p0 = np.concatenate([[initial_fwhm, max(np.min(y), 0.0)], np.full(n_comp, signal_guess / n_comp)])
        lower = np.concatenate([[max(bin_width * 0.5, 1e-4), 0.0], np.zeros(n_comp)])
        upper = np.concatenate([[window_width, np.inf], np.full(n_comp, max(np.sum(y) * 2.0, 1.0e9))])
        amp_offset = 2
    else:
        p0 = np.concatenate([[initial_fwhm], np.full(n_comp, signal_guess / n_comp)])
        lower = np.concatenate([[max(bin_width * 0.5, 1e-4)], np.zeros(n_comp)])
        upper = np.concatenate([[window_width], np.full(n_comp, max(np.sum(y) * 2.0, 1.0e9))])
        amp_offset = 1

    def _model(params):
        fwhm = float(params[0])
        if background_mode == "linear":
            bg = params[1] + params[2] * x_zero
        elif background_mode == "constant":
            bg = np.full_like(x, params[1], dtype=float)
        else:
            bg = np.zeros_like(x, dtype=float)
        amplitudes = np.asarray(params[amp_offset:], dtype=float)
        basis = np.vstack([_profile(x, center, fwhm, shape=shape) for center in centers])
        signal = amplitudes @ basis
        return np.maximum(0.0, bg + signal), basis, amplitudes

    def _residuals(params):
        model_y, _, _ = _model(params)
        weights = np.sqrt(np.maximum(y, 1.0))
        return (model_y - y) / weights

    fit = least_squares(_residuals, p0, bounds=(lower, upper), max_nfev=4000)
    fit_y, basis, amplitudes = _model(fit.x)
    if background_mode == "linear":
        bg_y = np.maximum(0.0, fit.x[1] + fit.x[2] * x_zero)
    elif background_mode == "constant":
        bg_y = np.full_like(x, max(fit.x[1], 0.0), dtype=float)
    else:
        bg_y = np.zeros_like(x, dtype=float)
    component_curves = amplitudes[:, None] * basis
    component_counts = amplitudes
    component_df["fitted_counts"] = component_counts
    component_df["fraction"] = component_counts / max(np.sum(component_counts), 1.0)
    grouped = component_df.groupby("label", as_index=False)["fitted_counts"].sum()
    grouped["fraction"] = grouped["fitted_counts"] / max(grouped["fitted_counts"].sum(), 1.0)
    return {
        "x": x,
        "y": y,
        "edges": edges,
        "fit_y": fit_y,
        "background_y": bg_y,
        "component_curves": component_curves,
        "components": component_df,
        "grouped_components": grouped.sort_values("fitted_counts", ascending=False).reset_index(drop=True),
        "fwhm": float(fit.x[0]),
        "shape": shape,
        "background_mode": background_mode,
        "success": bool(fit.success),
        "message": fit.message,
        "cost": float(fit.cost),
        "rmse": float(np.sqrt(np.mean((fit_y - y) ** 2))),
        "window_range": (left, right),
    }


__all__ = [
    "build_histogram",
    "estimate_background_ppm",
    "fit_local_linear_background",
    "fit_peak_deconvolution",
    "resolve_deconvolution_components",
    "suggest_sidebands",
]
