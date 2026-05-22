"""Peak finding and legend helpers for :mod:`mc_plot`."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.special import erf as _erf

_MRP_INTERNAL_BIN_SIZE = 0.01
_BOX_SELECTION_BIN_SIZE = 0.1
_MRP_REFERENCE_BIN_SIZE = 0.01
_MRP_MIN_BINS = 60
_MRP_EXPANSION_FACTOR = 3.0
_MRP_REFERENCE_GUARD_FACTOR = 3.0
_MRP_REFERENCE_MIN_WIDTH_RATIO = 0.35
# No single-run APT instrument physically achieves FWHM MRP above this value;
# anything higher is a fitting artefact from a narrow sub-peak or noise spike.
_MRP_PHYSICAL_CEILING = 1500.0
# Half-width (Da) of the auto-picked window centred on the tallest peak.
# Used by the "MRP" button auto-pick AND by the dominant-peak branch of
# calculate_mrp so the legend MRP and the on-demand MRP report operate on
# identical windows and therefore agree.
_AUTO_MRP_HALF_WIDTH = 0.8
# Histogram bin size used by the MRP fit inside gaussian_mrp_report. Must
# match the value the MRP button passes to gaussian_mrp_report, otherwise
# the legend and the report would fit on differently-binned histograms and
# produce slightly different MRPs.
_AUTO_MRP_BIN_SIZE = 0.001
# Coarser bin used by the auto-pick argmax search. The peak location within
# ±0.005 Da is plenty for the ±0.8 Da window, and a coarser histogram is
# orders of magnitude faster on multi-million-ion arrays. Using 0.001 here
# is what made the mass-spectrum plot slow (np.histogram with 800k bins on
# 5M points). The actual MRP *fit* still runs at 0.001 inside
# gaussian_mrp_report so the legend number stays exact.
_AUTO_MRP_PICK_BIN_SIZE = 0.01


def _auto_mrp_window_from_array(values, bin_size=_AUTO_MRP_PICK_BIN_SIZE, half_width=_AUTO_MRP_HALF_WIDTH):
    """Return ``(left, right, center)`` for the tallest histogram bin.

    Bin edges are anchored to multiples of ``bin_size`` from zero, so the
    argmax bin's midpoint does NOT depend on ``min``/``max`` of the input
    array. This makes the legend's dominant-peak auto-pick and the MRP
    button's auto-pick produce identical centers even when one operates on a
    filtered slice (e.g. ``hist[hist<40]``) and the other on the full series.
    """
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return None
    lo = float(np.min(data))
    hi = float(np.max(data))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    bs = max(bin_size, 1e-9)
    # Anchored grid: edges are multiples of bin_size irrespective of (lo, hi).
    start = float(np.floor(lo / bs) * bs)
    stop = float(np.ceil(hi / bs) * bs + bs)
    edges = np.arange(start, stop + 0.5 * bs, bs)
    if edges.size < 2:
        return None
    counts, edges = np.histogram(data, bins=edges)
    if counts.sum() == 0:
        return None
    peak_idx = int(np.argmax(counts))
    center = float(0.5 * (edges[peak_idx] + edges[peak_idx + 1]))
    return center - half_width, center + half_width, center


def _gaussian(x, amp, mu, sigma, bg):
    """Gaussian with constant background."""
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + bg


_FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))


def _fit_gaussian_mrp(x, y, peak_idx):
    """Fit a Gaussian around *peak_idx* and return MRP at 0.5, 0.1, 0.01."""
    nan3 = [float('nan')] * 3

    mu0 = x[peak_idx]
    amp0 = float(y[peak_idx])
    if amp0 <= 0:
        return nan3, False

    # Use a data-adaptive half-window: at least ±0.3 Da worth of bins,
    # so the fitting window is physically meaningful regardless of bin size.
    bin_step = float(x[1] - x[0]) if len(x) > 1 else 1.0
    min_hw_bins = max(15, int(np.ceil(0.3 / max(bin_step, 1e-9))))
    hw = min(min_hw_bins, peak_idx, len(x) - 1 - peak_idx)
    sl = slice(peak_idx - hw, peak_idx + hw + 1)
    xw, yw = x[sl], y[sl].astype(float)

    bg0 = float(np.min(yw))
    sigma0 = (x[min(peak_idx + 2, len(x) - 1)] - x[max(peak_idx - 2, 0)]) / _FWHM_FACTOR

    try:
        popt, _ = curve_fit(
            _gaussian,
            xw,
            yw,
            p0=[amp0 - bg0, mu0, sigma0, bg0],
            bounds=([0, xw[0], 1e-12, 0], [np.inf, xw[-1], xw[-1] - xw[0], np.inf]),
            maxfev=2000,
        )
    except (RuntimeError, ValueError):
        return nan3, False

    _, mu_fit, sigma_fit, _ = popt
    if sigma_fit <= 0:
        return nan3, False

    result = []
    for frac in [0.5, 0.1, 0.01]:
        fw = 2.0 * sigma_fit * np.sqrt(2.0 * np.log(1.0 / frac))
        result.append(round(float(mu_fit / fw), 2) if fw > 0 else float('nan'))
    return result, True


def _mrp_sides_from_values(center, mrp_values):
    """Return symmetric peak-width sides inferred from MRP values."""
    sides = []
    center = float(center)
    for value in mrp_values:
        if not np.isfinite(value) or value <= 0:
            sides.append([float('nan'), float('nan')])
            continue
        width = center / float(value)
        sides.append([center - width / 2.0, center + width / 2.0])
    return sides


def _mrp_width_from_fwhm(center, mrp_values):
    """Return the inferred FWHM width for the first MRP value."""
    if mrp_values is None or len(mrp_values) == 0:
        return float('nan')
    value = float(mrp_values[0])
    if not np.isfinite(value) or value <= 0:
        return float('nan')
    center = float(center)
    if not np.isfinite(center) or center <= 0:
        return float('nan')
    return center / value


def _format_mrp_value(value):
    """Format MRP values consistently for UI/reporting."""
    return 'NA' if not np.isfinite(value) else f'{float(value):.2f}'


def _sanitize_tail_widths(widths, mrp_values, full_span, min_resolution_step):
    """Mark unreliable low-intensity tail widths as unavailable."""
    clean_widths = list(widths)
    clean_mrp = list(mrp_values)
    for idx, width in enumerate(clean_widths):
        if not np.isfinite(width) or width <= 0:
            clean_widths[idx] = float('nan')
            clean_mrp[idx] = float('nan')
            continue
        if width >= full_span - min_resolution_step:
            clean_widths[idx] = float('nan')
            clean_mrp[idx] = float('nan')
            continue
        if idx > 0 and np.isfinite(clean_widths[idx - 1]) and width <= clean_widths[idx - 1] + min_resolution_step:
            clean_widths[idx] = float('nan')
            clean_mrp[idx] = float('nan')
    return clean_widths, clean_mrp


def _expand_mrp_window(calibration_array, x1, x2, bin_size):
    """Expand a selected peak window to include baseline on both sides."""
    values = np.asarray(calibration_array, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(x1), float(x2), False

    left = float(min(x1, x2))
    right = float(max(x1, x2))
    width = max(right - left, bin_size * 2.0)
    center = 0.5 * (left + right)
    target_width = max(width * _MRP_EXPANSION_FACTOR, _MRP_MIN_BINS * bin_size)

    data_min = float(np.min(values))
    data_max = float(np.max(values))
    expanded_left = max(data_min, center - target_width / 2.0)
    expanded_right = min(data_max, center + target_width / 2.0)

    current_width = expanded_right - expanded_left
    min_width = min(_MRP_MIN_BINS * bin_size, data_max - data_min)
    if current_width < min_width and data_max > data_min:
        deficit = min_width - current_width
        expanded_left = max(data_min, expanded_left - deficit / 2.0)
        expanded_right = min(data_max, expanded_right + deficit / 2.0)

    expanded = not np.isclose(expanded_left, left) or not np.isclose(expanded_right, right)
    return float(expanded_left), float(expanded_right), expanded


def _select_peak_index(x, peaks, requested_center):
    """Choose the detected peak nearest to the requested center."""
    if len(peaks) == 0:
        return None
    distances = np.abs(x[peaks] - float(requested_center))
    return int(np.argmin(distances))


def _voigt(x, amp, mu, sigma, gamma, bg):
    """Pseudo-Voigt approximation."""
    fg = _FWHM_FACTOR * sigma
    fl = 2.0 * gamma
    f5 = fg**5 + 2.69269 * fg**4 * fl + 2.42843 * fg**3 * fl**2 + 4.47163 * fg**2 * fl**3 + 0.07842 * fg * fl**4 + fl**5
    f_v = f5**0.2 if f5 > 0 else max(fg, fl)
    if f_v > 0:
        eta = 1.36603 * (fl / f_v) - 0.47719 * (fl / f_v) ** 2 + 0.11116 * (fl / f_v) ** 3
        eta = np.clip(eta, 0.0, 1.0)
    else:
        eta = 0.0
    gauss = np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    lorentz = gamma**2 / ((x - mu) ** 2 + gamma**2)
    return amp * ((1 - eta) * gauss + eta * lorentz) + bg


def _voigt_fwhm(sigma, gamma):
    """Compute FWHM of a pseudo-Voigt profile."""
    fg = _FWHM_FACTOR * sigma
    fl = 2.0 * gamma
    f5 = fg**5 + 2.69269 * fg**4 * fl + 2.42843 * fg**3 * fl**2 + 4.47163 * fg**2 * fl**3 + 0.07842 * fg * fl**4 + fl**5
    return f5**0.2 if f5 > 0 else max(fg, fl)


def _fit_voigt_mrp(x, y, peak_idx):
    """Fit a pseudo-Voigt around *peak_idx* and return MRP at 0.5, 0.1, 0.01."""
    nan3 = [float('nan')] * 3

    mu0 = x[peak_idx]
    amp0 = float(y[peak_idx])
    if amp0 <= 0:
        return nan3, False, float('nan'), 'unknown'

    # Use a data-adaptive half-window: at least ±0.3 Da worth of bins,
    # so the fitting window is physically meaningful regardless of bin size.
    bin_step = float(x[1] - x[0]) if len(x) > 1 else 1.0
    min_hw_bins = max(15, int(np.ceil(0.3 / max(bin_step, 1e-9))))
    hw = min(min_hw_bins, peak_idx, len(x) - 1 - peak_idx)
    sl = slice(peak_idx - hw, peak_idx + hw + 1)
    xw, yw = x[sl], y[sl].astype(float)

    bg0 = float(np.min(yw))
    sigma0 = (x[min(peak_idx + 2, len(x) - 1)] - x[max(peak_idx - 2, 0)]) / _FWHM_FACTOR
    gamma0 = sigma0

    try:
        popt, _ = curve_fit(
            _voigt,
            xw,
            yw,
            p0=[amp0 - bg0, mu0, sigma0, gamma0, bg0],
            bounds=([0, xw[0], 1e-12, 1e-12, 0], [np.inf, xw[-1], xw[-1] - xw[0], xw[-1] - xw[0], np.inf]),
            maxfev=4000,
        )
    except (RuntimeError, ValueError):
        return nan3, False, float('nan'), 'unknown'

    _, mu_fit, sigma_fit, gamma_fit, _ = popt
    fwhm = _voigt_fwhm(sigma_fit, gamma_fit)
    if fwhm <= 0:
        return nan3, False, float('nan'), 'unknown'

    fg = _FWHM_FACTOR * sigma_fit
    fl = 2.0 * gamma_fit
    if fwhm > 0:
        eta = 1.36603 * (fl / fwhm) - 0.47719 * (fl / fwhm) ** 2 + 0.11116 * (fl / fwhm) ** 3
        eta = float(np.clip(eta, 0.0, 1.0))
    else:
        eta = 0.0
    profile_type = 'Lorentzian-dominated' if eta > 0.5 else 'Gaussian-dominated'

    result = []
    widths = []
    fit_span = float(xw[-1] - xw[0])
    # MRP(0.5) (i.e. FWHM) is always within the fit window, so we keep the
    # original numerical sampling for it to preserve historical numerics that
    # downstream consumers (peak-window inference, residual calibration)
    # depend on. For the lower fractions (10% / 1% of max) the analytic Voigt
    # tail can extend well outside the fit window — for a Lorentzian the FWHM
    # at 1%-of-max is ~10x FWHM(0.5) — so we resample over a wider analytic
    # range driven by the fitted FWHM. This keeps MRP(0.5) numerically stable
    # while letting MRP(0.1) and MRP(0.01) report finite values when the
    # analytic curve genuinely drops below threshold within the wider range.
    base_edge_tol = max((xw[1] - xw[0]) * 2.0 if len(xw) > 1 else 0.0, fit_span / 400.0)
    wide_half_span = max(fit_span / 2.0, 15.0 * float(fwhm))
    for frac in [0.5, 0.1, 0.01]:
        if frac >= 0.5:
            x_fine = np.linspace(xw[0], xw[-1], 2000)
            edge_tol = base_edge_tol
        else:
            x_fine = np.linspace(
                float(mu_fit) - wide_half_span,
                float(mu_fit) + wide_half_span,
                4000,
            )
            sample_step = float(x_fine[1] - x_fine[0]) if len(x_fine) > 1 else 0.0
            edge_tol = sample_step
        y_fine = _voigt(x_fine, *popt) - popt[4]
        y_max = float(np.max(y_fine))
        if y_max <= 0:
            result.append(float('nan'))
            widths.append(float('nan'))
            continue
        threshold = frac * y_max
        above = x_fine[y_fine >= threshold]
        if len(above) < 2:
            result.append(float('nan'))
            widths.append(float('nan'))
            continue
        if above[0] <= x_fine[0] + edge_tol or above[-1] >= x_fine[-1] - edge_tol:
            result.append(float('nan'))
            widths.append(float('nan'))
            continue
        fw = float(above[-1] - above[0])
        widths.append(fw)
        result.append(round(float(mu_fit / fw), 2) if fw > 0 else float('nan'))

    # The sanitizer rejects widths that meet or exceed `full_span`. For
    # MRP(0.5) the width is always within the fit window, but MRP(0.1) and
    # MRP(0.01) widths can comfortably exceed `fit_span` for Lorentzian-
    # dominated profiles — that's the whole point of the wider analytic
    # sampling above. Use the wider analytic span so legitimate tail widths
    # are not falsely marked NaN.
    sanitize_full_span = max(fit_span, 2.0 * wide_half_span)
    widths, result = _sanitize_tail_widths(widths, result, sanitize_full_span, max(base_edge_tol, fit_span / 200.0))

    return result, True, float(fwhm), profile_type


# Asymmetric peak shape adapted from APyT's `error-expDecay` model
# (sebi-85/apyt: apyt/spectrum/fit.py). The rising edge is an error-function
# activation; the trailing edge is an exponential decay, which captures the
# delayed-evaporation/thermal tail characteristic of APT peaks better than a
# Gaussian or pseudo-Voigt.
_SQRT2 = float(np.sqrt(2.0))


def _err_exp_decay(x, amp, center, sigma, tail, bg):
    """Error-function rise multiplied by an exponential decay."""
    sigma = max(float(sigma), 1e-12)
    tail = max(float(tail), 1e-12)
    activation = 0.5 * (1.0 + _erf((x - center) / (_SQRT2 * sigma)))
    dx = x - center
    decay = np.where(dx > 0.0, np.exp(-dx / tail), 1.0)
    return amp * activation * decay + bg


def _fit_asymmetric_mrp(x, y, peak_idx):
    """Fit err*expDecay around *peak_idx* and return MRP at 0.5, 0.1, 0.01."""
    nan3 = [float('nan')] * 3

    mu0 = float(x[peak_idx])
    amp0 = float(y[peak_idx])
    if amp0 <= 0:
        return nan3, False, float('nan')

    bin_step = float(x[1] - x[0]) if len(x) > 1 else 1.0
    min_hw_bins = max(15, int(np.ceil(0.3 / max(bin_step, 1e-9))))
    hw = min(min_hw_bins, peak_idx, len(x) - 1 - peak_idx)
    sl = slice(peak_idx - hw, peak_idx + hw + 1)
    xw, yw = x[sl], y[sl].astype(float)
    if xw.size < 6:
        return nan3, False, float('nan')

    bg0 = float(np.min(yw))
    sigma0 = max((x[min(peak_idx + 2, len(x) - 1)] - x[max(peak_idx - 2, 0)]) / _FWHM_FACTOR, bin_step * 0.5)
    tail0 = sigma0
    span = float(xw[-1] - xw[0])

    try:
        popt, _ = curve_fit(
            _err_exp_decay,
            xw,
            yw,
            p0=[amp0 - bg0, mu0, sigma0, tail0, bg0],
            bounds=(
                [0.0, xw[0], 1e-12, 1e-12, 0.0],
                [np.inf, xw[-1], span, span, np.inf],
            ),
            maxfev=4000,
        )
    except (RuntimeError, ValueError):
        return nan3, False, float('nan')

    amp_fit, mu_fit, sigma_fit, tail_fit, bg_fit = popt
    if sigma_fit <= 0 or tail_fit <= 0 or amp_fit <= 0:
        return nan3, False, float('nan')

    # Resolve MRP fractions analytically on a fine grid: the asymmetric tail
    # may extend well past the fit window for shallow decays, so widen the
    # sampling like the Voigt path does.
    fit_span = span
    base_edge_tol = max((xw[1] - xw[0]) * 2.0 if len(xw) > 1 else 0.0, fit_span / 400.0)
    rough_fwhm = _FWHM_FACTOR * sigma_fit + 1.5 * tail_fit
    wide_half_span = max(fit_span / 2.0, 15.0 * rough_fwhm)

    result = []
    widths = []
    fwhm_value = float('nan')
    for frac in [0.5, 0.1, 0.01]:
        if frac >= 0.5:
            x_fine = np.linspace(xw[0], xw[-1], 2000)
            edge_tol = base_edge_tol
        else:
            x_fine = np.linspace(mu_fit - wide_half_span, mu_fit + wide_half_span, 4000)
            edge_tol = float(x_fine[1] - x_fine[0]) if len(x_fine) > 1 else 0.0
        y_fine = _err_exp_decay(x_fine, *popt) - bg_fit
        y_max = float(np.max(y_fine))
        if y_max <= 0:
            result.append(float('nan'))
            widths.append(float('nan'))
            continue
        above = x_fine[y_fine >= frac * y_max]
        if len(above) < 2:
            result.append(float('nan'))
            widths.append(float('nan'))
            continue
        if above[0] <= x_fine[0] + edge_tol or above[-1] >= x_fine[-1] - edge_tol:
            result.append(float('nan'))
            widths.append(float('nan'))
            continue
        fw = float(above[-1] - above[0])
        widths.append(fw)
        result.append(round(float(mu_fit / fw), 2) if fw > 0 else float('nan'))
        if frac == 0.5 and fw > 0:
            fwhm_value = fw

    sanitize_full_span = max(fit_span, 2.0 * wide_half_span)
    widths, result = _sanitize_tail_widths(widths, result, sanitize_full_span, max(base_edge_tol, fit_span / 200.0))

    return result, True, fwhm_value


def _plotter_peak_window(plotter, peak_array_index):
    """Return a robust search window for a plotted peak using raw-ion MRP resolution."""
    x_axis = plotter.x_centers if getattr(plotter, 'x_centers', None) is not None else plotter.x[:-1]
    left = float(np.interp(plotter.peak_widths[2][peak_array_index], np.arange(len(x_axis)), x_axis))
    right = float(np.interp(plotter.peak_widths[3][peak_array_index], np.arange(len(x_axis)), x_axis))
    center = float(x_axis[plotter.peaks[peak_array_index]])
    coarse_width = max(right - left, float(plotter.bin_width or 0.0), _MRP_INTERNAL_BIN_SIZE * _MRP_MIN_BINS)
    search_width = max(coarse_width * 4.0, float(plotter.bin_width or 0.0) * 8.0, _MRP_INTERNAL_BIN_SIZE * _MRP_MIN_BINS * 4.0)
    data = np.asarray(plotter.mc_tof, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return left, right, center
    search_left = max(float(np.min(data)), center - search_width / 2.0)
    search_right = min(float(np.max(data)), center + search_width / 2.0)
    return search_left, search_right, center


def _recommended_mrp_payload(report, above_ceiling_strategy='nan'):
    """Return the recommended robust MRP set from a report.

    Priority order: Asymmetric (err*expDecay, APyT-style) > Voigt > Histogram.
    The asymmetric fit captures the exponential tail of real APT peaks and is
    the primary number; Voigt covers symmetric tails; the histogram width is
    the final independent fallback. Gaussian is no longer used in the chain
    (its MRP for tailed APT peaks is systematically biased) but is still
    reported alongside the others for cross-checking.

    Parameters
    ----------
    above_ceiling_strategy : {'nan', 'voigt_capped', 'fit_capped'}, default 'nan'
        Behavior when the histogram-fallback MRP exceeds the physical ceiling
        (1500). The default 'nan' returns NaN for the recommended MRP (legacy
        behavior). 'voigt_capped' falls back to the Voigt MRP capped at the
        ceiling (or asymmetric if voigt unavailable). 'fit_capped' tries
        asymmetric -> voigt -> histogram, capping each at the ceiling.

        The non-default modes unblock the V+Bowl optimizer on already-well-
        calibrated data: with the default mode it sees NaN > X = False and
        reverts every improvement. With capping the optimizer can at least
        hold steady (ceiling-vs-ceiling tie) rather than reverting.
    """
    if report is None:
        return [float('nan')] * 3, 'unavailable'
    histogram_values = report.get('histogram_mrp', [float('nan')] * 3)
    histogram_valid = np.isfinite(histogram_values[0])
    voigt_values = report.get('voigt_mrp', [float('nan')] * 3)
    asym_values = report.get('asymmetric_mrp', [float('nan')] * 3)
    asym_finite = bool(report.get('asymmetric_ok')) and np.isfinite(asym_values[0])

    # Cross-check the asymmetric fit against the histogram width (the most
    # reliable independent reference). If histogram MRP isn't usable, skip the
    # cross-check rather than falling back to the Gaussian fit.
    if histogram_valid:
        asym_cross_check_ok = asym_values[0] <= histogram_values[0] * _MRP_REFERENCE_GUARD_FACTOR
    else:
        asym_cross_check_ok = True

    asym_valid = (
        asym_finite
        and asym_cross_check_ok
        and asym_values[0] <= _MRP_PHYSICAL_CEILING
    )
    if asym_valid:
        return report['asymmetric_mrp'], 'Asymmetric (err*expDecay)'

    # Cross-check Voigt the same way -- only against the histogram reference.
    if histogram_valid:
        voigt_cross_check_ok = voigt_values[0] <= histogram_values[0] * _MRP_REFERENCE_GUARD_FACTOR
    else:
        voigt_cross_check_ok = True

    voigt_valid = (
        report.get('voigt_ok')
        and np.isfinite(voigt_values[0])
        and voigt_cross_check_ok
        and voigt_values[0] <= _MRP_PHYSICAL_CEILING
    )
    if voigt_valid:
        return report['voigt_mrp'], f'Voigt ({report["profile_type"]})'

    # Above-physical-ceiling histogram-fallback path. Default keeps the
    # legacy NaN behavior. Non-default strategies pick the highest-quality
    # available fit and cap it at the physical ceiling instead, so the
    # V+Bowl optimizer can rank "very narrow" candidates as ties rather
    # than reverting them as "worse than baseline".
    use_strategy = above_ceiling_strategy in ('voigt_capped', 'fit_capped')
    if histogram_valid and histogram_values[0] > _MRP_PHYSICAL_CEILING:
        if use_strategy:
            if asym_finite and np.isfinite(asym_values[0]) and asym_values[0] > 0:
                return _cap_at_ceiling(asym_values), 'Asymmetric (capped at ceiling)'
            if report.get('voigt_ok') and np.isfinite(voigt_values[0]) and voigt_values[0] > 0:
                return _cap_at_ceiling(voigt_values), 'Voigt (capped at ceiling)'
            return _cap_at_ceiling(histogram_values), 'Histogram (capped at ceiling)'
        # Default: NaN-out for legacy callers.
        return [float('nan')] * 3, 'Histogram (above physical ceiling)'

    # ALSO handle the case where histogram fallback is unusable (e.g. it
    # returned NaN due to noise around a sub-pixel-narrow peak) but Voigt
    # is finite and above ceiling -- legacy code falls through to the
    # NaN histogram below. The 'voigt_capped' strategy promotes the
    # Voigt-or-asymmetric fit and caps it at the ceiling, which is the
    # right signal for the optimizer: "the peak is very narrow".
    if use_strategy and not histogram_valid:
        if asym_finite and np.isfinite(asym_values[0]) and asym_values[0] > 0:
            return _cap_at_ceiling(asym_values), 'Asymmetric (capped, no histogram fallback)'
        if report.get('voigt_ok') and np.isfinite(voigt_values[0]) and voigt_values[0] > 0:
            return _cap_at_ceiling(voigt_values), 'Voigt (capped, no histogram fallback)'

    return histogram_values, 'Histogram'


def _cap_at_ceiling(mrp_values):
    """Cap each MRP value at the physical ceiling; pass NaN through as-is."""
    out = []
    for v in mrp_values:
        if np.isfinite(v) and v > 0:
            out.append(min(float(v), float(_MRP_PHYSICAL_CEILING)))
        else:
            out.append(float(v))
    return out


def _selection_fraction_index(percent):
    """Map UI percent values onto the supported MRP fractions."""
    target = max(0.0, min(1.0, float(percent) / 100.0))
    fractions = [0.5, 0.1, 0.01]
    return int(np.argmin([abs(target - frac) for frac in fractions]))


def _internal_peak_seed(values, percent=50, bin_size=_MRP_INTERNAL_BIN_SIZE, center_hint=None):
    """Detect a dominant peak on a separate high-resolution histogram."""
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size < 10:
        return None

    data_min = float(np.min(data))
    data_max = float(np.max(data))
    if not np.isfinite(data_min) or not np.isfinite(data_max) or data_max <= data_min:
        return None

    n_bins = max(_MRP_MIN_BINS, int(np.ceil((data_max - data_min) / max(bin_size, 1e-9))))
    edges = np.linspace(data_min, data_max, n_bins + 1)
    y, edges = np.histogram(data, bins=edges)
    x = (edges[:-1] + edges[1:]) * 0.5
    if len(x) == 0:
        return None

    y_smooth = gaussian_filter1d(y.astype(float), sigma=1.0, mode='nearest')
    try:
        peaks, _ = find_peaks(y_smooth, height=max(1.0, float(np.max(y_smooth)) * 0.01))
    except ValueError:
        peaks = np.array([], dtype=int)
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(y_smooth))], dtype=int)

    if center_hint is not None:
        peak_choice = int(peaks[_select_peak_index(x, peaks, center_hint)])
    else:
        peak_choice = int(peaks[np.argmax(y_smooth[peaks])])

    rel_height = max(0.0, min(1.0, (100.0 - float(percent)) / 100.0))
    try:
        widths = peak_widths(y_smooth, np.array([peak_choice]), rel_height=rel_height)
        left = float(np.interp(widths[2][0], np.arange(len(x)), x))
        right = float(np.interp(widths[3][0], np.arange(len(x)), x))
    except (ValueError, IndexError):
        coarse_width = max(bin_size * _MRP_MIN_BINS, (data_max - data_min) * 0.02)
        left = float(max(data_min, x[peak_choice] - coarse_width / 2.0))
        right = float(min(data_max, x[peak_choice] + coarse_width / 2.0))

    if not np.isfinite(left) or not np.isfinite(right) or right <= left:
        coarse_width = max(bin_size * _MRP_MIN_BINS, (data_max - data_min) * 0.02)
        left = float(max(data_min, x[peak_choice] - coarse_width / 2.0))
        right = float(min(data_max, x[peak_choice] + coarse_width / 2.0))

    return {
        'left': left,
        'right': right,
        'center': float(x[peak_choice]),
        'height': float(y[peak_choice]) if 0 <= peak_choice < len(y) else 0.0,
    }


def _display_peak_height(plotter, center):
    """Estimate the plotted histogram height nearest to the requested center."""
    x_axis = plotter.x_centers if getattr(plotter, 'x_centers', None) is not None else plotter.x[:-1]
    if x_axis is None or plotter.y is None or len(x_axis) == 0 or len(plotter.y) == 0:
        return 0.0
    index = int(np.argmin(np.abs(x_axis - float(center))))
    index = min(max(index, 0), len(plotter.y) - 1)
    return float(plotter.y[index])


def _auto_peak_selection(plotter):
    """Choose the automatic selection box from a fixed coarse histogram, independent of plot bin size."""
    seed = _internal_peak_seed(
        plotter.mc_tof,
        percent=getattr(plotter, 'percent', 50),
        bin_size=_BOX_SELECTION_BIN_SIZE,
    )
    if seed is None:
        return None

    report = gaussian_mrp_report(
        plotter.mc_tof,
        seed['left'],
        seed['right'],
        bin_size=_MRP_INTERNAL_BIN_SIZE,
        peak_center=seed['center'],
    )
    if report is None:
        height = _display_peak_height(plotter, seed['center'])
        return {
            'left': float(seed['left']),
            'right': float(seed['right']),
            'center': float(seed['center']),
            'height': height,
            'report': None,
        }

    center = float(seed['center'])
    height = _display_peak_height(plotter, center)
    return {
        'left': float(seed['left']),
        'right': float(seed['right']),
        'center': center,
        'height': height,
        'report': report,
    }


def build_calibration_core_mask(calibration_array, x1, x2, calibration_mode='mc'):
    """Build a tighter internal calibration mask without changing the visible box."""
    values = np.asarray(calibration_array, dtype=float)
    values = values[np.isfinite(values)]
    left = float(min(x1, x2))
    right = float(max(x1, x2))
    if values.size == 0 or right <= left:
        return np.zeros_like(np.asarray(calibration_array, dtype=bool), dtype=bool)

    base_mask = np.logical_and(
        np.asarray(calibration_array, dtype=float) > left, np.asarray(calibration_array, dtype=float) < right
    )
    if calibration_mode != 'mc' or np.count_nonzero(base_mask) < 25:
        return base_mask

    selected_values = np.asarray(calibration_array, dtype=float)[base_mask]
    seed = _internal_peak_seed(
        selected_values,
        percent=50,
        bin_size=_BOX_SELECTION_BIN_SIZE,
        center_hint=0.5 * (left + right),
    )
    if seed is None:
        return base_mask

    center = float(seed['center'])
    core_left = max(left, center - 0.5 * _BOX_SELECTION_BIN_SIZE)
    core_right = min(right, center + 1.5 * _BOX_SELECTION_BIN_SIZE)
    if core_right <= core_left:
        return base_mask

    core_mask = np.logical_and(
        np.asarray(calibration_array, dtype=float) > core_left,
        np.asarray(calibration_array, dtype=float) < core_right,
    )
    if np.count_nonzero(core_mask) < max(25, int(np.count_nonzero(base_mask) * 0.05)):
        return base_mask
    return core_mask


def _refine_selection_window(plotter, peak_array_index):
    """Build the automatic selection window from an internal 0.001-bin MRP histogram."""
    peak_left, peak_right, peak_center = _plotter_peak_window(plotter, peak_array_index)
    report = gaussian_mrp_report(
        plotter.mc_tof,
        peak_left,
        peak_right,
        bin_size=_MRP_INTERNAL_BIN_SIZE,
        peak_center=peak_center,
    )
    if report is None:
        return peak_left, peak_right
    frac_index = _selection_fraction_index(getattr(plotter, 'percent', 50))
    candidate_sides = report['recommended_peak_sides'][frac_index]
    if np.all(np.isfinite(candidate_sides)) and candidate_sides[1] > candidate_sides[0]:
        return float(candidate_sides[0]), float(candidate_sides[1])
    return float(report['used_window'][0]), float(report['used_window'][1])


def apply_hist_info_legend(plotter, label='mc', mrp_all=False, background=None, legend_mode='long', loc='left'):
    """Plot histogram info legend on the provided plotter axis."""
    selected_x1 = getattr(plotter.variables, 'selected_x1', 0.0) if plotter.variables is not None else 0.0
    selected_x2 = getattr(plotter.variables, 'selected_x2', 0.0) if plotter.variables is not None else 0.0
    if selected_x2 > selected_x1:
        report = gaussian_mrp_report(
            plotter.mc_tof,
            selected_x1,
            selected_x2,
            bin_size=_MRP_INTERNAL_BIN_SIZE,
            peak_center=0.5 * (selected_x1 + selected_x2),
        )
    else:
        auto_selection = _auto_peak_selection(plotter)
        report = auto_selection['report'] if auto_selection is not None else None
    recommended_mrp, recommended_label = _recommended_mrp_payload(report)
    frac_index = {50: 0, 10: 1, 1: 2}.get(int(plotter.percent), 0)
    mrp = _format_mrp_value(recommended_mrp[frac_index] if frac_index < len(recommended_mrp) else float('nan'))
    if mrp_all:
        mrp_list = report['recommended_mrp'] if report is not None else [float('nan')] * 3
    if label in ('mc', 'mc_c'):
        if background is not None:
            if mrp_all:
                if legend_mode == 'long':
                    txt = (
                        'plot bin: %s Da\nnum atoms: %.2f$e^6$\nbackG: %s ppm/Da\nMRP(0.5): %s\nMRP(0.1): %s\nMRP(0.01): %s'
                        % (
                            plotter.bin_width,
                            len(plotter.mc_tof) / 1000000,
                            round(plotter.background_ppm),
                            _format_mrp_value(mrp_list[0]),
                            _format_mrp_value(mrp_list[1]),
                            _format_mrp_value(mrp_list[2]),
                        )
                    )
                else:
                    txt = 'MRP(0.5): %s\nMRP(0.1): %s\nMRP(0.01): %s' % (
                        _format_mrp_value(mrp_list[0]),
                        _format_mrp_value(mrp_list[1]),
                        _format_mrp_value(mrp_list[2]),
                    )
            else:
                if legend_mode == 'long':
                    txt = 'plot bin: %s Da\nnum atoms: %.2f$e^6$\nbackG: %s ppm/Da\nFW%d%%M: %s' % (
                        plotter.bin_width,
                        len(plotter.mc_tof) / 1000000,
                        round(plotter.background_ppm),
                        plotter.percent,
                        mrp,
                    )
                else:
                    txt = 'MRP(0.5): %s' % mrp
        else:
            if mrp_all:
                if legend_mode == 'long':
                    txt = 'plot bin: %s Da\nnum atoms: %.2f$e^6$\nMRP(0.5): %s\nMRP(0.1): %s\nMRP(0.01): %s' % (
                        plotter.bin_width,
                        len(plotter.mc_tof) / 1000000,
                        _format_mrp_value(mrp_list[0]),
                        _format_mrp_value(mrp_list[1]),
                        _format_mrp_value(mrp_list[2]),
                    )
                else:
                    txt = 'MRP(0.5): %s\nMRP(0.1): %s\nMRP(0.01): %s' % (
                        _format_mrp_value(mrp_list[0]),
                        _format_mrp_value(mrp_list[1]),
                        _format_mrp_value(mrp_list[2]),
                    )
            else:
                if legend_mode == 'long':
                    txt = 'plot bin: %s Da\nnum atoms: %.2f$e^6$\nMRP(%s): %s' % (
                        plotter.bin_width,
                        len(plotter.mc_tof) / 1000000,
                        plotter.percent / 100,
                        mrp,
                    )
                else:
                    txt = 'MRP(0.5): %s' % mrp

    elif label in ('tof', 'tof_c'):
        if background is not None:
            if mrp_all:
                if legend_mode == 'long':
                    txt = (
                        'plot bin: %s ns\nnum atoms: %.2f$e^6$\nbackG: %s ppm/ns\nMRP(0.5): %s\nMRP(0.1): %s\nMRP(0.01): %s'
                        % (
                            plotter.bin_width,
                            len(plotter.mc_tof) / 1000000,
                            round(plotter.background_ppm),
                            _format_mrp_value(mrp_list[0]),
                            _format_mrp_value(mrp_list[1]),
                            _format_mrp_value(mrp_list[2]),
                        )
                    )
                else:
                    txt = 'MRP(0.5): %s\nMRP(0.1): %s\nMRP(0.01): %s' % (
                        _format_mrp_value(mrp_list[0]),
                        _format_mrp_value(mrp_list[1]),
                        _format_mrp_value(mrp_list[2]),
                    )
            else:
                if legend_mode == 'long':
                    txt = 'plot bin: %s ns\nnum atoms: %.2f$e^6$\nbackG: %s ppm/ns\nMRP(%s): %s' % (
                        plotter.bin_width,
                        len(plotter.mc_tof) / 1000000,
                        round(plotter.background_ppm),
                        plotter.percent / 100,
                        mrp,
                    )
                else:
                    txt = 'MRP(0.5): %s' % mrp
        else:
            if mrp_all:
                if legend_mode == 'long':
                    txt = 'plot bin: %s ns\nnum atoms: %.2f$e^6$\nMRP(0.5): %s\nMRP(0.1): %s\nMRP(0.01): %s' % (
                        plotter.bin_width,
                        len(plotter.mc_tof) / 1000000,
                        _format_mrp_value(mrp_list[0]),
                        _format_mrp_value(mrp_list[1]),
                        _format_mrp_value(mrp_list[2]),
                    )
                else:
                    txt = 'MRP(0.5): %s\nMRP(0.1): %s\nMRP(0.01): %s' % (
                        _format_mrp_value(mrp_list[0]),
                        _format_mrp_value(mrp_list[1]),
                        _format_mrp_value(mrp_list[2]),
                    )
            else:
                if legend_mode == 'long':
                    txt = 'plot bin: %s ns\nnum atoms: %.2f$e^6$ \nMRP(%s): %s' % (
                        plotter.bin_width,
                        len(plotter.mc_tof) / 1000000,
                        plotter.percent / 100,
                        mrp,
                    )
                else:
                    txt = 'MRP(0.5): %s' % mrp
    else:
        raise ValueError(f"Unsupported label for legend: {label!r}")

    props = dict(boxstyle='round', facecolor='#CCCCCC', alpha=1)
    if loc == 'left':
        plotter.ax.text(
            0.01,
            0.95,
            txt,
            va='top',
            ma='left',
            transform=plotter.ax.transAxes,
            bbox=props,
            fontsize=10,
            alpha=1,
            horizontalalignment='left',
            verticalalignment='top',
        )
    elif loc == 'right':
        plotter.ax.text(
            0.98,
            0.95,
            txt,
            va='top',
            ma='left',
            transform=plotter.ax.transAxes,
            bbox=props,
            fontsize=10,
            alpha=1,
            horizontalalignment='right',
            verticalalignment='top',
        )


def calculate_mrp(plotter):
    """Calculate MRP values with a robust profile fit for the dominant peak."""
    mrp_range = [0.5, 0.1, 0.01]
    mrp_peak = []
    mrp = {}

    if plotter.peaks is None or plotter.peak_widths is None or plotter.prominences is None or len(plotter.peaks) == 0:
        return mrp_peak, mrp
    idx_max = int(np.argmax(plotter.prominences[0]))
    dominant_seed = _internal_peak_seed(
        plotter.mc_tof,
        percent=getattr(plotter, 'percent', 50),
        bin_size=_BOX_SELECTION_BIN_SIZE,
    )
    if dominant_seed is not None:
        x_axis = plotter.x_centers if getattr(plotter, 'x_centers', None) is not None else plotter.x[:-1]
        peak_centers = x_axis[plotter.peaks]
        idx_max = int(np.argmin(np.abs(peak_centers - float(dominant_seed['center']))))
    # Reuse the same auto-pick window the MRP button uses, so the legend value
    # matches the PEAK PROFILE MRP REPORT for the dominant peak.
    auto_dominant_window = _auto_mrp_window_from_array(plotter.mc_tof)
    peak_reports = []
    for i in range(len(plotter.peaks)):
        if i == idx_max:
            if auto_dominant_window is not None:
                peak_left, peak_right, peak_center = auto_dominant_window
            elif dominant_seed is None:
                peak_left, peak_right, peak_center = _plotter_peak_window(plotter, i)
            else:
                peak_left = float(dominant_seed['left'])
                peak_right = float(dominant_seed['right'])
                peak_center = float(dominant_seed['center'])
            report = gaussian_mrp_report(
                plotter.mc_tof,
                peak_left,
                peak_right,
                bin_size=_AUTO_MRP_BIN_SIZE,
                peak_center=peak_center,
            )
            if report is not None:
                peak_reports.append(report)
                continue

        peak_left, peak_right, peak_center = _plotter_peak_window(plotter, i)
        mrp_values = fast_mrp(
            plotter.mc_tof,
            peak_left,
            peak_right,
            bin_size=_MRP_INTERNAL_BIN_SIZE,
        )
        peak_reports.append(
            {
                'histogram_mrp': mrp_values,
                'histogram_peak_sides': _mrp_sides_from_values(peak_center, mrp_values),
            }
        )

    max_report = peak_reports[idx_max]
    if max_report is not None:
        mrp_peak, _ = _recommended_mrp_payload(max_report)
    else:
        mrp_peak = [float('nan')] * 3

    for frac_idx, mrp_s in enumerate(mrp_range):
        mrp_values = []
        peak_width_tmp = []
        for report in peak_reports:
            if report is None:
                mrp_values.append(float('nan'))
                peak_width_tmp.append([float('nan'), float('nan')])
                continue
            recommended_mrp, _ = _recommended_mrp_payload(report)
            mrp_values.append(recommended_mrp[frac_idx])
            peak_sides = report.get('recommended_peak_sides', report.get('histogram_peak_sides'))
            peak_width_tmp.append(peak_sides[frac_idx])
        mrp['MRP(%s)' % mrp_s] = mrp_values
        mrp['peak_sides(%s)' % mrp_s] = peak_width_tmp

    return mrp_peak, mrp


def fast_mrp(calibration_array, x1, x2, bin_size=0.1):
    """Calculate MRP at (0.5, 0.1, 0.01) using Gaussian peak fitting.

    If the requested *bin_size* produces too-sparse histograms (leading to all-NaN
    results), the function automatically retries with progressively larger bin sizes.
    """
    nan3 = [float('nan')] * 3
    data = calibration_array[(calibration_array > x1) & (calibration_array < x2)]
    if len(data) < 10:
        return nan3

    # Build a unique, ordered list of bin sizes to try.
    _candidates = [bin_size, bin_size * 3, bin_size * 10, 0.01, 0.05, 0.1]
    seen = set()
    fallback_sizes = []
    for bs in _candidates:
        rounded = round(bs, 8)
        if rounded not in seen and rounded > 0:
            seen.add(rounded)
            fallback_sizes.append(rounded)

    for attempt_bin in fallback_sizes:
        result = _fast_mrp_core(data, x1, x2, attempt_bin)
        if result is not None and any(np.isfinite(v) for v in result):
            return result
    return nan3


def _fast_mrp_core(data, x1, x2, bin_size):
    """Single-attempt MRP calculation for *fast_mrp*."""
    nan3 = [float('nan')] * 3

    n_bins = max(2, int((x2 - x1) / bin_size))
    y, edges = np.histogram(data, bins=n_bins)
    x = (edges[:-1] + edges[1:]) * 0.5

    try:
        peaks, _ = find_peaks(y, height=0)
    except ValueError:
        return None
    if len(peaks) == 0:
        return None

    prom = peak_prominences(y, peaks)
    idx = np.argmax(prom[0])

    gauss_mrp, gauss_ok = _fit_gaussian_mrp(x, y, peaks[idx])
    if gauss_ok:
        return gauss_mrp

    result = []
    for rel in [0.5, 0.9, 0.99]:
        try:
            pw = peak_widths(y, peaks, rel_height=rel)
            left = np.interp(pw[2][idx], np.arange(len(x)), x)
            right = np.interp(pw[3][idx], np.arange(len(x)), x)
            width = right - left
            result.append(round(float(x[peaks[idx]] / width), 2) if width > 0 else float('nan'))
        except (ValueError, IndexError):
            result.append(float('nan'))
    return result if any(np.isfinite(v) for v in result) else None


def gaussian_mrp_report(
    calibration_array,
    x1,
    x2,
    bin_size=_MRP_INTERNAL_BIN_SIZE,
    peak_center=None,
    _reference_guard=True,
    above_ceiling_strategy='nan',
):
    """Compute a robust high-resolution MRP report for the selected peak window.

    If the requested *bin_size* is too fine to find any peaks, the function
    retries with progressively larger bin sizes before giving up.

    ``above_ceiling_strategy`` is forwarded to ``_recommended_mrp_payload``;
    see that function's docstring. Default 'nan' preserves legacy behavior.
    """
    report = _gaussian_mrp_report_core(
        calibration_array, x1, x2, bin_size=bin_size, peak_center=peak_center,
        _reference_guard=_reference_guard, above_ceiling_strategy=above_ceiling_strategy,
    )
    if report is not None:
        return report

    # Retry with coarser bin sizes when the fine resolution fails
    _candidates = [bin_size * 3, bin_size * 10, _MRP_REFERENCE_BIN_SIZE, 0.05, 0.1]
    seen = {round(bin_size, 8)}
    for candidate in _candidates:
        rounded = round(candidate, 8)
        if rounded in seen or rounded <= 0:
            continue
        seen.add(rounded)
        report = _gaussian_mrp_report_core(
            calibration_array, x1, x2, bin_size=rounded, peak_center=peak_center,
            _reference_guard=_reference_guard, above_ceiling_strategy=above_ceiling_strategy,
        )
        if report is not None:
            return report
    return None


def _gaussian_mrp_report_core(
    calibration_array, x1, x2, bin_size=_MRP_INTERNAL_BIN_SIZE, peak_center=None,
    _reference_guard=True, above_ceiling_strategy='nan',
):
    """Single-attempt MRP report computation (inner implementation)."""
    values = np.asarray(calibration_array, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 10:
        return None

    requested_x1 = float(min(x1, x2))
    requested_x2 = float(max(x1, x2))
    requested_center = float(peak_center) if peak_center is not None else 0.5 * (requested_x1 + requested_x2)
    used_x1, used_x2, window_expanded = _expand_mrp_window(values, requested_x1, requested_x2, bin_size)
    data = values[(values > used_x1) & (values < used_x2)]
    if len(data) < 10:
        return None

    n_bins = max(_MRP_MIN_BINS, int(np.ceil((used_x2 - used_x1) / max(bin_size, 1e-6))))
    edges = np.linspace(used_x1, used_x2, n_bins + 1)
    y, edges = np.histogram(data, bins=edges)
    x = (edges[:-1] + edges[1:]) * 0.5

    try:
        peaks, _ = find_peaks(y, height=0)
    except ValueError:
        return None
    if len(peaks) == 0:
        return None

    peak_idx = _select_peak_index(x, peaks, requested_center)
    if peak_idx is None:
        return None

    gauss_mrp, gauss_ok = _fit_gaussian_mrp(x, y, peaks[peak_idx])
    voigt_mrp, voigt_ok, voigt_fwhm, profile_type = _fit_voigt_mrp(x, y, peaks[peak_idx])
    asym_mrp, asym_ok, asym_fwhm = _fit_asymmetric_mrp(x, y, peaks[peak_idx])

    # scipy.signal.peak_widths walks outward from the apex bin and interpolates
    # the first downward crossing. When the histogram bin is much finer than
    # the peak width (e.g. 0.001 Da bin against a tall, narrow peak), the
    # apex bin towers over its neighbors and the half-prominence crossing
    # lands inside the apex bin, returning a width of one bin or less. That
    # produces an absurdly large histogram-based MRP (e.g. 30,000+) and breaks
    # the cross-check that protects against over-narrow Voigt fits. We
    # therefore re-bin to a coarser comparison histogram whenever the input
    # bin is finer than the threshold below.
    HIST_COMPARISON_MIN_BIN = 0.05
    if bin_size < HIST_COMPARISON_MIN_BIN:
        coarse_bin = HIST_COMPARISON_MIN_BIN
        coarse_n_bins = max(_MRP_MIN_BINS, int(np.ceil((used_x2 - used_x1) / coarse_bin)))
        coarse_edges = np.linspace(used_x1, used_x2, coarse_n_bins + 1)
        coarse_y, coarse_edges = np.histogram(data, bins=coarse_edges)
        coarse_x = (coarse_edges[:-1] + coarse_edges[1:]) * 0.5
        try:
            coarse_peaks, _ = find_peaks(coarse_y, height=0)
        except ValueError:
            coarse_peaks = np.array([], dtype=int)
        coarse_peak_idx = _select_peak_index(coarse_x, coarse_peaks, requested_center) if len(coarse_peaks) else None
        if coarse_peak_idx is not None:
            hist_x = coarse_x
            hist_y = coarse_y
            hist_peaks = coarse_peaks
            hist_peak_idx = coarse_peak_idx
            hist_bin_size = coarse_bin
        else:
            hist_x, hist_y, hist_peaks, hist_peak_idx, hist_bin_size = x, y, peaks, peak_idx, bin_size
    else:
        hist_x, hist_y, hist_peaks, hist_peak_idx, hist_bin_size = x, y, peaks, peak_idx, bin_size

    hist_mrp = []
    histogram_peak_sides = []
    histogram_widths = []
    for rel in [0.5, 0.9, 0.99]:
        try:
            pw = peak_widths(hist_y, hist_peaks, rel_height=rel)
            left = np.interp(pw[2][hist_peak_idx], np.arange(len(hist_x)), hist_x)
            right = np.interp(pw[3][hist_peak_idx], np.arange(len(hist_x)), hist_x)
            width = right - left
            edge_tol = max(hist_bin_size * 2.0, (used_x2 - used_x1) / 400.0)
            if left <= used_x1 + edge_tol or right >= used_x2 - edge_tol:
                hist_mrp.append(float('nan'))
                histogram_peak_sides.append([float('nan'), float('nan')])
                histogram_widths.append(float('nan'))
            else:
                hist_mrp.append(round(float(hist_x[hist_peaks[hist_peak_idx]] / width), 2) if width > 0 else float('nan'))
                histogram_peak_sides.append([float(left), float(right)])
                histogram_widths.append(float(width))
        except (ValueError, IndexError):
            hist_mrp.append(float('nan'))
            histogram_peak_sides.append([float('nan'), float('nan')])
            histogram_widths.append(float('nan'))

    histogram_widths, hist_mrp = _sanitize_tail_widths(
        histogram_widths,
        hist_mrp,
        used_x2 - used_x1,
        max(hist_bin_size * 2.0, (used_x2 - used_x1) / 200.0),
    )
    histogram_peak_sides = [
        [float('nan'), float('nan')] if not np.isfinite(width) else sides
        for width, sides in zip(histogram_widths, histogram_peak_sides)
    ]

    peak_position = float(x[peaks[peak_idx]])
    gaussian_peak_sides = _mrp_sides_from_values(peak_position, gauss_mrp if gauss_ok else [float('nan')] * 3)
    voigt_peak_sides = _mrp_sides_from_values(peak_position, voigt_mrp if voigt_ok else [float('nan')] * 3)
    asymmetric_peak_sides = _mrp_sides_from_values(peak_position, asym_mrp if asym_ok else [float('nan')] * 3)
    recommended_mrp, recommended_label = _recommended_mrp_payload(
        {
            'gaussian_ok': gauss_ok,
            'gaussian_mrp': gauss_mrp if gauss_ok else [float('nan')] * 3,
            'voigt_ok': voigt_ok,
            'voigt_mrp': voigt_mrp if voigt_ok else [float('nan')] * 3,
            'asymmetric_ok': asym_ok,
            'asymmetric_mrp': asym_mrp if asym_ok else [float('nan')] * 3,
            'histogram_mrp': hist_mrp,
            'profile_type': profile_type,
        },
        above_ceiling_strategy=above_ceiling_strategy,
    )
    if recommended_label.startswith('Asymmetric'):
        recommended_peak_sides = asymmetric_peak_sides
    elif recommended_label.startswith('Voigt'):
        recommended_peak_sides = voigt_peak_sides
    elif recommended_label == 'Gaussian':
        recommended_peak_sides = gaussian_peak_sides
    else:
        recommended_peak_sides = histogram_peak_sides

    report_peak_position = peak_position
    robustness_warning = ''
    if _reference_guard and bin_size < _MRP_REFERENCE_BIN_SIZE:
        reference_report = gaussian_mrp_report(
            values,
            requested_x1,
            requested_x2,
            bin_size=_MRP_REFERENCE_BIN_SIZE,
            peak_center=requested_center,
            _reference_guard=False,
        )
        if reference_report is not None:
            highres_width = _mrp_width_from_fwhm(report_peak_position, recommended_mrp)
            reference_width = _mrp_width_from_fwhm(reference_report['peak_position'], reference_report['recommended_mrp'])
            suspicious_highres_width = (
                np.isfinite(highres_width)
                and np.isfinite(reference_width)
                and highres_width < reference_width * _MRP_REFERENCE_MIN_WIDTH_RATIO
            )
            suspicious_highres_mrp = (
                np.isfinite(recommended_mrp[0])
                and np.isfinite(reference_report['recommended_mrp'][0])
                and recommended_mrp[0] > reference_report['recommended_mrp'][0] * _MRP_REFERENCE_GUARD_FACTOR
            )
            if suspicious_highres_width or suspicious_highres_mrp:
                recommended_mrp = reference_report['recommended_mrp']
                recommended_label = f'{reference_report["recommended_label"]} ({_MRP_REFERENCE_BIN_SIZE:g} Da guard)'
                recommended_peak_sides = reference_report['recommended_peak_sides']
                report_peak_position = float(reference_report['peak_position'])
                robustness_warning = (
                    f'High-resolution MRP at {bin_size:g} Da resolved an implausibly narrow sub-peak; '
                    f'using the fixed {_MRP_REFERENCE_BIN_SIZE:g} Da reference histogram for the final MRP.'
                )

    return {
        'gaussian_mrp': gauss_mrp if gauss_ok else [float('nan')] * 3,
        'gaussian_ok': gauss_ok,
        'formatted_gaussian_mrp': [_format_mrp_value(value) for value in (gauss_mrp if gauss_ok else [float('nan')] * 3)],
        'gaussian_peak_sides': gaussian_peak_sides,
        'voigt_mrp': voigt_mrp if voigt_ok else [float('nan')] * 3,
        'voigt_ok': voigt_ok,
        'formatted_voigt_mrp': [_format_mrp_value(value) for value in (voigt_mrp if voigt_ok else [float('nan')] * 3)],
        'voigt_peak_sides': voigt_peak_sides,
        'voigt_fwhm': voigt_fwhm,
        'profile_type': profile_type,
        'asymmetric_mrp': asym_mrp if asym_ok else [float('nan')] * 3,
        'asymmetric_ok': asym_ok,
        'formatted_asymmetric_mrp': [_format_mrp_value(value) for value in (asym_mrp if asym_ok else [float('nan')] * 3)],
        'asymmetric_peak_sides': asymmetric_peak_sides,
        'asymmetric_fwhm': asym_fwhm,
        'histogram_mrp': hist_mrp,
        'formatted_histogram_mrp': [_format_mrp_value(value) for value in hist_mrp],
        'histogram_peak_sides': histogram_peak_sides,
        'recommended_mrp': recommended_mrp,
        'formatted_recommended_mrp': [_format_mrp_value(value) for value in recommended_mrp],
        'recommended_label': recommended_label,
        'recommended_peak_sides': recommended_peak_sides,
        'peak_position': report_peak_position,
        'num_bins': n_bins,
        'num_ions': len(data),
        'bin_size': bin_size,
        'requested_window': [requested_x1, requested_x2],
        'used_window': [used_x1, used_x2],
        'window_expanded': window_expanded,
        'requested_center': requested_center,
        'requested_num_ions': int(np.count_nonzero((values > requested_x1) & (values < requested_x2))),
        'window_warning': (
            f'Selected window was expanded from [{requested_x1:.4f}, {requested_x2:.4f}] '
            f'to [{used_x1:.4f}, {used_x2:.4f}] for stable MRP fitting.'
            if window_expanded
            else ''
        ),
        'robustness_warning': robustness_warning,
    }


def find_peaks_and_widths(plotter, prominence=None, distance=None, percent=50):
    """Find peaks and widths on histogram data and update plotter state."""
    plotter.percent = percent
    rel_percent = 100 - percent
    plotter.prominence = prominence
    plotter.distance = distance
    x_axis = plotter.x_centers if getattr(plotter, 'x_centers', None) is not None else plotter.x[:-1]
    try:
        plotter.peaks, plotter.properties = find_peaks(
            plotter.y, prominence=plotter.prominence, distance=plotter.distance, height=0
        )
        plotter.peak_widths = peak_widths(plotter.y, plotter.peaks, rel_height=(rel_percent / 100), prominence_data=None)
        plotter.prominences = peak_prominences(plotter.y, plotter.peaks, wlen=None)

        x_peaks = x_axis[plotter.peaks]
        y_peaks = plotter.y[plotter.peaks]
        plotter.variables.peak_x = x_peaks
        plotter.variables.peak_y = y_peaks
        index_max_ini = np.argmax(y_peaks)
        plotter.variables.max_peak = x_peaks[index_max_ini]
        plotter.variables.peak_widths = plotter.peak_widths

    except ValueError:
        print('Peak finding failed.')
        plotter.peaks = None
        plotter.properties = None
        plotter.peak_widths = None
        plotter.prominences = None
        plotter.variables.peak_x = None
        plotter.variables.peak_y = None
        plotter.variables.max_peak = None

    return plotter.peaks, plotter.properties, plotter.peak_widths, plotter.prominences


def draw_rectangle(plotter, initial=False):
    """Draw automatic selection rectangle around highest peak.

    Parameters
    ----------
    plotter : AptHistPlotter
        The histogram plotter instance.
    initial : bool, optional
        When *True* a wider minimum selection width is enforced so that the
        **initial** calibration step always operates on a physically meaningful
        number of ions.  For subsequent optimisation iterations this should
        remain *False* so that the precise peak-width measurement from
        ``scipy.signal.peak_widths`` is used as-is.
    """
    index_max_ini = np.argmax(plotter.prominences[0])
    left_idx = int(np.clip(round(plotter.peak_widths[2][index_max_ini]), 0, max(0, len(plotter.x) - 1)))
    right_idx = int(np.clip(round(plotter.peak_widths[3][index_max_ini]), 0, max(0, len(plotter.x) - 1)))
    sel_x1 = float(plotter.x[left_idx])
    sel_x2 = float(plotter.x[right_idx])

    if initial:
        # Enforce a minimum selection width so that the initial calibration
        # always operates on a physically meaningful number of ions.  The
        # minimum scales with the data range so that ToF spectra (range
        # ~1000 ns) get at least ~5 ns instead of the sub-nanosecond windows
        # that scipy peak_widths can produce when many sub-peaks cluster
        # together.
        bin_w = float(plotter.bin_width) if plotter.bin_width else 0.1
        data_min = float(np.min(plotter.mc_tof))
        data_max = float(np.max(plotter.mc_tof))
        data_range = max(data_max - data_min, 1.0)
        min_width = max(bin_w * 20.0, data_range * 0.005, 0.5)
        if sel_x2 - sel_x1 < min_width:
            center = 0.5 * (sel_x1 + sel_x2)
            sel_x1 = max(data_min, center - min_width / 2.0)
            sel_x2 = min(data_max, center + min_width / 2.0)

    plotter.variables.selected_x1 = sel_x1
    plotter.variables.selected_x2 = sel_x2
    plotter.variables.selected_y1 = 0
    plotter.variables.selected_y2 = float(plotter.prominences[0][index_max_ini])

    plotter.rectangle = plt.Rectangle(
        (sel_x1, plotter.variables.selected_y1),
        sel_x2 - sel_x1,
        plotter.variables.selected_y2,
        edgecolor='g',
        facecolor=(0, 1, 0, 0.5),
        linewidth=1,
    )
    plotter.ax.add_patch(plotter.rectangle)


__all__ = [
    'apply_hist_info_legend',
    'calculate_mrp',
    'fast_mrp',
    'gaussian_mrp_report',
    'find_peaks_and_widths',
    'draw_rectangle',
    '_fit_gaussian_mrp',
    '_fit_voigt_mrp',
    '_fit_asymmetric_mrp',
]
