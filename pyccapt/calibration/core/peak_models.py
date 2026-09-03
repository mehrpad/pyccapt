"""Numerical peak-shape models separated from plotting/UI state."""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf as _erf

_MRP_MIN_BINS = 60
_MRP_EXPANSION_FACTOR = 3.0

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
    # ``np.where`` evaluates both branches and used to overflow while fitting
    # points on the rising side, even though those values were discarded.
    decay = np.ones_like(dx, dtype=float)
    trailing = dx > 0.0
    decay[trailing] = np.exp(-dx[trailing] / tail)
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
