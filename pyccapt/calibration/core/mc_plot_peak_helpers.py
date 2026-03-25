"""Peak finding and legend helpers for :mod:`mc_plot`."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_prominences, peak_widths


def _gaussian(x, amp, mu, sigma, bg):
    """Gaussian with constant background."""
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + bg


_FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))


def _fit_gaussian_mrp(x, y, peak_idx):
    """Fit a Gaussian around *peak_idx* and return MRP at 0.5, 0.1, 0.01."""
    nan3 = [float('nan')] * 3

    mu0 = x[peak_idx]
    amp0 = float(y[peak_idx])
    if amp0 <= 0:
        return nan3, False

    hw = min(15, peak_idx, len(x) - 1 - peak_idx)
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


def _voigt(x, amp, mu, sigma, gamma, bg):
    """Pseudo-Voigt approximation."""
    fg = _FWHM_FACTOR * sigma
    fl = 2.0 * gamma
    f5 = fg**5 + 2.69269 * fg**4 * fl + 2.42843 * fg**3 * fl**2 + 4.47163 * fg**2 * fl**3 + 0.07842 * fg * fl**4 + fl**5
    f_v = f5 ** 0.2 if f5 > 0 else max(fg, fl)
    if f_v > 0:
        eta = 1.36603 * (fl / f_v) - 0.47719 * (fl / f_v) ** 2 + 0.11116 * (fl / f_v) ** 3
        eta = np.clip(eta, 0.0, 1.0)
    else:
        eta = 0.0
    gauss = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    lorentz = gamma ** 2 / ((x - mu) ** 2 + gamma ** 2)
    return amp * ((1 - eta) * gauss + eta * lorentz) + bg


def _voigt_fwhm(sigma, gamma):
    """Compute FWHM of a pseudo-Voigt profile."""
    fg = _FWHM_FACTOR * sigma
    fl = 2.0 * gamma
    f5 = fg**5 + 2.69269 * fg**4 * fl + 2.42843 * fg**3 * fl**2 + 4.47163 * fg**2 * fl**3 + 0.07842 * fg * fl**4 + fl**5
    return f5 ** 0.2 if f5 > 0 else max(fg, fl)


def _fit_voigt_mrp(x, y, peak_idx):
    """Fit a pseudo-Voigt around *peak_idx* and return MRP at 0.5, 0.1, 0.01."""
    nan3 = [float('nan')] * 3

    mu0 = x[peak_idx]
    amp0 = float(y[peak_idx])
    if amp0 <= 0:
        return nan3, False, float('nan'), 'unknown'

    hw = min(15, peak_idx, len(x) - 1 - peak_idx)
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
    for frac in [0.5, 0.1, 0.01]:
        x_fine = np.linspace(xw[0], xw[-1], 2000)
        y_fine = _voigt(x_fine, *popt) - popt[4]
        y_max = float(np.max(y_fine))
        if y_max <= 0:
            result.append(float('nan'))
            continue
        threshold = frac * y_max
        above = x_fine[y_fine >= threshold]
        if len(above) < 2:
            result.append(float('nan'))
            continue
        fw = float(above[-1] - above[0])
        result.append(round(float(mu_fit / fw), 2) if fw > 0 else float('nan'))

    return result, True, float(fwhm), profile_type


def apply_hist_info_legend(plotter, label='mc', mrp_all=False, background=None, legend_mode='long', loc='left'):
    """Plot histogram info legend on the provided plotter axis."""
    index_peak_max = np.argmax(plotter.prominences[0])
    if label in ('mc', 'mc_c'):
        g_mrp, g_ok = _fit_gaussian_mrp(plotter.x, plotter.y, plotter.peaks[index_peak_max])
        if g_ok:
            mrp = '{:.2f}'.format(g_mrp[0])
        else:
            mrp = '{:.2f}'.format(
                plotter.x[plotter.peaks][index_peak_max]
                / (plotter.x[round(plotter.peak_widths[3][index_peak_max])] - plotter.x[round(plotter.peak_widths[2][index_peak_max])])
            )
        if mrp_all:
            mrp_list, _ = calculate_mrp(plotter)
        if background is not None:
            if mrp_all:
                if legend_mode == 'long':
                    txt = (
                        'bin width: %s Da\nnum atoms: %.2f$e^6$\nbackG: %s ppm/Da\nMRP(0.5): %s\nMRP(0.1): %s\nMRP(0.01): %s'
                        % (plotter.bin_width, len(plotter.mc_tof) / 1000000, round(plotter.background_ppm), mrp_list[0], mrp_list[1], mrp_list[2])
                    )
                else:
                    txt = 'MRP(0.5): %s\nMRP(0.1): %s\nMRP(0.01): %s' % (mrp_list[0], mrp_list[1], mrp_list[2])
            else:
                if legend_mode == 'long':
                    txt = 'bin width: %s Da\nnum atoms: %.2f$e^6$\nbackG: %s ppm/Da\nFW%d%%M: %s' % (
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
                    txt = 'bin width: %s Da\nnum atoms: %.2f$e^6$\nMRP(0.5): %s\nMRP(0.1): %s\nMRP(0.01): %s' % (
                        plotter.bin_width,
                        len(plotter.mc_tof) / 1000000,
                        mrp_list[0],
                        mrp_list[1],
                        mrp_list[2],
                    )
                else:
                    txt = 'MRP(0.5): %s\nMRP(0.1): %s\nMRP(0.01): %s' % (mrp_list[0], mrp_list[1], mrp_list[2])
            else:
                if legend_mode == 'long':
                    txt = 'bin width: %s Da\nnum atoms: %.2f$e^6$\nMRP(%s): %s' % (
                        plotter.bin_width,
                        len(plotter.mc_tof) / 1000000,
                        plotter.percent / 100,
                        mrp,
                    )
                else:
                    txt = 'MRP(0.5): %s' % mrp

    elif label in ('tof', 'tof_c'):
        g_mrp, g_ok = _fit_gaussian_mrp(plotter.x, plotter.y, plotter.peaks[index_peak_max])
        if g_ok:
            mrp = '{:.2f}'.format(g_mrp[0])
        else:
            mrp = '{:.2f}'.format(
                plotter.x[plotter.peaks[index_peak_max]]
                / (plotter.x[round(plotter.peak_widths[3][index_peak_max])] - plotter.x[round(plotter.peak_widths[2][index_peak_max])])
            )
        if mrp_all:
            mrp_list, _ = calculate_mrp(plotter)
        if background is not None:
            if mrp_all:
                if legend_mode == 'long':
                    txt = (
                        'bin width: %s ns\nnum atoms: %.2f$e^6$\nbackG: %s ppm/ns\nMRP(0.5): %s\nMRP(0.1): %s\nMRP(0.01): %s'
                        % (plotter.bin_width, len(plotter.mc_tof) / 1000000, round(plotter.background_ppm), mrp_list[0], mrp_list[1], mrp_list[2])
                    )
                else:
                    txt = 'MRP(0.5): %s\nMRP(0.1): %s\nMRP(0.01): %s' % (mrp_list[0], mrp_list[1], mrp_list[2])
            else:
                if legend_mode == 'long':
                    txt = 'bin width: %s ns\nnum atoms: %.2f$e^6$\nbackG: %s ppm/ns\nMRP(%s): %s' % (
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
                    txt = 'bin width: %s ns\nnum atoms: %.2f$e^6$\nMRP(0.5): %s\nMRP(0.1): %s\nMRP(0.01): %s' % (
                        plotter.bin_width,
                        len(plotter.mc_tof) / 1000000,
                        mrp_list[0],
                        mrp_list[1],
                        mrp_list[2],
                    )
                else:
                    txt = 'MRP(0.5): %s\nMRP(0.1): %s\nMRP(0.01): %s' % (mrp_list[0], mrp_list[1], mrp_list[2])
            else:
                if legend_mode == 'long':
                    txt = 'bin width: %s ns\nnum atoms: %.2f$e^6$ \nMRP(%s): %s' % (
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
            .01,
            .95,
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
            .98,
            .95,
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
    """Calculate MRP values using Gaussian peak fitting for sub-bin accuracy."""
    mrp_range = [0.5, 0.1, 0.01]
    mrp_peak = []
    mrp = {}

    try:
        peaks_unc, _ = find_peaks(plotter.y, height=0)
        prom_unc = peak_prominences(plotter.y, peaks_unc, wlen=None)
    except ValueError:
        print('Peak finding failed (unconstrained)')
        return mrp_peak, mrp

    if len(peaks_unc) == 0:
        return mrp_peak, mrp
    idx_max_unc = np.argmax(prom_unc[0])

    gauss_mrp, gauss_ok = _fit_gaussian_mrp(plotter.x, plotter.y, peaks_unc[idx_max_unc])
    if gauss_ok:
        mrp_peak = gauss_mrp
    else:
        for mrp_s in mrp_range:
            try:
                pw = peak_widths(plotter.y, peaks_unc, rel_height=1 - mrp_s)
                left = plotter.x[round(pw[2][idx_max_unc])]
                right = plotter.x[round(pw[3][idx_max_unc])]
                denom = right - left
                mrp_peak.append(round(float(plotter.x[peaks_unc[idx_max_unc]] / denom), 2) if denom > 0 else float('nan'))
            except (ValueError, IndexError):
                mrp_peak.append(float('nan'))

    try:
        peaks_c, _ = find_peaks(
            plotter.y,
            prominence=plotter.prominence,
            distance=plotter.distance,
            height=0,
        )
    except ValueError:
        peaks_c = np.array([], dtype=int)

    for mrp_s in mrp_range:
        if len(peaks_c) == 0:
            mrp['MRP(%s)' % mrp_s] = []
            mrp['peak_sides(%s)' % mrp_s] = []
            continue

        mrp_tmp = []
        peak_width_tmp = []
        for i in range(len(peaks_c)):
            g_mrp, g_ok = _fit_gaussian_mrp(plotter.x, plotter.y, peaks_c[i])
            frac_idx = {0.5: 0, 0.1: 1, 0.01: 2}[mrp_s]

            if g_ok:
                mrp_tmp.append(g_mrp[frac_idx])
                mu = plotter.x[peaks_c[i]]
                sigma_est = mu / (g_mrp[0] * _FWHM_FACTOR) if g_mrp[0] > 0 else 0
                fw = 2.0 * sigma_est * np.sqrt(2.0 * np.log(1.0 / mrp_s)) if sigma_est > 0 else 0
                peak_width_tmp.append([mu - fw / 2, mu + fw / 2])
            else:
                try:
                    pw_c = peak_widths(plotter.y, peaks_c, rel_height=1 - mrp_s)
                    left = plotter.x[round(pw_c[2][i])]
                    right = plotter.x[round(pw_c[3][i])]
                    denom = right - left
                    mrp_tmp.append(round(float(plotter.x[peaks_c[i]] / denom), 2) if denom > 0 else float('nan'))
                    peak_width_tmp.append([left, right])
                except (ValueError, IndexError):
                    mrp_tmp.append(float('nan'))
                    peak_width_tmp.append([float('nan'), float('nan')])

        mrp['MRP(%s)' % mrp_s] = mrp_tmp
        mrp['peak_sides(%s)' % mrp_s] = peak_width_tmp

    return mrp_peak, mrp


def fast_mrp(calibration_array, x1, x2, bin_size=0.1):
    """Calculate MRP at (0.5, 0.1, 0.01) using Gaussian peak fitting."""
    nan3 = [float('nan')] * 3
    data = calibration_array[(calibration_array > x1) & (calibration_array < x2)]
    if len(data) < 10:
        return nan3

    n_bins = max(2, int((x2 - x1) / bin_size))
    y, edges = np.histogram(data, bins=n_bins)
    x = (edges[:-1] + edges[1:]) * 0.5

    try:
        peaks, _ = find_peaks(y, height=0)
    except ValueError:
        return nan3
    if len(peaks) == 0:
        return nan3

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
    return result


def gaussian_mrp_report(calibration_array, x1, x2, bin_size=0.01):
    """Compute MRP using Gaussian and Voigt fits with fine bins."""
    data = calibration_array[(calibration_array > x1) & (calibration_array < x2)]
    if len(data) < 10:
        return None

    n_bins = max(10, int((x2 - x1) / bin_size))
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
    voigt_mrp, voigt_ok, voigt_fwhm, profile_type = _fit_voigt_mrp(x, y, peaks[idx])

    hist_mrp = []
    for rel in [0.5, 0.9, 0.99]:
        try:
            pw = peak_widths(y, peaks, rel_height=rel)
            left = np.interp(pw[2][idx], np.arange(len(x)), x)
            right = np.interp(pw[3][idx], np.arange(len(x)), x)
            width = right - left
            hist_mrp.append(round(float(x[peaks[idx]] / width), 2) if width > 0 else float('nan'))
        except (ValueError, IndexError):
            hist_mrp.append(float('nan'))

    return {
        'gaussian_mrp': gauss_mrp if gauss_ok else [float('nan')] * 3,
        'gaussian_ok': gauss_ok,
        'voigt_mrp': voigt_mrp if voigt_ok else [float('nan')] * 3,
        'voigt_ok': voigt_ok,
        'voigt_fwhm': voigt_fwhm,
        'profile_type': profile_type,
        'histogram_mrp': hist_mrp,
        'peak_position': float(x[peaks[idx]]),
        'num_bins': n_bins,
        'num_ions': len(data),
        'bin_size': bin_size,
    }


def find_peaks_and_widths(plotter, prominence=None, distance=None, percent=50):
    """Find peaks and widths on histogram data and update plotter state."""
    plotter.percent = percent
    rel_percent = 100 - percent
    plotter.prominence = prominence
    plotter.distance = distance
    try:
        plotter.peaks, plotter.properties = find_peaks(plotter.y, prominence=plotter.prominence, distance=plotter.distance, height=0)
        plotter.peak_widths = peak_widths(plotter.y, plotter.peaks, rel_height=(rel_percent / 100), prominence_data=None)
        plotter.prominences = peak_prominences(plotter.y, plotter.peaks, wlen=None)

        x_peaks = plotter.x[plotter.peaks]
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


def draw_rectangle(plotter):
    """Draw automatic selection rectangle around highest peak."""
    index_max_ini = np.argmax(plotter.prominences[0])
    plotter.variables.selected_x1 = plotter.x[round(plotter.peak_widths[2][index_max_ini])]
    plotter.variables.selected_x2 = plotter.x[round(plotter.peak_widths[3][index_max_ini])]
    plotter.variables.selected_y1 = 0
    plotter.variables.selected_y2 = plotter.prominences[0][index_max_ini]

    plotter.rectangle = plt.Rectangle(
        (plotter.variables.selected_x1, plotter.variables.selected_y1),
        plotter.variables.selected_x2 - plotter.variables.selected_x1,
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
]
