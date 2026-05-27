"""Background fitting helpers for :mod:`mc_plot`."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pybaselines
from pybaselines import Baseline
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def exponential_decay_with_linear_and_dc(x, a, b, c, d):
    """Exponential decay function with linear term and DC offset."""
    return a * np.exp(-b * x) + c * x + d


def find_valley_anchors(x, y, bin_width=None, smooth_sigma_bins=8,
                        peak_prominence_factor=5.0, peak_distance_da=0.15,
                        min_anchor_gap_da=0.4):
    """Return (x_anchors, y_anchors) — local minima between detected peaks.

    Detects peaks in a lightly smoothed histogram, then places one anchor at
    the lowest bin in each inter-peak valley.  Anchors are forced at both
    spectrum edges so a spline through them is fully constrained.
    """
    if bin_width is None:
        bin_width = float(np.median(np.diff(x))) if len(x) > 1 else 0.01
    ys = gaussian_filter1d(y, sigma=smooth_sigma_bins)
    pos = ys[ys > 0]
    bg_median = float(np.median(pos)) if pos.size else 0.0
    peaks, _ = find_peaks(
        ys,
        prominence=peak_prominence_factor * max(bg_median, 1.0),
        distance=max(int(round(peak_distance_da / bin_width)), 1),
    )
    if peaks.size == 0:
        return x[[0, -1]], y[[0, -1]]
    anchor_idx = [0]
    for p_lo, p_hi in zip(peaks[:-1], peaks[1:]):
        if p_hi - p_lo < 3:
            continue
        anchor_idx.append(int(p_lo + np.argmin(ys[p_lo:p_hi])))
    p_last = int(peaks[-1])
    if p_last + 3 < len(x):
        anchor_idx.append(p_last + int(np.argmin(ys[p_last:])))
    anchor_idx.append(len(x) - 1)
    min_gap = max(1, int(round(min_anchor_gap_da / bin_width)))
    pruned = [anchor_idx[0]]
    for i in anchor_idx[1:]:
        if i - pruned[-1] >= min_gap:
            pruned.append(i)
    pruned = np.unique(pruned)
    half = max(2, int(round(0.05 / bin_width)))
    y_anchor = np.array([y[max(0, i - half):i + half + 1].min() for i in pruned])
    x_anchor = x[pruned]
    return x_anchor, y_anchor


def pchip_valley_spline(x, y, bin_width=None, **anchor_kwargs):
    """Monotone PCHIP spline through inter-peak valley minima.

    Tracks a mass-dependent background that rises and falls under peak
    clusters — unlike constant or single-curve methods (BG@4, 1/sqrt(x),
    aspls, fabc).  Returns (background_array, x_anchors, y_anchors).
    """
    xa, ya = find_valley_anchors(x, y, bin_width=bin_width, **anchor_kwargs)
    if xa.size < 2:
        return np.zeros_like(y, dtype=float), xa, ya
    pchip = PchipInterpolator(xa, ya, extrapolate=False)
    bg = pchip(x)
    bg[x < xa[0]] = ya[0]
    bg[x > xa[-1]] = ya[-1]
    bg = np.minimum(np.maximum(bg, 0.0), y)
    return bg, xa, ya


# ---------------------------------------------------------------------------
# Parametric decay backgrounds — high at low Da, decays to a floor at high Da.
# ---------------------------------------------------------------------------
def _decay_inv_x(x, a, b):
    """a / x + b"""
    return a / np.maximum(x, 1e-6) + b


def _decay_inv_sqrt(x, a, b):
    """a / sqrt(x) + b"""
    return a / np.sqrt(np.maximum(x, 1e-6)) + b


def _decay_exp(x, a, b, c):
    """a * exp(-b * x) + c"""
    return a * np.exp(-b * x) + c


_DECAY_FORMS = {
    'inv_x':     (_decay_inv_x,     'a/x + b'),
    'inv_sqrt':  (_decay_inv_sqrt,  'a/sqrt(x) + b'),
    'exp':       (_decay_exp,       'a*exp(-b*x) + c'),
}


def decay_background(x, y, form='inv_x', x_min=0.5, bin_width=None,
                     **anchor_kwargs):
    """Fit a parametric decay model to auto-detected valley anchors.

    The background in atom-probe spectra is typically high at low m/n and
    decays to a floor at high m/n.  This helper picks valley anchors between
    peaks (so peaks do not pull the fit upward), then fits a closed-form
    decay model through them.

    Parameters
    ----------
    x, y : array-like
        Histogram bin centres and counts.
    form : {'inv_x', 'inv_sqrt', 'exp'}
        Which model to fit:
          * ``inv_x``     -> ``a / x + b``       (what most users mean by 1/x decay)
          * ``inv_sqrt``  -> ``a / sqrt(x) + b`` (APT thermal-tail physics)
          * ``exp``       -> ``a * exp(-b*x) + c``
    x_min : float
        Anchors with ``x < x_min`` are excluded from the fit so the
        divergent low-x region does not dominate.
    bin_width : float, optional
        Histogram bin width; auto-detected from ``x`` if omitted.
    **anchor_kwargs
        Forwarded to :func:`find_valley_anchors` (peak detection tuning).

    Returns
    -------
    bg : ndarray
        Fitted background evaluated at each ``x``.
    popt : ndarray
        Fitted parameters of the chosen model.
    formula : str
        Human-readable formula string.
    anchors : tuple of (x_anchors, y_anchors)
        The valley anchors used in the fit.
    """
    if form not in _DECAY_FORMS:
        raise ValueError(f"Unknown decay form {form!r}; choose from {list(_DECAY_FORMS)}")
    fn, formula = _DECAY_FORMS[form]
    xa, ya = find_valley_anchors(x, y, bin_width=bin_width, **anchor_kwargs)
    sel = xa >= x_min
    xa_fit, ya_fit = xa[sel], ya[sel]
    if xa_fit.size < (3 if form == 'exp' else 2):
        raise ValueError(
            f"Not enough valley anchors (n={xa_fit.size}) above x_min={x_min} "
            f"to fit decay form {form!r}."
        )
    # Poisson-style weights so high-count points don't dominate.
    sigma = np.sqrt(np.maximum(ya_fit, 1.0))
    floor = float(np.percentile(ya_fit, 20))
    if form == 'inv_x':
        p0 = [max(ya_fit[0] * xa_fit[0], 1.0), floor]
        bounds = ([0.0, 0.0], [np.inf, np.inf])
    elif form == 'inv_sqrt':
        p0 = [max(ya_fit[0] * np.sqrt(xa_fit[0]), 1.0), floor]
        bounds = ([0.0, 0.0], [np.inf, np.inf])
    else:  # exp
        p0 = [max(ya_fit[0] - floor, 1.0), 0.1, floor]
        bounds = ([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf])
    popt, _ = curve_fit(fn, xa_fit, ya_fit, p0=p0, sigma=sigma,
                        absolute_sigma=False, bounds=bounds, maxfev=20000)
    bg = fn(x, *popt)
    bg = np.minimum(np.maximum(bg, 0.0), y)
    return bg, popt, formula, (xa, ya)


def plot_background(plotter, mode, non_peaks=None, lam=1e6, tol=1e-1, max_iter=100, num_std=3.0, plot=True, patch=True):
    """Plot and quantify histogram background with selected algorithm."""
    if mode == 'pchip_valley':
        x_centers = plotter.bins[:-1]
        bin_width = float(plotter.bins[1] - plotter.bins[0]) if len(plotter.bins) > 1 else 0.01
        fit_2, xa, ya = pchip_valley_spline(x_centers, plotter.y, bin_width=bin_width)
        plotter.background_ppm = round(
            float(np.sum(fit_2)) / len(plotter.mc_tof) * 1e6 / np.max(plotter.mc_tof), 2
        )
        if plot:
            plotter.ax.plot(x_centers, fit_2, color='tab:blue', linewidth=1.6,
                            label='PCHIP valley spline')
            if patch:
                plotter.ax.scatter(xa, ya, s=24, color='black', marker='x',
                                   zorder=10, label='valley anchors')
            handles, labels = plt.gca().get_legend_handles_labels()
            handles.append(plt.Line2D([], [], linestyle='none'))
            labels.append('Noise ppm: ' + str(plotter.background_ppm))
            plt.legend(handles, labels, frameon=False, loc='upper right', fontsize=8)
        plotter.pchip_background = fit_2
        plotter.pchip_anchors = (xa, ya)
        return plotter.mask_f

    if mode in ('decay_inv_x', 'decay_inv_sqrt', 'decay_exp'):
        x_centers = plotter.bins[:-1]
        bin_width = float(plotter.bins[1] - plotter.bins[0]) if len(plotter.bins) > 1 else 0.01
        form = mode.replace('decay_', '')
        fit_2, popt, formula, (xa, ya) = decay_background(
            x_centers, plotter.y, form=form, bin_width=bin_width
        )
        plotter.background_ppm = round(
            float(np.sum(fit_2)) / len(plotter.mc_tof) * 1e6 / np.max(plotter.mc_tof), 2
        )
        plotter.decay_background = fit_2
        plotter.decay_params = popt
        plotter.decay_form = formula
        plotter.pchip_anchors = (xa, ya)
        if plot:
            params_str = ', '.join(f'{p:.4g}' for p in popt)
            plotter.ax.plot(x_centers, fit_2, color='tab:red', linewidth=1.6,
                            label=f'decay fit ({formula})')
            if patch:
                plotter.ax.scatter(xa, ya, s=24, color='black', marker='x',
                                   zorder=10, label='valley anchors')
            handles, labels = plt.gca().get_legend_handles_labels()
            handles.append(plt.Line2D([], [], linestyle='none'))
            labels.append(f'params: ({params_str})')
            handles.append(plt.Line2D([], [], linestyle='none'))
            labels.append('Noise ppm: ' + str(plotter.background_ppm))
            plt.legend(handles, labels, frameon=False, loc='upper right', fontsize=8)
        return plotter.mask_f

    if mode == 'aspls':
        baseline_fitter = Baseline(x_data=plotter.bins[:-1])
        fit_2, params_2 = baseline_fitter.aspls(plotter.y, lam=lam, tol=tol, max_iter=max_iter)

    if mode == 'fabc':
        fit_2, params_2 = pybaselines.classification.fabc(plotter.y, lam=lam, num_std=num_std, pad_kwargs='edges')

    if mode == 'manual@4':
        upperLim = 4.5
        lowerLim = 3.5
        mask = np.logical_and((plotter.x >= lowerLim), (plotter.x <= upperLim))
        bg_4 = np.sum(plotter.y[np.array(mask[:-1])]) / (upperLim - lowerLim)
        plotter.background_ppm = round(bg_4 / len(plotter.mc_tof) * 1e6, 2)
        handles, labels = plt.gca().get_legend_handles_labels()
        handles.append(plt.Line2D([], [], linestyle='none'))
        labels.append('Noise ppm: ' + str(plotter.background_ppm))
        plt.legend(handles, labels, frameon=False, loc='upper left')

    if mode == 'manual@100':
        upperLim = 100.5
        lowerLim = 99.5
        mask = np.logical_and((plotter.x >= lowerLim), (plotter.x <= upperLim))
        bg_100 = np.sum(plotter.y[np.array(mask[:-1])]) / (upperLim - lowerLim)
        plotter.background_ppm = round(bg_100 / len(plotter.mc_tof) * 1e6, 2)
        handles, labels = plt.gca().get_legend_handles_labels()
        handles.append(plt.Line2D([], [], linestyle='none'))
        labels.append('Noise ppm: ' + str(plotter.background_ppm))
        plt.legend(handles, labels, frameon=False, loc='upper left')

    if plot:
        if mode == 'fabc':
            keys = list(params_2.keys())
            if 'mask' in keys:
                mask_2 = params_2['mask']
                noise = 0
                for i in range(len(mask_2)):
                    if mask_2[i]:
                        noise += plotter.y[i]
                handles, labels = plt.gca().get_legend_handles_labels()
                handles.append(plt.Line2D([], [], linestyle='none'))
                plotter.background_ppm = round(noise / len(plotter.mc_tof) * 1e6 / np.max(plotter.mc_tof), 2)
                labels.append('Noise ppm: ' + str(plotter.background_ppm))
                plt.legend(handles, labels, frameon=False, loc='upper left')

                if patch:
                    plotter.ax.plot(plotter.bins[:-1][mask_2], plotter.y[mask_2], 'o', color='orange')[0]
        elif mode == 'aspls':
            effective_heights = []
            for i in range(len(plotter.bins) - 1):
                background_height = fit_2[i]
                bin_height = plotter.y[i]
                effective_height = min(bin_height, background_height)
                effective_heights.append(effective_height)

            effective_heights = np.array(effective_heights)
            handles, labels = plt.gca().get_legend_handles_labels()
            handles.append(plt.Line2D([], [], linestyle='none'))
            plotter.background_ppm = round(np.sum(effective_heights) / len(plotter.mc_tof) * 1e6 / np.max(plotter.mc_tof), 2)
            labels.append('Noise ppm: ' + str(plotter.background_ppm))
            plt.legend(handles, labels, frameon=False, loc='upper left')

    if plot and mode != 'manual@4' and mode != 'manual@100':
        plotter.ax.plot(plotter.bins[:-1], fit_2, label='class')
        ax3 = plotter.ax.twiny()
        ax3.axis('off')
        ax3.plot(fit_2, label='aspls', color='blue')

    return plotter.mask_f


def manual_background_fit(plotter):
    """Interactively fit background curve on existing histogram."""
    if plotter.fig is None or plotter.ax is None:
        raise RuntimeError('No histogram plotted. Please run plot_histogram first.')

    selected_points = []
    point_markers = []

    def onclick(event):
        if event.button == 1 and event.inaxes == plotter.ax:
            x_value, y_value = event.xdata, event.ydata
            selected_points.append((x_value, y_value))
            (marker,) = plotter.ax.plot(x_value, y_value, 'ro')
            point_markers.append(marker)
            plotter.fig.canvas.draw()
        elif event.button == 3:
            plotter.fig.canvas.mpl_disconnect(cid)
            fit_and_plot_exponential_with_linear_and_dc()

    def fit_and_plot_exponential_with_linear_and_dc():
        if len(selected_points) < 2:
            print('At least 2 points are required to fit an exponential decay with a linear term and DC offset.')
            return

        x_points, y_points = zip(*selected_points)
        plotter.popt, _ = curve_fit(exponential_decay_with_linear_and_dc, x_points, y_points, maxfev=10000)
        a, b, c, d = plotter.popt

        x_vals = np.linspace(min(plotter.x), max(plotter.x), 500)
        y_vals = exponential_decay_with_linear_and_dc(x_vals, a, b, c, d)
        plotter.ax.plot(x_vals, y_vals, 'b-')

        for marker in point_markers:
            marker.remove()
        point_markers.clear()

        plotter.fig.canvas.draw()
        calculate_noise(plotter, plot_without_noise=False)

    cid = plotter.fig.canvas.mpl_connect('button_press_event', onclick)
    print('Left-click to select points on the plot. Right-click to fit the exponential decay with linear term and DC offset.')


def calculate_noise(plotter, fig_size=(9, 5), plot_without_noise=False):
    """Subtract fitted background and report noise level."""
    if plotter.popt is None:
        raise RuntimeError('No background fitted. Please fit the background first.')

    a, b, c, d = plotter.popt
    bin_edges = plotter.bins[:]

    effective_heights = []
    for i in range(len(bin_edges) - 1):
        y_left = exponential_decay_with_linear_and_dc(bin_edges[i], a, b, c, d)
        y_right = exponential_decay_with_linear_and_dc(bin_edges[i + 1], a, b, c, d)
        bin_height = plotter.y[i]
        height_under_curve = min(max(y_left, y_right), bin_height)
        effective_heights.append(height_under_curve)

    effective_heights = np.array(effective_heights)
    y_noise_removed = plotter.y - effective_heights

    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(plt.Line2D([], [], linestyle='none'))
    plotter.background_ppm = round(np.sum(effective_heights) / len(plotter.mc_tof) * 1e6 / np.max(plotter.mc_tof), 2)
    labels.append('Noise ppm: ' + str(plotter.background_ppm))
    plt.legend(handles, labels, frameon=False, loc='upper left')

    if plot_without_noise:
        new_fig, new_ax = plt.subplots(figsize=fig_size)
        new_ax.hist(
            plotter.x[:-1],
            plotter.x,
            weights=y_noise_removed,
            alpha=0.9,
            color='slategray',
            edgecolor='k',
            histtype='stepfilled',
        )
        new_ax.set_xlabel('Mass/Charge [Da]' if plotter.ax.get_xlabel() == 'Mass/Charge [Da]' else 'Time of Flight [ns]')
        new_ax.set_ylabel('Event Counts')
        new_ax.set_yscale('log' if plotter.ax.get_yscale() == 'log' else 'linear')
        if plotter.original_x_limits is not None:
            new_ax.set_xlim(plotter.original_x_limits)

        new_ax.legend()
        plt.tight_layout()
        plt.show()


__all__ = [
    'exponential_decay_with_linear_and_dc',
    'find_valley_anchors',
    'pchip_valley_spline',
    'decay_background',
    'plot_background',
    'manual_background_fit',
    'calculate_noise',
]
