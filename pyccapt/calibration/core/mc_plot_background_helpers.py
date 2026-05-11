"""Background fitting helpers for :mod:`mc_plot`."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pybaselines
from pybaselines import Baseline
from scipy.optimize import curve_fit


def exponential_decay_with_linear_and_dc(x, a, b, c, d):
    """Exponential decay function with linear term and DC offset."""
    return a * np.exp(-b * x) + c * x + d


def plot_background(plotter, mode, non_peaks=None, lam=1e6, tol=1e-1, max_iter=100, num_std=3.0, plot=True, patch=True):
    """Plot and quantify histogram background with selected algorithm."""
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
    'plot_background',
    'manual_background_fit',
    'calculate_noise',
]
