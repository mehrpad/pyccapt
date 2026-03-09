"""Peak finding and legend helpers for :mod:`mc_plot`."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences, peak_widths


def apply_hist_info_legend(plotter, label='mc', mrp_all=False, background=None, legend_mode='long', loc='left'):
    """Plot histogram info legend on the provided plotter axis."""
    index_peak_max = np.argmax(plotter.prominences[0])
    if label in ('mc', 'mc_c'):
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
    """Calculate MRP values for the configured plotter."""
    mrp_range = [0.5, 0.1, 0.01]
    mrp_peak = []
    mrp = {}
    for mrp_s in mrp_range:
        mrp_r = 1 - mrp_s
        try:
            peaks, _properties = find_peaks(plotter.y, prominence=None, distance=None, height=0)
            peak_width = peak_widths(plotter.y, peaks, rel_height=mrp_r, prominence_data=None)
            prominences = peak_prominences(plotter.y, peaks, wlen=None)
        except ValueError:
            print('Peak finding failed for MRP(%s)' % mrp_r)
            continue

        index_peak_max = np.argmax(prominences[0])
        mrp_tmp = plotter.x[peaks][index_peak_max] / (
            plotter.x[round(peak_width[3][index_peak_max])] - plotter.x[round(peak_width[2][index_peak_max])]
        )
        mrp_peak.append(round(mrp_tmp, 2))

        try:
            peaks, _properties = find_peaks(plotter.y, prominence=plotter.prominence, distance=plotter.distance, height=0)
            peak_width = peak_widths(plotter.y, peaks, rel_height=mrp_r, prominence_data=None)
        except ValueError:
            print('Peak finding failed for MRP(%s)' % mrp_r)
            continue

        mrp_tmp = []
        peak_width_tmp = []
        for i in range(len(peaks)):
            mrp_tmp_2 = plotter.x[peaks][i] / (plotter.x[round(peak_width[3][i])] - plotter.x[round(peak_width[2][i])])
            mrp_tmp.append(round(mrp_tmp_2, 2))
            peak_width_tmp.append([plotter.x[round(peak_width[2][i])], plotter.x[round(peak_width[3][i])]])
        mrp['MRP(%s)' % mrp_s] = mrp_tmp
        mrp['peak_sides(%s)' % mrp_s] = peak_width_tmp

    return mrp_peak, mrp


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
    'find_peaks_and_widths',
    'draw_rectangle',
]
