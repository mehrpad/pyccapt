"""Public histogram workflow API separated from the plotting class module."""

from __future__ import annotations

import numpy as np

from pyccapt.calibration.core.mc_plot import AptHistPlotter
from pyccapt.calibration.core.mc_plot_peak_helpers import _auto_peak_selection, _format_mrp_value, gaussian_mrp_report


def hist_plot(
    variables,
    bin_size,
    log,
    target,
    normalize,
    prominence,
    distance,
    percent,
    selector,
    figname,
    lim,
    peaks_find=True,
    peaks_find_plot=False,
    plot_ranged_peak=False,
    plot_ranged_colors=False,
    mrp_all=False,
    background=None,
    grid=False,
    ranging_mode=False,
    range_sequence=[],
    range_mc=[],
    range_detx=[],
    range_dety=[],
    range_x=[],
    range_y=[],
    range_z=[],
    range_vol=[],
    save_fig=True,
    print_info=True,
    legend_mode="long",
    draw_calib_rect=False,
    figure_size=(9, 5),
    plot_show=True,
    fast_calibration=False,
    fast_histogram=True,
    initial_peak_selection=False,
    compute_mrp=True,
):
    """Plot a mass spectrum or tof spectrum and report MRP statistics."""
    assert target in ["mc", "mc_uc", "tof", "tof_c", "tof_calib", "mc_calib"], "Invalid target"
    assert selector in ["peak", "rect", "range"], "Invalid selector"
    assert legend_mode in ["long", "short"], "Invalid legend mode"

    if target == "mc":
        hist = variables.mc
        label = "mc"
    elif target == "mc_uc":
        hist = variables.mc_uc
        label = "mc"
    elif target == "tof":
        hist = variables.dld_t
        label = "tof"
    elif target == "tof_c":
        hist = variables.dld_t_c
        label = "tof"
    elif target == "tof_calib":
        hist = variables.dld_t_calib
        label = "tof"
    elif target == "mc_calib":
        hist = variables.mc_calib
        label = "mc"

    if selector == "peak":
        variables.peaks_x_selected = []
        variables.peak_widths = []
        variables.peaks_index_list = []

    if range_sequence or range_mc or range_detx or range_dety or range_x or range_y or range_z or range_vol:
        if range_sequence:
            mask_sequence = np.zeros_like(hist, dtype=bool)
            mask_sequence[range_sequence[0] : range_sequence[1]] = True
        else:
            mask_sequence = np.ones_like(hist, dtype=bool)
        if range_detx and range_dety:
            mask_det_x = (variables.dld_x_det < range_detx[1]) & (variables.dld_x_det > range_detx[0])
            mask_det_y = (variables.dld_y_det < range_dety[1]) & (variables.dld_y_det > range_dety[0])
            mask_det = mask_det_x & mask_det_y
        else:
            mask_det = np.ones(len(variables.dld_x_det), dtype=bool)
        if range_mc:
            mask_mc = (variables.mc < range_mc[1]) & (variables.mc > range_mc[0])
        else:
            mask_mc = np.ones(len(variables.mc), dtype=bool)
        if range_x and range_y and range_z:
            mask_x = (variables.x < range_x[1]) & (variables.x > range_x[0])
            mask_y = (variables.y < range_y[1]) & (variables.y > range_y[0])
            mask_z = (variables.z < range_z[1]) & (variables.z > range_z[0])
            mask_3d = mask_x & mask_y & mask_z
        else:
            mask_3d = np.ones(len(variables.x), dtype=bool)
        if range_vol:
            mask_vol = (variables.dld_high_voltage < range_vol[1]) & (variables.dld_high_voltage > range_vol[0])
        else:
            mask_vol = np.ones(len(variables.dld_high_voltage), dtype=bool)
        mask = mask_sequence & mask_det & mask_mc & mask_3d & mask_vol
        print("The number of data sequence:", len(mask_sequence[mask_sequence == True]))
        print("The number of data mc:", len(mask_mc[mask_mc == True]))
        print("The number of data det:", len(mask_det[mask_det == True]))
        print("The number of data 3d:", len(mask_3d[mask_3d == True]))
        print("The number of data after cropping:", len(mask[mask == True]))
    else:
        mask = np.ones(len(variables.mc), dtype=bool)

    hist = hist[mask]

    if fast_calibration:
        hist = np.random.choice(hist, int(len(hist) * 0.1), replace=False)

    steps = "bar" if (plot_ranged_peak or plot_ranged_colors) else "stepfilled"
    use_fast = fast_histogram and steps != "bar"

    mc_hist = AptHistPlotter(hist[hist < lim], variables)
    y_values, x_values = mc_hist.plot_histogram(
        bin_width=bin_size,
        normalize=normalize,
        label=label,
        steps=steps,
        log=log,
        grid=grid,
        fig_size=figure_size,
        plot_show=plot_show,
        fast=use_fast,
    )

    variables.AptHistPlotter = mc_hist

    if ranging_mode:
        mc_hist.plot_peaks(range_data=None, mode="range")
        peaks = None
        peak_widths = None
        prominences = None

    elif not normalize and peaks_find and not plot_ranged_peak and not plot_ranged_colors:
        peaks, _properties, peak_widths, prominences = mc_hist.find_peaks_and_widths(
            prominence=prominence,
            distance=distance,
            percent=percent,
        )
        if draw_calib_rect:
            mc_hist.draw_rectangle(initial=initial_peak_selection)
        if peaks_find_plot:
            mc_hist.plot_peaks()
        mc_hist.plot_hist_info_legend(label=label, mrp_all=mrp_all, background=None, legend_mode=legend_mode, loc="right")

    elif plot_ranged_colors and not plot_ranged_peak:
        mc_hist.plot_range(range_data=variables.range_data, legend=True, legend_loc="upper right")
        mc_hist.adjust_labels()
        peaks = None
        peak_widths = None
        prominences = None

    elif plot_ranged_peak and not plot_ranged_colors:
        mc_hist.plot_peaks(range_data=variables.range_data, mode="peaks")
        mc_hist.adjust_labels()
        peaks, _properties, peak_widths, prominences = mc_hist.find_peaks_and_widths(
            prominence=prominence,
            distance=distance,
            percent=percent,
        )
        mc_hist.plot_hist_info_legend(label=label, mrp_all=mrp_all, background=None, legend_mode=legend_mode, loc="right")

    elif plot_ranged_colors and plot_ranged_peak:
        print("Both ranged colors and ranged peak labels were requested. Using ranged colors overlay.")
        mc_hist.plot_range(range_data=variables.range_data, legend=True, legend_loc="upper right")
        mc_hist.adjust_labels()
        peaks = None
        peak_widths = None
        prominences = None

    else:
        peaks = None
        peak_widths = None
        prominences = None

    mc_hist.selector(selector=selector)

    if save_fig:
        mc_hist.save_fig(label=label, fig_name=figname)

    mrp_list = []
    mrp_list_all_peak = {}
    if compute_mrp:
        mrp_list, mrp_list_all_peak = mc_hist.mrp_calculation()

    if background is not None:
        if background in ["aspls", "fabc", "manual@4", "manual@100"]:
            mc_hist.plot_background(mode=background)
        elif background == "user":
            mc_hist.manual_background_fit()

    if peaks is not None and print_info and compute_mrp:
        x_axis = mc_hist.x_centers if getattr(mc_hist, 'x_centers', None) is not None else x_values[:-1]
        auto_selection = _auto_peak_selection(mc_hist)
        if auto_selection is not None:
            peak_report = auto_selection["report"]
        else:
            peak_report = None
        report_label = peak_report["recommended_label"] if peak_report is not None else 'Unavailable'
        report_bin = peak_report["bin_size"] if peak_report is not None else 0.001
        print(f"MRP model: {report_label}")
        print(f"MRP bin size used: {report_bin}")
        print(
            "Mass resolving power for the auto-selected peak (MRP --> m/m_2-m_1):",
            _format_mrp_value(mrp_list[0] if mrp_list else float('nan')),
        )
        print("-----------------------------------------------------------------------------")
        percent_key = percent / 100.0
        for i in range(len(peaks)):
            if not mrp_all:
                peak_mrp_values = mrp_list_all_peak.get(f"MRP({percent_key})", [])
                peak_mrp_value = peak_mrp_values[i] if i < len(peak_mrp_values) else float('nan')
                print(
                    "Peaks ",
                    i + 1,
                    "is at location and height: ({:.2f}, {:.2f})".format(x_axis[int(peaks[i])], prominences[0][i]),
                    "peak_x window sides ({:.0f} percent-maximum) are: ({:.2f}, {:.2f})".format(
                        percent,
                        np.interp(peak_widths[2][i], np.arange(len(x_axis)), x_axis),
                        np.interp(peak_widths[3][i], np.arange(len(x_axis)), x_axis),
                    ),
                    "-> MRP: %s" % _format_mrp_value(peak_mrp_value),
                )
            if mrp_all:
                for percent_value in [0.5, 0.1, 0.01]:
                    print(
                        "Peaks ",
                        i + 1,
                        "is at location and height: ({:.2f}, {:.2f})".format(x_axis[int(peaks[i])], prominences[0][i]),
                        "peak_x window sides ({:.0f} percent-maximum) are: ({:.2f}, {:.2f})".format(
                            percent_value * 100,
                            mrp_list_all_peak[f"peak_sides({percent_value})"][i][0],
                            mrp_list_all_peak[f"peak_sides({percent_value})"][i][1],
                        ),
                        "-> MRP: %s" % _format_mrp_value(mrp_list_all_peak[f"MRP({percent_value})"][i]),
                    )

            print("------------------------------")

    return mrp_list




