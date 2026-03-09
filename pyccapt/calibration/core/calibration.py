from copy import copy
from itertools import product
import concurrent.futures

import fast_histogram
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, rcParams
from matplotlib.tri import Triangulation
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from pyccapt.calibration.core.exceptions import CalibrationInputError
from pyccapt.calibration.core.validation import (
    BOWL_FIT_MODES,
    BOWL_SAMPLE_METHODS,
    SAMPLE_METHODS,
    VOLTAGE_MODES,
    ensure_choice,
    ensure_matching_lengths,
    ensure_non_empty_array,
    ensure_positive,
    normalize_voltage_model,
)
from pyccapt.calibration.path_utils import save_figure
from pyccapt.calibration.core.correction_models import (
    _predict_bowl_model,
    _predict_voltage_model,
    bowl_corr,
    hybrid_calibration_model,
    robust_fit,
    robust_voltage_fit,
    voltage_corr,
)
from pyccapt.calibration.core.diagnostics import (
    initial_calibration,
    plot_fdm,
    plot_selected_statistic,
)


def _extract_peak_mask_and_values(variables, calibration_mode):
    """Return the selected peak mask and values for the requested calibration mode."""
    mask = variables.build_calibration_mask(calibration_mode)
    values = variables.get_calibration_array(calibration_mode)[mask]
    return mask, values


def _resolve_sample_size(sample_size, population_size):
    """Resolve a valid sample size, auto-scaling down if needed."""
    if population_size <= 0:
        raise CalibrationInputError("No ions available for calibration")
    requested = int(ensure_positive(sample_size, field_name="sample_size"))
    if requested > population_size:
        return max(1, population_size // 10)
    return requested


def _resolve_peak_location(values, method, bin_size, fast_calibration=False):
    """Resolve the reference peak location by histogram/mean/median."""
    ensure_choice(method, field_name="maximum_cal_method", allowed=SAMPLE_METHODS)
    data = ensure_non_empty_array(values, field_name="peak_values")
    if fast_calibration and data.size > 10:
        data = np.random.choice(data, int(data.size * 0.1), replace=False)

    if method == "mean":
        return float(np.mean(data))
    if method == "median":
        return float(np.median(data))

    bins = np.linspace(np.min(data), np.max(data), round(np.max(data) / bin_size))
    hist = fast_histogram.histogram1d(
        data,
        bins=round(np.max(data) / bin_size) - 1,
        range=(np.min(data), np.max(data)),
    )
    peaks, properties = find_peaks(hist, height=0)
    if len(peaks) == 0:
        return float(np.mean(data))
    index_peak_max_ini = np.argmax(properties["peak_heights"])
    return float(bins[peaks[index_peak_max_ini]])


def voltage_correction(dld_highVoltage_peak, dld_t_peak, variables, maximum_location, index_fig, figname, sample_size,
                       mode, calibration_mode, sample_range_max, bin_size, plot=True, save=False,
                       fig_size=(5, 5), model='curve_fit'):
    """
    Performs voltage correction and plots the graph based on the passed arguments.

    Parameters:
    - dld_highVoltage_peak (array): Array of high voltage peaks.
    - dld_t_peak (array): Array of t peaks.
    - maximum_location (float): Maximum location value.
    - index_fig (string): Index of the saved plot.
    - figname (string): Name of the saved plot image.
    - sample_size (string): Sample size.
    - mode (string): Mode ('ion_seq'/'voltage').
    - calibration_mode (string): Type of calibration mode (tof/mc).
    - sample_range_max (string): Type of peak_x mode (histogram/mean/median).
    - outlier_remove (bool): Indicates whether to remove outliers. Default is True.
    - plot (bool): Indicates whether to plot the graph. Default is True.
    - save (bool): Indicates whether to save the plot. Default is False.
    - fig_size (tuple): Figure size in inches. Default is (7, 5).
    - model (string): Type of model ('curve_fit'/'robust_fit'). Default is 'curve_fit'.
    - bin_size (float): Size of the bin.

    Returns:
    - fitresult (array): Corrected voltage array.

    """
    model = normalize_voltage_model(model)
    ensure_choice(mode, field_name="mode", allowed=VOLTAGE_MODES)
    ensure_choice(calibration_mode, field_name="calibration_mode", allowed=["tof", "mc"])
    ensure_choice(sample_range_max, field_name="sample_range_max", allowed=SAMPLE_METHODS)
    bin_size = ensure_positive(bin_size, field_name="bin_size")
    sample_size = int(ensure_positive(sample_size, field_name="sample_size"))

    dld_highVoltage_peak = ensure_non_empty_array(dld_highVoltage_peak, field_name="dld_highVoltage_peak")
    dld_t_peak = ensure_non_empty_array(dld_t_peak, field_name="dld_t_peak")
    ensure_matching_lengths(
        dld_highVoltage_peak,
        dld_t_peak,
        field_names=("dld_highVoltage_peak", "dld_t_peak"),
    )

    dld_t_peak_list = []
    high_voltage_mean_list = []
    if mode == 'ion_seq':
        for i in range(int(len(dld_highVoltage_peak) / sample_size) + 1):
            dld_highVoltage_peak_selected = dld_highVoltage_peak[i * sample_size:(i + 1) * sample_size]
            dld_t_peak_selected = dld_t_peak[i * sample_size:(i + 1) * sample_size]
            if sample_range_max == 'histogram':
                try:
                    bins = np.linspace(np.min(dld_t_peak_selected), np.max(dld_t_peak_selected),
                                       round(np.max(dld_t_peak_selected) / bin_size))
                    y, x = np.histogram(dld_t_peak_selected, bins=bins)
                    peaks, properties = find_peaks(y, height=0)
                    index_peak_max_ini = np.argmax(properties['peak_heights'])
                    max_peak = peaks[index_peak_max_ini]
                    dld_t_peak_list.append(x[max_peak] / maximum_location)
                    # dld_t_peak_list.append(maximum_location / x[max_peak])
                    mask_v = np.logical_and((dld_t_peak_selected >= x[max_peak] - bin_size)
                                            , (dld_t_peak_selected <= x[max_peak] + bin_size))
                    if len(dld_highVoltage_peak_selected[mask_v]) == 0:
                        mask_v = np.logical_and((dld_t_peak_selected >= x[max_peak] - 2 * bin_size)
                                                , (dld_t_peak_selected <= x[max_peak] + 2 * bin_size))
                        if len(dld_highVoltage_peak_selected[mask_v]) == 0:
                            print('length of mask voltage', len(dld_highVoltage_peak_selected[mask_v]))
                            mask_v = np.logical_and((dld_t_peak_selected >= x[max_peak] - 4 * bin_size)
                                                    , (dld_t_peak_selected <= x[max_peak] + 4 * bin_size))
                            print('length of mask voltage after increase the window',
                                  len(dld_highVoltage_peak_selected[mask_v]))
                    high_voltage_mean_list.append(np.mean(dld_highVoltage_peak_selected[mask_v]))
                except ValueError:
                    # print('cannot find the maximum')
                    dld_t_mean = np.mean(dld_t_peak_selected)
                    dld_t_peak_list.append(dld_t_mean / maximum_location)
                    # dld_t_peak_list.append(maximum_location / dld_t_mean)
                    high_voltage_mean = np.mean(dld_highVoltage_peak_selected)
                    high_voltage_mean_list.append(high_voltage_mean)
            elif sample_range_max == 'mean':
                dld_t_mean = np.mean(dld_t_peak_selected)
                dld_t_peak_list.append(dld_t_mean / maximum_location)
                # dld_t_peak_list.append(maximum_location / dld_t_mean)
                high_voltage_mean = np.mean(dld_highVoltage_peak_selected)
                high_voltage_mean_list.append(high_voltage_mean)
            elif sample_range_max == 'median':
                dld_t_mean = np.median(dld_t_peak_selected)
                dld_t_peak_list.append(dld_t_mean / maximum_location)
                high_voltage_mean = np.median(dld_highVoltage_peak_selected)
                high_voltage_mean_list.append(high_voltage_mean)

    elif mode == 'voltage':
        for i in range(int((np.max(dld_highVoltage_peak) - np.min(dld_highVoltage_peak)) / sample_size) + 1):
            mask = np.logical_and((dld_highVoltage_peak >= (np.min(dld_highVoltage_peak) + (i) * sample_size)),
                                  (dld_highVoltage_peak < (np.min(dld_highVoltage_peak) + (i + 1) * sample_size)))
            dld_highVoltage_peak_selected = dld_highVoltage_peak[mask]
            dld_t_peak_selected = dld_t_peak[mask]

            if sample_range_max == 'histogram':
                try:
                    bins = np.linspace(np.min(dld_t_peak_selected), np.max(dld_t_peak_selected),
                                       round(np.max(dld_t_peak_selected) / bin_size))
                    y, x = np.histogram(dld_t_peak_selected, bins=bins)
                    peaks, properties = find_peaks(y, height=0)
                    index_peak_max_ini = np.argmax(properties['peak_heights'])
                    max_peak = peaks[index_peak_max_ini]
                    dld_t_peak_list.append(x[max_peak] / maximum_location)
                    # dld_t_peak_list.append(maximum_location / x[max_peak])
                    mask_v = np.logical_and((dld_t_peak_selected >= x[max_peak] - bin_size)
                                            , (dld_t_peak_selected <= x[max_peak] + bin_size))
                    high_voltage_mean_list.append(np.mean(dld_highVoltage_peak_selected[mask_v]))
                except ValueError:
                    dld_t_mean = np.mean(dld_t_peak_selected)
                    dld_t_peak_list.append(dld_t_mean / maximum_location)
                    high_voltage_mean = np.mean(dld_highVoltage_peak_selected)
                    high_voltage_mean_list.append(high_voltage_mean)
            elif sample_range_max == 'mean':
                dld_t_mean = np.mean(dld_t_peak_selected)
                dld_t_peak_list.append(dld_t_mean / maximum_location)
                # dld_t_peak_list.append(maximum_location / dld_t_mean)
                high_voltage_mean = np.mean(dld_highVoltage_peak_selected)
                high_voltage_mean_list.append(high_voltage_mean)
            elif sample_range_max == 'median':
                dld_t_mean = np.median(dld_t_peak_selected)
                dld_t_peak_list.append(dld_t_mean / maximum_location)
                # dld_t_peak_list.append(maximum_location / dld_t_mean)
                high_voltage_mean = np.median(dld_highVoltage_peak_selected)
                high_voltage_mean_list.append(high_voltage_mean)


    if model == 'curve_fit':
        fitresult, _ = curve_fit(voltage_corr, np.array(high_voltage_mean_list), np.array(dld_t_peak_list))
    elif model == 'robust_fit':
        fitresult = robust_voltage_fit(np.array(high_voltage_mean_list), np.array(dld_t_peak_list))

    if plot or save:
        fig1, ax1 = plt.subplots(figsize=fig_size, constrained_layout=True)
        if calibration_mode == 'tof':
            ax1.set_ylabel("Time of Flight (ns)", fontsize=10)
            label = 't'
        elif calibration_mode == 'mc':
            ax1.set_ylabel("mc (Da)", fontsize=10)
            label = 'mc'

        x = plt.scatter(np.array(high_voltage_mean_list) / 1000, np.array(dld_t_peak_list) * maximum_location,
                        color="forestgreen", label=r"$%s_{wp}$" % label, s=5)
        # x = plt.scatter(np.array(high_voltage_mean_list) / 1000, maximum_location / np.array(dld_t_peak_list),
        #                 color="forestgreen", label=r"$%s_{wp}$" % label, s=5)
        ax1.set_xlabel("Voltage (kV)", fontsize=10)
        plt.grid(alpha=0.3, linestyle='-.', linewidth=0.4)

        ax2 = ax1.twinx()
        f_v = _predict_voltage_model(model, fitresult, np.array(high_voltage_mean_list))
        y = ax2.plot(np.array(high_voltage_mean_list) / 1000, np.sqrt(f_v), color='r', label=r"$C_V$")
        # y = ax2.plot(np.array(high_voltage_mean_list) / 1000, f_v, color='r', label=r"$C_V$")
        ax2.set_ylabel(r"$C_V$", color="red", fontsize=10)  # Get the current axis
        ax2.tick_params(axis='y', colors='red')  # Change color and thickness of tick labels on y-axis
        ax2.spines['right'].set_color('red')  # Change color of right border
        plt.legend(handles=[x, y[0]], loc='lower left', markerscale=5., prop={'size': 10})

        if save:
            # Enable rendering for text elements
            rcParams['svg.fonttype'] = 'none'
            save_figure(
                fig1,
                directory=variables.result_path,
                stem=f"vol_corr_{figname}_{index_fig}",
                formats=("svg", "png"),
                dpi=600,
            )

        if plot:
            plt.show()

    return fitresult


def voltage_corr_main(dld_highVoltage, variables, sample_size, mode, calibration_mode, index_fig, plot, save,
                      maximum_cal_method='mean', maximum_sample_method='mean', fig_size=(5, 5), fast_calibration=False,
                      bin_size=0.01, model='curve_fit', peak_maximum=0, calibration_apply=True):
    """
    Perform voltage correction on the given data.

    Args:
        dld_highVoltage (numpy.ndarray): Array of high voltages.
        sample_size (int): Size of the sample.
        mode (str): Mode of the correction.
        calibration_mode (str): Calibration mode ('tof' or 'mc').
        index_fig (int): Index of the figure.
        plot (bool): Whether to plot the results.
        save (bool): Whether to save the plots.
        noise_remove (bool, optional): Whether to remove noise. Defaults to True.
        maximum_cal_method (str, optional): Maximum calculation method ('mean', 'histogram', 'median').
        maximum_sample_method (str, optional): Sample range maximum ('mean', 'histogram', 'median').
        fig_size (tuple, optional): Size of the figure. Defaults to (5, 5).
        fast_calibration (bool, optional): Whether to perform fast calibration. Defaults to False.
        bin_size (float, optional): Size of the bin. Defaults to 0.01.
    """
    model = normalize_voltage_model(model)
    ensure_choice(mode, field_name="mode", allowed=VOLTAGE_MODES)
    ensure_choice(calibration_mode, field_name="calibration_mode", allowed=["tof", "mc"])
    ensure_choice(maximum_cal_method, field_name="maximum_cal_method", allowed=SAMPLE_METHODS)
    ensure_choice(maximum_sample_method, field_name="maximum_sample_method", allowed=SAMPLE_METHODS)
    bin_size = ensure_positive(bin_size, field_name="bin_size")
    sample_size = int(ensure_positive(sample_size, field_name="sample_size"))

    all_calibration_values = variables.get_calibration_array(calibration_mode)
    ensure_matching_lengths(
        dld_highVoltage,
        all_calibration_values,
        field_names=("dld_highVoltage", f"{calibration_mode}_calibration_values"),
    )

    print('The left and right side of the main peak is:', variables.selected_x1, variables.selected_x2)
    mask_temporal, dld_peak_b = _extract_peak_mask_and_values(variables, calibration_mode)
    dld_highVoltage_peak_v = np.asarray(dld_highVoltage)[mask_temporal]

    print('The number of ions is:', len(dld_highVoltage_peak_v))
    sample_size = _resolve_sample_size(sample_size, len(dld_highVoltage_peak_v))
    print('The number of samples is:', int(len(dld_highVoltage_peak_v) / sample_size))

    if peak_maximum == 0:
        maximum_location = _resolve_peak_location(
            dld_peak_b,
            method=maximum_cal_method,
            bin_size=bin_size,
            fast_calibration=fast_calibration,
        )
    else:
        maximum_location = float(peak_maximum)

    print('The maximum/mean/median of histogram is located at:', maximum_location)
    print('The high voltage ranges are:', np.min(dld_highVoltage_peak_v), np.max(dld_highVoltage_peak_v))
    mean_before = np.mean(dld_peak_b)
    print('The mean of tof/mc  before voltage calibration is:', mean_before)
    fitresult = voltage_correction(dld_highVoltage_peak_v, dld_peak_b, variables,
                                   maximum_location, index_fig=index_fig,
                                   figname='voltage_corr',
                                   sample_size=sample_size, mode=mode, calibration_mode=calibration_mode,
                                   sample_range_max=maximum_sample_method, bin_size=bin_size,
                                   plot=plot, save=save, fig_size=fig_size, model=model)

    calibration_mc_tof = np.copy(variables.dld_t_calib) if calibration_mode == 'tof' else np.copy(variables.mc_calib)
    print('The fit result is:', fitresult)
    mask_fv = np.ones_like(dld_highVoltage, dtype=bool)

    f_v = _predict_voltage_model(model, fitresult, np.asarray(dld_highVoltage)[mask_fv])

    f_v = np.sqrt(f_v)
    print("Maximum value of f_v:", np.max(f_v))
    print("Minimum value of f_v:", np.min(f_v))
    calibration_mc_tof[mask_fv] = calibration_mc_tof[mask_fv] / f_v

    if plot or save:
        # Plot how correction factor for selected peak_x
        fig1, ax1 = plt.subplots(figsize=fig_size, constrained_layout=True)
        if len(dld_highVoltage_peak_v) > 1000:
            mask = np.random.randint(0, len(dld_highVoltage_peak_v), 1000)
        else:
            mask = np.arange(len(dld_highVoltage_peak_v))
        x = plt.scatter(dld_highVoltage_peak_v[mask] / 1000, dld_peak_b[mask], color="blue", label=r"$t$", s=1)

        if calibration_mode == 'tof':
            ax1.set_ylabel("Time of Flight (ns)", fontsize=10)
        elif calibration_mode == 'mc':
            ax1.set_ylabel("mc (Da)", fontsize=10)
        ax1.set_xlabel("Voltage (V)", fontsize=10)
        plt.grid(alpha=0.3, linestyle='-.', linewidth=0.4)

        # Plot high voltage curve
        ax2 = ax1.twinx()
        f_v_plot = _predict_voltage_model(model, fitresult, dld_highVoltage_peak_v)

        y = ax2.plot(dld_highVoltage_peak_v / 1000, 1 / np.sqrt(f_v_plot), color='r', label=r"$C_{V}^{-1}$")
        # y = ax2.plot(dld_highVoltage_peak_v / 1000, f_v_plot, color='r', label=r"$C_{V}^{-1}$")
        ax2.set_ylabel(r"$C_{V}^{-1}$", color="red", fontsize=10)
        ax2.tick_params(axis='y', colors='red')  # Change color and thickness of tick labels on y-axis
        ax2.spines['right'].set_color('red')  # Change color of right border
        plt.legend(handles=[x, y[0]], loc='upper left', markerscale=5., prop={'size': 10})

        if save:
            # Enable rendering for text elements
            rcParams['svg.fonttype'] = 'none'
            save_figure(
                fig1,
                directory=variables.result_path,
                stem=f"vol_corr_{index_fig}",
                formats=("svg", "png"),
                dpi=600,
            )
        plt.show()

        # Plot corrected tof/mc vs. uncalibrated tof/mc
        fig1, ax1 = plt.subplots(figsize=fig_size, constrained_layout=True)
        x = plt.scatter(dld_highVoltage_peak_v[mask] / 1000, dld_peak_b[mask], color="blue", label='t', s=1)
        if calibration_mode == 'tof':
            ax1.set_ylabel("Time of Flight (ns)", fontsize=10)
        elif calibration_mode == 'mc':
            ax1.set_ylabel("mc (Da)", fontsize=10)
        ax1.set_xlabel("Voltage (kV)", fontsize=10)
        plt.grid(alpha=0.3, linestyle='-.', linewidth=0.4)

        dld_t_plot = dld_peak_b * (1 / np.sqrt(f_v_plot))
        # dld_t_plot = dld_peak_b * f_v_plot

        y = plt.scatter(dld_highVoltage_peak_v[mask] / 1000, dld_t_plot[mask], color="red", label=r"$t_{C_{V}}$",
                        s=1)

        plt.legend(handles=[x, y], loc='upper right', markerscale=5., prop={'size': 10})

        if save:
            # Enable rendering for text elements
            rcParams['svg.fonttype'] = 'none'
            save_figure(
                fig1,
                directory=variables.result_path,
                stem=f"peak_tof_V_corr_{index_fig}",
                formats=("svg", "png"),
                dpi=600,
            )
        if plot:
            plt.show()
    mean_after = np.mean(calibration_mc_tof[mask_temporal])
    print('The mean of tof/mc  after voltage calibration is:', mean_after)
    print('The difference between the mean of tof/mc before and after voltage calibration is:',
          mean_after - mean_before)
    if calibration_apply:
        if calibration_mode == 'tof':
            variables.dld_t_calib = calibration_mc_tof
        elif calibration_mode == 'mc':
            variables.mc_calib = calibration_mc_tof
    return f_v


def compute_sample(i, j, d, dld_x_bowl, dld_y_bowl, dld_t_bowl, maximum_location, sample_range_max, bin_size):
    """
    Compute the sample for the given data.

    Args:
        i (int): Index i.
        j (int): Index j.
        d (int): Sample size.
        dld_x_bowl (numpy.ndarray): X coordinates of the data points.
        dld_y_bowl (numpy.ndarray): Y coordinates of the data points.
        dld_t_bowl (numpy.ndarray): Time values of the data points.
        maximum_location (float): Maximum location for normalization.
        sample_range_max (str): Sample range maximum ('mean' or 'histogram').
        bin_size (float): Size of the bin.

    Returns:
        x_sample (float): X sample value.
        y_sample (float): Y sample value.
        dld_t_peak (float): Time peak value.
    """
    mask_x = np.logical_and((dld_x_bowl < j + d), (dld_x_bowl > j))
    mask_y = np.logical_and((dld_y_bowl < i + d), (dld_y_bowl > i))
    mask = np.logical_and(mask_x, mask_y)

    if len(mask[mask]) > 0:
        x_y_selected = np.vstack((dld_x_bowl[mask], dld_y_bowl[mask])).T
        x_sample = np.median(x_y_selected[:, 0])
        y_sample = np.median(x_y_selected[:, 1])

        if sample_range_max == 'mean':
            dld_t_peak = np.mean(dld_t_bowl[mask]) / maximum_location
            # dld_t_peak = maximum_location / np.mean(dld_t_bowl[mask])
        elif sample_range_max == 'histogram':
            try:
                dld_t_bowl_selected = dld_t_bowl[mask]
                # if len(dld_t_bowl_selected) > 2000000:
                #     dld_t_bowl_selected = np.random.choice(dld_t_bowl_selected, 2000000, replace=False)

                bins = np.linspace(np.min(dld_t_bowl_selected), np.max(dld_t_bowl_selected),
                                   round(np.max(dld_t_bowl_selected) / bin_size))

                y_hist = fast_histogram.histogram1d(dld_t_bowl_selected,
                                                    bins=round(np.max(dld_t_bowl_selected) / bin_size) - 1,
                                                    range=(np.min(dld_t_bowl_selected), np.max(dld_t_bowl_selected)))
                peaks, properties = find_peaks(y_hist, height=0)

                if len(peaks) > 0:
                    index_peak_max_ini = np.argmax(properties['peak_heights'])
                    max_peak = peaks[index_peak_max_ini]
                    dld_t_peak = bins[max_peak] / maximum_location
                    # dld_t_peak = maximum_location / bins[max_peak]
                else:
                    dld_t_peak = np.mean(dld_t_bowl[mask]) / maximum_location
                    # dld_t_peak = maximum_location / np.mean(dld_t_bowl[mask])
            except ValueError:
                print('cannot find the maximum for i, j:', i, j)
                dld_t_peak = np.mean(dld_t_bowl[mask]) / maximum_location
                # dld_t_peak = maximum_location / np.mean(dld_t_bowl[mask])

        return x_sample, y_sample, dld_t_peak
    return None, None, None  # Return None if no samples found


def bowl_correction(dld_x_bowl, dld_y_bowl, dld_t_bowl, variables, det_diam, maximum_location, sample_range_max,
                    sample_size, calibration_mode, fit_mode, index_fig, plot, save, fig_size=(7, 5), bin_size=0.01):
    """
    Perform bowl correction on the input data.

    Args:
        dld_x_bowl (numpy.ndarray): X coordinates of the data points.
        dld_y_bowl (numpy.ndarray): Y coordinates of the data points.
        dld_t_bowl (numpy.ndarray): Time values of the data points.
        det_diam (float): Diameter of the detector.
        maximum_location (float): Maximum location for normalization.
        sample_range_max (str, optional): Sample range maximum ('mean' or 'histogram').
        sample_size (int): Size of each rectangle in mm.
        calibration_mode (str): Calibration mode ('tof' or 'mc').
        fit_mode (str): Fit mode ('curve_fit' or 'hemisphere_fit').
        index_fig (int): Index for figure naming.
        plot (bool): Flag indicating whether to plot the surface.
        save (bool): Flag indicating whether to save the plot.
        fig_size (tuple): Size of the figure.
        bin_size (float): Size of the bin.

    Returns:
        parameters (numpy.ndarray): Optimized parameters of the bowl correction.
    """
    ensure_choice(calibration_mode, field_name="calibration_mode", allowed=["tof", "mc"])
    ensure_choice(fit_mode, field_name="fit_mode", allowed=BOWL_FIT_MODES)
    ensure_choice(sample_range_max, field_name="sample_range_max", allowed=BOWL_SAMPLE_METHODS)
    bin_size = ensure_positive(bin_size, field_name="bin_size")
    sample_size = int(ensure_positive(sample_size, field_name="sample_size"))

    dld_x_bowl = ensure_non_empty_array(dld_x_bowl, field_name="dld_x_bowl")
    dld_y_bowl = ensure_non_empty_array(dld_y_bowl, field_name="dld_y_bowl")
    dld_t_bowl = ensure_non_empty_array(dld_t_bowl, field_name="dld_t_bowl")
    ensure_matching_lengths(
        dld_x_bowl,
        dld_y_bowl,
        dld_t_bowl,
        field_names=("dld_x_bowl", "dld_y_bowl", "dld_t_bowl"),
    )

    x_sample_list = []
    y_sample_list = []
    dld_t_peak_list = []

    w1 = int(np.floor(np.min(dld_x_bowl)))
    w2 = int(np.ceil(np.max(dld_x_bowl)))
    h1 = int(np.floor(np.min(dld_y_bowl)))
    h2 = int(np.ceil(np.max(dld_y_bowl)))

    d = sample_size  # sample size is in mm
    grid = product(range(h1, h2 - h2 % d, d), range(w1, w2 - w2 % d, d))

    # Use ThreadPoolExecutor or ProcessPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(compute_sample, i, j, d, dld_x_bowl, dld_y_bowl, dld_t_bowl, maximum_location,
                                   sample_range_max, bin_size)
                   for i, j in grid]

        for future in concurrent.futures.as_completed(futures):
            x_sample, y_sample, dld_t_peak = future.result()
            if x_sample is not None:
                x_sample_list.append(x_sample)
                y_sample_list.append(y_sample)
                dld_t_peak_list.append(dld_t_peak)

    if not x_sample_list:
        raise CalibrationInputError('No detector windows contained ions for bowl correction')

    print('x_sample_list max and min:', np.max(x_sample_list), np.min(x_sample_list))
    print('y_sample_list max and min:', np.max(y_sample_list), np.min(y_sample_list))
    print('dld_t_peak_list max and min:', np.max(dld_t_peak_list), np.min(dld_t_peak_list))
    if fit_mode == 'curve_fit':
        parameters, covariance = curve_fit(bowl_corr, [np.array(x_sample_list), np.array(y_sample_list)],
                                           np.array(dld_t_peak_list))
    elif fit_mode == 'ml_fit':
        parameters = hybrid_calibration_model(np.array(x_sample_list), np.array(y_sample_list),
                                              np.array(dld_t_peak_list))
    elif fit_mode == 'robust_fit':
        parameters = robust_fit(np.array(x_sample_list), np.array(y_sample_list), np.array(dld_t_peak_list))

    if plot or save:
        if calibration_mode == 'tof':
            label = 't'
        elif calibration_mode == 'mc':
            label = 'mc'
        model_x_data = np.array(x_sample_list)
        model_y_data = np.array(y_sample_list)
        X, Y = np.meshgrid(model_x_data, model_y_data)
        Z = _predict_bowl_model(fit_mode, parameters, X.ravel(), Y.ravel()).reshape(X.shape)

        fig, ax = plt.subplots(figsize=fig_size, subplot_kw=dict(projection="3d"), constrained_layout=True)
        box = ax.get_position()
        ax.set_position([box.x0 + 0.1, box.y0 + 0.1, box.width * 0.75, box.height * 0.75])
        scat = ax.scatter(model_x_data, model_y_data, zs=1 / np.array(dld_t_peak_list), color="forestgreen",
                          label=r"$%s_{wp}$" % label, s=3)
        fig.add_axes(ax)
        cmap = copy(plt.cm.plasma)
        cmap.set_bad(cmap(0))
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        triangles = Triangulation(x_flat, y_flat)
        # surf = ax.plot_surface(X, Y, 1 / Z, color='red', alpha=0.05, label='bowl')
        surf = ax.plot_trisurf(
            x_flat, y_flat, 1 / z_flat, triangles=triangles.triangles, cmap=cmap, alpha=0.6
        )
        ax.set_xlabel(r'$X_{det}$ (mm)', fontsize=10, labelpad=10)
        ax.set_ylabel(r'$Y_{det}$ (mm)', fontsize=10, labelpad=10)
        ax.set_zlabel(r"${C_B}$", fontsize=10, labelpad=5, color='red')
        ax.zaxis.line.set_color('red')

        # Change z-axis tick label color
        for tick in ax.get_zaxis().get_ticklabels():
            tick.set_color('red')
        ax.view_init(elev=7, azim=-41)

        if save:
            # Enable rendering for text elements
            rcParams['svg.fonttype'] = 'none'
            save_figure(
                fig,
                directory=variables.result_path,
                stem=f"bowl_corr_{index_fig}",
                formats=("svg", "png"),
                dpi=600,
            )
        if plot:
            plt.show()

    print('The parameters of the bowl correction are:', parameters)
    return parameters


def bowl_correction_main(dld_x, dld_y, dld_highVoltage, variables, det_diam, sample_size, fit_mode, calibration_mode,
                         index_fig, plot, save, maximum_cal_method='mean', maximum_sample_method='mean',
                         fig_size=(5, 5), fast_calibration=False, bin_size=0.01, peak_maximum=0, calibration_apply=True):
    """
    Perform bowl correction on the input data and plot the results.

    Args:
        dld_x (numpy.ndarray): X positions.
        dld_y (numpy.ndarray): Y positions.
        dld_highVoltage (numpy.ndarray): High voltage values.
        det_diam (float): Detector diameter.
        sample_size (int): Sample size.
        fit_mode (str): Fit mode ('curve_fit', 'ml_fit', or 'robust_fit').
        calibration_mode (str): Calibration mode ('tof' or 'mc').
        index_fig (int): Index figure.
        plot (bool): Flag indicating whether to plot the results.
        save (bool): Flag indicating whether to save the plots.
        maximum_cal_method (str, optional): Maximum calculation method ('mean' or 'histogram').
        maximum_sample_method (str, optional): Sample range maximum ('mean' or 'histogram').
        fig_size (tuple, optional): Figure size.
        fast_calibration (bool, optional): Flag indicating whether to perform fast calibration.
        bin_size (float, optional): Size of the bin.

    Returns:
        None

    """
    ensure_choice(calibration_mode, field_name="calibration_mode", allowed=["tof", "mc"])
    ensure_choice(fit_mode, field_name="fit_mode", allowed=BOWL_FIT_MODES)
    ensure_choice(maximum_cal_method, field_name="maximum_cal_method", allowed=SAMPLE_METHODS)
    ensure_choice(maximum_sample_method, field_name="maximum_sample_method", allowed=BOWL_SAMPLE_METHODS)
    bin_size = ensure_positive(bin_size, field_name="bin_size")
    sample_size = int(ensure_positive(sample_size, field_name="sample_size"))

    dld_x = np.asarray(dld_x) * 10  # convert cm to mm
    dld_y = np.asarray(dld_y) * 10  # convert cm to mm
    dld_highVoltage = np.asarray(dld_highVoltage)

    all_calibration_values = variables.get_calibration_array(calibration_mode)
    ensure_matching_lengths(
        dld_x,
        dld_y,
        dld_highVoltage,
        all_calibration_values,
        field_names=("dld_x", "dld_y", "dld_highVoltage", f"{calibration_mode}_calibration_values"),
    )

    print('The left and right side of the main peak is:', variables.selected_x1, variables.selected_x2)
    mask_temporal, dld_peak = _extract_peak_mask_and_values(variables, calibration_mode)
    print('The number of ions is:', len(dld_peak))

    if peak_maximum == 0:
        maximum_location = _resolve_peak_location(
            dld_peak,
            method=maximum_cal_method,
            bin_size=bin_size,
            fast_calibration=fast_calibration,
        )
    else:
        maximum_location = float(peak_maximum)

    print('The maximum/mean of peak is located at:', maximum_location)
    dld_x_peak = dld_x[mask_temporal]
    dld_y_peak = dld_y[mask_temporal]
    dld_highVoltage_peak = dld_highVoltage[mask_temporal]

    mean_before = np.mean(dld_peak)
    print('The mean of tof  before bowl calibration is:', mean_before)
    parameters = bowl_correction(dld_x_peak, dld_y_peak, dld_peak, variables, det_diam, maximum_location,
                                 maximum_sample_method, sample_size=sample_size, calibration_mode=calibration_mode,
                                 fit_mode=fit_mode, index_fig=index_fig, plot=plot, save=save, fig_size=fig_size,
                                 bin_size=bin_size)
    print('The fit result is:', parameters)

    mask_fv = np.ones_like(dld_x, dtype=bool)

    f_bowl = _predict_bowl_model(fit_mode, parameters, dld_x[mask_fv], dld_y[mask_fv])

    calibration_mc_tof = np.copy(variables.dld_t_calib) if calibration_mode == 'tof' else np.copy(variables.mc_calib)

    print("Maximum value of f_bowl:", np.max(f_bowl))
    print("Minimum value of f_bowl:", np.min(f_bowl))
    calibration_mc_tof[mask_fv] = calibration_mc_tof[mask_fv] / f_bowl
    # calibration_mc_tof[mask_fv] = calibration_mc_tof[mask_fv] * f_bowl

    mean_after = np.mean(calibration_mc_tof[mask_temporal])
    print('The mean of tof/mc after bowl calibration is:', mean_after)
    print('The difference between the mean of tof before and after bowl calibration is:',
          mean_after - mean_before)

    if plot or save:
        # Plot how bowl correct tof/mc vs high voltage
        fig1, ax1 = plt.subplots(figsize=fig_size, constrained_layout=True)
        mask = np.random.randint(0, len(dld_highVoltage_peak), 10000)

        x = plt.scatter(dld_highVoltage_peak[mask] / 1000, dld_peak[mask], color="blue", label=r"$t$", s=1)

        if calibration_mode == 'tof':
            ax1.set_ylabel("Time of Flight (ns)", fontsize=10)
        elif calibration_mode == 'mc':
            ax1.set_ylabel("mc (Da)", fontsize=10)

        ax1.set_xlabel("Voltage (kV)", fontsize=10)
        plt.grid(alpha=0.3, linestyle='-.', linewidth=0.4)

        f_bowl_plot = _predict_bowl_model(fit_mode, parameters, dld_x_peak[mask], dld_y_peak[mask])
        dld_t_plot = dld_peak[mask] / f_bowl_plot

        y = plt.scatter(dld_highVoltage_peak[mask] / 1000, dld_t_plot, color="red", label=r"$t_{C_{B}}$", s=1)

        plt.legend(handles=[x, y], loc='upper right', markerscale=5., prop={'size': 10})

        if save:
            # Enable rendering for text elements
            rcParams['svg.fonttype'] = 'none'
            save_figure(
                fig1,
                directory=variables.result_path,
                stem=f"peak_tof_bowl_corr_{index_fig}",
                formats=("svg", "png"),
                dpi=600,
            )

        if plot:
            plt.show()

        # Plot how bowl correction correct tof/mc vs dld_x position
        fig1, ax1 = plt.subplots(figsize=fig_size, constrained_layout=True)
        f_bowl_plot = _predict_bowl_model(fit_mode, parameters, dld_x_peak[mask], dld_y_peak[mask])
        dld_t_plot = dld_peak[mask] / f_bowl_plot

        x = plt.scatter(dld_x_peak[mask], dld_peak[mask], color="blue", label=r"$t$", s=1, alpha=0.5)
        y = plt.scatter(dld_x_peak[mask], dld_t_plot, color="red", label=r"$t_{C_{B}}$", s=1, alpha=0.5)

        if calibration_mode == 'tof':
            ax1.set_ylabel("Time of Flight (ns)", fontsize=10)
        elif calibration_mode == 'mc':
            ax1.set_ylabel("mc (Da)", fontsize=10)

        ax1.set_xlabel(r"$X_{det}$ (mm)", fontsize=10)
        plt.grid(color='aqua', alpha=0.3, linestyle='-.', linewidth=0.4)
        plt.legend(handles=[x, y], loc='upper right', markerscale=5., prop={'size': 10})

        if save:
            # Enable rendering for text elements
            rcParams['svg.fonttype'] = 'none'
            save_figure(
                fig1,
                directory=variables.result_path,
                stem=f"peak_tof_bowl_corr_p_x_det_{index_fig}",
                formats=("svg", "png"),
                dpi=600,
            )
        if plot:
            plt.show()

        # Plot how bowl correction correct tof/mc vs dld_x position
        fig1, ax1 = plt.subplots(figsize=fig_size, constrained_layout=True)
        f_bowl_plot = _predict_bowl_model(fit_mode, parameters, dld_x_peak[mask], dld_y_peak[mask])
        dld_t_plot = dld_peak[mask] / f_bowl_plot

        x = plt.scatter(dld_y_peak[mask], dld_peak[mask], color="blue", label=r"$t$", s=1, alpha=0.5)
        y = plt.scatter(dld_y_peak[mask], dld_t_plot, color="red", label=r"$t_{C_{B}}$", s=1, alpha=0.5)

        if calibration_mode == 'tof':
            ax1.set_ylabel("Time of Flight (ns)", fontsize=10)
        elif calibration_mode == 'mc':
            ax1.set_ylabel("mc (Da)", fontsize=10)

        ax1.set_xlabel(r"$y_{det}$ (mm)", fontsize=10)
        plt.grid(color='aqua', alpha=0.3, linestyle='-.', linewidth=0.4)
        plt.legend(handles=[x, y], loc='upper right', markerscale=5., prop={'size': 10})

        if save:
            # Enable rendering for text elements
            rcParams['svg.fonttype'] = 'none'
            save_figure(
                fig1,
                directory=variables.result_path,
                stem=f"peak_tof_bowl_corr_p_y_det_{index_fig}",
                formats=("svg", "png"),
                dpi=600,
            )
        if plot:
            plt.show()

        fig, ax = plt.subplots(figsize=fig_size, subplot_kw=dict(projection="3d"), constrained_layout=True)
        # Adjust the subplot parameters to make the plot smaller
        box = ax.get_position()
        ax.set_position([box.x0 + 0.1, box.y0 + 0.1, box.width * 0.75, box.height * 0.75])
        mask = np.random.randint(0, len(dld_highVoltage_peak), 500)
        f_bowl_plot = _predict_bowl_model(fit_mode, parameters, dld_x_peak[mask], dld_y_peak[mask])
        dld_t_plot = dld_peak[mask] / f_bowl_plot

        scat_1 = ax.scatter(dld_x_peak[mask], dld_y_peak[mask], zs=dld_peak[mask], color="blue",
                            label=r"$t$", s=1)
        scat_2 = ax.scatter(dld_x_peak[mask], dld_y_peak[mask], zs=dld_t_plot, color="red",
                            label=r"$t_{C_{B}}$", s=1)
        plt.legend(handles=[scat_1, scat_2], loc='upper left', markerscale=5., prop={'size': 10})

        ax.set_xlabel(r'$X_{det}$ (mm)', fontsize=10, labelpad=10)
        ax.set_ylabel(r'$Y_{det}$ (mm)', fontsize=10, labelpad=10)
        ax.set_zlabel(r"Time of Flight (ns)", fontsize=10, labelpad=5)
        ax.view_init(elev=7, azim=-41)

        if save:
            # Enable rendering for text elements
            rcParams['svg.fonttype'] = 'none'
            save_figure(
                fig,
                directory=variables.result_path,
                stem=f"peak_tof_bowl_corr_3d_{index_fig}",
                formats=("svg", "png"),
                dpi=600,
            )
        if plot:
            plt.show()

    if calibration_apply:
        if calibration_mode == 'tof':
            variables.dld_t_calib = calibration_mc_tof
        elif calibration_mode == 'mc':
            variables.mc_calib = calibration_mc_tof
    return f_bowl



