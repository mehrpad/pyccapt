from copy import copy

import fast_histogram
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, rcParams
from matplotlib.tri import Triangulation
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_prominences, peak_widths

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
    normalize_sampling_mode,
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


def _build_histogram_bins(values, bin_size):
    """Build stable histogram bin edges/count from the actual data span."""
    data = ensure_non_empty_array(values, field_name="histogram_values")
    lower = float(np.min(data))
    upper = float(np.max(data))
    span = max(upper - lower, float(bin_size))
    n_bins = max(10, int(np.ceil(span / float(bin_size))))
    edges = np.linspace(lower, upper, n_bins + 1)
    return edges, n_bins


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

    bins, n_bins = _build_histogram_bins(data, bin_size)
    hist = fast_histogram.histogram1d(
        data,
        bins=n_bins,
        range=(np.min(data), np.max(data)),
    )
    peaks, properties = find_peaks(hist, height=0)
    if len(peaks) == 0:
        return float(np.mean(data))
    index_peak_max_ini = np.argmax(properties["peak_heights"])
    return float(bins[peaks[index_peak_max_ini]])


def _radial_bowl_corr(data_xy, a, b, c, d, e):
    """Radial-dominant bowl model where r^2 drives the primary curvature."""
    x = np.asarray(data_xy[0], dtype=float)
    y = np.asarray(data_xy[1], dtype=float)
    r2 = x**2 + y**2
    return a + b * r2 + c * (r2**2) + d * x + e * y


def voltage_correction(
    dld_highVoltage_peak,
    dld_t_peak,
    variables,
    maximum_location,
    index_fig,
    figname,
    sample_size,
    mode,
    calibration_mode,
    sample_range_max,
    bin_size,
    plot=True,
    save=False,
    fig_size=(5, 5),
    model='curve_fit',
):
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
            dld_highVoltage_peak_selected = dld_highVoltage_peak[i * sample_size : (i + 1) * sample_size]
            dld_t_peak_selected = dld_t_peak[i * sample_size : (i + 1) * sample_size]
            if sample_range_max == 'histogram':
                try:
                    bins, _ = _build_histogram_bins(dld_t_peak_selected, bin_size)
                    y, x = np.histogram(dld_t_peak_selected, bins=bins)
                    peaks, properties = find_peaks(y, height=0)
                    index_peak_max_ini = np.argmax(properties['peak_heights'])
                    max_peak = peaks[index_peak_max_ini]
                    dld_t_peak_list.append(x[max_peak] / maximum_location)
                    # dld_t_peak_list.append(maximum_location / x[max_peak])
                    mask_v = np.logical_and(
                        (dld_t_peak_selected >= x[max_peak] - bin_size), (dld_t_peak_selected <= x[max_peak] + bin_size)
                    )
                    if len(dld_highVoltage_peak_selected[mask_v]) == 0:
                        mask_v = np.logical_and(
                            (dld_t_peak_selected >= x[max_peak] - 2 * bin_size),
                            (dld_t_peak_selected <= x[max_peak] + 2 * bin_size),
                        )
                        if len(dld_highVoltage_peak_selected[mask_v]) == 0:
                            print('length of mask voltage', len(dld_highVoltage_peak_selected[mask_v]))
                            mask_v = np.logical_and(
                                (dld_t_peak_selected >= x[max_peak] - 4 * bin_size),
                                (dld_t_peak_selected <= x[max_peak] + 4 * bin_size),
                            )
                            print(
                                'length of mask voltage after increase the window', len(dld_highVoltage_peak_selected[mask_v])
                            )
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
            mask = np.logical_and(
                (dld_highVoltage_peak >= (np.min(dld_highVoltage_peak) + (i) * sample_size)),
                (dld_highVoltage_peak < (np.min(dld_highVoltage_peak) + (i + 1) * sample_size)),
            )
            dld_highVoltage_peak_selected = dld_highVoltage_peak[mask]
            dld_t_peak_selected = dld_t_peak[mask]

            if sample_range_max == 'histogram':
                try:
                    bins, _ = _build_histogram_bins(dld_t_peak_selected, bin_size)
                    y, x = np.histogram(dld_t_peak_selected, bins=bins)
                    peaks, properties = find_peaks(y, height=0)
                    index_peak_max_ini = np.argmax(properties['peak_heights'])
                    max_peak = peaks[index_peak_max_ini]
                    dld_t_peak_list.append(x[max_peak] / maximum_location)
                    # dld_t_peak_list.append(maximum_location / x[max_peak])
                    mask_v = np.logical_and(
                        (dld_t_peak_selected >= x[max_peak] - bin_size), (dld_t_peak_selected <= x[max_peak] + bin_size)
                    )
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

        x = plt.scatter(
            np.array(high_voltage_mean_list) / 1000,
            np.array(dld_t_peak_list) * maximum_location,
            color="forestgreen",
            label=r"$%s_{wp}$" % label,
            s=5,
        )
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
        plt.legend(handles=[x, y[0]], loc='lower left', markerscale=5.0, prop={'size': 10})

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


def voltage_corr_main(
    dld_highVoltage,
    variables,
    sample_size,
    mode,
    calibration_mode,
    index_fig,
    plot,
    save,
    maximum_cal_method='mean',
    maximum_sample_method='mean',
    fig_size=(5, 5),
    fast_calibration=False,
    bin_size=0.01,
    model='curve_fit',
    peak_maximum=0,
    calibration_apply=True,
):
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

    left_edge, right_edge = variables.get_calibration_peak_range(calibration_mode)
    print('The left and right side of the main peak is:', left_edge, right_edge)
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
    fitresult = voltage_correction(
        dld_highVoltage_peak_v,
        dld_peak_b,
        variables,
        maximum_location,
        index_fig=index_fig,
        figname='voltage_corr',
        sample_size=sample_size,
        mode=mode,
        calibration_mode=calibration_mode,
        sample_range_max=maximum_sample_method,
        bin_size=bin_size,
        plot=plot,
        save=save,
        fig_size=fig_size,
        model=model,
    )

    calibration_mc_tof = np.copy(variables.dld_t_calib) if calibration_mode == 'tof' else np.copy(variables.mc_calib)
    print('The fit result is:', fitresult)
    mask_fv = np.ones_like(dld_highVoltage, dtype=bool)

    f_v = _predict_voltage_model(model, fitresult, np.asarray(dld_highVoltage)[mask_fv])

    if calibration_mode == 'tof':
        correction_factor = np.sqrt(f_v)
    else:
        correction_factor = f_v
    print("Maximum value of correction factor:", np.max(correction_factor))
    print("Minimum value of correction factor:", np.min(correction_factor))
    calibration_mc_tof[mask_fv] = calibration_mc_tof[mask_fv] / correction_factor

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

        correction_plot = 1 / np.sqrt(f_v_plot) if calibration_mode == 'tof' else 1 / f_v_plot
        y = ax2.plot(dld_highVoltage_peak_v / 1000, correction_plot, color='r', label=r"$C_{V}^{-1}$")
        # y = ax2.plot(dld_highVoltage_peak_v / 1000, f_v_plot, color='r', label=r"$C_{V}^{-1}$")
        ax2.set_ylabel(r"$C_{V}^{-1}$", color="red", fontsize=10)
        ax2.tick_params(axis='y', colors='red')  # Change color and thickness of tick labels on y-axis
        ax2.spines['right'].set_color('red')  # Change color of right border
        plt.legend(handles=[x, y[0]], loc='upper left', markerscale=5.0, prop={'size': 10})

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

        dld_t_plot = dld_peak_b * correction_plot
        # dld_t_plot = dld_peak_b * f_v_plot

        y = plt.scatter(dld_highVoltage_peak_v[mask] / 1000, dld_t_plot[mask], color="red", label=r"$t_{C_{V}}$", s=1)

        plt.legend(handles=[x, y], loc='upper right', markerscale=5.0, prop={'size': 10})

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
    print('The difference between the mean of tof/mc before and after voltage calibration is:', mean_after - mean_before)
    if calibration_apply:
        if calibration_mode == 'tof':
            variables.dld_t_calib = calibration_mc_tof
        elif calibration_mode == 'mc':
            variables.mc_calib = calibration_mc_tof
    return f_v


def _resolve_sampling_mode(sampling_mode, variables=None):
    """Resolve sampling mode with a variable-level default fallback."""
    configured = sampling_mode
    if configured is None and variables is not None:
        configured = getattr(variables, "bowl_sampling_mode", "polar")
    mode = normalize_sampling_mode(configured)
    if variables is not None and hasattr(variables, "bowl_sampling_mode"):
        variables.bowl_sampling_mode = mode
    return mode


def _cell_peak_value(values, maximum_location, sample_range_max, bin_size):
    if sample_range_max == 'mean':
        return float(np.mean(values)) / maximum_location
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float('nan')
    value_span = float(np.max(values) - np.min(values))
    if not np.isfinite(value_span) or value_span <= 0:
        return float(np.mean(values)) / maximum_location
    bins, n_bins = _build_histogram_bins(values, bin_size)
    y_hist = fast_histogram.histogram1d(
        values,
        bins=n_bins,
        range=(np.min(values), np.max(values)),
    )
    peaks, properties = find_peaks(y_hist, height=0)
    if len(peaks) == 0:
        return float(np.mean(values)) / maximum_location
    index_peak_max_ini = np.argmax(properties['peak_heights'])
    return float(bins[peaks[index_peak_max_ini]]) / maximum_location


def _iter_polar_cells(radial_distance, sample_size, det_diam):
    r_max_data = float(np.max(radial_distance)) if len(radial_distance) else 0.0
    r_max_detector = max(0.0, float(det_diam) / 2.0)
    r_max = max(r_max_data, r_max_detector, float(sample_size))
    n_rings = max(1, int(np.ceil(r_max / float(sample_size))))
    ring_edges = np.linspace(0.0, r_max, n_rings + 1)

    for r_idx in range(n_rings):
        r_min_i = float(ring_edges[r_idx])
        r_max_i = float(ring_edges[r_idx + 1])
        r_mid = 0.5 * (r_min_i + r_max_i)
        circumference = 2.0 * np.pi * max(r_mid, 0.5 * float(sample_size))
        n_sectors = int(np.clip(np.ceil(circumference / float(sample_size)), 8, 96))
        for theta_idx in range(n_sectors):
            theta_min = float(theta_idx * 2.0 * np.pi / n_sectors)
            theta_max = float((theta_idx + 1) * 2.0 * np.pi / n_sectors)
            yield r_min_i, r_max_i, theta_min, theta_max


def _collect_spatial_samples(
    dld_x,
    dld_y,
    dld_t,
    maximum_location,
    sample_range_max,
    sample_size,
    bin_size,
    *,
    sampling_mode,
    det_diam,
    dld_v=None,
):
    x_samples = []
    y_samples = []
    t_samples = []
    v_samples = []

    if sampling_mode == 'polar':
        radial_distance = np.hypot(dld_x, dld_y)
        polar_angle = np.mod(np.arctan2(dld_y, dld_x), 2.0 * np.pi)
        for r_min_i, r_max_i, theta_min, theta_max in _iter_polar_cells(radial_distance, sample_size, det_diam):
            mask = (
                (radial_distance >= r_min_i)
                & (radial_distance < r_max_i)
                & (polar_angle >= theta_min)
                & (polar_angle < theta_max)
            )
            if not np.any(mask):
                continue
            cell_t = dld_t[mask]
            normalized_value = _cell_peak_value(cell_t, maximum_location, sample_range_max, bin_size)
            if not np.isfinite(normalized_value) or normalized_value <= 0:
                continue
            x_samples.append(float(np.median(dld_x[mask])))
            y_samples.append(float(np.median(dld_y[mask])))
            t_samples.append(normalized_value)
            if dld_v is not None:
                v_samples.append(float(np.mean(dld_v[mask])))
    else:
        cell_size = float(sample_size)
        x_min = float(np.floor(np.min(dld_x)))
        x_max = float(np.ceil(np.max(dld_x)))
        y_min = float(np.floor(np.min(dld_y)))
        y_max = float(np.ceil(np.max(dld_y)))
        n_cols = max(1, int(np.ceil((x_max - x_min) / cell_size)))
        n_rows = max(1, int(np.ceil((y_max - y_min) / cell_size)))
        x_bin = np.floor((dld_x - x_min) / cell_size).astype(int)
        y_bin = np.floor((dld_y - y_min) / cell_size).astype(int)
        valid = (x_bin >= 0) & (x_bin < n_cols) & (y_bin >= 0) & (y_bin < n_rows)
        if np.any(valid):
            cell_id = y_bin[valid] * n_cols + x_bin[valid]
            order = np.argsort(cell_id, kind='mergesort')
            x_sorted = dld_x[valid][order]
            y_sorted = dld_y[valid][order]
            t_sorted = dld_t[valid][order]
            v_sorted = dld_v[valid][order] if dld_v is not None else None
            cell_sorted = cell_id[order]
            _, starts, counts = np.unique(cell_sorted, return_index=True, return_counts=True)
            for start, count in zip(starts, counts):
                stop = start + count
                cell_t = t_sorted[start:stop]
                normalized_value = _cell_peak_value(cell_t, maximum_location, sample_range_max, bin_size)
                if not np.isfinite(normalized_value) or normalized_value <= 0:
                    continue
                x_samples.append(float(np.median(x_sorted[start:stop])))
                y_samples.append(float(np.median(y_sorted[start:stop])))
                t_samples.append(normalized_value)
                if v_sorted is not None:
                    v_samples.append(float(np.mean(v_sorted[start:stop])))

    result = {
        'x': np.asarray(x_samples, dtype=float),
        'y': np.asarray(y_samples, dtype=float),
        't': np.asarray(t_samples, dtype=float),
    }
    if dld_v is not None:
        result['v'] = np.asarray(v_samples, dtype=float)
    return result


def bowl_correction(
    dld_x_bowl,
    dld_y_bowl,
    dld_t_bowl,
    variables,
    det_diam,
    maximum_location,
    sample_range_max,
    sample_size,
    calibration_mode,
    fit_mode,
    index_fig,
    plot,
    save,
    fig_size=(7, 5),
    bin_size=0.01,
    sampling_mode='polar',
):
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

    sampling_mode = _resolve_sampling_mode(sampling_mode, variables)
    samples = _collect_spatial_samples(
        np.asarray(dld_x_bowl, dtype=float),
        np.asarray(dld_y_bowl, dtype=float),
        np.asarray(dld_t_bowl, dtype=float),
        maximum_location,
        sample_range_max,
        sample_size,
        bin_size,
        sampling_mode=sampling_mode,
        det_diam=det_diam,
    )

    x_samples = samples['x']
    y_samples = samples['y']
    t_samples = samples['t']
    if len(x_samples) == 0:
        raise CalibrationInputError('No detector windows contained ions for bowl correction')

    print('x_sample_list max and min:', np.max(x_samples), np.min(x_samples))
    print('y_sample_list max and min:', np.max(y_samples), np.min(y_samples))
    print('dld_t_peak_list max and min:', np.max(t_samples), np.min(t_samples))
    r2_samples = x_samples**2 + y_samples**2
    radial_design = np.column_stack(
        [
            np.ones(len(x_samples), dtype=float),
            r2_samples,
            r2_samples**2,
            x_samples,
            y_samples,
        ]
    )

    if fit_mode == 'curve_fit':
        if sampling_mode == 'polar':
            radial_parameters, _ = curve_fit(
                _radial_bowl_corr,
                [x_samples, y_samples],
                t_samples,
            )
            parameters = {
                'model': 'radial_curve_fit',
                'parameters': [float(value) for value in radial_parameters],
                'sampling_mode': sampling_mode,
            }
        else:
            parameters, _ = curve_fit(bowl_corr, [x_samples, y_samples], t_samples)
    elif fit_mode == 'ml_fit':
        parameters = hybrid_calibration_model(x_samples, y_samples, t_samples)
    elif fit_mode == 'robust_fit':
        if sampling_mode == 'polar':
            radial_parameters = _robust_joint_linear_fit(radial_design, t_samples)
            parameters = {
                'model': 'radial_linear',
                'parameters': [float(value) for value in radial_parameters],
                'sampling_mode': sampling_mode,
            }
        else:
            parameters = robust_fit(x_samples, y_samples, t_samples)

    if plot or save:
        if calibration_mode == 'tof':
            label = 't'
        elif calibration_mode == 'mc':
            label = 'mc'
        model_x_data = np.array(x_samples)
        model_y_data = np.array(y_samples)
        X, Y = np.meshgrid(model_x_data, model_y_data)
        Z = _predict_bowl_model(fit_mode, parameters, X.ravel(), Y.ravel()).reshape(X.shape)

        fig, ax = plt.subplots(figsize=fig_size, subplot_kw=dict(projection="3d"), constrained_layout=True)
        box = ax.get_position()
        ax.set_position([box.x0 + 0.1, box.y0 + 0.1, box.width * 0.75, box.height * 0.75])
        scat = ax.scatter(
            model_x_data, model_y_data, zs=1 / np.array(t_samples), color="forestgreen", label=r"$%s_{wp}$" % label, s=3
        )
        fig.add_axes(ax)
        cmap = copy(plt.cm.plasma)
        cmap.set_bad(cmap(0))
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        triangles = Triangulation(x_flat, y_flat)
        # surf = ax.plot_surface(X, Y, 1 / Z, color='red', alpha=0.05, label='bowl')
        surf = ax.plot_trisurf(x_flat, y_flat, 1 / z_flat, triangles=triangles.triangles, cmap=cmap, alpha=0.6)
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


def bowl_correction_main(
    dld_x,
    dld_y,
    dld_highVoltage,
    variables,
    det_diam,
    sample_size,
    fit_mode,
    calibration_mode,
    index_fig,
    plot,
    save,
    maximum_cal_method='mean',
    maximum_sample_method='mean',
    fig_size=(5, 5),
    fast_calibration=False,
    bin_size=0.01,
    peak_maximum=0,
    calibration_apply=True,
    sampling_mode='polar',
):
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

    left_edge, right_edge = variables.get_calibration_peak_range(calibration_mode)
    print('The left and right side of the main peak is:', left_edge, right_edge)
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
    parameters = bowl_correction(
        dld_x_peak,
        dld_y_peak,
        dld_peak,
        variables,
        det_diam,
        maximum_location,
        maximum_sample_method,
        sample_size=sample_size,
        calibration_mode=calibration_mode,
        fit_mode=fit_mode,
        index_fig=index_fig,
        plot=plot,
        save=save,
        fig_size=fig_size,
        bin_size=bin_size,
        sampling_mode=sampling_mode,
    )
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
    print('The difference between the mean of tof before and after bowl calibration is:', mean_after - mean_before)

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

        plt.legend(handles=[x, y], loc='upper right', markerscale=5.0, prop={'size': 10})

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
        plt.legend(handles=[x, y], loc='upper right', markerscale=5.0, prop={'size': 10})

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
        plt.legend(handles=[x, y], loc='upper right', markerscale=5.0, prop={'size': 10})

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

        scat_1 = ax.scatter(dld_x_peak[mask], dld_y_peak[mask], zs=dld_peak[mask], color="blue", label=r"$t$", s=1)
        scat_2 = ax.scatter(dld_x_peak[mask], dld_y_peak[mask], zs=dld_t_plot, color="red", label=r"$t_{C_{B}}$", s=1)
        plt.legend(handles=[scat_1, scat_2], loc='upper left', markerscale=5.0, prop={'size': 10})

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


def _auto_detect_peaks(calibration_array, n_peaks=3, prominence=100, distance=500, hist_bin_size=0.1):
    """Auto-detect the top N prominent peaks in the calibration spectrum."""
    arr = np.asarray(calibration_array, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 50:
        raise CalibrationInputError("Not enough valid ions for multi-peak calibration")
    mc_min = float(np.percentile(arr, 0.1))
    mc_max = float(np.percentile(arr, 99.9))
    trimmed = arr[(arr >= mc_min) & (arr <= mc_max)]
    if trimmed.size < 50:
        raise CalibrationInputError("Not enough ions remain after trimming for multi-peak calibration")
    hist_edges, _ = _build_histogram_bins(trimmed, max(hist_bin_size, 1e-6))
    hist_y = np.histogram(trimmed, bins=hist_edges)[0]
    hist_x = (hist_edges[:-1] + hist_edges[1:]) / 2
    if hist_y.size >= 5:
        kernel_size = min(9, hist_y.size if hist_y.size % 2 == 1 else hist_y.size - 1)
        kernel_size = max(3, kernel_size)
        kernel = np.ones(kernel_size, dtype=float) / float(kernel_size)
        hist_y_eval = np.convolve(hist_y.astype(float), kernel, mode='same')
    else:
        hist_y_eval = hist_y.astype(float)

    peaks_found = np.array([], dtype=int)
    prominence_trials = [
        max(1.0, float(prominence)),
        max(1.0, float(prominence) * 0.5),
        max(1.0, float(prominence) * 0.25),
        max(1.0, float(np.max(hist_y_eval)) * 0.05),
    ]
    for prominence_value in prominence_trials:
        peaks_found, _ = find_peaks(
            hist_y_eval,
            prominence=prominence_value,
            distance=max(1, int(distance)),
            height=0,
        )
        if len(peaks_found) > 0:
            break
    if len(peaks_found) == 0:
        peaks_found, _ = find_peaks(
            hist_y_eval,
            prominence=max(1.0, float(np.max(hist_y_eval)) * 0.02),
            distance=max(1, int(distance) // 2),
            height=0,
        )
    if len(peaks_found) == 0:
        raise CalibrationInputError("No peaks found for multi-peak calibration")

    prom = peak_prominences(hist_y_eval, peaks_found)
    pw = peak_widths(hist_y_eval, peaks_found, rel_height=0.5)

    sorted_idx = np.argsort(prom[0])[::-1]
    top_n = min(n_peaks, len(sorted_idx))

    results = []
    for pi in sorted_idx[:top_n]:
        peak_pos = float(hist_x[peaks_found[pi]])
        left = float(np.interp(pw[2][pi], np.arange(len(hist_x)), hist_x))
        right = float(np.interp(pw[3][pi], np.arange(len(hist_x)), hist_x))
        width = max(right - left, float(hist_bin_size) * 3)
        margin = max(float(hist_bin_size) * 2, width * 0.25)
        x1 = max(mc_min, left - margin)
        x2 = min(mc_max, right + margin)
        if x2 <= x1:
            continue
        candidate = {
            'position': peak_pos,
            'x1': x1,
            'x2': x2,
            'prominence': float(prom[0][pi]),
            'width': float(width),
        }
        if any(not (candidate['x2'] <= existing['x1'] or existing['x2'] <= candidate['x1']) for existing in results):
            continue
        results.append(candidate)
    if not results:
        raise CalibrationInputError("Peak windows could not be resolved for multi-peak calibration")
    return results


def auto_detect_reference_peaks(calibration_array, n_peaks=6, prominence=100, distance=500, hist_bin_size=0.1):
    """Public helper for stable multi-peak evaluation windows used by auto calibration."""
    return _auto_detect_peaks(
        calibration_array,
        n_peaks=n_peaks,
        prominence=prominence,
        distance=distance,
        hist_bin_size=hist_bin_size,
    )


def multi_peak_voltage_corr_main(
    dld_highVoltage,
    variables,
    calibration_mode='mc',
    model='robust_fit',
    bin_size=0.01,
    n_peaks=3,
    prominence=100,
    distance=500,
):
    """Voltage correction using multiple auto-detected peaks simultaneously."""
    model = normalize_voltage_model(model)
    calib_arr = variables.get_calibration_array(calibration_mode)
    dld_hv = np.asarray(dld_highVoltage, dtype=float)

    detected = _auto_detect_peaks(
        calib_arr,
        n_peaks=n_peaks,
        prominence=prominence,
        distance=distance,
    )

    all_v_means = []
    all_t_normalized = []
    peak_info = []

    for pk in detected:
        x1, x2, peak_pos = pk['x1'], pk['x2'], pk['position']
        mask = (calib_arr > x1) & (calib_arr < x2)
        peak_ions = calib_arr[mask]
        voltage_ions = dld_hv[mask]

        if len(peak_ions) < 100:
            continue

        maximum_location = _resolve_peak_location(peak_ions, 'histogram', bin_size)
        if maximum_location == 0:
            continue

        sample_size = max(1, int(len(voltage_ions) / 100))
        for i in range(0, len(voltage_ions), sample_size):
            chunk_v = voltage_ions[i : i + sample_size]
            chunk_t = peak_ions[i : i + sample_size]
            if len(chunk_t) < 5:
                continue
            mean_voltage = float(np.mean(chunk_v))
            normalized_value = float(np.mean(chunk_t)) / maximum_location
            if not np.isfinite(mean_voltage) or not np.isfinite(normalized_value) or normalized_value <= 0:
                continue
            all_v_means.append(mean_voltage)
            all_t_normalized.append(normalized_value)

        peak_info.append(
            {
                'position': peak_pos,
                'x1': x1,
                'x2': x2,
                'n_ions': int(np.sum(mask)),
            }
        )

    if len(all_v_means) < 3:
        raise CalibrationInputError("Not enough data points from multi-peak detection for voltage correction")

    v_arr = np.asarray(all_v_means)
    t_arr = np.asarray(all_t_normalized)

    if model == 'robust_fit':
        fitresult = robust_voltage_fit(v_arr, t_arr)
    elif model == 'curve_fit':
        fitresult, _ = curve_fit(voltage_corr, v_arr, t_arr)
    predicted_fit = _predict_voltage_model(model, fitresult, v_arr)
    if not np.all(np.isfinite(np.asarray(predicted_fit, dtype=float))):
        raise CalibrationInputError("Voltage fit returned invalid parameters")

    f_v = np.clip(
        _predict_voltage_model(model, fitresult, dld_hv),
        np.finfo(float).eps,
        None,
    )

    correction_factor = np.sqrt(f_v) if calibration_mode == 'tof' else f_v

    source = variables.dld_t_calib if calibration_mode == 'tof' else variables.mc_calib
    calibration_mc_tof = source / correction_factor

    if calibration_mode == 'tof':
        variables.dld_t_calib = calibration_mc_tof
    else:
        variables.mc_calib = calibration_mc_tof

    return fitresult, peak_info


def multi_peak_bowl_corr_main(
    dld_x,
    dld_y,
    dld_highVoltage,
    variables,
    det_diam,
    calibration_mode='mc',
    fit_mode='robust_fit',
    sample_size=9,
    bin_size=0.01,
    n_peaks=3,
    prominence=100,
    distance=500,
    sampling_mode='polar',
):
    """Bowl correction using multiple auto-detected peaks simultaneously."""
    ensure_choice(fit_mode, field_name="fit_mode", allowed=BOWL_FIT_MODES)
    calib_arr = variables.get_calibration_array(calibration_mode)
    sampling_mode = _resolve_sampling_mode(sampling_mode, variables)
    dld_x_mm = np.asarray(dld_x) * 10
    dld_y_mm = np.asarray(dld_y) * 10

    detected = _auto_detect_peaks(
        calib_arr,
        n_peaks=n_peaks,
        prominence=prominence,
        distance=distance,
    )

    all_x_samples = []
    all_y_samples = []
    all_t_normalized = []
    n_used = 0

    for pk in detected:
        x1, x2 = pk['x1'], pk['x2']
        mask = (calib_arr > x1) & (calib_arr < x2)
        peak_t = calib_arr[mask]
        peak_x = dld_x_mm[mask]
        peak_y = dld_y_mm[mask]

        if len(peak_t) < 100:
            continue

        maximum_location = _resolve_peak_location(peak_t, 'histogram', bin_size)
        if maximum_location == 0:
            continue

        peak_samples = _collect_spatial_samples(
            peak_x,
            peak_y,
            peak_t,
            maximum_location,
            'mean',
            sample_size,
            bin_size,
            sampling_mode=sampling_mode,
            det_diam=det_diam,
        )
        if len(peak_samples['x']) == 0:
            continue
        all_x_samples.extend(peak_samples['x'].tolist())
        all_y_samples.extend(peak_samples['y'].tolist())
        all_t_normalized.extend(peak_samples['t'].tolist())

        n_used += 1

    if len(all_x_samples) < 5:
        raise CalibrationInputError("Not enough spatial data from multi-peak detection for bowl correction")

    x_arr = np.array(all_x_samples)
    y_arr = np.array(all_y_samples)
    t_arr = np.array(all_t_normalized)

    if fit_mode == 'curve_fit':
        if sampling_mode == 'polar':
            radial_parameters, _ = curve_fit(_radial_bowl_corr, [x_arr, y_arr], t_arr)
            parameters = {
                'model': 'radial_curve_fit',
                'parameters': [float(value) for value in radial_parameters],
                'sampling_mode': sampling_mode,
            }
        else:
            parameters, _ = curve_fit(bowl_corr, [x_arr, y_arr], t_arr)
    elif fit_mode == 'ml_fit':
        parameters = hybrid_calibration_model(x_arr, y_arr, t_arr)
    elif fit_mode == 'robust_fit':
        if sampling_mode == 'polar':
            r2_arr = x_arr**2 + y_arr**2
            radial_design = np.column_stack(
                [
                    np.ones(len(x_arr), dtype=float),
                    r2_arr,
                    r2_arr**2,
                    x_arr,
                    y_arr,
                ]
            )
            radial_parameters = _robust_joint_linear_fit(radial_design, t_arr)
            parameters = {
                'model': 'radial_linear',
                'parameters': [float(value) for value in radial_parameters],
                'sampling_mode': sampling_mode,
            }
        else:
            parameters = robust_fit(x_arr, y_arr, t_arr)

    if isinstance(parameters, dict):
        is_finite = np.all(np.isfinite(np.asarray(parameters.get('parameters', []), dtype=float)))
    elif fit_mode == 'robust_fit':
        trial_prediction = _predict_bowl_model(fit_mode, parameters, x_arr, y_arr)
        is_finite = np.all(np.isfinite(np.asarray(trial_prediction, dtype=float)))
    else:
        is_finite = np.all(np.isfinite(np.asarray(parameters, dtype=float)))
    if not is_finite:
        raise CalibrationInputError("Bowl fit returned invalid parameters")

    f_bowl = np.clip(
        _predict_bowl_model(fit_mode, parameters, dld_x_mm, dld_y_mm),
        np.finfo(float).eps,
        None,
    )

    source = variables.dld_t_calib if calibration_mode == 'tof' else variables.mc_calib
    calibration_mc_tof = source / f_bowl

    if calibration_mode == 'tof':
        variables.dld_t_calib = calibration_mc_tof
    else:
        variables.mc_calib = calibration_mc_tof

    return parameters, n_used


def _joint_feature_matrix(voltage_values, x_values, y_values, voltage_center, voltage_scale, spatial_scale):
    """Build a smooth joint voltage-plus-detector feature matrix."""
    v = (np.asarray(voltage_values, dtype=float) - float(voltage_center)) / float(voltage_scale)
    x = np.asarray(x_values, dtype=float) / float(spatial_scale)
    y = np.asarray(y_values, dtype=float) / float(spatial_scale)
    r2 = x**2 + y**2
    return np.column_stack(
        [
            np.ones(len(v), dtype=float),
            v,
            v**2,
            x,
            y,
            x**2,
            y**2,
            x * y,
            r2,
            v * r2,
            v * x,
            v * y,
        ]
    )


def _robust_joint_linear_fit(feature_matrix, target):
    """Fit a smooth joint correction surface while rejecting large residual outliers."""
    x_data = np.asarray(feature_matrix, dtype=float)
    y_data = np.asarray(target, dtype=float)
    params, *_ = np.linalg.lstsq(x_data, y_data, rcond=None)

    for _ in range(3):
        residual = y_data - x_data @ params
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        scale = max(1.4826 * mad, np.finfo(float).eps)
        good = np.abs(residual - median) <= 3.5 * scale
        if np.count_nonzero(good) < x_data.shape[1]:
            break
        params, *_ = np.linalg.lstsq(x_data[good], y_data[good], rcond=None)
    return params


def joint_voltage_bowl_corr_main(
    dld_x,
    dld_y,
    dld_highVoltage,
    variables,
    det_diam,
    calibration_mode='mc',
    sample_size=9,
    bin_size=0.01,
    n_peaks=4,
    prominence=100,
    distance=500,
    sampling_mode='polar',
):
    """Fit a smooth combined voltage-plus-detector correction surface from several peaks."""
    ensure_choice(calibration_mode, field_name="calibration_mode", allowed=["tof", "mc"])
    calib_arr = variables.get_calibration_array(calibration_mode)
    sampling_mode = _resolve_sampling_mode(sampling_mode, variables)
    dld_x_mm = np.asarray(dld_x, dtype=float) * 10
    dld_y_mm = np.asarray(dld_y, dtype=float) * 10
    dld_highVoltage = np.asarray(dld_highVoltage, dtype=float)
    ensure_matching_lengths(
        dld_x_mm,
        dld_y_mm,
        dld_highVoltage,
        calib_arr,
        field_names=("dld_x", "dld_y", "dld_highVoltage", f"{calibration_mode}_calibration_values"),
    )
    sample_size = int(ensure_positive(sample_size, field_name="sample_size"))
    bin_size = ensure_positive(bin_size, field_name="bin_size")

    detected = _auto_detect_peaks(
        calib_arr,
        n_peaks=n_peaks,
        prominence=prominence,
        distance=distance,
        hist_bin_size=bin_size,
    )

    all_x_samples = []
    all_y_samples = []
    all_v_samples = []
    all_t_normalized = []
    peak_info = []

    for peak in detected:
        mask = (calib_arr > peak['x1']) & (calib_arr < peak['x2'])
        peak_t = calib_arr[mask]
        peak_x = dld_x_mm[mask]
        peak_y = dld_y_mm[mask]
        peak_v = dld_highVoltage[mask]
        if len(peak_t) < 100:
            continue

        maximum_location = _resolve_peak_location(peak_t, 'histogram', bin_size)
        if maximum_location <= 0 or not np.isfinite(maximum_location):
            continue

        peak_samples = _collect_spatial_samples(
            peak_x,
            peak_y,
            peak_t,
            maximum_location,
            'mean',
            sample_size,
            bin_size,
            sampling_mode=sampling_mode,
            det_diam=det_diam,
            dld_v=peak_v,
        )
        n_cells = len(peak_samples['x'])
        if n_cells == 0:
            continue

        all_x_samples.extend(peak_samples['x'].tolist())
        all_y_samples.extend(peak_samples['y'].tolist())
        all_v_samples.extend(peak_samples['v'].tolist())
        all_t_normalized.extend(peak_samples['t'].tolist())

        if n_cells > 0:
            peak_info.append(
                {
                    'position': peak['position'],
                    'x1': peak['x1'],
                    'x2': peak['x2'],
                    'n_ions': int(np.sum(mask)),
                    'n_cells': int(n_cells),
                }
            )

    if len(all_t_normalized) < 20:
        raise CalibrationInputError("Not enough samples from multi-peak detection for joint voltage/bowl refinement")

    v_arr = np.asarray(all_v_samples, dtype=float)
    x_arr = np.asarray(all_x_samples, dtype=float)
    y_arr = np.asarray(all_y_samples, dtype=float)
    t_arr = np.asarray(all_t_normalized, dtype=float)
    voltage_center = float(np.median(v_arr))
    voltage_scale = max(float(np.std(v_arr)), np.finfo(float).eps)
    spatial_scale = max(
        float(np.nanmax(np.sqrt(x_arr**2 + y_arr**2))) if len(x_arr) else 0.0,
        float(det_diam) / 2.0,
        1.0,
    )

    feature_matrix = _joint_feature_matrix(v_arr, x_arr, y_arr, voltage_center, voltage_scale, spatial_scale)
    parameters = _robust_joint_linear_fit(feature_matrix, t_arr)
    if not np.all(np.isfinite(parameters)):
        raise CalibrationInputError("Joint voltage/bowl fit returned invalid parameters")

    all_features = _joint_feature_matrix(
        dld_highVoltage,
        dld_x_mm,
        dld_y_mm,
        voltage_center,
        voltage_scale,
        spatial_scale,
    )
    correction = np.clip(all_features @ parameters, np.finfo(float).eps, None)
    source = variables.dld_t_calib if calibration_mode == 'tof' else variables.mc_calib
    corrected = source / correction

    if calibration_mode == 'tof':
        variables.dld_t_calib = corrected
    else:
        variables.mc_calib = corrected

    model = {
        'parameters': [float(value) for value in parameters],
        'feature_names': ['bias', 'v', 'v2', 'x', 'y', 'x2', 'y2', 'xy', 'r2', 'v_r2', 'v_x', 'v_y'],
        'voltage_center': voltage_center,
        'voltage_scale': voltage_scale,
        'spatial_scale': spatial_scale,
        'peaks_used': len(peak_info),
    }
    return model, peak_info
