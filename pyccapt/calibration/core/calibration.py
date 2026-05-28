"""Mass-calibration core for atom probe tomography data.

This module turns detector hits (time-of-flight, high voltage, detector
position) into a calibrated mass-to-charge (m/c) spectrum. The standard
pipeline, in order, is:

1. Initial calibration (``diagnostics.initial_calibration``): a global
   t0 / flight-path estimate converts time-of-flight to a first m/c.
2. Voltage correction (``voltage_corr_main``): corrects the per-event
   m/c for the slow change in standing voltage during evaporation, fit
   over ion-index or voltage segments.
3. Bowl correction (``bowl_correction_main``): corrects the residual
   position-dependent flight-time difference across the detector
   ("bowl"), sampled in Cartesian or polar cells.
4. Optional time-drift correction (``new_methods.voltage_corr_time_dependent``):
   a multiplicative per-ion-index correction that cancels HV / temperature
   drift left after steps 2-3 in long runs.
5. Adaptive residual calibration (``adaptive_residual_calibration``):
   a per-peak temporal + spatial residual fit that tightens mass
   resolution after the parametric corrections.

The notebook helpers expose three configuration presets that only change
what happens after the shared voltage+bowl stage:
``new`` (adaptive residual, default), ``best`` (adds step 4), and
``old`` (legacy adaptive residual without the coarse-to-fine speedup).
A separate NIST reference fit in the ion-list helper rescales the
already-calibrated m/c onto reference masses without re-fitting V/bowl/drift.

Peak locations are read from histograms at BIN CENTERS (not bin edges)
and histogram bins are anchored to the requested bin width. Hot paths
(bowl polar sampling, the voltage-correction segment loop) parallelise
across CPU cores via ``parallel_map`` when the workload is large enough.
"""

from collections.abc import Mapping
from copy import copy

import fast_histogram
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.tri import Triangulation
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_prominences, peak_widths

from pyccapt.calibration.core.exceptions import CalibrationInputError
from pyccapt.calibration.core.parallel import parallel_map
from pyccapt.calibration.core.validation import (
    BOWL_FIT_MODES,
    BOWL_SAMPLE_METHODS,
    CALIBRATION_MODES,
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
    refine_correction_nelder_mead,
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

def _resolve_peak_location(values, method, bin_size, fast_calibration=False, rng=None):
    """Resolve the reference peak location by histogram/mean/median.

    Parameters
    ----------
    rng : optional ``numpy.random.Generator``
        Generator used by the ``fast_calibration`` subsample path. Defaults
        to a seeded generator so the same input produces the same peak
        location across runs (the previous implementation used the global
        ``np.random`` state, making fast-mode calibrations non-reproducible).
    """
    ensure_choice(method, field_name="maximum_cal_method", allowed=SAMPLE_METHODS)
    data = ensure_non_empty_array(values, field_name="peak_values")
    if fast_calibration and data.size > 10:
        if rng is None:
            rng = np.random.default_rng(0)
        data = rng.choice(data, int(data.size * 0.1), replace=False)

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
    # ``bins`` is the edges array (length n_bins+1); ``peaks`` indexes the
    # counts array (length n_bins). bins[peak] returns the LEFT edge of the
    # bin, biasing the peak location low by half a bin width. Use the bin
    # center instead.
    peak_idx = int(peaks[index_peak_max_ini])
    return float(0.5 * (bins[peak_idx] + bins[peak_idx + 1]))

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

    # Each segment below is independent of every other and the inner work is
    # NumPy + scipy.signal.find_peaks -- both release the GIL, so we run the
    # per-segment evaluation through parallel_map(gil_releasing=True) and let
    # the auto-serial fallback kick in for tiny datasets.
    if mode == 'ion_seq':
        # Use ceildiv (not int(len/sample_size) + 1) so a trailing empty
        # segment is not generated on exact divisibility. The empty
        # segment is filtered downstream, but the printed segment count
        # then disagreed with the number of (V, t) points the fit saw.
        num_segments = (len(dld_highVoltage_peak) + sample_size - 1) // sample_size

        def _segment_indices(i: int):
            start = i * sample_size
            stop = start + sample_size
            return dld_highVoltage_peak[start:stop], dld_t_peak[start:stop]

        segments = [_segment_indices(i) for i in range(num_segments)]
    elif mode == 'voltage':
        v_min = np.min(dld_highVoltage_peak)
        v_max = np.max(dld_highVoltage_peak)
        # Ceildiv on the voltage range; sample_size is the V step in this
        # mode. Need to handle zero-range data (all the same voltage)
        # explicitly so we don't underflow to zero segments.
        v_range = float(v_max - v_min)
        if v_range <= 0:
            num_segments = 1
        else:
            num_segments = max(1, int(np.ceil(v_range / float(sample_size))))

        def _segment_by_voltage(i: int):
            lo = v_min + i * sample_size
            hi = v_min + (i + 1) * sample_size
            mask = np.logical_and(dld_highVoltage_peak >= lo, dld_highVoltage_peak < hi)
            return dld_highVoltage_peak[mask], dld_t_peak[mask]

        segments = [_segment_by_voltage(i) for i in range(num_segments)]
    else:
        segments = []

    def _evaluate_segment(segment):
        v_selected, t_selected = segment
        # Empty trailing segments are common when sample_size doesn't divide
        # the row count evenly; the original code returned NaN here, which
        # then broke curve_fit downstream. Drop them.
        if v_selected.size == 0 or t_selected.size == 0:
            return None
        if sample_range_max == 'histogram':
            try:
                bins, _ = _build_histogram_bins(t_selected, bin_size)
                y, x = np.histogram(t_selected, bins=bins)
                peaks, properties = find_peaks(y, height=0)
                # find_peaks returns an empty array when no local maxima
                # exist; np.argmax on the empty 'peak_heights' raises
                # ValueError. Make this branch explicit so we don't rely on
                # an exception handler for normal flow.
                if peaks.size == 0:
                    raise ValueError("no peaks in segment")
                index_peak_max_ini = np.argmax(properties['peak_heights'])
                max_peak = int(peaks[index_peak_max_ini])
                # ``x`` is the edges array (length n_bins+1); index it via
                # the bin CENTER, not the left edge.
                peak_center = 0.5 * (x[max_peak] + x[max_peak + 1])
                t_value = peak_center / maximum_location
                mask_v = np.logical_and(
                    t_selected >= peak_center - bin_size,
                    t_selected <= peak_center + bin_size,
                )
                if mode == 'ion_seq' and v_selected[mask_v].size == 0:
                    mask_v = np.logical_and(
                        t_selected >= peak_center - 2 * bin_size,
                        t_selected <= peak_center + 2 * bin_size,
                    )
                    if v_selected[mask_v].size == 0:
                        mask_v = np.logical_and(
                            t_selected >= peak_center - 4 * bin_size,
                            t_selected <= peak_center + 4 * bin_size,
                        )
                v_value = float(np.mean(v_selected[mask_v]))
            except ValueError:
                t_value = float(np.mean(t_selected)) / maximum_location
                v_value = float(np.mean(v_selected))
        elif sample_range_max == 'mean':
            t_value = float(np.mean(t_selected)) / maximum_location
            v_value = float(np.mean(v_selected))
        elif sample_range_max == 'median':
            t_value = float(np.median(t_selected)) / maximum_location
            v_value = float(np.median(v_selected))
        else:
            return None
        return t_value, v_value

    pairs = parallel_map(_evaluate_segment, segments, gil_releasing=True)
    dld_t_peak_list = [pair[0] for pair in pairs if pair is not None]
    high_voltage_mean_list = [pair[1] for pair in pairs if pair is not None]

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
            save_figure(
                fig1,
                directory=variables.result_path,
                stem=f"vol_corr_{figname}_{index_fig}",
                formats=("pdf", "png"),
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
    refine_nelder_mead=False,
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
    # Match the ceildiv used inside ``voltage_correction`` so the printed
    # count agrees with the actual number of (V, t) segments the fit sees.
    _printed_segments = (len(dld_highVoltage_peak_v) + sample_size - 1) // sample_size
    print('The number of samples is:', int(_printed_segments))

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

    # Optional per-stage Nelder-Mead refinement of the voltage polynomial,
    # adapted from APyT's `optimize_correction(mode='voltage')`
    # (sebi-85/apyt: apyt/spectrum/align.py). Only the curve_fit model exposes
    # interpretable polynomial coefficients; for other models the flag is a
    # no-op and a notice is printed.
    if refine_nelder_mead:
        if model == 'curve_fit':
            voltage_peak = np.asarray(dld_highVoltage_peak_v, dtype=float)
            peak_raw = np.asarray(dld_peak_b, dtype=float)
            sqrt_cal = calibration_mode == 'tof'

            def _apply_voltage_correction(coeffs):
                factor = voltage_corr(voltage_peak, *coeffs)
                factor = np.where(factor > 0, factor, np.nan)
                if sqrt_cal:
                    factor = np.sqrt(factor)
                return peak_raw / factor

            fitresult, refine_info = refine_correction_nelder_mead(
                np.asarray(fitresult, dtype=float),
                _apply_voltage_correction,
                peak_raw,
                fwhm_bin_size=float(bin_size) if float(bin_size) > 0 else 0.01,
            )
            print(
                'Nelder-Mead voltage refine: status=%s baseline_FWHM=%.6f refined_FWHM=%.6f'
                % (refine_info['status'], refine_info['baseline_fwhm'], refine_info['refined_fwhm'])
            )
            if refine_info['accepted']:
                print('Refined voltage coefficients:', fitresult)
        else:
            print(
                f"Nelder-Mead voltage refine requested but model='{model}' has no polynomial coefficients; skipping."
            )

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
            save_figure(
                fig1,
                directory=variables.result_path,
                stem=f"vol_corr_{index_fig}",
                formats=("pdf", "png"),
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
            save_figure(
                fig1,
                directory=variables.result_path,
                stem=f"peak_tof_V_corr_{index_fig}",
                formats=("pdf", "png"),
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
    # ``bins`` is the edges array (length n_bins+1); ``peaks`` indexes the
    # counts array. Use the bin CENTER, not the left edge.
    peak_idx = int(peaks[index_peak_max_ini])
    peak_center = 0.5 * (bins[peak_idx] + bins[peak_idx + 1])
    return float(peak_center) / maximum_location

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
    # The inner work for each cell is a histogram + peak find + mean / median
    # call on a small subarray. Every cell is independent of every other, and
    # the inner kernel is NumPy/SciPy (GIL-releasing), so this is the textbook
    # case for parallel_map(gil_releasing=True). On a 80 mm detector at 2 mm
    # sampling we get O(2000) cells per call -- typically 4-8x speedup with
    # threads, automatic serial fallback for tiny inputs.
    if sampling_mode == 'polar':
        radial_distance = np.hypot(dld_x, dld_y)
        polar_angle = np.mod(np.arctan2(dld_y, dld_x), 2.0 * np.pi)
        cells = list(_iter_polar_cells(radial_distance, sample_size, det_diam))

        def _eval_polar_cell(bounds):
            r_min_i, r_max_i, theta_min, theta_max = bounds
            mask = (
                (radial_distance >= r_min_i)
                & (radial_distance < r_max_i)
                & (polar_angle >= theta_min)
                & (polar_angle < theta_max)
            )
            if not np.any(mask):
                return None
            cell_t = dld_t[mask]
            normalized_value = _cell_peak_value(cell_t, maximum_location, sample_range_max, bin_size)
            if not np.isfinite(normalized_value) or normalized_value <= 0:
                return None
            row = (
                float(np.median(dld_x[mask])),
                float(np.median(dld_y[mask])),
                normalized_value,
                float(np.mean(dld_v[mask])) if dld_v is not None else None,
            )
            return row

        results = parallel_map(_eval_polar_cell, cells, gil_releasing=True)
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
        if not np.any(valid):
            results: list = []
        else:
            cell_id = y_bin[valid] * n_cols + x_bin[valid]
            order = np.argsort(cell_id, kind='mergesort')
            x_sorted = dld_x[valid][order]
            y_sorted = dld_y[valid][order]
            t_sorted = dld_t[valid][order]
            v_sorted = dld_v[valid][order] if dld_v is not None else None
            cell_sorted = cell_id[order]
            _, starts, counts = np.unique(cell_sorted, return_index=True, return_counts=True)
            # Stays serial: cartesian cells operate on pre-sorted contiguous
            # slices so each cell is sub-millisecond. Thread dispatch overhead
            # exceeds the per-cell work, so parallelism here regresses
            # wall-clock time on every realistic dataset.
            results = []
            for start, count in zip(starts, counts):
                stop = start + count
                cell_t = t_sorted[start:stop]
                normalized_value = _cell_peak_value(cell_t, maximum_location, sample_range_max, bin_size)
                if not np.isfinite(normalized_value) or normalized_value <= 0:
                    results.append(None)
                    continue
                results.append(
                    (
                        float(np.median(x_sorted[start:stop])),
                        float(np.median(y_sorted[start:stop])),
                        normalized_value,
                        float(np.mean(v_sorted[start:stop])) if v_sorted is not None else None,
                    )
                )

    x_samples: list = []
    y_samples: list = []
    t_samples: list = []
    v_samples: list = []
    for row in results:
        if row is None:
            continue
        x_samples.append(row[0])
        y_samples.append(row[1])
        t_samples.append(row[2])
        if dld_v is not None:
            v_samples.append(row[3])

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

    # Drop partial-recovered rows (NaN x_det or y_det). The bowl
    # polynomial fit cannot ingest NaN coordinates -- a single partial
    # ion poisons every coefficient. Partials retain their uncorrected
    # ``t (ns)`` downstream because the apply step uses an array-wise
    # mask elsewhere.
    _finite_xy = np.isfinite(np.asarray(dld_x_bowl, dtype=float)) & np.isfinite(np.asarray(dld_y_bowl, dtype=float))
    if not _finite_xy.all():
        _n_dropped = int((~_finite_xy).sum())
        print(
            f'[bowl_correction] Excluding {_n_dropped} partial-recovered rows '
            '(NaN x_det / y_det) from the fit; they keep their uncorrected t.'
        )
        dld_x_bowl = np.asarray(dld_x_bowl, dtype=float)[_finite_xy]
        dld_y_bowl = np.asarray(dld_y_bowl, dtype=float)[_finite_xy]
        dld_t_bowl = np.asarray(dld_t_bowl, dtype=float)[_finite_xy]

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
            save_figure(
                fig,
                directory=variables.result_path,
                stem=f"bowl_corr_{index_fig}",
                formats=("pdf", "png"),
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
    refine_nelder_mead=False,
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

    # Optional per-stage Nelder-Mead refinement of the bowl polynomial,
    # adapted from APyT's `optimize_correction(mode='flight')`
    # (sebi-85/apyt: apyt/spectrum/align.py). Only the standard 6-parameter
    # curve_fit model is refined here; radial/RANSAC/ML variants are skipped
    # with a notice.
    if refine_nelder_mead:
        coeffs_array = np.asarray(parameters, dtype=float).ravel() if not isinstance(parameters, Mapping) else None
        if fit_mode == 'curve_fit' and coeffs_array is not None and coeffs_array.size == 6:
            x_peak = np.asarray(dld_x_peak, dtype=float)
            y_peak = np.asarray(dld_y_peak, dtype=float)
            peak_raw = np.asarray(dld_peak, dtype=float)

            def _apply_bowl_correction(coeffs):
                factor = bowl_corr([x_peak, y_peak], *coeffs)
                factor = np.where(factor > 0, factor, np.nan)
                return peak_raw / factor

            refined_coeffs, refine_info = refine_correction_nelder_mead(
                coeffs_array,
                _apply_bowl_correction,
                peak_raw,
                fwhm_bin_size=float(bin_size) if float(bin_size) > 0 else 0.01,
            )
            print(
                'Nelder-Mead bowl refine: status=%s baseline_FWHM=%.6f refined_FWHM=%.6f'
                % (refine_info['status'], refine_info['baseline_fwhm'], refine_info['refined_fwhm'])
            )
            if refine_info['accepted']:
                parameters = refined_coeffs
                print('Refined bowl coefficients:', parameters)
        else:
            print(
                f"Nelder-Mead bowl refine requested but fit_mode='{fit_mode}' has no 6-param polynomial coefficients; skipping."
            )

    mask_fv = np.ones_like(dld_x, dtype=bool)
    # Partial-recovered rows (NaN x_det / y_det) cannot have a
    # position-dependent correction applied -- predicting bowl(NaN,NaN)
    # would return NaN and we'd lose the row's uncalibrated t/mc. Skip
    # them in the apply step so their value passes through unchanged.
    _finite_xy_apply = np.isfinite(dld_x) & np.isfinite(dld_y)
    mask_fv = mask_fv & _finite_xy_apply

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
            save_figure(
                fig1,
                directory=variables.result_path,
                stem=f"peak_tof_bowl_corr_{index_fig}",
                formats=("pdf", "png"),
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
            save_figure(
                fig1,
                directory=variables.result_path,
                stem=f"peak_tof_bowl_corr_p_x_det_{index_fig}",
                formats=("pdf", "png"),
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
            save_figure(
                fig1,
                directory=variables.result_path,
                stem=f"peak_tof_bowl_corr_p_y_det_{index_fig}",
                formats=("pdf", "png"),
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
            save_figure(
                fig,
                directory=variables.result_path,
                stem=f"peak_tof_bowl_corr_3d_{index_fig}",
                formats=("pdf", "png"),
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
    hist_edges, _n_bins = _build_histogram_bins(trimmed, max(hist_bin_size, 1e-6))
    # fast_histogram.histogram1d is ~10x faster than np.histogram for
    # evenly-spaced bins and dominates a large fraction of peak-detection
    # time when ``calibration_array`` is in the millions.
    hist_y = fast_histogram.histogram1d(
        trimmed, bins=int(_n_bins),
        range=(float(hist_edges[0]), float(hist_edges[-1])),
    )
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

def auto_detect_reference_peaks(
    calibration_array, n_peaks=6, prominence=100, distance=500, hist_bin_size=0.1,
    event_mask=None,
):
    """Public helper for stable multi-peak evaluation windows used by auto calibration.

    ``event_mask`` (optional) is a boolean array the same length as
    ``calibration_array``: True keeps the ion, False ignores it. Use this
    to detect peaks on calibration-trustworthy ions only (single-hit /
    excluded-correlations subsets from
    ``pyccapt.calibration.core.event_filters``). Default ``None``
    preserves legacy behavior.
    """
    arr = calibration_array
    if event_mask is not None:
        mask = np.asarray(event_mask, dtype=bool)
        arr_np = np.asarray(arr)
        if mask.size == arr_np.size:
            arr = arr_np[mask]
    return _auto_detect_peaks(
        arr,
        n_peaks=n_peaks,
        prominence=prominence,
        distance=distance,
        hist_bin_size=hist_bin_size,
    )


def recompute_peak_window(
    variables,
    calibration_mode: str = 'mc',
    *,
    prominence: float = 100,
    distance: int = 10,
    hist_bin_size: float = 0.05,
    lim=None,
    window_inflate: float = 2.0,
    target_position=None,
) -> tuple:
    """Lightweight peak-window recompute. NO Matplotlib.

    Detects peaks via ``auto_detect_reference_peaks`` on the currently
    calibrated array and updates ``variables.selected_x1/x2`` to the
    dominant peak (or, if ``target_position`` is set, the peak closest
    to that mass).

    This is the headless equivalent of ``mc_plot.hist_plot(...,
    plot_show=False)`` for use inside auto-calibration loops where the
    Matplotlib stack adds significant per-iteration overhead.

    Parameters
    ----------
    variables : Variables
        Calibration state container.
    calibration_mode : {'mc', 'tof'}
    prominence, distance, hist_bin_size : passed to peak detection.
    lim : float, optional
        Upper limit on values considered. Defaults to ``variables.max_tof``
        for 'tof' and 400.0 for 'mc'.
    window_inflate : float
        Multiplier on the FWHM-based window width. 2.0 keeps the window
        wide enough for stable MRP scoring on tight peaks.
    target_position : float, optional
        If supplied, pick the peak nearest this m/c instead of the most
        prominent one. Useful for keeping the same physical peak window
        across calibration iterations as the peak drifts.

    Returns
    -------
    (x1, x2, position) : tuple of floats
    """
    mode = ensure_choice(calibration_mode, field_name='calibration_mode',
                         allowed=CALIBRATION_MODES)
    arr = np.asarray(variables.get_calibration_array(mode), dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if lim is None:
        lim = float(variables.max_tof) if mode == 'tof' else 400.0
    arr = arr[arr < lim]
    if arr.size < 50:
        raise CalibrationInputError(
            "Not enough ions for recompute_peak_window"
        )

    try:
        peaks = auto_detect_reference_peaks(
            arr, n_peaks=8, prominence=float(prominence),
            distance=int(distance), hist_bin_size=float(hist_bin_size),
        )
    except CalibrationInputError:
        peaks = []

    if not peaks:
        # Fallback: histogram-max
        hist, edges = np.histogram(arr, bins=1000)
        idx = int(np.argmax(hist))
        center = 0.5 * (edges[idx] + edges[idx + 1])
        width = (edges[-1] - edges[0]) * 0.02
        x1, x2 = center - width, center + width
        position = center
    else:
        if target_position is not None:
            best = min(peaks, key=lambda p: abs(float(p['position']) - float(target_position)))
        else:
            best = max(peaks, key=lambda p: float(p.get('prominence', 0.0)))
        x1 = float(best['x1'])
        x2 = float(best['x2'])
        position = float(best['position'])
        if window_inflate != 1.0:
            half = (x2 - x1) * 0.5 * float(window_inflate)
            x1 = position - half
            x2 = position + half

    variables.selected_x1 = float(x1)
    variables.selected_x2 = float(x2)
    try:
        variables.set_calibration_peak_range(mode, float(x1), float(x2))
    except Exception:
        pass
    return float(x1), float(x2), float(position)

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

    # Track partial-recovered rows so the fit excludes them but the
    # apply step preserves their values.
    _finite_xy_mp = np.isfinite(dld_x_mm) & np.isfinite(dld_y_mm)
    if not _finite_xy_mp.all():
        print(
            f'[multi_peak_bowl_corr_main] Excluding {int((~_finite_xy_mp).sum())} '
            'partial-recovered rows (NaN x_det / y_det) from peak-sample collection.'
        )

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
        mask = (calib_arr > x1) & (calib_arr < x2) & _finite_xy_mp
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

    # Apply only on position-capable rows. Partials keep their
    # uncalibrated value rather than being divided by NaN.
    source = variables.dld_t_calib if calibration_mode == 'tof' else variables.mc_calib
    calibration_mc_tof = np.array(source, dtype=float, copy=True)
    if _finite_xy_mp.any():
        f_bowl_subset = np.clip(
            _predict_bowl_model(fit_mode, parameters, dld_x_mm[_finite_xy_mp], dld_y_mm[_finite_xy_mp]),
            np.finfo(float).eps,
            None,
        )
        calibration_mc_tof[_finite_xy_mp] = calibration_mc_tof[_finite_xy_mp] / f_bowl_subset

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

    # Track partial rows so they're excluded from the joint fit but
    # their values pass through the apply step unchanged.
    _finite_xy_jv = np.isfinite(dld_x_mm) & np.isfinite(dld_y_mm)
    if not _finite_xy_jv.all():
        print(
            f'[joint_voltage_bowl_corr_main] Excluding {int((~_finite_xy_jv).sum())} '
            'partial-recovered rows (NaN x_det / y_det) from peak-sample collection.'
        )

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
        mask = (calib_arr > peak['x1']) & (calib_arr < peak['x2']) & _finite_xy_jv
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

    # Apply only on position-capable rows. Partials keep their
    # uncalibrated value rather than being multiplied by NaN features.
    source = variables.dld_t_calib if calibration_mode == 'tof' else variables.mc_calib
    corrected = np.array(source, dtype=float, copy=True)
    if _finite_xy_jv.any():
        sub_features = _joint_feature_matrix(
            dld_highVoltage[_finite_xy_jv],
            dld_x_mm[_finite_xy_jv],
            dld_y_mm[_finite_xy_jv],
            voltage_center,
            voltage_scale,
            spatial_scale,
        )
        sub_correction = np.clip(sub_features @ parameters, np.finfo(float).eps, None)
        corrected[_finite_xy_jv] = corrected[_finite_xy_jv] / sub_correction

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
