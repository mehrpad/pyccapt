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
    - plot (bool): Indicates whether to plot the graph. Default is True.
    - save (bool): Indicates whether to save the plot. Default is False.
    - fig_size (tuple): Figure size in inches. Default is (7, 5).
    - model (string): Type of model ('curve_fit'/'robust_fit'). Default is 'curve_fit'.
    - bin_size (float): Size of the bin.

    Returns:
    - fitresult: Fit result -- the polynomial coefficient array for
      ``model='curve_fit'`` or the fitted sklearn pipeline for
      ``model='robust_fit'`` (NOT a corrected-value array).

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
            if i == num_segments - 1:
                # The final segment must be right-inclusive: a strict ``< hi``
                # drops every ion sitting exactly at v_max when the voltage
                # range divides evenly into sample_size, biasing the fit away
                # from the high-V end (where the correction matters most).
                mask = dld_highVoltage_peak >= lo
            else:
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
                x, n_bins = _build_histogram_bins(t_selected, bin_size)
                # fast_histogram.histogram1d is ~10x faster than np.histogram
                # for uniform bins and matches the rest of this module
                # (_resolve_peak_location, _cell_peak_value). ``x`` are the edges.
                y = fast_histogram.histogram1d(
                    t_selected, bins=n_bins, range=(float(x[0]), float(x[-1]))
                )
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

    # The quadratic voltage model has 3 parameters; curve_fit raises a cryptic
    # "number of func parameters=3 must not exceed the number of data points"
    # when too few segments survive (tiny peak / oversized sample_size). Fail
    # with an actionable message instead, mirroring the multi-peak variants.
    if len(high_voltage_mean_list) < 3:
        raise CalibrationInputError(
            "Not enough valid (V, t) segments for the voltage fit "
            f"({len(high_voltage_mean_list)} found, need >= 3); "
            "reduce sample_size or widen the calibration peak window."
        )

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

    f_v = _predict_voltage_model(model, fitresult, np.asarray(dld_highVoltage))
    # Guard against a poor/extrapolated quadratic predicting f_v <= 0, which
    # would make np.sqrt(f_v) NaN (tof) and poison the calibrated array on
    # division. Matches the np.clip(..., eps, None) in multi_peak_voltage_corr_main.
    f_v = np.clip(f_v, np.finfo(float).eps, None)

    if calibration_mode == 'tof':
        correction_factor = np.sqrt(f_v)
    else:
        correction_factor = f_v
    print("Maximum value of correction factor:", np.max(correction_factor))
    print("Minimum value of correction factor:", np.min(correction_factor))
    # In-place divide over the full array: the previous all-True boolean-mask
    # gather/scatter copied the whole calibrated vector twice for no selectivity.
    calibration_mc_tof /= correction_factor

    if plot or save:
        # Plot how correction factor for selected peak_x
        fig1, ax1 = plt.subplots(figsize=fig_size, constrained_layout=True)
        # Seeded, without-replacement subsample so saved diagnostic figures
        # are reproducible run-to-run (the module docstring calls this out)
        # and don't show duplicate points from sampling with replacement.
        _n_pts = len(dld_highVoltage_peak_v)
        if _n_pts > 1000:
            mask = np.random.default_rng(0).choice(_n_pts, size=1000, replace=False)
        else:
            mask = np.arange(_n_pts)
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
        fit_mode (str): Fit mode ('curve_fit', 'ml_fit', or 'robust_fit').
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
        numpy.ndarray: ``f_bowl``, the per-event bowl correction factor that
        was applied (the calibrated array is divided by it).

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

    valid_factor = np.isfinite(f_bowl) & (f_bowl > 0)
    if f_bowl.size:
        print("Maximum value of f_bowl:", np.max(f_bowl[valid_factor]) if valid_factor.any() else "n/a")
        print("Minimum value of f_bowl:", np.min(f_bowl[valid_factor]) if valid_factor.any() else "n/a")
    apply_indices = np.flatnonzero(mask_fv)[valid_factor]
    calibration_mc_tof[apply_indices] = calibration_mc_tof[apply_indices] / f_bowl[valid_factor]
    # calibration_mc_tof[mask_fv] = calibration_mc_tof[mask_fv] * f_bowl

    mean_after = np.mean(calibration_mc_tof[mask_temporal])
    print('The mean of tof/mc after bowl calibration is:', mean_after)
    print('The difference between the mean of tof before and after bowl calibration is:', mean_after - mean_before)

    if plot or save:
        # Plot how bowl correct tof/mc vs high voltage
        fig1, ax1 = plt.subplots(figsize=fig_size, constrained_layout=True)
        _n_pts = len(dld_highVoltage_peak)
        mask = np.random.default_rng(0).choice(_n_pts, size=min(10000, _n_pts), replace=False)

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
        _n_pts = len(dld_highVoltage_peak)
        mask = np.random.default_rng(0).choice(_n_pts, size=min(500, _n_pts), replace=False)
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

from pyccapt.calibration.core.calibration_advanced import (  # noqa: E402
    _auto_detect_peaks, _joint_feature_matrix, _robust_joint_linear_fit,
    auto_detect_reference_peaks, joint_voltage_bowl_corr_main,
    multi_peak_bowl_corr_main, multi_peak_voltage_corr_main, recompute_peak_window,
)
