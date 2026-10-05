"""Advanced multi-peak and joint calibration domain workflows.

This module is separated from the legacy plotting/orchestration facade so the
scientific domain code can evolve and be tested without growing calibration.py.
"""

from __future__ import annotations

import fast_histogram
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_prominences, peak_widths

from pyccapt.calibration.core.calibration import (
    _build_histogram_bins,
    _collect_spatial_samples,
    _radial_bowl_corr,
    _resolve_peak_location,
    _resolve_sampling_mode,
)
from pyccapt.calibration.core.correction_models import (
    _predict_bowl_model,
    _predict_voltage_model,
    bowl_corr,
    hybrid_calibration_model,
    robust_fit,
    robust_voltage_fit,
    voltage_corr,
)
from pyccapt.calibration.core.exceptions import CalibrationInputError
from pyccapt.calibration.core.validation import (
    BOWL_FIT_MODES,
    CALIBRATION_MODES,
    ensure_choice,
    ensure_matching_lengths,
    ensure_positive,
    normalize_voltage_model,
)

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
    elif fit_mode in {'robust_fit', 'ml_fit'}:
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
