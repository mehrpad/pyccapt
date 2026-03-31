"""
Physics-Constrained Iterative Co-Calibration (Joint ToF + m/c).

This module implements a novel calibration method that simultaneously
optimizes voltage and bowl corrections by working in both time-of-flight
(ToF) and mass-to-charge (m/c) domains.

Algorithm outline
-----------------
1. Apply initial t0 calibration (existing pipeline).
2. Convert raw ToF to uncalibrated m/c.
3. Identify reference peaks in **both** domains simultaneously
   (dual-space peak lock).
4. Joint optimization of voltage (f_v) and bowl (f_bowl) corrections,
   minimizing a combined loss that penalizes peak width in ToF space
   while constraining peak-position consistency in m/c space.
5. Apply the resulting correction through a single, consistent path.
"""

from __future__ import annotations

from copy import copy
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.signal import find_peaks, peak_prominences, peak_widths

from pyccapt.calibration.core.exceptions import (
    CalibrationInputError,
    CalibrationStateError,
)
from pyccapt.calibration.core.validation import ensure_choice, ensure_positive


# ---------------------------------------------------------------------------
# Physical constants (identical to mc_tools / tof_tools)
# ---------------------------------------------------------------------------
_E_CHARGE = 1.6e-19       # C
_AMU = 1.66e-27            # kg / Da
_ALPHA = 1.015             # pulse amplification factor
_BETA = 0.7                # pulse timing factor

_MIN_IONS_PER_PEAK = 50
_MIN_PEAK_COUNT = 2


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _effective_flight_path(x_det_cm, y_det_cm, flight_path_mm):
    """Return per-ion effective flight path length in metres."""
    x_m = np.asarray(x_det_cm, dtype=float) * 1e-2
    y_m = np.asarray(y_det_cm, dtype=float) * 1e-2
    l_m = float(flight_path_mm) * 1e-3
    return np.sqrt(x_m ** 2 + y_m ** 2 + l_m ** 2)


def _tof_to_mc(tof_ns, voltage, flight_path_m):
    """Convert ToF (ns) to m/c (Da) using simple idealized formula."""
    t_s = np.asarray(tof_ns, dtype=float) * 1e-9
    v = np.asarray(voltage, dtype=float)
    l_m = np.asarray(flight_path_m, dtype=float)
    mc_kg_per_c = 2.0 * _E_CHARGE * v * (t_s / l_m) ** 2
    return mc_kg_per_c / _AMU


def _mc_to_tof(mc_da, voltage, flight_path_m):
    """Convert m/c (Da) to ToF (ns) using simple idealized formula."""
    mc = np.asarray(mc_da, dtype=float)
    v = np.asarray(voltage, dtype=float)
    l_m = np.asarray(flight_path_m, dtype=float)
    t_s = np.sqrt(mc * _AMU * l_m ** 2 / (2.0 * _E_CHARGE * v))
    return t_s * 1e9


def _histogram_peak_center(arr, bin_size):
    """Return the histogram-mode of *arr* using the given bin width."""
    arr = arr[np.isfinite(arr)]
    if arr.size < 10:
        return float('nan')
    lo, hi = float(np.min(arr)), float(np.max(arr))
    span = hi - lo
    if span <= 0 or not np.isfinite(span):
        return float('nan')
    n_bins = max(5, int(np.ceil(span / max(bin_size, 1e-8))))
    n_bins = min(n_bins, 10000)  # cap to avoid memory issues
    counts, edges = np.histogram(arr, bins=n_bins)
    idx = int(np.argmax(counts))
    return float((edges[idx] + edges[idx + 1]) * 0.5)


def _peak_fwhm(arr, bin_size):
    """Estimate full-width at half-maximum from a 1-D sample."""
    arr = arr[np.isfinite(arr)]
    if arr.size < 20:
        return float('nan'), float('nan')
    lo, hi = float(np.min(arr)), float(np.max(arr))
    span = hi - lo
    if span <= 0 or not np.isfinite(span):
        return float('nan'), float('nan')
    n_bins = max(10, int(np.ceil(span / max(bin_size, 1e-8))))
    n_bins = min(n_bins, 10000)  # cap to avoid memory issues
    counts, edges = np.histogram(arr, bins=n_bins)
    centers = (edges[:-1] + edges[1:]) * 0.5
    peak_idx = int(np.argmax(counts))
    half_max = float(counts[peak_idx]) * 0.5
    # walk left
    left = centers[peak_idx]
    for i in range(peak_idx, -1, -1):
        if counts[i] <= half_max:
            left = float(centers[i])
            break
    # walk right
    right = centers[peak_idx]
    for i in range(peak_idx, len(counts)):
        if counts[i] <= half_max:
            right = float(centers[i])
            break
    fwhm = max(right - left, float(bin_size))
    return fwhm, float(centers[peak_idx])


# ---------------------------------------------------------------------------
# Step 3 – Dual-space peak detection
# ---------------------------------------------------------------------------

def _detect_peaks_1d(arr, n_peaks, prominence_threshold, distance, bin_size):
    """Detect up to *n_peaks* prominent peaks in a 1-D array."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < _MIN_IONS_PER_PEAK:
        return []
    lo = float(np.percentile(arr, 0.1))
    hi = float(np.percentile(arr, 99.9))
    trimmed = arr[(arr >= lo) & (arr <= hi)]
    if trimmed.size < _MIN_IONS_PER_PEAK:
        return []

    n_bins = max(10, int(np.ceil((hi - lo) / max(bin_size, 1e-8))))
    edges = np.linspace(lo, hi, n_bins + 1)
    counts = np.histogram(trimmed, bins=edges)[0].astype(float)
    centers = (edges[:-1] + edges[1:]) * 0.5

    # smooth
    if counts.size >= 5:
        k = min(9, counts.size if counts.size % 2 else counts.size - 1)
        k = max(3, k)
        kernel = np.ones(k) / float(k)
        counts_smooth = np.convolve(counts, kernel, mode='same')
    else:
        counts_smooth = counts

    for prom in [prominence_threshold, prominence_threshold * 0.5,
                 prominence_threshold * 0.25, max(1.0, float(np.max(counts_smooth)) * 0.05)]:
        found, _ = find_peaks(counts_smooth, prominence=max(1.0, prom),
                              distance=max(1, int(distance)), height=0)
        if len(found) > 0:
            break
    if len(found) == 0:
        return []

    prom_vals = peak_prominences(counts_smooth, found)[0]
    pw = peak_widths(counts_smooth, found, rel_height=0.5)
    order = np.argsort(prom_vals)[::-1]

    results = []
    for pi in order[:n_peaks * 2]:
        pos = float(centers[found[pi]])
        left = float(np.interp(pw[2][pi], np.arange(len(centers)), centers))
        right = float(np.interp(pw[3][pi], np.arange(len(centers)), centers))
        width = max(right - left, bin_size * 3)
        margin = max(bin_size * 2, width * 0.5)
        x1 = max(lo, left - margin)
        x2 = min(hi, right + margin)
        if x2 <= x1:
            continue
        cand = {'position': pos, 'x1': x1, 'x2': x2,
                'prominence': float(prom_vals[pi]), 'width': width}
        if any(not (cand['x2'] <= e['x1'] or e['x2'] <= cand['x1']) for e in results):
            continue
        results.append(cand)
        if len(results) >= n_peaks:
            break
    return results


def dual_space_peak_detection(
    tof_array,
    mc_array,
    voltage,
    flight_path_m,
    n_peaks=6,
    prominence=100,
    distance=500,
    bin_size_tof=1.0,
    bin_size_mc=0.1,
):
    """
    Detect reference peaks visible in **both** ToF and m/c domains.

    For each m/c peak the function checks whether ions falling inside that
    peak also cluster in ToF space.  Only peaks confirmed in both domains
    are returned, providing a more robust reference set than single-space
    detection alone.

    Parameters
    ----------
    tof_array : array-like
        Calibrated (or t0-corrected) time-of-flight values in ns.
    mc_array : array-like
        Corresponding mass-to-charge values in Da.
    voltage : array-like
        Per-ion high voltage in V.
    flight_path_m : array-like
        Per-ion effective flight path in m.
    n_peaks, prominence, distance : int
        Peak-detection parameters forwarded to the 1-D detector.
    bin_size_tof, bin_size_mc : float
        Histogram bin widths for ToF and m/c histograms respectively.

    Returns
    -------
    list[dict]
        Each entry contains ``mc_position``, ``mc_x1``, ``mc_x2``,
        ``tof_position``, ``tof_x1``, ``tof_x2``, ``n_ions``, and
        ``tof_fwhm``, ``mc_fwhm``.
    """
    tof_arr = np.asarray(tof_array, dtype=float)
    mc_arr = np.asarray(mc_array, dtype=float)
    volt = np.asarray(voltage, dtype=float)
    fp = np.asarray(flight_path_m, dtype=float)

    # Detect peaks in m/c space (primary domain)
    mc_peaks = _detect_peaks_1d(mc_arr, n_peaks * 2, prominence, distance, bin_size_mc)
    # Detect peaks in ToF space
    tof_peaks = _detect_peaks_1d(tof_arr, n_peaks * 2, prominence, distance, bin_size_tof)

    matched = []
    for mc_pk in mc_peaks:
        mask = (mc_arr > mc_pk['x1']) & (mc_arr < mc_pk['x2'])
        n_ions = int(np.sum(mask))
        if n_ions < _MIN_IONS_PER_PEAK:
            continue
        peak_tof = tof_arr[mask]
        tof_median = float(np.median(peak_tof))

        # Check whether a ToF peak bracket contains this median
        confirmed = False
        for tof_pk in tof_peaks:
            if tof_pk['x1'] <= tof_median <= tof_pk['x2']:
                confirmed = True
                break

        if not confirmed:
            # Fallback: check if peak_tof has low relative spread
            tof_fwhm_val, tof_center = _peak_fwhm(peak_tof, bin_size_tof)
            if np.isfinite(tof_fwhm_val) and tof_center > 0:
                relative_width = tof_fwhm_val / tof_center
                if relative_width < 0.05:
                    confirmed = True

        if not confirmed:
            continue

        tof_fwhm_val, tof_center = _peak_fwhm(peak_tof, bin_size_tof)
        mc_ions = mc_arr[mask]
        mc_fwhm_val, mc_center = _peak_fwhm(mc_ions, bin_size_mc)

        # Determine ToF window from the ions in this peak
        tof_q1, tof_q3 = float(np.percentile(peak_tof, 5)), float(np.percentile(peak_tof, 95))
        tof_margin = max(bin_size_tof * 2, (tof_q3 - tof_q1) * 0.3)

        matched.append({
            'mc_position': mc_pk['position'],
            'mc_x1': mc_pk['x1'],
            'mc_x2': mc_pk['x2'],
            'mc_fwhm': float(mc_fwhm_val) if np.isfinite(mc_fwhm_val) else 0.0,
            'tof_position': float(tof_center) if np.isfinite(tof_center) else tof_median,
            'tof_x1': tof_q1 - tof_margin,
            'tof_x2': tof_q3 + tof_margin,
            'tof_fwhm': float(tof_fwhm_val) if np.isfinite(tof_fwhm_val) else 0.0,
            'n_ions': n_ions,
        })
        if len(matched) >= n_peaks:
            break

    return matched


# ---------------------------------------------------------------------------
# Step 4 – Joint optimization helpers
# ---------------------------------------------------------------------------

def _build_correction_feature_matrix(voltage, x_det_mm, y_det_mm, v_center, v_scale, s_scale):
    """Build a 12-column joint voltage+spatial feature matrix (same as existing)."""
    v = (np.asarray(voltage, dtype=float) - v_center) / v_scale
    x = np.asarray(x_det_mm, dtype=float) / s_scale
    y = np.asarray(y_det_mm, dtype=float) / s_scale
    r2 = x ** 2 + y ** 2
    return np.column_stack([
        np.ones(len(v)),
        v, v ** 2,
        x, y,
        x ** 2, y ** 2, x * y,
        r2, v * r2, v * x, v * y,
    ])


def _joint_tof_mc_loss(
    params,
    tof_corrected,
    mc_uncalibrated,
    feature_matrix,
    matched_peaks,
    tof_weight,
    mc_weight,
    bin_size_tof,
    bin_size_mc,
):
    """
    Combined loss function for the joint ToF + m/c optimization.

    The loss has two terms:

    * **ToF term** – normalised FWHM of each peak in ToF space after
      applying the parametric correction.  Smaller FWHM means sharper
      peaks.
    * **m/c term** – variance of each peak centre in m/c space across
      voltage sub-groups, penalising inconsistent peak positions.

    Returns a scalar loss value.
    """
    correction = feature_matrix @ params
    # Clamp to avoid division by zero or negative corrections
    correction = np.clip(correction, np.finfo(float).eps, None)

    corrected_tof = tof_corrected / np.sqrt(correction)
    corrected_mc = mc_uncalibrated / correction

    loss_tof = 0.0
    loss_mc = 0.0
    n_valid = 0

    for pk in matched_peaks:
        mc_mask = (mc_uncalibrated > pk['mc_x1']) & (mc_uncalibrated < pk['mc_x2'])
        if np.sum(mc_mask) < _MIN_IONS_PER_PEAK:
            continue

        # ToF quality: normalised FWHM
        peak_tof = corrected_tof[mc_mask]
        fwhm_tof, center_tof = _peak_fwhm(peak_tof, bin_size_tof)
        if np.isfinite(fwhm_tof) and center_tof > 0:
            loss_tof += fwhm_tof / center_tof

        # m/c consistency: split by voltage quartiles, measure variance of peak centres
        peak_mc = corrected_mc[mc_mask]
        peak_v = feature_matrix[mc_mask, 1]  # normalised voltage column
        mc_center = _histogram_peak_center(peak_mc, bin_size_mc)
        if not np.isfinite(mc_center) or mc_center <= 0:
            n_valid += 1
            continue

        # Variance of peak centres across voltage sub-groups
        v_median = float(np.median(peak_v))
        lo_mask = peak_v <= v_median
        hi_mask = peak_v > v_median
        centres = []
        for sub_mask in [lo_mask, hi_mask]:
            if np.sum(sub_mask) >= 10:
                c = _histogram_peak_center(peak_mc[sub_mask], bin_size_mc)
                if np.isfinite(c):
                    centres.append(c)
        if len(centres) >= 2:
            spread = float(np.std(centres))
            loss_mc += spread / mc_center

        n_valid += 1

    if n_valid == 0:
        return 1e12

    return tof_weight * loss_tof / n_valid + mc_weight * loss_mc / n_valid


# ---------------------------------------------------------------------------
# Step 5 – Main public API
# ---------------------------------------------------------------------------

def joint_tof_mc_calibration(
    variables,
    flight_path_length,
    t0=0.0,
    det_diam=50.0,
    pulse_mode='voltage',
    n_peaks=6,
    prominence=100,
    distance=500,
    bin_size_mc=0.1,
    bin_size_tof=1.0,
    max_iterations=10,
    convergence_tol=1e-4,
    tof_weight=0.7,
    mc_weight=0.3,
    sample_size=9,
    sampling_mode='polar',
    verbose=True,
):
    """
    Physics-Constrained Iterative Co-Calibration.

    Performs a joint ToF + m/c calibration that identifies reference peaks
    in both domains and iteratively optimizes voltage and bowl corrections.

    Parameters
    ----------
    variables : Variables
        Shared calibration state.  Must have ``dld_t``, ``dld_high_voltage``,
        ``dld_x_det``, ``dld_y_det``, and optionally ``dld_pulse_v`` populated.
    flight_path_length : float
        Nominal flight-path length in **mm**.
    t0 : float
        Time-zero offset in **ns** (default 0).
    det_diam : float
        Detector diameter in **mm** (default 50).
    pulse_mode : str
        ``'voltage'`` or ``'laser'``.
    n_peaks : int
        Maximum number of dual-space reference peaks.
    prominence, distance : int
        Peak-detection sensitivity parameters.
    bin_size_mc, bin_size_tof : float
        Histogram bin widths for the two domains.
    max_iterations : int
        Maximum number of optimization iterations.
    convergence_tol : float
        Stop when relative loss improvement falls below this threshold.
    tof_weight, mc_weight : float
        Weights for the two loss terms (should sum to 1).
    sample_size, sampling_mode : int, str
        Spatial sampling parameters (forwarded to correction surface).
    verbose : bool
        Print progress information.

    Returns
    -------
    dict
        Result dictionary with keys ``'matched_peaks'``, ``'parameters'``,
        ``'loss_history'``, ``'n_iterations'``, ``'feature_names'``,
        ``'voltage_center'``, ``'voltage_scale'``, ``'spatial_scale'``.

    Raises
    ------
    CalibrationInputError
        If required data arrays are missing or too short.
    CalibrationStateError
        If the optimization fails to converge to a valid solution.
    """
    # ---- validate inputs ---------------------------------------------------
    tof_weight = float(tof_weight)
    mc_weight = float(mc_weight)
    if tof_weight < 0 or mc_weight < 0:
        raise CalibrationInputError("tof_weight and mc_weight must be non-negative")
    w_sum = tof_weight + mc_weight
    if w_sum <= 0:
        raise CalibrationInputError("tof_weight + mc_weight must be positive")
    tof_weight /= w_sum
    mc_weight /= w_sum

    ensure_positive(flight_path_length, field_name="flight_path_length")
    ensure_positive(det_diam, field_name="det_diam")
    ensure_choice(pulse_mode, field_name="pulse_mode", allowed=["voltage", "laser"])

    # ---- Step 1: extract raw arrays ----------------------------------------
    raw_tof = np.asarray(variables.dld_t, dtype=float)
    voltage = np.asarray(variables.dld_high_voltage, dtype=float)
    x_det = np.asarray(variables.dld_x_det, dtype=float)
    y_det = np.asarray(variables.dld_y_det, dtype=float)

    for name, arr in [("dld_t", raw_tof), ("dld_high_voltage", voltage),
                       ("dld_x_det", x_det), ("dld_y_det", y_det)]:
        if arr.size == 0:
            raise CalibrationInputError(f"{name} is empty")
    n_ions = raw_tof.size
    if not (voltage.size == n_ions == x_det.size == y_det.size):
        raise CalibrationInputError("All input arrays must have the same length")

    pulse_v = np.zeros(n_ions, dtype=float)
    if pulse_mode == 'voltage' and hasattr(variables, 'dld_pulse_v'):
        pv = np.asarray(variables.dld_pulse_v, dtype=float)
        if pv.size == n_ions:
            pulse_v = pv

    # ---- Step 1: t0 correction ---------------------------------------------
    tof_corrected = raw_tof - float(t0)
    # Remove non-positive ToF values
    valid = tof_corrected > 0
    if np.sum(valid) < _MIN_IONS_PER_PEAK * _MIN_PEAK_COUNT:
        raise CalibrationInputError("Not enough valid ions after t0 correction")

    # ---- Step 2: compute uncalibrated m/c ----------------------------------
    flight_path_m = _effective_flight_path(x_det, y_det, flight_path_length)

    if pulse_mode == 'voltage':
        eff_voltage = _ALPHA * (voltage + _BETA * pulse_v)
    else:
        eff_voltage = voltage

    mc_uncalibrated = _tof_to_mc(tof_corrected, eff_voltage, flight_path_m)

    # ---- Step 3: dual-space peak detection ---------------------------------
    matched_peaks = dual_space_peak_detection(
        tof_corrected,
        mc_uncalibrated,
        eff_voltage,
        flight_path_m,
        n_peaks=n_peaks,
        prominence=prominence,
        distance=distance,
        bin_size_tof=bin_size_tof,
        bin_size_mc=bin_size_mc,
    )
    if len(matched_peaks) < _MIN_PEAK_COUNT:
        raise CalibrationInputError(
            f"Dual-space peak detection found only {len(matched_peaks)} "
            f"matched peaks (minimum {_MIN_PEAK_COUNT} required)"
        )

    if verbose:
        print(f"[Joint ToF+m/c] Found {len(matched_peaks)} dual-space peaks:")
        for pk in matched_peaks:
            print(
                f"  m/c={pk['mc_position']:.2f} Da  "
                f"ToF={pk['tof_position']:.1f} ns  "
                f"ions={pk['n_ions']:,}"
            )

    # ---- Step 4: joint optimization ----------------------------------------
    x_det_mm = x_det * 10.0   # cm → mm
    y_det_mm = y_det * 10.0

    voltage_center = float(np.median(eff_voltage))
    voltage_scale = max(float(np.std(eff_voltage)), np.finfo(float).eps)
    spatial_scale = max(
        float(np.nanmax(np.sqrt(x_det_mm ** 2 + y_det_mm ** 2))),
        float(det_diam) / 2.0,
        1.0,
    )

    feature_matrix = _build_correction_feature_matrix(
        eff_voltage, x_det_mm, y_det_mm,
        voltage_center, voltage_scale, spatial_scale,
    )

    # Initial parameters: identity correction (bias = 1, rest = 0)
    n_features = feature_matrix.shape[1]
    params0 = np.zeros(n_features, dtype=float)
    params0[0] = 1.0

    loss_history = []
    best_params = params0.copy()
    best_loss = _joint_tof_mc_loss(
        params0, tof_corrected, mc_uncalibrated, feature_matrix,
        matched_peaks, tof_weight, mc_weight, bin_size_tof, bin_size_mc,
    )
    loss_history.append(float(best_loss))

    if verbose:
        print(f"[Joint ToF+m/c] Initial loss: {best_loss:.6f}")

    current_params = params0.copy()

    for iteration in range(1, max_iterations + 1):
        result = minimize(
            _joint_tof_mc_loss,
            current_params,
            args=(tof_corrected, mc_uncalibrated, feature_matrix,
                  matched_peaks, tof_weight, mc_weight, bin_size_tof, bin_size_mc),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10},
        )

        candidate_params = result.x
        candidate_loss = float(result.fun)
        loss_history.append(candidate_loss)

        if verbose:
            improvement = (best_loss - candidate_loss) / max(abs(best_loss), 1e-12)
            print(
                f"[Joint ToF+m/c] Iteration {iteration}: "
                f"loss={candidate_loss:.6f}  "
                f"improvement={improvement:.2e}"
            )

        if not np.all(np.isfinite(candidate_params)):
            if verbose:
                print("[Joint ToF+m/c] Invalid parameters; stopping.")
            break

        if candidate_loss < best_loss:
            relative_improvement = (best_loss - candidate_loss) / max(abs(best_loss), 1e-12)
            best_loss = candidate_loss
            best_params = candidate_params.copy()
            current_params = candidate_params.copy()

            if relative_improvement < convergence_tol:
                if verbose:
                    print(f"[Joint ToF+m/c] Converged (improvement {relative_improvement:.2e} < {convergence_tol})")
                break
        else:
            if verbose:
                print("[Joint ToF+m/c] No improvement; stopping.")
            break

        # Re-detect peaks in corrected space for next iteration
        correction_iter = np.clip(feature_matrix @ best_params, np.finfo(float).eps, None)
        mc_iter = mc_uncalibrated / correction_iter
        tof_iter = tof_corrected / np.sqrt(correction_iter)

        new_peaks = dual_space_peak_detection(
            tof_iter, mc_iter, eff_voltage, flight_path_m,
            n_peaks=n_peaks, prominence=prominence, distance=distance,
            bin_size_tof=bin_size_tof, bin_size_mc=bin_size_mc,
        )
        if len(new_peaks) >= _MIN_PEAK_COUNT:
            matched_peaks = new_peaks
            if verbose:
                print(f"[Joint ToF+m/c]   Re-detected {len(matched_peaks)} peaks")

    # ---- Step 5: apply final correction ------------------------------------
    if not np.all(np.isfinite(best_params)):
        raise CalibrationStateError("Joint optimization produced invalid parameters")

    final_correction = np.clip(feature_matrix @ best_params, np.finfo(float).eps, None)

    # Apply to m/c (the primary calibrated quantity)
    mc_calibrated = mc_uncalibrated / final_correction

    # Apply to ToF (consistent square-root correction)
    tof_calibrated = tof_corrected / np.sqrt(final_correction)

    # Store results in variables
    variables.mc_calib = mc_calibrated
    variables.dld_t_calib = tof_calibrated

    result_dict = {
        'matched_peaks': matched_peaks,
        'parameters': [float(v) for v in best_params],
        'feature_names': [
            'bias', 'v', 'v2', 'x', 'y',
            'x2', 'y2', 'xy', 'r2', 'v_r2', 'v_x', 'v_y',
        ],
        'voltage_center': voltage_center,
        'voltage_scale': voltage_scale,
        'spatial_scale': spatial_scale,
        'loss_history': loss_history,
        'n_iterations': len(loss_history) - 1,
        'final_loss': float(best_loss),
        'n_matched_peaks': len(matched_peaks),
        'tof_weight': tof_weight,
        'mc_weight': mc_weight,
    }

    if verbose:
        print(f"[Joint ToF+m/c] Done. Final loss: {best_loss:.6f}, "
              f"iterations: {result_dict['n_iterations']}, "
              f"peaks: {len(matched_peaks)}")

    return result_dict
