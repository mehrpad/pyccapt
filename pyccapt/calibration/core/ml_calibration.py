"""ML-based master calibration using Gradient-Boosted Correction Surfaces.

This module provides a self-supervised calibration approach that learns a unified
correction function from the full feature space (voltage, detector position,
ion sequence) using gradient-boosted regression trees.  Training data is
extracted from automatically detected reference peaks where the ideal position
is known; the model then generalises the correction to every ion.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

from pyccapt.calibration.core.exceptions import CalibrationInputError
from pyccapt.calibration.core.mc_plot_peak_helpers import gaussian_mrp_report


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def _build_feature_matrix(voltage, x_det, y_det, ion_seq_norm):
    """Construct the feature matrix used by the GBCS model.

    Features: voltage, x_det, y_det, ion_seq_norm, r^2, V^2, V*r^2

    All inputs must be 1-D arrays of equal length.
    Returns (X, feature_names).
    """
    voltage = np.asarray(voltage, dtype=float)
    x_det = np.asarray(x_det, dtype=float)
    y_det = np.asarray(y_det, dtype=float)
    ion_seq_norm = np.asarray(ion_seq_norm, dtype=float)

    r2 = x_det**2 + y_det**2
    v2 = voltage**2
    v_r2 = voltage * r2

    X = np.column_stack([voltage, x_det, y_det, ion_seq_norm, r2, v2, v_r2])
    feature_names = ["voltage", "x_det", "y_det", "ion_seq", "r2", "V2", "V_r2"]
    return X, feature_names


def _normalise_voltage(voltage):
    """Centre and scale voltage to roughly unit variance."""
    v = np.asarray(voltage, dtype=float)
    center = np.nanmedian(v)
    scale = np.nanstd(v)
    if scale < 1e-12:
        scale = 1.0
    return (v - center) / scale, center, scale


def _normalise_spatial(x_det, y_det):
    """Scale detector coordinates to roughly unit range."""
    x = np.asarray(x_det, dtype=float)
    y = np.asarray(y_det, dtype=float)
    scale = max(np.nanstd(x), np.nanstd(y), 1e-12)
    return x / scale, y / scale, scale


# ---------------------------------------------------------------------------
# Training data extraction
# ---------------------------------------------------------------------------


def _extract_training_data(
    calibration_array,
    voltage,
    x_det,
    y_det,
    peaks,
    max_samples_per_peak=20000,
    rng=None,
):
    """Extract (features, correction_ratio) pairs from reference peaks.

    For each peak at expected position *p*, every ion within the peak window
    contributes one training sample with target ``ion_value / p`` (the
    multiplicative correction ratio that should be close to 1.0).
    """
    if rng is None:
        rng = np.random.RandomState(42)

    n = len(calibration_array)
    ion_seq_norm = np.arange(n, dtype=float) / max(n - 1, 1)

    v_norm, v_center, v_scale = _normalise_voltage(voltage)
    x_norm, y_norm, s_scale = _normalise_spatial(x_det, y_det)

    all_X = []
    all_y = []
    peak_labels = []

    for peak in peaks:
        pos = float(peak["position"])
        x1, x2 = float(peak["x1"]), float(peak["x2"])
        mask = (calibration_array > x1) & (calibration_array < x2)
        indices = np.nonzero(mask)[0]
        if indices.size < 30:
            continue

        # Subsample large peaks
        if indices.size > max_samples_per_peak:
            indices = rng.choice(indices, max_samples_per_peak, replace=False)

        X_peak, _ = _build_feature_matrix(
            v_norm[indices],
            x_norm[indices],
            y_norm[indices],
            ion_seq_norm[indices],
        )
        ratio = calibration_array[indices] / pos
        all_X.append(X_peak)
        all_y.append(ratio)
        peak_labels.extend([pos] * len(indices))

    if not all_X:
        raise CalibrationInputError("No peaks with enough ions for ML training")

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    return X, y, np.array(peak_labels), (v_center, v_scale, s_scale)


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def _train_gbcs(X, y, n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, cv_folds=3, verbose=False):
    """Train a GradientBoostingRegressor and return model + diagnostics."""
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        loss="squared_error",
        random_state=42,
    )
    model.fit(X, y)

    # Cross-validation R^2
    cv_scores = cross_val_score(
        GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            loss="squared_error",
            random_state=42,
        ),
        X,
        y,
        cv=min(cv_folds, max(2, X.shape[0] // 100)),
        scoring="r2",
    )

    diagnostics = {
        "train_r2": float(model.score(X, y)),
        "cv_r2_mean": float(np.mean(cv_scores)),
        "cv_r2_std": float(np.std(cv_scores)),
        "n_samples": int(X.shape[0]),
    }

    if verbose:
        print(f"[GBCS] Train R2: {diagnostics['train_r2']:.4f}")
        print(f"[GBCS] CV R2:    {diagnostics['cv_r2_mean']:.4f} +/- {diagnostics['cv_r2_std']:.4f}")
        print(f"[GBCS] Samples:  {diagnostics['n_samples']}")

    return model, diagnostics


# ---------------------------------------------------------------------------
# Prediction & application
# ---------------------------------------------------------------------------


def _predict_correction(model, voltage, x_det, y_det, ion_seq_norm, norm_params, clamp_low=0.9, clamp_high=1.1):
    """Predict the correction factor for every ion and clamp for safety."""
    v_center, v_scale, s_scale = norm_params
    v_norm = (np.asarray(voltage, dtype=float) - v_center) / v_scale
    x_norm = np.asarray(x_det, dtype=float) / s_scale
    y_norm = np.asarray(y_det, dtype=float) / s_scale

    X, _ = _build_feature_matrix(v_norm, x_norm, y_norm, ion_seq_norm)
    raw_pred = model.predict(X)
    return np.clip(raw_pred, clamp_low, clamp_high)


# ---------------------------------------------------------------------------
# Quality evaluation
# ---------------------------------------------------------------------------


def _evaluate_peaks(calibration_array, peaks):
    """Compute weighted MRP score across peaks (same logic as adaptive residual)."""
    scores = []
    for peak in peaks:
        local_bin = max(1e-4, min(0.02, (peak["x2"] - peak["x1"]) / 80.0))
        report = gaussian_mrp_report(
            calibration_array,
            peak["x1"],
            peak["x2"],
            bin_size=local_bin,
        )
        if report is None:
            continue
        mrp_values = report["recommended_mrp"]
        weights = (0.6, 0.3, 0.1)
        s, w = 0.0, 0.0
        for wt, val in zip(weights, mrp_values):
            if np.isfinite(val):
                s += wt * max(0.0, float(val))
                w += wt
        if w > 0:
            n_ions = int(np.count_nonzero((calibration_array > peak["x1"]) & (calibration_array < peak["x2"])))
            scores.append((s / w, max(1.0, np.sqrt(n_ions))))
    if not scores:
        return float("nan")
    total_w = sum(w for _, w in scores)
    return sum(s * w for s, w in scores) / total_w


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ml_calibration(
    variables,
    calibration_mode="mc",
    n_peaks=6,
    prominence=100,
    distance=10,
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    clamp_low=0.9,
    clamp_high=1.1,
    max_samples_per_peak=20000,
    verbose=True,
):
    """Apply gradient-boosted correction surface calibration.

    Parameters
    ----------
    variables : Variables
        Shared state container with calibration arrays and detector data.
    calibration_mode : str
        ``'mc'`` or ``'tof'``.
    n_peaks : int
        Number of reference peaks to detect for training.
    prominence, distance : float
        Peak detection parameters passed to ``auto_detect_reference_peaks``.
    n_estimators, max_depth, learning_rate, subsample : float
        GBR hyper-parameters.
    clamp_low, clamp_high : float
        Safety bounds for the predicted correction factor.
    max_samples_per_peak : int
        Maximum training samples drawn from each peak.
    verbose : bool
        Print progress information.

    Returns
    -------
    dict
        Result dictionary with keys: ``baseline_score``, ``final_score``,
        ``feature_importances``, ``diagnostics``, ``model``.
    """
    from pyccapt.calibration.core import calibration as cal
    from pyccapt.calibration.core.adaptive_residual_calibration import _mode_aware_defaults

    calibration_key = "tof" if calibration_mode == "tof" else "mc"
    cal_array = np.asarray(variables.get_calibration_array(calibration_key), dtype=float)
    voltage = np.asarray(variables.dld_high_voltage, dtype=float)
    x_det = np.asarray(variables.dld_x_det, dtype=float)
    y_det = np.asarray(variables.dld_y_det, dtype=float)

    # Mode-aware histogram bin size for peak detection
    _, hist_bin_size = _mode_aware_defaults(cal_array, calibration_mode, 0.01)

    # 1. Detect reference peaks
    peaks = cal.auto_detect_reference_peaks(
        cal_array,
        n_peaks=max(3, int(n_peaks)),
        prominence=prominence,
        distance=distance,
        hist_bin_size=hist_bin_size,
    )
    if len(peaks) < 2:
        raise CalibrationInputError("ML calibration needs at least 2 detectable peaks")

    # Split into train / holdout for quality evaluation
    train_peaks = peaks[:-1] if len(peaks) > 2 else peaks
    holdout_peaks = peaks[-1:] if len(peaks) > 2 else []

    baseline_score = _evaluate_peaks(cal_array, peaks)
    if verbose:
        print(f"[ML-GBCS] Baseline MRP score: {baseline_score:.2f}")
        print(f"[ML-GBCS] Using {len(train_peaks)} training peaks, {len(holdout_peaks)} holdout peaks")

    # 2. Extract training data
    X, y, peak_labels, norm_params = _extract_training_data(
        cal_array,
        voltage,
        x_det,
        y_det,
        train_peaks,
        max_samples_per_peak=max_samples_per_peak,
    )

    if verbose:
        print(f"[ML-GBCS] Training samples: {X.shape[0]}")

    # 3. Train model
    model, diagnostics = _train_gbcs(
        X,
        y,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        verbose=verbose,
    )

    # 4. Predict correction for all ions
    n = len(cal_array)
    ion_seq_norm = np.arange(n, dtype=float) / max(n - 1, 1)

    correction = _predict_correction(
        model,
        voltage,
        x_det,
        y_det,
        ion_seq_norm,
        norm_params,
        clamp_low=clamp_low,
        clamp_high=clamp_high,
    )

    # 5. Apply correction
    calibrated = cal_array / correction

    # 6. Evaluate improvement
    final_score = _evaluate_peaks(calibrated, peaks)

    if verbose:
        print(f"[ML-GBCS] Final MRP score:    {final_score:.2f}")
        improvement = final_score - baseline_score
        print(f"[ML-GBCS] Improvement:        {improvement:+.2f}")

        # Feature importances
        _, feature_names = _build_feature_matrix(
            np.zeros(1),
            np.zeros(1),
            np.zeros(1),
            np.zeros(1),
        )
        importances = model.feature_importances_
        ranked = sorted(zip(feature_names, importances), key=lambda x: -x[1])
        print("[ML-GBCS] Feature importances:")
        for name, imp in ranked:
            print(f"  {name:>10s}: {imp:.4f}")

    # 7. Accept or reject
    if np.isfinite(final_score) and np.isfinite(baseline_score) and final_score > baseline_score:
        if calibration_key == "tof":
            variables.dld_t_calib = calibrated
        else:
            variables.mc_calib = calibrated
        accepted = True
        if verbose:
            print("[ML-GBCS] Correction accepted (MRP improved).")
    else:
        accepted = False
        if verbose:
            print("[ML-GBCS] Correction rejected (no MRP improvement). Keeping previous calibration.")

    _, feature_names = _build_feature_matrix(
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
    )

    return {
        "baseline_score": baseline_score,
        "final_score": final_score,
        "accepted": accepted,
        "feature_importances": dict(zip(feature_names, model.feature_importances_.tolist())),
        "diagnostics": diagnostics,
        "n_peaks_used": len(train_peaks),
        "n_holdout_peaks": len(holdout_peaks),
        "model": model,
        "norm_params": norm_params,
    }
