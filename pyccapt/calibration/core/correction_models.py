"""Model functions used by voltage and bowl calibration workflows."""

from __future__ import annotations

import numpy as np
from collections.abc import Mapping
from scipy.optimize import curve_fit, minimize
from scipy.signal import find_peaks, peak_widths


def voltage_corr(x, a, b, c):
    """Quadratic voltage correction model."""
    return a + b * x + c * (x**2)


def bowl_corr(data_xy, a, b, c, d, e, f):
    """Quadratic bowl correction model."""
    x = data_xy[0]
    y = data_xy[1]
    return a + b * x + c * y + d * (x**2) + e * x * y + f * (y**2)


def bowl_corr_radial(data_xy, a, b, c, d, e):
    """Radial-dominant bowl correction model using r^2 as the primary term."""
    x = np.asarray(data_xy[0], dtype=float)
    y = np.asarray(data_xy[1], dtype=float)
    r2 = x**2 + y**2
    return a + b * r2 + c * (r2**2) + d * x + e * y


def robust_voltage_fit(dld_high_voltage, dld_t):
    """Perform robust polynomial fitting using a RANSAC pipeline."""
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import RANSACRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures

    x_values = dld_high_voltage.reshape(-1, 1)
    y_values = dld_t
    polynomial = PolynomialFeatures(degree=2)
    feature_count = polynomial.fit_transform(np.zeros((1, x_values.shape[1]))).shape[1]
    if x_values.shape[0] <= feature_count:
        model = make_pipeline(polynomial, LinearRegression())
    else:
        model = make_pipeline(
            polynomial,
            RANSACRegressor(
                estimator=LinearRegression(),
                min_samples=min(x_values.shape[0], feature_count),
                random_state=42,
            ),
        )
    model.fit(x_values, y_values)
    return model


def hybrid_calibration_model(dld_x, dld_y, dld_t):
    """Train a random-forest regression model for bowl correction."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    x_values = np.column_stack((dld_x, dld_y))
    y_values = 1 / dld_t
    x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(x_train, y_train)

    score = model.score(x_test, y_test)
    print(f"Machine learning model R2 score: {score:.3f}")
    return model


def robust_fit(dld_x, dld_y, dld_t, degree=2):
    """Perform robust polynomial fitting for bowl correction."""
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import RANSACRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures

    x_values = np.column_stack((dld_x, dld_y))
    y_values = dld_t
    polynomial = PolynomialFeatures(degree=degree)
    feature_count = polynomial.fit_transform(np.zeros((1, x_values.shape[1]))).shape[1]
    if x_values.shape[0] <= feature_count:
        model = make_pipeline(polynomial, LinearRegression())
    else:
        model = make_pipeline(
            polynomial,
            RANSACRegressor(
                estimator=LinearRegression(),
                min_samples=min(x_values.shape[0], feature_count),
                random_state=42,
            ),
        )
    model.fit(x_values, y_values)
    return model


def _predict_voltage_model(model_name, fitresult, voltage_values):
    """Predict voltage correction values for the selected model."""
    if model_name == "curve_fit":
        return voltage_corr(voltage_values, *fitresult)
    return fitresult.predict(voltage_values.reshape(-1, 1))


def _predict_bowl_model(fit_mode, parameters, dld_x, dld_y):
    """Predict bowl correction values for the selected fitting mode."""
    if isinstance(parameters, Mapping) and parameters.get("model") in {"radial_curve_fit", "radial_linear"}:
        coeffs = np.asarray(parameters.get("parameters", ()), dtype=float)
        if coeffs.shape[0] != 5:
            raise ValueError("Radial bowl model requires exactly 5 parameters")
        return bowl_corr_radial([dld_x, dld_y], *coeffs)

    if fit_mode == "curve_fit":
        return bowl_corr([dld_x, dld_y], *parameters)
    return parameters.predict(np.column_stack((dld_x, dld_y)))


_FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))


def _estimate_peak_fwhm(values, bin_size=0.01):
    """Estimate FWHM of a 1-D distribution by Gaussian fit, with histogram fallback.

    Returns +inf when no peak is resolvable so the optimiser treats this as a
    bad candidate (NM will steer away from it).
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 20:
        return float('inf')

    data_min = float(np.min(values))
    data_max = float(np.max(values))
    if data_max <= data_min:
        return float('inf')

    bin_size = float(max(bin_size, 1e-6))
    n_bins = max(40, int(np.ceil((data_max - data_min) / bin_size)))
    edges = np.linspace(data_min, data_max, n_bins + 1)
    try:
        import fast_histogram as _fhist
        y = _fhist.histogram1d(values, bins=int(n_bins),
                               range=(data_min, data_max))
    except ImportError:
        y, edges = np.histogram(values, bins=edges)
    if y.max() <= 0:
        return float('inf')
    x = (edges[:-1] + edges[1:]) * 0.5

    peak_idx = int(np.argmax(y))
    amp0 = float(y[peak_idx])
    mu0 = float(x[peak_idx])
    bg0 = float(np.min(y))
    sigma0 = max(bin_size * 2.0, (data_max - data_min) / 50.0)

    def _gauss(xv, amp, mu, sigma, bg):
        return amp * np.exp(-((xv - mu) ** 2) / (2.0 * max(sigma, 1e-12) ** 2)) + bg

    try:
        popt, _ = curve_fit(
            _gauss,
            x,
            y.astype(float),
            p0=[amp0 - bg0, mu0, sigma0, bg0],
            bounds=([0.0, x[0], 1e-9, 0.0], [np.inf, x[-1], data_max - data_min, np.inf]),
            maxfev=2000,
        )
        sigma_fit = float(popt[2])
        if sigma_fit > 0:
            return _FWHM_FACTOR * sigma_fit
    except (RuntimeError, ValueError):
        pass

    try:
        peaks, _ = find_peaks(y, height=max(1.0, amp0 * 0.25))
        if peaks.size == 0:
            return float('inf')
        idx = int(peaks[np.argmin(np.abs(peaks - peak_idx))])
        widths = peak_widths(y, np.array([idx]), rel_height=0.5)
        left = float(np.interp(widths[2][0], np.arange(len(x)), x))
        right = float(np.interp(widths[3][0], np.arange(len(x)), x))
        width = right - left
        return width if width > 0 else float('inf')
    except (ValueError, IndexError):
        return float('inf')


def refine_correction_nelder_mead(
    initial_coeffs,
    apply_correction_fn,
    peak_values_raw,
    *,
    fwhm_bin_size=0.01,
    max_step_ratio=0.25,
    maxiter=80,
):
    """Refine a correction polynomial by Nelder-Mead minimisation of peak FWHM.

    Adapted from APyT's `optimize_correction` (sebi-85/apyt:
    apyt/spectrum/align.py), which sequentially fine-tunes voltage and flight
    corrections by minimising the width of the reference peak. The refinement
    is conservative: each coefficient is bounded to a fractional change of
    ``max_step_ratio`` around its initial value (or an absolute floor when the
    initial value is near zero), and the result is only accepted when the
    refined FWHM is strictly smaller than the baseline.

    Parameters
    ----------
    initial_coeffs : array-like
        Starting polynomial coefficients (e.g. ``[a, b, c]`` for voltage,
        ``[a, b, c, d, e, f]`` for full-2D bowl).
    apply_correction_fn : callable
        Function ``coeffs -> corrected_peak_values`` that applies the
        candidate correction to the **reference peak ions only** and returns
        the corrected m/c (or ToF) values.
    peak_values_raw : array-like
        Raw m/c or ToF values of the reference peak ions (only used for the
        baseline FWHM measurement).
    fwhm_bin_size : float
        Bin width for the FWHM histogram (defaults to 0.01, matching the
        existing MRP routines).
    max_step_ratio : float
        Soft per-coefficient step bound, expressed as a fraction of the
        initial coefficient magnitude. Keeps NM from straying too far from
        the polynomial fit.
    maxiter : int
        Maximum NM iterations.

    Returns
    -------
    refined_coeffs : numpy.ndarray
    info : dict
        Diagnostics: ``baseline_fwhm``, ``refined_fwhm``, ``accepted``,
        ``status`` (one of ``'improved'``, ``'no_improvement'``, ``'failed'``).
    """
    initial = np.asarray(initial_coeffs, dtype=float).copy()
    baseline_fwhm = _estimate_peak_fwhm(np.asarray(peak_values_raw, dtype=float), bin_size=fwhm_bin_size)

    # Compute a per-coefficient step floor so coefficients that are exactly
    # zero (or near zero) can still move freely without violating the soft
    # ratio bound.
    step_floor = np.maximum(np.abs(initial) * float(max_step_ratio), 1e-6)

    def objective(coeffs):
        # Soft penalty when a coefficient walks too far from its initial value.
        # Keeps NM exploring around the polynomial fit rather than blowing up.
        excess = np.maximum(0.0, np.abs(coeffs - initial) - step_floor)
        penalty = float(np.sum((excess / step_floor) ** 2))
        try:
            corrected = np.asarray(apply_correction_fn(coeffs), dtype=float)
        except Exception:
            return baseline_fwhm * 10.0 + penalty
        if corrected.size == 0:
            return baseline_fwhm * 10.0 + penalty
        fwhm = _estimate_peak_fwhm(corrected, bin_size=fwhm_bin_size)
        if not np.isfinite(fwhm):
            return baseline_fwhm * 10.0 + penalty
        return fwhm + 0.1 * baseline_fwhm * penalty

    try:
        result = minimize(
            objective,
            initial,
            method='Nelder-Mead',
            options={
                'xatol': 1e-5,
                'fatol': max(baseline_fwhm * 1e-4, 1e-9),
                'maxiter': int(maxiter) * max(1, initial.size),
                'adaptive': True,
            },
        )
    except Exception:
        return initial, {
            'baseline_fwhm': baseline_fwhm,
            'refined_fwhm': baseline_fwhm,
            'accepted': False,
            'status': 'failed',
        }

    refined_coeffs = np.asarray(result.x, dtype=float)
    refined_fwhm = _estimate_peak_fwhm(
        np.asarray(apply_correction_fn(refined_coeffs), dtype=float),
        bin_size=fwhm_bin_size,
    )

    accepted = bool(
        np.isfinite(refined_fwhm)
        and np.isfinite(baseline_fwhm)
        and refined_fwhm < baseline_fwhm
    )
    return (
        refined_coeffs if accepted else initial,
        {
            'baseline_fwhm': float(baseline_fwhm),
            'refined_fwhm': float(refined_fwhm),
            'accepted': accepted,
            'status': 'improved' if accepted else 'no_improvement',
            'nm_message': str(result.message),
        },
    )


__all__ = [
    "voltage_corr",
    "bowl_corr",
    "bowl_corr_radial",
    "robust_voltage_fit",
    "hybrid_calibration_model",
    "robust_fit",
    "refine_correction_nelder_mead",
    "_predict_voltage_model",
    "_predict_bowl_model",
]
