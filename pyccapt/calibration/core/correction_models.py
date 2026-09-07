"""Model functions used by voltage and bowl calibration workflows."""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections.abc import Mapping
from scipy.optimize import curve_fit, minimize
from scipy.signal import find_peaks, peak_widths


class CalibrationModel(ABC):
    """Serializable correction-model contract used by calibration workflows."""

    @abstractmethod
    def fit(self, features, target) -> "CalibrationModel":
        """Fit and return this model."""

    @abstractmethod
    def predict_factor(self, features) -> np.ndarray:
        """Predict the multiplicative/divisive correction factor."""

    @property
    @abstractmethod
    def valid_domain(self) -> dict[str, tuple[float, float]]:
        """Finite training domain for every feature."""

    @abstractmethod
    def provenance(self) -> dict:
        """Return JSON-serializable model parameters and domain."""


@dataclass
class PolynomialCorrectionModel(CalibrationModel):
    """Least-squares polynomial model for voltage or detector position."""

    kind: str
    degree: int = 2
    coefficients: np.ndarray | None = None
    _valid_domain: dict[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in {"voltage", "bowl"}:
            raise ValueError("Polynomial correction kind must be 'voltage' or 'bowl'")
        if int(self.degree) < 0:
            raise ValueError("Polynomial degree must be non-negative")
        if self.kind == "bowl" and int(self.degree) != 2:
            raise ValueError("The bowl correction currently supports degree=2 only")

    def _design(self, features) -> np.ndarray:
        values = np.asarray(features, dtype=float)
        if self.kind == "voltage":
            x = values.reshape(-1)
            return np.vander(x, N=int(self.degree) + 1, increasing=True)
        values = np.atleast_2d(values)
        if values.shape[1] != 2:
            raise ValueError("Bowl model requires [x, y] features")
        x, y = values[:, 0], values[:, 1]
        return np.column_stack((np.ones(x.size), x, y, x**2, x * y, y**2))

    def fit(self, features, target) -> "PolynomialCorrectionModel":
        values = np.asarray(features, dtype=float)
        target_values = np.asarray(target, dtype=float).reshape(-1)
        design = self._design(values)
        if design.shape[0] != target_values.size:
            raise ValueError("Feature and target row counts must match")
        finite = np.isfinite(target_values) & np.all(np.isfinite(design), axis=1)
        if finite.sum() < design.shape[1]:
            raise ValueError("Insufficient finite rows to fit correction model")
        self.coefficients = np.linalg.lstsq(design[finite], target_values[finite], rcond=None)[0]
        domain_values = values.reshape(-1, 1) if self.kind == "voltage" else np.atleast_2d(values)
        names = ("voltage",) if self.kind == "voltage" else ("x_det", "y_det")
        self._valid_domain = {
            name: (float(np.min(domain_values[finite, i])), float(np.max(domain_values[finite, i])))
            for i, name in enumerate(names)
        }
        return self

    def predict_factor(self, features) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("Model has not been fitted")
        return self._design(features) @ self.coefficients

    @property
    def valid_domain(self) -> dict[str, tuple[float, float]]:
        return dict(self._valid_domain)

    def provenance(self) -> dict:
        return {
            "model_type": type(self).__name__,
            "kind": self.kind,
            "degree": self.degree,
            "coefficients": [] if self.coefficients is None else self.coefficients.tolist(),
            "valid_domain": {key: list(value) for key, value in self._valid_domain.items()},
        }


@dataclass
class EstimatorCorrectionModel(CalibrationModel):
    """Adapter for sklearn-like estimators using the same model contract."""

    estimator: object
    feature_names: tuple[str, ...]
    _valid_domain: dict[str, tuple[float, float]] = field(default_factory=dict)

    def fit(self, features, target) -> "EstimatorCorrectionModel":
        values = np.asarray(features, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        target_values = np.asarray(target, dtype=float).reshape(-1)
        if values.shape[0] != target_values.size:
            raise ValueError("Feature and target row counts must match")
        if values.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features {self.feature_names}, got {values.shape[1]}"
            )
        finite = np.isfinite(target_values) & np.all(np.isfinite(values), axis=1)
        if not finite.any():
            raise ValueError("No finite rows are available to fit the estimator")
        self.estimator.fit(values[finite], target_values[finite])
        self._valid_domain = {
            name: (float(np.min(values[finite, i])), float(np.max(values[finite, i])))
            for i, name in enumerate(self.feature_names)
        }
        return self

    def predict_factor(self, features) -> np.ndarray:
        values = np.asarray(features, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        return np.asarray(self.estimator.predict(values), dtype=float)

    @property
    def valid_domain(self) -> dict[str, tuple[float, float]]:
        return dict(self._valid_domain)

    def provenance(self) -> dict:
        parameters = self.estimator.get_params(deep=False) if hasattr(self.estimator, "get_params") else {}
        serializable = {key: value for key, value in parameters.items() if isinstance(value, (str, int, float, bool, type(None)))}
        return {
            "model_type": type(self).__name__,
            "estimator_type": type(self.estimator).__name__,
            "parameters": serializable,
            "valid_domain": {key: list(value) for key, value in self._valid_domain.items()},
        }


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
    # The correction is applied as value / predicted_factor, so learn the
    # normalized time/mass factor itself (not its reciprocal).
    y_values = np.asarray(dld_t, dtype=float)
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
    if isinstance(fitresult, CalibrationModel):
        return fitresult.predict_factor(voltage_values)
    if model_name == "curve_fit":
        return voltage_corr(voltage_values, *fitresult)
    return fitresult.predict(voltage_values.reshape(-1, 1))


def _predict_bowl_model(fit_mode, parameters, dld_x, dld_y):
    """Predict bowl correction values for the selected fitting mode."""
    if isinstance(parameters, CalibrationModel):
        return parameters.predict_factor(np.column_stack((dld_x, dld_y)))
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
    "CalibrationModel",
    "PolynomialCorrectionModel",
    "EstimatorCorrectionModel",
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
