"""Optional physical multi-reference ToF model; no shared-state mutation.

The fitted offset has units ns. ``predict_factor`` returns a dimensionless,
positive ToF factor, NOT a mass factor. Reference identities are user inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from pyccapt.calibration.core.correction_models import CalibrationModel
from pyccapt.calibration.core.exceptions import CalibrationInputError, CalibrationStateError


FEATURE_NAMES = ("voltage_V", "x_cm", "y_cm", "event_fraction", "baseline_ns_per_sqrt_Da")


def physical_tof_baseline(voltage, x_cm, y_cm, flight_length_mm, pulse_voltage=None, pulse_mode="laser"):
    """Ideal ns/sqrt(Da), using the constants/convention in ``mc_tools.tof2mc``."""
    if not np.isfinite(flight_length_mm) or flight_length_mm <= 0:
        raise CalibrationInputError("flight_length_mm must be positive and finite")
    voltage, x_cm, y_cm = np.broadcast_arrays(voltage, x_cm, y_cm)
    if pulse_mode == "voltage":
        if pulse_voltage is None:
            raise CalibrationInputError("Voltage mode requires event-aligned pulse voltage")
        effective = 1.015 * (voltage + .7 * np.asarray(pulse_voltage))
    elif pulse_mode == "laser":
        effective = voltage
    else:
        raise CalibrationInputError("pulse_mode must be laser or voltage")
    length_m = np.sqrt((x_cm * .01)**2 + (y_cm * .01)**2 + (flight_length_mm * .001)**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return length_m * np.sqrt(1.66e-27 / (2 * 1.6e-19 * effective)) * 1e9


@dataclass(frozen=True)
class ToFModelConfig:
    voltage_term: bool = True
    spatial_terms: bool = True
    linear_drift: bool = False
    # Fractions of acquisition event order, supplied before looking at holdouts.
    block_edges: tuple[float, ...] = ()
    t0_bounds_ns: tuple[float, float] = (-200., 200.)
    factor_bounds: tuple[float, float] = (.5, 2.)
    domain_margin_fraction: float = .02
    robust_scale_ns: float = 1.
    max_condition_number: float = 1e7

    def __post_init__(self):
        object.__setattr__(self, "block_edges", tuple(self.block_edges))
        object.__setattr__(self, "t0_bounds_ns", tuple(self.t0_bounds_ns))
        object.__setattr__(self, "factor_bounds", tuple(self.factor_bounds))
        edges = np.asarray(self.block_edges)
        if np.any(~np.isfinite(edges)) or np.any((edges <= 0) | (edges >= 1)) or np.any(np.diff(edges) <= 0):
            raise CalibrationInputError("block_edges must be increasing fractions strictly inside (0, 1)")
        for name, bounds in (("t0_bounds_ns", self.t0_bounds_ns), ("factor_bounds", self.factor_bounds)):
            if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
                raise CalibrationInputError(f"Invalid {name}")
        if self.factor_bounds[0] <= 0 or not 0 <= self.domain_margin_fraction <= .25:
            raise CalibrationInputError("Positive factor bounds and domain margin in [0, .25] required")
        if not np.isfinite(self.robust_scale_ns) or self.robust_scale_ns <= 0:
            raise CalibrationInputError("robust_scale_ns must be positive and finite")
        if not np.isfinite(self.max_condition_number) or self.max_condition_number <= 1:
            raise CalibrationInputError("max_condition_number must be finite and greater than one")


class MultiReferenceToFModel(CalibrationModel):
    """Fit t = t0 + sqrt(m/q) * physical_baseline * exp(B beta).

    Features follow ``FEATURE_NAMES``; target columns are raw ToF in ns and
    independently assigned mass-to-charge in Da. No reference is inferred here.
    """
    def __init__(self, config=None):
        self.config = config or ToFModelConfig()
        self.coefficients = None
        self._valid_domain = {}

    @staticmethod
    def _features(features):
        values = np.asarray(features, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
            raise CalibrationInputError(f"Expected columns {FEATURE_NAMES}")
        if not np.all(np.isfinite(values)) or np.any(values[:, 0] <= 0) or np.any(values[:, 4] <= 0):
            raise CalibrationInputError("Model features must be finite with positive voltage/baseline")
        return values

    def _raw_design(self, values):
        v, x, y, event = ((values[:, i] - self.center[i]) / self.scale[i] for i in range(4))
        columns, names = [np.ones(len(values))], ["intercept"]
        if self.config.voltage_term:
            columns += [v]
            names += ["voltage"]
        if self.config.spatial_terms:
            columns += [x, y, x*x, x*y, y*y]
            names += ["x", "y", "x2", "xy", "y2"]
        if self.config.linear_drift:
            columns += [event]
            names += ["event_fraction"]
        block = np.searchsorted(self.config.block_edges, values[:, 3], side="right")
        for index in range(1, len(self.config.block_edges) + 1):
            columns.append((block == index).astype(float))
            names.append(f"block_{index}")
        return np.column_stack(columns), names

    def fit(self, features, target):
        # A failed refit must not leave old coefficients paired with new scaling.
        self.coefficients, self._valid_domain = None, {}
        for attribute in ("t0_ns", "fit_diagnostics", "term_names", "center", "scale", "keep"):
            if hasattr(self, attribute):
                delattr(self, attribute)
        values = self._features(features)
        target = np.asarray(target, dtype=float)
        if target.shape != (len(values), 2) or not np.all(np.isfinite(target)) or np.any(target[:, 1] <= 0):
            raise CalibrationInputError("Target must contain aligned finite [raw_t_ns, mass_da] rows")
        masses, counts = np.unique(target[:, 1], return_counts=True)
        if len(masses) < 2:
            raise CalibrationInputError("At least two independently identified reference masses are required")
        self.center = np.mean(values[:, :4], axis=0)
        self.scale = np.std(values[:, :4], axis=0)
        self.scale[self.scale < 1e-12] = 1.
        raw, names = self._raw_design(values)
        self.keep = np.r_[True, np.ptp(raw[:, 1:], axis=0) > 1e-12]
        design = raw[:, self.keep]
        self.term_names = np.asarray(names)[self.keep].tolist()
        if len(values) <= design.shape[1] + 2 or np.linalg.matrix_rank(design) < design.shape[1]:
            raise CalibrationInputError("Reference coverage cannot identify the requested model terms")
        baseline = np.sqrt(target[:, 1]) * values[:, 4]
        # Equal total weight per reference; abundant species cannot monopolize the fit.
        weights = np.sqrt(len(values) / (len(masses) * counts[np.searchsorted(masses, target[:, 1])]))
        def residual(parameters):
            return weights * (parameters[0] + baseline * np.exp(design @ parameters[1:]) - target[:, 0])
        low, high = self.config.t0_bounds_ns
        initial = np.zeros(design.shape[1] + 1)
        initial[0] = np.clip(0., low + 1e-9, high - 1e-9)
        result = least_squares(residual, initial, bounds=(np.r_[low, np.full(design.shape[1], -2.)],
                                                        np.r_[high, np.full(design.shape[1], 2.)]),
                               loss="soft_l1", f_scale=self.config.robust_scale_ns,
                               x_scale="jac", max_nfev=400)
        norms = np.linalg.norm(result.jac, axis=0)
        condition = np.linalg.cond(result.jac / np.maximum(norms, 1e-15))
        if not result.success or np.any(result.active_mask) or not np.isfinite(condition) or condition > self.config.max_condition_number:
            raise CalibrationInputError("Fit is unidentifiable, did not converge, or reached a parameter bound")
        factors = np.exp(design @ result.x[1:])
        if np.any(~np.isfinite(factors) | (factors < self.config.factor_bounds[0]) | (factors > self.config.factor_bounds[1])):
            raise CalibrationInputError("Fitted ToF factor exceeds the declared physical bounds")
        self.t0_ns = float(result.x[0])
        self.coefficients = result.x[1:].copy()
        self._valid_domain = {name: (float(values[:, i].min()), float(values[:, i].max()))
                              for i, name in enumerate(FEATURE_NAMES[:4])}
        variance = float(np.sum(result.fun**2) / (len(values) - len(result.x)))
        covariance = np.linalg.pinv(result.jac.T @ result.jac) * variance
        self.fit_diagnostics = dict(condition_number=float(condition), n_train=len(values),
                                    weighted_rmse_ns=float(np.sqrt(np.mean(result.fun**2))),
                                    parameter_standard_errors=np.sqrt(np.maximum(0, np.diag(covariance))).tolist(),
                                    uncertainty_kind="local model-conditional linearization; not coverage probabilities")
        return self

    def predict_factor(self, features):
        if self.coefficients is None:
            raise CalibrationStateError("Model has not been fitted")
        values = self._features(features)
        factor = np.exp(self._raw_design(values)[0][:, self.keep] @ self.coefficients)
        low, high = self.config.factor_bounds
        if np.any(~np.isfinite(factor) | (factor < low) | (factor > high)):
            raise CalibrationInputError("Predicted ToF factor exceeds the declared physical bounds")
        return factor

    def domain_mask(self, features):
        values = np.asarray(features, dtype=float)
        if self.coefficients is None:
            raise CalibrationStateError("Model has not been fitted")
        if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
            raise CalibrationInputError(f"Expected columns {FEATURE_NAMES}")
        mask = np.all(np.isfinite(values), axis=1) & (values[:, 0] > 0) & (values[:, 4] > 0)
        for i, (low, high) in enumerate(self._valid_domain.values()):
            margin = max((high - low) * self.config.domain_margin_fraction, 1e-12)
            mask &= (values[:, i] >= low - margin) & (values[:, i] <= high + margin)
        return mask

    def predict_mass_da(self, features, raw_t_ns):
        if self.coefficients is None:
            raise CalibrationStateError("Model has not been fitted")
        values = self._features(features)
        elapsed = np.asarray(raw_t_ns, dtype=float) - self.t0_ns
        if elapsed.shape != (len(values),) or np.any(~np.isfinite(elapsed) | (elapsed <= 0)):
            raise CalibrationInputError("Every predicted event must have finite positive t - t0")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            mass = (elapsed / (values[:, 4] * self.predict_factor(values)))**2
        if np.any(~np.isfinite(mass) | (mass <= 0)):
            raise CalibrationInputError("Predicted mass must be finite and positive")
        return mass

    @property
    def valid_domain(self):
        return dict(self._valid_domain)

    def provenance(self):
        from dataclasses import asdict
        return dict(model_type=type(self).__name__, config=asdict(self.config), feature_names=list(FEATURE_NAMES),
                    t0_ns=getattr(self, "t0_ns", None), factor_units="dimensionless ToF ratio",
                    coefficients=None if self.coefficients is None else self.coefficients.tolist(),
                    term_names=getattr(self, "term_names", []), center=getattr(self, "center", np.array([])).tolist(),
                    scale=getattr(self, "scale", np.array([])).tolist(), valid_domain=self.valid_domain,
                    diagnostics=getattr(self, "fit_diagnostics", {}))
