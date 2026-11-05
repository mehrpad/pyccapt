"""Optional empirical shared-shape isotope-family fitting for manual review.

The caller confirms the isotope identities and provides an isolated reference
shape. This module estimates a bounded common mass offset, nonnegative peak
areas and a constant background. It does not identify species, range events,
or change calibration Variables. All uncertainty is conditional on the supplied
identities, template, constant-background model and chosen fit interval.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.optimize import minimize, minimize_scalar, nnls

from pyccapt.calibration.core.exceptions import CalibrationInputError, CalibrationStateError


def _vector(values, name, *, minimum=1, nonnegative=False):
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise CalibrationInputError(f"{name} must be a numeric vector") from exc
    if result.ndim != 1 or len(result) < minimum or not np.isfinite(result).all():
        raise CalibrationInputError(f"{name} must contain at least {minimum} finite values")
    if nonnegative and np.any(result < 0):
        raise CalibrationInputError(f"{name} cannot contain negative values")
    return result


def _scalar(value, name, *, nonnegative=False):
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise CalibrationInputError(f"{name} must be a finite numeric scalar") from exc
    if array.ndim != 0 or not np.isfinite(array):
        raise CalibrationInputError(f"{name} must be a finite numeric scalar")
    result = float(array)
    if nonnegative and result < 0:
        raise CalibrationInputError(f"{name} cannot be negative")
    return result


def _edges(centres):
    midpoint = (centres[:-1]+centres[1:])/2
    return np.r_[centres[0]-(midpoint[0]-centres[0]), midpoint,
                 centres[-1]+(centres[-1]-midpoint[-1])]


@dataclass(frozen=True)
class EmpiricalTemplate:
    """Unit-area piecewise-linear density anchored to an explicit mass."""

    offsets: np.ndarray
    density: np.ndarray
    reference_mass: float

    def __post_init__(self):
        offsets = _vector(self.offsets, "template offsets", minimum=3).copy()
        density = _vector(self.density, "template density", minimum=3, nonnegative=True).copy()
        if len(offsets) != len(density) or np.any(np.diff(offsets) <= 0):
            raise CalibrationInputError("Template offsets must be increasing and aligned with density")
        reference_mass = _scalar(self.reference_mass, "Template reference mass")
        area = float(np.sum((density[1:]+density[:-1])*np.diff(offsets)/2))
        if not np.isfinite(area) or area <= 0:
            raise CalibrationInputError("Template requires finite positive area")
        density /= area
        offsets.setflags(write=False)
        density.setflags(write=False)
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "density", density)
        object.__setattr__(self, "reference_mass", reference_mass)

    def integral(self, values):
        """Exact antiderivative of the linear density, clamped outside support."""
        values = np.asarray(values, dtype=float)
        widths = np.diff(self.offsets)
        cumulative = np.r_[0., np.cumsum((self.density[:-1]+self.density[1:])*widths/2)]
        index = np.clip(np.searchsorted(self.offsets, values, side="right")-1, 0, len(widths)-1)
        distance = np.clip(values-self.offsets[index], 0., widths[index])
        slope = (self.density[index+1]-self.density[index])/widths[index]
        result = cumulative[index]+self.density[index]*distance+.5*slope*distance**2
        return np.where(values <= self.offsets[0], 0.,
                        np.where(values >= self.offsets[-1], 1., result))

    def provenance(self):
        digest = hashlib.sha256(self.offsets.tobytes()+self.density.tobytes()).hexdigest()
        return {"method": "empirical_piecewise_linear", "reference_mass": float(self.reference_mass),
                "support": [float(self.offsets[0]), float(self.offsets[-1])],
                "points": len(self.offsets), "shape_sha256": digest}


def build_empirical_template(mass_centres, counts, *, reference_mass, background=0.):
    """Build an isolated-reference shape; background is counts per reference bin.

    The supplied reference_mass anchors the shape; it is not inferred from a
    symmetric Gaussian or from the highest bin. The caller must exclude other
    peaks from this interval and ensure that the supplied anchor is appropriate.
    """
    mass = _vector(mass_centres, "reference mass centres", minimum=3)
    counts = _vector(counts, "reference counts", minimum=3, nonnegative=True)
    if len(mass) != len(counts) or np.any(np.diff(mass) <= 0):
        raise CalibrationInputError("Reference mass centres must be increasing and aligned with counts")
    background = _scalar(background, "Reference background", nonnegative=True)
    reference_mass = _scalar(reference_mass, "Reference mass")
    if not mass[0] <= reference_mass <= mass[-1]:
        raise CalibrationInputError("Reference mass must lie inside the isolated reference interval")
    widths = np.diff(_edges(mass))
    density = np.maximum(counts-background, 0.)/widths
    return EmpiricalTemplate(mass-float(reference_mass), density, float(reference_mass))


@dataclass(frozen=True)
class IsotopeFamilyFit:
    isotope_masses: np.ndarray
    fitted_centres: np.ndarray
    amplitudes: np.ndarray
    background_density: float
    mass_offset: float
    prediction: np.ndarray
    components: np.ndarray
    validation_mask: np.ndarray
    diagnostics: dict
    uncertainty: dict
    provenance: dict

    def summary(self):
        """Small JSON-serializable evidence record; predictions remain separate."""
        return {"isotope_masses": self.isotope_masses.tolist(),
                "fitted_centres": self.fitted_centres.tolist(), "amplitudes": self.amplitudes.tolist(),
                "background_density": self.background_density, "mass_offset": self.mass_offset,
                "diagnostics": self.diagnostics, "uncertainty": self.uncertainty,
                "provenance": self.provenance}


def _metrics(observed, predicted):
    if not len(observed):
        return None
    expected = np.maximum(predicted, np.finfo(float).tiny)
    nonzero = observed > 0
    terms = expected-observed
    terms[nonzero] += observed[nonzero]*(np.log(observed[nonzero])-np.log(expected[nonzero]))
    return {"bins": len(observed), "observed_counts": float(observed.sum()),
            "rmse_counts": float(np.sqrt(np.mean((observed-predicted)**2))),
            "poisson_deviance_per_bin": float(max(0., 2*terms.sum()/len(observed))),
            "pearson_per_bin": float(np.mean((observed-predicted)**2/np.maximum(predicted, 1.)))}


def fit_isotope_family(mass_centres, counts, *, isotope_masses, template,
                       shift_bounds=(-.05, .05), abundance_prior=None, prior_strength=0.,
                       validation_mask=None, shift_grid_size=81):
    """Fit user-confirmed isotope masses with a shared empirical template.

    Peak areas and constant background density are nonnegative. At each common
    offset, NNLS initializes a Poisson-deviance minimization. A coarse
    offset profile plus bounded local refinement avoids relying on one starting
    guess. By default every fifth histogram bin is held out; these are held-out
    bins, not independent acquisitions. Natural abundances are never assumed.

    Optional regularization is strength * ||(A-p*sum(A))/S||^2, where p is the
    caller's normalized abundance prior and S is observed training counts.
    A zero strength disables it, including for enriched specimens. Reported
    errors use the data-only local information matrix, not this prior.
    """
    mass = _vector(mass_centres, "mass centres", minimum=10)
    observed = _vector(counts, "counts", minimum=10, nonnegative=True)
    isotopes = _vector(isotope_masses, "confirmed isotope masses")
    if len(mass) != len(observed) or np.any(np.diff(mass) <= 0):
        raise CalibrationInputError("Mass centres must be increasing and aligned with counts")
    if len(np.unique(isotopes)) != len(isotopes):
        raise CalibrationInputError("Confirmed isotope masses must be distinct")
    if not isinstance(template, EmpiricalTemplate):
        raise CalibrationInputError("Provide an EmpiricalTemplate from an isolated reference")
    bounds = _vector(shift_bounds, "shift bounds", minimum=2)
    if len(bounds) != 2 or bounds[0] >= bounds[1]:
        raise CalibrationInputError("Shift bounds must be a finite increasing pair")
    prior_strength = _scalar(prior_strength, "Prior strength", nonnegative=True)
    if not isinstance(shift_grid_size, (int, np.integer)) or shift_grid_size < 9:
        raise CalibrationInputError("At least nine offset grid points are required")
    if observed.sum() <= 0:
        raise CalibrationInputError("A family fit requires positive observed counts")
    prior = None
    if abundance_prior is not None:
        prior = _vector(abundance_prior, "abundance prior", nonnegative=True)
        if len(prior) != len(isotopes) or prior.sum() <= 0:
            raise CalibrationInputError("Abundance prior must match isotope masses and have positive total")
        prior = prior/prior.sum()
    elif prior_strength:
        raise CalibrationInputError("A positive prior strength requires an explicit abundance prior")
    if validation_mask is None:
        validation = np.arange(len(mass)) % 5 == 4
    else:
        validation = np.asarray(validation_mask)
        if validation.dtype != bool or validation.shape != mass.shape:
            raise CalibrationInputError("Validation mask must be a boolean vector matching the histogram")
        validation = validation.copy()
    train = ~validation
    parameter_count = len(isotopes)+2
    if train.sum() <= parameter_count:
        raise CalibrationInputError("Too few training bins for the requested isotope family")
    if observed[train].sum() <= 0:
        raise CalibrationInputError("Training bins require positive observed counts; held-out counts cannot fit the family")
    edges = _edges(mass)
    widths = np.diff(edges)
    weight = 1/np.sqrt(np.maximum(observed[train], 1.))
    solver_evidence = {}

    def design(offset):
        cumulative = template.integral(edges[:, None]-isotopes[None, :]-offset)
        return np.column_stack((np.maximum(np.diff(cumulative, axis=0), 0.), widths))

    def solve(offset):
        matrix = design(offset)
        weighted = matrix[train]*weight[:, None]
        target = observed[train]*weight
        scale = max(float(observed[train].sum()), 1.)
        constraint = None
        if prior is not None and prior_strength:
            constraint = np.eye(len(isotopes))-prior[:, None]
            constraint = np.column_stack((constraint, np.zeros(len(isotopes))))/scale
        try:
            initial, _ = nnls(weighted, target, maxiter=100*matrix.shape[1])
        except RuntimeError as exc:
            raise CalibrationStateError("Isotope-family coefficient initialization did not converge") from exc
        # Comparable parameter scales make the positive linear Poisson model
        # well behaved even when areas and background densities differ greatly.
        scales = np.r_[np.full(len(isotopes), scale), scale/widths[train].sum()]
        initial[-1] = max(initial[-1], scales[-1]*1e-3)
        basis = matrix[train]*scales
        y = observed[train]
        positive = y > 0

        def objective(theta):
            expected = np.maximum(basis@theta, 1e-12)
            terms = expected-y
            terms[positive] += y[positive]*(np.log(y[positive])-np.log(expected[positive]))
            cost = 2*terms.sum()
            gradient = 2*basis.T@(1-y/expected)
            if constraint is not None:
                residual = constraint@(theta*scales)
                cost += prior_strength*float(residual@residual)
                gradient += 2*prior_strength*(constraint.T@residual)*scales
            return float(cost/scale), gradient/scale

        optimized = minimize(objective, initial/scales, method="L-BFGS-B", jac=True,
                             bounds=[(0., None)]*len(initial),
                             options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 300})
        projected_gradient = optimized.jac.copy()
        projected_gradient[(optimized.x <= 1e-10) & (projected_gradient > 0)] = 0.
        optimality = float(np.linalg.norm(projected_gradient, ord=np.inf))
        # Degenerate overlapping columns can make the line search stop even
        # when the bound-constrained first-order condition is satisfied.
        stationary = np.isfinite(optimality) and optimality <= 1e-6
        if not np.isfinite(optimized.fun) or (not optimized.success and not stationary):
            raise CalibrationStateError(f"Isotope-family coefficient fit did not converge: {optimized.message}")
        solver_evidence.update(projected_gradient_inf=optimality,
                               optimizer_success=bool(optimized.success),
                               optimizer_message=str(optimized.message))
        coefficients = optimized.x*scales
        return float(max(0., optimized.fun*scale)), coefficients, matrix

    grid = np.linspace(*bounds, shift_grid_size)
    profile = np.array([solve(offset)[0] for offset in grid])
    best = int(np.argmin(profile))
    low, high = grid[max(0, best-1)], grid[min(len(grid)-1, best+1)]
    optimum = minimize_scalar(lambda value: solve(value)[0], bounds=(low, high), method="bounded",
                              options={"xatol": max(1e-12, float(np.diff(bounds)[0])*1e-8)})
    candidates = [(float(grid[best]), float(profile[best]))]
    if optimum.success:
        candidates.append((float(optimum.x), float(optimum.fun)))
    offset = min(candidates, key=lambda pair: pair[1])[0]
    objective, coefficients, matrix = solve(offset)
    predicted = matrix@coefficients
    components = matrix[:, :-1]*coefficients[:-1]
    flags = []
    support = matrix[:, :-1].sum(axis=0)
    if np.any(support < .95):
        flags.append("template_support_truncated")
    if np.any(coefficients[:-1] <= max(float(coefficients[:-1].sum()), 1.)*1e-8):
        flags.append("one_or_more_component_areas_at_zero")
    if min(offset-bounds[0], bounds[1]-offset) <= (bounds[1]-bounds[0])*1e-3:
        flags.append("mass_offset_at_search_boundary")
    peak_columns = matrix[train, :-1]
    norms = np.linalg.norm(peak_columns, axis=0)
    correlations = peak_columns.T@peak_columns/np.maximum(np.outer(norms, norms), np.finfo(float).tiny)
    np.fill_diagonal(correlations, 0.)
    maximum_correlation = float(correlations.max()) if len(isotopes) > 1 else 0.
    if maximum_correlation > .98:
        flags.append("strongly_overlapping_component_shapes")
    # Conditional local errors: template/noise uncertainty and isotope-identity
    # uncertainty are not represented by this matrix.
    step = max(1e-9, min(float(np.median(widths))*.1, float(bounds[1]-bounds[0])*1e-4))
    derivative = ((design(offset+step)-design(offset-step))@coefficients)/(2*step)
    jacobian = np.column_stack((matrix, derivative))[train]/np.sqrt(np.maximum(predicted[train], 1.))[:, None]
    norms = np.linalg.norm(jacobian, axis=0)
    scaled = jacobian/np.maximum(norms, np.finfo(float).tiny)
    rank = int(np.linalg.matrix_rank(scaled))
    condition = float(np.linalg.cond(scaled))
    identifiable = rank == parameter_count and np.isfinite(condition) and condition < 1e8
    if not identifiable:
        flags.append("local_parameters_not_identifiable")
    if not validation.any():
        flags.append("no_held_out_bins")
    elif observed[validation].sum() == 0:
        flags.append("held_out_bins_have_no_observed_counts")
    other_minima = [i for i in range(1, len(grid)-1)
                    if profile[i] <= profile[i-1] and profile[i] <= profile[i+1] and abs(i-best) > 2]
    if any(profile[i] <= objective+max(1., .01*max(objective, 1.)) for i in other_minima):
        flags.append("competing_mass_offset_minima")
    errors = [None]*parameter_count
    if identifiable:
        # Scale columns before inversion so area and mass units do not make
        # the covariance numerically rank deficient.
        dispersion = max(1., float(np.sum((observed[train]-predicted[train])**2/
                                          np.maximum(predicted[train], 1.)))/(int(train.sum())-parameter_count))
        covariance = np.linalg.pinv(scaled.T@scaled)/np.outer(norms, norms)*dispersion
        errors = [float(value) for value in np.sqrt(np.maximum(np.diag(covariance), 0.))]
    for index, value in enumerate(coefficients):
        if value <= max(float(coefficients.sum()), 1.)*1e-10:
            errors[index] = None  # symmetric errors do not describe an active nonnegative bound
    if "mass_offset_at_search_boundary" in flags:
        errors[-1] = None
    uncertainty = {"scope": "conditional_local_linear_approximation",
                   "includes_template_or_identity_uncertainty": False,
                   "mass_offset_std": errors[-1], "amplitude_std": errors[:-2],
                   "background_density_std": errors[-2]}
    diagnostics = {"review_required": True, "ambiguous": bool(flags), "flags": flags,
                   "training": _metrics(observed[train], predicted[train]),
                   "validation": _metrics(observed[validation], predicted[validation]),
                   "template_area_in_fit_window": support.tolist(),
                   "maximum_component_shape_correlation": maximum_correlation,
                   "information_rank": rank, "information_condition": condition if np.isfinite(condition) else None,
                   "poisson_deviance_plus_penalty": objective, "coefficient_solver": dict(solver_evidence)}
    provenance = {"method": "empirical_isotope_family_v1", "template": template.provenance(),
                  "identities": "caller_confirmed", "shift_bounds": bounds.tolist(),
                  "prior_strength": float(prior_strength),
                  "abundance_prior": None if prior is None else prior.tolist(),
                  "validation": "held_out_histogram_bins", "objective": "Poisson deviance plus explicit prior penalty"}
    return IsotopeFamilyFit(isotopes.copy(), isotopes+offset, coefficients[:-1], float(coefficients[-1]),
                            offset, predicted, components, validation, diagnostics, uncertainty, provenance)
