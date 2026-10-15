# Empirical isotope-family fitting (optional)

`pyccapt.calibration.core.isotope_family` is an opt-in proposal tool for a
user-confirmed isotope family. It shares a measured asymmetric peak shape across
the supplied isotope masses and fits a common bounded mass offset, nonnegative
peak areas, and a constant background. It does not identify elements or molecular
ions, assign events to ranges, change `Variables`, or apply a calibration.

Use an isolated reference peak with enough counts to describe its tail, a known
mass anchor, and an interval excluding other species. The reference should come
from comparable acquisition conditions and a mass interval where sharing its
shape is defensible. A noisy or contaminated reference transfers its errors to
every fitted component. The method does not infer that a Gaussian peak maximum
is the true mass anchor, and it cannot establish a missing reference identity.

```python
from pyccapt.calibration.core.isotope_family import (
    build_empirical_template, fit_isotope_family,
)

# Inputs are histogram bin centres and counts; reference_mass and
# confirmed_isotope_masses must have been confirmed by the operator.
template = build_empirical_template(
    reference_bin_centres,
    reference_counts,
    reference_mass=confirmed_reference_mass,
    background=reference_background_counts_per_bin,
)
proposal = fit_isotope_family(
    family_bin_centres,
    family_counts,
    isotope_masses=confirmed_isotope_masses,
    template=template,
    shift_bounds=(-0.05, 0.05),
)
print(proposal.summary())
# Inspect proposal.prediction, proposal.components, residuals, fitted_centres
# and diagnostics before deciding whether these centres are useful references.
```

The mass axis, bounds, reference anchor and isotope masses must use the same
mass-to-charge units, normally Da/e. Background is counts per unit mass in the
family fit, whereas `build_empirical_template(background=...)` subtracts counts
per reference bin before converting to density. Nonuniform histogram bins are
supported: edges are inferred from adjacent bin centres.

## Model and objective

The reference defines a nonnegative piecewise-linear density `T`, normalized to
unit area. Its exact piecewise-quadratic antiderivative integrates each predicted
component over the target bin edges. For bin `i`, known isotope mass `M_j`,
common offset `delta`, nonnegative area `A_j`, and nonnegative background density
`b`, the expected count is:

```text
mu_i = b * bin_width_i
       + sum_j A_j * integral_bin_i T(m - M_j - delta) dm
```

At each trial offset, a nonnegative least-squares solution initializes a
constrained fit that minimizes the Poisson deviance:

```text
2 * sum_training_bins [mu_i - counts_i + counts_i * log(counts_i / mu_i)]
```

The logarithmic term is zero when the observed count is zero. An explicitly
enabled abundance penalty is added to this objective as described below.
An offset grid locates a promising basin, followed by bounded scalar refinement.
The algorithm reports competing grid minima and boundary solutions. Independent
Poisson counts are a modelling assumption; correlated or overdispersed data can
violate it. Treat amplitudes as fit diagnostics, not an unqualified composition
estimate. Shape variation across mass, changing thermal tails, unmodelled
species, background slope and detector effects can invalidate the common-template
assumption. The conditional solver reports its projected gradient residual,
including whether a bounded stationary solution required accepting that residual
after the optimizer reported a line-search failure.

Natural abundances are **not imposed**. `abundance_prior=None` and
`prior_strength=0` are the defaults, appropriate when enrichment is possible.
If the operator explicitly supplies prior fractions `p`, the optional penalty is:

```text
prior_strength * ||(A - p * sum(A)) / S||^2
```

Here `S` is the observed count in the training bins and `p` is normalized by the
function. This is transparent regularization, not a Bayesian posterior. A
supplied prior with zero strength has no effect. Strong priors can force an
incorrect answer and must not be used to conceal poor data or ambiguous species.

## Evidence and limitations

By default every fifth histogram bin is held out from all parameter fitting.
An explicit boolean `validation_mask` can override that split. Validation
reports count RMSE, Pearson residuals and Poisson deviance per bin. These are
**held-out bins**, not an independent acquisition or independent reference
template. Validation values never affect the fitted offset or areas.

The result always sets `review_required=True`. Flags report strongly overlapping
component shapes, truncated template support, zero-area components, offset
boundaries, competing minima, missing validation bins and rank-deficient local
information. Held-out bins containing no observed counts are explicitly flagged:
their residuals test predicted false positives but do not demonstrate peak
recovery. Training bins must contain positive counts. No identity confidence or
acceptance probability is returned.

Conditional standard errors come from the local data-only information matrix,
with an overdispersion floor. They exclude uncertainty in reference shape,
reference mass, isotope identity, background model and acquisition drift.
Singular fits return unavailable errors; active nonnegative and offset bounds
also suppress inappropriate symmetric errors. A small conditional error does
not prove that the reference identity or model is correct.

The JSON-compatible summary records the supplied isotope masses, optional prior,
offset bounds, template shape hash, fit objective, residual evidence and
uncertainty scope. Store the template arrays and the histogram source/selection
alongside the summary if the proposal must be reproducible. The hash identifies
the shape; it does not contain the source data.

Synthetic tests exercise a known asymmetric family with overlap, enriched
abundances, nonuniform bins, held-out-bin isolation, boundaries and nearly
coincident components. This is an experimental analysis option, not a universally
validated replacement for the existing calibration workflow. Confirm performance
on representative datasets and independent reference peaks before adopting it.
