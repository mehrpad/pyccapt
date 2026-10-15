# Guided multi-reference calibration: mathematics and use

This optional workflow complements the existing voltage, bowl and reference
optimizers. It uses independently identified reference masses to fit one
additive time offset and a positive, smooth multiplicative time scale. It is
an experimental model with explicit validation gates, not a guarantee that a
sharper peak has the correct identity or that a specimen reconstructs correctly.

Implementation: `calibration/core/multireference_tof.py`,
`calibration/core/guided_calibration.py`, and the notebook helper
`calibration/tutorials/tutorials_helpers/helper_guided_calibration.py`.

## Physical model and units

Let $m_i$ be the known mass-to-charge in Da, $t_i$ the **raw** measured
time in ns, $V_i$ the accelerating voltage in V and $x_i,y_i$ detector
coordinates in cm. For flight length $L$ in mm, define

$$
 \ell_i=\sqrt{(10^{-2}x_i)^2+(10^{-2}y_i)^2+(10^{-3}L)^2},\qquad
 s_i=10^9\ell_i\sqrt{u/(2eV_{\mathrm{eff},i})}.
$$

Here $s_i$ has units ns/$\sqrt{\mathrm{Da}}$. To remain exactly compatible
with the existing `mc_tools.tof2mc` convention, the implementation uses
$u=1.66\times10^{-27}$ kg and $e=1.6\times10^{-19}$ C. These are the
existing rounded constants, not a new precision claim. Laser mode uses
$V_{\mathrm{eff}}=V$; voltage-pulse mode uses the existing empirical convention
$V_{\mathrm{eff}}=1.015(V+0.7V_{\mathrm{pulse}})$. Changing those instrument
coefficients requires independent instrument characterization.

The fitted model is

$$
 t_i=t_0+s_i\sqrt{m_i}\,f_i+\epsilon_i,
 \qquad f_i=\exp(B_i\boldsymbol\beta)>0.
$$

`t0_ns` is additive and measured in ns. `predict_factor` returns the
**dimensionless ToF ratio** $f_i$, not its reciprocal or a mass ratio. The
design contains an intercept, optionally standardized voltage, standardized
$x,y,x^2,xy,y^2$, optional linear event-order drift, and optional offsets for
user-declared acquisition blocks. Standardization is learned on training
events only. Constant terms are removed; dependent terms cause rejection.

Block edges are fractions of event order strictly inside (0,1). They describe
known acquisition regimes selected before evaluating holdouts. There is no
automatic change-point discovery, and event order is not elapsed clock time.
Block offsets change multiplicative scale; this version has one global
additive $t_0$, not a separate drifting additive offset in each block.

The model uses a bounded robust nonlinear least-squares fit with soft-L1 loss.
For $J$ references, $n_j$ fitted events in reference $j$, $N=\sum_j n_j$,
raw time residual $r_i$ in ns and declared robust scale $c$ in ns, its objective is

$$
 \frac{c^2}{2}\sum_i a_i\rho((r_i/c)^2),\qquad
 a_i=\frac{N}{J n_{j(i)}},\qquad \rho(z)=2(\sqrt{1+z}-1).
$$

Each reference species has equal total fitting weight. The weight multiplies
the **loss**, not the residual before applying the nonlinear loss. The latter
would change the robust cutoff to $c/\sqrt{a_i}$ and make abundant and scarce
references use different outlier rules. A regression repeats one reference
twenty times while preserving its distribution and verifies an unchanged fit.
This corrects the objective; it does not identify an incorrectly labelled or
overlapping reference. Positivity comes from
the exponential; declared factor bounds are checked separately. An active
coefficient bound, failed convergence or poorly conditioned scaled Jacobian
rejects the fit. Coefficients have numerical bounds [-2,2] in standardized
coordinates; these are stabilization bounds, not physical prior probabilities.

## Why multiple references and independent validation matter

One reference can trade a time offset against a scale. Multiple masses with
overlapping voltage, detector and acquisition-time coverage help separate
those effects. Each train, selection and confirmation split must contain every
reference in every declared block, with a configurable minimum event count.
This prevents an apparently successful fit based on one species in one block
and another species in a different block. It does not prove identifiability
under arbitrary composition or detector correlations.

Fit time is bounded by sampling at most 5,000 training events per reference/model
block **without replacement**, using a seed fixed at session creation
(`fit_seed=0` by default). Selected indices are restored to acquisition order.
The sample hash, seed and available event count are recorded. Regular-stride
sampling is avoided: a synthetic periodic 5 ns tail is sampled exclusively by
the old evenly spaced subset, producing more than 2% median mass bias; seeded
sampling of the same population passes the unchanged validation limits.
Increase the cap when training/selection diagnostics justify it; do not search
seeds or sampling settings against confirmation data.
Selection and confirmation evaluate all supplied reference events in those
splits. They **never update coefficients**. Validation uses four fixed
acquisition-time bands by default (`validation_time_bins=4`) over the union of
reference and application extents. An application mask never erases held-out
reference evidence, and an empty evaluation cannot pass. Declared model-block boundaries are respected when
subdividing that extent, avoiding nearly coincident boundaries and empty
slivers. Each reference must have enough held-out events in every resulting
band occupied by the evaluated role; insufficient reference coverage there fails
explicitly. A contiguous later holdout need not contain events in earlier bands
reserved for training. Per-reference/time-band reports include
relative median bias, relative RMS mass error, empirical 2.5/50/97.5-percentile
errors, positive-elapsed-time validity and counts outside the training domain.
This catches equal/opposite temporal biases that cancel in an aggregate report.
The policy is fixed at session
creation; default limits are 0.5% absolute median bias and 2% RMS relative error.
These defaults are starting tolerances to review for the experiment, not
universal standards for adequate mass calibration.

Replacement validation also compares the candidate with a saved mass baseline
on **identical held-out events**. By default (`require_no_regression=True`),
neither absolute median bias nor relative RMSE may worsen in any reference/time
band; only a $10^{-12}$ numerical allowance is used. This is deliberately
conservative, not a statistical significance test. A pass within broad absolute
limits is insufficient to call a mixed result an improvement. A real-data check
on sample 2643 fails eight of twenty groups under this rule and leaves
confirmation unused. The source calibration remains untouched.

Supply `baseline_mass_da=accepted_mass_array` when replacing an accepted result.
The array is snapshotted and hashed. When omitted, a usable recorded `mc (Da)`
column is the fallback, **not** the legacy active array, which can initially
contain `mc_uc`. Unavailable/zero placeholder mass supplies no baseline;
comparison is then explicitly unavailable and the absolute gates still apply.
An existing baseline with missing values on a validation group fails the
comparison. Baseline agreement does not establish correct chemical identities.

Prediction is allowed only inside the rectangular training voltage/detector
feature bounds, expanded by the declared margin (default 2% of each feature's
training range). A fitted linear drift additionally requires event order inside
the training event-order range; extrapolation is rejected. Without a drift term,
event order is unused and a later holdout tests the model's explicit stationarity
assumption. It is not rejected merely for being later. Declared block offsets
require a block category observed in training, even in a stationary-within-block
model. Older snapshots without recorded categories retain their original
conservative event-domain restriction.
This is a coordinate-range check, not evidence of dense joint coverage. Inspect
residuals against voltage, detector position and acquisition order. A point can
lie inside every marginal range while occupying a sparse joint combination.
Prefer simpler models when coverage or conditioning is poor.

Reported parameter standard errors are a local model-conditional Jacobian
linearization. They omit identity mistakes, overlap, correlated events,
model misspecification and instrument-geometry uncertainty. Robust-loss and
sampling choices also affect them. They are **not probabilities that a model is
correct**, and empirical error quantiles are not confidence intervals for a
future specimen. The fitted model cannot validate an identity chosen from its
own residual alone. Review isotope/charge-family consistency independently.

## Deriving the outputs without double correction

The inferred mass is

$$
 \widehat m_i=\left[\frac{t_i-t_0}{s_i f_i}\right]^2.
$$

Mass mode writes this directly to `Variables.mc_calib` in Da. It does not
silently overwrite the displayed/committed `mc` or source dataframe; use the
existing workflow's explicit publication/save step when appropriate.

ToF mode instead writes one common reference scale:

$$
 T_{\mathrm{ref},i}=s_{\mathrm{ref}}\sqrt{\widehat m_i},\qquad
 s_{\mathrm{ref}}=10^9 L_{\mathrm{ref,m}}\sqrt{u/(2eV_{\mathrm{ref}})}.
$$

This is normalized ToF in ns, at the receipt's reference voltage and flight
length, on the detector axis, with zero additive offset. It is **not raw ToF**.
Do not subtract the previous $t_0$, reuse event-specific voltage/positions or
rerun initial geometric normalization. Call
`session.convert_applied_tof_to_mass()` (the panel's **Convert ToF to mass**)
to populate `mc_calib`. This calls the existing public `tof2mc` with the matching
reference geometry, zero offset and laser convention; known-truth tests verify
that it agrees with inferred mass. The existing legacy panels do not generally
read this new receipt, so use this explicit conversion for guided output.

Excluded rows become NaN in **both** derived modes, preserving event count and
order while avoiding mixed normalized/raw or differently calibrated values.
`guided_calibration_valid_mask` records coverage. Raw arrays and the source
dataframe are untouched; the previous active array is stored in its backup
only when application succeeds. All chunks are evaluated before assigning the
new array, so an invalid later chunk cannot leave a partially corrected result.
Chunking bounds model-evaluation memory; a full output array and the immutable
source snapshot still reside in memory.

## Notebook workflow

1. Load data through `Variables.sync_from_data`.
2. Optionally call `propose_reference_windows(variables, mode="mc", ...)` to
   reuse the existing peak detector. Every proposed identity is explicitly
   `unconfirmed`. Inspect tails, overlap/background and time coverage; identify
   the isotope/charge independently and supply its mass-to-charge in Da.
3. Create disjoint masks for training, model selection and untouched
   confirmation. Interleaved acquisition *blocks* can preserve coverage while
   reducing adjacent-event leakage; choose the split before comparing models.
4. Fit and inspect selection diagnostics. A failed confirmation requires fresh
   independent confirmation data for subsequent model development. Creating
   another session does not magically make an already inspected holdout new.
5. Accept, apply in the matching mode, and save the receipt with the dataset.

```python
from pyccapt.calibration.core.guided_calibration import (
    GuidedCalibrationSession, ValidationPolicy,
)
from pyccapt.calibration.core.multireference_tof import ToFModelConfig
from pyccapt.calibration.tutorials.tutorials_helpers.helper_guided_calibration import (
    propose_reference_windows, show_guided_calibration,
)

# Supply independently reviewed event-aligned boolean masks and known m/q.
references = {known_mass_1_Da: reference_mask_1, known_mass_2_Da: reference_mask_2}
roles = {"train": training_mask, "selection": selection_mask,
         "confirmation": untouched_confirmation_mask}
session = GuidedCalibrationSession(
    variables, references, roles, mode="mc", flight_length_mm=110.,
    pulse_mode="laser",  # Use voltage mode only with valid pulse voltages.
    model_config=ToFModelConfig(linear_drift=True, block_edges=()),
    policy=ValidationPolicy(max_abs_relative_bias=.005, max_relative_rmse=.02),
    baseline_mass_da=accepted_mass_array,  # Omit if there is no accepted mass baseline.
    fit_seed=0,  # Fix before looking at selection/confirmation results.
)
show_guided_calibration(session, receipt_path="guided_calibration_receipt.json")
# Equivalent script: preview(), fit(), validate_selection(), confirm(),
# accept(), apply(active_mode="mc"), save_receipt(path).
```

Session source arrays and masks are copied and read-only. Source changes,
reload/reordering, changed active calibration, externally changed model or
protocol invalidate the session. Confirmation is consumed once. Receipts
contain the model, units, normalization, domain, policy, diagnostics and hashes
of source arrays, reference labels, role masks and application mask. Save raw
data and masks separately: hashes verify identity but cannot reconstruct the
original event selections. JSON receipts are audit records, not resumable
serialized sessions.

## Inputs needed when a fit gives mixed results

Keep the prior calibration when the candidate worsens a reference; do not
automatically widen thresholds, relabel peaks or trim a window until it passes.
The useful additional inputs are independently identified isotope/charge
references across a wider mass span; windows with reviewed tails, background
and neighbouring species; measured event-aligned pulse voltage; checked detector
units and flight geometry; and acquisition regions reserved before model
selection. The bundled isotope weights are rounded, so sub-millidalton claims
would additionally need validated higher-precision reference masses.

A broad mass window is not a pure reference. In sample 2643 the historical Fe56
doubly charged window includes the nominal Fe57 doubly charged mass at 28.47 Da.
The diagnostic reports this possibility without deleting those ions or changing
their labels. Isotope-family evidence or narrower independently justified
windows can help review such a case; selecting them with the fitted residuals
alone creates circular evidence.

Raw ToF windows can also combine different masses when voltage or geometry
varies. Inspect a physically normalized mass view with checked instrument
settings when choosing event masks, then fit the original raw times. Declare
linear drift or acquisition blocks only when the instrument/history supports
them, and keep confirmation untouched. More model terms cannot repair an
incorrect identity. The fitted offset remains conditional on the labels and
model; it is not an independent measurement of hardware time zero.

## Reconstruction and ranging corrections kept separate

Mass calibration does not independently determine image compression factor
(ICF), field factor (kf), detector efficiency or density. The reconstruction
helper now suggests the observable near-axis ICF/kf ratio described below.
The previous numerical 2-D SDM sweep remains available as a low-level diagnostic;
zero-evidence, tied and search-boundary sweeps have no applicable best pair.
Saved parameters propagate into the reconstruction panel through explicit
acceptance/loading. Manual reconstruction remains available.

For the canonical reconstruction coordinates
$(x,y,z)=(Ru_x,Ru_y,H+R(1-u_z))$, a pole-plane normal is proportional to
$(-p_x,-p_y,p_z)$. This sign convention flattens the independently constructed
spherical-cap tangent. Detector angular separation uses the angle between
the two full rays $(x_1,y_1,L)$ and $(x_2,y_2,L)$, not the arctangent of
their planar separation. Missing HV is rejected before cumulative depth;
silently dropping such an event would alter all later volume increments.

Ranged mass intervals use `[low, high)`. The first valid range row owns an
overlap, matching existing composition-analysis semantics and avoiding duplicate
species counts. Range-table order therefore matters; resolve ambiguous overlaps
explicitly rather than interpreting this assignment rule as chemical evidence.

## Near-axis ratio suggestion adapted from APT3D

`reconstructions/ratio_calibration.py` implements the smallest useful observable
from the local APT3D work: a static near-axis depth ratio anchored to an
independently indexed plane spacing. **Here $q=\mathrm{ICF}/k_f$**, the user's
convention. The cited APT3D implementation often uses its reciprocal
$q_{\mathrm{APT3D}}=k_f/\mathrm{ICF}=1/q$. This implementation does not import
APT3D, train a neural model or claim to infer a unique ICF/kf pair.

With atomic density $\rho$ in atoms/nm³, field $F$ in V/nm, flight length $L$
in mm, detection efficiency $\eta$, and detector area $A$ in mm², the dominant
depth term in the current PyCCAPT reconstruction is

$$
 P_i=\sum_{j=0}^{i}\frac{1}{V_j^2},\qquad
 C=\frac{L^2F^2}{\rho\eta A},\qquad
 H_i(q)=\frac{C P_i}{q^2}.
$$

The prefix is accumulated **once over every loaded acquisition event from event
zero**, before species selection, detector ROI selection or subsampling. Missing
detector positions still contribute their valid voltage increment. Invalid HV
causes abstention; it is never silently dropped. A crop detected against the
loaded backup is refused. Arbitrary imported files cannot reveal whether earlier
events were omitted, so the UI requires explicit full-acquisition confirmation
and records that coverage as **user-declared**, not independently verified.

The cached area is the full observed circular detector extent, exactly matching
the existing PyCCAPT Geiser/Bas implementation. It is not independently measured
hardware area. Therefore the inferred ratio is conditional on that convention,
efficiency, density, field, projection and correctly indexed spacing. Changing
those assumptions changes the inferred ratio. Six contiguous acquisition bands
are used by default: even bands select q and odd bands validate that fixed
selection. A seeded uniform sample without replacement of up to 2,500 ROI events
per band bounds profile cost; their prefix values still include all intervening
events. Selected rows are sorted back into acquisition order. The sampling seed
and method are recorded. Every band needs at least 100 sampled events and five indexed periods.

Regularly taking every Nth ROI event is deliberately avoided: this can create
an artificial periodic sampling comb. A regression with uniformly spaced,
noncrystalline events gave approximately 0.99984 sampled coherence at a false
q=0.5 under regular-stride selection, versus approximately 0.0002 coherence in
the full event sequence. Seeded random selection abstains on that same input.
This changes sampling only; the full acquisition depth prefix remains exact.

For a declared spacing $d$, the selection objective is the absolute phase
coherence

$$
 R_b(q)=\left|\frac{1}{N_b}\sum_{i\in b}
     \exp\left(2\pi\mathrm{i}\,H_i(q)/d\right)\right|,
 \qquad R_{\mathrm{train}}(q)=\operatorname{median}_{b\;\mathrm{even}} R_b(q).
$$

This amplitude is invariant to a constant phase offset in each band. Neighboring
frequency ratios 0.82, 0.90, 1.10 and 1.18 supply a separate contrast check, using
APT3D's log exact/side coherence with a $1/\sqrt{N_b}$ floor. We deliberately
do **not** maximize that log ratio: known-truth tests showed that a sideband
denominator minimum can shift a short-band optimum away from the physical
period, including making boundary truth appear interior. Absolute exact-period
coherence selects the ruler; neighboring-frequency contrast tests its evidence.

The default coherent-amplitude floor is $\max(0.08,4/\sqrt{N_b})$ and the
log-contrast floor is 0.35 in every training and held-out band. These are explicit
heuristic gates, not p-values. Held-out optima must agree with the training
choice within 2%; they do not update the selected ratio. Weak evidence,
insufficient coverage, changing depth ratio, flat profiles, competing supported
branches and search-boundary support all abstain. Harmonics can produce multiple
ratio branches; narrowing the search is justified only by independent knowledge,
not by deleting an inconvenient alternative.

The descriptive support region is the connected training-profile interval
within 0.10 coherence units of the maximum, with linearly interpolated threshold
crossings between grid points. It is **not a confidence interval**.
The report also includes a 200-repeat training-band bootstrap percentile range,
which describes resampling variability with few bands; it is not calibrated
coverage and omits metrology/model uncertainty. Grid resolution limits both.
The complete profiles, thresholds and counts are retained for review.
The search estimates the required grid step from the maximum within-band depth
span: $\Delta q\le q_{\min}^3d/(4\,\Delta(CP))$. Coarser supplied grids are
refined automatically up to 4,001 points by default. Larger requirements produce
an explicit `grid_underresolved` abstention with the needed count, avoiding a
false weak-evidence conclusion when a long acquisition produces a narrow peak.

This is a **near-axis approximation**, not a general off-axis calibration. Before
profiling, the code screens the omitted terms across the declared ICF family
(five ICF samples and both q endpoints), using the corrected fixed-origin
coordinate

$$
 z_{p,i}=p_z H_i + \frac{V_i q}{\mathrm{ICF}F}
                 (p_z-\mathbf u_i\cdot\mathbf p).
$$

The full observed detector extent must also remain within the projection domain
and the front hemisphere for the entire declared ICF family, because accepted
parameters reconstruct all events, including events outside the pole ROI.
The maximum sampled within-band span of $z_p-H$ must stay below 0.15 periods,
and the pole-cosine contribution to relative q bias below 0.5%. This is a
screen based on the assumed spherical-cap model and sampled events, not a
rigorous bound on unknown specimen geometry. Large off-axis angles or cap
variation return an actionable abstention: choose a tighter central indexed
pole, a narrower independently justified ICF family, or use manual reconstruction.
No angular/depth-dependent surface is fitted from insufficient evidence.

For a supported q, the answer is the family

$$
 k_f=\frac{\mathrm{ICF}}{q},\qquad
 k_f(\mathrm{ICF})\in
 \left[\frac{\mathrm{ICF}}{q_{\mathrm{high}}},
       \frac{\mathrm{ICF}}{q_{\mathrm{low}}}\right].
$$

The helper displays this family and requires explicit confirmation of a chosen
manual/independent ICF anchor before acceptance. It preserves that ICF and sets
its compatible kf, stores the profile/assumptions/anchor in
`Variables.reconstruction_ratio_result`, and propagates the parameters through
the existing reconstruction panel. Source changes prevent acceptance; changed
geometry, lattice or profile inputs require rerunning. A subsequent manual
parameter override clears the ratio receipt instead of misrepresenting it as
evidence for the changed reconstruction.

Programmatic use is `prepare_ratio_geometry(...)` followed by
`profile_ratio_geometry(cache, spacing_nm, q_grid, icf_bounds=...)` when reusing
one source cache, or `profile_icf_kf_ratio(...)` for one call. Review `status`,
`diagnostics`, profiles and `compatible_parameter_family(result, anchors)`;
only `accept_ratio_family(variables, result, icf_anchor=...)` updates settings.

The notebook ratio preview runs as a background job with progress and cooperative
cancellation between profile points. Full-source prefix preparation remains one
array operation; cancelling during preparation discards its eventual result.
Parameter-changing controls are disabled during the job. Widget inputs are
captured before launch and source/input consistency is checked before showing a
result. Cancelled or stale results cannot become an accepted ratio, including
when cancellation arrives after computation but before notebook delivery.

Local source attribution (read-only comparison, no runtime dependency):

- `E:/apt3d/reconstruction_correction/orientation_workflow/near_axis_depth_ratio.py`:
  ratio-only observability, exact-period/neighbor-frequency diagnostics,
  observation bootstrap and depth-band transfer.
- `E:/apt3d/reconstruction_correction/orientation_workflow/exact_single_pole_phase.py`:
  `paraxial_ratio_only_coordinate_nm`, full-event prefix, exact launch geometry,
  and the corrected `fixed_origin_pole_surface_shape` term.
- `E:/apt3d/reconstruction_correction/orientation_workflow/angular_depth_ratio.py`:
  nested angular/depth models require more independently supported observations;
  deliberately not adopted by this bounded static-ratio workflow.

## Crop proposals and processing provenance

The editable automatic crop shapes are selection heuristics, not new mass or
reconstruction equations. The temporal proposal uses up to 512 acquisition windows:
local residual roughness is `1.4826 * median(abs(residual - median(residual)))` after
removing an endpoint trend. Jump/drop, roughness and finite-positive-voltage tests
mark bad windows; a one-window margin is added, and the longest passing interval
is proposed. Similar-length intervals are reported as ambiguous.

The detector proposal smooths a full-hit histogram (Gaussian sigma 1.2 bins),
uses Otsu separation in `log(1 + density)` when the density contrast supports a
core/halo interpretation, otherwise uses a footprint threshold. It fills enclosed
holes, retains the largest connected component, then maximizes distance to its
boundary in detector cm. Radius is reduced by 0.75 bin diagonals. Edge-connected
bites stay outside the support; these thresholds cannot certify an artifact-free
or composition-neutral ROI. A weak/degenerate result abstains. Details and manual
overrides are in the [workflow guide](guided_workflow.md#how-the-automatic-crop-proposal-works).

The shared journal records working fits, model/settings state, resets and final
output fingerprints. It does not change the voltage, bowl, residual, physical
ToF conversion or Geiser/Bas equations. Exact same-source restoration reads the
saved output checkpoint after verifying raw measurements and event IDs; it does
not refit or claim an interchangeable calibration for another acquisition.
