# Guided processing notebook

Open `pyccapt/calibration/tutorials/jupyter_files/data_processing_guided.ipynb`
with the calibration dependencies installed (`pip install -e ".[calibration]"`
or the project's full environment). Run the two code cells. The panel is a
separate, optional workflow; the original processing notebook is preserved.

## Accepted state and dependencies

The tabs follow Import → Inspect → Calibrate → Range → Reconstruct → Analyze →
Export. Opening a panel only renders controls. Dataset reads, fitting, acceptance,
and writes require explicit actions. Every derived stage has a preview and an
acceptance step. The header distinguishes accepted values, pending previews, and
review edits that have not been applied. Comparisons report the change in mass or
coordinates and the number of ranged/unranged events.

Accepting mass calibration clears the accepted ranges, geometry, and analysis.
Accepting revised ranges clears geometry and analysis. Accepting geometry clears
analysis. This conservative dependency rule prevents a mixture of results from
different processing states. **Undo accepted change** restores a complete prior
accepted state (the most recent 12 changes); **Undo review** restores a peak
decision (the most recent 50 changes). Draft peak decisions do not modify the
accepted assignment until Preview ranges → Accept ranges.

The workflow owns a copy of the imported event table and immutable event-aligned
derived arrays. Stable unsigned 64-bit event IDs retain exact values even above
2^53. Source identity is checked when shared Variables is replaced or reordered;
the full source hash is checked again before persistence. Do not edit the
workflow's imported source directly. Reimport changed data as a new workflow.

## Import, inspect, and calibrate

Import supports PyCCAPT processed/raw HDF5 and existing POS/EPOS/APT/ATO loaders.
Inspect reports measured capabilities and unknown positions. Unavailable fields
are not interchangeable with measured zero. Placeholder imported mass columns
cannot be accepted as calibrated mass. An available uncalibrated mass column is
labelled with its source column in the receipt; accepting it is an explicit
decision, not evidence that its instrument assumptions are correct.

If a legacy calibrated mass column contains no finite positive values, the
preview falls back to an available uncalibrated mass column and reports the
chosen column. Importing calibrated mass preserves an available corresponding
`t_c (ns)` vector; an uncalibrated mass fallback is never paired with calibrated
ToF. Measured finite zero ToF is retained in an otherwise usable vector, while
all-zero legacy ToF placeholders are omitted. Current control defaults are not
written as historical instrument settings when merely accepting imported mass.

Enter the actual pulse mode, flight length in mm, and time zero in ns. Physical
conversion previews raw ToF using the existing `tof2mc` convention and checks
finite geometry and positive voltage; unavailable output remains NaN. Imported
mass can instead be previewed and explicitly accepted. These controls' initial
values are examples, not detected instrument measurements.

Optional multi-reference calibration proposes windows in the raw-ToF spectrum.
Fill in independently known `mass_Da` identities, edit each `tof_min_ns` and
`tof_max_ns`, and confirm the identities before creating a session. Inspect
truncation, tails, overlap, and coverage first. The optional model jointly fits
time zero and positive voltage/spatial/time scale as described in
[calibration_math.md](calibration_math.md). The existing guided protocol enforces
training → selection validation → single confirmation → acceptance → application.
Capture the applied model as a workflow preview and accept it separately.
Reference-window construction excludes invalid source rows and reports the
excluded count; every required reference/validation group must still meet the
model's coverage rules.

Interleaved 60/20/20 role assignment shares acquisition conditions and is a
correlated holdout, not independent experimental validation. Contiguous 60/20/20
roles can reveal drift but must contain each reference in each required block;
domain and coverage failures remain failures. Neither split certifies chemistry
or real-data performance. Learned coefficients/trees/model configuration are
saved with role masks, reference labels, application mask, and receipts, rather
than reconstructing a model from settings alone.

## Faster peak review with explicit decisions

Choose expected elements, maximum charge, maximum atoms per molecule, histogram
bin size, prominence, mass limits, and candidate tolerance. **Propose peaks**
finds spectral maxima and proposes editable windows. It does not assign chemical
identities. Candidates are ordered by absolute mass residual, atom count, and a
stable tie-breaker. These are display rules, not calibrated probabilities.

The isotope catalogue uses canonical multisets (not atom-order permutations).
The cache is bounded to four immutable catalogues and its key includes elements,
charge set, molecular complexity, and the isotope-table content hash. Candidate
search uses sorted mass bounds. Requests exceeding the explicit catalogue budget
fail with guidance to reduce scope; no candidates are silently truncated during
generation. Individual peak displays show up to 30 nearby candidates. Natural
abundance products are metadata only, without an automatic abundance prior.

The queue shows the spectrum, proposed bounds, event count, candidate mass
residuals, prior decisions, and overlap/ambiguity warnings. Its actions are:

- **Choose identity**, **Accept identity**, and **Reject identity**. Rejection
  removes that candidate from the peak's available choices.
- **Not a peak**, **Defer**, **Next**, and **Undo review**.
- **Edit bounds**, **Split**, and **Merge selected**. Changed intervals require
  renewed acceptance; split/merged intervals start without an identity.
- **Accept family**, after choosing distinct isotope identities with the same
  elemental composition and charge, reviewing consistent mass residuals, and
  explicitly confirming family and ambiguity checks.

Locked accepted and not-a-peak decisions survive rerunning detection on the same
accepted mass spectrum. A new accepted calibration requires a fresh queue. On
rerun, unaccepted identity choices are cleared so an identity from an obsolete
catalogue cannot be accepted against newly proposed candidates. On
resume, split/merge lazily restore the original catalogue only if the installed
isotope-table hash matches; otherwise rerun proposals and review new candidates.
Changing an identity or window clears the earlier family approval and fit
evidence. Concurrent background results cannot overwrite newer decisions.

**Auto zoom** follows the selected peak. Turn it off and set View min/max to keep
a fixed spectrum view across decisions. Histograms use binary searches into one
sorted finite mass array and read only events within the plotted bounds. A
three-entry cache avoids recomputing repeated views. Counts are exactly those of
`numpy.histogram`, including its inclusive final edge. Integer assignments need
one event-sized vector; they do not allocate an event-by-range boolean matrix.
Intervals are lower-inclusive/upper-exclusive. Overlap uses first accepted row
priority and is explicitly warned; prefer splitting or editing when possible.

Optional **Profile evidence** uses existing asymmetric/Voigt peak diagnostics.
Optional **Mixture evidence** requires confirmed component identities and an
accepted isolated empirical reference peak. It displays fitted counts and
residuals alongside conditional uncertainty and ambiguity diagnostics from
[calibration_isotope_family.md](calibration_isotope_family.md). It does not change
event assignments or decide identities. Natural-abundance priors are disabled.

## Reconstruction and analysis

Manual Geiser/Bas geometry uses the entered ICF, kf, detector efficiency,
evaporation field in V/nm, atomic density in atoms/nm³, and flight length in mm.
Voltage-mode reconstruction uses the existing DC-plus-pulse voltage convention,
the same convention as ratio profiling. Calibration's effective acceleration
formula is a distinct documented model and is not substituted for reconstruction.

The optional **Review ICF/kf ratio** panel runs in a separate workspace. Review
the acquisition completeness, pole/ROI, indexed spacing, depth-band validation,
and parameter family. A ratio q = ICF/kf does not uniquely determine both
parameters; acceptance requires an independently anchored ICF. Capture ratio
geometry computes coordinates from those accepted settings, then requires
workflow acceptance. Ratio evidence and selected settings survive session resume.
The restored controls display saved accepted geometry rather than fresh defaults.

Analyze reports counts by ion and stoichiometric atomic fractions of assigned
atoms. It aggregates repeated ion labels and reports unranged events. These are
hard-interval results, not background-corrected concentration estimates or
overlap-deconvolved event labels.

## Progress, cancellation, save, and resume

Import, proposal generation, physical conversion, optional evidence, geometry
previews, and reference-model fitting run as notebook background jobs. Chunked
loops check cancellation. Some SciPy/NumPy kernels are not internally
interruptible: cancellation discards their result when the call returns. Accepted
state is unchanged. A concurrent accepted-stage change invalidates old previews.
The nested fit/review panels have their own Cancel buttons. Explicit file saves
are synchronous atomic writes; let them finish before closing the kernel.

**Save resumable session** stores a complete source snapshot, numeric dtypes,
source metadata/hash, stable event IDs, accepted outputs, preview, masks, ranges,
review history, learned model state when available, and undo history in versioned
HDF5. Identical arrays are deduplicated and hashes are computed in blocks. Safe
JSON plus numeric arrays are used; loading does not execute pickle or Python.
Resume validates integrity, source identity, aligned outputs, range assignments,
model state, and dependencies in a temporary workspace before replacing live
state. Session folder/search controls read compact headers; full verification
happens when a selected session is loaded.

**Export accepted HDF5** rebuilds mass, optional calibrated ToF, and optional XYZ
from accepted state. Raw source columns do not override accepted outputs; stale
geometry/ToF columns are omitted. It embeds the complete analysis session and the
available NeXus mapping. **Export NeXus mapping** creates the standalone mapping.
This is explicitly a partial NXapm-aligned representation; missing instrument,
acquisition, and coordinate-system context are reported, not fabricated. A
session may be saved before calibration, but derived-data export requires an
accepted mass spectrum.

## Validation and boundaries

Synthetic regression tests cover canonical catalogue enumeration/hash changes,
bounded exact histograms, locked decisions and undo, mandatory family review,
preview isolation, dependency invalidation, source replacement, transactional
resume, exact 64-bit identities, model-state persistence, integer assignments,
voltage-mode reconstruction parity, cancelled/stale jobs, and widget click paths.
The shipped notebook cells and a complete synthetic UI click path are executed
in-process, including fitting, acceptance, reviewed ranges, reconstruction,
analysis, export and resume. This is separate from testing a live Jupyter kernel
and browser rendering. `tools/validate_guided_notebook_ui.py` can replay an
explicitly supplied dataset and historical range file without fitting new
calibration or inventing apparatus settings. It reports unmatched proposals,
window agreement, exact imported-mass retention, source hashes and persistence
checks. No claim of real-acquisition calibration accuracy follows from these
software tests. Large datasets still require memory for the source, accepted
outputs, sorting, and bounded undo snapshots; this workflow does not claim
out-of-core processing.

The [same-dataset validation report](workflow_validation.md) covers the legacy
notebook's 5,842,702-event sample 2643, including exact calibration persistence,
reviewed-ranging comparisons, a conditional multireference fit and old/new
reconstruction suggestions. It also records where the evidence does not justify
a better calibration or a parameter suggestion. The replay runner accepts
`--mass-min` and `--mass-max` to bound peak detection without cropping source
events; proposals outside the window are not silently accepted or discarded
from saved data.
