# Guided processing notebook

Open `pyccapt/calibration/tutorials/jupyter_files/data_processing_guided.ipynb`
with the calibration dependencies installed (`pip install -e ".[calibration]"`
or the project's full environment). Run the two code cells. The panel is a
separate, optional workflow; the original processing notebook is preserved.

## If the Crop tab is missing

After updating PyCCAPT, choose **Kernel → Restart Kernel and Run All**. Python
keeps imported modules in the running kernel, so rerunning an ordinary import
cell can keep an older panel even when this notebook's instructions are current.
Save a resumable session first if you need to retain work from that kernel.

The startup cell now checks the loaded workflow and displays its source folder
and Python executable. The tabs must read **Import → Inspect → Crop → Calibrate →
Range → Reconstruct → Analyze → Export**. A stale or mixed runtime stops with an
instruction to restart. If the loaded source folder differs from your checkout,
select the kernel for that installation. Existing saved widget output may still
show an older panel until you rerun the launch cell.

For the raw 1891 Al file, reimport after restarting, accept your sequence/detector
crop, then enter **110 mm**, **38 ns**, and **voltage** mode. Tick the instrument
confirmation checkbox, click **Preview physical mass**, inspect the spectrum,
then **Accept calibration**. **Preview imported mass** requires a saved mass
column and is disabled for this raw source. The physical conversion supplies an
initial spectrum; optional reference fitting is a later refinement. Proposed
reference rows with `mass_Da: null` still require independently identified masses.
The comparison's `finite_accepted_values: 0` means you have not yet accepted a
mass calibration; it does not mean that the finite preview failed.

## Start here: raw data with no mass column

A raw event tells us **when an ion arrived**, its detector position and the
voltage. Calibration turns that flight time into a mass-to-charge estimate.
You do not need an `mc` column to begin.

1. **Import:** choose the HDF5 file and leave the mode on **Auto detect**. The
   workflow recognizes raw `/dld` hits and processed `/df` tables. Raw detector
   channel streams must first be converted into hits in the raw-data notebook.
2. **Inspect:** look at flight time and voltage versus event position and the
   detector map. Look for startup transients, interruptions and unwanted regions.
3. **Crop:** click **Draw crop regions**. Drag a rectangle across the part of the
   DC-voltage curve with a smooth increase. Its horizontal span fills the start
   and stop event positions; its height does not add a voltage filter. Then draw
   a circle inside the detector density map, avoiding sparse edges and mouse
   bites. Check the numeric fields, click **Preview crop**, check the count and
   plots, then **Accept crop**. These controls use event positions, not the
   hardware start counter. Start is included; stop is excluded.
4. **Calibrate:** enter the actual flight length, time zero and pulse mode. Check
   their confirmation box, click **Preview physical mass**, inspect the spectrum,
   then **Accept calibration**. This is the initial estimate. For processed data,
   use **Preview imported mass** to retain its existing calibration instead.
5. **Range:** choose plausible elements and charges, propose peaks, review each
   identity and its boundaries, then preview and accept the reviewed ranges.
6. **Reconstruct:** use reviewed material/instrument parameters. The optional
   ICF/kf panel can suggest a ratio only when a correctly indexed crystal feature
   supplies enough evidence. It explains an unsupported result.
7. **Analyze:** inspect the composition of accepted ranges and the unranged count.
8. **Export:** save a resumable session at any time. Export accepted HDF5 when you
   want a selected-event table plus metadata, NeXus mapping and embedded session.

**Preview** shows a proposal; **Accept** makes it current. If an earlier step
changes, dependent results are cleared. **Undo accepted change** restores the
previous state. A failed optional fit leaves your accepted calibration intact.

The inspection plots display a bounded sample for responsiveness; crop counts
and accepted data use every event. Cropping stores a mask and preserves the full
source in memory and in the session. Reconstruction still needs earlier events
for its cumulative depth calculation. The exported table contains only selected
rows; a cropped table by itself cannot supply missing acquisition history.

### Drawing the crop

The two drawings implement the sequence-plus-detector selection described in
the [PyCCAPT paper, Section 2.3 and Figure 3](https://doi.org/10.1002/jemt.70011).
You choose the region: a smooth voltage trace and a bright detector region help
identify useful data, but they do not establish composition or calibration quality.

- The sequence rectangle uses the full source's event positions. Fractional
  drag bounds are rounded outward (`floor(left)`, `ceil(right)`) and clipped to
  the source length. Exact start/stop fields remain editable.
- The FDM is a logarithmic density histogram using every finite detector hit in
  the draft sequence interval. This is not a subsampled scatter plot. It shows
  that interval before the circle, optional ToF window, or optional voltage
  window is applied, so excluded edges remain visible for review.
- Draw a box around the desired circular area; the displayed circle fits inside
  it. Its center and radius are in cm, with equal detector-axis scaling. Reverse
  drags work. Fine-tune the center and radius numerically if necessary.
- After typing a new interval, click **Update detector map**. Drawing an interval
  updates the map automatically. Reopen **Draw crop regions** to start a new
  drawing. Keep the plot toolbar's pan/zoom tools off while selecting.
- Drawing or typing changes only the draft. Editing a crop field clears the old
  preview, so you cannot accidentally accept a different crop than the displayed
  values. **Preview all events** resets the draft limits; **Undo accepted change**
  restores the previous accepted state and its fields.

The interactive canvas uses `ipympl`, included in the calibration dependencies.
Use a Jupyter frontend with widget support; no global Matplotlib backend change
is needed. Numeric crop controls remain usable without the interactive canvas.

## Accepted state and dependencies

The tabs follow Import → Inspect → Crop → Calibrate → Range → Reconstruct → Analyze →
Export. Opening a panel only renders controls. Dataset reads, fitting, acceptance,
and writes require explicit actions. Every derived stage has a preview and an
acceptance step. The header distinguishes accepted values, pending previews, and
review edits that have not been applied. Comparisons report the change in mass or
coordinates and the number of ranged/unranged events.

Accepting a crop clears calibration, ranges, geometry and analysis. Accepting
mass calibration clears the accepted ranges, geometry, and analysis.
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

Import supports PyCCAPT processed/raw HDF5, this project's partial NeXus mapping,
and existing POS/EPOS/APT/ATO loaders.
Raw hit import reads only the `/dld` group. It does not load the often much
larger `/tdc` channel arrays unless a separate raw-channel operation requests
them. This reduces unnecessary I/O and memory without changing hit values,
units or the downstream calibration equations.
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

Enter and confirm the actual pulse mode, flight length in mm, and time zero in ns. Physical
conversion previews raw ToF using the existing `tof2mc` convention and checks
finite geometry and positive voltage; unavailable output remains NaN. Imported
mass can instead be previewed and explicitly accepted. Known source metadata
prefills the controls; otherwise their initial values are examples, not detected
instrument measurements. Editing these fields clears confirmation. The mass-view
limits change the displayed spectrum only; they do not crop data.

Changing physical-conversion settings also discards its pending preview. A
calculation already running cannot publish a result if the settings or their
confirmation changed meanwhile, even if you change them back. Preview again
after confirming the current values. A resumed physical preview restores its
own settings and requires confirmation before acceptance. Imported mass is
independent of those controls and retains its recorded provenance.

Optional multi-reference calibration proposes windows in the accepted initial
mass spectrum (default), or in raw flight time. Fill in independently known
`mass_Da` identities and edit each window's `mass_min_Da` / `mass_max_Da` or
`tof_min_ns` / `tof_max_ns`. Mass windows require an accepted initial calibration;
they depend on it and are not independent reference evidence. Confirm the
identities and instrument settings before creating a session. The model fits its
own time zero; the current time-zero field supplies the initial conversion only. Inspect
truncation, tails, overlap, and coverage first. The optional model jointly fits
time zero and positive voltage/spatial/time scale as described in
[calibration_math.md](calibration_math.md). The existing guided protocol enforces
training → selection validation → single confirmation → acceptance → application.
Capture the applied model as a workflow preview and accept it separately.
For the button-by-button sequence:

1. Keep **Window basis** on **Accepted initial mass** and click **Propose reference
   windows**. Review the limits, remove contaminated proposals, and replace each
   `mass_Da: null` with an independently identified mass-to-charge value.
2. Confirm the identities and instrument settings, then click **Create reference
   session**. This opens the fitting panel below the main controls.
3. Click **Fit training**, **Validate selection**, then **Confirm once**, inspecting
   the reports at each step. A failing report stops the sequence.
4. When the checks pass, click **Accept model**, then **Apply accepted**. Leave
   **Apply mode** on **Mass (Da)** for this route.
5. Click **Capture applied model** in the main panel, inspect the new spectrum,
   then **Accept calibration** to replace the initial conversion.

The initial physical spectrum is enough to start selecting reference peaks; it
does not establish that voltage and detector-position corrections are complete.
Reference-window construction excludes invalid source rows and reports the
excluded count; every required reference/validation group must still meet the
model's coverage rules. Validation checks both absolute errors and changes
against the available baseline on identical events in fixed acquisition bands.
Worsening absolute median bias or relative RMSE prevents replacement. This is a
conservative comparison, not a statistical significance test or a guarantee of
chemical accuracy. The equal-reference robust loss and seeded fitting sample are
documented in [calibration_math.md](calibration_math.md).

Interleaved 60/20/20 role assignment shares acquisition conditions and is a
correlated holdout, not independent experimental validation. Contiguous 60/20/20
roles partition the selected usable events in their original order and can reveal
drift. Each role must contain each reference in its occupied validation bands.
A stationary model may test later events with in-domain voltage and detector
conditions; time-dependent models cannot extrapolate to unseen time ranges or
block categories. Domain and coverage failures remain failures. Neither split certifies chemistry
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

Use the detector map to review the ROI before profiling. Detected features are
unindexed candidates, not crystallographic assignments. Selecting one requires
review; changing the ROI or lattice information clears spacing confirmation.
The report shows evidence for each acquisition band, so a weak average cannot
hide a strong but inconsistent band. A table explicitly marked as a cropped
export cannot be treated as a complete acquisition by ticking a confirmation box.

For sample 2643 the tested central region has insufficient periodic evidence in
every band. Useful extra inputs are a confirmed phase/lattice and indexed plane
spacing (including diffraction order), an indexed near-axis pole/ROI, an
independent ICF anchor (for example two indexed poles with known angular
separation), field/density/efficiency estimates, and the complete original event
history. After searching ROIs, reserve a fresh interval for confirmation. See
[the dataset investigation](workflow_validation.md) for measured results.

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

**Export accepted HDF5** writes selected events and rebuilds mass, optional
calibrated ToF, and optional XYZ from accepted state. Raw source columns do not override accepted outputs; stale
geometry/ToF columns are omitted. It embeds the complete analysis session and the
available NeXus mapping. **Export NeXus mapping** creates the standalone mapping.
This is explicitly a partial NXapm-aligned representation; missing instrument,
acquisition, and coordinate-system context are reported, not fabricated.
Instrument metadata separates run-start settings, preview inputs and accepted
calibration settings, with units and provenance. A fitted time zero is labelled
as a model result, not a hardware measurement. Importing saved masses preserves
their historical metadata. Both this notebook and the original notebook can
read this project's mapping; legacy `save_data` includes it by default. A
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
