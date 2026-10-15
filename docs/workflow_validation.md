# Guided workflow validation — 15 October 2026

## Actual raw input supplied later: sample 1891

The user supplied
`D:/APT_data/1891_Jul-05-2024_12-10_Al_test_50um_1.h5`, an actual raw HDF5 file.
It contains **6,058,440 detector-hit rows**, **25,105,070 TDC channel rows** and
38,547 control observations, with no mass column. Unlike the derived 2643 replay
below, this file directly exercises the raw acquisition layout.

Inspection found that the eager DLD importer read every group, including all TDC
channel rows, before discarding the unrelated data. It now requests only `/dld`;
explicit TDC imports request only `/tdc`. The general `read_hdf5` default retains
its previous all-group behavior. Regression tests guard unrelated dataset reads
and preserve values, exact integer counters, legacy aliases and unit conversions.
This is an I/O optimization; no calibration mathematics changed in this follow-up.

The file has no stored attributes, instrument settings, units or completeness
receipt. Its units follow the recognized legacy PyCCAPT layout. Nonzero voltage
pulses and zero laser-pulse values suggested voltage pulsing. The user subsequently
confirmed **voltage mode, 110 mm flight length and 38 ns time zero**. The replay
uses these explicit inputs and records them as conversion settings; it does not
claim they were present in the original acquisition file.

`tools/validate_raw_guided_workflow.py` executes the shipped notebook cells and
widget callbacks. The full-file replay verified every imported DLD measurement
and exact unsigned counter against the source, then checked crop, accepted mass,
selected-table export, embedded session and standalone save/resume. Source SHA-256
before and after was
`7d44afdf1d403050cdefb5c3cf7cebd25773ba178075989626f37550dfd2ff9a`.

The **software-test crop** `[1000, 6057440)` retained 6,056,440 events; it is not a
scientifically selected smooth-voltage interval or detector ROI. Of these,
6,044,752 gave finite masses identical to the existing `tof2mc` function. The
11,688 selected events with ToF at or below 38 ns remained NaN, as did the 2,000
crop-excluded events in the full-source session. No peak identities, reference
fits, ranges or ICF/kf parameters were accepted. The initial conversion's median
mass was 31.13 Da and its 99th percentile was 1714.17 Da; the broad unfiltered
distribution is not evidence of a valid Al mass calibration.

The initial raw import/inspect took 4.20 s. The confirmed conversion, export and
resume replay took 80.98 s including hashing and setup on this machine. These
are observed timings, not a hardware-independent speed claim. Reports are under
`analysis_outputs/1891_raw_ui/confirmed_physical/`. Reproduce with:

```console
python tools/validate_raw_guided_workflow.py --data D:/APT_data/1891_Jul-05-2024_12-10_Al_test_50um_1.h5 --output-directory analysis_outputs/1891_raw_ui/replay_new --event-start 1000 --event-stop 6057440 --flight-mm 110 --t0-ns 38 --pulse-mode voltage
```

Choose a fresh output folder: the validator refuses existing result artifacts
so a failed callback cannot be mistaken for a successful earlier save.

Interactive cropping now follows the user's procedure: a rectangle chooses the
event interval on the DC-voltage plot, and a true circle selects the detector
region on a logarithmic FDM. Both remain drafts until preview and acceptance.
Regression tests cover old-preview invalidation after an edit, selection bounds,
source lifetime, busy callbacks, and undo restoring visible fields. These changes
do not alter the mass-conversion equations or infer a scientific ROI automatically.
Physical-preview tests also cover input edits during calculation, editing a
setting back to its original value, resumed-preview confirmation, and preserving
imported mass independently of the physical-input controls. Export preserves
recorded acquisition completeness instead of replacing known incomplete data
with an unverified label.

An additional actual-data drawing replay used the real `ipympl` canvas with
synthesized mouse events. An illustrative rectangle and circle retained
2,240,111 events; an independent all-row circle/intersection calculation matched
the preview and accepted masks exactly. All 3,635,065 finite detector hits in
that draft interval matched an independently calculated density histogram.
Acceptance cleared downstream calibration; undo restored the prior mass array
exactly, restored the visible crop fields, and disconnected the old drawing.
The original raw file and saved input session were unchanged. The script,
report and plotted canvases are under `analysis_outputs/1891_raw_ui/drawing_check/`.
This exercises the widget canvas and callbacks, not browser event transport.

The subsequent crop-usability update opens the interactive canvas when entering
Crop, with an event rectangle and an enclosing detector circle already visible.
On the full 1891 raw file, these starting regions retained all **6,058,440 events**.
Actual handle events shrank and enlarged the circle, moved its center and resized
both rectangle edges. An illustrative selection retained **3,227,778 events**;
its preview and accepted mask matched an independent all-row calculation exactly.
Undo restored all events, all source columns still matched the raw file, and the
input SHA-256 remained unchanged. The check took 25.53 s including hashing;
its script, initial/adjusted plots and report are under
`analysis_outputs/1891_raw_ui/automatic_selectors/`. The selected region is a
software example, not a scientifically recommended ROI. **110 integration tests**
passed in 33.38 s, including real mouse-handle events, crop-state preservation,
raw calibration, save/resume and the shipped notebook's full synthetic workflow.

The follow-up's combined targeted suite passed **374 tests** in 40.33 s, covering
raw import, guided calibration/cropping, plotting events, storage, metadata and
control-file preservation. Ruff and whitespace checks passed. The documentation
build succeeded with 16 warnings in existing API docstrings; the final incremental
guide/report build succeeded without warnings. No instrument was operated for
these checks.

All DLD values are finite. Hardware start counters decrease 63,799 times and are
not a global event coordinate. Control event counts sum to 6,151,914, which is
93,474 more than the stored DLD rows. The cause is unresolved: the old counter
semantics are not recorded, so this difference is not proof of lost hits or of
complete acquisition. The adjacent processed temporary file has 4,373,572 rows
and lacks recorded crop criteria and instrument settings; it cannot substitute
for the complete raw event history.

## Follow-up: mixed calibration, missing ratio evidence and raw-data support

The first comparison below is retained as a historical audit. The subsequent
investigation changed the optional reference model and its acceptance checks;
its current outcome on sample 2643 is **selection failed**, not a better accepted
calibration. No candidate was applied to the user's data.

### Why the calibration results were mixed

The original aggregate checks could pass even when some reference peaks moved
farther from their assumed target. Also, the equal-reference soft-L1 fit put
reference weights inside the residual, which changed the outlier cutoff with
reference population. Weights now multiply the robust loss itself. A replication
counterexample verifies that repeating one reference 20 times no longer changes
its robust influence: the corrected fitted offset agrees within 1.3e-11 ns,
whereas the old formulation shifted by about 0.00651 ns. This verifies the
weighting logic, not the physical accuracy of the real-data model.

Fitting uses fixed-seed sampling without replacement. Validation now reports
each reference in fixed acquisition bands, compares against the baseline on
the same events, and rejects worsening absolute median bias or relative RMSE.
The existing absolute error gates remain unchanged. Blocked holdouts work for
stationary models without allowing time-dependent extrapolation or unseen block
categories. The equations, sampling policy and limits are in
[calibration_math.md](calibration_math.md).

The corrected candidate fails in **8 of 20 reference/time groups**: three Ni58
bias groups, all four Ni60 bias groups and the first Mo96 RMSE group. Confirmation
is left untouched, and no acceptance/application occurs. The diagnostic took
11.94 s, including 0.554 s fitting. The historical Fe56 window also includes
nominal Fe57; this contamination and provisional identities limit what any fit
can establish. A change in peak location is not evidence of improved peak width.
Source hashes and active calibration arrays remain unchanged.

### Why there is no reliable ICF/kf pair

All six tested acquisition bands have scanned maximum coherence **0.0803–0.0957**,
below their unchanged required **0.1078–0.1146**. Checking the bands individually
does not reveal a strong signal hidden by averaging. Using the exact spherical
cap/pole correction at the replay's fixed ICF anchor gives **0.0804–0.0960**,
still with zero passing bands. The detector-feature finder at its standard
contrast setting returns no candidates. These findings apply to the tested
central region and assumptions; they do not prove that all other ROIs lack
usable planes.

The panel now exposes per-band evidence, detector-map review and specific failure
reasons. Detector features require manual crystallographic review. Evidence
thresholds are unchanged; the code does not turn a numerical maximum into a
supported answer. Even a well-supported ratio only fixes a family of pairs:
`kf = ICF / q`; an independent anchor is needed to select one.

### Additional inputs that would help

| Purpose | Needed input |
| --- | --- |
| Establish the specimen | Confirm whether this is the NiC1 specimen or the 316L described in the historical log; supply expected phases/composition. |
| Improve mass calibration | Clean, nonoverlapping, independently identified isotope/charge windows covering the voltage range, detector area and run duration; exclude or explicitly model contaminated peaks. Provide the actual flight length, pulse mode and voltage convention. A known time offset helps the initial physical conversion; a fitted offset remains conditional on references. |
| Establish reconstruction depth | Original acquisition from event zero, or documented prefix information, plus any interruptions, cropping and losses. Contiguous exported row IDs alone do not establish completeness. |
| Estimate q | An indexed near-axis pole/ROI and confirmed lattice spacing/order, with reviewed field, density and detector efficiency. Search for a stronger region without treating hotspots as indexed poles. |
| Choose ICF and kf separately | An independently constrained ICF, such as two indexed poles with known angular separation, or suitable independent specimen-geometry measurements. |
| Confirm a searched solution | Reserve an unscreened interval after exploring ROIs/models, or provide an independent acquisition. Interleaved events from this file are correlated holdouts. |

### Raw-data and cropping replay

No original raw acquisition HDF5 was found in the supplied folder. A local
**derived raw-style `/dld` file** was therefore made from this dataset's recorded
ToF, detector and voltage fields, omitting all mass and XYZ columns. This tests
the raw interface on real measurements; it does not prove original acquisition
completeness or replace testing the user's actual failing raw file.

The guided widget callbacks imported **5,842,702** events, cropped positions
`[1000, N-1000)` to **5,840,702**, and used explicitly confirmed 110 mm / 38 ns /
voltage-mode settings. The physical mass preview matches the source `mc_uc`
**exactly** for every selected event. The 2,000 excluded outputs remain NaN.
The selected export retains original event IDs and declares that its table is
cropped; the embedded session restores the full source, mask and cumulative
voltage context. Both input hashes remain unchanged. Import took 3.00 s,
conversion preview 3.70 s, export 26.94 s and resume 7.87 s on this machine.
No peak identities or reconstruction were accepted during this replay.

Regression tests additionally check crop boundaries, invalid/empty crops,
dependency invalidation and undo, full-prefix reconstruction parity, selected
role splitting, old-session migration, corrupt restored masks/criteria/previews,
and required instrument confirmation. Metadata tests cover original-workflow
NeXus import/export and control run-start settings with fake hardware objects.

Local detailed reports are under `analysis_outputs/2643_calibration_revision`,
`analysis_outputs/2643_ratio/investigation` and
`analysis_outputs/2643_derived_raw_ui`. They are ignored validation artifacts;
the reproducible diagnostic runners and this interpretation are versioned.

### Follow-up software validation

The final collection contains **1,278 tests**. Across the broad run and focused
reruns, **1,277 have passing results**; the optional performance/memory test is
skipped. The broad run passed 1,268 tests and exposed one empty-panel metadata
regression. That failure was fixed, with 39 legacy/metadata tests passing,
including a new no-mutation regression. The final guided workflow subset passed
70 tests, including the new physical-input and selected-mass fallback cases.

The unchanged adaptive peak-fitting test was isolated because it takes several
minutes; it passed in 230.47 s on supported Python 3.12.7 with NumPy 1.26.4 and
SciPy 1.13.1. Earlier interrupted/timeout-diagnostic attempts are not counted as
passes; one produced a native Windows access violation. The completed isolated
run used no timeout dumping and unchanged numerical/test settings. Native
OMP/OpenBLAS/MKL thread limits were set to one for these checks.

Ruff and staged whitespace checks passed. The full documentation build succeeded
with the existing docstring/cross-reference warnings; the final incremental
build after correcting the new link completed without warnings. Notebook
execution was disabled for documentation generation. No physical hardware
alignment or acquisition was performed. This validation preceded the separately
supplied sample 1891 raw file described above.

## Same-dataset comparison: sample 2643

The user supplied the local copy of the legacy notebook's selected dataset at
`D:/APT_data/2643_Sep-02-2026_17-35_NiC1_HPT_Dc_1`. The main HDF5 file under
`data_processing` contains **5,842,702 events**, matching the notebook's loaded
count. The matching 36-row range file is under
`data_processing/2643_Sep-02-2026_17-35_NiC1_HPT_Dc_1/data_processing`.
The tests below use this dataset, not the supplementary Nimonic/Al datasets.

The saved notebook records voltage mode, a 110 mm flight length and a 38 ns time
offset. Other cells contain older outputs from sample 2590, so the notebook is
not a complete calibration history. Neither the HDF5 nor saved widget state
provides the final calibration model, indexed pole or independently measured
reconstruction settings. The numerical comparison is complete; independent
physical accuracy cannot be established from these files alone. The legacy
notebook and input files were preserved, and no experimental fit or inferred
reconstruction was accepted into their state.

Source SHA-256:
`08cfadb18d0cd7203e2439a58c650fc0118c2f580a7c70caaee567e1da39c5d0`.
Range-file SHA-256:
`e9e41e915d07c2137df47c4c97cabd83b8a8960d7d3d972a1769466195e69c9d`.

### Input consistency and historical labels

The main and temporary HDF5 numerical datasets match throughout. The EPOS
export has the same event order and all corresponding fields agree after the
expected float32 and detector cm-to-mm conversions. Thus the temporary file
does not supply an earlier calibration baseline. All floating columns are
finite; raw time, mass and voltage are positive. The actual pulse-voltage array
is used: it is close to 20% of DC, but 3,235 events differ from that relationship.

The source's work log identifies the specimen as 316L despite the NiC1 folder
name. Its first-pass MATLAB ranges and the saved PyCCAPT ranges disagree about
several identities. The 36-row table has one N/Si interval overlap, duplicate
isotope labels, and 15 nominal masses outside their observed bounds. Its Fe56
window also includes a neighbouring Fe57 peak. These are **historical inputs,
not independently verified reference labels**; H/D molecular overlap likewise
prevents interpreting a D-window count directly as a deuterium concentration.

The exported row index is contiguous. The hardware `start_counter` resets
40,691 times and is not a global time coordinate; comparisons use preserved
event order. Neither fact establishes that the export contains every original
acquisition event. All-zero laser-energy values lack availability metadata and
are not evidence of measured zero laser energy.

### Initial comparison: calibration preservation, speed and optional fit

Using the notebook's 38 ns, 110 mm and voltage-mode settings reproduces saved
`mc_uc (Da)` **exactly for every event**. Guided acceptance of imported
`mc (Da)` preserves every calibrated mass and corresponding `t_c (ns)` value
through standalone-session resume, embedded-session resume and export.

The exact voltage-grouping optimization was compared with repeated interval
masks on the same 3,240,596 reference events, using 23 bins of 100 V. Every value
and its within-bin order matched. Median times over three repetitions were
0.1183 s versus 0.0460 s: **2.57× faster for this grouping operation**. This is
not an end-to-end calibration speedup claim. Both implementations include the
final interval endpoint, so the benchmark measures grouping rather than the
separate endpoint bug fix.

An additional diagnostic fitted the default physical multireference ToF model
using five explicitly assumed historical windows: rows 11, 15, 16, 17 and 26
(Cr52²⁺, Fe56²⁺, Ni58²⁺, Ni60²⁺ and Mo96²⁺). Targets come from the bundled
isotope catalogue, which stores rounded weights. A seeded 60/20/20 interleaved
split separated training, selection and confirmation; fitting used 23,635 of
1,944,094 eligible training events. The fit took 0.38 s, and the complete
diagnostic, including input loading and exact grouping checks, took 7.41 s.
The existing 0.5% median-bias and 2% relative-RMSE gates passed on selection and
confirmation without changing thresholds or adding drift terms.
All source events also fall within the fitted feature domain and have positive
elapsed time relative to the candidate offset. This is an applicability check,
not evidence that unlabelled masses are accurately calibrated.

Selection-event comparisons below use identical events for both calibrations.
Widths are central 90% window-population widths, **not** resolved-peak FWHM.

| Assumed reference | Median bias, saved → candidate (%) | Relative RMSE, saved → candidate (%) | Width, saved → candidate (Da) |
| --- | --- | --- | --- |
| Cr52²⁺, 25.970 Da | +0.311 → +0.047 | 0.589 → 0.425 | 0.3169 → 0.3196 |
| Fe56²⁺, 27.970 Da | +0.215 → −0.049 | 0.733 → 0.611 | 0.5226 → 0.5238 |
| Ni58²⁺, 28.970 Da | +0.127 → −0.134 | 0.529 → 0.460 | 0.4530 → 0.4520 |
| Ni60²⁺, 29.965 Da | +0.057 → −0.199 | 0.475 → 0.452 | 0.4665 → 0.4659 |
| Mo96²⁺, 47.955 Da | +0.068 → +0.008 | 0.166 → 0.154 | 0.2243 → 0.2334 |

RMSE improves against these assumed labels, but widths barely improve and two
absolute median biases worsen. The fitted offset, about 24.93 ns, is conditional
on those labels and the correction model; it is not a new measurement of the
instrument's time offset. The windows were selected using the old calibration,
and interleaved event holdouts share the same acquisition. Consequently passing
these gates does not establish a generally better calibration. The diagnostic
never calls acceptance or application, and all active/source arrays remain
unchanged. That initial validation did not change calibration mathematics; the
later weighting and validation corrections are described at the top of this report.

### Ranging and complete notebook persistence replay

The shipped notebook's cells and actual widget callbacks completed the
import → inspect → preview → explicit acceptance → peak review → save →
export → resume path on all 5,842,702 events.

| Measurement | Result |
| --- | --- |
| Explicit detection window | 0–65 Da; contains every historical interval |
| Events inside detection window | 4,388,645 |
| Events outside window, retained in source/session/export | 1,454,057 |
| Proposals | 302; zero automatic acceptances |
| Proposals with no candidate / multiple candidates | 194 / 31 |
| Historical identities available for explicit replay | 22 of 36 |
| Events assigned by replayed historical windows | 3,659,488 |
| Events assigned by all historical windows | 3,741,448 |
| Imported calibrated masses/ToF, event IDs, range assignments | Exact through export and both resume paths |
| Source and range-file hashes | Unchanged |

The final replay took 87.67 s: 0.54 s for proposals and 73.81 s for session save,
export and both resume checks. Input inspection/acceptance and scripted review
account for most of the remaining time.

Detection used 0.02 Da bins, prominence 20 counts, tolerance ±0.15 Da and
single-atom candidates. Every unmatched historical interval contains a detected
peak; its historical identity lies outside that candidate tolerance. Nearest
residuals are approximately 0.17–0.54 Da. This is a calibration/identity
consistency issue, not a failure to detect those peaks. The runner retains
unmatched rows and reasons instead of widening the tolerance automatically.

Replayed decisions use the historical bounds. Proposal-window overlap and counts
are reported separately; the 97.81% coverage of historically assigned events is
not an automatic-ranging accuracy score. The full-span first attempt correctly
refused 998 peaks at the existing 500-peak guard. The explicit mass-window
option in the validation runner makes this choice reproducible and does not
crop events or silently truncate peaks. Review time is scripted replay, not an
estimate of human decision time. Persistence dominates this dataset's run time.

### Reconstruction recipe and q = ICF/kf suggestions

With the notebook's 110 mm flight length, fitting the stored coordinates to the
existing Geiser equations recovers effective **ICF ≈ 1.4**, **kf × field ≈
168 V/nm**, and **density × efficiency ≈ 59.5 atoms/nm³**, using DC plus the
actual pulse-voltage array. Separate kf/field and density/efficiency values
cannot be identified from those products. Choosing field = 35 V/nm and
efficiency = 0.7 solely for replay gives kf ≈ 4.8, density ≈ 85 atoms/nm³ and
q ≈ 0.291667. These are an equivalent numerical recipe, not recommended or
independently measured settings.

The production reconstruction API processed every event in 0.52 s. Comparison
against 8,000 stored-coordinate samples gives maximum absolute x/y/z errors of
2.94×10⁻¹⁰, 3.13×10⁻¹⁰ and 2.81×10⁻⁹ nm. This verifies recipe parity, not the
physical accuracy of the stored reconstruction.

| Suggestion check | Result |
| --- | --- |
| New q method, acquisition completeness unconfirmed | Abstains before profiling. |
| Conditional full-file diagnostic, central 0.1 cm ROI, assumed FCC(002) spacing 0.1795 nm | 7,660 ROI events; no periodic evidence, so no accepted q or ICF/kf pair. Training-band coherences 0.018–0.067 remain below the 0.108–0.115 evidence floors. |
| Legacy peakiness objective, bounded nine-candidate production grid | Numerical maximum at kf ≈ 5.52, ICF ≈ 1.19, q ≈ 0.21558; measured spacing ≈ 0.07792 nm, inconsistent with the assumed 0.1795 nm. The current guard rejects the boundary maximum. |

The q profile took about 0.95 s and the production nine-candidate grid 4.99 s.
Their objectives, search ranges and budgets differ; these times do not establish
a general speedup or accuracy ranking. Complete acquisition coverage and a
correctly indexed pole were not established. The conditional coverage override
exists only in the diagnostic, and no pair was accepted. The practical outcome
on this dataset is an explicit lack of support, not a confident but unjustified
replacement pair.

### Reproduce the sample 2643 checks

From the repository root in PowerShell:

```powershell
$sample = '2643_Sep-02-2026_17-35_NiC1_HPT_Dc_1'
$processing = Join-Path (Join-Path 'D:/APT_data' $sample) 'data_processing'
$data = Join-Path $processing ($sample + '.h5')
$ranges = Join-Path (Join-Path (Join-Path $processing $sample) 'data_processing') ($sample + '_range.h5')
python tools/validate_guided_notebook_ui.py --data $data --ranges $ranges --mass-min 0 --mass-max 65 --output-directory analysis_outputs/2643_ui
python tools/validate_calibration_comparison.py --data $data --ranges $ranges --reference-rows 11 15 16 17 26 --flight-mm 110 --t0-ns 38 --pulse-mode voltage --output analysis_outputs/2643_calibration/summary.json
python tools/validate_saved_reconstruction.py --data $data --flight-mm 110 --output-directory analysis_outputs/2643_ratio
python tools/validate_saved_reconstruction.py --data $data --flight-mm 110 --diagnose-signal --output-directory analysis_outputs/2643_ratio/investigation
```

The JSON reports retain settings, per-reference errors, unmatched range reasons,
source fingerprints and timing. They are local validation outputs rather than
new scientific annotations on the input data.

### Changes made during the initial dataset comparison

No new production calibration/control error was reproduced in those initial checks.
The validation runner now accepts explicit detection bounds, retains original
widget-job errors, explains candidate-tolerance mismatches, and checks exact
calibrated ToF alongside mass and IDs. New diagnostics make the conditional
calibration and saved-reconstruction comparisons reproducible. An independent
review also found that pathname-only output checks missed hard links to inputs;
the runners now reject those aliases before writing, with regression coverage.
At that point the application equations, evidence gates and historical source
data were unchanged. The later fixes are documented in the follow-up above.

## Errors found during integration testing

Each fix has a regression test exercising the failing behavior.

| Area | Error and correction |
| --- | --- |
| Imported calibration | An all-zero or NaN `mc` column could hide usable `mc_uc`. Imported calibrated mass could also lose its corresponding calibrated ToF. Select a usable mass source, retain matching available calibrated ToF, and do not attach calibrated ToF to uncorrected mass. |
| Calibration input | The direct physical-fit API could use fields explicitly marked unavailable by an importer. Require recorded ToF, voltage and detector coordinates, plus recorded pulse voltage in voltage mode. Measured zeros remain distinct from unavailable fields. |
| Constant reference peak | Histogram peak estimation failed on a constant population because its histogram range had zero width. Return the exact constant location. |
| Preview/resume | A negative calibrated ToF preview could be accepted but could not be resumed. Enforce the same nonnegative, positive-evidence contract before acceptance. |
| Peak review | A changed catalogue could leave an unaccepted, previously chosen identity attached to a refreshed proposal. Clear that choice and its review evidence; locked decisions remain explicit. |
| Notebook state | Import/resume could retain nested panels for the previous source. Reset those panels. Reference masks now exclude invalid source rows. |
| Ratio projection | Checking only a small central ROI could certify an ICF family that fails on outer detector hits. Validate the full observed detector extent. |
| Ratio sampling | Regular-stride sampling could create an artificial periodic signal. Use seeded sampling without replacement within each band, then restore acquisition order. The cumulative depth prefix still includes every source event. |
| NeXus mapping | Compatibility zero placeholders could appear as measured values in mapped arrays. Mask unavailable/nonfinite measurements to NaN and provide adjacent status arrays while retaining exact original compatibility columns. |
| Software provenance | Installed package metadata can lag the running editable source. Record both versions and identify the running source version in NeXus program metadata. |
| Detector shutdown | Surface Concept callbacks queued at stop could be discarded without completeness disclosure. Preserve their copies as unattributed evidence and report incomplete acquisition; do not invent voltage metadata after safe-off. Crash recovery reports the same evidence. |

The sampling counterexample is especially significant: a uniformly spaced,
noncrystalline sequence had full-sequence coherence below 0.001, but regular
subsampling produced approximately 0.99984 coherence at a false ratio. The
new sampling method abstains on that same counterexample. Evidence thresholds
were not relaxed.

## Supplementary real-data notebook replay

The shipped notebook cells and real widget callbacks were exercised using
`R56_09163-v01_corrected.h5` and its explicitly supplied historical range file:

| Measurement | Result |
| --- | --- |
| Source events | 313,673 |
| Imported calibrated mass | Every value unchanged through acceptance, export and resume |
| Proposed peaks | 36; no identities automatically accepted |
| Historical range rows | 39 |
| Historical identities available for review replay | 30 |
| Events assigned by those replayed historical windows | 293,882 |
| Events assigned by all original historical windows | 294,362 |
| Session and embedded-session resume | Exact masses, integer event IDs and interval assignments retained |
| Source files | SHA-256 unchanged |

Detection used 0.02 Da bins, prominence 20 counts, tolerance 0.15 Da, and
single-atom candidates. Nine historical rows remained unmatched, including
molecular identities excluded by that candidate setting and weak/out-of-window
peaks. They were reported rather than assigned automatically. Matching rows
used the old reviewed bounds for the replay; proposal-bound comparisons are
reported separately. These historical labels are inputs, not independent
chemical ground truth.

The source lacks usable calibrated ToF, and no instrument/reference/reconstruction
settings were supplied for this check. Consequently it did not fit a new
calibration or accept a new reconstruction. Imported XYZ was omitted from the
accepted-output export rather than presented as newly accepted geometry.

On this machine, proposal generation took about 0.18 seconds and the entire
scripted replay about 13 seconds. This includes programmatic replay of known
decisions; it does not estimate the time for a person to review identities.

Source SHA-256:
`5e45bcbd5fcb73f253cd895d75c52538524d4c0aa31c7b2e10eff3d47aeb9eca`.
Range-file SHA-256:
`ec5250d66f353a29a2eab4bbc4288bec146790e006aa27cf8cbc299ed151695c`.

The reusable runner writes the detailed comparisons, timing, unmatched rows and
source fingerprints to `summary.json` in the chosen output directory:

```console
python tools/validate_guided_notebook_ui.py --data DATA.h5 --ranges REVIEWED_RANGES.h5 --output-directory analysis_outputs/workflow_validation
```

## Supplementary ratio comparison on a local Al dataset

Source: `Al_R56_09269-v01_corrected.h5`, 3,389,537 events.
File SHA-256:
`d1e7e2af4134551b12210898a2730234f56a3fa4ea7580458e6ce1176dc2e851`.

This is a conditional algorithm check. The local note supplied flight length
3.5 mm, field 19 V/nm, atomic density 60.2 atoms/nm³ and efficiency 0.4.
The check assumed a central ROI and Al(002) spacing 0.2025 nm. Complete original
acquisition coverage and correct pole indexing were **not independently
established**. These values must not be taken as validated metrology.

| Check | Outcome |
| --- | --- |
| Coverage not confirmed | Ratio method abstains. |
| Conditional coverage; ROI radius 0.005 cm | Geometry approximation rejected: phase span 0.1713 periods exceeds the 0.15 limit. |
| Conditional coverage; ROI radius 0.0045 cm | Geometry check passes; 831–938 ROI events per band. The periodic-evidence check fails, so no q or ICF/kf pair is accepted. |
| Evidence at the numerical profile maximum | Training coherence 0.05899, below the per-band required 0.1306–0.1388. The numerical maximum is not promoted to a suggestion. |
| Legacy peakiness objective, bounded nine-candidate grid | Maximum at ICF 1.7/kf 3.5, with measured SDM spacing 0.07833 nm rather than the assumed 0.2025 nm. The guarded grid also rejects this boundary maximum. |

The conditional ratio cache/profile took approximately 0.60 seconds and the
bounded grid check 4.65 seconds on this machine, excluding input loading.
The grid used at most 1,000 SDM atoms. These are different objectives and work
budgets, so this is **not** a general speedup benchmark or an accuracy ranking.
Neither method produced an accepted parameter pair in this check.

## Reproduction and interpretation

The shipped guided notebook is exercised through its code cells and widget
callbacks, including explicit acceptance, invalidation, undo, session save/resume
and accepted-output export. A synthetic end-to-end regression independently
generates labelled ToF data, fits/validates/confirms a physical model, reviews
identities and verifies exported event IDs, masses and reconstruction arrays.

These runs use in-process execution. Starting even a minimal independent Jupyter
kernel failed here in native ZeroMQ (`Bad file descriptor`, `epoll.cpp:73`), so
browser rendering and independent-kernel execution are not claimed as passed.

The control tests use fake SDKs, fault injection and simulator/replay inputs.
They do not establish performance or alignment accuracy on a physical instrument.
The new acceleration options remain opt-in; automatic Z approach requires its
independent response-calibration gate.

Run the regression suites with:

```console
python -m pytest -q --run-control --run-calibration
python -m ruff check pyccapt tests tools
python -m sphinx -b html -D nbsphinx_execute=never docs docs/_build/html
```

The implementation's combined regression run completed with **1,178 passed, 1 skipped**
in 388.56 seconds. Two input/output-collision tests added after that run's
collection also passed separately. The skipped optional two-million-ion speed
and memory regression was then explicitly enabled and passed (8-second CPU
and 350 MiB allocation limits). Thus all 1,181 collected/added checks have a
passing result across these runs. Ruff and whitespace checks passed. The full
HTML documentation build succeeded with 43 warnings; the final incremental
build covering the updated validation report completed without warnings.

After obtaining sample 2643, the calibration suite passed **631 tests** in
147.68 seconds. Six focused reconstruction-runner regressions also passed,
covering import behavior, reordered coordinate columns, input/output aliases
and Python 3.10-compatible source hashing.
The actual-data replays above additionally checked all 5,842,702 source events.
Control code was unchanged in this follow-up; its preceding combined-suite
result remains the relevant simulated validation.
Ruff and staged whitespace checks passed. The updated HTML documentation build
succeeded without warnings, with notebook execution disabled.

The NeXus export remains a labelled partial mapping, not complete NXapm
conformance. Checksummed storage proves stored integrity, not correctness of an
ion identity, calibration model, reconstruction or acquisition completeness.
