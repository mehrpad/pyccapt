# Guided workflow validation — 15 October 2026

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

### Calibration preservation, speed and optional fit

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
unchanged. No calibration mathematics changed during this validation follow-up.

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
```

The JSON reports retain settings, per-reference errors, unmatched range reasons,
source fingerprints and timing. They are local validation outputs rather than
new scientific annotations on the input data.

### Changes made after obtaining the dataset

No new production calibration/control error was reproduced in these checks.
The validation runner now accepts explicit detection bounds, retains original
widget-job errors, explains candidate-tolerance mismatches, and checks exact
calibrated ToF alongside mass and IDs. New diagnostics make the conditional
calibration and saved-reconstruction comparisons reproducible. An independent
review also found that pathname-only output checks missed hard links to inputs;
the runners now reject those aliases before writing, with regression coverage.
The application equations, evidence gates and historical source data are unchanged.

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
