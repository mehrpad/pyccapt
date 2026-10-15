# Guided workflow validation — 15 October 2026

## Scope and missing comparison data

The locally saved legacy notebook selects
`2643_Sep-02-2026_17-35_NiC1_HPT_Dc_1.h5` and its matching range file on a
`T:` drive that is unavailable in this environment. Its loaded-data output
reports 5,842,702 events, voltage mode, a 110 mm flight length and a 38 ns time
offset. Other saved cells contain outputs from sample 2590 and different row
counts. They cannot be combined into a reproducible calibration history.
The final accepted reconstruction settings and indexed pole are not recoverable
from the saved widget state.

**The requested same-dataset calibration/ranging/ICF–kf comparison remains
pending access to those files and the necessary reference/geometry settings.**
No result below is represented as that comparison. The legacy notebook was
preserved byte for byte and excluded from the implementation commit.

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

The final combined regression run completed with **1,178 passed, 1 skipped**
in 388.56 seconds. Two input/output-collision tests added after that run's
collection also passed separately. The skipped optional two-million-ion speed
and memory regression was then explicitly enabled and passed (8-second CPU
and 350 MiB allocation limits). Thus all 1,181 collected/added checks have a
passing result across these runs. Ruff and whitespace checks passed. The full
HTML documentation build succeeded with 43 warnings; the final incremental
build covering the updated validation report completed without warnings.

The NeXus export remains a labelled partial mapping, not complete NXapm
conformance. Checksummed storage proves stored integrity, not correctness of an
ion identity, calibration model, reconstruction or acquisition completeness.
