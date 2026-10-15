# PyCCAPT Calibration Data Structure

This document summarizes the data layout used by the calibration module and its
range files.

The canonical machine-readable schema for the raw acquisition groups
(`/dld`, `/tdc`, `/hsd`) — column names, order, dtypes, and the group
aliases accepted across pyccapt versions — is defined in
`pyccapt.calibration.data_tools.hdf5_schema`. The control-side writer
and the calibration-side reader both follow that module.

## Notation

- `(n,)`: one-dimensional array with length `n`
- Units and data type are written as `(unit, dtype)`
- `N/A` means the field has no physical unit

## Main Calibration Dataset (HDF5)

Typical calibrated dataset fields:

- `x (nm)`: `(n,)` `(nm, float64)` reconstructed x-coordinate
- `y (nm)`: `(n,)` `(nm, float64)` reconstructed y-coordinate
- `z (nm)`: `(n,)` `(nm, float64)` reconstructed z-coordinate
- `mc (Da)`: `(n,)` `(Da, float64)` calibrated mass-to-charge ratio
- `mc_uc (Da)`: `(n,)` `(Da, float64)` uncalibrated mass-to-charge ratio
- `high_voltage (V)`: `(n,)` `(V, float64)` detector high voltage
- `pulse`: `(n,)` `(V, float64)` or `(pJ, float64)` pulse voltage or laser energy
- `t (ns)`: `(n,)` `(ns, float64)` uncalibrated time-of-flight
- `t_c (ns)`: `(n,)` `(ns, float64)` calibrated time-of-flight
- `x_det (cm)`: `(n,)` `(cm, float64)` detector x hit position
- `y_det (cm)`: `(n,)` `(cm, float64)` detector y hit position
- `delta_p`: `(n,)` `(N/A, uint32)` pulses since previous detected event
- `multi`: `(n,)` `(N/A, uint32)` multiplicity per pulse
- `start_counter`: `(n,)` `(N/A, uint64 for native integer counters)` TDC counter
  value (wraps; not unique); legacy sources may provide a different numeric dtype
- `event_id`: `(n,)` `(N/A, uint64)` immutable event identity within a source;
  assigned once at the normalized dataset boundary and retained through crops.
  Identity is `(pyccapt_source_id, event_id)`, not the hardware pulse counter.
- `event_group_id` *(optional)*: `(n,)` `(N/A, int64)` shared id linking each
  dld row to the matching raw `/tdc` rows; present when the dataset was loaded
  with `load_tdc_raw=True` and preserved through every cropping step.

When partial-hit recovery (`data_tools.partial_recovery`) runs, the DLD
dataframe gains two columns: `dlts` `(N/A, int8)` — `4` for a native or
fully-recovered two-axis hit, `2` for a single-axis partial — and
`dlts_quality` `(N/A, string)` provenance label (`native` for original rows;
`recovered_xy` / `recovered_x` / `recovered_y` for recovered rows; and
`recovered_xy_3of4` for a full (x, y) hit rebuilt from a 3-channel pulse via
the delay-line time-sum constraint). Single-axis recovered rows hold `NaN` on
the unrecovered detector axis.

## Optional analysis metadata in processed HDF5

When a guided calibration has been accepted or reconstruction parameters have
been stored, the processed dataframe's `/df` storer includes the JSON attribute
`pyccapt_analysis_json` (`schema_version = "pyccapt-analysis-1"`). It records
the guided model receipt, its equations' parameters and units, validation
diagnostics, source/selection hashes, reconstruction parameters, and export row
count. This supplements the existing dataframe schema; it does not replace raw
acquisition metadata.

The receipt describes the original calibration session. An exported dataframe
may be a subset, so its row count and receipt scope are explicit. Hashes do not
store the original source or selection masks: retain those separately for
reproducibility. Guided outputs leave excluded derived rows as NaN. The session's
`guided_calibration_valid_mask` is event-aligned with its original source, not
implicitly with a cropped export.

## Resumable guided sessions and validity

The new `data_processing_guided.ipynb` writes a complete, checksummed
`/analysis_session` group (`pyccapt-session-2`), either as a standalone HDF5
session or alongside `/df`. JSON metadata names numeric arrays through a manifest;
arrays preserve their dtypes and carry SHA-256 digests. Source data, masks,
learned coefficients/tree state, accepted outputs, range-review decisions,
ICF/kf ratio evidence, dependency state and undo history survive resumption.
The old metadata receipt above remains readable but cannot by itself resume work.

Processed `/df` attributes include `pyccapt_source_json` when source identity or
missing-field provenance is available. `load_data` restores those attributes.
Legacy pulse-column zero fills retain their numeric behavior but are marked
unavailable, so they are not presented as measured zeros. Validity codes are
`0=measured/finite`, `1=unknown/nonfinite`, `2=unavailable`; missing coordinates
remain NaN. The NXapm mapping carries explicit status arrays and field availability.

An optional `/entry` representation targets NXapm v2026.01 as a **partial mapping**.
It preserves the existing groups, maps units explicitly, and lists missing
instrument/specimen/coordinate metadata instead of asserting full conformance.
See [session and export documentation](../../docs/analysis_sessions.md) for the
schema, migration, validation, benchmark and opt-in SWMR APIs.

## Range Dataset (HDF5 or Imported `.rrng` / `.rng`)

Range data defines identified ion windows in mass-to-charge space. The workflow
widgets can load either a saved PyCCAPT HDF5 range table (`.h5`), an IVAS
range file (`.rrng`), or a legacy LEAP/IVAS range file (`.rng`), which are all
normalized into the schema below.

- `name`: `(n,)` `(N/A, string)` ion label (plain text)
- `ion`: `(n,)` `(N/A, string)` ion label (LaTeX style)
- `mass`: `(n,)` `(Da, float64)` mass-to-charge from isotope composition
- `mc`: `(n,)` `(Da, float64)` detected peak center
- `mc_low`: `(n,)` `(Da, float64)` lower mass-to-charge bound
- `mc_up`: `(n,)` `(Da, float64)` upper mass-to-charge bound
- `color`: `(n,)` `(N/A, string)` display color (hex code)
- `element`: `(n,)` `(N/A, list[str])` element symbols for each range
- `complex`: `(n,)` `(N/A, list[uint32])` stoichiometric multiplicities
- `isotope`: `(n,)` `(N/A, list[uint32])` isotope identifiers
- `charge`: `(n,)` `(N/A, uint32)` ion charge state

## Interoperability

Calibration data can be imported from and exported to:

- HDF5
- EPOS
- POS
- ATO
- CSV

See tutorial notebooks under `pyccapt/calibration/tutorials` for examples.

