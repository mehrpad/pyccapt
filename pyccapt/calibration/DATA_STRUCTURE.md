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

The `/entry` representation targets NXapm v2026.01 as a **partial mapping**.
It preserves the existing groups, maps units explicitly, and lists missing
instrument/specimen/coordinate metadata instead of asserting full conformance.
See [session and export documentation](../../docs/analysis_sessions.md) for the
schema, migration, validation, benchmark and opt-in SWMR APIs.

## Instrument settings and NeXus interoperability

The original notebook's `save_data(..., hdf=True)` now writes the partial NeXus
mapping alongside `/df` by default. `nexus=False` keeps a legacy-only export;
instrument metadata still survives. Existing numerical columns are unchanged.
The original Save data widget keeps working without notebook changes. For
widget discovery, `save_data` exposes scalar controls and an `**export_metadata`
slot; optional `analysis_session` and `nexus_metadata` remain accepted as their
documented positional or keyword arguments, or explicit `fixed(...)` widget
arguments. Constructing the manual panel does not perform an export.
The mapping duplicates event arrays for interoperability, so its storage/time
cost is additional to the `/df` export. This is not full NXapm conformance:
no `definition=NXapm` assertion is written, and the mapping report lists missing
context. The mapping follows the distinction between acquisition and processing
in the official [NXapm definition](https://manual.nexusformat.org/classes/applications/NXapm.html),
with project-specific settings in labelled
[NXcollection groups](https://manual.nexusformat.org/classes/base_classes/NXcollection.html).

`dataframe.attrs['pyccapt_instrument_metadata']` (`pyccapt-instrument-1`) contains
separate contexts with `value`, `units`, `status` and `source` per setting:

- `acquisition`: configured instrument constants and observed run modes;
- `calibration`: explicit accepted calibration inputs; a model-fit t0 is labelled
  `fitted_model`, not a measured hardware offset;
- `legacy_workflow_inputs`: values explicitly passed to the original loader or
  calibration panel, without claiming that they produced an imported calibration;
- `import_inputs`: optional settings supplied when accepting imported masses in
  the guided API; the historical calibration context is retained;
- `conversion_preview`: inputs actually used by the original t0 preview;
- `uncorrected_conversion`: the preview inputs when copied to `mc_uc`, or explicit
  legacy converter assumptions (110 mm, 0 ns, zero pulse-voltage voltage-mode
  conversion). These assumptions are not acquisition facts.

Settings are also exposed as typed scalar datasets below
`instrument_metadata` and `/entry/atom_probe1/instrument_metadata`, with mm/ns
units. Acquisition and calibration contexts remain distinct. Unknown numeric
values are NaN in HDF5 and null in JSON. A mixed/combined-mode acquisition does
not offer a single pulse mode/t0. Older config-only files can recover configured
constants but cannot infer the run's actual pulse mode from its GUI default.

`pyccapt.instrument_metadata.extract_instrument_settings(frame)` returns only
known, supported flight length/t0/mode suggestions. A calibration context takes
precedence as a whole; missing values are not filled from a different context.
The helper does not confirm settings or apply any calibration.

Raw eager/lazy loads and processed loads preserve these attributes. All finite
JSON-compatible `pyccapt_` dataframe attributes, including acquisition
completeness, crop scope and accepted analysis receipts, survive `/df` saves.
Missing raw pulse columns remain compatibility zeros but are marked unavailable;
recorded zeros remain available measurements.

The original notebook's **Data mode → PyCCAPT NeXus mapping** imports this
project's `.nxs`, `.nx5` or `.h5` exports. The file picker exposes these formats.
`load_data(path, 'nexus')` also reads standalone `/entry` exports, preserving
integer event IDs and metadata. It deliberately rejects unrelated third-party
NeXus layouts rather than assuming their fields match this partial mapping.

## Shared processing history

Both processing workflows maintain `CalibrationMetadata` in
`Variables.calibration_metadata` and `dataframe.attrs['pyccapt_calibration_metadata']`.
Schema `pyccapt-calibration-history-1` stores software version, original raw-column
fingerprint, ordered UTC action/model records, and an export checkpoint. The
checkpoint fingerprints selected raw measurements and each saved mass/ToF/XYZ
column, preserving numeric types and stable uint64 event IDs. Working corrections
are labelled separately from accepted outputs. History starts at the current
import; earlier missing steps and arbitrary notebook edits are not inferred.

HDF5 saves always include UTF-8 JSON at `/calibration_provenance/json`, even with
NeXus disabled. Its NXcollection group is a project extension, not a full NXapm
conformance assertion. The partial NeXus mapping hard-links that same group below
`/entry/pyccapt_data`. Exchange exports write `<filename>.calibration.json` sidecars.
Models are serialized as safe JSON state; arbitrary Python objects are not loaded
via pickle by this journal. Existing Pandas range-table storage is unchanged.

`restore_saved_calibration(result_path, original_source_frame)` verifies the full
original analysis input/order and selected event measurements, then restores saved
derived arrays without a refit. It rejects mismatches and does not apply a recipe
to another experiment. See [the workflow guide](../../docs/guided_workflow.md).

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

