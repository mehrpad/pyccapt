# Reproducible sessions, event identity and interoperable exports

The guided notebook saves an accepted analysis as a versioned HDF5 snapshot.
It keeps the existing `/df`, `/tdc` and `/range` exports available. A session
can also be saved by itself and reopened without replaying notebook cells.

## Session contract

`pyccapt.calibration.data_tools.analysis_session` provides:

```python
save_analysis_session(path, metadata, arrays)
metadata, arrays = load_analysis_session(path)
report = validate_analysis_session(path)
matches = list_analysis_sessions(directory, query="sample name")
```

The guided workflow supplies the metadata and arrays through its state adapter.
The snapshot includes source columns and their dtypes, immutable event identity,
accepted calibrated/reconstructed arrays, reference and selection masks, range
identities and decisions, model records, reconstruction assumptions, dependency
state and processing history. Source and derived content remain distinguishable.
Changing calibration invalidates dependent range acceptance; changing
reconstruction invalidates spatial analysis results. Undo restores an accepted
state rather than silently editing its historical record.

The storage schema is `pyccapt-session-2`:

```text
/analysis_session
  @schema_version, @status, @created_utc, @software_json
  @metadata_sha256, @manifest_sha256
  metadata_json
  manifest_json
  arrays/a000000, arrays/a000001, ...
```

Array names are mapped through the manifest, so names containing slashes cannot
create arbitrary HDF5 paths. Numeric and Boolean arrays retain exact dtypes.
SHA-256 covers each array's dtype, shape and ordered bytes. Loading checks
metadata, manifest and every array before handing state to the workflow.
Unknown schema versions, object arrays, external links and incomplete snapshots
are rejected. No pickle, joblib or dynamically imported executable model is
loaded. `max_bytes=` can bound allocations when opening a session.

Saves stage a unique sibling file, flush and synchronize it, and then rename it
atomically. A failed save leaves an existing session intact. The prototype
`pyccapt-session-1` envelope has an explicit in-memory migration that preserves
arrays and adds a manifest checksum on the next save. An older
`pyccapt-analysis-1` metadata receipt is **not** a complete session and cannot
be treated as resumable.

The software record includes installed distribution versions, the running
PyCCAPT source version (which can differ in an editable checkout), the Git revision when
available, and whether tracked working files differ from it. A dirty flag does
not archive the changed code: retain the source revision/patch with research
results when exact environment recreation matters.

`list_analysis_sessions` searches file names and metadata without loading bulk
arrays. Its “header checked” status is distinct from a full integrity check.
The validation report describes stored integrity; it does not certify the
scientific model or completeness of the original acquisition.

## Learned models are more than settings

`model_state.export_model_state` records fitted coefficients, tree structures,
thresholds, leaf values, ensemble weights/initial prediction and preprocessing.
`restore_model_state` returns a numeric inference object. Polynomial, linear,
standard-scaling/polynomial pipelines, fitted RANSAC, random forests, extra trees,
gradient boosting and the physical multi-reference ToF model are supported. Regression tests
compare predictions before and after a JSON round trip, including sklearn's
float32 tree-input convention.

These are inference snapshots, not training checkpoints. An unsupported model
raises an explicit error; storing its hyperparameters does not silently create
a purportedly reproducible learned model. The workflow can still preserve an
accepted output while recording that no reusable model was provided. Physical
multi-reference calibration stores its fitted offset, coefficients, normalization,
active-term mask, feature order and domain. Its restored model reproduces both
mass prediction and domain checks, including terms dropped during fitting.
Reference/selection masks remain separate session arrays; do not infer them from hashes.

## Event identity and missing data

At the normalized dataset boundary, `event_id` is an unsigned 64-bit integer.
It is assigned once if absent and preserved by filtering and reordering.
Identity is the pair `(source_id, event_id)`. It is distinct from the wrapping
hardware `start_counter` and from pulse-group identifiers. A cropped session
preserves the source ID and original event IDs; its current-content hash changes.
Arbitrary already-cropped input files cannot prove which earlier ions were lost.
In particular, a reconstruction depth prefix requires the full acquisition.

`CalibrationDataset.validity_status(column)` uses:

| Code | Meaning |
| --- | --- |
| 0 | Measured/finite; a measured zero remains valid |
| 1 | Unknown or nonfinite value in an available field |
| 2 | Field unavailable in the source |

Unknown coordinates remain NaN. Status fields do not invent positions, voltages
or measurements. Very large integer counters never pass through a homogeneous
floating-point matrix. Source attrs and event IDs are included in guided sessions.

POS, EPOS (including streamed HDF conversion) and APT imports record fields
absent from their source in `pyccapt_unavailable_columns`. Existing zero
placeholders retain their numerical values for compatibility, but their status
is unavailable. A finite zero supplied by the source remains available. APT
availability follows the actual selected source section, including fallback
section names. Source metadata and previously recorded unavailability are
preserved. This records what the importer can establish; it cannot determine
whether an upstream program fabricated a value already present in a file.
Older files without availability metadata cannot recover that distinction
from zero values alone. Use the PyCCAPT processed loader to restore the explicit
HDF metadata; a bare `pandas.read_hdf` call returns the table without its attrs.

## Processed HDF5 and NeXus

`save_data(..., analysis_session=(metadata, arrays), nexus_metadata={...})`
embeds a resumable session and a NeXus mapping alongside the existing dataframe.
Saving a subset does not relabel the original session's masks as subset masks.
The source/accepted arrays and exported event IDs establish the relationship.

`nexus_export.export_nexus(path, frame, metadata=..., ranges=...,
reconstruction=...)` creates a standalone mapping. Mapping version
`pyccapt-nxapm-map-1` targets the **v2026.01** conventions of
[NXapm](https://manual.nexusformat.org/classes/applications/NXapm.html).
It maps available detector hits, ToF, mass-to-charge, reconstruction coordinates,
parameter units, source identity and range records under `/entry`.
Detector cm are converted explicitly to mm; reconstruction nm and ToF ns retain
their units. Integer identities remain integers.
Mapped measurement arrays replace unavailable placeholders and nonfinite values
with NaN and carry adjacent status arrays. The original compatibility columns
retain their exact source values and status, so a source zero remains distinct
from a fabricated importer zero even outside the PyCCAPT reader.

This is an **NXapm-aligned partial mapping**, not full application-definition
conformance. It deliberately does not write `definition=NXapm`. A machine-readable
`mapping_report_json` lists missing context: complete instrument metadata,
coordinate/rotation systems, specimen characterization, prescribed voxel
discretization and NeXus nuclide encoding where relevant. Missing experimental
timestamps are not replaced with export timestamps. Full NXDL validation and
completion of that context are still needed for a strict NXapm submission.

## Acquisition completeness and recovery

Control HDF5 provenance retains expected manifest rows, written/excluded rows,
internal sequence gaps, rejected groups, validation checks and recovery actions.
Quarantined files and valid-but-unpaired HSD groups are distinguished. Waveforms
and their voltage records are retained or excluded together. Recovery cannot
prove that a crashed detector had no uncommitted tail, so that uncertainty remains
visible. Structural validity and acquisition completeness are separate outcomes.

## Storage tuning and optional live readers

`storage_benchmark.benchmark_storage(directory, values)` measures lossless layouts
on the intended destination: sequential reads, contiguous ROI reads and strided
previews, plus write time and file size. Results include chunk/compression choices
and cache policy. The function removes its temporary files and does not change
production defaults. HDF5's chunk-level access means a small ROI may still read
whole chunks; benchmark the actual access pattern and storage device.
[h5py dataset guidance](https://docs.h5py.org/en/stable/high/dataset.html)

`live_hdf5.LiveEventWriter` is an opt-in single-writer SWMR utility with a fixed
numeric schema. It flushes every column before publishing `committed_rows`.
`read_live_events` refreshes datasets and returns only that committed prefix,
ignoring a partially appended tail. Existing files cannot be accidentally
overwritten. There is no automatic activation on the instrument and no extra
writer sharing an acquisition file. The checksummed acquisition journal remains
the recovery authority. Test the deployed local/network filesystem before use;
SWMR has filesystem and schema constraints described in
[h5py SWMR guidance](https://docs.h5py.org/en/stable/swmr.html).
