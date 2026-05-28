# Calibration Module

The `pyccapt.calibration` package provides workflows for atom probe tomography data preparation, calibration, reconstruction, and visualization.

![Calibration visualization](../pyccapt/files/readme_images/visualization_gif.gif)

## Core Workflows

Typical calibration workflows include:

1. Import and crop datasets (HDF5, EPOS, POS, ATO, CSV, and raw-detector workflows).
2. Correct time-of-flight and estimate `t0`/flight-path parameters.
3. Convert time-of-flight to mass-to-charge (`m/c`).
4. Apply voltage and bowl corrections.
5. Perform 3D reconstruction.
6. Define and apply ranging windows, including saved `.h5`, `.rrng`, and `.rng` range files.
7. Generate 2D/3D visualizations and analysis plots.

## Package Structure

- `core`: validation, shared state, and primary calibration logic
- `data_tools`: loading, conversion, and preprocessing utilities
- `mc`: mass-to-charge and time-of-flight helper functions
- `reconstructions`: reconstruction and structural analysis tools
- `clustering`: clustering and isosurface workflows
- `leap_tools`: LEAP/POS/EPOS/APT/RRNG/RNG readers, Cameca raw importers, and helper tools
- `tutorials`: notebooks and notebook helper modules

## Mass-Calibration Pipeline

Mass calibration converts detector hits (time-of-flight, standing high
voltage, detector position) into a calibrated mass-to-charge (`m/c`)
spectrum. The pipeline in `pyccapt.calibration.core.calibration` runs the
following stages in order:

1. **Initial calibration** — a global `t0` and flight-path estimate
   converts time-of-flight to a first `m/c`.
2. **Voltage correction** — corrects each event's `m/c` for the slow
   change in standing voltage during evaporation, fit over ion-index or
   voltage segments.
3. **Bowl correction** — corrects the residual position-dependent
   flight-time difference across the detector (the "bowl"), sampled in
   Cartesian or polar detector cells.
4. **Time-drift correction** *(optional)* — a multiplicative
   per-ion-index correction that cancels high-voltage or temperature
   drift remaining after steps 2-3 in long acquisitions.
5. **Adaptive residual calibration** — a per-peak temporal and spatial
   residual fit that tightens mass resolution after the parametric
   corrections.

Peak locations used to drive the fits are read at histogram **bin
centers**, and histogram bins are anchored to the requested bin width.

### Configuration presets

The data-processing notebook exposes a **Config preset** dropdown. All
presets share the same voltage and bowl stages; they differ only in what
runs afterward:

- **Adaptive residual (default)** — voltage + bowl, then adaptive
  residual with the coarse-to-fine peak-scoring speedup.
- **+ time-drift correction (recommended)** — adds the per-ion-index
  time-drift correction (stage 4) between bowl correction and the
  residual fit. Suited to long runs with measurable drift.
- **Legacy adaptive residual** — voltage + bowl, then adaptive residual
  without the coarse-to-fine speedup. Slower, and reproduces the
  pre-2026 reference behaviour for bit-comparable results.

### NIST reference fit

The ion-list helper provides a second **Fit method** that rescales the
already-calibrated `m/c` onto reference (NIST) masses. It only rescales
`m/c`; it does not re-fit the voltage, bowl, or drift corrections, so it
composes with the pipeline above rather than replacing it. The rescale is
applied to the current calibrated spectrum and is rejected automatically
if it introduces non-finite values or collapses the mass spread.

## Partial-Hit Recovery (Surface Concept)

A delay-line detector reports an event only when all delay-line ends fire
within the hardware coincidence window. Pulses that fired some, but not
all, channels are normally discarded. The Surface Concept raw-data
workflow can recover physically valid hits from these partial pulses:

- For each delay-line axis (x: channels 0+1, y: channels 2+3), candidate
  pairs are enumerated and a one-to-one assignment selects a consistent
  subset. Axes are cross-matched by time-of-flight agreement: a matched
  pair becomes a full 2-axis (`xy`) hit; unmatched pairs become
  single-axis (1-DLTS) partial hits.
- The diagnostics path (`extract_surface_concept_hits`) enumerates every
  reconstructible pair and flags each one's `in_detector` status against
  the configured detector limit, so out-of-detector reconstructions are
  reported rather than silently dropped.
- The live merge path
  (`data_tools.partial_recovery.merge_partial_tdc_into_dld`) appends the
  recovered rows to the `/dld` dataframe, tagging each with a `dlts`
  (2 or 4) and `dlts_quality` label. Single-axis recovered rows carry
  `NaN` on the unrecovered detector axis. Pulses are grouped by
  `event_group_id` (wrap-safe), not by the raw `start_counter`, which
  wraps during long runs.
- **3-of-4 (time-sum) recovery** (on by default in the merge path):
  pulses that fired one complete delay-line axis plus a single end of the
  other axis are promoted to full `(x, y)` hits. The crossed delay lines
  share the same total propagation time (`t0 + t1 = t2 + t3`), so the
  missing end is `sum_complete_axis - t_present`; the recovered hit is
  gated by non-negative recovered times, the detector radius, and the ToF
  window, and labelled `dlts_quality = recovered_xy_3of4`. The Surface
  Concept firmware "quadrupel finder" requires all four stops, so these
  hits exist only after this offline step. Set
  `recover_three_channel=False` to disable it.

Partial-hit recovery is available from the `raw_data_analysis.ipynb`
notebook and through the auto raw-analysis helper.

## Shared State and Validation

Calibration workflows use shared mutable state through `Variables` in `pyccapt.calibration.core.share_variables`.
Validation and state-related errors should raise explicit calibration exceptions.

## Cross-Platform Paths

Use `pyccapt.calibration.path_utils` helpers for output and figure paths:

- `ensure_directory`
- `build_output_path`
- `save_figure`

## Data Structures

Calibration and range-file schema details are documented in [Calibration_DATA_STRUCTURE.md](Calibration_DATA_STRUCTURE.md).

## Tutorials

Interactive examples are available in:

- `pyccapt/calibration/tutorials/jupyter_files`
- `pyccapt/calibration/tutorials/colab`

The main Jupyter widget workflows currently include:

- `data_processing.ipynb`
- `visualization.ipynb`
- `L_and_t0_determination.ipynb`
- `raw_data_analysis.ipynb`
- `cameca_raw_import.ipynb`
- `reflectron_correction.ipynb`
- `tapsim_node_builder.ipynb`

A batch CLI companion to the reflectron notebook is also available:
`python -m pyccapt.calibration.reflectron_correction.batch_cli` scans a folder
for `.epos` files and writes `<stem>_corrected.h5` (and optionally `.epos`)
beside each input for a chosen instrument preset. See the
[Reflectron Batch Correction (CLI)](tutorials/reflectron_batch_cli.rst)
tutorial page for arguments and examples.

### Speeding up calibration with multiple workers

A handful of calibration hot paths (bowl correction polar sampling, voltage
correction segment loop, Surface Concept raw-data diagnostics) automatically
parallelize across CPU cores when the workload is large enough to amortize
executor startup. The default is auto; set `PYCCAPT_PARALLEL_WORKERS=1` to
force serial (useful for reproducible benchmarks and CI). See the
[Parallel Execution in Calibration](tutorials/parallel_calibration.rst)
tutorial page for the env vars, measured speedups, and which paths benefit.

### Running on small-RAM machines

The calibration pipeline ships a memory-mapped I/O layer
(`pyccapt.calibration.data_tools.lazy_io`) so multi-gigabyte EPOS / POS /
pyccapt-raw HDF5 files can be loaded, reflectron-corrected, and analyzed on
machines with as little as 8 GB of RAM. The reflectron batch CLI uses it
automatically (peak heap drops from multiple GB to ~270 MB), and the
raw-data widget exposes a **Low memory** checkbox that routes the stats
through chunked code paths. See the
[Working with Big Datasets on Small RAM](tutorials/low_memory_mode.rst)
tutorial page for the full design, opt-in entry points, and measured
before/after numbers.

Google Colab support is currently provided for:

- `data_processing.ipynb`
- `visualization.ipynb`

The tutorial save steps can export processed datasets as `HDF5`, `EPOS`, `POS`, and `ATO`.
The visualization helpers also include Min-Max and Maximum-Separation clustering,
iso-surface generation, and proxigram analysis for selected precipitate populations.

### Linking and saving raw `/tdc` data alongside `/dld`

PyCCAPT acquisition files contain both a `/dld` group with reconstructed events
and a `/tdc` group with the raw delay-line timestamps (DLTS) from which those
events were derived. From the calibration tutorials you can opt in to loading
both groups together via the **Load raw tdc** dropdown, and at save time choose
**Save raw tdc** to write the still-relevant raw rows into a `/tdc` key inside
the calibrated `.h5` output.

Internally this is handled by a shared `event_group_id` column added to both
dataframes when the file is loaded with `load_tdc_raw=True`. The column rides
through every cropping step in the calibration workflow (TOF clip, ROI, FDM,
sequence range, manual mask drops); at save time the linked tdc rows for
deleted dld rows are dropped, while orphan tdc rows (pulses that never produced
a reconstructible dld event) are always preserved. See
[Calibration_DATA_STRUCTURE.md](Calibration_DATA_STRUCTURE.md) for the on-disk
schema of the optional `/tdc` group and the linking semantics.

The user-facing tutorial pages are grouped in the Tutorials section of this documentation set.

## Workflow Snapshots

![Mass spectrum](../pyccapt/files/readme_images/hist.png)

<p align="center">
  <img width="36%" src="../pyccapt/files/readme_images/fdm.png" alt="FDM">
  <img width="32%" src="../pyccapt/files/readme_images/detector.gif" alt="Detector GIF">
</p>

<p align="center">
  <img width="30%" src="../pyccapt/files/readme_images/vol_corr.png" alt="Voltage correction">
  <img width="30%" src="../pyccapt/files/readme_images/bowl_corr.png" alt="Bowl correction">
</p>

<p align="center">
  <img width="30%" src="../pyccapt/files/readme_images/tof_V_corr.png" alt="TOF versus voltage">
  <img width="30%" src="../pyccapt/files/readme_images/tof_bowl_corr_y_det.png" alt="TOF bowl correction">
</p>

![Ranged mass spectrum](../pyccapt/files/readme_images/mc.png)

Interactive 3D example:
[Nimonic 90 reconstruction](https://rawcdn.githack.com/mmonajem/pyccapt/52835bc47735ef12bffcf7e18ce90b556b07d12f/pyccapt/files/readme_images/3d_o.html)

<p align="center">
  <img width="40%" src="../pyccapt/files/readme_images/roto.gif" alt="3D rotation">
  <img width="40%" src="../pyccapt/files/readme_images/iso.gif" alt="3D isosurface">
</p>
