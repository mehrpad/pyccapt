# Automatic alignment

Automatic alignment runs a separate, fully finalized experiment for each saved
sample. The main window exposes **Alignment start voltage** (1500 V by default)
and **Alignment voltage increment** (200 V by default). These settings are frozen
when the sequence starts. Detection-rate fractions refer to the requested rate:
for a 1% experiment target, entry is 0.3% and completion is 0.8%.

Sample-stage coarse search is configured to **±50 µm per X/Y axis** around the
saved rough position. Fine alignment is limited to **±15 µm per X/Y axis** around
the position where stable coarse signal is found. Fine moves must also remain
inside the original ±50 µm envelope and the calibrated absolute stage bounds.
Reaching the fine range limit returns to coarse recovery. These are ranges,
not individual movement steps. Coarse XY uses 0.1 mm/s; fine XY uses 0.016 mm/s
and evaluates detector events after each 1 µm probe.
The coarse-grid guard allows up to 20,000 positions (the 1 µm grid at ±50 µm has
10,201). The experiment/alignment timeouts still apply, so a full grid is not
guaranteed to finish before the configured search deadline.

## Commissioning before physical movement

The configured bounds are X [-2, 6], Y [-3, 7] and Z [-10, 7] mm. Increasing
Z approaches the electrode; sample transfers retract to -4 mm before XY travel.
Every saved working sample Z must be greater than -4 mm. The startup validator
checks that and the XY envelope. Detector events cannot establish electrode
clearance or verify an unobstructed transfer path.

Measure and enter these values in `pyccapt/config.toml`:

- `alignment_bounds_mm`: three `[minimum, maximum]` pairs for X, Y, Z in the
  stage's referenced coordinates. These are absolute travel bounds, not a
  collision detector. Validate the local motion envelopes and transfer path below.
- `alignment_xy_range_um`: the permitted +/- X and Y search ranges about each
  saved position. Each sample's local XY envelope must be clear throughout its
  permitted Z advance, including intermediate positions.
- `alignment_fine_xy_range_um`: the permitted +/- X/Y travel from each fine
  alignment starting position, currently 15 µm; it does not enlarge the coarse envelope.
- `alignment_xy_jacobian_mm_per_um`: optional measured 2 by 2 mapping. Empty
  selects detector feedback: fine alignment probes one axis at a time, waits for
  independent detector windows and retains only centring improvements.
- `alignment_z_direction`: +1 or -1, toward the electrode.
- `alignment_transfer_z_mm`: a calibrated clearance Z reachable by retracting
  from every saved position. Transfers retract Z, traverse XY, then move Z to
  the next saved position. Simultaneous lateral and Z movement is rejected.
- `alignment_z_max_advance_um`: the permitted approach from a saved Z. The
  common limit must be safe for **every** sample, including positioning error
  and stopping distance. Choose the most restrictive clearance of the samples.
- Stage speeds, increments, settling tolerance and motion timeout. The included
  speeds/steps are commissioning candidates, not measured safety limits.

Keep `alignment_approach_enabled = false` until the approach limits and the
relationship between Z and footprint size have been verified. With approach
disabled the controller can centre laterally and raise voltage while stationary,
but cannot approach the electrode. The enabled XY/transfer configuration is not
proof of physical clearance; verify the path on the actual instrument.
Re-save sample positions after changing the holder or stage coordinate reference.

### Resolving “Automatic alignment is not calibrated”

This message is expected if the motion flag is disabled. Edit the alignment
entries in `pyccapt/config.toml`, then restart PyCCAPT so it reloads the file.

| Config key | Value to enter |
| --- | --- |
| `alignment_bounds_mm` | `[[Xmin, Xmax], [Ymin, Ymax], [Zmin, Zmax]]`, measured permissible absolute stage coordinates in **mm** |
| `alignment_xy_range_um` | `[Rx, Ry]`, permissible **±µm** about each saved sample position |
| `alignment_xy_jacobian_mm_per_um` | `[]` for live detector feedback, or a measured `[[a, b], [c, d]]` detector displacement in mm per µm of stage displacement |
| `alignment_z_direction` | `1` if increasing stage Z moves toward the electrode; otherwise `-1` |
| `alignment_transfer_z_mm` | Absolute referenced Z in **mm** where XY travel between all samples has verified clearance |
| `alignment_motion_calibrated` | `true` only once the above measurements, paths, speeds and local envelopes are verified |

The letters in this table are placeholders, **not literal TOML values**. Read
absolute coordinates from Stage Control; camera saved positions also display mm.
For an optional fixed detector mapping, at fixed voltage record a footprint centre `(u0,v0)`,
make a small verified +X displacement `dx` µm and measure `(ux,vx)` in detector mm.
Return to the original position, then measure `(uy,vy)` after +Y displacement
`dy` µm. Enter `a=(ux-u0)/dx`, `c=(vx-v0)/dx`, `b=(uy-u0)/dy`, `d=(vy-v0)/dy`.
Repeat to check signs/reproducibility. Values cannot be inferred from the aperture
diameter, and an identity matrix is not a measured calibration.

Keep approach disabled and maximum advance zero while commissioning XY. Before
enabling approach, enter the measured safe maximum advance and verify Z step and
speed. Transfer moves still use Z even when fine approach is disabled.

## Live 3D alignment window

Selecting Automatic Alignment opens a separate PyQtGraph/OpenGL window. It stays
available through sample changes and experiment shutdown; deselecting Automatic
Alignment hides it and stops its timer. Its close button minimizes it while the
mode is selected. The window is read-only and cannot command any movement.
Beside the 3D plot, a detector hitmap shows up to 2000 recent paired hits from
the current alignment observation. It uses the same detector diameter, red
boundary and millimetre coordinates as the visualization hitmap. The green
circle shows a valid fitted sample footprint. Reset clears only this display;
it does not clear the detector stream or interrupt alignment. Hits from the
previous stage position are cleared when the alignment observation changes.

- Horizontal axes: measured stage X/Y **offset from the sample's saved position**,
  in µm. The readout also shows absolute measured XYZ in mm.
- Vertical axis: measured detection rate in **percent**, not physical stage Z and
  not percent of target. A target of 1% and a measurement of 0.3% plots at 0.3%.
- Coloured points show coarse/fine/ramp/moving/aligned phases; a white marker shows
  the latest reading. The thin trace connects display readings, not interpolated
  surface estimates or a reconstructed continuous motor trajectory.
- Drag to rotate and scroll to zoom. Sample selector recalls each sample's own
  history; new samples are followed automatically and new sequences clear history.
- A small snapshot is published and rendered at most 5 Hz. The GUI retains at
  most 10,000 readings per sample, skips stale snapshots and does not read or
  drain acquisition rings. Render slowdown may skip display readings without
  changing experiment acquisition. Movement-phase rates can include events from
  the preceding position because detection rate is measured over a time window.

`PyOpenGL` is included in the control GUI/full installation extras. For an existing
environment, use `python -m pip install PyOpenGL` with the Python that runs PyCCAPT.
Without it the window explains the missing dependency. GPU/OpenGL support is
required for rendering; hardware-free tests validate data and lifecycle separately.

The service also checks the MCS2 referenced/sensor flags, controller fault bits,
position bounds, velocity limits, motion completion and settling. A GUI heartbeat
or position older than two seconds stops automatic alignment. Manual jog, Home
and Reference are blocked throughout a sequence; Stage Stop cancels the sequence.

## Sequence

1. Validate all selected sample positions and all run parameter sets up front.
   TextBox parameters are copied for each selected sample; TextLine blocks map
   to selected samples in sample-number order. Voltage mode with an enabled DC
   supply and Surface Concept or RoentDek position-resolving TDC is required.
2. With outputs shut down, move through the transfer path to the saved position.
   Wait for every movement/settling acknowledgement before normal experiment
   startup. The main status bar shows each of the three moves and remaining
   distance: retract Z to -4 mm, traverse XY, then move Z to the saved sample
   position. A positioning failure never starts the experiment.
3. Ramp DC to the entered alignment start voltage, retaining the first-event
   gate. If the measured detection rate reaches the full experiment target
   during this initial ramp, immediately hold the current voltage and collect
   fresh detector windows there. A valid footprint and stable target rate enter
   fine alignment directly at that voltage. If the signal cannot be confirmed
   within the configured observation duration, resume the initial ramp.
   DC regulation is **held**, including the PID integrator, throughout XY
   moves and observation periods. The experiment's ordinary Kp values are not
   modified. Upward voltage changes are only allowed during stationary ramps.
4. Search a bounded square-spiral XY grid at the saved Z. Each position receives
   a fresh observation window. A coherent footprint and at least 30% of the
   target rate, stable for the configured duration and across independent event
   windows, enter fine alignment. Rates above 50% also qualify.
5. If a full search fails, return to the saved origin and await acknowledgement,
   then increment voltage and repeat. The effective cap is the minimum of the
   alignment maximum (6000 V), experiment maximum and hardware maximum. The
   final increment is shortened to the cap (5900 -> 6000 for the defaults).
6. Fine alignment centres the footprint with bounded XY detector-feedback probes
   (or a measured XY mapping when supplied),
   then takes small Z steps if enabled. Each step settles before using fresh
   events. Centre + stable 80% target rate freezes all alignment movement for
   the remainder of the experiment. The 90% detector-area approach limit includes
   a conservative radius uncertainty; reaching it prevents additional approach.
   Voltage can still rise while the stage is stationary, within the alignment cap.
7. Sustained loss of sample signal during fine alignment triggers recovery:
   retract to the saved Z first, return to saved XY, then restart the XY search.
   The five-attempt limit counts fine-alignment entries, not voltage search steps.
8. Stop criteria remain active during alignment. All ions are saved, including
   alignment ions. Alignment does not reset experiment time/ion counters halfway
   through a sample.
9. Finish the normal experiment shutdown, detector join, data save, finalization,
   buffer cleanup and worker exit. Wait for visualization/camera final snapshots
   before positioning the next sample. Then run normal new-experiment startup,
   including visualization reset and new counters/output files.

An exhausted voltage search skips that sample. Five failed fine attempts,
alignment timeout, clearance/movement faults, detector errors, incomplete
shutdown and Operator Stop end the entire sequence. The existing ten-second
no-event detector safeguard remains active. A coarse scan does not bypass it.

## Detector analysis and metadata

The acquisition process publishes a bounded, paired XY snapshot separately from
the visualization's single-consumer rings. The alignment epoch changes after
every movement/settling transition. The publisher discards the straddling batch
and a short acquisition guard period; it does not reuse the previous footprint.
Only complete windows with bounded age authorize movement. Backlog behaviour
and acquisition timing must be verified in recorded-data/live-observation tests
on the installed detector before enabling physical motion.

The estimator uses 2000 events by default, masks the detector area, clips hot
pixels, estimates background, smooths the density, extracts a half-contrast
boundary and fits a circle with outlier rejection. Boundary residuals, angular
coverage, estimated signal fraction and clipping determine validity. Invalid or
stale fits cannot authorize an approach. The green hitmap circle shows the latest
accepted footprint; the red detector circle now uses the detector configuration.
This is an initial geometric model, requiring validation against this instrument's
actual evaporation patterns, crystallographic features and background distributions.

Each experiment stores:

- `meta_data/alignment.jsonl`: settings, sample/rough position, movement requests
  and completions, observations, voltage steps, retries and alignment outcome.
- HDF5 `provenance` attributes: `automatic_alignment`, `alignment_sample`,
  `alignment_sequence_id`, `alignment_selected_samples`,
  `alignment_saved_position_m`, `alignment_settings_json`,
  `alignment_final_status_json`, `alignment_outcome`, `alignment_transfer_json`.
- `parameters.txt`: alignment settings, sample, rough position and outcome.
- Existing `apt/stage_*` datasets: sampled actual stage positions.

Runtime settings are snapshotted, including the two GUI voltage values. Control
config provenance remains available in the HDF5 file as before.

## Validation

Run `pytest -q --run-control tests/control`. Alignment tests use a virtual clock,
simulated motor acknowledgements, synthetic detector patterns and an offscreen
Qt GUI. No test opens a hardware connection or enables voltage outputs. These
tests verify sequence behaviour and failure handling; they do not establish
physical clearance or qualify the image model for unattended instrument use.
