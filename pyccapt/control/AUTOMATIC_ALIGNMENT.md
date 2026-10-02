# Automatic alignment

Automatic alignment runs a separate, fully finalized experiment for each saved
sample. The main window exposes **Alignment start voltage** (1500 V by default)
and **Alignment voltage increment** (100 V by default). These settings are frozen
when the sequence starts. Sample finding compares repeatable evaporation jumps
between neighbouring positions at the same DC; it does not require 30% of the
requested detection rate. Completion still requires stable centring and 80% of
the requested rate (0.8% for a 1% experiment target).

Sample-stage coarse search is configured to **±50 µm per X/Y axis** around the
saved rough position. Fine alignment is limited to **±15 µm per X/Y axis** around
the position where stable coarse signal is found. Fine moves must also remain
inside the original ±50 µm envelope and the calibrated absolute stage bounds.
Reaching the fine range limit returns to coarse recovery. These are ranges,
not individual movement steps. Coarse XY uses 0.1 mm/s; fine XY uses 0.016 mm/s
and evaluates detector events after each 1 µm probe.
The first probes cover the saved position and its eight neighbours at ±10 µm.
If no candidate is confirmed, outer probe distances expand to 20, 40 and 50 µm,
with eight axial/diagonal points per distance: **33 positions maximum per pass**.
This is a sparse search, so a narrow peak between probes can be missed. Re-save
the rough position nearer that sample or reduce the first spacing if needed.
A nominal complete pass, origin return and voltage increment fit within
the 30-minute alignment timeout. Fine detector-feedback probes remain 1 µm.
Startup rejects settings whose estimated complete coarse sweep and first voltage
retry exceed the alignment timeout. The estimate includes dwell, settling,
configured travel speed and a polling allowance; hardware delays and time spent
in fine alignment and candidate rechecks can still exhaust the overall deadline.

## Commissioning before physical movement

### Current configured movement limits

These values describe the included `pyccapt/config.toml`. They are configured
software limits; physical clearance still requires instrument measurements.
For a saved position `(X0, Y0, Z0)` and the position `(Xf, Yf)` where fine
alignment begins:

| Movement | Permitted coordinates | Step or spacing | Speed |
| --- | --- | --- | --- |
| Absolute stage envelope | X [-2, 6], Y [-3, 7], Z [-10, 7] **mm** | Every target and measured position is checked | Phase-specific |
| Initial/sample transfer | Current XY at Z -4 mm, then saved XY at Z -4 mm, then saved Z | Three separate acknowledged moves | 300 µm/s = 0.3 mm/s |
| Coarse search | X0 ±50 µm, Y0 ±50 µm, Z = Z0 | Initial ±10 µm neighbourhood; double outer distance, clipped at 50 µm; 33 points | 100 µm/s = 0.1 mm/s |
| Fine search | Xf ±15 µm, Yf ±15 µm, also within the coarse envelope and absolute bounds | 1 µm single-axis detector-feedback probes | 16 µm/s = 0.016 mm/s |
| Optional mapped fine correction | Same fine/coarse/absolute limits | At most ±0.1 µm per X/Y axis per correction | 16 µm/s |
| Fine Z approach | **Enabled**, after stable XY centring and a valid circular footprint; total advance ≤20 µm from Z0 | +0.05 µm per step; never reset the allowance after a move | 0.1 µm/s |

`alignment_xy_jacobian_mm_per_um = []` selects the detector-feedback method.
Consequently, `alignment_fine_probe_step_um = 1.0` controls the current fine
probe size. `alignment_fine_xy_step_um = 0.1` limits the alternative mapped
correction; it is not the current feedback probe size. Increasing Z is configured
as approaching the electrode. The three transfer moves still retract and return Z.

Each movement requires stopped axes and the commanded axes within 0.02 µm
(20 nm) of target continuously for 0.5 s. The per-move timeout is 180 s.
The measured approach and XY search envelopes are checked before and throughout
motion, allowing only that 20 nm positioning tolerance beyond their nominal limits.

### Current detection and voltage decisions

| Config key | Default | Meaning |
| --- | --- | --- |
| `alignment_xy_step_um` | 10 | First neighbourhood radius; outer distances double |
| `alignment_coarse_dwell_s` / `alignment_coarse_max_dwell_s` | 2 / 6 | Minimum/maximum seconds for each neighbour or candidate observation |
| `alignment_search_min_events` | 200 | Minimum fresh paired hits and independent new events for rate/density comparison |
| `alignment_jump_ratio` / `alignment_jump_sigma` | 1.5 / 3 | Relative rate contrast and combined count-noise significance |
| `alignment_fine_loss_ratio` | 0.25 | Signal-loss fraction of the verified fine-entry rate |
| `alignment_voltage_increment` | 100 | Default stationary DC retry in volts; GUI value takes precedence |
| `alignment_approach_enabled` / `alignment_z_max_advance_um` | true / 20 | Fine Z enable and total advance cap in µm from saved Z |

The former `alignment_entry_fraction` and `alignment_loss_fraction` are no longer
used. Relative-search evidence controls finding and signal loss; the completion
goal remains configured by `alignment_finish_fraction`.

Settings snapshots from a GUI started before the relative-search update can
still contain `entry_fraction` and `loss_fraction`. The experiment, stage service
and sample-transfer GUI discard these two obsolete fields when reading a
snapshot; missing relative-search settings use their defaults. Explicit voltages,
motion limits and approach enable flags are preserved, and unknown fields or
invalid values still fail validation. Fully close and reopen PyCCAPT after a code
update so the GUI, stage service and experiment processes all load the new code
and configuration defaults.

At each coarse position, collect fresh paired hits for 2 s, extending to at most
6 s if evidence is insufficient. A usable observation requires 200 recent hits
and at least 200 newly arriving events after its first snapshot. Compare the
median measured detection rate against the closest observed neighbour at the
same voltage. The default candidate must be at least **1.5 times** that neighbour
and exceed it by **3 combined Poisson standard errors**, with a coherent dense
detector region or valid circular footprint. A single saturated pixel, diffuse
background and detector-edge clipping cannot supply that evidence.

Revisit the low neighbour, then revisit the candidate with new observation
epochs at unchanged DC. Enter fine alignment only if the relative contrast and
coherent region persist. This rejects single bursts and uniform temporal rate
increases. Both the original comparison and repeat measurements are logged.

Let `R` be the requested experiment detection rate, in percent. Completion
requires the footprint centre within 0.8 mm of the detector centre (2% of the
configured 40 mm detector radius) and at least `0.80 × R`, for at least 3 s with
2000 additional events. For `R = 1%`, that is 0.80%, not 80% absolute rate.

The detector window contains up to 2000 recent paired XY events, trimmed to a
maximum age of 10 s so lower-rate streams can still provide fresh density windows.
After a move or held
voltage change, the observation epoch is reset and the publisher discards the
straddling batch and a 0.5 s acquisition guard period. Old windows cannot
authorize a new movement. Dense-region centroids can guide XY without a complete
circular boundary. Z still requires the full circular-fit checks: coverage,
residual, background signal and detector-edge clipping.

DC is held throughout movement and observations. A complete failed
coarse pass returns to the saved origin before increasing DC by 100 V at up to
100 V/s. Start voltage defaults to 1500 V; the GUI values override the defaults.
The effective voltage limit is the minimum of 6000 V, the run maximum and the
hardware maximum, additionally capped by laser alignment when enabled together.
The first-detector-event gate still blocks upward DC changes before an event
has been received; the ordinary 10 s no-event shutdown remains active.

During fine feedback, establish a stable baseline, probe one axis by 1 µm,
and compare independent detector windows. Retain a probe only if its reduction
in detector-centre distance is at least the larger of 0.1 mm and 3% of baseline
distance. Each fine XY comparison waits 3 s and 200 independent new events.
Otherwise return to the baseline and try another X/Y direction.
The footprint centre determines which signs are tried first, but measured
improvement determines acceptance. Exhausting all permitted directions causes
recovery to saved Z and XY, then another coarse search.

After XY is centred with a valid circular footprint for 3 s and 2000 additional
events, Z can advance while the rate is below the completion goal, the conservative
footprint area including radius uncertainty is below 90%, and the next step is
within the total 20 µm allowance from saved Z. Recheck centring after every step.
If only a density centroid is available, Z remains fixed and stationary DC can
increase instead. **The 90% area setting is an approach limit, not a mandatory
completion condition.**

In fine alignment, an invalid region or rate below 25% of the verified fine-entry
signal (rather than a fraction of the requested target) must persist
beyond the observation dwell and then for another 3 s before recovery. The
five-attempt limit counts entries into fine alignment, not coarse grid points
or voltage increments. The overall alignment deadline is 1800 s (30 min),
starting when the experiment's alignment engine is created after transfer.
Normal experiment time/ion/voltage stop criteria remain active throughout.

The configured bounds are X [-2, 6], Y [-3, 7] and Z [-10, 7] mm. Increasing
Z approaches the electrode; sample transfers retract to -4 mm before XY travel.
All three positioning moves (Z retraction, XY transfer and return to saved Z)
use `alignment_transfer_speed_um_s = 300`, or 0.3 mm/s, three times the previous
0.1 mm/s transfer speed. Coarse and fine alignment speeds are configured separately.
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
or position older than two seconds stops automatic alignment. Alignment motion
also requires a running experiment, an experiment heartbeat no older than three
seconds, the electrode in, a healthy detector and a closed physical interlock.
These checks apply before and throughout motion. Transfers require completed
output shutdown and a closed physical interlock, without requiring a running
experiment. Manual jog, Home
and Reference are blocked throughout a sequence; Stage Stop cancels the sequence.

## Sequence

1. Validate all selected sample positions and all run parameter sets up front.
   [TOML plans](EXPERIMENT_PLANS.md) map each row's explicit `sample_id` to its
   saved Cameras position and run in queue order. IDs may repeat; saved samples
   omitted from the plan are not visited. TextBox parameters are copied for each
   selected sample.
   Voltage or Laser mode with an
   enabled DC supply and Surface Concept or RoentDek position-resolving TDC is
   required. For combined sample-stage and laser alignment, select Laser mode
   and enable **Align when experiment starts** in Laser Control. Validate laser
   calibration up front; both searches share the lower sample/laser DC ceiling.
   The sample stage aligns first, then the laser scan starts in the same run.
2. With outputs shut down, move through the transfer path to the saved position.
   Wait for every movement/settling acknowledgement before normal experiment
   startup. The main status bar shows each of the three moves and remaining
   distance: retract Z to -4 mm, traverse XY, then move Z to the saved sample
   position. A positioning failure never starts the experiment.
   Position tolerance and settling apply to the axes commanded in that step.
   For example, a Z-only retreat waits for Z within the configured 0.02 µm
   tolerance, rather than waiting for held X/Y readings to return to their
   initial values. All axes must be stopped and all measured XYZ positions
   must remain inside the calibrated bounds. Clearance checks still apply
   before lateral travel.
3. Ramp DC to the entered alignment start voltage, retaining the first-event
   gate. If the measured detection rate reaches the full experiment target
   during this initial ramp, immediately hold the current voltage and collect
   fresh detector windows there. A valid footprint and stable target rate enter
   fine alignment directly at that voltage. If the signal cannot be confirmed
   within the configured observation duration, resume the initial ramp.
   DC regulation is **held**, including the PID integrator, throughout XY
   moves and observation periods. The experiment's ordinary Kp values are not
   modified. Upward voltage changes are only allowed during stationary ramps.
4. Probe the local ±10 µm neighbourhood at saved Z, then expand outer distances
   within ±50 µm. Compare neighbouring fixed-DC observations, revisit both sides
   of any significant rate jump, and enter fine alignment around its confirmed
   high-density position. No requested-rate entry threshold is used.
5. If a full search fails, return to the saved origin and await acknowledgement,
   then increment voltage and repeat. The effective cap is the minimum of the
   alignment maximum (6000 V), experiment maximum and hardware maximum. The
   final increment is shortened to the cap (5900 -> 6000 for the defaults).
6. Fine alignment centres the footprint with bounded XY detector-feedback probes
   (or a measured XY mapping when supplied),
   then takes small Z steps only after stable XY centring and circular-fit
   validation, with a total maximum approach of 20 µm. Each step settles before using fresh
   events. Centre + stable 80% target rate freezes all alignment movement for
   the remainder of the experiment. The 90% detector-area approach limit includes
   a conservative radius uncertainty; reaching it prevents additional approach.
   Voltage can still rise while the stage is stationary, within the alignment cap.
   If all four permitted XY directions fail to improve centring, recover to the
   saved position and restart coarse search instead of repeating the same probes.
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
Surface Concept and RoentDek detection rates use their actual monotonic elapsed
measurement interval; a delayed update does not assume it still represents 0.5 s.

The estimator uses 2000 events by default, masks the detector area, clips hot
pixels, estimates background, smooths the density, extracts a half-contrast
boundary and fits a circle with outlier rejection. Boundary residuals, angular
coverage, estimated signal fraction and clipping determine validity. Invalid or
stale fits cannot authorize an approach. The green hitmap circle shows the latest
accepted circular footprint; an orange + marks a density centroid used for XY
only. The red detector circle uses the detector configuration.
This is an initial geometric model, requiring validation against this instrument's
actual evaporation patterns, crystallographic features and background distributions.

Each experiment stores:

- `meta_data/alignment_transfer.jsonl`: the initial retreat, XY transfer and saved-Z
  return, with UTC/monotonic times, requested axes, measured positions, position
  errors, settling transitions, completions, cancellation and faults.
- `meta_data/alignment.jsonl`: settings, sample/rough position, movement requests
  and completions, observations, voltage steps, retries and alignment outcome.
- HDF5 `provenance` attributes: `automatic_alignment`, `alignment_sample`,
  `alignment_sequence_id`, `alignment_selected_samples`,
  `alignment_saved_position_m`, `alignment_settings_json`,
  `alignment_final_status_json`, `alignment_outcome`, `alignment_transfer_json`,
  `alignment_transfer_journal_source`.
- `parameters.txt`: alignment settings, sample, rough position and outcome.
- Existing `apt/stage_*` datasets: sampled actual stage positions.

Runtime settings are snapshotted, including the two GUI voltage values. Control
config provenance remains available in the HDF5 file as before.

Before an experiment exists, the durable transfer journal is created under
`data/alignment_sequences/<sequence_id>/<queue_index>_sample_<sample_id>/`.
Successful transfers are copied into that experiment's metadata folder before
hardware initialization. Failed or cancelled initial transfers keep their journal
there even when no experiment is launched. Settling transitions are logged when
they change, plus position/error progress at least once a second during a move.
`alignment.jsonl` additionally records neighbour rates/counts/uncertainties,
candidate rechecks, density/circle model decisions and the verified fine-entry
reference rate. Laser sessions remain in `meta_data/laser_alignment.jsonl`.

## If positioning waits at -4 mm

The experiment starts only after the Z retreat, XY transfer and saved-Z return
have all completed. During each move, the main status bar reports the remaining
distance in µm, the configured tolerance, and whether the controller is moving,
waiting for the commanded position, or settling. Held-axis sensor drift does
not block completion of a Z-only or XY-only move.

If a commanded axis remains outside tolerance or the controller keeps reporting
motion, the move still times out and stops the sequence. The error includes
the commanded axes, the latest XYZ position errors in µm, and the tolerance.
The GUI reports controller faults directly and records them in
`pyccapt/files/logs/gui/gui_YYYY-MM-DD.log`, alongside each transfer request and
completion. Restart PyCCAPT after updating the motion code so the Stage GUI
uses the corrected completion check.

## Validation

Run `pytest -q --run-control tests/control`. Alignment tests use a virtual clock,
simulated motor acknowledgements, synthetic detector patterns and an offscreen
Qt GUI. No test opens a hardware connection or enables voltage outputs. These
tests verify sequence behaviour and failure handling; they do not establish
physical clearance or qualify the image model for unattended instrument use.
