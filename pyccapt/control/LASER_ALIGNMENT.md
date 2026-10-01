# Automatic laser alignment

The Laser Control window contains an embedded OpenGL XY response plot and a
separate Z-focus curve. These display **measured, settled positions**, never an
interpolated claim about unvisited positions. X/Y/Z coordinates are laser-stage
travel in micrometres. The vertical metric is detection rate (%) or DC voltage
(V); voltage is context, not a second implemented optimizer. Plot history is
bounded to 5,000 measurements and updates at most five times per second.

## Operator controls

- **Align when experiment starts**: start the full sequence in a Laser-mode
  experiment. If specimen alignment is also selected, finish specimen alignment
  first in that same Laser-mode experiment, then hand voltage control to the
  laser search. Both searches use the lower specimen/laser alignment DC ceiling.
  Each new experiment/sample starts a separate laser alignment session.
- **Coarse Alignment**: start coarse XY, fine XY, focus Z, then final fine XY
  during an existing Laser-mode experiment.
- **Fine Alignment**: enter at fine XY, then focus Z and final fine XY.
- **Focus Z**: enter at focus Z, then final fine XY.
- **Track during experiment**: after alignment, periodically probe a small XY
  cross, verify improvements, and return to the centre when improvement is not
  significant. Three failed fine scans trigger a coarse recovery.
- **Stop Alignment**: cancel laser motion and resume normal voltage regulation;
  acquisition continues. The laser-stage STOP also cancels alignment. Main Stop,
  interlocks, detector failure, and experiment completion stop alignment too.
- Editable coarse/fine/focus **± range** and **step**, in stage µm.
- **DC increase after no signal**, default **50 V**: after an entire unsuccessful
  coarse scan, return to that scan's centre, ramp DC by this amount, and repeat.
  The effective ceiling is the minimum of `laser_alignment_max_voltage`, the
  experiment maximum, and the system maximum. The final increment is clamped.

No button opens the laser output or starts an experiment implicitly. Laser output
must already have a valid enabled readback. Automatic operation currently supports
Laser pulse mode with a Surface Concept or RoentDek TDC, not Voltage + Laser/DRS.

## Search behavior

1. Hold DC and laser settings during each scan; suspend PID evaluation to avoid
   accumulating integral corrections. Use a serpentine XY raster, then successively
   smaller fine scans. Scan axes are clipped to the intersection of absolute
   calibrated limits and the session's permitted travel. An optimum at a boundary
   receives an inward scan, including centre baselines, without enlarging either
   limit. Z moves and XY moves are separate.
2. Wait for acknowledged stage position and a configured settling interval.
3. Request a new detector epoch. Discard its first packet because it may contain
   pre-settle events, then collect an independent count/TOF window. Visualization
   ring buffers remain single-consumer and are not drained by alignment.
4. Score actual ions per actual readback pulse frequency and elapsed time. Require
   a minimum event count and a rate more than three Poisson standard errors above
   the configured background/minimum rate. Repeat the scan centre to check drift;
   a claimed improvement must exceed statistical and fractional thresholds.
5. Revisit and measure the selected candidate before accepting it. A flat or
   statistically inconclusive improvement retains the original centre when that
   centre already has valid signal. A displaced candidate's verification must
   preserve a statistically and fractionally significant improvement over the
   higher of the initial and repeated centre responses. Repeatability relative
   to the candidate alone cannot authorize a move with a worse verified response.
6. After three failed fine/focus attempts, return to coarse search. The total
   recovery count and search duration are bounded. Failure requests the normal
   experiment shutdown, with an error visible in the main GUI and laser status.
7. Following alignment, ordinary detection-rate voltage control resumes. Tracking
   uses short fixed-DC comparisons; it does not infer drift merely from a falling
   count rate. Tracking pauses above the alignment voltage ceiling, without imposing
   that ceiling on a successfully aligned experiment's subsequent normal voltage ramp.

**The existing 10-second no-event shutdown remains active.** A scan does not reset
it. Consequently a completely silent detector can terminate a long coarse scan
before the voltage retry. Retries can occur when detector events continue but no
qualifying evaporation signal is found. The requested laser retry is a narrowly
scoped exception to the initial first-event voltage gate: only the explicit ramp
after a completed unsuccessful scan can increase voltage without a first event.
All ordinary experiment ramps retain the first-event gate.

Manual laser jogging, Home, Reference, laser setting changes and scan parameter
edits are locked while alignment owns the stage. Laser output-off controls remain
available. A change in readback settings, stale stage/laser telemetry, lost
experiment/GUI heartbeat, or a motion fault ends the search. A configured excessive
detection rate while scanning also stops the experiment.

## Commissioning required

`laser_alignment_calibrated` is deliberately **false** by default. Set actual
referenced laser-stage travel limits before enabling it:

| Key | Meaning |
| --- | --- |
| `laser_alignment_bounds_mm` | `[xmin,xmax,ymin,ymax,zmin,zmax]`, absolute referenced laser-stage coordinates |
| `laser_alignment_travel_um` | Maximum allowed ± X/Y/Z travel around this alignment session's initial position |
| `laser_alignment_xy_speed_um_s`, `laser_alignment_z_speed_um_s` | Calibrated speed ceilings |
| `laser_alignment_background_rate_percent` | Measured dark-count detection rate in absolute percent |
| `laser_alignment_min_rate_percent` | Minimum useful evaporation detection rate in absolute percent |
| `laser_alignment_settle_s`, `laser_alignment_dwell_s` | Settle time and measurement duration |
| `laser_alignment_max_voltage`, `laser_alignment_ramp_v_s` | Alignment DC ceiling and ramp speed |
| `laser_alignment_max_rate_multiple` | Stop scanning above this multiple of the experiment target rate |
| `laser_alignment_tracking_step_um`, `laser_alignment_tracking_interval_s` | Tracking probe size and interval |

The electrode aperture diameter does not establish laser-stage calibration. The
numerical range/speed defaults are editable commissioning values, not a statement
that those moves are safe on the connected optics. Measure beam displacement and
focus response versus stage displacement, including the Z axis mapping.

### Optional focus quality diagnostic

Set `laser_alignment_quality_peak_ns = [low, high]` around one isolated TOF peak
and configure `laser_alignment_quality_min_events`. Focus candidates must retain
acceptable robust peak width (10th–90th percentile) and late-tail fraction relative
to the baseline; insufficient peak statistics cause a retry rather than inventing
a quality score. This is a relative fixed-DC diagnostic, **not calibrated mass
resolving power**. A fixed TOF ROI must be appropriate for the voltage interval used;
changing material or voltage may require a different ROI.

With an empty ROI, focus optimization uses ion response only and the GUI explicitly
shows that peak quality is unavailable. No mass-resolution guarantee is inferred.

## Data and implementation

`meta_data/laser_alignment.jsonl` records every session, settings, movement target,
scan, DC retry, observation (position, count, uncertainty, rate, voltage and optional
TOF quality), acceptance and termination. HDF5 provenance retains the final run
settings/origin/status and tracking selection; normal acquisition retains stage
positions and laser telemetry. Multiple manual alignment sessions share the JSONL
log with distinct session IDs.

- `apt/laser_alignment.py`: deterministic scan state machine; no device I/O.
- `apt/laser_alignment_runtime.py`: experiment integration and sole DC-owner interface.
- `apt/laser_alignment_data.py`: independent bounded detector windows.
- `devices/laser_alignment_stage.py`: acknowledged motion using the existing connection.
- `gui/laser_alignment_gui.py`, `gui/laser_alignment_plot.py`: controls and display.

The design follows the bounded scan/recenter/verification concepts described in
[US7683318B2](https://patents.google.com/patent/US7683318B2/en), with focus-quality
motivation from [Koelling et al., 2013](https://doi.org/10.1016/j.ultramic.2013.03.003).
It is not a reproduction of CAMECA's current proprietary controller.

Software tests use simulated stage/detector responses. Live instrument commissioning
has not been performed by these tests.
