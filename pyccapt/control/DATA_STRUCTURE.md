# HDF5 Data Structure for `pyccapt.control`

This document describes the expected structure of control-side HDF5 output.

Laser alignment also writes `meta_data/laser_alignment.jsonl` with per-session
settings, stage targets, settled observations, DC retries and outcomes. HDF5
`provenance` attributes `laser_alignment_run_json` and
`laser_alignment_final_status_json` retain the last session configuration/origin
and final status; `laser_alignment_enabled` and `laser_alignment_tracking` record
the automatic-start and tracking selections. See [LASER_ALIGNMENT.md](LASER_ALIGNMENT.md).

Notation:

- `(n,)`: one-dimensional array with `n` samples
- units are listed in parentheses
- datatype is listed as NumPy/HDF5 type
- `N/A` means dimensionless or not directly unit-bearing

## Group `provenance`

Schema 2.0 files record `schema_version`, `pyccapt_version`, creation time, platform and Python versions, the normalized
control configuration plus its SHA-256, the chunk-manifest SHA-256, excluded-row count, and serialized calibration-model
provenance. Every numeric dataset also carries a `units` attribute (`1` for dimensionless quantities). These attributes
are checked by `pyccapt validate-hdf5` and let downstream tools distinguish schema evolution from data corruption.

## Group `apt`

Control-loop metadata recorded each iteration.

- `id` `(n,)` (`N/A`, `uint64`): control-loop iteration index
- `num_events` `(n,)` (`N/A`, `uint32`): number of detected ions in loop interval
- `num_raw_signals` `(n,)` (`N/A`, `uint32`): number of raw detector signals
- `temperature` `(n,)` (`K`, `float64`): sample temperature
- `experiment_chamber_vacuum` `(n,)` (`mbar`, `float64`): main-chamber vacuum
- `timestamps` `(n,)` (`UNIX s`, `float64`): acquisition timestamp
- `stage_x`, `stage_y`, `stage_z` `(n,)` (`m`, `float64`): specimen-stage position
  (SmarAct MCS2 `stage_smartact_main`), one sample per control-loop iteration
- `laser_x`, `laser_y`, `laser_z` `(n,)` (`m`, `float64`): laser-focusing-stage
  position (SmarAct MCS2 `stage_smartact_laser`), one sample per control-loop iteration

Laser telemetry recorded per control iteration uses additional `apt/laser_*`
datasets: `output_power_mw`, `ir_power_mw` (mW), `pulse_energy_nj` (nJ),
`base_frequency_hz`, `output_frequency_hz` (Hz), `divider`, `wavelength_index`,
`wavelength_nm` (nominal nm), `aom_percent` (%), `status_code`, and `valid`.
Each suffix is prefixed with `laser_`, e.g. `apt/laser_output_power_mw`.
Missing/stale telemetry is NaN, with `valid=0`. Values come from the laser's
internal monitor; they are not calibrated specimen-delivered energy.
`meta_data/laser_readings.jsonl` stores timestamped readbacks, source commands,
explicit returned units and the device frequency table during the run.

**Laser unit correction:** files with provenance `laser_readback_revision =
unit-aware-1` convert the GUI's nJ energy to pJ before writing `dld/laser_pulse`
and `tdc/laser_pulse`. Earlier code wrote nJ numbers under pJ labels and assumed
`e_mlp` was always mW; old recordings cannot be repaired reliably by applying
one scale factor without checking their original monitor responses/wavelength.

> Stage/laser positions are in **meters**. They are published by the Stage Control
> and Laser Control GUIs' poll timers and read by the experiment loop each
> iteration. If a SmarAct stage is not connected (or not referenced) during the
> run, its axes are written as the `0.0` default — see the GUI session log for the
> connection outcome.

## Group `dld`

Delay-line detector hit coordinates and synchronized high-voltage/pulse metadata.

- `x` `(n,)` (`cm`, `float64`): detector X hit position
- `y` `(n,)` (`cm`, `float64`): detector Y hit position
- `t` `(n,)` (`ns`, `float64`): time-of-flight
- `high_voltage` `(n,)` (`V`, `float64`): specimen DC voltage
- `voltage_pulse` `(n,)` (`V`, `float64`): pulse voltage
- `laser_pulse` `(n,)` (`pJ`, `float64`): laser pulse energy
- `start_counter` `(n,)` (`N/A`, `float64`): DLD/TDC start counter aligned to event stream

## Group `tdc`

Raw time-to-digital converter stream. Exact channel schema depends on the TDC backend.

### Surface Concept backend

- `start_counter` `(n,)` (`N/A`, `uint64`)
- `channel` `(n,)` (`N/A`, `uint32`)
- `time_data` `(n,)` (`N/A`, `uint64`)
- `high_voltage` `(n,)` (`V`, `float64`)
- `voltage_pulse` `(n,)` (`V`, `float64`)
- `laser_pulse` `(n,)` (`pJ`, `float64`)

### RoentDek backend

- `ch0..ch7` `(n,)` (`N/A`, `uint64`): per-channel raw counters
- `voltage_pulse` `(n,)` (`V`, `float64`)
- `laser_pulse` `(n,)` (`pJ`, `float64`)

## Group `hsd`

High-speed digitizer (DRS) waveforms and synchronized metadata.

- `ch0_time`, `ch1_time`, `ch2_time`, `ch3_time` `(n,)` (`ns`, `float64`)
- `ch0_wave`, `ch1_wave`, `ch2_wave`, `ch3_wave` `(n,)` (`V`, `float64`)
- `high_voltage` `(n,)` (`V`, `float64`)
- `voltage_pulse` `(n,)` (`V`, `float64`)
- `laser_pulse` `(n,)` (`pJ`, `float64`)

## Compatibility Notes

- Some historical files may use older dataset names.
- `control/control_data_tool.py` contains migration helpers for older structures.
