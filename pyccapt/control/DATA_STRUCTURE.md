# HDF5 Data Structure for `pyccapt.control`

This document describes the expected structure of control-side HDF5 output.

Notation:

- `(n,)`: one-dimensional array with `n` samples
- units are listed in parentheses
- datatype is listed as NumPy/HDF5 type
- `N/A` means dimensionless or not directly unit-bearing

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
