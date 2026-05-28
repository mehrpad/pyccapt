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
- `start_counter`: `(n,)` `(N/A, float64)` TDC counter value (wraps; not unique)
- `event_group_id` *(optional)*: `(n,)` `(N/A, int64)` shared id linking each
  dld row to the matching raw `/tdc` rows; present when the dataset was loaded
  with `load_tdc_raw=True` and preserved through every cropping step.

When partial-hit recovery (`data_tools.partial_recovery`) runs, the DLD
dataframe gains two columns: `dlts` `(N/A, int8)` — `4` for a native or
fully-recovered two-axis hit, `2` for a single-axis partial — and
`dlts_quality` `(N/A, string)` provenance label (`native` for original rows,
`recovered_xy` / `recovered_x` / `recovered_y` for recovered rows). Recovered
rows hold `NaN` on the unrecovered detector axis.

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

