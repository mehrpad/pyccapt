# Calibration and Range File Data Structure

This document summarizes the data layout used by the calibration module and its range files.

The canonical machine-readable definition of the raw acquisition groups
(`/dld`, `/tdc`, `/hsd`) — column names, order, dtypes, and the group
aliases accepted across pyccapt versions — lives in
`pyccapt.calibration.data_tools.hdf5_schema`. The writer
(`pyccapt.control.core.hdf5_creator`) and the reader
(`pyccapt.calibration.data_tools.data_loadcrop`) are both expected to
agree with that module; this page is the human-readable companion.

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
- `start_counter`: `(n,)` `(N/A, float64)` TDC counter value
- `event_group_id` *(optional)*: `(n,)` `(N/A, int64)` shared event-group id
  linking each dld row to the matching raw `/tdc` rows. Present only when the
  dataset was loaded with `load_tdc_raw=True`. Survives all downstream cropping
  steps so the link can be used at save time.

### Recovered partial-hit rows (optional)

When partial-hit recovery (`data_tools.partial_recovery`) runs, the whole DLD
dataframe gains two columns. Original rows are tagged as native; rows appended
for pulses that fired only some delay-line channels are tagged as recovered.

- `dlts`: `(n,)` `(N/A, int8)` number of delay-line timestamps behind the hit —
  `4` for a native or fully-recovered two-axis hit, `2` for a single-axis
  partial.
- `dlts_quality`: `(n,)` `(N/A, string)` provenance label: `native` for the
  original DLD rows; `recovered_xy` / `recovered_x` / `recovered_y` for hits
  rebuilt from a full 4-channel pulse or a single delay-line axis; and
  `recovered_xy_3of4` for a full (x, y) hit reconstructed from a 3-channel
  pulse via the delay-line time-sum constraint (`dlts == 4`, both axes
  present, one delay-line end inferred).

Recovered partial rows store `NaN` on the detector axis that was not
reconstructed (`x_det (cm)` for a y-only hit, and vice versa); their `mc`/
`mc_uc` use a centred-axis estimate so they remain visible in the mass
spectrum. Downstream steps that need exact positions filter them via
`dlts == 4` or the `NaN` detector mask.

## Linked Raw TDC Group `/tdc` (Optional)

When `load_tdc_raw=True` is selected at load and `save_tdc=True` at save, the
output `.h5` file also contains a `/tdc` group with the raw delay-line
timestamps that are still relevant after dld filtering. The group has the
columns of a Surface Concept tdc dataframe plus two link fields:

- `channel`: `(m,)` `(N/A, uint32)` delay-line channel index (0-3 for two
  delay lines, 0-5 for three)
- `start_counter`: `(m,)` `(N/A, uint32)` pulse-trigger id (wraps; not unique)
- `high_voltage (V)`, `pulse_v (V)`, `pulse_l (pJ)`, `time_data`: same
  semantics as the raw acquisition dataset
- `event_group_id`: `(m,)` `(N/A, int64)` shared id used to link each tdc row
  to the dld row(s) for the same pulse trigger; `-1` for orphan rows
- `has_dld_match`: `(m,)` `(N/A, bool)` `True` iff the pulse trigger produced
  at least one dld row at load time. Orphan rows (`False`) are always preserved
  during save filtering, regardless of which dld rows the user removed.

### Linking and filtering rules

- The link is built once at load time by walking the consecutive `start_counter`
  runs in both groups in time order. This is robust to counter wraparound, since
  the algorithm never compares counter values across different runs.
- A tdc row is kept on save iff `has_dld_match == False` OR its
  `event_group_id` is still present in the calibrated dld dataframe.
- Multi-hit pulses (multiple dld rows for the same trigger) are treated at the
  group level: if any dld row in the group survives filtering, all tdc rows for
  that group are preserved.

## Range Dataset (HDF5)

Range data defines identified ion windows in mass-to-charge space.

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
