# Cameca Raw Importers

This folder contains the Python equivalents of the RHIT/STR MATLAB import
workflow:

- `rhitLoad.m` -> `pyccapt.calibration.leap_tools.cameca_raw.rhit_tools.rhit_load`
- `rhitCalibrateFromEpos.m` -> `pyccapt.calibration.leap_tools.cameca_raw.rhit_tools.rhit_calibrate_from_epos`
- `rhitApplyCalibration.m` -> `pyccapt.calibration.leap_tools.cameca_raw.rhit_tools.rhit_apply_calibration`
- `rhitExtract.py` -> `pyccapt.calibration.leap_tools.cameca_raw.rhit_extract.py`
- `strLoad.m` -> `pyccapt.calibration.leap_tools.cameca_raw.str_tools.str_load`
- `strCalculatePositions.m` -> `pyccapt.calibration.leap_tools.cameca_raw.str_tools.str_calculate_positions`
- `strCalibrateFromRhit.m` -> `pyccapt.calibration.leap_tools.cameca_raw.str_tools.str_calibrate_from_rhit`

The notebook workflow lives in:

- `pyccapt/calibration/tutorials/jupyter_files/cameca_raw_import.ipynb`

The widget launcher used by that notebook lives in:

- `pyccapt/calibration/tutorials/tutorials_helpers/helper_cameca_raw_import.py`

## Raw-data interpretation guardrails

- RHIT `mc_direct_flight_approx` is a direct-flight calculation from raw TOF
  and VDC.  The compatibility `mc` column contains the same approximation.
  Neither is an IVAS-calibrated mass-to-charge value for a reflectron dataset.
  A verified decoder for Cameca's private `CCalibMass`/`CBowl` objects, or a
  measured reflectron transfer model, is required before using raw RHIT time
  for mass-tail or source-energy inference.
- STR payload words are unsigned 24-bit little-endian TDC/counter values.
  `str_load` preserves those words rather than converting valid timestamps
  above `2**23` into negative values.
- An STR `0x18` record can contain repeated delay-line tags.  The current
  rectangular table is a raw-record view, not a validated per-ion multi-hit
  pairing.  Do not use it for an experimental multi-hit distribution until a
  packet-level pairing rule has been checked against the acquisition's Cameca
  format specification or a matched accepted-hit stream.
