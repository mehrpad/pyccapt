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
