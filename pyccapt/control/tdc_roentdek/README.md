# RoentDek TDC8HP (hexanode) for PyCCAPT

## Files

| file | what |
|---|---|
| `wrapper_read_TDC8HP_x64.dll` | acquisition DLL, version 3 (Release x64, static C runtime; needs `hptdc_driver_3.5.2_x64.dll` like before) |
| `roentdek_resort_x64.dll` | reconstruction with RoentDek's **Resort64c** (the library CoboldPC uses); needs `resort64c_VS2010_x64.dll` from CoboldPC (see below) and the VC++ 2010 runtime (MSVCR100/MSVCP100, installed with CoboldPC) |
| `roentdek_resort.cfg` | Resort64c configuration made from the Cobold DAn ccf `source/cobold_ccf/DAn-part2ForLFIM.ccf` (run 283, 2026-11-09) and `source/cobold_ccf/correction_table_i_det_part3.ccf` |
| `tdc_roentdek.py` | acquisition process (reads all hits, reconstructs, streams ions + raw hits to chunks) |
| `roentdek_resort.py` | Python interface to Resort64c, `ccf_to_config()` |
| `roentdek_reco.py`, `roentdek_calibration.json` | fallback reconstruction, used only if Resort64c cannot be loaded |
| `check_run.py` | check a finished experiment: `python -m pyccapt.control.tdc_roentdek.check_run <folder>` |
| `global.cfg`, `CoboldPC2011R5-2_Header-Template_1TDC8HP.lmf` | TDC configuration and LMF header template. Copies are in `source/default_files`; they are restored automatically (with a warning) if missing |
| `tools/lmf_dump.exe` | LMF → binary dump (used by `check_run.py`) |
| `source/` | C++ sources and `build_wrapper.bat` (Visual Studio 2022), Cobold ccf files, test DLL (`source/test`) |

Hardware test without GUI: `python -m pyccapt.control.devices_test.tdc_roentdek_hardware_test 20`

## Resort64c library (not in the repository)

Resort64c is RoentDek's library. It comes with the detector and CoboldPC under RoentDek's licence, so it is
not distributed with PyCCAPT. Copy it from the CoboldPC installation on the instrument PC (search the CoboldPC
folder for `resort64c_VS2010_x64.dll`; the header and lib are in the same RoentDek Resort64c package):

* to run: `resort64c_VS2010_x64.dll` → `pyccapt/control/tdc_roentdek/`
* to rebuild `roentdek_resort_x64.dll` (only needed after changing `source/resort_wrapper.cpp`):
  `resort64c.h` and `resort64c_VS2010_x64.lib` → `pyccapt/control/tdc_roentdek/source/resort64c/`

These paths are in `.gitignore`. Without `resort64c_VS2010_x64.dll` the acquisition still works and still writes
the complete `output.lmf`, but prints a warning and uses the PyCCAPT reconstruction (`roentdek_reco.py`).
The LMF can always be re-evaluated later in CoboldPC.

## Acquisition DLL (`source/wrapper_read_TDC8HP.cpp`)

* The TDC is set up as in the RoentDek example: `global.cfg` from the working directory, then `Init` and
  `Reconfigure`.
* **Every group is written to `output.lmf` exactly like the previous DLLs.** This was verified by replaying
  run 198 through the DLL: the new file is byte-identical to the original LMF except the 5 bytes holding
  the start/stop time. The acquisition copies `output.lmf` into the experiment folder.
* `get_data_tdc_raw()` returns **all hits** (up to 10 per channel) of every group plus the 64-bit time stamp.
  It returns after 30000 groups or after 100 ms, also when no data arrive, so a stop always closes the LMF.
* Exceptions of the RoentDek driver are caught. `get_diagnostics()`, `get_last_words()` and
  `get_last_error()` report what the driver delivers.
* Fixed bugs of the previous DLLs:
  * X = Y = 0, caused by the integer division `(1 / 2)`;
  * a buffer overflow after 30000 groups;
  * a timeout that never fired;
  * only the first hit of each channel was returned;
  * TOF was computed with `abs()`, so a missing or late start produced fake TOF values.
  * `LMF_IO` is now created with (8 channels, 10 hits) to match the data array.

## Reconstruction

* Resort64c is configured like DAn from the Cobold ccf parameters 1101–1137 and 1421 (`roentdek_resort.cfg`).
  * TOF = particle time − laser start (Cobold channel 7). If a group contains several laser pulses, the start
    used is the last one before the ion (`start_selection 1`).
  * Groups without a laser start still give ions with x/y and **TOF = NaN**. `require_start 1` drops them,
    as Cobold does.
* Validation against the CoboldPC export of run 198 (Jan 2024, its own ccf + a fitted correction table):
  * 99.96 % of Cobold's ions are found;
  * the method is the same for 99.1 %;
  * TOF agrees within 0.03 ns for 97.8 %;
  * x/y agree within 0.1 mm for 99.0 %.
  * `tests/control/test_roentdek_reader.py` checks this on 3000 groups.
* After a new calibration in Cobold, regenerate the config:
  `python -m pyccapt.control.tdc_roentdek.roentdek_resort DAn-part2_xxx.ccf [correction_table.ccf] roentdek_resort.cfg`

## TDC configuration

`global.cfg`:
* MCP (TDC input 7 = Cobold channel 8) is the group trigger.
* Group window −6000…+1000 ns.

These are the settings recorded in the instrument's LMF headers (runs 2024–2026). A window that starts only
1000 ns before the MCP would lose every ion with TOF > 1 µs.

## Rebuild

`source\build_wrapper.bat` (Visual Studio 2022 C++ workload, no MFC) builds:
* both DLLs;
* `tools/lmf_dump.exe`;
* the test DLL `source/test/wrapper_read_TDC8HP_x64_mock.dll`, which replays recorded groups through the
  unchanged decoding code.

## Offline reconstruction without Resort64c

`pyccapt/calibration/data_tools/roentdek_hexanode.py` is an independent NumPy reconstruction of the
CoboldPC/DAn hexanode position calculation (no RoentDek library needed), for re-processing and
cross-checks. It reads every HDF5 layout (`tdc/ch0..7`, 8 rows per group, and the one-row-per-hit
table with the 2²³ offset) and CoboldPC `lmf2txt` dumps (`*_raw.txt`), and uses every hit per channel:

```python
from pyccapt.calibration.data_tools import roentdek_hexanode as hx

hits, extras = hx.load_roentdek_hits('run.h5')          # or hx.read_lmf_txt('run_raw.txt')
first = hx.first_particle(hx.reconstruct_hexanode_multihit(hits), len(hits))
```

Against the CoboldPC exports of runs 194–207 it finds 99.93–99.96 % of CoboldPC's first particles, with a
median position difference of 0.28–0.29 mm, or 0.04 mm with `source/hexanode_correction_run198.json`
(a smooth correction map fitted on run 198 with the `source/DAn-part2.ccf` constants). It finds far fewer
second particles than CoboldPC: it only accepts a later particle that is consistent on all three layers.
