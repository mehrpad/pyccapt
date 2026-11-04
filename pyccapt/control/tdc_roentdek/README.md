#  Python wrapper for Roentdec tdc

This folder contains dlls and python wrapper for Roentdec tdc.

## Position reconstruction

`wrapper_read_TDC8HP_x64.dll` returns 12 values per TDC group: the first-hit bins of
channels 0..7 (u1, u2, v1, v2, w1, w2, start, MCP), X [mm], Y [mm], TOF [ns] and the
absolute timestamp. `tdc_roentdek.py` does not trust the DLL's X/Y/TOF: it rebuilds them
from the raw channels with `pyccapt/calibration/data_tools/roentdek_hexanode.py`, the same
code used to re-process HDF5 files offline. Detector constants default to
`source/DAn-part2.ccf`; set `ro_hexanode_ccf` / `ro_hexanode_correction` in `config.toml`
to use another CCF or a correction map fitted against CoboldPC.

DLL builds made before 2026-11 always return X = Y = 0 (`(1 / 2)` is integer division in
C++), ignore missing hits and time sums, and their packet counter never resets, so after
the first full 30000-event packet they repeat events and write past the end of the buffer.
`source/wrapper_read_TDC8HP.cpp` is fixed. Rebuild the DLL with Visual Studio 2019+ (v142
toolset with MFC): open `source/simple_read_TDC8HP.sln` and build the `Debug|x64`
configuration, the only x64 configuration set to build a DLL. Copy the resulting
`simple_read_TDC8HP_x64.dll` over `wrapper_read_TDC8HP_x64.dll`.

## Re-processing existing files

```python
from pyccapt.calibration.data_tools import roentdek_hexanode as hx

frame = hx.reconstruct_roentdek_hdf5('data_198.h5')        # x_det (cm), y_det (cm), t (ns), ...

# Optional: learn CoboldPC's non-linearity correction from one run with a CoboldPC export
hits = hx.reconstruct_hexanode(hx.load_roentdek_tdc_channels('data_198.h5')['channels'])
cobold = hx.read_cobold_txt('data_198.txt')
correction = hx.fit_cobold_correction(hits, cobold)
print(hx.compare_with_cobold(hx.reconstruct_hexanode(
    hx.load_roentdek_tdc_channels('data_198.h5')['channels'], correction=correction), cobold))
```

`source/hexanode_correction_run198.json` is that correction, fitted on run 198
(2024-01-30) with the `DAn-part2.ccf` constants. To use it live, set
`ro_hexanode_correction = "control/tdc_roentdek/source/hexanode_correction_run198.json"`.
It is valid only while the detector and its CCF constants are unchanged.

On runs 194–207, the analytic positions agree with CoboldPC to a median of 0.29 mm. With a
correction fitted on run 198 the median is 0.04 mm on every run, with 98.4% of hits within
0.5 mm. 2–4% of CoboldPC's hits are missing because their first hits are reflections or
noise; CoboldPC recovers them from later hits, which are only in the LMF file.
