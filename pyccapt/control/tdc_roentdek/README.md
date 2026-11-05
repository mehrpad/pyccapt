#  Python wrapper for Roentdec tdc

This folder contains dlls and python wrapper for Roentdec tdc.

## What the DLL sends

`wrapper_read_TDC8HP_x64.dll` returns one row per TDC group (event):

| Values | Content |
|---|---|
| 0..7 | first-hit bins of channels 0..7 (u1, u2, v1, v2, w1, w2, start, MCP); 0 = no hit |
| 8..11 | X [mm], Y [mm], TOF [ns] of the first-hit reconstruction, absolute timestamp [bins] |
| 12..19 | number of hits per channel (as counted by the driver, up to 10) |
| 20..51 | up to 4 hits per channel, channel-major (0 = unused slot) |

`values_per_event()` (52) and `hits_per_channel()` (4) are exported so `tdc_roentdek.py`
can tell this build from older builds, which send only values 0..11; both work.

## Position reconstruction

`tdc_roentdek.py` does not trust the DLL's X/Y/TOF. It rebuilds them with
`pyccapt/calibration/data_tools/roentdek_hexanode.py`, the same code used offline: the
multi-hit reconstruction (`reconstruct_hexanode_multihit`) uses every hit, so an ion whose
first hit on a channel is a reflection or noise pulse is still found. The live `dld` row of
an event is its first (MCP-referenced) particle. Every hit is also streamed to the HDF5
`tdc` group (one row per hit), so later particles can be reconstructed offline.

Detector constants default to `source/DAn-part2.ccf`; set `ro_hexanode_ccf` /
`ro_hexanode_correction` in `config.toml` to use another CCF or a correction map. The
correction map may carry the time-sum walk (`sum_walk`) used to gate the live data.

DLL builds made before 2026-11 always return X = Y = 0 (`(1 / 2)` is integer division in
C++), ignore missing hits and time sums, and their packet counter never resets, so after
the first full 30000-event packet they repeat events and write past the end of the buffer.
`source/wrapper_read_TDC8HP.cpp` is fixed and `wrapper_read_TDC8HP_x64.dll` is rebuilt from
it. The build is a Release build with the C runtime linked statically: it depends only on
`hptdc_driver_3.5.2_x64.dll` (installed with the RoentDek TDC driver). To rebuild, run
`source/build_dll.bat`. It needs the Visual Studio 2019/2022 C++ build tools with the v142
(14.29) x64 toolset; MFC is not needed.

## Re-processing existing files

```python
from pyccapt.calibration.data_tools import roentdek_hexanode as hx

frame = hx.reconstruct_roentdek_hdf5('data_198.h5')        # one row per event, first hits
particles = hx.reconstruct_roentdek_hdf5('data_198.h5', all_particles=True)  # every stored hit

# Runs recorded before every hit was stored: read all hits from CoboldPC's lmf2txt dump
hits = hx.read_lmf_txt('data_198_raw.txt')
result = hx.reconstruct_hexanode_multihit(hits)             # per-particle arrays
first = hx.first_particle(result, len(hits))                # per-event view

# Optional: learn CoboldPC's non-linearity correction from one run with a CoboldPC export
cobold = hx.read_cobold_txt('data_198.txt')
correction = hx.fit_cobold_correction(hx.reconstruct_hexanode(hits.first_hits()), cobold)
correction.sum_walk = hx.fit_sum_walk(hits.first_hits())
```

`source/hexanode_correction_run198.json` is that correction (with its time-sum walk), fitted
on run 198 (2024-01-30) with the `DAn-part2.ccf` constants. To use it live, set
`ro_hexanode_correction = "control/tdc_roentdek/source/hexanode_correction_run198.json"`.
It is valid only while the detector and its CCF constants are unchanged.

## Agreement with CoboldPC (runs 194–207)

| | First hits only | All hits (`*_raw.txt`) |
|---|---|---|
| CoboldPC hits found | 95.6–97.6% | 99.93–99.96% |
| Hits CoboldPC does not have | 387–1774 per run | 447–2033 per run |
| Position, median (with correction) | 0.28–0.29 mm (0.04 mm) | 0.28–0.29 mm (0.04 mm) |

Where the two disagree on a recovered hit, the raw signals support pyccapt's position
(4–6 of the 6 anode signals at the predicted times versus about 0–2 for CoboldPC's).
CoboldPC reports many more second particles (6014 vs ~265 on run 198), but more than half
of them have none of their own signals in the raw data: their signals fall within the anode
dead time after the first particle's, so they cannot be measured independently. Options of
`reconstruct_hexanode_multihit` (`min_free_layers=2`, `share_signals`,
`single_signal_window_ns`) accept more second particles; they are off by default because
the extra particles mostly disagree with CoboldPC.
