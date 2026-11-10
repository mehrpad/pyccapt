# RoentDek test data

`run198_first3000_groups.npz`: the first 3000 TDC groups of run 198
(`data_198_Jan-30-2024_18-34_Powersweep.lmf`, TDC8HP + hexanode) and, row by row,
the CoboldPC/DAn export of the same groups (`cobold`, columns: event, -, s, s fraction,
n_hits, n_anode_signals, then per hit x [mm], y [mm], TOF [ns], method).

- `time_stamp` (uint64, TDC bins), `cnt` (uint8, hits per channel), `tdc` (int32, [n, 8, 10] TDC bins)
- `bin_ns`: TDC bin size of the run

Used by `tests/control/test_roentdek_reader.py`, which needs the real Windows DLLs to compare the
Resort64c reconstruction with Cobold.
