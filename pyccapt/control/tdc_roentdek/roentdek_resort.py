"""
RoentDek Resort64c reconstruction (the library CoboldPC / DAn uses) for PyCCAPT.

The work is done in tdc_roentdek/roentdek_resort_x64.dll (source/resort_wrapper.cpp),
which links RoentDek's resort64c_VS2010_x64.dll. The sorter is configured from
tdc_roentdek/roentdek_resort.cfg, which is made from the Cobold DAn ccf file:

    python -m pyccapt.control.tdc_roentdek.roentdek_resort  DAn-part2_16_07_25.ccf  [correction_table.ccf]  out.cfg

Output (same keys as roentdek_reco.RoentDekReconstructor.reconstruct, one row per ion):
    event, x [mm], y [mm], t = TOF [ns] (NaN without laser start), method (Resort/Cobold
    method number), time_stamp, hit
(the acquisition converts x, y to cm for dld/x, dld/y)
"""

import ctypes
import os
import sys

import numpy as np
from numpy.ctypeslib import ndpointer

TDC_DIR = os.path.dirname(os.path.abspath(__file__))
DLL_PATH = os.path.join(TDC_DIR, 'roentdek_resort_x64.dll')
DEFAULT_CFG = os.path.join(TDC_DIR, 'roentdek_resort.cfg')

# Cobold DAn ccf parameter -> key in roentdek_resort.cfg
CCF_MAP = {
    1101: 'use_hex', 1102: 'fu', 1103: 'fv', 1104: 'fw', 1105: 'w_offset',
    1106: 'pos_offset_x', 1107: 'pos_offset_y',
    1108: 'tsum_offset_u', 1109: 'tsum_offset_v', 1110: 'tsum_offset_w',
    1111: 'mcp_radius', 1112: 'rotation_center_x', 1113: 'rotation_center_y', 1114: 'rotation_deg',
    1115: 'tsum_half_width_u', 1116: 'tsum_half_width_v', 1117: 'tsum_half_width_w',
    1118: 'runtime_u', 1119: 'runtime_v', 1120: 'runtime_w',
    1121: 'dead_time_anode', 1122: 'dead_time_mcp',
    1123: 'advanced_reconstruction', 1124: 'use_sum_correction', 1125: 'use_pos_correction',
    1129: 'ch_u1', 1130: 'ch_u2', 1131: 'ch_v1', 1132: 'ch_v2', 1133: 'ch_w1', 1134: 'ch_w2', 1135: 'ch_mcp',
    1136: 'time_zero_offset_det', 1137: 'use_mcp',
    1400: 'time_zero_offset_all', 1421: 'ch_start',
}


def read_ccf_parameters(ccf_file):
    par = {}
    with open(ccf_file, 'r', errors='replace') as f:
        for line in f:
            s = line.strip()
            if not s.lower().startswith('parameter'):
                continue
            body = s[len('parameter'):].split(';')[0].strip()
            if ',' not in body:
                continue
            k, v = body.split(',', 1)
            try:
                par[int(k.strip())] = float(v.strip())
            except ValueError:
                pass
    return par


def ccf_to_config(ccf_file, out_cfg, correction_table=None, start_selection=1, require_start=0):
    """write roentdek_resort.cfg from a Cobold DAn ccf (+ optional correction table)"""
    par = read_ccf_parameters(ccf_file)
    missing = [p for p in CCF_MAP if p not in par and p not in (1136, 1400)]
    lines = ['# made from %s' % os.path.abspath(ccf_file)]
    if missing:
        lines.append('# parameters not found in the ccf (defaults used): %s' % missing)
    t0 = par.get(1136, 0.0) + par.get(1400, 0.0)
    for p, key in CCF_MAP.items():
        if p in (1136, 1400):
            continue
        if p in par:
            lines.append('%s %s' % (key, repr(par[p])))
    lines.append('time_zero_offset %r' % t0)
    lines.append('start_selection %d   # 1 = last start before the ion (several laser pulses per group), 0 = first start (Cobold)' % start_selection)
    lines.append('require_start %d   # 0 = groups without laser start give ions with x, y and TOF = NaN; 1 = dropped (Cobold)' % require_start)
    if correction_table:
        rel = os.path.relpath(os.path.abspath(correction_table), os.path.dirname(os.path.abspath(out_cfg)))
        lines.append('correction_table %s' % rel)   # relative to this config file
    with open(out_cfg, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return out_cfg


def read_config(cfg):
    out = {}
    with open(cfg) as f:
        for line in f:
            line = line.split('#')[0].strip()
            if not line:
                continue
            k, v = line.split(None, 1)
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v.strip()
    return out


class ResortReconstructor:
    """same interface as roentdek_reco.RoentDekReconstructor"""

    MAX_ROWS_PER_EVENT = 4

    def __init__(self, config_file=None, bin_ns=0.0251173):
        self.config_file = os.path.abspath(config_file or DEFAULT_CFG)
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(self.config_file)
        self.cfg = read_config(self.config_file)
        self.lib = ctypes.CDLL(DLL_PATH)
        L = self.lib
        L.rs_create.restype = ctypes.c_void_p
        L.rs_create.argtypes = [ctypes.c_char_p]
        L.rs_last_error.restype = ctypes.c_char_p
        L.rs_delete.argtypes = [ctypes.c_void_p]
        L.rs_table_points.argtypes = [ctypes.c_void_p]
        L.rs_version.argtypes = [ndpointer(np.int32, flags='C_CONTIGUOUS')]
        L.rs_process.restype = ctypes.c_int
        L.rs_process.argtypes = [ctypes.c_void_p, ndpointer(np.uint32, flags='C_CONTIGUOUS'),
                                 ndpointer(np.int32, flags='C_CONTIGUOUS'), ctypes.c_int, ctypes.c_double,
                                 ndpointer(np.float64, flags='C_CONTIGUOUS'), ctypes.c_int, ctypes.c_longlong]
        self.ncol = L.rs_out_cols()
        self.handle = L.rs_create(self.config_file.encode())
        if not self.handle:
            raise RuntimeError('Resort64c initialization failed: %s' % L.rs_last_error().decode(errors='replace'))
        v = np.zeros(4, np.int32)
        L.rs_version(v)
        self.version = '%d.%d.%d.%d' % tuple(v)
        self.table_points = L.rs_table_points(self.handle)
        self.bin_ns = bin_ns
        self.rot = np.deg2rad(float(self.cfg.get('rotation_deg', 0.0)))
        self.rot_c = (float(self.cfg.get('rotation_center_x', 0.0)), float(self.cfg.get('rotation_center_y', 0.0)))
        ch = lambda k, d: int(self.cfg.get(k, d)) - 1
        self.ch_start = ch('ch_start', 7)
        self.ch_mcp = ch('ch_mcp', 8)
        self.ch_names = {ch('ch_u1', 1): 'u1', ch('ch_u2', 2): 'u2', ch('ch_v1', 3): 'v1', ch('ch_v2', 4): 'v2',
                         ch('ch_w1', 5): 'w1', ch('ch_w2', 6): 'w2', self.ch_mcp: 'MCP', self.ch_start: 'laser start'}
        self.sum_shift = {'u': 0.0, 'v': 0.0, 'w': 0.0}
        self.reset_stats()

    def describe(self):
        c = self.cfg
        return ('Resort64c %s, config %s: advanced=%d sum_corr=%d pos_corr=%d (table points %d), fu/fv/fw %.4f/%.4f/%.4f, '
                'w_offset %.3f, offset x/y %.3f/%.3f, time sums %.2f/%.2f/%.2f, gates %.2f/%.2f/%.2f, radius %.1f, start ch %d'
                % (self.version, os.path.basename(self.config_file), c.get('advanced_reconstruction', 0),
                   c.get('use_sum_correction', 0), c.get('use_pos_correction', 0), self.table_points,
                   c.get('fu', 0), c.get('fv', 0), c.get('fw', 0), c.get('w_offset', 0), c.get('pos_offset_x', 0),
                   c.get('pos_offset_y', 0), c.get('tsum_offset_u', 0), c.get('tsum_offset_v', 0), c.get('tsum_offset_w', 0),
                   c.get('tsum_half_width_u', 0), c.get('tsum_half_width_v', 0), c.get('tsum_half_width_w', 0),
                   c.get('mcp_radius', 0), self.ch_start + 1))

    def reset_stats(self):
        self.stats = {'groups': 0, 'no_mcp': 0, 'no_start': 0, 'ions': 0, 'groups_with_ion': 0, 'ions_without_start': 0}
        self.channel_groups = np.zeros(8, np.int64)

    def stats_text(self):
        s = self.stats
        return ('groups %d | ions %d in %d groups (Resort64c), of these without laser start (TOF = NaN) %d | '
                'groups without MCP %d, without start %d'
                % (s['groups'], s['ions'], s['groups_with_ion'], s['ions_without_start'], s['no_mcp'], s['no_start']))

    def signal_warnings(self, min_fraction=0.3):
        n = self.stats['groups']
        if n < 100:
            return []
        return ['TDC input %d (%s, Cobold channel %d) has a hit in only %.1f %% of the groups'
                % (ch, self.ch_names.get(ch, '?'), ch + 1, 100 * self.channel_groups[ch] / n)
                for ch in range(8) if self.channel_groups[ch] / n < min_fraction]

    def reconstruct(self, cnt, tdc, time_stamp=None):
        cnt = np.ascontiguousarray(np.asarray(cnt).reshape(-1, 8), dtype=np.uint32)
        tdc = np.ascontiguousarray(np.asarray(tdc).reshape(-1, 8, 10), dtype=np.int32)
        n = cnt.shape[0]
        if time_stamp is None:
            time_stamp = np.zeros(n)
        time_stamp = np.asarray(time_stamp, dtype=np.float64)
        st = self.stats
        st['groups'] += n
        self.channel_groups += (cnt > 0).sum(axis=0)
        st['no_mcp'] += int(np.sum(cnt[:, self.ch_mcp] == 0)) if 0 <= self.ch_mcp < 8 else 0
        st['no_start'] += int(np.sum(cnt[:, self.ch_start] == 0)) if 0 <= self.ch_start < 8 else 0
        max_rows = max(16, n * self.MAX_ROWS_PER_EVENT)
        while True:
            out = np.zeros((max_rows, self.ncol), np.float64)
            rows = self.lib.rs_process(self.handle, cnt, tdc, n, float(self.bin_ns), out, max_rows, 0)
            if rows == -1:
                max_rows *= 4
                continue
            if rows < 0:
                raise RuntimeError('Resort64c processing failed: %s' % self.lib.rs_last_error().decode(errors='replace'))
            break
        out = out[:rows]
        ev = out[:, 0].astype(np.int64)
        x, y = out[:, 2], out[:, 3]
        if self.rot != 0.0:
            cx, cy = self.rot_c
            c, s = np.cos(self.rot), np.sin(self.rot)
            x, y = cx + c * (x - cx) - s * (y - cy), cy + s * (x - cx) + c * (y - cy)
        st['ions'] += rows
        st['ions_without_start'] += int(np.sum(~np.isfinite(out[:, 4])))
        st['groups_with_ion'] += int(np.unique(ev).size)
        return {'event': ev, 'hit': out[:, 1].astype(np.int8), 'x': x, 'y': y, 't': out[:, 4],
                'method': out[:, 5].astype(np.int8), 'time_stamp': time_stamp[ev] if n else np.zeros(0)}

    def __del__(self):
        try:
            if getattr(self, 'handle', None):
                self.lib.rs_delete(self.handle)
                self.handle = None
        except Exception:
            pass


def make_reconstructor(bin_ns=None, verbose=True, config_file=None):
    """Resort64c if available (exactly as Cobold), else the PyCCAPT reconstruction"""
    try:
        r = ResortReconstructor(config_file)
        if bin_ns:
            r.bin_ns = bin_ns
        if verbose:
            print('RoentDek reconstruction: ' + r.describe(), flush=True)
        return r
    except Exception as e:
        from pyccapt.control.tdc_roentdek.roentdek_reco import RoentDekReconstructor
        print('WARNING: Resort64c reconstruction not available (%s). Using the PyCCAPT reconstruction '
              '(roentdek_calibration.json) instead.' % e, flush=True)
        r = RoentDekReconstructor()
        if bin_ns:
            r.bin_ns = bin_ns
        return r


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    if len(sys.argv) == 3:
        print(ccf_to_config(sys.argv[1], sys.argv[2]))
    else:
        print(ccf_to_config(sys.argv[1], sys.argv[3], correction_table=sys.argv[2]))
