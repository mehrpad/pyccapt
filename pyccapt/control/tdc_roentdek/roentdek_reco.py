"""
RoentDek hexanode (HEX80) reconstruction that follows CoboldPC / DAn (Resort).

Input  : raw TDC8HP group data as delivered by the wrapper DLL (and as written
         to the .lmf file): number of hits per channel cnt[N, 8] and the hit
         times tdc[N, 8, 10] in TDC bins (MCP is the group trigger).
Output : one row per reconstructed ion: x [mm], y [mm], TOF [ns], method, ...

What is reproduced (checked against the CoboldPC export of run 198):
  * TOF  = t_MCP - t_start (first start hit), in ns, same sign convention as
           Cobold (negative values are kept, like Cobold does).
  * x, y = 3-layer hex average  X = (2u + v + w)/3,  Y = (w - v)/sqrt(3)
           with u = fu/2*(u1-u2), v = fv/2*(v1-v2), w = fw/2*(w1-w2+w_offset),
           per-layer non-linearity correction tables and the x/y offsets of
           the ccf.  For events where all signals are consistent (Cobold
           method 0 and 7, ~78 % of all ions) this agrees with Cobold to
           < 0.03 mm (TDC binning / Cobold anti-moire randomisation level).
  * Events with a missing or inconsistent signal are reconstructed with the
           remaining layers (2-layer hex formulas) or by rebuilding the missing
           signal from the time sum.  Cobold's Resort library uses an internal
           decision logic for these cases that is not public, so here the
           result can differ by a few 0.1 mm for these ions (see report).
  * Second/third ions in one group that have no own MCP signal (Cobold n>=2,
           ~2.5 % of ions) are not reconstructed.

This is the FALLBACK used only when the Resort64c DLL (roentdek_resort.py) cannot be
loaded. Its tables are fitted to the run-198 (Jan 2024) Cobold calibration.
The raw data are always written unchanged to the .lmf file and to tdc/ in the HDF5 file,
so every event can be re-evaluated later, e.g. with CoboldPC.
"""

import json
import os

import numpy as np

_DEFAULT_CAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'roentdek_calibration.json')

SQRT3 = np.sqrt(3.0)

# method codes written to the HDF5 file (own codes, not Cobold's numbers)
METHOD_ALL_SIGNALS = 0      # 6 anode signals + MCP consistent (Cobold 0/7)
METHOD_TWO_LAYERS = 1       # one layer missing/inconsistent -> 2-layer formula
METHOD_REBUILT_SIGNAL = 2   # one anode signal rebuilt from the time sum
METHOD_REBUILT_TWO_LAYERS = 3   # 2 layers, one of them with a rebuilt signal


class RoentDekReconstructor:
    def __init__(self, calibration_file=None):
        if calibration_file is None:
            calibration_file = os.path.normpath(_DEFAULT_CAL)
        with open(calibration_file, 'r') as f:
            cal = json.load(f)
        self.cal = cal
        self.bin_ns = float(cal['tdc_bin_ns'])
        self.f = {'u': cal['fu'] * 0.5, 'v': cal['fv'] * 0.5, 'w': cal['fw'] * 0.5}
        self.off = {'u': 0.0, 'v': 0.0, 'w': float(cal['w_offset_ns'])}
        self.x0 = float(cal['x_offset_mm'])
        self.y0 = float(cal['y_offset_mm'])
        self.rot = np.deg2rad(float(cal.get('rotation_deg', 0.0)))
        self.hw = {k: float(v) for k, v in cal['tsum_half_width_ns'].items()}
        ch = cal['channels']
        self.ch = {'u': (ch['u1'], ch['u2']), 'v': (ch['v1'], ch['v2']), 'w': (ch['w1'], ch['w2'])}
        self.ch_start = ch['start']
        self.ch_mcp = ch['mcp']
        self.pos_nodes = np.asarray(cal['pos_table_nodes_ns'], float)
        self.pos_tab = {k: np.asarray(v, float) for k, v in cal['pos_table'].items()}
        self.sum_nodes = np.asarray(cal['sum_table_nodes_ns'], float)
        self.sum_tab = {k: np.asarray(v, float) for k, v in cal['sum_table_ns'].items()}
        # max. allowed hex inconsistency |u - (v + w)| in mm for a 3-layer hit
        self.max_inconsistency_mm = float(cal.get('max_layer_inconsistency_mm', 1.0))
        # hits outside the active MCP area are rejected (ccf parameter 1111, like Cobold)
        self.mcp_radius = float(cal.get('mcp_radius_mm', 39.0))
        # time sums drift (cables, CFD, voltages): follow the measured time-sum offsets
        # of complete single-hit groups automatically (Cobold: ccf parameters 1108-1110)
        self.auto_sum_offset = bool(cal.get('auto_time_sum_offset', True))
        self.sum_shift = {'u': 0.0, 'v': 0.0, 'w': 0.0}
        self._sum_dev = {'u': np.zeros(0), 'v': np.zeros(0), 'w': np.zeros(0)}
        self.sum_track_window_ns = 20.0
        # start signal: with several laser pulses in one group the start belonging to
        # the ion is the last one before the MCP signal
        self.start_selection = cal.get('start_selection', 'last_before_mcp')
        # False: groups without laser start still give an ion (x, y) with TOF = NaN
        self.require_start = bool(cal.get('require_start', False))
        self.reset_stats()

    def reset_stats(self):
        self.stats = {k: 0 for k in ('groups', 'no_mcp', 'no_start', 'start_after_mcp_only', 'anode_incomplete',
                                     'single_complete', 'rejected_time_sum', 'rejected_radius_or_consistency',
                                     'ions_fast', 'ions_slow', 'slow_failed')}
        self.channel_groups = np.zeros(8, np.int64)   # groups with >= 1 hit per TDC channel

    def signal_warnings(self, min_fraction=0.3):
        """hardware check: channels that (almost) never have a hit"""
        n = self.stats['groups']
        if n < 100:
            return []
        names = {self.ch['u'][0]: 'u1', self.ch['u'][1]: 'u2', self.ch['v'][0]: 'v1', self.ch['v'][1]: 'v2',
                 self.ch['w'][0]: 'w1', self.ch['w'][1]: 'w2', self.ch_start: 'laser start', self.ch_mcp: 'MCP'}
        out = []
        for ch in range(8):
            frac = self.channel_groups[ch] / n
            if frac < min_fraction:
                out.append('TDC input %d (%s, Cobold channel %d) has a hit in only %.1f %% of the groups'
                           % (ch, names.get(ch, '?'), ch + 1, 100 * frac))
        return out

    def stats_text(self):
        st = self.stats
        sh = ' '.join('%s%+.2f' % (k, self.sum_shift[k]) for k in 'uvw')
        return ('groups %d | ions %d (fast %d, other %d) | no MCP %d, no start %d, anode incomplete %d | '
                'complete single groups %d: rejected by time sum %d, by radius/consistency %d | '
                'not reconstructable %d | time-sum shift [ns] %s'
                % (st['groups'], st['ions_fast'] + st['ions_slow'], st['ions_fast'], st['ions_slow'],
                   st['no_mcp'], st['no_start'], st['anode_incomplete'], st['single_complete'],
                   st['rejected_time_sum'], st['rejected_radius_or_consistency'], st['slow_failed'], sh))

    # ------------------------------------------------------------------ helpers
    def _layer_pos(self, k, d_ns):
        """corrected layer coordinate in mm from the time difference in ns"""
        d = d_ns + np.interp(d_ns, self.pos_nodes, self.pos_tab[k])
        return self.f[k] * d

    def _tsum(self, k, d_ns):
        return np.interp(d_ns, self.sum_nodes, self.sum_tab[k]) + self.sum_shift[k]

    def _track_sum_offset(self, k, dev_raw):
        """dev_raw: time sum - table (without shift) of complete single-hit groups"""
        d = dev_raw[np.abs(dev_raw - self.sum_shift[k]) <= self.sum_track_window_ns]
        if d.size == 0:
            d = dev_raw[np.abs(dev_raw) <= 3 * self.sum_track_window_ns]
        buf = np.concatenate([self._sum_dev[k], d])[-5000:]
        self._sum_dev[k] = buf
        if buf.size >= 200:
            new = float(np.median(buf))
            if abs(new - self.sum_shift[k]) > 0.5:
                print('RoentDek reconstruction: time sum %s offset changed by %+.2f ns (now %+.2f ns vs calibration)'
                      % (k, new - self.sum_shift[k], new), flush=True)
            self.sum_shift[k] = new

    def _xy(self, u, v, w, layers):
        if layers == 'uvw':
            x = (2.0 * u + v + w) / 3.0
            y = (w - v) / SQRT3
        elif layers == 'uv':
            x = u
            y = (u - 2.0 * v) / SQRT3
        elif layers == 'uw':
            x = u
            y = (2.0 * w - u) / SQRT3
        else:  # 'vw'
            x = v + w
            y = (w - v) / SQRT3
        if self.rot != 0.0:
            c, s = np.cos(self.rot), np.sin(self.rot)
            x, y = c * x - s * y, s * x + c * y
        return x + self.x0, y + self.y0

    # ------------------------------------------------------------------ public
    def reconstruct(self, cnt, tdc, time_stamp=None):
        """
        cnt : (N, 8) hits per channel,  tdc : (N, 8, 10) hit times in TDC bins,
        time_stamp : (N,) TDC time stamp of the group in bins (optional)

        returns dict of 1-D arrays (one entry per reconstructed ion):
            event, x, y, t, method, time_stamp
        """
        cnt = np.asarray(cnt).reshape(-1, 8).astype(np.int64)
        tdc = np.asarray(tdc).reshape(-1, 8, 10).astype(np.float64) * self.bin_ns
        n_ev = cnt.shape[0]
        if time_stamp is None:
            time_stamp = np.zeros(n_ev)
        time_stamp = np.asarray(time_stamp, dtype=np.float64)

        out_ev = np.full(n_ev, -1, np.int64)
        out_x = np.zeros(n_ev)
        out_y = np.zeros(n_ev)
        out_t = np.zeros(n_ev)
        out_m = np.zeros(n_ev, np.int8)

        has_mcp = cnt[:, self.ch_mcp] > 0
        has_start = cnt[:, self.ch_start] > 0
        has_ref = (has_start & has_mcp) if self.require_start else has_mcp
        mcp = tdc[:, self.ch_mcp, 0]
        starts = tdc[:, self.ch_start, :]
        valid_start = np.arange(10)[None, :] < cnt[:, self.ch_start][:, None]
        start = starts[:, 0].copy()
        if self.start_selection == 'last_before_mcp':
            before = valid_start & (starts <= mcp[:, None])
            last = np.where(before, starts, -np.inf).max(axis=1)
            use = np.isfinite(last)
            start[use] = last[use]
            self.stats['start_after_mcp_only'] += int(np.sum(has_ref & ~use))
        tof = mcp - start
        tof[~has_start] = np.nan
        st = self.stats
        st['groups'] += n_ev
        self.channel_groups += (cnt > 0).sum(axis=0)
        st['no_mcp'] += int(np.sum(~has_mcp))
        st['no_start'] += int(np.sum(has_mcp & ~has_start))
        anode_ok = np.ones(n_ev, bool)
        for k in 'uvw':
            a, b = self.ch[k]
            anode_ok &= (cnt[:, a] > 0) & (cnt[:, b] > 0)
        st['anode_incomplete'] += int(np.sum(has_ref & ~anode_ok))

        # ---------------- fast path: exactly one hit on every anode channel
        single = has_ref.copy()
        for k in 'uvw':
            a, b = self.ch[k]
            single &= (cnt[:, a] == 1) & (cnt[:, b] == 1)
        idx = np.nonzero(single)[0]
        done = np.zeros(n_ev, bool)
        if idx.size:
            lay = {}
            ok = np.ones(idx.size, bool)
            st['single_complete'] += idx.size
            for k in 'uvw':
                a, b = self.ch[k]
                t1 = tdc[idx, a, 0]
                t2 = tdc[idx, b, 0]
                d = t1 - t2 + self.off[k]
                s = t1 + t2 - 2.0 * mcp[idx]
                if self.auto_sum_offset:
                    self._track_sum_offset(k, s - np.interp(d, self.sum_nodes, self.sum_tab[k]))
                ok &= np.abs(s - self._tsum(k, d)) <= self.hw[k]
                lay[k] = self._layer_pos(k, d)
            st['rejected_time_sum'] += int(np.sum(~ok))
            ok_sum = ok.copy()
            ok &= np.abs(lay['u'] - lay['v'] - lay['w']) <= self.max_inconsistency_mm
            xa, ya = self._xy(lay['u'], lay['v'], lay['w'], 'uvw')
            ok &= np.hypot(xa, ya) <= self.mcp_radius
            st['rejected_radius_or_consistency'] += int(np.sum(ok_sum & ~ok))
            st['ions_fast'] += int(np.sum(ok))
            sel = idx[ok]
            x, y = xa[ok], ya[ok]
            out_ev[sel] = sel
            out_x[sel] = x
            out_y[sel] = y
            out_t[sel] = tof[sel]
            out_m[sel] = METHOD_ALL_SIGNALS
            done[sel] = True

        # ---------------- slow path: everything else that has start + MCP
        for e in np.nonzero(has_ref & ~done)[0]:
            r = self._reconstruct_event(cnt[e], tdc[e], mcp[e])
            if r is not None:
                out_ev[e] = e
                out_x[e], out_y[e], out_m[e] = r
                out_t[e] = tof[e]
                st['ions_slow'] += 1
            else:
                st['slow_failed'] += 1

        keep = out_ev >= 0
        return {'event': out_ev[keep], 'x': out_x[keep], 'y': out_y[keep], 't': out_t[keep],
                'method': out_m[keep], 'time_stamp': time_stamp[keep]}

    # ------------------------------------------------------------------
    def _layer_candidates(self, k, c, t, mcp):
        """all measured pairs and single-signal rebuilds of one layer.
        returns list of (layer_mm, |dev|/halfwidth, rebuilt_flag)"""
        a, b = self.ch[k]
        h1 = t[a, :min(c[a], 10)]
        h2 = t[b, :min(c[b], 10)]
        cands = []
        for t1 in h1:
            for t2 in h2:
                d = t1 - t2 + self.off[k]
                dev = abs(t1 + t2 - 2.0 * mcp - self._tsum(k, d)) / self.hw[k]
                if dev <= 1.0:
                    cands.append((self._layer_pos(k, d), dev, False))
        if not cands:
            # rebuild one missing signal from the time sum (iterate the
            # position-dependent sum twice)
            for t1 in h1:
                s = self._tsum(k, 0.0)
                for _ in range(3):
                    t2 = s + 2.0 * mcp - t1
                    s = self._tsum(k, t1 - t2 + self.off[k])
                t2 = s + 2.0 * mcp - t1
                cands.append((self._layer_pos(k, t1 - t2 + self.off[k]), 0.5, True))
            for t2 in h2:
                s = self._tsum(k, 0.0)
                for _ in range(3):
                    t1 = s + 2.0 * mcp - t2
                    s = self._tsum(k, t1 - t2 + self.off[k])
                t1 = s + 2.0 * mcp - t2
                cands.append((self._layer_pos(k, t1 - t2 + self.off[k]), 0.5, True))
        return cands

    def _reconstruct_event(self, c, t, mcp):
        cand = {k: self._layer_candidates(k, c, t, mcp) for k in 'uvw'}
        # 1) three layers, consistent
        best = None
        for cu in cand['u']:
            for cv in cand['v']:
                for cw in cand['w']:
                    inc = abs(cu[0] - cv[0] - cw[0])
                    if inc > self.max_inconsistency_mm:
                        continue
                    n_rebuilt = cu[2] + cv[2] + cw[2]
                    if n_rebuilt > 1:
                        continue
                    score = n_rebuilt * 10.0 + inc + cu[1] + cv[1] + cw[1]
                    if best is None or score < best[0]:
                        x, y = self._xy(cu[0], cv[0], cw[0], 'uvw')
                        if x * x + y * y > self.mcp_radius ** 2:
                            continue
                        best = (score, x, y, n_rebuilt)
        if best is not None:
            return best[1], best[2], (METHOD_REBUILT_SIGNAL if best[3] else METHOD_ALL_SIGNALS)
        # 2) two layers (measured preferred)
        best = None
        for pair in ('uv', 'uw', 'vw'):
            k1, k2 = pair
            for c1 in cand[k1]:
                for c2 in cand[k2]:
                    n_rebuilt = c1[2] + c2[2]
                    if n_rebuilt > 1:
                        continue
                    score = n_rebuilt * 10.0 + c1[1] + c2[1]
                    if best is None or score < best[0]:
                        vals = {'u': 0.0, 'v': 0.0, 'w': 0.0}
                        vals[k1] = c1[0]
                        vals[k2] = c2[0]
                        x, y = self._xy(vals['u'], vals['v'], vals['w'], pair)
                        if x * x + y * y > self.mcp_radius ** 2:
                            continue
                        best = (score, x, y, n_rebuilt)
        if best is None:
            return None
        return best[1], best[2], (METHOD_REBUILT_TWO_LAYERS if best[3] else METHOD_TWO_LAYERS)
