"""RoentDek hexanode (HEX) position reconstruction from raw TDC8HP channels.

This is a vectorised NumPy port of the position calculation CoboldPC/DAn
performs for a hexanode detector, so ``det_x``/``det_y``/``tof`` can be
recomputed from the raw TDC values that pyccapt stores in the HDF5 ``tdc``
group, or from CoboldPC ``lmf2txt`` dumps. Live acquisition reconstructs with
RoentDek's Resort64c (:mod:`pyccapt.control.tdc_roentdek.roentdek_resort`);
this module is an independent offline reconstruction (no RoentDek library
needed) for re-processing and cross-checking. Only NumPy is imported at
module level.

Channel layout (0-based, as stored in ``tdc/ch0..ch7`` or the flat
``tdc/time_data`` table)::

    0 u1   1 u2   2 v1   3 v2   4 w1   5 w2   6 start (laser/pulse)   7 MCP

The TDC8HP runs in group mode with the MCP channel as group trigger, so every
time is relative to the MCP signal (the stored MCP value is 0) and a value of
exactly 0 on any other channel means "no hit".

Layer coordinates follow the RoentDek "Pro" CCF convention::

    u = 0.5 * fu * (u1 - u2)
    v = 0.5 * fv * (v1 - v2)
    w = 0.5 * fw * (w1 - w2 + w_offset)

    Xuv = u        Yuv = (u - 2v) / sqrt(3)
    Xuw = u        Yuw = (2w - u) / sqrt(3)
    Xvw = v + w    Yvw = (w - v) / sqrt(3)

A layer is used only when its time sum ``t1 + t2 - 2*t_mcp + sum_offset``
lies inside the CCF gate. A layer with one missing signal is rebuilt from its
time sum, but only when at least one other layer is complete. Three good
layers are averaged; otherwise the matching pair formula is used.

CoboldPC's advanced resort routine and HEX non-linearity correction are
proprietary. On complete single hits the analytic positions above agree with
CoboldPC to ~0.3 mm RMS. :func:`fit_cobold_correction` learns the remaining
smooth distortion from a CoboldPC ASCII export of the same detector, which
brings the agreement to ~0.03 mm RMS.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

CHANNEL_NAMES = ('u1', 'u2', 'v1', 'v2', 'w1', 'w2', 'start', 'mcp')
LAYER_BITS = {'u': 1, 'v': 2, 'w': 4}
_SQRT3 = np.sqrt(3.0)


@dataclass(frozen=True)
class HexanodeParameters:
    """Detector constants, defaulting to ``tdc_roentdek/source/DAn-part2.ccf``."""

    # Fitted from CoboldPC TOF vs raw start bins (the LMF header rounds it to 0.025117).
    tdc_bin_ns: float = 0.0251172
    fu: float = 0.890
    fv: float = 0.917
    fw: float = 0.887
    w_offset_ns: float = -1.175
    x_offset_mm: float = -1.5
    y_offset_mm: float = 1.85
    sum_offset_u_ns: float = -113.98
    sum_offset_v_ns: float = -113.03
    sum_offset_w_ns: float = -114.85
    sum_halfwidth_u_ns: float = 1.5
    sum_halfwidth_v_ns: float = 1.2
    sum_halfwidth_w_ns: float = 2.3
    runtime_u_ns: float = 88.0
    runtime_v_ns: float = 86.0
    runtime_w_ns: float = 88.0
    mcp_radius_mm: float = 39.0
    rotation_deg: float = 0.0
    rotation_center_mm: tuple[float, float] = (0.0, 0.0)
    tof_offset_ns: float = 0.0
    # 0-based TDC channel indices: u1, u2, v1, v2, w1, w2.
    anode_channels: tuple[int, int, int, int, int, int] = (0, 1, 2, 3, 4, 5)
    start_channel: int | None = 6
    mcp_channel: int | None = 7
    # In group mode the MCP is the group trigger and sits at t = 0. Older
    # converted files also stored a copy of the start channel in ch7, so the
    # stored MCP value is ignored unless this is switched off.
    mcp_is_group_trigger: bool = True
    # CoboldPC reports no particle when the start channel did not fire.
    require_start: bool = True

    @classmethod
    def from_ccf(cls, path: str | Path, *, first_parameter: int = 1100, **overrides) -> 'HexanodeParameters':
        """Read detector constants from a CoboldPC DAn ``.ccf`` command file.

        ``first_parameter`` is the parameter index of the detector block
        (1100 for the first detector). Parameters missing from the file keep
        their defaults; ``overrides`` win over both.
        """
        values = _parse_ccf_parameters(path)
        base = first_parameter
        defaults = cls()
        kwargs = {}

        def take(offset, name, convert=float):
            key = base + offset
            if key in values:
                kwargs[name] = convert(values[key])

        take(2, 'fu')
        take(3, 'fv')
        take(4, 'fw')
        take(5, 'w_offset_ns')
        take(6, 'x_offset_mm')
        take(7, 'y_offset_mm')
        take(8, 'sum_offset_u_ns')
        take(9, 'sum_offset_v_ns')
        take(10, 'sum_offset_w_ns')
        take(11, 'mcp_radius_mm')
        take(14, 'rotation_deg')
        take(15, 'sum_halfwidth_u_ns')
        take(16, 'sum_halfwidth_v_ns')
        take(17, 'sum_halfwidth_w_ns')
        take(18, 'runtime_u_ns')
        take(19, 'runtime_v_ns')
        take(20, 'runtime_w_ns')
        if base + 12 in values or base + 13 in values:
            kwargs['rotation_center_mm'] = (values.get(base + 12, 0.0), values.get(base + 13, 0.0))
        runtime_u = kwargs.get('runtime_u_ns', defaults.runtime_u_ns)
        for name in ('runtime_v_ns', 'runtime_w_ns'):
            if kwargs.get(name) == 0:  # CCF: 0 = same as u
                kwargs[name] = runtime_u
        anode_keys = [base + offset for offset in range(29, 35)]
        if all(key in values for key in anode_keys):
            kwargs['anode_channels'] = tuple(int(values[key]) - 1 for key in anode_keys)
        if base + 35 in values:
            mcp = int(values[base + 35])
            kwargs['mcp_channel'] = mcp - 1 if mcp > 0 else None
        tof_offset = values.get(base + 36, 0.0) + values.get(1400, 0.0)
        if tof_offset:
            kwargs['tof_offset_ns'] = tof_offset
        if 1421 in values:
            start = int(values[1421])
            kwargs['start_channel'] = start - 1 if start > 0 else None
        kwargs.update(overrides)
        return replace(defaults, **kwargs)

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_ccf_parameters(path: str | Path) -> dict[int, float]:
    pattern = re.compile(r'^\s*Parameter\s+(\d+)\s*,\s*([-+0-9.eE]+)', re.IGNORECASE)
    values = {}
    with open(path, 'r', encoding='utf-8', errors='replace') as handle:
        for line in handle:
            match = pattern.match(line)
            if match:
                values[int(match.group(1))] = float(match.group(2))
    return values


@dataclass
class HexanodeCorrection:
    """Smooth 2-D polynomial map from analytic to CoboldPC positions (mm).

    ``sum_walk`` optionally carries the time-sum walk of the same detector
    (:func:`fit_sum_walk`), so small data sets can gate on walk-corrected
    sums without fitting it themselves.
    """

    degree: int
    coef_x: np.ndarray
    coef_y: np.ndarray
    scale_mm: float = 40.0
    stats: dict = field(default_factory=dict)
    sum_walk: np.ndarray | None = None

    def _design(self, x, y):
        xs = np.asarray(x, dtype=np.float64) / self.scale_mm
        ys = np.asarray(y, dtype=np.float64) / self.scale_mm
        return np.stack([xs ** i * ys ** j for i in range(self.degree + 1)
                         for j in range(self.degree + 1 - i)], axis=-1)

    def apply(self, x_mm, y_mm):
        design = self._design(x_mm, y_mm)
        return design @ self.coef_x, design @ self.coef_y

    def to_dict(self) -> dict:
        data = {'degree': self.degree, 'coef_x': self.coef_x.tolist(), 'coef_y': self.coef_y.tolist(),
                'scale_mm': self.scale_mm, 'stats': self.stats}
        if self.sum_walk is not None:
            data['sum_walk'] = np.asarray(self.sum_walk).tolist()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'HexanodeCorrection':
        sum_walk = data.get('sum_walk')
        return cls(int(data['degree']), np.asarray(data['coef_x'], dtype=np.float64),
                   np.asarray(data['coef_y'], dtype=np.float64), float(data.get('scale_mm', 40.0)),
                   dict(data.get('stats', {})), None if sum_walk is None else np.asarray(sum_walk, dtype=np.float64))


def _finish_positions(x, y, params: HexanodeParameters, correction: HexanodeCorrection | None):
    """Detector offsets, rotation and optional correction map (NaN stays NaN)."""
    x = np.asarray(x, dtype=np.float64) + params.x_offset_mm
    y = np.asarray(y, dtype=np.float64) + params.y_offset_mm
    if params.rotation_deg:
        cx, cy = params.rotation_center_mm
        angle = np.deg2rad(params.rotation_deg)
        dx, dy = x - cx, y - cy
        x = cx + dx * np.cos(angle) - dy * np.sin(angle)
        y = cy + dx * np.sin(angle) + dy * np.cos(angle)
    valid = np.isfinite(x)
    if correction is not None and valid.any():
        x_corr, y_corr = correction.apply(x[valid], y[valid])
        x[valid], y[valid] = x_corr, y_corr
    return x, y


def reconstruct_hexanode(
    channels,
    params: HexanodeParameters | None = None,
    correction: HexanodeCorrection | None = None,
) -> dict[str, np.ndarray]:
    """Reconstruct hit positions and TOF from first-hit TDC bins.

    Args:
        channels: ``(n, 8)`` integer TDC bins (see module docstring for the
            channel layout). 0 means "no hit" except on the trigger channel.
        params: Detector constants; defaults to :class:`HexanodeParameters`.
        correction: Optional map from :func:`fit_cobold_correction`.

    Returns:
        Dict of ``(n,)`` arrays: ``x_mm``, ``y_mm`` (NaN when no position
        could be built), ``tof_ns`` (NaN without a start signal), ``valid``,
        ``layer_mask`` (u=1, v=2, w=4 of the layers used), ``n_reconstructed``
        (signals rebuilt from a time sum), ``sum_u_ns``/``sum_v_ns``/
        ``sum_w_ns`` (offset-corrected time sums) and ``hex_delta_mm``
        (``u - (v + w)``, the hexanode redundancy residual).
    """
    params = params or HexanodeParameters()
    raw = np.asarray(channels)
    if raw.ndim != 2 or raw.shape[1] < 6:
        raise ValueError(f'expected an (n, 8) channel array, got shape {raw.shape}')
    n = raw.shape[0]
    times = raw.astype(np.float64) * params.tdc_bin_ns
    present = raw != 0

    if params.mcp_is_group_trigger or params.mcp_channel is None:
        t_mcp = np.zeros(n)
    else:
        t_mcp = np.where(present[:, params.mcp_channel], times[:, params.mcp_channel], np.nan)

    layer_specs = (
        ('u', params.fu, 0.0, params.sum_offset_u_ns, params.sum_halfwidth_u_ns, params.runtime_u_ns),
        ('v', params.fv, 0.0, params.sum_offset_v_ns, params.sum_halfwidth_v_ns, params.runtime_v_ns),
        ('w', params.fw, params.w_offset_ns, params.sum_offset_w_ns, params.sum_halfwidth_w_ns, params.runtime_w_ns),
    )
    positions, full, rebuilt, sums = {}, {}, {}, {}
    for index, (name, factor, offset, sum_offset, halfwidth, runtime) in enumerate(layer_specs):
        ch1, ch2 = params.anode_channels[2 * index], params.anode_channels[2 * index + 1]
        t1, t2 = times[:, ch1], times[:, ch2]
        p1, p2 = present[:, ch1], present[:, ch2]
        both = p1 & p2
        corrected_sum = np.where(both, t1 + t2 - 2.0 * t_mcp + sum_offset, np.nan)
        # A single missing signal is rebuilt from the nominal time sum.
        target = 2.0 * t_mcp - sum_offset
        t1 = np.where(p2 & ~p1, target - t2, t1)
        t2 = np.where(p1 & ~p2, target - t1, t2)
        diff = t1 - t2
        in_runtime = np.abs(diff) <= runtime + halfwidth
        full[name] = both & (np.abs(corrected_sum) <= halfwidth) & in_runtime
        rebuilt[name] = (p1 ^ p2) & np.isfinite(t_mcp) & in_runtime
        positions[name] = 0.5 * factor * (diff + offset)
        sums[name] = corrected_sum

    n_full = full['u'].astype(int) + full['v'].astype(int) + full['w'].astype(int)
    use = {}
    for name in 'uvw':
        # Rebuilt layers only count next to at least one complete layer, and
        # are not needed once two complete layers exist.
        use[name] = full[name] | (rebuilt[name] & (n_full == 1))
    mask = sum(use[name].astype(np.int64) * bit for name, bit in LAYER_BITS.items())
    u, v, w = positions['u'], positions['v'], positions['w']

    x = np.full(n, np.nan)
    y = np.full(n, np.nan)
    for layer_mask, x_formula, y_formula in (
        (7, lambda: (2.0 * u + v + w) / 3.0, lambda: (w - v) / _SQRT3),
        (3, lambda: u, lambda: (u - 2.0 * v) / _SQRT3),
        (5, lambda: u, lambda: (2.0 * w - u) / _SQRT3),
        (6, lambda: v + w, lambda: (w - v) / _SQRT3),
    ):
        selected = mask == layer_mask
        if selected.any():
            x[selected] = x_formula()[selected]
            y[selected] = y_formula()[selected]

    if params.start_channel is None:
        tof = np.full(n, np.nan)
    else:
        start = params.start_channel
        tof = np.where(present[:, start], t_mcp - times[:, start] + params.tof_offset_ns, np.nan)
        if params.require_start:
            x[~present[:, start]] = np.nan
            y[~present[:, start]] = np.nan
    valid = np.isfinite(x)
    mask = np.where(valid, mask, 0)
    x, y = _finish_positions(x, y, params, correction)

    n_reconstructed = sum(((mask & bit) > 0) & ~full[name] for name, bit in LAYER_BITS.items())
    return {
        'x_mm': x,
        'y_mm': y,
        'tof_ns': tof,
        'valid': valid,
        'layer_mask': mask.astype(np.int8),
        'n_reconstructed': n_reconstructed.astype(np.int8),
        'sum_u_ns': sums['u'],
        'sum_v_ns': sums['v'],
        'sum_w_ns': sums['w'],
        'hex_delta_mm': u - (v + w),
    }


@dataclass
class RoentdekHits:
    """Every TDC hit of every event in compressed-row (CSR) form.

    The hits of event ``i`` are ``channel[offsets[i]:offsets[i + 1]]`` and
    ``value[...]`` (TDC bins relative to the MCP group trigger), grouped by
    channel and in arrival order within a channel. Zero values are dropped
    (pyccapt uses 0 for "no hit"); the MCP trigger itself is implicit.
    """

    offsets: np.ndarray
    channel: np.ndarray
    value: np.ndarray
    timestamp_ns: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def event_index(self) -> np.ndarray:
        return np.repeat(np.arange(len(self)), np.diff(self.offsets))

    def counts(self, n_channels: int = 8) -> np.ndarray:
        """``(n, n_channels)`` number of hits per event and channel."""
        flat = self.event_index() * n_channels + self.channel
        return np.bincount(flat, minlength=len(self) * n_channels).reshape(len(self), n_channels)

    def first_hits(self, n_channels: int = 8) -> np.ndarray:
        """``(n, n_channels)`` first hit per channel, 0 when the channel did not fire."""
        event = self.event_index()
        first = np.ones(len(event), dtype=bool)
        first[1:] = (event[1:] != event[:-1]) | (self.channel[1:] != self.channel[:-1])
        out = np.zeros((len(self), n_channels), dtype=np.int64)
        out[event[first], self.channel[first]] = self.value[first]
        return out

    @classmethod
    def from_rows(cls, event, channel, value, n_events: int, timestamp_ns=None) -> 'RoentdekHits':
        """Build from one row per hit (``event`` ascending); drops zero values."""
        event = np.asarray(event, dtype=np.int64)
        channel = np.asarray(channel, dtype=np.int64)
        value = np.asarray(value, dtype=np.int64)
        keep = value != 0
        event, channel, value = event[keep], channel[keep], value[keep]
        order = np.lexsort((channel, event))  # stable: arrival order kept within a channel
        event, channel, value = event[order], channel[order], value[order]
        offsets = np.zeros(n_events + 1, dtype=np.int64)
        np.cumsum(np.bincount(event, minlength=n_events), out=offsets[1:])
        return cls(offsets, channel.astype(np.int8), value, timestamp_ns)

    @classmethod
    def from_first_hits(cls, channels) -> 'RoentdekHits':
        """Wrap an ``(n, 8)`` first-hit array (older pyccapt files)."""
        channels = np.asarray(channels, dtype=np.int64)
        n, width = channels.shape
        event = np.repeat(np.arange(n), width)
        return cls.from_rows(event, np.tile(np.arange(width), n), channels.reshape(-1), n)


def read_lmf_txt(path: str | Path) -> RoentdekHits:
    """Read every hit from an ``lmf2txt`` dump of a CoboldPC LMF file (``*_raw.txt``).

    Event blocks look like ``------- #N -------``, ``T  = <ns> ns = ...`` and
    one ``chan <c> <n> <bin> <bin> ...`` line per channel (1-based channel,
    hit count, then the hits in TDC bins).
    """
    events, channels, values, stamps = [], [], [], []
    event = -1
    with open(path, 'r', encoding='utf-8', errors='replace') as handle:
        for line in handle:
            if line.startswith('chan'):
                parts = line.split()
                count = int(parts[2])
                if count:
                    hits = parts[3:3 + count]
                    events.extend([event] * len(hits))
                    channels.extend([int(parts[1]) - 1] * len(hits))
                    values.extend(hits)
            elif line.startswith('------- #'):
                event += 1
                stamps.append(np.nan)
            elif line.startswith('T ') and event >= 0:
                try:
                    stamps[event] = float(line.split('=')[1].split()[0])
                except (IndexError, ValueError):
                    pass
    return RoentdekHits.from_rows(np.asarray(events, dtype=np.int64), np.asarray(channels, dtype=np.int64),
                                  np.asarray(values, dtype=np.int64), event + 1, np.asarray(stamps))


def _layer_specs(params: HexanodeParameters):
    channels = params.anode_channels
    return (
        (channels[0], channels[1], params.fu, 0.0, params.sum_offset_u_ns, params.sum_halfwidth_u_ns, params.runtime_u_ns),
        (channels[2], channels[3], params.fv, 0.0, params.sum_offset_v_ns, params.sum_halfwidth_v_ns, params.runtime_v_ns),
        (channels[4], channels[5], params.fw, params.w_offset_ns, params.sum_offset_w_ns, params.sum_halfwidth_w_ns,
         params.runtime_w_ns),
    )


def _xy_from_layers(mask: int, u: float, v: float, w: float) -> tuple[float, float]:
    if mask == 7:
        return (2.0 * u + v + w) / 3.0, (w - v) / _SQRT3
    if mask == 3:
        return u, (u - 2.0 * v) / _SQRT3
    if mask == 5:
        return u, (2.0 * w - u) / _SQRT3
    return v + w, (w - v) / _SQRT3


def fit_sum_walk(channels, params: HexanodeParameters | None = None, *, degree: int = 4,
                 min_events: int = 2000) -> np.ndarray:
    """Time-sum walk of each layer: corrected sum (ns) as a polynomial of the layer position (mm).

    The anode time sum drifts across the detector (here by up to ~2.7 ns on
    the w layer), which pushes real hits at the detector edge out of the
    time-sum gate. CoboldPC gates on walk-corrected sums; this fits the walk
    from first-hit events with three complete, mutually consistent layers.
    Returns a ``(3, degree + 1)`` array of ``np.polyval`` coefficients for
    u, v, w (all zero when fewer than ``min_events`` clean events exist).
    """
    params = params or HexanodeParameters()
    channels = np.asarray(channels)
    result = reconstruct_hexanode(channels, params)
    clean = (result['layer_mask'] == 7) & (result['n_reconstructed'] == 0) & (np.abs(result['hex_delta_mm']) < 1.0)
    walk = np.zeros((3, degree + 1))
    if clean.sum() < min_events:
        return walk
    times = channels[clean].astype(np.float64) * params.tdc_bin_ns
    for k, (a, b, factor, offset, _sum_offset, _halfwidth, _runtime) in enumerate(_layer_specs(params)):
        position = 0.5 * factor * (times[:, a] - times[:, b] + offset)
        corrected_sum = result[f'sum_{"uvw"[k]}_ns'][clean]
        keep = np.abs(corrected_sum - np.median(corrected_sum)) < 3 * corrected_sum.std()
        walk[k] = np.polyfit(position[keep], corrected_sum[keep], degree)
    return walk


def _horner(coefficients, x: float) -> float:
    value = 0.0
    for c in coefficients:
        value = value * x + c
    return value


def _event_particles(times, params, layers, walk, max_particles, hex_tolerance_mm, max_pairs, min_free_layers,
                     share_signals, single_signal_window_ns, duplicate_mm=2.0, duplicate_ns=2.0):
    """Particles of one event from per-channel hit times (ns); see reconstruct_hexanode_multihit."""
    used = [[False] * len(channel_times) for channel_times in times]
    start = params.start_channel
    t_start = times[start][0] if start is not None and times[start] else None
    if params.require_start and start is not None and t_start is None:
        return []
    half = [0.5 * layer[5] for layer in layers]

    def pairs(k, t_mcp):
        # (|arrival - t_mcp|, arrival, position, hits) of complete pairs of layer k
        a, b, factor, offset, sum_offset, halfwidth, runtime = layers[k]
        out = []
        for i, ta in enumerate(times[a]):
            if used[a][i]:
                continue
            for j, tb in enumerate(times[b]):
                if used[b][j] or abs(ta - tb) > runtime + halfwidth:
                    continue
                position = 0.5 * factor * (ta - tb + offset)
                arrival = 0.5 * (ta + tb + sum_offset - _horner(walk[k], position))
                if t_mcp is not None and abs(arrival - t_mcp) > half[k]:
                    # Keep what the plain (walk-free) gate accepts, as the first-hit reconstruction does.
                    plain = 0.5 * (ta + tb + sum_offset)
                    if abs(plain - t_mcp) > half[k]:
                        continue
                    arrival = plain
                out.append((abs(arrival - t_mcp) if t_mcp is not None else 0.0, arrival, position, ((a, i), (b, j))))
        if t_mcp is None:
            return sorted(out, key=lambda item: item[1])
        out.sort(key=lambda item: item[0])
        return out[:max_pairs]

    def singles(k, t_mcp, need_empty_partner, allow_used=False):
        # (position, hits) of layer k rebuilt from one signal at arrival time t_mcp
        a, b, factor, offset, sum_offset, halfwidth, runtime = layers[k]
        free_a = [i for i, flag in enumerate(used[a]) if allow_used or not flag]
        free_b = [j for j, flag in enumerate(used[b]) if allow_used or not flag]
        out = []
        for channel, free, sign, partner in ((a, free_a, 1.0, free_b), (b, free_b, -1.0, free_a)):
            if need_empty_partner and partner:
                continue
            for i in free:
                position = 0.0
                for _ in range(2):  # the walk depends on the position being rebuilt
                    target = 2.0 * t_mcp - sum_offset + _horner(walk[k], position)
                    diff = sign * (2.0 * times[channel][i] - target)
                    position = 0.5 * factor * (diff + offset)
                if abs(diff) <= runtime + halfwidth:
                    out.append((position, ((channel, i),)))
        return out

    def three_layers(cands_or_singles, base_score, arrival, n_rebuilt, best):
        # cands_or_singles: per layer a list of (score, position, hits)
        for cu in cands_or_singles[0]:
            for cv in cands_or_singles[1]:
                for cw in cands_or_singles[2]:
                    delta = abs(cu[1] - (cv[1] + cw[1]))
                    if delta > hex_tolerance_mm:
                        continue
                    score = base_score + cu[0] + cv[0] + cw[0] + delta / hex_tolerance_mm
                    if best is None or score < best[0]:
                        best = (score, 7, (cu[1], cv[1], cw[1]), cu[2] + cv[2] + cw[2], n_rebuilt, arrival)
        return best

    def as_scored(cands, k):
        return [(c[0] / half[k], c[2], c[3]) for c in cands]

    def as_rebuilt(rebuilt):
        return [(1.0, r[0], r[1]) for r in rebuilt]

    def best_with_mcp():
        cands = [pairs(k, 0.0) for k in range(3)]
        full = [k for k in range(3) if cands[k]]
        best = None
        if len(full) == 3:
            best = three_layers([as_scored(cands[k], k) for k in range(3)], 0.0, 0.0, 0, None)
            if best is not None:
                return best
        if len(full) >= 2:
            for k1, k2 in ((0, 1), (0, 2), (1, 2)):
                for c1 in cands[k1]:
                    for c2 in cands[k2]:
                        score = c1[0] / half[k1] + c2[0] / half[k2]
                        if best is None or score < best[0]:
                            position = [0.0, 0.0, 0.0]
                            position[k1], position[k2] = c1[2], c2[2]
                            best = (score, (1 << k1) | (1 << k2), tuple(position), c1[3] + c2[3], 0, 0.0)
            return best
        if len(full) == 1:
            k = full[0]
            others = [m for m in range(3) if m != k]
            per_layer = [None, None, None]
            per_layer[k] = as_scored(cands[k], k)
            per_layer[others[0]] = as_rebuilt(singles(others[0], 0.0, False))
            per_layer[others[1]] = as_rebuilt(singles(others[1], 0.0, False))
            best = three_layers(per_layer, 0.0, 0.0, 2, None)
            if best is not None:
                return best
            lone = cands[k][0]
            for m in others:
                rebuilt = singles(m, 0.0, True)
                if rebuilt:
                    position = [0.0, 0.0, 0.0]
                    position[k], position[m] = lone[2], rebuilt[0][0]
                    return (0.0, (1 << k) | (1 << m), tuple(position), lone[3] + rebuilt[0][1], 1, 0.0)
            return None
        # No complete layer: three single signals rebuilt against the MCP (CoboldPC
        # does the same); only accepted when the hexanode redundancy holds.
        return three_layers([as_rebuilt(singles(k, 0.0, False)) for k in range(3)], 3.0, 0.0, 3, None)

    def best_free():
        # A later ion arrives after the MCP trigger: an earlier one would have
        # triggered the group itself.
        earliest = -min(half)
        cands = [[c for c in pairs(k, None) if c[1] >= earliest] for k in range(3)]
        best = None
        for k1, k2 in ((0, 1), (0, 2), (1, 2)):
            k3 = 3 - k1 - k2
            for c1 in cands[k1]:
                for c2 in cands[k2]:
                    if abs(c1[1] - c2[1]) > half[k1] + half[k2]:
                        continue
                    arrival = 0.5 * (c1[1] + c2[1])
                    per_layer = [None, None, None]
                    per_layer[k1] = [(abs(c1[1] - arrival) / half[k1], c1[2], c1[3])]
                    per_layer[k2] = [(abs(c2[1] - arrival) / half[k2], c2[2], c2[3])]
                    third = [(abs(c[1] - arrival) / half[k3], c[2], c[3]) for c in cands[k3]
                             if abs(c[1] - arrival) <= half[k3] and not set(c[3]) & set(c1[3] + c2[3])]
                    if third:
                        per_layer[k3] = third
                        best = three_layers(per_layer, 0.0, arrival, 0, best)
                    else:
                        per_layer[k3] = as_rebuilt(singles(k3, arrival, False, share_signals))
                        best = three_layers(per_layer, 1.0, arrival, 1, best)
                    if min_free_layers <= 2:
                        # Two layers only: no redundancy check, so rank below any three-layer match.
                        score = 10.0 + abs(c1[1] - c2[1]) / (half[k1] + half[k2])
                        if best is None or score < best[0]:
                            position = [0.0, 0.0, 0.0]
                            position[k1], position[k2] = c1[2], c2[2]
                            best = (score, (1 << k1) | (1 << k2), tuple(position), c1[3] + c2[3], 0, arrival)
        # One complete layer gives the arrival time; the other two layers are
        # rebuilt from single signals (the partner signals of a close double
        # hit are often lost in the anode dead time) and must pass the hexanode check.
        for k in range(3):
            others = [m for m in range(3) if m != k]
            for c in cands[k]:
                per_layer = [None, None, None]
                per_layer[k] = [(0.0, c[2], c[3])]
                for m in others:
                    per_layer[m] = as_rebuilt(singles(m, c[1], False, share_signals))
                best = three_layers(per_layer, 2.0, c[1], 2, best)
        if best is None and single_signal_window_ns:
            best = three_singles()
        return best

    def three_singles():
        # One signal per layer and no complete layer: u = v + w fixes the arrival
        # time. There is no redundancy left, so accept only a unique solution.
        options = []
        for k in range(3):
            a, b, factor, offset, sum_offset, halfwidth, runtime = layers[k]
            options.append([(channel, i, sign) for channel, sign in ((a, 1.0), (b, -1.0))
                            for i, flag in enumerate(used[channel]) if not flag])
        solutions = []
        for pick in ((x, y, z) for x in options[0] for y in options[1] for z in options[2]):
            walk_terms = [0.0, 0.0, 0.0]
            for _ in range(2):
                intercept, slope = [], []
                for k, (channel, i, sign) in enumerate(pick):
                    factor, offset, sum_offset = layers[k][2], layers[k][3], layers[k][4]
                    # position_k = intercept_k - slope_k * arrival
                    intercept.append(0.5 * factor * (sign * (2.0 * times[channel][i] + sum_offset - walk_terms[k]) + offset))
                    slope.append(factor * sign)
                arrival = (intercept[0] - intercept[1] - intercept[2]) / (slope[0] - slope[1] - slope[2])
                position = [intercept[k] - slope[k] * arrival for k in range(3)]
                walk_terms = [_horner(walk[k], position[k]) for k in range(3)]
            if not -min(half) <= arrival <= single_signal_window_ns:
                continue
            if any(abs(2.0 * position[k] / layers[k][2] - layers[k][3]) > layers[k][6] + layers[k][5] for k in range(3)):
                continue
            solutions.append((3.0, 7, tuple(position), tuple((c, i) for c, i, _ in pick), 3, arrival))
        return solutions[0] if len(solutions) == 1 else None

    particles = []
    found = best_with_mcp()
    if found is not None:
        particles.append(found)
    while len(particles) < max_particles:
        if found is not None:
            for channel, index in found[3]:
                used[channel][index] = True
        found = best_free()
        if found is None:
            break
        x_new, y_new = _xy_from_layers(found[1], *found[2])
        if any(abs(found[5] - p[5]) < duplicate_ns and np.hypot(x_new - px, y_new - py) < duplicate_mm
               for p, (px, py) in zip(particles, (_xy_from_layers(p[1], *p[2]) for p in particles))):
            break  # same ion seen again through shared signals
        particles.append(found)
    out = []
    for score, mask, (u, v, w), hits, n_rebuilt, arrival in particles:
        x, y = _xy_from_layers(mask, u, v, w)
        tof = arrival - t_start + params.tof_offset_ns if t_start is not None else np.nan
        out.append((x, y, tof, mask, n_rebuilt, arrival))
    return out  # the MCP-referenced particle first, then later particles in search order


def reconstruct_hexanode_multihit(
    hits: RoentdekHits,
    params: HexanodeParameters | None = None,
    correction: HexanodeCorrection | None = None,
    *,
    sum_walk=None,
    max_particles: int = 4,
    hex_tolerance_mm: float = 1.5,
    max_pairs: int = 4,
    min_free_layers: int = 3,
    share_signals: bool = False,
    single_signal_window_ns: float = 0.0,
) -> dict[str, np.ndarray]:
    """Reconstruct every particle of every event from all TDC hits (LMF-like).

    Time sums are corrected for their position dependence (``sum_walk``:
    ``None`` uses ``correction.sum_walk`` when present and otherwise fits it
    from this data with :func:`fit_sum_walk`, ``False`` disables it, or pass
    the ``(3, d)`` coefficients). A pair the plain time-sum gate accepts is
    always kept, as in :func:`reconstruct_hexanode`.

    The first particle is referenced to the MCP group trigger (t = 0): per
    layer the hit pair with the smallest walk-corrected time sum inside the
    gate is taken; three layers must agree within ``hex_tolerance_mm``
    (``u = v + w``), otherwise the best two layers are used. With one
    complete layer the missing layers are rebuilt from single signals and the
    time sum; with none, three single signals (one per layer) are accepted
    only when they satisfy the hexanode check. This recovers events whose
    first hit on a channel is a reflection or noise pulse.

    Later particles have no MCP signal of their own (the TDC records only the
    trigger), so, like CoboldPC, their arrival time is rebuilt from the anode
    time sums of complete layers built from unused hits: two layers whose
    arrival times agree within the half-gates plus a third layer (complete, or
    rebuilt from one signal at that arrival time), or one complete layer plus
    two layers rebuilt from single signals; every three-layer particle must
    pass the hexanode check, and must arrive after the MCP trigger (an earlier
    ion would have triggered the group). Options, all off by default because
    the extra particles mostly disagree with CoboldPC on runs 194-207:
    ``min_free_layers=2`` accepts two complete layers without a third;
    ``share_signals`` lets a rebuilt layer reuse a signal of an earlier
    particle; ``single_signal_window_ns > 0`` accepts one signal per layer
    with no complete layer when ``u = v + w`` gives a unique arrival time
    within that window. A later particle within 2 mm and 2 ns of an earlier one
    is treated as the same ion and ends the search.

    CoboldPC reports many more second particles (6014 vs ~180 on run 198),
    but more than half of them have none of their own signals in the raw data
    (their predicted signals fall within the anode dead time of the first
    particle), so they cannot be measured independently.

    Events with at most one hit per channel and a valid first-hit position go
    through the vectorised :func:`reconstruct_hexanode`.

    Returns:
        Per-particle arrays ``event``, ``x_mm``, ``y_mm``, ``tof_ns``,
        ``layer_mask``, ``n_reconstructed`` and ``arrival_ns`` (relative to
        the MCP trigger; 0 for the trigger particle), ordered by event then
        search order (the MCP-referenced particle first), plus per-event
        ``n_particles`` and the ``sum_walk`` coefficients used.
    """
    params = params or HexanodeParameters()
    if params.rotation_deg:
        raise NotImplementedError('rotation is not supported by the multi-hit reconstruction yet')
    n = len(hits)
    counts = hits.counts()
    first = hits.first_hits()
    if sum_walk is None and correction is not None and correction.sum_walk is not None:
        walk = np.asarray(correction.sum_walk, dtype=np.float64)
    elif sum_walk is None:
        walk = fit_sum_walk(first, params)
    elif sum_walk is False:
        walk = np.zeros((3, 1))
    else:
        walk = np.asarray(sum_walk, dtype=np.float64)
    walk_rows = [list(row) for row in walk]

    anode = list(params.anode_channels)
    single_channels = anode + ([params.start_channel] if params.start_channel is not None else [])
    simple = (counts[:, single_channels] <= 1).all(axis=1)
    base = reconstruct_hexanode(first[simple], params)
    base_valid = base['valid']
    simple_events = np.flatnonzero(simple)
    keep = simple_events[base_valid]
    # Positions from reconstruct_hexanode already carry the detector offsets;
    # undo them so every particle goes through one _finish_positions pass.
    rows_event = [keep]
    rows = [np.column_stack([base['x_mm'][base_valid] - params.x_offset_mm, base['y_mm'][base_valid] - params.y_offset_mm,
                             base['tof_ns'][base_valid], base['layer_mask'][base_valid],
                             base['n_reconstructed'][base_valid], np.zeros(base_valid.sum()),
                             np.zeros(base_valid.sum())])]

    has_start = counts[:, params.start_channel] > 0 if params.start_channel is not None else np.ones(n, dtype=bool)
    retry = np.zeros(n, dtype=bool)
    retry[simple_events[~base_valid]] = True
    retry &= has_start & (counts[:, anode].sum(axis=1) >= 3)
    loop_events = np.flatnonzero(~simple | retry)

    layers = _layer_specs(params)
    bin_ns = params.tdc_bin_ns
    offsets, channel, value = hits.offsets, hits.channel, hits.value
    loop_rows, loop_event = [], []
    for event in loop_events:
        lo, hi = offsets[event], offsets[event + 1]
        times = [[] for _ in range(8)]
        for c, t in zip(channel[lo:hi].tolist(), value[lo:hi].tolist()):
            if c < 8:
                times[c].append(t * bin_ns)
        particles = _event_particles(times, params, layers, walk_rows, max_particles, hex_tolerance_mm, max_pairs,
                                     min_free_layers, share_signals, single_signal_window_ns)
        for rank, particle in enumerate(particles):
            loop_rows.append(particle + (rank,))
            loop_event.append(event)
    if loop_rows:
        rows.append(np.asarray(loop_rows, dtype=np.float64))
        rows_event.append(np.asarray(loop_event, dtype=np.int64))

    table = np.concatenate(rows)
    event = np.concatenate(rows_event)
    order = np.lexsort((table[:, 6], event))  # MCP particle first, then later particles
    table, event = table[order], event[order]
    x, y = _finish_positions(table[:, 0], table[:, 1], params, correction)
    return {
        'event': event,
        'x_mm': x,
        'y_mm': y,
        'tof_ns': table[:, 2],
        'layer_mask': table[:, 3].astype(np.int8),
        'n_reconstructed': table[:, 4].astype(np.int8),
        'arrival_ns': table[:, 5],
        'n_particles': np.bincount(event, minlength=n),
        'sum_walk': walk,
    }


def first_particle(result: dict[str, np.ndarray], n_events: int) -> dict[str, np.ndarray]:
    """Per-event view (``x_mm``, ``y_mm``, ``tof_ns``, ``valid``) of the first (MCP-referenced) particle."""
    event = result['event']
    first = np.ones(len(event), dtype=bool)
    first[1:] = event[1:] != event[:-1]
    out = {'valid': np.zeros(n_events, dtype=bool)}
    for key in ('x_mm', 'y_mm', 'tof_ns'):
        out[key] = np.full(n_events, np.nan)
        out[key][event[first]] = result[key][first]
    out['valid'][event[first]] = True
    out['layer_mask'] = np.zeros(n_events, dtype=np.int8)
    out['layer_mask'][event[first]] = result['layer_mask'][first]
    return out


def _read_tdc_group(path: str | Path):
    """Raw RoentDek ``tdc`` group: ``('per_channel', (n, 8) bins, {})`` or
    ``('flat', (time_data, channel, event, n_events), per-event extras)``."""
    import h5py

    with h5py.File(path, 'r') as handle:
        if 'tdc/ch0' in handle:
            columns = [np.asarray(handle[f'tdc/ch{i}'][:]) for i in range(8)]
            return 'per_channel', np.stack([_as_signed(column) for column in columns], axis=1), {}
        if 'tdc/time_data' not in handle or 'tdc/channel' not in handle:
            raise KeyError(f'{path} has no RoentDek tdc channels (tdc/ch0..ch7 or tdc/time_data)')
        dataset = handle['tdc/time_data']
        offset = dataset.attrs.get('roentdek_time_offset')
        if offset is not None:
            # Reader v3 (2026-11): signed bins + offset, every row is one real hit.
            time_data = np.asarray(dataset[:]).astype(np.int64) - int(offset)
        else:
            time_data = _as_signed(np.asarray(dataset[:]))
        channel = np.asarray(handle['tdc/channel'][:]).astype(np.int64)
        extras = {name: np.asarray(handle[f'tdc/{name}'][:])
                  for name in ('start_counter', 'high_voltage', 'voltage_pulse', 'laser_pulse')
                  if f'tdc/{name}' in handle}

    if len(channel) % 8 == 0 and np.array_equal(channel, np.tile(np.arange(8), len(channel) // 8)):
        starts = np.arange(0, len(channel), 8)
    elif 'start_counter' in extras:
        start_counter = extras['start_counter'].astype(np.int64)
        starts = np.r_[0, np.flatnonzero(np.diff(start_counter) != 0) + 1] if len(start_counter) else np.zeros(0, np.int64)
    else:
        raise ValueError('flat tdc table is not 8 rows per event and has no start_counter to group by')
    event = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(channel)]))
    return 'flat', (time_data, channel, event, len(starts)), {name: values[starts] for name, values in extras.items()}


def load_roentdek_tdc_channels(path: str | Path) -> dict[str, np.ndarray]:
    """Load first-hit RoentDek channels from a pyccapt HDF5 file.

    Supports every layout pyccapt has written:

    * legacy per-channel datasets ``tdc/ch0`` .. ``tdc/ch7``;
    * the flat ``tdc/channel`` + ``tdc/time_data`` table, either 8 rows per
      event (first hits, 0 = no hit) or one row per hit grouped by
      ``start_counter`` (all hits, see :func:`load_roentdek_hits`).
      ``time_data`` is stored as uint64. The RoentDek reader v3 (Resort64c,
      attribute ``roentdek_time_offset``) stores signed bins + 2**23; older
      files stored negative bins (the start channel precedes the MCP
      trigger) two's-complement, recovered by reinterpreting as int64.

    Returns:
        ``channels`` as an ``(n, 8)`` int64 array plus any per-event
        ``start_counter``, ``high_voltage``, ``voltage_pulse`` and
        ``laser_pulse`` arrays found in the ``tdc`` group.
    """
    layout, data, extras = _read_tdc_group(path)
    if layout == 'per_channel':
        return {'channels': data}
    time_data, channel, event, n_events = data
    channels = np.zeros((n_events, 8), dtype=np.int64)
    keep = np.flatnonzero((channel >= 0) & (channel < 8) & (time_data != 0))
    # Reverse order so the first value per (event, channel) wins the write.
    order = keep[::-1]
    channels[event[order], channel[order]] = time_data[order]
    return {'channels': channels, **extras}


def load_roentdek_hits(path: str | Path) -> tuple[RoentdekHits, dict[str, np.ndarray]]:
    """Load every RoentDek hit of a pyccapt HDF5 file (any layout).

    Files written before all hits were stored only hold the first hit per
    channel; they load as one hit per channel. Returns the hits and the
    per-event ``tdc`` metadata (``start_counter``, ``high_voltage``, ...).
    """
    layout, data, extras = _read_tdc_group(path)
    if layout == 'per_channel':
        return RoentdekHits.from_first_hits(data), extras
    time_data, channel, event, n_events = data
    keep = (channel >= 0) & (channel < 8)
    return RoentdekHits.from_rows(event[keep], channel[keep], time_data[keep], n_events), extras


def _as_signed(values: np.ndarray) -> np.ndarray:
    if values.dtype == np.uint64:
        return values.view(np.int64)
    if np.issubdtype(values.dtype, np.floating):
        return np.rint(values).astype(np.int64)
    return values.astype(np.int64)


def reconstruct_roentdek_hdf5(
    path: str | Path,
    params: HexanodeParameters | None = None,
    correction: HexanodeCorrection | None = None,
    *,
    invalid_value: float = np.nan,
    all_particles: bool = False,
    **multihit_options,
):
    """Rebuild ``x_det``/``y_det``/``t`` for every TDC event of an HDF5 file.

    Returns a DataFrame with one row per TDC event (the same order as the
    ``tdc`` group and as CoboldPC's ASCII export of the matching LMF):
    ``x_det (cm)``, ``y_det (cm)``, ``t (ns)``, ``valid``, ``layer_mask``,
    ``n_reconstructed`` and the per-event ``tdc`` metadata when present.
    Rows without a position get ``invalid_value`` for x, y and t (pass 0.0
    to match CoboldPC's convention for events with no particle).

    With ``all_particles=True`` every stored hit is used
    (:func:`reconstruct_hexanode_multihit`, ``multihit_options`` passed
    through) and the frame has one row per particle with its ``event`` index,
    ``particle`` number (0 = MCP-referenced) and ``arrival (ns)``; events
    without a particle have no row.
    """
    import pandas as pd

    if all_particles:
        hits, extras = load_roentdek_hits(path)
        result = reconstruct_hexanode_multihit(hits, params, correction, **multihit_options)
        event = result['event']
        first = np.ones(len(event), dtype=bool)
        first[1:] = event[1:] != event[:-1]
        starts = np.flatnonzero(first)
        particle = np.arange(len(event)) - np.repeat(starts, np.diff(np.r_[starts, len(event)]))
        frame = pd.DataFrame({
            'event': event,
            'particle': particle,
            'x_det (cm)': result['x_mm'] / 10.0,
            'y_det (cm)': result['y_mm'] / 10.0,
            't (ns)': result['tof_ns'],
            'layer_mask': result['layer_mask'],
            'n_reconstructed': result['n_reconstructed'],
            'arrival (ns)': result['arrival_ns'],
        })
        for name, column in (('start_counter', 'start_counter'), ('high_voltage', 'high_voltage (V)'),
                             ('voltage_pulse', 'pulse_v (V)'), ('laser_pulse', 'pulse_l (pJ)')):
            if name in extras:
                frame[column] = extras[name][event]
        return frame

    loaded = load_roentdek_tdc_channels(path)
    result = reconstruct_hexanode(loaded['channels'], params, correction)
    valid = result['valid']
    frame = pd.DataFrame({
        'x_det (cm)': np.where(valid, result['x_mm'] / 10.0, invalid_value),
        'y_det (cm)': np.where(valid, result['y_mm'] / 10.0, invalid_value),
        't (ns)': np.where(valid, result['tof_ns'], invalid_value),
        'valid': valid,
        'layer_mask': result['layer_mask'],
        'n_reconstructed': result['n_reconstructed'],
    })
    for name, column in (('start_counter', 'start_counter'), ('high_voltage', 'high_voltage (V)'),
                         ('voltage_pulse', 'pulse_v (V)'), ('laser_pulse', 'pulse_l (pJ)')):
        if name in loaded:
            frame[column] = loaded[name]
    return frame


def read_cobold_txt(path: str | Path, max_particles: int = 4):
    """Read a CoboldPC DAn "ASCII special" export (``Parameter 1008,3``).

    Each row is one LMF event: event counter, two unused columns, absolute
    event time (s), number of particles, consistence indicator, then
    ``x [mm], y [mm], tof [ns], reconstruction method`` per particle.
    """
    import pandas as pd

    names = ['event', 'col2', 'col3', 'abs_time_s', 'n_particles', 'consistence']
    for k in range(1, max_particles + 1):
        names += [f'x{k}_mm', f'y{k}_mm', f'tof{k}_ns', f'method{k}']
    return pd.read_csv(path, sep=r'\s+', header=None, names=names, usecols=range(len(names)),
                       engine='c', dtype=np.float64)


def compare_with_cobold(result: dict[str, np.ndarray], cobold) -> dict[str, float]:
    """Agreement of :func:`reconstruct_hexanode` output with a CoboldPC export.

    Rows are matched by index; the shorter of the two sets the length.
    """
    n = min(len(result['x_mm']), len(cobold))
    cob = cobold.iloc[:n]
    has_particle = cob['n_particles'].to_numpy() >= 1
    valid = result['valid'][:n]
    both = valid & has_particle
    dx = result['x_mm'][:n][both] - cob['x1_mm'].to_numpy()[both]
    dy = result['y_mm'][:n][both] - cob['y1_mm'].to_numpy()[both]
    distance = np.hypot(dx, dy)
    single = both & (cob['n_particles'].to_numpy() == 1)
    single_distance = np.hypot(result['x_mm'][:n][single] - cob['x1_mm'].to_numpy()[single],
                               result['y_mm'][:n][single] - cob['y1_mm'].to_numpy()[single])

    def pct(values, limit):
        return float(np.mean(values <= limit)) if len(values) else float('nan')

    return {
        'events': int(n),
        'cobold_with_particle': int(has_particle.sum()),
        'pyccapt_valid': int(valid.sum()),
        'both': int(both.sum()),
        'cobold_only': int((has_particle & ~valid).sum()),
        'pyccapt_only': int((valid & ~has_particle).sum()),
        'rms_x_mm': float(np.sqrt(np.mean(dx ** 2))) if len(dx) else float('nan'),
        'rms_y_mm': float(np.sqrt(np.mean(dy ** 2))) if len(dy) else float('nan'),
        'median_distance_mm': float(np.median(distance)) if len(distance) else float('nan'),
        'within_0.1mm': pct(distance, 0.1),
        'within_0.5mm': pct(distance, 0.5),
        'within_1mm': pct(distance, 1.0),
        'single_within_0.5mm': pct(single_distance, 0.5),
        'single_within_1mm': pct(single_distance, 1.0),
    }


def fit_cobold_correction(
    result: dict[str, np.ndarray],
    cobold,
    *,
    degree: int = 5,
    outlier_mm: float = 1.0,
) -> HexanodeCorrection:
    """Fit a polynomial map from analytic positions to CoboldPC positions.

    ``result`` must come from :func:`reconstruct_hexanode` *without* a
    correction, on the same events (same order) as the CoboldPC export
    ``cobold`` (see :func:`read_cobold_txt`). Only single-particle events
    with all three layers complete are used, and a second pass drops events
    where CoboldPC picked a different hit than the first one (residual above
    ``outlier_mm``). The map is detector-specific: refit it whenever the CCF
    constants change.
    """
    n = min(len(result['x_mm']), len(cobold))
    cob = cobold.iloc[:n]
    selected = (result['valid'][:n] & (result['layer_mask'][:n] == 7)
                & (result['n_reconstructed'][:n] == 0)
                & (cob['n_particles'].to_numpy() == 1))
    x, y = result['x_mm'][:n][selected], result['y_mm'][:n][selected]
    target_x, target_y = cob['x1_mm'].to_numpy()[selected], cob['y1_mm'].to_numpy()[selected]
    if len(x) < 1000:
        raise ValueError(f'only {len(x)} complete single hits to fit a correction; need at least 1000')

    correction = HexanodeCorrection(degree, np.zeros(0), np.zeros(0))
    design = correction._design(x, y)
    keep = np.hypot(target_x - x, target_y - y) <= 5.0 * outlier_mm
    for _ in range(2):
        correction.coef_x = np.linalg.lstsq(design[keep], target_x[keep], rcond=None)[0]
        correction.coef_y = np.linalg.lstsq(design[keep], target_y[keep], rcond=None)[0]
        fit_x, fit_y = design @ correction.coef_x, design @ correction.coef_y
        residual = np.hypot(target_x - fit_x, target_y - fit_y)
        keep = residual <= outlier_mm
    correction.stats = {
        'events_used': int(keep.sum()),
        'events_rejected': int((~keep).sum()),
        'rms_before_mm': float(np.sqrt(np.mean((target_x - x)[keep] ** 2 + (target_y - y)[keep] ** 2))),
        'rms_after_mm': float(np.sqrt(np.mean(residual[keep] ** 2))),
    }
    return correction
