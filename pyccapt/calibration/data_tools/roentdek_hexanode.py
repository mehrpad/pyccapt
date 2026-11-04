"""RoentDek hexanode (HEX) position reconstruction from raw TDC8HP channels.

This is a vectorised NumPy port of the position calculation CoboldPC/DAn
performs for a hexanode detector, so ``det_x``/``det_y``/``tof`` can be
recomputed from the raw first-hit TDC values that pyccapt stores in the HDF5
``tdc`` group. It is shared by the live acquisition loop
(:mod:`pyccapt.control.tdc_roentdek.tdc_roentdek`) and offline re-processing,
so only NumPy is imported at module level.

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
    """Smooth 2-D polynomial map from analytic to CoboldPC positions (mm)."""

    degree: int
    coef_x: np.ndarray
    coef_y: np.ndarray
    scale_mm: float = 40.0
    stats: dict = field(default_factory=dict)

    def _design(self, x, y):
        xs = np.asarray(x, dtype=np.float64) / self.scale_mm
        ys = np.asarray(y, dtype=np.float64) / self.scale_mm
        return np.stack([xs ** i * ys ** j for i in range(self.degree + 1)
                         for j in range(self.degree + 1 - i)], axis=-1)

    def apply(self, x_mm, y_mm):
        design = self._design(x_mm, y_mm)
        return design @ self.coef_x, design @ self.coef_y

    def to_dict(self) -> dict:
        return {'degree': self.degree, 'coef_x': self.coef_x.tolist(), 'coef_y': self.coef_y.tolist(),
                'scale_mm': self.scale_mm, 'stats': self.stats}

    @classmethod
    def from_dict(cls, data: dict) -> 'HexanodeCorrection':
        return cls(int(data['degree']), np.asarray(data['coef_x'], dtype=np.float64),
                   np.asarray(data['coef_y'], dtype=np.float64), float(data.get('scale_mm', 40.0)),
                   dict(data.get('stats', {})))


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

    x = x + params.x_offset_mm
    y = y + params.y_offset_mm
    if params.rotation_deg:
        cx, cy = params.rotation_center_mm
        angle = np.deg2rad(params.rotation_deg)
        dx, dy = x - cx, y - cy
        x = cx + dx * np.cos(angle) - dy * np.sin(angle)
        y = cy + dx * np.sin(angle) + dy * np.cos(angle)
    if correction is not None and valid.any():
        x_corr, y_corr = correction.apply(x[valid], y[valid])
        x[valid], y[valid] = x_corr, y_corr

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


def load_roentdek_tdc_channels(path: str | Path) -> dict[str, np.ndarray]:
    """Load first-hit RoentDek channels from a pyccapt HDF5 file.

    Supports both layouts pyccapt has written:

    * legacy per-channel datasets ``tdc/ch0`` .. ``tdc/ch7``;
    * the flat ``tdc/channel`` + ``tdc/time_data`` table with 8 rows per event.
      ``time_data`` is stored as uint64; negative TDC bins (the start channel
      precedes the MCP trigger) are recovered by reinterpreting it as int64.

    Returns:
        ``channels`` as an ``(n, 8)`` int64 array plus any per-event
        ``start_counter``, ``high_voltage``, ``voltage_pulse`` and
        ``laser_pulse`` arrays found in the ``tdc`` group.
    """
    import h5py

    with h5py.File(path, 'r') as handle:
        if 'tdc/ch0' in handle:
            columns = [np.asarray(handle[f'tdc/ch{i}'][:]) for i in range(8)]
            channels = np.stack([_as_signed(column) for column in columns], axis=1)
            return {'channels': channels}
        if 'tdc/time_data' not in handle or 'tdc/channel' not in handle:
            raise KeyError(f'{path} has no RoentDek tdc channels (tdc/ch0..ch7 or tdc/time_data)')
        time_data = _as_signed(np.asarray(handle['tdc/time_data'][:]))
        channel = np.asarray(handle['tdc/channel'][:]).astype(np.int64)
        extras = {name: np.asarray(handle[f'tdc/{name}'][:])
                  for name in ('start_counter', 'high_voltage', 'voltage_pulse', 'laser_pulse')
                  if f'tdc/{name}' in handle}

    if len(channel) % 8 == 0 and np.array_equal(channel, np.tile(np.arange(8), len(channel) // 8)):
        result = {'channels': time_data.reshape(-1, 8)}
        result.update({name: values[::8] for name, values in extras.items()})
        return result

    if 'start_counter' not in extras:
        raise ValueError('flat tdc table is not 8 rows per event and has no start_counter to group by')
    start_counter = extras['start_counter']
    starts = np.r_[0, np.flatnonzero(np.diff(start_counter.astype(np.int64)) != 0) + 1]
    event = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(start_counter)]))
    channels = np.zeros((len(starts), 8), dtype=np.int64)
    keep = (channel >= 0) & (channel < 8)
    # Reverse order so the first value per (event, channel) wins the write.
    order = np.flatnonzero(keep)[::-1]
    channels[event[order], channel[order]] = time_data[order]
    result = {'channels': channels}
    result.update({name: values[starts] for name, values in extras.items()})
    return result


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
):
    """Rebuild ``x_det``/``y_det``/``t`` for every TDC event of an HDF5 file.

    Returns a DataFrame with one row per TDC event (the same order as the
    ``tdc`` group and as CoboldPC's ASCII export of the matching LMF):
    ``x_det (cm)``, ``y_det (cm)``, ``t (ns)``, ``valid``, ``layer_mask``,
    ``n_reconstructed`` and the per-event ``tdc`` metadata when present.
    Rows without a position get ``invalid_value`` for x, y and t (pass 0.0
    to match CoboldPC's convention for events with no particle).
    """
    import pandas as pd

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
