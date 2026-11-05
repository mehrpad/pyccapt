"""Surface Concept delay-line detector geometry (TDC bins -> detector cm).

The FPGA quadrupel finder (``control/tdc_surface_concept/tdc_gpx3.ini``)
accepts a stop pair only when ``TimeDif1Min <= t0 - t1 <= TimeDif1Max``
(x; ``TimeDif2*`` for y), and reports the hardware-binned pixel

    x_bin = floor((t0 - t1 - TimeDif1Min) / HardwareBinningX)

so the detector centre (``t0 = t1``) is pixel ``-TimeDif1Min / binning``
(1201 in x, 1190 in y), and the active window spans 2402 x 2380 pixels of
``pix_size_x`` x ``pix_size_y`` mm (0.033306 x 0.033613 mm, i.e. 80 mm).
Coordinates are reported at the pixel centre:

    x_cm = (x_bin + 0.5 - centre_x) * pix_size_x * 0.1

Files written before 2026-11-05 used ``(bin - 1225) * 80 / 2450 mm`` on
both axes instead (centre off by 0.8 / 1.15 mm, scale 2-3 % small). Those
values still encode the exact pixel, so :func:`legacy_cm_to_cm` converts
them losslessly. New files carry ``GEOMETRY_ATTR`` on ``/dld``.

This module only imports NumPy so the acquisition process can use it.
"""

from __future__ import annotations

import numpy as np

TIME_DIF_MIN_X = -2402
TIME_DIF_MIN_Y = -2380
BINNING = 2
PIXEL_SIZE_X_MM = 0.033306
PIXEL_SIZE_Y_MM = 0.033613

# Pre-2026-11-05 convention: 4900 bins over 80 mm, binning 2, centre 1225.
LEGACY_CENTRE_BIN = 4900 / 2 / 2
LEGACY_PIXEL_SIZE_MM = 80.0 / 4900 * 2

GEOMETRY_ATTR = 'sc_detector_geometry'
GEOMETRY_NAME = 'firmware_window_v1'


def geometry_from_config(conf=None) -> dict:
    """Geometry constants, with per-rig overrides from ``config.toml``.

    Keys read: ``sc_time_dif1_min``, ``sc_time_dif2_min``,
    ``sc_detector_binning_factor``, ``sc_pixel_size_x_mm``,
    ``sc_pixel_size_y_mm`` (keep them equal to ``tdc_gpx3.ini``).
    """
    def get(key, default):
        try:
            value = conf[key] if conf is not None else None
        except (KeyError, TypeError):
            value = None
        return default if value is None else value

    return {
        'time_dif_min_x': int(get('sc_time_dif1_min', TIME_DIF_MIN_X)),
        'time_dif_min_y': int(get('sc_time_dif2_min', TIME_DIF_MIN_Y)),
        'binning_factor': int(get('sc_detector_binning_factor', BINNING)),
        'pixel_size_x_mm': float(get('sc_pixel_size_x_mm', PIXEL_SIZE_X_MM)),
        'pixel_size_y_mm': float(get('sc_pixel_size_y_mm', PIXEL_SIZE_Y_MM)),
    }


DEFAULT_GEOMETRY = geometry_from_config()


def centre_bin(axis: str, geometry: dict | None = None) -> float:
    geometry = geometry or DEFAULT_GEOMETRY
    return -geometry[f'time_dif_min_{axis}'] / geometry['binning_factor']


def bins_to_cm(position_bin, axis: str, geometry: dict | None = None):
    """Firmware pixel (``dif1`` / ``dif2``) -> detector coordinate in cm."""
    geometry = geometry or DEFAULT_GEOMETRY
    value = (np.asarray(position_bin, dtype=np.float64) + 0.5 - centre_bin(axis, geometry))
    value = value * geometry[f'pixel_size_{axis}_mm'] * 0.1
    return float(value) if np.ndim(value) == 0 else value


def stops_to_bin(first_time, second_time, axis: str, geometry: dict | None = None):
    """Firmware pixel of one axis from its two raw stop times (unbinned TDC bins)."""
    geometry = geometry or DEFAULT_GEOMETRY
    difference = np.asarray(first_time, dtype=np.float64) - np.asarray(second_time, dtype=np.float64)
    return np.floor((difference - geometry[f'time_dif_min_{axis}']) / geometry['binning_factor'])


def stops_to_cm(first_time, second_time, axis: str, geometry: dict | None = None):
    """Detector coordinate (cm) from raw stops; identical to the dld value."""
    return bins_to_cm(stops_to_bin(first_time, second_time, axis, geometry), axis, geometry)


def legacy_cm_to_bin(value_cm):
    """Recover the firmware pixel from a pre-2026-11-05 coordinate."""
    return np.rint(np.asarray(value_cm, dtype=np.float64) * 10.0 / LEGACY_PIXEL_SIZE_MM + LEGACY_CENTRE_BIN)


def legacy_cm_to_cm(value_cm, axis: str, geometry: dict | None = None):
    """Convert a pre-2026-11-05 coordinate to the firmware geometry (exact)."""
    return bins_to_cm(legacy_cm_to_bin(value_cm), axis, geometry)


def looks_like_legacy_cm(x_cm, y_cm, *, sample: int = 100_000) -> bool:
    """True when every sampled x/y is an exact legacy pixel (bins 0..4095).

    Legacy Surface Concept coordinates are ``(bin - 1225) * 80 / 2450 mm``
    with an integer ``bin``; CoboldPC/RoentDek and processed coordinates are
    not on that grid.
    """
    x = np.asarray(x_cm, dtype=np.float64).reshape(-1)
    y = np.asarray(y_cm, dtype=np.float64).reshape(-1)
    if x.size == 0 or x.size != y.size:
        return False
    step = max(1, x.size // sample)
    values = np.concatenate([x[::step], y[::step]])
    values = values[np.isfinite(values)]
    if not np.any(values != 0):
        return False
    scaled = values * 10.0 / LEGACY_PIXEL_SIZE_MM + LEGACY_CENTRE_BIN
    on_grid = np.abs(scaled - np.rint(scaled)) < 1e-6
    in_range = (scaled > -0.5) & (scaled < 4096.5)
    return bool(np.all(on_grid & in_range))
