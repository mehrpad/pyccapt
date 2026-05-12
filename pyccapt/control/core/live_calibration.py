"""Real-time TOF / mass calibration for the live visualization GUI.

Pipeline (mirrors ``pyccapt.calibration``)
------------------------------------------
The fits and formulas here are deliberately the same as the offline
calibration submodule -- only the orchestration is different. The two
modes use the order documented in
``pyccapt.calibration.tutorials.tutorials_helpers.helper_calibration``
(the "Initial calibration" button handler):

* **TOF**: ``initial_calibration(t,V,x,y)`` -- a "naive" prescaling
  that brings every event to a common voltage and a common
  flight-path length -- followed by ``voltage_correction`` and
  ``bowl_correction`` polynomial fits.
* **MC**: a single ``bowl_correction`` is the initial step (because
  ``mc = 2eV(t/L)^2`` already accounts for V), followed by a
  ``voltage_correction`` and a final ``bowl_correction``.

The exact formulas, taken verbatim from the calibration submodule:

::

    # Initial calibration for TOF (diagnostics.initial_calibration):
    #   d^2 = (10*x_cm)^2 + (10*y_cm)^2 + L^2     # mm
    #   t1 = t * mean(d^2)/d^2 * sqrt(V/mean(V))

    # Voltage correction:
    #   ratio = t_peak / maximum_location  (sampled per event-window)
    #   ratio = a + b*V + c*V^2            (curve_fit)
    #   For TOF: t_v  = t1 / sqrt(ratio_at_event_V)
    #   For MC:  mc_v = mc / ratio_at_event_V       (no sqrt)

    # Bowl correction (per polar-cell sampling, then global poly fit):
    #   ratio = a + b*x + c*y + d*x^2 + e*x*y + f*y^2   (x, y in mm)
    #   t_b   = t_v / ratio_at_event_xy

Architecture
------------
Two phases, deliberately decoupled so the GUI thread never blocks:

1. **Fitting** (slow, infrequent, runs in a background ``QThread``).
   Every ``refit_interval_s`` seconds the worker:
   - takes a snapshot of the most recent events from the GUI's
     100 000-event ring buffer (under the buffer lock),
   - runs the same algorithmic steps as the offline calibration
     submodule but on numpy arrays only, no plotting, no Variables
     object,
   - emits the new ``CalibrationParams`` if the fit passes the
     R² gate, otherwise drops the result so the previous good
     parameters stay in effect.

2. **Applying** (fast, every render tick, runs in the GUI thread).
   :func:`apply_corrections` evaluates the cached polynomials on the
   raw event arrays in vectorized NumPy. ~1 ms per 100 000 events.

EMA smoothing
-------------
A naïve "swap in the freshly-fitted params" pattern produces visible
peak jitter on the live spectrum because each ~10 s refit lands on a
slightly different ``a, b, c, ...`` due to event-pool noise. We keep
the last ``ema_window`` (default 3) successful parameter sets in a
deque and the *applied* parameters are the arithmetic mean across the
window. Equivalent to a length-N moving average; damps jitter while
keeping the response fast (after N successful refits the window is
full and the displayed spectrum tracks the latest data again).

Trade-offs (be honest with operators)
-------------------------------------
* The displayed spectrum reflects parameters that are at most
  ``refit_interval_s`` * ``ema_window`` seconds stale. Fine for live
  monitoring, not for analysis -- post-experiment offline calibration
  using the full multi-peak iterative pipeline still produces the
  authoritative results.
* The live fit is **single-peak** (the dominant peak in a
  configurable mass window). The offline pipeline uses multi-peak
  joint fits which are markedly more accurate but markedly slower.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, fields
from typing import Callable, Deque, Optional

import numpy as np
from PyQt6 import QtCore
from scipy.optimize import curve_fit

from pyccapt.calibration.core.correction_models import (
    bowl_corr,
    voltage_corr,
)
from pyccapt.control.core import tof2mc_simple

_log = logging.getLogger("pyccapt.live_cal")


# ---------------------------------------------------------------- params


@dataclass(frozen=True)
class CalibrationParams:
    """Cached calibration coefficients used by the apply path.

    The fields mirror the calibration submodule's two model functions:

    * ``voltage_corr(V, a, b, c)``   -> 3 params
    * ``bowl_corr([x, y], a..f)``    -> 6 params per bowl stage

    Mode-specific ordering:

    * For ``mode == 'tof'``: only ``vol`` and ``bowl_final`` are used;
      ``bowl_init`` is set to identity (1, 0, 0, 0, 0, 0).
    * For ``mode == 'mc'``: all three stages are populated; the apply
      path divides by ``bowl_init`` first, then by ``vol``, then by
      ``bowl_final`` (this is the exact order the offline pipeline
      uses for MC).
    """

    mode: str  # 'tof' or 'mc'
    t_0: float  # ns
    flight_path_length_mm: float
    mean_d2_mm2: float  # for the TOF initial step
    mean_v: float  # for the TOF initial step
    # Voltage poly: 3 floats (a, b, c)  s.t. ratio = a + b*V + c*V^2
    vol: tuple  # (a, b, c)
    # Bowl polys: 6 floats each
    bowl_init: tuple  # MC only -- identity for TOF
    bowl_final: tuple
    # Diagnostics (carried for the GUI status label and EMA decisions)
    maximum_location: float  # peak location after the previous step
    mc_scale: float  # so the peak lands at target_mass
    fit_quality: float  # R^2 of the voltage stage
    timestamp: float
    num_events_used: int


_BOWL_IDENTITY = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_VOLT_IDENTITY = (1.0, 0.0, 0.0)


# ---------------------------------------------------------------- apply


def _voltage_factor(v: np.ndarray, params_vol: tuple) -> np.ndarray:
    a, b, c = params_vol
    return voltage_corr(v, a, b, c)


def _bowl_factor(x_mm: np.ndarray, y_mm: np.ndarray, params_bowl: tuple) -> np.ndarray:
    # bowl_corr expects [x, y] sequence -- pass as a 2-tuple of arrays.
    return bowl_corr([x_mm, y_mm], *params_bowl)


def apply_corrections(
    t_ns: np.ndarray,
    v_dc: np.ndarray,
    x_cm: np.ndarray,
    y_cm: np.ndarray,
    params: Optional[CalibrationParams],
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Apply the cached calibration to raw events.

    Vectorized; runs on the GUI thread on every refresh tick. Returns
    ``(t_corrected_ns, mc_corrected_Da)`` or ``None`` when ``params``
    is missing or any intermediate value is non-finite (so the caller
    transparently falls back to the uncalibrated raw path).
    """
    if params is None:
        return None
    if t_ns.size == 0:
        return t_ns.copy(), np.empty(0, dtype=np.float64)

    t = np.asarray(t_ns, dtype=np.float64)
    v = np.asarray(v_dc, dtype=np.float64)
    # Bowl model fits use mm (the offline pipeline multiplies cm * 10
    # at the start of bowl_correction_main); we do the same here.
    x_mm = np.asarray(x_cm, dtype=np.float64) * 10.0
    y_mm = np.asarray(y_cm, dtype=np.float64) * 10.0

    mode = params.mode

    if mode == 'tof':
        # Step 1: initial calibration -- bring every event to a common
        # voltage and common flight-path length. Mirrors
        # diagnostics.initial_calibration verbatim.
        d2 = x_mm * x_mm + y_mm * y_mm + params.flight_path_length_mm**2
        init_flight_factor = params.mean_d2_mm2 / d2
        mean_v = params.mean_v if params.mean_v > 0 else float(np.mean(v))
        init_voltage_factor = np.sqrt(np.maximum(v / mean_v, 0.0))
        t1 = t * init_flight_factor * init_voltage_factor

        # Step 2: voltage correction. Same as the offline pipeline:
        #   correction = sqrt(f_v); t_v = t1 / correction
        f_v = _voltage_factor(v, params.vol)
        if not np.all(np.isfinite(f_v)) or np.any(f_v <= 0):
            return None
        t_v = t1 / np.sqrt(f_v)

        # Step 3: bowl correction. ratio = bowl_corr(x_mm, y_mm, ...)
        # t_b = t_v / ratio.
        f_bowl = _bowl_factor(x_mm, y_mm, params.bowl_final)
        if not np.all(np.isfinite(f_bowl)) or np.any(f_bowl <= 0):
            return None
        t_corr = t_v / f_bowl

        # Convert to mc *only* with the simple formula -- the GUI MC
        # view in TOF-calibrated mode shows mc derived from the
        # corrected t. mc_scale lets us anchor the dominant peak at
        # the configured target mass.
        mc_corr = (
            tof2mc_simple.tof_2_mc(
                t_corr,
                params.t_0,
                v,
                x_cm,
                y_cm,
                flightPathLength=params.flight_path_length_mm,
            )
            * params.mc_scale
        )
        return t_corr, mc_corr

    if mode == 'mc':
        # MC has no naive prescaling -- mc = 2eV(t/L)^2 already accounts
        # for V. The first stage in the offline MC pipeline is
        # bowl_correction (acting on raw mc).
        mc = tof2mc_simple.tof_2_mc(
            t,
            params.t_0,
            v,
            x_cm,
            y_cm,
            flightPathLength=params.flight_path_length_mm,
        )
        f_bowl_init = _bowl_factor(x_mm, y_mm, params.bowl_init)
        if not np.all(np.isfinite(f_bowl_init)) or np.any(f_bowl_init <= 0):
            return None
        mc1 = mc / f_bowl_init

        # Voltage correction. For MC the offline code divides by f_v
        # directly (NO sqrt -- that's the TOF convention because t ~
        # 1/sqrt(V) but mc is linear in V).
        f_v = _voltage_factor(v, params.vol)
        if not np.all(np.isfinite(f_v)) or np.any(f_v <= 0):
            return None
        mc2 = mc1 / f_v

        # Final bowl correction.
        f_bowl_final = _bowl_factor(x_mm, y_mm, params.bowl_final)
        if not np.all(np.isfinite(f_bowl_final)) or np.any(f_bowl_final <= 0):
            return None
        mc3 = mc2 / f_bowl_final * params.mc_scale
        return t, mc3

    return None


# ---------------------------------------------------------------- helpers


def _polar_cells_iter(radial: np.ndarray, sample_size: float, det_diam_mm: float):
    """Yield (r_min, r_max, th_min, th_max) cells covering the detector.

    Mirrors ``calibration._iter_polar_cells``: ring count grows with
    the detector radius, and each ring is split into 8..96 equal
    angular sectors.
    """
    r_max_data = float(np.max(radial)) if radial.size else 0.0
    r_max_detector = max(0.0, float(det_diam_mm) / 2.0)
    r_max = max(r_max_data, r_max_detector, float(sample_size))
    n_rings = max(1, int(np.ceil(r_max / float(sample_size))))
    ring_edges = np.linspace(0.0, r_max, n_rings + 1)
    for r_idx in range(n_rings):
        r_min_i = float(ring_edges[r_idx])
        r_max_i = float(ring_edges[r_idx + 1])
        r_mid = 0.5 * (r_min_i + r_max_i)
        circumference = 2.0 * np.pi * max(r_mid, 0.5 * float(sample_size))
        n_sectors = int(np.clip(np.ceil(circumference / float(sample_size)), 8, 96))
        for theta_idx in range(n_sectors):
            theta_min = float(theta_idx * 2.0 * np.pi / n_sectors)
            theta_max = float((theta_idx + 1) * 2.0 * np.pi / n_sectors)
            yield r_min_i, r_max_i, theta_min, theta_max


def _collect_polar_samples(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    t_or_mc: np.ndarray,
    v_dc: np.ndarray,
    maximum_location: float,
    sample_size_mm: float,
    det_diam_mm: float,
) -> dict[str, np.ndarray]:
    """Per-tile peak-position sampling, identical in spirit to the offline
    ``_collect_spatial_samples(sampling_mode='polar')`` -- one sample
    per non-empty polar cell, using the cell's mean ``t/maximum_location``.
    Returns dict with arrays ``x``, ``y``, ``t`` (=ratio), ``v``.
    """
    if x_mm.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return {"x": empty, "y": empty, "t": empty, "v": empty}

    radial = np.hypot(x_mm, y_mm)
    angle = np.mod(np.arctan2(y_mm, x_mm), 2.0 * np.pi)

    xs, ys, ts, vs = [], [], [], []
    for r_lo, r_hi, th_lo, th_hi in _polar_cells_iter(radial, sample_size_mm, det_diam_mm):
        mask = (radial >= r_lo) & (radial < r_hi) & (angle >= th_lo) & (angle < th_hi)
        count = int(mask.sum())
        if count < 5:
            continue
        cell_t = t_or_mc[mask]
        # Use the mean of the cell as the cell-peak proxy. This matches
        # ``_cell_peak_value`` with sample_range_max='mean', which is the
        # default in the live path.
        ratio = float(np.mean(cell_t)) / maximum_location
        if not np.isfinite(ratio) or ratio <= 0:
            continue
        xs.append(float(np.median(x_mm[mask])))
        ys.append(float(np.median(y_mm[mask])))
        ts.append(ratio)
        vs.append(float(np.mean(v_dc[mask])))

    return {
        "x": np.asarray(xs, dtype=np.float64),
        "y": np.asarray(ys, dtype=np.float64),
        "t": np.asarray(ts, dtype=np.float64),
        "v": np.asarray(vs, dtype=np.float64),
    }


def _voltage_window_samples(
    v: np.ndarray,
    t_or_mc: np.ndarray,
    maximum_location: float,
    sample_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin events into ``sample_size``-event windows, return (V_mean, t/max).

    Mirrors ``voltage_correction(mode='ion_seq', sample_range_max='mean')``
    from the offline pipeline.
    """
    n = v.size
    if n == 0 or sample_size <= 0:
        return np.empty(0), np.empty(0)
    v_means, ratios = [], []
    for i in range(int(np.ceil(n / sample_size))):
        v_chunk = v[i * sample_size : (i + 1) * sample_size]
        t_chunk = t_or_mc[i * sample_size : (i + 1) * sample_size]
        if v_chunk.size < 5:
            continue
        v_means.append(float(np.mean(v_chunk)))
        ratios.append(float(np.mean(t_chunk)) / maximum_location)
    return np.asarray(v_means, dtype=np.float64), np.asarray(ratios, dtype=np.float64)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _fit_voltage_polynomial(
    v: np.ndarray,
    t_or_mc: np.ndarray,
    maximum_location: float,
    sample_size: int,
) -> tuple[Optional[tuple[float, float, float]], float]:
    v_means, ratios = _voltage_window_samples(v, t_or_mc, maximum_location, sample_size)
    if v_means.size < 5:
        return None, 0.0
    if float(np.max(v_means) - np.min(v_means)) < 50.0:
        return None, 0.0
    try:
        coeffs, _ = curve_fit(voltage_corr, v_means, ratios)
    except Exception:
        return None, 0.0
    a, b, c = (float(v) for v in coeffs)
    pred = voltage_corr(v_means, a, b, c)
    return (a, b, c), _r2(ratios, pred)


def _fit_bowl_polynomial(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    t_or_mc: np.ndarray,
    v: np.ndarray,
    maximum_location: float,
    sample_size_mm: float,
    det_diam_mm: float,
) -> Optional[tuple[float, float, float, float, float, float]]:
    samples = _collect_polar_samples(
        x_mm,
        y_mm,
        t_or_mc,
        v,
        maximum_location,
        sample_size_mm,
        det_diam_mm,
    )
    if samples["x"].size < 6:
        return None
    try:
        coeffs, _ = curve_fit(bowl_corr, [samples["x"], samples["y"]], samples["t"])
    except Exception:
        return None
    if coeffs.size != 6 or not np.all(np.isfinite(coeffs)):
        return None
    return tuple(float(c) for c in coeffs)


# ---------------------------------------------------------------- EMA


def _mean_params(window: Deque[CalibrationParams]) -> Optional[CalibrationParams]:
    """Return the arithmetic mean of every numeric field in ``window``.

    Non-numeric fields (``mode``, ``timestamp``, ``num_events_used``)
    take the value from the most recent entry. ``vol``, ``bowl_init``,
    ``bowl_final`` are tuples and are averaged element-wise.
    """
    if not window:
        return None
    if len(window) == 1:
        return window[0]
    latest = window[-1]
    avg_kwargs = {}
    # Tuple fields -- average element-wise.
    tuple_fields = {"vol", "bowl_init", "bowl_final"}
    # Pure-numeric scalar fields -- arithmetic mean.
    scalar_fields = {
        "t_0",
        "flight_path_length_mm",
        "mean_d2_mm2",
        "mean_v",
        "maximum_location",
        "mc_scale",
        "fit_quality",
    }
    for f in fields(CalibrationParams):
        if f.name in tuple_fields:
            stacked = np.array([getattr(p, f.name) for p in window], dtype=np.float64)
            avg_kwargs[f.name] = tuple(float(v) for v in np.mean(stacked, axis=0))
        elif f.name in scalar_fields:
            avg_kwargs[f.name] = float(np.mean([getattr(p, f.name) for p in window]))
        else:
            avg_kwargs[f.name] = getattr(latest, f.name)
    return CalibrationParams(**avg_kwargs)


# ---------------------------------------------------------------- worker


class LiveCalibrationWorker(QtCore.QThread):
    """Background fitter for the live spectrum.

    Owns no GUI widgets. Emits ``parameters_updated(CalibrationParams)``
    with the EMA-smoothed parameters whenever a fit passes the quality
    gate, and ``status_changed(str)`` with human-readable progress text.
    """

    parameters_updated = QtCore.pyqtSignal(object)
    status_changed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        snapshot_callback: Callable[[], Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]],
        conf: dict,
    ) -> None:
        super().__init__()
        self._snapshot = snapshot_callback
        self._conf = conf
        self._stop_event = threading.Event()
        # Cadence + thresholds (overridable via config.toml).
        self._refit_interval_s = float(conf.get("live_calibration_refit_interval_s", 15.0))
        self._min_events = int(conf.get("live_calibration_min_events", 5000))
        self._min_fit_quality = float(conf.get("live_calibration_min_fit_quality", 0.7))
        self._max_consecutive_failures = max(
            1, int(conf.get("live_calibration_max_consecutive_failures", 5))
        )
        self._consecutive_failures = 0
        self._mass_window_min = float(conf.get("live_calibration_mass_window_min", 1.0))
        self._mass_window_max = float(conf.get("live_calibration_mass_window_max", 60.0))
        self._target_mass = conf.get("live_calibration_target_mass")
        self._mode = str(conf.get("live_calibration_mode", "tof")).strip().lower()
        if self._mode not in ("tof", "mc"):
            self._mode = "tof"
        self._bin_size_da = float(conf.get("bin_size", 0.1))
        # Voltage-bin chunk size (events per ion_seq window in the offline pipeline).
        self._volt_sample_size = int(conf.get("live_calibration_volt_sample_size", 500))
        # Polar-tile sample size in mm (matches the offline default range).
        self._bowl_sample_size_mm = float(conf.get("live_calibration_bowl_sample_size_mm", 4.0))
        # EMA window -- average the last N successful fits.
        self._ema_window_size = max(1, int(conf.get("live_calibration_ema_window", 3)))
        self._ema_window: Deque[CalibrationParams] = deque(maxlen=self._ema_window_size)

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        _log.info(
            "Live calibration worker started (mode=%s, refit %.1f s, min %d events, min R²=%.2f, EMA=%d)",
            self._mode,
            self._refit_interval_s,
            self._min_events,
            self._min_fit_quality,
            self._ema_window_size,
        )
        while not self._stop_event.is_set():
            # Sleep first so the very first tick has events to look at.
            deadline = time.time() + self._refit_interval_s
            while time.time() < deadline:
                if self._stop_event.is_set():
                    return
                self.msleep(200)
            try:
                fit = self._refit_once()
            except Exception as exc:
                _log.exception("Live calibration refit raised: %s", exc)
                self.status_changed.emit(f"refit error: {exc.__class__.__name__}")
                fit = None
            if fit is None:
                # The previous params (if any) might have been borderline
                # and are now visibly destroying peak structure in the
                # GUI. After enough consecutive failures, throw them
                # out so the spectrum falls back to the raw view rather
                # than continuing to display a stale bad calibration.
                self._consecutive_failures += 1
                if (
                    self._consecutive_failures >= self._max_consecutive_failures
                    and self._ema_window
                ):
                    self._ema_window.clear()
                    self.status_changed.emit(
                        f"no good fit in {self._consecutive_failures} attempts; "
                        "falling back to raw"
                    )
                    self.parameters_updated.emit(None)
                continue
            # Push the freshly fitted parameters into the EMA window and
            # emit the *averaged* parameters so the displayed spectrum
            # updates smoothly across successive refits.
            self._consecutive_failures = 0
            self._ema_window.append(fit)
            averaged = _mean_params(self._ema_window)
            if averaged is not None:
                self.parameters_updated.emit(averaged)

    # ------------------------------------------------------------------
    # One refit pass.
    # ------------------------------------------------------------------

    def _refit_once(self) -> Optional[CalibrationParams]:
        snapshot = self._snapshot()
        if snapshot is None:
            self.status_changed.emit("waiting for events…")
            return None
        t_ns, v_dc, x_cm, y_cm = snapshot
        if t_ns.size < self._min_events:
            self.status_changed.emit(f"calibrating… ({t_ns.size}/{self._min_events} events)")
            return None

        t_0 = float(
            self._conf.get(
                "t_0_voltage" if not self._conf.get("_active_pulse_mode_is_laser") else "t_0_laser",
                0.0,
            )
        )
        L_mm = float(self._conf["flight_path_length"])
        det_diam_mm = float(self._conf.get("detector_diameter", 80.0))
        x_mm = x_cm.astype(np.float64) * 10.0
        y_mm = y_cm.astype(np.float64) * 10.0

        if self._mode == "tof":
            return self._refit_tof(t_ns, v_dc, x_cm, y_cm, x_mm, y_mm, t_0, L_mm, det_diam_mm)
        return self._refit_mc(t_ns, v_dc, x_cm, y_cm, x_mm, y_mm, t_0, L_mm, det_diam_mm)

    # ----- TOF -------------------------------------------------------------

    def _refit_tof(self, t, v, x_cm, y_cm, x_mm, y_mm, t_0, L_mm, det_diam_mm):
        # Step 1: initial calibration -- exactly as
        # diagnostics.initial_calibration does it.
        d2 = x_mm * x_mm + y_mm * y_mm + L_mm**2
        mean_d2 = float(np.mean(d2))
        mean_v = float(np.mean(v))
        init_flight = mean_d2 / d2
        init_volt = np.sqrt(np.maximum(v / mean_v, 0.0))
        t1 = t * init_flight * init_volt

        # Find the dominant peak in the configured mass window. We
        # convert t1 to mc only to locate the peak in mass space, then
        # work in t1 (ns) for the actual fits -- the offline pipeline
        # does the equivalent via variables.dld_t_calib.
        mc1 = tof2mc_simple.tof_2_mc(
            t1,
            t_0,
            v,
            x_cm,
            y_cm,
            flightPathLength=L_mm,
        )
        peak_mass = self._dominant_peak_mass(mc1)
        if peak_mass is None:
            self.status_changed.emit("no clear peak in mass window")
            return None
        mc_mask = (mc1 >= peak_mass * 0.95) & (mc1 <= peak_mass * 1.05)
        if mc_mask.sum() < max(self._volt_sample_size, 200):
            self.status_changed.emit("not enough events at the dominant peak")
            return None
        t_peak = t1[mc_mask]
        v_peak = v[mc_mask]
        x_mm_peak = x_mm[mc_mask]
        y_mm_peak = y_mm[mc_mask]

        # Step 2: voltage correction. maximum_location = peak position
        # in t1; ratio = mean(t_chunk) / maximum_location.
        maximum_location_t = float(np.median(t_peak))
        if maximum_location_t <= 0:
            self.status_changed.emit("invalid peak location for voltage fit")
            return None
        vol_fit, r2 = _fit_voltage_polynomial(
            v_peak,
            t_peak,
            maximum_location_t,
            self._volt_sample_size,
        )
        if vol_fit is None or r2 < self._min_fit_quality:
            self.status_changed.emit(f"voltage fit weak (R²={r2:.2f}); keeping previous params")
            return None

        # Apply voltage correction to ALL events at the peak so the
        # bowl fit sees voltage-flat data, exactly as the offline
        # pipeline does (vol corr is run before bowl corr).
        f_v_peak = voltage_corr(v_peak, *vol_fit)
        t_v_peak = t_peak / np.sqrt(np.maximum(f_v_peak, 1e-30))

        # Step 3: bowl correction. maximum_location = peak position
        # after voltage correction.
        maximum_location_t_v = float(np.median(t_v_peak))
        if maximum_location_t_v <= 0:
            self.status_changed.emit("invalid peak location for bowl fit")
            return None
        bowl_fit = _fit_bowl_polynomial(
            x_mm_peak,
            y_mm_peak,
            t_v_peak,
            v_peak,
            maximum_location_t_v,
            self._bowl_sample_size_mm,
            det_diam_mm,
        )
        if bowl_fit is None:
            bowl_fit = _BOWL_IDENTITY  # voltage-only is still useful

        # Initial mass scale.
        target = self._target_mass if self._target_mass is not None else peak_mass
        try:
            target = float(target)
        except (TypeError, ValueError):
            target = peak_mass
        mc_scale = target / peak_mass if peak_mass > 0 else 1.0

        params = CalibrationParams(
            mode="tof",
            t_0=t_0,
            flight_path_length_mm=L_mm,
            mean_d2_mm2=mean_d2,
            mean_v=mean_v,
            vol=vol_fit,
            bowl_init=_BOWL_IDENTITY,
            bowl_final=bowl_fit,
            maximum_location=maximum_location_t_v,
            mc_scale=mc_scale,
            fit_quality=r2,
            timestamp=time.time(),
            num_events_used=int(t.size),
        )
        self.status_changed.emit(f"tof: peak {peak_mass:.2f} Da → {target:.2f} Da, R²={r2:.2f}, n={t.size}")
        return params

    # ----- MC --------------------------------------------------------------

    def _refit_mc(self, t, v, x_cm, y_cm, x_mm, y_mm, t_0, L_mm, det_diam_mm):
        # Step 0: raw MC.
        mc = tof2mc_simple.tof_2_mc(
            t,
            t_0,
            v,
            x_cm,
            y_cm,
            flightPathLength=L_mm,
        )
        peak_mass = self._dominant_peak_mass(mc)
        if peak_mass is None:
            self.status_changed.emit("no clear peak in mass window")
            return None
        mc_mask = (mc >= peak_mass * 0.95) & (mc <= peak_mass * 1.05)
        if mc_mask.sum() < max(self._volt_sample_size, 200):
            self.status_changed.emit("not enough events at the dominant peak")
            return None
        mc_peak = mc[mc_mask]
        v_peak = v[mc_mask]
        x_mm_peak = x_mm[mc_mask]
        y_mm_peak = y_mm[mc_mask]

        # Step 1: initial bowl correction (acts on raw mc, in MC mode this
        # replaces the TOF "naive prescaling").
        maximum_location_mc = float(np.median(mc_peak))
        if maximum_location_mc <= 0:
            self.status_changed.emit("invalid peak location for initial bowl fit")
            return None
        bowl_init = _fit_bowl_polynomial(
            x_mm_peak,
            y_mm_peak,
            mc_peak,
            v_peak,
            maximum_location_mc,
            self._bowl_sample_size_mm,
            det_diam_mm,
        )
        if bowl_init is None:
            bowl_init = _BOWL_IDENTITY
        f_b_init_peak = bowl_corr([x_mm_peak, y_mm_peak], *bowl_init)
        mc1_peak = mc_peak / np.maximum(f_b_init_peak, 1e-30)

        # Step 2: voltage correction (NO sqrt for MC).
        maximum_location_mc1 = float(np.median(mc1_peak))
        vol_fit, r2 = _fit_voltage_polynomial(
            v_peak,
            mc1_peak,
            maximum_location_mc1,
            self._volt_sample_size,
        )
        if vol_fit is None or r2 < self._min_fit_quality:
            self.status_changed.emit(f"voltage fit weak (R²={r2:.2f}); keeping previous params")
            return None
        f_v_peak = voltage_corr(v_peak, *vol_fit)
        mc2_peak = mc1_peak / np.maximum(f_v_peak, 1e-30)

        # Step 3: final bowl correction.
        maximum_location_mc2 = float(np.median(mc2_peak))
        bowl_final = _fit_bowl_polynomial(
            x_mm_peak,
            y_mm_peak,
            mc2_peak,
            v_peak,
            maximum_location_mc2,
            self._bowl_sample_size_mm,
            det_diam_mm,
        )
        if bowl_final is None:
            bowl_final = _BOWL_IDENTITY

        target = self._target_mass if self._target_mass is not None else peak_mass
        try:
            target = float(target)
        except (TypeError, ValueError):
            target = peak_mass
        mc_scale = target / peak_mass if peak_mass > 0 else 1.0

        params = CalibrationParams(
            mode="mc",
            t_0=t_0,
            flight_path_length_mm=L_mm,
            mean_d2_mm2=0.0,  # unused in MC apply path
            mean_v=0.0,  # unused in MC apply path
            vol=vol_fit,
            bowl_init=bowl_init,
            bowl_final=bowl_final,
            maximum_location=maximum_location_mc2,
            mc_scale=mc_scale,
            fit_quality=r2,
            timestamp=time.time(),
            num_events_used=int(t.size),
        )
        self.status_changed.emit(f"mc: peak {peak_mass:.2f} Da → {target:.2f} Da, R²={r2:.2f}, n={t.size}")
        return params

    # ----- shared ----------------------------------------------------------

    def _dominant_peak_mass(self, mc_arr: np.ndarray) -> Optional[float]:
        mask = (mc_arr >= self._mass_window_min) & (mc_arr <= self._mass_window_max)
        if mask.sum() < 200:
            return None
        bins = np.arange(
            self._mass_window_min,
            self._mass_window_max + self._bin_size_da,
            self._bin_size_da,
        )
        hist, edges = np.histogram(mc_arr[mask], bins=bins)
        if hist.max() < 50:
            return None
        idx = int(np.argmax(hist))
        return float(0.5 * (edges[idx] + edges[idx + 1]))
