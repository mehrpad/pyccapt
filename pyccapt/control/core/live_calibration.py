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
	bowl_corr_radial,
    voltage_corr,
)
from pyccapt.control.core import tof2mc_simple

_log = logging.getLogger("pyccapt.live_cal")


# ---------------------------------------------------------------- params


@dataclass(frozen=True)
class CalibrationParams:
    """Cached calibration coefficients used by the apply path.

    The fields mirror the calibration submodule's two model functions:

    * ``voltage_corr(V, a, b, c)``        -> 3 params
    * ``bowl_corr_radial([x, y], a..e)``  -> 5 params (radial bowl)
      (the offline ``sampling_mode='polar'`` default: a + b*r^2 + c*r^4
      + d*x + e*y, which stays well-behaved across the full detector
      instead of blowing up at the rim like the free Cartesian model).

    Both modes follow the offline pipeline: an INITIAL step, then ONE
    voltage correction, then ONE bowl correction.

    * ``mode == 'tof'``: initial = naive prescale (mean_d2 / mean_v carry
      it), then ``t /= sqrt(vol)`` then ``t /= bowl``.
    * ``mode == 'mc'``: initial = ``tof_2_mc`` (uses t_0 + per-event V and
      flight path, so the peak already sits near the true mass), then
      ``mc /= vol`` (no sqrt) then ``mc /= bowl``.
    """

    mode: str  # 'tof' or 'mc'
    t_0: float  # ns
    flight_path_length_mm: float
    mean_d2_mm2: float  # for the TOF initial step
    mean_v: float  # for the TOF initial step
    # Voltage poly: 3 floats (a, b, c)  s.t. ratio = a + b*V + c*V^2
    vol: tuple  # (a, b, c)
    # Radial bowl poly: 5 floats (a, b, c, d, e)
    bowl_final: tuple
    # Diagnostics (carried for the GUI status label and EMA decisions)
    maximum_location: float  # peak location after the previous step
    mc_scale: float  # so the peak lands at target_mass
    fit_quality: float  # R^2 of the voltage stage
    timestamp: float
    num_events_used: int


# Radial bowl identity: a=1, all higher terms 0 -> factor == 1 everywhere.
_BOWL_IDENTITY = (1.0, 0.0, 0.0, 0.0, 0.0)
_VOLT_IDENTITY = (1.0, 0.0, 0.0)

# Sane band for any applied correction factor. Real voltage/bowl
# corrections sit within a few percent of 1.0; clamping to this band is a
# safety net so a bad-but-finite fit can never collapse the mass axis
# toward 0 (the symptom this guards against). Also used as the bowl
# surface-sanity band when accepting/rejecting a fit.
_FACTOR_MIN = 0.5
_FACTOR_MAX = 2.0


# ---------------------------------------------------------------- apply


def _voltage_factor(v: np.ndarray, params_vol: tuple) -> np.ndarray:
    a, b, c = params_vol
    return voltage_corr(v, a, b, c)


def _bowl_factor(x_mm: np.ndarray, y_mm: np.ndarray, params_bowl: tuple) -> np.ndarray:
	# Radial-dominant bowl model (matches the offline polar default).
	return bowl_corr_radial([x_mm, y_mm], *params_bowl)


def _clamp_factor(f: np.ndarray) -> Optional[np.ndarray]:
	"""Clamp a correction-factor array to the sane band.

	Returns None if any value is non-finite (caller falls back to the raw
	path); otherwise clips into ``[_FACTOR_MIN, _FACTOR_MAX]`` so a few
	rim events with an extrapolated factor can neither veto the whole
	batch nor shrink the mass toward 0.
	"""
	if not np.all(np.isfinite(f)):
		return None
	return np.clip(f, _FACTOR_MIN, _FACTOR_MAX)


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
        f_v = _clamp_factor(_voltage_factor(v, params.vol))
        if f_v is None:
            return None
        t_v = t1 / np.sqrt(f_v)

        # Step 3: bowl correction. ratio = bowl_corr_radial(x_mm, y_mm, ...)
        # t_b = t_v / ratio.
        f_bowl = _clamp_factor(_bowl_factor(x_mm, y_mm, params.bowl_final))
        if f_bowl is None:
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
	    # Initial MC calibration = the physics conversion, which already
	    # accounts for V (the 2eV factor) and per-event geometry/flight
	    # path, and uses t_0 so the peak lands near the true mass. Then
	    # ONE voltage + ONE bowl correction refine it (offline MC pipeline).
        mc = tof2mc_simple.tof_2_mc(
            t,
            params.t_0,
            v,
            x_cm,
            y_cm,
            flightPathLength=params.flight_path_length_mm,
        )
        # Voltage correction. For MC the offline code divides by f_v
        # directly (NO sqrt -- that's the TOF convention because t ~
        # 1/sqrt(V) but mc is linear in V).
	    f_v = _clamp_factor(_voltage_factor(v, params.vol))
	    if f_v is None:
            return None
	    mc_v = mc / f_v

	    # Bowl correction on the voltage-corrected mc.
	    f_bowl = _clamp_factor(_bowl_factor(x_mm, y_mm, params.bowl_final))
	    if f_bowl is None:
            return None
	    mc_corr = mc_v / f_bowl * params.mc_scale
	    return t, mc_corr

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
        # Cap iterations so a pathological input (e.g. all events at
        # the same voltage during a vdc_hold) cannot stall the live
        # calibration thread for the seconds-to-minutes it takes
        # scipy to give up. The QThread polls ``_stop_event`` only
        # between fits, so an unbounded ``maxfev`` blocks shutdown.
        coeffs, _ = curve_fit(voltage_corr, v_means, ratios, maxfev=2000)
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
) -> Optional[tuple[float, float, float, float, float]]:
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
	    # Radial-dominant model (offline polar default). Unlike the free
	    # Cartesian quadratic, this does not extrapolate to huge factors
	    # at the detector rim when fit only on the peak-ion footprint.
        coeffs, _ = curve_fit(
	        bowl_corr_radial, [samples["x"], samples["y"]], samples["t"], maxfev=2000
        )
    except Exception:
        return None
    if coeffs.size != 5 or not np.all(np.isfinite(coeffs)):
        return None
    return tuple(float(c) for c in coeffs)


def _bowl_surface_ok(bowl: tuple, det_diam_mm: float) -> bool:
	"""True if the bowl correction stays in the sane band across the detector.

	The fit is constrained near 1.0 only inside the peak-ion footprint;
	this evaluates it on a coarse grid over the full detector disk and
	rejects it (-> caller uses identity) if the factor leaves
	``[_FACTOR_MIN, _FACTOR_MAX]`` anywhere, the way the collapse arose.
	"""
	r = max(float(det_diam_mm) / 2.0, 1.0)
	g = np.linspace(-r, r, 15)
	gx, gy = np.meshgrid(g, g)
	disk = (gx * gx + gy * gy) <= r * r
	f = bowl_corr_radial([gx[disk], gy[disk]], *bowl)
	return bool(np.all(np.isfinite(f)) and f.min() >= _FACTOR_MIN and f.max() <= _FACTOR_MAX)


# ---------------------------------------------------------------- EMA


def _mean_params(window: Deque[CalibrationParams]) -> Optional[CalibrationParams]:
	"""EMA-smooth the SCALAR fields; keep the latest polynomial tuples.

	Averaging polynomial *coefficients* across fits done on different
	event pools / peak medians is not the same as averaging the
	correction *surfaces* and biases the divisor away from 1 (a key
	contributor to the old MC-cal collapse). So ``vol`` and ``bowl_final``
	take the most-recent fit verbatim; only the scalar anchors (peak
	location, mass scale, etc.) are smoothed.
	"""
    if not window:
        return None
    if len(window) == 1:
        return window[0]
    latest = window[-1]
    avg_kwargs = {}
	# Polynomial-coefficient tuples are NOT averaged (see docstring) -- they
	# take the latest fit via the else-branch below.
	tuple_fields: set = set()
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
		    event_count_callback: Optional[Callable[[], int]] = None,
    ) -> None:
        super().__init__()
        self._snapshot = snapshot_callback
        self._conf = conf
        self._stop_event = threading.Event()
        # Cadence. Preferred: re-fit every N NEW events (event_count_callback
        # returns the GUI's running ion count). Falls back to the legacy
        # time interval when no event counter is supplied.
        self._event_count = event_count_callback
        self._refit_event_interval = max(1, int(conf.get("live_calibration_refit_event_interval", 1000)))
        self._last_refit_count = 0
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
	        "Live calibration worker started (mode=%s, cadence=%s, min %d events, min R²=%.2f, EMA=%d)",
            self._mode,
	        ("every %d events" % self._refit_event_interval) if self._event_count is not None
	        else ("every %.1f s" % self._refit_interval_s),
            self._min_events,
            self._min_fit_quality,
            self._ema_window_size,
        )
        while not self._stop_event.is_set():
	        if self._event_count is not None:
		        # Event-count cadence: re-fit every _refit_event_interval new
		        # ions. Poll the GUI's running counter on a short tick.
		        try:
			        cur = int(self._event_count())
		        except Exception:
			        cur = self._last_refit_count
		        if cur < self._last_refit_count:
			        # GUI cleared its counter (new session) -- resync.
			        self._last_refit_count = 0
		        if cur - self._last_refit_count < self._refit_event_interval:
			        self.msleep(200)
			        continue
		        self._last_refit_count = cur
	        else:
		        # Legacy time cadence. Sleep first so the very first tick has
		        # events to look at. Monotonic clock so an NTP/manual time
		        # change can't skip or stack refits.
		        deadline = time.monotonic() + self._refit_interval_s
		        while time.monotonic() < deadline:
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
        # SAFETY: the snapshot callback's contract does NOT say the
        # returned arrays must be copies, and the GUI side historically
        # has handed over views into the live writer's arrays. If the
        # writer extends one of those arrays while curve_fit is
        # iterating, the result is either a noisy fit or a
        # mid-iteration ValueError caught by the broad `except` in
        # ``run()`` -- both of which the user sees as "refit error" /
        # "calibrating…" without any clue why. Take a defensive deep
        # copy here so the fit sees frozen inputs.
        try:
            t_ns = np.array(snapshot[0], dtype=np.float64, copy=True)
            v_dc = np.array(snapshot[1], dtype=np.float64, copy=True)
            x_cm = np.array(snapshot[2], dtype=np.float64, copy=True)
            y_cm = np.array(snapshot[3], dtype=np.float64, copy=True)
        except Exception as exc:  # malformed snapshot
            self.status_changed.emit(f"snapshot error: {exc.__class__.__name__}")
            return None
        if not (t_ns.size == v_dc.size == x_cm.size == y_cm.size):
            self.status_changed.emit(
                f"snapshot length mismatch: t={t_ns.size}, v={v_dc.size}, "
                f"x={x_cm.size}, y={y_cm.size}"
            )
            return None
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
	        # A weak/flat voltage dependence (e.g. during a vdc_hold, or too
	        # few windows) is NOT a reason to abort the whole refit -- fall
	        # back to identity voltage (no V correction is always safe) so
	        # the dominant bowl correction still runs. Mirrors the offline
	        # pipeline, which tolerates a near-flat voltage term.
	        vol_fit = _VOLT_IDENTITY
	        r2 = 0.0

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
        # Reject a bowl whose surface leaves the sane band over the full
        # detector (would distort/destroy rim events); voltage-only is
        # still useful.
        if bowl_fit is None or not _bowl_surface_ok(bowl_fit, det_diam_mm):
	        bowl_fit = _BOWL_IDENTITY

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

        # The initial MC calibration is the tof_2_mc conversion above (it
        # already uses t_0 + per-event V/geometry, so the peak sits near the
        # true mass). Refine with ONE voltage then ONE bowl correction --
        # the offline MC pipeline (initial -> voltage -> bowl).

        # Step 1: voltage correction on the raw mc (MC divides by f_v
        # directly -- NO sqrt; mc is linear in V).
        maximum_location_mc = float(np.median(mc_peak))
        if maximum_location_mc <= 0:
	        self.status_changed.emit("invalid peak location for voltage fit")
            return None
        vol_fit, r2 = _fit_voltage_polynomial(
	        v_peak,
            mc_peak,
            maximum_location_mc,
            self._volt_sample_size,
        )
        if vol_fit is None or r2 < self._min_fit_quality:
	        # Weak/flat voltage dependence -> identity (skip V correction),
	        # keep going so the bowl correction still applies. The MC
	        # voltage target (mc/median) is often near-flat, so aborting
	        # here would leave MC permanently uncalibrated (the reason the
	        # status banner only ever showed a 'tof:' entry).
	        vol_fit = _VOLT_IDENTITY
	        r2 = 0.0
        f_v_peak = voltage_corr(v_peak, *vol_fit)
        mc_v_peak = mc_peak / np.maximum(f_v_peak, 1e-30)

        # Step 2: bowl correction on the voltage-corrected mc.
        maximum_location_mc_v = float(np.median(mc_v_peak))
        bowl_final = _fit_bowl_polynomial(
            x_mm_peak,
            y_mm_peak,
	        mc_v_peak,
            v_peak,
	        maximum_location_mc_v,
            self._bowl_sample_size_mm,
            det_diam_mm,
        )
        if bowl_final is None or not _bowl_surface_ok(bowl_final, det_diam_mm):
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
            bowl_final=bowl_final,
	        maximum_location=maximum_location_mc_v,
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
