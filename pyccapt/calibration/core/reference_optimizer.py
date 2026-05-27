"""Reference-constrained calibration optimizer.

Jointly fits voltage, bowl, drift, and scale corrections to align
detected peaks with known isotope/charge-state reference lines. Uses
``scipy.optimize.least_squares`` (NOT torch SGD). Designed to be run
ON TOP OF a closed-form warm-start (legacy V+Bowl or joint M1).

Algorithm
---------

Given a calibration array ``raw`` (the warm-started spectrum) and a list
of expected ``ReferenceLine`` (m/z, tolerance, weight), the optimizer:

1. Detects the strongest peaks in ``raw``.
2. Matches each detected peak to its nearest reference within tolerance
   (Hungarian-style: each reference at most once).
3. For each matched peak, splits the ions in its window into
   ``n_index_chunks`` chunks along the ion-index axis. For each
   (peak × chunk) cell, computes the observed peak centre.
4. Fits a smooth multiplicative correction surface::

      factor(V, x, y, t_idx) = scale
                              * (1 + a1*Vn + a2*Vn^2)             # V correction
                              * (1 + bowl_poly(xn, yn))            # detector bowl
                              * (1 + c1*tn + c2*tn^2)              # drift

   via ``scipy.optimize.least_squares`` against the residuals
   ``(observed_centre / factor) - reference_mz``. Bounds keep
   ``factor`` in ``[0.95, 1.05]`` so the optimizer can't violently warp
   the spectrum.

5. Predicts the per-ion correction factor and applies::

      corrected_mc  = raw / factor                  (mc)
      corrected_tof = raw / sqrt(factor)            (tof)

6. Acceptance gate: revert if ``n_peaks`` drops, if ``ref_hit`` drops,
   or if geomean MRP drops beyond ``mrp_tolerance``.

Event filtering: when ``data`` and ``contamination_policy`` are passed,
peak detection and per-cell centre estimation use only the calibration-
trustworthy subset (e.g. ``single_hit_only`` -> ion_pp == 1). The
correction factor is still applied to ALL ions; the filter only affects
which ions VOTE for the fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import least_squares

from pyccapt.calibration.core import calibration as _cal
from pyccapt.calibration.core import event_filters as _ef
from pyccapt.calibration.core.mc_plot_peak_helpers import fast_mrp


# ---------------------------------------------------------------------------
# Reference lines


@dataclass
class ReferenceLine:
    """Single expected isotope/charge-state line.

    Parameters
    ----------
    label : str
        Human-readable label, e.g. ``'Ni-58'``.
    mz : float
        Expected mass-to-charge (Da for mc, ns for tof if a tof-domain
        reference list is used).
    tolerance : float
        Max distance from a detected peak to count as a match (Da or ns).
    weight : float
        Per-line weight in the residual sum. Use 1.0 unless one line is
        much more reliable than others.
    """

    label: str
    mz: float
    tolerance: float = 0.15
    weight: float = 1.0


# Nimonic NiC1 reference. Includes the dominant 2+ charge-state lines
# that actually appear in the spectrum (these are what auto_detect_reference_peaks
# finds first at this dataset's mass range), plus the 1+ lines used by
# peak_quality_audit.py for high-mass coverage.
DEFAULT_NIMONIC_MC = [
    # 2+ states (low-mass, dominant peaks in this dataset)
    ReferenceLine("Ti-48 2+", 23.97, tolerance=0.20),
    ReferenceLine("Ti-49 2+", 24.47, tolerance=0.20),
    ReferenceLine("Ti-50 2+", 24.97, tolerance=0.20),
    ReferenceLine("Cr-52 2+", 25.97, tolerance=0.20),
    ReferenceLine("Cr-53 2+", 26.47, tolerance=0.20),
    ReferenceLine("Cr-54 2+", 26.97, tolerance=0.20),
    ReferenceLine("Fe-56 2+", 27.97, tolerance=0.20),
    ReferenceLine("Fe-57 2+", 28.47, tolerance=0.20),
    ReferenceLine("Ni-58 2+", 28.98, tolerance=0.20),
    ReferenceLine("Co-59 2+", 29.46, tolerance=0.20),
    ReferenceLine("Ni-60 2+", 29.97, tolerance=0.20),
    ReferenceLine("Ni-61 2+", 30.46, tolerance=0.20),
    ReferenceLine("Ni-62 2+", 30.97, tolerance=0.20),
    ReferenceLine("Ni-64 2+", 31.97, tolerance=0.20),
    # 1+ states (high-mass coverage)
    ReferenceLine("Al-27", 26.98),
    ReferenceLine("Ti-46", 45.95), ReferenceLine("Ti-47", 46.95),
    ReferenceLine("Ti-48", 47.95), ReferenceLine("Ti-49", 48.95),
    ReferenceLine("Ti-50", 49.94),
    ReferenceLine("Cr-52", 51.94), ReferenceLine("Cr-53", 52.94),
    ReferenceLine("Cr-54", 53.94),
    ReferenceLine("Fe-56", 55.93), ReferenceLine("Fe-57", 56.94),
    ReferenceLine("Fe-58", 57.94),
    ReferenceLine("Co-59", 58.93),
    ReferenceLine("Ni-58", 57.94), ReferenceLine("Ni-60", 59.93),
    ReferenceLine("Ni-61", 60.93),
    ReferenceLine("Ni-62", 61.93), ReferenceLine("Ni-64", 63.93),
    # Mo (substantial in many NiC1 datasets)
    ReferenceLine("Mo-92 2+", 45.96), ReferenceLine("Mo-94 2+", 46.95),
    ReferenceLine("Mo-95 2+", 47.45), ReferenceLine("Mo-96 2+", 47.95),
    ReferenceLine("Mo-97 2+", 48.45), ReferenceLine("Mo-98 2+", 48.95),
    ReferenceLine("Mo-100 2+", 49.95),
    ReferenceLine("Mo-92", 91.91), ReferenceLine("Mo-94", 93.91),
    ReferenceLine("Mo-95", 94.91), ReferenceLine("Mo-96", 95.90),
    ReferenceLine("Mo-97", 96.91), ReferenceLine("Mo-98", 97.91),
    ReferenceLine("Mo-100", 99.91),
]


# ---------------------------------------------------------------------------
# Helpers


def _match_peaks_to_references(
    detected_positions: np.ndarray,
    reference_lines: Sequence[ReferenceLine],
):
    """Greedy nearest-neighbor assignment of detected -> references.

    Each reference matches at most one detected peak, the closest one
    within ``ref.tolerance``. Returns list of (det_idx, ref_idx,
    distance) tuples.
    """
    if len(detected_positions) == 0 or not reference_lines:
        return []
    refs = list(reference_lines)
    matches = []
    used_det = set()
    # Sort refs by descending weight so high-weight lines claim first
    ref_order = sorted(range(len(refs)), key=lambda i: -refs[i].weight)
    for r_i in ref_order:
        ref = refs[r_i]
        best_det = -1
        best_dist = float("inf")
        for d_i, pos in enumerate(detected_positions):
            if d_i in used_det:
                continue
            dist = abs(pos - ref.mz)
            if dist < best_dist and dist <= ref.tolerance:
                best_dist = dist
                best_det = d_i
        if best_det >= 0:
            used_det.add(best_det)
            matches.append((best_det, r_i, best_dist))
    matches.sort(key=lambda m: m[0])  # by detected order
    return matches


def _bowl_basis(xn: np.ndarray, yn: np.ndarray, degree: int) -> np.ndarray:
    """2-D polynomial basis without the constant term.

    Returns columns [xn, yn, xn^2, xn*yn, yn^2, ...] up to total degree.
    """
    feats = []
    for d in range(1, degree + 1):
        for i in range(d + 1):
            feats.append((xn ** (d - i)) * (yn ** i))
    if not feats:
        return np.zeros((len(xn), 0), dtype=float)
    return np.column_stack(feats)


def _bowl_n_terms(degree: int) -> int:
    return sum(d + 1 for d in range(1, degree + 1))


def _factor_from_params(
    params: np.ndarray, vn: np.ndarray, xn: np.ndarray, yn: np.ndarray,
    tn: np.ndarray, bowl_degree: int,
) -> np.ndarray:
    """Compute the multiplicative correction factor at each ion / bin.

    params layout: [scale, a1, a2, *bowl_coefs, c1, c2]
    """
    scale = params[0]
    a1, a2 = params[1], params[2]
    bowl_n = _bowl_n_terms(bowl_degree)
    bowl_coefs = params[3:3 + bowl_n]
    c1, c2 = params[3 + bowl_n], params[4 + bowl_n]
    f_v = 1.0 + a1 * vn + a2 * vn ** 2
    bowl_feats = _bowl_basis(xn, yn, bowl_degree)
    bowl_pert = bowl_feats @ bowl_coefs if bowl_feats.size else np.zeros_like(vn)
    f_b = 1.0 + bowl_pert
    f_t = 1.0 + c1 * tn + c2 * tn ** 2
    return scale * f_v * f_b * f_t


def _cell_observations(
    calib_array: np.ndarray,
    voltage: np.ndarray,
    x_det: np.ndarray,
    y_det: np.ndarray,
    matches,
    detected_peaks,
    references: Sequence[ReferenceLine],
    n_index_chunks: int,
    mask: np.ndarray,
):
    """Build (V_mean, x_mean, y_mean, t_mean, observed_centre, target,
    weight) tuples for each (peak × ion-index chunk) cell.

    Only ions inside the peak window AND inside ``mask`` contribute.
    """
    n_ions = len(calib_array)
    obs_v, obs_x, obs_y, obs_t = [], [], [], []
    obs_centre, target, weight = [], [], []
    # Partial-recovered rows have NaN x_det / y_det; their np.mean per
    # chunk would propagate NaN to obs_x/obs_y, blowing up the joint
    # voltage+spatial least-squares fit. Restrict the in-peak mask to
    # position-capable rows before chunk aggregation.
    finite_xy_global = np.isfinite(np.asarray(x_det, dtype=float)) & np.isfinite(np.asarray(y_det, dtype=float))
    for det_idx, ref_idx, _dist in matches:
        peak = detected_peaks[det_idx]
        ref = references[ref_idx]
        x1, x2 = float(peak["x1"]), float(peak["x2"])
        in_peak = (calib_array > x1) & (calib_array < x2) & mask & finite_xy_global
        idx_in = np.where(in_peak)[0]
        if idx_in.size < 200:
            continue
        # Split by ion index into n_index_chunks bins
        chunk_size = max(50, int(np.ceil(idx_in.size / max(1, n_index_chunks))))
        for start in range(0, idx_in.size, chunk_size):
            chunk_idx = idx_in[start:start + chunk_size]
            if chunk_idx.size < 30:
                continue
            chunk_values = calib_array[chunk_idx]
            # Centre estimate: trimmed mean is robust + linear in ions
            centre = _trimmed_mean(chunk_values, trim=0.10)
            obs_v.append(float(np.mean(voltage[chunk_idx])))
            obs_x.append(float(np.mean(x_det[chunk_idx])))
            obs_y.append(float(np.mean(y_det[chunk_idx])))
            obs_t.append(float(np.mean(chunk_idx)) / max(1.0, n_ions - 1))
            obs_centre.append(centre)
            target.append(float(ref.mz))
            weight.append(float(ref.weight) * float(np.sqrt(chunk_idx.size)))
    if not obs_centre:
        return None
    return {
        "v": np.asarray(obs_v),
        "x": np.asarray(obs_x),
        "y": np.asarray(obs_y),
        "t": np.asarray(obs_t),
        "obs": np.asarray(obs_centre),
        "tgt": np.asarray(target),
        "w": np.asarray(weight),
    }


def _trimmed_mean(values: np.ndarray, trim: float = 0.10) -> float:
    if values.size == 0:
        return 0.0
    s = np.sort(values)
    k = int(values.size * trim)
    if 2 * k >= values.size:
        return float(np.mean(s))
    return float(np.mean(s[k:values.size - k]))


def _audit_simple(calib_array: np.ndarray, lim: float, references: Sequence[ReferenceLine]):
    """Quick audit: n_peaks above prominence + geomean MRP + ref_hit count."""
    arr = np.asarray(calib_array, dtype=float)
    arr = arr[(arr > 0) & np.isfinite(arr) & (arr < lim)]
    try:
        peaks = _cal.auto_detect_reference_peaks(
            arr, n_peaks=20, prominence=25, distance=10, hist_bin_size=0.05,
        )
    except Exception:
        return {"n_peaks": 0, "geomean_mrp": float("nan"), "ref_hit": 0}
    if not peaks:
        return {"n_peaks": 0, "geomean_mrp": float("nan"), "ref_hit": 0}
    positions = np.asarray([float(p["position"]) for p in peaks])
    matches = _match_peaks_to_references(positions, references)
    mrps = []
    for p in peaks:
        m = fast_mrp(arr, p["x1"], p["x2"], bin_size=0.01)
        m0 = float(m[0]) if m and len(m) > 0 and np.isfinite(m[0]) else float("nan")
        if np.isfinite(m0) and m0 > 0:
            mrps.append(min(m0, 5000.0))
    finite = np.asarray(mrps)
    geomean = float(np.exp(np.mean(np.log(finite)))) if finite.size else float("nan")
    return {"n_peaks": len(peaks), "geomean_mrp": geomean, "ref_hit": len(matches)}


# ---------------------------------------------------------------------------
# Main entry point


def fit_reference_constrained(
    variables,
    calibration_mode: str = "mc",
    reference_lines: Iterable[ReferenceLine] | None = None,
    *,
    bowl_degree: int = 2,
    n_index_chunks: int = 12,
    correction_bounds: tuple = (0.99, 1.01),
    width_penalty: float = 0.0,
    reg_bowl: float = 1e-1,
    reg_drift: float = 1e-1,
    max_nfev: int = 200,
    contamination_policy: str = "single_hit_only",
    mrp_drop_tolerance: float = 0.05,
    apply: bool = True,
    verbose: bool = True,
):
    """Apply reference-constrained calibration to ``variables``.

    Returns a dict with diagnostics. When ``apply=True`` and the
    acceptance gate is passed, writes back to
    ``variables.{mc_calib | dld_t_calib}``.

    Parameters
    ----------
    reference_lines : list of ReferenceLine, optional
        Expected isotope positions. Defaults to ``DEFAULT_NIMONIC_MC`` for
        mc mode. Pass an empty list to disable reference matching (then
        the fit collapses to a width-minimisation, which we DON'T do --
        see Method 1).
    bowl_degree : int
        Polynomial degree for the 2-D bowl term (default 4).
    n_index_chunks : int
        Number of ion-index chunks per peak for the drift fit.
    correction_bounds : (low, high)
        Hard bounds on the per-ion correction factor. Default (0.95, 1.05).
    width_penalty : float
        Weight on the "per-cell FWHM" penalty (0 = pure mass-error fit).
    reg_bowl, reg_drift : float
        L2 regularisation on the bowl/drift coefficients.
    max_nfev : int
        Maximum scipy.optimize.least_squares function evaluations.
    contamination_policy : str
        Forwarded to ``event_filters.build_calibration_mask``.
    mrp_drop_tolerance : float
        Acceptance gate: revert if geomean MRP drops by more than this
        fraction.
    apply : bool
        Write back to ``variables`` if True.
    """
    if calibration_mode not in ("mc", "tof"):
        raise ValueError(f"calibration_mode must be 'mc' or 'tof', got {calibration_mode!r}")
    if reference_lines is None:
        if calibration_mode == "mc":
            reference_lines = list(DEFAULT_NIMONIC_MC)
        else:
            reference_lines = []
    reference_lines = list(reference_lines)

    raw = np.asarray(variables.get_calibration_array(calibration_mode), dtype=float).copy()
    voltage = np.asarray(variables.dld_high_voltage, dtype=float)
    if getattr(variables, "pulse_mode", None) == "voltage":
        voltage = voltage + 0.7 * np.asarray(variables.dld_pulse_v, dtype=float)
    x_det = np.asarray(variables.dld_x_det, dtype=float)
    y_det = np.asarray(variables.dld_y_det, dtype=float)
    n_ions = raw.size

    # Build calibration mask (single-hit etc.). The correction factor
    # is fit using the masked ions but applied to ALL ions.
    mask = _ef.build_calibration_mask(getattr(variables, "data", None),
                                       policy=contamination_policy)
    if mask.size != n_ions:
        mask = np.ones(n_ions, dtype=bool)
    if verbose:
        n_keep = int(mask.sum())
        print(f"[ref-opt] {calibration_mode}: contamination_policy={contamination_policy!r} "
              f"keep {n_keep:,}/{n_ions:,} ions for fitting")

    # Detect peaks on the masked subset
    lim = variables.max_tof if calibration_mode == "tof" else 400.0
    # Use threaded event_mask API for clarity. The mask + finite/positive
    # filter is combined into a single boolean array.
    finite_mask = mask & (raw > 0) & (raw < lim) & np.isfinite(raw)
    arr_for_detect = raw[finite_mask]
    try:
        detected_peaks = _cal.auto_detect_reference_peaks(
            arr_for_detect, n_peaks=20, prominence=25, distance=10, hist_bin_size=0.05,
        )
    except Exception as exc:
        if verbose:
            print(f"[ref-opt] {calibration_mode}: peak detection failed: {exc!r}")
        return {"ok": False, "reason": f"peak_detect_error: {exc!r}"}
    if len(detected_peaks) < 2:
        if verbose:
            print(f"[ref-opt] {calibration_mode}: only {len(detected_peaks)} peaks; aborting")
        return {"ok": False, "reason": "insufficient_peaks"}

    detected_positions = np.asarray([p["position"] for p in detected_peaks])
    matches = _match_peaks_to_references(detected_positions, reference_lines)
    if not matches:
        if verbose:
            print(f"[ref-opt] {calibration_mode}: 0 reference matches; aborting")
        return {"ok": False, "reason": "no_reference_matches"}
    if verbose:
        pairs = []
        for det_idx, ref_idx, dist in matches[:6]:
            ref = reference_lines[ref_idx]
            pos = detected_positions[det_idx]
            pairs.append(f"{ref.label}({pos:.3f}->{ref.mz:.3f}, d={dist:.3f})")
        more = f" ... +{len(matches) - 6}" if len(matches) > 6 else ""
        print(f"[ref-opt] {calibration_mode}: matched {len(matches)} peaks: " + ", ".join(pairs) + more)

    # Adaptive model complexity: don't fit parameters we can't constrain.
    n_matches = len(matches)
    if n_matches < 4:
        # Very sparse: scale only (1 param)
        effective_bowl_degree = 0
        fit_v = False
        fit_drift = False
    elif n_matches < 8:
        # Modest: scale + V (3 params)
        effective_bowl_degree = 0
        fit_v = True
        fit_drift = False
    elif n_matches < 14:
        # Reasonable: scale + V + bowl deg 2 (8 params)
        effective_bowl_degree = min(2, int(bowl_degree))
        fit_v = True
        fit_drift = False
    else:
        # Plenty: full model
        effective_bowl_degree = int(bowl_degree)
        fit_v = True
        fit_drift = True

    if verbose:
        print(f"[ref-opt] {calibration_mode}: adaptive model - "
              f"bowl_deg={effective_bowl_degree} fit_v={fit_v} fit_drift={fit_drift} "
              f"(based on {n_matches} matches)")

    # Build per-cell observations
    cells = _cell_observations(
        raw, voltage, x_det, y_det, matches, detected_peaks,
        reference_lines, n_index_chunks=int(n_index_chunks), mask=mask,
    )
    if cells is None:
        if verbose:
            print(f"[ref-opt] {calibration_mode}: no usable cells; aborting")
        return {"ok": False, "reason": "no_cells"}

    # Normalise features
    v_mean = float(np.mean(voltage)); v_std = max(float(np.std(voltage)), 1e-6)
    x_mean = float(np.mean(x_det)); x_std = max(float(np.std(x_det)), 1e-6)
    y_mean = float(np.mean(y_det)); y_std = max(float(np.std(y_det)), 1e-6)

    def _norm(arr_val, mean, std):
        return (arr_val - mean) / std

    vn_cell = _norm(cells["v"], v_mean, v_std)
    xn_cell = _norm(cells["x"], x_mean, x_std)
    yn_cell = _norm(cells["y"], y_mean, y_std)
    tn_cell = 2.0 * cells["t"] - 1.0

    bowl_n = _bowl_n_terms(effective_bowl_degree)
    n_params = 1 + 2 + bowl_n + 2  # scale, a1, a2, bowl..., c1, c2

    def _residuals(params):
        f = _factor_from_params(params, vn_cell, xn_cell, yn_cell, tn_cell, effective_bowl_degree)
        corrected = cells["obs"] / np.maximum(f, 1e-6)
        residual = (corrected - cells["tgt"]) * np.sqrt(cells["w"])
        # L2 regularisation on bowl + drift
        reg = []
        for i in range(3, 3 + bowl_n):
            reg.append(np.sqrt(reg_bowl) * params[i])
        for i in (3 + bowl_n, 4 + bowl_n):
            reg.append(np.sqrt(reg_drift) * params[i])
        # Pull scale toward 1 (small penalty)
        reg.append(0.01 * (params[0] - 1.0) * np.sqrt(cells["obs"].size))
        return np.concatenate([residual, np.asarray(reg)])

    # Bounds: scale in [0.9, 1.1], poly coeffs unconstrained, but the
    # factor itself is clipped at application time.
    # Tight bounds: this is a fine-tuning step on top of a closed-form
    # warm start. The correction factor itself is clipped to
    # correction_bounds at application time, which is the hard guarantee.
    # Tight bounds: this is a fine-tuning step on top of a closed-form
    # warm start. The correction factor itself is clipped to
    # correction_bounds at application time, which is the hard guarantee.
    lb = np.full(n_params, -np.inf)
    ub = np.full(n_params, np.inf)
    lb[0] = 0.99; ub[0] = 1.01    # scale (always fit)
    # scipy.optimize.least_squares requires lb < ub strictly. For "fixed
    # at zero", we use a tiny non-zero range so the param is effectively pinned.
    EPS = 1e-9
    if fit_v:
        lb[1] = -0.02; ub[1] = 0.02   # a1
        lb[2] = -0.02; ub[2] = 0.02   # a2
    else:
        lb[1] = -EPS; ub[1] = EPS
        lb[2] = -EPS; ub[2] = EPS
    for i in range(3, 3 + bowl_n):
        lb[i] = -0.01; ub[i] = 0.01
    if fit_drift:
        lb[3 + bowl_n] = -0.01; ub[3 + bowl_n] = 0.01
        lb[4 + bowl_n] = -0.01; ub[4 + bowl_n] = 0.01
    else:
        lb[3 + bowl_n] = -EPS; ub[3 + bowl_n] = EPS
        lb[4 + bowl_n] = -EPS; ub[4 + bowl_n] = EPS

    x0 = np.zeros(n_params)
    x0[0] = 1.0

    try:
        result = least_squares(
            _residuals, x0, bounds=(lb, ub),
            max_nfev=int(max_nfev), method="trf",
        )
    except Exception as exc:
        if verbose:
            print(f"[ref-opt] {calibration_mode}: least_squares failed: {exc!r}")
        return {"ok": False, "reason": f"lsq_error: {exc!r}"}

    params = result.x
    # Predict per-ion factor
    vn_full = _norm(voltage, v_mean, v_std)
    xn_full = _norm(x_det, x_mean, x_std)
    yn_full = _norm(y_det, y_mean, y_std)
    tn_full = (np.arange(n_ions, dtype=float) / max(1.0, n_ions - 1)) * 2.0 - 1.0
    factor = _factor_from_params(params, vn_full, xn_full, yn_full, tn_full, effective_bowl_degree)
    factor = np.clip(factor, correction_bounds[0], correction_bounds[1])

    if calibration_mode == "tof":
        factor_applied = np.sqrt(np.maximum(factor, 1e-6))
    else:
        factor_applied = factor
    candidate = raw / np.maximum(factor_applied, 1e-6)

    # Acceptance gate: compare candidate against raw on the audit metric
    pre = _audit_simple(raw, lim, reference_lines)
    post = _audit_simple(candidate, lim, reference_lines)
    accepted = (
        post["n_peaks"] >= pre["n_peaks"]
        and post["ref_hit"] >= pre["ref_hit"]
        and (
            not np.isfinite(pre["geomean_mrp"])
            or not np.isfinite(post["geomean_mrp"])
            or post["geomean_mrp"] >= pre["geomean_mrp"] * (1.0 - float(mrp_drop_tolerance))
        )
    )
    info = {
        "ok": bool(accepted),
        "calibration_mode": calibration_mode,
        "lsq_cost": float(result.cost),
        "lsq_nfev": int(result.nfev),
        "n_matches": len(matches),
        "n_cells": int(cells["obs"].size),
        "params": params.tolist(),
        "factor_min": float(factor.min()),
        "factor_max": float(factor.max()),
        "pre": pre,
        "post": post,
    }
    if verbose:
        print(
            f"[ref-opt] {calibration_mode}: pre n={pre['n_peaks']} mrp={pre['geomean_mrp']:.0f} ref={pre['ref_hit']}  "
            f"-> post n={post['n_peaks']} mrp={post['geomean_mrp']:.0f} ref={post['ref_hit']}  "
            f"factor_range=[{info['factor_min']:.4f}, {info['factor_max']:.4f}]  "
            f"{'ACCEPTED' if accepted else 'REVERTED'}"
        )

    if apply and accepted:
        if calibration_mode == "tof":
            variables.dld_t_calib = candidate
        else:
            variables.mc_calib = candidate
    return info
