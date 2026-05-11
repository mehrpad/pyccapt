"""RHIT mass-to-charge calibration helpers.

Split out from :mod:`rhit_tools` to keep that module under the calibration
1250-line policy. Re-exported via :mod:`rhit_tools` so external callers continue
to use ``from pyccapt.calibration.leap_tools.cameca_raw import rhit_tools`` and
its ``rhit_calibrate_from_epos`` / ``rhit_apply_calibration`` entry points
unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pyccapt.calibration.leap_tools import leap_tools

ELECTRON_CHARGE_C = 1.602176634e-19
ATOMIC_MASS_UNIT_KG = 1.66053906660e-27
DEFAULT_FLIGHT_PATH_M = 0.382
DEFAULT_T0_NS = 45.0
DEFAULT_KF = 1.03


def _normalize_epos_calibration_dataframe(epos: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(epos, (str, Path)):
        epos_df = leap_tools.read_epos(str(epos))
    else:
        epos_df = epos.copy()

    normalized = pd.DataFrame(index=epos_df.index)
    column_map = {
        "mc": ["mc", "m/n (Da)"],
        "tof": ["tof", "TOF (ns)"],
        "VDC": ["VDC", "HV_DC (V)"],
        "detx": ["detx", "det_x (mm)"],
        "dety": ["dety", "det_y (mm)"],
    }
    for output_name, candidates in column_map.items():
        for candidate in candidates:
            if candidate in epos_df.columns:
                normalized[output_name] = epos_df[candidate].to_numpy(dtype=float)
                break
        else:
            raise ValueError(f"EPOS data is missing a required column for calibration: {candidates[0]}")
    return normalized


def _find_rhit_anchor_for_epos_start(
    hits: pd.DataFrame, epos: pd.DataFrame, sample_size: int = 200
) -> int:
    """Locate the RHIT index where the EPOS file appears to begin.

    The MATLAB ``rhitCalibrateFromEpos`` assumes ``epos[0]`` ↔ ``hits[0]``, but
    EPOS files are often produced after an initial voltage-ramp/burn-in is
    discarded. This finds the RHIT index whose VDC is closest to ``epos[0:N]``
    median VDC, so we can pair RHIT[anchor + i] with EPOS[i] downstream.
    """
    if len(hits) == 0 or len(epos) == 0:
        return 0
    sample_size = int(min(sample_size, len(epos)))
    target_vdc = float(np.median(epos["VDC"].to_numpy(dtype=float)[:sample_size]))
    hits_vdc = hits["VDC"].to_numpy(dtype=float)
    above = np.where(hits_vdc >= target_vdc)[0]
    if len(above) == 0:
        return int(np.argmin(np.abs(hits_vdc - target_vdc)))
    return int(above[0])


def _estimate_icf(
    hits: pd.DataFrame,
    epos: pd.DataFrame,
    verbose: bool = True,
    rhit_offset: int = 0,
) -> float:
    """Estimate the image-compression factor from a paired sample of events.

    Mirrors the MATLAB ``rhitCalibrateFromEpos`` step:
    ``ICF = median(epos.detx / hits.detx)`` for ~200 paired events where
    ``|hits.detx| > 1`` and ``|ratio| < 2``. The ``rhit_offset`` argument lets
    callers anchor the pairing at an arbitrary RHIT index when EPOS[0] does not
    correspond to RHIT[0].
    """
    sample_size = int(min(200, len(hits) - rhit_offset, len(epos)))
    if sample_size <= 0:
        return 1.0
    hit_detx = hits["detx"].to_numpy(dtype=float)[rhit_offset : rhit_offset + sample_size]
    epos_detx = epos["detx"].to_numpy(dtype=float)[:sample_size]
    ratios = np.full(sample_size, np.nan, dtype=float)
    valid_nonzero = np.abs(hit_detx) > 1e-12
    ratios[valid_nonzero] = epos_detx[valid_nonzero] / hit_detx[valid_nonzero]
    valid = np.abs(hit_detx) > 1.0
    valid &= np.isfinite(ratios) & (np.abs(ratios) < 2.0)
    if verbose:
        n_valid = int(np.count_nonzero(valid))
        if n_valid > 0:
            valid_ratios = ratios[valid]
            print(
                f"  ICF estimation (rhit_offset={rhit_offset}): "
                f"{n_valid}/{sample_size} usable pairs; "
                f"ratio median={float(np.median(valid_ratios)):.4f}, "
                f"std={float(np.std(valid_ratios)):.4f}"
            )
        else:
            print(
                f"  ICF estimation (rhit_offset={rhit_offset}): "
                f"no usable pairs in {sample_size} samples; falling back to ICF=1.0"
            )
    if not np.any(valid):
        return 1.0
    return float(np.median(ratios[valid]))


def _chunked_match_events(
    hits: pd.DataFrame,
    epos: pd.DataFrame,
    icf: float,
    *,
    vdc_tol: float = 0.1,
    det_tol: float = 0.2,
    rhit_anchor: int = 0,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair RHIT and EPOS events by VDC + detector position.

    ``rhit_anchor`` is the RHIT index that corresponds to ``epos[0]``. EPOS
    files frequently start partway through a RHIT (after voltage ramp-up), so
    callers should pass the anchor returned by
    :func:`_find_rhit_anchor_for_epos_start` rather than the implicit 0.
    """
    n_rhit = len(hits)
    n_epos = len(epos)
    if n_rhit == 0 or n_epos == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    max_pairs = min(n_rhit - rhit_anchor, n_epos)
    if max_pairs <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    chunk_size = min(2000, max_pairs)
    if max_pairs <= chunk_size + 200:
        chunk_starts = np.array([0], dtype=int)
    else:
        n_chunks = min(50, max(1, max_pairs // chunk_size))
        chunk_starts = np.round(np.linspace(0, max_pairs - chunk_size - 1, n_chunks)).astype(int)

    matched_epos: list[int] = []
    matched_rhit: list[int] = []
    rhit_offset = rhit_anchor
    hits_vdc = hits["VDC"].to_numpy(dtype=float)
    hits_detx = hits["detx"].to_numpy(dtype=float)
    hits_dety = hits["dety"].to_numpy(dtype=float)
    epos_vdc = epos["VDC"].to_numpy(dtype=float)
    epos_detx = epos["detx"].to_numpy(dtype=float)
    epos_dety = epos["dety"].to_numpy(dtype=float)

    for start_epos in chunk_starts:
        start_rhit = int(np.clip(start_epos + rhit_offset, 0, n_rhit - 1))
        sync_range = range(max(0, start_rhit - 500), min(n_rhit, start_rhit + 501))
        best_distance = np.inf
        for candidate in sync_range:
            distance = abs(epos_vdc[start_epos] - hits_vdc[candidate])
            if distance < best_distance:
                best_distance = distance
                start_rhit = candidate
            if distance < vdc_tol:
                break
        rhit_offset = start_rhit - start_epos

        rolling_rhit = start_rhit
        stop_epos = min(start_epos + chunk_size, n_epos)
        for epos_index in range(start_epos, stop_epos):
            search_start = max(0, rolling_rhit)
            search_stop = min(n_rhit, rolling_rhit + 16)
            for rhit_index in range(search_start, search_stop):
                d_vdc = abs(epos_vdc[epos_index] - hits_vdc[rhit_index])
                d_detx = abs(epos_detx[epos_index] - hits_detx[rhit_index] * icf)
                d_dety = abs(epos_dety[epos_index] - hits_dety[rhit_index] * icf)
                if d_vdc < vdc_tol and d_detx < det_tol and d_dety < det_tol:
                    matched_epos.append(epos_index)
                    matched_rhit.append(rhit_index)
                    rolling_rhit = rhit_index + 1
                    break

    if verbose:
        print(
            f"  Match attempt rhit_anchor={rhit_anchor}, vdc_tol={vdc_tol} V, "
            f"det_tol={det_tol} mm, icf={icf:.4f}: "
            f"{len(matched_epos):,} matches across {len(chunk_starts)} chunks"
        )

    if not matched_epos:
        return np.array([], dtype=int), np.array([], dtype=int)

    unique_pairs = list(dict.fromkeys(zip(matched_epos, matched_rhit)))
    matched_epos = np.array([pair[0] for pair in unique_pairs], dtype=int)
    matched_rhit = np.array([pair[1] for pair in unique_pairs], dtype=int)
    return matched_epos, matched_rhit


def _adaptive_match(
    hits: pd.DataFrame,
    epos: pd.DataFrame,
    icf: float,
    rhit_anchor: int = 0,
    min_matches: int = 50,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Match RHIT/EPOS events, progressively loosening tolerances if needed."""
    candidates = [
        (icf, 0.1, 0.2),
        (icf, 0.5, 0.5),
        (icf, 1.0, 1.0),
        (1.0, 0.5, 0.5),
        (-icf, 0.5, 0.5),
        (1.0, 2.0, 2.0),
    ]
    seen: set[tuple[float, float, float]] = set()
    for trial_icf, vdc_tol, det_tol in candidates:
        key = (round(trial_icf, 6), vdc_tol, det_tol)
        if key in seen:
            continue
        seen.add(key)
        matched_epos, matched_rhit = _chunked_match_events(
            hits, epos, trial_icf, vdc_tol=vdc_tol, det_tol=det_tol,
            rhit_anchor=rhit_anchor, verbose=verbose,
        )
        if len(matched_epos) >= min_matches:
            return matched_epos, matched_rhit, {
                "icf": trial_icf, "vdc_tol": vdc_tol, "det_tol": det_tol,
                "rhit_anchor": rhit_anchor,
            }

    matched_epos, matched_rhit = _chunked_match_events(
        hits, epos, icf, rhit_anchor=rhit_anchor, verbose=False
    )
    return matched_epos, matched_rhit, {
        "icf": icf, "vdc_tol": 0.1, "det_tol": 0.2, "rhit_anchor": rhit_anchor,
    }


def rhit_apply_calibration(hits: pd.DataFrame, calibration: dict[str, Any]) -> pd.DataFrame:
    """Apply a previously derived RHIT calibration to RHIT hits."""
    calibrated = hits.copy()
    radius_sq = calibrated["detx"].to_numpy(dtype=float) ** 2 + calibrated["dety"].to_numpy(dtype=float) ** 2
    c_poly = np.asarray(calibration["C_poly"], dtype=float)
    c_values = c_poly[0] + c_poly[1] * radius_sq + c_poly[2] * radius_sq**2
    tof_corrected = calibrated["tof"].to_numpy(dtype=float) - float(calibration["t_offset"])
    calibrated["mc"] = c_values * calibrated["VDC"].to_numpy(dtype=float) * tof_corrected**2
    return calibrated


def apply_rhit_calibration(hits: pd.DataFrame, calibration: dict[str, Any]) -> pd.DataFrame:
    """Alias for :func:`rhit_apply_calibration`."""
    return rhit_apply_calibration(hits, calibration)


def _calibrate_via_event_match(
    hits_df: pd.DataFrame,
    epos_df: pd.DataFrame,
    *,
    verbose: bool,
) -> dict[str, Any]:
    """MATLAB-style per-event matcher: pair RHIT/EPOS by VDC + detector position."""
    rhit_anchor = _find_rhit_anchor_for_epos_start(hits_df, epos_df)
    if verbose and rhit_anchor != 0:
        print(
            f"  RHIT anchor: pairing EPOS[0] with RHIT[{rhit_anchor:,}] "
            f"(RHIT VDC at anchor = {hits_df['VDC'].iloc[rhit_anchor]:.1f} V)"
        )

    icf = _estimate_icf(hits_df, epos_df, verbose=verbose, rhit_offset=rhit_anchor)
    matched_epos, matched_rhit, used_tols = _adaptive_match(
        hits_df, epos_df, icf, rhit_anchor=rhit_anchor, min_matches=50, verbose=verbose
    )
    if len(matched_epos) < 50 and rhit_anchor != 0:
        if verbose:
            print("  Anchor-based matching short on hits; retrying with rhit_anchor=0")
        fallback = _adaptive_match(
            hits_df, epos_df,
            _estimate_icf(hits_df, epos_df, verbose=False, rhit_offset=0),
            rhit_anchor=0, min_matches=50, verbose=verbose,
        )
        if len(fallback[0]) > len(matched_epos):
            matched_epos, matched_rhit, used_tols = fallback
    icf = float(used_tols["icf"])
    if len(matched_epos) < 50:
        raise ValueError(
            f"RHIT/EPOS matching found only {len(matched_epos)} events even after adaptive widening "
            f"(last try: vdc_tol={used_tols['vdc_tol']} V, det_tol={used_tols['det_tol']} mm, "
            f"icf={used_tols['icf']:.3f}, rhit_anchor={used_tols['rhit_anchor']}). "
            f"Try method='spectrum_match' instead — it doesn't need event-by-event correspondence."
        )

    tof_rhit = hits_df.iloc[matched_rhit]["tof"].to_numpy(dtype=float)
    tof_epos = epos_df.iloc[matched_epos]["tof"].to_numpy(dtype=float)
    t_offset = float(np.median(tof_rhit - tof_epos))

    mc_epos = epos_df.iloc[matched_epos]["mc"].to_numpy(dtype=float)
    vdc_epos = epos_df.iloc[matched_epos]["VDC"].to_numpy(dtype=float)
    valid = (mc_epos > 0.5) & (mc_epos < 200.0) & (tof_epos > 10.0) & np.isfinite(mc_epos) & np.isfinite(vdc_epos)
    if np.count_nonzero(valid) < 50:
        raise ValueError("Too few matched RHIT/EPOS events survived the mc/tof validity filter")

    c_per_event = mc_epos[valid] / (vdc_epos[valid] * tof_epos[valid] ** 2)
    detx = hits_df.iloc[matched_rhit[valid]]["detx"].to_numpy(dtype=float)
    dety = hits_df.iloc[matched_rhit[valid]]["dety"].to_numpy(dtype=float)
    radius_sq = detx**2 + dety**2

    median_c = float(np.median(c_per_event))
    good = np.abs(c_per_event - median_c) < 0.2 * median_c
    if np.count_nonzero(good) < 25:
        good = np.ones_like(c_per_event, dtype=bool)

    design = np.column_stack((np.ones(np.count_nonzero(good)), radius_sq[good], radius_sq[good] ** 2))
    c_poly, *_ = np.linalg.lstsq(design, c_per_event[good], rcond=None)
    residuals = c_per_event[good] - design @ c_poly

    return {
        "method": "event_match",
        "t_offset": t_offset,
        "C_poly": [float(value) for value in c_poly],
        "ICF": float(icf),
        "residual_std": float(np.std(residuals)),
        "matched_events": int(len(matched_epos)),
    }


def _seed_constants_from_metadata(
    hits_df: pd.DataFrame, metadata: dict[str, Any]
) -> tuple[float, float, float, float]:
    """Return (flight_path_m, t0_meta_ns, kf_seed, base_C0) from metadata."""
    instrument_params = metadata.get("instrumentParams", {}) if metadata else {}
    flight_path_m = float(instrument_params.get("flight_path_mm", DEFAULT_FLIGHT_PATH_M * 1000.0)) / 1000.0
    t0_meta = float(instrument_params.get("t0_ns", DEFAULT_T0_NS))
    if "Vref" in hits_df.columns:
        vdc = hits_df["VDC"].to_numpy(dtype=float)
        valid_v = np.abs(vdc) > 0
        if np.any(valid_v):
            kf_seed = float(np.median(hits_df.loc[valid_v, "Vref"] / vdc[valid_v]))
        else:
            kf_seed = DEFAULT_KF
    else:
        kf_seed = DEFAULT_KF
    base_c0 = 2.0 * ELECTRON_CHARGE_C / (ATOMIC_MASS_UNIT_KG * flight_path_m**2) * 1e-18 / np.sqrt(kf_seed)
    return flight_path_m, t0_meta, kf_seed, base_c0


def _build_reference_pmass(rhit_histograms: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    """Pull the Cameca pMass histogram out of the RHIT file as (edges, values)."""
    if not rhit_histograms:
        return None
    mass = rhit_histograms.get("massSpectrum")
    if mass is None:
        return None
    values = np.asarray(mass.get("values"), dtype=float)
    edges = np.asarray(mass.get("edges"), dtype=float)
    if values.size == 0 or edges.size != values.size + 1:
        return None
    return edges, values


def _spectrum_match_two_stage(
    *,
    tof_sub: np.ndarray,
    vdc_sub: np.ndarray,
    radius_sq_sub: np.ndarray,
    reference_hist: np.ndarray,
    mc_edges: np.ndarray,
    base_c0: float,
    t0_meta: float,
    verbose: bool,
) -> tuple[np.ndarray, float]:
    """Two-stage spectrum-match optimiser.

    Stage 1: optimise ``(t_offset, c0)`` only with ``c1 = c2 = 0`` over a 7x3
    seed grid; pick the best by ``-corr(rhit_hist, reference_hist)``.

    Stage 2: refine all four parameters from Stage 1's optimum, with the bowl
    coefficients allowed to drift but constrained to a small fractional impact
    on the largest-radius events.

    Returns ``(parameters, best_cost)`` where parameters = [t_offset, c0, c1, c2].
    """
    from scipy.optimize import minimize

    radius_sq_max = float(np.nanmax(radius_sq_sub)) if radius_sq_sub.size else 0.0
    bowl_scale_c1 = base_c0 / max(radius_sq_max, 1.0) if radius_sq_max > 0 else 0.0
    bowl_scale_c2 = base_c0 / max(radius_sq_max**2, 1.0) if radius_sq_max > 0 else 0.0

    def _cost_stage1(parameters: np.ndarray) -> float:
        t_offset = parameters[0]
        c0 = parameters[1]
        if c0 <= 0.0:
            return np.inf
        tof_corrected = tof_sub - t_offset
        trial_mc = c0 * vdc_sub * tof_corrected * tof_corrected
        valid = (trial_mc > 0.0) & (trial_mc < mc_edges[-1])
        if np.count_nonzero(valid) < 1000:
            return np.inf
        hist, _ = np.histogram(trial_mc[valid], bins=mc_edges)
        if hist.std() == 0:
            return np.inf
        corr = np.corrcoef(hist.astype(float), reference_hist.astype(float))[0, 1]
        if not np.isfinite(corr):
            return np.inf
        return -float(corr)

    def _cost_stage2(parameters: np.ndarray) -> float:
        t_offset = parameters[0]
        c0 = parameters[1]
        c1 = parameters[2]
        c2 = parameters[3]
        if c0 <= 0.0:
            return np.inf
        c_values = c0 + c1 * radius_sq_sub + c2 * radius_sq_sub * radius_sq_sub
        if np.any(c_values <= 0.0):
            return np.inf
        max_bowl_offset = abs(c1) * radius_sq_max + abs(c2) * radius_sq_max * radius_sq_max
        if max_bowl_offset > 0.5 * c0:
            return np.inf
        tof_corrected = tof_sub - t_offset
        trial_mc = c_values * vdc_sub * tof_corrected * tof_corrected
        valid = (trial_mc > 0.0) & (trial_mc < mc_edges[-1])
        if np.count_nonzero(valid) < 1000:
            return np.inf
        hist, _ = np.histogram(trial_mc[valid], bins=mc_edges)
        if hist.std() == 0:
            return np.inf
        corr = np.corrcoef(hist.astype(float), reference_hist.astype(float))[0, 1]
        if not np.isfinite(corr):
            return np.inf
        return -float(corr)

    best_stage1 = (np.inf, np.array([t0_meta, base_c0], dtype=float))
    t0_seeds = np.array([t0_meta - 20.0, t0_meta - 5.0, t0_meta - 1.0, t0_meta,
                         t0_meta + 1.0, t0_meta + 5.0, t0_meta + 20.0], dtype=float)
    c0_seeds = np.array([0.5 * base_c0, base_c0, 2.0 * base_c0], dtype=float)
    if verbose:
        print(f"  Stage 1: t0+C0 only, seeds {len(t0_seeds)} x {len(c0_seeds)}")
    for t0_seed in t0_seeds:
        for c0_seed in c0_seeds:
            result = minimize(
                _cost_stage1,
                np.array([t0_seed, c0_seed], dtype=float),
                method="Nelder-Mead",
                options={"xatol": 1e-9, "fatol": 1e-8, "maxiter": 800, "adaptive": True},
            )
            if result.fun < best_stage1[0]:
                best_stage1 = (float(result.fun), np.asarray(result.x, dtype=float))
                if verbose:
                    print(
                        f"    t0={t0_seed:.2f}, C0_factor={c0_seed/base_c0:.2f} -> "
                        f"corr={-result.fun:.4f}, t_offset={result.x[0]:.3f}, C0={result.x[1]:.3e}"
                    )

    stage1_t, stage1_c0 = best_stage1[1]
    best_stage2 = (best_stage1[0], np.array([stage1_t, stage1_c0, 0.0, 0.0], dtype=float))
    if radius_sq_max > 0:
        if verbose:
            print(f"  Stage 2: refining bowl (C1, C2) from Stage 1 optimum")
        for c1_seed_factor in (0.0, 0.05, -0.05):
            for c2_seed_factor in (0.0, 0.05, -0.05):
                x0 = np.array([
                    stage1_t,
                    stage1_c0,
                    c1_seed_factor * bowl_scale_c1,
                    c2_seed_factor * bowl_scale_c2,
                ], dtype=float)
                result = minimize(
                    _cost_stage2,
                    x0,
                    method="Nelder-Mead",
                    options={"xatol": 1e-10, "fatol": 1e-9, "maxiter": 1200, "adaptive": True},
                )
                if result.fun < best_stage2[0]:
                    best_stage2 = (float(result.fun), np.asarray(result.x, dtype=float))
                    if verbose:
                        print(
                            f"    bowl seed ({c1_seed_factor:+.2f},{c2_seed_factor:+.2f}) -> "
                            f"corr={-result.fun:.4f}, t={result.x[0]:.3f}, C0={result.x[1]:.3e}, "
                            f"C1={result.x[2]:.3e}, C2={result.x[3]:.3e}"
                        )

    return best_stage2[1], best_stage2[0]


def _calibrate_via_spectrum_match(
    hits_df: pd.DataFrame,
    epos_df: pd.DataFrame | None,
    *,
    metadata: dict[str, Any],
    verbose: bool,
    method_label: str = "spectrum_match",
    reference_source: tuple[np.ndarray, np.ndarray] | None = None,
    reference_label: str = "EPOS mc histogram",
) -> dict[str, Any]:
    """Optimise t_offset and the radial bowl polynomial against a reference mc histogram."""
    flight_path_m, t0_meta, kf_seed, base_c0 = _seed_constants_from_metadata(hits_df, metadata)

    if reference_source is not None:
        mc_edges, reference_hist = reference_source
        mc_edges = np.asarray(mc_edges, dtype=float)
        reference_hist = np.asarray(reference_hist, dtype=float)
        n_reference_samples = int(reference_hist.sum())
    else:
        mc_edges = np.arange(0.0, 200.02, 0.02, dtype=float)
        if epos_df is None:
            raise ValueError("spectrum_match requires either an EPOS dataframe or a reference_source")
        mc_epos = epos_df["mc"].to_numpy(dtype=float)
        finite_epos = mc_epos[np.isfinite(mc_epos) & (mc_epos > 0.0) & (mc_epos < 200.0)]
        reference_hist, _ = np.histogram(finite_epos, bins=mc_edges)
        n_reference_samples = int(len(finite_epos))
    if reference_hist.std() == 0 or reference_hist.sum() == 0:
        raise ValueError("Reference mc histogram is empty or constant; cannot fit")

    stride = max(1, len(hits_df) // 200_000)
    tof_sub = hits_df["tof"].to_numpy(dtype=float)[::stride]
    vdc_sub = hits_df["VDC"].to_numpy(dtype=float)[::stride]
    detx_sub = hits_df["detx"].to_numpy(dtype=float)[::stride]
    dety_sub = hits_df["dety"].to_numpy(dtype=float)[::stride]
    radius_sq_sub = detx_sub * detx_sub + dety_sub * dety_sub
    finite_mask = (
        np.isfinite(tof_sub) & np.isfinite(vdc_sub)
        & np.isfinite(detx_sub) & np.isfinite(dety_sub)
    )
    tof_sub = tof_sub[finite_mask]
    vdc_sub = vdc_sub[finite_mask]
    radius_sq_sub = radius_sq_sub[finite_mask]
    if len(tof_sub) < 1000:
        raise ValueError("Too few RHIT events with finite tof/VDC/detx/dety for histogram matching")

    if verbose:
        bin_width = float(mc_edges[1] - mc_edges[0])
        print(
            f"  {method_label}: histogramming {len(tof_sub):,} subsampled RHIT events "
            f"(stride={stride}) against {reference_label} "
            f"({n_reference_samples:,} samples, bin width {bin_width:.3f} Da)"
        )
        print(
            f"  Seed: flight_path={flight_path_m*1000:.1f} mm, "
            f"t0={t0_meta:.2f} ns, kf={kf_seed:.4f}, base C0={base_c0:.3e}"
        )

    parameters, best_cost = _spectrum_match_two_stage(
        tof_sub=tof_sub,
        vdc_sub=vdc_sub,
        radius_sq_sub=radius_sq_sub,
        reference_hist=reference_hist,
        mc_edges=mc_edges,
        base_c0=base_c0,
        t0_meta=t0_meta,
        verbose=verbose,
    )

    t_offset, c0, c1, c2 = (float(value) for value in parameters)
    return {
        "method": method_label,
        "t_offset": t_offset,
        "C_poly": [c0, c1, c2],
        "ICF": 1.0,
        "spectrum_correlation": float(-best_cost) if np.isfinite(best_cost) else float("nan"),
        "matched_events": n_reference_samples,
        "kf": float(kf_seed),
        "flight_path_m": float(flight_path_m),
        "reference": reference_label,
    }


def _calibrate_via_pmass_match(
    hits_df: pd.DataFrame,
    rhit_histograms: dict[str, Any],
    *,
    metadata: dict[str, Any],
    verbose: bool,
) -> dict[str, Any]:
    """Spectrum-match against the Cameca pMass histogram stored inside the RHIT.

    pMass is IVAS's own bowl-corrected mass spectrum so it makes a more reliable
    reference than EPOS mc when the EPOS file is filtered or re-ordered. No
    separate EPOS file is needed.
    """
    pmass_data = _build_reference_pmass(rhit_histograms)
    if pmass_data is None:
        raise ValueError("RHIT file has no usable pMass histogram; pmass_match unavailable")
    edges, values = pmass_data
    if verbose:
        peak_idx = int(np.argmax(values))
        peak_centre = 0.5 * (edges[peak_idx] + edges[peak_idx + 1])
        print(
            f"  Reference: Cameca pMass histogram, {len(values):,} bins "
            f"({edges[0]:.2f} ... {edges[-1]:.2f} Da), dominant peak at {peak_centre:.3f} Da"
        )
    return _calibrate_via_spectrum_match(
        hits_df,
        epos_df=None,
        metadata=metadata,
        verbose=verbose,
        method_label="pmass_match",
        reference_source=(edges, values),
        reference_label="RHIT pMass histogram",
    )


def rhit_calibrate_from_epos(
    hits: pd.DataFrame,
    epos: pd.DataFrame | str | Path | None = None,
    *,
    method: str = "spectrum_match",
    metadata: dict[str, Any] | None = None,
    rhit_histograms: dict[str, Any] | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Calibrate RHIT mass-to-charge.

    Parameters
    ----------
    epos
        Required for ``method='spectrum_match'`` and ``method='event_match'``.
        Ignored for ``method='pmass_match'``.

    method : {"spectrum_match", "pmass_match", "event_match"}
        ``"spectrum_match"`` (default): optimise ``t_offset`` and a radial bowl
        polynomial in two stages so the RHIT-derived mc histogram correlates
        maximally with the EPOS mc histogram. No event-by-event correspondence
        is required and it survives EPOS files that have been filtered or
        re-ordered.

        ``"pmass_match"``: same algorithm but uses the Cameca pMass histogram
        stored inside the RHIT file as the reference. **No EPOS file is
        needed.** This is the most reliable spectrum-based mode because pMass
        is IVAS's own fully calibrated spectrum baked into the run file.

        ``"event_match"``: the MATLAB-equivalent algorithm — pair events by
        VDC + detector position and fit the bowl polynomial from those pairs.
        Most accurate when RHIT and EPOS share an event-by-event ordering, but
        fragile on filtered EPOS.

    metadata, rhit_histograms : dict, optional
        The metadata and histograms dicts returned by :func:`rhit_load`. Used
        by spectrum-based modes to seed the optimiser and (for ``pmass_match``)
        as the reference histogram.
    """
    method = str(method).lower()

    epos_df: pd.DataFrame | None
    if method == "pmass_match":
        epos_df = None
    else:
        if epos is None:
            raise ValueError(f"method={method!r} requires an EPOS path or dataframe")
        epos_df = _normalize_epos_calibration_dataframe(epos)

    hits_df = hits.copy()
    if len(hits_df) == 0:
        raise ValueError("RHIT input has no events")
    if epos_df is not None and len(epos_df) == 0:
        raise ValueError("EPOS input has no events")

    if verbose:
        epos_n = 0 if epos_df is None else len(epos_df)
        print(f"  Calibrating: {len(hits_df):,} RHIT events vs {epos_n:,} EPOS events  (method={method})")
        head = min(200, len(hits_df))
        rhit_head = hits_df.iloc[:head]
        print(
            f"  RHIT first {head}: VDC=[{rhit_head['VDC'].min():.1f},{rhit_head['VDC'].max():.1f}] V, "
            f"detx=[{rhit_head['detx'].min():.2f},{rhit_head['detx'].max():.2f}] mm, "
            f"dety=[{rhit_head['dety'].min():.2f},{rhit_head['dety'].max():.2f}] mm"
        )
        if epos_df is not None:
            epos_head = epos_df.iloc[: min(200, len(epos_df))]
            print(
                f"  EPOS first {len(epos_head)}: VDC=[{epos_head['VDC'].min():.1f},{epos_head['VDC'].max():.1f}] V, "
                f"detx=[{epos_head['detx'].min():.2f},{epos_head['detx'].max():.2f}] mm, "
                f"dety=[{epos_head['dety'].min():.2f},{epos_head['dety'].max():.2f}] mm"
            )

    if method == "event_match":
        calibration = _calibrate_via_event_match(hits_df, epos_df, verbose=verbose)
    elif method == "spectrum_match":
        calibration = _calibrate_via_spectrum_match(
            hits_df, epos_df, metadata=metadata or {}, verbose=verbose
        )
    elif method == "pmass_match":
        calibration = _calibrate_via_pmass_match(
            hits_df, rhit_histograms or {}, metadata=metadata or {}, verbose=verbose
        )
    else:
        raise ValueError(
            f"Unknown calibration method {method!r}. "
            f"Use 'spectrum_match', 'pmass_match', or 'event_match'."
        )

    calibrated_hits = rhit_apply_calibration(hits_df, calibration)
    return calibrated_hits, calibration
