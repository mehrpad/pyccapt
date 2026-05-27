"""Calibration-data filtering: build masks/weights so multihit, DC,
and dissociation tracks don't steer the V+Bowl+drift fits.

The pyccapt HDF5 carries three useful columns for this:

  - ``ion_pp``         : ions-per-pulse (multiplicity). ion_pp == 1 means
                         a clean single-hit event.
  - ``pulse_pi``       : pulse index. Ions sharing a pulse_pi were
                         evaporated by the same laser shot.
  - ``start_counter``  : TOF start-time counter; another grouping signal
                         for correlated events.

Key principle: **never drop ions from the dataset**. Build a calibration
mask/weight, fit the correction on the clean subset, then apply the
correction to ALL ions. The mask/weight is consumed by ``gaussian_mrp_report``,
``auto_detect_reference_peaks`` and the reference-constrained optimizer
via a uniform ``data`` and ``policy`` argument.

Public API
----------

``build_calibration_mask(data, policy="single_hit_only")``
    Return a boolean array of length ``len(data)``. True ions count for
    calibration; False ones are ignored.

``build_calibration_weights(data, policy="downweight_multihit")``
    Return a float array of length ``len(data)`` (weights in [0, 1]).
    Designed to be multiplied into histogram counts during scoring.

``filter_for_mrp(values, data, policy="single_hit_only")``
    Convenience: returns the calibration-trustworthy subset of
    ``values`` according to the chosen policy. Equivalent to
    ``values[build_calibration_mask(data, policy)]``.

Policies
--------

``"none"``                       (default in legacy paths)
    Every ion participates. Includes multihit, DC, dissociation tracks.

``"single_hit_only"``
    Keep ions where ``ion_pp == 1``. Drops ~13% of ions on the tested
    Nimonic NiC1 dataset, removes most dissociation/correlation
    contamination. Safe default for clean datasets.

``"downweight_multihit"``
    Weight = 1 / sqrt(max(ion_pp, 1)). Single-hit ions get full weight,
    doubles get 0.71, triples 0.58, etc. Use when single-hit data is
    too sparse to fit a reliable correction.

``"exclude_correlation_tracks"``
    Group by ``pulse_pi``. For events with ion_pp > 1, identify
    pairs whose mass-to-charge values fall on the
    "mc_i + mc_j = const" diagonal — these are dissociation tracks
    (one parent molecule split into two daughter ions). Exclude both
    daughters from calibration scoring. Combined with single-hit
    filter under the hood, then dissociation exclusion on top of the
    multihit events that survive.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    pd = None
    _HAS_PANDAS = False


VALID_POLICIES = (
    "none",
    "single_hit_only",
    "downweight_multihit",
    "exclude_correlation_tracks",
)


def _as_dataframe_or_none(data):
    if data is None:
        return None
    if _HAS_PANDAS and isinstance(data, pd.DataFrame):
        return data
    # Allow a Variables instance: it carries a ``.data`` DataFrame
    candidate = getattr(data, "data", None)
    if _HAS_PANDAS and isinstance(candidate, pd.DataFrame):
        return candidate
    return None


def _ion_pp_series(data):
    df = _as_dataframe_or_none(data)
    if df is None or "ion_pp" not in df.columns:
        return None
    return np.asarray(df["ion_pp"].to_numpy(), dtype=np.int64)


def _length(data) -> int:
    df = _as_dataframe_or_none(data)
    if df is not None:
        return int(len(df))
    if hasattr(data, "dld_t"):
        return int(len(data.dld_t))
    return 0


# ---------------------------------------------------------------------------
# Public API


def build_calibration_mask(data, policy: str = "single_hit_only") -> np.ndarray:
    """Return a boolean mask of ions trustworthy for calibration scoring.

    Length equals the number of ions in the dataset. True = keep, False = ignore.
    Defaults to "single_hit_only" because that's the conservative starting
    point; explicit "none" returns all-True if the legacy behavior is desired.

    Falls back to all-True when the dataset lacks the required columns.
    """
    if policy not in VALID_POLICIES:
        raise ValueError(f"Unknown policy {policy!r}; valid: {VALID_POLICIES}")
    n = _length(data)
    if n == 0:
        return np.ones(0, dtype=bool)
    if policy == "none":
        return np.ones(n, dtype=bool)

    ion_pp = _ion_pp_series(data)
    if ion_pp is None:
        # Column missing — degrade to "none" gracefully so existing
        # pipelines don't crash on legacy datasets.
        return np.ones(n, dtype=bool)
    if len(ion_pp) != n:
        return np.ones(n, dtype=bool)

    if policy == "single_hit_only":
        return ion_pp <= 1
    if policy == "downweight_multihit":
        # Mask is "is this ion non-zero-weight?" — all are, so all True.
        return np.ones(n, dtype=bool)
    if policy == "exclude_correlation_tracks":
        # Start with single-hit; then layer dissociation-track exclusion.
        mask = ion_pp <= 1
        df = _as_dataframe_or_none(data)
        if df is None or "pulse_pi" not in df.columns or "mc_uc (Da)" not in df.columns:
            return mask
        diss = _detect_dissociation_tracks(df, ion_pp)
        # diss is a boolean of ions to EXCLUDE from calibration scoring
        mask &= ~diss
        return mask
    return np.ones(n, dtype=bool)


def build_calibration_weights(data, policy: str = "downweight_multihit") -> np.ndarray:
    """Return per-ion weights for calibration scoring.

    Weights are in [0, 1]. Multiply into histogram counts during MRP /
    reference-line scoring. Length equals number of ions.

    Defaults to ``downweight_multihit`` because that preserves all the
    ions but de-emphasises noisy ones.
    """
    if policy not in VALID_POLICIES:
        raise ValueError(f"Unknown policy {policy!r}; valid: {VALID_POLICIES}")
    n = _length(data)
    if n == 0:
        return np.ones(0, dtype=float)
    if policy == "none":
        return np.ones(n, dtype=float)

    ion_pp = _ion_pp_series(data)
    if ion_pp is None:
        return np.ones(n, dtype=float)
    if len(ion_pp) != n:
        return np.ones(n, dtype=float)

    if policy == "single_hit_only":
        return (ion_pp <= 1).astype(float)
    if policy == "downweight_multihit":
        # 1 / sqrt(max(ion_pp, 1))
        clipped = np.maximum(ion_pp, 1).astype(float)
        return 1.0 / np.sqrt(clipped)
    if policy == "exclude_correlation_tracks":
        mask = build_calibration_mask(data, policy="exclude_correlation_tracks")
        return mask.astype(float)
    return np.ones(n, dtype=float)


def filter_for_mrp(values, data, policy: str = "single_hit_only") -> np.ndarray:
    """Return ``values[mask]`` where mask comes from ``build_calibration_mask``.

    Convenience wrapper used by the MRP scorers when they want to operate
    on a clean subset.
    """
    mask = build_calibration_mask(data, policy=policy)
    arr = np.asarray(values)
    if len(mask) != len(arr):
        return arr
    return arr[mask]


# ---------------------------------------------------------------------------
# Dissociation-track detection (used by "exclude_correlation_tracks")


def _detect_dissociation_tracks(
    df, ion_pp, mc_column: str = "mc_uc (Da)", track_tol: float = 0.5,
) -> np.ndarray:
    """Flag ions on dissociation tracks (mc_i + mc_j ≈ const within a pulse).

    Returns a boolean array of length ``len(df)``; True means the ion
    is part of a likely dissociation pair and should be excluded from
    calibration scoring.

    Heuristic (light, single-pass):
      - For each pulse with ion_pp == 2, compute mc_i + mc_j.
      - Build a histogram of these sums across all such pulses.
      - Sums that fall on a sharp peak (above the median + 5 sigma) are
        dissociation tracks; flag BOTH ions in those pulses.

    This catches the "parent molecule split into 2 daughters" case which
    is the dominant dissociation pattern in laser-pulsed APT. Triple-hit
    and higher are not handled here; downweight_multihit picks them up.
    """
    n = len(df)
    flagged = np.zeros(n, dtype=bool)
    if "pulse_pi" not in df.columns or mc_column not in df.columns:
        return flagged
    mc = df[mc_column].to_numpy(dtype=float)
    pulse = df["pulse_pi"].to_numpy(dtype=np.int64)
    multi_mask = ion_pp == 2
    if not multi_mask.any():
        return flagged
    # Group rows by pulse_pi where ion_pp == 2
    idx = np.where(multi_mask)[0]
    sub_pulse = pulse[idx]
    order = np.argsort(sub_pulse)
    idx_sorted = idx[order]
    sub_pulse_sorted = sub_pulse[order]
    # Group boundaries: consecutive ions sharing the same pulse_pi
    sums = []
    pair_indices = []
    i = 0
    n_sub = len(idx_sorted)
    while i < n_sub - 1:
        if sub_pulse_sorted[i] == sub_pulse_sorted[i + 1]:
            a, b = idx_sorted[i], idx_sorted[i + 1]
            sums.append(mc[a] + mc[b])
            pair_indices.append((a, b))
            i += 2
        else:
            i += 1
    if not sums:
        return flagged
    sums_arr = np.asarray(sums, dtype=float)
    # Histogram of sums; dissociation tracks form sharp peaks
    if sums_arr.size < 50:
        return flagged
    bins = np.arange(sums_arr.min(), sums_arr.max() + track_tol, track_tol)
    hist, edges = np.histogram(sums_arr, bins=bins)
    if hist.size < 4 or hist.sum() == 0:
        return flagged
    median = float(np.median(hist[hist > 0]))
    threshold = median + 5.0 * float(np.sqrt(max(median, 1.0)))
    hot_bins = np.where(hist > threshold)[0]
    if hot_bins.size == 0:
        return flagged
    # Which sums fall in hot bins?
    sum_bin = np.clip(np.searchsorted(edges, sums_arr, side="right") - 1, 0, len(hist) - 1)
    pair_in_track = np.isin(sum_bin, hot_bins)
    for keep, (a, b) in zip(pair_in_track, pair_indices):
        if keep:
            flagged[a] = True
            flagged[b] = True
    return flagged


# ---------------------------------------------------------------------------
# Diagnostics


def policy_diagnostics(data, policy: str) -> dict:
    """Return summary stats for the given policy on ``data``.

    Useful for printing what fraction of ions a policy keeps and how
    the weights are distributed.
    """
    n = _length(data)
    if n == 0:
        return {"n_ions": 0, "policy": policy}
    mask = build_calibration_mask(data, policy=policy)
    weights = build_calibration_weights(data, policy=policy)
    return {
        "n_ions": n,
        "policy": policy,
        "mask_kept": int(mask.sum()),
        "mask_kept_fraction": float(mask.sum() / max(n, 1)),
        "weight_sum": float(weights.sum()),
        "weight_mean": float(weights.mean()),
    }
