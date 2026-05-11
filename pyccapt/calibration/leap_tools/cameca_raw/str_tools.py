"""STR and HITS import helpers for Cameca LEAP raw data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ELECTRON_CHARGE_C = 1.602176634e-19
ATOMIC_MASS_UNIT_KG = 1.66053906660e-27


def _int24_little_endian_to_int32(raw4: np.ndarray) -> np.ndarray:
    values = raw4[:, 0].astype(np.int32)
    values += raw4[:, 1].astype(np.int32) * 256
    values += raw4[:, 2].astype(np.int32) * 65536
    sign_mask = values >= 8388608
    values[sign_mask] -= 16777216
    return values


def _parse_str_header(tag_vec: np.ndarray, value_vec: np.ndarray) -> dict[str, Any]:
    metadata: dict[str, Any] = {}

    metadata["version"] = int(value_vec[0] & 255) if len(tag_vec) and int(tag_vec[0]) == 0xA0 else 0

    labels = []
    for tag in range(0x27, 0x2D):
        match = np.where(tag_vec[: min(50, len(tag_vec))] == tag)[0]
        labels.append(chr(int(value_vec[match[0]])) if len(match) else "\x00")
    metadata["detectorLabels"] = "".join(labels)

    thresholds = []
    for tag in range(0x2D, 0x30):
        match = np.where(tag_vec[: min(50, len(tag_vec))] == tag)[0]
        thresholds.append(int(value_vec[match[0]]) if len(match) else 0)
    metadata["thresholds"] = thresholds

    walk = []
    for tag in range(0x33, 0x39):
        match = np.where(tag_vec[: min(50, len(tag_vec))] == tag)[0]
        walk.append(int(value_vec[match[0]]) if len(match) else 0)
    metadata["walkCorrections"] = walk

    voltage_match = np.where(tag_vec[: min(100, len(tag_vec))] == 0x1B)[0]
    if len(voltage_match):
        metadata["voltage"] = float(value_vec[voltage_match[0]])

    return metadata


def _extract_hits_v3(tag_vec: np.ndarray, value_vec: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    hit_tag_lo = np.array([0x3B, 0x3C, 0x3D, 0x3E, 0x3F], dtype=np.uint8)
    hit_tag_hi = np.array([0x4C, 0x4D, 0x4E, 0x4F, 0x50], dtype=np.uint8)

    event_positions = np.where(tag_vec == 0x3B)[0]
    if len(event_positions) == 0:
        return pd.DataFrame(), {"nEvents": 0, "note": "No HITS v3 header events were found"}

    ch_lo = np.full((len(event_positions), 5), np.nan, dtype=float)
    ch_hi = np.full((len(event_positions), 5), np.nan, dtype=float)

    for event_index, start in enumerate(event_positions):
        stop = min(start + 16, len(tag_vec))
        for field_index in range(start, stop):
            tag = tag_vec[field_index]
            for channel_index in range(5):
                if tag == hit_tag_lo[channel_index]:
                    ch_lo[event_index, channel_index] = float(value_vec[field_index])
                elif tag == hit_tag_hi[channel_index]:
                    ch_hi[event_index, channel_index] = float(value_vec[field_index])

    ch_full = ch_lo.copy()
    has_hi = np.isfinite(ch_hi) & (ch_hi != 0)
    ch_full[has_hi] = ch_lo[has_hi] + ch_hi[has_hi] * 16777216.0

    valid = np.isfinite(ch_full[:, 0]) & np.isfinite(ch_full[:, 3])
    cv = ch_full[valid]
    hits = pd.DataFrame(
        {
            "ionIdx": np.arange(1, len(cv) + 1, dtype=int),
            "detx": cv[:, 0],
            "dety": cv[:, 1],
            "tof": cv[:, 3],
            "quality": np.zeros(len(cv)),
            "ch1": cv[:, 0],
            "ch2": cv[:, 1],
            "ch3": cv[:, 2],
            "ch4": cv[:, 3],
            "ch5": cv[:, 4],
        }
    )
    metadata = {
        "nEvents": int(len(cv)),
        "note": "HITS v3: only header-region TLV events were decoded; bulk data remains unsupported.",
    }
    return hits, metadata


def str_load(file_path: str | Path, verbose: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a Cameca STR or HITS file into a dataframe of raw delay-line timings.

    With ``verbose=True`` (default), prints the file's TLV-tag inventory, the
    detected version, and per-channel raw event counts so users can see what
    the binary parser found before any downstream calibration.
    """
    file_path = Path(file_path)
    raw = np.fromfile(file_path, dtype=np.uint8)
    n_fields = len(raw) // 4
    if n_fields == 0:
        raise ValueError(f"File does not contain any 4-byte TLV records: {file_path}")

    raw4 = raw[: n_fields * 4].reshape(-1, 4)
    tag_vec = raw4[:, 3]
    value_vec = _int24_little_endian_to_int32(raw4)

    metadata = _parse_str_header(tag_vec, value_vec)
    metadata["fileName"] = str(file_path)
    metadata["nFields"] = int(n_fields)

    if verbose:
        print(f"  STR/HITS file: {file_path}")
        print(f"  File size: {file_path.stat().st_size:,} bytes; TLV records: {n_fields:,}")
        print(f"  Header version byte: 0x{int(metadata.get('version', 0)):02x} "
              f"({'STR v2' if metadata.get('version') == 2 else 'HITS v3' if metadata.get('version') == 3 else 'unknown'})")
        labels = metadata.get("detectorLabels", "").replace("\x00", "?")
        print(f"  Detector channel labels: {labels!r}")
        print(f"  Thresholds: {metadata.get('thresholds')}")
        print(f"  Walk corrections: {metadata.get('walkCorrections')}")
        if "voltage" in metadata:
            print(f"  Initial specimen voltage: {metadata['voltage']:.1f} V")
        unique_tags, tag_counts = np.unique(tag_vec, return_counts=True)
        order = np.argsort(-tag_counts)
        print("  Per-tag occurrence counts (top 25 by frequency):")
        named = {
            0x01: "detxt1", 0x02: "detxt2", 0x03: "detyt1", 0x04: "detyt2",
            0x21: "detwt1", 0x22: "detwt2", 0x18: "quality/event-end",
            0x05: "pulse-counter-A", 0x0B: "pulse-counter-B",
            0xA0: "version", 0x1B: "voltage",
        }
        for slot in order[:25]:
            tag_value = int(unique_tags[slot])
            count = int(tag_counts[slot])
            label = named.get(tag_value, "")
            label_part = f" ({label})" if label else ""
            print(f"    tag 0x{tag_value:02x}{label_part}: {count:,}")

    if metadata.get("version") == 3:
        hits, extra_metadata = _extract_hits_v3(tag_vec, value_vec)
        metadata.update(extra_metadata)
        if verbose:
            print(f"  HITS v3: extracted {len(hits):,} initial events from header region only.")
        return hits, metadata

    is_end = tag_vec == 0x18
    event_id = np.cumsum(is_end)
    event_id = event_id - is_end + 1
    n_events = int(event_id.max())

    channel_tags = np.array([0x01, 0x02, 0x03, 0x04, 0x21, 0x22], dtype=np.uint8)
    channel_data = np.full((n_events, 6), np.nan, dtype=float)
    for column_index, channel_tag in enumerate(channel_tags):
        mask = tag_vec == channel_tag
        if not np.any(mask):
            continue
        indexes = np.where(mask)[0]
        events = event_id[indexes] - 1
        channel_data[events, column_index] = value_vec[indexes].astype(float)

    quality_data = value_vec[is_end].astype(float)
    if len(quality_data) < n_events:
        quality_data = np.pad(quality_data, (0, n_events - len(quality_data)))

    has_any_channel = np.any(np.isfinite(channel_data), axis=1)
    channel_data = channel_data[has_any_channel]
    quality_data = quality_data[has_any_channel]

    hits = pd.DataFrame(
        {
            "ionIdx": np.arange(1, len(channel_data) + 1, dtype=int),
            "detxt1": channel_data[:, 0],
            "detxt2": channel_data[:, 1],
            "detyt1": channel_data[:, 2],
            "detyt2": channel_data[:, 3],
            "detwt1": channel_data[:, 4],
            "detwt2": channel_data[:, 5],
            "quality": quality_data,
        }
    )
    metadata["nEvents"] = int(len(hits))
    metadata["nFull6Channels"] = int(np.sum(np.all(np.isfinite(channel_data), axis=1)))
    if verbose:
        per_channel_counts = {
            "detxt1": int(np.sum(np.isfinite(channel_data[:, 0]))),
            "detxt2": int(np.sum(np.isfinite(channel_data[:, 1]))),
            "detyt1": int(np.sum(np.isfinite(channel_data[:, 2]))),
            "detyt2": int(np.sum(np.isfinite(channel_data[:, 3]))),
            "detwt1": int(np.sum(np.isfinite(channel_data[:, 4]))),
            "detwt2": int(np.sum(np.isfinite(channel_data[:, 5]))),
        }
        print(f"  Real events kept (>=1 channel): {len(hits):,} of {n_events:,} 0x18 markers")
        print(f"  Events with all 6 channels: {metadata['nFull6Channels']:,}")
        print(f"  Per-channel finite counts: {per_channel_counts}")
        print(f"  Output dataframe columns: {list(hits.columns)}")
    return hits, metadata


def _safe_geometry_fit(detx_raw: np.ndarray, dety_raw: np.ndarray, detw_raw: np.ndarray) -> tuple[float, float, float]:
    if len(detx_raw) >= 10:
        design = np.column_stack((detx_raw, dety_raw, np.ones(len(detx_raw))))
        fit, *_ = np.linalg.lstsq(design, detw_raw, rcond=None)
        return float(fit[0]), float(fit[1]), float(fit[2])
    c = float(np.nanmedian(detw_raw - (detx_raw + dety_raw) / np.sqrt(2.0))) if len(detw_raw) else 0.0
    return float(1.0 / np.sqrt(2.0)), float(1.0 / np.sqrt(2.0)), c


def str_calculate_positions(hits: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Compute raw detector positions and TDC TOF from STR/HITS delay-line timings."""
    result = hits.copy()

    has_x = np.isfinite(result["detxt1"]) & np.isfinite(result["detxt2"])
    has_y = np.isfinite(result["detyt1"]) & np.isfinite(result["detyt2"])
    has_w = np.isfinite(result["detwt1"]) & np.isfinite(result["detwt2"])

    detx_raw = np.full(len(result), np.nan, dtype=float)
    dety_raw = np.full(len(result), np.nan, dtype=float)
    detw_raw = np.full(len(result), np.nan, dtype=float)
    detx_raw[has_x] = result.loc[has_x, "detxt1"].to_numpy(dtype=float) - result.loc[has_x, "detxt2"].to_numpy(dtype=float)
    dety_raw[has_y] = result.loc[has_y, "detyt1"].to_numpy(dtype=float) - result.loc[has_y, "detyt2"].to_numpy(dtype=float)
    detw_raw[has_w] = result.loc[has_w, "detwt1"].to_numpy(dtype=float) - result.loc[has_w, "detwt2"].to_numpy(dtype=float)

    full = has_x & has_y & has_w
    sum_x = result["detxt1"].to_numpy(dtype=float) + result["detxt2"].to_numpy(dtype=float)
    sum_y = result["detyt1"].to_numpy(dtype=float) + result["detyt2"].to_numpy(dtype=float)
    sum_w = result["detwt1"].to_numpy(dtype=float) + result["detwt2"].to_numpy(dtype=float)

    fit_indexes = np.where(full)[0]
    if len(fit_indexes):
        d_xy = sum_x[fit_indexes] - sum_y[fit_indexes]
        d_xw = sum_x[fit_indexes] - sum_w[fit_indexes]
        median_xy = np.nanmedian(d_xy)
        median_xw = np.nanmedian(d_xw)
        sum_good = np.abs(d_xy - median_xy) < 50.0
        sum_good &= np.abs(d_xw - median_xw) < 50.0
        fit_indexes = fit_indexes[sum_good]
    if len(fit_indexes) > 500000:
        fit_indexes = fit_indexes[:500000]

    a, b, c = _safe_geometry_fit(detx_raw[fit_indexes], dety_raw[fit_indexes], detw_raw[fit_indexes])

    miss_x = (~has_x) & has_y & has_w
    miss_y = has_x & (~has_y) & has_w
    miss_w = has_x & has_y & (~has_w)
    if np.any(miss_x) and abs(a) > 1e-12:
        detx_raw[miss_x] = (detw_raw[miss_x] - b * dety_raw[miss_x] - c) / a
    if np.any(miss_y) and abs(b) > 1e-12:
        dety_raw[miss_y] = (detw_raw[miss_y] - a * detx_raw[miss_y] - c) / b
    if np.any(miss_w):
        detw_raw[miss_w] = a * detx_raw[miss_w] + b * dety_raw[miss_w] + c

    if len(fit_indexes):
        mean_sum = (sum_x[fit_indexes] + sum_y[fit_indexes] + sum_w[fit_indexes]) / 3.0
        off_x = float(np.nanmedian(sum_x[fit_indexes] - mean_sum))
        off_y = float(np.nanmedian(sum_y[fit_indexes] - mean_sum))
        off_w = float(np.nanmedian(sum_w[fit_indexes] - mean_sum))
    else:
        off_x = 0.0
        off_y = 0.0
        off_w = 0.0

    tof = np.full(len(result), np.nan, dtype=float)
    n_delay_lines = np.zeros(len(result), dtype=int)
    tof_accum = np.zeros(len(result), dtype=float)

    if np.any(has_x):
        corr_x = (sum_x - off_x) / 2.0
        tof_accum[has_x] += corr_x[has_x]
        n_delay_lines[has_x] += 1
    if np.any(has_y):
        corr_y = (sum_y - off_y) / 2.0
        tof_accum[has_y] += corr_y[has_y]
        n_delay_lines[has_y] += 1
    if np.any(has_w):
        corr_w = (sum_w - off_w) / 2.0
        tof_accum[has_w] += corr_w[has_w]
        n_delay_lines[has_w] += 1
    has_any_tof = n_delay_lines > 0
    tof[has_any_tof] = tof_accum[has_any_tof] / n_delay_lines[has_any_tof]

    result["detxRaw"] = detx_raw
    result["detyRaw"] = dety_raw
    result["detwRaw"] = detw_raw
    result["tof"] = tof
    result["hitType"] = (has_x.astype(np.uint8) + has_y.astype(np.uint8) + has_w.astype(np.uint8)).astype(np.uint8)
    result["geometry_a"] = a
    result["geometry_b"] = b
    result["geometry_c"] = c
    result["tof_offset_x"] = off_x
    result["tof_offset_y"] = off_y
    result["tof_offset_w"] = off_w
    if verbose:
        hit_type_counts = dict(pd.Series(result["hitType"].to_numpy()).value_counts().sort_index().items())
        print(
            f"  STR positions: hitType counts={hit_type_counts}, "
            f"geometry w = {a:.4f}*x + {b:.4f}*y + {c:.1f}, "
            f"TOF DL offsets x={off_x:+.1f}, y={off_y:+.1f}, w={off_w:+.1f}"
        )
        print(
            f"  Rows with detx/dety: {int(np.sum(np.isfinite(detx_raw) & np.isfinite(dety_raw))):,}; "
            f"with TOF: {int(np.sum(np.isfinite(tof))):,}"
        )
    return result


def calculate_str_positions(hits: pd.DataFrame) -> pd.DataFrame:
    """Alias for :func:`str_calculate_positions`."""
    return str_calculate_positions(hits)


def _normalize_rhit_inputs(
    rhit_hits: pd.DataFrame,
    rhit_histograms: dict[str, Any] | None,
    rhit_metadata: dict[str, Any] | None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    histograms = {} if rhit_histograms is None else rhit_histograms
    metadata = {} if rhit_metadata is None else rhit_metadata
    if "instrumentParams" not in metadata:
        metadata["instrumentParams"] = {}
    return rhit_hits.copy(), histograms, metadata


def _mc_correlation_cost(parameters: np.ndarray, tof_tdc: np.ndarray, vdc: np.ndarray, constant: float,
                         reference_hist: np.ndarray, mc_edges: np.ndarray) -> float:
    t0_tdc = parameters[0]
    clock_ns = parameters[1]
    if clock_ns <= 0.0:
        return np.inf
    trial_mc = constant * vdc * ((tof_tdc - t0_tdc) * clock_ns) ** 2
    hist, _ = np.histogram(trial_mc[np.isfinite(trial_mc)], bins=mc_edges)
    if np.std(hist) == 0 or np.std(reference_hist) == 0:
        return np.inf
    corr = np.corrcoef(hist.astype(float), reference_hist.astype(float))[0, 1]
    if not np.isfinite(corr):
        return np.inf
    return -float(corr)


def estimate_str_detector_scale_from_rhit(str_hits: pd.DataFrame, rhit_hits: pd.DataFrame) -> dict[str, float]:
    """Estimate a linear STR raw-count to RHIT millimeter detector mapping."""
    valid = np.isfinite(str_hits["detxRaw"]) & np.isfinite(str_hits["detyRaw"]) & (str_hits["hitType"] >= 2)
    if not np.any(valid) or len(rhit_hits) == 0:
        return {"x_scale": 0.0, "x_offset": 0.0, "y_scale": 0.0, "y_offset": 0.0}

    str_subset = str_hits.loc[valid].copy()
    fractions = (np.arange(len(str_subset), dtype=float) + 0.5) / len(str_hits)
    rhit_indexes = np.clip(np.round(fractions * (len(rhit_hits) - 1)).astype(int), 0, len(rhit_hits) - 1)
    rhit_subset = rhit_hits.iloc[rhit_indexes]

    def _fit_axis(raw_values: np.ndarray, target_values: np.ndarray) -> tuple[float, float]:
        finite = np.isfinite(raw_values) & np.isfinite(target_values)
        finite &= np.abs(raw_values) > 1e-12
        if np.count_nonzero(finite) < 20:
            return 0.0, 0.0
        design = np.column_stack((raw_values[finite], np.ones(np.count_nonzero(finite))))
        fit, *_ = np.linalg.lstsq(design, target_values[finite], rcond=None)
        return float(fit[0]), float(fit[1])

    x_scale, x_offset = _fit_axis(str_subset["detxRaw"].to_numpy(dtype=float), rhit_subset["detx"].to_numpy(dtype=float))
    y_scale, y_offset = _fit_axis(str_subset["detyRaw"].to_numpy(dtype=float), rhit_subset["dety"].to_numpy(dtype=float))
    return {
        "x_scale": x_scale,
        "x_offset": x_offset,
        "y_scale": y_scale,
        "y_offset": y_offset,
    }


def str_calibrate_from_rhit(
    hits: pd.DataFrame,
    rhit_hits: pd.DataFrame,
    rhit_histograms: dict[str, Any] | None = None,
    rhit_metadata: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Calibrate STR TOF and mass-to-charge using a matching RHIT file."""
    str_hits = hits.copy()
    rhit_hits, rhit_histograms, rhit_metadata = _normalize_rhit_inputs(rhit_hits, rhit_histograms, rhit_metadata)
    if len(str_hits) == 0 or len(rhit_hits) == 0:
        raise ValueError("STR and RHIT inputs must both contain events for calibration")

    fractions = (np.arange(len(str_hits), dtype=float) + 1.0) / max(len(str_hits), 1)
    rhit_indexes = np.clip(np.round(fractions * len(rhit_hits)).astype(int) - 1, 0, len(rhit_hits) - 1)
    str_hits["VDC"] = rhit_hits.iloc[rhit_indexes]["VDC"].to_numpy(dtype=float)

    instrument_params = rhit_metadata.get("instrumentParams", {})
    flight_path_m = float(instrument_params.get("flight_path_mm", 382.0)) / 1000.0
    if "Vref" in rhit_hits.columns and np.any(np.abs(rhit_hits["VDC"].to_numpy(dtype=float)) > 0):
        valid_v = np.abs(rhit_hits["VDC"].to_numpy(dtype=float)) > 0
        kf = float(np.median(rhit_hits.loc[valid_v, "Vref"] / rhit_hits.loc[valid_v, "VDC"]))
    else:
        kf = 1.03
    constant = 2.0 * ELECTRON_CHARGE_C / (ATOMIC_MASS_UNIT_KG * flight_path_m**2) * 1e-18 / np.sqrt(kf)

    mc_edges = np.arange(0.0, 120.05, 0.05)
    if "massSpectrum" in rhit_histograms and "values" in rhit_histograms["massSpectrum"]:
        reference_hist = np.asarray(rhit_histograms["massSpectrum"]["values"], dtype=float)
        reference_edges = np.asarray(rhit_histograms["massSpectrum"]["edges"], dtype=float)
        if len(reference_edges) == len(reference_hist) + 1:
            mc_edges = reference_edges
    else:
        reference_hist, _ = np.histogram(rhit_hits["mc"].to_numpy(dtype=float), bins=mc_edges)

    stride = 10 if len(str_hits) > 5000 else 1
    tof_sub = str_hits["tof"].to_numpy(dtype=float)[::stride]
    vdc_sub = str_hits["VDC"].to_numpy(dtype=float)[::stride]
    valid = np.isfinite(tof_sub) & np.isfinite(vdc_sub)
    tof_sub = tof_sub[valid]
    vdc_sub = vdc_sub[valid]
    if len(tof_sub) < 100:
        raise ValueError("Too few STR events with valid TOF survived for calibration")

    best_cost = np.inf
    best_parameters = np.array([0.0, 0.03], dtype=float)
    for t0_guess in np.linspace(-5000.0, 30000.0, 20):
        for clock_guess in (0.025, 0.030, 0.035, 0.040, 0.050):
            result = minimize(
                _mc_correlation_cost,
                np.array([t0_guess, clock_guess], dtype=float),
                args=(tof_sub, vdc_sub, constant, reference_hist, mc_edges),
                method="Nelder-Mead",
                options={"maxiter": 500, "xatol": 1e-12, "fatol": 1e-10},
            )
            if not result.success:
                continue
            t0_tdc, clock_ns = result.x
            if result.fun < best_cost and 0.01 < clock_ns < 0.1:
                best_cost = float(result.fun)
                best_parameters = result.x

    t0_tdc = float(best_parameters[0])
    clock_ns = float(best_parameters[1])
    str_hits["tof_ns"] = (str_hits["tof"].to_numpy(dtype=float) - t0_tdc) * clock_ns
    str_hits["mc"] = constant * str_hits["VDC"].to_numpy(dtype=float) * str_hits["tof_ns"].to_numpy(dtype=float) ** 2

    detector_scale = estimate_str_detector_scale_from_rhit(str_hits, rhit_hits)
    str_hits["detx"] = detector_scale["x_scale"] * str_hits["detxRaw"].to_numpy(dtype=float) + detector_scale["x_offset"]
    str_hits["dety"] = detector_scale["y_scale"] * str_hits["detyRaw"].to_numpy(dtype=float) + detector_scale["y_offset"]

    calibration = {
        "t0_tdc": t0_tdc,
        "clock_ns": clock_ns,
        "kf": float(kf),
        "flight_path_m": float(flight_path_m),
        "spectrum_correlation": float(-best_cost) if np.isfinite(best_cost) else float("nan"),
        "detector_scale": detector_scale,
    }
    return str_hits, calibration


def calibrate_str_from_rhit(
    hits: pd.DataFrame,
    rhit_hits: pd.DataFrame,
    rhit_histograms: dict[str, Any] | None = None,
    rhit_metadata: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Alias for :func:`str_calibrate_from_rhit`."""
    return str_calibrate_from_rhit(hits, rhit_hits, rhit_histograms, rhit_metadata)


def str_to_ccapt(hits: pd.DataFrame, drop_invalid: bool = True) -> pd.DataFrame:
    """Convert STR/HITS data into a processed PyCCAPT-style dataframe.

    With ``drop_invalid=True`` (default), rows with NaN ``detx`` / ``dety`` /
    ``mc`` / ``tof_ns`` / ``VDC`` are dropped — these can never produce useful
    downstream analysis.
    """
    required = {"mc", "tof_ns", "VDC", "detx", "dety"}
    missing = required.difference(hits.columns)
    if missing:
        raise ValueError(
            "STR data must be calibrated before conversion to a PyCCAPT dataset. "
            f"Missing columns: {sorted(missing)}"
        )

    if drop_invalid:
        valid = pd.Series(True, index=hits.index)
        for column in ("detx", "dety", "mc", "tof_ns", "VDC"):
            valid &= np.isfinite(hits[column].to_numpy(dtype=float))
        hits = hits.loc[valid]

    length = len(hits)
    start_counter = hits["ionIdx"].to_numpy(dtype=int) - 1 if "ionIdx" in hits.columns else np.arange(length, dtype=int)
    return pd.DataFrame(
        {
            "x (nm)": np.zeros(length),
            "y (nm)": np.zeros(length),
            "z (nm)": np.zeros(length),
            "mc (Da)": hits["mc"].to_numpy(dtype=float),
            "mc_uc (Da)": hits["mc"].to_numpy(dtype=float),
            "high_voltage (V)": hits["VDC"].to_numpy(dtype=float),
            "pulse_v (V)": np.zeros(length),
            "pulse_l (pJ)": np.zeros(length),
            "t (ns)": hits["tof_ns"].to_numpy(dtype=float),
            "t_c (ns)": hits["tof_ns"].to_numpy(dtype=float),
            "x_det (cm)": hits["detx"].to_numpy(dtype=float) / 10.0,
            "y_det (cm)": hits["dety"].to_numpy(dtype=float) / 10.0,
            "delta_p": np.zeros(length, dtype=int),
            "multi": np.ones(length, dtype=int),
            "start_counter": start_counter,
        }
    )


def _require_h5py():
    try:
        import h5py  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional runtime package
        raise ImportError(
            "STR raw-HDF5 export requires the optional 'h5py' package. "
            "Install it with 'pip install h5py'."
        ) from exc
    return h5py


# Channel-tag mapping for the STR delay-line ends.
# Order chosen so the raw-analysis Roentdek pipeline reads
# x-pair, then y-pair, then w-pair — matching its DLD convention.
_STR_CHANNEL_COLUMNS: tuple[tuple[int, str], ...] = (
    (0, "detxt1"),
    (1, "detxt2"),
    (2, "detyt1"),
    (3, "detyt2"),
    (4, "detwt1"),
    (5, "detwt2"),
)


def _build_str_tdc_table(
    hits: pd.DataFrame,
    *,
    pulse_mode: str,
) -> dict[str, np.ndarray]:
    """Flatten STR per-channel TDC counts into a single tdc-style hit table.

    For each event with a finite value in ``detxt1..detwt2``, emit a row with
    ``channel`` = 0..5, ``time_data`` = TDC count, ``start_counter`` = event
    index, ``high_voltage`` / ``pulse`` / ``laser_pulse`` = that event's value.

    Rows are returned in **event-major order** (all channel hits for event
    ``ionIdx=k`` are consecutive in the table, then all hits for ``k+1``, ...).
    This is the layout expected by
    :func:`pyccapt.calibration.data_tools.data_loadcrop.build_event_group_mapping`,
    which pairs each tdc run of equal ``start_counter`` with one dld row.
    """
    n = len(hits)
    if "ionIdx" in hits.columns:
        event_index = hits["ionIdx"].to_numpy(dtype=np.uint32)
    else:
        event_index = np.arange(n, dtype=np.uint32)
    high_voltage_per_event = (
        hits["VDC"].to_numpy(dtype=float)
        if "VDC" in hits.columns
        else np.zeros(n, dtype=float)
    )
    if pulse_mode == "laser":
        pulse_v_per_event = np.zeros(n, dtype=float)
        laser_per_event = (
            hits["laserpower"].to_numpy(dtype=float)
            if "laserpower" in hits.columns
            else np.zeros(n, dtype=float)
        )
    else:
        pulse_v_per_event = (
            hits["pulse"].to_numpy(dtype=float)
            if "pulse" in hits.columns
            else np.zeros(n, dtype=float)
        )
        laser_per_event = np.zeros(n, dtype=float)

    # Stack per-channel arrays as columns so we can iterate "down events, across
    # channels" without materializing six separate Python loops.
    channel_columns_present: list[tuple[int, str]] = [
        (idx, col) for idx, col in _STR_CHANNEL_COLUMNS if col in hits.columns
    ]
    if not channel_columns_present:
        return {}

    n_channels = len(channel_columns_present)
    values_matrix = np.full((n, n_channels), np.nan, dtype=float)
    channel_ids = np.empty(n_channels, dtype=np.uint32)
    for column_position, (channel_index, column) in enumerate(channel_columns_present):
        values_matrix[:, column_position] = hits[column].to_numpy(dtype=float)
        channel_ids[column_position] = channel_index
    finite_mask = np.isfinite(values_matrix)  # (n_events, n_channels)
    n_per_event = finite_mask.sum(axis=1)     # (n_events,) hits per event
    total_rows = int(n_per_event.sum())
    if total_rows == 0:
        return {}

    # Order of finite cells when scanned event-by-event, channel-by-channel.
    # np.where on a 2D array returns the row indices first, in row-major order,
    # which is exactly the event-major flattening we need.
    event_pos, channel_pos = np.where(finite_mask)
    return {
        "channel": channel_ids[channel_pos].reshape(-1, 1),
        "start_counter": event_index[event_pos].reshape(-1, 1).astype(np.uint32),
        "high_voltage": high_voltage_per_event[event_pos].reshape(-1, 1).astype(float),
        "pulse": pulse_v_per_event[event_pos].reshape(-1, 1).astype(float),
        "laser_pulse": laser_per_event[event_pos].reshape(-1, 1).astype(float),
        "time_data": values_matrix[event_pos, channel_pos].reshape(-1, 1).astype(np.uint32),
    }


def str_to_raw_hdf5(
    hits: pd.DataFrame,
    output_path: str | Path,
    *,
    pulse_mode: str = "voltage",
    drop_invalid: bool = True,
) -> Path:
    """Write `dld/` and `tdc/` groups to an HDF5 readable by the PyCCAPT raw-data workflow.

    * ``dld/...`` — already-decoded events (high_voltage, pulse, laser_intensity,
      start_counter, t, x, y). Requires calibrated columns
      (``VDC`` / ``tof_ns`` / ``detx`` / ``dety``) — produced by
      :func:`str_calibrate_from_rhit`. Loadable via
      ``fetch_dataset_from_dld_grp(..., extract_mode='dld')``.
    * ``tdc/...`` — per-DL-end TDC counts as a flat 6-channel hit stream
      (``channel`` 0=detxt1, 1=detxt2, 2=detyt1, 3=detyt2, 4=detwt1, 5=detwt2;
      ``time_data`` = TDC counts). Always written if the channel columns are
      present. Loadable via ``extract_mode='tdc_ro'`` or ``'tdc_sc'`` for
      Roentdek/Surface-Concept-style raw analysis.

    With ``drop_invalid=True`` (default), the ``dld/`` rows where ``detx`` /
    ``dety`` / ``tof_ns`` / ``VDC`` are NaN are dropped. The ``tdc/`` table
    naturally drops NaN per-channel since each row is one finite hit.
    """
    h5py = _require_h5py()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    has_calibrated = {"VDC", "tof_ns", "detx", "dety"}.issubset(hits.columns)

    if has_calibrated:
        if drop_invalid:
            valid = (
                np.isfinite(hits["VDC"].to_numpy(dtype=float))
                & np.isfinite(hits["tof_ns"].to_numpy(dtype=float))
                & np.isfinite(hits["detx"].to_numpy(dtype=float))
                & np.isfinite(hits["dety"].to_numpy(dtype=float))
            )
            keep = hits.loc[valid].reset_index(drop=True)
        else:
            keep = hits.reset_index(drop=True)
        n_valid = len(keep)
        n_dropped = len(hits) - n_valid
    else:
        keep = hits.reset_index(drop=True)
        n_valid = 0
        n_dropped = 0

    tdc_table = _build_str_tdc_table(hits.reset_index(drop=True), pulse_mode=pulse_mode)

    if not has_calibrated and not tdc_table:
        raise ValueError(
            "STR hits dataframe has neither calibrated columns "
            "(VDC, tof_ns, detx, dety) nor channel columns "
            "(detxt1..detwt2). Nothing to export."
        )

    with h5py.File(output_path, "w") as handle:
        if has_calibrated and n_valid > 0:
            high_voltage = keep["VDC"].to_numpy(dtype=float).reshape(-1, 1)
            tof_ns = keep["tof_ns"].to_numpy(dtype=float).reshape(-1, 1)
            det_x_cm = (keep["detx"].to_numpy(dtype=float) / 10.0).reshape(-1, 1)
            det_y_cm = (keep["dety"].to_numpy(dtype=float) / 10.0).reshape(-1, 1)
            pulse_v = np.zeros((n_valid, 1), dtype=float)
            pulse_l = np.zeros((n_valid, 1), dtype=float)
            if "ionIdx" in keep.columns:
                start_counter = keep["ionIdx"].to_numpy(dtype=np.uint32).reshape(-1, 1)
            else:
                start_counter = np.arange(n_valid, dtype=np.uint32).reshape(-1, 1)

            grp = handle.create_group("dld")
            grp.create_dataset("high_voltage", data=high_voltage, compression="gzip", compression_opts=4)
            grp.create_dataset("pulse", data=pulse_v, compression="gzip", compression_opts=4)
            grp.create_dataset("laser_intensity", data=pulse_l, compression="gzip", compression_opts=4)
            grp.create_dataset("start_counter", data=start_counter, compression="gzip", compression_opts=4)
            grp.create_dataset("t", data=tof_ns, compression="gzip", compression_opts=4)
            grp.create_dataset("x", data=det_x_cm, compression="gzip", compression_opts=4)
            grp.create_dataset("y", data=det_y_cm, compression="gzip", compression_opts=4)
            grp.attrs["num_entries"] = n_valid
            grp.attrs["num_dropped_nan"] = int(n_dropped)
            grp.attrs["units"] = (
                "high_voltage=V, pulse=V, laser_intensity=pJ, t=ns, x=cm, y=cm, start_counter=uint32"
            )
            grp.attrs["source"] = "str_to_raw_hdf5"
            grp.attrs["pulse_mode"] = pulse_mode

        if tdc_table:
            tdc = handle.create_group("tdc")
            for name, array in tdc_table.items():
                tdc.create_dataset(name, data=array, compression="gzip", compression_opts=4)
            tdc.attrs["num_entries"] = int(tdc_table["channel"].shape[0])
            tdc.attrs["channel_count"] = 6
            tdc.attrs["channel_map"] = (
                "0=detxt1, 1=detxt2, 2=detyt1, 3=detyt2, 4=detwt1, 5=detwt2"
            )
            tdc.attrs["units"] = (
                "channel=uint32, time_data=TDC counts (uint32), high_voltage=V, "
                "pulse=V, laser_pulse=pJ, start_counter=uint32"
            )
            tdc.attrs["source"] = "str_to_raw_hdf5"
            tdc.attrs["pulse_mode"] = pulse_mode

    return output_path
