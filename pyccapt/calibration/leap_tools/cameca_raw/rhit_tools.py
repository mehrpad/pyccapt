"""RHIT import and calibration helpers for Cameca LEAP raw data."""

from __future__ import annotations

import struct
import tempfile
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pyccapt.calibration.leap_tools import leap_tools

ELECTRON_CHARGE_C = 1.602176634e-19
ATOMIC_MASS_UNIT_KG = 1.66053906660e-27
DEFAULT_LSB_TO_MM = 18.5 / 750.0
DEFAULT_FLIGHT_PATH_M = 0.382
DEFAULT_T0_NS = 45.0
DEFAULT_KF = 1.03


def _require_uproot():
    try:
        import uproot  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional runtime package
        raise ImportError(
            "RHIT support requires the optional 'uproot' package. Install it with "
            "'pip install uproot'."
        ) from exc
    return uproot


def _require_h5py():
    try:
        import h5py  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional runtime package
        raise ImportError(
            "HDF5 export for RHIT support requires the optional 'h5py' package. "
            "Install it with 'pip install h5py'."
        ) from exc
    return h5py


def _select_latest_key(root_file, object_name: str):
    candidates = []
    for key in root_file.keys():
        if not key.startswith(f"{object_name};"):
            continue
        try:
            cycle = int(str(key).split(";")[1])
        except (IndexError, ValueError):
            cycle = 0
        candidates.append((cycle, key))
    if not candidates:
        return None
    return max(candidates)[1]


def _decompress_tkey(raw, offset: int) -> bytes:
    raw.seek(offset)
    nbytes = struct.unpack(">i", raw.read(4))[0]
    raw.seek(offset + 14)
    keylen = struct.unpack(">h", raw.read(2))[0]

    raw.seek(offset + keylen)
    compressed = raw.read(nbytes - keylen)

    chunks = []
    position = 0
    while position < len(compressed):
        if compressed[position:position + 2] == b"ZL":
            compressed_size = int.from_bytes(compressed[position + 3:position + 6], "little")
            chunk = zlib.decompress(compressed[position + 9:position + 9 + compressed_size])
            chunks.append(chunk)
            position += 9 + compressed_size
        else:
            chunks.append(compressed[position:])
            break
    return b"".join(chunks)


def _decode_run_header(file_path: str | Path) -> dict[str, Any]:
    with open(file_path, "rb") as raw:
        raw.seek(4)
        raw.read(4)
        file_begin = struct.unpack(">I", raw.read(4))[0]
        file_end = struct.unpack(">I", raw.read(4))[0]

        offset = file_begin
        best_offset = None
        best_cycle = -1

        while offset < file_end:
            raw.seek(offset)
            nbytes_raw = raw.read(4)
            if len(nbytes_raw) < 4:
                break
            nbytes = struct.unpack(">i", nbytes_raw)[0]
            if nbytes == 0:
                break
            if nbytes < 0:
                offset += abs(nbytes)
                continue

            raw.seek(offset)
            header = raw.read(min(nbytes, 200))
            if len(header) < 27:
                offset += nbytes
                continue

            cycle = struct.unpack(">h", header[16:18])[0]
            class_name_length = header[26]
            if 27 + class_name_length <= len(header):
                class_name = header[27:27 + class_name_length].decode("ascii", errors="replace")
                if class_name == "CRunHeader" and cycle > best_cycle:
                    best_cycle = cycle
                    best_offset = offset
            offset += nbytes

        if best_offset is None:
            return {}

        data = _decompress_tkey(raw, best_offset)

    rest = data[6:]
    params: dict[str, Any] = {}

    try:
        for index in range(30, 70):
            if rest[index:index + 1].isdigit():
                null_offset = rest[index:index + 20].find(b"\x00")
                end = index + null_offset if null_offset >= 0 else index + 15
                candidate = rest[index:end].decode("ascii", errors="replace")
                if "." in candidate and len(candidate) > 5:
                    params["ivas_version"] = candidate.rstrip("\x00")
                    break
    except (IndexError, ValueError):
        pass

    for month in (b"Jan", b"Feb", b"Mar", b"Apr", b"May", b"Jun", b"Jul", b"Aug", b"Sep", b"Oct", b"Nov", b"Dec"):
        index = rest.find(month)
        if index > 0:
            params["run_date"] = rest[index:index + 20].split(b"\x00")[0].decode("ascii", errors="replace").strip()
            break

    float_fields = {
        384: "detector_param1",
        388: "detector_param2",
        432: "max_voltage_V",
        444: "detector_param3",
        448: "detector_halfsize_mm",
        452: "t0_ns",
        456: "flight_path_mm",
        504: "mcp_gain_voltage_V",
        508: "anode_accel_voltage_V",
    }
    for offset_in_rest, field_name in float_fields.items():
        if offset_in_rest + 4 > len(rest):
            continue
        value = struct.unpack(">f", rest[offset_in_rest:offset_in_rest + 4])[0]
        if np.isfinite(value) and abs(value) < 1e8:
            params[field_name] = float(value)

    coefficients = []
    for index in range(1952, min(2112, len(rest)), 8):
        value = struct.unpack(">d", rest[index:index + 8])[0]
        if np.isfinite(value) and abs(value) < 1e15:
            coefficients.append(float(value))
    if coefficients:
        params["bowl_correction_coefficients"] = coefficients

    params["lsb_to_mm"] = float(params.get("detector_halfsize_mm", 18.5)) / 750.0
    return params


def _flatten_rhit_tree(tree) -> pd.DataFrame:
    columns: dict[str, np.ndarray] = {}
    for branch_name in tree.keys():
        try:
            data = tree[branch_name].array(library="np")
        except Exception:
            continue
        if hasattr(data, "dtype") and data.dtype.names:
            for field_name in data.dtype.names:
                columns[f"{branch_name}_{field_name}"] = np.asarray(data[field_name])
        else:
            columns[branch_name] = np.asarray(data)
    return pd.DataFrame(columns)


def _read_histogram_1d(root_file, root_key: str) -> dict[str, Any] | None:
    try:
        histogram = root_file[root_key]
    except Exception:
        return None
    return {
        "values": np.asarray(histogram.values()),
        "edges": np.asarray(histogram.axis().edges()),
        "title": getattr(histogram, "title", ""),
    }


def _read_histogram_2d(root_file, root_key: str) -> dict[str, Any] | None:
    try:
        histogram = root_file[root_key]
    except Exception:
        return None
    return {
        "values": np.asarray(histogram.values()),
        "xedges": np.asarray(histogram.axis(0).edges()),
        "yedges": np.asarray(histogram.axis(1).edges()),
        "title": getattr(histogram, "title", ""),
    }


def _collect_rhit_histograms(root_file) -> dict[str, dict[str, Any]]:
    histogram_map = {
        "voltageHistory": "phV;1",
        "erateHistory": "phE;1",
        "massSpectrum": "pMass;1",
        "tofRaw": "pTofRaw;1",
    }
    histograms: dict[str, dict[str, Any]] = {}
    for output_name, root_key in histogram_map.items():
        histogram = _read_histogram_1d(root_file, root_key)
        if histogram is not None:
            histograms[output_name] = histogram
    detector_map = _read_histogram_2d(root_file, "pXyHist;1")
    if detector_map is not None:
        histograms["detectorXY"] = detector_map
    return histograms


def _collect_pelf(root_file) -> dict[str, np.ndarray]:
    latest_key = _select_latest_key(root_file, "pElf")
    if latest_key is None:
        return {}
    try:
        ntuple = root_file[latest_key]
    except Exception:
        return {}
    pelf = {}
    for branch_name in ntuple.keys():
        try:
            pelf[branch_name] = np.asarray(ntuple[branch_name].array(library="np"))
        except Exception:
            continue
    return pelf


def _compute_rhit_mc(hits: pd.DataFrame, instrument_params: dict[str, Any]) -> np.ndarray:
    flight_path_m = float(instrument_params.get("flight_path_mm", DEFAULT_FLIGHT_PATH_M * 1000.0)) / 1000.0
    t0_ns = float(instrument_params.get("t0_ns", DEFAULT_T0_NS))
    if "Vref" in hits.columns and "VDC" in hits.columns:
        valid = np.isfinite(hits["Vref"]) & np.isfinite(hits["VDC"]) & (np.abs(hits["VDC"]) > 0)
        kf = float(np.median(hits.loc[valid, "Vref"] / hits.loc[valid, "VDC"])) if np.any(valid) else DEFAULT_KF
    else:
        kf = DEFAULT_KF
    constant = 2.0 * ELECTRON_CHARGE_C / (ATOMIC_MASS_UNIT_KG * flight_path_m**2) * 1e-18
    tof = hits["tof"].to_numpy(dtype=float)
    vdc = hits["VDC"].to_numpy(dtype=float)
    return constant * vdc * (tof - t0_ns) ** 2 / np.sqrt(kf)


def _convert_rhit_to_pos_format(raw_hits: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    hits = raw_hits.copy()
    instrument_params = metadata.get("instrumentParams", {})
    lsb_to_mm = float(instrument_params.get("lsb_to_mm", DEFAULT_LSB_TO_MM))

    if "x" not in hits.columns or "y" not in hits.columns:
        raise ValueError("RHIT tree does not contain the expected detector x/y branches")
    if "tof" not in hits.columns or "v" not in hits.columns:
        raise ValueError("RHIT tree does not contain the expected tof/v branches")

    hits["detxRaw"] = hits["x"].to_numpy(dtype=float)
    hits["detyRaw"] = hits["y"].to_numpy(dtype=float)
    hits["detx"] = hits["detxRaw"] * lsb_to_mm
    hits["dety"] = hits["detyRaw"] * lsb_to_mm
    hits = hits.drop(columns=["x", "y"])

    if "z" in hits.columns:
        hits["ionIdx"] = hits["z"].to_numpy(dtype=float) + 1.0
        hits = hits.drop(columns=["z"])
    else:
        hits["ionIdx"] = np.arange(1, len(hits) + 1, dtype=float)

    hits["VDC"] = hits["v"].to_numpy(dtype=float)
    hits = hits.drop(columns=["v"])
    hits["mc"] = _compute_rhit_mc(hits, instrument_params)

    standard_columns = ["ionIdx", "detx", "dety", "mc", "tof", "VDC"]
    if "Vref" in hits.columns:
        standard_columns.append("Vref")
    ordered_columns = standard_columns + [col for col in hits.columns if col not in standard_columns]
    return hits.loc[:, ordered_columns]


def rhit_load(file_path: str | Path) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, Any]]:
    """Load a Cameca RHIT file into a dataframe, histogram dictionary, and metadata."""
    uproot = _require_uproot()

    file_path = Path(file_path)
    root_file = uproot.open(file_path)
    latest_tree_key = _select_latest_key(root_file, "nth")
    if latest_tree_key is None:
        raise ValueError(f"No 'nth' tree was found in RHIT file: {file_path}")

    tree = root_file[latest_tree_key]
    raw_hits = _flatten_rhit_tree(tree)
    histograms = _collect_rhit_histograms(root_file)
    instrument_params = _decode_run_header(file_path)
    metadata = {
        "fileName": str(file_path),
        "format": "RHIT (Cameca ROOT)",
        "numHits": int(tree.num_entries),
        "instrumentParams": instrument_params,
        "rootKeys": list(root_file.keys()),
    }
    pelf = _collect_pelf(root_file)
    if pelf:
        metadata["pElf"] = pelf

    hits = _convert_rhit_to_pos_format(raw_hits, metadata)
    return hits, histograms, metadata


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


def _estimate_icf(hits: pd.DataFrame, epos: pd.DataFrame) -> float:
    sample_size = int(min(200, len(hits), len(epos)))
    if sample_size == 0:
        return 1.0
    hit_detx = hits["detx"].to_numpy(dtype=float)[:sample_size]
    epos_detx = epos["detx"].to_numpy(dtype=float)[:sample_size]
    ratios = np.full(sample_size, np.nan, dtype=float)
    valid_nonzero = np.abs(hit_detx) > 1e-12
    ratios[valid_nonzero] = epos_detx[valid_nonzero] / hit_detx[valid_nonzero]
    valid = np.abs(hit_detx) > 1.0
    valid &= np.isfinite(ratios) & (np.abs(ratios) < 2.0)
    if not np.any(valid):
        return 1.0
    return float(np.median(ratios[valid]))


def _chunked_match_events(hits: pd.DataFrame, epos: pd.DataFrame, icf: float) -> tuple[np.ndarray, np.ndarray]:
    n_rhit = len(hits)
    n_epos = len(epos)
    if n_rhit == 0 or n_epos == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    max_idx = min(n_rhit, n_epos)
    chunk_size = min(2000, max_idx)
    if max_idx <= chunk_size + 200:
        chunk_starts = np.array([0], dtype=int)
    else:
        n_chunks = min(50, max(1, max_idx // chunk_size))
        chunk_starts = np.round(np.linspace(0, max_idx - chunk_size - 1, n_chunks)).astype(int)

    matched_epos: list[int] = []
    matched_rhit: list[int] = []
    rhit_offset = 0
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
            if distance < 0.1:
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
                if d_vdc < 0.1 and d_detx < 0.2 and d_dety < 0.2:
                    matched_epos.append(epos_index)
                    matched_rhit.append(rhit_index)
                    rolling_rhit = rhit_index + 1
                    break

    if not matched_epos:
        return np.array([], dtype=int), np.array([], dtype=int)

    unique_pairs = list(dict.fromkeys(zip(matched_epos, matched_rhit)))
    matched_epos = np.array([pair[0] for pair in unique_pairs], dtype=int)
    matched_rhit = np.array([pair[1] for pair in unique_pairs], dtype=int)
    return matched_epos, matched_rhit


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


def rhit_calibrate_from_epos(
    hits: pd.DataFrame,
    epos: pd.DataFrame | str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Calibrate RHIT mass-to-charge using a matching EPOS file."""
    epos_df = _normalize_epos_calibration_dataframe(epos)
    hits_df = hits.copy()
    if len(hits_df) == 0 or len(epos_df) == 0:
        raise ValueError("RHIT and EPOS inputs must both contain events for calibration")

    icf = _estimate_icf(hits_df, epos_df)
    matched_epos, matched_rhit = _chunked_match_events(hits_df, epos_df, icf)
    if len(matched_epos) < 50:
        raise ValueError(
            "RHIT/EPOS matching found too few events for a stable calibration. "
            "Check that the files come from the same run."
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

    calibration = {
        "t_offset": t_offset,
        "C_poly": [float(value) for value in c_poly],
        "ICF": float(icf),
        "residual_std": float(np.std(residuals)),
        "matched_events": int(len(matched_epos)),
    }
    calibrated_hits = rhit_apply_calibration(hits_df, calibration)
    return calibrated_hits, calibration


def rhit_to_ccapt(hits: pd.DataFrame) -> pd.DataFrame:
    """Convert RHIT hit data into a processed PyCCAPT-style dataframe."""
    length = len(hits)
    pulse = hits["pulse"].to_numpy(dtype=float) if "pulse" in hits.columns else np.zeros(length)
    start_counter = (
        hits["tElapsed"].to_numpy(dtype=float) if "tElapsed" in hits.columns else np.arange(length, dtype=float)
    )
    return pd.DataFrame(
        {
            "x (nm)": np.zeros(length),
            "y (nm)": np.zeros(length),
            "z (nm)": np.zeros(length),
            "mc (Da)": hits["mc"].to_numpy(dtype=float),
            "mc_uc (Da)": hits["mc"].to_numpy(dtype=float),
            "high_voltage (V)": hits["VDC"].to_numpy(dtype=float),
            "pulse_v (V)": pulse,
            "pulse_l (pJ)": np.zeros(length),
            "t (ns)": hits["tof"].to_numpy(dtype=float),
            "t_c (ns)": hits["tof"].to_numpy(dtype=float),
            "x_det (cm)": hits["detx"].to_numpy(dtype=float) / 10.0,
            "y_det (cm)": hits["dety"].to_numpy(dtype=float) / 10.0,
            "delta_p": np.zeros(length, dtype=int),
            "multi": np.ones(length, dtype=int),
            "start_counter": np.asarray(start_counter).astype(int, copy=False),
        }
    )


def extract_rhit_to_hdf5(file_path: str | Path, output_path: str | Path) -> Path:
    """Extract RHIT content to HDF5 for interchange or external inspection."""
    hits, histograms, metadata = rhit_load(file_path)
    h5py = _require_h5py()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as handle:
        hits_group = handle.create_group("hits")
        for column in hits.columns:
            hits_group.create_dataset(column, data=np.asarray(hits[column].to_numpy()), compression="gzip")
        hits_group.attrs["num_entries"] = len(hits)

        hist_group = handle.create_group("histograms")
        for name, histogram in histograms.items():
            sub_group = hist_group.create_group(name)
            for key, value in histogram.items():
                if isinstance(value, str):
                    sub_group.attrs[key] = value
                else:
                    sub_group.create_dataset(key, data=np.asarray(value), compression="gzip")

        params_group = handle.create_group("instrumentParams")
        for key, value in metadata.get("instrumentParams", {}).items():
            if isinstance(value, (list, tuple, np.ndarray)):
                params_group.create_dataset(key, data=np.asarray(value))
            elif isinstance(value, str):
                params_group.attrs[key] = value
            else:
                params_group.attrs[key] = value

        for key, value in metadata.items():
            if key in {"instrumentParams", "pElf"}:
                continue
            if isinstance(value, (list, tuple)):
                handle.attrs[key] = ",".join(str(item) for item in value)
            else:
                handle.attrs[key] = value

        if "pElf" in metadata:
            pelf_group = handle.create_group("pElf")
            for key, value in metadata["pElf"].items():
                pelf_group.create_dataset(key, data=np.asarray(value), compression="gzip")

    return output_path


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Extract a RHIT file to HDF5.")
    parser.add_argument("input_path", help="Input RHIT path")
    parser.add_argument("output_path", nargs="?", help="Output HDF5 path")
    args = parser.parse_args(argv)

    output_path = Path(args.output_path) if args.output_path else Path(tempfile.mkstemp(suffix=".h5")[1])
    extract_rhit_to_hdf5(args.input_path, output_path)
    print(f"Extracted RHIT data to: {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI helper
    raise SystemExit(_main())
