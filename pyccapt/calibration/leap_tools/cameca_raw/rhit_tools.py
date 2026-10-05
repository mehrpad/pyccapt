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
            "RHIT support requires the optional 'uproot' package. Install it with 'pip install uproot'."
        ) from exc
    return uproot


def _require_h5py():
    try:
        import h5py  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional runtime package
        raise ImportError(
            "HDF5 export for RHIT support requires the optional 'h5py' package. Install it with 'pip install h5py'."
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
        if compressed[position : position + 2] == b"ZL":
            compressed_size = int.from_bytes(compressed[position + 3 : position + 6], "little")
            chunk = zlib.decompress(compressed[position + 9 : position + 9 + compressed_size])
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
                class_name = header[27 : 27 + class_name_length].decode("ascii", errors="replace")
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
            if rest[index : index + 1].isdigit():
                null_offset = rest[index : index + 20].find(b"\x00")
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
            params["run_date"] = rest[index : index + 20].split(b"\x00")[0].decode("ascii", errors="replace").strip()
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
        value = struct.unpack(">f", rest[offset_in_rest : offset_in_rest + 4])[0]
        if np.isfinite(value) and abs(value) < 1e8:
            params[field_name] = float(value)

    # CRunHeader is a private Cameca ROOT class.  Earlier code treated a fixed
    # byte range as big-endian doubles and exposed the result as bowl
    # coefficients.  On complete LEAP files that produces obvious byte-garbage
    # (for example 1e-315 and 1e14 values), so it must not be presented as a
    # calibration.  A verified CCalibMass/CBowl class decoder is required.
    params["bowl_correction_status"] = "not decoded: proprietary CRunHeader/CBowl layout requires a verified class decoder"

    params["lsb_to_mm"] = float(params.get("detector_halfsize_mm", 18.5)) / 750.0
    return params


def _flatten_rhit_tree(tree, verbose: bool = True) -> pd.DataFrame:
    """Flatten an RHIT TTree into a DataFrame.

    Reads each branch with uproot's ``library="np"`` and groups columns by their
    natural length. The largest-by-branch-count group becomes the returned
    DataFrame, so ``pd.DataFrame`` never has to resolve a length mismatch
    silently. With ``verbose=True`` (the default) every branch and field is
    printed with its length and dtype, mirroring the inspection snippet
    users typically run against the file directly.
    """

    expected_entries = int(tree.num_entries)
    by_length: dict[int, dict[str, np.ndarray]] = {}
    failed: list[tuple[str, str]] = []
    branch_log: list[tuple[str, int, str]] = []

    def _record(name: str, values: np.ndarray) -> None:
        bucket = by_length.setdefault(len(values), {})
        bucket[name] = values
        branch_log.append((name, len(values), str(values.dtype)))

    for branch_name in tree.keys():
        try:
            data = tree[branch_name].array(library="np")
        except Exception as exc:
            failed.append((branch_name, str(exc)))
            continue
        if hasattr(data, "dtype") and data.dtype.names:
            # Note the parent struct length/dtype too so it's visible in the log.
            try:
                parent_length = len(data)
            except TypeError:
                parent_length = -1
            branch_log.append((f"{branch_name} (struct)", parent_length, str(data.dtype)))
            for field_name in data.dtype.names:
                _record(f"{branch_name}_{field_name}", np.asarray(data[field_name]))
        else:
            _record(branch_name, np.asarray(data))

    if verbose:
        print(f"  Tree num_entries: {expected_entries:,}")
        print(f"  Per-branch array lengths:")
        for name, length, dtype in branch_log:
            print(f"    {name}: len={length:,}, dtype={dtype}")
        for name, reason in failed:
            print(f"    {name}: FAILED -> {reason}")

    if not by_length:
        raise ValueError(f"Could not read any branches from RHIT tree (expected {expected_entries:,} entries)")

    primary_length = max(by_length.keys(), key=lambda length: len(by_length[length]))
    primary_columns = by_length[primary_length]
    other_lengths = {length: cols for length, cols in by_length.items() if length != primary_length}

    print(f"  Building dataframe from {len(primary_columns)} columns at length {primary_length:,}")
    if primary_length != expected_entries:
        print(f"  Note: dataframe length ({primary_length:,}) differs from tree.num_entries ({expected_entries:,}).")
    for length, cols in other_lengths.items():
        preview = ", ".join(sorted(cols.keys())[:6])
        extra = " ..." if len(cols) > 6 else ""
        print(f"  Skipping {len(cols)} columns at length {length:,}: {preview}{extra}")

    return pd.DataFrame(primary_columns)


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
    """Return a direct-flight m/q proxy, not an IVAS or reflectron calibration."""
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
    # Keep the long-standing ``mc`` compatibility column, but make the
    # approximation explicit as well.  Reflectron RHIT files need their stored
    # CCalibMass/CBowl calibration (or a separately validated transfer model)
    # before this quantity can be called calibrated m/q.
    hits["mc_direct_flight_approx"] = _compute_rhit_mc(hits, instrument_params)
    hits["mc"] = hits["mc_direct_flight_approx"]

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
    nth_keys = [key for key in root_file.keys() if str(key).startswith("nth;")]
    if nth_keys:
        print(f"  Available 'nth' cycles in file: {nth_keys}")
    latest_tree_key = _select_latest_key(root_file, "nth")
    if latest_tree_key is None:
        raise ValueError(f"No 'nth' tree was found in RHIT file: {file_path}")
    print(f"  Selected RHIT tree: {latest_tree_key}")

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
        "massToCharge": {
            "column": "mc_direct_flight_approx",
            "status": "direct-flight approximation from raw RHIT TOF/VDC; not IVAS-calibrated m/q",
            "limitation": "Do not use for reflectron TOF, energy, bowl, or mass-tail inference without a verified CCalibMass/CBowl decoder or instrument transfer model.",
        },
    }
    pelf = _collect_pelf(root_file)
    if pelf:
        metadata["pElf"] = pelf

    hits = _convert_rhit_to_pos_format(raw_hits, metadata)
    return hits, histograms, metadata


# Calibration helpers live in a sibling module to keep this file under the
# 1250-line policy. They are re-exported here so existing
# ``rhit_tools.rhit_calibrate_from_epos`` / ``rhit_tools.rhit_apply_calibration``
# call sites keep working unchanged.
from pyccapt.calibration.leap_tools.cameca_raw._rhit_calibration import (  # noqa: E402
    apply_rhit_calibration,
    rhit_apply_calibration,
    rhit_calibrate_from_epos,
)


def rhit_to_ccapt(hits: pd.DataFrame, drop_invalid: bool = True) -> pd.DataFrame:
    """Convert RHIT hit data into a processed PyCCAPT-style dataframe.

    With ``drop_invalid=True`` (default) rows missing detx, dety, mc, tof, or
    VDC are dropped — these can never produce useful downstream analysis.
    """
    if drop_invalid:
        required = ["detx", "dety", "mc", "tof", "VDC"]
        mask = pd.Series(True, index=hits.index)
        for column in required:
            if column not in hits.columns:
                continue
            values = hits[column].to_numpy(dtype=float)
            mask &= np.isfinite(values)
        hits = hits.loc[mask]

    length = len(hits)
    pulse = hits["pulse"].to_numpy(dtype=float) if "pulse" in hits.columns else np.zeros(length)
    start_counter = hits["tElapsed"].to_numpy(dtype=float) if "tElapsed" in hits.columns else np.arange(length, dtype=float)
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


def rhit_to_raw_hdf5(
    hits: pd.DataFrame,
    output_path: str | Path,
    *,
    pulse_mode: str = "voltage",
    drop_invalid: bool = True,
) -> Path:
    """Write `dld/` and `tdc/` group HDF5 readable by the PyCCAPT raw-data analysis workflow.

    Two groups are written so the file is a complete drop-in for the raw
    analysis tooling:

    * ``dld/...`` — already-decoded events (high_voltage, pulse, laser_intensity,
      start_counter, t, x, y). Loadable via ``fetch_dataset_from_dld_grp(..., extract_mode='dld')``.
    * ``tdc/...`` — the same events as a single-channel TDC stream
      (channel=0, time_data=tof in ns). RHIT files don't carry channel-level
      delay-line counts, so this is a single-channel mirror of ``dld/`` rather
      than true per-channel hits. Loadable via ``extract_mode='tdc_sc'``.

    With ``drop_invalid=True`` (default), rows where ``detx`` / ``dety`` /
    ``tof`` / ``VDC`` are NaN are dropped before writing.
    """
    h5py = _require_h5py()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if drop_invalid:
        valid = (
            np.isfinite(hits["VDC"].to_numpy(dtype=float))
            & np.isfinite(hits["tof"].to_numpy(dtype=float))
            & np.isfinite(hits["detx"].to_numpy(dtype=float))
            & np.isfinite(hits["dety"].to_numpy(dtype=float))
        )
        hits = hits.loc[valid]

    n = len(hits)
    high_voltage = hits["VDC"].to_numpy(dtype=float).reshape(-1, 1)
    tof_ns = hits["tof"].to_numpy(dtype=float).reshape(-1, 1)
    det_x_cm = (hits["detx"].to_numpy(dtype=float) / 10.0).reshape(-1, 1)
    det_y_cm = (hits["dety"].to_numpy(dtype=float) / 10.0).reshape(-1, 1)
    pulse_value = hits["pulse"].to_numpy(dtype=float) if "pulse" in hits.columns else np.zeros(n, dtype=float)
    pulse_value = pulse_value.reshape(-1, 1)
    laser_value = (
        hits["laserpower"].to_numpy(dtype=float) if "laserpower" in hits.columns else np.zeros(n, dtype=float)
    ).reshape(-1, 1)
    if pulse_mode == "laser":
        pulse_v = np.zeros_like(pulse_value)
        pulse_l = laser_value
    else:
        pulse_v = pulse_value
        pulse_l = laser_value
    if "ionIdx" in hits.columns:
        start_counter = hits["ionIdx"].to_numpy(dtype=np.uint32).reshape(-1, 1)
    else:
        start_counter = np.arange(n, dtype=np.uint32).reshape(-1, 1)

    with h5py.File(output_path, "w") as handle:
        grp = handle.create_group("dld")
        grp.create_dataset("high_voltage", data=high_voltage, compression="gzip", compression_opts=4)
        grp.create_dataset("pulse", data=pulse_v, compression="gzip", compression_opts=4)
        grp.create_dataset("laser_intensity", data=pulse_l, compression="gzip", compression_opts=4)
        grp.create_dataset("start_counter", data=start_counter, compression="gzip", compression_opts=4)
        grp.create_dataset("t", data=tof_ns, compression="gzip", compression_opts=4)
        grp.create_dataset("x", data=det_x_cm, compression="gzip", compression_opts=4)
        grp.create_dataset("y", data=det_y_cm, compression="gzip", compression_opts=4)
        grp.attrs["num_entries"] = n
        grp.attrs["units"] = "high_voltage=V, pulse=V, laser_intensity=pJ, t=ns, x=cm, y=cm, start_counter=uint32"
        grp.attrs["source"] = "rhit_to_raw_hdf5"
        grp.attrs["pulse_mode"] = pulse_mode

        # tdc/ group: one channel per RHIT event (the file doesn't carry
        # per-DL-end timings; use tof as the time_data and channel=0).
        tdc = handle.create_group("tdc")
        tdc.create_dataset("channel", data=np.zeros((n, 1), dtype=np.uint32), compression="gzip", compression_opts=4)
        tdc.create_dataset("start_counter", data=start_counter, compression="gzip", compression_opts=4)
        tdc.create_dataset("high_voltage", data=high_voltage, compression="gzip", compression_opts=4)
        tdc.create_dataset("pulse", data=pulse_v, compression="gzip", compression_opts=4)
        tdc.create_dataset("laser_pulse", data=pulse_l, compression="gzip", compression_opts=4)
        # time_data in raw analysis is uint32 TDC counts; we don't have those
        # for RHIT, so write tof in ns as the placeholder time_data.
        tdc.create_dataset("time_data", data=tof_ns.astype(np.uint32), compression="gzip", compression_opts=4)
        tdc.attrs["num_entries"] = n
        tdc.attrs["channel_count"] = 1
        tdc.attrs["channel_map"] = "0=tof_ns"
        tdc.attrs["units"] = "channel=uint32, time_data=ns (cast to uint32 to match raw analysis)"
        tdc.attrs["source"] = "rhit_to_raw_hdf5 (single-channel mirror)"
    return output_path


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
