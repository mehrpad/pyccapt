"""Validation and crash-recovery utilities for PyCCAPT control data."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

import h5py
import numpy as np

from pyccapt.control.core import hdf5_creator, read_files


CHUNK_DATASETS = (
    ("dld/x", "x", np.float64),
    ("dld/y", "y", np.float64),
    ("dld/t", "t", np.float64),
    ("dld/high_voltage", "voltage", np.float64),
    ("dld/voltage_pulse", "voltage_pulse", np.float64),
    ("dld/laser_pulse", "laser_pulse", np.float64),
    ("dld/start_counter", "start_counter", np.uint64),
    ("tdc/channel", "channel", np.uint32),
    ("tdc/time_data", "time", np.uint64),
    ("tdc/start_counter", "tdc_start_counter", np.uint64),
    ("tdc/high_voltage", "voltage_tdc", np.float64),
    ("tdc/voltage_pulse", "voltage_pulse_tdc", np.float64),
    ("tdc/laser_pulse", "laser_pulse_tdc", np.float64),
)


def validate_hdf5(path: str | Path) -> dict[str, object]:
    """Validate required groups, one-dimensional datasets, and aligned lengths."""
    file_path = Path(path).expanduser().resolve()
    issues: list[str] = []
    lengths: dict[str, int] = {}
    with h5py.File(file_path, "r") as handle:
        for group_name in ("apt", "dld"):
            if group_name not in handle:
                issues.append(f"missing required group: {group_name}")
        for group_name in ("apt", "dld", "tdc", "hsd"):
            if group_name not in handle:
                continue
            group_lengths = []
            for name, dataset in handle[group_name].items():
                if not isinstance(dataset, h5py.Dataset):
                    continue
                dataset_path = f"{group_name}/{name}"
                if dataset.ndim != 1:
                    issues.append(f"{dataset_path} is {dataset.ndim}D; expected 1D")
                    continue
                lengths[dataset_path] = int(dataset.shape[0])
                group_lengths.append(int(dataset.shape[0]))
            # HSD voltage arrays are per waveform while channel arrays are per
            # sample, so only enforce equality on the other synchronized groups.
            if group_name in {"apt", "dld", "tdc"} and group_lengths and len(set(group_lengths)) != 1:
                issues.append(f"{group_name} dataset lengths differ: {sorted(set(group_lengths))}")
    return {"path": str(file_path), "valid": not issues, "issues": issues, "lengths": lengths}


def validate_config(path: str | Path) -> dict[str, object]:
    """Parse and normalize a control TOML file without importing the GUI."""
    file_path = Path(path).expanduser().resolve()
    issues: list[str] = []
    try:
        config = read_files.normalize_control_config(read_files.read_toml_file(file_path))
    except Exception as exc:
        return {"path": str(file_path), "valid": False, "issues": [f"{exc.__class__.__name__}: {exc}"]}
    for key in ("tdc", "v_dc", "v_p", "laser", "signal_generator", "gauges", "cryo"):
        if key not in config:
            issues.append(f"missing setting: {key}")
        elif config[key] not in {"on", "off"}:
            issues.append(f"{key} must normalize to on/off, got {config[key]!r}")
    return {"path": str(file_path), "valid": not issues, "issues": issues}


def recover_chunks(chunk_directory: str | Path, output_path: str | Path) -> dict[str, object]:
    """Build a new HDF5 file from atomically completed detector chunks."""
    chunk_dir = Path(chunk_directory).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not chunk_dir.is_dir():
        raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")
    temporary = output.with_suffix(output.suffix + ".tmp")
    written: dict[str, int] = {}
    try:
        with h5py.File(temporary, "w") as handle:
            provenance = handle.require_group("provenance")
            provenance.attrs["recovered_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            provenance.attrs["chunk_directory"] = str(chunk_dir)
            for dataset_name, stem, dtype in CHUNK_DATASETS:
                files = hdf5_creator._sorted_chunk_files(chunk_dir, stem)
                if not files:
                    continue
                handle.require_group(dataset_name.split("/", 1)[0])
                hdf5_creator._write_chunked_dataset(handle, dataset_name, files, dtype)
                written[dataset_name] = int(handle[dataset_name].shape[0])
        if not written:
            raise ValueError(f"No recognized complete chunk files found in {chunk_dir}")
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    result = validate_hdf5(output)
    # A crash-recovery file may lack apt metadata, so report it but distinguish
    # successful detector recovery from full-schema validity.
    result.update({"recovered": True, "datasets_written": written})
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pyccapt-data", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    hdf_parser = subparsers.add_parser("validate-hdf", help="validate a PyCCAPT HDF5 file")
    hdf_parser.add_argument("path")
    config_parser = subparsers.add_parser("validate-config", help="validate control config TOML")
    config_parser.add_argument("path")
    recover_parser = subparsers.add_parser("recover-chunks", help="recover detector chunks into a new HDF5 file")
    recover_parser.add_argument("chunk_directory")
    recover_parser.add_argument("output_path")
    args = parser.parse_args(argv)
    if args.command == "validate-hdf":
        result = validate_hdf5(args.path)
    elif args.command == "validate-config":
        result = validate_config(args.path)
    else:
        result = recover_chunks(args.chunk_directory, args.output_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("valid") or result.get("recovered") else 1


if __name__ == "__main__":
    raise SystemExit(main())
