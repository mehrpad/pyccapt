"""Validation and crash-recovery utilities for PyCCAPT control data."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

from pyccapt.control.core import chunk_store, detector_hdf, hdf5_creator, read_files


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
    """Validate schema metadata, dimensions, units, and aligned stream lengths."""
    file_path = Path(path).expanduser().resolve()
    issues: list[str] = []
    lengths: dict[str, int] = {}
    with h5py.File(file_path, "r") as handle:
        if "apt" not in handle:
            issues.append("missing required group: apt")
        elif 'id' not in handle['apt']:
            issues.append('missing required dataset: apt/id')
        schema_version = ""
        completeness = 'unverified'
        config = {}
        if "provenance" in handle:
            schema_version = str(handle["provenance"].attrs.get("schema_version", ""))
        if schema_version.startswith("2"):
            for key in (
                "control_config_sha256", "calibration_input_sha256",
                "pyccapt_version", "created_utc", "model_provenance_json",
            ):
                if key not in handle["provenance"].attrs:
                    issues.append(f"missing provenance attribute: {key}")
            config_hash = str(handle["provenance"].attrs.get("control_config_sha256", ""))
            if len(config_hash) != 64 or any(character not in "0123456789abcdef" for character in config_hash.lower()):
                issues.append("control_config_sha256 is not a SHA-256 hex digest")
            raw_config = str(handle['provenance'].attrs.get('control_config_json', ''))
            if hashlib.sha256(raw_config.encode('utf-8')).hexdigest() != config_hash:
                issues.append('control_config_sha256 does not match control_config_json')
            try:
                config = json.loads(raw_config)
                if not isinstance(config, dict):
                    raise ValueError('expected an object')
            except (ValueError, TypeError):
                config = {}
                issues.append('control_config_json is not a JSON object')
            try:
                json.loads(str(handle["provenance"].attrs.get("model_provenance_json", "")))
            except json.JSONDecodeError:
                issues.append("model_provenance_json is not valid JSON")
            required = {path for _, path, _ in hdf5_creator._APT_CHUNK_STEMS[:6]}
            if config.get('tdc') == 'on':
                try:
                    model = hdf5_creator.normalize_tdc_model(config.get('tdc_model'))
                except ValueError as exc:
                    model = ''
                    issues.append(str(exc))
                if model in {'Surface_Concept', 'RoentDek'}:
                    required.update(path for path, _, _ in CHUNK_DATASETS)
                elif model == 'HSD':
                    required.update(path for path in hdf5_creator._DATASET_UNITS if path.startswith('hsd/'))
            for name in sorted(required):
                if name not in handle:
                    issues.append(f'missing required dataset: {name}')
            expected_types = {path: np.dtype(dtype) for path, _, dtype in CHUNK_DATASETS}
            expected_types.update({path: np.dtype(dtype) for _, path, dtype in hdf5_creator._APT_CHUNK_STEMS})
            expected_types.update({path: np.dtype('float32' if '/ch' in path else 'float64')
                                   for path in hdf5_creator._DATASET_UNITS if path.startswith('hsd/')})
            for name, dtype in expected_types.items():
                if name in handle and isinstance(handle[name], h5py.Dataset) and handle[name].dtype != dtype:
                    issues.append(f'{name} has dtype {handle[name].dtype}; expected {dtype}')
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
                if schema_version.startswith("2") and not str(dataset.attrs.get("units", "")):
                    issues.append(f"{dataset_path} has no units metadata")
                expected_unit = hdf5_creator._DATASET_UNITS.get(dataset_path)
                if schema_version.startswith('2') and expected_unit and dataset.attrs.get('units') != expected_unit:
                    issues.append(f'{dataset_path} has incorrect units; expected {expected_unit}')
            # HSD voltage arrays are per waveform while channel arrays are per
            # sample, so only enforce equality on the other synchronized groups.
            if group_name in {"apt", "dld", "tdc"} and group_lengths and len(set(group_lengths)) != 1:
                issues.append(f"{group_name} dataset lengths differ: {sorted(set(group_lengths))}")
        hsd_samples = [lengths[name] for name in lengths if name.startswith('hsd/ch')]
        hsd_events = [lengths[name] for name in ('hsd/high_voltage', 'hsd/voltage_pulse') if name in lengths]
        if hsd_samples and len(set(hsd_samples)) != 1:
            issues.append('hsd waveform channel lengths differ')
        if hsd_events and len(set(hsd_events)) != 1:
            issues.append('hsd per-waveform voltage lengths differ')
        if hsd_samples and hsd_events and any(samples != 1024 * events for samples in hsd_samples for events in hsd_events):
            issues.append('hsd waveforms must have 1024 samples per voltage event')
        structurally_valid = not issues
        if 'provenance' in handle:
            completeness = str(handle['provenance'].attrs.get('acquisition_completeness', 'unverified'))
            if completeness == 'incomplete':
                issues.append('acquisition incomplete; inspect provenance/chunk_integrity_json')
    return {
        "path": str(file_path), "valid": not issues, "issues": issues,
        "lengths": lengths, "schema_version": schema_version,
        'structurally_valid': structurally_valid, 'acquisition_completeness': completeness,
    }


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
    manifest_files = chunk_store.validated_files_by_stem(chunk_dir)
    manifest_files = detector_hdf.align_hsd_chunks(manifest_files)
    valid_manifest = manifest_files.valid_records if manifest_files is not None else []
    invalid_manifest = manifest_files.invalid_records if manifest_files is not None else []
    temporary = output.with_suffix(output.suffix + ".tmp")
    written: dict[str, int] = {}
    try:
        with h5py.File(temporary, "w") as handle:
            provenance = handle.require_group("provenance")
            provenance.attrs["recovered_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            provenance.attrs["chunk_directory"] = str(chunk_dir)
            report = (manifest_files.integrity_report() if manifest_files is not None else
                      {'status': 'unverified', 'scope': 'legacy chunks without manifests'})
            unattributed = sorted(path.name for path in (chunk_dir.parent / 'unattributed').glob('*.npz'))
            report['unattributed_packets'] = unattributed
            if unattributed:
                report['status'] = 'incomplete'
            # Recovery can verify committed records, never prove a crashed
            # acquisition had no uncommitted tail. Preserve that distinction.
            provenance.attrs['acquisition_completeness'] = ('incomplete' if report['status'] == 'incomplete' else 'unverified')
            provenance.attrs['chunk_integrity_json'] = json.dumps(report, sort_keys=True)
            backend_written = (detector_hdf.write_hsd_chunks(handle, chunk_dir, manifest_files)
                               or detector_hdf.write_roentdek_chunks(handle, chunk_dir, manifest_files))
            if not backend_written:
                for dataset_name, stem, dtype in CHUNK_DATASETS:
                    files = hdf5_creator._sorted_chunk_files(chunk_dir, stem, manifest_files)
                    if not files:
                        continue
                    handle.require_group(dataset_name.split("/", 1)[0])
                    hdf5_creator._write_chunked_dataset(handle, dataset_name, files, dtype)
            for group_name in ('dld', 'tdc', 'hsd'):
                if group_name in handle:
                    written.update({f'{group_name}/{name}': int(dataset.shape[0])
                                    for name, dataset in handle[group_name].items() if isinstance(dataset, h5py.Dataset)})
        if not written:
            raise ValueError(f"No recognized complete chunk files found in {chunk_dir}")
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    result = validate_hdf5(output)
    # A crash-recovery file may lack apt metadata, so report it but distinguish
    # successful detector recovery from full-schema validity.
    result.update(
        {
            "recovered": True,
            "datasets_written": written,
            "manifest_records": len(valid_manifest),
            "quarantined_records": sum("quarantine" in record for record in invalid_manifest),
            "excluded_manifest_records": len(invalid_manifest),
        }
    )
    return result


def main(argv: list[str] | None = None, *, prog: str = "pyccapt-data") -> int:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    hdf_parser = subparsers.add_parser(
        "validate-hdf5", aliases=["validate-hdf"], help="validate a PyCCAPT HDF5 file"
    )
    hdf_parser.add_argument("path")
    config_parser = subparsers.add_parser("validate-config", help="validate control config TOML")
    config_parser.add_argument("path")
    recover_parser = subparsers.add_parser(
        "recover-run", aliases=["recover-chunks"], help="recover detector chunks into a new HDF5 file"
    )
    recover_parser.add_argument("chunk_directory")
    recover_parser.add_argument("output_path")
    args = parser.parse_args(argv)
    if args.command in {"validate-hdf", "validate-hdf5"}:
        result = validate_hdf5(args.path)
    elif args.command == "validate-config":
        result = validate_config(args.path)
    else:
        result = recover_chunks(args.chunk_directory, args.output_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("valid") or result.get("recovered") else 1


if __name__ == "__main__":
    raise SystemExit(main())
