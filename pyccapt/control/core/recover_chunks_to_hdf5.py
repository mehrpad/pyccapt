"""
Chunk-to-HDF5 recovery script for PyCCAPT experiments.

Use this when an experiment finished but the final .h5 file is missing
(e.g. the control PC crashed during finalization, or hdf_creator threw
an exception after the run completed).

The control software writes detector data to:
    <exp_path>/temp_data/chunks/<stem>_chunk_<N>.npy   (chunked, primary)
    <exp_path>/temp_data/<stem>.npy                    (flat, older fallback)

This script reassembles those files into a valid .h5 in <exp_path>/,
following the same schema that hdf_creator.hdf_creator() produces.

Usage
-----
    python recover_chunks_to_hdf5.py <exp_path>

    # Example (run ON THE CONTROL COMPUTER):
    python recover_chunks_to_hdf5.py "D:\\pyccapt\\pyccapt\\data\\2512_Jun-10-2026_16-01_NiC1_C3"

    # Without arguments: uses current working directory
    python recover_chunks_to_hdf5.py

Robustness
----------
- Missing stems: skipped with a warning; the HDF5 group is omitted rather
  than written with empty/garbage data.
- Corrupted or zero-byte chunks: logged and skipped; the remaining chunks
  in that stem are still loaded.
- Dtype mismatch across chunks: each chunk is cast to the expected dtype
  with a warning.
- Unequal array lengths within a group: all datasets in a group (dld/*
  or tdc/*) are truncated to the shortest present array, so the file stays
  self-consistent. A report is printed showing how many rows were dropped.
- Partial chunks from an interrupted experiment: handled transparently since
  chunks are sorted by ID and concatenated individually.

What is recovered
-----------------
    dld/x, y, t, high_voltage, voltage_pulse, laser_pulse, start_counter
    tdc/channel, time_data, start_counter, high_voltage, voltage_pulse, laser_pulse
    apt/id, num_events, num_raw_signals  -- reconstructed from dld/start_counter
    apt/temperature, vacuum, timestamps  -- zeros (held in RAM, not in chunks)

The apt/* zeroed fields are not used by the calibration pipeline.
"""

from __future__ import annotations

import os
import re
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Dataset schema
# Each tuple: (hdf5_path, chunk_stem, numpy_dtype)
# ---------------------------------------------------------------------------

# dld/* arrays must all share the same first-axis length (one entry per hit)
DLD_MAPPING: list[tuple[str, str, str]] = [
    ("dld/x",             "x",             "float64"),
    ("dld/y",             "y",             "float64"),
    ("dld/t",             "t",             "float64"),
    ("dld/high_voltage",  "voltage",       "float64"),
    ("dld/voltage_pulse", "voltage_pulse", "float64"),
    ("dld/laser_pulse",   "laser_pulse",   "float64"),
    ("dld/start_counter", "start_counter", "uint64"),
]

# tdc/* arrays must all share the same first-axis length (one entry per TDC word)
TDC_MAPPING: list[tuple[str, str, str]] = [
    ("tdc/channel",       "channel",            "uint32"),
    ("tdc/time_data",     "time",               "uint64"),
    ("tdc/start_counter", "tdc_start_counter",  "uint64"),
    ("tdc/high_voltage",  "voltage_tdc",        "float64"),
    ("tdc/voltage_pulse", "voltage_pulse_tdc",  "float64"),
    ("tdc/laser_pulse",   "laser_pulse_tdc",    "float64"),
]

# apt/* metadata chunks written by APT_Exp_Control._flush_apt_meta_chunks()
# These are all the same length (one entry per main-loop step).
APT_MAPPING: list[tuple[str, str, str]] = [
    ("apt/id",                        "apt_id",              "uint64"),
    ("apt/timestamps",                "apt_timestamps",      "float64"),
    ("apt/num_events",                "apt_num_events",      "uint32"),
    ("apt/num_raw_signals",           "apt_num_raw_signals", "uint32"),
    ("apt/temperature",               "apt_temperature",     "float64"),
    ("apt/experiment_chamber_vacuum", "apt_vacuum",          "float64"),
]

ALL_MAPPING = DLD_MAPPING + TDC_MAPPING


# ---------------------------------------------------------------------------
# Chunk discovery and loading
# ---------------------------------------------------------------------------

def _sorted_chunk_files(chunk_dir: Path, stem: str) -> list[Path]:
    """Return chunk files for *stem* sorted by their numeric chunk index."""
    pattern = re.compile(rf"^{re.escape(stem)}_chunk_(\d+)\.npy$")
    found: list[tuple[int, Path]] = []
    for p in chunk_dir.glob(f"{stem}_chunk_*.npy"):
        m = pattern.match(p.name)
        if m:
            found.append((int(m.group(1)), p))
    return [p for _, p in sorted(found)]


def _load_one_npy(path: Path, target_dtype: np.dtype) -> np.ndarray | None:
    """Load a single .npy file; return None on any error."""
    try:
        if path.stat().st_size == 0:
            print(f"    SKIP (zero bytes): {path.name}")
            return None
        arr = np.load(path, mmap_mode="r")
        if arr.ndim == 0 or arr.size == 0:
            print(f"    SKIP (empty array): {path.name}")
            return None
        if arr.dtype != target_dtype:
            print(f"    CAST {arr.dtype} -> {target_dtype}: {path.name}")
            arr = arr.astype(target_dtype)
        return arr.copy()
    except Exception as exc:
        print(f"    ERROR loading {path.name}: {exc}")
        return None


def load_from_chunks(chunk_dir: Path, stem: str, dtype: str) -> np.ndarray | None:
    """Concatenate all valid chunk files for *stem*; return None if none found."""
    if not chunk_dir.is_dir():
        return None
    files = _sorted_chunk_files(chunk_dir, stem)
    if not files:
        return None
    target = np.dtype(dtype)
    parts: list[np.ndarray] = []
    skipped = 0
    for f in files:
        arr = _load_one_npy(f, target)
        if arr is not None:
            parts.append(arr)
        else:
            skipped += 1
    if not parts:
        return None
    result = np.concatenate(parts)
    if skipped:
        print(f"    ({skipped}/{len(files)} chunk(s) skipped for {stem})")
    return result


def load_from_flat(temp_dir: Path, stem: str, dtype: str) -> np.ndarray | None:
    """Load a single flat temp_data/<stem>.npy file."""
    p = temp_dir / f"{stem}.npy"
    if not p.exists():
        return None
    return _load_one_npy(p, np.dtype(dtype))


def load_dataset(chunk_dir: Path, temp_dir: Path, stem: str, dtype: str) -> np.ndarray | None:
    """Try chunked files first, then the flat temp_data file."""
    arr = load_from_chunks(chunk_dir, stem, dtype)
    if arr is not None:
        return arr
    return load_from_flat(temp_dir, stem, dtype)


# ---------------------------------------------------------------------------
# Length reconciliation
# ---------------------------------------------------------------------------

def reconcile_lengths(
    group_name: str,
    datasets: dict[str, np.ndarray],
    mapping: list[tuple[str, str, str]],
) -> dict[str, np.ndarray]:
    """
    Truncate all arrays in *mapping* that are present in *datasets* to the
    length of the shortest one.  Logs any truncation clearly.

    Returns a new dict with reconciled arrays (only the datasets from
    *mapping* that were already in *datasets*).
    """
    present = {ds: datasets[ds] for ds, _, _ in mapping if ds in datasets}
    if not present:
        return {}

    lengths = {ds: len(arr) for ds, arr in present.items()}
    min_len = min(lengths.values())
    max_len = max(lengths.values())

    if min_len != max_len:
        print(f"\n  LENGTH MISMATCH in {group_name}:")
        for ds, length in sorted(lengths.items()):
            flag = " <- shortest (truncating others to this)" if length == min_len else ""
            print(f"    {ds}: {length:,}{flag}")
        print(f"  Truncating all {group_name} arrays to {min_len:,} rows.")

    return {ds: arr[:min_len] for ds, arr in present.items()}


# ---------------------------------------------------------------------------
# apt/* reconstruction
# ---------------------------------------------------------------------------

def load_apt_from_chunks(chunk_dir: Path, temp_dir: Path) -> dict[str, np.ndarray]:
    """Load apt/* metadata from dedicated apt_* chunk files written during the run."""
    result: dict[str, np.ndarray] = {}
    for ds_name, stem, dtype in APT_MAPPING:
        arr = load_dataset(chunk_dir, temp_dir, stem, dtype)
        if arr is not None and len(arr) > 0:
            result[ds_name] = arr
    return result


def synthesize_apt_group(datasets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Synthesize apt/* metadata from dld/tdc chunk data when apt_* chunks are absent.

    apt/id          -- unique DLD start_counter values (proxy for loop steps)
    apt/num_events  -- event count per step from dld/start_counter
    apt/num_raw_signals -- raw TDC trigger count per step from tdc/start_counter
    apt/temperature, apt/experiment_chamber_vacuum, apt/timestamps
                    -- zeros: these lived in RAM and were not in old-format chunks.
    """
    apt: dict[str, np.ndarray] = {}

    dld_sc = datasets.get("dld/start_counter")
    tdc_sc = datasets.get("tdc/start_counter")

    if dld_sc is not None and len(dld_sc) > 0:
        unique_steps, counts = np.unique(dld_sc, return_counts=True)
        apt["apt/id"]         = unique_steps.astype(np.uint64)
        apt["apt/num_events"] = counts.astype(np.uint32)
        n_steps = len(unique_steps)
    else:
        print("  WARNING: dld/start_counter not available -- apt group will be minimal.")
        n_steps = 0
        apt["apt/id"]         = np.zeros(0, dtype=np.uint64)
        apt["apt/num_events"] = np.zeros(0, dtype=np.uint32)

    if tdc_sc is not None and len(tdc_sc) > 0:
        _, raw_counts = np.unique(tdc_sc, return_counts=True)
        if len(raw_counts) > n_steps:
            raw_counts = raw_counts[:n_steps]
        elif len(raw_counts) < n_steps:
            raw_counts = np.pad(raw_counts, (0, n_steps - len(raw_counts)))
        apt["apt/num_raw_signals"] = raw_counts.astype(np.uint32)
    else:
        apt["apt/num_raw_signals"] = np.zeros(n_steps, dtype=np.uint32)

    apt["apt/temperature"]               = np.zeros(n_steps, dtype=np.float64)
    apt["apt/experiment_chamber_vacuum"] = np.zeros(n_steps, dtype=np.float64)
    apt["apt/timestamps"]                = np.arange(n_steps, dtype=np.float64)

    return apt


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------

def write_hdf5(
    output_path: Path,
    datasets: dict[str, np.ndarray],
    apt: dict[str, np.ndarray],
) -> None:
    """Write the recovered datasets to an HDF5 file via an atomic .tmp rename."""
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    print(f"\nWriting to {tmp_path} …")
    with h5py.File(tmp_path, "w") as hf:
        # apt/* group: keys are already full HDF5 paths (apt/id, apt/timestamps, …)
        for ds_path, arr in apt.items():
            hf.create_dataset(ds_path, data=arr)
        # dld/* and tdc/* groups
        for ds_name, arr in datasets.items():
            hf.create_dataset(ds_name, data=arr)
    os.replace(tmp_path, output_path)
    print(f"Renamed  → {output_path}")


# ---------------------------------------------------------------------------
# Log helpers
# ---------------------------------------------------------------------------

def _tail_log(path: Path, n_lines: int = 50) -> None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        print(f"\n--- last {min(n_lines, len(lines))} lines of {path} ---")
        for line in lines[-n_lines:]:
            print(line)
        print("--- end ---")
    except Exception as exc:
        print(f"  Could not read {path}: {exc}")


def _check_logs(exp_dir: Path) -> None:
    apt_log = exp_dir / "meta_data" / "apt.log"
    if apt_log.exists():
        _tail_log(apt_log, n_lines=50)
    else:
        print(f"\nNo per-experiment log found at {apt_log}")

    # Try to find today's GUI log from the project root two levels up
    # (<project_root>/pyccapt/data/<exp>)
    project_root = exp_dir.parent.parent
    import datetime
    today = datetime.date.today().strftime("%Y-%m-%d")
    gui_log = project_root / "files" / "logs" / "gui" / f"gui_{today}.log"
    if gui_log.exists():
        print(f"\nGUI log found at {gui_log}")
        print("Search it for 'ERROR', 'CRITICAL', 'Traceback', or 'hdf_creator'.")
    else:
        # Give the operator the expected path even if we can't reach it
        print(f"\nExpected GUI log (not accessible from here):")
        print(f"  {gui_log}")
        print("Search it for 'ERROR', 'CRITICAL', 'Traceback', or 'hdf_creator'.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(exp_path: str | None = None) -> int:
    if exp_path is None:
        exp_path = sys.argv[1] if len(sys.argv) > 1 else "."

    exp_dir = Path(exp_path).resolve()
    if not exp_dir.is_dir():
        print(f"ERROR: experiment directory not found: {exp_dir}")
        return 1

    exp_name = exp_dir.name
    print("=" * 70)
    print(f"PyCCAPT chunk-to-HDF5 recovery")
    print(f"  Experiment : {exp_name}")
    print(f"  Directory  : {exp_dir}")
    print("=" * 70)

    # ---- check for an existing HDF5 -------------------------------------
    existing = list(exp_dir.glob("*.h5"))
    if existing:
        print("\nNOTE: HDF5 file(s) already present:")
        for p in existing:
            print(f"  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")
        answer = input("Continue and overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return 0

    # ---- locate source files --------------------------------------------
    temp_dir  = exp_dir / "temp_data"
    chunk_dir = temp_dir / "chunks"

    if not temp_dir.exists():
        print(f"\nERROR: temp_data/ not found under {exp_dir}")
        print("The control software may not have created any data yet, or the path is wrong.")
        _check_logs(exp_dir)
        return 1

    has_chunks = chunk_dir.is_dir() and any(chunk_dir.glob("*_chunk_*.npy"))
    has_flat   = any(temp_dir.glob("*.npy"))

    if not has_chunks and not has_flat:
        print(f"\nERROR: no .npy files found in {temp_dir} (neither chunks/ nor flat files).")
        _check_logs(exp_dir)
        return 1

    print(f"\nData source:")
    print(f"  chunked (temp_data/chunks/) : {'yes' if has_chunks else 'no'}")
    print(f"  flat    (temp_data/*.npy)   : {'yes' if has_flat else 'no'}")

    # ---- load all stems -------------------------------------------------
    all_datasets: dict[str, np.ndarray] = {}
    print("\nLoading datasets:")
    for ds_name, stem, dtype in ALL_MAPPING:
        arr = load_dataset(chunk_dir, temp_dir, stem, dtype)
        if arr is None or len(arr) == 0:
            print(f"  MISSING  {ds_name}")
        else:
            print(f"  OK       {ds_name:35s}  {arr.shape}  {arr.dtype}")
            all_datasets[ds_name] = arr

    if not all_datasets:
        print("\nERROR: no datasets could be loaded. Aborting.")
        _check_logs(exp_dir)
        return 1

    # ---- reconcile group lengths ----------------------------------------
    print("\nReconciling group lengths …")
    dld_clean = reconcile_lengths("dld", all_datasets, DLD_MAPPING)
    tdc_clean = reconcile_lengths("tdc", all_datasets, TDC_MAPPING)

    reconciled: dict[str, np.ndarray] = {**dld_clean, **tdc_clean}

    # Keep any datasets that weren't part of a group (shouldn't happen, but
    # be safe so nothing already loaded is silently discarded).
    known = {ds for ds, _, _ in ALL_MAPPING}
    for ds, arr in all_datasets.items():
        if ds not in known:
            reconciled[ds] = arr

    # ---- summary before writing -----------------------------------------
    n_dld = len(dld_clean.get("dld/x", np.zeros(0)))
    n_tdc = len(tdc_clean.get("tdc/time_data", np.zeros(0)))
    print(f"\nRecovered hits:")
    print(f"  DLD hits : {n_dld:,}")
    print(f"  TDC words: {n_tdc:,}")

    missing = [ds for ds, _, _ in ALL_MAPPING if ds not in reconciled]
    if missing:
        print(f"\nMissing datasets (will be absent from the output file):")
        for ds in missing:
            print(f"  {ds}")

    # ---- load or synthesize apt/* group ---------------------------------
    print("\nLoading apt/* metadata chunks …")
    apt = load_apt_from_chunks(chunk_dir, temp_dir)
    if apt:
        # Reconcile apt group lengths the same way as dld/tdc
        apt_mapping_present = [(ds, s, dt) for ds, s, dt in APT_MAPPING if ds in apt]
        apt = reconcile_lengths("apt", apt, apt_mapping_present)
        print(f"  {len(apt)} apt/* stem(s) recovered from chunk files.")
        zeroed = [ds for ds, _, _ in APT_MAPPING if ds not in apt]
    else:
        print("  No apt_* chunk files found (experiment predates chunk flushing).")
        print("  Synthesizing apt/* from dld/start_counter …")
        apt = synthesize_apt_group(reconciled)
        zeroed = ["apt/temperature", "apt/experiment_chamber_vacuum", "apt/timestamps"]

    for ds_path, arr in apt.items():
        note = " [ZEROED]" if ds_path in zeroed else ""
        print(f"  {ds_path:45s}  {arr.shape}{note}")

    # ---- check logs for failure reason ----------------------------------
    _check_logs(exp_dir)

    # ---- write HDF5 -----------------------------------------------------
    output_path = exp_dir / f"{exp_name}.h5"
    try:
        write_hdf5(output_path, reconciled, apt)
    except Exception as exc:
        print(f"\nERROR writing HDF5: {exc}")
        traceback.print_exc()
        return 1

    sz_mb = output_path.stat().st_size / 1e6
    print(f"\nRecovery complete.  Output: {output_path}  ({sz_mb:.1f} MB)")
    if zeroed:
        print()
        print("NOTE: the following apt/* fields were zero-filled (no chunk data found):")
        for ds in zeroed:
            print(f"  {ds}")
        print("These fields are not used by the calibration pipeline.")

    stats = exp_dir / "meta_data" / f"{exp_name}.txt"
    if stats.exists():
        print(f"\nExperiment statistics: {stats}")
        print("(contains total ions, run time, and voltage summary)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
