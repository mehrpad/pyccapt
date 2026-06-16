from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import h5py
import numpy as np

from pyccapt.control.apt.detector_models import normalize_tdc_model

logger = logging.getLogger("apt")

_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _sanitize_for_path(name: str) -> str:
    """Replace characters that are illegal in Windows file names.

    The experiment name is composed from user-typed fields (electrode,
    hdf5_data_name) and may legitimately contain spaces, but a stray ``/`` or
    ``:`` typed by the operator would otherwise break path creation.
    """
    cleaned = _INVALID_FILENAME_CHARS.sub('_', str(name)).rstrip(' .')
    return cleaned or 'experiment'


def _sorted_chunk_files(chunk_dir: Path, stem: str) -> list[Path]:
    pattern = re.compile(rf"^{re.escape(stem)}_chunk_(\d+)\.npy$")
    files_with_ids: list[tuple[int, Path]] = []
    for path in chunk_dir.glob(f"{stem}_chunk_*.npy"):
        match = pattern.match(path.name)
        if match is not None:
            files_with_ids.append((int(match.group(1)), path))
    return [path for _, path in sorted(files_with_ids)]


# Stems written by APT_Exp_Control._flush_apt_meta_chunks() during the run.
# Each maps to: (hdf5_dataset_path, numpy_dtype)
_APT_CHUNK_STEMS: list[tuple[str, str, str]] = [
    ("apt_id",              "apt/id",                          "uint64"),
    ("apt_timestamps",      "apt/timestamps",                  "float64"),
    ("apt_num_events",      "apt/num_events",                  "uint32"),
    ("apt_num_raw_signals", "apt/num_raw_signals",             "uint32"),
    ("apt_temperature",     "apt/temperature",                 "float64"),
    ("apt_vacuum",          "apt/experiment_chamber_vacuum",   "float64"),
	# Stage positions in meters, logged once per experiment iteration.
	("apt_laser_x", "apt/laser_x", "float64"),
	("apt_laser_y", "apt/laser_y", "float64"),
	("apt_laser_z", "apt/laser_z", "float64"),
	("apt_stage_x", "apt/stage_x", "float64"),
	("apt_stage_y", "apt/stage_y", "float64"),
	("apt_stage_z", "apt/stage_z", "float64"),
]


def _load_apt_from_chunks(chunk_dir: Path) -> dict[str, np.ndarray] | None:
    """Load apt/* metadata from chunk files.

    Returns a dict {hdf5_path: array} if at least one apt chunk stem is present,
    or None if no apt chunk files exist (so callers can fall back to in-memory data).
    """
    result: dict[str, np.ndarray] = {}
    any_found = False
    for stem, ds_path, dtype in _APT_CHUNK_STEMS:
        files = _sorted_chunk_files(chunk_dir, stem)
        if not files:
            continue
        any_found = True
        target = np.dtype(dtype)
        parts: list[np.ndarray] = []
        for f in files:
            try:
                arr = np.load(f, mmap_mode="r")
                if arr.size == 0:
                    continue
                if arr.dtype != target:
                    arr = arr.astype(target)
                parts.append(arr.copy())
                del arr
            except Exception:
                pass
        if parts:
            result[ds_path] = np.concatenate(parts)
    return result if any_found else None


def _coerce_chunk_to_target(values: np.ndarray, target_dtype: np.dtype,
                            chunk_file: Path, dataset_name: str) -> np.ndarray:
    """Return *values* as *target_dtype*, casting only when it is lossless.

    The detector chunk writer (tdc_surface_concept.save_chunk_worker) builds
    integer counter/channel/time arrays from Python ints, so older chunks were
    saved as the platform default int64 while the HDF5 schema declares
    uint64/uint32.  That widening is lossless for the non-negative values
    acquisition produces, so we perform it rather than refusing the whole file.

    A cast that would actually lose information is still refused -- that signals
    genuinely corrupt or incompatible data, not the benign int64-vs-uint64 label
    difference:
      * negative value into an unsigned dataset, or any out-of-range overflow
      * a fractional float into an integer dataset
    """
    src_dtype = np.dtype(values.dtype)
    if src_dtype == target_dtype:
        return values

    if np.issubdtype(src_dtype, np.floating) and np.issubdtype(target_dtype, np.integer):
        if not np.all(np.isfinite(values)) or np.any(values != np.rint(values)):
            raise ValueError(
                f"Chunk {chunk_file.name} for dataset {dataset_name!r} holds "
                f"non-integer values incompatible with {target_dtype}. "
                f"Refusing to truncate acquisition data."
            )

    if np.issubdtype(target_dtype, np.integer) and values.size:
        info = np.iinfo(target_dtype)
        vmin = int(values.min())
        vmax = int(values.max())
        if vmin < info.min or vmax > info.max:
            raise ValueError(
                f"Chunk {chunk_file.name} for dataset {dataset_name!r} holds "
                f"values [{vmin}, {vmax}] outside the {target_dtype} range "
                f"[{info.min}, {info.max}]. Refusing to wrap acquisition data."
            )

    return values.astype(target_dtype)


# HDF5 chunk size (in elements) for compressed 1-D datasets. ~8 MiB per
# chunk for an 8-byte dtype: large enough that lzf compresses the
# channel-0-dominated raw arrays and the tiled per-pulse columns well,
# small enough that the calibration pipeline's partial reads stay cheap.
_HDF5_COMPRESS_CHUNK = 1 << 20  # 1,048,576 elements


def _compression_opts(n_elements: int) -> dict:
	"""h5py create_dataset kwargs enabling lzf compression for large arrays.

	Compression requires chunked storage, which adds overhead that isn't
	worth it for small datasets (apt/*, short dld/*). Only arrays at least
	one chunk long are compressed. lzf is h5py's BUILT-IN filter (no
	external dependency, always readable wherever h5py is installed) and is
	fast enough not to bottleneck finalization. Read is transparent -- the
	calibration pipeline needs no change.
	"""
	if n_elements < _HDF5_COMPRESS_CHUNK:
		return {}
	return {"compression": "lzf", "chunks": (_HDF5_COMPRESS_CHUNK,)}


def _write_chunked_dataset(hdf_file, dataset_name: str, chunk_files: list[Path], dtype) -> None:
    # First pass: size each chunk via a memory-mapped header read (no full
    # load) and cache the sizes so the write pass doesn't re-open files.
    target_dtype = np.dtype(dtype)
    chunk_sizes: list[int] = []
    total_size = 0
    for chunk_file in chunk_files:
        chunk_array = np.load(chunk_file, mmap_mode="r")
        size = int(chunk_array.shape[0])
        chunk_sizes.append(size)
        total_size += size
        # Release the mmap handle promptly.
        del chunk_array

    dataset = hdf_file.create_dataset(
	    dataset_name, (total_size,), dtype=target_dtype, **_compression_opts(total_size)
    )
    offset = 0
    cast_from: np.dtype | None = None
    for chunk_file, chunk_size in zip(chunk_files, chunk_sizes):
        # Stream the chunk via mmap so the whole file isn't pulled into RAM at
        # once.  A chunk whose dtype differs from the destination (e.g. the
        # int64 counters older writers produced, or a run restarted under new
        # code) is cast *only when that cast is provably lossless* -- otherwise
        # _coerce_chunk_to_target raises rather than silently wrap/truncate.
        chunk_array = np.load(chunk_file, mmap_mode="r")
        if np.dtype(chunk_array.dtype) != target_dtype:
            cast_from = np.dtype(chunk_array.dtype)
        values = _coerce_chunk_to_target(
            np.asarray(chunk_array), target_dtype, chunk_file, dataset_name
        )
        dataset[offset: offset + chunk_size] = values
        offset += chunk_size
        del chunk_array, values

    if cast_from is not None:
        logger.warning(
            "Dataset %r: chunk dtype %s differed from schema %s; values were "
            "losslessly cast on write (chunk-writer dtype drift).",
            dataset_name, cast_from, target_dtype,
        )


def _coerce_numeric_array(data, dtype):
    target_dtype = np.dtype(dtype)
    values = np.asarray(data)
    needs_string_conversion = values.dtype.kind in {"U", "S", "O"} and (
        np.issubdtype(target_dtype, np.floating) or np.issubdtype(target_dtype, np.integer)
    )
    if not needs_string_conversion:
        return values.astype(target_dtype, copy=False)

    def _normalize_string(value):
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        text = str(value).strip()
        return text

    def _convert_value(value):
        text = _normalize_string(value)
        if not text or text.lower() in {"n/a", "nan", "none"}:
            return np.nan if np.issubdtype(target_dtype, np.floating) else 0
        try:
            numeric_value = float(text)
        except (TypeError, ValueError):
            return np.nan if np.issubdtype(target_dtype, np.floating) else 0
        if np.issubdtype(target_dtype, np.integer):
            return int(numeric_value)
        return numeric_value

    if values.ndim == 0:
        return np.asarray(_convert_value(values.item()), dtype=target_dtype)

    flat_values = [_convert_value(value) for value in values.reshape(-1)]
    return np.asarray(flat_values, dtype=target_dtype).reshape(values.shape)


def _create_dataset(hdf_file, dataset_name: str, data, dtype) -> None:
    dataset_data = _coerce_numeric_array(data, dtype)
    hdf_file.create_dataset(
	    dataset_name, data=dataset_data, dtype=dtype, **_compression_opts(dataset_data.size)
    )


def _write_surface_concept_detector_data(hdf_file, variables) -> None:
	chunk_dir = Path(variables.path) / "temp_data" / "chunks"
	chunk_dir_exists = chunk_dir.is_dir()

	# (hdf5 dataset, chunk stem, in-memory fallback attr on `variables`, dtype).
	# The DLD and raw/TDC streams have DIFFERENT lengths (one DLD event can
	# yield several raw channel hits, and many raw hits never complete a DLD
	# event) and are chunked INDEPENDENTLY by the acquisition process. Each
	# dataset is therefore resolved on its own: use its chunk files if any
	# exist, otherwise fall back to the in-memory array. This correctly
	# handles a run that chunked the fast raw stream but not the smaller DLD
	# stream (or vice versa) -- the old all-or-nothing chunk_mode wrote
	# NOTHING for the un-chunked stream, silently dropping it.
	combined_mapping = (
		("dld/x", "x", "x", np.float64),
		("dld/y", "y", "y", np.float64),
		("dld/t", "t", "t", np.float64),
		("dld/high_voltage", "voltage", "main_v_dc_dld", np.float64),
		("dld/voltage_pulse", "voltage_pulse", "main_v_p_dld", np.float64),
		("dld/laser_pulse", "laser_pulse", "main_l_p_dld", np.float64),
		("dld/start_counter", "start_counter", "dld_start_counter", np.uint64),
		("tdc/channel", "channel", "channel", np.uint32),
		("tdc/time_data", "time", "time_data", np.uint64),
		("tdc/start_counter", "tdc_start_counter", "tdc_start_counter", np.uint64),
		("tdc/high_voltage", "voltage_tdc", "main_v_dc_tdc", np.float64),
		("tdc/voltage_pulse", "voltage_pulse_tdc", "main_v_p_tdc", np.float64),
		("tdc/laser_pulse", "laser_pulse_tdc", "main_l_p_tdc", np.float64),
    )

	for dataset_name, chunk_stem, var_attr, dtype in combined_mapping:
		chunk_files = _sorted_chunk_files(chunk_dir, chunk_stem) if chunk_dir_exists else []
		if chunk_files:
			_write_chunked_dataset(hdf_file, dataset_name, chunk_files, dtype)
		else:
			# Lazy: only fetch the (possibly large, Manager-backed) array when
			# this stream was NOT chunked, so a fully-chunked run never pulls
			# the bulk acquisition arrays back through the Manager.
			_create_dataset(hdf_file, dataset_name, getattr(variables, var_attr), dtype)


def hdf_creator(variables, conf, time_counter, time_ex):
    """
    Save experiment data to an HDF5 file.

    Args:
            variables (object): An object containing experiment variables.
            conf (dict): A dictionary containing configuration settings.
            time_counter (list): A list of time counter data.
            time_ex (list): A list of timestamp of iteration.

    Returns:
            None
    """

    safe_name = _sanitize_for_path(variables.exp_name)
    path = Path(variables.path) / f"{safe_name}.h5"
    # Write to a sibling .tmp file first and rename atomically. If the write
    # fails or the process dies mid-write, the previous .h5 (if any) remains
    # intact, and we are left with at most a partial .tmp that can be
    # deleted manually.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tdc_model = normalize_tdc_model(conf.get("tdc_model")) if conf.get("tdc") == "on" else ""
    chunk_dir = Path(variables.path) / "temp_data" / "chunks"
    try:
        with h5py.File(tmp_path, "w") as hdf_file:
            # apt/* group: prefer chunk files written during the run (crash-safe),
            # fall back to the in-memory lists for backwards-compatibility with
            # experiments that ran before chunk flushing was introduced.
            apt_from_chunks = _load_apt_from_chunks(chunk_dir)
            if apt_from_chunks is not None:
                for ds_path, arr in apt_from_chunks.items():
                    hdf_file.create_dataset(ds_path, data=arr)
                # Fill any stems that had no chunk files with in-memory data so
                # the apt group is always structurally complete.
                written = set(apt_from_chunks.keys())
                if "apt/id"                         not in written:
                    _create_dataset(hdf_file, "apt/id", time_counter, np.uint64)
                if "apt/num_events"                 not in written:
                    _create_dataset(hdf_file, "apt/num_events", variables.main_counter, np.uint32)
                if "apt/num_raw_signals"            not in written:
                    _create_dataset(hdf_file, "apt/num_raw_signals", variables.main_raw_counter, np.uint32)
                if "apt/temperature"                not in written:
                    _create_dataset(hdf_file, "apt/temperature", variables.main_temperature, np.float64)
                if "apt/experiment_chamber_vacuum"  not in written:
                    _create_dataset(hdf_file, "apt/experiment_chamber_vacuum", variables.main_chamber_vacuum, np.float64)
                if "apt/timestamps"                 not in written:
                    _create_dataset(hdf_file, "apt/timestamps", time_ex, np.float64)
            else:
                # No apt chunks: use the in-memory lists (pre-chunk-flush experiments).
                _create_dataset(hdf_file, "apt/id", time_counter, np.uint64)
                _create_dataset(hdf_file, "apt/num_events", variables.main_counter, np.uint32)
                _create_dataset(hdf_file, "apt/num_raw_signals", variables.main_raw_counter, np.uint32)
                _create_dataset(hdf_file, "apt/temperature", variables.main_temperature, np.float64)
                _create_dataset(hdf_file, "apt/experiment_chamber_vacuum", variables.main_chamber_vacuum, np.float64)
                _create_dataset(hdf_file, "apt/timestamps", time_ex, np.float64)

            if conf["tdc"] == "on" and tdc_model == "Surface_Concept" and variables.counter_source == "TDC":
                _write_surface_concept_detector_data(hdf_file, variables)

            elif conf["tdc"] == "on" and tdc_model == "RoentDek" and variables.counter_source == "TDC":
                _create_dataset(hdf_file, "dld/x", variables.x, np.float64)
                _create_dataset(hdf_file, "dld/y", variables.y, np.float64)
                _create_dataset(hdf_file, "dld/t", variables.t, np.float64)
                _create_dataset(hdf_file, "dld/high_voltage", variables.main_v_dc_dld, np.float64)
                _create_dataset(hdf_file, "dld/voltage_pulse", variables.main_v_p_dld, np.float64)
                _create_dataset(hdf_file, "dld/laser_pulse", variables.main_l_p_dld, np.float64)
                _create_dataset(hdf_file, "dld/start_counter", variables.time_stamp, np.uint64)
                # RoentDek raw: convert per-channel arrays ch0..ch7 (one entry
                # per pulse trigger per channel) into the same flat
                # (channel, time_data, start_counter, ...) layout that
                # Surface Concept uses, so the calibration loader at
                # data_loadcrop.fetch_dataset_from_dld_grp(extract_mode='tdc_ro')
                # can read both detectors with one code path. Each event
                # contributes 8 rows (one per channel); rows where the
                # channel did not fire (raw value == 0) are kept so the
                # per-event grouping by start_counter stays intact, but the
                # downstream partial-hit recovery can filter them via
                # ``time_data != 0``.
                _ch_stack = np.column_stack(
                    [
                        np.asarray(variables.ch0, dtype=np.uint64).reshape(-1),
                        np.asarray(variables.ch1, dtype=np.uint64).reshape(-1),
                        np.asarray(variables.ch2, dtype=np.uint64).reshape(-1),
                        np.asarray(variables.ch3, dtype=np.uint64).reshape(-1),
                        np.asarray(variables.ch4, dtype=np.uint64).reshape(-1),
                        np.asarray(variables.ch5, dtype=np.uint64).reshape(-1),
                        np.asarray(variables.ch6, dtype=np.uint64).reshape(-1),
                        np.asarray(variables.ch7, dtype=np.uint64).reshape(-1),
                    ]
                )  # shape (n_events, 8)
                _n_events = _ch_stack.shape[0]

                # Validate per-event auxiliary arrays are long enough BEFORE
                # slicing. Previously ``[:_n_events]`` silently truncated a
                # short array, writing tdc/* groups of inconsistent length
                # that crash the calibration reader much later. Fail loudly
                # at write time instead.
                _start_counter_per_event = np.asarray(variables.time_stamp, dtype=np.uint64).reshape(-1)
                _hv_per_event = np.asarray(variables.main_v_dc_tdc, dtype=np.float64).reshape(-1)
                _vp_per_event = np.asarray(variables.main_v_p_tdc, dtype=np.float64).reshape(-1)
                _lp_per_event = np.asarray(variables.main_l_p_tdc, dtype=np.float64).reshape(-1)
                for _name, _arr in (
                    ("time_stamp", _start_counter_per_event),
                    ("main_v_dc_tdc", _hv_per_event),
                    ("main_v_p_tdc", _vp_per_event),
                    ("main_l_p_tdc", _lp_per_event),
                ):
                    if _arr.shape[0] < _n_events:
                        raise ValueError(
                            f"RoentDek auxiliary array {_name!r} has length "
                            f"{_arr.shape[0]} < {_n_events} events; cannot build "
                            f"consistent tdc/* groups. (Did the per-event and "
                            f"per-channel buffers desync during acquisition?)"
                        )

                # MEMORY: build-write-free each flat array in turn so peak
                # RAM is ~ _ch_stack + ONE 8*n_events flat array, not all six
                # at once (~17 GB for 50M events with the old up-front build).
                # tdc/time_data is a view of _ch_stack, so keep _ch_stack
                # alive until that one is written.
                _create_dataset(hdf_file, "tdc/time_data", _ch_stack.reshape(-1), np.uint64)
                del _ch_stack

                _channel_flat = np.tile(np.arange(8, dtype=np.uint32), _n_events)
                _create_dataset(hdf_file, "tdc/channel", _channel_flat, np.uint32)
                del _channel_flat

                _create_dataset(
                    hdf_file, "tdc/start_counter",
                    np.repeat(_start_counter_per_event[:_n_events], 8), np.uint64,
                )
                _create_dataset(
                    hdf_file, "tdc/high_voltage",
                    np.repeat(_hv_per_event[:_n_events], 8), np.float64,
                )
                _create_dataset(
                    hdf_file, "tdc/voltage_pulse",
                    np.repeat(_vp_per_event[:_n_events], 8), np.float64,
                )
                _create_dataset(
                    hdf_file, "tdc/laser_pulse",
                    np.repeat(_lp_per_event[:_n_events], 8), np.float64,
                )

            elif conf["tdc"] == "on" and tdc_model == "HSD" and variables.counter_source == "HSD":
                # DRS readout: GetTime returns ns and GetWave returns mV as
                # C float — both are signed real values, NOT unsigned ints.
                # Casting to uint64 (the previous behaviour) silently
                # truncated fractional ns and wrapped negative mV samples
                # (range ±500 mV at SetInputRange(0)) to ~1.8e19, ruining
                # every saved HSD file. Persist as float32 to match the
                # native dtype.
                _create_dataset(hdf_file, "hsd/ch0_time", variables.ch0_time, np.float32)
                _create_dataset(hdf_file, "hsd/ch0_wave", variables.ch0_wave, np.float32)
                _create_dataset(hdf_file, "hsd/ch1_time", variables.ch1_time, np.float32)
                _create_dataset(hdf_file, "hsd/ch1_wave", variables.ch1_wave, np.float32)
                _create_dataset(hdf_file, "hsd/ch2_time", variables.ch2_time, np.float32)
                _create_dataset(hdf_file, "hsd/ch2_wave", variables.ch2_wave, np.float32)
                _create_dataset(hdf_file, "hsd/ch3_time", variables.ch3_time, np.float32)
                _create_dataset(hdf_file, "hsd/ch3_wave", variables.ch3_wave, np.float32)
                # ch4/ch5 and laser_pulse are not produced by drs.experiment_measure,
                # so we don't write empty datasets for them.
                _create_dataset(hdf_file, "hsd/high_voltage", variables.main_v_dc_drs, np.float64)
                _create_dataset(hdf_file, "hsd/voltage_pulse", variables.main_v_p_drs, np.float64)
        # h5py has flushed and closed the file. Atomically replace any prior
        # .h5 file in this folder. ``os.replace`` is atomic on POSIX and
        # atomic-or-best-effort on Windows.
        os.replace(tmp_path, path)
    except Exception:
        # Tidy the partial file so the experiment folder is not littered
        # with .h5.tmp leftovers.
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise
