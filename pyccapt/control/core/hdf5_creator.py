from __future__ import annotations

import os
import re
from pathlib import Path

import h5py
import numpy as np

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

    dataset = hdf_file.create_dataset(dataset_name, (total_size,), dtype=target_dtype)
    offset = 0
    for chunk_file, chunk_size in zip(chunk_files, chunk_sizes):
        # Stream the chunk via mmap so the whole file isn't pulled into
        # RAM at once; h5py copies from the mapped view. Assert the chunk
        # dtype matches the destination -- a mismatched chunk (e.g. an
        # interrupted run restarted under new code) would otherwise be
        # silently up/down-cast or wrap.
        chunk_array = np.load(chunk_file, mmap_mode="r")
        if np.dtype(chunk_array.dtype) != target_dtype:
            raise ValueError(
                f"Chunk {chunk_file.name} has dtype {chunk_array.dtype}, "
                f"expected {target_dtype} for dataset {dataset_name!r}. "
                f"Refusing to silently cast acquisition data."
            )
        dataset[offset : offset + chunk_size] = chunk_array[...]
        offset += chunk_size
        del chunk_array


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
    hdf_file.create_dataset(dataset_name, data=dataset_data, dtype=dtype)


def _write_surface_concept_detector_data(hdf_file, variables) -> None:
    temp_data_dir = Path(variables.path) / "temp_data"
    chunk_dir = temp_data_dir / "chunks"
    chunk_mode = chunk_dir.is_dir() and any(chunk_dir.glob("*_chunk_*.npy"))

    chunk_mapping = (
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
    fallback_mapping = (
        ("dld/x", variables.x, np.float64),
        ("dld/y", variables.y, np.float64),
        ("dld/t", variables.t, np.float64),
        ("dld/high_voltage", variables.main_v_dc_dld, np.float64),
        ("dld/voltage_pulse", variables.main_v_p_dld, np.float64),
        ("dld/laser_pulse", variables.main_l_p_dld, np.float64),
        ("dld/start_counter", variables.dld_start_counter, np.uint64),
        ("tdc/channel", variables.channel, np.uint32),
        ("tdc/time_data", variables.time_data, np.uint64),
        ("tdc/start_counter", variables.tdc_start_counter, np.uint64),
        ("tdc/high_voltage", variables.main_v_dc_tdc, np.float64),
        ("tdc/voltage_pulse", variables.main_v_p_tdc, np.float64),
        ("tdc/laser_pulse", variables.main_l_p_tdc, np.float64),
    )

    if chunk_mode:
        for dataset_name, chunk_stem, dtype in chunk_mapping:
            chunk_files = _sorted_chunk_files(chunk_dir, chunk_stem)
            if chunk_files:
                _write_chunked_dataset(hdf_file, dataset_name, chunk_files, dtype)
    else:
        for dataset_name, values, dtype in fallback_mapping:
            _create_dataset(hdf_file, dataset_name, values, dtype)


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
    try:
        with h5py.File(tmp_path, "w") as hdf_file:
            _create_dataset(hdf_file, "apt/id", time_counter, np.uint64)
            _create_dataset(hdf_file, "apt/num_events", variables.main_counter, np.uint32)
            _create_dataset(hdf_file, "apt/num_raw_signals", variables.main_raw_counter, np.uint32)
            _create_dataset(hdf_file, "apt/temperature", variables.main_temperature, np.float64)
            _create_dataset(hdf_file, "apt/experiment_chamber_vacuum", variables.main_chamber_vacuum, np.float64)
            _create_dataset(hdf_file, "apt/timestamps", time_ex, np.float64)

            if conf["tdc"] == "on" and conf["tdc_model"] == "Surface_Consept" and variables.counter_source == "TDC":
                _write_surface_concept_detector_data(hdf_file, variables)

            elif conf["tdc"] == "on" and conf["tdc_model"] == "RoentDek" and variables.counter_source == "TDC":
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

            elif conf["tdc"] == "on" and conf["tdc_model"] == "HSD" and variables.counter_source == "HSD":
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
