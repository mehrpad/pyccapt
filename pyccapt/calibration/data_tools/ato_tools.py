from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pandas as pd


def _optional_column(data: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    """Return a numeric column when present, otherwise a default-filled array."""
    if column in data.columns:
        return pd.to_numeric(data[column], errors="coerce").fillna(default).to_numpy()
    return np.full(len(data), default, dtype=float)


def _detector_column_cm(data: pd.DataFrame, column_cm: str, column_mm: str) -> np.ndarray:
    """Return detector coordinates in centimeters from either cm or mm columns."""
    if column_cm in data.columns:
        return _optional_column(data, column_cm)
    if column_mm in data.columns:
        return _optional_column(data, column_mm) / 10.0
    return np.full(len(data), 0.0, dtype=float)


def ccapt_to_ato(data: pd.DataFrame, path: str | None = None, name: str | None = None) -> bytes:
    """Convert a PyCCAPT dataframe to the ATO v6 binary layout used by this project."""
    required_columns = {"mc (Da)", "t (ns)", "high_voltage (V)"}
    missing = sorted(required_columns - set(data.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"ATO export requires these columns: {missing_text}")

    x_nm = _optional_column(data, "x (nm)")
    y_nm = _optional_column(data, "y (nm)")
    z_nm = _optional_column(data, "z (nm)")
    mc_da = _optional_column(data, "mc (Da)")
    tof_ns = _optional_column(data, "t (ns)")
    voltage_v = _optional_column(data, "high_voltage (V)")
    delta_p = _optional_column(data, "delta_p").astype(np.int32)
    x_det_cm = _detector_column_cm(data, "x_det (cm)", "x_det (mm)")
    y_det_cm = _detector_column_cm(data, "y_det (cm)", "y_det (mm)")
    mcp_amp = _optional_column(data, "mcp_amp").astype(np.uint16)
    atom_ids = np.arange(1, len(data) + 1, dtype=np.uint32)

    payload = bytearray()
    payload.extend(struct.pack("iii", 0, 6, len(data)))

    for index in range(len(data)):
        x_ato = int(np.clip(np.rint(x_nm[index]), np.iinfo(np.int16).min, np.iinfo(np.int16).max))
        y_ato = int(np.clip(np.rint(y_nm[index]), np.iinfo(np.int16).min, np.iinfo(np.int16).max))
        z_ato = float(z_nm[index] * 10.0)
        tof_ato = float(tof_ns[index] / 1000.0)
        x_det_mm = float(x_det_cm[index] * 10.0)
        y_det_mm = float(y_det_cm[index] * 10.0)
        x_det_ato = int(np.clip(np.rint(x_det_mm / 0.01), np.iinfo(np.int16).min, np.iinfo(np.int16).max))
        y_det_ato = int(np.clip(np.rint(y_det_mm / 0.01), np.iinfo(np.int16).min, np.iinfo(np.int16).max))
        voltage_ato = int(np.clip(np.rint(voltage_v[index] / 0.5), 0, np.iinfo(np.uint16).max))
        payload.extend(struct.pack("I", int(atom_ids[index])))
        payload.extend(struct.pack("i", int(delta_p[index])))
        payload.extend(struct.pack("h", x_ato))
        payload.extend(struct.pack("h", y_ato))
        payload.extend(struct.pack("f", z_ato))
        payload.extend(struct.pack("f", float(mc_da[index])))
        payload.extend(struct.pack("f", tof_ato))
        payload.extend(struct.pack("h", x_det_ato))
        payload.extend(struct.pack("h", y_det_ato))
        payload.extend(struct.pack("H", voltage_ato))
        payload.extend(struct.pack("H", int(mcp_amp[index])))
        payload.extend(struct.pack("B", 0))
        payload.extend(struct.pack("H", 0))

    ato_bytes = bytes(payload)
    if path is not None and name is not None:
        target_path = Path(path) / name
        with open(target_path, "wb") as file_handle:
            file_handle.write(ato_bytes)
    return ato_bytes


def ato_to_ccapt(file_path: str, mode: str) -> pd.DataFrame:
    """
    Read data from an .ato file version 6 and convert it into a pandas DataFrame.

    Args:
        file_path: Path to the .ato file
        mode: Type of mode (oxcart/ato)

    Returns:
        Pandas DataFrame containing the converted data
    """
    with open(file_path, 'rb') as f:
        data = f.read()

        zero = struct.unpack('i', data[:4])
        version = struct.unpack('i', data[4:8])
        num_atoms = struct.unpack('i', data[8:12])

        # NOTE: the per-row 35-byte stride below assumes ``num_cluster == 0``
        # for every row. A row with non-zero cluster count is wider than
        # 35 bytes (33 header + num_cluster*2 cluster IDs), so the
        # ``bias = 12 + (35 * (i + 1))`` formula desynchronises and the
        # parser silently returns garbage. We detect that condition and
        # raise before producing wrong data. Variable-stride parsing
        # would require a sequential reader; not yet implemented.
        n = int(num_atoms[0])
        if len(data) >= 12 + 33:
            sample_num_cluster = struct.unpack('B', data[12 + 32 : 12 + 33])[0]
            if sample_num_cluster > 0:
                raise NotImplementedError(
                    "ato_to_ccapt: file uses non-zero num_cluster on the "
                    "first record; the current fixed-stride parser would "
                    "produce silently-wrong output. Variable-stride .ato "
                    "parsing is not implemented yet."
                )
        expected_size = 12 + 35 * n
        if len(data) < expected_size:
            raise ValueError(
                f"ato_to_ccapt: file truncated -- expected at least "
                f"{expected_size} bytes for {n} fixed-stride records, "
                f"got {len(data)}."
            )

        # Fast vectorised path: build a structured dtype matching the
        # fixed 35-byte record layout and use np.frombuffer to parse all
        # rows in one C-level call. This replaces ~28M individual
        # ``struct.unpack`` calls per million atoms (minutes -> seconds).
        record_dtype = np.dtype([
            ('atom_id', '<u4'),
            ('delta_p', '<i4'),
            ('x_raw', '<i2'),
            ('y_raw', '<i2'),
            ('z_raw', '<f4'),
            ('mc_raw', '<f4'),
            ('tof_raw', '<f4'),
            ('x_det_raw', '<i2'),
            ('y_det_raw', '<i2'),
            ('dc_voltage_raw', '<u2'),
            ('mcp_amp', '<u2'),
            ('num_cluster', '<u1'),
            ('_cluster_id_low', '<u1'),
            ('_cluster_id_high', '<u1'),
        ])
        assert record_dtype.itemsize == 35, "record stride must be 35 bytes"
        records = np.frombuffer(data, dtype=record_dtype, count=n, offset=12)

        # All rows assume num_cluster == 0 by the early raise above; verify
        # the rest of the file confirms that to catch corrupt files.
        if records['num_cluster'].any():
            raise NotImplementedError(
                "ato_to_ccapt: encountered non-zero num_cluster mid-stream "
                "after a zero header; the file likely uses variable-stride "
                "records and the fixed-stride parser is not safe to use."
            )

        atom_id = records['atom_id']
        delta_p = records['delta_p']
        x = records['x_raw'].astype(np.int32)
        y = records['y_raw'].astype(np.int32)
        z = records['z_raw'].astype(np.float32) * np.float32(0.1)
        mc = records['mc_raw']
        tof = records['tof_raw'].astype(np.float32) * np.float32(1000.0)
        x_det = records['x_det_raw'].astype(np.float32) * np.float32(0.01)
        y_det = records['y_det_raw'].astype(np.float32) * np.float32(0.01)
        dc_voltage = records['dc_voltage_raw'].astype(np.float32) * np.float32(0.5)
        mcp_amp = records['mcp_amp']
        num_cluster = records['num_cluster']
        cluster_id = [[] for _ in range(n)]  # always empty in fixed-stride mode

        if mode == 'ato':
            data_f = pd.DataFrame(
                {
                    'atom_id': atom_id,
                    'delta_p': delta_p,
                    'x (nm)': x,
                    'y (nm)': y,
                    'z (nm)': z,
                    'mc (Da)': mc,
                    'tof (ns)': tof,
                    'x_det (mm)': x_det,
                    'y_det (mm)': y_det,
                    'dc_voltage (V)': dc_voltage,
                    'mcp_amp': mcp_amp,
                    'num_cluster': num_cluster,
                    'cluster_id': cluster_id,
                }
            )
        elif mode == 'pyccapt':
            # The previous dict had ``'mc (Da)'`` twice (once with zeros,
            # once with the parsed values); the second silently shadowed
            # the first. Keep the parsed mc under 'mc (Da)' and put the
            # zero-initialised calibrated column under 'mc_c (Da)' only.
            n_rows = len(dc_voltage)
            data_f = pd.DataFrame(
                {
                    'x (nm)': np.zeros(n_rows),
                    'y (nm)': np.zeros(n_rows),
                    'z (nm)': np.zeros(n_rows),
                    'mc_c (Da)': np.zeros(n_rows),
                    'high_voltage (V)': dc_voltage,
                    'pulse': np.zeros(n_rows),
                    'start_counter': np.zeros(n_rows),
                    't_c (ns)': np.zeros(n_rows),
                    't (ns)': tof,
                    'mc (Da)': mc,
                    'x_det (mm)': x_det,
                    'y_det (mm)': y_det,
                    'delta_p': delta_p,
                    'multi': np.zeros(n_rows),
                }
            )
        else:
            raise ValueError(f"ato_to_ccapt: unknown mode {mode!r}")
    return data_f
