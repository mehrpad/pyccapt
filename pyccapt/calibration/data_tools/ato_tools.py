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

        atom_id = []
        delta_p = []
        x = []
        y = []
        z = []
        mc = []
        tof = []
        x_det = []
        y_det = []
        dc_voltage = []
        mcp_amp = []
        num_cluster = []
        cluster_id = []
        bias = 12

        for i in range(num_atoms[0]):
            atom_id.append(struct.unpack('I', data[bias:bias+4])[0])
            delta_p.append(struct.unpack('i', data[bias+4:bias+8])[0])
            x.append(struct.unpack('h', data[bias+8:bias+10])[0])
            y.append(struct.unpack('h', data[bias+10:bias+12])[0])
            z.append(struct.unpack('f', data[bias+12:bias+16])[0] * 0.1)
            mc.append(struct.unpack('f', data[bias+16:bias+20])[0])
            tof.append(struct.unpack('f', data[bias+20:bias+24])[0] * 1000)
            x_det.append(struct.unpack('h', data[bias+24:bias+26])[0] * 0.01)
            y_det.append(struct.unpack('h', data[bias+26:bias+28])[0] * 0.01)
            dc_voltage.append(struct.unpack('H', data[bias+28:bias+30])[0] * 0.5)
            mcp_amp.append(struct.unpack('H', data[bias+30:bias+32])[0])
            num_cluster_tmp = struct.unpack('B', data[bias+32:bias+33])[0]
            num_cluster.append(num_cluster_tmp)

            if num_cluster_tmp > 0:
                cluster_id.append(list(struct.unpack('H' * num_cluster_tmp, data[bias+33:bias+33+num_cluster_tmp*2])))
            else:
                cluster_id.append([])

            bias = 12 + (35 * (i+1))

        if mode == 'ato':
            data_f = pd.DataFrame({'atom_id': atom_id,
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
                                   'cluster_id': cluster_id})
        elif mode == 'pyccapt':
            data_f = pd.DataFrame({
                'x (nm)': np.zeros(len(dc_voltage)),
                'y (nm)': np.zeros(len(dc_voltage)),
                'z (nm)': np.zeros(len(dc_voltage)),
                'mc_c (Da)': np.zeros(len(dc_voltage)),
                'mc (Da)': np.zeros(len(dc_voltage)),
                'high_voltage (V)': dc_voltage,
                'pulse': np.zeros(len(dc_voltage)),
                'start_counter': np.zeros(len(dc_voltage)),
                't_c (ns)': np.zeros(len(dc_voltage)),
                't (ns)': tof,
                'mc (Da)': mc,
                'x_det (mm)': x_det,
                'y_det (mm)': y_det,
                'delta_p': delta_p,
                'multi': np.zeros(len(dc_voltage))})
    return data_f

