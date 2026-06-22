import struct
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd

# Local module and scripts
from pyccapt.calibration.leap_tools import leap_tools


def ccapt_to_pos(data, path=None, name=None):
    """
    Convert CCAPT data to POS format.

    Args:
        data (pandas.DataFrame): CCAPT data.
        path (str): Optional. Path to save the POS file.
        name (str): Optional. Name of the POS file.

    Returns:
        bytes: POS data.

    """
    dd = data[['x (nm)', 'y (nm)', 'z (nm)', 'mc (Da)']]
    dd = dd.astype(np.single)
    records = dd.to_records(index=False)
    list_records = list(records)
    d = tuple(chain(*list_records))
    pos = struct.pack('>' + 'ffff' * len(dd), *d)
    if name is not None:
        with open(path + name, 'w+b') as f:
            f.write(pos)
    return pos


def ccapt_to_epos(data, path=None, name=None, chunk_size=1_000_000):
    """
    Convert CCAPT data to EPOS format, processing in chunks to avoid memory errors.

    Args:
        data (pandas.DataFrame): CCAPT data.
        path (str): Optional. Path to save the EPOS file.
        name (str): Optional. Name of the EPOS file.
        chunk_size (int): Number of rows to process in each chunk.

    Returns:
        None: Writes EPOS data to file if path and name are provided.
    """

    dd = data[
        [
            'x (nm)',
            'y (nm)',
            'z (nm)',
            'mc (Da)',
            't (ns)',
            'high_voltage (V)',
            'pulse_v (V)',
            'x_det (cm)',
            'y_det (cm)',
            'delta_p',
            'multi',
        ]
    ]
    dd['x_det (cm)'] = dd['x_det (cm)'] * 10
    dd['y_det (cm)'] = dd['y_det (cm)'] * 10

    dd = dd.astype(np.single)
    dd = dd.astype({'delta_p': np.uintc})
    dd = dd.astype({'multi': np.uintc})

    if name is not None:
        with open(path + name, 'w+b') as f:
            for i in range(0, len(dd), chunk_size):
                chunk = dd.iloc[i : i + chunk_size]
                records = chunk.to_records(index=False)
                list_records = list(records)
                d = tuple(chain(*list_records))
                epos_chunk = struct.pack('>' + 'fffffffffII' * len(chunk), *d)
                f.write(epos_chunk)
    else:
        epos = b''
        for i in range(0, len(dd), chunk_size):
            chunk = dd.iloc[i : i + chunk_size]
            records = chunk.to_records(index=False)
            list_records = list(records)
            d = tuple(chain(*list_records))
            epos_chunk = struct.pack('>' + 'fffffffffII' * len(chunk), *d)
            epos += epos_chunk
        return epos


def pos_to_ccapt(file_path):
    """
    Convert POS data to CCAPT format.

    Args:
        file_path: POS data file_path.

    Returns:
        pandas.DataFrame: CCAPT data.

    """
    pos = leap_tools.read_pos(file_path)
    length = len(pos)
    ccapt = pd.DataFrame(
        {
            'x (nm)': pos['x (nm)'].to_numpy(dtype=np.float32, copy=False),
            'y (nm)': pos['y (nm)'].to_numpy(dtype=np.float32, copy=False),
            'z (nm)': pos['z (nm)'].to_numpy(dtype=np.float32, copy=False),
            'mc (Da)': pos['m/n (Da)'].to_numpy(dtype=np.float32, copy=False),
            'mc_uc (Da)': np.zeros(length, dtype=np.float32),
            'high_voltage (V)': np.zeros(length, dtype=np.float32),
            'pulse_v (V)': np.zeros(length, dtype=np.float32),
            'pulse_l (pJ)': np.zeros(length, dtype=np.float32),
            't (ns)': np.zeros(length, dtype=np.float32),
            't_c (ns)': np.zeros(length, dtype=np.float32),
            'x_det (cm)': np.zeros(length, dtype=np.float32),
            'y_det (cm)': np.zeros(length, dtype=np.float32),
            'delta_p': np.zeros(length, dtype=np.int32),
            'multi': np.zeros(length, dtype=np.int32),
            'start_counter': np.zeros(length, dtype=np.int32),
        }
    )
    return ccapt


def epos_to_ccapt(file_path):
    """
    Convert EPOS data to PyCCAPT format.

    Args:
        file_path: EPOS data file path.

    Returns:
        pandas.DataFrame: CCAPT data.

    """
    epos = leap_tools.read_epos(file_path)
    length = len(epos)
    ccapt = pd.DataFrame(
        {
            'x (nm)': epos['x (nm)'].to_numpy(dtype=np.float32, copy=False),
            'y (nm)': epos['y (nm)'].to_numpy(dtype=np.float32, copy=False),
            'z (nm)': epos['z (nm)'].to_numpy(dtype=np.float32, copy=False),
            'mc (Da)': epos['m/n (Da)'].to_numpy(dtype=np.float32, copy=False),
            'mc_uc (Da)': np.zeros(length, dtype=np.float32),
            'high_voltage (V)': epos['HV_DC (V)'].to_numpy(dtype=np.float32, copy=False),
            'pulse_v (V)': epos['pulse (V)'].to_numpy(dtype=np.float32, copy=False),
            'pulse_l (pJ)': np.zeros(length, dtype=np.float32),
            't (ns)': epos['TOF (ns)'].to_numpy(dtype=np.float32, copy=False),
            't_c (ns)': np.zeros(length, dtype=np.float32),
            'x_det (cm)': epos['det_x (mm)'].to_numpy(dtype=np.float32, copy=False) / 10,
            'y_det (cm)': epos['det_y (mm)'].to_numpy(dtype=np.float32, copy=False) / 10,
            'delta_p': epos['pslep'].to_numpy(dtype=np.int32, copy=False),
            'multi': epos['ipp'].to_numpy(dtype=np.int32, copy=False),
            'start_counter': np.zeros(length, dtype=np.int32),
        }
    )
    return ccapt


def epos_lazy_to_ccapt_chunks(epos_table, chunk_size: int = 1 << 20):
    """Stream a memory-mapped EPOS table as PyCCAPT-format DataFrame chunks.

    Args:
        epos_table: ``LazyTable`` from
            :func:`pyccapt.calibration.leap_tools.leap_tools.read_epos_lazy`.
        chunk_size: Number of rows per yielded DataFrame.

    Yields:
        pandas.DataFrame: A PyCCAPT-format chunk with the standard 15 columns.

    The conversion is the same as :func:`epos_to_ccapt` but never holds the
    whole file in RAM; peak resident memory is bounded by ``chunk_size`` rows
    (about 60 bytes/row).
    """
    n_rows = epos_table.n_rows
    if n_rows == 0:
        # Yield one empty frame so downstream writers see the right schema.
        yield pd.DataFrame(
            {
                'x (nm)': np.empty(0, dtype=np.float32),
                'y (nm)': np.empty(0, dtype=np.float32),
                'z (nm)': np.empty(0, dtype=np.float32),
                'mc (Da)': np.empty(0, dtype=np.float32),
                'mc_uc (Da)': np.empty(0, dtype=np.float32),
                'high_voltage (V)': np.empty(0, dtype=np.float32),
                'pulse_v (V)': np.empty(0, dtype=np.float32),
                'pulse_l (pJ)': np.empty(0, dtype=np.float32),
                't (ns)': np.empty(0, dtype=np.float32),
                't_c (ns)': np.empty(0, dtype=np.float32),
                'x_det (cm)': np.empty(0, dtype=np.float32),
                'y_det (cm)': np.empty(0, dtype=np.float32),
                'delta_p': np.empty(0, dtype=np.int32),
                'multi': np.empty(0, dtype=np.int32),
                'start_counter': np.empty(0, dtype=np.int32),
            }
        )
        return
    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        length = stop - start
        yield pd.DataFrame(
            {
                'x (nm)': epos_table['x (nm)'][start:stop].astype(np.float32, copy=False),
                'y (nm)': epos_table['y (nm)'][start:stop].astype(np.float32, copy=False),
                'z (nm)': epos_table['z (nm)'][start:stop].astype(np.float32, copy=False),
                'mc (Da)': epos_table['m/n (Da)'][start:stop].astype(np.float32, copy=False),
                'mc_uc (Da)': np.zeros(length, dtype=np.float32),
                'high_voltage (V)': epos_table['HV_DC (V)'][start:stop].astype(np.float32, copy=False),
                'pulse_v (V)': epos_table['pulse (V)'][start:stop].astype(np.float32, copy=False),
                'pulse_l (pJ)': np.zeros(length, dtype=np.float32),
                't (ns)': epos_table['TOF (ns)'][start:stop].astype(np.float32, copy=False),
                't_c (ns)': np.zeros(length, dtype=np.float32),
                'x_det (cm)': epos_table['det_x (mm)'][start:stop].astype(np.float32, copy=False) / 10.0,
                'y_det (cm)': epos_table['det_y (mm)'][start:stop].astype(np.float32, copy=False) / 10.0,
                'delta_p': epos_table['pslep'][start:stop].astype(np.int32, copy=False),
                'multi': epos_table['ipp'][start:stop].astype(np.int32, copy=False),
                'start_counter': np.zeros(length, dtype=np.int32),
            }
        )


def epos_to_ccapt_h5_streaming(epos_path, h5_output_path, *, chunk_size: int = 1 << 20, progress_callback=None):
    """Stream-convert an EPOS file to an uncorrected PyCCAPT HDF5.

    This is the plain (no reflectron correction) sibling of
    :func:`pyccapt.calibration.reflectron_correction.core.correct_epos_streaming`.
    It memory-maps the EPOS via :func:`leap_tools.read_epos_lazy`, converts it to
    the PyCCAPT 15-column convention one chunk at a time, and appends each chunk
    to the output HDF5 with ``format='table'``. Peak resident memory is bounded
    by ``chunk_size`` rows instead of the whole file, so a multi-GB EPOS converts
    comfortably on a small-RAM machine.

    The output is written under key ``"df"`` -- the same key the calibration
    data loader expects -- and is readable both with ``pd.read_hdf(path, key='df')``
    and in iterator mode (``store.select('df', iterator=True, chunksize=...)``).

    Args:
        epos_path: Path to the ``.epos`` input.
        h5_output_path: Destination ``.h5`` path.
        chunk_size: Number of rows per chunk (default 1<<20 ~ 1M rows).
        progress_callback: Optional ``callable(rows_done, total_rows)`` fired
            after each chunk is written.

    Returns:
        dict with ``'h5'`` (output path) and ``'rows'`` (total rows written).
    """
    epos_path = Path(epos_path).expanduser()
    h5_output_path = Path(h5_output_path).expanduser()
    h5_output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    empty_schema = None
    with leap_tools.read_epos_lazy(epos_path) as epos_table:
        total_rows = epos_table.n_rows
        with pd.HDFStore(str(h5_output_path), mode="w") as store:
            for chunk in epos_lazy_to_ccapt_chunks(epos_table, chunk_size=chunk_size):
                if len(chunk) == 0:
                    if empty_schema is None:
                        empty_schema = chunk
                    continue
                store.append("df", chunk, format="table", index=False)
                rows_written += len(chunk)
                if progress_callback is not None:
                    progress_callback(rows_written, total_rows)
            if rows_written == 0:
                # No non-empty chunk was written (empty / truncated .epos).
                # append(format='table') does NOT create the key for a 0-row
                # frame, so the file would have no 'df' dataset and the
                # documented pd.read_hdf(path, key='df') consumer would raise
                # KeyError. Persist the 0-row schema frame explicitly via put().
                if empty_schema is None:
                    empty_schema = pd.DataFrame()
                try:
                    store.put("df", empty_schema, format="table")
                except (ValueError, TypeError):
                    store.put("df", empty_schema, format="fixed")

    return {"h5": str(h5_output_path), "rows": rows_written}


def apt_to_ccapt(file_path):
    """
    Convert APT data to PyCCAPT format.

    Args:
        file_path: APT data file path.

    Returns:
        pandas.DataFrame: CCAPT data.
    """

    data = leap_tools.read_apt(file_path)
    length_data = len(data["Mass"])

    def pick_first(*keys, default=0.0, dtype=float):
        for key in keys:
            if key in data.columns:
                return data[key].to_numpy()
        return np.full(length_data, default, dtype=dtype)

    if "z" in data.columns:
        z_values = data["z"].to_numpy()
    elif "zs" in data.columns:
        z_values = -1 * data["zs"].to_numpy()
    else:
        z_values = np.zeros(length_data)

    data_dict = {
        'x (nm)': pick_first('x', 'xs'),
        'y (nm)': pick_first('y', 'ys'),
        'z (nm)': z_values,
        'mc (Da)': pick_first('Mass'),
        'high_voltage (V)': pick_first('Voltage', 'Vref'),
        'pulse_v (V)': pick_first('Vap', 'pulse'),
        'pulse_l (pJ)': pick_first('laserpower'),
        't (ns)': pick_first('Epos ToF', 'tof'),
        't_c (ns)': pick_first('tofc'),
        'x_det (cm)': pick_first('XDet_mm'),
        'y_det (cm)': pick_first('YDet_mm'),
        'delta_p': pick_first('Delta Pulse', 'pulseDelta', dtype=int),
        'multi': pick_first('Multiplicity', dtype=int),
        'start_counter': pick_first('tElapsed', dtype=int),
    }

    df = pd.DataFrame(data_dict)
    df.insert(loc=4, column='mc_uc (Da)', value=np.zeros(length_data))
    df['x_det (cm)'] = df['x_det (cm)'] / 10
    df['y_det (cm)'] = df['y_det (cm)'] / 10
    df['delta_p'] = df['delta_p'].astype(int)
    df['multi'] = df['multi'].astype(int)
    df['start_counter'] = df['start_counter'].astype(int)

    return df
