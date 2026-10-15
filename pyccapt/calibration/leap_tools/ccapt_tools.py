from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

# Local module and scripts
from pyccapt.calibration.leap_tools import leap_tools


_EPOS_SOURCE_COLUMNS = {
    'x (nm)': 'x (nm)', 'y (nm)': 'y (nm)', 'z (nm)': 'z (nm)',
    'mc (Da)': 'm/n (Da)', 'high_voltage (V)': 'HV_DC (V)',
    'pulse_v (V)': 'pulse (V)', 't (ns)': 'TOF (ns)',
    'x_det (cm)': 'det_x (mm)', 'y_det (cm)': 'det_y (mm)',
    'delta_p': 'pslep', 'multi': 'ipp',
}


def _with_import_availability(frame, source, source_columns):
    """Record absent source fields without changing legacy numerical placeholders.

    ``source_columns`` maps output fields to the actual source fields used. A
    present, finite zero remains available; no availability is inferred from
    its numerical value. Carry existing source metadata through renaming.
    """
    frame.attrs = deepcopy(getattr(source, 'attrs', {}))
    inherited = set(frame.attrs.get('pyccapt_unavailable_columns', []))
    unavailable = set(frame.columns).difference(source_columns)
    unavailable.update(name for name in inherited if name in frame.columns)
    unavailable.update(target for target, origin in source_columns.items() if origin in inherited)
    frame.attrs['pyccapt_unavailable_columns'] = sorted(unavailable)
    return frame


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
    # POS is four interleaved big-endian float32 values per ion.  Converting
    # the matrix once avoids materialising millions of Python tuples and
    # passing each scalar through ``struct.pack``.
    pos = np.ascontiguousarray(
        data[['x (nm)', 'y (nm)', 'z (nm)', 'mc (Da)']].to_numpy(dtype='>f4', copy=True)
    ).tobytes()
    if name is not None:
        with open(path + name, 'w+b') as f:
            f.write(pos)
    return pos


_EPOS_DTYPE = np.dtype(
    [
        ('x', '>f4'), ('y', '>f4'), ('z', '>f4'), ('mc', '>f4'),
        ('tof', '>f4'), ('high_voltage', '>f4'), ('pulse_voltage', '>f4'),
        ('detector_x', '>f4'), ('detector_y', '>f4'),
        ('delta_p', '>u4'), ('multi', '>u4'),
    ]
)


def _epos_chunk_bytes(chunk):
    """Encode one DataFrame slice as interleaved big-endian EPOS records."""
    records = np.empty(len(chunk), dtype=_EPOS_DTYPE)
    float_fields = (
        ('x', 'x (nm)', 1.0), ('y', 'y (nm)', 1.0), ('z', 'z (nm)', 1.0),
        ('mc', 'mc (Da)', 1.0), ('tof', 't (ns)', 1.0),
        ('high_voltage', 'high_voltage (V)', 1.0),
        ('pulse_voltage', 'pulse_v (V)', 1.0),
        ('detector_x', 'x_det (cm)', 10.0),
        ('detector_y', 'y_det (cm)', 10.0),
    )
    for field, column, scale in float_fields:
        records[field] = chunk[column].to_numpy(dtype=np.float32, copy=False) * scale
    records['delta_p'] = chunk['delta_p'].to_numpy(dtype=np.uint32, copy=False)
    records['multi'] = chunk['multi'].to_numpy(dtype=np.uint32, copy=False)
    return records.tobytes()


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

    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError('chunk_size must be a positive integer')

    if name is not None:
        with open(path + name, 'w+b') as f:
            for i in range(0, len(data), chunk_size):
                f.write(_epos_chunk_bytes(data.iloc[i : i + chunk_size]))
    else:
        chunks = (
            _epos_chunk_bytes(data.iloc[i : i + chunk_size])
            for i in range(0, len(data), chunk_size)
        )
        return b''.join(chunks)


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
    return _with_import_availability(ccapt, pos, {
        'x (nm)': 'x (nm)', 'y (nm)': 'y (nm)', 'z (nm)': 'z (nm)', 'mc (Da)': 'm/n (Da)',
    })


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
    return _with_import_availability(ccapt, epos, _EPOS_SOURCE_COLUMNS)


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
        frame = pd.DataFrame(
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
        yield _with_import_availability(frame, epos_table, _EPOS_SOURCE_COLUMNS)
        return
    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        length = stop - start
        frame = pd.DataFrame(
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
        yield _with_import_availability(frame, epos_table, _EPOS_SOURCE_COLUMNS)


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
    source_attrs = {}
    with leap_tools.read_epos_lazy(epos_path) as epos_table:
        total_rows = epos_table.n_rows
        with pd.HDFStore(str(h5_output_path), mode="w") as store:
            for chunk in epos_lazy_to_ccapt_chunks(epos_table, chunk_size=chunk_size):
                source_attrs = {key: chunk.attrs[key] for key in
                                ('pyccapt_source_id', 'pyccapt_unavailable_columns') if key in chunk.attrs}
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
                if 'df' not in store:
                    store.put('df', empty_schema, format='fixed')
            # Pandas does not persist DataFrame.attrs in its HDF representation.
            # Use the same explicit metadata contract as the processed loader.
            from pyccapt.calibration.data_tools.analysis_session import canonical_json
            store.get_storer('df').attrs.pyccapt_source_json = canonical_json(source_attrs)

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
    source_columns = {}

    def pick_first(target, *keys, default=0.0, dtype=float):
        for key in keys:
            if key in data.columns:
                source_columns[target] = key
                return data[key].to_numpy()
        return np.full(length_data, default, dtype=dtype)

    if "z" in data.columns:
        z_values = data["z"].to_numpy()
        source_columns['z (nm)'] = 'z'
    elif "zs" in data.columns:
        z_values = -1 * data["zs"].to_numpy()
        source_columns['z (nm)'] = 'zs'
    else:
        z_values = np.zeros(length_data)

    data_dict = {
        'x (nm)': pick_first('x (nm)', 'x', 'xs'),
        'y (nm)': pick_first('y (nm)', 'y', 'ys'),
        'z (nm)': z_values,
        'mc (Da)': pick_first('mc (Da)', 'Mass'),
        'high_voltage (V)': pick_first('high_voltage (V)', 'Voltage', 'Vref'),
        'pulse_v (V)': pick_first('pulse_v (V)', 'Vap', 'pulse'),
        'pulse_l (pJ)': pick_first('pulse_l (pJ)', 'laserpower'),
        't (ns)': pick_first('t (ns)', 'Epos ToF', 'tof'),
        't_c (ns)': pick_first('t_c (ns)', 'tofc'),
        'x_det (cm)': pick_first('x_det (cm)', 'XDet_mm'),
        'y_det (cm)': pick_first('y_det (cm)', 'YDet_mm'),
        'delta_p': pick_first('delta_p', 'Delta Pulse', 'pulseDelta', dtype=int),
        'multi': pick_first('multi', 'Multiplicity', dtype=int),
        'start_counter': pick_first('start_counter', 'tElapsed', dtype=int),
    }

    df = pd.DataFrame(data_dict)
    df.insert(loc=4, column='mc_uc (Da)', value=np.zeros(length_data))
    df['x_det (cm)'] = df['x_det (cm)'] / 10
    df['y_det (cm)'] = df['y_det (cm)'] / 10
    df['delta_p'] = df['delta_p'].astype(int)
    df['multi'] = df['multi'].astype(int)
    df['start_counter'] = df['start_counter'].astype(int)

    return _with_import_availability(df, data, source_columns)
