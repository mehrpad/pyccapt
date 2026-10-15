"""Lightweight format hints and metadata-only HDF5 layout detection."""
from __future__ import annotations

from pathlib import Path

from pyccapt.calibration.core.exceptions import CalibrationInputError


FORMAT_LABELS = {'pyccapt': 'PyCCAPT HDF5', 'nexus': 'PyCCAPT NeXus mapping',
                 'leap_epos': 'LEAP epos', 'leap_pos': 'LEAP pos',
                 'leap_apt': 'LEAP apt', 'ato_v6': 'ATO v6'}
EXTENSION_FORMATS = {'.h5': 'pyccapt', '.hdf5': 'pyccapt', '.hdf': 'pyccapt',
                     '.nxs': 'nexus', '.nx5': 'nexus', '.epos': 'leap_epos',
                     '.pos': 'leap_pos', '.apt': 'leap_apt', '.ato': 'ato_v6'}
HDF_FORMATS = {'pyccapt', 'nexus'}


def format_from_path(path):
    """Return a filename hint without reading the file or changing shared state."""
    return EXTENSION_FORMATS.get(Path(str(path).strip()).suffix.lower())


def resolve_dataset_format(path, *, data_type='auto', mode='auto'):
    """Resolve supported readers; inspect container keys without reading event arrays.

    Filename extensions choose an exchange reader. HDF5 layout takes precedence
    over its suffix in automatic mode. Explicit format/mode choices remain usable.
    No raw detector-channel stream is silently interpreted as a hit table.
    """
    if mode not in {'auto', 'processed', 'raw'}:
        raise CalibrationInputError(f'Unsupported HDF5 mode: {mode!r}')
    automatic = data_type == 'auto'
    if automatic:
        data_type = format_from_path(path)
        if data_type is None:
            raise CalibrationInputError('Cannot detect this dataset format from its extension. '
                'Choose Format manually for a renamed supported file; raw channel streams need the raw-data workflow.')
    if data_type not in FORMAT_LABELS:
        raise CalibrationInputError(f'Unsupported dataset format: {data_type!r}')
    if data_type not in HDF_FORMATS:
        return data_type, 'processed'  # HDF5 mode has no meaning for exchange readers.
    if data_type in HDF_FORMATS and mode == 'auto':
        import h5py
        with h5py.File(Path(path).expanduser(), 'r') as handle:
            if 'df' in handle:
                return 'pyccapt' if automatic else data_type, 'processed'
            if 'dld' in handle and (automatic or data_type == 'pyccapt'):
                return 'pyccapt', 'raw'
            if 'entry' in handle:
                return 'nexus', 'processed'
        raise CalibrationInputError('Expected processed /df, raw /dld hits or a PyCCAPT NeXus mapping; '
                                    'convert raw detector channels into hits first')
    return data_type, 'processed' if mode == 'auto' else mode
