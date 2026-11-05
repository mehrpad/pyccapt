"""Opt-in SWMR event stream with a committed-row publication marker.

One writer must own a file. This reader facility does not replace the control
acquisition journal or enable itself on an instrument. Verify the deployed
filesystem's SWMR support before attaching live readers.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from pyccapt.calibration.core.exceptions import CalibrationInputError, CalibrationStateError


class LiveEventWriter:
    def __init__(self, path, columns, *, units=None, chunk_rows=65536):
        if not columns or not isinstance(chunk_rows, int) or chunk_rows <= 0:
            raise CalibrationInputError('Declare columns and a positive chunk_rows before entering SWMR')
        self.dtypes = {name: np.dtype(dtype) for name, dtype in columns.items()}
        if any(not isinstance(name, str) or not name or '/' in name or dtype.kind not in 'biufc'
               for name, dtype in self.dtypes.items()):
            raise CalibrationInputError('Live columns must have simple names and numeric dtypes')
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(destination, 'x', libver='latest')
        self._failed = False
        try:
            self.file.attrs['schema_version'] = 'pyccapt-live-1'
            self.file.attrs['completeness_scope'] = 'committed preview rows; raw acquisition journal remains authoritative'
            group = self.file.create_group('events')
            for name, dtype in self.dtypes.items():
                dataset = group.create_dataset(name, shape=(0,), maxshape=(None,), chunks=(chunk_rows,), dtype=dtype)
                if units and name in units:
                    dataset.attrs['units'] = str(units[name])
            self.file.create_dataset('committed_rows', data=np.array([0], dtype=np.uint64))
            self.file.create_dataset('writer_state', data=np.array([0], dtype=np.uint8))
            self.file['writer_state'].attrs['codes'] = '0=writing;1=closed;2=failed'
            self.file.flush()
            self.file.swmr_mode = True
        except Exception:
            self.file.close()
            raise

    def append(self, columns):
        if self._failed or not self.file.id.valid:
            raise CalibrationStateError('Live writer is failed or closed')
        if set(columns) != set(self.dtypes):
            raise CalibrationInputError('Every declared live column must be supplied together')
        arrays = {key: np.asarray(value) for key, value in columns.items()}
        if any(value.ndim != 1 or value.dtype != self.dtypes[key] for key, value in arrays.items()):
            raise CalibrationInputError('Live arrays must be one-dimensional and match the declared dtype exactly')
        sizes = {len(value) for value in arrays.values()}
        if len(sizes) != 1:
            raise CalibrationInputError('Live columns must have equal row counts')
        start = int(self.file['committed_rows'][0])
        stop = start + sizes.pop()
        try:
            for name, value in arrays.items():
                dataset = self.file['events'][name]
                dataset.resize((stop,))
                dataset[start:stop] = value
                dataset.flush()
            self.file['committed_rows'][0] = stop
            self.file['committed_rows'].flush()
            return stop
        except Exception:
            self._failed = True
            self.file['writer_state'][0] = 2
            self.file['writer_state'].flush()
            raise

    def close(self):
        if self.file.id.valid:
            self.file['writer_state'][0] = 2 if self._failed else 1
            self.file.flush()
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        self._failed |= exc_type is not None
        self.close()


def read_live_events(path, *, start=0):
    if not isinstance(start, int) or start < 0:
        raise CalibrationInputError('start must be a nonnegative row offset')
    with h5py.File(path, 'r', libver='latest', swmr=True) as handle:
        if handle.attrs.get('schema_version') != 'pyccapt-live-1':
            raise CalibrationInputError('Unsupported live stream schema')
        # Read terminal state before the publication marker. If close races a
        # read, returning "writing" is conservative; returning "closed" with
        # an older prefix would make a reader stop and silently miss final rows.
        handle['writer_state'].refresh()
        writer_state = int(handle['writer_state'][0])
        marker = handle['committed_rows']
        marker.refresh()
        stop = int(marker[0])
        if start > stop:
            raise CalibrationInputError('Reader offset exceeds the committed stream')
        arrays = {}
        for name, dataset in handle['events'].items():
            dataset.refresh()
            if len(dataset) < stop:
                raise CalibrationStateError('Committed marker exceeds available event rows')
            arrays[name] = dataset[start:stop]
        return arrays, {'committed_rows': stop, 'writer_state': writer_state}
