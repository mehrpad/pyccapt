"""Measure HDF5 layouts on the actual destination without changing defaults."""
from __future__ import annotations

from pathlib import Path
import tempfile
import time

import h5py
import numpy as np

from pyccapt.calibration.core.exceptions import CalibrationInputError


def benchmark_storage(directory, values, *, chunk_rows=(16384, 65536, 262144),
                      compressions=(None, 'lzf', 'gzip'), repeats=3):
    """Compare exact full, contiguous ROI and strided preview reads.

    Timings include filesystem cache effects and are local measurements, not
    cross-machine promises. Files are temporary; no acquisition defaults change.
    """
    values = np.asarray(values)
    if values.ndim != 1 or values.size < 2 or values.dtype.kind not in 'biufc' or repeats < 1:
        raise CalibrationInputError('Supply a numeric vector with at least two rows and positive repeats')
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = []
    selectors = {'sequential': slice(None), 'roi': slice(len(values)//3, min(len(values), len(values)//3+10000)),
                 'preview': slice(None, None, max(1, len(values)//1000))}
    with tempfile.TemporaryDirectory(prefix='pyccapt-storage-benchmark-', dir=root) as temporary:
        for chunk in chunk_rows:
            for compression in compressions:
                if int(chunk) <= 0 or compression not in (None, 'lzf', 'gzip'):
                    raise CalibrationInputError('Invalid chunk/compression benchmark configuration')
                path = Path(temporary) / 'candidate.h5'
                options = {'chunks': (min(int(chunk), len(values)),), 'compression': compression,
                           'shuffle': compression is not None}
                if compression == 'gzip':
                    options['compression_opts'] = 4
                started = time.perf_counter()
                with h5py.File(path, 'w') as handle:
                    handle.create_dataset('value', data=values, **options)
                write_s = time.perf_counter()-started
                timings = {}
                for name, selection in selectors.items():
                    durations = []
                    for _ in range(repeats):
                        started = time.perf_counter()
                        with h5py.File(path, 'r') as handle:
                            actual = handle['value'][selection]
                        durations.append(time.perf_counter()-started)
                        np.testing.assert_array_equal(actual, values[selection])
                    timings[name+'_median_s'] = float(np.median(durations))
                records.append({'chunk_rows': min(int(chunk), len(values)), 'compression': compression,
                                'file_bytes': path.stat().st_size, 'write_s': write_s,
                                'exact_values': True, **timings})
    return {'schema': 'pyccapt-storage-benchmark-1', 'rows': len(values), 'dtype': values.dtype.str,
            'repeats': repeats, 'destination': str(root), 'cache_policy': 'OS cache not evicted', 'results': records}
