"""Bounded-memory finalization for typed RoentDek and HSD acquisition chunks."""
from pathlib import Path

import numpy as np

from pyccapt.control.core.chunk_store import ValidatedChunkIndex


def _declared(index, stem, files):
    return bool(files) or isinstance(index, ValidatedChunkIndex) and index.declares(stem)


def align_hsd_chunks(index):
    """Exclude unmatched HSD groups without shifting sample/voltage identity.

    The producer commits 64 waveforms and their 64 voltage records under the
    same chunk ID (with matching residual groups). Keep valid-but-unpaired
    files on disk for forensic recovery, and include their exclusion in the
    same integrity report as failed checksum groups.
    """
    if not isinstance(index, ValidatedChunkIndex):
        return index
    required = {
        'hsd_waveforms': {f'ch{i}_{kind}' for i in range(4) for kind in ('time', 'wave')},
        'hsd_events': {'voltage_drs', 'pulse_drs'},
    }
    records = {name: {} for name in required}
    for record in index.valid_records:
        if record.get('stream') in records:
            records[record['stream']][int(record['chunk_id'])] = record
    if not any(records.values()):
        return index

    def rows(record, fields):
        metadata = record.get('fields', {})
        if not fields.issubset(metadata):
            return None
        shapes = [metadata[name].get('shape', []) for name in fields]
        if any(len(shape) != 1 for shape in shapes) or len({shape[0] for shape in shapes}) != 1:
            return None
        return shapes[0][0]

    paired = set()
    for chunk_id in records['hsd_waveforms'].keys() & records['hsd_events'].keys():
        samples = rows(records['hsd_waveforms'][chunk_id], required['hsd_waveforms'])
        events = rows(records['hsd_events'][chunk_id], required['hsd_events'])
        if samples is not None and events is not None and samples == 1024 * events:
            paired.add(chunk_id)
    valid, excluded = [], []
    for record in index.valid_records:
        if record.get('stream') in records and int(record['chunk_id']) not in paired:
            excluded.append({**record, 'error': 'HSD waveform/voltage group missing or not aligned at 1024 samples per event'})
        else:
            valid.append(record)
    if not excluded:
        return index
    excluded_files = {metadata['file'] for record in excluded for metadata in record.get('fields', {}).values()}
    files = {stem: [path for path in paths if path.name not in excluded_files] for stem, paths in index.items()}
    return ValidatedChunkIndex(files, valid, index.invalid_records + excluded)


def write_hsd_chunks(handle, directory, index):
    from pyccapt.control.core import hdf5_creator as hdf
    directory = Path(directory)
    index = align_hsd_chunks(index)
    if not _declared(index, 'ch0_time', hdf._sorted_chunk_files(directory, 'ch0_time', index)):
        return False
    mapping = [(f'ch{i}_{kind}', f'hsd/ch{i}_{kind}', np.float32)
               for i in range(4) for kind in ('time', 'wave')]
    mapping += [('voltage_drs', 'hsd/high_voltage', np.float64),
                ('pulse_drs', 'hsd/voltage_pulse', np.float64)]
    for stem, name, dtype in mapping:
        files = hdf._sorted_chunk_files(directory, stem, index)
        hdf._write_chunked_dataset(handle, name, files, dtype)
    return True


def write_roentdek_chunks(handle, directory, index):
    from pyccapt.control.core import hdf5_creator as hdf
    directory = Path(directory)
    channel_files = hdf._sorted_chunk_files(directory, 'ch0', index)
    if not _declared(index, 'ch0', channel_files):
        return False
    mapping = [('x', 'dld/x', np.float64), ('y', 'dld/y', np.float64),
               ('t', 'dld/t', np.float64), ('voltage', 'dld/high_voltage', np.float64),
               ('voltage_pulse', 'dld/voltage_pulse', np.float64),
               ('laser_pulse', 'dld/laser_pulse', np.float64),
               ('time_stamp', 'dld/start_counter', np.uint64)]
    for stem, name, dtype in mapping:
        hdf._write_chunked_dataset(handle, name, hdf._sorted_chunk_files(directory, stem, index), dtype)
    stems = {f'ch{i}': np.uint64 for i in range(8)}
    stems.update(time_stamp=np.uint64, voltage=np.float64, voltage_pulse=np.float64, laser_pulse=np.float64)
    by_id = {}
    for stem in stems:
        by_id[stem] = {int(path.stem.rsplit('_', 1)[1]): path
                       for path in hdf._sorted_chunk_files(directory, stem, index)}
    ids = sorted(by_id['ch0'])
    if any(set(files) != set(ids) for files in by_id.values()):
        raise ValueError('RoentDek chunk fields do not share the same event groups')
    sizes = []
    for chunk_id in ids:
        values = np.load(by_id['ch0'][chunk_id], mmap_mode='r')
        sizes.append(len(values))
        del values
    total = sum(sizes) * 8
    outputs = {'channel': np.uint32, 'time_data': np.uint64, 'start_counter': np.uint64,
               'high_voltage': np.float64, 'voltage_pulse': np.float64, 'laser_pulse': np.float64}
    datasets = {}
    for field, dtype in outputs.items():
        name = 'tdc/' + field
        datasets[field] = handle.create_dataset(name, (total,), dtype=dtype, **hdf._compression_opts(total))
        hdf._annotate_dataset(datasets[field], name)
    offset = 0
    for chunk_id, size in zip(ids, sizes):
        arrays = {}
        for stem, dtype in stems.items():
            path = by_id[stem][chunk_id]
            values = np.load(path, mmap_mode='r')
            if values.ndim != 1 or len(values) != size:
                raise ValueError(f'RoentDek event alignment mismatch: {path.name}')
            arrays[stem] = hdf._coerce_chunk_to_target(values, np.dtype(dtype), path, stem)
        stop = offset + size * 8
        datasets['time_data'][offset:stop] = np.column_stack([arrays[f'ch{i}'] for i in range(8)]).reshape(-1)
        datasets['channel'][offset:stop] = np.tile(np.arange(8, dtype=np.uint32), size)
        for output, stem in [('start_counter', 'time_stamp'), ('high_voltage', 'voltage'),
                             ('voltage_pulse', 'voltage_pulse'), ('laser_pulse', 'laser_pulse')]:
            datasets[output][offset:stop] = np.repeat(arrays[stem], 8)
        offset = stop
        del arrays, values
    return True
