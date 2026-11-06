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


ROENTDEK_RAW_TIME_OFFSET = 1 << 23   # must match tdc_roentdek.RAW_TIME_OFFSET
ROENTDEK_RECONSTRUCTION_FILE = 'roentdek_reconstruction.txt'   # must match tdc_roentdek.RECONSTRUCTION_FILE


def _write_roentdek_ion_and_hit_chunks(handle, directory, index):
    """Layout of the RoentDek reader since 2026-11 (wrapper DLL v3 + Resort64c).

    ``roentdek`` stream: one row per reconstructed ion -> dld/*
    ``roentdek_raw`` stream: one row per TDC hit of every group -> tdc/*
    (tdc/time_data = signed TDC bins + ROENTDEK_RAW_TIME_OFFSET)
    """
    from pyccapt.control.core import hdf5_creator as hdf
    ion_map = [('x', 'dld/x', np.float64), ('y', 'dld/y', np.float64), ('t', 'dld/t', np.float64),
               ('voltage', 'dld/high_voltage', np.float64), ('voltage_pulse', 'dld/voltage_pulse', np.float64),
               ('laser_pulse', 'dld/laser_pulse', np.float64), ('time_stamp', 'dld/start_counter', np.uint64),
               ('ro_reco_method', 'dld/reco_method', np.int16)]
    raw_map = [('ro_raw_channel', 'tdc/channel', np.uint32), ('ro_raw_time', 'tdc/time_data', np.uint64),
               ('ro_raw_time_stamp', 'tdc/start_counter', np.uint64),
               ('ro_raw_voltage', 'tdc/high_voltage', np.float64),
               ('ro_raw_voltage_pulse', 'tdc/voltage_pulse', np.float64),
               ('ro_raw_laser_pulse', 'tdc/laser_pulse', np.float64)]
    for mapping, label in ((ion_map, 'ion'), (raw_map, 'raw hit')):
        files = {stem: hdf._sorted_chunk_files(directory, stem, index) for stem, _, _ in mapping}
        ids = {stem: [int(path.stem.rsplit('_', 1)[1]) for path in paths] for stem, paths in files.items()}
        first = next(iter(ids.values()))
        if any(chunk_ids != first for chunk_ids in ids.values()):
            raise ValueError(f'RoentDek {label} chunk fields do not share the same chunk groups')
        for stem, name, dtype in mapping:
            hdf._write_chunked_dataset(handle, name, files[stem], dtype)
    dld_lengths = {len(handle[name]) for _, name, _ in ion_map}
    tdc_lengths = {len(handle[name]) for _, name, _ in raw_map}
    if len(dld_lengths) != 1 or len(tdc_lengths) != 1:
        raise ValueError('RoentDek dld/ or tdc/ datasets have different lengths')
    time_data = handle['tdc/time_data']
    time_data.attrs['roentdek_time_offset'] = ROENTDEK_RAW_TIME_OFFSET
    time_data.attrs['encoding'] = ('RoentDek TDC8HP: signed TDC bins relative to the group trigger + '
                                   'roentdek_time_offset (2**23); every row is one real hit')
    handle['dld/reco_method'].attrs['description'] = ('RoentDek Resort64c/CoboldPC reconstruction method '
                                                     'number of each ion (PyCCAPT codes 0-3 if the Resort64c '
                                                     'DLL was not available)')
    handle['dld/t'].attrs['missing_start'] = 'NaN: the group had no laser start signal (TOF unknown, x/y valid)'
    # which reconstruction produced dld/x, y, t (written by tdc_roentdek.record_reconstruction)
    reconstruction = ''
    try:
        info_path = Path(directory).parent / ROENTDEK_RECONSTRUCTION_FILE
        if info_path.is_file():
            reconstruction = info_path.read_text().strip()
            handle['dld'].attrs['roentdek_reconstruction'] = reconstruction
    except Exception:
        reconstruction = ''
    if not reconstruction or reconstruction.startswith('Resort64c'):
        try:
            from pyccapt.control.tdc_roentdek.roentdek_resort import DEFAULT_CFG
            handle['dld'].attrs['roentdek_resort_cfg'] = Path(DEFAULT_CFG).read_text()
        except Exception:
            pass
    return True


def write_roentdek_chunks(handle, directory, index):
    from pyccapt.control.core import hdf5_creator as hdf
    directory = Path(directory)
    if _declared(index, 'ro_raw_channel', hdf._sorted_chunk_files(directory, 'ro_raw_channel', index)):
        return _write_roentdek_ion_and_hit_chunks(handle, directory, index)
    # previous layout: one row per TDC group with the first hit of each channel (ch0..ch7)
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
