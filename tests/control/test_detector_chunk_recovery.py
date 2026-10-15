"""Recovery and damaged-group replay preserve waveform/metadata identity."""
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from pyccapt.control.core import chunk_store, data_integrity, detector_hdf
from pyccapt.control.core.hdf5_creator import hdf_creator


def _hsd_chunks(directory, chunk_id, marker, events=1):
    samples = np.full(1024*events, marker, dtype=np.float32)
    chunk_store.atomic_write_chunk_group(directory, stream_name='hsd_waveforms', chunk_id=chunk_id,
        arrays={f'ch{i}_{kind}': samples+i for i in range(4) for kind in ('time', 'wave')})
    chunk_store.atomic_write_chunk_group(directory, stream_name='hsd_events', chunk_id=chunk_id,
        arrays={'voltage_drs': np.full(events, marker*100., dtype=np.float64),
                'pulse_drs': np.full(events, marker*10., dtype=np.float64)})


def test_hsd_recovery_preserves_signed_samples_and_event_voltage(tmp_path):
    directory = tmp_path/'chunks'
    _hsd_chunks(directory, 1, -.25, events=2)
    output = tmp_path/'recovered.h5'
    result = data_integrity.recover_chunks(directory, output)
    assert result['recovered']
    assert result['acquisition_completeness'] == 'unverified'
    with h5py.File(output) as handle:
        np.testing.assert_array_equal(handle['hsd/ch0_wave'], np.full(2048, -.25, dtype=np.float32))
        np.testing.assert_array_equal(handle['hsd/high_voltage'], [-25., -25.])
        assert handle['hsd/ch0_wave'].dtype == np.dtype('float32')


def test_recovery_reports_unattributed_tail_without_claiming_complete_chunks_are_all_events(tmp_path):
    from pyccapt.control.apt.detector_chunks import preserve_unattributed_packet
    directory = tmp_path/'chunks'
    _hsd_chunks(directory, 1, 1.)
    evidence = preserve_unattributed_packet(tmp_path, 'hsd_shutdown', {'waveforms': np.ones((8, 1024))})
    output = tmp_path/'recovered.h5'
    result = data_integrity.recover_chunks(directory, output)
    assert result['recovered'] and result['acquisition_completeness'] == 'incomplete'
    with h5py.File(output) as handle:
        report = json.loads(handle['provenance'].attrs['chunk_integrity_json'])
        assert report['unattributed_packets'] == [evidence.name]
        assert len(handle['hsd/high_voltage']) == 1
    assert evidence.exists()


@pytest.mark.parametrize('damaged_stem', ['ch0_wave', 'voltage_drs'])
def test_hsd_corrupt_group_excludes_its_partner_without_metadata_shift(tmp_path, damaged_stem):
    directory = tmp_path/'chunks'
    for i in (1, 2, 3):
        _hsd_chunks(directory, i, float(i))
    (directory/f'{damaged_stem}_chunk_2.npy').write_bytes(b'corrupt')
    output = tmp_path/'recovered.h5'
    result = data_integrity.recover_chunks(directory, output)
    assert result['acquisition_completeness'] == 'incomplete'
    assert result['quarantined_records'] == 1
    assert result['excluded_manifest_records'] == 2
    with h5py.File(output) as handle:
        np.testing.assert_array_equal(handle['hsd/high_voltage'], [100., 300.])
        np.testing.assert_array_equal(handle['hsd/ch0_wave'][::1024], [1., 3.])
        report = json.loads(handle['provenance'].attrs['chunk_integrity_json'])
        assert report['streams']['hsd_events']['excluded_rows'] == 1
        assert report['streams']['hsd_waveforms']['excluded_rows'] == 1024
    # Unpaired valid files remain available for forensic recovery.
    partner = 'voltage_drs' if damaged_stem == 'ch0_wave' else 'ch0_wave'
    assert (directory/f'{partner}_chunk_2.npy').exists()


def test_hsd_partial_commit_and_wrong_sample_count_are_excluded(tmp_path):
    directory = tmp_path/'chunks'
    _hsd_chunks(directory, 1, 1.)
    samples = np.ones(1024, dtype=np.float32)
    chunk_store.atomic_write_chunk_group(directory, stream_name='hsd_waveforms', chunk_id=2,
        arrays={f'ch{i}_{kind}': samples for i in range(4) for kind in ('time', 'wave')})
    index = detector_hdf.align_hsd_chunks(chunk_store.validated_files_by_stem(directory))
    assert len(index['ch0_wave']) == 1
    assert index.integrity_report()['status'] == 'incomplete'
    assert detector_hdf.align_hsd_chunks(index).integrity_report() == index.integrity_report()


def test_hsd_finalization_records_pair_exclusions_before_provenance(tmp_path):
    directory = tmp_path/'temp_data'/'chunks'
    for i in (1, 2):
        _hsd_chunks(directory, i, float(i))
    (directory/'ch0_wave_chunk_1.npy').write_bytes(b'corrupt')
    variables = SimpleNamespace(path=str(tmp_path), exp_name='run', counter_source='HSD',
                                main_counter=[1], main_raw_counter=[1],
                                main_temperature=[40.], main_chamber_vacuum=[1e-8])
    hdf_creator(variables, {'tdc': 'on', 'tdc_model': 'HSD'}, [1], [0.])
    with h5py.File(tmp_path/'run.h5') as handle:
        np.testing.assert_array_equal(handle['hsd/high_voltage'], [200.])
        np.testing.assert_array_equal(handle['hsd/ch0_wave'][::1024], [2.])
        assert handle['provenance'].attrs['acquisition_completeness'] == 'incomplete'
        assert handle['provenance'].attrs['excluded_row_count'] == 1
    result = data_integrity.validate_hdf5(tmp_path/'run.h5')
    assert result['structurally_valid'] and not result['valid']


def test_roentdek_recovery_maps_timestamp_and_all_raw_channels(tmp_path):
    directory = tmp_path/'chunks'
    ids = np.array([2, 3], dtype=np.uint64)
    arrays = {name: ids.astype(float) for name in ('x', 'y', 't', 'voltage', 'voltage_pulse', 'laser_pulse')}
    arrays['time_stamp'] = ids+np.uint64(2**53)
    arrays.update({f'ch{i}': ids*10+i for i in range(8)})
    chunk_store.atomic_write_chunk_group(directory, stream_name='roentdek', chunk_id=1, arrays=arrays)
    output = tmp_path/'recovered.h5'
    result = data_integrity.recover_chunks(directory, output)
    assert result['recovered']
    with h5py.File(output) as handle:
        np.testing.assert_array_equal(handle['dld/start_counter'], arrays['time_stamp'])
        np.testing.assert_array_equal(handle['tdc/start_counter'], np.repeat(arrays['time_stamp'], 8))
        np.testing.assert_array_equal(handle['tdc/time_data'], np.column_stack([arrays[f'ch{i}'] for i in range(8)]).ravel())


def test_hsd_validator_rejects_misaligned_sample_and_event_counts(tmp_path):
    path = tmp_path/'broken.h5'
    with h5py.File(path, 'w') as handle:
        handle.create_dataset('apt/id', data=[1])
        handle.create_dataset('hsd/ch0_wave', data=np.zeros(1024))
        handle.create_dataset('hsd/ch1_wave', data=np.zeros(1023))
        handle.create_dataset('hsd/high_voltage', data=[100., 200.])
        handle.create_dataset('hsd/voltage_pulse', data=[10.])
    issues = data_integrity.validate_hdf5(path)['issues']
    assert 'hsd waveform channel lengths differ' in issues
    assert 'hsd per-waveform voltage lengths differ' in issues
    assert 'hsd waveforms must have 1024 samples per voltage event' in issues
