"""Files must expose loss and retain detector layout/dtype under streaming."""
import json
from types import SimpleNamespace

import h5py
import numpy as np

from pyccapt.control.core import chunk_store
from pyccapt.control.core.hdf5_creator import hdf_creator
from pyccapt.control.core.data_integrity import validate_hdf5


def _variables(path, **kwargs):
    defaults = dict(path=str(path), exp_name='run', counter_source='TDC', main_counter=[1],
                    main_raw_counter=[1], main_temperature=[40.], main_chamber_vacuum=[1e-8])
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _dld(i):
    return dict(x=np.array([float(i)]), y=np.array([-float(i)]), t=np.array([100. + i]),
                voltage=np.array([500.]), voltage_pulse=np.array([50.]), laser_pulse=np.array([0.]),
                start_counter=np.array([i], dtype=np.uint64))


def test_quarantined_chunk_loss_survives_finalization_and_validation(tmp_path):
    directory = tmp_path/'temp_data'/'chunks'
    for i in (1, 2, 3):
        chunk_store.atomic_write_chunk_group(directory, stream_name='dld', chunk_id=i, arrays=_dld(i))
    np.save(directory/'x_chunk_2.npy', [99.])
    variables = _variables(tmp_path, channel=[], time_data=[], tdc_start_counter=[],
                           main_v_dc_tdc=[], main_v_p_tdc=[], main_l_p_tdc=[])
    hdf_creator(variables, {'tdc': 'on', 'tdc_model': 'Surface_Concept'}, [1], [0.])
    output = tmp_path/'run.h5'
    with h5py.File(output) as file:
        np.testing.assert_array_equal(file['dld/x'], [1., 3.])
        attrs = file['provenance'].attrs
        assert attrs['excluded_row_count'] == 1
        assert attrs['acquisition_completeness'] == 'incomplete'
        report = json.loads(attrs['chunk_integrity_json'])
        assert report['streams']['dld']['expected_manifest_rows'] == 3
        assert report['streams']['dld']['written_rows'] == 2
        assert report['rejected_records'][0]['chunk_id'] == 2
    result = validate_hdf5(output)
    assert result['structurally_valid']
    assert not result['valid']
    assert result['acquisition_completeness'] == 'incomplete'


def test_all_corrupt_chunks_do_not_resurrect_memory_fallback(tmp_path):
    directory = tmp_path/'temp_data'/'chunks'
    chunk_store.atomic_write_chunk_group(directory, stream_name='dld', chunk_id=1, arrays=_dld(1))
    (directory/'x_chunk_1.npy').write_bytes(b'corrupt')
    variables = _variables(tmp_path, x=[999.], y=[999.], t=[999.], main_v_dc_dld=[999.],
                           main_v_p_dld=[999.], main_l_p_dld=[999.], dld_start_counter=[999],
                           channel=[], time_data=[], tdc_start_counter=[], main_v_dc_tdc=[],
                           main_v_p_tdc=[], main_l_p_tdc=[])
    hdf_creator(variables, {'tdc': 'on', 'tdc_model': 'Surface_Concept'}, [1], [0.])
    with h5py.File(tmp_path/'run.h5') as file:
        assert len(file['dld/x']) == 0
        assert file['provenance'].attrs['excluded_row_count'] == 1


def test_empty_apt_group_is_not_valid_acquisition(tmp_path):
    path = tmp_path/'empty.h5'
    with h5py.File(path, 'w') as file:
        file.create_group('apt')
    assert not validate_hdf5(path)['valid']


def test_changed_configuration_hash_and_wrong_units_are_detected(tmp_path):
    hdf_creator(_variables(tmp_path), {'tdc': 'off'}, [1], [0.])
    path = tmp_path/'run.h5'
    assert validate_hdf5(path)['valid']
    with h5py.File(path, 'r+') as file:
        file['provenance'].attrs['control_config_json'] = '{"tdc":"on"}'
        file['apt/temperature'].attrs['units'] = 'C'
    issues = validate_hdf5(path)['issues']
    assert any('does not match' in issue for issue in issues)
    assert any('incorrect units' in issue for issue in issues)


def test_manifest_gap_and_detector_failure_mark_incomplete(tmp_path):
    directory = tmp_path/'temp_data'/'chunks'
    for i in (1, 3):
        chunk_store.atomic_write_chunk_group(directory, stream_name='dld', chunk_id=i, arrays=_dld(i))
    report = chunk_store.validated_files_by_stem(directory).integrity_report()
    assert report['status'] == 'incomplete'
    assert report['streams']['dld']['missing_chunk_id_ranges'] == [[2, 2]]
    hdf_creator(_variables(tmp_path, detector_error='writer failed'), {'tdc': 'off'}, [1], [0.])
    assert validate_hdf5(tmp_path/'run.h5')['acquisition_completeness'] == 'incomplete'


def test_roentdek_chunk_finalization_matches_existing_flat_layout(tmp_path):
    directory = tmp_path/'temp_data'/'chunks'
    expected = []
    for chunk_id, values in enumerate(([2., 3.], [5.]), 1):
        ids = np.asarray(values)
        arrays = {name: ids + offset for offset, name in enumerate(('x', 'y', 't', 'voltage', 'voltage_pulse', 'laser_pulse'))}
        arrays['time_stamp'] = ids.astype(np.uint64) + np.uint64(2**53)
        arrays.update({f'ch{i}': ids.astype(np.uint64) * 10 + i for i in range(8)})
        expected.append(arrays)
        chunk_store.atomic_write_chunk_group(directory, stream_name='roentdek', chunk_id=chunk_id, arrays=arrays)
    hdf_creator(_variables(tmp_path), {'tdc': 'on', 'tdc_model': 'RoentDek'}, [1], [0.])
    with h5py.File(tmp_path/'run.h5') as file:
        np.testing.assert_array_equal(file['dld/x'], [2., 3., 5.])
        rows = np.concatenate([np.column_stack([a[f'ch{i}'] for i in range(8)]) for a in expected])
        np.testing.assert_array_equal(file['tdc/time_data'], rows.ravel())
        np.testing.assert_array_equal(file['tdc/channel'], np.tile(np.arange(8), 3))
        for output, stem in [('start_counter', 'time_stamp'), ('high_voltage', 'voltage'),
                             ('voltage_pulse', 'voltage_pulse'), ('laser_pulse', 'laser_pulse')]:
            np.testing.assert_array_equal(file['tdc/'+output], np.repeat(np.concatenate([a[stem] for a in expected]), 8))
        assert file['tdc/start_counter'].dtype == np.dtype('uint64')
    assert validate_hdf5(tmp_path/'run.h5')['valid']


def test_hsd_chunk_finalization_preserves_fractional_signed_samples(tmp_path):
    directory = tmp_path/'temp_data'/'chunks'
    samples = np.resize(np.array([-.25, .125, -1.75], dtype=np.float32), 1024)
    arrays = {f'ch{i}_{kind}': samples + i for i in range(4) for kind in ('time', 'wave')}
    chunk_store.atomic_write_chunk_group(directory, stream_name='hsd_waveforms', chunk_id=1, arrays=arrays)
    chunk_store.atomic_write_chunk_group(directory, stream_name='hsd_events', chunk_id=1,
                                         arrays={'voltage_drs': np.array([500.]), 'pulse_drs': np.array([50.])})
    hdf_creator(_variables(tmp_path, counter_source='HSD'),
                {'tdc': 'on', 'tdc_model': 'HSD', 'drs_acquisition_mode': 'internal_calibration'}, [1], [0.])
    with h5py.File(tmp_path/'run.h5') as file:
        for stem, values in arrays.items():
            np.testing.assert_array_equal(file['hsd/'+stem], values)
            assert file['hsd/'+stem].dtype == np.dtype('float32')
        np.testing.assert_array_equal(file['hsd/high_voltage'], [500.])
        assert file['provenance'].attrs['drs_acquisition_mode'] == 'internal_calibration'
