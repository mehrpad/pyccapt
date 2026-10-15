from __future__ import annotations

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from pyccapt.control.core.data_integrity import validate_hdf5
from pyccapt.control.core.hdf5_creator import hdf_creator
from pyccapt.control.core import chunk_store
from pyccapt.instrument_metadata import acquisition_metadata, record_mode_observation


@pytest.mark.parametrize('modes, selected, offset', [
    (('Voltage',), 'voltage', 31.),
    (('Laser',), 'laser', 725.),
    (('VoltageLaser',), 'voltage_laser', None),
    (('Voltage', 'Laser', 'Voltage'), 'mixed', None),
    ((), '', None),
])
def test_instrument_metadata_is_additive_to_full_control_file(tmp_path, modes, selected, offset):
    """Exercise the real HDF writer without detector SDKs or hardware access."""
    conf = {'tdc': 'on', 'tdc_model': 'Surface_Concept', 'flight_path_length': 132.5,
            't_0_voltage': 31., 't_0_laser': 725., 'detector_diameter': 80.,
            'device_name': 'synthetic instrument', 'existing_extra_setting': 'retained'}
    metadata = acquisition_metadata(conf, pulse_mode=modes[0] if modes else None,
                                    observed_at='2026-10-16T12:00:00+00:00')
    for index, mode in enumerate(modes[1:], start=1):
        metadata = record_mode_observation(metadata, mode,
            observed_at=f'2026-10-16T12:00:0{index}+00:00', control_iteration=index)
    arrays = {
        'apt/num_events': ('main_counter', [1, 1]),
        'apt/num_raw_signals': ('main_raw_counter', [2, 2]),
        'apt/temperature': ('main_temperature', [40., 41.]),
        'apt/experiment_chamber_vacuum': ('main_chamber_vacuum', [1e-9, 2e-9]),
        'dld/x': ('x', [.1, -.2]), 'dld/y': ('y', [-.3, .4]),
        'dld/t': ('t', [100., 200.]),
        'dld/high_voltage': ('main_v_dc_dld', [5000., 5100.]),
        'dld/voltage_pulse': ('main_v_p_dld', [700., 710.]),
        'dld/laser_pulse': ('main_l_p_dld', [0., 0.]),
        'dld/start_counter': ('dld_start_counter', [2**60+1, 2**60+2]),
        'tdc/channel': ('channel', [0, 1, 0, 1]),
        'tdc/time_data': ('time_data', [101, 102, 201, 202]),
        'tdc/start_counter': ('tdc_start_counter', [2**60+1]*2+[2**60+2]*2),
        'tdc/high_voltage': ('main_v_dc_tdc', [5000., 5000., 5100., 5100.]),
        'tdc/voltage_pulse': ('main_v_p_tdc', [700., 700., 710., 710.]),
        'tdc/laser_pulse': ('main_l_p_tdc', [0., 0., 0., 0.]),
    }
    variables = SimpleNamespace(path=str(tmp_path), exp_name='additive', counter_source='TDC',
        acquisition_instrument_metadata=metadata, electrode='specimen label',
        calibration_model_provenance={'existing_model': 'retained'},
        **{attribute: values for attribute, values in arrays.values()})
    final_conf = {**conf, 'flight_path_length': 999., 't_0_voltage': 999., 't_0_laser': 999.}
    hdf_creator(variables, final_conf, [0, 1], [1700000000., 1700000001.])
    with h5py.File(tmp_path/'additive.h5', 'r') as handle:
        assert {'apt', 'dld', 'tdc', 'provenance', 'instrument_metadata'} == set(handle)
        for path, (_, expected) in arrays.items():
            np.testing.assert_array_equal(handle[path][:], expected)
            assert 'units' in handle[path].attrs
        np.testing.assert_array_equal(handle['apt/id'][:], [0, 1])
        np.testing.assert_array_equal(handle['apt/timestamps'][:], [1700000000., 1700000001.])
        assert handle['dld/start_counter'].dtype == np.uint64
        provenance = handle['provenance'].attrs
        assert json.loads(provenance['control_config_json']) == final_conf
        assert json.loads(provenance['model_provenance_json']) == {'existing_model': 'retained'}
        assert provenance['electrode_name'] == 'specimen label'
        assert provenance['acquisition_completeness'] == 'unverified'  # no manifests in this legacy fixture
        group = handle['instrument_metadata']
        assert group.attrs['NX_class'] == 'NXcollection'
        assert 'not full NXapm' in group.attrs['conformance']
        assert 'definition' not in handle and 'entry' not in handle
        assert json.loads(group['json'].asstr()[()]) == metadata
        acquisition = group['acquisition']
        for field, expected, unit, key in [
            ('flight_path_length_mm', 132.5, 'mm', 'flight_path_length'),
            ('t0_voltage_ns', 31., 'ns', 't_0_voltage'),
            ('t0_laser_ns', 725., 'ns', 't_0_laser'),
            ('detector_diameter_mm', 80., 'mm', 'detector_diameter'),
        ]:
            assert acquisition[field][()] == expected
            assert dict(acquisition[field].attrs) == {'units': unit, 'status': 'configured', 'source': 'control.config.'+key}
        assert acquisition['pulse_mode'].asstr()[()] == selected
        if offset is None:
            assert np.isnan(acquisition['t0_ns'][()])
            assert acquisition['t0_ns'].attrs['status'] == 'unknown'
        else:
            assert acquisition['t0_ns'][()] == offset
        assert len(metadata['acquisition']['pulse_mode_history']) == len(modes)


def test_instrument_configuration_exposes_physical_inputs_without_defaults(tmp_path):
    from pyccapt.control.core.read_files import load_config_file
    path = tmp_path/'instrument.toml'
    path.write_text('device_name="user instrument"\nflight_path_length=144.25\n'
                    't_0_voltage=15.5\nt_0_laser=821.75\ndetector_diameter=90.0\n')
    configured = acquisition_metadata(load_config_file(path), pulse_mode='Voltage')['acquisition']
    assert configured['flight_path_length_mm']['value'] == 144.25
    assert configured['t0_voltage_ns']['value'] == 15.5
    assert configured['t0_laser_ns']['value'] == 821.75
    assert configured['instrument_name']['value'] == 'user instrument'
    for key in ('flight_path_length', 't_0_voltage', 't_0_laser', 'detector_diameter'):
        invalid = acquisition_metadata({key: True}, pulse_mode='Voltage')['acquisition']
        for field in ('flight_path_length_mm', 't0_voltage_ns', 't0_laser_ns', 'detector_diameter_mm', 't0_ns'):
            assert invalid[field]['value'] is None
            assert invalid[field]['status'] == 'unknown'


def test_hdf5_keeps_run_snapshot_and_mode_history_not_final_defaults(tmp_path):
    initial = {'tdc': 'off', 'flight_path_length': 110., 't_0_voltage': 38., 't_0_laser': 800.}
    metadata = acquisition_metadata(initial, pulse_mode='Voltage', observed_at='2026-10-15T12:00:00+00:00')
    metadata = record_mode_observation(metadata, 'Laser', control_iteration=5)
    variables = SimpleNamespace(path=str(tmp_path), exp_name='instrument', main_counter=[1],
        main_raw_counter=[1], main_temperature=[45.], main_chamber_vacuum=[1e-8],
        acquisition_instrument_metadata=metadata, pulse_mode='Laser')
    hdf_creator(variables, {**initial, 'flight_path_length': 999.}, [0], [0.])
    with h5py.File(tmp_path/'instrument.h5', 'r') as handle:
        group = handle['instrument_metadata/acquisition']
        assert group['flight_path_length_mm'][()] == 110.
        assert group['flight_path_length_mm'].attrs['units'] == 'mm'
        assert group['pulse_mode'].asstr()[()] == 'mixed'
        assert np.isnan(group['t0_ns'][()])
        assert group['t0_ns'].attrs['status'] == 'unknown'
        assert group['t0_voltage_ns'][()] == 38.
        assert json.loads(handle['instrument_metadata/json'].asstr()[()]) == metadata


def test_hdf5_records_schema_units_hashes_and_model_provenance(tmp_path):
    variables = SimpleNamespace(
        path=str(tmp_path),
        exp_name="provenance:test",
        electrode="NiC1",
        main_counter=[1, 2],
        main_raw_counter=[4, 8],
        main_temperature=[45.0, 46.0],
        main_chamber_vacuum=[1e-8, 2e-8],
        excluded_row_count=3,
        dataset=SimpleNamespace(source_hash="a" * 64),
        calibration_model_provenance={"model_type": "golden", "parameters": [1, 2]},
    )
    conf = {"tdc": "off", "v_dc": "off"}
    hdf_creator(variables, conf, [0, 1], [0.0, 0.1])
    output = tmp_path / "provenance_test.h5"

    with h5py.File(output, "r") as handle:
        provenance = handle["provenance"].attrs
        assert provenance["schema_version"] == "2.0"
        assert provenance["electrode_name"] == "NiC1"
        assert len(provenance["control_config_sha256"]) == 64
        assert provenance["calibration_input_sha256"] == "a" * 64
        assert provenance["excluded_row_count"] == 3
        assert json.loads(provenance["model_provenance_json"])["model_type"] == "golden"
        assert handle["apt/temperature"].attrs["units"] == "K"
        assert handle["apt/id"].attrs["units"] == "1"

    result = validate_hdf5(output)
    assert result["valid"], result["issues"]
    assert result["schema_version"] == "2.0"


def test_finalization_validates_chunks_once_for_metadata_and_detector(tmp_path, monkeypatch):
    chunks = tmp_path/'temp_data'/'chunks'
    for chunk_id in (1, 2):
        chunk_store.atomic_write_chunk_group(chunks, stream_name='apt', chunk_id=chunk_id,
            arrays={'apt_id': np.array([chunk_id], dtype=np.uint64),
                    'apt_temperature': np.array([40.+chunk_id])})
        chunk_store.atomic_write_chunk_group(chunks, stream_name='dld', chunk_id=chunk_id,
            arrays={'x': np.array([float(chunk_id)]), 'y': np.array([-float(chunk_id)])})
    original = chunk_store.validate_manifest_records
    calls = []
    def validate(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(chunk_store, 'validate_manifest_records', validate)
    variables = SimpleNamespace(path=str(tmp_path), exp_name='indexed', counter_source='TDC',
        main_counter=[1, 1], main_raw_counter=[1, 1], main_temperature=[0., 0.],
        main_chamber_vacuum=[1e-8, 1e-8], t=[0., 0.], main_v_dc_dld=[1500., 1500.],
        main_v_p_dld=[300., 300.], main_l_p_dld=[0., 0.], dld_start_counter=[1, 2],
        channel=[0, 0], time_data=[0, 0], tdc_start_counter=[1, 2],
        main_v_dc_tdc=[1500., 1500.], main_v_p_tdc=[300., 300.], main_l_p_tdc=[0., 0.])
    hdf_creator(variables, {'tdc': 'on', 'tdc_model': 'Surface_Concept'}, [0, 0], [0., 1.])
    assert len(calls) == 1
    with h5py.File(tmp_path/'indexed.h5', 'r') as handle:
        np.testing.assert_array_equal(handle['apt/id'][:], [1, 2])
        np.testing.assert_array_equal(handle['apt/temperature'][:], [41., 42.])
        np.testing.assert_array_equal(handle['dld/x'][:], [1., 2.])
        np.testing.assert_array_equal(handle['dld/y'][:], [-1., -2.])


def test_hdf5_keeps_alignment_settings_sample_and_outcome(tmp_path):
    variables = SimpleNamespace(
        path=str(tmp_path), exp_name='aligned', electrode='NiC1', main_counter=[1],
        main_raw_counter=[1], main_temperature=[45.], main_chamber_vacuum=[1e-8],
        automatic_alignment_enabled=True, alignment_sample=2,
        alignment_sample_position=(.001, .002, .003), alignment_outcome='aligned',
        alignment_settings={'start_voltage': 1500., 'voltage_increment': 125.},
        alignment_status={'phase': 'aligned', 'attempt': 2}, alignment_transfer_log=[{'kind': 'transfer'}],
    )
    hdf_creator(variables, {'tdc': 'off', 'v_dc': 'off'}, [0], [0.])
    with h5py.File(tmp_path/'aligned.h5', 'r') as handle:
        attrs = handle['provenance'].attrs
        assert attrs['automatic_alignment']
        assert attrs['alignment_sample'] == 2
        assert attrs['alignment_outcome'] == 'aligned'
        assert json.loads(attrs['alignment_settings_json'])['voltage_increment'] == 125.
        assert json.loads(attrs['alignment_transfer_json']) == [{'kind': 'transfer'}]


def test_hdf5_laser_telemetry_units_and_unknown_readings(tmp_path):
    import numpy as np
    variables = SimpleNamespace(path=str(tmp_path), exp_name='laser', main_counter=[1],
        main_raw_counter=[1], main_temperature=[45.], main_chamber_vacuum=[1e-8],
        laser_telemetry={'wavelength': 'DUV', 'wavelength_nm': 257.5,
                         'output_power_mw': float('nan'), 'valid': False},
        laser_alignment_enabled=True, laser_alignment_tracking=True,
        laser_alignment_run={'id': 'laser-session', 'settings': {'voltage_increment': 75.}},
        laser_alignment_status={'phase': 'stopped', 'reason': 'Experiment ended.'})
    chunks = tmp_path/'temp_data'/'chunks'
    chunks.mkdir(parents=True)
    for key, value in [('output_power_mw', 530.), ('pulse_energy_nj', 1325.),
                       ('output_frequency_hz', 400000.), ('wavelength_nm', 257.5), ('valid', 1.)]:
        np.save(chunks/f'apt_laser_{key}_chunk_1.npy', np.array([value], dtype=np.float64))
    hdf_creator(variables, {'tdc': 'off', 'v_dc': 'off'}, [0], [0.])
    with h5py.File(tmp_path/'laser.h5', 'r') as handle:
        assert handle['apt/laser_output_power_mw'][0] == 530.
        assert handle['apt/laser_output_power_mw'].attrs['units'] == 'mW'
        assert handle['apt/laser_pulse_energy_nj'].attrs['units'] == 'nJ'
        assert handle['apt/laser_output_frequency_hz'].attrs['units'] == 'Hz'
        assert handle['apt/laser_wavelength_nm'].attrs['units'] == 'nm'
        assert handle['provenance'].attrs['laser_readback_revision'] == 'unit-aware-1'
        metadata = json.loads(handle['provenance'].attrs['laser_final_readback_json'])
        assert metadata['output_power_mw'] is None
        assert metadata['wavelength'] == 'DUV'
        attrs = handle['provenance'].attrs
        assert attrs['laser_alignment_enabled'] and attrs['laser_alignment_tracking']
        assert json.loads(attrs['laser_alignment_run_json'])['settings']['voltage_increment'] == 75.
        assert json.loads(attrs['laser_alignment_final_status_json'])['phase'] == 'stopped'
