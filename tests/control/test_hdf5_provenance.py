from __future__ import annotations

import json
from types import SimpleNamespace

import h5py
import numpy as np

from pyccapt.control.core.data_integrity import validate_hdf5
from pyccapt.control.core.hdf5_creator import hdf_creator
from pyccapt.control.core import chunk_store


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
