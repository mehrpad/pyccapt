from __future__ import annotations

import json
from types import SimpleNamespace

import h5py

from pyccapt.control.core.data_integrity import validate_hdf5
from pyccapt.control.core.hdf5_creator import hdf_creator


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
