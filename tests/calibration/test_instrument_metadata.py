"""Instrument provenance survives raw/legacy/NeXus paths without invented settings."""
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

from pyccapt.instrument_metadata import (
    ATTR, acquisition_metadata, extract_instrument_settings, record_calibration_settings,
    record_mode_observation, write_instrument_metadata,
)
from pyccapt.calibration.data_tools import data_tools, data_loadcrop
from pyccapt.calibration.data_tools.nexus_export import export_nexus


def metadata():
    return acquisition_metadata({'flight_path_length': 110., 't_0_voltage': 38.,
                                 't_0_laser': 800., 'device_name': 'synthetic'}, pulse_mode='Voltage')


def frame():
    data = pd.DataFrame({'event_id': np.array([2**60+7, 2**60+9], dtype=np.uint64),
                         'mc (Da)': [10., 20.], 't (ns)': [100., 200.],
                         'high_voltage (V)': [5000., 5100.], 'pulse_v (V)': [0., 0.],
                         'pulse_l (pJ)': [0., 0.]})
    data.attrs[ATTR] = metadata()
    data.attrs['pyccapt_acquisition_metadata'] = {'acquisition_completeness': 'incomplete'}
    data.attrs['pyccapt_selection'] = {'schema_version': '1.0', 'source_events': 10,
                                      'selected_events': 2, 'preserves_full_source': False}
    return data


def test_mode_and_offset_sources_are_not_defaulted_or_mixed():
    data = frame()
    assert extract_instrument_settings(data) == {'flight_path_length_mm': 110., 't0_ns': 38., 'pulse_mode': 'voltage'}
    data.attrs[ATTR] = record_mode_observation(data.attrs[ATTR], 'Laser', control_iteration=42)
    assert extract_instrument_settings(data) == {'flight_path_length_mm': 110.}
    assert data.attrs[ATTR]['acquisition']['pulse_mode']['value'] == 'mixed'
    assert data.attrs[ATTR]['acquisition']['pulse_mode_history'][-1]['control_iteration'] == 42
    record_calibration_settings(data, flight_path_length_mm=120., pulse_mode='laser', source='user')
    assert extract_instrument_settings(data) == {'flight_path_length_mm': 120., 'pulse_mode': 'laser'}
    assert data.attrs[ATTR]['calibration']['t0_ns']['status'] == 'unknown'


@pytest.mark.parametrize('bad', [None, float('nan'), float('inf'), -2., 'unknown'])
def test_unknown_or_invalid_flight_length_is_not_a_suggestion(bad):
    data = pd.DataFrame()
    record_calibration_settings(data, flight_path_length_mm=bad, t0_ns=0., pulse_mode='voltage', source='test')
    assert 'flight_path_length_mm' not in extract_instrument_settings(data)
    assert extract_instrument_settings(data)['t0_ns'] == 0.


def test_legacy_save_roundtrip_and_default_nexus_keep_settings_ids_and_crop(tmp_path):
    data = frame()
    record_calibration_settings(data, flight_path_length_mm=112., t0_ns=37., pulse_mode='voltage',
                                status='fitted_model', source='accepted synthetic model')
    attrs = dict(data.attrs)
    variables = SimpleNamespace(result_data_path=str(tmp_path), result_data_name='legacy')
    data_tools.save_data(data, variables)
    output = tmp_path/'legacy.h5'
    loaded = data_tools.load_data(output, 'pyccapt')
    pd.testing.assert_frame_equal(loaded, data)
    assert {key: loaded.attrs[key] for key in attrs} == attrs
    assert loaded.attrs['pyccapt_calibration_metadata']['checkpoint']['input']['rows'] == len(loaded)
    assert data.attrs == attrs
    with h5py.File(output, 'r') as handle:
        assert 'df' in handle and 'entry' in handle
        assert 'definition' not in handle['entry']
        path = 'entry/atom_probe1/instrument_metadata/calibration/t0_ns'
        assert handle[path][()] == 37.
        assert handle[path].attrs['units'] == 'ns'
        assert handle[path].attrs['status'] == 'fitted_model'
        assert handle['instrument_metadata'].id == handle['entry/atom_probe1/instrument_metadata'].id
        report = json.loads(handle['entry/mapping_report_json'].asstr()[()])
        assert report['acquisition_completeness'] == 'incomplete'
        assert report['selection']['source_events'] == 10


def test_standalone_partial_nexus_roundtrip_and_unknown_foreign_layout(tmp_path):
    data = frame()
    output = tmp_path/'mapped.nxs'
    export_nexus(output, data)
    loaded = data_tools.load_data(output, 'nexus')
    pd.testing.assert_frame_equal(loaded, data)
    for key, value in data.attrs.items():
        assert loaded.attrs[key] == value
    foreign = tmp_path/'foreign.nxs'
    with h5py.File(foreign, 'w') as handle:
        handle.create_group('entry')
    with pytest.raises(ValueError, match='Unsupported NeXus'):
        data_tools.load_data(foreign, 'nexus')


def raw_file(path, *, pulses=True, explicit_metadata=True):
    with h5py.File(path, 'w') as handle:
        for name, values in {'x': [.1, .2], 'y': [.2, .3], 't': [100., 200.],
                             'high_voltage': [5000., 5100.]}.items():
            handle.create_dataset('dld/'+name, data=values)
        if pulses:
            for name in ('voltage_pulse', 'laser_pulse', 'start_counter'):
                handle.create_dataset('dld/'+name, data=[0., 0.])
        if explicit_metadata:
            write_instrument_metadata(handle, metadata())
        else:
            provenance = handle.create_group('provenance')
            provenance.attrs['control_config_json'] = json.dumps({
                'flight_path_length': 110., 't_0_voltage': 38., 'default_pulse_mode': 'Voltage'})


def test_raw_eager_and_lazy_loaders_retain_metadata_and_measured_zero(tmp_path):
    path = tmp_path/'raw.h5'
    raw_file(path)
    eager = data_tools.load_data(path, 'pyccapt', mode='raw')
    assert extract_instrument_settings(eager)['t0_ns'] == 38.
    assert 'pulse_v (V)' not in eager.attrs.get('pyccapt_unavailable_columns', [])
    with data_loadcrop.fetch_dataset_from_dld_grp(path, lazy=True) as lazy:
        assert lazy.attrs[ATTR] == eager.attrs[ATTR]
        assert extract_instrument_settings(lazy.head(1))['pulse_mode'] == 'voltage'
        assert extract_instrument_settings(lazy.to_dataframe())['flight_path_length_mm'] == 110.


def test_raw_compatibility_zeros_and_legacy_default_mode_are_not_measurements(tmp_path):
    path = tmp_path/'old.h5'
    raw_file(path, pulses=False, explicit_metadata=False)
    data = data_tools.load_data(path, 'pyccapt', mode='raw')
    assert set(data.attrs['pyccapt_unavailable_columns']) == {'pulse_v (V)', 'pulse_l (pJ)', 'start_counter'}
    assert extract_instrument_settings(data) == {'flight_path_length_mm': 110.}
    np.testing.assert_array_equal(data['pulse_v (V)'], [0., 0.])


def test_nexus_opt_out_still_retains_instrument_metadata(tmp_path):
    variables = SimpleNamespace(result_data_path=str(tmp_path), result_data_name='oldformat')
    data_tools.save_data(frame(), variables, nexus=False)
    with h5py.File(tmp_path/'oldformat.h5', 'r') as handle:
        assert 'entry' not in handle
        assert 'instrument_metadata' in handle


def test_original_notebook_picker_and_loader_expose_nexus(tmp_path):
    from pyccapt.calibration.core.share_variables import Variables
    from pyccapt.calibration.core.widgets import dataset_instrument_specification_selection
    from pyccapt.calibration.data_tools.dataset_path_qt import DATASET_FILTER
    from pyccapt.calibration.tutorials.tutorials_helpers import helper_data_loader
    widgets = dataset_instrument_specification_selection()
    widgets[0].value = 'nexus'
    assert widgets[0].label == 'PyCCAPT NeXus mapping'
    assert '*.nxs' in DATASET_FILTER and '*.nx5' in DATASET_FILTER
    data = frame()
    output = tmp_path/'original.nxs'
    export_nexus(output, data)
    variables = Variables()
    helper_data_loader.load_data(str(output), 400., 110., 'voltage', 'nexus', variables,
                                 show_progress=False, merge_partial_tdc=False)
    np.testing.assert_array_equal(variables.data.event_id, data.event_id)
    assert variables.data.attrs[ATTR]['acquisition']['t0_ns']['value'] == 38.
    assert variables.data.attrs[ATTR]['legacy_workflow_inputs']['t0_ns']['status'] == 'unknown'


def test_parameter_status_distinguishes_fitted_offset_from_fixed_geometry():
    data = frame()
    record_calibration_settings(data, flight_path_length_mm=110., t0_ns=37., pulse_mode='voltage',
        source='accepted receipt', status={'t0_ns': 'fitted_model', 'flight_path_length_mm': 'user_supplied'})
    settings = data.attrs[ATTR]['calibration']
    assert settings['t0_ns']['status'] == 'fitted_model'
    assert settings['flight_path_length_mm']['status'] == 'user_supplied'
    assert settings['pulse_mode']['status'] == 'user_supplied'


def test_empty_legacy_calibration_panel_does_not_create_data_or_metadata(monkeypatch):
    from pyccapt import instrument_metadata
    from pyccapt.calibration.core.share_variables import Variables
    from pyccapt.calibration.tutorials.tutorials_helpers import helper_calibration as helper
    variables = Variables()
    variables.max_tof = 1000
    original = {name: getattr(variables, name) for name in
                ('data', 'dataset', 'mc', 'mc_uc', 'mc_calib', 'dld_t', 'dld_t_calib', 'range_data')}
    assert variables.data is None
    displayed = []
    monkeypatch.setattr(helper, 'display', lambda *objects: displayed.extend(objects))
    monkeypatch.setattr(helper.mc_plot, 'hist_plot', lambda *args, **kwargs: None)
    monkeypatch.setattr(instrument_metadata, 'record_calibration_settings',
                        lambda *args, **kwargs: pytest.fail('No loaded frame exists for metadata'))
    helper.call_voltage_bowl_calibration(variables, 80., 110., 'laser', t0=38.)
    assert displayed
    for name, value in original.items():
        assert getattr(variables, name) is value
    assert ATTR not in vars(variables)
    for panel in displayed:
        if hasattr(panel, 'close'):
            panel.close()


def test_imported_mass_acceptance_preserves_historical_calibration_context(tmp_path):
    from pyccapt.calibration.core.guided_workflow import GuidedWorkflow
    from pyccapt.calibration.core.share_variables import Variables
    data = frame()
    record_calibration_settings(data, flight_path_length_mm=111., t0_ns=35., pulse_mode='voltage',
                                source='historical fitted model', status='fitted_model')
    historical = data.attrs[ATTR]['calibration'].copy()
    variables = Variables()
    variables.sync_from_data(data)
    workflow = GuidedWorkflow(variables)
    assert workflow.working_variables().pulse_mode == 'voltage'
    workflow.preview_imported_mass(flight_length_mm=120., pulse_mode='laser')
    workflow.accept('calibration')
    exported = workflow.export_frame()
    assert exported.attrs[ATTR]['calibration'] == historical
    assert exported.attrs[ATTR]['import_inputs']['t0_ns']['status'] == 'unknown'
    assert extract_instrument_settings(exported)['t0_ns'] == 35.
    path = tmp_path/'preserved.nxs'
    export_nexus(path, exported)
    with h5py.File(path, 'r') as handle:
        group = handle['entry/atom_probe1/instrument_metadata']
        assert group['calibration/t0_ns'][()] == 35.
        assert group['import_inputs/flight_path_length_mm'][()] == 120.


def test_further_export_slice_updates_scope_without_changing_full_source_metadata(tmp_path):
    variables = SimpleNamespace(result_data_path=str(tmp_path), result_data_name='subset')
    data = frame()
    data_tools.save_data(data, variables, start_index=1)
    loaded = data_tools.load_data(tmp_path/'subset.h5', 'pyccapt')
    selection = loaded.attrs['pyccapt_selection']
    assert selection['source_events'] == 10
    assert selection['selected_events'] == 1
    assert selection['export_input_events'] == 2
    assert not selection['preserves_full_source']
    assert data.attrs['pyccapt_selection']['selected_events'] == 2
    np.testing.assert_array_equal(loaded.event_id, [2**60+9])
