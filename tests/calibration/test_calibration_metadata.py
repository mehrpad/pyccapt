"""Shared provenance, exact same-source restoration, and rejection of mismatches."""
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

from pyccapt.calibration.core.calibration_metadata import ATTR, CalibrationMetadata, record_step, restore_saved_calibration
from pyccapt.calibration.core.exceptions import CalibrationInputError
from pyccapt.calibration.core.share_variables import Variables
from pyccapt.calibration.data_tools.data_tools import load_data, save_data
from pyccapt.calibration.data_tools.nexus_export import export_nexus


def raw_frame():
    return pd.DataFrame({'event_id': np.arange(30, dtype=np.uint64) + np.uint64(2**60),
        't (ns)': np.linspace(200., 700., 30), 'high_voltage (V)': np.linspace(3000., 5000., 30),
        'x_det (cm)': np.linspace(-1., 1., 30), 'y_det (cm)': np.zeros(30),
        'pulse_v (V)': np.full(30, 500.), 'start_counter': np.arange(30, dtype=np.uint64)})


def result_frame():
    source = raw_frame()
    variables = Variables()
    variables.sync_from_data(source)
    record_step(variables, 'voltage_correction', status='working', parameters={'units': 'ns'}, model=np.array([1., .1]))
    frame = variables.data.iloc[3:27].copy().reset_index(drop=True)
    frame['mc (Da)'] = np.linspace(10., 30., len(frame))
    frame['t_c (ns)'] = frame['t (ns)'] - 2.
    return source, frame


@pytest.mark.parametrize('nexus_only', [False, True])
def test_both_hdf_exports_keep_shared_journal_and_restore_exact_large_ids(tmp_path, nexus_only):
    source, frame = result_frame()
    path = tmp_path/'result.h5'
    if nexus_only:
        export_nexus(path, frame)
    else:
        save_data(frame, SimpleNamespace(result_data_path=str(tmp_path), result_data_name='result'))
    with h5py.File(path, 'r') as handle:
        journal = json.loads(handle['calibration_provenance/json'].asstr()[()])
        assert handle['entry/pyccapt_data/calibration_provenance'].id == handle['calibration_provenance'].id
    assert journal['steps'][0]['model'] == [1., .1]
    assert journal['steps'][0]['status'] == 'working'
    restored = restore_saved_calibration(path, source)
    for name in frame:
        np.testing.assert_array_equal(restored[name], frame[name])
    assert restored.attrs['pyccapt_calibration_restoration'] == {'source_verified': True, 'refit': False}


@pytest.mark.parametrize('change', ['tof', 'voltage', 'order', 'missing'])
def test_restore_refuses_different_original_measurements_or_order(tmp_path, change):
    source, frame = result_frame()
    save_data(frame, SimpleNamespace(result_data_path=str(tmp_path), result_data_name='result'), nexus=False)
    if change == 'tof':
        source.loc[0, 't (ns)'] += 1.
    elif change == 'voltage':
        source.loc[10, 'high_voltage (V)'] += 1.
    elif change == 'order':
        source = source.iloc[::-1].reset_index(drop=True)
    else:
        source = source.iloc[1:].reset_index(drop=True)
    with pytest.raises(CalibrationInputError, match='original analysis input/order'):
        restore_saved_calibration(tmp_path/'result.h5', source)


def test_corrupt_saved_output_is_rejected(tmp_path):
    source, frame = result_frame()
    path = tmp_path/'result.h5'
    save_data(frame, SimpleNamespace(result_data_path=str(tmp_path), result_data_name='result'), nexus=False)
    with pd.HDFStore(path, 'a') as store:
        altered = store['df']
        altered.loc[0, 'mc (Da)'] = 999.
        store.put('df', altered)
    # Rewriting via pandas may drop storer attrs; preserve the original manifest explicitly.
    with pd.HDFStore(path, 'a') as store:
        original = frame.copy()
        metadata = CalibrationMetadata.from_frame(original)
        metadata.checkpoint(original)
        store.get_storer('df').attrs.pyccapt_source_json = json.dumps(original.attrs)
    with pytest.raises(CalibrationInputError, match='fingerprint'):
        restore_saved_calibration(path, source)


def test_csv_gets_automatic_sidecar_and_hdf_without_nexus_keeps_metadata(tmp_path):
    _, frame = result_frame()
    variables = SimpleNamespace(result_data_path=str(tmp_path), result_data_name='result')
    save_data(frame, variables, csv=True, nexus=False)
    loaded = load_data(str(tmp_path/'result.h5'), 'pyccapt')
    sidecar = json.loads((tmp_path/'result.csv.calibration.json').read_text())
    assert loaded.attrs[ATTR] == sidecar
    assert sidecar['checkpoint']['outputs']['mc (Da)']


def test_guided_state_and_undo_keep_processing_history():
    from tests.calibration.test_guided_workflow_review import workflow
    from pyccapt.calibration.core.guided_workflow import GuidedWorkflow
    session = workflow()
    session.preview_crop(event_start=10, event_stop=100)
    session.accept('crop')
    session.undo()
    resumed = GuidedWorkflow.from_state(*session.to_state())
    steps = resumed.variables.calibration_metadata.state['steps']
    assert [step['operation'] for step in steps][-2:] == ['guided_accept_crop', 'guided_undo']
    assert resumed.active_mask.all()


@pytest.mark.parametrize('apply', [True, False])
def test_bowl_journal_tracks_only_applied_working_fits_and_preserves_math(monkeypatch, apply):
    from pyccapt.calibration.core import calibration
    frame = raw_frame()
    variables = Variables()
    variables.sync_from_data(frame)
    variables.mc_calib = np.full(len(frame), 110.)
    variables.set_calibration_peak_range('mc', 105., 115.)
    monkeypatch.setattr(calibration, 'bowl_correction', lambda *a, **k: np.array([1.1, 0., 0., 0., 0., 0.]))
    factors = calibration.bowl_correction_main(variables.dld_x_det, variables.dld_y_det, variables.dld_high_voltage,
        variables, 80., 4, 'curve_fit', 'mc', 0, False, False, calibration_apply=apply)
    np.testing.assert_allclose(factors, 1.1)
    np.testing.assert_allclose(variables.mc_calib, 100. if apply else 110.)
    steps = variables.calibration_metadata.state['steps']
    assert bool(steps) == apply
    if apply:
        assert steps[-1]['operation'] == 'bowl_correction' and steps[-1]['status'] == 'working'
        assert steps[-1]['parameters']['detector_input_units'] == 'cm converted to mm'
