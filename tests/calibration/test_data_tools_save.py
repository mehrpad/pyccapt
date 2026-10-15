import pandas as pd
import pytest
import inspect

from pyccapt.calibration.data_tools import data_tools


def test_slice_data_for_export_uses_inclusive_end_index():
    data = pd.DataFrame({"value": [10, 11, 12, 13, 14]})

    sliced = data_tools._slice_data_for_export(data, start_index=1, end_index=3)

    assert sliced["value"].tolist() == [11, 12, 13]


def test_slice_data_for_export_supports_minus_one_for_last_row():
    data = pd.DataFrame({"value": [10, 11, 12]})

    sliced = data_tools._slice_data_for_export(data, start_index=1, end_index=-1)

    assert sliced["value"].tolist() == [11, 12]


def test_slice_data_for_export_rejects_reversed_bounds():
    data = pd.DataFrame({"value": [10, 11, 12]})

    with pytest.raises(ValueError, match="end_index must be greater than or equal to start_index"):
        data_tools._slice_data_for_export(data, start_index=2, end_index=1)


def test_save_data_uses_widget_friendly_export_defaults():
    signature = inspect.signature(data_tools.save_data)

    assert signature.parameters["start_index"].default == 0
    assert signature.parameters["end_index"].default == -1


def test_original_notebook_save_panel_builds_without_saving(monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    import ipywidgets as widgets
    import ipywidgets.widgets.interaction as interaction

    notebook = Path(__file__).resolve().parents[2] / 'pyccapt/calibration/tutorials/jupyter_files/data_processing.ipynb'
    original = notebook.read_bytes()
    cells = json.loads(original)['cells']
    save_cell = next(''.join(cell['source']) for cell in cells
                     if 'interact_manual_data(data_tools.save_data' in ''.join(cell.get('source', [])))
    variables = SimpleNamespace(data=pd.DataFrame({'mc (Da)': [10., 20.]}),
                                result_data_name='synthetic', data_tdc=None)
    monkeypatch.setattr(interaction, 'display', lambda *args: None)
    monkeypatch.setattr(data_tools, '_resolve_variable_output_file',
                        lambda *args, **kwargs: pytest.fail('Constructing Save data must not save'))
    namespace = dict(data_tools=data_tools, variables=variables, widgets=widgets,
                     fixed=widgets.fixed, interact_manual=widgets.interact_manual)
    exec(compile(save_cell, str(notebook), 'exec'), namespace)
    panel = data_tools.save_data.widget
    try:
        controls = {widget._kwarg: widget for widget in panel.kwargs_widgets}
        assert panel.manual and panel.manual_button.description == 'Save data'
        assert controls['nexus'].value is True
        assert 'analysis_session' not in controls and 'nexus_metadata' not in controls
        assert notebook.read_bytes() == original
    finally:
        panel.close()
        del data_tools.save_data.widget


def test_save_widget_accepts_explicit_fixed_metadata_without_saving(monkeypatch):
    import ipywidgets as widgets
    state, metadata = ({'source': 'synthetic'}, {}), {'experiment_alias': 'explicit'}
    monkeypatch.setattr(data_tools, '_resolve_variable_output_file',
                        lambda *args, **kwargs: pytest.fail('Constructing Save data must not save'))
    panel = widgets.interactive(data_tools.save_data, {'manual': True},
                                data=widgets.fixed(None), variables=widgets.fixed(None),
                                name=widgets.fixed('synthetic'), analysis_session=widgets.fixed(state),
                                nexus_metadata=widgets.fixed(metadata))
    try:
        controls = {widget._kwarg: widget for widget in panel.kwargs_widgets}
        assert controls['analysis_session'].value is state
        assert controls['nexus_metadata'].value is metadata
    finally:
        panel.close()


def test_save_data_retains_positional_metadata_arguments(tmp_path):
    from types import SimpleNamespace
    import h5py
    import numpy as np
    from pyccapt.calibration.data_tools.analysis_session import load_analysis_session
    variables = SimpleNamespace(result_data_name='unused', result_data_path=str(tmp_path))
    frame = pd.DataFrame({'event_id': np.array([7, 9], dtype=np.uint64), 'mc (Da)': [10., 20.]})
    state = ({'source': 'test'}, {'event_ids': frame.event_id.to_numpy()})
    data_tools.save_data(frame, variables, 'positional', True, False, False, False, False,
                         False, 0, -1, False, False, state, {'experiment_alias': 'synthetic'}, True)
    path = tmp_path/'positional.h5'
    metadata, arrays = load_analysis_session(path)
    assert metadata == state[0]
    np.testing.assert_array_equal(arrays['event_ids'], frame.event_id.to_numpy())
    with h5py.File(path, 'r') as handle:
        assert handle['entry/experiment_alias'].asstr()[()] == 'synthetic'


def test_save_data_preserves_accepted_math_and_subset_scope(tmp_path):
    import json
    from types import SimpleNamespace
    receipt = {'status': 'applied', 'source_hash': 'a' * 64,
               'model': {'t0_ns': 12., 'coefficients': [.1, .2]},
               'output': {'units': 'ns', 'convention': 'normalized reference ToF'}}
    variables = SimpleNamespace(result_data_name='accepted', result_data_path=str(tmp_path),
                                guided_calibration_receipt=receipt,
                                reconstruction_parameters={'kf': 4., 'icf': 1.4})
    frame = pd.DataFrame({'mc (Da)': [10., 20., 30.]})
    data_tools.save_data(frame, variables, start_index=1)
    path = tmp_path/'accepted.h5'
    with pd.HDFStore(path, 'r') as store:
        metadata = json.loads(store.get_storer('df').attrs.pyccapt_analysis_json)
        pd.testing.assert_frame_equal(store['df'], frame.iloc[1:].reset_index(drop=True))
    assert metadata['guided_calibration'] == receipt
    assert metadata['reconstruction_parameters'] == {'kf': 4., 'icf': 1.4}
    assert metadata['export_rows'] == 2
    assert 'subset' in metadata['receipt_scope']


def test_save_data_embeds_resumable_session_and_nexus_without_changing_dataframe(tmp_path):
    from types import SimpleNamespace
    import numpy as np
    import h5py
    from pyccapt.calibration.data_tools.analysis_session import load_analysis_session
    variables = SimpleNamespace(result_data_name='complete', result_data_path=str(tmp_path))
    frame = pd.DataFrame({'event_id': np.array([2**60+7, 2**60+9], dtype=np.uint64), 'mc (Da)': [10., 20.]})
    state = ({'source': 'test'}, {'event_ids': frame.event_id.to_numpy(), 'mass': frame['mc (Da)'].to_numpy()})
    data_tools.save_data(frame, variables, analysis_session=state, nexus_metadata={'experiment_alias': 'synthetic'})
    path = tmp_path/'complete.h5'
    pd.testing.assert_frame_equal(pd.read_hdf(path, 'df'), frame)
    metadata, arrays = load_analysis_session(path)
    assert metadata == state[0]
    np.testing.assert_array_equal(arrays['event_ids'], frame.event_id.to_numpy())
    with h5py.File(path, 'r') as handle:
        assert handle['entry'].attrs['application_definition_target'] == 'NXapm'


def test_processed_crop_roundtrip_retains_source_identity_and_missing_field_status(tmp_path):
    import numpy as np
    from types import SimpleNamespace
    from pyccapt.calibration.core.dataset import CalibrationDataset
    source = CalibrationDataset.from_frame(pd.DataFrame({'mc (Da)': [10., 20., 30.], 'pulse_l (pJ)': [0., 0., 0.]}))
    variables = SimpleNamespace(result_data_name='crop', result_data_path=str(tmp_path))
    data_tools.save_data(source.frame, variables, start_index=1)
    with pytest.warns(RuntimeWarning, match='pulse'):
        loaded = data_tools.load_data(tmp_path/'crop.h5', 'pyccapt')
    restored = CalibrationDataset.from_frame(loaded)
    assert restored.source_id == source.source_id
    np.testing.assert_array_equal(restored.event_ids, [1, 2])
    np.testing.assert_array_equal(restored.validity_status('pulse_v (V)'), [2, 2])
    np.testing.assert_array_equal(restored.validity_status('pulse_l (pJ)'), [0, 0])
    # Numeric compatibility fills remain, but they are not reported as measurements.
    np.testing.assert_array_equal(restored.frame['pulse_v (V)'], [0., 0.])
    data_tools.save_data(restored.frame, variables)
    loaded_again = CalibrationDataset.from_frame(data_tools.load_data(tmp_path/'crop.h5', 'pyccapt'))
    np.testing.assert_array_equal(loaded_again.validity_status('pulse_v (V)'), [2, 2])


def test_failed_session_embedding_preserves_existing_export_and_unrelated_temp(tmp_path):
    import numpy as np
    from types import SimpleNamespace
    from pyccapt.calibration.core.exceptions import CalibrationInputError
    variables = SimpleNamespace(result_data_name='safe', result_data_path=str(tmp_path))
    frame = pd.DataFrame({'mc (Da)': [10., 20.]})
    data_tools.save_data(frame, variables)
    original = (tmp_path/'safe.h5').read_bytes()
    unrelated = tmp_path/'safe.h5.tmp'
    unrelated.write_bytes(b'owned by another writer')
    with pytest.raises(CalibrationInputError):
        data_tools.save_data(frame*2, variables, analysis_session=({}, {'unsupported': np.array([object()])}))
    assert (tmp_path/'safe.h5').read_bytes() == original
    assert unrelated.read_bytes() == b'owned by another writer'
    assert list(tmp_path.glob('.*.tmp')) == []
