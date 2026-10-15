"""Guided exports retain known acquisition completeness instead of downgrading it."""
from copy import deepcopy
import json

import h5py
import pytest

from pyccapt.calibration.core.guided_workflow import GuidedWorkflow
from pyccapt.calibration.core.share_variables import Variables
from pyccapt.calibration.tutorials.tutorials_helpers import helper_guided_workflow as helper
from tests.calibration.test_guided_instrument_ui import session


@pytest.mark.parametrize('format', ['hdf', 'nexus'])
@pytest.mark.parametrize('recorded', ['incomplete', 'complete', None])
def test_guided_export_preserves_recorded_completeness(tmp_path, monkeypatch, format, recorded):
    frame = session().source.copy(deep=True)
    if recorded is not None:
        frame.attrs['pyccapt_acquisition_metadata'] = {
            'acquisition_completeness': recorded, 'experiment_name': 'synthetic acquisition'}
    variables = Variables()
    variables.sync_from_data(frame)
    workflow = GuidedWorkflow(variables)
    workflow.preview_crop(event_start=2, event_stop=20)
    workflow.accept('crop')
    workflow.preview_imported_mass()
    workflow.accept('calibration')
    before = deepcopy(workflow.source.attrs)
    monkeypatch.setattr(helper, 'display', lambda *_: None)
    panel = helper.show_guided_workflow(workflow, display_panel=False)
    target = tmp_path / ('accepted.h5' if format == 'hdf' else 'accepted.nxs')
    panel.controls['export_path'].value = str(target)
    panel.controls['Export accepted HDF5' if format == 'hdf' else 'Export NeXus mapping'].click()
    assert target.is_file()
    expected = 'unverified' if recorded is None else recorded
    with h5py.File(target, 'r') as handle:
        report = json.loads(handle['entry/mapping_report_json'].asstr()[()])
        experiment = json.loads(handle['entry/pyccapt_data/experiment_metadata_json'].asstr()[()])
        provenance = json.loads(handle['entry/pyccapt_data/dataframe_provenance_json'].asstr()[()])
        assert report['acquisition_completeness'] == experiment.get('acquisition_completeness', 'unverified') == expected
        assert report['selection']['selected_events'] == 18
        if recorded is not None:
            assert provenance['pyccapt_acquisition_metadata']['acquisition_completeness'] == recorded
        if format == 'hdf':
            saved_attrs = json.loads(handle['df'].attrs['pyccapt_source_json'])
            if recorded is not None:
                assert saved_attrs['pyccapt_acquisition_metadata']['acquisition_completeness'] == recorded
            assert 'analysis_session' in handle
    assert workflow.source.attrs == before
