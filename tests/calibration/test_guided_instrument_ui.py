"""Physical previews stay tied to the instrument inputs the user reviewed."""
from threading import Event

import numpy as np
import pandas as pd
import pytest

from pyccapt.calibration.core.guided_workflow import GuidedWorkflow
from pyccapt.calibration.core.share_variables import Variables
from pyccapt.calibration.tutorials.tutorials_helpers import helper_guided_workflow as helper
from tests.calibration.test_guided_notebook_ui import ImmediateJob


def session():
    n = 24
    frame = pd.DataFrame({'event_id': np.arange(n, dtype=np.uint64)+2**60,
        'mc (Da)': np.linspace(26.9, 27.1, n), 't (ns)': np.linspace(350., 360., n),
        'high_voltage (V)': np.full(n, 5000.), 'pulse_v (V)': np.full(n, 500.),
        'x_det (cm)': np.linspace(-1., 1., n), 'y_det (cm)': np.zeros(n)})
    variables = Variables()
    variables.sync_from_data(frame)
    return GuidedWorkflow(variables)


def panel_for(monkeypatch, workflow, job_type=ImmediateJob):
    monkeypatch.setattr(helper, 'NotebookJob', job_type)
    monkeypatch.setattr(helper, 'display', lambda *_: None)
    panel = helper.show_guided_workflow(workflow, display_panel=False)
    panel.controls['flight'].value = 112.
    panel.controls['t0'].value = 11.
    panel.controls['pulse_mode'].value = 'laser'
    return panel


@pytest.mark.parametrize('field,value', [('flight', 113.), ('t0', 12.), ('pulse_mode', 'voltage')])
def test_geometry_edit_discards_physical_preview_before_acceptance(monkeypatch, field, value):
    workflow = session()
    panel = panel_for(monkeypatch, workflow)
    panel.controls['instrument_confirmed'].value = True
    panel.controls['Preview physical mass'].click()
    assert 'calibration' in workflow.preview
    panel.controls[field].value = value
    assert not panel.controls['instrument_confirmed'].value
    assert 'calibration' not in workflow.preview
    assert panel.controls['Accept calibration'].disabled
    panel.controls['Accept calibration'].click()
    assert workflow.accepted['calibration'] is None


class DeferredJob:
    def __init__(self):
        self.running = False
        self.errors = []

    def start(self, operation, *, on_result, on_error, on_progress=lambda *_: None, on_finished=lambda: None):
        self.running = True
        self.pending = operation, on_result, on_error, on_progress, on_finished

    def finish(self):
        operation, consume, error, progress, finished = self.pending
        try:
            consume(operation(Event(), progress))
        except Exception as caught:
            self.errors.append(caught)
            error(caught)
        finally:
            self.running = False
            finished()

    def cancel(self):
        pass


@pytest.mark.parametrize('change', ['geometry_roundtrip', 'confirmation_roundtrip'])
def test_pending_physical_result_rejected_after_edit_and_reconfirmation(monkeypatch, change):
    workflow = session()
    panel = panel_for(monkeypatch, workflow, DeferredJob)
    panel.controls['instrument_confirmed'].value = True
    panel.controls['Preview physical mass'].click()
    job = panel.controls['job']
    assert job.running
    if change == 'geometry_roundtrip':
        panel.controls['t0'].value = 12.
        panel.controls['t0'].value = 11.
    else:
        panel.controls['instrument_confirmed'].value = False
    panel.controls['instrument_confirmed'].value = True
    job.finish()
    assert 'calibration' not in workflow.preview
    assert workflow.accepted['calibration'] is None
    assert panel.controls['Accept calibration'].disabled


def test_imported_mass_preview_survives_unrelated_instrument_edits(monkeypatch):
    workflow = session()
    panel = panel_for(monkeypatch, workflow)
    panel.controls['Preview imported mass'].click()
    imported = workflow.preview['calibration']
    panel.controls['t0'].value = 88.
    panel.controls['flight'].value = 123.
    panel.controls['pulse_mode'].value = 'voltage'
    assert workflow.preview['calibration'] is imported
    assert not panel.controls['Accept calibration'].disabled
    panel.controls['Accept calibration'].click()
    assert workflow.accepted['calibration']['receipt']['method'] == 'imported mass'
    assert 't0_ns' not in workflow.accepted['calibration']['receipt']


def test_resumed_pending_physical_preview_restores_inputs_and_requires_confirmation(monkeypatch):
    workflow = session()
    panel = panel_for(monkeypatch, workflow)
    panel.controls['pulse_mode'].value = 'voltage'
    panel.controls['instrument_confirmed'].value = True
    panel.controls['Preview physical mass'].click()
    restored = GuidedWorkflow.from_state(*workflow.to_state())
    resumed = helper.show_guided_workflow(restored, display_panel=False)
    assert resumed.controls['flight'].value == 112.
    assert resumed.controls['t0'].value == 11.
    assert resumed.controls['pulse_mode'].value == 'voltage'
    assert not resumed.controls['instrument_confirmed'].value
    assert resumed.controls['Accept calibration'].disabled
    resumed.controls['Accept calibration'].click()
    assert restored.accepted['calibration'] is None
    resumed.controls['instrument_confirmed'].value = True
    assert not resumed.controls['Accept calibration'].disabled
    resumed.controls['Accept calibration'].click()
    assert restored.accepted['calibration']['receipt']['t0_ns'] == 11.
