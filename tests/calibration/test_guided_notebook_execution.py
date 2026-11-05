"""Execute shipped notebook cells and explicit synthetic UI actions in-process."""
import ast
import asyncio
import json
from pathlib import Path

import nbformat

from tests.calibration.test_guided_notebook_ui import ImmediateJob


def test_shipped_guided_notebook_runs_full_synthetic_click_workflow(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    source = root / "pyccapt/calibration/tutorials/jupyter_files/data_processing_guided.ipynb"
    notebook = nbformat.read(source, as_version=4)
    # Keep the shipped cells unchanged. Additional verification cells represent
    # explicit human actions on independently generated, labelled synthetic data.
    cells = [f"artifact_directory = Path({json.dumps(str(tmp_path))})", r'''
import asyncio
import json
import numpy as np
import pandas as pd
from pyccapt.calibration.core.isotope_catalogue import get_isotope_catalogue

async def wait_job(job):
    deadline = asyncio.get_running_loop().time() + 45
    while job.running:
        assert asyncio.get_running_loop().time() < deadline, "Notebook job timed out"
        await asyncio.sleep(.025)

rng = np.random.default_rng(953)
n = 1800
index = np.arange(n)
ions = get_isotope_catalogue('Fe', (1, 2), 1)
reference_masses = sorted(candidate.mass for candidate in ions.candidates if candidate.isotope == (54,))
known_mass = np.asarray(reference_masses)[index % 2]
voltage = rng.uniform(4400, 5600, n)
x, y = rng.uniform(-1.5, 1.5, (2, n))
flight_m = np.sqrt(.110**2 + (.01*x)**2 + (.01*y)**2)
ideal_t = np.sqrt(known_mass*1.66e-27/(2*1.6e-19*voltage))*flight_m*1e9
scale = np.exp(.015 + .00001*(voltage-5000) + .001*x + .0015*y + .0002*x*x - .0001*x*y + .0003*y*y)
raw_t = 17.4 + ideal_t*scale + rng.normal(0, .02, n)
frame = pd.DataFrame({'mc (Da)': known_mass*1.1, 'mc_uc (Da)': known_mass*1.1,
    't (ns)': raw_t, 'high_voltage (V)': voltage, 'pulse_v (V)': np.zeros(n),
    'x_det (cm)': x, 'y_det (cm)': y, 'event_id': index.astype(np.uint64)+2**60})
input_path = artifact_directory / 'synthetic_input.h5'
frame.to_hdf(input_path, key='df')
guided_panel.controls['path'].value = str(input_path)
guided_panel.controls['flight'].value = 110.
guided_panel.controls['Import dataset'].click()
await wait_job(guided_panel.controls['job'])
workflow = guided_panel.controller['workflow']
assert workflow is not None and workflow.n == n
guided_panel.controls['Inspect source'].click()
guided_panel.controls['Preview imported mass'].click()
assert workflow.accepted['calibration'] is None
guided_panel.controls['Accept calibration'].click()
np.testing.assert_allclose(workflow.variables.mc, known_mass*1.1)
''', r'''
reference_windows = [{'mass_Da': float(mass), 'tof_min_ns': float(raw_t[known_mass == mass].min()-.001),
    'tof_max_ns': float(raw_t[known_mass == mass].max()+.001)} for mass in reference_masses]
assert reference_windows[0]['tof_max_ns'] < reference_windows[1]['tof_min_ns']
guided_panel.controls['reference_windows'].value = json.dumps(reference_windows)
guided_panel.controls['reference_confirmed'].value = True
guided_panel.controls['instrument_confirmed'].value = True
guided_panel.controls['Create reference session'].click()
calibration_panel = guided_panel.controller['calibration_panel']
assert calibration_panel is not None
calibration_panel.controls['Fit training'].click()
await wait_job(calibration_panel.controls['job'])
model_session = guided_panel.controller['calibration_session']
assert model_session.status == 'fitted'
for label, status in [('Validate selection', 'selection_passed'), ('Confirm once', 'confirmed'),
                      ('Accept model', 'accepted'), ('Apply accepted', 'applied')]:
    assert not calibration_panel.controls[label].disabled, label
    calibration_panel.controls[label].click()
    assert model_session.status == status, (label, model_session.status)
guided_panel.controls['Capture applied model'].click()
np.testing.assert_allclose(workflow.variables.mc, known_mass*1.1)  # still the previously accepted mass
guided_panel.controls['Accept calibration'].click()
assert np.max(np.abs(workflow.variables.mc-known_mass)) < .02
assert workflow.accepted['calibration']['model_state']['learned_model']
''', r'''
guided_panel.controls['Open peak-review queue'].click()
range_panel = guided_panel.controller['range_panel']
range_panel.controls['elements'].value = 'Fe'
range_panel.controls['Propose peaks'].click()
await wait_job(range_panel.controls['job'])
assert len(workflow.review.records) == 2
assert workflow.review.ranges().empty
for peak in list(workflow.review.records):
    range_panel.controls['peaks'].value = peak['id']
    selected = min(peak['candidates'], key=lambda candidate: abs(candidate['mass']-peak['center']))
    assert selected['isotope'] == [54]
    range_panel.controls['candidate'].value = selected['key']
    range_panel.controls['reviewed'].value = True
    range_panel.controls['Accept identity'].click()
assert workflow.accepted['ranges'] is None
range_panel.controls['Preview ranges'].click()
range_panel.controls['Accept ranges'].click()
assert len(workflow.accepted['ranges']['rows']) == 2
assert np.count_nonzero(workflow.accepted['ranges']['assignment'] >= 0) > .99*n
guided_panel.controls['Preview reconstruction'].click()
await wait_job(guided_panel.controls['job'])
assert workflow.accepted['reconstruction'] is None
guided_panel.controls['Accept reconstruction'].click()
assert np.isfinite(workflow.accepted['reconstruction']['xyz']).all()
guided_panel.controls['Analyze accepted ranges'].click()
assert workflow.accepted['analysis']['atomic_fractions_of_assigned_atoms'] == {'Fe': 1.0}
''', r'''
accepted_mass = workflow.variables.mc.copy()
accepted_xyz = workflow.accepted['reconstruction']['xyz'].copy()
saved_path = artifact_directory / 'resumable_session.h5'
exported_path = artifact_directory / 'accepted_outputs.h5'
guided_panel.controls['session_path'].value = str(saved_path)
guided_panel.controls['Save resumable session'].click()
assert saved_path.is_file()
guided_panel.controls['export_path'].value = str(exported_path)
guided_panel.controls['Export accepted HDF5'].click()
assert exported_path.is_file()
exported = pd.read_hdf(exported_path, key='df')
np.testing.assert_array_equal(exported['event_id'], frame['event_id'])
np.testing.assert_array_equal(exported['mc (Da)'], accepted_mass)
np.testing.assert_array_equal(exported[['x (nm)', 'y (nm)', 'z (nm)']].to_numpy(), accepted_xyz)
guided_panel.controls['Resume saved session'].click()
await wait_job(guided_panel.controls['job'])
restored = guided_panel.controller['workflow']
assert restored is not workflow
assert guided_panel.controller['calibration_panel'] is None
assert guided_panel.controller['range_panel'] is None
np.testing.assert_array_equal(restored.variables.mc, accepted_mass)
assert restored.accepted['analysis']['atomic_fractions_of_assigned_atoms'] == {'Fe': 1.0}
assert guided_panel.controls['flight'].value == 110.
guided_panel.controls['Undo accepted change'].click()
assert restored.accepted['analysis'] is None
assert restored.accepted['reconstruction'] is not None
print(json.dumps({'synthetic_only': True, 'events': n, 'maximum_mass_error_Da': float(np.max(np.abs(accepted_mass-known_mass))),
    'reviewed_ranges': len(restored.accepted['ranges']['rows']), 'stable_first_event_id': int(restored.event_ids[0]),
    'workflow_status': restored.status()}, indent=2))
''']
    notebook.cells.extend(nbformat.v4.new_code_cell(cell) for cell in cells)
    from pyccapt.calibration.tutorials.tutorials_helpers import helper_guided_workflow, helper_peak_review, guided_jobs
    monkeypatch.setattr(helper_guided_workflow, "NotebookJob", ImmediateJob)
    monkeypatch.setattr(helper_peak_review, "NotebookJob", ImmediateJob)
    monkeypatch.setattr(guided_jobs, "NotebookJob", ImmediateJob)
    namespace = {}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            code = compile(cell.source, str(source), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
            result = eval(code, namespace)
            if asyncio.iscoroutine(result):
                asyncio.run(result)
