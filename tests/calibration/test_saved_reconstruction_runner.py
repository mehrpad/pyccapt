"""The read-only numerical audit must preserve input and honor named columns."""
import importlib.util
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest


def _runner():
    path = Path(__file__).resolve().parents[2] / 'tools' / 'validate_saved_reconstruction.py'
    spec = importlib.util.spec_from_file_location('saved_reconstruction_runner_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_does_not_parse_cli_or_run(monkeypatch):
    monkeypatch.setattr('sys.argv', ['unused'])
    module = _runner()
    assert callable(module.main)


def test_reordered_named_coordinate_columns_are_read_correctly(tmp_path):
    module = _runner()
    source = tmp_path / 'saved.h5'
    # Saved x/y/z deliberately occupy separated, non-leading positions.
    columns = ['pulse_v (V)', 'z (nm)', 'high_voltage (V)', 'x_det (cm)',
               'y (nm)', 'y_det (cm)', 'x (nm)']
    values = np.arange(112*len(columns), dtype=float).reshape(112, len(columns))
    with h5py.File(source, 'w') as stream:
        frame = stream.create_group('df')
        frame.create_dataset('block0_items', data=np.asarray(columns, dtype='S'))
        frame.create_dataset('block0_values', data=values)
        frame.attrs['pyccapt_source_json'] = json.dumps({'pyccapt_source_id': 'stable-source'})
    before = source.read_bytes()
    variables, sample, xyz, metadata = module.load(source)
    np.testing.assert_array_equal(xyz, values[np.ix_(sample, [6, 4, 1])])
    np.testing.assert_array_equal(variables.dld_high_voltage, values[:, 2])
    assert metadata['rows'] == 112
    assert variables.data.attrs['pyccapt_source_id'] == 'stable-source'
    assert source.read_bytes() == before


@pytest.mark.parametrize('suffix', ['.json', '.png', '_detector.png'])
def test_input_output_hardlink_alias_rejected_without_writes(tmp_path, suffix):
    module = _runner()
    source = tmp_path / 'source.h5'
    original = b'original acquisition must not be overwritten'
    source.write_bytes(original)
    name = Path(module.REPORT_NAME)
    output = tmp_path / (name.with_suffix(suffix) if not suffix.startswith('_') else name.stem+suffix)
    os.link(source, output)
    with pytest.raises(SystemExit) as raised:
        module.parse_args(['--data', str(source), '--output-directory', str(tmp_path), '--flight-mm', '110', '--diagnose-signal'])
    assert raised.value.code == 2
    assert source.read_bytes() == original
    assert output.read_bytes() == original


def test_identical_input_output_rejected(tmp_path):
    module = _runner()
    source = tmp_path / module.REPORT_NAME
    source.write_bytes(b'unchanged')
    with pytest.raises(SystemExit):
        module.parse_args(['--data', str(source), '--output-directory', str(tmp_path), '--flight-mm', '110'])
    assert source.read_bytes() == b'unchanged'


def test_source_hash_does_not_require_python311_file_digest(tmp_path, monkeypatch):
    module = _runner()
    source = tmp_path / 'source.bin'
    # More than one block verifies accumulation, not just the first read.
    contents = b'acquisition' * 100000
    source.write_bytes(contents)
    monkeypatch.delattr(module.hashlib, 'file_digest', raising=False)
    assert module.file_sha256(source) == module.hashlib.sha256(contents).hexdigest()


def test_diagnostic_loader_preserves_cropped_export_receipt_for_prefix_guard(tmp_path):
    module = _runner()
    source = tmp_path/'cropped.h5'
    names = ['x (nm)','y (nm)','z (nm)','x_det (cm)','y_det (cm)','high_voltage (V)','pulse_v (V)']
    with h5py.File(source,'w') as stream:
        frame = stream.create_group('df')
        frame.create_dataset('block0_items',data=np.asarray(names,dtype='S'))
        frame.create_dataset('block0_values',data=np.ones((100,len(names))))
        frame.attrs['pyccapt_source_json'] = json.dumps({'pyccapt_selection': {
            'source_events': 200, 'selected_events': 100, 'preserves_full_source': False}})
    variables, _, _, _ = module.load(source)
    with pytest.raises(module.CalibrationInputError,match='cropped export'):
        module.sc.validate_full_source_metadata(variables)
