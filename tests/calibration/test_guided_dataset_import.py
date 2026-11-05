"""Automatic format choice and a reversible, explicit notebook Browse/import flow."""
import h5py
import pytest

from pyccapt.calibration.core.exceptions import CalibrationInputError
from pyccapt.calibration.core.guided_workflow import GuidedWorkflow
from pyccapt.calibration.data_tools import data_tools, file_dialog
from pyccapt.calibration.data_tools.dataset_formats import format_from_path, resolve_dataset_format
from pyccapt.calibration.tutorials.tutorials_helpers import helper_guided_workflow as helper
from tests.calibration.test_guided_notebook_ui import ImmediateJob
from tests.calibration.test_guided_workflow_review import workflow


@pytest.mark.parametrize('name,expected', [('example.EPOS', 'leap_epos'), ('example.pos', 'leap_pos'),
    ('sample.APT', 'leap_apt'), ('sample.ATO', 'ato_v6'), ('sample.HDF5', 'pyccapt'),
    ('sample.nx5', 'nexus'), ('folder with spaces/example.nxs', 'nexus')])
def test_filename_hints_are_case_insensitive_and_require_no_io(name, expected):
    assert format_from_path(name) == expected
    if expected not in {'pyccapt', 'nexus'}:
        assert resolve_dataset_format(name) == (expected, 'processed')


@pytest.mark.parametrize('suffix,groups,expected', [
    ('.h5', ['dld'], ('pyccapt', 'raw')), ('.hdf5', ['df'], ('pyccapt', 'processed')),
    ('.h5', ['entry'], ('nexus', 'processed')), ('.nxs', ['dld'], ('pyccapt', 'raw')),
    ('.nx5', ['df', 'entry'], ('pyccapt', 'processed')),
    ('.h5', ['df', 'dld', 'entry'], ('pyccapt', 'processed'))])
def test_hdf_layout_is_detected_before_import_without_loading_event_arrays(tmp_path, monkeypatch, suffix, groups, expected):
    path = tmp_path/('sample'+suffix)
    with h5py.File(path, 'w') as handle:
        for name in groups:
            handle.create_group(name)
    calls = []
    source = workflow().source
    def load(filename, data_type, mode):
        calls.append((filename, data_type, mode))
        return source.copy()
    monkeypatch.setattr(data_tools, 'load_data', load)
    imported = GuidedWorkflow.import_file(path, data_type='auto')
    assert calls == [(str(path), *expected)]
    assert imported.n == len(source)
    assert imported.source.attrs['pyccapt_import_format'] == dict(zip(('data_type', 'mode'), expected))
    panel = helper.show_guided_workflow(imported, display_panel=False)
    panel.controls['path'].value = str(path)
    assert 'imported' in panel.controls['format_hint'].value
    assert ('raw /dld' in panel.controls['format_hint'].value) == (expected[1] == 'raw')


def test_manual_format_mode_overrides_remain_available_and_unknown_files_do_not_guess(tmp_path):
    with pytest.raises(CalibrationInputError, match='Cannot detect'):
        resolve_dataset_format('renamed.bin')
    assert resolve_dataset_format('renamed.bin', data_type='leap_epos') == ('leap_epos', 'processed')
    assert resolve_dataset_format('run.epos', mode='raw') == ('leap_epos', 'processed')
    # Explicit raw mode bypasses layout discovery; the raw reader performs validation.
    assert resolve_dataset_format('run.h5', mode='raw') == ('pyccapt', 'raw')
    nexus = tmp_path/'run.nxs'
    with h5py.File(nexus, 'w') as handle:
        handle.create_group('entry')
    assert resolve_dataset_format(nexus, data_type='nexus') == ('nexus', 'processed')
    with pytest.raises(CalibrationInputError, match='Unsupported HDF5 mode'):
        resolve_dataset_format(nexus, mode='unknown')
    channels = tmp_path/'channels.h5'
    with h5py.File(channels, 'w') as handle:
        handle.create_group('tdc')
    with pytest.raises(CalibrationInputError, match='convert raw detector channels'):
        resolve_dataset_format(channels)


def test_legacy_api_default_still_reads_hdf_with_nonstandard_extension(tmp_path, monkeypatch):
    path = tmp_path/'renamed.bin'
    with h5py.File(path, 'w') as handle:
        handle.create_group('dld')
    source = workflow().source
    calls = []
    monkeypatch.setattr(data_tools, 'load_data', lambda path, data_type, mode:
                        calls.append((data_type, mode)) or source.copy())
    GuidedWorkflow.import_file(path)
    assert calls == [('pyccapt', 'raw')]


def test_browse_sets_path_and_auto_format_without_importing_or_replacing_session(monkeypatch):
    chosen, calls = 'D:/folder with spaces/Run.EPOS', []
    monkeypatch.setattr(helper, 'NotebookJob', ImmediateJob)
    monkeypatch.setattr(file_dialog, 'choose_file_path',
                        lambda initial, **kw: calls.append((initial, kw)) or chosen)
    existing = workflow()
    panel = helper.show_guided_workflow(existing, display_panel=False)
    controls = panel.controls
    controls['path'].value = 'D:/previous.h5'
    controls['kind'].value = 'pyccapt'
    controls['mode'].value = 'raw'
    controls['Browse dataset'].click()
    assert calls == [('D:/previous.h5', {'file_kind': 'guided_dataset'})]
    assert controls['path'].value == chosen and controls['kind'].value == controls['mode'].value == 'auto'
    assert controls['mode'].disabled and 'LEAP epos' in controls['format_hint'].value
    assert panel.controller['workflow'] is existing  # Browse is not Import.
    imports = []
    monkeypatch.setattr(GuidedWorkflow, 'import_file', lambda path, **kw: imports.append((path, kw)) or existing)
    controls['Import dataset'].click()
    assert imports == [(chosen, dict(data_type='auto', mode='auto'))]


@pytest.mark.parametrize('cancel', [True, False])
def test_browse_cancel_or_dialog_failure_preserves_path_and_workflow(monkeypatch, cancel):
    def choose(*args, **kwargs):
        if cancel:
            return None
        raise RuntimeError('Picker unavailable')
    monkeypatch.setattr(file_dialog, 'choose_file_path', choose)
    existing = workflow()
    panel = helper.show_guided_workflow(existing, display_panel=False)
    panel.controls['path'].value = 'current.h5'
    panel.controls['kind'].value = 'pyccapt'
    panel.controls['Browse dataset'].click()
    assert panel.controller['workflow'] is existing
    assert panel.controls['path'].value == 'current.h5' and panel.controls['kind'].value == 'pyccapt'
    assert not panel.controls['Browse dataset'].disabled


def test_new_panel_can_browse_before_import_and_hint_does_not_read_file(monkeypatch):
    panel = helper.show_guided_workflow(display_panel=False)
    assert panel.controls['kind'].value == panel.controls['mode'].value == 'auto'
    assert not panel.controls['Browse dataset'].disabled
    monkeypatch.setattr(h5py, 'File', lambda *a, **k: pytest.fail('A filename hint must not open a file'))
    panel.controls['path'].value = 'not_imported_yet.h5'
    assert not panel.controls['mode'].disabled
    assert 'layout is checked on import' in panel.controls['format_hint'].value
    panel.controls['job'].running = True
    panel.controls['refresh']()
    assert panel.controls['Browse dataset'].disabled and panel.controls['mode'].disabled


def test_failed_automatic_import_preserves_existing_session_and_reports_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(helper, 'NotebookJob', ImmediateJob)
    existing = workflow()
    panel = helper.show_guided_workflow(existing, display_panel=False)
    panel.controls['path'].value = str(tmp_path/'not-supported.rhit')
    panel.controls['Import dataset'].click()
    assert panel.controller['workflow'] is existing
    assert panel.controls['progress'].bar_style == 'warning'
