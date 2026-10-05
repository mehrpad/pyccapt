"""Build the complete laser GUI with both device connection paths replaced."""
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PyQt6 import QtWidgets

from pyccapt.control.core.read_files import read_toml_file
from pyccapt.control.core.share_variables import Variables
from pyccapt.control.gui.gui_laser_control import Ui_Laser_Control


@pytest.fixture
def gui(monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    conf = read_toml_file(Path(__file__).parents[2]/'pyccapt/config.toml')
    v = Variables(conf, SimpleNamespace())
    monkeypatch.setattr(Ui_Laser_Control, '_open_laser_cli', lambda *a, **k: None)
    monkeypatch.setattr(Ui_Laser_Control, '_connect_stage_device', lambda *a: None)
    window = QtWidgets.QWidget()
    ui = Ui_Laser_Control(v, conf)
    ui.setupUi(window)
    ui.error_message = Mock()
    yield ui, v, window, app
    ui._laser_alignment_timer.stop()
    ui._laser_status_timer.stop()
    window.deleteLater()


def test_gui_has_editable_increment_and_plots(gui):
    ui, v, _, _ = gui
    field = ui.laser_alignment_fields['voltage_increment']
    assert field.value() == 50
    field.setValue(75)
    assert v.laser_alignment_settings['voltage_increment'] == 75
    assert ui.alignment_plot.tabs.count() == 2
    assert not ui.laser_alignment_buttons[0].isEnabled()


def test_alignment_locks_manual_motion_and_settings(gui):
    ui, v, _, _ = gui
    v.start_flag = True
    v.experiment_state = 'running'
    v.laser_alignment_status = {'active': True, 'phase': 'fine'}
    ui._tick_laser_alignment()
    assert not ui.laser_left.isEnabled()
    assert not ui.laser_home.isEnabled()
    assert not ui.laser_power.isEnabled()
    assert not ui.laser_alignment_fields['voltage_increment'].isEnabled()
    assert ui.laser_alignment_stop.isEnabled()
    ui._cancel_laser_alignment()
    assert v.laser_alignment_cancel
    assert v.laser_alignment_command['mode'] == 'stop'
    assert not v.stop_flag


def test_uncalibrated_auto_enable_shows_error(gui):
    ui, v, _, _ = gui
    ui.laser_auto_alignment.setChecked(True)
    assert not v.laser_alignment_enabled
    assert not ui.laser_auto_alignment.isChecked()
    assert ui.error_message.called


def test_plot_history_resets_between_alignment_sessions(gui):
    ui, v, _, _ = gui
    base = dict(origin_m=(0, 0, 0), position_m=(1e-6, 0, 0), rate_percent=.4,
                voltage=1500, phase='coarse', time=1, session='one', quality={})
    ui.alignment_plot.append(base)
    ui.alignment_plot.append(base)
    assert len(ui.alignment_plot.records) == 1
    ui.alignment_plot.append({**base, 'time': 2, 'session': 'two'})
    assert len(ui.alignment_plot.records) == 1
    assert ui.alignment_plot.session == 'two'


@pytest.mark.parametrize('first_poll_fails', [False, True])
def test_legacy_status_worker_survives_success_and_transient_failure(monkeypatch, caplog, first_poll_fails):
    from pyccapt.control.gui.gui_laser_control import Worker
    calls = []
    def poll():
        calls.append(True)
        if len(calls) == 2:
            worker.stop()
        elif first_poll_fails:
            raise OSError('serial timeout')
    worker = Worker(poll)
    monkeypatch.setattr(worker, 'msleep', lambda *_: None)
    worker.run()
    assert len(calls) == 2
    assert ('Laser status poll failed' in caplog.text) == first_poll_fails
