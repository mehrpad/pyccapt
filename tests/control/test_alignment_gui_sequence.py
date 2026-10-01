"""Offscreen GUI tests with simulated stage and experiment launch only."""
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest

pytest.importorskip('cv2')
pytest.importorskip('nidaqmx')
from PyQt6 import QtCore, QtWidgets

from pyccapt.control.core import runtime, share_variables
from pyccapt.control.devices.alignment_stage import AlignmentStageService
from pyccapt.control.gui.gui_main import Ui_PyCCAPT


class Motor:
    def __init__(self):
        self.position = dict(x=0., y=0., z=0.)
        self.moves = []

    def get_position(self):
        return self.position.copy()

    def is_moving(self):
        return False

    def validate_alignment_state(self):
        pass

    def stop(self):
        pass

    def move_absolute(self, **kwargs):
        self.moves.append(kwargs)
        for axis in 'xyz':
            if kwargs[axis+'_m'] is not None:
                self.position[axis] = kwargs[axis+'_m']


@pytest.fixture
def gui(monkeypatch, tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    conf, _ = runtime.load_project_config(change_cwd=False)
    conf.update(alignment_motion_calibrated=True, alignment_bounds_mm=[[-1, 1]]*3,
                alignment_xy_range_um=[1., 1.], alignment_xy_jacobian_mm_per_um=[[1., 0.], [0., 1.]],
                alignment_z_direction=1, alignment_transfer_z_mm=-.5, v_dc='on', tdc='on')
    variables = share_variables.Variables(conf, SimpleNamespace())
    variables.sample_rough_positions = {1: (0., 0., 0.), 2: (.0002, .0001, 0.)}
    monkeypatch.setattr(Ui_PyCCAPT, 'wins_init', lambda self: None)
    from pyccapt.control.gui.alignment_plot import AlignmentPlotWindow
    # Exercise the real plot widget/data path without needing an offscreen GL context.
    monkeypatch.setattr(AlignmentPlotWindow, 'showNormal', lambda self: None)
    counter = tmp_path/'counter.txt'; counter.write_text('1')
    monkeypatch.setattr(runtime, 'ensure_counter_file', lambda: counter)
    ui = Ui_PyCCAPT(variables, conf, None, None, None, None)
    window = QtWidgets.QMainWindow()
    ui.setupUi(window)
    for timer in window.findChildren(QtCore.QTimer):
        timer.stop()
    for value in vars(ui).values():
        if isinstance(value, QtCore.QTimer):
            value.stop()
    motor = Motor()
    service = AlignmentStageService(variables, lambda: motor)
    ui.gui_stage_control = SimpleNamespace(stage_device=motor, _reference_worker=None, alignment_service=service)
    ui.errors = []
    ui.error_message = ui.errors.append
    ui.automatic_alignment_button.setChecked(True)
    ui.variables.electrode_out = False
    ui.started = []
    def start():
        ui.started.append((variables.alignment_sample, motor.position.copy(), variables.ex_name))
        variables.start_flag = True
        return True
    ui.start_experiment_worker = start
    clock = [0.]
    monkeypatch.setattr('time.monotonic', lambda: clock[0])
    def tick():
        clock[0] += .5
        service.tick(now=clock[0])
        ui._alignment_batch_tick()
    yield ui, variables, motor, tick
    if ui._alignment_plot_window is not None:
        ui._alignment_plot_window.set_monitoring(False)
    window.close()
    for q in (ui.camera_command_queue, ui.visualization_command_queue, ui.experiment_command_queue,
              ui.experiment_status_queue, ui.experiment_completion_queue):
        q.close()


def test_editable_voltage_fields_and_positioning_precede_launch(gui):
    ui, v, motor, tick = gui
    assert ui.alignment_start_voltage.value() == 1500
    assert ui.alignment_voltage_increment.value() == 200
    ui.alignment_voltage_increment.setValue(125)
    ui._start_alignment_batch((1, 2))
    assert not ui.errors
    assert not ui.started
    assert not ui.alignment_voltage_increment.isEnabled()
    for _ in range(30):
        tick()
        if ui.started:
            break
    assert ui.started == [(1, dict(x=0., y=0., z=0.), 'test')]
    assert len(motor.moves) == 3
    assert all(move['velocity_m_s'] == pytest.approx(300e-6) for move in motor.moves)
    assert v.alignment_settings['voltage_increment'] == 125
    assert len(v.alignment_transfer_log) == 3
    assert all(move['speed_um_s'] == 300. for move in v.alignment_transfer_log)


def test_combined_alignment_launches_in_laser_mode_with_shared_voltage_ceiling(gui):
    ui, v, motor, tick = gui
    ui.conf.update(laser_alignment_calibrated=True, laser_alignment_bounds_mm=[-1., 1.]*3,
                   laser_alignment_travel_um=[30., 30., 10.], laser_alignment_max_voltage=3000.)
    v.laser_stage_snapshot = (0., 0., 0., 0.)
    v.laser_alignment_enabled = True
    v.pulse_mode = 'Laser'
    ui._start_alignment_batch((1,))
    assert not ui.errors
    assert v.alignment_settings['max_voltage'] == 3000.
    for _ in range(30):
        tick()
        if ui.started:
            break
    assert ui.started and v.pulse_mode == 'Laser'
    assert v.automatic_alignment_enabled


@pytest.mark.parametrize('invalid', ['Voltage mode', 'uncalibrated laser', 'start voltage above laser ceiling'])
def test_combined_alignment_validates_laser_before_sample_transfer(gui, invalid):
    ui, v, motor, tick = gui
    ui.conf.update(laser_alignment_calibrated=True, laser_alignment_bounds_mm=[-1., 1.]*3,
                   laser_alignment_travel_um=[30., 30., 10.])
    v.laser_stage_snapshot = (0., 0., 0., 0.)
    v.laser_alignment_enabled = True
    v.pulse_mode = 'Laser'
    if invalid == 'Voltage mode':
        v.pulse_mode = 'Voltage'
    elif invalid == 'uncalibrated laser':
        ui.conf['laser_alignment_calibrated'] = False
    else:
        ui.conf['laser_alignment_max_voltage'] = 1000.
    ui._start_alignment_batch((1,))
    assert ui.errors and not motor.moves and not ui.started


def test_single_start_completes_retract_xy_and_saved_z_before_experiment(gui):
    ui, v, motor, tick = gui
    motor.position.update(x=0., y=0., z=.0008)
    v.sample_rough_positions = {1: (.0002, .0001, .0006)}
    ui._start_alignment_batch((1,))
    for _ in range(30):
        tick()
        if ui.started:
            break
    assert not ui.errors
    assert len(ui.started) == 1
    assert len(motor.moves) == 3
    assert motor.moves[0]['z_m'] == -.0005
    assert motor.moves[1]['x_m'] == .0002
    assert motor.moves[1]['y_m'] == .0001
    assert motor.moves[1]['z_m'] is None
    assert motor.moves[2]['z_m'] == .0006
    assert ui.started[0][1] == dict(x=.0002, y=.0001, z=.0006)


def test_second_sample_waits_for_full_visualization_cleanup(gui):
    ui, v, motor, tick = gui
    ui._start_alignment_batch((1, 2))
    for _ in range(30):
        tick()
        if ui.started:
            break
    v.start_flag = False
    v.hardware_safe = True
    v.alignment_outcome = 'voltage_limit'
    v.last_screen_shot = True
    ui._alignment_sample_finished('')
    for _ in range(4):
        tick()
    assert len(motor.moves) == 3
    assert len(ui.started) == 1
    v.last_screen_shot = False
    for _ in range(30):
        tick()
        if len(ui.started) == 2:
            break
    assert ui.started[1] == (2, dict(x=.0002, y=.0001, z=0.), 'test')
    assert v.alignment_outcome == ''
    assert v.alignment_events == {}


@pytest.mark.parametrize('reason,error,safe', [('attempt_limit','',True), ('fault','',True),
                                             ('aligned','Detector failed',True), ('aligned','',False)])
def test_failed_sample_does_not_launch_next(gui, reason, error, safe):
    ui, v, motor, tick = gui
    ui._start_alignment_batch((1, 2))
    for _ in range(30):
        tick()
        if ui.started:
            break
    v.start_flag = False
    v.hardware_safe = safe
    v.alignment_outcome = reason
    ui._alignment_sample_finished(error)
    for _ in range(5):
        tick()
    assert len(ui.started) == 1
    assert not ui._alignment_batch


def test_stop_during_initial_positioning_never_starts_experiment(gui):
    ui, v, motor, tick = gui
    ui._start_alignment_batch((1, 2))
    tick()
    ui.stop_experiment_clicked()
    for _ in range(5):
        tick()
    assert not ui.started
    assert not ui._alignment_batch
    assert not v.sample_selection_locked


def test_missing_calibration_blocks_before_any_move(gui):
    ui, v, motor, tick = gui
    ui.conf['alignment_motion_calibrated'] = False
    ui._start_alignment_batch((1, 2))
    assert 'calibrated' in ui.errors[-1]
    assert not motor.moves
    assert not ui.started


def test_textline_parameters_map_to_saved_samples(gui):
    ui, v, motor, tick = gui
    ui.parameters_source.setCurrentText('TextLine')
    ui.alignment_start_voltage.setValue(2700)
    ui._start_alignment_batch((1, 2))
    assert not ui.errors
    for _ in range(30):
        tick()
        if ui.started:
            break
    assert ui.started[0][2] == 'test1'
    v.start_flag = False
    v.alignment_outcome = 'aligned'
    v.hardware_safe = True
    ui._alignment_sample_finished('')
    for _ in range(30):
        tick()
        if len(ui.started) == 2:
            break
    assert ui.started[1][2] == 'test2'
    assert v.detection_rate == 2
    assert v.vdc_max == 3000


def test_completion_handler_waits_for_worker_exit_before_advancing(gui, tmp_path):
    from unittest.mock import Mock
    ui, v, motor, tick = gui
    ui._start_alignment_batch((1, 2))
    for _ in range(30):
        tick()
        if ui.started:
            break
    v.flag_end_experiment = True
    v.hardware_safe = True
    v.alignment_outcome = 'voltage_limit'
    v.last_screen_shot = True
    v.path = str(tmp_path)
    ui.experiment_process = Mock()
    ui.experiment_process.is_alive.return_value = True
    ui.on_stop_experiment_worker()
    assert ui._alignment_batch_index == 0
    assert v.flag_end_experiment
    ui.experiment_process.is_alive.return_value = False
    ui.on_stop_experiment_worker()
    assert ui._alignment_batch_index == 1
    assert ui._alignment_waiting_cleanup
    assert not v.flag_end_experiment
    assert not v.start_flag
    assert len(ui.started) == 1


def test_new_worker_does_not_inherit_previous_stop_or_status(gui, monkeypatch):
    import queue
    from unittest.mock import Mock
    from pyccapt.control.gui import gui_main
    ui, v, motor, tick = gui
    for name in ('experiment_command_queue', 'experiment_status_queue', 'experiment_completion_queue'):
        getattr(ui, name).close()
        channel = queue.Queue()
        channel.close = lambda: None
        channel.put('stale previous-run message')
        setattr(ui, name, channel)
    ui.latest_completion_ack = object()
    ui.latest_experiment_status = object()
    monkeypatch.setattr(gui_main.device_checks, 'collect_startup_device_issues', lambda *args, **kwargs: [])
    ui.process_coordinator = SimpleNamespace(start_experiment=Mock(return_value=SimpleNamespace(pid=99)))
    assert Ui_PyCCAPT.start_experiment_worker(ui)
    assert ui.experiment_command_queue.empty()
    assert ui.experiment_status_queue.empty()
    assert ui.experiment_completion_queue.empty()
    assert ui.latest_completion_ack is None
    assert ui.latest_experiment_status is None
    assert v.sample_selection_locked
    for button in (ui.start_button, ui.electrode_button, ui.flat_test_button, ui.automatic_alignment_button):
        assert not button.isEnabled()
    # Refreshing electrode display must not unlock a running experiment.
    ui._sync_electrode_controls()
    assert not ui.electrode_button.isEnabled()
    before = v.electrode_out
    ui.toggle_electrode()
    assert v.electrode_out == before


def test_alignment_plot_lifetime_and_sample_switch(gui):
    ui, v, motor, tick = gui
    window = ui._alignment_plot_window
    assert window.monitoring and window.timer.isActive()
    v.automatic_alignment_enabled = v.start_flag = True
    for sample in (1, 2):
        tick()
        v.alignment_plot_snapshot = dict(time=v.alignment_stage_heartbeat, sequence_id='batch', sample=sample,
            origin_m=(0., 0., 0.), position_m=(1e-6, 2e-6, 0.), rate_percent=.3,
            target_percent=1., voltage=1500., phase='coarse')
        window.refresh()
    assert set(window.history.samples) == {1, 2}
    assert window.sample_selector.currentData() == 2
    assert window.history.coordinates(2)[0][0].tolist() == [1., 2., .3]
    v.alignment_window_epoch = 'position-one'
    v.alignment_events = {'epoch': 'position-one', 'sequence': 2,
                          'points_mm': [[1., 2.], [3., 4.]]}
    window.refresh()
    assert window.hitmap_count.text() == '2'
    assert window.detector_hitmap is not None
    window.hitmap_reset.click()
    assert window.hitmap_count.text() == '0'
    window.refresh()
    assert window.hitmap_count.text() == '0'
    v.alignment_events = {'epoch': 'position-one', 'sequence': 3,
                          'points_mm': [[1., 2.], [3., 4.], [5., 6.]]}
    window.refresh()
    assert window.hitmap_count.text() == '1'
    v.alignment_window_epoch = 'position-two'
    window.refresh()
    assert window.hitmap_count.text() == '0'
    v.start_flag = v.automatic_alignment_enabled = False
    window.refresh()
    assert window.monitoring  # Remain open after experiment shutdown.
    ui.automatic_alignment_button.setChecked(False)
    assert not window.monitoring and not window.timer.isActive()


@pytest.mark.parametrize('flag', ['start_flag', 'sample_selection_locked', 'automatic_alignment_enabled'])
def test_camera_sample_buttons_lock_and_handler_rejects_changes(gui, flag):
    from pyccapt.control.gui.gui_cameras import Ui_Cameras_Alignment
    ui, v, motor, tick = gui
    camera = Ui_Cameras_Alignment.__new__(Ui_Cameras_Alignment)
    camera.variables = v
    camera.sample_buttons = {i: QtWidgets.QPushButton() for i in (1, 2, 3)}
    camera.saved_sample_positions = dict(v.sample_rough_positions)
    for button in camera.sample_buttons.values():
        button.setCheckable(True)
    setattr(v, flag, True)
    camera._refresh_sample_selection_lock()
    assert all(not button.isEnabled() for button in camera.sample_buttons.values())
    before = dict(v.sample_rough_positions)
    camera._save_sample_position(1)
    assert dict(v.sample_rough_positions) == before
    assert camera.sample_buttons[1].isChecked()
    setattr(v, flag, False)
    camera._refresh_sample_selection_lock()
    assert all(button.isEnabled() for button in camera.sample_buttons.values())


def _prepare_flat_test(ui, variables, fraction):
    ui.automatic_alignment_button.setChecked(False)
    ui.start_button.setEnabled(True)
    variables.flag_main_gate = False
    variables.stage_pos_updated_at = 0.0  # Fixture's monotonic clock starts at zero.
    for axis in 'xyz':
        setattr(variables, 'stage_pos_'+axis, float(ui.conf.get('stage_home_'+axis+'_mm', 0))*1e-3)
    ui.pulse_fraction.setText(str(fraction))
    variables.pulse_fraction = fraction


@pytest.mark.parametrize('outcome', ['success', 'stopped', 'error'])
def test_flat_test_restores_previous_fraction_after_worker_exit(gui, monkeypatch, tmp_path, outcome):
    from unittest.mock import Mock
    ui, v, motor, tick = gui
    _prepare_flat_test(ui, v, 23)
    ui.start_flat_test()
    assert ui._flat_test_running and v.flat_test_active
    assert ui.pulse_fraction.text() == '5' and v.pulse_fraction == 5
    ui.experiment_process = Mock()
    ui.experiment_process.is_alive.return_value = True
    v.flag_end_experiment = True
    ui.on_stop_experiment_worker()
    assert v.pulse_fraction == 5  # Do not change the setting while acquisition/finalization is alive.
    ui.experiment_process.is_alive.return_value = False
    v.path = str(tmp_path)
    v.flat_test_reached_max = outcome == 'success'
    v.flat_test_peak_rate = .1
    v.experiment_error = 'Detector failed' if outcome == 'error' else ''
    ui.success_message = Mock()
    monkeypatch.setattr(QtWidgets.QApplication, 'primaryScreen', lambda: Mock())
    ui.on_stop_experiment_worker()
    assert ui.pulse_fraction.text() == '23' and v.pulse_fraction == 23
    assert not ui._flat_test_running and not v.flat_test_active
    assert ui._flat_test_previous_pulse_fraction is None
    # A subsequent test must remember the new setting, not a previous test's snapshot.
    _prepare_flat_test(ui, v, 18)
    ui.start_flat_test()
    assert ui._flat_test_previous_pulse_fraction == ('18', 18)


@pytest.mark.parametrize('raises', [False, True])
def test_flat_test_restores_fraction_if_launch_fails(gui, raises):
    ui, v, motor, tick = gui
    _prepare_flat_test(ui, v, 17)
    def failed_start():
        if raises:
            raise RuntimeError('Launch failed')
        return False
    ui.start_experiment_worker = failed_start
    if raises:
        with pytest.raises(RuntimeError, match='Launch failed'):
            ui.start_flat_test()
    else:
        ui.start_flat_test()
    assert ui.pulse_fraction.text() == '17' and v.pulse_fraction == 17
    assert not ui._flat_test_running and not v.flat_test_active
