"""Offscreen GUI tests with simulated stage and experiment launch only."""
import os
from pathlib import Path
from types import SimpleNamespace
import json

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
    project_path = runtime.project_path
    monkeypatch.setattr(runtime, 'project_path', lambda *parts:
                        tmp_path.joinpath(*parts) if parts[:2] == ('data', 'alignment_sequences')
                        else project_path(*parts))
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


@pytest.mark.parametrize('legacy', [False, True])
def test_editable_voltage_fields_and_positioning_precede_launch(gui, monkeypatch, legacy):
    ui, v, motor, tick = gui
    if legacy:
        position_sample = ui._position_alignment_sample
        def publish_old_settings_then_position():
            # Reproduce the snapshot held by a GUI started before the update.
            v.alignment_settings = {**v.alignment_settings, 'entry_fraction': .3, 'loss_fraction': .1}
            position_sample()
        monkeypatch.setattr(ui, '_position_alignment_sample', publish_old_settings_then_position)
    assert ui.alignment_start_voltage.value() == 1500
    assert ui.alignment_voltage_increment.value() == 100
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


def test_z_retreat_completes_when_uncommanded_xy_drifts_by_50_nm(gui):
    """Reproduce the first-attempt stall using the October 2 transfer positions."""
    ui, v, motor, tick = gui
    ui.conf.update(alignment_transfer_z_mm=-4.,
                   alignment_bounds_mm=[[-2., 6.], [-3., 7.], [-10., 7.]])
    motor.position.update(x=.002448013562, y=.002579671224, z=.000728608972)
    target = (.002448071847, .002579713089, .000728608972)
    v.sample_rough_positions = {1: target}
    move = motor.move_absolute
    def with_sensor_drift(**kwargs):
        move(**kwargs)
        if len(motor.moves) == 1:
            motor.position['x'] -= 51.497e-9
            motor.position['y'] -= 47.054e-9
    motor.move_absolute = with_sensor_drift
    ui._start_alignment_batch((1,))
    for _ in range(30):
        tick()
        if ui.started:
            break
    assert not ui.errors
    assert len(ui.started) == 1
    assert len(motor.moves) == 3
    assert motor.moves[0]['x_m'] is None and motor.moves[0]['y_m'] is None
    assert motor.moves[0]['z_m'] == -.004
    assert ui.started[0][1] == dict(zip('xyz', target))


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


def test_stage_transfer_fault_is_shown_in_main_gui(gui):
    ui, v, motor, tick = gui
    def failing_move(**kwargs):
        raise RuntimeError('Simulated controller communication failure')
    motor.move_absolute = failing_move
    ui._start_alignment_batch((1,))
    tick()  # Publish the first transfer request.
    tick()  # The stage service faults and requests cancellation.
    assert ui.errors == ['Simulated controller communication failure']
    assert not ui.started and not ui._alignment_batch
    journal = Path(v.alignment_transfer_path)
    assert journal.is_file()  # Preserved even though no experiment was launched.
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    assert any(record['event'] == 'motion_error' and
               record['error'] == 'Simulated controller communication failure' for record in records)
    assert records[-1]['event'] == 'transfer_stopped'


def test_initial_transfer_journal_contains_settling_and_is_copied_into_dataset(gui, tmp_path):
    from pyccapt.control.core.alignment_diagnostics import copy_transfer_journal
    ui, v, motor, tick = gui
    ui._start_alignment_batch((1,))
    for _ in range(30):
        tick()
        if ui.started:
            break
    records = [json.loads(line) for line in Path(v.alignment_transfer_path).read_text().splitlines()]
    assert records[0]['event'] == 'transfer_start' and records[-1]['event'] == 'transfer_complete'
    settled = [item for item in records if item['event'] == 'motion_settled']
    assert len(settled) == 3
    assert all(item['settled_s'] >= .5 and 'error_um' in item for item in settled)
    assert any(item.get('wait_reason') == 'settling' for item in records)
    metadata = tmp_path/'dataset'/'meta_data'
    metadata.mkdir(parents=True)
    copy_transfer_journal(v, metadata)
    assert (metadata/'alignment_transfer.jsonl').read_bytes() == Path(v.alignment_transfer_path).read_bytes()


def test_dense_xy_centroid_is_distinguished_from_fitted_circle_in_hitmap(gui):
    ui, v, motor, tick = gui
    plot = ui._alignment_plot_window
    v.automatic_alignment_enabled = True
    v.alignment_status = {'footprint': dict(valid=True, model='density', centre_mm=(5., -4.), radius_mm=8.)}
    plot.redraw_hitmap()
    assert not plot.footprint_circle.isVisible()
    assert list(plot.density_centroid.getData()[0]) == [5.]
    v.alignment_status = {'footprint': dict(valid=True, model='circle', centre_mm=(5., -4.), radius_mm=8.)}
    plot.redraw_hitmap()
    assert plot.footprint_circle.isVisible()
    assert len(plot.density_centroid.getData()[0]) == 0


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


def test_plan_parameters_map_to_saved_samples(gui):
    ui, v, motor, tick = gui
    ui.load_experiment_plan(Path(__file__).resolve().parents[2] /
                            'pyccapt/files/experiment_plan.example.toml')
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


def test_toml_plan_maps_reordered_samples_and_publishes_each_row(gui):
    from pyccapt.control.core import experiment_plan
    ui, v, motor, tick = gui
    rows = experiment_plan.load_plan(Path(__file__).resolve().parents[2] /
                                    'pyccapt/files/experiment_plan.example.toml')
    ui.plan_items = rows[::-1]
    ui._refresh_plan_table()
    ui.parameters_source.setCurrentText('TOML Plan')
    ui._freeze_experiment_queue()
    ui.alignment_start_voltage.setValue(2700)
    ui._start_alignment_batch((2, 1))
    assert not ui.errors
    assert not ui.plan_buttons['Load TOML'].isEnabled()
    assert not ui.plan_buttons['Edit'].isEnabled()
    for _ in range(30):
        tick()
        if ui.started:
            break
    assert ui.started[0][0] == 2
    assert ui.started[0][2] == 'test2'
    assert v.experiment_plan_snapshot['experiment']['sample_id'] == 2
    v.start_flag = False
    v.alignment_outcome = 'aligned'
    v.hardware_safe = True
    ui._alignment_sample_finished('')
    for _ in range(30):
        tick()
        if len(ui.started) == 2:
            break
    assert ui.started[1][0] == 1
    assert v.experiment_plan_snapshot['queue_index'] == 2
    assert v.experiment_plan_snapshot['experiment']['sample_id'] == 1


def test_plan_load_failure_preserves_queue_and_plan_is_frozen_at_start(gui, tmp_path):
    ui, v, motor, tick = gui
    path = Path(__file__).resolve().parents[2] / 'pyccapt/files/experiment_plan.example.toml'
    ui.load_experiment_plan(path)
    assert ui.parameters_source.currentText() == 'TOML Plan'
    assert ui.plan_table.rowCount() == 2
    invalid = tmp_path/'invalid.toml'
    invalid.write_text('schema_version = 2')
    with pytest.raises(ValueError):
        ui.load_experiment_plan(invalid)
    assert ui.plan_items[0]['ex_name'] == 'test1'
    ui._freeze_experiment_queue()
    ui.plan_items[1]['ex_name'] = 'changed after start'
    v.experiment_plan_index = 1
    ui.apply_current_plan_experiment()
    assert v.ex_name == 'test2'
    assert v.experiment_plan_snapshot['queue_index'] == 2
    v.start_flag = True
    with pytest.raises(ValueError, match='Wait'):
        ui.load_experiment_plan(path)


def test_queue_edit_duplicate_reorder_remove_and_save(gui, monkeypatch, tmp_path):
    from pyccapt.control.core import experiment_plan
    ui, v, motor, tick = gui
    path = Path(__file__).resolve().parents[2] / 'pyccapt/files/experiment_plan.example.toml'
    ui.load_experiment_plan(path)
    edited = dict(ui.plan_items[0], ex_name='edited')
    monkeypatch.setattr(ui, '_edit_plan_item', lambda item: edited)
    ui._edit_plan_row()
    ui._duplicate_plan_row()
    assert [row['ex_name'] for row in ui.plan_items] == ['edited', 'edited', 'test2']
    ui._move_plan_row(1)
    assert [row['ex_name'] for row in ui.plan_items] == ['edited', 'test2', 'edited']
    ui._remove_plan_row()
    target = tmp_path/'saved.toml'
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName', lambda *args: (str(target), ''))
    ui._save_plan_dialog()
    assert experiment_plan.load_plan(target) == ui.plan_items
    ui._add_plan_row()
    assert len(ui.plan_items) == 3
    assert ui.plan_items[-1]['ex_name'] == 'edited'


def test_plan_missing_position_or_invalid_second_row_blocks_before_launch(gui, monkeypatch):
    ui, v, motor, tick = gui
    ui.load_experiment_plan(Path(__file__).resolve().parents[2] /
                            'pyccapt/files/experiment_plan.example.toml')
    monkeypatch.setattr(ui, '_confirm_warning_dialog', lambda *args: True)
    v.sample_rough_positions = {1: (0., 0., 0.)}
    ui.start_experiment_clicked()
    assert 'saved Cameras' in ui.errors[-1]
    assert not motor.moves and not ui.started
    ui.plan_items[1]['vdc_max'] = ui.conf['max_vdc']+1
    ui.start_experiment_clicked()
    assert 'Experiment 2' in ui.errors[-1]
    assert not motor.moves and not ui.started


def test_plan_runs_without_stage_mapping_when_alignment_disabled(gui, monkeypatch):
    from pyccapt.control.core import experiment_plan
    ui, v, motor, tick = gui
    ui.automatic_alignment_button.setChecked(False)
    ui.plan_items = experiment_plan.load_plan(Path(__file__).resolve().parents[2] /
                                             'pyccapt/files/experiment_plan.example.toml')
    del ui.plan_items[0]['sample_id']
    del ui.plan_items[1]['sample_id']
    ui._refresh_plan_table()
    ui.parameters_source.setCurrentText('TOML Plan')
    monkeypatch.setattr(ui, '_confirm_start_parameter_warnings', lambda: True)
    ui.start_experiment_clicked()
    assert ui.started[0][2] == 'test1'
    assert not motor.moves
    assert v.experiment_plan_snapshot['experiment']['ex_name'] == 'test1'


def test_experiment_editor_preserves_types_and_units(gui):
    from pyccapt.control.core import experiment_plan
    from pyccapt.control.gui.experiment_plan_gui import ExperimentEditor
    ui, v, motor, tick = gui
    item = experiment_plan.load_plan(Path(__file__).resolve().parents[2] /
                                     'pyccapt/files/experiment_plan.example.toml')[0]
    editor = ExperimentEditor(item, ui.centralwidget)
    assert editor.item() == item
    editor.fields['sample_id'].setCurrentIndex(0)
    editor.fields['criteria_time'].setChecked(False)
    editor.fields['criteria_ions'].setChecked(True)
    editor.fields['vdc_steps_up'].setText('0.25')
    edited = editor.item()
    assert 'sample_id' not in edited
    assert edited['vdc_steps_up'] == .25
    assert edited['criteria_ions'] is True
    editor.close()


@pytest.mark.parametrize('outcome', ['clean', 'stop', 'failed', 'unsafe'])
def test_plan_queue_continues_only_after_clean_worker_exit(gui, tmp_path, outcome):
    from unittest.mock import Mock
    ui, v, motor, tick = gui
    ui.automatic_alignment_button.setChecked(False)
    ui.load_experiment_plan(Path(__file__).resolve().parents[2] /
                            'pyccapt/files/experiment_plan.example.toml')
    ui._freeze_experiment_queue()
    v.experiment_plan_index = 0
    ui.apply_current_plan_experiment()
    ui.start_experiment_worker()
    v.sample_selection_locked = True
    v.flag_end_experiment = True
    v.hardware_safe = outcome != 'unsafe'
    v.experiment_error = 'worker failed' if outcome == 'failed' else ''
    ui._operator_stopped = outcome == 'stop'
    v.path = str(tmp_path)
    ui.experiment_process = Mock()
    ui.experiment_process.is_alive.return_value = True
    ui.on_stop_experiment_worker()
    assert len(ui.started) == 1
    assert v.flag_end_experiment
    ui.experiment_process.is_alive.return_value = False
    ui.on_stop_experiment_worker()
    if outcome == 'clean':
        assert len(ui.started) == 2
        assert ui.started[-1][2] == 'test2'
        assert v.experiment_plan_snapshot['queue_index'] == 2
        # The second experiment finishes the queue without a third launch.
        v.flag_end_experiment = True
        ui.on_stop_experiment_worker()
        assert len(ui.started) == 2
    else:
        assert len(ui.started) == 1
    assert ui._batch_items is None
    assert v.experiment_plan_index == 0
    assert ui.plan_buttons['Load TOML'].isEnabled()
    assert ui.plan_buttons['Edit'].isEnabled()
    assert not motor.moves


def test_single_run_form_unlocks_after_worker_finishes(gui, tmp_path):
    from unittest.mock import Mock
    ui, v, motor, tick = gui
    ui.automatic_alignment_button.setChecked(False)
    ui.start_experiment_worker()
    v.sample_selection_locked = True
    v.flag_end_experiment = True
    v.hardware_safe = True
    v.path = str(tmp_path)
    ui.experiment_process = Mock()
    ui.experiment_process.is_alive.return_value = False
    ui.on_stop_experiment_worker()
    assert not v.sample_selection_locked
    assert ui.vdc_max.isEnabled()
    assert ui.parameters_source.isEnabled()


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
    ui.advanced_settings_button.click()
    assert ui.advanced_settings_dialog.isVisible()
    assert not ui.counter_source.isEnabled()
    assert not ui.ex_freq.isEnabled()
    assert ui.control_algorithm.isEnabled()
    ui.advanced_settings_dialog.close()
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
