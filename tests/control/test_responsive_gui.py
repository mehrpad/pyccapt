"""Screen fitting and readable controls, using offscreen Qt and no hardware."""
import os
import gc
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PyQt6 import QtCore, QtGui, QtWidgets

from pyccapt.control.core import runtime, share_variables
from pyccapt.control.gui import (
    gui_baking, gui_cameras, gui_gates, gui_laser_control, gui_main,
    gui_pumps_vacuum, gui_stage_control, gui_visualization,
)
from pyccapt.control.gui.alignment_plot import AlignmentPlotWindow
from pyccapt.control.gui.responsive import ResponsiveWindow, make_window_responsive


@pytest.fixture
def app(monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    # The offscreen Windows plugin has no system font discovery.
    if os.name == 'nt':
        from pathlib import Path
        fonts = Path(os.environ['WINDIR']) / 'Fonts'
        QtGui.QFontDatabase.addApplicationFont(str(fonts / 'segoeui.ttf'))
        QtGui.QFontDatabase.addApplicationFont(str(fonts / 'seguisym.ttf'))
        QtGui.QFontDatabase.addApplicationFont(str(fonts / 'arial.ttf'))
        app.setFont(QtGui.QFont('Segoe UI', 9))
    app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    gc.collect()
    # A virtual full-size work area, independent of the CI machine's screen.
    def fit(controller):
        controller._pending = False
        controller.fit_to_screen(QtCore.QRect(0, 0, 1920, 1040))
    monkeypatch.setattr(ResponsiveWindow, '_fit_current_screen', fit)
    return app


@pytest.mark.parametrize('main_window', [False, True])
def test_scroll_fallback_preserves_layout_controls_and_monitor_bounds(app, main_window):
    window = QtWidgets.QMainWindow() if main_window else QtWidgets.QWidget()
    content = QtWidgets.QWidget() if main_window else window
    if main_window:
        window.setCentralWidget(content)
        window.menuBar().addMenu('File')
    layout = QtWidgets.QVBoxLayout(content)
    panel = QtWidgets.QWidget()
    panel.setMinimumSize(950, 600)
    layout.addWidget(panel)
    stop = QtWidgets.QPushButton('Stop')
    stop.setFixedSize(140, 28)
    layout.addWidget(stop)
    window.resize(1100, 750)
    make_window_responsive(window)
    controller = window._responsive_window
    make_window_responsive(window)
    assert controller is window._responsive_window
    window.show()
    app.processEvents()
    assert window.size() == QtCore.QSize(1100, 750)
    assert controller.scroll.verticalScrollBar().maximum() == 0
    font = stop.font()
    # Negative desktop coordinates represent a smaller secondary monitor.
    area = QtCore.QRect(-1280, 0, 960, 540)
    controller.fit_to_screen(area)
    app.processEvents()
    assert area.contains(window.frameGeometry())
    assert controller.scroll.horizontalScrollBar().maximum() > 0
    assert controller.scroll.verticalScrollBar().maximum() > 0
    assert stop.size() == QtCore.QSize(140, 28)
    assert stop.font() == font
    controller.scroll.ensureWidgetVisible(stop)
    app.processEvents()
    point = stop.mapTo(controller.scroll.viewport(), QtCore.QPoint())
    assert controller.scroll.viewport().rect().intersects(QtCore.QRect(point, stop.size()))
    window.resize(1100, 750)
    app.processEvents()
    assert controller.scroll.horizontalScrollBar().maximum() == 0
    assert controller.scroll.verticalScrollBar().maximum() == 0
    window.close()


def test_embedded_panel_uses_parent_viewport(app):
    parent = QtWidgets.QWidget()
    child = QtWidgets.QWidget(parent)
    QtWidgets.QVBoxLayout(child).addWidget(QtWidgets.QPushButton('Gate'))
    make_window_responsive(child)
    assert not hasattr(child, '_responsive_window')
    parent.close()


class WebPlot(QtWidgets.QWidget):
    """WebEngine stand-in with the same Qt geometry, without a browser process."""
    loadFinished = QtCore.pyqtSignal(bool)

    def setHtml(self, *_args, **_kwargs):
        pass

    def load(self, *_args):
        pass

    def page(self):
        return self

    def runJavaScript(self, *_args):
        pass


@pytest.fixture
def instrument_window(request, app, monkeypatch, tmp_path):
    conf, _ = runtime.load_project_config(change_cwd=False)
    conf.update(gauges='off', camera='off', stage='off', laser='off')
    variables = share_variables.Variables(conf, SimpleNamespace())
    monkeypatch.setattr(gui_main.Ui_PyCCAPT, 'wins_init', lambda self: None)
    monkeypatch.setattr(gui_stage_control.Ui_Stage_Control, '_connect_device', lambda self: None)
    monkeypatch.setattr(gui_laser_control.Ui_Laser_Control, '_open_laser_cli', lambda *args, **kwargs: None)
    monkeypatch.setattr(gui_laser_control.Ui_Laser_Control, '_connect_stage_device', lambda self: None)
    monkeypatch.setattr(gui_cameras.Ui_Cameras_Alignment, '_initialise_illumination', lambda self: None)
    monkeypatch.setattr(gui_cameras.Ui_Cameras_Alignment, 'initialize_camera_thread', lambda self: None)
    monkeypatch.setattr(gui_visualization.Ui_Visualization, '_start_live_calibration_worker', lambda self: None)
    monkeypatch.setattr(gui_baking.Ui_Baking, 'read', lambda self: None)
    monkeypatch.setattr(gui_pumps_vacuum, 'QWebEngineView', WebPlot)
    monkeypatch.setattr(gui_pumps_vacuum.Ui_Pumps_Vacuum, '_set_vent_valve', lambda *args: None)
    counter = tmp_path / 'counter.txt'
    counter.write_text('1')
    monkeypatch.setattr(runtime, 'ensure_counter_file', lambda: counter)
    project_path = runtime.project_path
    monkeypatch.setattr(runtime, 'project_path', lambda *parts:
                        tmp_path.joinpath(*parts) if parts[:3] == ('files', 'logs', 'baking')
                        else project_path(*parts))
    name = request.param
    window = QtWidgets.QMainWindow() if name == 'main' else QtWidgets.QWidget()
    if name == 'main':
        ui = gui_main.Ui_PyCCAPT(variables, conf, None, None, None, None)
    elif name == 'stage':
        ui = gui_stage_control.Ui_Stage_Control(variables, conf)
    elif name == 'laser':
        ui = gui_laser_control.Ui_Laser_Control(variables, conf)
    elif name == 'cameras':
        ui = gui_cameras.Ui_Cameras_Alignment(variables, conf, gui_cameras.SignalEmitter())
    elif name == 'visualization':
        ui = gui_visualization.Ui_Visualization(variables, conf, [], [], [], [])
    elif name == 'baking':
        ui = gui_baking.Ui_Baking(variables, conf, gui_pumps_vacuum.SignalEmitter())
    elif name in ('vacuum', 'combined'):
        ui = gui_pumps_vacuum.Ui_Pumps_Vacuum(variables, conf, gui_pumps_vacuum.SignalEmitter())
    elif name == 'gates':
        ui = gui_gates.Ui_Gates(variables, conf)
    else:
        window = AlignmentPlotWindow(variables)
        ui = window
    if ui is not window:
        ui.setupUi(window)
    if name == 'combined':
        gates = QtWidgets.QWidget(window)
        gates_ui = gui_gates.Ui_Gates(variables, conf, parent=window)
        gates_ui.setupUi(gates)
        gates_ui.attach_load_lock_temperature_controls(ui)
        ui.gridLayout_9.addWidget(gates, 0, 1)
        window.resize(1280, 640)
        gates_ui.diagram_timer.stop()
    for timer in window.findChildren(QtCore.QTimer):
        timer.stop()
    for value in vars(ui).values():
        if isinstance(value, QtCore.QTimer):
            value.stop()
    yield window, ui, app
    if name == 'main':
        app.aboutToQuit.disconnect(ui.cleanup)
        for queue in (ui.camera_command_queue, ui.visualization_command_queue,
                      ui.experiment_command_queue, ui.experiment_status_queue,
                      ui.experiment_completion_queue):
            queue.close()
    window.close()
    # Unregister pyqtgraph views before Qt deletes their context menus.
    import pyqtgraph as pg
    for view in window.findChildren(pg.ViewBox):
        view.unregister()
    window.deleteLater()
    app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


@pytest.mark.parametrize('instrument_window', ['main'], indirect=True)
def test_main_plan_layout_is_compact_and_queue_fits(instrument_window, tmp_path):
    from pathlib import Path
    window, ui, app = instrument_window
    window.show()
    for _ in range(5):
        app.processEvents()
    assert window.width() == 760
    assert window.height() == 670
    assert [ui.parameters_source.itemText(index) for index in
            range(ui.parameters_source.count())] == ['TextBox', 'TOML Plan']
    assert ui.plan_panel.isHidden()
    assert window._responsive_window.scroll.verticalScrollBar().maximum() == 0
    window.grab().save(str(tmp_path/'single-form.png'))
    ui.load_experiment_plan(Path(__file__).resolve().parents[2] /
                            'pyccapt/files/experiment_plan.example.toml')
    for _ in range(5):
        app.processEvents()
    assert ui.plan_panel.isVisible()
    assert ui.advanced_settings_label.isHidden()
    assert ui.advanced_settings_button.isHidden()
    assert ui.run_controls_panel.isVisible()
    assert ui.experiment_actions_panel.isVisible()
    assert ui.electrode_button.isVisible()
    assert ui.stop_button.isVisible()
    assert window._responsive_window.scroll.viewport().rect().contains(
        QtCore.QRect(ui.run_controls_panel.mapTo(window._responsive_window.scroll.viewport(), QtCore.QPoint()),
                     ui.run_controls_panel.size()))
    assert window._responsive_window.scroll.horizontalScrollBar().maximum() == 0
    assert window._responsive_window.scroll.verticalScrollBar().maximum() == 0
    assert window._responsive_window.scroll.viewport().rect().contains(
        QtCore.QRect(ui.plan_panel.mapTo(window._responsive_window.scroll.viewport(), QtCore.QPoint()),
                     ui.plan_panel.size()))
    window.grab().save(str(tmp_path/'plan-queue.png'))
    print(f'GUI previews: {tmp_path}')


@pytest.mark.parametrize('instrument_window', ['main'], indirect=True)
def test_main_advanced_dialog_and_run_controls_layout(instrument_window, tmp_path):
    window, ui, app = instrument_window
    window.show()
    for _ in range(5):
        app.processEvents()
    advanced = (ui.counter_source, ui.control_algorithm, ui.ex_freq, ui.vp_min,
                ui.vp_max, ui.vdc_steps_up, ui.vdc_steps_down)
    assert all(widget.window() is ui.advanced_settings_dialog for widget in advanced)
    assert not ui.advanced_settings_dialog.isVisible()
    assert ui.line_3.isHidden() and ui.line_4.isHidden()
    def position(widget):
        return widget.mapTo(ui.centralwidget, QtCore.QPoint())
    assert ui.advanced_settings_label.text() == 'Advanced settings'
    assert position(ui.detection_rate_init).y() < position(ui.advanced_settings_label).y()
    assert position(ui.label_188).x() == position(ui.advanced_settings_label).x()
    assert position(ui.detection_rate_init).x() == position(ui.advanced_settings_button).x()
    assert position(ui.advanced_settings_label).x() < position(ui.advanced_settings_button).x()
    assert position(ui.advanced_settings_label).y() == position(ui.advanced_settings_button).y()
    assert ui.statistics_separator.isVisible()
    assert ui.statistics_separator.frameShape() == QtWidgets.QFrame.Shape.HLine
    assert position(ui.detection_rate).y() + ui.detection_rate.height() <= position(ui.statistics_separator).y()
    assert position(ui.statistics_separator).y() < position(ui.electrode_button).y()
    assert position(ui.electrode_button).x() < position(ui.flat_test_button).x()
    assert position(ui.electrode_button).y() == position(ui.flat_test_button).y()
    assert ui.electrode_separator.isVisible()
    assert ui.electrode_separator.frameShape() == QtWidgets.QFrame.Shape.HLine
    assert position(ui.electrode_button).y() + ui.electrode_button.height() <= position(ui.electrode_separator).y()
    assert position(ui.electrode_separator).y() < position(ui.alignment_group).y()
    assert ui.alignment_group.title() == 'Auto Alignment'
    for widget in (ui.alignment_start_voltage_label, ui.alignment_start_voltage,
                   ui.alignment_voltage_increment_label, ui.alignment_voltage_increment,
                   ui.automatic_alignment_button):
        assert ui.alignment_group.isAncestorOf(widget)
    assert position(ui.alignment_start_voltage).y() < position(ui.alignment_voltage_increment).y()
    assert position(ui.electrode_button).y() < position(ui.alignment_start_voltage).y()
    assert position(ui.alignment_voltage_increment).y() < position(ui.automatic_alignment_button).y()
    assert position(ui.start_button).y() + ui.start_button.height() < position(ui.stop_button).y()
    assert position(ui.start_button).x() == position(ui.stop_button).x()
    assert position(ui.start_button).y() >= position(ui.alignment_group).y() + ui.alignment_group.height() + 8
    assert position(ui.start_button).y() > position(ui.advanced_settings_button).y()
    assert position(ui.stop_button).y() + ui.stop_button.height() < position(ui.run_controls_separator).y()
    ui.advanced_settings_button.click()
    for _ in range(3):
        app.processEvents()
    assert ui.advanced_settings_dialog.isVisible()
    assert all(widget.isVisible() for widget in advanced)
    ui.vdc_steps_up.setText('0.25')
    ui.vdc_steps_up.editingFinished.emit()
    assert ui.variables.vdc_step_up == .25
    ui.advanced_settings_dialog.grab().save(str(tmp_path/'advanced-settings.png'))
    ui.advanced_settings_dialog.close()
    ui.advanced_settings_button.click()
    assert ui.vdc_steps_up.text() == '0.25'
    ui.parameters_source.setCurrentText('TOML Plan')
    assert not ui.advanced_settings_dialog.isVisible()
    assert not ui.advanced_settings_button.isEnabled()
    ui.parameters_source.setCurrentText('TextBox')
    assert ui.advanced_settings_button.isEnabled()
    print(f'Advanced dialog preview: {tmp_path}')


@pytest.mark.parametrize('instrument_window', ['vacuum', 'combined'], indirect=True)
def test_pump_groups_keep_displays_and_controls_visible(instrument_window, tmp_path):
    window, ui, app = instrument_window
    window.show()
    for _ in range(5):
        app.processEvents()
    combined = isinstance(ui.temp_ll.parentWidget(), QtWidgets.QGroupBox)
    assert window.width() <= (1280 if combined else 840)
    assert window.height() <= (640 if combined else 720)
    gauges = (ui.vacuum_buffer, ui.vacuum_buffer_back, ui.vacuum_load_lock,
              ui.vacuum_load_lock_back, ui.vacuum_cryo_load_lock, ui.vacuum_cryo_load_lock_back)
    for lcd in gauges:
        assert ui.vacuum_gauges_group.isAncestorOf(lcd)
        assert lcd.size() == QtCore.QSize(150, 50)
    for widget in (ui.temp_stage, ui.temp_cryo_head, ui.temp_cryo_head_inside,
                   ui.set_temperature_cryo, ui.target_tempreature_cryo):
        assert ui.cryo_temperature_group.isAncestorOf(widget)
    for button in (ui.pump_cryo_load_lock_switch, ui.vent_cryo_load_lock_partial_switch,
                   ui.pump_load_lock_switch):
        assert ui.venting_group.isAncestorOf(button)
    ui.emitter.temp_stage.emit(42.5)
    ui.emitter.temp_cryo_head.emit(50.25)
    ui.emitter.temp_cryo_head_inside.emit(49.75)
    assert ui.temp_stage.value() == 42.5
    assert ui.temp_cryo_head.value() == 50.25
    assert ui.temp_cryo_head_inside.value() == 49.75
    assert not ui.pump_cryo_load_lock_switch.isEnabled()
    viewport = window._responsive_window.scroll.viewport()
    assert window._responsive_window.scroll.horizontalScrollBar().maximum() == 0
    assert window._responsive_window.scroll.verticalScrollBar().maximum() == 0
    for widget in window.findChildren((QtWidgets.QLCDNumber, QtWidgets.QPushButton, QtWidgets.QSpinBox)):
        assert widget.isVisible()
        assert viewport.rect().contains(QtCore.QRect(widget.mapTo(viewport, QtCore.QPoint()), widget.size()))
    window.grab().save(str(tmp_path/'pumps.png'))
    print(f'Pump GUI preview: {tmp_path}')


@pytest.mark.parametrize('instrument_window', ['stage'], indirect=True)
def test_stage_controls_fit_compact_window_and_keep_readable_values(instrument_window, tmp_path):
    window, ui, app = instrument_window
    window.show()
    for _ in range(5):
        app.processEvents()
    assert window.width() <= 880
    assert window.height() <= 220
    for lcd in (ui.stage_x_mm, ui.stage_x_um, ui.stage_x_nm,
                ui.stage_y_mm, ui.stage_y_um, ui.stage_y_nm,
                ui.stage_z_mm, ui.stage_z_um, ui.stage_z_nm):
        assert lcd.size() == QtCore.QSize(64, 28)
        assert lcd.digitCount() == 5
        assert lcd.isVisible()
    for selector, label in ((ui.stage_speed_x, ui.stage_speed_x_label),
                            (ui.stage_speed_y, ui.stage_speed_y_label),
                            (ui.stage_speed_z, ui.stage_speed_z_label)):
        for index in range(selector.count()):
            selector.setCurrentIndex(index)
            app.processEvents()
            assert selector.width() >= selector.fontMetrics().horizontalAdvance(selector.currentText()) + 42
            assert label.width() >= label.fontMetrics().horizontalAdvance(label.text())
        selector.setValue(ui._speed_default)
    viewport = window._responsive_window.scroll.viewport()
    assert window._responsive_window.scroll.horizontalScrollBar().maximum() == 0
    assert window._responsive_window.scroll.verticalScrollBar().maximum() == 0
    for button in window.findChildren(QtWidgets.QPushButton):
        assert button.isVisible()
        assert viewport.rect().contains(QtCore.QRect(button.mapTo(viewport, QtCore.QPoint()), button.size()))
    assert not ui.stage_reference.isEnabled()
    assert ui.stage_stop.text() == 'STOP'
    window.grab().save(str(tmp_path/'stage.png'))
    print(f'Stage GUI preview: {tmp_path}')


@pytest.mark.parametrize('instrument_window', ['cameras'], indirect=True)
@pytest.mark.parametrize('detected', [False, True])
def test_camera_compact_layout_preserves_views_and_all_controls(instrument_window, tmp_path, detected):
    window, ui, app = instrument_window
    if detected:
        ui.camera_list_empty_label.hide()
        for slot in range(3):
            entry = ui._make_camera_row({'model': 'a2A1920-160uc', 'serial': f'1234567{slot}',
                                         'slot': slot, 'attached': True, 'user_disabled': False})
            ui._camera_row_widgets[str(slot)] = entry
    window.show()
    for _ in range(8):
        app.processEvents()
    assert window.width() <= 900
    assert window.height() <= 750
    for view in (ui.cam_s_o, ui.cam_s_d, ui.cam_b_o, ui.cam_b_d, ui.cam_angle_o, ui.cam_angle_d):
        assert view.isVisible()
        assert view.width() >= view.minimumWidth()
        assert view.height() >= 160
    fields = (ui.exposure_time_cam_1, ui.exposure_time_cam_2, ui.exposure_time_cam_3)
    for field, slider in zip(fields, ui.exposure_sliders):
        assert field.size() == QtCore.QSize(90, 25)
        assert field.isVisible() and slider.isVisible()
        assert not field.isEnabled() and not slider.isEnabled()
    ui.emitter.cams_exposure_time_current.emit([2_000_000, 1_000_000, 500_000])
    assert [field.text() for field in fields] == ['2000000', '1000000', '500000']
    viewport = window._responsive_window.scroll.viewport()
    assert window._responsive_window.scroll.horizontalScrollBar().maximum() == 0
    assert window._responsive_window.scroll.verticalScrollBar().maximum() == 0
    controls = [ui.superuser, ui.light, ui.illumination_percent, ui.auto_exposure_time,
                ui.default_exposure_time, *fields, *ui.exposure_sliders,
                *ui.sample_buttons.values(), *ui.camera_monitor_lcds.values()]
    controls.extend(field for row in ui.sample_position_fields.values() for field in row)
    controls.extend(button for entry in ui._camera_row_widgets.values()
                    for button in (entry['connect_btn'], entry['disconnect_btn']))
    for widget in controls:
        assert widget.isVisible()
        assert viewport.rect().contains(QtCore.QRect(widget.mapTo(viewport, QtCore.QPoint()), widget.size()))
    for entry in ui._camera_row_widgets.values():
        label = entry['label']
        assert label.height() >= label.heightForWidth(label.width())
    window.grab().save(str(tmp_path/'cameras.png'))
    print(f'Camera GUI preview: {tmp_path}')


@pytest.mark.parametrize('instrument_window', ['laser'], indirect=True)
def test_laser_compact_layout_preserves_controls_and_readouts(instrument_window, tmp_path):
    window, ui, app = instrument_window
    # Exercise the persistent banner as well as the wrapped alignment status.
    ui.laser_connection_banner.setText('Laser unavailable: reconnect the configured CLI port.')
    ui.laser_connection_banner.show()
    if ui.alignment_plot.view is not None:
        ui.alignment_plot.tabs.setCurrentIndex(1)
    window.show()
    for _ in range(8):
        app.processEvents()
    print(f'Laser GUI size: {window.size()}, preview: {tmp_path}')
    assert window.width() <= 1000
    assert window.height() <= 700
    viewport = window._responsive_window.scroll.viewport()
    assert window._responsive_window.scroll.horizontalScrollBar().maximum() == 0
    assert window._responsive_window.scroll.verticalScrollBar().maximum() == 0
    controls = [ui.laser_wavelegnth, ui.laser_wavelegnth_nm_label, ui.laser_power,
                ui.laser_rate, ui.laser_divition_factor, ui.laser_enable, ui.laser_on,
                ui.laser_standby, ui.laser_listen, ui.laser_auto_alignment, ui.laser_tracking,
                *ui.laser_alignment_fields.values(), *ui.laser_alignment_buttons,
                ui.laser_alignment_stop, ui.laser_alignment_label, ui.alignment_plot,
                ui.laser_home, ui.laser_stage_reference, ui.laser_stage_stop, ui.laser_stage_superuser,
                ui.switch_to_cli_button, ui.nktpbus_mode_switch, ui.laser_connection_banner,
                ui.laser_up, ui.laser_down, ui.laser_left, ui.laser_right,
                ui.laser_forward, ui.laser_backward, ui.laser_power_disp,
                ui.laser_pulse_energy_disp, ui.laser_repetion_rate_disp]
    for axis in ('x', 'y', 'z'):
        controls.extend(getattr(ui, f'laser_{axis}_{unit}') for unit in ('mm', 'um', 'nm'))
        selector = getattr(ui, f'laser_speed_{axis}')
        label = getattr(ui, f'laser_speed_{axis}_label')
        controls.extend((selector, label))
        for index in range(selector.count()):
            selector.setCurrentIndex(index)
            app.processEvents()
            assert selector.width() >= selector.fontMetrics().horizontalAdvance(selector.currentText()) + 42
            assert label.width() >= label.fontMetrics().horizontalAdvance(label.text())
        selector.setValue(ui._speed_default)
    for widget in controls:
        assert widget.isVisible()
        assert viewport.rect().contains(QtCore.QRect(widget.mapTo(viewport, QtCore.QPoint()), widget.size()))
    for field in ui.laser_alignment_fields.values():
        field.setValue(field.maximum())
        assert field.width() >= field.fontMetrics().horizontalAdvance(field.text()) + 30
    assert not ui.laser_stage_reference.isEnabled()
    assert not ui.switch_to_cli_button.isEnabled()
    assert ui.label_9.text() == 'Selected output (W)'
    assert ui.label_10.text() == 'Pulse energy (µJ)'
    window.grab().save(str(tmp_path/'laser.png'))


@pytest.mark.parametrize('instrument_window', ['visualization'], indirect=True)
def test_visualization_compact_controls_and_equal_plot_dimensions(instrument_window, tmp_path):
    window, ui, app = instrument_window
    window.show()
    for _ in range(8):
        app.processEvents()
    print(f'Visualization GUI size: {window.size()}, preview: {tmp_path}')
    assert window.width() <= 1000
    assert window.height() <= 620
    viewport = window._responsive_window.scroll.viewport()
    assert window._responsive_window.scroll.horizontalScrollBar().maximum() == 0
    assert window._responsive_window.scroll.verticalScrollBar().maximum() == 0
    for widget in (ui.voltage, ui.detection_rate, ui.hitmap_count, ui.fdm_count, ui.dc_hold,
                   ui.set_dc_voltage, ui.set_dc_voltage_value, ui.detection_rate_range_switch,
                   ui.reset_heatmap_v, ui.hitmap_plot_size, ui.hit_displayed,
                   ui.fdm_last_events_switch, ui.fdm_max_ions, ui.experiment_status_led,
                   ui.experiment_status_text, ui.histogram, ui.btn_view_mc_cal, ui.btn_view_mc,
                   ui.btn_view_tof_cal, ui.btn_view_tof, ui.calib_status_label,
                   ui.spectrum_last_events_switch, ui.num_last_events, ui.max_mc, ui.max_tof):
        assert widget.isVisible()
        assert viewport.rect().contains(QtCore.QRect(widget.mapTo(viewport, QtCore.QPoint()), widget.size()))
    plots = (ui.vdc_time, ui.detection_rate_viz, ui.detector_heatmap, ui.detector_fdm)
    assert ui.data_line_vdc in ui.vdc_time.listDataItems()
    assert ui.data_line_dtec in ui.detection_rate_viz.listDataItems()
    window.grab().save(str(tmp_path/'visualization.png'))
    for width, height in ((981, 620), (1283, 720), (1601, 900), (800, 480), (980, 620)):
        window.resize(width, height)
        for _ in range(5):
            app.processEvents()
        assert len({(plot.width(), plot.height()) for plot in plots}) == 1
        assert all(plot.isVisible() and plot.width() >= 220 and plot.height() >= 220 for plot in plots)
        assert len({plot.y() for plot in plots}) == 1
        for left, right in zip(plots, plots[1:]):
            assert left.geometry().right() < right.geometry().left()


@pytest.mark.parametrize('instrument_window', [
    'main', 'stage', 'laser', 'cameras', 'visualization', 'baking',
    'vacuum', 'combined', 'gates', 'alignment',
], indirect=True)
def test_instrument_windows_fit_without_shrinking_controls(instrument_window):
    window, ui, app = instrument_window
    controller = window._responsive_window
    window.show()
    app.processEvents()
    app.processEvents()
    assert controller.scroll.horizontalScrollBar().maximum() == 0
    assert controller.scroll.verticalScrollBar().maximum() == 0
    if isinstance(ui, AlignmentPlotWindow):
        assert window.size() == QtCore.QSize(660, 350)
    buttons = [(button, button.font(), button.minimumSize())
               for button in window.findChildren(QtWidgets.QPushButton)]
    for width, height in ((1366, 728), (1280, 680), (960, 540)):
        area = QtCore.QRect(0, 0, width, height)
        controller.fit_to_screen(area)
        app.processEvents()
        assert area.contains(window.frameGeometry())
        for button, font, minimum in buttons:
            assert button.font() == font
            assert button.width() >= minimum.width()
            assert button.height() >= minimum.height()
        # Every control can be reached by scrolling, even on the smallest screen.
        for button, _, _ in buttons:
            if button.isVisible():
                controller.scroll.ensureWidgetVisible(button, 0, 0)
                app.processEvents()
                point = button.mapTo(controller.scroll.viewport(), QtCore.QPoint())
                assert controller.scroll.viewport().rect().intersects(
                    QtCore.QRect(point, button.size()))
    window.resize(1800, 950)
    app.processEvents()
    assert controller.scroll.horizontalScrollBar().maximum() == 0
    assert controller.scroll.verticalScrollBar().maximum() == 0
