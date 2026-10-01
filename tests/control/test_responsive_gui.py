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
        window.resize(1480, 650)
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
    assert window.height() == 640
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
    assert ui.alignment_group.title() == 'Alignment'
    for widget in (ui.alignment_start_voltage_label, ui.alignment_start_voltage,
                   ui.alignment_voltage_increment_label, ui.alignment_voltage_increment,
                   ui.automatic_alignment_button):
        assert ui.alignment_group.isAncestorOf(widget)
    assert position(ui.alignment_start_voltage).y() < position(ui.alignment_voltage_increment).y()
    assert position(ui.electrode_button).y() < position(ui.alignment_start_voltage).y()
    assert position(ui.alignment_voltage_increment).y() < position(ui.automatic_alignment_button).y()
    assert position(ui.start_button).y() == position(ui.stop_button).y()
    assert position(ui.start_button).x() < position(ui.stop_button).x()
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
