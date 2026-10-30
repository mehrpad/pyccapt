"""Regression tests for the 2026-10 control safety audit fixes."""
import os
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest

from pyccapt.control.apt.apt_exp_control import APT_Exp_Control
from pyccapt.control.apt.experiment_state import ExperimentState
from pyccapt.control.devices.initialize_devices import read_tpg_pressure


def test_held_dc_setpoint_above_vdc_max_cannot_lift_the_step_floor(monkeypatch):
    from pyccapt.control.apt import apt_exp_control as module
    v = SimpleNamespace(temperature=0, vacuum_main=0, laser_pos_x=0, laser_pos_y=0,
                        laser_pos_z=0, stage_pos_x=0, stage_pos_y=0, stage_pos_z=0,
                        control_algorithm="Proportional", detection_rate_current=0.,
                        vdc_step_down=4, vdc_step_up=4, stop_flag=False, vdc_hold=False)
    # Operator set 9000 V while held; the run's ceiling is 7000 V.
    control = SimpleNamespace(variables=v, total_ions=100, count_last=100, _tdc_first_event_seen=True,
                              total_raw_signals=100, count_raw_signals_last=100, main_laser_telemetry={},
                              control_algorithm="Proportional", detection_rate=1, laser_alignment_runtime=None,
                              alignment=None, specimen_voltage=6990., vdc_min=9000., vdc_max=7000.,
                              _vdc_active=lambda: True, com_port_v_dc=object(), pulse_mode="Laser")
    for key in ("main_counter", "main_raw_counter", "main_temperature", "main_chamber_vacuum",
                "main_laser_x", "main_laser_y", "main_laser_z", "main_stage_x", "main_stage_y", "main_stage_z"):
        setattr(control, key, [])
    monkeypatch.setattr(module, "fresh_snapshot", lambda _: {})
    command = Mock()
    monkeypatch.setattr(module.apt_exp_control_func, "command_v_dc", command)

    module.APT_Exp_Control.main_ex_loop(control)

    assert control.specimen_voltage == 7000.
    command.assert_called_once_with(control.com_port_v_dc, ">S0 7000.0")


def test_set_dc_voltage_field_is_capped_by_the_run_vdc_max():
    gui_visualization = pytest.importorskip('pyccapt.control.gui.gui_visualization')
    ui = SimpleNamespace(conf={'default_vdc_min': 500, 'default_vdc_max': 9000},
                         variables=SimpleNamespace(vdc_max=7000))
    assert gui_visualization.Ui_Visualization._dc_voltage_limits(ui) == (500, 7000)
    ui.variables.vdc_max = 0  # No run parameters yet: config limits apply.
    assert gui_visualization.Ui_Visualization._dc_voltage_limits(ui) == (500, 9000)


def _control_with_safe_off(variables, side_effects):
    control = APT_Exp_Control(variables, {}, None, None, None, None, None)
    control.log_apt = Mock()
    control.safe_outputs_off = Mock(side_effect=side_effects)
    return control


def test_failed_safe_off_is_recorded_without_aborting_finalization():
    variables = SimpleNamespace(ex_freq=1, experiment_state=ExperimentState.STOPPING.value,
                                experiment_error="", hardware_safe=False)
    error = RuntimeError("Hardware safe-off incomplete: signal generator: VISA timeout")
    control = _control_with_safe_off(variables, [error, error])

    control._safe_off_before_finalization()  # Must not raise.

    assert control.safe_outputs_off.call_count == 2
    assert "VISA timeout" in control._run_failure
    assert "VISA timeout" in variables.experiment_error
    control._enter_finalization_state()  # STOPPING -> FINALIZING would be invalid.
    assert variables.experiment_state == ExperimentState.FAILED.value
    assert "VISA timeout" in variables.experiment_error


def test_transient_safe_off_error_is_retried_and_run_finalizes_normally():
    variables = SimpleNamespace(ex_freq=1, experiment_state=ExperimentState.STOPPING.value,
                                experiment_error="", hardware_safe=False)

    attempts = []

    def safe_off():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("transient")
        variables.hardware_safe = True
        variables.experiment_state = ExperimentState.SAFE_OFF.value

    control = _control_with_safe_off(variables, safe_off)

    control._safe_off_before_finalization()
    control._enter_finalization_state()

    assert control._run_failure == ""
    assert variables.experiment_state == ExperimentState.FINALIZING.value


class _Tpg:
    def __init__(self, replies):
        self.replies = replies

    def pressure_gauge(self, channel):
        reply = self.replies[channel]
        if isinstance(reply, Exception):
            raise reply
        return reply, 0


def test_gauge_read_error_publishes_unknown_instead_of_killing_the_thread():
    reported, cleared = [], []
    tpg = _Tpg({1: "2.5e-8", 2: IOError("NAK")})

    main = read_tpg_pressure(tpg, 2, "vacuum_main", lambda key, msg: reported.append(key), cleared.append)
    buffer = read_tpg_pressure(tpg, 1, "vacuum_buffer", lambda key, msg: reported.append(key), cleared.append)

    assert main == -1.0  # The gate interlock treats <= 0 as "pressure unknown".
    assert reported == ["vacuum_main"]
    assert buffer == pytest.approx(2.5e-8)
    assert cleared == ["vacuum_buffer"]


def test_gauge_garbled_reply_publishes_unknown():
    tpg = _Tpg({2: "garbage"})
    reported = []
    assert read_tpg_pressure(tpg, 2, "vacuum_main", lambda key, msg: reported.append(key), lambda key: None) == -1.0
    assert reported == ["vacuum_main"]


def _start_stub(flag_main_gate, super_user):
    ui = SimpleNamespace(
        variables=SimpleNamespace(electrode_out=False, flag_main_gate=flag_main_gate,
                                  laser_alignment_enabled=False, pulse_mode='Voltage'),
        conf={}, flag_super_user=super_user, errors=[],
    )
    ui.error_message = ui.errors.append
    return ui


def test_every_experiment_start_rechecks_the_main_gate(monkeypatch):
    gui_main = pytest.importorskip('pyccapt.control.gui.gui_main')
    reached_parameters = Mock(side_effect=gui_main.main_parameters.ParameterError('stop here'))
    monkeypatch.setattr(gui_main.main_parameters, 'validate_run_parameters', reached_parameters)

    # Chained starts (next TOML-plan row, next alignment sample) call the
    # worker directly, so the gate check must live there.
    ui = _start_stub(flag_main_gate=True, super_user=False)
    assert gui_main.Ui_PyCCAPT.start_experiment_worker(ui) is False
    assert 'main gate' in ui.errors[0]
    reached_parameters.assert_not_called()

    ui = _start_stub(flag_main_gate=True, super_user=True)
    assert gui_main.Ui_PyCCAPT.start_experiment_worker(ui) is False
    assert ui.errors == ['stop here']
    reached_parameters.assert_called_once()


def test_main_gate_cannot_open_between_chained_runs():
    QtWidgets = pytest.importorskip('PyQt6.QtWidgets')
    from PyQt6 import QtCore
    from pyccapt.control.core import runtime, share_variables
    from pyccapt.control.gui import gui_gates

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    conf, _ = runtime.load_project_config(change_cwd=False)
    conf = {**conf, 'gates': 'off', 'gauges': 'off'}
    variables = share_variables.Variables(conf, SimpleNamespace())
    window = QtWidgets.QWidget()
    ui = gui_gates.Ui_Gates(variables, conf)
    ui.setupUi(window)
    for timer in window.findChildren(QtCore.QTimer):
        timer.stop()
    ui.errors = []
    ui.error_message = ui.errors.append
    variables.flag_pump_load_lock = True
    variables.flag_pump_cryo_load_lock = True
    variables.flag_load_gate = variables.flag_cryo_gate = variables.flag_main_gate = False
    variables.start_flag = False

    variables.sample_selection_locked = True  # A plan/batch is between runs.
    ui.gates(1)
    assert variables.flag_main_gate is False
    assert ui.errors == ["!!! An experiment is running !!!"]

    variables.sample_selection_locked = False
    ui.gates(1)
    assert variables.flag_main_gate is True
    ui.timer.stop()
    window.close()
    app.processEvents()
