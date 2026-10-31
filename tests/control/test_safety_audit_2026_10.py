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


# --- second batch ------------------------------------------------------------

def test_unparseable_laser_status_does_not_abort_the_experiment_loop():
    import math
    from pyccapt.control.nkt_photonics.readback import TELEMETRY_UNITS, telemetry_value
    snapshot = {'status_code': None, 'valid': False, 'divider': 4, 'output_power_mw': float('nan')}
    values = {key: telemetry_value(snapshot, key) for key in TELEMETRY_UNITS}
    assert math.isnan(values['status_code'])
    assert values['valid'] == 0.0 and values['divider'] == 4.0
    assert math.isnan(values['wavelength_nm'])  # missing key
    assert telemetry_value({}, 'valid') == 0.0


def _vent_panel():
    from pyccapt.control.gui import gui_pumps_vacuum
    panel = SimpleNamespace(flag_vent_cll_partial=False, _cll_vent_cycle=0,
                            conf={'cll_backing_valve_line': 6, 'cll_turbo_valve_line': 7},
                            variables=SimpleNamespace(), valves=[])
    panel._set_cll_valve_6525 = lambda line, state: panel.valves.append((line, state))
    panel._set_vent_valve = lambda state: panel.valves.append(('vent', state))
    cls = gui_pumps_vacuum.Ui_Pumps_Vacuum
    return panel, cls


def test_stale_cll_deselect_timers_cannot_reopen_turbo_in_a_later_cycle():
    pytest.importorskip('PyQt6')
    panel, cls = _vent_panel()
    # Deselect at t=0 schedules turbo-on for cycle 1; the operator then vents
    # again (cycle 2) and deselects again (cycle 3) before t=90 s.
    first_deselect, panel._cll_vent_cycle = 1, 3
    cls._vent_cll_deselect_turbo_on(panel, first_deselect)
    cls._vent_cll_deselect_backing_off(panel, first_deselect)
    assert panel.valves == []
    cls._vent_cll_deselect_turbo_on(panel, 3)
    cls._vent_cll_deselect_backing_off(panel, 3)
    assert panel.valves == [(7, True), (6, False)]


def test_stale_cll_vent_press_timer_cannot_vent_a_later_cycle():
    pytest.importorskip('PyQt6')
    panel, cls = _vent_panel()
    panel.flag_vent_cll_partial, panel._cll_vent_cycle = True, 3
    cls._vent_cll_press_delayed(panel, 1)
    assert panel.valves == []
    cls._vent_cll_press_delayed(panel, 3)
    assert panel.valves == [('vent', False)]


class _FakeCtl:
    class Property:
        REFERENCING_OPTIONS, MOVE_VELOCITY, CHANNEL_STATE = 1, 2, 3

    class ChannelState:
        ACTIVELY_MOVING = 1

    class Error(Exception):
        pass

    def __init__(self, moving):
        self.moving, self.stopped = set(moving), []

    def SetProperty_i32(self, *args):
        pass

    def SetProperty_i64(self, *args):
        pass

    def Reference(self, handle, channel, _):
        pass

    def GetProperty_i32(self, handle, channel, prop):
        return 1 if channel in self.moving else 0

    def Stop(self, handle, channel):
        self.stopped.append(channel)
        self.moving.discard(channel)


def _stage(monkeypatch, moving):
    from pyccapt.control.smaract_mcs2 import mcs2_stage
    fake = _FakeCtl(moving)
    monkeypatch.setattr(mcs2_stage, 'ctl', fake)
    stage = mcs2_stage.SmarActStage.__new__(mcs2_stage.SmarActStage)
    stage._handle = 1
    return mcs2_stage, stage, fake


def test_reference_timeout_stops_every_axis(monkeypatch):
    mcs2_stage, stage, fake = _stage(monkeypatch, moving={0, 1, 2})
    with pytest.raises(mcs2_stage.SmarActStageError, match='did not stop'):
        stage.find_reference(timeout_s=0.05)
    assert sorted(fake.stopped) == [0, 1, 2]


def test_cancelled_reference_is_not_reported_as_complete(monkeypatch):
    import threading
    mcs2_stage, stage, fake = _stage(monkeypatch, moving={0, 1, 2})
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(mcs2_stage.SmarActReferenceCancelled):
        stage.find_reference(timeout_s=1.0, cancel_event=cancel)
    assert {0, 1, 2} <= set(fake.stopped)
    assert issubclass(mcs2_stage.SmarActReferenceCancelled, mcs2_stage.SmarActStageError)


def test_gauges_are_polled_when_cryo_monitoring_is_off(monkeypatch):
    import threading
    from pyccapt.control.devices import initialize_devices as module

    class Tpg:
        def __init__(self, port):
            pass

        def pressure_gauge(self, channel):
            return ('3e-9' if channel == 2 else '4e-8'), 0

    stop = threading.Event()
    monkeypatch.setattr(module, 'TPG362', Tpg)
    monkeypatch.setattr(module.time, 'sleep', lambda _s: stop.set())
    conf = {'gauges': 'on', 'cryo': 'off', 'COM_PORT_gauge_mc': 'COM5', 'COM_PORT_gauge_bc': 'off',
            'COM_PORT_gauge_ll': 'off', 'COM_PORT_gauge_cll': 'off', 'pump_ll': 'off', 'pump_cll': 'off',
            'log_time_time_interval': 1e9, 'min_temperature_ll': 250}
    variables = SimpleNamespace(COM_PORT_gauge_mc='COM5', vacuum_main=0.0, vacuum_buffer=0.0)
    emitter = SimpleNamespace(**{name: Mock() for name in ('vacuum_main', 'vacuum_buffer')})

    module.state_update(conf, variables, emitter, stop_event=stop)

    assert variables.vacuum_main == pytest.approx(3e-9)
    assert variables.vacuum_buffer == pytest.approx(4e-8)
