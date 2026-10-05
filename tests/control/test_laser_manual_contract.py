"""Manual-derived protocol and GUI regressions. Never open a serial port."""
import math
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt6 import QtGui, QtWidgets

from pyccapt.control.nkt_photonics import readback
from pyccapt.control.nkt_photonics.origamiClassCLI import origClass
from pyccapt.control.gui.gui_laser_control import Ui_Laser_Control


@pytest.mark.parametrize('reply,value', [
    ('ly_oxp2_power 4.65', 4.65), ('ly_oxp2_dev_status 129', 129),
    ('Notice: Division factor for PCAOM is 10000', 10000),
    ('>e_freq?\r\nFrequency index parameter: 6\r\n?>', 6),
    ('Setted laser power in relative unit: 4000', 4000),
    ('e_mode of AOM: 2', 2), ('Error 5', None), ('ly_oxp2_power?', None),
])
def test_numeric_parser_does_not_parse_command_name_digits(reply, value):
    assert readback.scalar(reply) == value


@pytest.mark.parametrize('reply,mw,nj', [
    ('e_mlp\n12300 nJ', 4920., 12300.),
    ('e_mlp 4.9 W', 4900., 12250.),
    ('e_mlp 4900mW', 4900., 12250.),
    ('NKT FHG Output power=0.53W', 530., 1325.),
    ('12.3 uJ', 4920., 12300.),
])
def test_unit_aware_power_energy_conversion(reply, mw, nj):
    power, energy = readback.optical_values(reply, 400000.)
    assert power == pytest.approx(mw)
    assert energy == pytest.approx(nj)


def test_unitless_monitor_is_unknown_and_energy_is_converted_to_pj():
    assert all(math.isnan(v) for v in readback.optical_values('e_mlp 12300', 400000.))
    import time
    v = SimpleNamespace(pulse_mode='Laser', pulse_frequency=200,
        laser_telemetry=dict(monotonic=time.monotonic(), pulse_energy_nj=1325., output_frequency_hz=40000.))
    assert readback.pulse_energy_pj(v) == 1325000.
    assert readback.experiment_frequency_hz(v) == 40000.
    v.laser_telemetry['monotonic'] -= 11
    assert math.isnan(readback.pulse_energy_pj(v))


def test_complete_multiline_cli_response_and_correct_query_syntax():
    class Port:
        def __init__(self):
            self.pending = b''
            self.commands = []
        def reset_input_buffer(self):
            self.pending = b''
        def write(self, cmd):
            self.commands.append(cmd)
            if cmd == b'ly_oxp2_dev_status?\r\n':
                self.pending = cmd+b'ly_oxp2_dev_status 33\r\n>'
            else:
                self.pending = cmd+b'Available repetition rates:\r\ne_freq=4 --> 400000 Hz\r\ne_freq=6 --> 579710 Hz\r\n>'
        @property
        def in_waiting(self):
            return min(5, len(self.pending))
        def read(self, count):
            data, self.pending = self.pending[:count], self.pending[count:]
            return data
    device = origClass('FAKE')
    device.ser = Port()
    assert device.StatusRead() == 'ly_oxp2_dev_status 33'
    assert readback.frequency_table(device.freq_avaliable()) == {4: 400000., 6: 579710.}


def test_missing_response_is_a_timeout_not_unbound_local_error():
    device = origClass('FAKE')
    device.ser = Mock(in_waiting=0)
    device.ser.read.return_value = b''
    with pytest.raises(TimeoutError):
        device._query('e_mlp?', timeout=.001)


class FragmentedPort:
    """Bounded simulated reads, including pauses between response lines."""
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.elapsed = 0.
        self.commands = []
        self.closed = False

    def reset_input_buffer(self): pass
    def write(self, command): self.commands.append(command)
    def close(self): self.closed = True
    @property
    def in_waiting(self): return 0

    def read(self, count):
        self.elapsed += .05
        return self.chunks.pop(0) if self.chunks else b''


def fragmented_device(monkeypatch, chunks):
    from pyccapt.control.nkt_photonics import origamiClassCLI
    port = FragmentedPort(chunks)
    monkeypatch.setattr(origamiClassCLI, 'time', SimpleNamespace(monotonic=lambda: port.elapsed))
    device = origClass('FAKE')
    device.ser = port
    return device


@pytest.mark.parametrize('echo', [b'', b'ly_oxp2_dev_status?\n', b'>ly_oxp2_dev_status?\r\n'])
def test_status_without_prompt_accepts_real_instrument_reply(monkeypatch, echo):
    device = fragmented_device(monkeypatch, [echo+b'ly_oxp2_dev_status 9\n'])
    assert device.StatusRead() == 'ly_oxp2_dev_status 9'


def test_promptless_multiline_response_is_not_cut_at_first_newline(monkeypatch):
    device = fragmented_device(monkeypatch, [
        b'e_freq_available?\nAvailable repetition rates:\n', b'',
        b'\t e_freq=4\t--> 400000 Hz\n', b'',
        b'\t e_freq=5\t--> 500000 Hz\n', b'\t e_freq=6\t--> 579710 Hz\n',
    ])
    assert readback.frequency_table(device.freq_avaliable()) == {4: 400000., 5: 500000., 6: 579710.}


@pytest.mark.parametrize('response', [b'e_mlp?\n', b'e_mlp?\n123 m', b''])
def test_echo_or_unterminated_response_is_not_success(monkeypatch, response):
    device = fragmented_device(monkeypatch, [response])
    with pytest.raises(TimeoutError) as error:
        device._query('e_mlp?', timeout=.5)
    assert 'received' in str(error.value).lower()


def test_setter_acknowledgement_without_newline(monkeypatch):
    device = fragmented_device(monkeypatch, [b'e_div=10=ok>'])
    assert device.Div(10) == 'e_div=10=ok'


@pytest.mark.parametrize('method,command', [
    ('Listen', 'ly_oxp2_listen'), ('Standby', 'ly_oxp2_standby'),
    ('Enable', 'ly_oxp2_enabled'), ('AOMEnable', 'ly_oxp2_output_enable'),
    ('AOMDisable', 'ly_oxp2_output_disable'),
])
def test_state_commands_can_return_echo_without_confirming_state(monkeypatch, method, command):
    device = fragmented_device(monkeypatch, [(command+'\r\n').encode('ascii')])
    assert getattr(device, method)() == ''
    # The echo does not serve as a status reading or imply the desired state.
    with pytest.raises(TimeoutError):
        device.StatusRead(timeout=.5)


def test_silent_setter_still_fails(monkeypatch):
    device = fragmented_device(monkeypatch, [])
    with pytest.raises(TimeoutError):
        device.Listen()


def test_echo_only_divider_setter_is_confirmed_by_separate_readback(monkeypatch):
    device = fragmented_device(monkeypatch, [])
    port = device.ser
    def write(command):
        port.commands.append(command)
        port.chunks = [command]
        if command == b'e_div?\r\n':
            port.chunks.append(b'Notice: Division factor for PCAOM is 4\n')
    port.write = write
    assert device.Div(4) == ''
    assert readback.scalar(device.DivRead()) == 4
    assert port.commands == [b'e_div=4\r\n', b'e_div?\r\n']


@pytest.mark.parametrize('reply,expected', [
    (b'ly_oxp2_dev_status?\nly_oxp2_dev_status 9\n', True),
    (b'ly_oxp2_dev_status?\n', False),
    (b'ly_oxp2_dev_status?\nUnknown command\n', False),
    (b'ly_oxp2_dev_status?\nly_oxp2_dev_status 999\n', False),
])
def test_cli_probe_requires_valid_status_and_closes_port(monkeypatch, reply, expected):
    from pyccapt.control.nkt_photonics import nktpbus_switch, origamiClassCLI
    device = fragmented_device(monkeypatch, [reply])
    port = device.ser
    monkeypatch.setattr(origamiClassCLI.serial, 'Serial', lambda **kwargs: port)
    assert nktpbus_switch.is_cli_responding('FAKE', timeout_s=.5) is expected
    assert port.commands == [b'ly_oxp2_dev_status?\r\n']
    assert port.closed


class Laser:
    def __init__(self):
        self.calls = []
        self.code = 33
        self.aom = 800
        self.harmonic = 'NKT FHG Output power=0.53W'
    def freq_avaliable(self): return 'e_freq=4 --> 400000 Hz\ne_freq=6 --> 579710 Hz'
    def FreqRead(self): return 'Frequency index parameter: 4'
    def DivRead(self): return 'Notice: Division factor for PCAOM is 10'
    def AOMRead(self): return f'Setted laser power in relative unit: {self.aom}'
    def wavelength_read(self): return 'NKT FHG Current position=DUV'
    def read_average_power(self): return '12300 nJ'
    def power_read_dv_green(self): return self.harmonic
    def StatusRead(self): return f'ly_oxp2_dev_status {self.code}'
    def ModeRead(self): return 'e_mode of AOM: 2'
    def AOM(self, value): self.calls.append(('AOM', value)); self.aom = value
    def Div(self, value): self.calls.append(('Div', value))  # Simulate refusing requested divider.
    def Freq(self, value): self.calls.append(('Freq', value))
    def wavelength_change(self, value): self.calls.append(('wavelength', value))
    def Enable(self): self.calls.append(('Enable',)); self.code = 129
    def AOMEnable(self): self.calls.append(('AOMEnable',)); self.code = 129
    def AOMDisable(self): self.calls.append(('AOMDisable',)); self.code = 65
    def Listen(self): self.calls.append(('Listen',)); self.code = 9
    def Standby(self): self.calls.append(('Standby',)); self.code = 33


@pytest.fixture
def laser_gui():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ui = Ui_Laser_Control.__new__(Ui_Laser_Control)
    ui.variables = SimpleNamespace(start_flag=False, laser_telemetry={})
    ui.conf = {'laser_aom_max_percent': 100.}
    ui.laser_device = Laser()
    ui.errors = []
    ui.error_message = ui.errors.append
    ui.laser_rate = QtWidgets.QComboBox()
    ui.laser_wavelegnth = QtWidgets.QComboBox()
    ui.laser_wavelegnth.addItems(['IR', 'Green', 'DUV'])
    ui.laser_wavelegnth_nm_label = QtWidgets.QLabel()
    ui.laser_power = QtWidgets.QDoubleSpinBox(); ui.laser_power.setRange(0, 100)
    ui.laser_divition_factor = QtWidgets.QSpinBox(); ui.laser_divition_factor.setRange(1, 10000000)
    for name in ('laser_power_disp', 'laser_pulse_energy_disp', 'laser_repetion_rate_disp'):
        setattr(ui, name, QtWidgets.QLCDNumber())
    for name in ('laser_listen', 'laser_standby', 'laser_on', 'laser_enable'):
        setattr(ui, name, QtWidgets.QPushButton())
    for name in ('led_laser_listen', 'led_laser_laser_standby', 'led_laser_on', 'led_laser_enable'):
        setattr(ui, name, QtWidgets.QLabel())
    ui.led_green = ui.led_red = QtGui.QPixmap(1, 1)
    for flag in ('listen_mode', 'standby_mode', 'on_mode', 'enable_ouput_mode',
                 'change_laser_wavelegnth', 'change_laser_power', 'change_laser_rate', 'change_laser_divition_factor'):
        setattr(ui, flag, False)
    ui.index = 0
    yield ui


def test_actual_harmonic_frequency_divider_and_power_are_displayed(laser_gui):
    ui = laser_gui
    ui._sync_controls_from_device(initial=True)
    data = ui.variables.laser_telemetry
    assert data['wavelength'] == 'DUV'
    assert data['wavelength_nm'] == 257.5
    assert data['output_power_mw'] == 530.
    assert data['pulse_energy_nj'] == 13250.  # 0.53 W / 40 kHz.
    assert ui.laser_power.value() == 20.
    assert ui.laser_repetion_rate_disp.value() == 40.
    assert ui.laser_power_disp.value() == .53
    assert ui.laser_pulse_energy_disp.value() == 13.25
    assert ui.laser_rate.itemData(1) == 6
    assert ui.laser_rate.itemText(0) == '400'
    assert ui.laser_rate.itemText(1) == '579.71'
    assert data['base_frequency_hz'] == 400000.
    assert not ui.laser_device.calls


def test_rejected_divider_uses_actual_readback_and_does_not_open_aom(laser_gui):
    ui = laser_gui
    ui._sync_controls_from_device(initial=True)
    ui.laser_divition_factor.setValue(20)
    ui.change_laser_divition_factor = True
    ui.check_laser_status()
    assert ui.variables.laser_division_factor == 10
    assert ui.laser_divition_factor.value() == 10
    assert ui.laser_device.calls == [('Div', 20)]


def test_unsupported_harmonic_monitor_never_substitutes_ir_power(laser_gui, caplog):
    ui = laser_gui
    ui.laser_device.harmonic = 'Unknown command'
    ui._sync_controls_from_device(initial=True)
    assert not ui.variables.laser_telemetry['valid']
    assert math.isnan(ui.variables.laser_average_power)
    assert 'Unknown command' in ui.variables.laser_telemetry['error']
    assert 'DUV monitor' in ui.laser_power_disp.toolTip()
    assert 'ls_output_power?' in ui.laser_power_disp.toolTip()
    assert 'Pulse energy is unavailable' in ui.laser_pulse_energy_disp.toolTip()
    assert ui.laser_repetion_rate_disp.value() == 40.
    assert ui.variables.laser_telemetry['raw']['ls_output_power'] == 'Unknown command'
    assert len([record for record in caplog.records if 'optical readback unavailable' in record.message]) == 1
    ui._sync_controls_from_device()
    assert len([record for record in caplog.records if 'optical readback unavailable' in record.message]) == 1
    ui.laser_device.harmonic = 'NKT FHG Output power=0.53W'
    ui._sync_controls_from_device()
    assert ui.variables.laser_telemetry['valid']
    assert 'error' not in ui.variables.laser_telemetry
    assert 'Unknown command' not in ui.laser_power_disp.toolTip()


def test_true_zero_harmonic_reading_is_distinct_from_missing_monitor(laser_gui):
    ui = laser_gui
    ui.laser_device.harmonic = 'NKT FHG Output power=0.00W'
    ui._sync_controls_from_device(initial=True)
    assert ui.variables.laser_telemetry['valid']
    assert ui.variables.laser_average_power == 0.
    assert ui.variables.laser_pulse_energy == 0.
    assert ui.laser_power_disp.value() == 0.
    assert ui.laser_pulse_energy_disp.value() == 0.
    assert 'error' not in ui.variables.laser_telemetry


def test_laser_on_tracks_output_enabled_without_full_power_write(laser_gui):
    ui = laser_gui
    ui.on_mode = True
    ui.check_laser_status()
    assert ui.laser_device.code == 129
    assert ui.laser_device.calls == [('Enable',)]
    assert not ui.laser_rate.isEnabled()
    assert not ui.laser_wavelegnth.isEnabled()


def test_frequency_changes_are_blocked_during_experiment(laser_gui):
    ui = laser_gui
    ui.variables.start_flag = True
    ui.change_laser_rate = ui.change_laser_wavelegnth = ui.change_laser_divition_factor = True
    ui._apply_laser_settings(33)
    assert not ui.laser_device.calls
    assert len(ui.errors) == 3


@pytest.mark.parametrize('code', [65, 129])
def test_divider_edit_is_allowed_while_on_but_base_rate_is_blocked(laser_gui, code):
    ui = laser_gui
    ui.laser_device.code = code
    ui._sync_controls_from_device(initial=True)
    assert ui.laser_divition_factor.isEnabled()
    assert not ui.laser_rate.isEnabled()
    ui.laser_divition_factor.setValue(4)
    ui.change_laser_divition_factor = True
    ui.change_laser_rate = True
    ui.check_laser_status()
    assert ui.laser_device.calls == [('Div', 4)]
    assert any('Listen or Standby' in error for error in ui.errors)


def test_khz_dropdown_preserves_factory_indexes_and_one_mhz_option(laser_gui):
    ui = laser_gui
    ui.laser_device.freq_avaliable = lambda: 'e_freq=4 --> 400000 Hz\ne_freq=10 --> 1000000 Hz'
    ui._sync_controls_from_device(initial=True)
    assert [ui.laser_rate.itemText(i) for i in range(2)] == ['400', '1000']
    assert [ui.laser_rate.itemData(i) for i in range(2)] == [4, 10]
    ui.laser_rate.setCurrentIndex(1)
    ui.change_laser_rate = True
    ui._apply_laser_settings(33)
    assert ui.laser_device.calls == [('Freq', 10)]


def test_listen_cancels_pending_emission_request(laser_gui):
    ui = laser_gui
    ui.on_mode = ui.listen_mode = True
    ui.check_laser_status()
    assert ui.laser_device.calls == [('Listen',)]
    assert not ui.on_mode


def test_already_standby_request_still_cancels_queued_on(laser_gui):
    ui = laser_gui
    ui.laser_device.code = 33
    ui.standby_mode = ui.on_mode = True
    ui.check_laser_status()
    assert ui.laser_device.calls == [('Standby',)]
    assert not ui.on_mode


@pytest.mark.parametrize('code,actions', [
    (9, {'standby'}), (17, {'listen'}), (33, {'listen', 'on'}),
    (65, {'listen', 'standby', 'output'}), (129, {'listen', 'standby', 'on', 'output'}),
    (1, {'listen'}), (3, {'listen'}), (5, {'listen'}), (None, {'listen'}), (255, {'listen'}),
])
def test_laser_action_permissions_match_observed_state(laser_gui, code, actions):
    ui = laser_gui
    ui.laser_state_label = QtWidgets.QLabel()
    ui._apply_button_locks_for_status(f'ly_oxp2_dev_status {code}' if code is not None else None)
    actual = {name for name, widget in (
        ('listen', ui.laser_listen), ('standby', ui.laser_standby),
        ('on', ui.laser_on), ('output', ui.laser_enable),
    ) if widget.isEnabled()}
    assert actual == actions
    if code == 17:
        assert 'warming' in ui.laser_state_label.text()


def test_disconnected_laser_has_no_actions(laser_gui):
    ui = laser_gui
    ui.laser_device = None
    ui._apply_button_locks_for_status('ly_oxp2_dev_status 33')
    assert not any(button.isEnabled() for button in (ui.laser_listen, ui.laser_standby, ui.laser_on, ui.laser_enable))


def test_standby_warmup_keeps_listen_available_then_enables_on_when_ready(laser_gui):
    ui = laser_gui
    ui.laser_device.code = 9
    def warmup():
        ui.laser_device.calls.append(('Standby',))
        ui.laser_device.code = 17
    ui.laser_device.Standby = warmup
    ui.standby_mode = True
    ui.check_laser_status()
    assert ui.laser_listen.isEnabled()
    assert not ui.laser_on.isEnabled()
    assert not ui.variables.laser_telemetry['valid']
    ui.laser_device.code = 33
    ui.check_laser_status()
    assert ui.laser_on.isEnabled()
    assert ui.laser_listen.isEnabled()


def test_can_return_to_listen_from_warmup(laser_gui):
    ui = laser_gui
    ui.laser_device.code = 17
    ui.listen_mode = True
    ui.check_laser_status()
    assert ui.laser_device.calls == [('Listen',)]
    assert ui.laser_standby.isEnabled()


def test_standby_click_immediately_allows_listen_before_status_changes(laser_gui):
    ui = laser_gui
    ui.laser_device.code = 9
    ui._apply_button_locks_for_status('ly_oxp2_dev_status 9')
    assert not ui.laser_listen.isEnabled()
    ui.laser_standby_clicked()
    assert ui.laser_listen.isEnabled()
    assert not ui.laser_standby.isEnabled()
    ui._apply_button_locks_for_status('ly_oxp2_dev_status 9')
    assert ui.laser_listen.isEnabled()
    ui.laser_listen_clicked()
    ui.check_laser_status()
    assert ui.laser_device.calls == [('Listen',)]
    assert not ui._laser_standby_pending
    assert ui.laser_standby.isEnabled()


def test_pending_standby_preserves_cancel_until_ready_status(laser_gui):
    ui = laser_gui
    ui.laser_device.code = 9
    ui._apply_button_locks_for_status('ly_oxp2_dev_status 9')
    ui.laser_standby_clicked()
    # Simulate firmware still reporting Listen just after accepting Standby.
    ui.laser_device.Standby = lambda: ui.laser_device.calls.append(('Standby',))
    ui.check_laser_status()
    assert ui.laser_listen.isEnabled()
    assert not ui.laser_on.isEnabled()
    assert ui._laser_standby_pending
    ui.laser_device.code = 17
    ui.check_laser_status()
    assert ui.laser_listen.isEnabled()
    ui.laser_device.code = 33
    ui.check_laser_status()
    assert not ui._laser_standby_pending
    assert ui.laser_listen.isEnabled()
    assert ui.laser_on.isEnabled()


def test_on_request_can_be_cancelled_to_standby_before_status_changes(laser_gui):
    ui = laser_gui
    ui._apply_button_locks_for_status('ly_oxp2_dev_status 33')
    ui.laser_on_clicked()
    assert ui.laser_listen.isEnabled()
    assert ui.laser_standby.isEnabled()
    assert not ui.laser_on.isEnabled()
    ui.laser_standby_clicked()
    ui.check_laser_status()
    assert ui.laser_device.calls == [('Standby',)]
    assert not ui._laser_emission_pending


def test_sent_on_request_keeps_lower_states_available(laser_gui):
    ui = laser_gui
    ui._apply_button_locks_for_status('ly_oxp2_dev_status 33')
    ui.laser_device.Enable = lambda: ui.laser_device.calls.append(('Enable',))
    ui.laser_on_clicked()
    ui.check_laser_status()
    assert ui.laser_listen.isEnabled()
    assert ui.laser_standby.isEnabled()
    ui.laser_listen_clicked()
    ui.check_laser_status()
    assert ui.laser_device.calls == [('Enable',), ('Listen',)]


def test_on_transition_setup_can_be_cancelled_to_standby(laser_gui):
    ui = laser_gui
    ui._apply_button_locks_for_status('ly_oxp2_dev_status 33')
    def enabling():
        ui.laser_device.calls.append(('Enable',))
        ui.laser_device.code = 17
    ui.laser_device.Enable = enabling
    ui.laser_on_clicked()
    ui.check_laser_status()
    assert ui.laser_listen.isEnabled()
    assert ui.laser_standby.isEnabled()
    ui.laser_standby_clicked()
    ui.check_laser_status()
    assert ui.laser_device.calls == [('Enable',), ('Standby',)]
    assert not ui._laser_emission_pending


@pytest.mark.parametrize('send_first', [False, True])
def test_pending_output_enable_can_be_closed_without_replaying_enable(laser_gui, send_first):
    ui = laser_gui
    ui.laser_device.code = 65
    ui._apply_button_locks_for_status('ly_oxp2_dev_status 65')
    ui.laser_device.AOMEnable = lambda: ui.laser_device.calls.append(('AOMEnable',))
    ui.laser_enable_clicked()
    assert ui.laser_enable.text() == 'Close Output'
    assert ui.laser_listen.isEnabled()
    assert ui.laser_standby.isEnabled()
    if send_first:
        ui.check_laser_status()
        assert ui.laser_enable.text() == 'Close Output'
    ui.laser_enable_clicked()
    ui.check_laser_status()
    assert ui.laser_device.calls == ([('AOMEnable',)] if send_first else [])+[('AOMDisable',)]
    assert not ui._laser_emission_pending


def test_confirmed_open_output_has_all_lower_state_controls(laser_gui):
    ui = laser_gui
    ui.laser_device.code = 129
    ui._apply_button_locks_for_status('ly_oxp2_dev_status 129')
    assert ui.laser_listen.isEnabled()
    assert ui.laser_standby.isEnabled()
    assert ui.laser_enable.text() == 'Close Output'
    ui.laser_enable_clicked()
    ui.check_laser_status()
    assert ui.laser_device.calls == [('AOMDisable',)]


def test_listen_is_attempted_even_if_status_reads_fail(laser_gui):
    ui = laser_gui
    ui._laser_status_in_progress = False
    ui.laser_device.StatusRead = Mock(side_effect=TimeoutError('Status unavailable'))
    ui.listen_mode = True
    ui._poll_laser_status()
    assert ui.laser_device.calls == [('Listen',)]
    assert ui.laser_listen.isEnabled()
    assert not ui.laser_on.isEnabled()


def test_failed_status_does_not_replay_queued_emission_after_recovery(laser_gui):
    ui = laser_gui
    ui._laser_status_in_progress = False
    ui.laser_device.StatusRead = Mock(side_effect=TimeoutError('Status unavailable'))
    ui.on_mode = True
    ui._poll_laser_status()
    assert not ui.on_mode
    assert not ui.laser_device.calls
    ui.laser_device.StatusRead = lambda: 'ly_oxp2_dev_status 33'
    ui.check_laser_status()
    assert not ui.laser_device.calls


def test_setting_permissions_use_state_after_emission_command(laser_gui):
    ui = laser_gui
    ui._sync_controls_from_device(initial=True)
    ui.on_mode = True
    ui.change_laser_rate = True
    ui.check_laser_status()
    assert ui.laser_device.calls == [('Enable',)]
    assert any('Listen or Standby' in error for error in ui.errors)


def test_pending_aom_edit_is_rejected_while_warming(laser_gui):
    ui = laser_gui
    ui.laser_device.code = 17
    ui.change_laser_power = True
    ui.check_laser_status()
    assert not ui.laser_device.calls
    assert any('stable laser state' in error for error in ui.errors)


def test_queued_emission_is_cancelled_while_warming(laser_gui):
    ui = laser_gui
    ui.laser_device.code = 17
    ui.on_mode = True
    ui.check_laser_status()
    assert not ui.laser_device.calls
    assert not ui.on_mode
    ui.laser_device.code = 33
    ui.check_laser_status()
    assert not ui.laser_device.calls


def test_failed_readback_clears_previous_wavelength_and_values(laser_gui):
    ui = laser_gui
    ui._sync_controls_from_device(initial=True)
    assert '257.5' in ui.laser_wavelegnth_nm_label.text()
    ui._invalidate_laser_readouts('Serial timeout')
    assert ui.laser_wavelegnth_nm_label.text() == '(not read back)'
    assert not ui.variables.laser_telemetry['valid']
    assert math.isnan(ui.variables.laser_pulse_energy)
    assert math.isnan(ui.variables.laser_average_power)


def test_connection_failure_reports_evidence_without_guessing_bus_mode(laser_gui, monkeypatch):
    from pyccapt.control.nkt_photonics import origamiClassCLI
    device = Mock(last_error=None)
    device.open_port.return_value = 0
    device.StatusRead.side_effect = TimeoutError('received 0 bytes: b\'\'')
    monkeypatch.setattr(origamiClassCLI, 'origClass', lambda port: device)
    ui = laser_gui
    ui._set_laser_disconnected_banner = Mock()
    assert not ui._open_laser_cli('FAKE')
    reason = ui._set_laser_disconnected_banner.call_args.args[0]
    assert 'status query failed' in reason
    assert 'received 0 bytes' in reason
    assert 'Interface mode has not been determined' in reason
    assert 'NKTPBus' not in reason
    device.close_port.assert_called_once()
    device.Listen.assert_not_called()


def test_cli_switch_releases_existing_session_before_probe(laser_gui, monkeypatch):
    from pyccapt.control.nkt_photonics import nktpbus_switch
    ui = laser_gui
    ui.com_port_laser = 'FAKE'
    ui.laser_device.close_port = Mock()
    old_device = ui.laser_device
    ui._open_laser_cli = Mock(return_value=True)

    def probe(port):
        assert port == 'FAKE'
        old_device.close_port.assert_called_once()
        assert ui.laser_device is None
        return True

    monkeypatch.setattr(nktpbus_switch, 'is_cli_responding', probe)
    switch = Mock()
    monkeypatch.setattr(nktpbus_switch, 'switch_to_cli', switch)
    ui.switch_to_cli_clicked()
    ui._open_laser_cli.assert_called_once_with('FAKE')
    switch.assert_not_called()
