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
        self.close_port = Mock()
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
    ui.laser_connection_banner = QtWidgets.QLabel()
    ui._reset_laser_message_scroll = Mock()
    ui.laser_rate = QtWidgets.QComboBox()
    ui.laser_wavelegnth = QtWidgets.QComboBox()
    ui.laser_wavelegnth.addItems(['IR', 'Green', 'DUV'])
    ui.laser_wavelegnth_nm_label = QtWidgets.QLabel()
    ui.laser_power = QtWidgets.QDoubleSpinBox(); ui.laser_power.setRange(0, 100)
    ui.laser_divition_factor = QtWidgets.QSpinBox(); ui.laser_divition_factor.setRange(1, 10000000)
    ui.laser_ir_power_label = QtWidgets.QLabel()
    for name in ('laser_ir_power_disp', 'laser_power_disp', 'laser_pulse_energy_disp', 'laser_repetion_rate_disp'):
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
    assert ui.laser_ir_power_disp.value() == pytest.approx(.492)
    assert ui.laser_ir_power_label.text() == 'IR power estimate (W)'
    assert 'not a direct watt measurement' in ui.laser_ir_power_disp.toolTip()
    assert ui.laser_rate.itemData(1) == 6
    assert ui.laser_rate.itemText(0) == '400'
    assert ui.laser_rate.itemText(1) == '579.71'
    assert data['base_frequency_hz'] == 400000.
    assert not ui.laser_device.calls


@pytest.mark.parametrize('reply,watts', [('4.65 W', 4.65), ('4650 mW', 4.65), ('0 mW', 0.)])
def test_measured_ir_power_is_separate_from_selected_output_and_aom(laser_gui, reply, watts):
    ui = laser_gui
    ui.laser_device.read_average_power = lambda: reply
    ui._sync_controls_from_device(initial=True)
    assert ui.laser_ir_power_disp.value() == pytest.approx(watts)
    assert ui.laser_ir_power_label.text() == 'Measured IR power (W)'
    assert ui.laser_power_disp.value() == .53
    assert ui.laser_pulse_energy_disp.value() == 13.25
    assert ui.laser_power.value() == 20.
    assert 'e_mlp?' in ui.laser_ir_power_disp.toolTip()
    assert 'not a direct watt measurement' not in ui.laser_ir_power_disp.toolTip()
    assert not ui.laser_device.calls


@pytest.mark.parametrize('failure', ['unsupported', 'timeout'])
def test_unavailable_harmonic_monitor_keeps_independent_ir_reading(laser_gui, failure):
    ui = laser_gui
    ui.laser_device.read_average_power = lambda: '4.65 W'
    if failure == 'unsupported':
        ui.laser_device.harmonic = 'Unknown command'
    else:
        ui.laser_device.power_read_dv_green = Mock(side_effect=TimeoutError('harmonic timeout'))
    ui.laser_power_disp.display = Mock(wraps=ui.laser_power_disp.display)
    ui.laser_pulse_energy_disp.display = Mock(wraps=ui.laser_pulse_energy_disp.display)
    ui._sync_controls_from_device(initial=True)
    assert ui.laser_ir_power_disp.value() == 4.65
    assert not ui.variables.laser_telemetry['valid']
    ui.laser_power_disp.display.assert_called_with('-----')
    ui.laser_pulse_energy_disp.display.assert_called_with('-----')
    assert 'harmonic timeout' not in ui.laser_ir_power_disp.toolTip()
    assert 'Unknown command' not in ui.laser_ir_power_disp.toolTip()


@pytest.mark.parametrize('failure', ['unsupported', 'timeout'])
def test_unavailable_ir_monitor_keeps_selected_output_reading(laser_gui, failure):
    ui = laser_gui
    ui.laser_device.read_average_power = (lambda: 'Unknown command') if failure == 'unsupported' else Mock(
        side_effect=TimeoutError('IR timeout'))
    ui.laser_ir_power_disp.display = Mock(wraps=ui.laser_ir_power_disp.display)
    ui._sync_controls_from_device(initial=True)
    ui.laser_ir_power_disp.display.assert_called_with('-----')
    assert ui.variables.laser_telemetry['valid']
    assert ui.laser_power_disp.value() == .53
    assert ui.laser_pulse_energy_disp.value() == 13.25
    assert 'no usable unit-bearing reading' in ui.laser_ir_power_disp.toolTip()


def test_separate_ir_readout_is_cleared_after_disconnect_or_stale_reading(laser_gui):
    ui = laser_gui
    ui.laser_device.read_average_power = lambda: '4.65 W'
    ui._sync_controls_from_device(initial=True)
    ui.laser_ir_power_disp.display = Mock(wraps=ui.laser_ir_power_disp.display)
    ui.variables.laser_telemetry['monotonic'] -= 11
    ui._recompute_derived_readouts()
    ui.laser_ir_power_disp.display.assert_called_with('-----')
    assert 'No fresh IR monitor' in ui.laser_ir_power_disp.toolTip()
    ui._invalidate_laser_readouts('CLI disconnected')
    ui.laser_ir_power_disp.display.assert_called_with('-----')
    assert 'CLI disconnected' in ui.laser_ir_power_disp.toolTip()


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

    def probe(port, *, raise_on_port_error):
        assert port == 'FAKE'
        assert raise_on_port_error
        old_device.close_port.assert_called_once()
        assert ui.laser_device is None
        return True

    monkeypatch.setattr(nktpbus_switch, 'is_cli_responding', probe)
    switch = Mock()
    monkeypatch.setattr(nktpbus_switch, 'switch_to_cli', switch)
    ui.switch_to_cli_clicked()
    ui._open_laser_cli.assert_called_once_with('FAKE')
    switch.assert_not_called()


@pytest.fixture
def mode_gui(laser_gui):
    from PyQt6 import QtCore
    ui = laser_gui
    ui.com_port_laser = 'FAKE'
    ui.flag_super_user_stage = True
    ui._laser_status_in_progress = False
    ui.laser_device.close_port = Mock()
    ui.laser_connection_banner = QtWidgets.QLabel()
    ui._reset_laser_message_scroll = Mock()
    ui.nktpbus_mode_switch = QtWidgets.QPushButton()
    ui.switch_to_cli_button = QtWidgets.QPushButton()
    ui._laser_status_timer = QtCore.QTimer()
    ui._laser_status_timer.start(1000)
    yield ui
    ui._laser_status_timer.stop()


@pytest.mark.parametrize('connected', [True, False])
def test_gui_bus_handover_releases_cli_and_does_not_need_a_reply(mode_gui, monkeypatch, connected):
    from pyccapt.control.nkt_photonics import nktpbus_switch
    ui = mode_gui
    old = ui.laser_device
    if not connected:
        ui.laser_device = None
    ui.on_mode = ui.change_laser_power = True
    if connected:
        ui._sync_controls_from_device(initial=True)

    def handover(port):
        assert port == 'FAKE'
        assert ui.laser_device is None
        if connected:
            old.close_port.assert_called_once()
        assert not ui._laser_status_timer.isActive()
        assert not ui.nktpbus_mode_switch.isEnabled()
        assert not ui.switch_to_cli_button.isEnabled()
        assert not ui.on_mode and not ui.change_laser_power
        # Simulate a queued timer callback and a second button click while
        # Qt processes events: neither can open/poll the port mid-handover.
        ui.check_laser_status = Mock()
        ui._poll_laser_status()
        ui.check_laser_status.assert_not_called()
        ui.switch_to_cli_clicked()

    bus = Mock(side_effect=handover)
    cli = Mock()
    monkeypatch.setattr(nktpbus_switch, 'switch_to_nktpbus', bus)
    monkeypatch.setattr(nktpbus_switch, 'is_cli_responding', cli)
    ui.switch_to_nktpbus_mode()
    bus.assert_called_once_with('FAKE')
    cli.assert_not_called()
    assert ui.laser_device is None
    assert not ui.variables.flag_laser_connected
    assert not ui.variables.laser_telemetry['valid']
    assert not ui.laser_on.isEnabled()
    assert ui._laser_status_timer.isActive()
    assert ui.switch_to_cli_button.isEnabled()
    assert 'released' in ui.errors[-1]


def test_cli_handover_closes_sdk_before_reconnect_and_retries(mode_gui, monkeypatch):
    from pyccapt.control.nkt_photonics import nktpbus_switch
    ui = mode_gui
    events = []
    old = ui.laser_device
    monkeypatch.setattr(nktpbus_switch, 'is_cli_responding', Mock(return_value=False))

    def sdk_switch(port, **options):
        assert not ui._laser_status_timer.isActive()
        assert ui.laser_device is None
        old.close_port.assert_called_once()
        events.append('sdk released')

    monkeypatch.setattr(nktpbus_switch, 'switch_to_cli', sdk_switch)
    ui._wait_with_gui_pump = Mock()

    def reopen(port):
        assert events[0] == 'sdk released'
        assert not ui._laser_status_timer.isActive()
        events.append('reopen')
        if len(events) == 2:
            return False
        ui.laser_device = old
        ui._last_laser_status = 'ly_oxp2_dev_status 9'
        ui._set_laser_disconnected_banner('')
        return True

    ui._open_laser_cli = Mock(side_effect=reopen)
    ui.switch_to_cli_clicked()
    assert events == ['sdk released', 'reopen', 'reopen']
    assert ui.variables.flag_laser_connected
    assert ui.laser_standby.isEnabled()
    assert ui._laser_status_timer.isActive()
    assert not ui._laser_mode_switch_in_progress


@pytest.mark.parametrize('method', ['switch_to_nktpbus_mode', 'switch_to_cli_clicked'])
def test_handover_failure_is_retryable_without_restart(mode_gui, monkeypatch, method):
    from pyccapt.control.nkt_photonics import nktpbus_switch
    ui = mode_gui
    failed = Mock(side_effect=nktpbus_switch.NKTPSwitchError('Disconnect NKT CONTROL'))
    monkeypatch.setattr(nktpbus_switch, 'switch_to_nktpbus', failed)
    monkeypatch.setattr(nktpbus_switch, 'is_cli_responding', failed)
    getattr(ui, method)()
    assert ui.laser_device is None
    assert not ui._laser_mode_switch_in_progress
    assert ui._laser_status_timer.isActive()
    assert ui.nktpbus_mode_switch.isEnabled()
    assert ui.switch_to_cli_button.isEnabled()
    assert 'Disconnect NKT CONTROL' in ui.errors[-1]
    # Retry in the same GUI instance.
    monkeypatch.setattr(nktpbus_switch, 'switch_to_nktpbus', Mock())
    ui.switch_to_nktpbus_mode()
    assert 'released' in ui.errors[-1]


def test_close_failure_prevents_opening_another_handle_and_is_retryable(mode_gui, monkeypatch):
    from pyccapt.control.nkt_photonics import nktpbus_switch
    ui = mode_gui
    ui.laser_device.close_port.side_effect = OSError('close failed')
    bus = Mock()
    monkeypatch.setattr(nktpbus_switch, 'switch_to_nktpbus', bus)
    ui.switch_to_nktpbus_mode()
    bus.assert_not_called()
    assert 'close failed' in ui.errors[-1]
    assert not ui._laser_mode_switch_in_progress
    assert ui._laser_status_timer.isActive()
    ui.laser_device.close_port.side_effect = None
    ui.switch_to_nktpbus_mode()
    bus.assert_called_once_with('FAKE')


@pytest.mark.parametrize('method', ['switch_to_nktpbus_mode', 'switch_to_cli_clicked'])
@pytest.mark.parametrize('busy', ['experiment', 'alignment', 'poll', 'shutdown'])
def test_handover_does_not_interrupt_other_laser_work(mode_gui, monkeypatch, method, busy):
    from pyccapt.control.nkt_photonics import nktpbus_switch
    ui = mode_gui
    if busy == 'experiment':
        ui.variables.start_flag = True
    elif busy == 'alignment':
        ui.variables.laser_alignment_status = {'active': True}
    elif busy == 'poll':
        ui._laser_status_in_progress = True
    else:
        ui._laser_stopping = True
    bus, probe = Mock(), Mock()
    monkeypatch.setattr(nktpbus_switch, 'switch_to_nktpbus', bus)
    monkeypatch.setattr(nktpbus_switch, 'is_cli_responding', probe)
    old = ui.laser_device
    getattr(ui, method)()
    bus.assert_not_called()
    probe.assert_not_called()
    old.close_port.assert_not_called()


def test_shutdown_during_settle_does_not_reopen_port(mode_gui, monkeypatch):
    from pyccapt.control.nkt_photonics import nktpbus_switch
    ui = mode_gui
    monkeypatch.setattr(nktpbus_switch, 'is_cli_responding', Mock(return_value=False))
    monkeypatch.setattr(nktpbus_switch, 'switch_to_cli', lambda port, **options: options['wait'](1.))
    def shutdown(seconds):
        ui._laser_stopping = True
    ui._wait_with_gui_pump = shutdown
    ui._open_laser_cli = Mock()
    ui.switch_to_cli_clicked()
    ui._open_laser_cli.assert_not_called()
    assert not ui._laser_status_timer.isActive()
    assert not ui.switch_to_cli_button.isEnabled()


def test_failed_listen_handshake_releases_cli_instead_of_claiming_connection(mode_gui, monkeypatch):
    from pyccapt.control.nkt_photonics import origamiClassCLI
    ui = mode_gui
    device = Mock(last_error=None)
    device.open_port.return_value = 0
    device.StatusRead.return_value = 'ly_oxp2_dev_status 33'
    device.Listen.side_effect = TimeoutError('no Listen reply')
    monkeypatch.setattr(origamiClassCLI, 'origClass', lambda port: device)
    assert not ui._open_laser_cli('FAKE')
    device.close_port.assert_called_once()
    assert ui.laser_device is None
    assert not ui.variables.flag_laser_connected
    assert 'initialization failed' in ui.laser_connection_banner.text()


@pytest.mark.parametrize('switch_result', [0, 6])
def test_repeated_gui_round_trips_with_exclusive_port_and_external_control(mode_gui, monkeypatch, switch_result):
    from pyccapt.control.nkt_photonics import nktpbus_switch, origamiClassCLI
    ui = mode_gui
    state = SimpleNamespace(owner=None, mode='cli', elapsed=0.)
    commands = []

    class Port:
        def __init__(self, **options):
            if state.owner is not None:
                raise PermissionError('Access denied: port is already owned')
            assert options['baudrate'] == 38400
            state.owner = self
            self.pending = b''

        def reset_input_buffer(self):
            self.pending = b''

        def write(self, command):
            commands.append(command)
            if state.mode != 'cli':
                return len(command)
            if command == b'ly_oxp2_nktpbus=1\n':
                state.mode = 'bus'
                return len(command)  # Real firmware supplies no CLI reply.
            replies = {
                b'ly_oxp2_dev_status?\r\n': 'ly_oxp2_dev_status 9',
                b'ly_oxp2_listen\r\n': 'ly_oxp2_listen=ok',
                b'e_freq_available?\r\n': 'e_freq=4 --> 400000 Hz',
                b'e_freq?\r\n': 'Frequency index parameter: 4',
                b'e_div?\r\n': 'Division factor for PCAOM is 10',
                b'e_power?\r\n': 'Setted laser power in relative unit: 800',
                b'ls_wavelength?\r\n': 'NKT FHG Current position=DUV',
                b'e_mlp?\r\n': '12300 nJ',
                b'ls_output_power?\r\n': 'NKT FHG Output power=0.53W',
            }
            self.pending = (replies[command]+'\n>').encode('ascii')
            return len(command)

        @property
        def in_waiting(self):
            return len(self.pending)

        def read(self, count):
            state.elapsed += .05
            data, self.pending = self.pending[:count], self.pending[count:]
            return data

        def flush(self):
            pass

        def close(self):
            assert state.owner is self
            state.owner = None

    class SDK:
        def openPorts(self, port, auto, live):
            assert state.owner is None and state.mode == 'bus'
            state.owner = self
            return 0

        def deviceGetAllTypes(self, port):
            return 0, b'\x00\x95' + bytes(253)

        def registerWriteU8(self, port, address, register, value, index):
            assert state.owner is self
            assert (register, value, index) == (0x39, 1, -1)
            state.mode = 'cli'
            return switch_result

        def RegisterResultTypes(self, result):
            return '6:RegResultTimeout' if result == 6 else '0:RegResultSuccess'

        def closePorts(self, port):
            assert state.owner is self
            state.owner = None
            return 0

    monkeypatch.setattr(origamiClassCLI.serial, 'Serial', Port)
    monkeypatch.setattr(origamiClassCLI, 'time', SimpleNamespace(monotonic=lambda: state.elapsed))
    monkeypatch.setattr(nktpbus_switch, '_load_nktp_dll_module', SDK)
    monkeypatch.setattr(nktpbus_switch.time, 'sleep', lambda seconds: None)
    ui._wait_with_gui_pump = Mock()
    assert ui._open_laser_cli('FAKE')
    for _ in range(3):
        ui.on_mode = ui.change_laser_power = True
        ui.switch_to_nktpbus_mode()
        assert state.mode == 'bus' and state.owner is None
        assert not ui.on_mode and not ui.change_laser_power
        assert not ui.variables.flag_laser_connected
        # Another application can acquire the port while PyCCAPT stays open.
        state.owner = 'NKT CONTROL'
        ui.switch_to_cli_clicked()
        assert state.owner == 'NKT CONTROL' and state.mode == 'bus'
        assert 'Disconnect the laser in NKT CONTROL' in ui.errors[-1]
        state.owner = None
        ui.switch_to_cli_clicked()
        assert state.mode == 'cli' and state.owner is ui.laser_device.ser
        assert ui.variables.flag_laser_connected
        assert ui._laser_status_timer.isActive()
        assert ui.laser_standby.isEnabled()
        assert ui.variables.laser_telemetry['valid']
    assert commands.count(b'ly_oxp2_nktpbus=1\n') == 3
    assert not any(command.startswith(b'ly_oxp2_enabled') for command in commands)
    ui._close_laser_device()
    assert state.owner is None


@pytest.fixture
def watt_gui(mode_gui, monkeypatch):
    from pyccapt.control.devices import nkt_ir_power
    ui = mode_gui
    ui._laser_stopping = False
    ui._sync_controls_from_device(initial=True)
    ui.laser_ir_power_control = QtWidgets.QDoubleSpinBox()
    ui.laser_ir_power_control.setRange(0, 5.2)
    ui.laser_ir_power_control.setDecimals(3)
    ui.laser_ir_power_control.setValue(1.)
    ui.laser_ir_power_apply = QtWidgets.QPushButton()
    ui._wait_with_gui_pump = Mock()
    device = ui.laser_device
    conversion = Mock(return_value=(1., 640))
    monkeypatch.setattr(nkt_ir_power, 'calibrated_aom', conversion)
    def reconnect(port):
        assert ui.laser_device is None
        assert not ui._laser_status_timer.isActive()
        ui.laser_device = device
        device.code = 9
        ui._set_laser_disconnected_banner('')
        return True
    ui._open_laser_cli = Mock(side_effect=reconnect)
    return ui, device, conversion


def test_watt_apply_uses_firmware_conversion_and_confirms_aom_in_listen(watt_gui):
    ui, device, conversion = watt_gui
    ui._apply_ir_power_watts()
    assert device.calls == [('AOM', 640)]
    assert ui.laser_power.value() == 16.
    assert ui.laser_ir_power_control.value() == 1.
    assert ui.laser_device is device and device.code == 9
    assert ui._laser_status_timer.isActive()
    assert ui.laser_ir_power_apply.isEnabled()
    assert '1 W applied as calibrated AOM 16%' in ui.errors[-1]
    options = conversion.call_args.kwargs
    assert options['base_hz'] == 400000 and options['divider'] == 10
    assert options['max_watts'] == 5.2
    assert ui.laser_power_disp.value() == .53  # Selected DUV monitor stays separate.


@pytest.mark.parametrize('busy', ['experiment', 'alignment', 'emitting', 'external', 'pending', 'switch'])
def test_watt_request_is_blocked_without_touching_hardware_when_not_allowed(watt_gui, busy):
    ui, device, conversion = watt_gui
    if busy == 'experiment':
        ui.variables.start_flag = True
    elif busy == 'alignment':
        ui.variables.laser_alignment_status = {'active': True}
    elif busy == 'emitting':
        device.code = 129
    elif busy == 'external':
        device.ModeRead = lambda: 'e_mode of AOM: 3'
    elif busy == 'pending':
        ui.change_laser_power = True
    else:
        ui._laser_mode_switch_in_progress = True
    ui._apply_ir_power_watts()
    conversion.assert_not_called()
    device.close_port.assert_not_called()
    assert not device.calls


@pytest.mark.parametrize('failure', ['conversion', 'timing', 'wavelength', 'aom', 'reconnect'])
def test_failed_watt_request_is_not_reported_as_applied(watt_gui, failure):
    ui, device, conversion = watt_gui
    if failure == 'conversion':
        conversion.side_effect = ValueError('Firmware refused watts')
    elif failure in ('timing', 'wavelength'):
        def changing(port):
            if failure == 'timing':
                device.FreqRead = lambda: 'Frequency index parameter: 6'
            else:
                device.wavelength_read = lambda: 'Current position=IR'
            ui.laser_device = device
            device.code = 9
            return True
        ui._open_laser_cli.side_effect = changing
    elif failure == 'aom':
        device.AOM = lambda value: device.calls.append(('AOM', value))  # Simulate firmware refusing it.
    else:
        ui._open_laser_cli.side_effect = None
        ui._open_laser_cli.return_value = False
    ui._apply_ir_power_watts()
    assert 'applied as calibrated' not in ui.errors[-1]
    assert not ui._laser_mode_switch_in_progress
    assert ui._laser_status_timer.isActive()
    if failure != 'aom':
        assert not device.calls
    if failure == 'conversion':
        ui._open_laser_cli.assert_called_once()
        assert 'Firmware refused watts' in ui.errors[-1]
        assert ui.variables.flag_laser_connected
