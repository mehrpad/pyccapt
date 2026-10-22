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
        self.syncing = False

    def reset_input_buffer(self): pass
    def write(self, command):
        self.commands.append(command)
        self.syncing = command == b'\n'
    def flush(self): pass
    def close(self): self.closed = True
    @property
    def in_waiting(self): return 0

    def read(self, count):
        self.elapsed += .05
        if self.syncing:
            return b''
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
    assert port.commands == [b'\n', b'ly_oxp2_dev_status?\r\n']
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
    for name, color in (('led_red', 'red'), ('led_green', 'green'), ('led_orange', 'orange')):
        pixmap = QtGui.QPixmap(2, 2)
        pixmap.fill(QtGui.QColor(color))
        setattr(ui, name, pixmap)
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


@pytest.mark.parametrize('wavelength,reply,watts', [
    ('Green', 'NKT SHG\tOutput power=120mW\nNKT FHG\tOutput power=7mW', .120),
    ('DUV', 'NKT SHG\tOutput power=120mW\nNKT FHG\tOutput power=7mW', .007),
    ('Green', 'NKT SHG\tOutput power=0mW\nNKT FHG\tOutput power=0mW', 0.),
    ('DUV', 'NKT SHG\tOutput power=0mW\nNKT FHG\tOutput power=0mW\nListen=ok', 0.),
    ('Green', 'NKT SHG\tOutput power=10mW\nNKT FHG\tUnknown command', .010),
    ('UV', 'NKT SHG\tOutput power=120mW\nNKT THG\tOutput power=3mW', .003),
    ('DUV', 'Output power=0.53W\nListen=ok', .53),
])
def test_selected_harmonic_is_read_from_combined_controller_reply(laser_gui, wavelength, reply, watts):
    ui = laser_gui
    ui.laser_device.wavelength_read = lambda: f'Current position={wavelength}'
    ui.laser_device.harmonic = reply
    ui._sync_controls_from_device(initial=True)
    data = ui.variables.laser_telemetry
    assert data['valid'] and 'error' not in data
    assert ui.laser_power_disp.value() == pytest.approx(watts)
    assert ui.laser_pulse_energy_disp.value() == pytest.approx(watts / 40000 * 1e6)
    assert data['raw']['ls_output_power'] == reply
    assert data['raw']['ls_selected_output_power']
    assert not ui.laser_device.calls


@pytest.mark.parametrize('reply', [
    'NKT FHG\tOutput power=0.53W',  # Not a Green monitor, even if it is the only reading.
    'NKT SHG\tOutput power=0mW\nNKT SHG\tOutput power=0mW\nNKT FHG\tOutput power=1W',
    'NKT SHG\tOutput power=unknown\nNKT FHG\tOutput power=1W',
    'NKT SHG\tOutput power=3nJ\nNKT FHG\tOutput power=1W',
])
def test_ambiguous_or_missing_selected_harmonic_never_uses_other_module(laser_gui, reply):
    ui = laser_gui
    ui.laser_device.wavelength_read = lambda: 'Current position=Green'
    ui.laser_device.harmonic = reply
    ui._sync_controls_from_device(initial=True)
    data = ui.variables.laser_telemetry
    assert not data['valid'] and math.isnan(data['output_power_mw'])
    assert math.isnan(data['pulse_energy_nj'])


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
    (65, {'listen', 'standby', 'on', 'output'}), (129, {'listen', 'standby', 'on', 'output'}),
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


@pytest.mark.parametrize('start,handler,command,target,led,green', [
    (9, 'laser_standby_clicked', 'Standby', 33, 'led_laser_laser_standby', True),
    (33, 'laser_on_clicked', 'Enable', 129, 'led_laser_on', True),
    (65, 'laser_enable_clicked', 'AOMEnable', 129, 'led_laser_enable', True),
    (129, 'laser_enable_clicked', 'AOMDisable', 65, 'led_laser_enable', False),
    (129, 'laser_listen_clicked', 'Listen', 9, 'led_laser_listen', True),
    (129, 'laser_on_clicked', 'Standby', 33, 'led_laser_on', False),
])
def test_pending_leds_blink_until_fresh_target_readback(laser_gui, start, handler, command, target, led, green):
    ui = laser_gui
    device = ui.laser_device
    device.code = start
    setattr(device, command, lambda: device.calls.append((command,)))  # Delayed firmware transition.
    ui._apply_button_locks_for_status(f'ly_oxp2_dev_status {start}')
    getattr(ui, handler)()
    ui.check_laser_status()
    widget = getattr(ui, led)
    before = widget.pixmap().cacheKey()
    ui._blink_laser_indicators()
    after = widget.pixmap().cacheKey()
    assert {before, after} == {ui.led_red.cacheKey(), ui.led_orange.cacheKey()}
    # Re-rendering a cached UI state must not confirm a hardware request.
    ui._apply_button_locks_for_status(f'ly_oxp2_dev_status {target}')
    before = widget.pixmap().cacheKey()
    ui._blink_laser_indicators()
    assert {before, widget.pixmap().cacheKey()} == {ui.led_red.cacheKey(), ui.led_orange.cacheKey()}
    assert device.calls == [(command,)]  # Animation never performs hardware I/O.
    device.code = target
    ui.check_laser_status()
    expected = ui.led_green if green else ui.led_red
    assert widget.pixmap().cacheKey() == expected.cacheKey()
    ui._blink_laser_indicators()
    assert widget.pixmap().cacheKey() == expected.cacheKey()


def test_failed_status_stops_blinking_and_discards_requests(laser_gui):
    ui = laser_gui
    parent = QtWidgets.QWidget()
    ui._setup_laser_indicators(parent)
    ui.laser_device.code = 9
    ui._apply_button_locks_for_status('ly_oxp2_dev_status 9')
    ui.laser_standby_clicked()
    assert ui._laser_led_timer.isActive()
    before = ui.led_laser_laser_standby.pixmap().cacheKey()
    ui._laser_led_timer.timeout.emit()
    assert {before, ui.led_laser_laser_standby.pixmap().cacheKey()} == {
        ui.led_red.cacheKey(), ui.led_orange.cacheKey()}
    assert not ui.laser_device.calls
    ui.laser_device.StatusRead = Mock(side_effect=TimeoutError('Lost status'))
    ui._laser_status_in_progress = False
    ui._poll_laser_status()
    assert not ui._laser_led_timer.isActive()
    assert not ui._laser_indicator_requests
    assert ui.led_laser_laser_standby.pixmap().cacheKey() == ui.led_red.cacheKey()
    parent.deleteLater()


def test_listen_keeps_blinking_through_warning_until_stopped_fsm(laser_gui):
    ui = laser_gui
    parent = QtWidgets.QWidget()
    ui._setup_laser_indicators(parent)
    device = ui.laser_device
    device.protocol = 'nktpbus'
    device.code = 5
    device.status_fault = 'interlock waiting for reset'
    device.status_details = dict(fsm=2, error=0, interlock=1)
    device.Listen = lambda: device.calls.append(('Listen',))
    ui.laser_state_label = QtWidgets.QLabel()
    ui._apply_button_locks_for_status(device.StatusRead())
    ui.laser_listen_clicked()
    for _ in range(2):
        ui.check_laser_status()
        assert ui._laser_led_timer.isActive()
        before = ui.led_laser_listen.pixmap().cacheKey()
        ui._laser_led_timer.timeout.emit()
        assert {before, ui.led_laser_listen.pixmap().cacheKey()} == {
            ui.led_red.cacheKey(), ui.led_orange.cacheKey()}
    assert device.calls == [('Listen',)]
    device.code = 9  # Native FSM now says Listen; interlock still needs reset.
    device.status_details['fsm'] = 1
    ui.check_laser_status()
    assert not ui._laser_led_timer.isActive()
    assert ui.led_laser_listen.pixmap().cacheKey() == ui.led_green.cacheKey()
    assert ui.laser_listen.isEnabled()
    assert not ui.laser_standby.isEnabled() and not ui.laser_on.isEnabled()
    assert not ui.laser_power.isEnabled()
    assert 'interlock waiting for reset' in ui.laser_state_label.text()
    # A queued/programmatic request cannot bypass the fault lock.
    ui.standby_mode = True
    ui.check_laser_status()
    assert device.calls == [('Listen',)]
    parent.deleteLater()


@pytest.mark.parametrize('code', [65,129])
def test_laser_off_and_output_gate_have_distinct_actions(laser_gui, code):
    ui = laser_gui
    ui.laser_device.code = code
    ui._apply_button_locks_for_status(f'ly_oxp2_dev_status {code}')
    assert ui.laser_on.text() == 'Laser Off'
    assert ui.laser_enable.text() == ('Close Output' if code == 129 else 'Open Output')
    ui.laser_on_clicked()
    ui.check_laser_status()
    assert ui.laser_device.calls == [('Standby',)]
    assert ui.laser_device.code == 33


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


def test_startup_retries_a_settling_cli_without_changing_protocol(mode_gui, monkeypatch):
    from pyccapt.control.nkt_photonics import origamiClassCLI
    ui = mode_gui
    device = Laser()
    device.open_port = Mock(return_value=0)
    device.StatusRead = Mock(side_effect=[TimeoutError('first reply missing'),
                                         'ly_oxp2_dev_status 9', 'ly_oxp2_dev_status 9',
                                         'ly_oxp2_dev_status 9'])
    monkeypatch.setattr(origamiClassCLI, 'origClass', lambda port: device)
    assert ui._open_laser_cli('FAKE', initial_open=True)
    assert ui.laser_device is device
    assert not device.calls
    device.open_port.assert_called_once()
    device.close_port.assert_not_called()


@pytest.mark.parametrize('blocked', [None, 'connected', 'shutdown', 'handover', 'experiment', 'alignment'])
def test_startup_recovery_only_retries_cli_without_switching_protocol(mode_gui, blocked):
    ui = mode_gui
    ui.laser_device = ui.laser_device if blocked == 'connected' else None
    ui._laser_stopping = blocked == 'shutdown'
    ui._laser_mode_switch_in_progress = blocked == 'handover'
    ui.variables.start_flag = blocked == 'experiment'
    ui.variables.laser_alignment_status = {'active': blocked == 'alignment'}
    ui.switch_to_cli_clicked = Mock(side_effect=AssertionError('No startup protocol changes'))
    ui._open_laser_cli = Mock(return_value=False)
    ui._recover_laser_startup()
    assert ui._open_laser_cli.call_count == (1 if blocked is None else 0)
    ui.switch_to_cli_clicked.assert_not_called()


def test_manual_mode_handover_cancels_pending_startup_recovery(mode_gui):
    from PyQt6 import QtCore
    ui = mode_gui
    ui._laser_startup_retry_timer = QtCore.QTimer()
    ui._laser_startup_retry_timer.setSingleShot(True)
    ui._laser_startup_retry_timer.start(1000)
    with ui._laser_mode_transition('manual test') as port:
        assert port == 'FAKE'
        assert not ui._laser_startup_retry_timer.isActive()


def test_cli_switch_releases_existing_session_before_probe(laser_gui, monkeypatch):
    from pyccapt.control.nkt_photonics import nktpbus_switch
    ui = laser_gui
    ui.flag_super_user_stage = True
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


@pytest.mark.parametrize('code', [9, 17, 33, 65, 129])
def test_manual_cli_reconnect_preserves_observed_state_without_setters(mode_gui, monkeypatch, code):
    from pyccapt.control.nkt_photonics import origamiClassCLI
    ui = mode_gui
    ui._laser_last_stable_code = 9  # A previous connection must not color this one.
    device = Laser()
    device.code = code
    device.open_port = Mock(return_value=0)
    device.Listen = Mock(side_effect=AssertionError('Connecting must not request Listen'))
    monkeypatch.setattr(origamiClassCLI, 'origClass', lambda port: device)
    assert ui._open_laser_cli('FAKE')
    assert ui.laser_device is device and device.code == code
    assert not device.calls
    device.Listen.assert_not_called()
    device.close_port.assert_not_called()
    if code == 17:
        assert ui.led_laser_listen.pixmap().cacheKey() == ui.led_red.cacheKey()


@pytest.mark.parametrize('code', [17, 33, 129])
def test_cli_connection_keeps_latest_status_during_settings_refresh(mode_gui, monkeypatch, code):
    from pyccapt.control.nkt_photonics import origamiClassCLI
    ui = mode_gui
    device = Laser()
    device.open_port = Mock(return_value=0)
    device.StatusRead = Mock(side_effect=['ly_oxp2_dev_status 9', f'ly_oxp2_dev_status {code}'])
    monkeypatch.setattr(origamiClassCLI, 'origClass', lambda port: device)
    assert ui._open_laser_cli('FAKE', initial_open=True)
    assert readback.scalar(ui._last_laser_status) == code
    assert ui.laser_on.isEnabled() is (code in (33, 129))
    assert not device.calls


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
            if command == b'\n':
                self.pending = b''
                return len(command)
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


@pytest.mark.parametrize('code', [9,33,65,129])
def test_cli_watt_control_is_disabled_without_protocol_handover(watt_gui, code):
    ui, device, conversion = watt_gui
    device.code = code
    ui._apply_button_locks_for_status(f'ly_oxp2_dev_status {code}')
    assert not ui.laser_ir_power_control.isEnabled()
    ui._apply_ir_power_watts()
    conversion.assert_not_called()
    device.close_port.assert_not_called()
    ui._open_laser_cli.assert_not_called()
    assert not device.calls
    assert ui.laser_device is device and device.code == code
    assert ui._laser_status_timer.isActive()
    assert 'power selector' in ui.errors[-1]


@pytest.mark.parametrize('busy', ['experiment', 'alignment', 'pending', 'switch', 'stopping', 'poll'])
def test_watt_request_is_blocked_without_touching_hardware_when_not_allowed(watt_gui, busy):
    ui, device, conversion = watt_gui
    device.protocol = 'nktpbus'
    device.set_ir_power_watts = Mock()
    if busy == 'experiment':
        ui.variables.start_flag = True
    elif busy == 'alignment':
        ui.variables.laser_alignment_status = {'active': True}
    elif busy == 'pending':
        ui.change_laser_power = True
    elif busy == 'switch':
        ui._laser_mode_switch_in_progress = True
    elif busy == 'stopping':
        ui._laser_stopping = True
    else:
        ui._laser_status_in_progress = True
    ui._apply_ir_power_watts()
    conversion.assert_not_called()
    device.set_ir_power_watts.assert_not_called()
    device.close_port.assert_not_called()
    assert not device.calls


def test_failed_native_watt_request_is_not_replayed_or_reported_as_applied(watt_gui):
    ui, device, conversion = watt_gui
    device.protocol = 'nktpbus'
    device.set_ir_power_watts = Mock(side_effect=ValueError('Firmware refused watts'))
    ui._mark_laser_watt_edit(1.)
    ui._commit_laser_watt_edit()
    ui._commit_laser_watt_edit()  # Another focus-out must not repeat a refused write.
    device.set_ir_power_watts.assert_called_once_with(1.)
    assert 'Firmware refused watts' in ui.errors[-1]
    assert 'accepted' not in ui.errors[-1]
    assert not getattr(ui, '_laser_mode_switch_in_progress', False)
    assert ui._laser_status_timer.isActive()
    assert not device.calls
    conversion.assert_not_called()
    ui._open_laser_cli.assert_not_called()
    device.close_port.assert_not_called()


@pytest.mark.parametrize('code', [65,129])
@pytest.mark.parametrize('wavelength', ['IR', 'Green', 'DUV'])
def test_native_gui_watts_while_on_do_not_handover_or_send_state_commands(watt_gui, code, wavelength):
    ui, device, conversion = watt_gui
    device.protocol = 'nktpbus'
    device.code = code
    device.wavelength_basis = f'{wavelength} selected before native handover'
    device.supports_wavelength_changes = False
    device.wavelength_read = lambda: f'CLI-verified position={wavelength}'
    device.read_average_power = lambda: '1.0 W'
    device.power_read_dv_green = lambda: 'Unknown: native harmonic monitor scaling unverified'
    device.set_ir_power_watts = Mock(return_value=1.)
    ui._apply_button_locks_for_status(f'ly_oxp2_dev_status {code}')
    assert ui.laser_ir_power_control.isEnabled()
    ui._apply_ir_power_watts()
    device.set_ir_power_watts.assert_called_once_with(1.)
    assert device.code == code
    assert not device.calls
    device.close_port.assert_not_called()
    conversion.assert_not_called()
    ui._open_laser_cli.assert_not_called()
    assert 'operating state retained' in ui.errors[-1]
    assert 'Listen' not in ui.laser_ir_power_control.toolTip()
    assert ui.laser_wavelegnth.currentText() == wavelength
    assert not ui.laser_wavelegnth.isEnabled()
    data = ui.variables.laser_telemetry
    assert data['ir_power_mw'] == 1000.
    assert wavelength in ui.laser_wavelegnth.toolTip()
    if wavelength != 'IR':
        assert not data['valid']
        assert math.isnan(data['output_power_mw']) and math.isnan(data['pulse_energy_nj'])
        assert 'unavailable' in data['source']


def test_native_handover_is_refused_while_warming(watt_gui, monkeypatch):
    from pyccapt.control.nkt_photonics import nktpbus_switch
    ui, device, conversion = watt_gui
    device.code = 17
    device.wavelength_read = lambda: 'Current position=IR'
    switch = Mock()
    monkeypatch.setattr(nktpbus_switch, 'switch_to_nktpbus', switch)
    ui._use_native_ir_control()
    switch.assert_not_called()
    device.close_port.assert_not_called()
    assert 'stable laser state' in ui.errors[-1]


def test_native_handover_requires_verified_wavelength_selection(watt_gui, monkeypatch):
    from pyccapt.control.nkt_photonics import nktpbus_switch
    ui, device, conversion = watt_gui
    device.wavelength_read = lambda: 'Moving'
    switch = Mock()
    monkeypatch.setattr(nktpbus_switch, 'switch_to_nktpbus', switch)
    ui._use_native_ir_control()
    switch.assert_not_called()
    device.close_port.assert_not_called()
    assert 'valid selected wavelength' in ui.errors[-1]


@pytest.mark.parametrize('wavelength,index', [('IR',0), ('Green',1), ('DUV',3)])
@pytest.mark.parametrize('code', [33,129])
def test_native_handover_opens_one_owned_sdk_session_then_restores_polling(watt_gui, monkeypatch, wavelength, index, code):
    from pyccapt.control.gui import laser_power_control
    from pyccapt.control.nkt_photonics import nktpbus_switch
    ui, old, conversion = watt_gui
    old.wavelength_read = lambda: f'Current position={wavelength}'
    old.code = code
    ui.flag_super_user_stage = False
    native = Laser()
    native.code = 9
    native.protocol = 'nktpbus'
    native.wavelength_basis = f'{wavelength} verified before handover'
    native.supports_wavelength_changes = False
    native.wavelength_read = lambda: f'CLI-verified position={wavelength}'
    native.open_port = Mock(return_value=0)
    monkeypatch.setattr(nktpbus_switch, '_load_nktp_dll_module', Mock())
    def switch(port):
        old.close_port.assert_called_once()
        assert ui.laser_device is None
        assert not ui._laser_status_timer.isActive()
    monkeypatch.setattr(nktpbus_switch, 'switch_to_nktpbus', switch)
    factory = Mock(return_value=native)
    monkeypatch.setattr(laser_power_control, 'NativeLaser', factory)
    ui._use_native_ir_control()
    assert ui.laser_device is native
    assert factory.call_args.kwargs['wavelength'] == index
    native.open_port.assert_called_once()
    native.close_port.assert_not_called()
    assert ui.variables.flag_laser_connected
    assert ui._laser_status_timer.isActive()
    assert not ui.laser_wavelegnth.isEnabled()
    assert native.calls == [('Listen',)]  # Connecting explicitly stops; never resumes emission.
    assert native.code == 9
    assert ui.laser_wavelegnth.currentText() == wavelength


@pytest.mark.parametrize('wavelength', ['IR', 'Green', 'DUV'])
@pytest.mark.parametrize('optics_available', [True, False])
def test_cli_standby_and_laser_on_keep_the_working_connection(watt_gui, monkeypatch, wavelength, optics_available):
    from pyccapt.control.gui import laser_power_control
    from pyccapt.control.nkt_photonics import nktpbus_switch
    ui, device, _ = watt_gui
    device.code = 9
    device.wavelength_read = lambda: f'Current position={wavelength}'
    if not optics_available:
        device.read_average_power = Mock(side_effect=ValueError('Monitor unavailable'))
    def warmup():
        device.calls.append(('Standby',))
        device.code = 17
    device.Standby = warmup
    native_factory = Mock(side_effect=AssertionError('Standby must not open native watts'))
    load_sdk = Mock(side_effect=AssertionError('CLI controls must not require the SDK'))
    bus_switch = Mock(side_effect=AssertionError('State buttons must not switch protocols'))
    monkeypatch.setattr(laser_power_control, 'NativeLaser', native_factory)
    monkeypatch.setattr(nktpbus_switch, '_load_nktp_dll_module', load_sdk)
    monkeypatch.setattr(nktpbus_switch, 'switch_to_nktpbus', bus_switch)
    ui._publish_laser_state(device.StatusRead())
    ui._apply_button_locks_for_status(device.StatusRead())
    ui.laser_standby_clicked()
    ui.check_laser_status()
    assert device.code == 17 and ui._laser_standby_pending
    assert device.calls == [('Standby',)]
    assert ui.led_laser_listen.pixmap().cacheKey() == ui.led_green.cacheKey()
    before = ui.led_laser_laser_standby.pixmap().cacheKey()
    ui._blink_laser_indicators()
    assert {before, ui.led_laser_laser_standby.pixmap().cacheKey()} == {
        ui.led_red.cacheKey(), ui.led_orange.cacheKey()}
    device.code = 33
    ui.check_laser_status()
    assert ui.laser_on.isEnabled()
    assert ui.led_laser_laser_standby.pixmap().cacheKey() == ui.led_green.cacheKey()
    ui.laser_on_clicked()
    ui.check_laser_status()
    assert device.code == 129 and ui.laser_device is device
    assert device.calls == [('Standby',), ('Enable',)]
    assert not ui.laser_ir_power_control.isEnabled()  # Explicit native connection is separate.
    device.close_port.assert_not_called()
    ui._open_laser_cli.assert_not_called()
    native_factory.assert_not_called()
    load_sdk.assert_not_called()
    bus_switch.assert_not_called()


@pytest.mark.parametrize('accepted', [True, False])
def test_explicit_watt_connection_restores_verified_timing_in_listen(watt_gui, monkeypatch, accepted):
    from pyccapt.control.gui import laser_power_control
    from pyccapt.control.nkt_photonics import nktpbus_switch
    ui, old, _ = watt_gui
    old.code = 9
    old.DivRead = lambda: 'Division factor: 4'
    divider = [1]  # Simulate the reset restoring a default divider.
    native = Laser()
    native.protocol = 'nktpbus'
    native.code = 9
    native.wavelength_basis = 'DUV read back before handover'
    native.open_port = Mock(return_value=0)
    native.DivRead = lambda: f'Division factor: {divider[0]}'
    def set_divider(value):
        assert native.code == 9
        native.calls.append(('Div', value))
        if accepted:
            divider[0] = value
    native.Div = set_divider
    monkeypatch.setattr(laser_power_control, 'NativeLaser', Mock(return_value=native))
    monkeypatch.setattr(nktpbus_switch, '_load_nktp_dll_module', Mock())
    monkeypatch.setattr(nktpbus_switch, 'switch_to_nktpbus', Mock())
    ui._apply_button_locks_for_status('ly_oxp2_dev_status 9')
    connected = bool(ui._use_native_ir_control())
    assert native.calls == [('Listen',), ('Div', 4)]
    assert not ui.on_mode
    assert connected is accepted
    assert not ui.standby_mode
    if accepted:
        assert ui.laser_divition_factor.value() == 4
    else:
        assert 'timing was not restored' in ui.errors[-1]


@pytest.mark.parametrize('code,label', [(9,'Connect W control'), (33,'Connect W control'),
                                      (129,'Connect W (stops output)')])
def test_watt_connection_does_not_require_stage_override(watt_gui, code, label):
    ui, device, _ = watt_gui
    ui.flag_super_user_stage = False
    ui.laser_native_control_button = QtWidgets.QPushButton()
    device.code = code
    ui._apply_button_locks_for_status(f'ly_oxp2_dev_status {code}')
    assert ui.laser_native_control_button.isEnabled()
    assert ui.laser_native_control_button.text() == label
    assert not ui.switch_to_cli_button.isEnabled()
    assert not ui.laser_ir_power_control.isEnabled()
    ui.variables.start_flag = True
    ui._update_laser_mode_buttons()
    assert not ui.laser_native_control_button.isEnabled()
    assert not ui.switch_to_cli_button.isEnabled()


def test_watt_connection_does_not_claim_stopped_if_listen_is_not_confirmed(watt_gui, monkeypatch):
    from pyccapt.control.gui import laser_power_control
    from pyccapt.control.nkt_photonics import nktpbus_switch
    ui, old, _ = watt_gui
    native = Laser()
    native.protocol = 'nktpbus'
    native.code = 129
    native.open_port = Mock(return_value=0)
    native.Listen = Mock()  # Firmware acknowledges but does not change its FSM.
    monkeypatch.setattr(nktpbus_switch, '_load_nktp_dll_module', Mock())
    monkeypatch.setattr(nktpbus_switch, 'switch_to_nktpbus', Mock())
    monkeypatch.setattr(laser_power_control, 'NativeLaser', Mock(return_value=native))
    ui._use_native_ir_control()
    assert ui.laser_device is native
    native.Listen.assert_called_once()
    native.close_port.assert_not_called()
    assert 'Listen was not confirmed' in ui.errors[-1]
    assert 'Output stopped' not in ui.errors[-1]
    assert not native.calls


def test_native_handover_confirms_listen_and_reports_interlock_without_starting(watt_gui, monkeypatch):
    from pyccapt.control.gui import laser_power_control
    from pyccapt.control.nkt_photonics import nktpbus_switch
    ui, old, _ = watt_gui
    native = Laser()
    native.protocol = 'nktpbus'
    native.code = 9
    native.wavelength_basis = 'DUV read back before handover'
    native.open_port = Mock(return_value=0)
    native.status_fault = 'interlock waiting for reset'
    native.status_details = dict(fsm=1, error=0, interlock=1)
    monkeypatch.setattr(nktpbus_switch, '_load_nktp_dll_module', Mock())
    monkeypatch.setattr(nktpbus_switch, 'switch_to_nktpbus', Mock())
    monkeypatch.setattr(laser_power_control, 'NativeLaser', Mock(return_value=native))
    ui._apply_button_locks_for_status('ly_oxp2_dev_status 9')
    ui._use_native_ir_control()
    assert ui.laser_device is native and ui.variables.flag_laser_connected
    assert native.calls == [('Listen',)]
    assert not ui.standby_mode and not ui.on_mode
    assert 'Listen confirmed' in ui.errors[-1]
    assert 'interlock waiting for reset' in ui.errors[-1]
    assert not ui.laser_ir_power_control.isEnabled()
    assert ui.led_laser_listen.pixmap().cacheKey() == ui.led_green.cacheKey()
    assert ui.laser_listen.isEnabled() and not ui.laser_standby.isEnabled()
    assert not ui.variables.laser_telemetry['valid']
    # After the fault actually clears, a new explicit Standby uses this session.
    native.status_fault = ''
    ui.check_laser_status()
    assert ui.laser_standby.isEnabled()
    ui.laser_standby_clicked()
    ui.check_laser_status()
    assert native.calls == [('Listen',), ('Standby',)]
    native.open_port.assert_called_once()


def test_native_handover_waits_for_queued_wavelength_request(watt_gui, monkeypatch):
    from pyccapt.control.nkt_photonics import nktpbus_switch
    ui, device, _ = watt_gui
    ui.change_laser_wavelegnth = True
    switch = Mock()
    monkeypatch.setattr(nktpbus_switch, 'switch_to_nktpbus', switch)
    ui._use_native_ir_control()
    switch.assert_not_called()
    device.close_port.assert_not_called()
    assert 'pending laser settings' in ui.errors[-1]
    assert ui.change_laser_wavelegnth


def test_native_release_to_official_software_stops_output_and_releases_handle(watt_gui, monkeypatch):
    from pyccapt.control.nkt_photonics import nktpbus_switch
    ui, device, conversion = watt_gui
    device.protocol = 'nktpbus'
    device.code = 129
    switch = Mock()
    monkeypatch.setattr(nktpbus_switch, 'switch_to_nktpbus', switch)
    ui.switch_to_nktpbus_mode()
    switch.assert_not_called()  # Already in the binary protocol; no reset needed.
    assert device.calls == [('Listen',)]
    device.close_port.assert_called_once()
    assert ui.laser_device is None


@pytest.mark.parametrize('editing', [False, True])
def test_native_watt_readback_updates_field_without_overwriting_active_edit(watt_gui, monkeypatch, editing):
    ui, device, _ = watt_gui
    device.protocol = 'nktpbus'
    device.wavelength_basis = 'IR selected before native handover'
    device.wavelength_read = lambda: 'Current position=IR'
    device.ir_power_setpoint_watts = lambda: 1.4
    monkeypatch.setattr(ui.laser_ir_power_control, 'hasFocus', lambda: editing)
    changed = Mock()
    ui.laser_ir_power_control.valueChanged.connect(changed)
    ui._sync_controls_from_device()
    assert ui.variables.laser_telemetry['ir_power_setpoint_w'] == 1.4
    assert ui.laser_ir_power_control.value() == (1. if editing else 1.4)
    changed.assert_not_called()
    assert not device.calls
