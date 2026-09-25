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


def test_unsupported_harmonic_monitor_never_substitutes_ir_power(laser_gui):
    ui = laser_gui
    ui.laser_device.harmonic = 'Unknown command'
    ui._sync_controls_from_device(initial=True)
    assert not ui.variables.laser_telemetry['valid']
    assert math.isnan(ui.variables.laser_average_power)


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


def test_listen_cancels_pending_emission_request(laser_gui):
    ui = laser_gui
    ui.on_mode = ui.listen_mode = True
    ui.check_laser_status()
    assert ui.laser_device.calls == [('Listen',)]
    assert not ui.on_mode


def test_failed_readback_clears_previous_wavelength_and_values(laser_gui):
    ui = laser_gui
    ui._sync_controls_from_device(initial=True)
    assert '257.5' in ui.laser_wavelegnth_nm_label.text()
    ui._invalidate_laser_readouts('Serial timeout')
    assert ui.laser_wavelegnth_nm_label.text() == '(not read back)'
    assert not ui.variables.laser_telemetry['valid']
    assert math.isnan(ui.variables.laser_pulse_energy)
    assert math.isnan(ui.variables.laser_average_power)
