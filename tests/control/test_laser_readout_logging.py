"""Separate expected optical readout gaps from real connection/data faults."""
import logging
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pyccapt.control.gui.laser_readouts import LaserReadoutMixin


@pytest.mark.parametrize('reason', [
    'Waiting for laser readback',
    'Detecting laser interface on FAKE; CLI has not responded yet. Checking the existing protocol.',
    'Switching to CLI on FAKE...',
    'Switching to NKTPBus on FAKE...',
    'Switching to native watt control on FAKE...',
    'Laser status 1.0: Power / booting',
    'Laser status 17.0: Setup / warming',
    'Laser state 17.0 is not stable; optical readings unavailable.',
    'Current wavelength not verified; selected output power and pulse energy are unavailable.',
    'Native selected-harmonic power monitor is unavailable; its register scaling has not been verified.',
])
def test_expected_readout_gaps_are_info_and_deduplicated(reason, caplog):
    ui = LaserReadoutMixin()
    with caplog.at_level(logging.INFO, logger='pyccapt.laser'):
        ui._report_optical_readback_error(reason)
        ui._report_optical_readback_error(reason)
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.INFO
    assert reason in caplog.text


@pytest.mark.parametrize('reason', [
    'CLI disconnected',
    'Serial timeout',
    'CLI handover failed: no serial response',
    'Laser status 3.0: Error',
    'Laser status 5.0: Warning',
    'Laser status None: Status unavailable',
    'Incomplete laser frequency, divider, AOM or wavelength readback',
    'Green monitor returned an unsupported or ambiguous power reading',
    'interlock waiting for reset',
    'Laser state 17.0 is not stable; optical readings unavailable. Read failed: serial timeout',
])
def test_true_readout_failures_remain_warnings_and_deduplicated(reason, caplog):
    ui = LaserReadoutMixin()
    with caplog.at_level(logging.INFO, logger='pyccapt.laser'):
        ui._report_optical_readback_error(reason)
        ui._report_optical_readback_error(reason)
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING


def test_recovery_is_logged_once_and_a_new_occurrence_is_reported(caplog):
    ui = LaserReadoutMixin()
    with caplog.at_level(logging.INFO, logger='pyccapt.laser'):
        ui._report_optical_readback_error('Serial timeout')
        ui._report_optical_readback_error('')
        ui._report_optical_readback_error('')
        ui._report_optical_readback_error('Serial timeout')
    assert [record.levelno for record in caplog.records] == [logging.WARNING, logging.INFO, logging.WARNING]
    assert caplog.records[1].message == 'Laser optical readback restored'


@pytest.fixture
def readout_ui():
    ui = LaserReadoutMixin()
    ui.variables = SimpleNamespace(laser_telemetry={})
    ui.error_message = Mock()
    ui._apply_button_locks_for_status = Mock()
    for name in ('laser_rate', 'laser_wavelegnth', 'laser_wavelegnth_nm_label',
                 'laser_power', 'laser_divition_factor', 'laser_ir_power_disp',
                 'laser_power_disp', 'laser_pulse_energy_disp', 'laser_repetion_rate_disp',
                 'laser_ir_power_label'):
        setattr(ui, name, Mock())
    ui.laser_wavelegnth.findText.return_value = 0
    ui.laser_device = SimpleNamespace(
        protocol='nktpbus', wavelength_basis='Current wavelength not verified at native startup',
        freq_avaliable=Mock(return_value='e_freq=4 --> 400000 Hz'),
        FreqRead=Mock(return_value='Frequency index: 4'),
        DivRead=Mock(return_value='Divider: 4'),
        AOMRead=Mock(return_value='AOM: 800'),
        wavelength_read=Mock(return_value='Current wavelength not verified'),
        read_average_power=Mock(return_value='1 W'),
        power_read_dv_green=Mock(side_effect=AssertionError('Unknown harmonic must not be queried')),
        StatusRead=Mock(return_value='ly_oxp2_dev_status 33'),
        ir_power_setpoint_watts=Mock(return_value=1.25),
    )
    return ui


@pytest.mark.parametrize('code', [1, 17])
def test_initial_boot_or_warmup_skips_settings_and_optical_queries(readout_ui, caplog, code):
    ui = readout_ui
    status = f'ly_oxp2_dev_status {code}'
    with caplog.at_level(logging.INFO, logger='pyccapt.laser'):
        assert ui._sync_controls_from_device(initial=True, status_text=status) is False
    for method in vars(ui.laser_device).values():
        if isinstance(method, Mock):
            method.assert_not_called()
    ui._apply_button_locks_for_status.assert_called_once_with(status)
    ui.error_message.assert_not_called()
    assert ui.variables.laser_telemetry['valid'] is False
    assert [record.levelno for record in caplog.records] == [logging.INFO]


def test_real_fault_while_warming_is_still_a_warning(readout_ui, caplog):
    ui = readout_ui
    ui.laser_device.status_fault = 'external module interlock open'
    with caplog.at_level(logging.INFO, logger='pyccapt.laser'):
        ui._sync_controls_from_device(initial=True, status_text='ly_oxp2_dev_status 17')
    assert [record.levelno for record in caplog.records] == [logging.WARNING]
    assert 'external module interlock open' in caplog.text


def test_native_unknown_wavelength_keeps_verified_ir_and_timing(readout_ui, caplog):
    ui = readout_ui
    with caplog.at_level(logging.INFO, logger='pyccapt.laser'):
        assert ui._sync_controls_from_device(initial=True) is True
    data = ui.variables.laser_telemetry
    assert data['wavelength_index'] is None and data['wavelength_nm'] is None
    assert not data['valid']
    assert data['ir_power_mw'] == 1000.
    assert data['ir_power_setpoint_w'] == 1.25
    assert data['aom_percent'] == 20.
    assert data['output_frequency_hz'] == 100000.
    assert math.isnan(data['output_power_mw']) and math.isnan(data['pulse_energy_nj'])
    ui.laser_ir_power_disp.display.assert_called_with(1.)
    ui.laser_repetion_rate_disp.display.assert_called_with(100.)
    ui.laser_power_disp.display.assert_called_with('-----')
    ui.laser_pulse_energy_disp.display.assert_called_with('-----')
    ui.laser_wavelegnth.setCurrentIndex.assert_called_once_with(-1)
    ui.laser_wavelegnth.setCurrentText.assert_not_called()
    ui.laser_wavelegnth_nm_label.setText.assert_called_with('(not read back)')
    ui.laser_device.power_read_dv_green.assert_not_called()
    ui._apply_button_locks_for_status.assert_called_once_with('ly_oxp2_dev_status 33')
    ui.error_message.assert_not_called()
    assert [record.levelno for record in caplog.records] == [logging.INFO]


def test_cli_malformed_wavelength_remains_invalid_and_a_warning(readout_ui, caplog):
    ui = readout_ui
    ui.laser_device.protocol = 'cli'
    with caplog.at_level(logging.INFO, logger='pyccapt.laser'):
        assert ui._sync_controls_from_device(initial=True) is False
    assert [record.levelno for record in caplog.records] == [logging.WARNING]
    assert 'Incomplete laser frequency, divider, AOM or wavelength readback' in caplog.text
    ui.error_message.assert_called_once()


def test_native_unknown_wavelength_does_not_suppress_transport_failure(readout_ui, caplog):
    ui = readout_ui
    ui.laser_device.StatusRead.side_effect = ConnectionError('Native register: RegResultPortNotFound')
    with caplog.at_level(logging.INFO, logger='pyccapt.laser'):
        assert ui._sync_controls_from_device(initial=True) is False
    assert [record.levelno for record in caplog.records] == [logging.WARNING]
    assert 'RegResultPortNotFound' in caplog.text
    ui.error_message.assert_called_once()


@pytest.mark.parametrize('ir_failure', ['timeout', 'unitless'])
def test_unverified_wavelength_keeps_independent_ir_failures_as_warnings(readout_ui, caplog, ir_failure):
    ui = readout_ui
    if ir_failure == 'timeout':
        ui.laser_device.read_average_power.side_effect = TimeoutError('IR serial timeout')
    else:
        ui.laser_device.read_average_power.return_value = '12345'
    with caplog.at_level(logging.INFO, logger='pyccapt.laser'):
        assert ui._sync_controls_from_device(initial=True) is True
    assert [record.levelno for record in caplog.records] == [logging.WARNING]
    assert 'IR monitor has no usable unit-bearing reading' in caplog.text
    assert 'Current wavelength not verified' in caplog.text
    assert math.isnan(ui.variables.laser_telemetry['ir_power_mw'])
    assert not ui.variables.laser_telemetry['valid']


@pytest.mark.parametrize('ir_failed', [False, True])
def test_native_harmonic_capability_gap_is_info_but_ir_failures_remain_warnings(readout_ui, caplog, ir_failed):
    ui = readout_ui
    ui.laser_device.wavelength_read.return_value = 'Current position=Green'
    ui.laser_device.supports_harmonic_power_readback = False
    if ir_failed:
        ui.laser_device.read_average_power.side_effect = TimeoutError('IR serial timeout')
    with caplog.at_level(logging.INFO, logger='pyccapt.laser'):
        assert ui._sync_controls_from_device(initial=True) is True
    assert [record.levelno for record in caplog.records] == [logging.WARNING if ir_failed else logging.INFO]
    assert 'Native selected-harmonic power monitor is unavailable' in caplog.text
    ui.laser_device.power_read_dv_green.assert_not_called()
    data = ui.variables.laser_telemetry
    assert data['wavelength'] == 'Green'
    assert math.isnan(data['output_power_mw']) and math.isnan(data['pulse_energy_nj'])
    if ir_failed:
        assert 'IR serial timeout' in caplog.text
        assert math.isnan(data['ir_power_mw'])
    else:
        assert data['ir_power_mw'] == 1000.


def test_warmup_status_does_not_hide_an_actual_ir_read_failure(readout_ui, caplog):
    ui = readout_ui
    ui.laser_device.wavelength_read.return_value = 'Current position=IR'
    ui.laser_device.read_average_power.side_effect = TimeoutError('IR serial timeout')
    ui.laser_device.StatusRead.return_value = 'ly_oxp2_dev_status 17'
    with caplog.at_level(logging.INFO, logger='pyccapt.laser'):
        ui._sync_controls_from_device(initial=True)
    assert [record.levelno for record in caplog.records] == [logging.WARNING]
    assert 'IR serial timeout' in caplog.text
    assert 'IR serial timeout' in ui.variables.laser_telemetry['ir_power_error']


def test_malformed_cli_harmonic_reply_remains_a_warning(readout_ui, caplog):
    ui = readout_ui
    ui.laser_device.protocol = 'cli'
    ui.laser_device.wavelength_read.return_value = 'Current position=Green'
    ui.laser_device.power_read_dv_green = Mock(return_value='Output power=unknown')
    with caplog.at_level(logging.INFO, logger='pyccapt.laser'):
        assert ui._sync_controls_from_device(initial=True) is True
    assert [record.levelno for record in caplog.records] == [logging.WARNING]
    assert 'unsupported or ambiguous power reading' in caplog.text
