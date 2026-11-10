"""In-flight optical reads must not outlive their GUI/device ownership."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pyccapt.control.gui.laser_readouts import LaserReadoutMixin


class ClosedVariables:
    """Record any access that would hit the closed multiprocessing Manager."""
    def __init__(self):
        object.__setattr__(self, 'accesses', [])

    def __getattr__(self, name):
        self.accesses.append(('read', name))
        raise BrokenPipeError('Manager pipe is closed')

    def __setattr__(self, name, value):
        self.accesses.append(('write', name))
        raise BrokenPipeError('Manager pipe is closed')


READS = (
    ('freq_avaliable', 'e_freq=4 --> 400000 Hz'),
    ('FreqRead', 'Frequency index: 4'),
    ('DivRead', 'Division factor: 4'),
    ('AOMRead', 'AOM setting: 800'),
    ('wavelength_read', 'Current position=Green'),
    ('read_average_power', '1 W'),
    ('power_read_dv_green', 'NKT SHG Output power=0.2 W'),
    ('StatusRead', 'ly_oxp2_dev_status 33'),
    ('ir_power_setpoint_watts', 1.),
)


def readout_gui():
    ui = LaserReadoutMixin()
    ui._laser_stopping = False
    ui._frequency_table = {9: 900000.}
    ui.variables = SimpleNamespace(laser_telemetry={})
    ui.error_message = Mock()
    ui._report_optical_readback_error = Mock()
    ui._apply_button_locks_for_status = Mock()
    for name in ('laser_rate', 'laser_wavelegnth', 'laser_wavelegnth_nm_label',
                 'laser_power', 'laser_divition_factor', 'laser_ir_power_disp',
                 'laser_power_disp', 'laser_pulse_energy_disp', 'laser_repetion_rate_disp',
                 'laser_ir_power_label'):
        setattr(ui, name, Mock())
    ui.laser_wavelegnth.findText.return_value = 0
    ui.laser_device = SimpleNamespace(protocol='nktpbus', wavelength_basis='CLI selection locked')
    for name, reply in READS:
        setattr(ui.laser_device, name, Mock(return_value=reply))
    return ui


@pytest.mark.parametrize('cancel_at', [name for name, _ in READS])
@pytest.mark.parametrize('cancel', ['shutdown', 'replace_device'])
@pytest.mark.parametrize('raises', [False, True])
def test_inflight_read_stops_without_publication_after_shutdown_or_replacement(cancel_at, cancel, raises):
    ui = readout_gui()
    original_device = ui.laser_device
    closed_variables = ClosedVariables()
    calls = []

    def query(name, reply):
        calls.append(name)
        if name == cancel_at:
            ui.variables = closed_variables
            if cancel == 'shutdown':
                ui._laser_stopping = True
            else:
                ui.laser_device = object()
            if raises:
                raise ConnectionError('Native register: RegResultPortNotFound')
        return reply

    for name, reply in READS:
        getattr(original_device, name).side_effect = lambda name=name, reply=reply: query(name, reply)

    assert ui._sync_controls_from_device(initial=True) is False
    assert calls == [name for name, _ in READS][:calls.index(cancel_at)+1]
    assert closed_variables.accesses == []
    assert ui._frequency_table == {9: 900000.}
    ui.error_message.assert_not_called()
    ui._report_optical_readback_error.assert_not_called()
    ui._apply_button_locks_for_status.assert_not_called()
    ui.laser_rate.clear.assert_not_called()
    ui.laser_power.setValue.assert_not_called()
    ui.laser_ir_power_disp.display.assert_not_called()


@pytest.mark.parametrize('method,args', [
    ('_sync_controls_from_device', ()),
    ('_invalidate_laser_readouts', ('late query failure',)),
    ('_recompute_derived_readouts', ()),
    ('_update_wavelength_nm_label', ()),
])
def test_late_readout_callbacks_do_not_touch_the_manager_after_stop(method, args):
    ui = readout_gui()
    ui._laser_stopping = True
    ui.variables = ClosedVariables()
    getattr(ui, method)(*args)
    assert ui.variables.accesses == []
    for name, _ in READS:
        getattr(ui.laser_device, name).assert_not_called()


def test_active_read_failure_still_invalidates_and_reports_the_readout():
    ui = readout_gui()
    ui.laser_device.DivRead.side_effect = ConnectionError('read timeout')
    assert ui._sync_controls_from_device(initial=True) is False
    assert ui.variables.laser_telemetry['valid'] is False
    assert ui.variables.laser_telemetry['error'] == 'read timeout'
    ui.error_message.assert_called_once_with('Laser readback unavailable: read timeout')
    ui.laser_ir_power_disp.display.assert_called_with('-----')
    ui.laser_device.AOMRead.assert_not_called()


def test_complete_current_readout_is_published_after_all_queries():
    ui = readout_gui()
    assert ui._sync_controls_from_device(initial=True) is True
    assert ui.variables.laser_telemetry['wavelength'] == 'Green'
    assert ui.variables.laser_telemetry['ir_power_setpoint_w'] == 1.
    assert ui.variables.laser_telemetry['output_power_mw'] == 200.
    assert ui._frequency_table == {4: 400000.}
    ui.laser_rate.clear.assert_called_once()
    ui.error_message.assert_not_called()
