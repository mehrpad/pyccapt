"""Explicit IR watt requests alongside the independent relative AOM control."""
import math

from PyQt6 import QtWidgets

from pyccapt.control.devices import nkt_ir_power
from pyccapt.control.nkt_photonics import readback


class LaserPowerControlMixin:
    def _setup_ir_power_control(self, parent):
        self.laser_ir_power_control_label = QtWidgets.QLabel('IR power control (W)', parent)
        self.laser_ir_power_control_label.setFont(self.label.font())
        self.laser_ir_power_control = QtWidgets.QDoubleSpinBox(parent)
        self.laser_ir_power_control.setObjectName('laser_ir_power_control')
        maximum = float(self.conf.get('laser_ir_max_power_w', 5.2))
        if not math.isfinite(maximum) or not 0 < maximum <= 65.535:
            raise ValueError('laser_ir_max_power_w must be finite and within 0..65.535 W.')
        self.laser_ir_power_control.setRange(0., maximum)
        self.laser_ir_power_control.setDecimals(3)
        self.laser_ir_power_control.setSingleStep(.01)
        self.laser_ir_power_control.setKeyboardTracking(False)
        self.laser_ir_power_control.setFixedSize(159, 25)
        self.laser_ir_power_apply = QtWidgets.QPushButton('Apply W (Listen)', parent)
        self.laser_ir_power_apply.setObjectName('laser_ir_power_apply')
        self.laser_ir_power_apply.setFixedHeight(25)
        tooltip = ('Request IR watts using the NKT firmware calibration. Stop the experiment and use Listen or Standby. '
                   'Applying changes protocols and leaves the laser in Listen; enable emission separately. '
                   'The entered value is a requested setpoint, not measured power. AOM edits override it.')
        for widget in (self.laser_ir_power_control, self.laser_ir_power_apply):
            widget.setToolTip(tooltip)
            widget.setEnabled(False)
        self.gridLayout_3.addWidget(self.laser_ir_power_control_label, 4, 0)
        self.gridLayout_3.addWidget(self.laser_ir_power_control, 4, 1)
        self.gridLayout_3.addWidget(self.laser_ir_power_apply, 4, 2)
        self.laser_ir_power_apply.clicked.connect(self._apply_ir_power_watts)

    def _ir_power_conditions(self):
        device = self.laser_device
        return (readback.scalar(device.StatusRead()), readback.scalar(device.ModeRead()),
                readback.scalar(device.FreqRead()), readback.scalar(device.DivRead()),
                readback.wavelength(device.wavelength_read()))

    def _apply_ir_power_watts(self):
        if (getattr(self, '_laser_mode_switch_in_progress', False) or getattr(self, '_laser_stopping', False)
                or getattr(self, '_laser_status_in_progress', False)):
            return
        if self.variables.start_flag or self._laser_alignment_busy() or self.laser_device is None:
            self.error_message('IR watt control requires a connected laser and a stopped experiment/alignment.')
            return
        if any(getattr(self, flag, False) for flag in ('listen_mode', 'standby_mode', 'on_mode',
                'enable_ouput_mode', 'close_output_mode', 'change_laser_power', 'change_laser_rate',
                'change_laser_divition_factor', 'change_laser_wavelegnth')):
            self.error_message('Wait for pending laser settings to be read back before applying IR watts.')
            return
        watts = self.laser_ir_power_control.value()
        try:
            status, mode, index, divider, wavelength = self._ir_power_conditions()
            base_hz = self.get_frequency(index)
            if status not in (9, 33) or mode != 2:
                raise ValueError('Select Listen or ready Standby with internal power control before applying IR watts.')
            if not math.isfinite(base_hz) or divider is None or divider < 1 or wavelength is None:
                raise ValueError('Read valid laser timing and wavelength before applying IR watts.')
        except Exception as exc:
            self.error_message(f'IR watt request rejected: {exc}')
            return
        with self._laser_mode_transition('IR watt control') as port:
            if not port:
                return
            try:
                accepted, aom = nkt_ir_power.calibrated_aom(
                    port, watts, max_watts=self.laser_ir_power_control.maximum(),
                    max_aom_percent=float(self.conf.get('laser_aom_max_percent', 100.)),
                    base_hz=base_hz, divider=divider, wait=self._wait_with_gui_pump,
                    cancelled=lambda: self._laser_stopping or self.variables.start_flag,
                )
            finally:
                # Even a refused setpoint must release SDK ownership and recover
                # a fresh CLI session; never reuse a pre-switch serial handle.
                self._wait_with_gui_pump(self.CLI_SWITCH_INITIAL_DELAY_S)
                reconnected = False
                for _ in range(self.CLI_SWITCH_NUM_ATTEMPTS):
                    if self._laser_stopping:
                        break
                    if self._open_laser_cli(port):
                        reconnected = True
                        break
                    self._wait_with_gui_pump(self.CLI_SWITCH_RETRY_DELAY_S)
                if not reconnected:
                    raise ConnectionError('IR power handover could not reconnect; use Switch to CLI to recover.')
            current = self._ir_power_conditions()
            if (self.variables.start_flag or self._laser_stopping or current[:2] != (9, 2)
                    or current[2:] != (index, divider, wavelength)):
                raise ValueError('Laser state/timing/wavelength changed; calibrated AOM value was not applied.')
            self.laser_device.AOM(aom)
            actual = readback.scalar(self.laser_device.AOMRead())
            if actual != aom:
                raise ValueError(f'IR watt request was not confirmed by AOM readback ({actual!r}, expected {aom}).')
            self._sync_controls_from_device(initial=True)
            self.laser_ir_power_control.setValue(accepted)
            self.error_message(
                f'IR request {accepted:g} W applied as calibrated AOM {aom / 40:g}%. '
                'Laser remains in Listen. Measured power is displayed separately.'
            )
