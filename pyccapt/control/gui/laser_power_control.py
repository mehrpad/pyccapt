"""Primary watt control with an advanced relative AOM override."""
import logging
import math

from PyQt6 import QtWidgets

from pyccapt.control.devices.nkt_native import NativeLaser
from pyccapt.control.nkt_photonics import nktpbus_switch, readback


class LaserPowerControlMixin:
    def _setup_ir_power_control(self, parent):
        self.laser_ir_power_control_label = QtWidgets.QLabel('Laser power (W)', parent)
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
        self.laser_native_control_button = QtWidgets.QPushButton('Connect W control', parent)
        self.laser_native_control_button.setObjectName('laser_native_control_button')
        self.laser_native_control_button.setMinimumWidth(180)
        self.laser_native_control_button.setFixedHeight(25)
        self.laser_native_control_button.setEnabled(False)
        self.laser_native_control_button.setToolTip(
            'Connect once to enable laser power in watts. This stops output and leaves the laser in Listen. '
            'Then select Standby and Laser On; watt changes use the same connection while on. '
            'The selected wavelength is retained from CLI and locked until returning to CLI. '
            'Stage Override Access is not required.'
        )
        self.laser_native_control_button.clicked.connect(self._use_native_ir_control)
        self.laser_ir_power_control.setEnabled(False)
        # Move the relative control out of the primary power row. Keep the
        # original widget/signals for alignment and existing AOM commands.
        self.gridLayout_3.removeWidget(self.label)
        self.gridLayout_3.removeWidget(self.laser_power)
        self.laser_aom_advanced_button = QtWidgets.QToolButton(parent)
        self.laser_aom_advanced_button.setObjectName('laser_aom_advanced_button')
        self.laser_aom_advanced_button.setText('Advanced AOM (%)')
        self.laser_aom_advanced_button.setFixedHeight(25)
        self.laser_aom_advanced_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.laser_aom_menu = QtWidgets.QMenu(self.laser_aom_advanced_button)
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout(panel)
        layout.addWidget(self.label, 0, 0)
        layout.addWidget(self.laser_power, 0, 1)
        help_text = QtWidgets.QLabel('Relative AOM override. Changing this overrides the watt setpoint.', panel)
        help_text.setWordWrap(True)
        help_text.setMaximumWidth(330)
        layout.addWidget(help_text, 1, 0, 1, 2)
        action = QtWidgets.QWidgetAction(self.laser_aom_menu)
        action.setDefaultWidget(panel)
        self.laser_aom_menu.addAction(action)
        self.laser_aom_advanced_button.setMenu(self.laser_aom_menu)
        self.gridLayout_3.addWidget(self.laser_ir_power_control_label, 1, 0)
        self.gridLayout_3.addWidget(self.laser_ir_power_control, 1, 1)
        self.gridLayout_3.addWidget(self.laser_aom_advanced_button, 4, 0, 1, 2)
        self.gridLayout_3.addWidget(self.laser_native_control_button, 4, 2)
        self._laser_watt_edit_pending = False
        self.laser_ir_power_control.valueChanged.connect(self._mark_laser_watt_edit)
        self.laser_ir_power_control.editingFinished.connect(self._commit_laser_watt_edit)

    def _mark_laser_watt_edit(self, _value):
        self._laser_watt_edit_pending = True

    def _commit_laser_watt_edit(self):
        if not getattr(self, '_laser_watt_edit_pending', False):
            return
        # A rejected request must not be replayed on a later focus change.
        self._laser_watt_edit_pending = False
        self._apply_ir_power_watts()

    def _use_native_ir_control(self):
        if getattr(self.laser_device, 'protocol', None) == 'nktpbus':
            return
        if (self.laser_device is None or self.variables.start_flag or self._laser_alignment_busy()
                or getattr(self, '_laser_mode_switch_in_progress', False)
                or getattr(self, '_laser_stopping', False) or getattr(self, '_laser_status_in_progress', False)):
            self.error_message('Native watt handover requires a connected, idle laser outside an experiment/alignment.')
            return
        if any(getattr(self, flag, False) for flag in ('listen_mode', 'standby_mode', 'on_mode',
                'enable_ouput_mode', 'close_output_mode', 'change_laser_power', 'change_laser_rate',
                'change_laser_divition_factor', 'change_laser_wavelegnth')):
            self.error_message('Wait for pending laser settings to be read back before entering native watt control.')
            return
        try:
            status, mode, index, divider, wavelength = self._ir_power_conditions()
            if status not in (9, 33, 65, 129):
                raise ValueError('Wait for a stable laser state before connecting watt control; connecting stops output.')
            if wavelength not in readback.WAVELENGTHS:
                raise ValueError('Read back a valid selected wavelength through CLI before entering native watt control.')
            nktpbus_switch._load_nktp_dll_module()  # Preflight before releasing CLI.
        except Exception as exc:
            self.error_message(f'Native watt handover rejected: {exc}')
            return
        with self._laser_mode_transition('native watt control') as port:
            if not port:
                return
            nktpbus_switch.switch_to_nktpbus(port)
            self._wait_with_gui_pump(self.CLI_SWITCH_INITIAL_DELAY_S)
            if self._laser_stopping:
                return
            device = NativeLaser(port, wavelength=wavelength,
                max_watts=self.laser_ir_power_control.maximum(),
                max_aom_percent=self.conf.get('laser_aom_max_percent', 100.))
            self.laser_device = device  # Retain ownership if SDK cleanup fails.
            if device.open_port() != 0:
                self._close_laser_device()
                raise ConnectionError(f'Native watt connection failed: {device.last_error}; use Switch to CLI to recover.')
            # The Connect W action explicitly stops output. Never resume an
            # emission request automatically after changing port ownership.
            device.Listen()
            for _ in range(5):
                if self._laser_stopping:
                    return
                stopped_status = readback.scalar(device.StatusRead())
                if stopped_status == 9:
                    break
                self._wait_with_gui_pump(.2)
            else:
                raise ConnectionError(f'W control opened, but Listen was not confirmed (status {stopped_status}); '
                                      'check the laser state and request Listen again.')
            logging.getLogger('pyccapt.laser').info(
                'W control connected on %s; Listen confirmed; CLI-selected wavelength=%s',
                port, readback.WAVELENGTHS[wavelength][0])
            self._sync_controls_from_device(initial=True)
            self._set_laser_disconnected_banner('')
            if (readback.scalar(device.FreqRead()), readback.scalar(device.DivRead())) != (index, divider):
                self.error_message('Native watt control connected; timing changed during reset. Review base rate/divider before Laser On.')
            else:
                name = readback.WAVELENGTHS[wavelength][0]
                self.error_message(f'W control connected ({name} selected in CLI). Output stopped; select Standby, then Laser On. '
                                   'Further watt changes keep the connection and operating state.')

    def _apply_native_ir_watts(self):
        try:
            accepted = self.laser_device.set_ir_power_watts(self.laser_ir_power_control.value())
            self._sync_controls_from_device()
            previous = self.laser_ir_power_control.blockSignals(True)
            try:
                self.laser_ir_power_control.setValue(accepted)
            finally:
                self.laser_ir_power_control.blockSignals(previous)
            logging.getLogger('pyccapt.laser').info(
                'IR watt setpoint accepted on %s: %g W', getattr(self, 'com_port_laser', '?'), accepted)
            self.error_message(f'IR request {accepted:g} W accepted; native connection and operating state retained.')
        except Exception as exc:
            logging.getLogger('pyccapt.laser').warning(
                'IR watt request failed on %s: %s', getattr(self, 'com_port_laser', '?'), exc)
            self.error_message(f'IR watt request failed: {exc}')

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
        if getattr(self.laser_device, 'protocol', None) != 'nktpbus':
            self.error_message('Click Connect W control below the power field. Connecting stops output; then select Standby and Laser On.')
            return
        self._apply_native_ir_watts()
