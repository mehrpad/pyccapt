"""Native watt-control helpers retained independently of the primary AOM GUI."""
import logging
import time

from pyccapt.control.devices.nkt_native import NativeLaser
from pyccapt.control.nkt_photonics import nktpbus_switch, readback


class LaserPowerControlMixin:
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
            if getattr(self.laser_device, 'interlock_reset_available', False):
                return self._reset_native_interlock()
            return getattr(self.laser_device, 'connected', True)
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
            self._observe_laser_indicators(status)
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
            self._queue_laser_indicator('listen', (9,))
            device.Listen()
            self._mark_laser_indicators_sent()
            for _ in range(50):
                if self._laser_stopping:
                    return
                status_reply = device.StatusRead()
                stopped_status = readback.scalar(status_reply)
                self._publish_laser_state(status_reply)
                self._apply_button_locks_for_status(status_reply)
                if stopped_status == 9:
                    break
                self._wait_with_gui_pump(.2)
            else:
                raise ConnectionError(f'W control opened, but Listen was not confirmed (status {stopped_status}); '
                                      f'{getattr(device, "status_fault", "") or "check the laser state"}; '
                                      'request Listen again.')
            logging.getLogger('pyccapt.laser').info(
                'W control connected on %s; Listen confirmed; CLI-selected wavelength=%s',
                port, readback.WAVELENGTHS[wavelength][0])
            # The protocol reset may restore default timing. Restore only
            # verified timing while Listen is confirmed; never restore power.
            if readback.scalar(device.FreqRead()) != index:
                device.Freq(index)
            if readback.scalar(device.DivRead()) != divider:
                device.Div(divider)
            if (readback.scalar(device.FreqRead()), readback.scalar(device.DivRead())) != (index, divider):
                raise ValueError('Laser timing was not restored after handover; review base rate/divider before continuing.')
            settings_confirmed = self._sync_controls_from_device(initial=True)
            if not settings_confirmed or readback.scalar(getattr(self, '_last_laser_status', None)) != 9:
                raise ConnectionError('W control settings/status readback is incomplete; emission was not requested.')
            self._set_laser_disconnected_banner('')
            fault = getattr(device, 'status_fault', '')
            if fault:
                recovery = ('Select Standby to begin warm-up; Reset interlock is also available as a separate action.'
                            if getattr(device, 'interlock_reset_available', False)
                            else 'Resolve the reported interlock/error before requesting Standby.')
                self.error_message(f'Listen confirmed; W control connected. {fault}. '
                                   f'{recovery} No interlock reset requested automatically.')
                return False
            name = readback.WAVELENGTHS[wavelength][0]
            self.error_message(f'W control connected ({name} selected in CLI). '
                'Output stopped; select Standby, then Laser On. Further watt changes keep the connection and operating state.')
            return True

    def _reset_native_interlock(self):
        """Explicit operator reset; status polling confirms the relay later."""
        if (self.laser_device is None or self.variables.start_flag or self._laser_alignment_busy()
                or getattr(self, '_laser_mode_switch_in_progress', False)
                or getattr(self, '_laser_stopping', False) or getattr(self, '_laser_status_in_progress', False)
                or getattr(self, '_laser_interlock_reset_deadline', None) is not None):
            return False
        try:
            self.laser_device.reset_interlock()
        except Exception as exc:
            self.error_message(f'Interlock reset not requested: {exc}')
            logging.getLogger('pyccapt.laser').warning('Interlock reset not requested: %s', exc)
            self._apply_button_locks_for_status(getattr(self, '_last_laser_status', None))
            return False
        self._laser_interlock_reset_deadline = time.monotonic() + 5.
        logging.getLogger('pyccapt.laser').info('Interlock reset requested; waiting for device readback')
        self.error_message('Interlock reset requested; waiting for confirmation. Standby and emission were not requested.')
        self._update_laser_mode_buttons()
        return True

    def _check_native_interlock_reset(self):
        deadline = getattr(self, '_laser_interlock_reset_deadline', None)
        if deadline is None:
            return
        device = self.laser_device
        details = getattr(device, 'status_details', {})
        if not getattr(device, 'status_fault', '') and details.get('interlock') == 2:
            self._laser_interlock_reset_deadline = None
            logging.getLogger('pyccapt.laser').info('Interlock reset confirmed by device readback')
            self.error_message('Interlock reset confirmed. Select Standby, then Laser On when ready.')
        elif (details.get('error') or details.get('interlock') != 1 or time.monotonic() >= deadline):
            self._laser_interlock_reset_deadline = None
            reason = getattr(device, 'status_fault', '') or 'interlock readback unavailable'
            logging.getLogger('pyccapt.laser').warning('Interlock reset was not confirmed: %s', reason)
            self.error_message(f'Interlock reset was not confirmed: {reason}. Standby and emission were not requested.')

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
