"""Persistent OXPS NKTPBus control, using the vendor's 0x95 register map.

Main-board registers are documented in 800-621-01 pp163-166. The IR setpoint
controls the fundamental feeding any selected harmonic. The wavelength is
read back via CLI before handover and locked for this session; it is not
queried through an unverified harmonic register map.
Opening/reading never enables emission or resets an interlock.
"""
import logging
import math
import struct
import threading
import time

from pyccapt.control.nkt_photonics import nktpbus_switch, readback
from pyccapt.control.nkt_photonics.origamiClassCLI import origClass


class NativeLaser:
    protocol = 'nktpbus'
    supports_wavelength_changes = False

    def __init__(self, port, *, wavelength, max_watts=5.2, max_aom_percent=100.):
        self.comPort = port
        self._wavelength_index = origClass._integer(wavelength, 0, 3)
        name = readback.WAVELENGTHS[self._wavelength_index][0]
        self.wavelength_basis = (
            f'{name} selection read back via CLI before native handover; '
            'selection locked; no live native harmonic-position readback')
        self.max_watts = float(max_watts)
        self.max_aom_percent = float(max_aom_percent)
        if not math.isfinite(self.max_watts) or not 0 < self.max_watts <= 65.535:
            raise ValueError('Invalid native IR watt ceiling')
        if not math.isfinite(self.max_aom_percent) or not 0 <= self.max_aom_percent <= 100:
            raise ValueError('Invalid native AOM percentage ceiling')
        self.nkt = None
        self.address = None
        self.last_error = None
        self._opened = False
        self.connected = False
        self.status_details = {}
        self.status_fault = ''
        self._port_missing = False
        self._lock = threading.RLock()

    def open_port(self):
        with self._lock:
            if self._opened:
                if self.connected:
                    return 0
                self.last_error = ConnectionError('Close the previous native session before reopening')
                return -1
            try:
                self.nkt = nktpbus_switch._load_nktp_dll_module()
                result = self.nkt.openPorts(self.comPort, 1, 0)
                self._opened = True  # Also clean up partial SDK opens.
                if result != 0:
                    raise ConnectionError(f'Native port open failed: {self.nkt.PortResultTypes(result)}')
                self.address = nktpbus_switch._find_oxps_address(self.nkt, self.comPort, None)
                self.StatusRead()
                self.connected = True
                self.last_error = None
                return 0
            except Exception as exc:
                self.last_error = exc
                self.close_port()
                return -1

    def close_port(self):
        with self._lock:
            self.connected = False
            if self._opened:
                result = self.nkt.closePorts(self.comPort)
                if result not in (0, 2):  # OPPortNotFound also confirms no owned handle.
                    raise ConnectionError(f'Native port close failed: {self.nkt.PortResultTypes(result)}')
                self._opened = False
                self.address = None

    def _read(self, register, kind='U16', index=-1):
        with self._lock:
            if not self._opened or self.address is None:
                raise ConnectionError('Native laser is not connected')
            result, value = getattr(self.nkt, 'registerRead'+kind)(self.comPort, self.address, register, index)
            if result != 0:
                if result == 13:  # SDK auto-scan may have closed an empty port.
                    self._port_missing = True
                    self.connected = False
                raise ConnectionError(f'Native register 0x{register:02X}: {self.nkt.RegisterResultTypes(result)}')
            self._port_missing = False
            return value

    def _write(self, register, value, kind='U8'):
        with self._lock:
            if not self._opened or self.address is None:
                raise ConnectionError('Native laser is not connected')
            result = getattr(self.nkt, 'registerWrite'+kind)(self.comPort, self.address, register, value, -1)
            if result != 0:
                if result == 13:
                    self._port_missing = True
                    self.connected = False
                raise ConnectionError(f'Native register 0x{register:02X}: {self.nkt.RegisterResultTypes(result)}')
            return ''

    def StatusRead(self, timeout=2.):
        with self._lock:
            return self._read_status()

    def _read_status(self):
        previous = self.status_details
        self.status_details = {}
        self.status_fault = 'Native status unavailable'
        state = self._read(0x01, 'U8')
        error = self._read(0x67, 'U8')
        interlock = self._read(0x32)
        code = {0: 1, 1: 9, 2: 17, 3: 33, 4: 17, 5: 65, 6: 129}.get(state)
        if code is None:
            raise ValueError(f'Unknown native laser FSM state {state}')
        interlock_text = {0: 'interlock loop open', 1: 'interlock waiting for reset',
                          2: '', 0x3000: 'external module interlock open',
                          0x4000: 'front-panel interlock open'}.get(
                              interlock, f'interlock 0x{interlock:04X}')
        self.status_fault = '; '.join(filter(None, (
            f'laser error {error}' if error else '', interlock_text)))
        self.status_details = dict(fsm=state, error=error, interlock=interlock)
        self.connected = True
        if self.status_details != previous:
            logging.getLogger('pyccapt.laser').info(
                'Native laser on %s: FSM=%s, error=%s, interlock=0x%04X%s',
                self.comPort, state, error, interlock,
                f' ({self.status_fault})' if self.status_fault else '')
        # Listen is a real stopped state even with an interlock awaiting reset.
        # Retain its fault separately; it must still block upward commands.
        if error and state != 1:
            code = 3
        elif interlock != 2 and state != 1:
            code = 5  # Recovery Listen remains available; never auto-reset it.
        return f'ly_oxp2_dev_status {code}'

    def Listen(self):
        # Reacquire a lost SDK handle only for this explicit stop request.
        # Polling and power commands must never reconnect/replay themselves.
        if self._port_missing:
            self.close_port()
            if self.open_port() != 0:
                raise ConnectionError(f'Cannot reconnect for Listen: {self.last_error}')
        return self._write(0x30, 1)

    def Standby(self):
        return self._standby(reset_interlock=False)

    def standby_with_interlock_reset(self):
        """Operator Standby action; never used by shutdown or connection code."""
        return self._standby(reset_interlock=True)

    def _standby(self, *, reset_interlock):
        self.StatusRead()
        if reset_interlock and self.standby_reset_available:
            # NKT CONTROL 1.3.7.1569 PubOrigamiLib: the reset slot writes
            # value 1 to its register 0x32. A closed loop awaiting reset is
            # acknowledged only by this explicit operator Standby action.
            self._write(0x32, 1, 'U16')
            for _ in range(10):
                self.StatusRead()
                if self.status_details['error'] or self.status_details['interlock'] not in (1, 2):
                    raise ValueError(self.status_fault)
                if self.status_details['interlock'] == 2:
                    break
                time.sleep(.05)
            else:
                raise ValueError('Interlock reset was not confirmed; Standby was not requested')
        if self.status_fault and self.status_details['fsm'] < 3:
            raise ValueError(self.status_fault)
        return self._write(0x30, 3)

    @property
    def standby_reset_available(self):
        return self.status_details == dict(fsm=1, error=0, interlock=1)

    def _require_clear_interlock(self):
        self.StatusRead()
        if self.status_fault:
            raise ValueError(self.status_fault)

    def Enable(self):
        self._require_clear_interlock()
        return self._write(0x30, 6)  # Only explicit Laser On action.

    def AOMDisable(self): return self._write(0x34, 0)
    def AOMEnable(self):
        self._require_clear_interlock()
        return self._write(0x34, 1)
    def AOMRead(self): return f'AOM relative setting: {self._read(0x05)}'
    def FreqRead(self): return f'Frequency index parameter: {self._read(0x35, "U8")}'
    def DivRead(self): return f'Division factor: {self._read(0x36, "U32")}'
    def ModeRead(self):
        external = self._read(0x37, 'U8')
        if external not in (0, 1):
            raise ValueError(f'Unknown native modulation mode {external}')
        return f'e_mode of AOM: {3 if external else 2}'

    def AOM(self, value):
        value = origClass._integer(value, 0, min(4000, round(self.max_aom_percent * 40)))
        if value:
            self._require_clear_interlock()
        return self._write(0x05, value, 'U16')

    def Freq(self, value): return self._write(0x35, origClass._integer(value, 0, 12))
    def Div(self, value): return self._write(0x36, origClass._integer(value, 1, 1000000), 'U32')

    def freq_avaliable(self):
        with self._lock:
            result, data = self.nkt.registerRead(self.comPort, self.address, 0x04, -1)
            if result != 0:
                raise ConnectionError(f'Native frequency table: {self.nkt.RegisterResultTypes(result)}')
            if not data or len(data) % 4 or len(data) > 52:
                raise ValueError('Invalid native frequency table')
            return '\n'.join(f'e_freq={i} --> {hz} Hz' for i, (hz,) in enumerate(struct.iter_unpack('<I', data)) if hz)

    def wavelength_read(self):
        return f'CLI-verified position={readback.WAVELENGTHS[self._wavelength_index][0]}'
    def wavelength_change(self, value):
        raise ValueError('Return to CLI to change wavelength, then reconnect native watt control.')
    def read_average_power(self): return f'{self._read(0x10) / 1000:g} W'
    def read_pulse_energy(self): return f'{self._read(0x11) * 10:g} nJ'
    def ir_power_setpoint_watts(self): return self._read(0x20) / 1000
    def power_read_dv_green(self): return 'Unknown: native harmonic monitor scaling is unverified'

    def set_ir_power_watts(self, watts):
        """One power-register write. No state, shutter, protocol or mode writes."""
        with self._lock:
            watts = float(watts)
            if not math.isfinite(watts) or not 0 <= watts <= self.max_watts:
                raise ValueError('IR watt request exceeds laser_ir_max_power_w')
            if self.ModeRead() != 'e_mode of AOM: 2':
                raise ValueError('Native watt adjustment requires internal power control')
            if self.StatusRead() not in tuple(f'ly_oxp2_dev_status {i}' for i in (9, 33, 65, 129)):
                raise ValueError('Wait for a stable laser state before adjusting watts')
            if self.status_fault:
                raise ValueError(self.status_fault)
            # The firmware applies its calibrated AOM immediately. A restrictive
            # relative cap cannot be predicted from watts without calibration.
            if watts and self.max_aom_percent < 100:
                raise ValueError('A relative AOM cap is configured; use AOM percent or a verified native watt ceiling.')
            requested = round(watts * 1000)
            self._write(0x20, requested, 'U16')
            accepted = self._read(0x20)
            if abs(accepted - requested) > 1:
                raise ValueError(f'IR watt request was not accepted: read back {accepted / 1000:g} W')
            return accepted / 1000
