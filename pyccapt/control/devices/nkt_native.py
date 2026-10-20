"""Persistent OXPS NKTPBus control, using the vendor's 0x95 register map.

Main-board registers are documented in 800-621-01 pp163-166. This adapter
currently supports an IR output selected and verified before handover.
Harmonic selection remains a CLI operation until its register map is verified.
Opening/reading never enables emission or resets an interlock.
"""
import math
import struct
import threading

from pyccapt.control.nkt_photonics import nktpbus_switch
from pyccapt.control.nkt_photonics.origamiClassCLI import origClass


class NativeLaser:
    protocol = 'nktpbus'
    supports_wavelength_changes = False
    wavelength_basis = 'IR selection verified before native handover; wavelength control locked'

    def __init__(self, port, *, max_watts=5.2, max_aom_percent=100.):
        self.comPort = port
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
                if result != 0:
                    raise ConnectionError(f'Native port close failed: {self.nkt.PortResultTypes(result)}')
                self._opened = False
                self.address = None

    def _read(self, register, kind='U16', index=-1):
        with self._lock:
            if not self._opened or self.address is None:
                raise ConnectionError('Native laser is not connected')
            result, value = getattr(self.nkt, 'registerRead'+kind)(self.comPort, self.address, register, index)
            if result != 0:
                raise ConnectionError(f'Native register 0x{register:02X}: {self.nkt.RegisterResultTypes(result)}')
            return value

    def _write(self, register, value, kind='U8'):
        with self._lock:
            if not self._opened or self.address is None:
                raise ConnectionError('Native laser is not connected')
            result = getattr(self.nkt, 'registerWrite'+kind)(self.comPort, self.address, register, value, -1)
            if result != 0:
                raise ConnectionError(f'Native register 0x{register:02X}: {self.nkt.RegisterResultTypes(result)}')
            return ''

    def StatusRead(self, timeout=2.):
        state = self._read(0x01, 'U8')
        error = self._read(0x67, 'U8')
        interlock = self._read(0x32)
        if error:
            code = 3
        elif interlock != 2:
            code = 5  # Recovery Listen remains available; never auto-reset it.
        else:
            code = {0: 1, 1: 9, 2: 17, 3: 33, 4: 17, 5: 65, 6: 129}.get(state)
            if code is None:
                raise ValueError(f'Unknown native laser FSM state {state}')
        return f'ly_oxp2_dev_status {code}'

    def Listen(self): return self._write(0x30, 1)
    def Standby(self): return self._write(0x30, 3)
    def Enable(self): return self._write(0x30, 6)  # Only explicit Laser On action.
    def AOMDisable(self): return self._write(0x34, 0)
    def AOMEnable(self): return self._write(0x34, 1)
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

    def wavelength_read(self): return 'Current position=IR'
    def wavelength_change(self, value):
        raise ValueError('Change wavelength using CLI; native control currently supports IR only.')
    def read_average_power(self): return f'{self._read(0x10) / 1000:g} W'
    def read_pulse_energy(self): return f'{self._read(0x11) * 10:g} nJ'
    def ir_power_setpoint_watts(self): return self._read(0x20) / 1000
    def power_read_dv_green(self): return 'Unknown: harmonic monitor unavailable in native IR control'

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
