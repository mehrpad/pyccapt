"""Persistent OXPS NKTPBus control, using the vendor's 0x95 register map.

Main-board registers are documented in 800-621-01 pp163-166. The IR setpoint
controls the fundamental feeding any selected harmonic. Native register 0x0E
reports and selects the wavelength on supported firmware. Selection changes
require confirmed Listen; older firmware retains the CLI-verified selection.
Opening/reading never enables emission or resets an interlock.
"""
import logging
import math
import struct
import threading
import time

from pyccapt.control.nkt_photonics import nktpbus_switch, readback
from pyccapt.control.nkt_photonics.origamiClassCLI import origClass
from pyccapt.control.devices.nkt_harmonic import NativeHarmonicReadoutMixin


class NativeLaser(NativeHarmonicReadoutMixin):
    protocol = 'nktpbus'

    def __init__(self, port, *, wavelength=None, max_watts=5.2, max_aom_percent=100.):
        self.comPort = port
        self._wavelength_index = None if wavelength is None else origClass._integer(wavelength, 0, 3)
        self._native_wavelength_supported = False
        if self._wavelength_index is None:
            self.wavelength_basis = 'Current native wavelength has not been verified; return to CLI to read/change it.'
        else:
            name = readback.WAVELENGTHS[self._wavelength_index][0]
            self.wavelength_basis = (
                f'{name} selection read back via CLI before native handover; '
                'native wavelength support has not yet been checked')
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
                # The SDK's live monitoring maintains the module watchdog for
                # the lifetime of this persistent connection. Protocol-switch
                # helpers use a short-lived, non-monitoring session instead.
                result = self.nkt.openPorts(self.comPort, 1, 1)
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

    def _read(self, register, kind='U16', index=-1, *, allow_unsupported=False, address=None):
        with self._lock:
            if not self._opened or self.address is None:
                raise ConnectionError('Native laser is not connected')
            target = self.address if address is None else address
            for attempt in range(4):
                result, value = getattr(self.nkt, 'registerRead'+kind)(self.comPort, target, register, index)
                if result != 15 or attempt == 3:
                    break
                # Retry reads only; never replay a state/power write or reuse
                # cached status as though it were a fresh hardware reply.
                time.sleep(.02)
            if allow_unsupported and result in (4, 8, 11):  # NACK, type unsupported, register absent.
                return None
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
        interlock_text = {0: 'interlock loop open', 1: 'interlock waiting for reset', 2: '',
                          0x0100: 'key-switch interlock open', 0x0200: 'door interlock open',
                          0x0300: 'external module interlock open', 0x0400: 'interlock power failure',
                          0x0500: 'internal controller interlock open',
                          0x1000: 'interlock power failure', 0x2000: 'internal interlock open',
                          0x3000: 'external module interlock open', 0x4000: 'front-panel interlock open'}.get(
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
        # Retain its fault separately; emission still requires a clear interlock.
        if error and state != 1:
            code = 3
        elif interlock != 2 and state != 1 and not (interlock == 1 and state in (2, 4)):
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

    def zero_power_in_listen(self):
        """Clear stored watts and relative power only after Listen is read back."""
        with self._lock:
            if readback.scalar(self.StatusRead()) != 9:
                raise ValueError('Power can only be cleared after Listen is confirmed')
            # Zero is a reduction, allowed even while the interlock awaits reset.
            # Never use the normal positive-power path to bypass that interlock.
            self._write(0x20, 0, 'U16')
            self._write(0x05, 0, 'U16')
            if self._read(0x20) != 0 or self._read(0x05) != 0:
                raise ValueError('Zero watt/AOM settings were not confirmed by laser readback')

    def Standby(self):
        self.StatusRead()
        if self.status_fault and self.status_details['fsm'] < 3 and not self.standby_available:
            raise ValueError(self.status_fault)
        return self._write(0x30, 3)

    @property
    def standby_available(self):
        # Vendor manual 800-621-01 Procedure 7 (pp85-86) and CONTROL's
        # standbyChanged handler request target 3 directly from Listen.
        # Closed-loop "waiting for reset" is not a prerequisite failure.
        return (self.connected and self.status_details.get('fsm') == 1
                and self.status_details.get('error') == 0
                and self.status_details.get('interlock') in (1, 2))

    @property
    def interlock_reset_available(self):
        return self.connected and self.status_details == dict(fsm=1, error=0, interlock=1)

    def reset_interlock(self):
        """Request only the closed-loop reset; later status reads confirm it.

        The installed NKT CONTROL Origami plugin writes U8=1 to 0x32
        asynchronously. It does not require the relay to close in the command
        transaction. Never request a state transition or enable output here.
        """
        with self._lock:
            self.StatusRead()
            if not self.interlock_reset_available:
                raise ValueError(self.status_fault or 'Interlock reset is only available in Listen while awaiting reset')
            return self._write(0x32, 1, 'U8')

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

    @property
    def supports_wavelength_changes(self):
        return self._native_wavelength_supported

    @property
    def wavelength_change_available(self):
        return (self.connected and self.supports_wavelength_changes
                and self.status_details.get('fsm') == 1
                and self.status_details.get('error') == 0
                and self.status_details.get('interlock') in (1, 2))

    def wavelength_read(self):
        with self._lock:
            # NKT CONTROL's onRegDynamicWavelengthRead uses main-board 0x0E
            # (U8): 0=IR, 1=SHG, 2=THG, 3=FHG. Older firmware may NACK it.
            selected = self._read(0x0E, 'U8', allow_unsupported=True)
            self._native_wavelength_supported = selected in readback.WAVELENGTHS
            if selected in readback.WAVELENGTHS:
                self._wavelength_index = selected
                name = readback.WAVELENGTHS[selected][0]
                self.wavelength_basis = f'{name} selected wavelength read from native register 0x0E'
                return f'Native wavelength position={name}'
            if selected is not None:
                self._wavelength_index = None
                raise ValueError(f'Unknown native wavelength position {selected}')
            if self._wavelength_index is None:
                return 'Unknown: current native wavelength has not been verified'
            self.wavelength_basis = ('Last verified wavelength retained; native wavelength register is unsupported '
                                     'and wavelength edits are unavailable')
            return f'Last verified position={readback.WAVELENGTHS[self._wavelength_index][0]}'

    def wavelength_change(self, value):
        with self._lock:
            value = origClass._integer(value, 0, 3)
            self.StatusRead()
            self.wavelength_read()  # Confirm this firmware supports register 0x0E.
            if not self.wavelength_change_available:
                raise ValueError('Native wavelength changes require confirmed Listen, a closed interlock loop '
                                 'and supported wavelength readback.')
            # Installed NKT CONTROL PubOrigamiLib: dynamicWavelengthChanged
            # encodes SetInt8 and writes the register created at main-board 0x0E.
            # No protocol, FSM, interlock-reset or power command is needed.
            self._write(0x0E, value, 'U8')
            self._wavelength_index = None
            self.wavelength_basis = 'Wavelength change requested; waiting for native register 0x0E readback'
            return ''
    def read_average_power(self): return f'{self._read(0x10) / 1000:g} W'
    def read_pulse_energy(self): return f'{self._read(0x11) * 10:g} nJ'
    def ir_power_setpoint_watts(self): return self._read(0x20) / 1000

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
