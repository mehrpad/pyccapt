"""Origami XPS laser control over the vendor CLI protocol.

Vendor / origin
---------------
The CLI command set used in this module (``ly_oxp2_*``, ``e_freq``,
``e_div``, ``e_mode``, ``e_power``, ``e_mlp``, ``ls_wavelength``, etc.)
is defined and owned by **NKT Photonics A/S** for their Origami XP /
XPS series femtosecond lasers. This file is a thin Python wrapper
around that ASCII serial protocol; the protocol itself is documented
in NKT's CLI reference and the Origami QSG (see
``T:/Monajem/Oxcart_laser_manual/`` on the lab network and
``800-621-0X.pdf`` series).

The original wrapper (`origClass`) was authored by **Ian Baker (NKT
Photonics), version 1.1** and shipped as an example with the NKT SDK.
Subsequent edits in this repository are local extensions for pyccapt
integration; the underlying CLI commands remain NKT's.

Use of the Origami / OXPS hardware and its CLI protocol is subject to
NKT Photonics' licence terms; consult NKT's documentation before
distributing this file outside this project.
"""

import math
import re
import threading
import time

import serial

from pyccapt.control.nkt_photonics.readback import scalar


class origClass:
    """NKT CLI wrapper. Transactions return complete replies, including units."""
    # This instrument returns LF-terminated lines without a trailing prompt.
    # Allow gaps between lines (e.g. the repetition-rate table), rather than
    # treating the first newline or command echo as a complete response.
    REPLY_IDLE_SECONDS = .2

    def __init__(self, comPort):
        self.comPort = comPort
        self.ser = None
        self.last_error = None
        self._lock = threading.RLock()

    def open_port(self):
        try:
            self.ser = serial.Serial(port=self.comPort, baudrate=38400,
                stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS,
                rtscts=False, timeout=.05, write_timeout=.5)
            self.last_error = None
            return 0
        except Exception as exc:
            self.last_error = exc
            return -1

    def close_port(self):
        with self._lock:
            if self.ser is not None:
                self.ser.close()
                self.ser = None

    def _query(self, command, timeout=2., *, allow_echo=False):
        with self._lock:
            if self.ser is None:
                raise ConnectionError('Laser CLI is not connected')
            self.ser.reset_input_buffer()
            self.ser.write((command+'\r\n').encode('ascii'))
            deadline = time.monotonic()+timeout
            response = bytearray()
            last_received = None
            while time.monotonic() < deadline:
                chunk = self.ser.read(max(1, self.ser.in_waiting))
                if chunk:
                    response.extend(chunk)
                    last_received = time.monotonic()
                if re.search(rb'(?:[\r\n]\??>|=ok>)\s*$', response):
                    break
                if (last_received is not None
                        and time.monotonic()-last_received >= self.REPLY_IDLE_SECONDS
                        and response.endswith(b'\n')
                        and not self.ser.in_waiting
                        and (self._reply_text(response, command)
                             or (allow_echo and response.decode('ascii', errors='replace').strip().lstrip('>') == command))):
                    break
            else:
                raise TimeoutError(
                    f'Incomplete or missing laser reply to {command}; '
                    f'received {len(response)} bytes: {bytes(response[:256])!r}'
                )
            return self._reply_text(response, command)

    def _command(self, command):
        """Send a setter which this firmware may answer with only an echo.

        An empty result means no acknowledgement, not confirmed success. The
        caller must read back status/settings. Queries never accept echo alone.
        """
        return self._query(command, allow_echo=True)

    @staticmethod
    def _reply_text(response, command):
        """Keep data lines, removing only the exact command echo and prompt."""
        text = response.decode('utf-8', errors='replace')
        lines = [line.strip().lstrip('>') for line in text.splitlines()]
        lines = [line for line in lines if line and line not in (command, '>', '?>')]
        return '\n'.join(lines).rstrip('>').strip()

    @staticmethod
    def _integer(value, low, high):
        value = float(value)
        if not math.isfinite(value) or value != int(value) or not low <= value <= high:
            raise ValueError(f'Laser value must be an integer in {low}..{high}')
        return int(value)

    def Power(self, pulse_energy_nj):
        """Legacy pre-5.0 command: pulse energy in nJ, NOT watts (manual p119)."""
        value = float(pulse_energy_nj)
        if not math.isfinite(value) or value < 0:
            raise ValueError('Pulse energy must be finite and non-negative')
        return self._command(f'ly_oxp2_power={value:g}')

    def StatusRead(self, timeout=2.):
        response = self._query('ly_oxp2_dev_status?', timeout=timeout)
        value = scalar(response)
        if value is None or value != int(value) or not 0 <= value <= 255:
            raise ValueError(f'Unrecognised laser status: {response!r}')
        return f'ly_oxp2_dev_status {int(value)}'

    def Temp(self, comPort=None):
        return self._query('ly_oxp2_temp_status')

    def Listen(self):
        return self._command('ly_oxp2_listen')

    def Standby(self):
        return self._command('ly_oxp2_standby')

    def Enable(self):
        return self._command('ly_oxp2_enabled')

    def PowerRead(self):
        return self._query('ly_oxp2_power?')

    def AOMRead(self):
        return self._query('e_power?')

    def FreqRead(self):
        return self._query('e_freq?')

    def DivRead(self):
        return self._query('e_div?')

    def ModeRead(self):
        return self._query('e_mode?')

    def StatusMode(self):
        return self._query('ly_oxp2_mode?')

    def ServiceMode(self):
        return self._command('ly_oxp2_service_mode')

    def DigitalGateLogicRead(self):
        return self._query('ly_oxp2_digiop?')

    def AOMEnable(self):
        return self._command('ly_oxp2_output_enable')

    def AOMDisable(self):
        return self._command('ly_oxp2_output_disable')

    def AOMState(self):
        return self._query('ly_oxp2_output?')

    def InterbusEnable(self):
        return self._command('ly_oxp2_nktpbus=1')

    def wavelength_read(self):
        return self._query('ls_wavelength?')

    def read_average_power(self):
        return self._query('e_mlp?')

    def freq_avaliable(self):
        return self._query('e_freq_available?')

    def power_read_dv_green(self):
        return self._query('ls_output_power?')

    def status_led(self):
        return self._query('ly_oxp2_mode?')

    def AOM(self, value):
        value = self._integer(value, 0, 4000)
        return self._command(f"e_power={value}")

    def Freq(self, value):
        value = self._integer(value, 0, 11)
        return self._command(f"e_freq={value}")

    def Div(self, value):
        value = self._integer(value, 1, 10000000)
        return self._command(f"e_div={value}")

    def Mode(self, value):
        value = self._integer(value, 2, 8)
        if value not in (2, 3, 8):
            raise ValueError("Laser mode must be 2, 3 or 8")
        return self._command(f"e_mode={value}")

    def DigitalGateLogic(self, value):
        value = self._integer(value, 0, 1)
        return self._command(f"ly_oxp2_digiop={value}")

    def wavelength_change(self, value):
        value = self._integer(value, 0, 3)
        return self._command(f"ls_wavelength={value}")
