"""Driver for Pfeiffer TPG 362 (and family) vacuum-gauge controllers.

Vendor / origin
---------------
The mnemonic command set used in this module (``PR1``, ``PR2``,
``UNI``, ``ETX``, ``ENQ``, gauge-status codes, etc.) is defined and
owned by **Pfeiffer Vacuum GmbH** for their TPG 26x / TPG 36x series
single- and dual-channel vacuum measurement and control units. This
file is a thin Python wrapper around that ASCII serial protocol.

See Pfeiffer's "TPG 261 / 262 Communication Protocol" and
"TPG 361 / 362 Operating Instructions" for the authoritative
specification.
"""

import time

import serial

# Code translations constants
MEASUREMENT_STATUS = {
    0: 'Measurement data okay',
    1: 'Underrange',
    2: 'Overrange',
    3: 'Sensor error',
    4: 'Sensor off (IKR, PKR, IMR, PBR)',
    5: 'No sensor (output: 5,2.0000E-2 [mbar])',
    6: 'Identification error',
}
GAUGE_IDS = {
    'TPR': 'Pirani Gauge or Pirani Capacitive gauge',
    'IKR9': 'Cold Cathode Gauge 10E-9 ',
    'IKR11': 'Cold Cathode Gauge 10E-11 ',
    'PKR': 'FullRange CC Gauge',
    'PBR': 'FullRange BA Gauge',
    'IMR': 'Pirani / High Pressure Gauge',
    'CMR': 'Linear gauge',
    'noSEn': 'no SEnsor',
    'noid': 'no identifier',
}
PRESSURE_UNITS = {0: 'mbar/bar', 1: 'Torr', 2: 'Pascal'}


class TPG26x(object):
    """
    Abstract class that implements the common driver for the TPG 261 and
    TPG 262 dual channel measurement and control unit.
    """

    ETX = chr(3)  # \x03
    CR = chr(13)
    LF = chr(10)
    ENQ = chr(5)  # \x05
    ACK = chr(6)  # \x06
    NAK = chr(21)  # \x15

    def __init__(self, port='/dev/ttyUSB0', baudrate=9600):
        """
        Initialize the TPG26x driver.

        Args:
                port (str or int): The COM port to open.
                baudrate (int): Data transmission rate.
        """
        self.serial = serial.Serial(port=port, baudrate=baudrate, timeout=1)

    def _cr_lf(self, string):
        """
        Pad carriage return and line feed to a string.

        Args:
                string (str): String to pad.

        Returns:
                str: Padded string.
        """
        return string + self.CR + self.LF

    def _send_command(self, command):
        """
        Send a command and check if it is positively acknowledged.

        Args:
                command (str): The command to be sent.

        Raises:
                IOError: If the negative acknowledged or an unknown response is returned.
        """
        self.serial.write(self._cr_lf(command).encode())
        response = self.serial.readline().decode()
        if response == self._cr_lf(self.NAK):
            raise IOError('Serial communication returned negative acknowledge')
        elif response != self._cr_lf(self.ACK):
            raise IOError(f'Serial communication returned unknown response: {response!r}')

    def _get_data(self):
        """
        Get the data that is ready on the device.

        Returns:
                str: Raw data from serial communication line.
        """
        self.serial.write(self.ENQ.encode())
        data = self.serial.readline().decode()
        return data.rstrip(self.LF).rstrip(self.CR)

    def _clear_output_buffer(self):
        """
        Clear the output buffer.
        """
        time.sleep(0.1)
        just_read = 'start value'
        out = ''
        while just_read != '':
            just_read = self.serial.read()
            out += just_read
        return out

    def program_number(self):
        """
        Return the firmware version.

        Returns:
                str: The firmware version.
        """
        self._send_command('PNR')
        return self._get_data()

    def pressure_gauge(self, gauge=1):
        """
        Return the pressure measured by gauge X.

        Args:
                gauge (int): The gauge number, 1 or 2.

        Returns:
                tuple: The value of pressure along with status code and message.
        """
        if gauge not in [1, 2]:
            raise ValueError("The input gauge number can only be 1 or 2")

        self._send_command(f'PR{gauge}')
        reply = self._get_data()
        status_code = int(reply.split(',')[0])
        value = float(reply.split(',')[1])
        return value, (status_code, MEASUREMENT_STATUS[status_code])

    def pressure_gauges(self):
        """
        Return the pressures measured by the gauges.

        Returns:
                tuple: The values of both gauges along with their status codes and messages.
        """
        self._send_command('PRX')
        reply = self._get_data()
        status_code1 = int(reply.split(',')[0])
        value1 = float(reply.split(',')[1])
        status_code2 = int(reply.split(',')[2])
        value2 = float(reply.split(',')[3])
        return (
            value1,
            (status_code1, MEASUREMENT_STATUS[status_code1]),
            value2,
            (status_code2, MEASUREMENT_STATUS[status_code2]),
        )

    def gauge_identification(self):
        """
        Return the gauge identification.

        Returns:
                tuple: Identification codes and messages for both gauges.
        """
        self._send_command('TID')
        reply = self._get_data()
        id1, id2 = reply.split(',')
        return id1, GAUGE_IDS[id1], id2, GAUGE_IDS[id2]

    def pressure_unit(self):
        """
        Return the pressure unit.

        Returns:
                str: The pressure unit.
        """
        self._send_command('UNI')
        unit_code = int(self._get_data())
        return PRESSURE_UNITS[unit_code]

    def rs232_communication_test(self):
        """
        Test the RS232 communication.

        Returns:
                bool: The status of the communication test.
        """
        self._send_command('RST')
        self.serial.write(self.ENQ)
        self._clear_output_buffer()
        test_string_out = ''
        for char in 'a1':
            self.serial.write(char)
            test_string_out += self._get_data().rstrip(self.ENQ)
        self._send_command(self.ETX)
        return test_string_out == 'a1'


class TPG362(TPG26x):
    """
    Driver for the TPG 261 dual channel measurement and control unit.
    Inherits from TPG26x.
    """

    def __init__(self, port='/dev/ttyUSB0', baudrate=9600):
        """
        Initialize the TPG362 driver.

        Args:
                port (str or int): The COM port to open.
                baudrate (int): Data transmission rate.
        """
        super(TPG362, self).__init__(port=port, baudrate=baudrate)
