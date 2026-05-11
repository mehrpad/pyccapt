import serial


class EdwardsAGC:
    """Primitive driver for Edwards Active Gauge Controller.

    Complete manual can be found at:
    http://www.idealvac.com/files/brochures/Edwards_AGC_D386-52-880_IssueM.pdf
    """

    def __init__(self, port='COM1'):
        self.serial = serial.Serial(port, baudrate=9600, timeout=0.5)

    def comm(self, command):
        """Implements basic communication."""
        comm = command + "\r\n"
        self.serial.write(comm.encode())
        complete_string = self.serial.readline().decode().strip()
        return complete_string

    def gauge_type(self, gauge_number):
        """Return the type of gauge."""
        types = {
            0: 'Not Fitted',
            1: '590 CM capacitance manometer',
            15: 'Active strain gauge',
            5: 'Pirani L low pressure',
            20: 'Wide range gauge',
        }

        type_number = int(self.comm('?GV ' + str(gauge_number)))
        gauge_type = types.get(type_number, 'Unknown Type')
        return gauge_type

    def read_pressure(self, gauge_number):
        """Read the pressure of a gauge."""
        pressure_string = self.comm('?V ' + str(gauge_number))
        pressure_value = float(pressure_string)
        return pressure_value

    def pressure_unit(self, gauge_number):
        """Read the unit of a gauge."""
        units = {0: 'mbar', 1: 'torr'}
        unit_string = self.comm('?NU ' + str(gauge_number))
        unit_number = int(unit_string)
        unit = units.get(unit_number, 'Unknown Unit')
        return unit

    def current_error(self):
        """Read the current error code."""
        error_code = self.comm('?SY')
        return error_code

    def my_commands(self):
        """Return the software version of the controller."""
        return self.comm('?V940')

    # Other commands can be added here


if __name__ == '__main__':
    E_AGC = EdwardsAGC('COM6')
    # Example usage of the methods
    print(E_AGC.my_commands())

    # print(E_AGC.gauge_type(4))
    # print(E_AGC.my_commands())
    # print(E_AGC.read_pressure(1))
    # print(E_AGC.read_pressure(2))
    # print(E_AGC.read_pressure(3))
    # print(E_AGC.read_pressure(4))
    # print(E_AGC.pressure_unit(1))
    # print(E_AGC.current_error())
