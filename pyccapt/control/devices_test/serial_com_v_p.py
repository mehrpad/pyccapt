import time

import serial


def command(ser, cmd):
    cmd = f'{cmd}\n'
    ser.write(cmd.encode())


if __name__ == '__main__':
    # Replace 'COM4' with the appropriate serial port identifier
    serial_port = 'COM4'

    # Establish a serial connection
    ser = serial.Serial(serial_port, baudrate=115200, timeout=0.01)

    # Query instrument identity
    command(ser, '*IDN?')
    # ser.write(b'*IDN?\n')
    response = ser.readline().decode().strip()
    print("Instrument Identity:", response)

    # ser.write(b'OUTPut ON\n')
    command(ser, 'OUTPut ON')
    response = ser.readline().decode().strip()
    print("Instrument Identity:", response)
    # Query system ownership
    ser.write(b'SYST:LOCK:OWN?\n')
    response = ser.readline().decode().strip()
    print("System Ownership:", response)

    # Set voltage to 0
    ser.write(b'VOLT 0\n')
    ser.write(b'VOLT?\n')
    response = ser.readline().decode().strip()
    print("Voltage:", response)

    # Set voltage to 15
    ser.write(b'VOLT 30\n')
    ser.write(b'VOLT?\n')
    response = ser.readline().decode().strip()
    print("Voltage:", response)
    time.sleep(1)

    # Turn output on
    ser.write(b'OUTPut ON\n')
    time.sleep(1)

    # Query voltage after turning on output
    ser.write(b'VOLT?\n')
    response = ser.readline().decode().strip()
    print("Voltage:", response)
    time.sleep(5)

    # Turn output off
    ser.write(b'OUTPut OFF\n')

    # Close the serial connection
    ser.close()
