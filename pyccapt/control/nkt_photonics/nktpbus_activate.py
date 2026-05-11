"""Standalone helper: switch an Origami OXPS from CLI mode to NKTPBus.

Vendor / origin
---------------
The ``ly_oxp2_nktpbus`` CLI command sent below is part of the
**NKT Photonics A/S** Origami XPS CLI protocol. This file is adapted
from an example shipped with the NKT SDK (originally authored by
Ian Baker, NKT Photonics).

For the *reverse* direction (NKTPBus -> CLI from Python) see
``nktpbus_switch.py`` in this folder, which uses NKT's NKTPDLL.
"""

import serial

# Example Program using the control object
# Version 1.1
#
# Author: Ian Baker (NKT Photonics)
# Origin: NKT Photonics SDK example, adapted for pyccapt.

if __name__ == "__main__":
    # Customise your comport to your Origami address
    comPort = "COM9"

    # Open the port
    ser = serial.Serial(
        port=comPort, baudrate=38400, stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS, rtscts=False, timeout=1
    )

    # Command to be sent
    cmd = "ly_oxp2_nktpbus=1\n"

    try:
        # Write the command to the serial port
        ser.write(cmd.encode())

        # Read and print the response
        response = ser.readline().decode("utf-8")
        print("Response:", response)

    except Exception as e:
        print("Exception:", e)
    except serial.SerialException as e:
        print("Error: ", e)

    finally:
        # Close the serial port
        ser.close()
