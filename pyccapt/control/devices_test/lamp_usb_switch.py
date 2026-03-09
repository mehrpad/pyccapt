import os
import time
from ctypes import *


class USBSwitch:
    """
    This class is used to control the USB switch.
    """

    def __init__(self, dll_path):
        """
        Initialize the USB switch.

        Args:
            dll_path (str): Path to the USBaccessX64.dll file.

        Returns:
            None
        """
        self.dll_path = os.path.abspath(dll_path)
        self.device = windll.LoadLibrary(self.dll_path)
        cw = self.device.FCWInitObject()
        devCnt = self.device.FCWOpenCleware(0)
        serNum = self.device.FCWGetSerialNumber(0, 0)
        devType = self.device.FCWGetUSBType(0, 0)
        self.deviceID = 0  # When more than 1 device is connected, the serial number could be used

    def switch_on(self, switch_number):
        """
        Switch on the USB switch.

        Args:
            switch_number (int): Switch number. It is 16 first the first switch.

        Returns:
            None
        """
        state = 1  # 1=on
        self.device.FCWSetSwitch(0, self.deviceID, switch_number, state)

    def switch_off(self, switch_number):
        """
        Switch off the USB switch.

        Args:
            switch_number (int): Switch number. It is 16 first the first switch.

        Returns:
            None
        """
        state = 0  # 0=off
        self.device.FCWSetSwitch(0, self.deviceID, switch_number, state)


if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    switch = USBSwitch(os.path.join(current_directory, "../", "usb_switch", "USBaccessX64.dll"))
    # switch.open_connection()

    switch_number = 16  # Change this to the desired switch number
    switch.switch_on(switch_number)
    print("Switch is on")
    time.sleep(3)
    switch.switch_off(switch_number)
    print("Switch is off")
