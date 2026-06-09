import os
from ctypes import windll


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
        self._closed = False

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

    def close(self):
        """Release the Cleware DLL handle.

        Without this the USB handle stayed open until process exit;
        a re-init in the same session would orphan the previous handle.
        Idempotent and safe to call from cleanup paths.
        """
        if self._closed:
            return
        try:
            # Vendor SDK names: FCWCloseCleware releases the open device,
            # FCWUninitObject tears down the underlying init context.
            if hasattr(self.device, 'FCWCloseCleware'):
                self.device.FCWCloseCleware(self.deviceID)
            if hasattr(self.device, 'FCWUninitObject'):
                self.device.FCWUninitObject()
        except Exception as exc:
            print(f"USBSwitch.close failed (non-fatal): {exc}")
        finally:
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
