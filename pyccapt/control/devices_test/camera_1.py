import sys
import time

from pypylon import genicam
from pypylon import pylon

try:
    import cv2
except ImportError:
    print('Please install opencv2')
import numpy as np


def camera():
    """
    This function demonstrates grabbing and processing images from multiple cameras using the pypylon library.
    """
    # Number of images to be grabbed.
    countOfImagesToGrab = 10

    # Limits the amount of cameras used for grabbing.
    # It is important to manage the available bandwidth when grabbing with multiple cameras.
    # This applies, for instance, if two GigE cameras are connected to the same network adapter via a switch.
    # To manage the bandwidth, the GevSCPD interpacket delay parameter and the GevSCFTD transmission delay
    # parameter can be set for each GigE camera device.
    # The "Controlling Packet Transmission Timing with the Interpacket and Frame Transmission Delays on Basler GigE Vision Cameras"
    # Application Notes (AW000649xx000)
    # provide more information about this topic.
    # The bandwidth used by a FireWire camera device can be limited by adjusting the packet size.
    maxCamerasToUse = 2

    # The exit code of the sample application.
    exitCode = 0
    img0 = []
    img1 = []
    windowName = 'title'

    try:
        # Get the transport layer factory.
        tlFactory = pylon.TlFactory.GetInstance()

        # Get all attached devices and exit application if no device is found.
        devices = tlFactory.EnumerateDevices()
        for device in devices:
            print('devices name:', device.GetFriendlyName())
        if len(devices) == 0:
            raise pylon.RUNTIME_EXCEPTION("No camera present.")

        # Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
        cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))

        # Create and attach all Pylon Devices.
        for i, cam in enumerate(cameras):
            cam.Attach(tlFactory.CreateDevice(devices[i]))
            cam.Open()
            cam.Width = 2448
            cam.Height = 2048
            cam.ExposureTime.SetValue(800000)

        # Starts grabbing for all cameras starting with index 0. The grabbing
        # is started for one camera after the other. That's why the images of all
        # cameras are not taken at the same time.
        # However, a hardware trigger setup can be used to cause all cameras to grab images synchronously.
        # According to their default configuration, the cameras are
        # set up for free-running continuous acquisition.
        cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        # converting to opencv bgr format
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        while cameras.IsGrabbing():
            if not cameras.IsGrabbing():
                break

            grabResult = cameras.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)

            cameraContextValue = grabResult.GetCameraContext()

            if grabResult.GrabSucceeded():
                image = converter.Convert(grabResult)
                if cameraContextValue == 0:
                    img0 = image.GetArray()
                else:
                    img1 = image.GetArray()

                if len(img1) == 0:
                    img1 = img0

                img0_zoom = cv2.resize(img0[800:1100, 1800:2300], dsize=(2448, 1000), interpolation=cv2.INTER_CUBIC)
                img1_zoom = cv2.resize(img1[1100:1300, 1000:1500], dsize=(2448, 1000), interpolation=cv2.INTER_CUBIC)
                img0_f = np.concatenate((img0, img0_zoom), axis=0)
                img1_f = np.concatenate((img1, img1_zoom), axis=0)
                vis = np.concatenate((img0_f, img1_f), axis=1)
                cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(windowName, 1500, 600)
                cv2.imshow(windowName, vis)
                k = cv2.waitKey(1)
                if k == 27:
                    print('ESC')
                    cv2.destroyAllWindows()
                    break

            else:
                print("Error: ", grabResult.ErrorCode)
            grabResult.Release()
            time.sleep(0.05)

            if cv2.getWindowProperty(windowName, 0) < 0:
                cv2.destroyAllWindows()
                break

    except genicam.GenericException as e:
        print("An exception occurred.", e.GetDescription())
        exitCode = 1

    sys.exit(exitCode)


if __name__ == '__main__':
    camera()
