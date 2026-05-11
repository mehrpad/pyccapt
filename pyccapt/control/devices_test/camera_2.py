from PyQt6 import QtWidgets
from pyqtgraph import PlotWidget, ImageView
import pyqtgraph as pg
from threading import Thread
import numpy as np
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
    Demonstrates grabbing and processing images from multiple cameras using the pypylon library.
    """
    # Number of images to be grabbed.
    countOfImagesToGrab = 10

    # Limits the amount of cameras used for grabbing.
    maxCamerasToUse = 2

    # The exit code of the sample application.
    exitCode = 0
    img0 = []
    img1 = []
    windowName = 'title'

    try:
        tlFactory = pylon.TlFactory.GetInstance()  # Get the transport layer factory.
        devices = tlFactory.EnumerateDevices()  # Get all attached devices

        # Print device names
        for device in devices:
            print('Device name:', device.GetFriendlyName())

        if len(devices) == 0:
            raise pylon.RUNTIME_EXCEPTION("No camera present.")

        cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))  # Create instant camera array

        # Create and attach Pylon Devices
        for i, cam in enumerate(cameras):
            cam.Attach(tlFactory.CreateDevice(devices[i]))
            cam.Open()
            cam.Width = 2448
            cam.Height = 2048
            cam.ExposureTime.SetValue(800000)
            print("Using device", cam.GetDeviceInfo().GetModelName())

        cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)  # Start grabbing images

        # Converter for image format
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        app = QtWidgets.QApplication(sys.argv)
        main = MainWindow()
        thread_read = Thread(target=camera_update, args=(cameras, main, converter, img0, img1))
        thread_read.daemon = True
        thread_read.start()
        main.image_show()
        sys.exit(app.exec())

    except genicam.GenericException as e:
        # Error handling
        print("An exception occurred.", e.GetDescription())
        exitCode = 1


def camera_update(cameras, main, converter, img0, img1):
    """
    Update camera images and display using threading.
    """
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

            img0_zoom = cv2.resize(img0[750:1150, 1650:2250], dsize=(2448, 1000), interpolation=cv2.INTER_CUBIC)
            img1_zoom = cv2.resize(img1[600:1000, 1600:2200], dsize=(2448, 1000), interpolation=cv2.INTER_CUBIC)
            img0_f = np.concatenate((img0, img0_zoom), axis=0)
            img1_f = np.concatenate((img1, img1_zoom), axis=0)
            vis = np.concatenate((img0_f, img1_f), axis=1)
            vis = np.einsum('ijk->kij', vis / 255)
            main.set_image(vis.T)

        else:
            print("Error:", grabResult.ErrorCode)
        grabResult.Release()
        time.sleep(0.05)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)
        self.imv = ImageView()

    def set_image(self, image):
        self.imv.setImage(image, autoRange=False)

    def image_show(self):
        self.imv.show()


if __name__ == '__main__':
    camera()
