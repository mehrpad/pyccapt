import threading
import time

import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSlot, pyqtSignal

try:
    from pypylon import pylon
except Exception as exc:  # pragma: no cover - depends on local Basler runtime
    pylon = None
    _PYPYLON_IMPORT_ERROR = exc
else:
    _PYPYLON_IMPORT_ERROR = None


def check_camera_availability(required_cameras: int = 2) -> tuple[bool, str]:
    """Return whether the Basler camera backend has enough connected devices."""

    if pylon is None:
        return False, f"Camera backend is unavailable ({_PYPYLON_IMPORT_ERROR})"

    try:
        devices = pylon.TlFactory.GetInstance().EnumerateDevices()
    except Exception as exc:  # pragma: no cover - backend specific
        return False, f"Unable to enumerate cameras ({exc})"

    count = len(devices)
    if count < required_cameras:
        plural = "s" if count != 1 else ""
        return (
            False,
            f"Detected {count} Basler camera{plural}; at least {required_cameras} are required.",
        )

    return True, f"Detected {count} Basler cameras."


class CameraWorker(QObject):
    """
    This class is used to control the BASLER Cameras.
    """
    finished = pyqtSignal()  # Define the finished signal
    def __init__(self, variables, emitter):
        """
        Constructor function which initializes and setups all variables
        and parameters for the class.

        Args:
            variables: The class object of the Variables class.
            emitter: The class object of the Emitter class.

        Return:
            None
        """
        super().__init__()
        self.flag_default_exposure_time = None
        self.exposure_auto = None
        self.emitter = emitter
        self.variables = variables
        self.camera_available = False
        self.camera_status_message = ""

        self.running = False
        self.index_save_image = 0
        self.exposure_time_cam_1 = 400000
        self.exposure_time_cam_1_light = 10000
        self.exposure_time_cam_2 = 1000000
        self.exposure_time_cam_2_light = 20000
        self.exposure_time_cam_3 = 400000
        self.exposure_time_cam_3_light = 10000
        self.emitter.cam_1_exposure_time = emitter.cam_1_exposure_time
        self.emitter.cam_2_exposure_time = emitter.cam_2_exposure_time
        self.emitter.cam_3_exposure_time = emitter.cam_3_exposure_time
        self.emitter.default_exposure_time = emitter.default_exposure_time

        self.emitter.cam_1_exposure_time.connect(self.set_exposure_time_1)
        self.emitter.cam_2_exposure_time.connect(self.set_exposure_time_2)
        self.emitter.cam_3_exposure_time.connect(self.set_exposure_time_3)
        self.emitter.default_exposure_time.connect(self.set_default_exposure_time)
        self.emitter.auto_exposure_time.connect(self.set_auto_exposure_time)

        self.initialize_cameras()

    def start_capturing(self):
        if not self.camera_available:
            self.finished.emit()
            return
        self.running = True
        self.thread = threading.Thread(target=self.update_cameras)
        self.thread.start()

    def stop_capturing(self):
        self.running = False

    @pyqtSlot(bool)
    def set_default_exposure_time(self):
        """
        This class method sets

        Args:
            None

        Return:
            None
        """
        if not self.exposure_auto:
            self.exposure_time_cam_1 = 400000
            self.exposure_time_cam_1_light = 10000
            self.exposure_time_cam_2 = 1000000
            self.exposure_time_cam_2_light = 20000
            self.exposure_time_cam_3 = 400000
            self.exposure_time_cam_3_light = 10000

            self.flag_default_exposure_time = True
            if self.variables.light:
                exposure_times = [self.exposure_time_cam_1_light, self.exposure_time_cam_2_light,
                                  self.exposure_time_cam_3_light]
                self.emitter.cams_exposure_time_default.emit(exposure_times)
            else:
                exposure_times = [self.exposure_time_cam_1, self.exposure_time_cam_2,
                                  self.exposure_time_cam_3]
                self.emitter.cams_exposure_time_default.emit(exposure_times)
        else:
            print('Cannot set the default exposure time when auto exposure is on')

    @pyqtSlot(bool)
    def set_auto_exposure_time(self):
        """
        This class method sets

        Args:
            None

        Return:
            None
        """
        if not self.exposure_auto:
            self.exposure_mode = 'Continuous'
            self.exposure_auto = True
        elif self.exposure_auto:
            self.exposure_mode = 'Off'
            self.exposure_auto = False

    @pyqtSlot(int)
    def set_exposure_time_1(self, exposure_time):
        """
        This class method sets

        Args:
            exposure_time: The exposure time for the camera.

        Return:
            None
        """
        self.exposure_time_cam_1 = exposure_time

    @pyqtSlot(int)
    def set_exposure_time_2(self, exposure_time):
        """
        This class method sets

        Args:
            exposure_time: The exposure time for the camera.

        Return:
            None
        """
        self.exposure_time_cam_2 = exposure_time

    @pyqtSlot(int)
    def set_exposure_time_3(self, exposure_time):
        """
        This class method sets

        Args:
            exposure_time: The exposure time for the camera.

        Return:
            None
        """
        self.exposure_time_cam_3 = exposure_time

    def initialize_cameras(self):
        """
        Initializes and sets up the cameras.

        Args:
            None

        Return:
            None
        """
        available, message = check_camera_availability(required_cameras=2)
        self.camera_available = available
        self.camera_status_message = message
        self.cameras = None
        if not available:
            print(message)
            return

        try:
            maxCamerasToUse = 2
            self.tlFactory = pylon.TlFactory.GetInstance()
            self.devices = self.tlFactory.EnumerateDevices()
            self.cameras = pylon.InstantCameraArray(min(len(self.devices), maxCamerasToUse))

            for i, cam in enumerate(self.cameras):
                cam.Attach(self.tlFactory.CreateDevice(self.devices[i]))
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            self.cameras[0].Open()
            self.cameras[0].ExposureAuto.SetValue('Off')
            self.cameras[0].ExposureTime.SetValue(self.exposure_time_cam_1)
            self.cameras[1].Open()
            self.cameras[1].ExposureAuto.SetValue('Off')
            self.cameras[1].ExposureTime.SetValue(self.exposure_time_cam_2)
            self.exposure_auto = False
        except Exception as e:
            self.camera_available = False
            self.camera_status_message = f"Error in initializing the camera class ({e})"
            self.cameras = None
            print(self.camera_status_message)

    def update_cameras(self):
        """
        This class method sets up the cameras to capture the required images.

        Args:
            None

        Return:
            None
        """
        if not self.camera_available or self.cameras is None:
            self.finished.emit()
            return

        retry_attempts = 5
        tmp_exposure_time_cam_1 = self.exposure_time_cam_1
        tmp_exposure_time_cam_2 = self.exposure_time_cam_2
        # tmp_exposure_time_cam_3 = self.exposure_time_cam_3
        # set the auto exposure mode off
        self.exposure_mode = 'Off'
        tmp_exposure_mode = self.exposure_mode
        for attempt in range(retry_attempts):
            try:
                self.cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                start_time = time.time()
                while self.cameras.IsGrabbing() and self.running:
                    if tmp_exposure_mode != self.exposure_mode:
                        tmp_exposure_mode = self.exposure_mode
                        # self.cameras[0].open()
                        self.cameras[0].ExposureAuto.SetValue(self.exposure_mode)
                        # self.cameras[1].open()
                        self.cameras[1].ExposureAuto.SetValue(self.exposure_mode)
                        # self.cameras[2].open()
                        # self.cameras[2].ExposureAuto.SetValue(self.exposure_mode)
                    if tmp_exposure_time_cam_1 != self.exposure_time_cam_1:
                        tmp_exposure_time_cam_1 = self.exposure_time_cam_1
                        # self.cameras[0].open()
                        self.cameras[0].ExposureTime.SetValue(self.exposure_time_cam_1)
                    if tmp_exposure_time_cam_2 != self.exposure_time_cam_2:
                        tmp_exposure_time_cam_2 = self.exposure_time_cam_2
                        # self.cameras[1].open()
                        self.cameras[1].ExposureTime.SetValue(self.exposure_time_cam_2)
                    # if tmp_exposure_time_cam_3 != self.exposure_time_cam_3:
                    #     tmp_exposure_time_cam_3 = self.exposure_time_cam_3
                    #     self.cameras[2].open()
                    #     self.cameras[2].ExposureTime.SetValue(self.exposure_time_cam_3)
                    current_time = time.time()
                    try:
                        grabResult0 = self.cameras[0].RetrieveResult(8000, pylon.TimeoutHandling_ThrowException)
                        grabResult1 = self.cameras[1].RetrieveResult(8000, pylon.TimeoutHandling_ThrowException)
                        image0 = self.converter.Convert(grabResult0)
                        img0 = image0.GetArray()
                        image1 = self.converter.Convert(grabResult1)
                        img1 = image1.GetArray()
                    except Exception as e:
                        print(f"Error in grabbing the images from the camera: {e}")
                        break

                    self.img0_orig = img0
                    self.img1_orig = img1
                    self.img2_orig = img0

                    self.emitter.img0_orig.emit(np.swapaxes(self.img0_orig, 0, 1))
                    self.emitter.img1_orig.emit(np.swapaxes(self.img1_orig, 0, 1))
                    self.emitter.img2_orig.emit(np.swapaxes(self.img2_orig, 0, 1))

                    if self.variables.clear_index_save_image:
                        self.variables.clear_index_save_image = False
                        self.index_save_image = 0

                    if current_time - start_time >= self.variables.save_meta_interval_camera and self.variables.start_flag:
                        start_time = time.time()
                        path_meta = self.variables.path_meta
                        cv2.imwrite(path_meta + "/camera_side_%s.png" % self.index_save_image, self.img0_orig)
                        cv2.imwrite(path_meta + '/camera_top_%s.png' % self.index_save_image, self.img1_orig)
                        cv2.imwrite(path_meta + '/camera_45_%s.png' % self.index_save_image, self.img2_orig)
                        self.index_save_image += 1
                        time.sleep(0.5)

                    grabResult0.Release()
                    grabResult1.Release()

                    if self.variables.light_switch or self.flag_default_exposure_time:
                        self.light_switch()
                        self.variables.light_switch = False
                        self.flag_default_exposure_time = False

                    time.sleep(0.5)

                    if not self.variables.flag_camera_grab:
                        break
                break  # Exit the retry loop if successful
            except Exception as e:
                print(f"Error during update_cameras attempt {attempt + 1}: {e}")
                self.initialize_cameras()
                time.sleep(1)
        self.finished.emit()  # Emit the finished signal when done

    def light_switch(self):
        """
        This class method sets the Exposure time based on a flag.

        Args:
            None

        Return:
            None
        """
        if not self.exposure_auto:
            try:
                if self.variables.light:
                    # self.cameras[0].Open()
                    self.cameras[0].ExposureTime.SetValue(self.exposure_time_cam_1_light)
                    # self.cameras[1].Open()
                    self.cameras[1].ExposureTime.SetValue(self.exposure_time_cam_2_light)
                else:
                    # self.cameras[0].Open()
                    self.cameras[0].ExposureTime.SetValue(self.exposure_time_cam_1)
                    # self.cameras[1].Open()
                    self.cameras[1].ExposureTime.SetValue(self.exposure_time_cam_2)
            except Exception as e:
                print(f"Error in switching the light: {e}")
