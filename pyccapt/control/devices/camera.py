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


def check_camera_backend() -> tuple[bool, str]:
    """Return whether the Basler camera backend is usable on this host.

    A True result means we can enumerate Basler cameras at runtime. The
    actual *number* of connected cameras is allowed to fluctuate while the
    GUI is running, so the worker handles that dynamically; this check is
    only about whether pypylon itself is available.
    """

    if pylon is None:
        return False, f"Camera backend is unavailable ({_PYPYLON_IMPORT_ERROR})"

    try:
        devices = pylon.TlFactory.GetInstance().EnumerateDevices()
    except Exception as exc:  # pragma: no cover - backend specific
        return False, f"Unable to enumerate cameras ({exc})"

    count = len(devices)
    if count == 0:
        return True, "No Basler cameras detected; GUI will retry while running."
    plural = "s" if count != 1 else ""
    return True, f"Detected {count} Basler camera{plural}."


def check_camera_availability(required_cameras: int = 1) -> tuple[bool, str]:
    """Compatibility wrapper.

    Older call sites used this to gate whether the camera GUI could open at
    all. The GUI now always opens when the backend is loaded and the worker
    handles missing/disconnected cameras dynamically, so this just forwards
    to :func:`check_camera_backend`.
    """

    del required_cameras  # unused; kept for backward compatibility
    return check_camera_backend()


class CameraWorker(QObject):
    """Drives Basler camera capture with hot-reconnect support.

    The worker keeps a fixed number of slots (one per logical camera in the
    GUI) and binds each slot to a physical camera by serial number the first
    time it sees one. If a camera disappears, its slot is freed and the
    worker keeps trying to (re)attach it. Other slots keep streaming.
    """

    finished = pyqtSignal()

    SLOT_COUNT = 2
    RECONNECT_INTERVAL = 3.0

    def __init__(self, variables, emitter):
        super().__init__()
        self.flag_default_exposure_time = None
        self.exposure_auto = False
        self.exposure_mode = 'Off'
        self.emitter = emitter
        self.variables = variables

        self.running = False
        self.index_save_image = 0
        self.exposure_time_cam_1 = 400000
        self.exposure_time_cam_1_light = 10000
        self.exposure_time_cam_2 = 1000000
        self.exposure_time_cam_2_light = 20000
        self.exposure_time_cam_3 = 400000
        self.exposure_time_cam_3_light = 10000

        self.emitter.cam_1_exposure_time.connect(self.set_exposure_time_1)
        self.emitter.cam_2_exposure_time.connect(self.set_exposure_time_2)
        self.emitter.cam_3_exposure_time.connect(self.set_exposure_time_3)
        self.emitter.default_exposure_time.connect(self.set_default_exposure_time)
        self.emitter.auto_exposure_time.connect(self.set_auto_exposure_time)

        self._slots = [None] * self.SLOT_COUNT
        self._slot_serials = [None] * self.SLOT_COUNT
        self._applied_exposure = [None] * self.SLOT_COUNT
        self._applied_exposure_mode = [None] * self.SLOT_COUNT
        self._last_reconnect_attempt = 0.0
        self._converter = None
        self._announced_no_backend = False
        # Per-slot dedup of error messages emitted from the hot loop.
        # _reconcile_slots / _apply_exposure_changes / grab all run every
        # iteration; without dedup a single persistent error floods
        # stdout with thousands of identical lines.  Each tracker is
        # cleared whenever the corresponding operation succeeds.
        # Attach errors are keyed by device-id (the unique device path
        # pylon reports) instead of slot, because reconcile_slots tries
        # every device against every empty slot — keying on slot means
        # the cached message gets clobbered on each device tried in the
        # same iteration and dedup never matches.
        self._last_attach_error_by_device = {}
        self._last_exposure_mode_error = [None] * self.SLOT_COUNT
        self._last_exposure_error = [None] * self.SLOT_COUNT
        self._last_grab_error = [None] * self.SLOT_COUNT
        # Serials the user explicitly disconnected from the camera GUI;
        # _reconcile_slots skips these so they don't auto-reattach.
        self._user_disabled_serials = set()
        # Most recent human-readable status / error message, displayed in
        # the camera GUI's bottom banner. Updated by _set_status() so the
        # GUI can poll it on a timer.
        self.latest_status = ""

        self.camera_available, self.camera_status_message = check_camera_backend()
        self.latest_status = self.camera_status_message
        if self.camera_available:
            self._init_backend()

    def _init_backend(self):
        try:
            self._tl_factory = pylon.TlFactory.GetInstance()
            self._converter = pylon.ImageFormatConverter()
            self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        except Exception as e:
            self.camera_available = False
            self.camera_status_message = f"Error initializing the camera backend ({e})"
            print(self.camera_status_message)

    def initialize_cameras(self):
        # backwards-compat alias for any caller still using the old name
        self._reconcile_slots(force=True)

    def start_capturing(self):
        if not self.camera_available:
            self.finished.emit()
            return
        self.running = True
        self.thread = threading.Thread(target=self.update_cameras, daemon=True)
        self.thread.start()

    def stop_capturing(self):
        self.running = False

    @pyqtSlot(bool)
    def set_default_exposure_time(self):
        if not self.exposure_auto:
            self.exposure_time_cam_1 = 400000
            self.exposure_time_cam_1_light = 10000
            self.exposure_time_cam_2 = 1000000
            self.exposure_time_cam_2_light = 20000
            self.exposure_time_cam_3 = 400000
            self.exposure_time_cam_3_light = 10000

            self.flag_default_exposure_time = True
            if self.variables.light:
                exposure_times = [
                    self.exposure_time_cam_1_light,
                    self.exposure_time_cam_2_light,
                    self.exposure_time_cam_3_light,
                ]
            else:
                exposure_times = [self.exposure_time_cam_1, self.exposure_time_cam_2, self.exposure_time_cam_3]
            self.emitter.cams_exposure_time_default.emit(exposure_times)
        else:
            print('Cannot set the default exposure time when auto exposure is on')

    @pyqtSlot(bool)
    def set_auto_exposure_time(self):
        if not self.exposure_auto:
            self.exposure_mode = 'Continuous'
            self.exposure_auto = True
        else:
            self.exposure_mode = 'Off'
            self.exposure_auto = False

    @pyqtSlot(int)
    def set_exposure_time_1(self, exposure_time):
        self.exposure_time_cam_1 = exposure_time

    @pyqtSlot(int)
    def set_exposure_time_2(self, exposure_time):
        self.exposure_time_cam_2 = exposure_time

    @pyqtSlot(int)
    def set_exposure_time_3(self, exposure_time):
        self.exposure_time_cam_3 = exposure_time

    def _exposure_for_slot(self, slot):
        if slot == 0:
            return self.exposure_time_cam_1
        if slot == 1:
            return self.exposure_time_cam_2
        return self.exposure_time_cam_3

    def _close_slot(self, slot):
        cam = self._slots[slot]
        self._slots[slot] = None
        self._applied_exposure[slot] = None
        self._applied_exposure_mode[slot] = None
        if cam is None:
            return
        try:
            if cam.IsGrabbing():
                cam.StopGrabbing()
        except Exception:
            pass
        try:
            if cam.IsOpen():
                cam.Close()
        except Exception:
            pass

    def _reconcile_slots(self, force=False):
        """Fill empty slots from currently-enumerated devices.

        Slots are bound by serial number: a slot remembers the first serial
        it was assigned to and will only re-attach to that same physical
        camera. Brand-new serials fill the lowest empty slot that has never
        been claimed.
        """

        if pylon is None:
            return
        now = time.time()
        if not force and now - self._last_reconnect_attempt < self.RECONNECT_INTERVAL:
            return
        self._last_reconnect_attempt = now

        try:
            devices = self._tl_factory.EnumerateDevices()
        except Exception as e:
            if not self._announced_no_backend:
                print(f"Camera enumeration failed: {e}")
                self._announced_no_backend = True
            return
        self._announced_no_backend = False

        by_serial = {}
        for dev in devices:
            try:
                serial_number = dev.GetSerialNumber()
            except Exception:
                serial_number = None
            if serial_number:
                by_serial[serial_number] = dev

        used_serials = set()
        for slot in range(self.SLOT_COUNT):
            if self._slots[slot] is not None:
                try:
                    if self._slots[slot].IsOpen():
                        used_serials.add(self._slot_serials[slot])
                        continue
                except Exception:
                    pass
                self._close_slot(slot)
            sn = self._slot_serials[slot]
            if sn in self._user_disabled_serials:
                continue
            if sn is not None and sn in by_serial:
                if self._attach_slot(slot, by_serial[sn]):
                    used_serials.add(sn)

        for sn, dev in by_serial.items():
            if sn in used_serials or sn in self._user_disabled_serials:
                continue
            for slot in range(self.SLOT_COUNT):
                if self._slots[slot] is not None:
                    continue
                if self._slot_serials[slot] is not None:
                    continue
                if self._attach_slot(slot, dev):
                    self._slot_serials[slot] = sn
                    used_serials.add(sn)
                    break

    def _device_key(self, device_info):
        """Return a stable identifier for *device_info*.

        Prefers the serial number; falls back to the device's full path
        (which still uniquely identifies the physical USB endpoint).
        """
        for getter in ("GetSerialNumber", "GetFullName", "GetDeviceID"):
            fn = getattr(device_info, getter, None)
            if fn is None:
                continue
            try:
                value = fn()
            except Exception:
                continue
            if value:
                return str(value)
        return repr(device_info)

    def _attach_slot(self, slot, device_info):
        device_key = self._device_key(device_info)
        try:
            cam = pylon.InstantCamera(self._tl_factory.CreateDevice(device_info))
            cam.Open()
            try:
                cam.ExposureAuto.SetValue('Off')
            except Exception:
                pass
            try:
                cam.ExposureTime.SetValue(self._exposure_for_slot(slot))
            except Exception:
                pass
            cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        except Exception as e:
            # Dedup per-device. The reconcile loop tries every visible
            # device against every empty slot, so a single stuck device
            # would otherwise emit one message per (slot, iteration).
            msg = str(e)
            if self._last_attach_error_by_device.get(device_key) != msg:
                self._last_attach_error_by_device[device_key] = msg
                print(f"Could not attach camera (slot {slot}): {msg}")
                self._set_status(f"Camera {device_key}: {msg}")
            return False
        # Successful attach - reset the dedup state for this device so
        # the next failure (if any) prints again.
        self._last_attach_error_by_device.pop(device_key, None)
        self._slots[slot] = cam
        self._applied_exposure[slot] = self._exposure_for_slot(slot)
        self._applied_exposure_mode[slot] = 'Off'
        try:
            sn = device_info.GetSerialNumber()
        except Exception:
            sn = self._slot_serials[slot]
        print(f"Camera attached in slot {slot} (serial={sn}).")
        self._set_status(f"Camera {sn} attached (slot {slot}).")
        return True

    def _set_status(self, message):
        """Publish a human-readable status / error string for the GUI."""
        self.latest_status = message or ""

    # ------------------------------------------------------------ public API

    def list_cameras(self):
        """Return one dict per currently-detected Basler camera.

        Each dict has ``serial``, ``model``, ``slot`` (or None when the
        device is detected but not bound), ``attached`` (bool), and
        ``user_disabled`` (bool — user explicitly disconnected it).
        """

        if pylon is None or not self.camera_available:
            return []
        try:
            devices = self._tl_factory.EnumerateDevices()
        except Exception as e:
            self._set_status(f"Camera enumeration failed: {e}")
            return []

        # serial -> slot, derived from currently-open slots
        slot_by_serial = {}
        for slot in range(self.SLOT_COUNT):
            cam = self._slots[slot]
            sn = self._slot_serials[slot]
            if cam is not None and sn is not None:
                slot_by_serial[sn] = slot

        out = []
        seen = set()
        for dev in devices:
            try:
                sn = dev.GetSerialNumber()
            except Exception:
                sn = None
            if not sn or sn in seen:
                continue
            seen.add(sn)
            try:
                model = dev.GetModelName()
            except Exception:
                model = ""
            out.append(
                {
                    "serial": sn,
                    "model": model,
                    "slot": slot_by_serial.get(sn),
                    "attached": sn in slot_by_serial,
                    "user_disabled": sn in self._user_disabled_serials,
                }
            )
        return out

    def disconnect_serial(self, serial):
        """Close any open slot bound to *serial* and prevent auto-reattach.

        Use :meth:`connect_serial` to re-enable it.
        """
        if not serial:
            return
        self._user_disabled_serials.add(serial)
        for slot in range(self.SLOT_COUNT):
            if self._slot_serials[slot] == serial and self._slots[slot] is not None:
                self._close_slot(slot)
                self._set_status(f"Disconnected camera {serial} from slot {slot}.")
                return
        self._set_status(f"Camera {serial} marked disconnected.")

    def connect_serial(self, serial):
        """Permit *serial* to attach again and force a reconcile pass."""
        if not serial:
            return
        self._user_disabled_serials.discard(serial)
        # Drop the cached attach error so the next failure (if any)
        # prints fresh.
        self._last_attach_error_by_device.pop(serial, None)
        try:
            self._reconcile_slots(force=True)
        except Exception as e:
            self._set_status(f"Could not connect {serial}: {e}")
            return
        for slot in range(self.SLOT_COUNT):
            if self._slot_serials[slot] == serial and self._slots[slot] is not None:
                self._set_status(f"Connected camera {serial} (slot {slot}).")
                return
        self._set_status(f"Camera {serial} could not be attached — see terminal log.")

    def _apply_exposure_changes(self):
        for slot in range(self.SLOT_COUNT):
            cam = self._slots[slot]
            if cam is None:
                continue
            target_mode = self.exposure_mode
            if self._applied_exposure_mode[slot] != target_mode:
                try:
                    cam.ExposureAuto.SetValue(target_mode)
                    self._applied_exposure_mode[slot] = target_mode
                    self._last_exposure_mode_error[slot] = None
                except Exception as e:
                    msg = str(e)
                    if self._last_exposure_mode_error[slot] != msg:
                        self._last_exposure_mode_error[slot] = msg
                        print(f"Slot {slot} exposure-auto change failed: {msg}")
            target_exposure = self._exposure_for_slot(slot)
            if self._applied_exposure[slot] != target_exposure:
                try:
                    cam.ExposureTime.SetValue(target_exposure)
                    self._applied_exposure[slot] = target_exposure
                    self._last_exposure_error[slot] = None
                except Exception as e:
                    msg = str(e)
                    if self._last_exposure_error[slot] != msg:
                        self._last_exposure_error[slot] = msg
                        print(f"Slot {slot} exposure change failed: {msg}")

    def update_cameras(self):
        if not self.camera_available:
            self.finished.emit()
            return

        last_save_time = time.time()
        self._reconcile_slots(force=True)

        while self.running:
            self._reconcile_slots()
            self._apply_exposure_changes()

            any_open = False
            grabbed_images = [None] * self.SLOT_COUNT
            for slot in range(self.SLOT_COUNT):
                cam = self._slots[slot]
                if cam is None:
                    continue
                any_open = True
                try:
                    if not cam.IsGrabbing():
                        cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                    grab = cam.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)
                    try:
                        if grab.GrabSucceeded():
                            image = self._converter.Convert(grab)
                            grabbed_images[slot] = image.GetArray()
                    finally:
                        grab.Release()
                except Exception as e:
                    msg = str(e)
                    if self._last_grab_error[slot] != msg:
                        self._last_grab_error[slot] = msg
                        print(f"Slot {slot} grab failed: {msg}; will try to reconnect.")
                        self._set_status(f"Slot {slot} grab failed: {msg}")
                    self._close_slot(slot)
                else:
                    # Successful grab - clear the dedup state.
                    self._last_grab_error[slot] = None

            self._emit_images(grabbed_images)

            if self.variables.clear_index_save_image:
                self.variables.clear_index_save_image = False
                self.index_save_image = 0

            now = time.time()
            if now - last_save_time >= self.variables.save_meta_interval_camera and self.variables.start_flag:
                last_save_time = now
                self._save_screenshots(grabbed_images)

            if self.variables.light_switch or self.flag_default_exposure_time:
                self.light_switch()
                self.variables.light_switch = False
                self.flag_default_exposure_time = False

            time.sleep(0.5)

            if not self.variables.flag_camera_grab:
                break

        for slot in range(self.SLOT_COUNT):
            self._close_slot(slot)
        self.finished.emit()

    def _emit_images(self, images):
        # Slot 0 -> img0 (side overview), Slot 1 -> img1 (top overview).
        # img2 mirrors slot 0 so the existing 'angle' view doesn't go blank
        # when only one camera is connected.
        img0 = images[0] if len(images) > 0 else None
        img1 = images[1] if len(images) > 1 else None
        if img0 is not None:
            self.emitter.img0_orig.emit(np.swapaxes(img0, 0, 1))
        if img1 is not None:
            self.emitter.img1_orig.emit(np.swapaxes(img1, 0, 1))
        angle_src = img0 if img0 is not None else img1
        if angle_src is not None:
            self.emitter.img2_orig.emit(np.swapaxes(angle_src, 0, 1))

    def _save_screenshots(self, images):
        path_meta = self.variables.path_meta
        labels = ("camera_side", "camera_top", "camera_45")
        save_sources = list(images)
        if len(save_sources) >= 1 and (len(save_sources) < 3 or save_sources[2] is None):
            while len(save_sources) < 3:
                save_sources.append(None)
            save_sources[2] = save_sources[0]
        for label, img in zip(labels, save_sources):
            if img is None:
                continue
            try:
                cv2.imwrite(f"{path_meta}/{label}_{self.index_save_image}.png", img)
            except Exception as e:
                print(f"Could not save {label} screenshot: {e}")
        self.index_save_image += 1
        time.sleep(0.5)

    def light_switch(self):
        if self.exposure_auto:
            return
        try:
            light_on = bool(self.variables.light)
            slot_targets = (
                self.exposure_time_cam_1_light if light_on else self.exposure_time_cam_1,
                self.exposure_time_cam_2_light if light_on else self.exposure_time_cam_2,
                self.exposure_time_cam_3_light if light_on else self.exposure_time_cam_3,
            )
            for slot in range(self.SLOT_COUNT):
                cam = self._slots[slot]
                if cam is None:
                    continue
                target = slot_targets[slot]
                cam.ExposureTime.SetValue(target)
                self._applied_exposure[slot] = target
        except Exception as e:
            print(f"Error in switching the light: {e}")
