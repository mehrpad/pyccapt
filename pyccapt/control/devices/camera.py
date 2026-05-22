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
        # Cameras start with ExposureAuto = Continuous so the user gets a
        # usable image even before they touch the exposure inputs. The
        # GUI's auto-exposure button toggles this back to manual ('Off').
        self.exposure_auto = True
        self.exposure_mode = 'Continuous'
        self.emitter = emitter
        self.variables = variables

        self.running = False
        self.index_save_image = 0
        # Defaults for the "light off" case (microseconds). 2,000,000 µs
        # (2 s) is the value the user previously dialled in by hand to
        # see the puck in the dark — matches the new auto-exposure
        # upper limit and the default reset button.
        self.exposure_time_cam_1 = 2_000_000
        self.exposure_time_cam_1_light = 10000
        self.exposure_time_cam_2 = 2_000_000
        self.exposure_time_cam_2_light = 20000
        self.exposure_time_cam_3 = 2_000_000
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
            self.exposure_time_cam_1 = 2_000_000
            self.exposure_time_cam_1_light = 10000
            self.exposure_time_cam_2 = 2_000_000
            self.exposure_time_cam_2_light = 20000
            self.exposure_time_cam_3 = 2_000_000
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
                cam.ExposureAuto.SetValue(self.exposure_mode)
            except Exception:
                pass
            if not self.exposure_auto:
                try:
                    cam.ExposureTime.SetValue(self._exposure_for_slot(slot))
                except Exception:
                    pass
            self._apply_quality_settings(cam)
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
        # If we're attaching in auto mode we didn't write ExposureTime, so
        # leave the cache empty — the manual-mode branch in
        # _apply_exposure_changes will push the configured value the
        # first time the user disables auto.
        self._applied_exposure[slot] = None if self.exposure_auto else self._exposure_for_slot(slot)
        self._applied_exposure_mode[slot] = self.exposure_mode
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

    @staticmethod
    def _try_set(node, *values):
        """Try a list of candidate values on a pylon node, accept the first one that sticks.

        Returns True on success, False otherwise. Used for image-quality
        features that exist on some Basler models but not others (e.g.
        BslSharpnessEnhancement on dart/ace2 cameras only).
        """
        if node is None:
            return False
        for value in values:
            try:
                node.SetValue(value)
                return True
            except Exception:
                continue
        return False

    def _apply_quality_settings(self, cam):
        """Push image-quality tweaks that help with sample alignment.

        All operations are best-effort: every Basler model exposes a
        slightly different feature set, so each tweak is wrapped so that
        an unsupported camera silently keeps its defaults rather than
        bailing out of the attach path.

        The goals are:
        * Pick the highest mono / colour pixel format the camera supports
          (12-bit if available) for better tonal range before we push to
          the BGR8 converter.
        * Enable Basler's PGI / sharpness enhancement and gamma so dark
          alignment features (specimen edges, puck contours) lift out of
          the background.
        * Disable auto-white-balance jitter on colour cameras — for
          alignment we want a stable image, not one that re-balances on
          every frame.
        """
        # Pixel format: prefer 12-bit. The pyqtgraph view will still
        # render 8-bit (via the converter) but a wider source format
        # gives the auto-level step more headroom.
        try:
            pf = cam.PixelFormat
            current = pf.GetValue()
            preferred = (
                "BayerRG12",
                "BayerBG12",
                "BayerGR12",
                "BayerGB12",
                "Mono12",
                "BayerRG8",
                "BayerBG8",
                "Mono8",
            )
            available = set()
            try:
                available = {entry.GetSymbolic() for entry in pf.GetEntries() if entry.IsAvailable()}
            except Exception:
                pass
            for cand in preferred:
                if cand in available and cand != current:
                    try:
                        pf.SetValue(cand)
                        break
                    except Exception:
                        continue
        except Exception:
            pass

        # Sharpness / PGI: Basler's image enhancement pipeline.
        # BslSharpnessEnhancement is the modern node; older firmware
        # exposes SharpnessEnhancement (no Bsl prefix) and the legacy
        # PgiMode toggle.
        try:
            self._try_set(getattr(cam, "BslSharpnessEnhancement", None), 1.5)
        except Exception:
            pass
        try:
            self._try_set(getattr(cam, "SharpnessEnhancement", None), 1.5)
        except Exception:
            pass
        try:
            self._try_set(getattr(cam, "PgiMode", None), "On")
        except Exception:
            pass

        # Gamma < 1 lifts shadows — useful when the light is off and
        # the specimen detail lives in the darker half of the histogram.
        try:
            gsel = getattr(cam, "GammaSelector", None)
            if gsel is not None:
                self._try_set(gsel, "User")
            gamma = getattr(cam, "Gamma", None)
            if gamma is not None:
                gamma.SetValue(0.7)
        except Exception:
            pass

        # Lock auto-white-balance off so the image is stable while the
        # operator is centring the specimen.
        try:
            wb = getattr(cam, "BalanceWhiteAuto", None)
            if wb is not None:
                self._try_set(wb, "Off")
        except Exception:
            pass

        # Black-level: keep at default but make sure it isn't pinned to
        # an inherited high value from a previous run.
        try:
            bl = getattr(cam, "BlackLevel", None)
            if bl is not None:
                bl.SetValue(0)
        except Exception:
            pass

        # Auto-exposure bounds.
        #
        # Sample alignment with the light off is very dark, and the
        # user was bumping ExposureTime up to ~2,000,000 µs by hand to
        # see the puck. Out of the box Basler caps AutoExposureTime at
        # something much shorter (often 100,000–500,000 µs), so the
        # firmware auto-loop hits the ceiling and gives up before the
        # image is bright enough. Raising the ceiling lets the auto
        # loop keep extending exposure for dark scenes, while a higher
        # target brightness (≈0.6 vs. the default 0.5) shifts the
        # set-point a notch brighter so dim specimen edges remain
        # visible. Feature names vary across Basler model families
        # (ace2/dart use AutoExposureTimeUpperLimit, older ace uses
        # AutoExposureTimeAbsUpperLimit, …), so each set is wrapped.
        DARK_EXPOSURE_UPPER_US = 3_000_000  # 3 s — covers "light off" alignment.
        SHORT_EXPOSURE_LOWER_US = 100       # 100 µs — fast end for "light on".
        TARGET_BRIGHTNESS = 0.6             # 0..1, default ~0.5; lift dark scenes.

        for name in ("AutoExposureTimeUpperLimit", "AutoExposureTimeAbsUpperLimit"):
            node = getattr(cam, name, None)
            if node is not None:
                try:
                    node.SetValue(DARK_EXPOSURE_UPPER_US)
                    break
                except Exception:
                    continue
        for name in ("AutoExposureTimeLowerLimit", "AutoExposureTimeAbsLowerLimit"):
            node = getattr(cam, name, None)
            if node is not None:
                try:
                    node.SetValue(SHORT_EXPOSURE_LOWER_US)
                    break
                except Exception:
                    continue
        for name in ("AutoTargetBrightness", "AutoTargetValue", "BslAutoTargetBrightness"):
            node = getattr(cam, name, None)
            if node is not None:
                try:
                    # Some firmwares expect 0..1, some 0..255. Try both.
                    self._try_set(node, TARGET_BRIGHTNESS, int(TARGET_BRIGHTNESS * 255))
                    break
                except Exception:
                    continue

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
                    # Gain follows exposure: in auto mode, let the camera
                    # also auto-tune the gain so low-light / high-light
                    # scenes (light on/off) converge faster and cleaner.
                    # In manual mode, GainAuto must be Off or the camera
                    # will keep overriding the user's exposure value.
                    try:
                        cam.GainAuto.SetValue(target_mode)
                    except Exception:
                        pass
                    # Auto-mode wrote its own value into ExposureTime; the
                    # cache no longer reflects the camera. Invalidate so
                    # the manual block below always re-pushes the user's
                    # configured exposure on the next iteration.
                    self._applied_exposure[slot] = None
                    self._applied_exposure_mode[slot] = target_mode
                    self._last_exposure_mode_error[slot] = None
                except Exception as e:
                    msg = str(e)
                    if self._last_exposure_mode_error[slot] != msg:
                        self._last_exposure_mode_error[slot] = msg
                        print(f"Slot {slot} exposure-auto change failed: {msg}")
            # Skip pushing manual exposure values while the camera is in
            # auto mode — the camera owns the value and writing here just
            # races with the firmware loop.
            if self.exposure_mode != 'Off':
                continue
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
