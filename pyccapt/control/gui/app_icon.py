"""Application and Windows taskbar icon setup shared by all GUI launchers."""

import sys

from PyQt6 import QtCore, QtGui, QtWidgets

from pyccapt.control.core import runtime


# Incrementing the suffix avoids Windows reusing the stale icon cache that was
# associated with the original taskbar identity.
WINDOWS_APP_USER_MODEL_ID = "OXCART.PyCCAPT.Control.2"
ICON_CANDIDATES = ("pyccapt.ico", "logo.png")


class _WindowTaskbarIdentity(QtCore.QObject):
    """Apply native properties and release their Shell storage on destruction."""
    def __init__(self, window):
        super().__init__(window)
        self.hwnd = None
        window.winId()  # Create a hidden native window before binding its Shell identity.
        self.handle = window.windowHandle()
        self.handle.installEventFilter(self)
        self.apply()
        QtWidgets.QApplication.instance().aboutToQuit.connect(self.clear)

    def apply(self):
        from pyccapt.control.gui.windows_taskbar import set_window_identity

        hwnd = int(self.handle.winId())
        path = runtime.project_path('files', 'pyccapt.ico')
        if set_window_identity(hwnd, WINDOWS_APP_USER_MODEL_ID, path):
            self.hwnd = hwnd

    def clear(self):
        if self.hwnd is not None:
            from pyccapt.control.gui.windows_taskbar import set_window_identity

            set_window_identity(self.hwnd, None)
            self.hwnd = None

    def eventFilter(self, watched, event):
        if event.type() == QtCore.QEvent.Type.PlatformSurface:
            if event.surfaceEventType() == QtGui.QPlatformSurfaceEvent.SurfaceEventType.SurfaceAboutToBeDestroyed:
                self.clear()
            else:
                self.apply()
        return False


def set_windows_app_user_model_id(app_id=WINDOWS_APP_USER_MODEL_ID):
    """Give this process its own Windows taskbar identity."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        setter = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID
        setter.argtypes = [ctypes.c_wchar_p]
        setter.restype = ctypes.c_long
        result = setter(app_id)
        return result == 0
    except Exception:
        return False


def load_application_icon():
    """Load the multi-resolution Windows icon, with the PNG as fallback."""
    for filename in ICON_CANDIDATES:
        path = runtime.project_path("files", filename)
        if not path.is_file():
            continue
        icon = QtGui.QIcon(str(path))
        if not icon.isNull():
            return icon
    return QtGui.QIcon()


def apply_application_icon(app=None, window=None):
    """Apply PyCCAPT identity/icon at both application and window levels."""
    set_windows_app_user_model_id()
    app = app or QtWidgets.QApplication.instance()
    icon = load_application_icon()
    if icon.isNull():
        return False
    if app is not None:
        app.setApplicationName("PyCCAPT")
        app.setApplicationDisplayName("PyCCAPT")
        app.setOrganizationName("PyCCAPT")
        app.setWindowIcon(icon)
    if window is not None:
        window.setWindowIcon(icon)
        if sys.platform == 'win32' and app is not None and app.platformName() == 'windows' and window.isWindow():
            identity = getattr(window, '_pyccapt_taskbar_identity', None)
            if identity is None:
                window._pyccapt_taskbar_identity = _WindowTaskbarIdentity(window)
            else:
                identity.apply()
    return True
