"""Application and Windows taskbar icon setup shared by all GUI launchers."""

import sys

from PyQt6 import QtGui, QtWidgets

from pyccapt.control.core import runtime


# Incrementing the suffix avoids Windows reusing the stale icon cache that was
# associated with the original taskbar identity.
WINDOWS_APP_USER_MODEL_ID = "OXCART.PyCCAPT.Control.2"
ICON_CANDIDATES = ("pyccapt.ico", "logo.png")


def set_windows_app_user_model_id(app_id=WINDOWS_APP_USER_MODEL_ID):
    """Give this process its own Windows taskbar identity."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        result = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
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
    return True
