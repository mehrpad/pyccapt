import multiprocessing
import signal
import sys

from PyQt6 import QtGui, QtWidgets

from pyccapt.control.core import runtime
from pyccapt.control.gui import gui_main


def _set_windows_app_id(app_id: str = "OXCART.PyCCAPT.Control.1") -> None:
    """Register an explicit Windows Application User Model ID.

    Windows groups taskbar buttons by a process's AppUserModelID. A Python
    GUI launched via ``python``/``pythonw`` inherits the interpreter's id,
    so the taskbar shows the Python icon even though the window's title-bar
    icon is set. Declaring our own id makes Windows treat PyCCAPT as a
    distinct application and use the window icon in the taskbar. No-op on
    non-Windows platforms.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        # Older Windows or a locked-down environment; the taskbar just
        # falls back to the interpreter icon -- not worth failing startup.
        pass


def main():
    """
    Load the GUI based on the configuration file.

    This function reads the configuration file, initializes global experiment variables, and
    shows the GUI window.

    Args:
            None

    Returns:
            None
    """
    try:
        conf, _ = runtime.load_project_config()
    except Exception as exc:
        print("Cannot load the configuration file")
        print(exc)
        sys.exit()

    shared = runtime.create_shared_context(conf)

    # Set the Windows taskbar app id BEFORE the QApplication is created so
    # the taskbar uses the PyCCAPT logo instead of the Python icon.
    _set_windows_app_id()

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    # Application-level icon: Qt uses this as the default for every
    # top-level window (title bar) and, together with the app id above,
    # for the taskbar button. The main window also sets it explicitly in
    # retranslateUi, so this is the belt-and-suspenders default.
    try:
        app.setWindowIcon(QtGui.QIcon(str(runtime.project_path("files", "logo.png"))))
    except Exception as exc:
        print(f"Could not load application icon (non-fatal): {exc}")
    window = gui_main.MyPyCCAPT(
        shared.variables,
        conf,
        shared.x_plot,
        shared.y_plot,
        shared.t_plot,
        shared.main_v_dc_plot,
    )

    # Release the shared context (ring buffers, Manager) when the Qt
    # event loop exits, regardless of how exit was triggered. Previously
    # ``release_shared_context`` was never invoked, so named shared
    # memory blocks (pyccapt_xplot_*) leaked into /dev/shm on Linux and
    # the Manager subprocess could persist after the GUI closed.
    def _on_about_to_quit():
        try:
            runtime.release_shared_context(shared)
        except Exception as exc:
            print(f"release_shared_context failed (non-fatal): {exc}")

    app.aboutToQuit.connect(_on_about_to_quit)

    # Wire Ctrl-C in the terminal to a graceful app.quit -- without this
    # the Qt event loop swallows SIGINT and the only way to exit is the
    # X button. ``app.quit()`` triggers the aboutToQuit signal above so
    # cleanup runs the same way as a normal close.
    def _handle_sigint(signum, frame):
        print("SIGINT received; requesting Qt event-loop shutdown.")
        QtWidgets.QApplication.quit()

    try:
        signal.signal(signal.SIGINT, _handle_sigint)
    except (ValueError, OSError):
        # signal.signal only works from the main thread; if we're in
        # a context where it doesn't (test harnesses, embedded use),
        # skip silently.
        pass

    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    # Required on Windows when packaged with PyInstaller / cx_Freeze
    # (or any other ``spawn``-based multiprocessing setup) so child
    # processes do not re-import the GUI and pop a second window.
    multiprocessing.freeze_support()
    main()
