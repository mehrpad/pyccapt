import multiprocessing
import signal
import sys

from PyQt6 import QtWidgets

from pyccapt.control.core import runtime
from pyccapt.control.gui import gui_main


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

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
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
