"""Cooperative notebook jobs: keep the kernel responsive and discard cancelled results."""
from __future__ import annotations

from threading import Event, Thread

from pyccapt.calibration.core.exceptions import CalibrationStateError


class NotebookJob:
    def __init__(self, *, dispatch=None):
        self.cancel_event = Event()
        self.running = False
        self.thread = None
        if dispatch is None:
            try:
                from IPython import get_ipython
                loop = get_ipython().kernel.io_loop
                dispatch = loop.add_callback
            except (AttributeError, RuntimeError):
                dispatch = lambda callback: callback()
        self.dispatch = dispatch

    def start(self, operation, *, on_result, on_error, on_progress=lambda fraction, text: None,
              on_finished=lambda: None):
        if self.running:
            raise CalibrationStateError("A notebook job is already running")
        self.running = True
        self.cancel_event.clear()

        def work():
            def progress(fraction, text):
                self.dispatch(lambda: on_progress(float(fraction), text))
            try:
                result = operation(self.cancel_event, progress)
                def deliver():
                    try:
                        if self.cancel_event.is_set():
                            raise CalibrationStateError("Cancelled; computed result was discarded")
                        on_result(result)
                    except Exception as error:
                        on_error(error)
                    finally:
                        self.running = False
                        on_finished()
                self.dispatch(deliver)
            except Exception as error:
                def failure(caught=error):
                    try:
                        on_error(caught)
                    finally:
                        self.running = False
                        on_finished()
                self.dispatch(failure)
        self.thread = Thread(target=work, daemon=True, name="pyccapt-guided-preview")
        self.thread.start()

    def cancel(self):
        self.cancel_event.set()
