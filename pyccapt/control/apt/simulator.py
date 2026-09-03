"""Hardware-free detector backend for development and integration tests."""

from __future__ import annotations

import time

import numpy as np


def experiment_measure(variables, x_plot, y_plot, t_plot, main_v_dc_plot, stop_event) -> None:
    """Generate deterministic synthetic detector events until stopped."""
    rng = np.random.default_rng(42)
    event_count = 0
    raw_count = 0
    try:
        while not variables.stop_flag and not variables.flag_stop_tdc and not stop_event.is_set():
            batch_size = 128
            x = rng.normal(0.0, 0.8, batch_size)
            y = rng.normal(0.0, 0.8, batch_size)
            tof = rng.normal(100.0, 2.0, batch_size)
            voltage = np.full(batch_size, float(variables.specimen_voltage))
            x_plot.write(x)
            y_plot.write(y)
            t_plot.write(tof)
            main_v_dc_plot.write(voltage)
            event_count += batch_size
            raw_count += batch_size * 4
            variables.total_ions = event_count
            variables.total_raw_signals = raw_count
            pulse_frequency = max(float(variables.pulse_frequency) * 1000.0, 1.0)
            variables.detection_rate_current = batch_size * 100.0 / pulse_frequency
            variables.detection_rate_current_plot = variables.detection_rate_current
            time.sleep(0.02)
    except Exception as exc:
        variables.flag_tdc_failure = True
        variables.detector_error = f"Simulator failed: {exc.__class__.__name__}: {exc}"
    finally:
        variables.flag_finished_tdc = True
