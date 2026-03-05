"""Detector process orchestration for APT experiments."""

from __future__ import annotations

import multiprocessing
from dataclasses import dataclass
from typing import Any, Callable

from pyccapt.control.drs import drs
from pyccapt.control.tdc_roentdek import tdc_roentdek
from pyccapt.control.tdc_surface_concept import tdc_surface_concept


@dataclass
class DetectorRuntime:
    """Holds worker process handles for detector backends."""

    stop_event: Any | None = None
    tdc_process: Any | None = None
    hsd_process: Any | None = None


def start_detector_processes(
    conf: dict[str, Any],
    variables: Any,
    x_plot: Any,
    y_plot: Any,
    t_plot: Any,
    main_v_dc_plot: Any,
    *,
    process_factory: Callable[..., Any] = multiprocessing.Process,
    event_factory: Callable[[], Any] = multiprocessing.Event,
) -> DetectorRuntime:
    """Start the detector worker process based on configuration."""
    runtime = DetectorRuntime()

    if conf.get("tdc") != "on":
        return runtime

    if conf.get("tdc_model") == "Surface_Consept" and variables.counter_source == "TDC":
        runtime.stop_event = event_factory()
        runtime.tdc_process = process_factory(
            target=tdc_surface_concept.experiment_measure,
            args=(variables, x_plot, y_plot, t_plot, main_v_dc_plot, runtime.stop_event),
        )
        runtime.tdc_process.start()
        return runtime

    if conf.get("tdc_model") == "RoentDek" and variables.counter_source == "TDC":
        runtime.tdc_process = process_factory(target=tdc_roentdek.experiment_measure, args=(variables,))
        runtime.tdc_process.start()
        return runtime

    if conf.get("tdc_model") == "HSD" and variables.counter_source == "HSD":
        runtime.hsd_process = process_factory(target=drs.experiment_measure, args=(variables,))
        runtime.hsd_process.start()

    return runtime


def join_detector_processes(conf: dict[str, Any], variables: Any, runtime: DetectorRuntime) -> None:
    """Join detector worker process handles if they exist."""
    if conf.get("tdc") != "on":
        return

    if variables.counter_source == "TDC" and runtime.tdc_process is not None:
        runtime.tdc_process.join(2)
        if runtime.tdc_process.is_alive():
            runtime.tdc_process.join(1)
    elif variables.counter_source == "HSD" and runtime.hsd_process is not None:
        runtime.hsd_process.join(2)
        if runtime.hsd_process.is_alive():
            runtime.hsd_process.join(1)
