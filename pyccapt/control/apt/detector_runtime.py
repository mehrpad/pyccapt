"""Detector process orchestration for APT experiments."""

from __future__ import annotations

import multiprocessing
from dataclasses import dataclass
from typing import Any, Callable

from pyccapt.control.apt.detector_models import (
    HSD_MODEL,
    ROENTDEK_MODEL,
    SIMULATOR_MODEL,
    normalize_counter_source,
    normalize_tdc_model,
)
from pyccapt.control.drs import drs
from pyccapt.control.apt import simulator
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

    tdc_model = normalize_tdc_model(conf.get("tdc_model"))
    counter_source = normalize_counter_source(variables.counter_source)
    variables.counter_source = counter_source

    if tdc_model == "Surface_Concept" and counter_source == "TDC":
        runtime.stop_event = event_factory()
        runtime.tdc_process = process_factory(
            target=tdc_surface_concept.experiment_measure,
            args=(variables, x_plot, y_plot, t_plot, main_v_dc_plot, runtime.stop_event),
        )
        runtime.tdc_process.start()
        return runtime

    if tdc_model == ROENTDEK_MODEL and counter_source == "TDC":
        runtime.stop_event = event_factory()
        runtime.tdc_process = process_factory(
            target=tdc_roentdek.experiment_measure,
            args=(variables, x_plot, y_plot, t_plot, main_v_dc_plot, runtime.stop_event),
        )
        runtime.tdc_process.start()
        return runtime

    if tdc_model == SIMULATOR_MODEL:
        runtime.stop_event = event_factory()
        runtime.tdc_process = process_factory(
            target=simulator.experiment_measure,
            args=(variables, x_plot, y_plot, t_plot, main_v_dc_plot, runtime.stop_event),
        )
        runtime.tdc_process.start()
        return runtime

    if tdc_model == HSD_MODEL and counter_source == "HSD":
        runtime.stop_event = event_factory()
        runtime.hsd_process = process_factory(
            target=drs.experiment_measure,
            args=(variables, x_plot, y_plot, t_plot, main_v_dc_plot, runtime.stop_event),
        )
        runtime.hsd_process.start()

    return runtime


def join_detector_processes(conf: dict[str, Any], variables: Any, runtime: DetectorRuntime) -> None:
    """Join detector worker process handles if they exist."""
    if conf.get("tdc") != "on":
        return

    process = runtime.tdc_process if variables.counter_source == "TDC" else runtime.hsd_process
    if process is None:
        return
    process.join(2)
    if process.is_alive():
        process.terminate()
        process.join(2)
