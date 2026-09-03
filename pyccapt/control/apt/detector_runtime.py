"""Detector backend interface and process orchestration."""

from __future__ import annotations

import json
import multiprocessing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from pyccapt.control.apt import simulator
from pyccapt.control.apt.detector_models import (
    HSD_MODEL,
    ROENTDEK_MODEL,
    SIMULATOR_MODEL,
    normalize_counter_source,
    normalize_tdc_model,
)
from pyccapt.control.drs import drs
from pyccapt.control.tdc_roentdek import tdc_roentdek
from pyccapt.control.tdc_surface_concept import tdc_surface_concept


@dataclass(frozen=True)
class DetectorHealth:
    name: str
    running: bool
    exit_code: int | None
    stop_requested: bool
    message: str = ""


class DetectorBackend(ABC):
    """Common lifecycle, health, and chunk contract for every detector."""

    name: str
    counter_source: str

    @abstractmethod
    def start(self) -> None:
        """Start acquisition."""

    @abstractmethod
    def stop(self) -> None:
        """Request cooperative stop."""

    @abstractmethod
    def join(self, timeout: float = 2.0) -> None:
        """Wait for stop and escalate if needed."""

    @abstractmethod
    def health(self) -> DetectorHealth:
        """Return a low-rate health snapshot."""

    @abstractmethod
    def stream_chunks(self) -> Iterator[dict[str, Any]]:
        """Yield completed chunk-manifest records."""


class ProcessDetectorBackend(DetectorBackend):
    """Detector implemented by a child process and cooperative stop event."""

    def __init__(
        self,
        *,
        name: str,
        counter_source: str,
        target: Callable[..., Any],
        args: tuple[Any, ...],
        variables: Any,
        process_factory: Callable[..., Any],
        event_factory: Callable[[], Any],
    ) -> None:
        self.name = name
        self.counter_source = counter_source
        self.variables = variables
        self.stop_event = event_factory()
        self.process = process_factory(target=target, args=(*args, self.stop_event))

    def start(self) -> None:
        self.process.start()

    def stop(self) -> None:
        self.stop_event.set()

    def join(self, timeout: float = 2.0) -> None:
        self.process.join(timeout)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout)

    def health(self) -> DetectorHealth:
        running = bool(self.process.is_alive())
        exit_code = getattr(self.process, "exitcode", None)
        requested = bool(self.stop_event.is_set()) if hasattr(self.stop_event, "is_set") else False
        message = "running" if running else ("stopped" if exit_code in {None, 0} else f"exited with code {exit_code}")
        return DetectorHealth(self.name, running, exit_code, requested, message)

    def stream_chunks(self) -> Iterator[dict[str, Any]]:
        path_value = str(getattr(self.variables, "path", ""))
        if not path_value:
            return
        chunk_dir = Path(path_value) / "temp_data" / "chunks"
        for manifest in sorted(chunk_dir.glob("manifest*.jsonl")):
            with manifest.open("r", encoding="utf-8") as stream:
                for line in stream:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("status") == "complete":
                        yield record


class SurfaceConceptBackend(ProcessDetectorBackend):
    pass


class RoentDekBackend(ProcessDetectorBackend):
    pass


class HSDBackend(ProcessDetectorBackend):
    pass


class SimulatorBackend(ProcessDetectorBackend):
    pass


@dataclass
class DetectorRuntime:
    """Backend plus legacy process aliases retained for GUI compatibility."""

    backend: DetectorBackend | None = None
    stop_event: Any | None = None
    tdc_process: Any | None = None
    hsd_process: Any | None = None


def _build_backend(
    conf: dict[str, Any],
    variables: Any,
    x_plot: Any,
    y_plot: Any,
    t_plot: Any,
    main_v_dc_plot: Any,
    process_factory: Callable[..., Any],
    event_factory: Callable[[], Any],
) -> ProcessDetectorBackend | None:
    model = normalize_tdc_model(conf.get("tdc_model"))
    source = normalize_counter_source(variables.counter_source)
    variables.counter_source = source
    args = (variables, x_plot, y_plot, t_plot, main_v_dc_plot)
    common = dict(variables=variables, args=args, process_factory=process_factory, event_factory=event_factory)

    if model == "Surface_Concept" and source == "TDC":
        return SurfaceConceptBackend(
            name="Surface Concept", counter_source="TDC",
            target=tdc_surface_concept.experiment_measure, **common,
        )
    if model == ROENTDEK_MODEL and source == "TDC":
        return RoentDekBackend(
            name="RoentDek", counter_source="TDC",
            target=tdc_roentdek.experiment_measure, **common,
        )
    if model == HSD_MODEL and source == "HSD":
        return HSDBackend(
            name="HSD", counter_source="HSD", target=drs.experiment_measure, **common,
        )
    if model == SIMULATOR_MODEL:
        # Copy only explicit simulator controls into the process namespace.
        # These knobs make dry runs deterministic and allow CI to exercise a
        # real worker-failure path without any detector SDK.
        for key, default in (
            ("simulator_seed", 42),
            ("simulator_batch_size", 128),
            ("simulator_interval_s", 0.02),
            ("simulator_fail_after_batches", 0),
        ):
            setattr(variables, key, conf.get(key, default))
        return SimulatorBackend(
            name="Simulator", counter_source=source, target=simulator.experiment_measure, **common,
        )
    return None


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
    """Construct and start the configured detector backend."""
    if conf.get("tdc") != "on":
        return DetectorRuntime()
    backend = _build_backend(
        conf, variables, x_plot, y_plot, t_plot, main_v_dc_plot,
        process_factory, event_factory,
    )
    if backend is None:
        return DetectorRuntime()
    backend.start()
    process = backend.process
    return DetectorRuntime(
        backend=backend,
        stop_event=backend.stop_event,
        tdc_process=process if backend.counter_source == "TDC" else None,
        hsd_process=process if backend.counter_source == "HSD" else None,
    )


def join_detector_processes(conf: dict[str, Any], variables: Any, runtime: DetectorRuntime) -> None:
    """Stop and join the selected backend, with legacy-handle fallback."""
    if conf.get("tdc") != "on":
        return
    if runtime.backend is not None:
        runtime.backend.stop()
        runtime.backend.join(2.0)
        return
    process = runtime.tdc_process if variables.counter_source == "TDC" else runtime.hsd_process
    if process is None:
        return
    process.join(2)
    if process.is_alive():
        process.terminate()
        process.join(2)
