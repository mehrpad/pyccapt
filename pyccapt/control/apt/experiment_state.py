"""Experiment data/state helpers for APT control."""

from __future__ import annotations

import datetime
import re
from enum import Enum
from pathlib import Path
from typing import Any

from pyccapt.control.apt.detector_models import normalize_tdc_model
from pyccapt.control.core import runtime


class ExperimentState(str, Enum):
    """Authoritative lifecycle states shared by the GUI and workers."""

    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPING = "stopping"
    SAFE_OFF = "safe_off"
    FINALIZING = "finalizing"
    COMPLETE = "complete"
    FAILED = "failed"


_ALLOWED_TRANSITIONS = {
    ExperimentState.IDLE: {ExperimentState.INITIALIZING},
    ExperimentState.INITIALIZING: {
        ExperimentState.RUNNING,
        ExperimentState.STOPPING,
        ExperimentState.SAFE_OFF,
    },
    ExperimentState.RUNNING: {ExperimentState.STOPPING, ExperimentState.SAFE_OFF},
    ExperimentState.STOPPING: {ExperimentState.SAFE_OFF},
    ExperimentState.SAFE_OFF: {ExperimentState.FINALIZING},
    ExperimentState.FINALIZING: {ExperimentState.COMPLETE},
    ExperimentState.COMPLETE: {ExperimentState.IDLE, ExperimentState.INITIALIZING},
    ExperimentState.FAILED: {ExperimentState.IDLE, ExperimentState.INITIALIZING},
}


class InvalidExperimentTransition(RuntimeError):
    """Raised when a caller attempts an impossible lifecycle transition."""


def set_experiment_state(variables: Any, state: ExperimentState, error: str = "", *, strict: bool = True) -> None:
    """Validate and publish lifecycle state before related wake-up events."""
    current_value = str(getattr(variables, "experiment_state", ExperimentState.IDLE.value))
    try:
        current = ExperimentState(current_value)
    except ValueError:
        current = ExperimentState.IDLE
    if state != current and state != ExperimentState.FAILED:
        if state not in _ALLOWED_TRANSITIONS[current] and strict:
            raise InvalidExperimentTransition(f"Invalid experiment transition: {current.value} -> {state.value}")
    variables.experiment_state = state.value
    if error or state != ExperimentState.FAILED:
        variables.experiment_error = error


def _safe_path_component(value: Any, fallback: str) -> str:
    """Return a portable, traversal-safe directory-name component."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    cleaned = cleaned.strip(" ._-")
    return cleaned[:80] or fallback

_CLEAR_LIST_FIELDS = (
    "x",
    "y",
    "t",
    "channel",
    "time_data",
    "tdc_start_counter",
    "dld_start_counter",
    "time_stamp",
    "ch0",
    "ch1",
    "ch2",
    "ch3",
    "ch4",
    "ch5",
    "ch6",
    "ch7",
    "laser_intensity",
    "ch0_time",
    "ch0_wave",
    "ch1_time",
    "ch1_wave",
    "ch2_time",
    "ch2_wave",
    "ch3_time",
    "ch3_wave",
    "main_v_p",
    "main_counter",
    "main_raw_counter",
    "main_temperature",
    "main_chamber_vacuum",
    "main_v_dc_dld",
    "main_v_p_dld",
    "main_l_p_dld",
    "main_v_dc_tdc",
    "main_v_p_tdc",
    "main_l_p_tdc",
    "main_v_dc_drs",
    "main_v_p_drs",
    "main_l_p_drs",
)


def prepare_experiment_output_paths(variables: Any) -> tuple[Path, Path]:
    """Create experiment output path and metadata path values."""
    now = datetime.datetime.now()
    counter = _safe_path_component(variables.counter, "0")
    electrode = _safe_path_component(variables.electrode, "unknown-electrode")
    data_name = _safe_path_component(variables.hdf5_data_name, "experiment")
    # Microseconds make rapid retries unique while retaining a readable name.
    variables.exp_name = f"{counter}_{now.strftime('%b-%d-%Y_%H-%M-%S-%f')}_{electrode}_{data_name}"

    data_root = runtime.project_path("data").resolve()
    data_path = (data_root / variables.exp_name).resolve()
    if data_path.parent != data_root:
        raise ValueError("Experiment output path escapes the project data directory")
    meta_path = data_path / "meta_data"
    variables.path = str(data_path)
    variables.path_meta = str(meta_path)
    variables.log_path = variables.path_meta
    return data_path, meta_path


def ensure_output_directories(data_path: Path, meta_path: Path) -> None:
    """Create a fresh output directory; never merge two experiment runs."""
    data_path.mkdir(mode=0o777, parents=True, exist_ok=False)
    meta_path.mkdir(mode=0o777, parents=False, exist_ok=False)


def append_main_loop_results(
    variables: Any,
    main_counter: list[Any],
    main_raw_counter: list[Any],
    main_temperature: list[Any],
    main_chamber_vacuum: list[Any],
) -> None:
    """Push loop-side accumulation buffers into shared state."""
    variables.extend_to("main_counter", main_counter)
    variables.extend_to("main_raw_counter", main_raw_counter)
    variables.extend_to("main_temperature", main_temperature)
    variables.extend_to("main_chamber_vacuum", main_chamber_vacuum)


def _warn_on_mismatch(log_apt: Any, label: str, arrays: list[Any]) -> None:
    lengths = [len(values) for values in arrays]
    non_zero_lengths = [length for length in lengths if length > 0]
    if not non_zero_lengths:
        return
    if len(set(non_zero_lengths)) != 1 or len(non_zero_lengths) != len(lengths):
        log_apt.warning("%s data do not have the same length: %s", label, lengths)


def validate_detector_data_lengths(variables: Any, conf: dict[str, Any], log_apt: Any) -> None:
    """Validate synchronized detector list lengths and emit warnings."""
    try:
        tdc_model = normalize_tdc_model(conf.get("tdc_model"))
    except ValueError:
        tdc_model = str(conf.get("tdc_model", "")).strip()

    if variables.counter_source == "TDC" and tdc_model == "Surface_Concept":
        _warn_on_mismatch(
            log_apt,
            "dld",
            [
                variables.x,
                variables.y,
                variables.t,
                variables.dld_start_counter,
                variables.main_v_dc_dld,
                variables.main_v_p_dld,
                variables.main_l_p_dld,
            ],
        )
        _warn_on_mismatch(
            log_apt,
            "tdc",
            [
                variables.channel,
                variables.time_data,
                variables.tdc_start_counter,
                variables.main_v_dc_tdc,
                variables.main_v_p_tdc,
                variables.main_l_p_tdc,
            ],
        )

    elif variables.counter_source == "TDC" and tdc_model == "RoentDek":
        _warn_on_mismatch(
            log_apt,
            "roentdek_dld",
            [
                variables.x,
                variables.y,
                variables.t,
                variables.time_stamp,
                variables.main_v_dc_dld,
                variables.main_v_p_dld,
                variables.main_l_p_dld,
            ],
        )
        _warn_on_mismatch(
            log_apt,
            "roentdek_raw",
            [
                variables.ch0,
                variables.ch1,
                variables.ch2,
                variables.ch3,
                variables.ch4,
                variables.ch5,
                variables.ch6,
                variables.ch7,
                variables.main_v_dc_tdc,
                variables.main_v_p_tdc,
                variables.main_l_p_tdc,
            ],
        )

    elif variables.counter_source == "HSD":
        # The DRS records 1024 waveform samples per acquisition and one
        # DC / pulse voltage reading per acquisition, so the two groups
        # have different lengths by design (channel = 1024 * N, voltage
        # = N). Validate them as separate groups instead of mixing them.
        # main_l_p_drs is not produced by drs.experiment_measure, so it
        # is intentionally excluded.
        _warn_on_mismatch(
            log_apt,
            "hsd-channels",
            [
                variables.ch0_time,
                variables.ch0_wave,
                variables.ch1_time,
                variables.ch1_wave,
                variables.ch2_time,
                variables.ch2_wave,
                variables.ch3_time,
                variables.ch3_wave,
            ],
        )
        _warn_on_mismatch(
            log_apt,
            "hsd-voltages",
            [
                variables.main_v_dc_drs,
                variables.main_v_p_drs,
            ],
        )


def reset_runtime_variables(
    variables: Any,
    x_plot: Any,
    y_plot: Any,
    t_plot: Any,
    main_v_dc_plot: Any,
) -> None:
    """Reset process-shared run variables and clear queue/list buffers."""
    variables.flag_finished_tdc = False
    variables.detection_rate_current = 0.0
    variables.count = 0
    variables.index_plot = 0
    variables.index_save_image = 0
    variables.index_wait_on_plot_start = 0
    variables.index_plot_save = 0
    variables.index_plot = 0
    variables.specimen_voltage = 0
    variables.specimen_voltage_plot = 0
    variables.pulse_voltage = 0

    # Plot pipes are now SharedRingBuffer instances - reset their indices
    # in one O(1) call instead of draining sample-by-sample.
    for buf in (x_plot, y_plot, t_plot, main_v_dc_plot):
        try:
            buf.reset()
        except AttributeError:
            # Backwards-compat: if a queue-style object is still passed,
            # drain it the old way.
            while not buf.empty():
                buf.get()

    for field in _CLEAR_LIST_FIELDS:
        variables.clear_to(field)
