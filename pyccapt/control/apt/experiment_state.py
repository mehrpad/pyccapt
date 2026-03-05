"""Experiment data/state helpers for APT control."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

from pyccapt.control.core import runtime


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
    variables.exp_name = (
        f"{variables.counter}_{now.strftime('%b-%d-%Y_%H-%M')}_{variables.electrode}_{variables.hdf5_data_name}"
    )

    data_path = runtime.project_path("data", variables.exp_name)
    meta_path = data_path / "meta_data"
    variables.path = str(data_path)
    variables.path_meta = str(meta_path)
    variables.log_path = variables.path_meta
    return data_path, meta_path


def ensure_output_directories(data_path: Path, meta_path: Path) -> None:
    """Ensure data and metadata directories exist."""
    data_path.mkdir(mode=0o777, parents=True, exist_ok=True)
    meta_path.mkdir(mode=0o777, parents=True, exist_ok=True)


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


def validate_detector_data_lengths(variables: Any, log_apt: Any) -> None:
    """Validate synchronized detector list lengths and emit warnings."""
    if variables.counter_source == "TDC":
        if all(
            len(lst) == len(variables.x)
            for lst in [
                variables.x,
                variables.y,
                variables.t,
                variables.dld_start_counter,
                variables.main_v_dc_dld,
                variables.main_v_p_dld,
                variables.main_l_p_dld,
            ]
        ):
            log_apt.warning("dld data have not same length")

        if all(
            len(lst) == len(variables.channel)
            for lst in [
                variables.channel,
                variables.time_data,
                variables.tdc_start_counter,
                variables.main_v_dc_tdc,
                variables.main_v_p_tdc,
                variables.main_l_p_tdc,
            ]
        ):
            log_apt.warning("tdc data have not same length")

    elif variables.counter_source == "DRS":
        if all(
            len(lst) == len(variables.ch0_time)
            for lst in [
                variables.ch0_wave,
                variables.ch1_time,
                variables.ch1_wave,
                variables.ch2_time,
                variables.ch2_wave,
                variables.ch3_time,
                variables.ch3_wave,
                variables.main_v_dc_drs,
                variables.main_v_p_drs,
                variables.main_l_p_drs,
            ]
        ):
            log_apt.warning("tdc data have not same length")


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

    while not x_plot.empty() or not y_plot.empty() or not t_plot.empty() or not main_v_dc_plot.empty():
        if not x_plot.empty():
            x_plot.get()
        if not y_plot.empty():
            y_plot.get()
        if not t_plot.empty():
            t_plot.get()
        if not main_v_dc_plot.empty():
            main_v_dc_plot.get()

    for field in _CLEAR_LIST_FIELDS:
        variables.clear_to(field)

