"""Integration-style process boundary tests for control module."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from pyccapt.control.apt import detector_runtime
from pyccapt.control.gui.process_coordinator import ProcessCoordinator


def _dummy_variables(counter_source: str) -> SimpleNamespace:
    return SimpleNamespace(counter_source=counter_source)


def test_surface_concept_detector_process_contract():
    process = Mock()
    process_factory = Mock(return_value=process)
    event_obj = object()
    event_factory = Mock(return_value=event_obj)

    conf = {"tdc": "on", "tdc_model": "Surface_Consept"}
    variables = _dummy_variables("TDC")

    runtime_state = detector_runtime.start_detector_processes(
        conf,
        variables,
        x_plot=object(),
        y_plot=object(),
        t_plot=object(),
        main_v_dc_plot=object(),
        process_factory=process_factory,
        event_factory=event_factory,
    )

    assert runtime_state.stop_event is event_obj
    assert runtime_state.tdc_process is process
    process.start.assert_called_once()


def test_roentdek_detector_process_contract():
    process = Mock()
    process_factory = Mock(return_value=process)

    conf = {"tdc": "on", "tdc_model": "RoentDek"}
    variables = _dummy_variables("TDC")

    runtime_state = detector_runtime.start_detector_processes(
        conf,
        variables,
        x_plot=None,
        y_plot=None,
        t_plot=None,
        main_v_dc_plot=None,
        process_factory=process_factory,
    )

    assert runtime_state.tdc_process is process
    assert runtime_state.stop_event is None
    process.start.assert_called_once()


def test_hsd_detector_process_contract():
    process = Mock()
    process_factory = Mock(return_value=process)

    conf = {"tdc": "on", "tdc_model": "HSD"}
    variables = _dummy_variables("HSD")

    runtime_state = detector_runtime.start_detector_processes(
        conf,
        variables,
        x_plot=None,
        y_plot=None,
        t_plot=None,
        main_v_dc_plot=None,
        process_factory=process_factory,
    )

    assert runtime_state.hsd_process is process
    process.start.assert_called_once()


def test_process_coordinator_starts_experiment_process():
    process = Mock()
    process_factory = Mock(return_value=process)

    def _dummy_target(*_args, **_kwargs):
        return None

    coordinator = ProcessCoordinator(
        process_factory=process_factory,
        experiment_target=_dummy_target,
    )

    coordinator.start_experiment(
        variables=object(),
        conf={"x": 1},
        experiment_finished_event=object(),
        x_plot=object(),
        y_plot=object(),
        t_plot=object(),
        main_v_dc_plot=object(),
    )

    process_factory.assert_called_once()
    assert process_factory.call_args.kwargs["target"] is _dummy_target
    process.start.assert_called_once()


def test_join_detector_processes_contract():
    process = Mock()
    process.is_alive.return_value = True

    runtime_state = detector_runtime.DetectorRuntime(tdc_process=process)
    detector_runtime.join_detector_processes(
        conf={"tdc": "on", "tdc_model": "RoentDek"},
        variables=_dummy_variables("TDC"),
        runtime=runtime_state,
    )

    assert process.join.call_count == 2
