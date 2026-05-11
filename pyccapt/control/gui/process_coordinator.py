"""Process coordination helpers for main control GUI."""

from __future__ import annotations

import multiprocessing
from typing import Any, Callable


class ProcessCoordinator:
    """Create and start worker processes used by the control GUI."""

    def __init__(
        self,
        process_factory: Callable[..., Any] = multiprocessing.Process,
        *,
        experiment_target: Callable[..., Any] | None = None,
        camera_target: Callable[..., Any] | None = None,
        visualization_target: Callable[..., Any] | None = None,
    ) -> None:
        self._process_factory = process_factory
        self._experiment_target = experiment_target
        self._camera_target = camera_target
        self._visualization_target = visualization_target

    def _get_experiment_target(self) -> Callable[..., Any]:
        if self._experiment_target is not None:
            return self._experiment_target
        from pyccapt.control.apt import apt_exp_control

        return apt_exp_control.run_experiment

    def _get_camera_target(self) -> Callable[..., Any]:
        if self._camera_target is not None:
            return self._camera_target
        from pyccapt.control.gui import gui_cameras

        return gui_cameras.run_camera_window

    def _get_visualization_target(self) -> Callable[..., Any]:
        if self._visualization_target is not None:
            return self._visualization_target
        from pyccapt.control.gui import gui_visualization

        return gui_visualization.run_visualization_window

    def start_experiment(
        self,
        variables: Any,
        conf: dict[str, Any],
        experiment_finished_event: Any,
        x_plot: Any,
        y_plot: Any,
        t_plot: Any,
        main_v_dc_plot: Any,
    ) -> Any:
        process = self._process_factory(
            target=self._get_experiment_target(),
            args=(variables, conf, experiment_finished_event, x_plot, y_plot, t_plot, main_v_dc_plot),
        )
        process.start()
        return process

    def start_camera(
        self,
        variables: Any,
        conf: dict[str, Any],
        camera_closed_event: Any,
        camera_command_queue: Any,
    ) -> Any:
        """Spawn the camera subprocess.

        Args:
            camera_command_queue: ``multiprocessing.Queue`` of typed string
                commands ("show", "show_front", "hide") sent from the main
                GUI to the camera window.  Replaces the old (Event +
                flag_camera_win_show) handshake.
        """
        process = self._process_factory(
            target=self._get_camera_target(),
            args=(variables, conf, camera_closed_event, camera_command_queue),
        )
        process.start()
        return process

    def start_visualization(
        self,
        variables: Any,
        conf: dict[str, Any],
        visualization_closed_event: Any,
        visualization_command_queue: Any,
        x_plot: Any,
        y_plot: Any,
        t_plot: Any,
        main_v_dc_plot: Any,
    ) -> Any:
        """Spawn the visualization subprocess.

        Args:
            visualization_command_queue: ``multiprocessing.Queue`` of typed
                string commands ("show", "show_front", "hide") sent from
                the main GUI to the visualization window.  Replaces the
                old (Event + flag_visualization_win_show) handshake.
        """
        process = self._process_factory(
            target=self._get_visualization_target(),
            args=(
                variables,
                conf,
                visualization_closed_event,
                visualization_command_queue,
                x_plot,
                y_plot,
                t_plot,
                main_v_dc_plot,
            ),
        )
        process.start()
        return process
