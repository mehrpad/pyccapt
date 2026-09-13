"""Notebook controls for rectangular-ROI concentration profiles."""

from __future__ import annotations

import matplotlib.pyplot as plt
from IPython.display import clear_output
import ipywidgets as widgets
from matplotlib.patches import Rectangle
import numpy as np

from pyccapt.calibration.core.concentration_profile import (
    calculate_roi_concentration_profile,
    plot_roi_concentration_profile,
    profile_species_options,
)
from pyccapt.calibration.path_utils import build_output_path, save_figure


def _coordinate_arrays(variables) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return validated reconstruction coordinates from shared calibration state."""
    coordinates = tuple(np.asarray(getattr(variables, name, []), dtype=float).reshape(-1) for name in ("x", "y", "z"))
    if not coordinates[0].size or len({array.size for array in coordinates}) != 1:
        raise ValueError("Run reconstruction before creating an ROI concentration profile")
    return coordinates


def build_roi_concentration_profile_panel(variables, *, label_layout=None):
    """Build a numeric rectangular-ROI composition-profile panel.

    The user chooses the profile direction and a rectangular transverse ROI;
    the profile spans the selected axis in fixed-width spatial bins.
    """
    label_layout = label_layout or widgets.Layout(width="200px")
    try:
        x_values, y_values, z_values = _coordinate_arrays(variables)
        coordinates_available = True
    except ValueError:
        x_values = y_values = z_values = np.empty(0, dtype=float)
        coordinates_available = False
    options = profile_species_options(variables.range_data)
    selected_default = tuple(value for _, value in options[:1])

    axis = widgets.Dropdown(options=[("x", "x"), ("y", "y"), ("z", "z")], value="z")
    center_first_label = widgets.Label(layout=label_layout)
    center_second_label = widgets.Label(layout=label_layout)
    width_first_label = widgets.Label(layout=label_layout)
    width_second_label = widgets.Label(layout=label_layout)
    center_first = widgets.FloatText()
    center_second = widgets.FloatText()
    width_first = widgets.BoundedFloatText(value=3.0, min=np.finfo(float).eps, step=0.5)
    width_second = widgets.BoundedFloatText(value=3.0, min=np.finfo(float).eps, step=0.5)
    profile_start = widgets.FloatText()
    profile_end = widgets.FloatText()
    bin_width = widgets.BoundedFloatText(value=1.0, min=np.finfo(float).eps, step=0.25)
    selected_species = widgets.SelectMultiple(
        options=options,
        value=selected_default,
        rows=min(10, max(4, len(options))),
        layout=widgets.Layout(width="360px"),
    )
    figure_name = widgets.Text(value="roi_concentration_profile")
    figure_width = widgets.FloatText(value=9.0)
    figure_height = widgets.FloatText(value=5.0)
    save_result = widgets.Dropdown(options=[("True", True), ("False", False)], value=False)
    preview_button = widgets.Button(description="Preview ROI", button_style="info")
    plot_button = widgets.Button(description="Plot ROI concentration", button_style="primary")
    clear_plot_button = widgets.Button(description="Clear plot", button_style="warning")
    output = widgets.Output()
    current_figure = [None]

    coordinate_names = ("x", "y", "z")
    coordinate_map = {"x": x_values, "y": y_values, "z": z_values}

    def _set_current_figure(figure):
        if current_figure[0] is not None:
            plt.close(current_figure[0])
        current_figure[0] = figure

    def _clear_plot(_button):
        if current_figure[0] is not None:
            plt.close(current_figure[0])
            current_figure[0] = None
        with output:
            clear_output(wait=True)

    def _update_axis_controls(*_args):
        profile_axis = axis.value
        transverse = tuple(name for name in coordinate_names if name != profile_axis)
        center_first_label.value = f"ROI centre {transverse[0]} [nm]:"
        center_second_label.value = f"ROI centre {transverse[1]} [nm]:"
        width_first_label.value = f"ROI width {transverse[0]} [nm]:"
        width_second_label.value = f"ROI width {transverse[1]} [nm]:"
        if not coordinates_available:
            return
        for widget, name in zip((center_first, center_second), transverse):
            values = coordinate_map[name]
            finite = values[np.isfinite(values)]
            widget.value = float(np.median(finite)) if finite.size else 0.0
        values = coordinate_map[profile_axis]
        finite = values[np.isfinite(values)]
        profile_start.value = float(np.min(finite)) if finite.size else 0.0
        profile_end.value = float(np.max(finite)) if finite.size else 1.0

    axis.observe(_update_axis_controls, names="value")
    _update_axis_controls()

    def _roi_parameters():
        profile_axis = axis.value
        transverse = tuple(name for name in coordinate_names if name != profile_axis)
        return profile_axis, transverse, (center_first.value, center_second.value), (width_first.value, width_second.value)

    def _preview(_button):
        try:
            x, y, z = _coordinate_arrays(variables)
            profile_axis, transverse, center, size = _roi_parameters()
            values = {"x": x, "y": y, "z": z}
            first, second = (values[name] for name in transverse)
            finite = np.isfinite(first) & np.isfinite(second)
            if not finite.any():
                raise ValueError("No finite reconstructed coordinates are available")
            indices = np.flatnonzero(finite)
            if indices.size > 100_000:
                indices = indices[np.linspace(0, indices.size - 1, 100_000, dtype=int)]
            with output:
                clear_output(wait=True)
                fig, plot_axis = plt.subplots(figsize=(6.0, 6.0))
                plot_axis.scatter(first[indices], second[indices], s=1, alpha=0.18, rasterized=True)
                plot_axis.add_patch(
                    Rectangle(
                        (center[0] - size[0] / 2.0, center[1] - size[1] / 2.0),
                        size[0], size[1], fill=False, color="crimson", linewidth=2,
                    )
                )
                plot_axis.set_xlabel(f"{transverse[0]} [nm]")
                plot_axis.set_ylabel(f"{transverse[1]} [nm]")
                plot_axis.set_title(f"ROI cross-section for {profile_axis}-axis profile")
                plot_axis.set_aspect("equal", adjustable="box")
                plot_axis.grid(True, alpha=0.25)
                _set_current_figure(fig)
                plt.show()
        except Exception as exc:
            with output:
                clear_output(wait=True)
                print(f"ROI preview could not be created: {exc}")

    def _plot(_button):
        plot_button.disabled = True
        try:
            x, y, z = _coordinate_arrays(variables)
            profile_axis, _transverse, center, size = _roi_parameters()
            with output:
                clear_output(wait=True)
                profile = calculate_roi_concentration_profile(
                    x, y, z, variables.mc, variables.range_data, selected_species.value,
                    axis=profile_axis,
                    transverse_center=center,
                    transverse_size=size,
                    bin_width=bin_width.value,
                    profile_start=profile_start.value,
                    profile_end=profile_end.value,
                )
                variables.roi_concentration_profile_data = profile
                fig, _plot_axis = plot_roi_concentration_profile(
                    profile, figure_size=(figure_width.value, figure_height.value)
                )
                _set_current_figure(fig)
                if save_result.value:
                    if not variables.result_path:
                        raise ValueError("Select a result directory before saving the profile")
                    stem = figure_name.value.strip() or "roi_concentration_profile"
                    paths = save_figure(fig, directory=variables.result_path, stem=stem)
                    csv_path = build_output_path(variables.result_path, f"{stem}.csv")
                    profile.to_csv(csv_path, index=False)
                    print("Saved:", ", ".join(str(path) for path in [*paths, csv_path]))
                print(f"ROI contains {profile.attrs['roi_event_count']:,} detected events.")
                plt.show()
        except Exception as exc:
            with output:
                print(f"ROI concentration profile could not be created: {exc}")
        finally:
            plot_button.disabled = False

    preview_button.on_click(_preview)
    plot_button.on_click(_plot)
    clear_plot_button.on_click(_clear_plot)
    return widgets.VBox(
        [
            widgets.HTML(
                "<b>ROI concentration profile</b><br>Choose an axis and a rectangular cross-section "
                "perpendicular to it. The selected ions are binned along the profile axis. "
                "For an x-axis line profile, set the y/z centre and widths (for example, 1 × 1 nm)."
            ),
            widgets.HBox([widgets.Label("Profile axis:", layout=label_layout), axis]),
            widgets.HBox([center_first_label, center_first]),
            widgets.HBox([center_second_label, center_second]),
            widgets.HBox([width_first_label, width_first]),
            widgets.HBox([width_second_label, width_second]),
            widgets.HBox([widgets.Label("Profile start [nm]:", layout=label_layout), profile_start]),
            widgets.HBox([widgets.Label("Profile end [nm]:", layout=label_layout), profile_end]),
            widgets.HBox([widgets.Label("Bin width [nm]:", layout=label_layout), bin_width]),
            widgets.HBox([widgets.Label("Materials to plot:", layout=label_layout), selected_species]),
            widgets.HBox([widgets.Label("Figure name:", layout=label_layout), figure_name]),
            widgets.HBox([widgets.Label("Figure size:", layout=label_layout), figure_width, figure_height]),
            widgets.HBox([widgets.Label("Save fig:", layout=label_layout), save_result]),
            widgets.HBox([preview_button, plot_button, clear_plot_button]),
            output,
        ]
    )


__all__ = ["build_roi_concentration_profile_panel"]
