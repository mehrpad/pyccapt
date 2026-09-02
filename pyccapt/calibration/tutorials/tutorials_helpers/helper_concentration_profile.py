"""Notebook controls for sequential concentration profiles."""

from __future__ import annotations

import matplotlib.pyplot as plt
from IPython.display import clear_output
import ipywidgets as widgets

from pyccapt.calibration.core.concentration_profile import (
    calculate_concentration_profile,
    plot_concentration_profile,
    profile_species_options,
)
from pyccapt.calibration.path_utils import build_output_path, save_figure


def build_concentration_profile_panel(variables, *, label_layout=None):
    """Build the Visualization-tab panel for acquisition-order composition."""
    label_layout = label_layout or widgets.Layout(width="200px")
    options = profile_species_options(variables.range_data)
    selected_default = tuple(value for _, value in options[:1])

    selected_species = widgets.SelectMultiple(
        options=options,
        value=selected_default,
        rows=min(10, max(4, len(options))),
        layout=widgets.Layout(width="360px"),
        description="",
    )
    window_size = widgets.BoundedIntText(value=50_000, min=1, max=100_000_000, step=1_000)
    include_partial = widgets.Checkbox(value=True, description="Include final partial window")
    figure_name = widgets.Text(value="concentration_profile")
    figure_width = widgets.FloatText(value=9.0)
    figure_height = widgets.FloatText(value=5.0)
    save_result = widgets.Dropdown(options=[("True", True), ("False", False)], value=False)
    plot_button = widgets.Button(description="Plot concentration profile", button_style="primary")
    output = widgets.Output()

    def _plot(_button):
        plot_button.disabled = True
        try:
            with output:
                clear_output(wait=True)
                profile = calculate_concentration_profile(
                    variables.mc,
                    variables.range_data,
                    selected_species.value,
                    window_size=window_size.value,
                    include_partial_window=include_partial.value,
                )
                variables.concentration_profile_data = profile
                fig, _axis = plot_concentration_profile(
                    profile,
                    figure_size=(figure_width.value, figure_height.value),
                )
                if save_result.value:
                    if not variables.result_path:
                        raise ValueError("Select a result directory before saving the profile")
                    stem = figure_name.value.strip() or "concentration_profile"
                    paths = save_figure(fig, directory=variables.result_path, stem=stem)
                    csv_path = build_output_path(variables.result_path, f"{stem}.csv")
                    profile.to_csv(csv_path, index=False)
                    print("Saved:", ", ".join(str(path) for path in [*paths, csv_path]))
                plt.show()
        except Exception as exc:
            with output:
                print(f"Concentration profile could not be created: {exc}")
        finally:
            plot_button.disabled = False

    plot_button.on_click(_plot)

    controls = widgets.VBox(
        [
            widgets.HTML(
                "<b>Sequential concentration profile</b><br>"
                "Each point uses a fixed acquisition-order event window. "
                "Percentages include ranged atoms and unranged events in every window; only "
                "the selected elements, ions, or Unranged series are plotted. "
                "Use Ctrl/Cmd-click to select multiple materials."
            ),
            widgets.HBox([widgets.Label("Materials to plot:", layout=label_layout), selected_species]),
            widgets.HBox([widgets.Label("Window length (events):", layout=label_layout), window_size]),
            widgets.HBox([widgets.Label("Final window:", layout=label_layout), include_partial]),
            widgets.HBox([widgets.Label("Figure name:", layout=label_layout), figure_name]),
            widgets.HBox(
                [widgets.Label("Figure size:", layout=label_layout), figure_width, figure_height]
            ),
            widgets.HBox([widgets.Label("Save fig:", layout=label_layout), save_result]),
            plot_button,
            output,
        ]
    )
    return controls


__all__ = ["build_concentration_profile_panel"]
