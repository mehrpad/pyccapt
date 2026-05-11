"""Widget workflow for reflectron detector-distortion correction."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import ipywidgets as widgets
import numpy as np
from IPython.display import display
from ipywidgets import Output

from pyccapt.calibration.core import share_variables
from pyccapt.calibration.data_tools import file_dialog
from pyccapt.calibration.reflectron_correction import core as reflectron_core


label_layout = widgets.Layout(width="220px")
path_layout = widgets.Layout(width="430px")
field_layout = widgets.Layout(width="240px")


def _display_figure(figure):
    if figure is None:
        return
    display(figure)
    plt.close(figure)


def call_reflectron_correction_workflow(variables=None):
    """Render the reflectron-correction workflow."""
    if variables is None:
        variables = share_variables.Variables()

    preset_options = [(preset.display_name, key) for key, preset in reflectron_core.BUILTIN_REFLECTRON_PRESETS.items()]
    state = {"raw": None, "corrected": None}
    out = Output()

    dataset_path = widgets.Text(value="", description="", layout=path_layout)
    browse_button = widgets.Button(description="Browse")
    load_button = widgets.Button(description="Load EPOS")
    preset_dropdown = widgets.Dropdown(options=preset_options, description="", layout=path_layout)
    preview_button = widgets.Button(description="Preview raw")
    correct_button = widgets.Button(description="Apply correction")
    load_into_workflow_button = widgets.Button(description="Load corrected")
    save_button = widgets.Button(description="Save corrected")
    output_directory = widgets.Text(value="", description="", layout=path_layout)
    output_directory_browse = widgets.Button(description="Browse")
    output_stem = widgets.Text(value="", description="", layout=path_layout)
    output_stem_browse = widgets.Button(description="Browse")
    save_epos = widgets.Dropdown(options=[("True", True), ("False", False)], value=True)
    save_h5 = widgets.Dropdown(options=[("True", True), ("False", False)], value=True)
    bins = widgets.IntText(value=256, description="", layout=field_layout)

    def _autofill_outputs_from_dataset(path_value: str) -> None:
        """Pre-fill output directory and stem from the EPOS path.

        Only fills fields the user has not already edited so we never clobber a
        custom choice.
        """
        if not path_value:
            return
        try:
            path = Path(path_value).expanduser()
        except Exception:
            return
        if not path.name:
            return
        if not output_directory.value:
            output_directory.value = str(path.parent)
        if not output_stem.value:
            output_stem.value = f"{path.stem}_corrected"

    def on_browse(_):
        try:
            selected_path = file_dialog.choose_file_path(
                file_dialog.resolve_initial_directory(
                    dataset_path.value,
                    getattr(variables, "last_directory", None),
                ),
                file_kind="dataset",
            )
            if selected_path:
                dataset_path.value = selected_path
                variables.last_directory = str(Path(selected_path).parent)
        except Exception as exc:
            with out:
                print(f"File chooser failed: {exc}")

    def on_browse_output_directory(_):
        try:
            initial_dir = output_directory.value or (
                str(Path(dataset_path.value).parent) if dataset_path.value else None
            )
            selected_dir = file_dialog.choose_directory_path(
                file_dialog.resolve_initial_directory(
                    initial_dir,
                    getattr(variables, "last_directory", None),
                ),
            )
            if selected_dir:
                output_directory.value = selected_dir
                variables.last_directory = selected_dir
        except Exception as exc:
            with out:
                print(f"Directory chooser failed: {exc}")

    def on_browse_output_stem(_):
        try:
            initial_dir = output_directory.value or (
                str(Path(dataset_path.value).parent) if dataset_path.value else None
            )
            selected_path = file_dialog.choose_file_path(
                file_dialog.resolve_initial_directory(
                    initial_dir,
                    getattr(variables, "last_directory", None),
                ),
                file_kind="dataset",
            )
            if selected_path:
                picked = Path(selected_path)
                output_directory.value = str(picked.parent)
                output_stem.value = picked.stem
                variables.last_directory = str(picked.parent)
        except Exception as exc:
            with out:
                print(f"File chooser failed: {exc}")

    def on_dataset_path_change(change):
        _autofill_outputs_from_dataset(change.get("new", ""))

    def on_load(_):
        load_button.disabled = True
        with out:
            out.clear_output()
            try:
                input_path = Path(dataset_path.value.strip()).expanduser()
                if input_path.suffix.lower() != ".epos":
                    raise ValueError("Reflectron correction currently expects an .epos file, matching the MATLAB workflow.")
                raw_data = reflectron_core.load_epos_for_reflectron_correction(input_path)
                state["raw"] = raw_data
                state["corrected"] = None
                _autofill_outputs_from_dataset(str(input_path))
                variables.last_directory = str(input_path.parent)
                print(f"Loaded EPOS dataset: {input_path}")
                print(f"Ions: {len(raw_data):,}")
                display(raw_data.head(20))
            except Exception as exc:
                print(f"EPOS loading failed: {exc}")
        load_button.disabled = False

    def on_preview(_):
        preview_button.disabled = True
        with out:
            if state["raw"] is None:
                print("Load an EPOS dataset first.")
            else:
                figure = reflectron_core.plot_detector_maps(
                    state["raw"],
                    state["raw"],
                    bins=max(32, int(bins.value)),
                    title_prefix="Reflectron correction preview (raw detector map)",
                )
                _display_figure(figure)
        preview_button.disabled = False

    def on_correct(_):
        correct_button.disabled = True
        with out:
            out.clear_output()
            try:
                if state["raw"] is None:
                    raise ValueError("Load an EPOS dataset first.")
                mesh = reflectron_core.load_builtin_preset(preset_dropdown.value)
                corrected = reflectron_core.apply_reflectron_correction_to_ccapt(state["raw"], mesh)
                state["corrected"] = corrected
                raw_detx = state["raw"]["x_det (cm)"].to_numpy() * 10.0
                raw_dety = state["raw"]["y_det (cm)"].to_numpy() * 10.0
                corrected_detx = corrected["x_det (cm)"].to_numpy() * 10.0
                corrected_dety = corrected["y_det (cm)"].to_numpy() * 10.0
                shifts = np.sqrt((corrected_detx - raw_detx) ** 2 + (corrected_dety - raw_dety) ** 2)
                print(f"Applied reflectron correction preset: {mesh.preset.display_name}")
                print(f"Mean detector shift: {float(np.mean(shifts)):.6f} mm")
                print(f"Max detector shift: {float(np.max(shifts)):.6f} mm")
                figure = reflectron_core.plot_detector_maps(
                    state["raw"],
                    corrected,
                    bins=max(32, int(bins.value)),
                    title_prefix=f"Reflectron correction: {mesh.preset.display_name}",
                )
                _display_figure(figure)
                display(corrected.head(20))
            except Exception as exc:
                print(f"Reflectron correction failed: {exc}")
        correct_button.disabled = False

    def on_save(_):
        save_button.disabled = True
        with out:
            if state["corrected"] is None:
                print("Apply the reflectron correction first.")
            else:
                try:
                    outputs = reflectron_core.save_corrected_outputs(
                        state["corrected"],
                        output_directory=output_directory.value.strip() or dataset_path.value.strip(),
                        output_stem=output_stem.value.strip(),
                        save_epos=bool(save_epos.value),
                        save_h5=bool(save_h5.value),
                    )
                    if not outputs:
                        print("No output format was selected.")
                    else:
                        for name, path in outputs.items():
                            print(f"Saved {name.upper()}: {path}")
                except Exception as exc:
                    print(f"Saving corrected outputs failed: {exc}")
        save_button.disabled = False

    def on_load_into_workflow(_):
        load_into_workflow_button.disabled = True
        with out:
            if state["corrected"] is None:
                print("Apply the reflectron correction first.")
            else:
                corrected = state["corrected"].copy()
                variables.data = corrected
                variables.data_backup = corrected.copy()
                variables.dataset_name = output_stem.value.strip() or Path(dataset_path.value.strip()).stem
                if output_directory.value.strip():
                    variables.set_result_directory(output_directory.value.strip())
                    variables.set_result_data_directory(output_directory.value.strip())
                variables.sync_from_data(corrected, update_backups=True)
                print("Loaded the corrected dataset into the active workflow variables.")
        load_into_workflow_button.disabled = False

    browse_button.on_click(on_browse)
    output_directory_browse.on_click(on_browse_output_directory)
    output_stem_browse.on_click(on_browse_output_stem)
    dataset_path.observe(on_dataset_path_change, names="value")
    load_button.on_click(on_load)
    preview_button.on_click(on_preview)
    correct_button.on_click(on_correct)
    save_button.on_click(on_save)
    load_into_workflow_button.on_click(on_load_into_workflow)

    workflow = widgets.VBox(
        [
            widgets.HTML(
                "<b>Reflectron correction workflow</b><br>"
                "This mirrors the MATLAB dataset workflow: choose a built-in instrument preset, "
                "load an EPOS file, apply the affine detector correction, preview the detector maps, "
                "and save the corrected dataset."
            ),
            widgets.HBox([widgets.Label(value="EPOS dataset:", layout=label_layout), dataset_path, browse_button, load_button]),
            widgets.HBox([widgets.Label(value="Instrument preset:", layout=label_layout), preset_dropdown]),
            widgets.HBox([widgets.Label(value="Detector-map bins:", layout=label_layout), bins]),
            widgets.HBox([preview_button, correct_button]),
            widgets.HBox([widgets.Label(value="Output directory:", layout=label_layout), output_directory, output_directory_browse]),
            widgets.HBox([widgets.Label(value="Output stem:", layout=label_layout), output_stem, output_stem_browse]),
            widgets.HBox([widgets.Label(value="Save EPOS:", layout=label_layout), save_epos]),
            widgets.HBox([widgets.Label(value="Save HDF5:", layout=label_layout), save_h5]),
            widgets.HBox([save_button, load_into_workflow_button]),
            out,
        ]
    )
    display(workflow)


__all__ = ["call_reflectron_correction_workflow"]
