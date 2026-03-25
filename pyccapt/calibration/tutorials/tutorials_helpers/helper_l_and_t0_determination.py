"""Widget workflow for instrument t0 and flight-path determination."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import Output, fixed, interact_manual

from pyccapt.calibration.core import flight_path_t0, mc_plot, share_variables, widgets as core_widgets
from pyccapt.calibration.data_tools import data_loadcrop, file_dialog
from pyccapt.calibration.tutorials.tutorials_helpers import helper_data_loader, helper_t_0_tune

label_layout = widgets.Layout(width="230px")
path_layout = widgets.Layout(width="430px")
medium_layout = widgets.Layout(width="180px")
small_layout = widgets.Layout(width="120px")
wide_layout = widgets.Layout(width="320px")


def _display_figure(figure):
    if figure is None:
        return
    display(figure)
    plt.close(figure)


def _browse_into(text_widget: widgets.Text, status_output: Output, variables) -> None:
    try:
        selected_path = file_dialog.choose_file_path(
            file_dialog.resolve_initial_directory(
                text_widget.value,
                getattr(variables, "last_directory", None),
            )
        )
        if selected_path:
            text_widget.value = selected_path
            if variables is not None:
                variables.last_directory = str(Path(selected_path).parent)
    except Exception as exc:
        with status_output:
            print(f"File chooser failed: {exc}")


def _merge_peak_summaries(selected_summary: pd.DataFrame, used_summary: pd.DataFrame) -> pd.DataFrame:
    selected = selected_summary.rename(columns={"num_ions": "num_ions_selected"})
    used = used_summary.rename(columns={"num_ions": "num_ions_used"})
    merged = selected.merge(
        used,
        on=["peak_label", "measured_mc", "ideal_mc", "window_left", "window_right"],
        how="left",
    )
    merged["num_ions_used"] = merged["num_ions_used"].fillna(0).astype(int)
    merged["usage_fraction"] = merged["num_ions_used"] / merged["num_ions_selected"].clip(lower=1)
    return merged.sort_values("measured_mc").reset_index(drop=True)


def call_l_and_t0_determination_workflow(variables=None):
    """Render the user-friendly t0 and flight-path determination workflow."""
    if variables is None:
        variables = share_variables.Variables()

    isotope_table = flight_path_t0.load_isotope_table()
    isotope_options, isotope_lookup = flight_path_t0.format_isotope_options(isotope_table)
    state = {"peak_rows": [], "results": None}

    status_out = Output()
    interactive_out = Output()
    assignments_out = Output()

    dataset_path = widgets.Text(value="", description="", layout=path_layout)
    browse_button = widgets.Button(description="browse")
    load_button = widgets.Button(description="load dataset")
    crop_button = widgets.Button(description="experiment history")
    tune_button = widgets.Button(description="fine tune t0")
    plot_peaks_button = widgets.Button(description="plot / select peaks")
    sync_peaks_button = widgets.Button(description="load selected peaks")
    estimate_button = widgets.Button(description="estimate t0")
    apply_button = widgets.Button(description="apply fitted values")

    tdc, pulse_mode, flight_path_length, t0_guess, max_mc, det_diam = core_widgets.dataset_instrument_specification_selection()
    detector_center_radius = widgets.FloatText(value=0.5, description="center r (cm):", layout=medium_layout)
    voltage_tolerance = widgets.FloatText(value=0.0, description="dV +/- (V):", layout=medium_layout)
    max_ions_per_peak = widgets.IntText(value=5000, description="max ions:", layout=medium_layout)
    random_seed = widgets.IntText(value=13, description="seed:", layout=medium_layout)

    hist_bin_size = widgets.FloatText(value=0.1, description="bin size:", layout=medium_layout)
    hist_prominence = widgets.IntText(value=100, description="prominence:", layout=medium_layout)
    hist_distance = widgets.IntText(value=100, description="distance:", layout=medium_layout)
    hist_percent = widgets.IntText(value=50, description="percent:", layout=medium_layout)
    hist_limit = widgets.FloatText(value=80.0, description="plot max:", layout=medium_layout)
    preview_mc_limit = widgets.FloatText(value=80.0, description="preview max:", layout=medium_layout)

    peak_rows_box = widgets.VBox([])

    def _clear_peak_rows():
        state["peak_rows"] = []
        peak_rows_box.children = []

    def _build_assignment_rows():
        peak_ranges = flight_path_t0.build_selected_peak_ranges(variables)
        rows = []
        containers = []
        for index, peak in enumerate(peak_ranges, start=1):
            charge_widget = widgets.Dropdown(
                options=[("1+", 1), ("2+", 2), ("3+", 3), ("4+", 4)],
                value=1,
                description="charge:",
                layout=medium_layout,
            )
            isotope_widget = widgets.Combobox(
                options=isotope_options,
                value="",
                description="isotope:",
                ensure_option=False,
                placeholder="optional isotope / manual search",
                layout=wide_layout,
            )
            ideal_widget = widgets.FloatText(
                value=float(peak["measured_mc"]),
                description="ideal mc:",
                layout=medium_layout,
            )
            suggest_button = widgets.Button(description="nearest isotope")

            def _update_ideal(_=None, *, isotope_control=isotope_widget, charge_control=charge_widget,
                              ideal_control=ideal_widget):
                ideal_mass = flight_path_t0.reference_mass_from_selection(
                    isotope_control.value.strip(),
                    charge_control.value,
                    isotope_lookup,
                )
                if ideal_mass is not None:
                    ideal_control.value = round(float(ideal_mass), 6)

            def _suggest(_=None, *, isotope_control=isotope_widget, charge_control=charge_widget, measured=peak["measured_mc"]):
                suggestion = flight_path_t0.suggest_isotope_option(float(measured), charge_control.value, isotope_table)
                isotope_control.value = suggestion
                _update_ideal()

            isotope_widget.observe(lambda change, updater=_update_ideal: updater() if change["name"] == "value" else None, names="value")
            charge_widget.observe(lambda change, updater=_update_ideal: updater() if change["name"] == "value" else None, names="value")
            suggest_button.on_click(_suggest)
            _suggest()

            metadata = dict(peak)
            row = {
                "meta": metadata,
                "charge_widget": charge_widget,
                "isotope_widget": isotope_widget,
                "ideal_widget": ideal_widget,
            }
            rows.append(row)
            containers.append(
                widgets.VBox(
                    [
                        widgets.HTML(
                            value=(
                                f"<b>Peak {index}</b> | measured {float(peak['measured_mc']):.4f} Da | "
                                f"window {float(peak['left']):.4f} to {float(peak['right']):.4f} | "
                                f"ions in window {int(peak['num_ions']):,}"
                            )
                        ),
                        widgets.HBox([charge_widget, isotope_widget, ideal_widget, suggest_button]),
                    ]
                )
            )
        state["peak_rows"] = rows
        peak_rows_box.children = containers

    def on_browse(_):
        _browse_into(dataset_path, status_out, variables)

    def on_load_dataset(_):
        load_button.disabled = True
        with status_out:
            status_out.clear_output()
            try:
                path = dataset_path.value.strip()
                if not path:
                    raise ValueError("Choose a dataset first.")

                variables.last_directory = str(Path(path).expanduser().resolve().parent)
                helper_data_loader.load_data(
                    path,
                    max_mc.value,
                    flight_path_length.value,
                    pulse_mode.value,
                    tdc.value,
                    variables,
                )
                flight_path_t0.configure_result_directories(variables, path)
                helper_data_loader.add_columns(variables, max_mc)
                variables.sync_from_data(update_backups=True)
                variables.peaks_x_selected = []
                variables.peaks_index_list = []
                _clear_peak_rows()

                print(f"Loaded dataset: {path}")
                print(f"Ions available after cleanup: {len(variables.data):,}")
                print(f"Result directory: {variables.result_path}")
                display(variables.data.head(20))
            except Exception as exc:
                print(f"Dataset loading failed: {exc}")
        load_button.disabled = False

    def on_show_crop(_):
        crop_button.disabled = True
        with interactive_out:
            interactive_out.clear_output()
            if variables.data is None:
                print("Load a dataset first.")
            else:
                print("Review the experiment history here before selecting peaks.")
                display(
                    interact_manual(
                        data_loadcrop.plot_crop_experiment_history,
                        data=fixed(variables.data),
                        variables=fixed(variables),
                        max_tof=widgets.FloatText(value=float(variables.max_tof)),
                        frac=widgets.FloatText(value=1.0),
                        bins=fixed((1200, 800)),
                        figure_size=fixed((7, 3)),
                        draw_rect=fixed(False),
                        data_crop=fixed(False),
                        pulse_plot=widgets.Dropdown(options=[("False", False), ("True", True)]),
                        dc_plot=widgets.Dropdown(options=[("True", True), ("False", False)]),
                        pulse_mode=widgets.Dropdown(options=[("voltage", "voltage"), ("laser", "laser")]),
                        save=widgets.Dropdown(options=[("False", False), ("True", True)]),
                        figname=widgets.Text(value="exp_hist"),
                    )
                )
        crop_button.disabled = False

    def on_show_tune(_):
        tune_button.disabled = True
        with interactive_out:
            interactive_out.clear_output()
            if variables.data is None:
                print("Load a dataset first.")
            else:
                helper_t_0_tune.call_fine_tune_t_0(variables, flight_path_length, pulse_mode, t0_guess)
        tune_button.disabled = False

    def on_plot_peaks(_):
        plot_peaks_button.disabled = True
        with interactive_out:
            interactive_out.clear_output()
            if variables.data is None:
                print("Load a dataset first.")
            else:
                variables.peaks_x_selected = []
                variables.peaks_index_list = []
                _clear_peak_rows()
                print("Click peaks with the left mouse button to include them. Right click removes a peak.")
                mc_plot.hist_plot(
                    variables,
                    hist_bin_size.value,
                    log=True,
                    target="mc",
                    normalize=False,
                    prominence=hist_prominence.value,
                    distance=hist_distance.value,
                    percent=hist_percent.value,
                    selector="peak",
                    figname="l_and_t0_peak_selection",
                    lim=hist_limit.value if hist_limit.value > 0 else max_mc.value,
                    peaks_find=True,
                    peaks_find_plot=True,
                    save_fig=False,
                    print_info=False,
                    draw_calib_rect=False,
                    plot_show=True,
                )
        plot_peaks_button.disabled = False

    def on_sync_peaks(_):
        sync_peaks_button.disabled = True
        with assignments_out:
            assignments_out.clear_output()
            try:
                _build_assignment_rows()
                print(f"Loaded {len(state['peak_rows'])} selected peaks. Adjust the ideal masses if needed.")
            except Exception as exc:
                print(f"Peak loading failed: {exc}")
        sync_peaks_button.disabled = False

    def on_estimate(_):
        estimate_button.disabled = True
        with status_out:
            status_out.clear_output()
            try:
                if variables.data is None:
                    raise ValueError("Load a dataset first.")
                if not state["peak_rows"]:
                    raise ValueError("Plot/select peaks and then load the selected peaks first.")

                assignments = []
                for index, row in enumerate(state["peak_rows"], start=1):
                    ideal_mc = float(row["ideal_widget"].value)
                    if ideal_mc <= 0:
                        raise ValueError(f"Peak {index} uses a non-positive ideal mass-to-charge value.")
                    label = row["isotope_widget"].value.strip() or f"Peak {index}"
                    assignments.append(
                        {
                            **row["meta"],
                            "label": label,
                            "ideal_mc": ideal_mc,
                        }
                    )

                selected_table = flight_path_t0.build_peak_regression_table(
                    variables,
                    assignments,
                    max_ions_per_peak=max_ions_per_peak.value,
                    random_seed=random_seed.value,
                )
                used_table = flight_path_t0.filter_peak_regression_table(
                    selected_table,
                    center_radius_cm=detector_center_radius.value,
                    voltage_tolerance_v=voltage_tolerance.value,
                )
                fixed_result = flight_path_t0.estimate_fixed_path_t0(
                    used_table,
                    flight_path_length_mm=flight_path_length.value,
                    pulse_mode=pulse_mode.value,
                )
                regression_result = flight_path_t0.fit_flight_path_and_t0(
                    used_table,
                    pulse_mode=pulse_mode.value,
                )
                selected_summary = flight_path_t0.summarize_peak_table(selected_table)
                used_summary = flight_path_t0.summarize_peak_table(used_table)
                merged_summary = _merge_peak_summaries(selected_summary, used_summary)

                state["results"] = {
                    "selected_table": selected_table,
                    "used_table": used_table,
                    "fixed": fixed_result,
                    "regression": regression_result,
                    "summary": merged_summary,
                }

                print("=" * 60)
                print("INSTRUMENT T0 DETERMINATION")
                print("=" * 60)
                print(f"Selected ions before center filter: {len(selected_table):,}")
                print(f"Used ions after center/voltage filters: {len(used_table):,}")
                print(f"Fixed-path t0: {fixed_result['t0_ns']:.4f} ns  (std {fixed_result['t0_std_ns']:.4f} ns)")
                print(
                    f"Fitted t0: {regression_result['t0_ns']:.4f} ns | "
                    f"Fitted flight path: {regression_result['flight_path_length_mm']:.4f} mm"
                )
                print(
                    f"Regression RMSE: {regression_result['rmse_ns']:.4f} ns | "
                    f"R^2: {regression_result['r_squared']:.6f}"
                )
                print("=" * 60)
                display(merged_summary)

                _display_figure(
                    flight_path_t0.plot_detector_selection(
                        selected_table,
                        used_table,
                        center_radius_cm=detector_center_radius.value,
                    )
                )
                _display_figure(
                    flight_path_t0.plot_regression_results(
                        fixed_result,
                        regression_result,
                    )
                )
                _display_figure(
                    flight_path_t0.preview_mass_spectrum_after_t0(
                        variables,
                        t0_ns=regression_result["t0_ns"],
                        flight_path_length_mm=regression_result["flight_path_length_mm"],
                        pulse_mode=pulse_mode.value,
                        max_mc=preview_mc_limit.value,
                        bin_width=hist_bin_size.value,
                    )
                )
            except Exception as exc:
                print(f"t0 estimation failed: {exc}")
        estimate_button.disabled = False

    def on_apply(_):
        apply_button.disabled = True
        with status_out:
            if state["results"] is None:
                print("Run the t0 estimation first.")
            else:
                regression_result = state["results"]["regression"]
                t0_guess.value = round(float(regression_result["t0_ns"]), 6)
                flight_path_length.value = round(float(regression_result["flight_path_length_mm"]), 6)
                print(
                    f"Applied fitted values to the workflow inputs: "
                    f"t0={t0_guess.value:.6f} ns, flight path={flight_path_length.value:.6f} mm"
                )
        apply_button.disabled = False

    browse_button.on_click(on_browse)
    load_button.on_click(on_load_dataset)
    crop_button.on_click(on_show_crop)
    tune_button.on_click(on_show_tune)
    plot_peaks_button.on_click(on_plot_peaks)
    sync_peaks_button.on_click(on_sync_peaks)
    estimate_button.on_click(on_estimate)
    apply_button.on_click(on_apply)

    workflow = widgets.VBox(
        [
            widgets.HTML(
                "<b>Instrument t0 and flight-path workflow</b><br>"
                "Load a processed dataset, check the experiment history, fine tune a rough t0, "
                "select several known peaks, assign their ideal masses, and fit the instrument t0."
            ),
            widgets.HBox([widgets.Label(value="Dataset file:", layout=label_layout), dataset_path, browse_button, load_button]),
            widgets.HBox([widgets.Label(value="Instrument mode:", layout=label_layout), widgets.HBox([tdc, pulse_mode])]),
            widgets.HBox([widgets.Label(value="Geometry and limits:", layout=label_layout), widgets.HBox([flight_path_length, t0_guess, max_mc, det_diam])]),
            widgets.HBox([widgets.Label(value="Fit filtering:", layout=label_layout), widgets.HBox([detector_center_radius, voltage_tolerance, max_ions_per_peak, random_seed])]),
            widgets.HBox([widgets.Label(value="Peak plot settings:", layout=label_layout), widgets.HBox([hist_bin_size, hist_prominence, hist_distance, hist_percent, hist_limit, preview_mc_limit])]),
            widgets.HBox([crop_button, tune_button, plot_peaks_button, sync_peaks_button, estimate_button, apply_button]),
            widgets.HTML("<b>Selected peak assignments</b>"),
            peak_rows_box,
            assignments_out,
            status_out,
            interactive_out,
        ]
    )
    display(workflow)


__all__ = ["call_l_and_t0_determination_workflow"]
