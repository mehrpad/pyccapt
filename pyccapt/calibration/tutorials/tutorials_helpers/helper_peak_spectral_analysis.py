"""Visualization helper for peak background, ppm estimation, and deconvolution."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import Output

from pyccapt.calibration.core import peak_spectral_analysis


def build_peak_spectral_analysis_panel(variables, label_layout=None):
    """Build a combined spectral-analysis panel for visualization notebooks."""
    label_layout = label_layout or widgets.Layout(width="200px")
    out = Output()
    out_status = Output()

    target = widgets.Dropdown(
        options=[("mc", "mc"), ("mc_uc", "mc_uc"), ("tof_c", "tof_c"), ("tof", "tof")],
        value="mc",
    )
    bin_size = widgets.FloatText(value=0.02)
    lim_value = widgets.FloatText(value=150.0)
    log_hist = widgets.Dropdown(options=[("True", True), ("False", False)], value=True)
    fig_w = widgets.FloatText(value=9.0)
    fig_h = widgets.FloatText(value=5.0)

    peak_left = widgets.FloatText(value=0.0)
    peak_right = widgets.FloatText(value=0.0)
    bg_gap = widgets.FloatText(value=0.2)
    bg_width = widgets.FloatText(value=0.5)
    bg_left_start = widgets.FloatText(value=0.0)
    bg_left_end = widgets.FloatText(value=0.0)
    bg_right_start = widgets.FloatText(value=0.0)
    bg_right_end = widgets.FloatText(value=0.0)
    ppm_bg_left = widgets.FloatText(value=3.5)
    ppm_bg_right = widgets.FloatText(value=4.5)
    ppm_target_left = widgets.FloatText(value=0.0)
    ppm_target_right = widgets.FloatText(value=0.0)
    deconv_candidates = widgets.Textarea(value="", layout=widgets.Layout(width="320px", height="90px"))
    deconv_shape = widgets.Dropdown(options=[("gaussian", "gaussian"), ("pseudo_voigt", "pseudo_voigt")], value="pseudo_voigt")
    deconv_background = widgets.Dropdown(options=[("linear", "linear"), ("constant", "constant"), ("none", "none")], value="linear")
    deconv_fwhm = widgets.FloatText(value=0.08)

    plot_button = widgets.Button(description="Plot hist")
    load_selection_button = widgets.Button(description="Load selection")
    suggest_bg_button = widgets.Button(description="Suggest sidebands")
    copy_peak_to_ppm_button = widgets.Button(description="Copy peak -> ppm")
    fit_background_button = widgets.Button(description="Fit peak background")
    estimate_ppm_button = widgets.Button(description="Estimate background ppm")
    deconvolve_button = widgets.Button(description="Deconvolve peak")
    clear_button = widgets.Button(description="Clear plots")

    def _resolve_values():
        if variables.data is None or len(variables.data) == 0:
            raise ValueError("No dataset is loaded")
        if target.value == "mc":
            arr = np.asarray(getattr(variables, "mc_calib", np.array([])), dtype=float)
            if arr.size == len(variables.data):
                return arr, "Mass-to-charge [Da]"
            if "mc (Da)" in variables.data:
                return variables.data["mc (Da)"].to_numpy(dtype=float), "Mass-to-charge [Da]"
            return variables.data["mc_uc (Da)"].to_numpy(dtype=float), "Mass-to-charge [Da]"
        if target.value == "mc_uc":
            return variables.data["mc_uc (Da)"].to_numpy(dtype=float), "Mass-to-charge [Da]"
        if target.value == "tof_c":
            arr = np.asarray(getattr(variables, "dld_t_calib", np.array([])), dtype=float)
            if arr.size == len(variables.data):
                return arr, "Time of flight [ns]"
            if "t_c (ns)" in variables.data:
                return variables.data["t_c (ns)"].to_numpy(dtype=float), "Time of flight [ns]"
            return variables.data["t (ns)"].to_numpy(dtype=float), "Time of flight [ns]"
        return variables.data["t (ns)"].to_numpy(dtype=float), "Time of flight [ns]"

    def _selection_available():
        return getattr(variables, "selected_x2", 0.0) > getattr(variables, "selected_x1", 0.0)

    def _peak_window():
        left = float(peak_left.value)
        right = float(peak_right.value)
        if right <= left:
            raise ValueError("Peak window must have positive width")
        return left, right

    def _current_figure_size():
        return (max(3.0, float(fig_w.value)), max(3.0, float(fig_h.value)))

    def _plot_histogram(ax=None, annotate=True):
        values, xlabel = _resolve_values()
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=_current_figure_size())
        x_max = float(lim_value.value) if lim_value.value > 0 else None
        x, y, _ = peak_spectral_analysis.build_histogram(values, bin_size.value, x_min=0.0, x_max=x_max)
        ax.step(x, y, where="mid", color="slategray", linewidth=1.1)
        if log_hist.value:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Counts")
        ax.set_xlim(0, x_max if x_max is not None else float(np.nanmax(x)))
        if annotate:
            if peak_right.value > peak_left.value:
                ax.axvspan(peak_left.value, peak_right.value, color="#FDB863", alpha=0.22, label="peak window")
            if bg_left_end.value > bg_left_start.value:
                ax.axvspan(bg_left_start.value, bg_left_end.value, color="#B2ABD2", alpha=0.18, label="left background")
            if bg_right_end.value > bg_right_start.value:
                ax.axvspan(bg_right_start.value, bg_right_end.value, color="#B2ABD2", alpha=0.18, label="right background")
            if ppm_bg_right.value > ppm_bg_left.value:
                ax.axvspan(ppm_bg_left.value, ppm_bg_right.value, color="#80CDC1", alpha=0.12, label="ppm window")
            if ppm_target_right.value > ppm_target_left.value:
                ax.axvspan(ppm_target_left.value, ppm_target_right.value, color="#018571", alpha=0.08, label="ppm target")
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                dedup = dict(zip(labels, handles))
                ax.legend(dedup.values(), dedup.keys(), frameon=False, loc="upper right")
        ax.set_title("Local spectral analysis")
        return fig, ax

    def _load_selection(_):
        with out_status:
            out_status.clear_output()
            if not _selection_available():
                print("No active selection found. Draw/select a peak in the mc tab first.")
                return
            peak_left.value = float(variables.selected_x1)
            peak_right.value = float(variables.selected_x2)
            ppm_target_left.value = peak_left.value
            ppm_target_right.value = peak_right.value
            print(f"Loaded selected window: {peak_left.value:.4f} to {peak_right.value:.4f}")

    def _suggest_sidebands(_):
        with out_status:
            out_status.clear_output()
            try:
                values, _ = _resolve_values()
                windows = peak_spectral_analysis.suggest_sidebands(
                    peak_left.value,
                    peak_right.value,
                    gap=bg_gap.value,
                    width=bg_width.value,
                    min_x=float(np.nanmin(values)),
                    max_x=float(np.nanmax(values)),
                )
                bg_left_start.value, bg_left_end.value = windows["left"]
                bg_right_start.value, bg_right_end.value = windows["right"]
                print(
                    f"Suggested sidebands: left=({bg_left_start.value:.4f}, {bg_left_end.value:.4f}), "
                    f"right=({bg_right_start.value:.4f}, {bg_right_end.value:.4f})"
                )
            except Exception as exc:
                print(f"Unable to suggest sidebands: {exc}")

    def _copy_peak_to_ppm(_):
        with out_status:
            out_status.clear_output()
            try:
                left, right = _peak_window()
                ppm_target_left.value = left
                ppm_target_right.value = right
                print("Copied peak window into ppm target range.")
            except Exception as exc:
                print(f"Unable to copy peak window: {exc}")

    def _fit_background(_):
        fit_background_button.disabled = True
        try:
            with out:
                out.clear_output()
                values, _ = _resolve_values()
                result = peak_spectral_analysis.fit_local_linear_background(
                    values,
                    bin_size.value,
                    (peak_left.value, peak_right.value),
                    (bg_left_start.value, bg_left_end.value),
                    (bg_right_start.value, bg_right_end.value),
                )
                fig, ax = plt.subplots(figsize=_current_figure_size())
                ax.step(result["x"], result["y"], where="mid", color="black", linewidth=1.1, label="histogram")
                ax.plot(result["x"], result["background_y"], color="#D73027", linewidth=2, label="linear background")
                ax.axvspan(*result["peak_range"], color="#FDB863", alpha=0.22)
                ax.axvspan(*result["bg_left_range"], color="#B2ABD2", alpha=0.18)
                ax.axvspan(*result["bg_right_range"], color="#B2ABD2", alpha=0.18)
                if log_hist.value:
                    ax.set_yscale("log")
                ax.set_xlabel("Mass-to-charge [Da]" if "mc" in target.value else "Time of flight [ns]")
                ax.set_ylabel("Counts")
                ax.legend(frameon=False, loc="upper right")
                ax.set_title("Peak background correction")
                plt.tight_layout()
                plt.show()
                summary = pd.DataFrame(
                    [
                        ("Peak position", result["peak_position"]),
                        ("Observed peak counts", result["observed_peak_counts"]),
                        ("Background in peak", result["background_peak_counts"]),
                        ("Corrected counts", result["corrected_counts"]),
                        ("Corrected ppm", result["corrected_ppm"]),
                        ("Background fraction", result["background_fraction"]),
                        ("Background ppm/Da", result["background_ppm_per_da"]),
                    ],
                    columns=["metric", "value"],
                )
                display(summary)
        except Exception as exc:
            with out_status:
                out_status.clear_output()
                print(f"Background correction failed: {exc}")
        finally:
            fit_background_button.disabled = False

    def _estimate_ppm(_):
        estimate_ppm_button.disabled = True
        try:
            with out:
                out.clear_output()
                values, xlabel = _resolve_values()
                result = peak_spectral_analysis.estimate_background_ppm(
                    values,
                    (ppm_bg_left.value, ppm_bg_right.value),
                    (ppm_target_left.value, ppm_target_right.value),
                )
                fig, ax = plt.subplots(figsize=_current_figure_size())
                _plot_histogram(ax=ax, annotate=False)
                ax.axvspan(*result["background_range"], color="#80CDC1", alpha=0.18, label="background window")
                ax.axvspan(*result["target_range"], color="#018571", alpha=0.10, label="target range")
                ax.set_xlabel(xlabel)
                ax.legend(frameon=False, loc="upper right")
                ax.set_title("Background ppm estimation")
                plt.tight_layout()
                plt.show()
                summary = pd.DataFrame(
                    [
                        ("Background counts", result["background_counts"]),
                        ("Background width", result["background_width"]),
                        ("Background ppm/Da", result["ppm_per_da"]),
                        ("ppm/Da low", result["ppm_per_da_ci"][0]),
                        ("ppm/Da high", result["ppm_per_da_ci"][1]),
                        ("Expected background counts", result["expected_background_counts"]),
                        ("Expected bg low", result["expected_background_counts_ci"][0]),
                        ("Expected bg high", result["expected_background_counts_ci"][1]),
                    ],
                    columns=["metric", "value"],
                )
                display(summary)
        except Exception as exc:
            with out_status:
                out_status.clear_output()
                print(f"Background ppm estimation failed: {exc}")
        finally:
            estimate_ppm_button.disabled = False

    def _deconvolve(_):
        deconvolve_button.disabled = True
        try:
            with out:
                out.clear_output()
                values, xlabel = _resolve_values()
                components = peak_spectral_analysis.resolve_deconvolution_components(
                    getattr(variables, "range_data", None),
                    _peak_window(),
                    deconv_candidates.value,
                )
                result = peak_spectral_analysis.fit_peak_deconvolution(
                    values,
                    bin_size.value,
                    _peak_window(),
                    components,
                    shape=deconv_shape.value,
                    background_mode=deconv_background.value,
                    initial_fwhm=deconv_fwhm.value,
                )
                fig, ax = plt.subplots(figsize=_current_figure_size())
                ax.step(result["x"], result["y"], where="mid", color="black", linewidth=1.1, label="histogram")
                ax.plot(result["x"], result["fit_y"], color="#2166AC", linewidth=2, label="total fit")
                if np.any(result["background_y"] > 0):
                    ax.plot(result["x"], result["background_y"], color="#D6604D", linestyle="--", linewidth=1.6, label="background")
                colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(result["component_curves"]))))
                for idx, curve in enumerate(result["component_curves"]):
                    label = result["components"].iloc[idx]["label"]
                    center = result["components"].iloc[idx]["center"]
                    ax.plot(result["x"], curve + result["background_y"], color=colors[idx % len(colors)], alpha=0.85, linewidth=1.1, label=f"{label}@{center:.3f}")
                if log_hist.value:
                    ax.set_yscale("log")
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Counts")
                ax.legend(frameon=False, loc="upper right", fontsize=8)
                ax.set_title("Peak overlap deconvolution")
                plt.tight_layout()
                plt.show()
                print(f"Fit success: {result['success']} | shape={result['shape']} | background={result['background_mode']}")
                print(f"Shared FWHM: {result['fwhm']:.5f} | RMSE: {result['rmse']:.4f}")
                display(result["components"][["label", "center", "fitted_counts", "fraction"]].sort_values("fitted_counts", ascending=False).reset_index(drop=True))
                display(result["grouped_components"])
        except Exception as exc:
            with out_status:
                out_status.clear_output()
                print(f"Peak deconvolution failed: {exc}")
        finally:
            deconvolve_button.disabled = False

    def _plot(_):
        plot_button.disabled = True
        try:
            with out:
                out.clear_output()
                fig, _ = _plot_histogram()
                plt.tight_layout()
                plt.show()
        except Exception as exc:
            with out_status:
                out_status.clear_output()
                print(f"Histogram plotting failed: {exc}")
        finally:
            plot_button.disabled = False

    def _clear(_):
        with out:
            out.clear_output()
        with out_status:
            out_status.clear_output()

    plot_button.on_click(_plot)
    load_selection_button.on_click(_load_selection)
    suggest_bg_button.on_click(_suggest_sidebands)
    copy_peak_to_ppm_button.on_click(_copy_peak_to_ppm)
    fit_background_button.on_click(_fit_background)
    estimate_ppm_button.on_click(_estimate_ppm)
    deconvolve_button.on_click(_deconvolve)
    clear_button.on_click(_clear)

    description = widgets.HTML(
        value=(
            "<b>Peak analysis</b><br>"
            "Combines local peak background correction, background ppm estimation, and overlap deconvolution. "
            "The background workflow follows the same sideband-style local linear fit used in Peter Felfer's MATLAB toolbox, "
            "while ppm reporting uses a background-density estimate with Poisson confidence limits and deconvolution uses a "
            "shared-width constrained local mixture fit."
        ),
        layout=widgets.Layout(width="960px"),
    )

    left = widgets.VBox([
        widgets.HBox([widgets.Label(value="target:", layout=label_layout), target]),
        widgets.HBox([widgets.Label(value="bin size:", layout=label_layout), bin_size]),
        widgets.HBox([widgets.Label(value="lim:", layout=label_layout), lim_value]),
        widgets.HBox([widgets.Label(value="log hist:", layout=label_layout), log_hist]),
        widgets.HBox([widgets.Label(value="fig size:", layout=label_layout), widgets.HBox([fig_w, fig_h])]),
        widgets.HBox([widgets.Label(value="peak left:", layout=label_layout), peak_left]),
        widgets.HBox([widgets.Label(value="peak right:", layout=label_layout), peak_right]),
        widgets.HBox([load_selection_button, suggest_bg_button]),
        widgets.HBox([widgets.Label(value="bg gap:", layout=label_layout), bg_gap]),
        widgets.HBox([widgets.Label(value="bg width:", layout=label_layout), bg_width]),
        widgets.HBox([widgets.Label(value="bg left start:", layout=label_layout), bg_left_start]),
        widgets.HBox([widgets.Label(value="bg left end:", layout=label_layout), bg_left_end]),
        widgets.HBox([widgets.Label(value="bg right start:", layout=label_layout), bg_right_start]),
        widgets.HBox([widgets.Label(value="bg right end:", layout=label_layout), bg_right_end]),
    ])

    center = widgets.VBox([
        widgets.HTML(value="<b>Background ppm</b>"),
        widgets.HBox([widgets.Label(value="ppm bg left:", layout=label_layout), ppm_bg_left]),
        widgets.HBox([widgets.Label(value="ppm bg right:", layout=label_layout), ppm_bg_right]),
        widgets.HBox([widgets.Label(value="ppm target left:", layout=label_layout), ppm_target_left]),
        widgets.HBox([widgets.Label(value="ppm target right:", layout=label_layout), ppm_target_right]),
        copy_peak_to_ppm_button,
        widgets.HTML(value="<b>Peak deconvolution</b>"),
        widgets.HBox([widgets.Label(value="Candidates:", layout=label_layout), deconv_candidates]),
        widgets.HBox([widgets.Label(value="Shape:", layout=label_layout), deconv_shape]),
        widgets.HBox([widgets.Label(value="Background mode:", layout=label_layout), deconv_background]),
        widgets.HBox([widgets.Label(value="Initial FWHM:", layout=label_layout), deconv_fwhm]),
    ])

    right = widgets.VBox([
        plot_button,
        fit_background_button,
        estimate_ppm_button,
        deconvolve_button,
        clear_button,
    ])

    return widgets.VBox([description, widgets.HBox([left, center, right]), widgets.VBox([out, out_status])])


__all__ = ["build_peak_spectral_analysis_panel"]
