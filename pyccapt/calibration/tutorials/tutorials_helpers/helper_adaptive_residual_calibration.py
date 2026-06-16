"""Jupyter helper for adaptive residual calibration."""

from __future__ import annotations

import io
import time
import traceback
from contextlib import ExitStack, nullcontext, redirect_stdout

import ipywidgets as widgets
import numpy as np
from ipywidgets import Output

from pyccapt.calibration.core import calibration, mc_plot
from pyccapt.calibration.core.adaptive_residual_calibration import adaptive_residual_calibration
from pyccapt.calibration.core.mc_plot_peak_helpers import gaussian_mrp_report

_LABEL_LAYOUT = widgets.Layout(width="300px")


def build_adaptive_residual_calibration_panel(
    variables,
    det_diam,
    flight_path_length,
    pulse_mode,
    run_mc_auto_calibration=None,
    run_tof_auto_calibration=None,
):
    out = Output()
    out_status = Output()

    target = widgets.Dropdown(
        options=[("mass_to_charge", "mc_calib"), ("time_of_flight", "tof_calib")],
        value="mc_calib",
        description="Target:",
        layout=_LABEL_LAYOUT,
    )
    bin_size = widgets.FloatText(value=0.1, description="Bin size:", layout=_LABEL_LAYOUT)
    lim_value = widgets.IntText(value=400, description="Lim tof/mc:", layout=_LABEL_LAYOUT)
    percent = widgets.IntText(value=50, description="Percent MRP:", layout=_LABEL_LAYOUT)
    bin_fdm = widgets.IntText(value=256, description="Bin FDM:", layout=_LABEL_LAYOUT)
    plot_peak = widgets.Dropdown(
        options=[("True", True), ("False", False)], value=True, description="Plot peak", layout=_LABEL_LAYOUT
    )
    index_fig = widgets.IntText(value=1, description="Fig save index:", layout=_LABEL_LAYOUT)
    save_fig = widgets.Dropdown(
        options=[("False", False), ("True", True)], value=False, description="Save fig:", layout=_LABEL_LAYOUT
    )
    fig_w = widgets.FloatText(value=9.0, description="Fig. size W:", layout=_LABEL_LAYOUT)
    fig_h = widgets.FloatText(value=5.0, description="Fig. size H:", layout=_LABEL_LAYOUT)
    n_peaks = widgets.IntText(value=6, description="Max peaks:", layout=_LABEL_LAYOUT)
    prominence = widgets.IntText(value=100, description="Prominence:", layout=_LABEL_LAYOUT)
    distance = widgets.IntText(value=10, description="Peak distance:", layout=_LABEL_LAYOUT)
    n_windows = widgets.IntText(value=24, description="Index windows:", layout=_LABEL_LAYOUT)
    overlap = widgets.FloatSlider(
        value=0.5, min=0.0, max=0.8, step=0.05, description="Overlap:", layout=_LABEL_LAYOUT, readout_format=".2f"
    )
    template_bin = widgets.FloatText(value=0.01, description="Template bin:", layout=_LABEL_LAYOUT)
    smoothing = widgets.FloatText(value=0.5, description="Smoothness:", layout=_LABEL_LAYOUT)
    apply_spatial = widgets.Dropdown(
        options=[("True", True), ("False", False)], value=True, description="Spatial pass:", layout=_LABEL_LAYOUT
    )
    spatial_grid = widgets.IntText(value=12, description="Spatial grid:", layout=_LABEL_LAYOUT)
    min_window_ions = widgets.IntText(value=40, description="Min win ions:", layout=_LABEL_LAYOUT)
    min_cell_ions = widgets.IntText(value=35, description="Min cell ions:", layout=_LABEL_LAYOUT)
    verbose = widgets.Dropdown(
        options=[("True", True), ("False", False)], value=False, description="Verbose:", layout=_LABEL_LAYOUT
    )

    plot_button = widgets.Button(description="Plot hist", layout=_LABEL_LAYOUT, button_style="primary")
    auto_calib_button = widgets.Button(description="Auto calibration", layout=_LABEL_LAYOUT, button_style="info")
    run_button = widgets.Button(description="Run adaptive residual calibration", layout=_LABEL_LAYOUT, button_style="success")
    save_button = widgets.Button(description="Save correction", layout=_LABEL_LAYOUT)
    back_button = widgets.Button(description="Back to saved", layout=_LABEL_LAYOUT, button_style="warning")
    reset_button = widgets.Button(description="Reset correction", layout=_LABEL_LAYOUT, button_style="danger")
    clear_button = widgets.Button(description="Clear plots", layout=_LABEL_LAYOUT, button_style="warning")
    gaussian_button = widgets.Button(description="MRP", layout=_LABEL_LAYOUT)
    stat_button = widgets.Button(description="Plot stat", layout=_LABEL_LAYOUT, button_style="primary")

    _all_buttons = [
        plot_button, auto_calib_button, run_button, save_button,
        back_button, reset_button, clear_button, gaussian_button, stat_button,
    ]

    def _set_all_disabled(state: bool) -> None:
        for btn in _all_buttons:
            btn.disabled = state

    def _verbosity_context():
        return nullcontext() if verbose.value else redirect_stdout(io.StringIO())

    def _target_key():
        return "tof" if target.value == "tof_calib" else "mc"

    def _sync_lim(*_):
        lim_value.value = variables.max_tof if target.value == "tof_calib" else 400

    def _save_current():
        if target.value == "tof_calib":
            variables.dld_t_calib_backup = np.copy(variables.dld_t_calib)
        else:
            variables.mc_calib_backup = np.copy(variables.mc_calib)

    def _restore_saved():
        if target.value == "tof_calib":
            variables.dld_t_calib = np.copy(variables.dld_t_calib_backup)
        else:
            variables.mc_calib = np.copy(variables.mc_calib_backup)

    def _reset_current():
        if target.value == "tof_calib":
            variables.dld_t_calib = variables.data["t (ns)"].to_numpy()
        else:
            variables.mc_calib = variables.data["mc_uc (Da)"].to_numpy()

    def _current_voltage():
        if pulse_mode == "voltage":
            return variables.dld_high_voltage + (0.7 * variables.dld_pulse_v)
        return variables.dld_high_voltage

    def _plot_hist():
        with out, _verbosity_context():
            out.clear_output()
            mc_plot.hist_plot(
                variables,
                bin_size.value,
                log=True,
                target=target.value,
                normalize=False,
                prominence=prominence.value,
                distance=distance.value,
                percent=percent.value,
                selector="rect",
                figname=index_fig.value,
                lim=lim_value.value,
                save_fig=save_fig.value,
                peaks_find_plot=plot_peak.value,
                draw_calib_rect=True,
                print_info=verbose.value,
                mrp_all=True,
                figure_size=(fig_w.value, fig_h.value),
                fast_calibration=False,
            )

    def _run_initial():
        with out_status, _verbosity_context():
            out_status.clear_output()
            if variables.selected_x2 <= variables.selected_x1:
                print("Please first select a peak for the initial correction.")
                return
            if target.value == "tof_calib":
                variables.dld_t_calib = calibration.initial_calibration(variables.data, flight_path_length)
                print("Initial ToF calibration is done")
            calibration.bowl_correction_main(
                variables.dld_x_det,
                variables.dld_y_det,
                _current_voltage(),
                variables,
                det_diam,
                sample_size=5,
                fit_mode="robust_fit",
                calibration_mode=_target_key(),
                index_fig=index_fig.value,
                plot=False,
                save=False,
                maximum_cal_method="mean",
                maximum_sample_method="mean",
                fig_size=(fig_w.value, fig_h.value),
                fast_calibration=False,
                bin_size=max(0.01, min(float(bin_size.value), 0.05)),
                sampling_mode=getattr(variables, "bowl_sampling_mode", "cartesian"),
            )
            if target.value == "tof_calib":
                print("Initial ToF calibration + bowl correction is done")
            else:
                print("Initial m/c tab action applied bowl correction")

    def _run_gaussian():
        with out_status, _verbosity_context():
            out_status.clear_output()
            if variables.selected_x2 <= variables.selected_x1:
                print("Please first select a peak")
                return
            result = gaussian_mrp_report(
                variables.get_calibration_array(_target_key()),
                variables.selected_x1,
                variables.selected_x2,
                bin_size=0.001,
            )
            if result is None:
                print("Gaussian MRP: insufficient data in selected range")
                return
            print(f'MRP model: {result["recommended_label"]}')
            print(f'MRP bin size used: {result["bin_size"]} ({result["num_bins"]} bins)')
            if result["window_warning"]:
                print(result["window_warning"])
            print(f'Peak position: {result["peak_position"]:.4f}')
            print(f'Recommended FWHM MRP: {result["formatted_recommended_mrp"]}')

    def _run_stat():
        with out:
            out.clear_output()
            calibration.plot_selected_statistic(
                variables,
                bin_fdm.value,
                index_fig.value,
                calibration_mode=_target_key(),
                save=True,
            )

    def _run_adaptive(_, mirror_output=None):
        _set_all_disabled(True)
        try:
            with ExitStack() as stack:
                stack.enter_context(out_status)
                if mirror_output is not None and mirror_output is not out_status:
                    stack.enter_context(mirror_output)
                stack.enter_context(_verbosity_context())
                out_status.clear_output()
                print(f"--- Adaptive residual calibration: target={_target_key()} ---")
                start = time.time()
                try:
                    result = adaptive_residual_calibration(
                        variables,
                        calibration_mode=_target_key(),
                        n_peaks=n_peaks.value,
                        prominence=prominence.value,
                        distance=distance.value,
                        n_windows=n_windows.value,
                        overlap=overlap.value,
                        template_bin_size=template_bin.value,
                        temporal_smoothing=smoothing.value,
                        apply_spatial=apply_spatial.value,
                        spatial_grid=spatial_grid.value,
                        min_window_ions=min_window_ions.value,
                        min_cell_ions=min_cell_ions.value,
                        verbose=verbose.value,
                        above_ceiling_strategy=getattr(
                            variables, "voltage_bowl_above_ceiling_strategy", "nan"
                        ),
                        fast_candidate_score=getattr(
                            variables, "residual_fast_candidate_score", False
                        ),
                        # Opt-in coarse-to-fine candidate selection.
                        # Audited at 2M ions: top_k=1 gives 12.6x mc / 2.8x tof
                        # speedup AND improves quality (+8 mc peaks, +26% mc MRP,
                        # +10 tof peaks). Set via variables.residual_coarse_to_fine_top_k
                        # (the 'best' / 'ref' presets set this automatically).
                        coarse_to_fine_top_k=getattr(
                            variables, "residual_coarse_to_fine_top_k", None
                        ),
                    )
                except Exception as exc:
                    elapsed = time.time() - start
                    print(f"Adaptive residual calibration FAILED after {elapsed:.1f}s: {exc}")
                    traceback.print_exc()
                    return
                elapsed = time.time() - start
                print(f"Adaptive residual calibration finished in {elapsed:.1f}s")
                print(f"Accepted steps: {result['accepted_steps'] or ['none']}")
                print(f"Iterations: {result['n_iterations']} ({result['stop_reason']})")
                print(
                    f"Train score: {result['baseline_quality']['train_score']:.2f} -> {result['final_quality']['train_score']:.2f}"
                )
                print(
                    f"Holdout score: {result['baseline_quality']['holdout_score']:.2f} -> {result['final_quality']['holdout_score']:.2f}"
                )
        finally:
            _set_all_disabled(False)

    def _run_auto_calibration(_):
        _set_all_disabled(True)
        try:
            if target.value == "tof_calib":
                if run_tof_auto_calibration is not None:
                    run_tof_auto_calibration()
            else:
                if run_mc_auto_calibration is not None:
                    run_mc_auto_calibration()
        finally:
            _set_all_disabled(False)

    target.observe(_sync_lim, names="value")
    _sync_lim()

    plot_button.on_click(lambda _: _plot_hist())
    auto_calib_button.on_click(_run_auto_calibration)
    run_button.on_click(_run_adaptive)
    save_button.on_click(lambda _: _save_current())
    back_button.on_click(lambda _: _restore_saved())
    reset_button.on_click(lambda _: _reset_current())
    clear_button.on_click(lambda _: (out.clear_output(), out_status.clear_output()))
    gaussian_button.on_click(lambda _: _run_gaussian())
    stat_button.on_click(lambda _: _run_stat())

    description = widgets.HTML(
        value=(
            "<b>Adaptive residual calibration</b><br>"
            "Learns peak templates from the current spectrum, estimates residual drift across overlapping ion-index windows, "
            "optionally applies a detector-space residual map, and keeps only steps that improve held-out Gaussian MRP."
        ),
        layout=widgets.Layout(width="920px"),
    )
    warning_label = widgets.HTML(
        value=(
            '<div style="color: red; font-weight: bold; padding: 4px 0;">'
            "&#9888; A base calibration (voltage + bowl correction) must be completed in the mc or tof tab "
            "before running adaptive residual calibration. "
            "Use the <em>Auto calibration</em> button below, or run it in the mc / tof tab first."
            "</div>"
        ),
        layout=widgets.Layout(width="920px"),
    )
    left = widgets.VBox([target, bin_size, lim_value, percent, bin_fdm, plot_peak, index_fig, save_fig, fig_w, fig_h])
    center = widgets.VBox([n_peaks, prominence, distance, n_windows, overlap, template_bin, smoothing, verbose])
    right = widgets.VBox(
        [
            apply_spatial,
            spatial_grid,
            min_window_ions,
            min_cell_ions,
            plot_button,
            auto_calib_button,
            run_button,
            save_button,
            back_button,
            reset_button,
            clear_button,
            gaussian_button,
            stat_button,
        ]
    )
    panel = widgets.VBox([description, warning_label, widgets.HBox([left, center, right]), widgets.VBox([out, out_status])])

    def run_for_mode(mode_key, mirror_output=None):
        """Programmatically run the adaptive residual button for one mode.

        Switches the panel's target dropdown to ``mode_key`` ("mc" or "tof")
        and invokes the same click handler the user would trigger manually.
        ``mirror_output``, when given, receives the same prints the adaptive
        tab's status area sees -- so callers on another tab (e.g. the mc+tof
        BEST button) still get live progress and the log file path.
        """
        target_value = "tof_calib" if mode_key == "tof" else "mc_calib"
        previous_target = target.value
        target.value = target_value
        try:
            _run_adaptive(None, mirror_output=mirror_output)
        finally:
            target.value = previous_target

    def apply_profile(profile_name):
        """Apply a named parameter preset to this panel's widgets.

        'old' reverts to the committed defaults. 'new' is identical to
        'old' except n_windows=32 (was 24) -- the only single-knob change
        in the audited Nimonic NiC1 sweep that beat OLD on mc peak count
        AND geomean MRP simultaneously (+3 peaks, +31% MRP) without
        gaming the metric by merging peaks. Earlier 'new' values
        (prominence=50, template_bin=0.005, apply_spatial=False) were
        withdrawn after the peak-quality audit at 2M ions showed they
        either merged peaks or were no-ops.
        """
        # Common defaults (both profiles use these values)
        n_peaks.value = 6
        prominence.value = 100
        distance.value = 10
        overlap.value = 0.5
        template_bin.value = 0.01
        smoothing.value = 0.5
        apply_spatial.value = True
        spatial_grid.value = 12
        min_window_ions.value = 40
        min_cell_ions.value = 35
        # Single difference: n_windows
        n_windows.value = 32 if profile_name == "new" else 24

    return panel, run_for_mode, apply_profile
