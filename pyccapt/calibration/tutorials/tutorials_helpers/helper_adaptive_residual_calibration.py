"""Jupyter helper for adaptive residual calibration."""

from __future__ import annotations

import io
import os
import time
from contextlib import contextmanager, nullcontext, redirect_stdout

import ipywidgets as widgets
import numpy as np
from ipywidgets import Output

from pyccapt.calibration.core import calibration, mc_plot
from pyccapt.calibration.core.adaptive_residual_calibration import adaptive_residual_calibration
from pyccapt.calibration.core.mc_plot_peak_helpers import gaussian_mrp_report

_LABEL_LAYOUT = widgets.Layout(width="300px")

_LOG_FILENAME = "adaptive_residual_calibration.log"


class _FanoutStream:
    """Write text to a log file and mirror it to one or more Output widgets."""

    def __init__(self, log_file, output_widgets):
        self._log_file = log_file
        self._widgets = [w for w in output_widgets if w is not None]

    def write(self, text):
        if not text:
            return len(text) if text is not None else 0
        if self._log_file is not None:
            try:
                self._log_file.write(text)
                self._log_file.flush()
            except Exception:
                pass
        for widget in self._widgets:
            try:
                widget.append_stdout(text)
            except Exception:
                pass
        return len(text)

    def flush(self):
        if self._log_file is not None:
            try:
                self._log_file.flush()
            except Exception:
                pass


@contextmanager
def _adaptive_logging(output_widgets):
    """Redirect stdout to a log file + the given Output widgets.

    Returns the absolute log path so callers can announce where the run was
    recorded. The file is appended to so successive runs accumulate history.
    """
    log_path = os.path.abspath(_LOG_FILENAME)
    log_file = None
    try:
        log_file = open(log_path, "a", encoding="utf-8", buffering=1)
        header = f"\n===== adaptive residual run @ {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n"
        log_file.write(header)
        log_file.flush()
    except Exception:
        log_file = None
    sink = _FanoutStream(log_file, output_widgets)
    try:
        with redirect_stdout(sink):
            yield log_path
    finally:
        if log_file is not None:
            try:
                log_file.close()
            except Exception:
                pass


def build_adaptive_residual_calibration_panel(variables, det_diam, flight_path_length, pulse_mode):
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
        options=[("True", True), ("False", False)], value=True, description="Verbose:", layout=_LABEL_LAYOUT
    )

    plot_button = widgets.Button(description="Plot hist", layout=_LABEL_LAYOUT)
    run_button = widgets.Button(description="Run adaptive residual calibration", layout=_LABEL_LAYOUT, button_style="success")
    save_button = widgets.Button(description="Save correction", layout=_LABEL_LAYOUT)
    back_button = widgets.Button(description="Back to saved", layout=_LABEL_LAYOUT)
    reset_button = widgets.Button(description="Reset correction", layout=_LABEL_LAYOUT)
    clear_button = widgets.Button(description="Clear plots", layout=_LABEL_LAYOUT)
    gaussian_button = widgets.Button(description="MRP", layout=_LABEL_LAYOUT)
    stat_button = widgets.Button(description="Plot stat", layout=_LABEL_LAYOUT)

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
        run_button.disabled = True
        sinks = [out_status]
        if mirror_output is not None and mirror_output is not out_status:
            sinks.append(mirror_output)
        out_status.clear_output()
        try:
            with _adaptive_logging(sinks) as log_path:
                print(f"--- Adaptive residual calibration: target={_target_key()} ---")
                print(f"Log file: {log_path}")
                print(
                    f"Params: n_peaks={n_peaks.value}, prominence={prominence.value}, "
                    f"distance={distance.value}, n_windows={n_windows.value}, "
                    f"overlap={overlap.value}, template_bin={template_bin.value}, "
                    f"smoothing={smoothing.value}, apply_spatial={apply_spatial.value}, "
                    f"spatial_grid={spatial_grid.value}, min_window_ions={min_window_ions.value}, "
                    f"min_cell_ions={min_cell_ions.value}, verbose={verbose.value}"
                )
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
                    )
                except Exception as exc:
                    elapsed = time.time() - start
                    print(f"Adaptive residual calibration FAILED after {elapsed:.1f}s: {exc}")
                    import traceback
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
            run_button.disabled = False

    target.observe(_sync_lim, names="value")
    _sync_lim()

    plot_button.on_click(lambda _: _plot_hist())
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
    left = widgets.VBox([target, bin_size, lim_value, percent, bin_fdm, plot_peak, index_fig, save_fig, fig_w, fig_h])
    center = widgets.VBox([n_peaks, prominence, distance, n_windows, overlap, template_bin, smoothing, verbose])
    right = widgets.VBox(
        [
            apply_spatial,
            spatial_grid,
            min_window_ions,
            min_cell_ions,
            plot_button,
            run_button,
            save_button,
            back_button,
            reset_button,
            clear_button,
            gaussian_button,
            stat_button,
        ]
    )
    panel = widgets.VBox([description, widgets.HBox([left, center, right]), widgets.VBox([out, out_status])])

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

    return panel, run_for_mode
