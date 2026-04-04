"""Shared mc+tof calibration panel used by the main calibration workflow."""

from __future__ import annotations

from contextlib import contextmanager

import ipywidgets as widgets

from pyccapt.calibration.core import calibration, mc_plot
from pyccapt.calibration.core.adaptive_residual_calibration import adaptive_residual_calibration


def build_combined_mc_tof_calibration_panel(
        variables, out, out_status, calibration_mode, label_layout,
        bin_size, percent, bin_fdm, plot_peak, index_fig, save, verbose, figure_mc_size_x, figure_mc_size_y,
        flight_path_length, auto_select_peak_for_mode, selected_peak_ready, verbosity_context,
        run_initial_current, run_auto_current, run_hybrid_current,
        save_both_corrections, restore_both_corrections, reset_both_corrections,
        clear_plots, print_gaussian_for_current_mode):
    """Build the simple combined mc+tof calibration tab."""
    combined_bin_size = widgets.FloatText(value=bin_size.value, description='bin size:', layout=label_layout)
    combined_lim_tof = widgets.IntText(value=variables.max_tof, description='lim ToF:', layout=label_layout)
    combined_lim_mc = widgets.IntText(value=400, description='lim m/c:', layout=label_layout)
    combined_percent = widgets.IntText(value=percent.value, description='percent MRP:', layout=label_layout)
    combined_bin_fdm = widgets.IntText(value=bin_fdm.value, description='bin FDM:', layout=label_layout)
    combined_plot_peak = widgets.Dropdown(options=plot_peak.options, value=plot_peak.value, description='plot peak', layout=label_layout)
    combined_index_fig = widgets.IntText(value=index_fig.value, description='fig save index:', layout=label_layout)
    combined_save = widgets.Dropdown(options=save.options, value=save.value, description='save fig:', layout=label_layout)
    combined_verbose = widgets.Dropdown(options=verbose.options, value=False, description='verbose:', layout=label_layout)
    combined_fig_w = widgets.FloatText(value=figure_mc_size_x.value, description='Fig. size W:', layout=label_layout)
    combined_fig_h = widgets.FloatText(value=figure_mc_size_y.value, description='Fig. size H:', layout=label_layout)

    for src, dst in [
        (combined_bin_size, bin_size), (combined_percent, percent), (combined_bin_fdm, bin_fdm),
        (combined_plot_peak, plot_peak), (combined_index_fig, index_fig), (combined_save, save),
        (combined_verbose, verbose), (combined_fig_w, figure_mc_size_x), (combined_fig_h, figure_mc_size_y),
    ]:
        widgets.link((src, 'value'), (dst, 'value'))

    plot_button = widgets.Button(description='Plot hist', layout=label_layout)
    auto_fast_button = widgets.Button(description='Auto calibration fast', layout=label_layout)
    auto_best_button = widgets.Button(description='Auto calibration best', layout=label_layout)
    save_button = widgets.Button(description='Save correction', layout=label_layout)
    back_button = widgets.Button(description='Back to saved', layout=label_layout)
    reset_button = widgets.Button(description='Reset correction', layout=label_layout)
    clear_button = widgets.Button(description='Clear plots', layout=label_layout)
    gaussian_button = widgets.Button(description='Gaussian MRP', layout=label_layout)
    stat_button = widgets.Button(description='Plot stat', layout=label_layout)

    def _mode_specs():
        return (('mc_calib', 'm/c', combined_lim_mc.value), ('tof_calib', 'ToF', combined_lim_tof.value))

    all_buttons = [
        plot_button, auto_fast_button, auto_best_button, save_button,
        back_button, reset_button, clear_button, gaussian_button, stat_button,
    ]

    @contextmanager
    def _lock_buttons():
        """Disable every action button while an operation is running."""
        for btn in all_buttons:
            btn.disabled = True
        try:
            yield
        finally:
            for btn in all_buttons:
                btn.disabled = False

    def _plot_histograms(_=None):
        with _lock_buttons():
            previous_mode = calibration_mode.value
            with out, verbosity_context():
                out.clear_output()
                for mode_value, title, lim_override in _mode_specs():
                    calibration_mode.value = mode_value
                    print(f'--- {title} histogram ---')
                    mc_plot.hist_plot(
                        variables, bin_size.value, log=True, target=mode_value, normalize=False,
                        prominence=100, distance=500, percent=percent.value, selector='rect',
                        figname=index_fig.value, lim=lim_override, save_fig=save.value,
                        peaks_find_plot=plot_peak.value, draw_calib_rect=True, print_info=verbose.value,
                        mrp_all=True, figure_size=(figure_mc_size_x.value, figure_mc_size_y.value),
                        fast_calibration=False, fast_histogram=True,
                    )
            calibration_mode.value = previous_mode

    def _run_auto_fast(_):
        """Initial → Voltage → Bowl → lightweight temporal residual for each mode (fast, ~5s)."""
        with _lock_buttons():
            previous_mode = calibration_mode.value
            try:
                with out_status, verbosity_context():
                    out_status.clear_output()
                    for mode_value, title, lim_override in _mode_specs():
                        calibration_mode.value = mode_value
                        mode_key = 'tof' if mode_value == 'tof_calib' else 'mc'
                        print(f'--- {title} auto calibration (fast) ---')
                        auto_select_peak_for_mode(mode_value, lim_override, initial_peak_selection=True)
                        if not selected_peak_ready():
                            print('Unable to auto-select a peak')
                            continue
                        print(
                            'Initial stage window: '
                            f'({float(variables.selected_x1):.4f}, {float(variables.selected_x2):.4f})'
                        )
                        run_initial_current()

                        auto_select_peak_for_mode(mode_value, lim_override)
                        if not selected_peak_ready():
                            print('Unable to auto-select a peak after initial calibration')
                            continue
                        print(
                            'Auto V+Bowl stage window: '
                            f'({float(variables.selected_x1):.4f}, {float(variables.selected_x2):.4f})'
                        )
                        run_auto_current()

                        # Lightweight temporal residual correction to remove
                        # left-shoulder drift that global V+Bowl cannot fix.
                        # Skips spatial correction for speed (~2-3 s).
                        print(f'Running lightweight temporal residual on {mode_key}...')
                        try:
                            result = adaptive_residual_calibration(
                                variables,
                                calibration_mode=mode_key,
                                n_peaks=3,
                                prominence=100,
                                distance=10,
                                n_windows=16,
                                overlap=0.5,
                                template_bin_size=0.01,
                                temporal_smoothing=0.5,
                                apply_spatial=False,
                                max_rounds=2,
                                min_window_ions=40,
                                min_cell_ions=35,
                                verbose=True,
                            )
                            if result['accepted_steps']:
                                print(f"Temporal residual steps: {result['accepted_steps']}")
                            else:
                                print('Temporal residual found no additional drift to correct.')
                        except Exception as exc:
                            print(f'Temporal residual skipped: {exc}')

                        print(f'{title} fast calibration done.')
            finally:
                calibration_mode.value = previous_mode

    def _run_auto_best(_):
        """Initial → Adaptive Residual for each mode (best quality, ~2 min)."""
        with _lock_buttons():
            previous_mode = calibration_mode.value
            try:
                with out_status, verbosity_context():
                    out_status.clear_output()
                    for mode_value, title, lim_override in _mode_specs():
                        calibration_mode.value = mode_value
                        mode_key = 'tof' if mode_value == 'tof_calib' else 'mc'
                        print(f'--- {title} auto calibration (best) ---')

                        # Step 1: Initial calibration
                        auto_select_peak_for_mode(mode_value, lim_override, initial_peak_selection=True)
                        if not selected_peak_ready():
                            print('Unable to auto-select a peak')
                            continue
                        print(
                            'Initial stage window: '
                            f'({float(variables.selected_x1):.4f}, {float(variables.selected_x2):.4f})'
                        )
                        run_initial_current()

                        # Step 2: Adaptive residual calibration
                        print(f'Running adaptive residual calibration on {mode_key}...')
                        try:
                            result = adaptive_residual_calibration(
                                variables,
                                calibration_mode=mode_key,
                                n_peaks=6,
                                prominence=100,
                                distance=10,
                                n_windows=24,
                                overlap=0.5,
                                template_bin_size=0.01,
                                temporal_smoothing=0.5,
                                apply_spatial=True,
                                spatial_grid=12,
                                min_window_ions=40,
                                min_cell_ions=35,
                                max_rounds=8,
                                verbose=True,
                            )
                            if result['accepted_steps']:
                                print(f"Accepted steps: {result['accepted_steps']}")
                            else:
                                print('Adaptive residual accepted no additional steps.')
                        except Exception as exc:
                            print(f'Adaptive residual failed: {exc}')
                            print('Keeping the initial calibration result.')

                        print(f'{title} best calibration done.')
            finally:
                calibration_mode.value = previous_mode

    def _run_gaussian_all():
        with _lock_buttons():
            previous_mode = calibration_mode.value
            with out_status:
                out_status.clear_output()
                for mode_value, title, lim_override in _mode_specs():
                    calibration_mode.value = mode_value
                    auto_select_peak_for_mode(mode_value, lim_override)
                    print_gaussian_for_current_mode(title)
            calibration_mode.value = previous_mode

    def _run_stat_all():
        with _lock_buttons():
            previous_mode = calibration_mode.value
            with out, verbosity_context():
                out.clear_output()
                for mode_value, title, _ in _mode_specs():
                    calibration_mode.value = mode_value
                    print(f'--- {title} statistics ---')
                    calibration.plot_selected_statistic(
                        variables, bin_fdm.value, index_fig.value,
                        calibration_mode=('tof' if mode_value == 'tof_calib' else 'mc'), save=True,
                    )
            calibration_mode.value = previous_mode

    plot_button.on_click(_plot_histograms)
    auto_fast_button.on_click(_run_auto_fast)
    auto_best_button.on_click(_run_auto_best)

    def _on_save(_):
        with _lock_buttons():
            save_both_corrections()

    def _on_back(_):
        with _lock_buttons():
            restore_both_corrections()
            variables.clear_calibration_selection_mask()
            variables.clear_calibration_peak_range()

    def _on_reset(_):
        with _lock_buttons():
            reset_both_corrections()
            variables.clear_calibration_selection_mask()
            variables.clear_calibration_peak_range()

    def _on_clear(_):
        with _lock_buttons():
            clear_plots()

    save_button.on_click(_on_save)
    back_button.on_click(_on_back)
    reset_button.on_click(_on_reset)
    clear_button.on_click(_on_clear)
    gaussian_button.on_click(lambda _: _run_gaussian_all())
    stat_button.on_click(lambda _: _run_stat_all())

    controls = widgets.VBox([
        combined_bin_size, combined_lim_tof, combined_lim_mc, combined_percent, combined_bin_fdm,
        combined_plot_peak, combined_index_fig, combined_save, combined_verbose, combined_fig_w, combined_fig_h,
    ])
    actions = widgets.VBox([
        plot_button, auto_fast_button, auto_best_button, save_button, back_button,
        reset_button, clear_button, gaussian_button, stat_button,
    ])
    return widgets.VBox([widgets.HBox([controls, actions]), widgets.VBox([out, out_status])])
