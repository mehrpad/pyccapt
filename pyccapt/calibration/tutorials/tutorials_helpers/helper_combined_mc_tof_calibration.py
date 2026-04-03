"""Shared mc+tof calibration panel used by the main calibration workflow."""

from __future__ import annotations

import ipywidgets as widgets

from pyccapt.calibration.core import calibration, mc_plot


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
    initial_button = widgets.Button(description='Initial calibration', layout=label_layout)
    auto_button = widgets.Button(description='Auto calibration', layout=label_layout)
    save_button = widgets.Button(description='Save correction', layout=label_layout)
    back_button = widgets.Button(description='Back to saved', layout=label_layout)
    reset_button = widgets.Button(description='Reset correction', layout=label_layout)
    clear_button = widgets.Button(description='Clear plots', layout=label_layout)
    gaussian_button = widgets.Button(description='Gaussian MRP', layout=label_layout)
    stat_button = widgets.Button(description='Plot stat', layout=label_layout)

    def _mode_specs():
        return (('mc_calib', 'm/c', combined_lim_mc.value), ('tof_calib', 'ToF', combined_lim_tof.value))

    def _plot_histograms():
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
                    mrp_all=True, figure_size=(figure_mc_size_x.value, figure_mc_size_y.value), fast_calibration=False,
                )
        calibration_mode.value = previous_mode

    def _run_initial_all():
        previous_mode = calibration_mode.value
        with out_status, verbosity_context():
            out_status.clear_output()
            for mode_value, title, lim_override in _mode_specs():
                calibration_mode.value = mode_value
                print(f'--- {title} initial calibration ---')
                auto_select_peak_for_mode(mode_value, lim_override)
                if not selected_peak_ready():
                    print('Unable to auto-select a peak')
                    continue
                run_initial_current()
        calibration_mode.value = previous_mode

    def _run_gaussian_all():
        previous_mode = calibration_mode.value
        with out_status:
            out_status.clear_output()
            for mode_value, title, lim_override in _mode_specs():
                calibration_mode.value = mode_value
                auto_select_peak_for_mode(mode_value, lim_override)
                print_gaussian_for_current_mode(title)
        calibration_mode.value = previous_mode

    def _run_stat_all():
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

    def _run_auto_all(_):
        auto_button.disabled = True
        previous_mode = calibration_mode.value
        try:
            with out_status, verbosity_context():
                out_status.clear_output()
                for mode_value, title, lim_override in _mode_specs():
                    calibration_mode.value = mode_value
                    print(f'--- {title} combined auto calibration ---')
                    auto_select_peak_for_mode(mode_value, lim_override)
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
                        'Auto stage window: '
                        f'({float(variables.selected_x1):.4f}, {float(variables.selected_x2):.4f})'
                    )
                    run_auto_current()

                    auto_select_peak_for_mode(mode_value, lim_override)
                    if not selected_peak_ready():
                        print('Unable to auto-select a peak after auto calibration')
                        continue
                    print(
                        'Hybrid stage window: '
                        f'({float(variables.selected_x1):.4f}, {float(variables.selected_x2):.4f})'
                    )
                    run_hybrid_current()
        finally:
            calibration_mode.value = previous_mode
            auto_button.disabled = False

    plot_button.on_click(lambda _: _plot_histograms())
    initial_button.on_click(lambda _: _run_initial_all())
    auto_button.on_click(_run_auto_all)
    save_button.on_click(lambda _: save_both_corrections())
    back_button.on_click(
        lambda _: (restore_both_corrections(), variables.clear_calibration_selection_mask(), variables.clear_calibration_peak_range())
    )
    reset_button.on_click(
        lambda _: (reset_both_corrections(), variables.clear_calibration_selection_mask(), variables.clear_calibration_peak_range())
    )
    clear_button.on_click(lambda _: clear_plots())
    gaussian_button.on_click(lambda _: _run_gaussian_all())
    stat_button.on_click(lambda _: _run_stat_all())

    controls = widgets.VBox([
        combined_bin_size, combined_lim_tof, combined_lim_mc, combined_percent, combined_bin_fdm,
        combined_plot_peak, combined_index_fig, combined_save, combined_verbose, combined_fig_w, combined_fig_h,
    ])
    actions = widgets.VBox([
        plot_button, initial_button, auto_button, save_button, back_button,
        reset_button, clear_button, gaussian_button, stat_button,
    ])
    return widgets.VBox([widgets.HBox([controls, actions]), widgets.VBox([out, out_status])])
