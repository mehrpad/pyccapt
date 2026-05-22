"""Shared mc+tof calibration panel used by the main calibration workflow."""

from __future__ import annotations

from contextlib import contextmanager

import ipywidgets as widgets

from pyccapt.calibration.core import calibration, mc_plot


def build_combined_mc_tof_calibration_panel(
    variables,
    out,
    out_status,
    calibration_mode,
    label_layout,
    bin_size,
    percent,
    bin_fdm,
    plot_peak,
    index_fig,
    save,
    verbose,
    figure_mc_size_x,
    figure_mc_size_y,
    flight_path_length,
    auto_select_peak_for_mode,
    selected_peak_ready,
    verbosity_context,
    run_initial_current,
    run_auto_current,
    run_hybrid_current,
    save_both_corrections,
    restore_both_corrections,
    reset_both_corrections,
    clear_plots,
    print_gaussian_for_current_mode,
    run_adaptive_for_mode,
    ensure_initial_calibration,
    run_mc_hybrid_auto_residual,
    run_tof_hybrid_auto_residual,
    run_mc_auto_calibration,
    run_tof_auto_calibration,
):
    """Build the simple combined mc+tof calibration tab."""
    combined_bin_size = widgets.FloatText(value=bin_size.value, description='Bin size:', layout=label_layout)
    combined_lim_tof = widgets.IntText(value=variables.max_tof, description='Lim ToF:', layout=label_layout)
    combined_lim_mc = widgets.IntText(value=400, description='Lim m/c:', layout=label_layout)
    combined_percent = widgets.IntText(value=percent.value, description='Percent MRP:', layout=label_layout)
    combined_bin_fdm = widgets.IntText(value=bin_fdm.value, description='Bin FDM:', layout=label_layout)
    combined_plot_peak = widgets.Dropdown(
        options=plot_peak.options, value=plot_peak.value, description='Plot peak', layout=label_layout
    )
    combined_index_fig = widgets.IntText(value=index_fig.value, description='Fig save index:', layout=label_layout)
    combined_save = widgets.Dropdown(options=save.options, value=save.value, description='Save fig:', layout=label_layout)
    combined_verbose = widgets.Dropdown(options=verbose.options, value=False, description='Verbose:', layout=label_layout)
    combined_fig_w = widgets.FloatText(value=figure_mc_size_x.value, description='Fig. size W:', layout=label_layout)
    combined_fig_h = widgets.FloatText(value=figure_mc_size_y.value, description='Fig. size H:', layout=label_layout)

    for src, dst in [
        (combined_bin_size, bin_size),
        (combined_percent, percent),
        (combined_bin_fdm, bin_fdm),
        (combined_plot_peak, plot_peak),
        (combined_index_fig, index_fig),
        (combined_save, save),
        (combined_verbose, verbose),
        (combined_fig_w, figure_mc_size_x),
        (combined_fig_h, figure_mc_size_y),
    ]:
        widgets.link((src, 'value'), (dst, 'value'))

    plot_button = widgets.Button(description='Plot hist', layout=label_layout)
    auto_fast_button = widgets.Button(description='Auto calibration fast', layout=label_layout)
    auto_best_button = widgets.Button(description='Auto calibration best', layout=label_layout)
    save_button = widgets.Button(description='Save correction', layout=label_layout)
    back_button = widgets.Button(description='Back to saved', layout=label_layout)
    reset_button = widgets.Button(description='Reset correction', layout=label_layout)
    clear_button = widgets.Button(description='Clear plots', layout=label_layout)
    gaussian_button = widgets.Button(description='MRP', layout=label_layout)
    stat_button = widgets.Button(description='Plot stat', layout=label_layout)

    def _mode_specs():
        return (('mc_calib', 'm/c', combined_lim_mc.value), ('tof_calib', 'ToF', combined_lim_tof.value))

    all_buttons = [
        plot_button,
        auto_fast_button,
        auto_best_button,
        save_button,
        back_button,
        reset_button,
        clear_button,
        gaussian_button,
        stat_button,
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
                        variables,
                        bin_size.value,
                        log=True,
                        target=mode_value,
                        normalize=False,
                        prominence=100,
                        distance=500,
                        percent=percent.value,
                        selector='rect',
                        figname=index_fig.value,
                        lim=lim_override,
                        save_fig=save.value,
                        peaks_find_plot=plot_peak.value,
                        draw_calib_rect=True,
                        print_info=verbose.value,
                        mrp_all=True,
                        figure_size=(figure_mc_size_x.value, figure_mc_size_y.value),
                        fast_calibration=False,
                        fast_histogram=True,
                    )
            calibration_mode.value = previous_mode

    def _run_auto_fast(_):
        """FAST = mc Auto calibration, then tof Auto calibration.

        Calls the SAME canonical functions the per-tab Auto buttons call
        (``run_mc_auto_calibration`` / ``run_tof_auto_calibration``), so
        the three entry points (mc tab Auto, tof tab Auto, FAST) cannot
        drift apart. Same shape as BEST: plot current-mode histogram
        before each per-mode auto call so ``selected_x1/x2`` are
        recomputed on the correct mode's peak.
        """
        mode_specs = _mode_specs()
        mc_lim = next(spec[2] for spec in mode_specs if spec[0] == 'mc_calib')
        tof_lim = next(spec[2] for spec in mode_specs if spec[0] == 'tof_calib')

        with _lock_buttons():
            previous_mode = calibration_mode.value
            try:
                out_status.append_stdout('\n=== FAST: m/c (Auto calibration) ===\n')
                calibration_mode.value = 'mc_calib'
                try:
                    with out, verbosity_context():
                        auto_select_peak_for_mode('mc_calib', mc_lim)
                except Exception as exc:
                    out_status.append_stdout(f'mc histogram plot failed: {exc} -- continuing.\n')
                try:
                    run_mc_auto_calibration()
                except Exception as exc:
                    out_status.append_stdout(f'm/c auto calibration failed: {exc}\n')

                out_status.append_stdout('\n=== FAST: ToF (Auto calibration) ===\n')
                calibration_mode.value = 'tof_calib'
                try:
                    with out, verbosity_context():
                        auto_select_peak_for_mode('tof_calib', tof_lim)
                except Exception as exc:
                    out_status.append_stdout(f'tof histogram plot failed: {exc} -- continuing.\n')
                try:
                    run_tof_auto_calibration()
                except Exception as exc:
                    out_status.append_stdout(f'ToF auto calibration failed: {exc}\n')
            finally:
                calibration_mode.value = previous_mode
        try:
            _plot_histograms()
        except Exception:
            pass

    def _run_auto_best(_):
        """BEST = mc Hybrid auto + residual, then tof Hybrid auto + residual.

        Calls the SAME canonical functions the per-tab Hybrid buttons call
        (``run_mc_hybrid_auto_residual`` / ``run_tof_hybrid_auto_residual``).

        Between mc and tof we just re-plot the tof histogram. That single
        call runs peak detection on the current ``dld_t_calib`` and
        populates ``variables.selected_x1/x2`` from the dominant tof peak,
        which is what the tof Hybrid handler needs to see at entry. Without
        this, ``selected_x1/x2`` would still hold mc-domain values from the
        just-finished mc Hybrid and tof Hybrid would evaluate MRP on a
        stale, wrong-domain window.
        """
        mode_specs = _mode_specs()
        mc_lim = next(spec[2] for spec in mode_specs if spec[0] == 'mc_calib')
        tof_lim = next(spec[2] for spec in mode_specs if spec[0] == 'tof_calib')

        with _lock_buttons():
            previous_mode = calibration_mode.value
            try:
                out_status.append_stdout('\n=== BEST: m/c (Hybrid auto + residual) ===\n')
                calibration_mode.value = 'mc_calib'
                try:
                    with out, verbosity_context():
                        auto_select_peak_for_mode('mc_calib', mc_lim)
                except Exception as exc:
                    out_status.append_stdout(f'mc histogram plot failed: {exc} -- continuing.\n')
                try:
                    run_mc_hybrid_auto_residual()
                except Exception as exc:
                    out_status.append_stdout(f'm/c best calibration failed: {exc}\n')

                out_status.append_stdout('\n=== BEST: ToF (Hybrid auto + residual) ===\n')
                calibration_mode.value = 'tof_calib'
                try:
                    with out, verbosity_context():
                        auto_select_peak_for_mode('tof_calib', tof_lim)
                except Exception as exc:
                    out_status.append_stdout(f'tof histogram plot failed: {exc} -- continuing.\n')
                try:
                    run_tof_hybrid_auto_residual()
                except Exception as exc:
                    out_status.append_stdout(f'ToF best calibration failed: {exc}\n')
            finally:
                calibration_mode.value = previous_mode
        try:
            _plot_histograms()
        except Exception:
            pass

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
                        variables,
                        bin_fdm.value,
                        index_fig.value,
                        calibration_mode=('tof' if mode_value == 'tof_calib' else 'mc'),
                        save=True,
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

    controls = widgets.VBox(
        [
            combined_bin_size,
            combined_lim_tof,
            combined_lim_mc,
            combined_percent,
            combined_bin_fdm,
            combined_plot_peak,
            combined_index_fig,
            combined_save,
            combined_verbose,
            combined_fig_w,
            combined_fig_h,
        ]
    )
    actions = widgets.VBox(
        [
            plot_button,
            auto_fast_button,
            auto_best_button,
            save_button,
            back_button,
            reset_button,
            clear_button,
            gaussian_button,
            stat_button,
        ]
    )
    return widgets.VBox([widgets.HBox([controls, actions]), widgets.VBox([out, out_status])])
