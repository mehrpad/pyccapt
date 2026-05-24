import io, time
from contextlib import nullcontext, redirect_stdout

import ipywidgets as widgets
import numpy as np
from IPython.display import clear_output, display
from ipywidgets import Output

from pyccapt.calibration.core.adaptive_residual_calibration import adaptive_residual_calibration
from pyccapt.calibration.core import calibration, mc_plot
from pyccapt.calibration.core.mc_plot_peak_helpers import fast_mrp, gaussian_mrp_report
from pyccapt.calibration.tutorials.tutorials_helpers.helper_adaptive_residual_calibration import (
    build_adaptive_residual_calibration_panel,
)
from pyccapt.calibration.tutorials.tutorials_helpers.helper_combined_mc_tof_calibration import (
    build_combined_mc_tof_calibration_panel,
)

# Public utilities and pure helpers are kept in a sibling module so the host
# file stays under the calibration module-length policy. They are re-exported
# below to preserve every existing import path.
from pyccapt.calibration.tutorials.tutorials_helpers._helper_calibration_pure import (
    clear_plot_on_click,
    label_layout,
    peaks_overlap as _peaks_overlap,
    reset_back_on_click,
    reset_on_click,
    save_on_click,
    score_improved as _score_improved,
    score_not_worse as _score_not_worse,
)


def call_voltage_bowl_calibration(variables, det_diam, flight_path_length, pulse_mode, t0=0.0):
    out = Output()
    out_status = Output()

    plot_button = widgets.Button(description='Plot hist', layout=label_layout)
    plot_stat_button = widgets.Button(description='Plot stat', layout=label_layout)
    reset_back_button = widgets.Button(description='Back to saved', layout=label_layout)
    reset_button = widgets.Button(description='Reset correction', layout=label_layout)
    save_button = widgets.Button(description='Save correction', layout=label_layout)
    bowl_button = widgets.Button(description='Bowl correction', layout=label_layout)
    vol_button = widgets.Button(description='Voltage correction', layout=label_layout)
    auto_button = widgets.Button(description='Auto calibration', layout=label_layout)
    auto_button_bowl = widgets.Button(description='Auto bowl calibration', layout=label_layout)
    gaussian_mrp_button = widgets.Button(description='MRP', layout=label_layout)
    hybrid_button = widgets.Button(description='Hybrid auto + residual', layout=label_layout)
    initial_calib_button = widgets.Button(description='Initial calibration', layout=label_layout)
    clear_plot = widgets.Button(description="Clear plots", layout=label_layout)

    calibration_mode = widgets.Dropdown(
        options=[('mass_to_charge', 'mc_calib'), ('time_of_flight', 'tof_calib')], description='Calibration mode:'
    )

    bin_size = widgets.FloatText(value=0.1, description='Bin size:', layout=label_layout)
    prominence = widgets.IntText(value=100, description='Peak prominance:', layout=label_layout)
    distance = widgets.IntText(value=500, description='Peak distance:', layout=label_layout)
    lim_tof = widgets.IntText(value=variables.max_tof, description='Lim tof/mc:', layout=label_layout)
    percent = widgets.IntText(value=50, description='Percent MRP:', layout=label_layout)
    index_fig = widgets.IntText(value=1, description='Fig save index:', layout=label_layout)
    plot_peak = widgets.Dropdown(
        options=[('True', True), ('False', False)],
        description='Plot peak',
        layout=label_layout,
    )
    save = widgets.Dropdown(
        options=[('False', False), ('True', True)],
        description='Save fig:',
        layout=label_layout,
    )
    verbose = widgets.Dropdown(options=[('True', True), ('False', False)], description='Verbose:', layout=label_layout)
    figure_mc_size_x = widgets.FloatText(value=9.0, description="Fig. size W:", layout=label_layout)
    figure_mc_size_y = widgets.FloatText(value=5.0, description="Fig. size H:", layout=label_layout)

    sample_size_v = widgets.IntText(value=10000, description='Sample size:', layout=label_layout)
    index_fig_v = widgets.IntText(value=1, description='Fig index:', layout=label_layout)
    plot_v = widgets.Dropdown(
        options=[('False', False), ('True', True)],
        description='Plot fig:',
        layout=label_layout,
    )
    save_v = widgets.Dropdown(
        options=[('False', False), ('True', True)],
        description='Save fig:',
        layout=label_layout,
    )
    mode_v = widgets.Dropdown(
        options=[('ion_seq', 'ion_seq'), ('voltage', 'voltage')],
        description='Sample mode:',
        layout=label_layout,
    )
    maximum_cal_method_v = widgets.Dropdown(
        options=[('mean', 'mean'), ('histogram', 'histogram'), ('median', 'median')],
        description='Peak max:',
        layout=label_layout,
    )
    model_v = widgets.Dropdown(
        options=[('robust_fit', 'robust_fit'), ('curve_fit', 'curve_fit')],
        description='Fit mode:',
        layout=label_layout,
    )
    maximum_sample_method_v = widgets.Dropdown(
        options=[('histogram', 'histogram'), ('mean', 'mean'), ('median', 'median')],
        description='Sample max:',
        layout=label_layout,
    )
    bin_size_v = widgets.FloatText(value=0.01, description='Bin size:', layout=label_layout)
    figure_v_size_x = widgets.FloatText(value=5.0, description="Fig. size W:", layout=label_layout)
    figure_v_size_y = widgets.FloatText(value=5.0, description="Fig. size H:", layout=label_layout)

    sample_size_b = widgets.IntText(value=5, description='Sample size:', layout=label_layout)
    sample_size_b_help = widgets.HTML(
        value=(
            '<span style="font-size:11px; color:#555;">'
            'In <b>polar</b> mode, sample size sets ring width and target sector arc length. '
            'In <b>cartesian</b> mode, it is the grid cell width (mm).'
            '</span>'
        ),
        layout=widgets.Layout(width='300px'),
    )
    fit_mode_b = widgets.Dropdown(
        options=[('robust_fit', 'robust_fit'), ('curve_fit', 'curve_fit')],
        description='Fit mode:',
        layout=label_layout,
    )
    # Bowl method: polynomial (legacy default) or RBF spline (experimental).
    # When 'spline' is chosen, _run_bowl_correction calls new_methods.bowl_correction_spline
    # which uses a thin-plate-spline RBF interpolator, clipped + FWHM-gated.
    # Audited on Nimonic NiC1: mc MRP +139% vs polynomial at a cost of 1 mc peak.
    bowl_method_b = widgets.Dropdown(
        options=[
            ('polynomial (default)', 'polynomial'),
            ('spline (RBF, experimental)', 'spline'),
        ],
        value='polynomial',
        description='Bowl method:',
        layout=label_layout,
    )
    sampling_mode_b = widgets.Dropdown(
        options=[('cartesian (default)', 'cartesian'), ('polar', 'polar')],
        value=getattr(variables, 'bowl_sampling_mode', 'cartesian'),
        description='Sampling mode:',
        layout=label_layout,
    )
    index_fig_b = widgets.IntText(value=1, description='Fig index:', layout=label_layout)
    bin_size_b = widgets.FloatText(value=0.01, description='Bin size:', layout=label_layout)
    maximum_cal_method_b = widgets.Dropdown(
        options=[('mean', 'mean'), ('histogram', 'histogram')],
        description='Peak max:',
        layout=label_layout,
    )
    maximum_sample_method_b = widgets.Dropdown(
        options=[('histogram', 'histogram'), ('mean', 'mean')],
        description='Sample max:',
        layout=label_layout,
    )
    plot_b = widgets.Dropdown(
        options=[('False', False), ('True', True)],
        description='Plot fig:',
        layout=label_layout,
    )
    save_b = widgets.Dropdown(
        options=[('False', False), ('True', True)],
        description='Save fig:',
        layout=label_layout,
    )
    # Widgets in the right-most advanced column have longer labels (e.g.
    # "Auto window update", "Refine NM (FWHM)"). The default description_width
    # truncates them, so use a wider layout + 'initial' description width so
    # the full text is visible.
    wide_label_layout = widgets.Layout(width='320px')
    wide_label_style = {'description_width': 'initial'}

    fast_calibration = widgets.Dropdown(
        options=[('False', False), ('True', True)],
        description='Fast calibration:',
        layout=wide_label_layout,
        style=wide_label_style,
    )
    # Per-stage Nelder-Mead FWHM refinement (adapted from APyT's
    # apyt/spectrum/align.py `optimize_correction`). Off by default so the
    # baseline polynomial fit remains the reference behaviour.
    refine_nelder_mead_widget = widgets.Dropdown(
        options=[('False', False), ('True', True)],
        value=False,
        description='Refine NM (FWHM):',
        layout=wide_label_layout,
        style=wide_label_style,
    )
    automatic_window_update = widgets.Dropdown(
        options=[('True', True), ('False', False)],
        value=True,
        description='Auto window update:',
        layout=wide_label_layout,
        style=wide_label_style,
    )
    lock_peak_selection = widgets.Dropdown(
        options=[('False', False), ('True', True)],
        value=False,
        description='Lock peak ions:',
        layout=wide_label_layout,
        style=wide_label_style,
    )
    peak_val = widgets.FloatText(
        value=0,
        description='Peak value:',
        layout=wide_label_layout,
        style=wide_label_style,
    )
    figure_b_size_x = widgets.FloatText(value=5.0, description="Fig. size W:", layout=label_layout)
    figure_b_size_y = widgets.FloatText(value=5.0, description="Fig. size H:", layout=label_layout)

    pb_bowl = widgets.HTML(value=" ", placeholder='Status:', description='Status:', layout=label_layout)
    pb_vol = widgets.HTML(value=" ", placeholder='Status:', description='Status:', layout=label_layout)
    bin_fdm = widgets.IntText(value=256, description='Bin FDM:', layout=label_layout)

    def _calibration_mode_key():
        return 'tof' if calibration_mode.value == 'tof_calib' else 'mc'

    def _current_voltage():
        if pulse_mode == 'voltage':
            return variables.dld_high_voltage + (0.7 * variables.dld_pulse_v)
        return variables.dld_high_voltage

    def _capture_state():
        if calibration_mode.value == 'tof_calib':
            return np.copy(variables.dld_t_calib)
        return np.copy(variables.mc_calib)

    def _restore_state(state):
        if calibration_mode.value == 'tof_calib':
            variables.dld_t_calib = np.copy(state)
        else:
            variables.mc_calib = np.copy(state)

    def _capture_selection():
        mode_key = _calibration_mode_key()
        calibration_range = variables.calibration_peak_ranges.get(mode_key)
        calibration_mask = variables.calibration_selection_masks.get(mode_key)
        return {
            'mode': mode_key,
            'selected_x1': float(variables.selected_x1),
            'selected_x2': float(variables.selected_x2),
            'calibration_range': None if calibration_range is None else tuple(calibration_range),
            'calibration_mask': None if calibration_mask is None else np.copy(calibration_mask),
        }

    def _restore_selection(selection):
        mode_key = selection['mode']
        variables.selected_x1 = selection['selected_x1']
        variables.selected_x2 = selection['selected_x2']
        if selection['calibration_range'] is None:
            variables.clear_calibration_peak_range(mode_key)
        else:
            variables.set_calibration_peak_range(mode_key, *selection['calibration_range'])
        if selection['calibration_mask'] is None:
            variables.clear_calibration_selection_mask(mode_key)
        else:
            variables.set_calibration_selection_mask(mode_key, selection['calibration_mask'])

    def _get_calibration_array():
        if calibration_mode.value == 'tof_calib':
            return variables.dld_t_calib
        return variables.mc_calib

    def _prepare_locked_selection():
        calibration_key = _calibration_mode_key()
        if not _selected_peak_ready():
            variables.clear_calibration_peak_range(calibration_key)
            variables.clear_calibration_selection_mask(calibration_key)
            return

        variables.set_calibration_peak_range(calibration_key, variables.selected_x1, variables.selected_x2)
        if not lock_peak_selection.value:
            variables.clear_calibration_selection_mask(calibration_key)
            return
        data = _get_calibration_array()
        mask = np.logical_and(data > variables.selected_x1, data < variables.selected_x2)
        if np.any(mask):
            variables.set_calibration_selection_mask(calibration_key, mask)
        else:
            variables.clear_calibration_selection_mask(calibration_key)

    def _sampling_mode_value():
        variables.bowl_sampling_mode = sampling_mode_b.value
        return sampling_mode_b.value

    def _verbosity_context():
        return nullcontext() if verbose.value else redirect_stdout(io.StringIO())

    def _state_is_valid(state):
        state = np.asarray(state, dtype=float)
        return state.size > 0 and np.all(np.isfinite(state)) and np.nanstd(state) > 0

    def _selected_peak_ready():
        return (
            not (variables.selected_x1 == 0 and variables.selected_x2 == 0) and variables.selected_x2 > variables.selected_x1
        )

    def _evaluate_mrp_values():
        return (
            [float('nan')] * 3
            if not _selected_peak_ready()
            else fast_mrp(_get_calibration_array(), variables.selected_x1, variables.selected_x2, bin_size=0.001)
        )

    def _print_mrp(prefix):
        mrp = _evaluate_mrp_values()
        print(f'{prefix} MRP(0.5, 0.1, 0.01): {mrp}')
        return mrp

    def _selected_peak_entry():
        if not _selected_peak_ready():
            return None
        data = _get_calibration_array()
        mask = np.logical_and(data > variables.selected_x1, data < variables.selected_x2)
        n_ions = int(np.count_nonzero(mask))
        if n_ions == 0:
            return None
        return {
            'position': float((variables.selected_x1 + variables.selected_x2) * 0.5),
            'x1': float(variables.selected_x1),
            'x2': float(variables.selected_x2),
            'label': 'selected',
            'n_ions': n_ions,
            'weight': max(1.0, float(np.sqrt(n_ions))),
        }

    def _collect_reference_peaks(max_peaks=6, holdout_count=2):
        current_data = _get_calibration_array()
        selected_peak = _selected_peak_entry()
        auto_peaks = []
        try:
            auto_peaks = calibration.auto_detect_reference_peaks(
                current_data,
                n_peaks=max_peaks,
                prominence=prominence.value,
                distance=distance.value,
                hist_bin_size=max(0.01, float(bin_size.value)),
            )
        except Exception:
            auto_peaks = []

        reference_peaks = []
        if selected_peak is not None:
            reference_peaks.append(selected_peak)

        for peak in auto_peaks:
            candidate_peak = dict(peak)
            if selected_peak is not None and _peaks_overlap(selected_peak, candidate_peak):
                continue
            peak_mask = np.logical_and(current_data > candidate_peak['x1'], current_data < candidate_peak['x2'])
            n_ions = int(np.count_nonzero(peak_mask))
            if n_ions < 25:
                continue
            candidate_peak['label'] = f'auto@{candidate_peak["position"]:.2f}'
            candidate_peak['n_ions'] = n_ions
            candidate_peak['weight'] = max(1.0, float(np.sqrt(n_ions)))
            reference_peaks.append(candidate_peak)

        auto_only = [peak for peak in reference_peaks if peak.get('label') != 'selected']
        holdout_n = min(holdout_count, max(0, len(auto_only) // 3))
        if holdout_n > 0:
            train_auto = auto_only[:-holdout_n]
            holdout = auto_only[-holdout_n:]
        else:
            train_auto = auto_only
            holdout = []

        train = []
        if selected_peak is not None:
            train.append(selected_peak)
        train.extend(train_auto)
        if not train and holdout:
            train.append(holdout.pop(0))
        return {'train': train, 'holdout': holdout}

    def _peak_quality_score(peak):
        local_bin_size = max(1e-4, min(0.02, (peak['x2'] - peak['x1']) / 80.0))
        # Opt-in: setting variables.voltage_bowl_above_ceiling_strategy to
        # 'voigt_capped' makes narrow-peak candidates saturate at the
        # physical ceiling instead of returning NaN. Unblocks the V+Bowl
        # optimizer on already-well-calibrated data where every candidate
        # would otherwise be reverted as 'worse than baseline'.
        _strategy = getattr(variables, 'voltage_bowl_above_ceiling_strategy', 'nan')
        report = gaussian_mrp_report(
            _get_calibration_array(),
            peak['x1'],
            peak['x2'],
            bin_size=local_bin_size,
            above_ceiling_strategy=_strategy,
        )
        if report is None:
            return float('nan'), None

        # Use the recommended MRP which already applies the physical ceiling
        # and cross-checks Voigt/Gaussian/histogram values for robustness.
        # Previously this used raw gaussian_mrp/histogram_mrp, letting absurd
        # values (e.g. 938,559) flow into the scoring and destabilise the
        # _optimize_sequence holdout comparisons.
        mrp_values = report['recommended_mrp']
        weights = [0.6, 0.3, 0.1]
        score = 0.0
        weight_sum = 0.0
        for weight, value in zip(weights, mrp_values):
            if np.isfinite(value):
                score += weight * float(value)
                weight_sum += weight
        if weight_sum == 0:
            return float('nan'), report
        return score / weight_sum, report

    def _evaluate_peak_group(peaks):
        weighted_scores = []
        details = []
        for peak in peaks:
            score, report = _peak_quality_score(peak)
            if not np.isfinite(score):
                continue
            weight = float(peak.get('weight', 1.0))
            weighted_scores.append((score, weight))
            details.append(
                {
                    'label': peak.get('label', f'{peak["position"]:.2f}'),
                    'score': float(score),
                    'weight': weight,
                    'position': float(peak['position']),
                    'num_ions': int(peak.get('n_ions', report['num_ions'] if report is not None else 0)),
                }
            )
        if not weighted_scores:
            return float('nan'), details
        total_weight = sum(weight for _, weight in weighted_scores)
        group_score = sum(score * weight for score, weight in weighted_scores) / total_weight
        return float(group_score), details

    def _evaluate_quality(reference_peaks):
        train_score, train_details = _evaluate_peak_group(reference_peaks['train'])
        holdout_score, holdout_details = _evaluate_peak_group(reference_peaks['holdout'])
        selected_score = float('nan')
        selected_peak = _selected_peak_entry()
        if selected_peak is not None:
            selected_score, _ = _peak_quality_score(selected_peak)
        return {
            'train_score': train_score,
            'train_details': train_details,
            'holdout_score': holdout_score,
            'holdout_details': holdout_details,
            'selected_score': selected_score,
        }

    def _print_quality(prefix, reference_peaks):
        quality = _evaluate_quality(reference_peaks)
        print(
            f'{prefix} weighted Gaussian score '
            f'(train={quality["train_score"]:.2f}, holdout={quality["holdout_score"]:.2f}, '
            f'selected={quality["selected_score"]:.2f})'
        )
        return quality

    def _force_reselect_peak_window(initial_peak_selection=True):
        """Re-run peak detection on current data and auto-select a new peak window.

        When ``initial_peak_selection=True`` (default) the histogram draws a
        wider rectangle around the dominant peak.  This matches the manual
        workflow where the user re-plots with a coarse bin size (0.1) between
        calibration steps, producing a wider and more stable peak window.
        """
        mc_plot.hist_plot(
            variables,
            bin_size.value,
            log=True,
            target=calibration_mode.value,
            normalize=False,
            prominence=prominence.value,
            distance=distance.value,
            percent=percent.value,
            selector='rect',
            figname=index_fig.value,
            lim=lim_tof.value,
            save_fig=False,
            peaks_find_plot=False,
            draw_calib_rect=True,
            print_info=False,
            mrp_all=False,
            figure_size=(figure_mc_size_x.value, figure_mc_size_y.value),
            plot_show=False,
            fast_calibration=False,
            fast_histogram=True,
            initial_peak_selection=initial_peak_selection,
        )

    def _update_peak_window(figure_size):
        if not automatic_window_update.value or lock_peak_selection.value:
            return
        mc_plot.hist_plot(
            variables,
            bin_size.value,
            log=True,
            target=calibration_mode.value,
            normalize=False,
            prominence=prominence.value,
            distance=distance.value,
            percent=percent.value,
            selector='rect',
            figname=index_fig.value,
            lim=lim_tof.value,
            save_fig=False,
            peaks_find_plot=False,
            draw_calib_rect=True,
            print_info=False,
            mrp_all=False,
            figure_size=figure_size,
            plot_show=False,
            fast_calibration=fast_calibration.value,
            fast_histogram=True,
        )
        _prepare_locked_selection()

    def _refresh_peak_window_plot(figure_size):
        if not automatic_window_update.value or lock_peak_selection.value:
            return False
        before_selection = _capture_selection()
        with out:
            out.clear_output()
            mc_plot.hist_plot(
                variables,
                bin_size.value,
                log=True,
                target=calibration_mode.value,
                normalize=False,
                prominence=prominence.value,
                distance=distance.value,
                percent=percent.value,
                selector='rect',
                figname=index_fig.value,
                lim=lim_tof.value,
                save_fig=False,
                peaks_find_plot=plot_peak.value,
                draw_calib_rect=True,
                print_info=False,
                mrp_all=False,
                figure_size=figure_size,
                fast_calibration=fast_calibration.value,
                fast_histogram=True,
            )
        _prepare_locked_selection()
        after_selection = _capture_selection()
        return after_selection != before_selection and _selected_peak_ready()

    def _run_with_mode(mode_value, callback):
        previous_mode = calibration_mode.value
        calibration_mode.value = mode_value
        try:
            return callback()
        finally:
            calibration_mode.value = previous_mode

    def _save_both_corrections():
        variables.dld_t_calib_backup = np.copy(variables.dld_t_calib)
        variables.mc_calib_backup = np.copy(variables.mc_calib)

    def _restore_both_corrections():
        variables.dld_t_calib = np.copy(variables.dld_t_calib_backup)
        variables.mc_calib = np.copy(variables.mc_calib_backup)

    def _reset_both_corrections():
        variables.dld_t_calib = variables.data['t (ns)'].to_numpy()
        variables.mc_calib = variables.data['mc_uc (Da)'].to_numpy()
        # Both calibrations are back to raw data; clear the per-mode initial
        # calibration flags so the next auto-* button re-runs initial.
        variables.initial_calibration_done_tof = False
        variables.initial_calibration_done_mc = False

    def _auto_select_peak_for_mode(mode_value, lim_value_override, initial_peak_selection=False):
        _run_with_mode(
            mode_value,
            lambda: mc_plot.hist_plot(
                variables,
                bin_size.value,
                log=True,
                target=mode_value,
                normalize=False,
                prominence=prominence.value,
                distance=distance.value,
                percent=percent.value,
                selector='rect',
                figname=index_fig.value,
                lim=lim_value_override,
                save_fig=False,
                peaks_find_plot=plot_peak.value,
                draw_calib_rect=True,
                print_info=False,
                mrp_all=False,
                figure_size=(figure_mc_size_x.value, figure_mc_size_y.value),
                plot_show=False,
                fast_calibration=False,
                fast_histogram=True,
                initial_peak_selection=initial_peak_selection,
            ),
        )

    def _print_gaussian_for_current_mode(title):
        print(f'--- {title} Gaussian MRP ---')
        if not _selected_peak_ready():
            print('Please first select a peak')
            return
        result = gaussian_mrp_report(_get_calibration_array(), variables.selected_x1, variables.selected_x2, bin_size=0.001)
        if result is None:
            print('Gaussian MRP: insufficient data in selected range')
            return
        print(f'MRP model: {result["recommended_label"]}')
        print(f'MRP bin size used: {result["bin_size"]} ({result["num_bins"]} bins)')
        if result['window_warning']:
            print(result['window_warning'])
        print(f'Peak position: {result["peak_position"]:.4f}')
        print(f'Recommended FWHM MRP: {result["formatted_recommended_mrp"]}')

    def sample_size_v_set(sample_size_widget):
        if calibration_mode.value == 'tof_calib':
            lim_tof.value = variables.max_tof
        else:
            lim_tof.value = 400

        mc_plot.hist_plot(
            variables,
            bin_size.value,
            log=True,
            target=calibration_mode.value,
            normalize=False,
            prominence=prominence.value,
            distance=distance.value,
            percent=percent.value,
            selector='rect',
            figname=index_fig.value,
            lim=lim_tof.value,
            save_fig=save.value,
            peaks_find_plot=plot_peak.value,
            draw_calib_rect=True,
            print_info=False,
            mrp_all=False,
            figure_size=(figure_mc_size_x.value, figure_mc_size_y.value),
            plot_show=False,
            fast_histogram=True,
        )

        if calibration_mode.value == 'tof_calib':
            mask_temporal = np.logical_and(
                (variables.dld_t_calib > variables.selected_x1), (variables.dld_t_calib < variables.selected_x2)
            )
        else:
            mask_temporal = np.logical_and(
                (variables.mc_calib > variables.selected_x1), (variables.mc_calib < variables.selected_x2)
            )

        sample_size = max(1, int(len(variables.dld_high_voltage[mask_temporal]) / 100)) if np.any(mask_temporal) else 1
        sample_size_widget.value = sample_size

    calibration_mode.observe(lambda change: sample_size_v_set(sample_size_v), names='value')

    def hist_plot(_, variables, output, calibration_mode_widget):
        plot_button.disabled = True
        figure_size = (figure_mc_size_x.value, figure_mc_size_y.value)
        clear_output(wait=True)
        with out_status:
            out_status.clear_output()
        with output:
            output.clear_output()
            mc_plot.hist_plot(
                variables,
                bin_size.value,
                log=True,
                target=calibration_mode_widget.value,
                normalize=False,
                prominence=prominence.value,
                distance=distance.value,
                percent=percent.value,
                selector='rect',
                figname=index_fig.value,
                lim=lim_tof.value,
                save_fig=save.value,
                peaks_find_plot=plot_peak.value,
                draw_calib_rect=True,
                print_info=verbose.value,
                mrp_all=True,
                figure_size=figure_size,
                fast_calibration=False,
                fast_histogram=True,
            )
        plot_button.disabled = False

    def _run_voltage_correction(plot_override=None, save_override=None):
        if not _selected_peak_ready():
            raise ValueError('Please first select a peak')

        calibration.voltage_corr_main(
            _current_voltage(),
            variables,
            sample_size=sample_size_v.value,
            calibration_mode=_calibration_mode_key(),
            index_fig=index_fig_v.value,
            plot=plot_v.value if plot_override is None else plot_override,
            save=save_v.value if save_override is None else save_override,
            mode=mode_v.value,
            maximum_cal_method=maximum_cal_method_v.value,
            maximum_sample_method=maximum_sample_method_v.value,
            fig_size=(figure_v_size_x.value, figure_v_size_y.value),
            fast_calibration=fast_calibration.value,
            model=model_v.value,
            bin_size=bin_size_v.value,
            peak_maximum=peak_val.value,
            refine_nelder_mead=refine_nelder_mead_widget.value,
        )

    def _run_bowl_correction(plot_override=None, save_override=None):
        if not _selected_peak_ready():
            raise ValueError('Please first select a peak')

        # Bowl method: 'polynomial' (legacy default) or 'spline' (RBF
        # thin-plate-spline, clipped + FWHM-gated). User-controlled via
        # the 'Bowl method' dropdown on the mc/tof advanced tab.
        if bowl_method_b.value == 'spline':
            from pyccapt.calibration.core import new_methods as _nm
            try:
                _nm.bowl_correction_spline(
                    variables.dld_x_det,
                    variables.dld_y_det,
                    _current_voltage(),
                    variables,
                    det_diam,
                    calibration_mode=_calibration_mode_key(),
                    sample_size=sample_size_b.value,
                    bin_size=bin_size_b.value,
                    use_rbf=True,
                )
                print('Spline bowl correction applied.')
                return
            except Exception as exc:
                print(f'Spline bowl failed ({exc}); falling back to polynomial bowl.')

        calibration.bowl_correction_main(
            variables.dld_x_det,
            variables.dld_y_det,
            _current_voltage(),
            variables,
            det_diam,
            sample_size=sample_size_b.value,
            fit_mode=fit_mode_b.value,
            maximum_cal_method=maximum_cal_method_b.value,
            maximum_sample_method=maximum_sample_method_b.value,
            fig_size=(figure_b_size_x.value, figure_b_size_y.value),
            calibration_mode=_calibration_mode_key(),
            index_fig=index_fig_b.value,
            plot=plot_b.value if plot_override is None else plot_override,
            save=save_b.value if save_override is None else save_override,
            fast_calibration=fast_calibration.value,
            bin_size=bin_size_b.value,
            peak_maximum=peak_val.value,
            sampling_mode=_sampling_mode_value(),
            refine_nelder_mead=refine_nelder_mead_widget.value,
        )

    def vol_correction(_, variables, output, status_output, calibration_mode_widget, pulse_mode_value):
        vol_button.disabled = True
        with status_output, _verbosity_context():
            status_output.clear_output()
            pb_vol.value = "<b>Starting...</b>"
            try:
                _prepare_locked_selection()
                left_edge, right_edge = variables.get_calibration_peak_range(_calibration_mode_key())
                print('Selected mc ranges are: (%s, %s)' % (left_edge, right_edge))
                print('----------------Voltage Calibration-------------------')
                _run_voltage_correction()
                print('Voltage calibration finished successfully.')
            except Exception as exc:
                print(f'Voltage calibration failed: {exc}')
            pb_vol.value = "<b>Finished</b>"
        vol_button.disabled = False

    def bowl_correction(_, variables, output, status_output, calibration_mode_widget, pulse_mode_value):
        bowl_button.disabled = True
        with status_output, _verbosity_context():
            status_output.clear_output()
            pb_bowl.value = "<b>Starting...</b>"
            try:
                _prepare_locked_selection()
                left_edge, right_edge = variables.get_calibration_peak_range(_calibration_mode_key())
                print('Selected mc ranges are: (%s, %s)' % (left_edge, right_edge))
                print('------------------Bowl Calibration---------------------')
                _run_bowl_correction()
                print('Bowl calibration finished successfully.')
            except Exception as exc:
                print(f'Bowl calibration failed: {exc}')
            pb_bowl.value = "<b>Finished</b>"
        bowl_button.disabled = False

    def _mark_initial_done(mode_value):
        if mode_value == 'tof_calib':
            variables.initial_calibration_done_tof = True
        else:
            variables.initial_calibration_done_mc = True

    def _initial_done(mode_value):
        if mode_value == 'tof_calib':
            return bool(getattr(variables, 'initial_calibration_done_tof', False))
        return bool(getattr(variables, 'initial_calibration_done_mc', False))

    def initial_calibration(_, variables, calibration_mode_widget, flight_path_length_value):
        initial_calib_button.disabled = True
        with out, _verbosity_context():
            out.clear_output()
            if calibration_mode_widget.value == 'tof_calib':
                # Step 1: Naive flight-path + voltage factor correction
                variables.dld_t_calib = calibration.initial_calibration(variables.data, flight_path_length_value)
                print('Initial ToF calibration is done')
                # Step 2: Re-select peak window on the corrected data
                _force_reselect_peak_window()
                _prepare_locked_selection()
                # Step 3: Voltage correction
                try:
                    _run_voltage_correction(plot_override=False, save_override=False)
                except Exception as exc:
                    print(f'Voltage correction during initial calibration skipped: {exc}')
                # Step 4: Re-select peak window after voltage correction
                _force_reselect_peak_window()
                _prepare_locked_selection()
                # Step 5: Bowl correction
                _run_bowl_correction(plot_override=False, save_override=False)
                print('Initial ToF calibration + bowl correction is done')
            else:
                # Always re-pick the peak window for *this* mode before
                # running bowl correction. Otherwise, when called from the
                # combined mc+tof FAST/BEST flow, variables.selected_x1/x2
                # may still hold the previous mode's window (e.g. tof units
                # while we're now in mc), and the bowl fit would run on the
                # wrong slice of the data.
                _force_reselect_peak_window()
                _prepare_locked_selection()
                _run_bowl_correction(plot_override=False, save_override=False)
                print('Initial m/c tab action applied bowl correction')
        _mark_initial_done(calibration_mode_widget.value)
        initial_calib_button.disabled = False

    def _ensure_initial_calibration():
        """Run the per-mode initial calibration if it hasn't been done yet,
        then refresh the auto-picked peak window from the current histogram.

        Used by every auto-* button so users don't need to remember to click
        "Initial calibration" first. Tracking is per-mode (mc vs tof) because
        the two have different initial-calibration steps.

        The trailing ``_force_reselect_peak_window()`` always re-runs peak
        detection on the post-calibration data, so the auto routine starts
        from a fresh peak window regardless of whether we just ran the
        initial calibration or skipped it.
        """
        if not _initial_done(calibration_mode.value):
            mode_label = 'ToF' if calibration_mode.value == 'tof_calib' else 'm/c'
            print(f'Auto-running initial calibration for {mode_label} (had not been done yet).')
            initial_calibration(None, variables, calibration_mode, flight_path_length)
        # Always recompute the peak window after initial calibration so the
        # auto routine starts from a freshly-detected dominant peak. Use
        # initial_peak_selection=False so the resulting window matches what
        # the manual workflow produces when the user presses the "Plot"
        # button between Initial calibration and Auto calibration. With
        # initial_peak_selection=True the rectangle is intentionally widened,
        # which changes the optimizer trajectory and yields a different MRP.
        try:
            _force_reselect_peak_window(initial_peak_selection=False)
        except Exception as exc:
            print(f'Peak window refresh after initial calibration skipped: {exc}')

    def stat_plot(_, variables, calibration_mode_widget, output):
        calibration_mode_t = 'tof' if calibration_mode_widget.value == 'tof_calib' else 'mc'
        with output:
            output.clear_output()
            calibration.plot_selected_statistic(
                variables, bin_fdm.value, index_fig.value, calibration_mode=calibration_mode_t, save=True
            )

    def _optimize_sequence(
        action_specs, title, figure_size, max_iterations=10, max_no_improve=3, retry_peak_window_on_stall=False
    ):
        if not _selected_peak_ready():
            print('Please first select a peak')
            return

        reference_peaks = _collect_reference_peaks()
        print(
            f'Using {len(reference_peaks["train"])} training peaks and '
            f'{len(reference_peaks["holdout"])} held-out peaks for optimization scoring.'
        )
        best_state = _capture_state()
        best_selection = _capture_selection()
        initial_mrp = _print_mrp('Initial')
        initial_quality = _print_quality('Initial', reference_peaks)
        best_train_score = initial_quality['train_score']
        best_holdout_score = initial_quality['holdout_score']
        best_selected_score = initial_quality['selected_score']
        no_improve_count = 0
        start_time = time.perf_counter()

        for iteration in range(1, max_iterations + 1):
            print('=======================================================')
            print(
                f'{title}: iteration {iteration} '
                f'(best train={best_train_score:.2f}, holdout={best_holdout_score:.2f}, '
                f'selected={best_selected_score:.2f})'
            )
            improved_this_round = False

            for action_name, action in action_specs:
                before_state = _capture_state()
                before_selection = _capture_selection()
                try:
                    action()
                    candidate_state = _capture_state()
                    if not _state_is_valid(candidate_state):
                        _restore_state(before_state)
                        _restore_selection(before_selection)
                        print(f'{action_name} produced an invalid calibration state; reverted this step.')
                        continue
                    _update_peak_window(figure_size)
                    candidate_selection = _capture_selection()
                    candidate_quality = _evaluate_quality(reference_peaks)

                    # Re-evaluate the pre-action state using the SAME post-action
                    # peak window. Critical in ToF: after Vol+Bowl the dominant
                    # peak can shift by several ns; ``_update_peak_window``
                    # refreshes ``variables.selected_x1/x2`` to the new peak
                    # position, but ``best_selected_score`` was scored on the
                    # OLD window. Comparing those two scores compared peaks at
                    # different positions and routinely rejected real
                    # improvements in ToF (where the peak moves more than in
                    # m/c). Now both sides of the comparison are scored on the
                    # same window for an apples-to-apples decision.
                    _restore_state(before_state)
                    before_quality_on_new_window = _evaluate_quality(reference_peaks)
                    _restore_state(candidate_state)
                    _restore_selection(candidate_selection)
                    quality = candidate_quality

                    print(
                        f'After {action_name}: '
                        f'train={quality["train_score"]:.2f}, '
                        f'holdout={quality["holdout_score"]:.2f}, '
                        f'selected={quality["selected_score"]:.2f} '
                        f'(baseline on same window: '
                        f'selected={before_quality_on_new_window["selected_score"]:.2f})'
                    )

                    has_valid_signal = (np.isfinite(quality['train_score']) and quality['train_score'] > 0) or (
                        np.isfinite(quality['selected_score']) and quality['selected_score'] > 0
                    )
                    if not has_valid_signal:
                        _restore_state(before_state)
                        _restore_selection(before_selection)
                        print(f'{action_name} produced an invalid quality score; reverted this step.')
                        continue

                    # Acceptance is gated by the SELECTED (dominant) peak only.
                    # Train / holdout scores are still computed and printed for
                    # diagnostics, but they no longer veto a step -- a low-
                    # intensity held-out peak that wiggles by a few percent
                    # was previously discarding real improvements on the main
                    # peak. The held-out tolerance is also loosened from
                    # 0.01 -> 0.05 so the diagnostic "unstable on held-out
                    # peaks" message only fires for genuinely large regressions.
                    baseline_selected = before_quality_on_new_window['selected_score']
                    baseline_train = before_quality_on_new_window['train_score']
                    baseline_holdout = before_quality_on_new_window['holdout_score']
                    selected_improved = _score_improved(quality['selected_score'], baseline_selected)
                    holdout_stable = _score_not_worse(
                        quality['holdout_score'], baseline_holdout, tolerance_ratio=0.05
                    )

                    # Opt-in: when the dominant peak's score is unreliable
                    # (NaN or both ties at the physical ceiling), fall back
                    # to the weighted multi-peak (train) score so the
                    # optimizer can still differentiate candidates.
                    #
                    # Mode is read from
                    # ``variables.voltage_bowl_optimizer_metric``:
                    #   'selected' (default) -> legacy behavior (selected only)
                    #   'train'              -> always use the multi-peak train score
                    #   'auto'               -> use train when selected is NaN
                    _metric_mode = getattr(variables, 'voltage_bowl_optimizer_metric', 'selected')
                    if _metric_mode == 'train':
                        accepted = _score_improved(quality['train_score'], baseline_train)
                    elif _metric_mode == 'auto':
                        if not (np.isfinite(quality['selected_score'])
                                and np.isfinite(baseline_selected)):
                            accepted = _score_improved(quality['train_score'], baseline_train)
                        else:
                            accepted = selected_improved
                    else:  # 'selected' (default, legacy behavior)
                        accepted = selected_improved

                    if accepted:
                        best_train_score = quality['train_score']
                        best_holdout_score = quality['holdout_score']
                        best_selected_score = quality['selected_score']
                        best_state = candidate_state
                        best_selection = candidate_selection
                        improved_this_round = True
                        warn = (
                            ' (note: held-out score regressed by >5%)'
                            if reference_peaks['holdout'] and not holdout_stable
                            else ''
                        )
                        print(
                            f'Accepted {action_name}; best scores are now '
                            f'train={best_train_score:.2f}, holdout={best_holdout_score:.2f}, '
                            f'selected={best_selected_score:.2f}{warn}'
                        )
                    else:
                        _restore_state(before_state)
                        _restore_selection(before_selection)
                        print(f'No MRP improvement on the dominant peak after {action_name}; reverted this step.')
                except Exception as exc:
                    _restore_state(before_state)
                    _restore_selection(before_selection)
                    print(f'{action_name} failed: {exc}')

            if improved_this_round:
                no_improve_count = 0
            else:
                if retry_peak_window_on_stall:
                    refreshed = _refresh_peak_window_plot(figure_size)
                    if refreshed:
                        print('No stable improvement detected; refreshed the peak window from the histogram and continuing.')
                        continue
                no_improve_count += 1
                print(f'No improvement round count: {no_improve_count}/{max_no_improve}')
                if no_improve_count >= max_no_improve:
                    break

        _restore_state(best_state)
        _restore_selection(best_selection)
        runtime_s = time.perf_counter() - start_time
        final_mrp = _print_mrp('Restored best')
        final_quality = _print_quality('Restored best', reference_peaks)
        print(f'{title} finished in {runtime_s:.2f}s')
        print(
            f'Best weighted Gaussian scores: '
            f'train={final_quality["train_score"]:.2f}, '
            f'holdout={final_quality["holdout_score"]:.2f}, '
            f'selected={final_quality["selected_score"]:.2f}'
        )
        print(f'Final selected window MRP values: {final_mrp}')

    def automatic_bowl_calibration(_, variables, output, status_output, calibration_mode_widget, pulse_mode_value):
        auto_button_bowl.disabled = True
        with status_output, _verbosity_context():
            status_output.clear_output()
            _ensure_initial_calibration()
            _prepare_locked_selection()
            _optimize_sequence(
                [('Bowl correction', lambda: _run_bowl_correction(plot_override=False, save_override=False))],
                title='Auto bowl calibration',
                figure_size=(figure_b_size_x.value, figure_b_size_y.value),
                max_iterations=10,
                max_no_improve=3,
                retry_peak_window_on_stall=False,
            )
        index_fig_v.value = 1
        index_fig_b.value = 1
        auto_button_bowl.disabled = False
        # Auto-plot the result so the user can see the corrected histogram
        # without having to click Plot.
        try:
            plot_button.click()
        except Exception:
            pass

    def _run_voltage_then_bowl():
        """Run voltage correction followed by bowl correction as one atomic step.

        Treating (Voltage, Bowl) as a single action lets ``_optimize_sequence``
        accept or reject the pair on the combined effect, matching the manual
        workflow (the user always presses Vol then Bowl together, even when
        Vol alone temporarily worsens the peak before Bowl compensates).
        Splitting them into two separate actions made the loop revert the
        voltage half before the bowl half had a chance to recover it,
        especially in ToF mode.
        """
        _run_voltage_correction(plot_override=False, save_override=False)
        _run_bowl_correction(plot_override=False, save_override=False)

    def automatic_calibration(_, variables, output, status_output, calibration_mode_widget, pulse_mode_value):
        # Auto / FAST button = the legacy iterative Voltage + Bowl
        # optimizer. Same behavior across every preset. The fancy
        # joint V+Bowl, time-dep V, reference-optimizer pipeline lives
        # ONLY in the Hybrid (BEST) button because each of those stages
        # depends on the adaptive-residual cleanup that follows.
        auto_button.disabled = True
        simple_auto_button.disabled = True
        _t_auto_start = time.perf_counter()
        with status_output, _verbosity_context():
            status_output.clear_output()
            print(f"[Auto calibration] start (mode={calibration_mode_widget.value})")
            _ensure_initial_calibration()
            _prepare_locked_selection()
            _optimize_sequence(
                [('Voltage + Bowl correction', _run_voltage_then_bowl)],
                title='Auto calibration',
                figure_size=(figure_mc_size_x.value, figure_mc_size_y.value),
                max_iterations=10,
                max_no_improve=3,
                retry_peak_window_on_stall=False,
            )
            print(f"[Auto calibration] finished in {time.perf_counter() - _t_auto_start:.1f}s")
        index_fig_v.value = 1
        index_fig_b.value = 1
        auto_button.disabled = False
        simple_auto_button.disabled = False
        try:
            plot_button.click()
        except Exception:
            pass

    def run_auto_for_mode(mode_key):
        """Canonical 'Auto calibration' entry point for one mode. Same
        pattern as ``run_hybrid_for_mode``: explicit mode-key arg so the
        mc tab, tof tab, and combined-FAST paths all delegate to one
        function and cannot drift apart.
        """
        calibration_mode.value = mode_key
        automatic_calibration(None, variables, out, out_status, calibration_mode, pulse_mode)

    def run_mc_auto_calibration():
        run_auto_for_mode('mc_calib')

    def run_tof_auto_calibration():
        run_auto_for_mode('tof_calib')

    def on_hybrid_auto_residual(_):
        hybrid_button.disabled = True
        simple_hybrid_button.disabled = True
        _t_hybrid_start = time.perf_counter()
        try:
            with out_status, _verbosity_context():
                out_status.clear_output()
                if not _selected_peak_ready():
                    print('Please first select a peak')
                    return
                print(f"[Hybrid auto + residual] start (mode={calibration_mode.value})")
                _ensure_initial_calibration()
                _prepare_locked_selection()
                _use_joint = bool(getattr(variables, 'use_joint_vbowl', False))
                _use_time_dep_v = bool(getattr(variables, 'use_time_dep_v', False))
                mode_key = _calibration_mode_key()
                if _use_joint:
                    try:
                        from pyccapt.calibration.core import calibration as _cal
                        _cal.joint_voltage_bowl_corr_main(
                            variables.dld_x_det, variables.dld_y_det,
                            _current_voltage(),
                            variables, det_diam,
                            calibration_mode=mode_key,
                            sample_size=9, bin_size=0.05, n_peaks=4,
                            prominence=100, distance=500,
                            sampling_mode=_sampling_mode_value(),
                        )
                        print('Joint V+Bowl correction applied.')
                    except Exception as exc:
                        print(f'Joint V+Bowl failed ({exc}); falling back to legacy V+Bowl loop.')
                        _use_joint = False
                if _use_joint and _use_time_dep_v:
                    # M3v2 refinement: time-dependent V residual on top of
                    # joint V+Bowl. Safe here because the adaptive residual
                    # below cleans up any chunk-to-chunk drift it introduces.
                    try:
                        from pyccapt.calibration.core import new_methods as _nm
                        _nm.voltage_corr_time_dependent(
                            _current_voltage(), variables,
                            calibration_mode=mode_key,
                            n_time_bins=12, bin_size=0.05, sample_size=100,
                            use_legacy_v=False,
                        )
                        print('Time-dependent V refinement applied.')
                    except Exception as exc:
                        print(f'Time-dep V refinement failed ({exc}); skipping.')
                if not _use_joint:
                    # Legacy iterative V+Bowl loop (atomic Voltage+Bowl step
                    # so the optimizer accepts/reverts the pair on its
                    # combined effect; needed in ToF where Vol alone often
                    # regresses before Bowl recovers it).
                    _optimize_sequence(
                        [('Voltage + Bowl correction', _run_voltage_then_bowl)],
                        title='Hybrid auto + residual',
                        figure_size=(figure_mc_size_x.value, figure_mc_size_y.value),
                        max_iterations=10,
                        max_no_improve=3,
                        retry_peak_window_on_stall=False,
                    )
                post_auto_state = _capture_state()
                post_auto_selection = _capture_selection()
                print('-------------------------------------------------------')
                print(
                    f'Running adaptive residual refinement on {mode_key} '
                    f'with tuned defaults (peaks=6, prominence=100, distance=10).'
                )
                # Hybrid's adaptive-residual stage reads its parameters from
                # the calibration profile. 'old' = legacy defaults.
                # 'new' = identical to 'old' except n_windows=32 (was 24) --
                # the single change the peak-quality audit on the Nimonic
                # NiC1 dataset showed beats OLD on mc on BOTH n_peaks AND
                # MRP simultaneously. Earlier 'new' values (prominence=50,
                # template_bin=0.005, apply_spatial=False, fast_score=True)
                # were withdrawn after the audit showed they merged peaks
                # or were no-ops at 2M ions.
                _profile = getattr(variables, 'calibration_profile', 'old')
                _n_windows = 32 if _profile == 'new' else 24
                _res_kwargs = dict(
                    n_peaks=6, prominence=100, distance=10, n_windows=_n_windows,
                    overlap=0.5, template_bin_size=0.01,
                    temporal_smoothing=0.5, apply_spatial=True,
                    spatial_grid=12, min_window_ions=40, min_cell_ions=35,
                    max_rounds=8,
                    # 'best' / 'ref' presets set residual_coarse_to_fine_top_k=1
                    # which makes the residual ~12x faster on mc at audit-equivalent
                    # quality (rank candidates by fast_mrp, Voigt-verify only top 1).
                    coarse_to_fine_top_k=getattr(variables, 'residual_coarse_to_fine_top_k', None),
                )
                try:
                    result = adaptive_residual_calibration(
                        variables,
                        calibration_mode=mode_key,
                        verbose=verbose.value,
                        above_ceiling_strategy=getattr(
                            variables, 'voltage_bowl_above_ceiling_strategy', 'nan'
                        ),
                        **_res_kwargs,
                    )
                except Exception as exc:
                    _restore_state(post_auto_state)
                    _restore_selection(post_auto_selection)
                    print(f'Adaptive residual refinement failed; restored the auto-calibration result: {exc}')
                    return

                # Reference-fit calibration was moved out of the
                # V+Bowl pipeline. It now lives in the ion-list step
                # (helper_ion_list.call_ion_list) where the user has
                # already supplied an expected-element list -- that's
                # the right place for absolute-m/c alignment.

                print(
                    'Hybrid final weighted Gaussian score '
                    f"(train={result['final_quality']['train_score']:.2f}, "
                    f"holdout={result['final_quality']['holdout_score']:.2f})"
                )
                if result['accepted_steps']:
                    print(f"Adaptive residual accepted steps: {result['accepted_steps']}")
                else:
                    print('Adaptive residual accepted no additional steps; auto-calibration result was kept.')
        finally:
            try:
                with out_status:
                    print(f"[Hybrid auto + residual] finished in {time.perf_counter() - _t_hybrid_start:.1f}s")
            except Exception:
                pass
            hybrid_button.disabled = False
            simple_hybrid_button.disabled = False
            try:
                plot_button.click()
            except Exception:
                pass

    def run_hybrid_for_mode(mode_key):
        """Canonical 'Hybrid auto + residual' entry point for one mode.

        Used by every entry point so the three paths (mc tab Hybrid, tof tab
        Hybrid, combined BEST) cannot drift apart:

        - mc tab's Hybrid button -> run_hybrid_for_mode('mc_calib')
        - tof tab's Hybrid button -> run_hybrid_for_mode('tof_calib')
        - combined BEST button -> run_hybrid_for_mode('mc_calib') then
          run_hybrid_for_mode('tof_calib')

        Body of the per-tab Hybrid handler is reused verbatim via
        ``on_hybrid_auto_residual`` -- the only thing this wrapper adds is
        explicitly setting ``calibration_mode.value`` before the call, so
        callers don't need to remember to do it.
        """
        calibration_mode.value = mode_key
        on_hybrid_auto_residual(None)

    def run_mc_hybrid_auto_residual():
        run_hybrid_for_mode('mc_calib')

    def run_tof_hybrid_auto_residual():
        run_hybrid_for_mode('tof_calib')

    def on_gaussian_mrp(_):
        # The user explicitly clicked the Gaussian MRP button to see the
        # report. The verbose dropdown only controls passive logging from
        # automatic calibration steps; an explicit button click should always
        # render its output regardless of that dropdown's value.
        gaussian_mrp_button.disabled = True
        with out_status:
            out_status.clear_output()
            if not _selected_peak_ready():
                print('Please first select a peak')
            else:
                result = gaussian_mrp_report(
                    _get_calibration_array(),
                    variables.selected_x1,
                    variables.selected_x2,
                    bin_size=0.001,
                )
                if result is None:
                    print('Gaussian MRP: insufficient data in selected range')
                else:
                    print('=' * 60)
                    print('PEAK PROFILE MRP REPORT')
                    print('=' * 60)
                    print(f'MRP model: {result["recommended_label"]}')
                    print(f'MRP bin size used: {result["bin_size"]} ({result["num_bins"]} bins)')
                    print(f'Peak position: {result["peak_position"]:.4f}')
                    print(f'Ions in range: {result["num_ions"]:,}')
                    print(f'Recommended FWHM MRP: {result["formatted_recommended_mrp"][0]}')
                    if result['window_warning']:
                        print(result['window_warning'])
                    print()
                    if result['gaussian_ok']:
                        print('Gaussian fit MRP (sub-bin accuracy):')
                        print(f'  MRP(0.5)  = {result["formatted_gaussian_mrp"][0]}')
                        print(f'  MRP(0.1)  = {result["formatted_gaussian_mrp"][1]}')
                        print(f'  MRP(0.01) = {result["formatted_gaussian_mrp"][2]}')
                    else:
                        print('Gaussian fit FAILED')
                    print()
                    if result['voigt_ok']:
                        print(f'Voigt fit MRP ({result["profile_type"]}):')
                        print(f'  MRP(0.5)  = {result["formatted_voigt_mrp"][0]}')
                        print(f'  MRP(0.1)  = {result["formatted_voigt_mrp"][1]}')
                        print(f'  MRP(0.01) = {result["formatted_voigt_mrp"][2]}')
                        print(f'  Voigt FWHM = {result["voigt_fwhm"]:.6f}')
                    else:
                        print('Voigt fit FAILED')
                    print()
                    if result.get('asymmetric_ok'):
                        print('Asymmetric (err*expDecay) fit MRP:')
                        print(f'  MRP(0.5)  = {result["formatted_asymmetric_mrp"][0]}')
                        print(f'  MRP(0.1)  = {result["formatted_asymmetric_mrp"][1]}')
                        print(f'  MRP(0.01) = {result["formatted_asymmetric_mrp"][2]}')
                        asym_fwhm = result.get('asymmetric_fwhm', float('nan'))
                        if np.isfinite(asym_fwhm):
                            print(f'  Asymmetric FWHM = {asym_fwhm:.6f}')
                    else:
                        print('Asymmetric (err*expDecay) fit FAILED')
                    print()
                    print('Histogram-based MRP (for comparison):')
                    print(f'  MRP(0.5)  = {result["formatted_histogram_mrp"][0]}')
                    print(f'  MRP(0.1)  = {result["formatted_histogram_mrp"][1]}')
                    print(f'  MRP(0.01) = {result["formatted_histogram_mrp"][2]}')
                    print('=' * 60)
        gaussian_mrp_button.disabled = False

    plot_button.on_click(lambda b: hist_plot(b, variables, out, calibration_mode))
    plot_stat_button.on_click(lambda b: stat_plot(b, variables, calibration_mode, out))
    reset_back_button.on_click(lambda b: reset_back_on_click(variables, calibration_mode))
    reset_button.on_click(lambda b: reset_on_click(variables, calibration_mode))
    save_button.on_click(lambda b: save_on_click(variables, calibration_mode))
    vol_button.on_click(lambda b: vol_correction(b, variables, out, out_status, calibration_mode, pulse_mode))
    bowl_button.on_click(lambda b: bowl_correction(b, variables, out, out_status, calibration_mode, pulse_mode))
    clear_plot.on_click(lambda b: clear_plot_on_click(out, out_status))
    auto_button.on_click(lambda b: run_auto_for_mode(calibration_mode.value))
    auto_button_bowl.on_click(
        lambda b: automatic_bowl_calibration(b, variables, out, out_status, calibration_mode, pulse_mode)
    )
    initial_calib_button.on_click(lambda b: initial_calibration(b, variables, calibration_mode, flight_path_length))
    gaussian_mrp_button.on_click(on_gaussian_mrp)
    hybrid_button.on_click(lambda b: run_hybrid_for_mode(calibration_mode.value))
    sampling_mode_b.observe(
        lambda change: setattr(variables, 'bowl_sampling_mode', change['new']),
        names='value',
    )

    plot_button.click()
    sample_size_v_set(sample_size_v)

    column11 = widgets.VBox(
        [
            bin_size,
            lim_tof,
            prominence,
            distance,
            percent,
            bin_fdm,
            plot_peak,
            index_fig,
            save,
            verbose,
            figure_mc_size_x,
            figure_mc_size_y,
        ]
    )
    column12 = widgets.VBox(
        [
            plot_button,
            save_button,
            reset_back_button,
            reset_button,
            clear_plot,
            gaussian_mrp_button,
            plot_stat_button,
        ]
    )
    column22 = widgets.VBox(
        [
            sample_size_b,
            sample_size_b_help,
            bin_size_b,
            fit_mode_b,
            bowl_method_b,
            sampling_mode_b,
            maximum_cal_method_b,
            maximum_sample_method_b,
            plot_b,
            index_fig_b,
            save_b,
            figure_b_size_x,
            figure_b_size_y,
        ]
    )
    column21 = widgets.VBox([bowl_button, pb_bowl])
    column33 = widgets.VBox(
        [
            sample_size_v,
            bin_size_v,
            model_v,
            maximum_cal_method_v,
            maximum_sample_method_v,
            mode_v,
            plot_v,
            index_fig_v,
            save_v,
            figure_v_size_x,
            figure_v_size_y,
        ]
    )
    initial_calib_hint = widgets.HTML(
        value=(
            '<span style="font-size:11px; color:#a00;">'
            'Note: <b>Initial calibration</b> must be run before <b>Voltage correction</b>. '
            'The auto-* buttons below will run it for you automatically if it has not been done.'
            '</span>'
        ),
        layout=widgets.Layout(width='420px'),
    )
    column32 = widgets.HBox(
        [
            widgets.VBox([vol_button, pb_vol]),
            widgets.VBox([initial_calib_button, initial_calib_hint]),
        ]
    )
    column34 = widgets.VBox(
        [fast_calibration, refine_nelder_mead_widget, automatic_window_update, lock_peak_selection, peak_val]
    )

    layout1 = widgets.HBox([column11, column22, column33, column34])
    layout2 = widgets.HBox([column12, column21, column32])
    advanced_action_row = widgets.HBox(
        [
            auto_button,
            auto_button_bowl,
            hybrid_button,
        ]
    )
    advanced_panel = widgets.VBox([layout1, layout2, advanced_action_row, widgets.VBox([out, out_status])])

    simple_bin_size = widgets.FloatText(value=bin_size.value, description='Bin size:', layout=label_layout)
    simple_lim = widgets.IntText(value=lim_tof.value, description='Lim tof/mc:', layout=label_layout)
    simple_percent = widgets.IntText(value=percent.value, description='Percent MRP:', layout=label_layout)
    simple_bin_fdm = widgets.IntText(value=bin_fdm.value, description='Bin FDM:', layout=label_layout)
    simple_plot_peak = widgets.Dropdown(
        options=plot_peak.options, value=plot_peak.value, description='Plot peak', layout=label_layout
    )
    simple_index_fig = widgets.IntText(value=index_fig.value, description='Fig save index:', layout=label_layout)
    simple_save = widgets.Dropdown(options=save.options, value=save.value, description='Save fig:', layout=label_layout)
    simple_fig_w = widgets.FloatText(value=figure_mc_size_x.value, description='Fig. size W:', layout=label_layout)
    simple_fig_h = widgets.FloatText(value=figure_mc_size_y.value, description='Fig. size H:', layout=label_layout)

    widgets.link((simple_bin_size, 'value'), (bin_size, 'value'))
    widgets.link((simple_lim, 'value'), (lim_tof, 'value'))
    widgets.link((simple_percent, 'value'), (percent, 'value'))
    widgets.link((simple_bin_fdm, 'value'), (bin_fdm, 'value'))
    widgets.link((simple_plot_peak, 'value'), (plot_peak, 'value'))
    widgets.link((simple_index_fig, 'value'), (index_fig, 'value'))
    widgets.link((simple_save, 'value'), (save, 'value'))
    widgets.link((simple_fig_w, 'value'), (figure_mc_size_x, 'value'))
    widgets.link((simple_fig_h, 'value'), (figure_mc_size_y, 'value'))

    simple_plot_button = widgets.Button(description='Plot hist', layout=label_layout)
    simple_save_button = widgets.Button(description='Save correction', layout=label_layout)
    simple_reset_back_button = widgets.Button(description='Back to saved', layout=label_layout)
    simple_reset_button = widgets.Button(description='Reset correction', layout=label_layout)
    simple_clear_button = widgets.Button(description='Clear plots', layout=label_layout)
    simple_gaussian_button = widgets.Button(description='MRP', layout=label_layout)
    simple_plot_stat_button = widgets.Button(description='Plot stat', layout=label_layout)
    simple_auto_button = widgets.Button(description='Auto calibration', layout=label_layout)
    simple_hybrid_button = widgets.Button(description='Hybrid auto + residual', layout=label_layout)

    simple_plot_button.on_click(lambda _: hist_plot(None, variables, out, calibration_mode))
    simple_save_button.on_click(lambda _: save_on_click(variables, calibration_mode))
    simple_reset_back_button.on_click(lambda _: reset_back_on_click(variables, calibration_mode))
    simple_reset_button.on_click(lambda _: reset_on_click(variables, calibration_mode))
    simple_clear_button.on_click(lambda _: clear_plot_on_click(out, out_status))
    simple_gaussian_button.on_click(on_gaussian_mrp)
    simple_plot_stat_button.on_click(lambda _: stat_plot(_, variables, calibration_mode, out))
    simple_auto_button.on_click(lambda _: run_auto_for_mode(calibration_mode.value))
    simple_hybrid_button.on_click(lambda b: run_hybrid_for_mode(calibration_mode.value))
    simple_controls = widgets.VBox(
        [
            simple_bin_size,
            simple_lim,
            simple_percent,
            simple_bin_fdm,
            simple_plot_peak,
            simple_index_fig,
            simple_save,
            verbose,
            simple_fig_w,
            simple_fig_h,
        ]
    )
    simple_common_actions = widgets.VBox(
        [
            simple_plot_button,
            simple_save_button,
            simple_reset_back_button,
            simple_reset_button,
            simple_clear_button,
            simple_gaussian_button,
            simple_plot_stat_button,
        ]
    )
    simple_mode_actions = widgets.VBox()
    simple_panel = widgets.VBox(
        [
            widgets.HBox([simple_controls, simple_common_actions, simple_mode_actions]),
            widgets.VBox([out, out_status]),
        ]
    )
    subtab_placeholders = [widgets.VBox(), widgets.VBox()]
    sub_tabs = widgets.Tab(children=subtab_placeholders)
    sub_tabs.set_title(0, 'simple')
    sub_tabs.set_title(1, 'advance')

    def _render_subtab_content():
        selected_panel = simple_panel if sub_tabs.selected_index == 0 else advanced_panel
        for index, placeholder in enumerate(subtab_placeholders):
            placeholder.children = (selected_panel,) if index == sub_tabs.selected_index else ()

    def _sync_mode_ui(*_):
        if mode_tabs.selected_index not in (1, 2):
            _render_top_content()
            return
        mode_key = 'mc_calib' if mode_tabs.selected_index == 1 else 'tof_calib'
        calibration_mode.value = mode_key
        lim_tof.value = variables.max_tof if mode_key == 'tof_calib' else 400
        simple_mode_actions.children = (
            simple_auto_button,
            simple_hybrid_button,
        )
        _render_subtab_content()
        _render_top_content()

    adaptive_panel, run_adaptive_for_mode, _adaptive_apply_profile = build_adaptive_residual_calibration_panel(
        variables, det_diam, flight_path_length, pulse_mode
    )
    combined_panel = build_combined_mc_tof_calibration_panel(
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
        _auto_select_peak_for_mode,
        _selected_peak_ready,
        _verbosity_context,
        lambda: initial_calibration(None, variables, calibration_mode, flight_path_length),
        lambda: automatic_calibration(None, variables, out, out_status, calibration_mode, pulse_mode),
        lambda: on_hybrid_auto_residual(None),
        _save_both_corrections,
        _restore_both_corrections,
        _reset_both_corrections,
        lambda: clear_plot_on_click(out, out_status),
        _print_gaussian_for_current_mode,
        run_adaptive_for_mode,
        _ensure_initial_calibration,
        run_mc_hybrid_auto_residual,
        run_tof_hybrid_auto_residual,
        run_mc_auto_calibration,
        run_tof_auto_calibration,
    )

    top_placeholders = [widgets.VBox(), widgets.VBox(), widgets.VBox(), widgets.VBox()]
    mode_tabs = widgets.Tab(children=top_placeholders)
    mode_tabs.set_title(0, 'mc + tof calibration')
    mode_tabs.set_title(1, 'mc calibration')
    mode_tabs.set_title(2, 'tof calibration')
    mode_tabs.set_title(3, 'adaptive residual')

    def _render_top_content():
        mapping = {0: combined_panel, 1: sub_tabs, 2: sub_tabs, 3: adaptive_panel}
        for index, placeholder in enumerate(top_placeholders):
            placeholder.children = (mapping[index],) if index == mode_tabs.selected_index else ()

    mode_tabs.observe(_sync_mode_ui, names='selected_index')
    sub_tabs.observe(lambda change: _render_subtab_content(), names='selected_index')

    def _sync_lim_to_mode(change):
        # Keep lim_tof aligned with the current calibration mode so callers
        # that switch calibration_mode programmatically (e.g. the combined
        # mc+tof tab's fast / best buttons looping over both modes) don't
        # plot tof with the m/c limit or vice versa.
        new_value = change.get('new', calibration_mode.value)
        lim_tof.value = variables.max_tof if new_value == 'tof_calib' else 400

    calibration_mode.observe(_sync_lim_to_mode, names='value')

    # ------------------------------------------------------------------
    # Calibration profile selector — opt into the benchmark-derived "new"
    # configuration without overwriting the committed defaults. See
    # ``benchmark_results/REPORT.md`` for which knobs change.
    # ------------------------------------------------------------------
    calibration_profile = widgets.Dropdown(
        options=[
            ('Joint V+Bowl (default)', 'new'),
            ('Joint V+Bowl + time-drift correction (recommended)', 'best'),
            ('Sequential V+Bowl (old)', 'old'),
        ],
        value='new',
        description='Config preset:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='720px'),
    )

    def _apply_profile(change):
        new_profile = change.get('new', calibration_profile.value) if isinstance(change, dict) else change

        # Reset shared widget values to OLD defaults first; the per-profile
        # branches below override only the knobs they need to.
        def _reset_widgets_to_old():
            sample_size_b.value = 5
            sampling_mode_b.value = 'cartesian'
            bin_size_b.value = 0.01
            fit_mode_b.value = 'robust_fit'
            sample_size_v.value = 10000
            bin_size_v.value = 0.01
            model_v.value = 'robust_fit'
            refine_nelder_mead_widget.value = False
            bin_size.value = 0.1

        def _clear_opt_in_attrs():
            for attr in (
                'voltage_bowl_above_ceiling_strategy',
                'voltage_bowl_optimizer_metric',
                'residual_fast_candidate_score',
                'residual_coarse_to_fine_top_k',
                'use_joint_vbowl',
                'use_time_dep_v',
            ):
                if hasattr(variables, attr):
                    try:
                        delattr(variables, attr)
                    except AttributeError:
                        pass

        if new_profile == 'new':
            # "Joint V+Bowl": replaces legacy iterative V+Bowl with the
            # one-shot joint multi-peak solver. Adaptive residual keeps
            # legacy defaults (n_windows=24); the final stack audit showed
            # n_windows=32 only helps on the OLD V+Bowl baseline and
            # actively hurts when joint V+Bowl is used.
            _reset_widgets_to_old()
            _clear_opt_in_attrs()
            variables.calibration_profile = 'new'
            variables.use_joint_vbowl = True
            # Coarse-to-fine residual: rank candidates with cheap fast_mrp,
            # Voigt-verify only the top 1. Audited Path 2 at 2M ions on the
            # joint-V+Bowl warm-start gave 12.6x mc / 2.8x tof speedup AND
            # +8 mc peaks, +26% mc MRP. Same speedup applies here because
            # 'new' uses the same joint V+Bowl warm-start.
            variables.residual_coarse_to_fine_top_k = 1
            _adaptive_apply_profile('old')
        elif new_profile == 'best':
            # M1+M3v2 best preset:
            #   - Joint V+Bowl                            -- M1
            #   - Time-dep V refinement on top            -- M3v2
            #   - Adaptive residual n_windows=24 with coarse-to-fine top_k=1
            #
            # n_windows=24 (NOT 32) is the audited best for the M1 stack.
            # coarse-to-fine top_k=1 is the Path 2 speedup: scores every
            # residual candidate with fast_mrp, then Voigt-verifies only
            # the single best. Audit at 2M ions: 12.6x mc speedup AND
            # +8 mc peaks (77->85), +26% mc MRP (883->1113), +10 tof peaks.
            # Surprising win: top_k=1 outperforms top_k=2/3 because it
            # acts as implicit regularisation against Voigt-score noise.
            _reset_widgets_to_old()
            _clear_opt_in_attrs()
            variables.calibration_profile = 'best'
            variables.use_joint_vbowl = True
            variables.use_time_dep_v = True
            variables.residual_coarse_to_fine_top_k = 1
            _adaptive_apply_profile('old')  # n_windows=24, NOT 32
        else:
            # Sequential V+Bowl (old): legacy iterative V+Bowl + adaptive
            # residual. Coarse-to-fine top_k=1 is enabled here too because
            # it speeds the residual ~3x on the legacy warm-start as well
            # (the c2f trick is warm-start-agnostic: it just replaces the
            # all-candidate Voigt scoring with fast_mrp ranking + 1-of-K
            # Voigt verification). If quality regresses on this preset for
            # a given dataset, unset variables.residual_coarse_to_fine_top_k
            # before pressing Hybrid.
            _reset_widgets_to_old()
            _clear_opt_in_attrs()
            variables.calibration_profile = 'old'
            variables.residual_coarse_to_fine_top_k = 1
            _adaptive_apply_profile('old')

    calibration_profile.observe(_apply_profile, names='value')
    # Initialize variables.calibration_profile (defaults to 'old' = current behavior).
    variables.calibration_profile = 'old'

    # User-facing summary of what each preset actually changes. Kept in sync
    # with _apply_profile above and the per-button code paths
    # (automatic_calibration, on_hybrid_auto_residual, initial_calibration).
    profile_note = widgets.HTML(
        value=(
            '<div style="font-size:11px; color:#444; '
            'background:#f7f7f7; border:1px solid #ddd; padding:6px 8px; '
            'border-radius:4px; max-width:720px; line-height:1.45;">'
            '<b>What this preset changes</b><br>'
            'The preset only affects the <b>Hybrid auto + residual</b> button '
            '(and the combined <b>BEST</b> button which delegates to it). '
            'All other buttons &mdash; <b>Initial calibration</b>, '
            '<b>Voltage correction</b>, <b>Bowl correction</b>, '
            '<b>Auto calibration</b>, <b>Auto bowl calibration</b>, '
            '<b>FAST</b>, <b>Adaptive residual</b> &mdash; always run the '
            'legacy sequential V+Bowl path and ignore this dropdown.'
            '<ul style="margin:4px 0 4px 16px;">'
            '<li><b>Sequential V+Bowl (old)</b>: legacy iterative '
            'Voltage&rarr;Bowl optimiser. Used by every button.</li>'
            '<li><b>Joint V+Bowl (default)</b>: Hybrid replaces its '
            'V+Bowl stage with one joint multi-peak constrained fit. '
            'Other buttons unchanged.</li>'
            '<li><b>+ time-drift correction</b> (recommended): Hybrid adds '
            'a per-ion-index voltage residual after joint V+Bowl. Cancels '
            'voltage / temperature drift that leaves residual mass shifts '
            'in long runs. See explanation below the tabs.</li>'
            '</ul>'
            'Switch any time; widget settings are not touched.'
            '</div>'
        ),
        layout=widgets.Layout(width='720px'),
    )

    method_help = widgets.HTML(
        value=(
            '<div style="font-size:11px; color:#444; '
            'background:#fafcff; border:1px solid #d6e3f3; padding:6px 8px; '
            'border-radius:4px; max-width:720px; line-height:1.5;">'
            '<b>Time-drift correction (M3v2)</b><br>'
            'Atom-probe runs are long (hours). The HV supply, laser power, '
            'and specimen temperature drift slowly during a run, which '
            'shifts the reference peak position as a function of acquisition '
            'time (ion index). The classical V correction fits ONE global '
            'voltage&ndash;m/c scaling for the whole dataset, so this slow '
            'drift leaks into the calibrated spectrum and broadens peaks.'
            '<br><br>'
            'This step splits the run into ~12 equal time bins, measures '
            'the dominant peak\'s centre in each bin, and fits a smooth '
            'low-degree polynomial of <i>time-index &rarr; multiplicative '
            'correction</i>. The correction is then applied per ion, on top '
            'of joint V+Bowl. Safe to apply because (a) the polynomial is '
            'low-degree and gently regularised, and (b) the subsequent '
            'adaptive residual stage cleans up any residual chunk-to-chunk '
            'wobble.'
            '<br><br>'
            'Reference-fit (NIST) is no longer a calibration preset &mdash; '
            'it now lives in the ion-list cell as a second "Fit method" '
            'option alongside the existing parametric fit. That cell is '
            'the natural place for peak labelling + absolute m/c alignment.'
            '</div>'
        ),
        layout=widgets.Layout(width='720px'),
    )

    profile_panel = widgets.VBox([
        widgets.HTML('<b>Calibration profile</b>'),
        calibration_profile,
        profile_note,
        method_help,
    ], layout=widgets.Layout(border='1px solid #ccc', padding='6px', margin='0 0 8px 0'))

    mode_tabs.selected_index = 0
    _sync_mode_ui()
    _render_top_content()
    display(widgets.VBox([profile_panel, mode_tabs]))
