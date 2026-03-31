import time

import ipywidgets as widgets
import numpy as np
from IPython.display import clear_output
from IPython.display import display
from ipywidgets import Output

from pyccapt.calibration.core import calibration, mc_plot
from pyccapt.calibration.core.mc_plot_peak_helpers import fast_mrp, gaussian_mrp_report

label_layout = widgets.Layout(width='300px')


def reset_on_click(variables, calibration_mode):
    if calibration_mode.value == 'tof_calib':
        variables.dld_t_calib = variables.data['t (ns)'].to_numpy()
    elif calibration_mode.value == 'mc_calib':
        variables.mc_calib = variables.data['mc_uc (Da)'].to_numpy()


def reset_back_on_click(variables, calibration_mode):
    if calibration_mode.value == 'tof_calib':
        variables.dld_t_calib = np.copy(variables.dld_t_calib_backup)
    elif calibration_mode.value == 'mc_calib':
        variables.mc_calib = np.copy(variables.mc_calib_backup)


def save_on_click(variables, calibration_mode):
    if calibration_mode.value == 'tof_calib':
        variables.dld_t_calib_backup = np.copy(variables.dld_t_calib)
    elif calibration_mode.value == 'mc_calib':
        variables.mc_calib_backup = np.copy(variables.mc_calib)


def clear_plot_on_click(out, out_status):
    with out:
        out.clear_output()
    with out_status:
        out_status.clear_output()
    clear_output(wait=True)


def call_voltage_bowl_calibration(variables, det_diam, flight_path_length, pulse_mode):
    out = Output()
    out_status = Output()

    plot_button = widgets.Button(description='plot hist', layout=label_layout)
    plot_stat_button = widgets.Button(description='plot stat', layout=label_layout)
    reset_back_button = widgets.Button(description='back to saved', layout=label_layout)
    reset_button = widgets.Button(description='reset correction', layout=label_layout)
    save_button = widgets.Button(description='save correction', layout=label_layout)
    bowl_button = widgets.Button(description='bowl correction', layout=label_layout)
    vol_button = widgets.Button(description='voltage correction', layout=label_layout)
    auto_button = widgets.Button(description='auto calibration', layout=label_layout)
    auto_button_bowl = widgets.Button(description='auto bowl calibration', layout=label_layout)
    gaussian_mrp_button = widgets.Button(description='Gaussian MRP', layout=label_layout)
    multi_peak_button = widgets.Button(description='Multi-Peak Calibrate', layout=label_layout)
    auto_optimize_button = widgets.Button(
        description='Auto Optimize MRP',
        layout=widgets.Layout(width='620px'),
        button_style='success',
    )
    initial_calib_button = widgets.Button(description='initial calibration', layout=label_layout)
    clear_plot = widgets.Button(description="clear plots", layout=label_layout)

    calibration_mode = widgets.Dropdown(
        options=[('mass_to_charge', 'mc_calib'), ('time_of_flight', 'tof_calib')],
        description='calibration mode:'
    )

    bin_size = widgets.FloatText(value=0.1, description='bin size:', layout=label_layout)
    prominence = widgets.IntText(value=100, description='peak prominance:', layout=label_layout)
    distance = widgets.IntText(value=500, description='peak distance:', layout=label_layout)
    lim_tof = widgets.IntText(value=variables.max_tof, description='lim tof/mc:', layout=label_layout)
    percent = widgets.IntText(value=50, description='percent MRP:', layout=label_layout)
    index_fig = widgets.IntText(value=1, description='fig save index:', layout=label_layout)
    plot_peak = widgets.Dropdown(
        options=[('True', True), ('False', False)],
        description='plot peak',
        layout=label_layout,
    )
    save = widgets.Dropdown(
        options=[('False', False), ('True', True)],
        description='save fig:',
        layout=label_layout,
    )
    figure_mc_size_x = widgets.FloatText(value=9.0, description="Fig. size W:", layout=label_layout)
    figure_mc_size_y = widgets.FloatText(value=5.0, description="Fig. size H:", layout=label_layout)

    sample_size_v = widgets.IntText(value=10000, description='sample size:', layout=label_layout)
    index_fig_v = widgets.IntText(value=1, description='fig index:', layout=label_layout)
    plot_v = widgets.Dropdown(
        options=[('False', False), ('True', True)],
        description='plot fig:',
        layout=label_layout,
    )
    save_v = widgets.Dropdown(
        options=[('False', False), ('True', True)],
        description='save fig:',
        layout=label_layout,
    )
    mode_v = widgets.Dropdown(
        options=[('ion_seq', 'ion_seq'), ('voltage', 'voltage')],
        description='sample mode:',
        layout=label_layout,
    )
    maximum_cal_method_v = widgets.Dropdown(
        options=[('mean', 'mean'), ('histogram', 'histogram'), ('median', 'median')],
        description='peak max:',
        layout=label_layout,
    )
    model_v = widgets.Dropdown(
        options=[('robust_fit', 'robust_fit'), ('curve_fit', 'curve_fit')],
        description='fit mode:',
        layout=label_layout,
    )
    maximum_sample_method_v = widgets.Dropdown(
        options=[('histogram', 'histogram'), ('mean', 'mean'), ('median', 'median')],
        description='sample max:',
        layout=label_layout,
    )
    bin_size_v = widgets.FloatText(value=0.01, description='bin size:', layout=label_layout)
    figure_v_size_x = widgets.FloatText(value=5.0, description="Fig. size W:", layout=label_layout)
    figure_v_size_y = widgets.FloatText(value=5.0, description="Fig. size H:", layout=label_layout)

    sample_size_b = widgets.IntText(value=5, description='sample size:', layout=label_layout)
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
        description='fit mode:',
        layout=label_layout,
    )
    sampling_mode_b = widgets.Dropdown(
        options=[('polar (default)', 'polar'), ('cartesian (legacy)', 'cartesian')],
        value=getattr(variables, 'bowl_sampling_mode', 'polar'),
        description='sampling mode:',
        layout=label_layout,
    )
    index_fig_b = widgets.IntText(value=1, description='fig index:', layout=label_layout)
    bin_size_b = widgets.FloatText(value=0.01, description='bin size:', layout=label_layout)
    maximum_cal_method_b = widgets.Dropdown(
        options=[('mean', 'mean'), ('histogram', 'histogram')],
        description='peak max:',
        layout=label_layout,
    )
    maximum_sample_method_b = widgets.Dropdown(
        options=[('histogram', 'histogram'), ('mean', 'mean')],
        description='sample max:',
        layout=label_layout,
    )
    plot_b = widgets.Dropdown(
        options=[('False', False), ('True', True)],
        description='plot fig:',
        layout=label_layout,
    )
    save_b = widgets.Dropdown(
        options=[('False', False), ('True', True)],
        description='save fig:',
        layout=label_layout,
    )
    fast_calibration = widgets.Dropdown(
        options=[('False', False), ('True', True)],
        description='fast calibration:',
        layout=label_layout,
    )
    automatic_window_update = widgets.Dropdown(
        options=[('False', False), ('True', True)],
        description='auto window update:',
        layout=label_layout,
    )
    lock_peak_selection = widgets.Dropdown(
        options=[('True', True), ('False', False)],
        description='lock peak ions:',
        layout=label_layout,
    )
    peak_val = widgets.FloatText(value=0, description='peak value:', layout=label_layout)
    figure_b_size_x = widgets.FloatText(value=5.0, description="Fig. size W:", layout=label_layout)
    figure_b_size_y = widgets.FloatText(value=5.0, description="Fig. size H:", layout=label_layout)

    pb_bowl = widgets.HTML(value=" ", placeholder='Status:', description='Status:', layout=label_layout)
    pb_vol = widgets.HTML(value=" ", placeholder='Status:', description='Status:', layout=label_layout)
    bin_fdm = widgets.IntText(value=256, description='bin FDM:', layout=label_layout)

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
        return float(variables.selected_x1), float(variables.selected_x2)

    def _restore_selection(selection):
        variables.selected_x1, variables.selected_x2 = selection

    def _get_calibration_array():
        if calibration_mode.value == 'tof_calib':
            return variables.dld_t_calib
        return variables.mc_calib

    def _prepare_locked_selection():
        calibration_key = _calibration_mode_key()
        if not lock_peak_selection.value:
            variables.clear_calibration_selection_mask(calibration_key)
            return
        if not _selected_peak_ready():
            variables.clear_calibration_selection_mask(calibration_key)
            return
        data = _get_calibration_array()
        mask = np.logical_and(data > variables.selected_x1, data < variables.selected_x2)
        if np.any(mask):
            variables.set_calibration_selection_mask(calibration_key, mask)

    def _sampling_mode_value():
        variables.bowl_sampling_mode = sampling_mode_b.value
        return sampling_mode_b.value

    def _state_is_valid(state):
        state = np.asarray(state, dtype=float)
        return state.size > 0 and np.all(np.isfinite(state)) and np.nanstd(state) > 0

    def _selected_peak_ready():
        return not (variables.selected_x1 == 0 and variables.selected_x2 == 0) and variables.selected_x2 > variables.selected_x1

    def _evaluate_mrp_values():
        if not _selected_peak_ready():
            return [float('nan')] * 3
        return fast_mrp(
            _get_calibration_array(),
            variables.selected_x1,
            variables.selected_x2,
            bin_size=max(1e-6, float(bin_size.value)),
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

    @staticmethod
    def _peaks_overlap(first_peak, second_peak):
        return not (first_peak['x2'] <= second_peak['x1'] or second_peak['x2'] <= first_peak['x1'])

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
        report = gaussian_mrp_report(
            _get_calibration_array(),
            peak['x1'],
            peak['x2'],
            bin_size=local_bin_size,
        )
        if report is None:
            return float('nan'), None

        mrp_values = report['gaussian_mrp'] if report['gaussian_ok'] else report['histogram_mrp']
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
            details.append({
                'label': peak.get('label', f'{peak["position"]:.2f}'),
                'score': float(score),
                'weight': weight,
                'position': float(peak['position']),
                'num_ions': int(peak.get('n_ions', report['num_ions'] if report is not None else 0)),
            })
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

    @staticmethod
    def _score_improved(candidate, best):
        if not np.isfinite(candidate):
            return False
        if not np.isfinite(best):
            return True
        return candidate > best + max(0.1, abs(best) * 0.002)

    @staticmethod
    def _score_not_worse(candidate, best, tolerance_ratio=0.01):
        if not np.isfinite(best):
            return True
        if not np.isfinite(candidate):
            return False
        return candidate >= best - max(0.1, abs(best) * tolerance_ratio)

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
        )

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
        )

        if calibration_mode.value == 'tof_calib':
            mask_temporal = np.logical_and((variables.dld_t_calib > variables.selected_x1), (variables.dld_t_calib < variables.selected_x2))
        else:
            mask_temporal = np.logical_and((variables.mc_calib > variables.selected_x1), (variables.mc_calib < variables.selected_x2))

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
                print_info=True,
                mrp_all=True,
                figure_size=figure_size,
                fast_calibration=False,
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
        )

    def _run_bowl_correction(plot_override=None, save_override=None):
        if not _selected_peak_ready():
            raise ValueError('Please first select a peak')

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
        )

    def _run_multi_peak_calibration():
        fit_v, peak_info_v = calibration.multi_peak_voltage_corr_main(
            _current_voltage(),
            variables,
            calibration_mode=_calibration_mode_key(),
            model=model_v.value,
            bin_size=bin_size_v.value,
            n_peaks=3,
            prominence=prominence.value,
            distance=distance.value,
        )
        print(f'Used {len(peak_info_v)} peaks for voltage correction:')
        for peak in peak_info_v:
            print(f'  Peak at {peak["position"]:.2f}, {peak["n_ions"]:,} ions')
        print('Voltage fit parameters:', fit_v)

        fit_b, n_peaks_b = calibration.multi_peak_bowl_corr_main(
            variables.dld_x_det,
            variables.dld_y_det,
            _current_voltage(),
            variables,
            det_diam,
            calibration_mode=_calibration_mode_key(),
            fit_mode=fit_mode_b.value,
            sample_size=max(1, int(sample_size_b.value)),
            bin_size=bin_size_b.value,
            n_peaks=3,
            prominence=prominence.value,
            distance=distance.value,
            sampling_mode=_sampling_mode_value(),
        )
        print(f'Used {n_peaks_b} peaks for bowl correction')
        print('Bowl fit parameters:', fit_b)

    def _run_joint_refinement():
        joint_model, peak_info = calibration.joint_voltage_bowl_corr_main(
            variables.dld_x_det,
            variables.dld_y_det,
            _current_voltage(),
            variables,
            det_diam,
            calibration_mode=_calibration_mode_key(),
            sample_size=max(1, int(sample_size_b.value)),
            bin_size=min(float(bin_size_v.value), float(bin_size_b.value)),
            n_peaks=4,
            prominence=prominence.value,
            distance=distance.value,
            sampling_mode=_sampling_mode_value(),
        )
        print(f'Used {len(peak_info)} peaks for joint voltage+bowl refinement')
        print('Joint model parameters:', joint_model['parameters'])

    def vol_correction(_, variables, output, status_output, calibration_mode_widget, pulse_mode_value):
        vol_button.disabled = True
        with status_output:
            status_output.clear_output()
            pb_vol.value = "<b>Starting...</b>"
            try:
                _prepare_locked_selection()
                print('Selected mc ranges are: (%s, %s)' % (variables.selected_x1, variables.selected_x2))
                print('----------------Voltage Calibration-------------------')
                _run_voltage_correction()
                print('Voltage calibration finished successfully.')
            except Exception as exc:
                print(f'Voltage calibration failed: {exc}')
            pb_vol.value = "<b>Finished</b>"
        vol_button.disabled = False

    def bowl_correction(_, variables, output, status_output, calibration_mode_widget, pulse_mode_value):
        bowl_button.disabled = True
        with status_output:
            status_output.clear_output()
            pb_bowl.value = "<b>Starting...</b>"
            try:
                _prepare_locked_selection()
                print('Selected mc ranges are: (%s, %s)' % (variables.selected_x1, variables.selected_x2))
                print('------------------Bowl Calibration---------------------')
                _run_bowl_correction()
                print('Bowl calibration finished successfully.')
            except Exception as exc:
                print(f'Bowl calibration failed: {exc}')
            pb_bowl.value = "<b>Finished</b>"
        bowl_button.disabled = False

    def initial_calibration(_, variables, calibration_mode_widget, flight_path_length_value):
        initial_calib_button.disabled = True
        with out:
            if calibration_mode_widget.value == 'tof_calib':
                variables.dld_t_calib = calibration.initial_calibration(variables.data, flight_path_length_value)
                print('Initial calibration is done')
            else:
                print('Initial mc calibration is not needed when calibrating directly the mc')
        initial_calib_button.disabled = False

    def stat_plot(_, variables, calibration_mode_widget, output):
        calibration_mode_t = 'tof' if calibration_mode_widget.value == 'tof_calib' else 'mc'
        with output:
            output.clear_output()
            calibration.plot_selected_statistic(variables, bin_fdm.value, index_fig.value, calibration_mode=calibration_mode_t, save=True)

    def _optimize_sequence(action_specs, title, figure_size, max_iterations=10, max_no_improve=3):
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
                    quality = _evaluate_quality(reference_peaks)
                    print(
                        f'After {action_name}: '
                        f'train={quality["train_score"]:.2f}, '
                        f'holdout={quality["holdout_score"]:.2f}, '
                        f'selected={quality["selected_score"]:.2f}'
                    )

                    has_valid_signal = (
                        np.isfinite(quality['train_score']) and quality['train_score'] > 0
                    ) or (
                        np.isfinite(quality['selected_score']) and quality['selected_score'] > 0
                    )
                    if not has_valid_signal:
                        _restore_state(before_state)
                        _restore_selection(before_selection)
                        print(f'{action_name} produced an invalid quality score; reverted this step.')
                        continue

                    train_improved = _score_improved(quality['train_score'], best_train_score)
                    selected_improved = _score_improved(quality['selected_score'], best_selected_score)
                    holdout_improved = _score_improved(quality['holdout_score'], best_holdout_score)
                    holdout_stable = _score_not_worse(quality['holdout_score'], best_holdout_score, tolerance_ratio=0.01)
                    train_stable = _score_not_worse(quality['train_score'], best_train_score, tolerance_ratio=0.01)

                    accepted = False
                    if train_improved and holdout_stable:
                        accepted = True
                    elif holdout_improved and train_stable:
                        accepted = True
                    elif not np.isfinite(best_train_score) and selected_improved and holdout_stable:
                        accepted = True

                    if accepted:
                        best_train_score = quality['train_score']
                        best_holdout_score = quality['holdout_score']
                        best_selected_score = quality['selected_score']
                        best_state = _capture_state()
                        best_selection = _capture_selection()
                        improved_this_round = True
                        print(
                            f'Accepted {action_name}; best scores are now '
                            f'train={best_train_score:.2f}, holdout={best_holdout_score:.2f}, '
                            f'selected={best_selected_score:.2f}'
                        )
                    else:
                        _restore_state(before_state)
                        _restore_selection(before_selection)
                        if reference_peaks['holdout'] and not holdout_stable:
                            print(f'{action_name} looked unstable on held-out peaks; reverted this step.')
                        else:
                            print(f'No stable improvement after {action_name}; reverted this step.')
                except Exception as exc:
                    _restore_state(before_state)
                    _restore_selection(before_selection)
                    print(f'{action_name} failed: {exc}')

            if improved_this_round:
                no_improve_count = 0
            else:
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
        with status_output:
            status_output.clear_output()
            _prepare_locked_selection()
            _optimize_sequence(
                [('Bowl correction', lambda: _run_bowl_correction(plot_override=False, save_override=False))],
                title='Auto bowl calibration',
                figure_size=(figure_b_size_x.value, figure_b_size_y.value),
                max_iterations=10,
                max_no_improve=3,
            )
        index_fig_v.value = 1
        index_fig_b.value = 1
        auto_button_bowl.disabled = False

    def automatic_calibration(_, variables, output, status_output, calibration_mode_widget, pulse_mode_value):
        auto_button.disabled = True
        with status_output:
            status_output.clear_output()
            _prepare_locked_selection()
            _optimize_sequence(
                [
                    ('Voltage correction', lambda: _run_voltage_correction(plot_override=False, save_override=False)),
                    ('Bowl correction', lambda: _run_bowl_correction(plot_override=False, save_override=False)),
                    ('Joint voltage+bowl refinement', _run_joint_refinement),
                ],
                title='Auto calibration',
                figure_size=(figure_mc_size_x.value, figure_mc_size_y.value),
                max_iterations=10,
                max_no_improve=3,
            )
        index_fig_v.value = 1
        index_fig_b.value = 1
        auto_button.disabled = False

    def on_gaussian_mrp(_):
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
                    bin_size=0.01,
                )
                if result is None:
                    print('Gaussian MRP: insufficient data in selected range')
                else:
                    print('=' * 60)
                    print('PEAK PROFILE MRP REPORT')
                    print('=' * 60)
                    print(f'Peak position: {result["peak_position"]:.4f}')
                    print(f'Ions in range: {result["num_ions"]:,}')
                    print(f'Bin size used: {result["bin_size"]} ({result["num_bins"]} bins)')
                    print()
                    if result['gaussian_ok']:
                        print('Gaussian fit MRP (sub-bin accuracy):')
                        print(f'  MRP(0.5)  = {result["gaussian_mrp"][0]:.2f}')
                        print(f'  MRP(0.1)  = {result["gaussian_mrp"][1]:.2f}')
                        print(f'  MRP(0.01) = {result["gaussian_mrp"][2]:.2f}')
                    else:
                        print('Gaussian fit FAILED')
                    print()
                    if result['voigt_ok']:
                        print(f'Voigt fit MRP ({result["profile_type"]}):')
                        print(f'  MRP(0.5)  = {result["voigt_mrp"][0]:.2f}')
                        print(f'  MRP(0.1)  = {result["voigt_mrp"][1]:.2f}')
                        print(f'  MRP(0.01) = {result["voigt_mrp"][2]:.2f}')
                        print(f'  Voigt FWHM = {result["voigt_fwhm"]:.6f}')
                    else:
                        print('Voigt fit FAILED')
                    print()
                    print('Histogram-based MRP (for comparison):')
                    print(f'  MRP(0.5)  = {result["histogram_mrp"][0]:.2f}')
                    print(f'  MRP(0.1)  = {result["histogram_mrp"][1]:.2f}')
                    print(f'  MRP(0.01) = {result["histogram_mrp"][2]:.2f}')
                    print('=' * 60)
        gaussian_mrp_button.disabled = False

    def on_multi_peak_calibrate(_):
        multi_peak_button.disabled = True
        with out_status:
            out_status.clear_output()
            print('=' * 60)
            print('MULTI-PEAK CALIBRATION')
            print('=' * 60)
            try:
                start_time = time.perf_counter()
                _run_multi_peak_calibration()
                runtime_s = time.perf_counter() - start_time
                print(f'Total runtime: {runtime_s:.2f} s')
                print('MRP after multi-peak calibration:', _evaluate_mrp_values())
                print('=' * 60)
                print('Multi-peak calibration complete!')
            except Exception as exc:
                print(f'Multi-peak calibration error: {exc}')
                print('=' * 60)
        multi_peak_button.disabled = False

    def on_auto_optimize(_):
        auto_optimize_button.disabled = True
        with out_status:
            out_status.clear_output()
            _prepare_locked_selection()
            _optimize_sequence(
                [
                    ('Voltage correction', lambda: _run_voltage_correction(plot_override=False, save_override=False)),
                    ('Bowl correction', lambda: _run_bowl_correction(plot_override=False, save_override=False)),
                    ('Multi-peak calibration', _run_multi_peak_calibration),
                    ('Joint voltage+bowl refinement', _run_joint_refinement),
                ],
                title='Auto optimize MRP',
                figure_size=(figure_mc_size_x.value, figure_mc_size_y.value),
                max_iterations=20,
                max_no_improve=3,
            )
        auto_optimize_button.disabled = False

    plot_button.on_click(lambda b: hist_plot(b, variables, out, calibration_mode))
    plot_stat_button.on_click(lambda b: stat_plot(b, variables, calibration_mode, out))
    reset_back_button.on_click(lambda b: reset_back_on_click(variables, calibration_mode))
    reset_button.on_click(lambda b: reset_on_click(variables, calibration_mode))
    save_button.on_click(lambda b: save_on_click(variables, calibration_mode))
    vol_button.on_click(lambda b: vol_correction(b, variables, out, out_status, calibration_mode, pulse_mode))
    bowl_button.on_click(lambda b: bowl_correction(b, variables, out, out_status, calibration_mode, pulse_mode))
    clear_plot.on_click(lambda b: clear_plot_on_click(out, out_status))
    auto_button.on_click(lambda b: automatic_calibration(b, variables, out, out_status, calibration_mode, pulse_mode))
    auto_button_bowl.on_click(lambda b: automatic_bowl_calibration(b, variables, out, out_status, calibration_mode, pulse_mode))
    initial_calib_button.on_click(lambda b: initial_calibration(b, variables, calibration_mode, flight_path_length))
    gaussian_mrp_button.on_click(on_gaussian_mrp)
    multi_peak_button.on_click(on_multi_peak_calibrate)
    auto_optimize_button.on_click(on_auto_optimize)
    sampling_mode_b.observe(
        lambda change: setattr(variables, 'bowl_sampling_mode', change['new']),
        names='value',
    )

    plot_button.click()
    sample_size_v_set(sample_size_v)

    column11 = widgets.VBox([
        bin_size,
        lim_tof,
        prominence,
        distance,
        percent,
        bin_fdm,
        plot_peak,
        index_fig,
        save,
        figure_mc_size_x,
        figure_mc_size_y,
    ])
    column12 = widgets.VBox([
        plot_button,
        save_button,
        reset_back_button,
        reset_button,
        clear_plot,
        gaussian_mrp_button,
        plot_stat_button,
    ])
    column22 = widgets.VBox([
        sample_size_b,
        sample_size_b_help,
        bin_size_b,
        fit_mode_b,
        sampling_mode_b,
        maximum_cal_method_b,
        maximum_sample_method_b,
        plot_b,
        index_fig_b,
        save_b,
        figure_b_size_x,
        figure_b_size_y,
    ])
    column21 = widgets.VBox([bowl_button, pb_bowl])
    column33 = widgets.VBox([
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
    ])
    column32 = widgets.VBox([vol_button, pb_vol])
    column34 = widgets.VBox([calibration_mode, fast_calibration, automatic_window_update, lock_peak_selection, peak_val])

    layout1 = widgets.HBox([column11, column22, column33, column34])
    layout2 = widgets.HBox([column12, column21, column32])
    layout3 = widgets.HBox([
        initial_calib_button,
        auto_button,
        auto_button_bowl,
        multi_peak_button,
    ])
    layout4 = widgets.HBox([auto_optimize_button])
    layout = widgets.VBox([layout1, layout2, layout3, layout4])

    display(layout)
    display(widgets.VBox([out, out_status]))
