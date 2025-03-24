import ipywidgets as widgets
import numpy as np
from IPython.display import display
from ipywidgets import Output
from IPython.display import clear_output

from pyccapt.calibration.calibration import calibration, mc_plot

# Define a layout for labels to make them a fixed width
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

    plot_button = widgets.Button(
        description='plot hist',
        layout=label_layout
    )
    plot_stat_button = widgets.Button(
        description='plot stat',
        layout=label_layout
    )
    reset_back_button = widgets.Button(
        description='back to saved',
        layout=label_layout
    )
    reset_button = widgets.Button(
        description='reset correction',
        layout=label_layout
    )
    save_button = widgets.Button(
        description='save correction',
        layout=label_layout
    )
    bowl_button = widgets.Button(
        description='bowl correction',
        layout=label_layout
    )
    vol_button = widgets.Button(
        description='voltage correction',
        layout=label_layout
    )

    auto_button = widgets.Button(
        description='auto calibration',
        layout=label_layout
    )
    auto_button_bowl = widgets.Button(
        description='auto bowl calibration',
        layout=label_layout
    )
    initial_calib_button = widgets.Button(
        description='initial calibration',
        layout=label_layout
    )
    calibration_mode = widgets.Dropdown(
        options=[('mass_to_charge', 'mc_calib'), ('time_of_flight', 'tof_calib')],
        description='calibration mode:')

    def sample_size_v_set(sample_size_v):
        # Create a button widget to voltage correction function
        if calibration_mode.value == 'tof_calib':
            lim_tof.value = variables.max_tof
        elif calibration_mode.value == 'mc_calib':
            lim_tof.value = 400
        mc_plot.hist_plot(variables, bin_size.value, log=True, target=calibration_mode.value, normalize=False,
                          prominence=prominence.value, distance=distance.value, percent=percent.value,
                          selector='rect', figname=index_fig.value, lim=lim_tof.value, save_fig=save.value,
                          peaks_find_plot=plot_peak.value, draw_calib_rect=True, print_info=False, mrp_all=False,
                          figure_size=(figure_mc_size_x.value, figure_mc_size_y.value), plot_show=False)

        if calibration_mode.value == 'tof_calib':
            mask_temporal = np.logical_and(
                (variables.dld_t_calib > variables.selected_x1),
                (variables.dld_t_calib < variables.selected_x2)
            )
        elif calibration_mode.value == 'mc_calib':
            mask_temporal = np.logical_and(
                (variables.mc_calib > variables.selected_x1),
                (variables.mc_calib < variables.selected_x2)

            )
        sample_size = int(len(variables.dld_high_voltage[mask_temporal]) / 100)
        sample_size_v.value = sample_size

    calibration_mode.observe(lambda change: sample_size_v_set(sample_size_v), names='value')

    clear_plot = widgets.Button(description="clear plots", layout=label_layout)

    plot_button.on_click(lambda b: hist_plot(b, variables, out, calibration_mode))
    plot_stat_button.on_click(lambda b: stat_plot(b, variables, calibration_mode, out))
    reset_back_button.on_click(lambda b: reset_back_on_click(variables, calibration_mode))
    reset_button.on_click(lambda b: reset_on_click(variables, calibration_mode))
    save_button.on_click(lambda b: save_on_click(variables, calibration_mode))
    vol_button.on_click(lambda b: vol_correction(b, variables, out, out_status, calibration_mode, pulse_mode))
    bowl_button.on_click(lambda b: bowl_correction(b, variables, out, out_status, calibration_mode, pulse_mode))
    clear_plot.on_click(lambda b: clear_plot_on_click(out, out_status))
    auto_button.on_click(lambda b: automatic_calibration(b, variables, out, out_status, calibration_mode, pulse_mode))
    auto_button_bowl.on_click(lambda b: automatic_bowl_calibration(b, variables, out, out_status, calibration_mode,
                                                                   pulse_mode))
    initial_calib_button.on_click(lambda b: initial_calibration(b, variables, calibration_mode, flight_path_length))

    # Define widgets and labels for histplot function
    bin_size = widgets.FloatText(value=0.1, description='bin size:', layout=label_layout)
    prominence = widgets.IntText(value=100, description='peak prominance:', layout=label_layout)
    distance = widgets.IntText(value=500, description='peak distance:', layout=label_layout)
    lim_tof = widgets.IntText(value=variables.max_tof, description='lim tof/mc:', layout=label_layout)
    percent = widgets.IntText(value=50, description='percent MRP:', layout=label_layout)
    index_fig = widgets.IntText(value=1, description='fig save index:', layout=label_layout)
    plot_peak = widgets.Dropdown(
        options=[('True', True), ('False', False)],
        description='plot peak',
        layout=label_layout
    )
    save = widgets.Dropdown(
        options=[('False', False), ('True', True)],
        description='save fig:',
        layout=label_layout
    )
    figure_mc_size_x = widgets.FloatText(value=9.0, description="Fig. size W:", layout=label_layout)
    figure_mc_size_y = widgets.FloatText(value=5.0, description="Fig. size H:", layout=label_layout)

    def hist_plot(b, variables, out, calibration_mode):
        plot_button.disabled = True
        figure_size = (figure_mc_size_x.value, figure_mc_size_y.value)
        clear_output(wait=True)
        with out_status:
            out_status.clear_output()
        with out:
            out.clear_output()
            mc_plot.hist_plot(variables, bin_size.value, log=True, target=calibration_mode.value, normalize=False,
                              prominence=prominence.value, distance=distance.value, percent=percent.value,
                              selector='rect', figname=index_fig.value, lim=lim_tof.value, save_fig=save.value,
                              peaks_find_plot=plot_peak.value, draw_calib_rect=True, print_info=True, mrp_all=True,
                              figure_size=figure_size, fast_calibration=False)
        plot_button.disabled = False

    plot_button.click()

    sample_size_v = widgets.IntText(value=10000, description='sample size:', layout=label_layout)
    sample_size_v_set(sample_size_v)


    index_fig_v = widgets.IntText(value=1, description='fig index:', layout=label_layout)
    plot_v = widgets.Dropdown(
        options=[('False', False), ('True', True)],
        description='plot fig:',
        layout=label_layout
    )
    save_v = widgets.Dropdown(
        options=[('False', False), ('True', True)],
        description='save fig:',
        layout=label_layout
    )
    mode_v = widgets.Dropdown(
        options=[('ion_seq', 'ion_seq'), ('voltage', 'voltage')],
        description='sample mode:',
        layout=label_layout
    )
    maximum_cal_method_v = widgets.Dropdown(
        options=[('mean', 'mean'), ('histogram', 'histogram'), ('median', 'median')],
        description='peak max:',
        layout=label_layout
    )
    model_v = widgets.Dropdown(
        options=[('robust_fit', 'robust_fit'), ('curve_fit', 'curve_fit')],
        description='fit mode:',
        layout=label_layout
    )
    maximum_sample_method_v = widgets.Dropdown(
        options=[('histogram', 'histogram'), ('mean', 'mean'), ('median', 'median')],
        description='sample max:',
        layout=label_layout
    )

    bin_size_v = widgets.FloatText(value=0.01, description='bin size:', layout=label_layout)

    figure_v_size_x = widgets.FloatText(value=5.0, description="Fig. size W:", layout=label_layout)
    figure_v_size_y = widgets.FloatText(value=5.0, description="Fig. size H:", layout=label_layout)

    def vol_correction(b, variables, out, out_status, calibration_mode, pulse_mode):
        vol_button.disabled = True
        with out_status:
            pb_vol.value = "<b>Starting...</b>"
            if variables.selected_x1 == 0 or variables.selected_x2 == 0:
                print('Please first select a peak')
            else:
                print('Selected mc ranges are: (%s, %s)' % (variables.selected_x1, variables.selected_x2))
                print('----------------Voltage Calibration-------------------')
                figure_size = (figure_v_size_x.value, figure_v_size_y.value)
                sample_size_p = sample_size_v.value
                index_fig_p = index_fig_v.value
                plot_p = plot_v.value
                save_p = save_v.value
                mode_p = mode_v.value
                maximum_cal_method_p = maximum_cal_method_v.value
                maximum_sample_method_p = maximum_sample_method_v.value
                if calibration_mode.value == 'tof_calib':
                    calibration_mode_t = 'tof'
                elif calibration_mode.value == 'mc_calib':
                    calibration_mode_t = 'mc'
                if pulse_mode == 'voltage':
                    voltage = variables.dld_high_voltage + (0.7 * variables.dld_pulse_v)
                elif pulse_mode == 'laser':
                    voltage = variables.dld_high_voltage

                calibration.voltage_corr_main(voltage, variables, sample_size=sample_size_p,
                                              calibration_mode=calibration_mode_t,
                                              index_fig=index_fig_p, plot=plot_p, save=save_p,
                                              mode=mode_p,
                                              maximum_cal_method=maximum_cal_method_p,
                                              maximum_sample_method=maximum_sample_method_p,
                                              fig_size=figure_size, fast_calibration=fast_calibration.value,
                                              model=model_v.value, bin_size=bin_size_v.value,
                                              peak_maximum=peak_val.value)
            pb_vol.value = "<b>Finished</b>"
        vol_button.disabled = False

    sample_size_b = widgets.IntText(value=5, description='sample size:', layout=label_layout)
    fit_mode_b = widgets.Dropdown(options=[('robust_fit', 'robust_fit'), ('curve_fit', 'curve_fit')],
                                  description='fit mode:', layout=label_layout)
    index_fig_b = widgets.IntText(value=1, description='fig index:', layout=label_layout)
    bin_size_b = widgets.FloatText(value=0.01, description='bin size:', layout=label_layout)
    maximum_cal_method_b = widgets.Dropdown(
        options=[('mean', 'mean'), ('histogram', 'histogram')],
        description='peak max:',
        layout=label_layout
    )
    maximum_sample_method_b = widgets.Dropdown(
        options=[('histogram', 'histogram'), ('mean', 'mean')],
        description='sample max:',
        layout=label_layout
    )
    plot_b = widgets.Dropdown(
        options=[('False', False), ('True', True)],
        description='plot fig:',
        layout=label_layout
    )

    save_b = widgets.Dropdown(
        options=[('False', False), ('True', True)],
        description='save fig:',
        layout=label_layout
    )
    fast_calibration = widgets.Dropdown(
        options=[('False', False), ('True', True)],
        description='fast calibration:',
        layout=label_layout
    )
    automatic_window_update = widgets.Dropdown(
        options=[('False', False), ('True', True)],
        description='auto window update:',
        layout=label_layout
    )
    peak_val = widgets.FloatText(value=0, description='peak value:', layout=label_layout)


    figure_b_size_x = widgets.FloatText(value=5.0, description="Fig. size W:", layout=label_layout)
    figure_b_size_y = widgets.FloatText(value=5.0, description="Fig. size H:", layout=label_layout)

    def initial_calibration(b, variables, calibration_mode, flight_path_length):
        initial_calib_button.disabled = True
        with out:
            if calibration_mode.value == 'tof_calib':
                variables.dld_t_calib = calibration.initial_calibration(variables.data, flight_path_length)
                print('Initial calibration is done')

            elif calibration_mode.value == 'mc_calib':
                print('Initial mc calibration is not needed when calibrating directly the mc')
        initial_calib_button.disabled = False

    def bowl_correction(b, variables, out, out_status, calibration_mode, pulse_mode):
        bowl_button.disabled = True
        with out_status:
            pb_bowl.value = "<b>Starting...</b>"
            if variables.selected_x1 == 0 or variables.selected_x2 == 0:
                print('Please first select a peak')
            else:
                print('Selected mc ranges are: (%s, %s)' % (variables.selected_x1, variables.selected_x2))
                sample_size_p = sample_size_b.value
                fit_mode_p = fit_mode_b.value
                index_fig_p = index_fig_b.value
                plot_p = plot_b.value
                save_p = save_b.value
                maximum_cal_method_p = maximum_cal_method_b.value
                maximum_sample_method_p = maximum_sample_method_b.value
                figure_size = (figure_b_size_x.value, figure_b_size_y.value)
                if calibration_mode.value == 'tof_calib':
                    calibration_mode_t = 'tof'
                elif calibration_mode.value == 'mc_calib':
                    calibration_mode_t = 'mc'
                if pulse_mode == 'voltage':
                    voltage = variables.dld_high_voltage + (0.7 * variables.dld_pulse_v)
                elif pulse_mode == 'laser':
                    voltage = variables.dld_high_voltage
                print('------------------Bowl Calibration---------------------')
                calibration.bowl_correction_main(variables.dld_x_det, variables.dld_y_det,
                                                 voltage,
                                                 variables, det_diam,
                                                 sample_size=sample_size_p, fit_mode=fit_mode_p,
                                                 maximum_cal_method=maximum_cal_method_p,
                                                 maximum_sample_method=maximum_sample_method_p,
                                                 fig_size=figure_size,
                                                 calibration_mode=calibration_mode_t, index_fig=index_fig_p,
                                                 plot=plot_p, save=save_p, fast_calibration=fast_calibration.value,
                                                 bin_size=bin_size_b.value,
                                                 peak_maximum=peak_val.value)

            pb_bowl.value = "<b>Finished</b>"
        bowl_button.disabled = False

    def stat_plot(b, variables, calibration_mode, out):
        if calibration_mode.value == 'tof_calib':
            calibration_mode_t = 'tof'
        elif calibration_mode.value == 'mc_calib':
            calibration_mode_t = 'mc'
        with out:
            out.clear_output()
            calibration.plot_selected_statistic(variables, bin_fdm.value, index_fig.value,
                                                calibration_mode=calibration_mode_t, save=True)


    def automatic_bowl_calibration(b, variables, out, out_status, calibration_mode, pulse_mode):
        auto_button_bowl.disabled = True
        counter = 1
        try_counter = 0
        index_fig_val = index_fig.value
        continue_calibration = True
        figure_size = (figure_b_size_x.value, figure_b_size_y.value)
        mrp_last = 0
        if calibration_mode.value == 'tof_calib':
            back_tof_mc = np.copy(variables.dld_t_calib)
        elif calibration_mode.value == 'mc_calib':
            back_tof_mc = np.copy(variables.mc_calib)
        while continue_calibration:
            with out_status:
                print('=======================================================')
                print('Starting calibration number %s' % counter)
                mrp = mc_plot.hist_plot(variables, bin_size.value, log=True, target=calibration_mode.value,
                                        normalize=False,
                                        prominence=prominence.value, distance=distance.value, percent=percent.value,
                                        selector='rect', figname=index_fig_val, lim=lim_tof.value, save_fig=save.value,
                                        peaks_find_plot=plot_peak.value, print_info=False, figure_size=figure_size,
                                        plot_show=False, fast_calibration=fast_calibration.value,
                                        draw_calib_rect=automatic_window_update.value, mrp_all=True)
                print('The MRPs at (0.5, 0.1, 0.01) are:', mrp)

                counter += 1

                if mrp_last - mrp[0] > 0:
                    try_counter += 1
                    if try_counter == 2:
                        print('*********************************************************')
                        print('Calibration is not improving, stopping', try_counter)
                        print('*********************************************************')
                        continue_calibration = False
                        break

                elif mrp_last < mrp[0]:
                    if calibration_mode.value == 'tof_calib':
                        back_tof_mc = np.copy(variables.dld_t_calib)
                    elif calibration_mode.value == 'mc_calib':
                        back_tof_mc = np.copy(variables.mc_calib)
                    mrp_last = mrp[0]
                    try_counter = 0
                if counter > 10:
                    print('*********************************************************')
                    print('Calibration is stopped at iteration', counter)
                    print('*********************************************************')
                    continue_calibration = False
                    break

                index_fig_v.value = index_fig_val
                index_fig_b.value = index_fig_val
                bowl_correction(b, variables, out, out_status, calibration_mode, pulse_mode)
                index_fig_val += 1

        if calibration_mode.value == 'tof_calib':
            variables.dld_t_calib = np.copy(back_tof_mc)
        elif calibration_mode.value == 'mc_calib':
            variables.mc_calib = np.copy(back_tof_mc)
        index_fig_v.value = 1
        index_fig_b.value = 1
        auto_button_bowl.disabled = False

    def automatic_calibration(b, variables, out, out_status, calibration_mode, pulse_mode):

        auto_button.disabled = True
        counter = 1
        try_counter = 0
        index_fig_val = index_fig.value
        continue_calibration = True
        figure_size = (figure_mc_size_x.value, figure_mc_size_y.value)
        mrp_last = 0
        if calibration_mode.value == 'tof_calib':
            back_tof_mc = np.copy(variables.dld_t_calib)
        elif calibration_mode.value == 'mc_calib':
            back_tof_mc = np.copy(variables.mc_calib)
        while continue_calibration:
            with out_status:
                print('=======================================================')
                print('Starting calibration number %s' % counter)

                mrp = mc_plot.hist_plot(variables, bin_size.value, log=True, target=calibration_mode.value,
                                        normalize=False,
                                  prominence=prominence.value, distance=distance.value, percent=percent.value,
                                        selector='rect', figname=index_fig_val, lim=lim_tof.value, save_fig=save.value,
                                  peaks_find_plot=plot_peak.value, print_info=False, figure_size=figure_size,
                                  plot_show=False, fast_calibration=fast_calibration.value,
                                        draw_calib_rect=automatic_window_update.value, mrp_all=True)
                print('The MRPs at (0.5, 0.1, 0.01) are:', mrp)
                counter += 1

                if mrp_last - mrp[0] > 0:
                    try_counter += 1
                    if try_counter == 2:
                        print('*********************************************************')
                        print('Calibration is not improving, stopping')
                        print('*********************************************************')
                        continue_calibration = False
                        break
                elif mrp_last < mrp[0]:
                    if calibration_mode.value == 'tof_calib':
                        back_tof_mc = np.copy(variables.dld_t_calib)
                    elif calibration_mode.value == 'mc_calib':
                        back_tof_mc = np.copy(variables.mc_calib)
                    mrp_last = mrp[0]
                    try_counter = 0

                if counter > 10:
                    print('*********************************************************')
                    print('Calibration is stopped at iteration', counter)
                    print('*********************************************************')
                    continue_calibration = False
                    break

                index_fig_v.value = index_fig_val
                index_fig_b.value = index_fig_val
                vol_correction(b, variables, out, out_status, calibration_mode, pulse_mode)
                # mrp = mc_plot.hist_plot(variables, bin_size.value, log=True, target=calibration_mode.value,
                #                         normalize=False,
                #                         prominence=prominence.value, distance=distance.value, percent=percent.value,
                #                         selector='rect', figname=index_fig_val, lim=lim_tof.value, save_fig=save.value,
                #                         peaks_find_plot=plot_peak.value, print_info=False, figure_size=figure_size,
                #                         plot_show=False, fast_calibration=fast_calibration.value,
                #                         draw_calib_rect=automatic_window_update.value, mrp_all=True)
                bowl_correction(b, variables, out, out_status, calibration_mode, pulse_mode)
                index_fig_val += 1

        if calibration_mode.value == 'tof_calib':
            variables.dld_t_calib = np.copy(back_tof_mc)
        elif calibration_mode.value == 'mc_calib':
            variables.mc_calib = np.copy(back_tof_mc)
        index_fig_v.value = 1
        index_fig_b.value = 1
        auto_button.disabled = False


    # Create a button widget to trigger the function
    pb_bowl = widgets.HTML(
        value=" ",
        placeholder='Status:',
        description='Status:',
        layout=label_layout
    )
    pb_vol = widgets.HTML(
        value=" ",
        placeholder='Status:',
        description='Status:',
        layout=label_layout
    )

    bin_fdm = widgets.IntText(value=256, description='bin FDM:', layout=label_layout)




    # Create the layout with three columns
    column11 = widgets.VBox([bin_size, lim_tof, prominence, distance, percent, bin_fdm, plot_peak, index_fig, save,
                             figure_mc_size_x, figure_mc_size_y])
    column12 = widgets.VBox([plot_button, auto_button, auto_button_bowl, initial_calib_button, save_button,
                             reset_back_button, reset_button, clear_plot, plot_stat_button])
    column22 = widgets.VBox([sample_size_b, bin_size_b, fit_mode_b, maximum_cal_method_b, maximum_sample_method_b,
                             plot_b,
                             index_fig_b, save_b, figure_b_size_x, figure_b_size_y])
    column21 = widgets.VBox([bowl_button, pb_bowl])
    column33 = widgets.VBox([sample_size_v, bin_size_v, model_v, maximum_cal_method_v, maximum_sample_method_v,
                             mode_v, plot_v, index_fig_v, save_v, figure_v_size_x, figure_v_size_y])
    column32 = widgets.VBox([vol_button, pb_vol])
    column34 = widgets.VBox([calibration_mode, fast_calibration, automatic_window_update, peak_val])

    # Create the overall layout by arranging the columns side by side
    layout1 = widgets.HBox([column11, column22, column33, column34])
    layout2 = widgets.HBox([column12, column21, column32])

    layout = widgets.VBox([layout1, layout2])

    # Display the layout
    display(layout)

    out_put_layout = widgets.VBox([out, out_status])
    display(out_put_layout)