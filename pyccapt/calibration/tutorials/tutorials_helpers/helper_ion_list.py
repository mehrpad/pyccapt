import functools

import ipywidgets as widgets
import numpy as np
from IPython.display import clear_output, display
from ipywidgets import Output
from scipy.optimize import curve_fit

from pyccapt.calibration.core import mc_plot, widgets as wd
from pyccapt.calibration.data_tools import data_tools

# Define a layout for labels to make them a fixed width
label_layout = widgets.Layout(width='200px')


def call_ion_list(variables, selector, path='../../../files/'):
    try:
        isotopeTableFile = path + 'isotopeTable.h5'
        dataframe = data_tools.read_range(isotopeTableFile)
    except Exception as e:
        print(f"Error in loading the isotopeTable file: {e}")
        print("Trying to load the isotopeTable file from the pyccapt package")
        try:
            isotopeTableFile = './pyccapt/files/isotopeTable.h5'
            dataframe = data_tools.read_range(isotopeTableFile)
        except Exception as e:
            print(f"Error in loading the isotopeTable file: {e}")

    elementsList = dataframe['element']
    elementIsotopeList = dataframe['isotope']
    elementMassList = dataframe['weight']
    abundanceList = dataframe['abundance']


    elements = list(zip(elementsList, elementIsotopeList, elementMassList, abundanceList))
    dropdownList = []
    for element in elements:
        tupleElement = ("{} ({}) ({:.2f})".format(element[0], element[1], element[3]),
                        "{}({})[{}]".format(element[0], element[1], element[2]))
        dropdownList.append(tupleElement)

    chargeList = [(1, 1,), (2, 2,), (3, 3,), (4, 4,)]
    dropdown = wd.dropdownWidget(dropdownList, "Elements")
    dropdown.observe(wd.on_change)

    chargeDropdown = wd.dropdownWidget(chargeList, "Charge")
    chargeDropdown.observe(wd.on_change_charge)

    wd.compute_element_isotope_values_according_to_selected_charge()

    buttonAdd = wd.buttonWidget("ADD")
    buttonDelete = wd.buttonWidget("DELETE")
    buttonReset = wd.buttonWidget("RESET")

    def buttonAdd_f(b, variables):
        with out_ion_list:
            clear_output(True)
            wd.onClickAdd(b, variables)
            display()

    def buttonDelete_f(b, variables):
        with out_ion_list:
            clear_output(True)
            wd.onClickDelete(b, variables)
            display()

    def buttonResett_f(b, variables):
        with out_ion_list:
            clear_output(True)
            wd.onClickReset(b, variables)
            display()

    buttonAdd.on_click(functools.partial(buttonAdd_f, variables=variables))
    buttonDelete.on_click(functools.partial(buttonDelete_f, variables=variables))
    buttonReset.on_click(functools.partial(buttonResett_f, variables=variables))

    out_ion_list = Output()
    out_mc = Output()

    # Define widgets for fine_tune_t_0 function
    bin_size_widget = widgets.FloatText(value=0.1)
    log_widget = widgets.Dropdown(options=[('True', True), ('False', False)])
    mode_widget = widgets.Dropdown(options=[('False', False), ('True', True)])
    prominence_widget = widgets.IntText(value=80)
    distance_widget = widgets.IntText(value=100)
    lim_widget = widgets.IntText(value=10000)
    percent_widget = widgets.IntText(value=50)
    figname_widget = widgets.Text(value='hist')
    figure_mc_size_x = widgets.FloatText(value=9.0)
    figure_mc_size_y = widgets.FloatText(value=5.0)

    # Create a button widget to trigger the function
    button_plot = widgets.Button(description="plot")
    reset_back_button = widgets.Button(
        description='reset back correction',
        layout=label_layout
    )
    button_fit = widgets.Button(description="fit")
    calibration_mode = widgets.Dropdown(
        options=[('mass_to_charge', 'mc_calib'), ('time_of_flight', 'tof_calib')])

    def parametric_fit(variables, calibration_mode, out_mc):

        button_fit.disabled = True
        peaks_chos = np.array(variables.peaks_x_selected)

        if len(peaks_chos) != len(variables.list_material):
            with out_mc:
                print('Number of peaks and number of materials are not equal')
        else:
            if calibration_mode.value == 'tof_calib':
                def parametric(t, t0, c, d):
                    return c * ((t - t0) ** 2) + d * t

                def parametric_calib(t, mc_ideal):
                    fitresult, _ = curve_fit(parametric, t, mc_ideal, maxfev=2000)
                    return fitresult
                if len(peaks_chos) > 2:
                    fitresult = parametric_calib(peaks_chos, variables.list_material)

                    variables.mc_calib = parametric(variables.dld_t_calib, *fitresult)
                else:
                    print('Number of peaks is less than 3. Select more peaks at least 3 peaks')

            elif calibration_mode.value == 'mc_calib':
                def shift_3(mc, a, b, c):
                    return mc ** a + b * mc + c
                    # return a * mc + b
                def shift_calib_3(mc, mc_ideal):
                    fitresult, _ = curve_fit(shift_3, mc, mc_ideal, maxfev=2000)
                    return fitresult
                def shift_2(mc, a, b):
                    return mc ** a + b
                def shift_calib_2(mc, mc_ideal):
                    fitresult, _ = curve_fit(shift_2, mc, mc_ideal, maxfev=2000)
                    return fitresult
                if len(peaks_chos) > 2:
                    fitresult = shift_calib_3(peaks_chos, variables.list_material)
                    variables.mc_calib = shift_3(variables.mc_calib_backup, *fitresult)
                elif len(peaks_chos) == 2:
                    fitresult = shift_calib_2(peaks_chos, variables.list_material)
                    variables.mc_calib = shift_2(variables.mc_calib_backup, *fitresult)
                else:
                    print('Number of peaks is less than 2. Select more peaks at least 2 peaks')

            with out_mc:
                print('parametric fit done')
        button_fit.disabled = False

    button_plot_result = widgets.Button(description="plot result")

    def plot_fit_result(b, variables, calibration_mode, out_mc):
        button_plot_result.disabled = True
        # Get the values from the widgets
        bin_size_value = bin_size_widget.value
        log_value = log_widget.value
        mode_value = mode_widget.value
        target_value = 'mc_c'
        prominence_value = prominence_widget.value
        distance_value = distance_widget.value
        percent_value = percent_widget.value
        figname_value = figname_widget.value
        lim_value = lim_widget.value
        figure_size = (figure_mc_size_x.value, figure_mc_size_y.value)
        with out_mc:  # Capture the output within the 'out' widget
            # Call the function
            mc_hist = mc_plot.AptHistPlotter(variables.mc_calib[variables.mc_calib < lim_value], variables)
            mc_hist.plot_histogram(bin_width=bin_size_value, normalize=mode_value, label='mc', steps='stepfilled',
                                   log=log_value, fig_size=figure_size)

            if mode_value != 'normalized':
                mc_hist.find_peaks_and_widths(prominence=prominence_value, distance=distance_value,
                                              percent=percent_value)
                mc_hist.plot_peaks()
                mc_hist.plot_hist_info_legend(label=target_value, background=None, loc='right')

            mc_hist.save_fig(label=target_value, fig_name=figname_value)

        # Enable the button when the code is finished
        button_plot_result.disabled = False

    def on_button_click(b, variables, selector):
        # Disable the button while the code is running
        button_plot.disabled = True
        variables.peaks_x_selected = []
        # Get the values from the widgets
        bin_size_value = bin_size_widget.value
        log_value = log_widget.value
        mode_value = mode_widget.value
        target_value = calibration_mode.value
        prominence_value = prominence_widget.value
        distance_value = distance_widget.value
        percent_value = percent_widget.value
        figname_value = figname_widget.value
        lim_value = lim_widget.value
        figure_size = (figure_mc_size_x.value, figure_mc_size_y.value)
        with out_mc:  # Capture the output within the 'out' widget
            out_mc.clear_output()  # Clear any previous output
            # Call the function
            if target_value == 'mc_calib':
                mc_hist = mc_plot.AptHistPlotter(variables.mc_calib[variables.mc_calib < lim_value], variables)
                mc_hist.plot_histogram(bin_width=bin_size_value, normalize=mode_value, label='mc', steps='stepfilled',
                                       log=log_value, fig_size=figure_size)
            elif target_value == 'tof_calib':
                mc_hist = mc_plot.AptHistPlotter(variables.dld_t_calib[variables.dld_t_calib < lim_value], variables)
                mc_hist.plot_histogram(bin_width=bin_size_value, normalize=mode_value, label='tof', steps='stepfilled',
                                       log=log_value, fig_size=figure_size)

            if not mode_value:
                mc_hist.find_peaks_and_widths(prominence=prominence_value, distance=distance_value,
                                              percent=percent_value)
                mc_hist.plot_peaks()
                mc_hist.plot_hist_info_legend(label='mc', background=None, loc='right')

            mc_hist.selector(selector=selector)  # rect, peak_x, range
            mc_hist.save_fig(label=target_value, fig_name=figname_value)

        # Enable the button when the code is finished
        button_plot.disabled = False

    button_plot.on_click(lambda b: on_button_click(b, variables, selector))
    button_fit.on_click(lambda b: parametric_fit(variables, calibration_mode, out_mc))
    reset_back_button.on_click(lambda b: reset_back_on_click(variables))
    button_plot_result.on_click(lambda b: plot_fit_result(b, variables, calibration_mode, out_mc))

    widget_container = widgets.VBox([
        widgets.HBox([widgets.Label(value="Calibration mde:", layout=label_layout), calibration_mode]),
        widgets.HBox([widgets.Label(value="Bin Size:", layout=label_layout), bin_size_widget]),
        widgets.HBox([widgets.Label(value="Log:", layout=label_layout), log_widget]),
        widgets.HBox([widgets.Label(value="Normalize:", layout=label_layout), mode_widget]),
        widgets.HBox([widgets.Label(value="Prominence:", layout=label_layout), prominence_widget]),
        widgets.HBox([widgets.Label(value="Distance:", layout=label_layout), distance_widget]),
        widgets.HBox([widgets.Label(value="Lim:", layout=label_layout), lim_widget]),
        widgets.HBox([widgets.Label(value="Percent:", layout=label_layout), percent_widget]),
        widgets.HBox([widgets.Label(value="Figname:", layout=label_layout), figname_widget]),
        widgets.HBox([widgets.Label(value="Fig. size W:", layout=label_layout), figure_mc_size_x]),
        widgets.HBox([widgets.Label(value="Fig. size H:", layout=label_layout), figure_mc_size_y]),
        widgets.HBox([button_plot, button_fit, button_plot_result, reset_back_button]),
    ])

    ion_list_box = widgets.VBox([dropdown, chargeDropdown, buttonAdd, buttonDelete, buttonReset])

    with out_ion_list:
        clear_output(True)
        print("Updated List: ", variables.list_material)
        print("Updated element List: ", variables.element)
        print("Updated isotope List: ", variables.isotope)
        print("Updated charge List: ", variables.charge)

    output_layout = widgets.HBox([out_mc, out_ion_list])
    display_layout = widgets.HBox([widget_container, ion_list_box])
    display(display_layout, output_layout)


def reset_back_on_click(variables):
    variables.dld_t_calib = np.copy(variables.dld_t_calib_backup)
    variables.mc_calib = np.copy(variables.mc_calib_backup)

