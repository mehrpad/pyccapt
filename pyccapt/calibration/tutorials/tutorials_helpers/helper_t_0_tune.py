import ipywidgets as widgets
from IPython.display import display
from ipywidgets import Output

from pyccapt.calibration.calibration import mc_plot
from pyccapt.calibration.mc import mc_tools

# Define a layout for labels to make them a fixed width
label_layout = widgets.Layout(width='200px')

def call_fine_tune_t_0(variables, flightPathLength, pulse_mode, t0):

    # Define widgets for fine_tune_t_0 function
    t0_d_widget = widgets.FloatText(value=t0.value)
    flightPathLength = widgets.FloatText(value=flightPathLength.value)
    bin_size_widget = widgets.FloatText(value=0.1)
    log_widget = widgets.Dropdown(options=[('True', True), ('False', False)])
    mode_widget = widgets.Dropdown(options=[('False', False), ('True', True)])
    target_widget = widgets.Dropdown(options=[('mc', 'mc'), ('tof', 'tof')])
    prominence_widget = widgets.IntText(value=10)
    distance_widget = widgets.IntText(value=100)
    lim_widget = widgets.IntText(value=400)
    percent_widget = widgets.IntText(value=50)
    figure_size_x = widgets.FloatText(value=9.0)
    figure_size_y = widgets.FloatText(value=5.0)
    save_figure_widget = widgets.Dropdown(options=[('False', False), ('True', True)])
    figure_size_label = widgets.Label(value="Figure Size (X, Y):", layout=label_layout)
    fig_name = widgets.Text(value='t0_tune')

    # Create a button widget to trigger the function
    button_plot = widgets.Button(description="plot")

    out = Output()
    def on_button_click(b, variables):
        # Disable the button while the code is running
        button_plot.disabled = True

        # Get the values from the widgets
        t0_d_value = t0_d_widget.value
        flightPathLength_value = flightPathLength.value
        bin_size_value = bin_size_widget.value
        log_value = log_widget.value
        mode_value = mode_widget.value
        target_value = target_widget.value
        prominence_value = prominence_widget.value
        distance_value = distance_widget.value
        percent_value = percent_widget.value
        lim_value = lim_widget.value

        with out:  # Capture the output within the 'out' widget
            out.clear_output()  # Clear any previous output
            # Call the function
            figure_size = (figure_size_x.value, figure_size_y.value)
            variables.mc_uc = mc_tools.tof2mc(variables.dld_t, t0_d_value, variables.dld_high_voltage,
                                                 variables.dld_x_det, variables.dld_y_det, flightPathLength_value,
                                                 variables.dld_pulse_v, mode=pulse_mode.value)
            if target_value == 'mc':
                mc_hist = mc_plot.AptHistPlotter(variables.mc_uc[variables.mc_uc < lim_value], variables)
                mc_hist.plot_histogram(bin_width=bin_size_value, normalize=mode_value, label='mc', steps='stepfilled',
                                       log=log_value, fig_size=figure_size)
            elif target_value == 'tof':
                mc_hist = mc_plot.AptHistPlotter(variables.dld_t[variables.dld_t < lim_value], variables)
                mc_hist.plot_histogram(bin_width=bin_size_value, normalize=mode_value, label='tof', steps='stepfilled',
                                       log=log_value, fig_size=figure_size)

            if not mode_value:
                mc_hist.find_peaks_and_widths(prominence=prominence_value, distance=distance_value,
                                              percent=percent_value)
                mc_hist.plot_peaks()
                mc_hist.plot_hist_info_legend(label=target_value, background=None, loc='right')

            if save_figure_widget.value:
                mc_hist.save_fig(target_value, fig_name.value)


        # Enable the button when the code is finished
        button_plot.disabled = False

    button_plot.on_click(lambda b: on_button_click(b, variables))

    widget_container = widgets.VBox([
        widgets.HBox([widgets.Label(value="t0:", layout=label_layout), t0_d_widget]),
        widgets.HBox([widgets.Label(value="Flight Path Length:", layout=label_layout), flightPathLength]),
        widgets.HBox([widgets.Label(value="Bin Size:", layout=label_layout), bin_size_widget]),
        widgets.HBox([widgets.Label(value="Log:", layout=label_layout), log_widget]),
        widgets.HBox([widgets.Label(value="Normalize:", layout=label_layout), mode_widget]),
        widgets.HBox([widgets.Label(value="Target:", layout=label_layout), target_widget]),
        widgets.HBox([widgets.Label(value="Prominence:", layout=label_layout), prominence_widget]),
        widgets.HBox([widgets.Label(value="Distance:", layout=label_layout), distance_widget]),
        widgets.HBox([widgets.Label(value="Lim:", layout=label_layout), lim_widget]),
        widgets.HBox([widgets.Label(value="Percent:", layout=label_layout), percent_widget]),
        widgets.HBox([widgets.Label(value="Save Figure:", layout=label_layout), save_figure_widget]),
        widgets.HBox([widgets.Label(value="Figure Name:", layout=label_layout), fig_name]),
        widgets.HBox([figure_size_label, widgets.HBox([figure_size_x, figure_size_y])]),
        widgets.HBox([button_plot]),
    ])

    display(widget_container)
    display(out)
