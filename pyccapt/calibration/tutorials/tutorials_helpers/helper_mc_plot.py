import ipywidgets as widgets
from IPython.display import display
from ipywidgets import Output

from pyccapt.calibration.core import mc_plot

# Define a layout for labels to make them a fixed width
label_layout = widgets.Layout(width='200px')
def call_mc_plot(variables, selector):
    out = Output()
    # Define widgets for fine_tune_t_0 function
    bin_size_widget = widgets.FloatText(value=0.1)
    log_widget = widgets.Dropdown(options=[('True', True), ('False', False)])
    grid_widget = widgets.Dropdown(options=[('True', True), ('False', False)])
    mode_widget = widgets.Dropdown(options=[('False', False), ('True', True)])
    mrp_all_widget = widgets.Dropdown(options=[('True', True), ('False', False)])
    prominence_widget = widgets.IntText(value=10)
    distance_widget = widgets.IntText(value=100)
    lim_widget = widgets.IntText(value=10000)
    percent_widget = widgets.IntText(value=50)
    figname_widget = widgets.Text(value='hist')
    figure_mc_size_x = widgets.FloatText(value=9.0)
    figure_mc_size_y = widgets.FloatText(value=5.0)
    target_mode = widgets.Dropdown(options=[('mc', 'mc'), ('tof', 'tof'), ('mc_uc', 'mc_uc'), ('tof_c', 'tof_c')])
    plot_peak = widgets.Dropdown(options=[('True', True), ('False', False)])
    save = widgets.Dropdown(options=[('False', False), ('True', True)])

    # Create a button widget to trigger the function
    button_plot = widgets.Button(description="plot")

    def on_button_click(b, variables, selector):
        # Disable the button while the code is running
        button_plot.disabled = True

        # Get the values from the widgets
        bin_size_value = bin_size_widget.value
        log_value = log_widget.value
        grid_value = grid_widget.value
        mode_value = mode_widget.value
        target_value = target_mode.value
        prominence_value = prominence_widget.value
        distance_value = distance_widget.value
        percent_value = percent_widget.value
        figname_value = figname_widget.value
        lim_value = lim_widget.value
        figure_size = (figure_mc_size_x.value, figure_mc_size_y.value)
        with out:  # Capture the output within the 'out' widget
            out.clear_output()  # Clear any previous output
            # Call the function
            if target_value == 'mc_uc':
                hist = variables.data['mc_uc (Da)']
            elif target_value == 'tof_c':
                hist = variables.data['t_c (ns)']
            elif target_value == 'mc':
                hist = variables.data['mc (Da)']
            elif target_value == 'tof':
                hist = variables.data['t (ns)']
            mc_hist = mc_plot.AptHistPlotter(hist[hist < lim_value], variables)
            mc_hist.plot_histogram(bin_width=bin_size_value, normalize=mode_value, label='mc', steps='stepfilled',
                                   log=log_value, grid=grid_value, fig_size=figure_size)
            if mode_value != 'normalized':
                mc_hist.find_peaks_and_widths(prominence=prominence_value, distance=distance_value,
                                              percent=percent_value)
                if plot_peak.value:
                    mc_hist.plot_peaks()
                mc_hist.plot_hist_info_legend(label='mc', mrp_all=mrp_all_widget.value, background=None, loc='right')

            mc_hist.selector(selector=selector)  # rect, peak_x, range
            if save.value:
                mc_hist.save_fig(label=target_value, fig_name=figname_value)

        # Enable the button when the code is finished
        button_plot.disabled = False

    button_plot.on_click(lambda b: on_button_click(b, variables, selector))

    widget_container = widgets.VBox([
        widgets.HBox([widgets.Label(value="target:", layout=label_layout), target_mode]),
        widgets.HBox([widgets.Label(value="Bin Size:", layout=label_layout), bin_size_widget]),
        widgets.HBox([widgets.Label(value="Log:", layout=label_layout), log_widget]),
        widgets.HBox([widgets.Label(value="Grid:", layout=label_layout), grid_widget]),
        widgets.HBox([widgets.Label(value="Normalize:", layout=label_layout), mode_widget]),
        widgets.HBox([widgets.Label(value="Prominence:", layout=label_layout), prominence_widget]),
        widgets.HBox([widgets.Label(value="Distance:", layout=label_layout), distance_widget]),
        widgets.HBox([widgets.Label(value="Lim:", layout=label_layout), lim_widget]),
        widgets.HBox([widgets.Label(value="Percent:", layout=label_layout), percent_widget]),
        widgets.HBox([widgets.Label(value="MRP all:", layout=label_layout), mrp_all_widget]),
        widgets.HBox([widgets.Label(value="Plot peak:", layout=label_layout), plot_peak]),
        widgets.HBox([widgets.Label(value="Save:", layout=label_layout), save]),
        widgets.HBox([widgets.Label(value="Fig name:", layout=label_layout), figname_widget]),
        widgets.HBox([widgets.Label(value="Fig size:", layout=label_layout),
                      widgets.HBox([figure_mc_size_x, figure_mc_size_y])]),
        widgets.HBox([button_plot]),
    ])

    display(widget_container)
    display(out)
