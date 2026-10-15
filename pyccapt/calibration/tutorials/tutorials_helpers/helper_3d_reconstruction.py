import ast

import ipywidgets as widgets
from IPython.display import display
from ipywidgets import Output

from pyccapt.calibration.reconstructions import reconstruction
from pyccapt.calibration.reconstructions.species_display import (
    default_element_controls,
    resolve_element_controls,
)

# Define a layout for labels to make them a fixed width
label_layout = widgets.Layout(width='200px')


def call_x_y_z_calculation(variables, flight_path_length, element_selected, colab=False, *, display_panel=True):
    out = Output()
    avg_dens = element_selected.value[1]
    field_evap = element_selected.value[2]

    element_percentage = str(default_element_controls(variables.range_data, 0.01))

    # Create widgets with initial values
    kf_widget = widgets.FloatText(value=4, step=0.1)
    det_eff_widget = widgets.FloatText(value=0.7, step=0.1)
    icf_widget = widgets.FloatText(value=1.4, step=0.1)
    field_evap_widget = widgets.FloatText(value=field_evap)
    avg_dens_widget = widgets.FloatText(value=avg_dens)
    element_percentage_widget = widgets.Textarea(value=element_percentage)
    figname_widget = widgets.Text(value='3d')
    mode_widget = widgets.Dropdown(options=[('Geiser', 'Geiser'), ('Bas', 'Bas')])
    opacity_widget = widgets.BoundedFloatText(value=0.5, min=0, max=1, step=0.1)
    save_widget = widgets.Dropdown(options=[('True', True), ('False', False)], value=False)
    display_widget = widgets.Dropdown(options=[('In this cell', 'inline'), ('Separate browser HTML', 'browser')], value='inline')
    max_points = widgets.BoundedIntText(value=50000, min=100, max=1000000)
    status = widgets.HTML()

    # Create a button widget to trigger the function
    button_calculate_plot = widgets.Button(description="Reconstruct & plot", button_style='primary')
    button_load = widgets.Button(description='Load accepted parameters')
    button_load.layout.width = 'auto'
    button_load.tooltip = 'Copy an explicitly accepted tuning result into the reconstruction fields.'
    parameter_widgets = dict(kf=kf_widget, icf=icf_widget, det_eff=det_eff_widget,
                             field_evap=field_evap_widget, avg_dens=avg_dens_widget,
                             mode=mode_widget, flight_path_length=flight_path_length)

    def load_accepted(_=None):
        accepted = getattr(variables, 'reconstruction_parameters', {})
        for name, value in accepted.items():
            name = 'mode' if name == 'projection' else name
            if name in parameter_widgets:
                parameter_widgets[name].value = value
        button_load.disabled = not bool(accepted)
        status.value = ('Accepted parameters loaded into the reconstruction controls.' if accepted else
                        'No accepted parameter set yet. Tune in the first tab, or enter reviewed values here.')

    load_accepted()
    button_load.on_click(load_accepted)

    def on_button_click(b, variables, flight_path_length):
        # Disable the button while the code is running
        button_calculate_plot.disabled = True

        # Get the values from the widgets
        kf_value = kf_widget.value
        det_eff_value = det_eff_widget.value
        icf_value = icf_widget.value
        field_evap_value = field_evap_widget.value
        avg_dens_value = avg_dens_widget.value
        element_percentage_value = element_percentage_widget.value
        figname_value = figname_widget.value
        mode_value = mode_widget.value
        opacity_value = opacity_widget.value
        save_value = save_widget.value

        try:
            with out:
                out.clear_output()
                element_percentage_dic = ast.literal_eval(element_percentage_value)
                element_percentage_list, unranged_fraction = resolve_element_controls(
                    variables.range_data, element_percentage_dic, 0.1, 0.01
                )
                reconstruction.x_y_z_calculation_and_plot(
                    kf=kf_value, det_eff=det_eff_value, icf=icf_value,
                    field_evap=field_evap_value, avg_dens=avg_dens_value,
                    element_percentage=element_percentage_list, rotary_fig_save=False,
                    variables=variables, flight_path_length=flight_path_length.value,
                    figname=figname_value, mode=mode_value, opacity=opacity_value,
                    save=save_value, colab=colab, cluster_result=None,
                    unranged_fraction=unranged_fraction,
                    display_mode=display_widget.value, max_display_points=max_points.value,
                )
                actual_parameters = {name: widget.value for name, widget in parameter_widgets.items()}
                if actual_parameters != variables.reconstruction_parameters:
                    variables.reconstruction_ratio_result = {}
                variables.reconstruction_parameters = actual_parameters
                button_load.disabled = False
                status.value = 'Reconstruction complete with the parameters shown above.'
                from pyccapt.calibration.core.calibration_metadata import record_step
                record_step(variables, 'reconstruction', parameters=actual_parameters)
        finally:
            button_calculate_plot.disabled = False

    button_calculate_plot.on_click(lambda b: on_button_click(b, variables, flight_path_length))

    widget_container = widgets.VBox(
        [
            widgets.HBox([widgets.Label(value='KF:', layout=label_layout), kf_widget, widgets.Label(value='(dimensionless)')]),
            widgets.HBox([widgets.Label(value='Det_eff:', layout=label_layout), det_eff_widget]),
            widgets.HBox([widgets.Label(value='ICF:', layout=label_layout), icf_widget]),
            widgets.HBox(
                [widgets.Label(value='Field_evap:', layout=label_layout), field_evap_widget, widgets.Label(value='(V/nm)')]
            ),
            widgets.HBox(
                [widgets.Label(value='Avg_dens:', layout=label_layout), avg_dens_widget, widgets.Label(value='(atoms/nm^3)')]
            ),
            widgets.HBox([widgets.Label(value='Display fraction:', layout=label_layout), element_percentage_widget]),
            widgets.HBox([widgets.Label(value='Fig name:', layout=label_layout), figname_widget]),
            widgets.HBox([widgets.Label(value='Save fig:', layout=label_layout), save_widget]),
            widgets.HBox([widgets.Label(value='Mode:', layout=label_layout), mode_widget]),
            widgets.HBox([widgets.Label(value='Opacity:', layout=label_layout), opacity_widget]),
            widgets.HBox([widgets.Label(value='Show plot:', layout=label_layout), display_widget]),
            widgets.HBox([widgets.Label(value='Max displayed ions:', layout=label_layout), max_points]),
            widgets.HBox([button_load, button_calculate_plot]),
            status, out,
        ]
    )

    from pyccapt.calibration.tutorials.tutorials_helpers.helper_sdm_calibration import call_sdm_icf_kf_calibration
    def transfer(parameters):
        load_accepted()
        tabs.selected_index = 1
    tuning = call_sdm_icf_kf_calibration(variables, flight_path_length, element_selected, colab,
                                       display_panel=False, on_parameters=transfer)
    tabs = widgets.Tab(children=[tuning, widget_container])
    tabs.set_title(0, '1. Find / tune ICF and kf')
    tabs.set_title(1, '2. Reconstruct and plot')
    panel = widgets.VBox([widgets.HTML('Tune or explicitly choose geometry in the first tab, then reconstruct. Display sampling does not change coordinates.'), tabs])
    panel.reconstruction_tabs = tabs
    panel.controls = {'load': button_load, 'reconstruct': button_calculate_plot, 'display_mode': display_widget,
                      'max_points': max_points, 'parameters': parameter_widgets, 'status': status, 'tuning': tuning}
    variables._reconstruction_panel = panel
    if display_panel:
        display(panel)
        return None
    return panel
