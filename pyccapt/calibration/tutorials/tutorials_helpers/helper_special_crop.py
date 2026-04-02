import ipywidgets as widgets
from IPython.display import display
from ipywidgets import Output

from pyccapt.calibration.data_tools import data_loadcrop

label_layout = widgets.Layout(width='200px')


def reset(variables, out):
    variables.restore_data_from_backup()
    with out:
        out.clear_output()
        print('Reset the crop')


def _apply_manual_spatial_values(variables, center_x, center_y, radius):
    variables.selected_x_fdm = float(center_x)
    variables.selected_y_fdm = float(center_y)
    variables.roi_fdm = float(radius)


def apply_crop(variables, out, use_manual_values, center_x_widget, center_y_widget, radius_widget):
    with out:
        out.clear_output()
        try:
            if use_manual_values.value:
                _apply_manual_spatial_values(
                    variables,
                    center_x_widget.value,
                    center_y_widget.value,
                    radius_widget.value,
                )

            if variables.roi_fdm <= 0:
                print('Please draw a spatial crop or enable manual values with a positive radius.')
                return

            variables.sync_from_data(
                data_loadcrop.crop_data_after_selection(variables.data.copy(), variables),
                update_backups=False,
                clear_selection=False,
            )
            print(
                'The crop for center x:',
                round(float(variables.selected_x_fdm), 4),
                'center y:',
                round(float(variables.selected_y_fdm), 4),
                'and radius:',
                round(float(variables.roi_fdm), 4),
                'is applied',
            )
            print('Remaining rows after crop:', len(variables.data))
        except ValueError as exc:
            print(f'Invalid spatial crop: {exc}')


def call_plot_crop_fdm(variables):
    frac_widget = widgets.FloatText(value=1.0)
    frac_label = widgets.Label(value="Fraction:", layout=label_layout)

    bins_x = widgets.IntText(value=256)
    bins_y = widgets.IntText(value=256)
    bins_label = widgets.Label(value="Bins (X, Y):", layout=label_layout)

    figure_size_x = widgets.FloatText(value=5.0)
    figure_size_y = widgets.FloatText(value=4.0)
    figure_size_label = widgets.Label(value="Figure Size (X, Y):", layout=label_layout)

    mode_selector_widget = widgets.Dropdown(options=[('circle', 'circle'), ('ellipse', 'ellipse')], value='circle')
    mode_selector_label = widgets.Label(value="Selector:", layout=label_layout)

    use_manual_values = widgets.Checkbox(value=False, description='Use manual crop values')
    use_manual_label = widgets.Label(value="Manual Crop:", layout=label_layout)
    center_x_widget = widgets.FloatText(value=float(variables.selected_x_fdm))
    center_y_widget = widgets.FloatText(value=float(variables.selected_y_fdm))
    radius_widget = widgets.FloatText(value=float(variables.roi_fdm))

    center_label = widgets.Label(value="Center (X, Y):", layout=label_layout)
    radius_label = widgets.Label(value="Radius:", layout=label_layout)

    save_widget = widgets.Dropdown(options=[('True', True), ('False', False)], value=False)
    save_label = widgets.Label(value="Save:", layout=label_layout)

    figname_widget = widgets.Text(value='fdm_ini')
    figname_label = widgets.Label(value="Figure Name:", layout=label_layout)

    button_plot = widgets.Button(description="Plot")
    button_apply = widgets.Button(description="Apply crop")
    button_reset = widgets.Button(description="Reset")
    button_sync = widgets.Button(description="Load drawn values")

    out = Output()

    def sync_from_drawn(_):
        center_x_widget.value = float(variables.selected_x_fdm)
        center_y_widget.value = float(variables.selected_y_fdm)
        radius_widget.value = float(variables.roi_fdm)
        with out:
            out.clear_output()
            print('Loaded the current drawn crop values into the manual fields.')

    def on_button_click(_, variables):
        button_plot.disabled = True

        frac = frac_widget.value
        bins = (bins_x.value, bins_y.value)
        figure_size = (figure_size_x.value, figure_size_y.value)
        save = save_widget.value
        figname = figname_widget.value
        mode_selector = mode_selector_widget.value

        if use_manual_values.value:
            _apply_manual_spatial_values(
                variables,
                center_x_widget.value,
                center_y_widget.value,
                radius_widget.value,
            )

        with out:
            out.clear_output()
            try:
                data = variables.data.copy()
                data_loadcrop.plot_crop_fdm(
                    data['x_det (cm)'].to_numpy(),
                    data['y_det (cm)'].to_numpy(),
                    bins,
                    frac,
                    axis_mode='normal',
                    figure_size=figure_size,
                    variables=variables,
                    range_sequence=[],
                    range_mc=[],
                    range_detx=[],
                    range_dety=[],
                    range_x=[],
                    range_y=[],
                    range_z=[],
                    range_vol=[],
                    data_crop=True,
                    draw_circle=use_manual_values.value and variables.roi_fdm > 0,
                    mode_selector=mode_selector,
                    save=save,
                    figname=figname,
                )
                if use_manual_values.value:
                    print('Manual spatial crop preview updated on the FDM plot.')
                else:
                    print('Draw a crop on the plot or enable manual values and re-plot for a preview.')
            except ValueError as exc:
                print(f'Invalid spatial crop preview: {exc}')

        button_plot.disabled = False

    button_plot.on_click(lambda b: on_button_click(b, variables))
    button_apply.on_click(
        lambda b: apply_crop(variables, out, use_manual_values, center_x_widget, center_y_widget, radius_widget)
    )
    button_reset.on_click(lambda b: reset(variables, out))
    button_sync.on_click(sync_from_drawn)

    widget_container = widgets.VBox([
        widgets.HBox([frac_label, frac_widget]),
        widgets.HBox([bins_label, widgets.HBox([bins_x, bins_y])]),
        widgets.HBox([figure_size_label, widgets.HBox([figure_size_x, figure_size_y])]),
        widgets.HBox([mode_selector_label, mode_selector_widget]),
        widgets.HBox([use_manual_label, use_manual_values]),
        widgets.HBox([center_label, widgets.HBox([center_x_widget, center_y_widget])]),
        widgets.HBox([radius_label, radius_widget]),
        widgets.HBox([save_label, save_widget]),
        widgets.HBox([figname_label, figname_widget]),
        widgets.HBox([button_plot, button_sync, button_apply, button_reset]),
    ])
    display(widget_container)
    display(out)
