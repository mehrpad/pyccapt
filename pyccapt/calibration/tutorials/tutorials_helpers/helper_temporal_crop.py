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


def _apply_manual_temporal_values(variables, start_idx, end_idx):
    variables.selected_x1 = int(start_idx)
    variables.selected_x2 = int(end_idx)


def apply_crop(variables, out, use_manual_values, start_widget, end_widget):
    with out:
        out.clear_output()
        try:
            if use_manual_values.value:
                _apply_manual_temporal_values(variables, start_widget.value, end_widget.value)

            variables.sync_from_data(
                data_loadcrop.crop_dataset(variables.data, variables),
                update_backups=False,
                clear_selection=False,
            )
            print(
                'The crop for start index:',
                int(min(variables.selected_x1, variables.selected_x2)),
                'and end index:',
                int(max(variables.selected_x1, variables.selected_x2)),
                'is applied',
            )
            print('Remaining rows after crop:', len(variables.data))
        except ValueError as exc:
            print(f'Invalid temporal crop: {exc}')


def call_plot_crop_experiment(variables, pulse_mode):
    max_tof_widget = widgets.FloatText(value=variables.max_tof)
    max_tof_label = widgets.Label(value="Max TOF:", layout=label_layout)

    frac_widget = widgets.FloatText(value=1.0)
    frac_label = widgets.Label(value="Fraction:", layout=label_layout)

    bins_x = widgets.IntText(value=1200)
    bins_y = widgets.IntText(value=800)
    bins_label = widgets.Label(value="Bins (X, Y):", layout=label_layout)

    figure_size_x = widgets.FloatText(value=7.0)
    figure_size_y = widgets.FloatText(value=3.0)
    figure_size_label = widgets.Label(value="Figure Size (X, Y):", layout=label_layout)

    draw_rect_widget = widgets.fixed(False)
    data_crop_widget = widgets.fixed(True)
    pulse_plot_widget = widgets.Dropdown(options=[('False', False), ('True', True)], value=False)
    pulse_plot_label = widgets.Label(value="Pulse Plot:", layout=label_layout)

    dc_plot_widget = widgets.Dropdown(options=[('True', True), ('False', False)], value=True)
    dc_plot_label = widgets.Label(value="DC Plot:", layout=label_layout)

    pulse_mode_widget = widgets.Dropdown(options=[('voltage', 'voltage'), ('laser', 'laser')], value=pulse_mode)
    pulse_mode_label = widgets.Label(value="Pulse Mode:", layout=label_layout)

    use_manual_values = widgets.Checkbox(value=False, description='Use manual crop indices')
    use_manual_label = widgets.Label(value="Manual Crop:", layout=label_layout)
    start_widget = widgets.IntText(value=int(min(variables.selected_x1, variables.selected_x2)))
    end_widget = widgets.IntText(value=int(max(variables.selected_x1, variables.selected_x2)))
    index_label = widgets.Label(value="Index Range:", layout=label_layout)

    save_widget = widgets.Dropdown(options=[('True', True), ('False', False)], value=False)
    save_label = widgets.Label(value="Save:", layout=label_layout)

    figname_widget = widgets.Text(value='hist_ini')
    figname_label = widgets.Label(value="Figure Name:", layout=label_layout)

    button_plot = widgets.Button(description="Plot")
    button_apply = widgets.Button(description="Apply crop")
    button_reset = widgets.Button(description="Reset")
    button_sync = widgets.Button(description="Load drawn values")

    out = Output()

    def sync_from_drawn(_):
        start_widget.value = int(min(variables.selected_x1, variables.selected_x2))
        end_widget.value = int(max(variables.selected_x1, variables.selected_x2))
        with out:
            out.clear_output()
            print('Loaded the current drawn crop values into the manual fields.')

    def on_button_click(_, variables):
        button_plot.disabled = True

        data = variables.data.copy()
        max_tof = max_tof_widget.value
        frac = frac_widget.value
        bins = (bins_x.value, bins_y.value)
        figure_size = (figure_size_x.value, figure_size_y.value)
        draw_rect = draw_rect_widget.value
        data_crop = data_crop_widget.value
        pulse_plot = pulse_plot_widget.value
        dc_plot = dc_plot_widget.value
        pulse_mode_value = pulse_mode_widget.value
        save = save_widget.value
        figname = figname_widget.value

        if use_manual_values.value:
            _apply_manual_temporal_values(variables, start_widget.value, end_widget.value)
            draw_rect = True

        with out:
            out.clear_output()
            try:
                data_loadcrop.plot_crop_experiment_history(
                    data,
                    variables,
                    max_tof,
                    frac,
                    bins,
                    figure_size,
                    draw_rect,
                    data_crop,
                    pulse_plot,
                    dc_plot,
                    pulse_mode_value,
                    save,
                    figname,
                )
                if use_manual_values.value:
                    print('Manual temporal crop preview updated on the experiment-history plot.')
                else:
                    print('Draw a rectangle on the plot or enable manual indices and re-plot for a preview.')
            except ValueError as exc:
                print(f'Invalid temporal crop preview: {exc}')

        button_plot.disabled = False

    button_plot.on_click(lambda b: on_button_click(b, variables))
    button_apply.on_click(lambda b: apply_crop(variables, out, use_manual_values, start_widget, end_widget))
    button_reset.on_click(lambda b: reset(variables, out))
    button_sync.on_click(sync_from_drawn)

    widget_container = widgets.VBox(
        [
            widgets.HBox([max_tof_label, max_tof_widget]),
            widgets.HBox([frac_label, frac_widget]),
            widgets.HBox([bins_label, widgets.HBox([bins_x, bins_y])]),
            widgets.HBox([figure_size_label, widgets.HBox([figure_size_x, figure_size_y])]),
            widgets.HBox([pulse_plot_label, pulse_plot_widget]),
            widgets.HBox([dc_plot_label, dc_plot_widget]),
            widgets.HBox([pulse_mode_label, pulse_mode_widget]),
            widgets.HBox([use_manual_label, use_manual_values]),
            widgets.HBox([index_label, widgets.HBox([start_widget, end_widget])]),
            widgets.HBox([save_label, save_widget]),
            widgets.HBox([figname_label, figname_widget]),
            widgets.HBox([button_plot, button_sync, button_apply, button_reset]),
        ]
    )
    display(widget_container)
    display(out)
