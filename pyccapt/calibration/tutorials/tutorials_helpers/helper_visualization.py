import ast
import json
import re

import ipywidgets as widgets
import matplotlib.colors as mcolors
import numpy as np
from IPython.display import display, clear_output, HTML
from ipywidgets import Output

from pyccapt.calibration.calibration import mc_plot, ion_selection
from pyccapt.calibration.data_tools import data_loadcrop
from pyccapt.calibration.reconstructions import reconstruction, sdm, rdf, density_map
from pyccapt.calibration.reconstructions import iso_surface

# Define a layout for labels to make them a fixed width
label_layout = widgets.Layout(width='200px')


def call_visualization(variables, colab=False):
    plot_mc_button = widgets.Button(description='plot mc')
    plot_3d_button = widgets.Button(description='plot 3D')
    plot_3d_button_iso = widgets.Button(description='plot 3D iso surface')
    plot_heatmap_button = widgets.Button(description='plot heatmap')
    plot_projection_button = widgets.Button(description='plot projection')
    clear_button = widgets.Button(description='Clear plots')
    plot_fdm_button = widgets.Button(description="plot FDM")
    plot_experiment_button = widgets.Button(description="plot experiment history")
    density_map_button = widgets.Button(description="plot density map")
    plot_sdm_button = widgets.Button(description='plot SDM')
    plot_rdf_button = widgets.Button(description='plot RDF')
    show_color = widgets.Button(description='show color')
    change_color = widgets.Button(description='change color')


    if variables.range_data.empty or variables.range_data['ion'].iloc[0] == 'unranged':
        element_percentage = str({'unranged': 0.01})
    else:
        # Iterate over unique elements in the "element" column
        element_percentage = {}
        for element_list in variables.range_data['element']:
            for element in element_list:
                # Check if the element is already a key in the dictionary
                if element not in element_percentage:
                    element_percentage[element] = 0.01
        element_percentage = str(element_percentage)

    #############
    peak_find_plot = widgets.Dropdown(options=[('True', True), ('False', False)])
    peaks_find = widgets.Dropdown(options=[('True', True), ('False', False)])
    plot_ranged_colors = widgets.Dropdown(options=[('False', False), ('True', True)])
    plot_ranged_peak = widgets.Dropdown(options=[('False', False), ('True', True)])
    target_mode = widgets.Dropdown(options=[('mc', 'mc'), ('mc_uc', 'mc_uc'), ('tof_c', 'tof_c'), ('tof', 'tof')])
    print_info = widgets.Dropdown(options=[('False', False), ('True', True)])
    log_widget = widgets.Dropdown(options=[('True', True), ('False', False)])
    normalize = widgets.Dropdown(options=[('False', False), ('True', True)])
    legend_widget = widgets.Dropdown(options=[('long', 'long'), ('short', 'short')])
    # hist_steps = widgets.Dropdown(options=[('bar', 'bar'), ('stepfilled', 'stepfilled')])
    bin_size_pm = widgets.FloatText(value=0.1)
    lim_mc_pm = widgets.IntText(value=150)
    prominence = widgets.IntText(value=50)
    distance = widgets.IntText(value=50)
    mrp_all = widgets.Dropdown(options=[('False', False), ('True', True)], value=False)
    percent = widgets.IntText(value=50)
    background_mc = widgets.Dropdown(options=[('None', None), ('aspls', 'aspls'), ('fabc', 'fabc'),
                                              ('manual@4', 'manual@4'),
                                              ('manual@100', 'manual@100'), ('manual', 'manual')])
    figname_mc = widgets.Text(value='mc')
    figure_mc_size_x_mc = widgets.FloatText(value=9.0)
    figure_mc_size_y_mc = widgets.FloatText(value=5.0)
    save_mc = widgets.Dropdown(options=[('True', True), ('False', False)], value=False)
    grid_mc = widgets.Dropdown(options=[('False', False), ('True', True)])
    range_sequence_mc = widgets.Textarea(value='[0,0]')
    range_detx_mc = widgets.Textarea(value='[0,0]')
    range_dety_mc = widgets.Textarea(value='[0,0]')
    range_mc_mc = widgets.Textarea(value='[0,0]')
    range_x_mc = widgets.Textarea(value='[0,0]')
    range_y_mc = widgets.Textarea(value='[0,0]')
    range_z_mc = widgets.Textarea(value='[0,0]')
    range_vol_mc = widgets.Textarea(value='[0,0]')

    plot_mc_button.on_click(lambda b: plot_mc(b, variables, out))

    def plot_mc(b, variables, out):
        plot_mc_button.disabled = True
        figure_size = (figure_mc_size_x_mc.value, figure_mc_size_y_mc.value)
        with out:
            try:
                # Use json.loads to convert the entered string to a list
                range_sequence = json.loads(range_sequence_mc.value)
                range_mc = json.loads(range_mc_mc.value)
                range_detx = json.loads(range_detx_mc.value)
                range_dety = json.loads(range_dety_mc.value)
                range_x = json.loads(range_x_mc.value)
                range_y = json.loads(range_y_mc.value)
                range_z = json.loads(range_z_mc.value)
                range_vol = json.loads(range_vol_mc.value)
                if range_sequence == [0, 0]:
                    range_sequence = []
                if range_mc == [0, 0]:
                    range_mc = []
                if range_detx == [0, 0] or range_dety == [0, 0]:
                    range_detx = []
                    range_dety = []

                if range_x == [0, 0] or range_y == [] or range_z == [0, 0]:
                    range_x = []
                    range_y = []
                    range_z = []
                if range_vol == [0, 0]:
                    range_vol = []

            except json.JSONDecodeError:
                # Handle invalid input
                print(f"Invalid range input")
            mc_plot.hist_plot(variables, bin_size_pm.value, log=log_widget.value, target=target_mode.value,
                              normalize=normalize.value,
                              prominence=prominence.value, distance=distance.value, percent=percent.value,
                              selector='rect',
                              figname=figname_mc.value, lim=lim_mc_pm.value, peaks_find=peaks_find.value,
                              peaks_find_plot=peak_find_plot.value, plot_ranged_colors=plot_ranged_colors.value,
                              plot_ranged_peak=plot_ranged_peak.value, mrp_all=mrp_all.value,
                              background=background_mc.value, grid=grid_mc.value,
                              range_sequence=range_sequence, range_mc=range_mc, range_detx=range_detx,
                              range_dety=range_dety, range_x=range_x, range_y=range_y, range_z=range_z,
                              range_vol=range_vol, print_info=print_info.value, figure_size=figure_size,
                              save_fig=save_mc.value, legend_mode=legend_widget.value)

        plot_mc_button.disabled = False

    #############
    # Define widgets for each parameter
    frac_fdm_widget = widgets.FloatText(value=1.0)
    bins_fdm = widgets.Textarea(value='[256,256]')
    axis_mode_fdm = widgets.Dropdown(options=['normal', 'scalebar'], value='normal')
    figure_size_x_fdm = widgets.FloatText(value=5.0)
    figure_size_y_fdm = widgets.FloatText(value=4.0)
    save_fdm_widget = widgets.Dropdown(options=[('True', True), ('False', False)], value=False)
    figname_fdm_widget = widgets.Text(value='fdm_ini')
    range_sequence_fdm = widgets.Textarea(value='[0,0]')
    range_detx_fdm = widgets.Textarea(value='[0,0]')
    range_dety_fdm = widgets.Textarea(value='[0,0]')
    range_mc_fdm = widgets.Textarea(value='[0,0]')
    range_x_fdm = widgets.Textarea(value='[0,0]')
    range_y_fdm = widgets.Textarea(value='[0,0]')
    range_z_fdm = widgets.Textarea(value='[0,0]')
    range_vol_fdm = widgets.Textarea(value='[0,0]')

    plot_fdm_button.on_click(lambda b: plot_fdm(b, variables, out))

    def plot_fdm(b, variables, out):
        plot_fdm_button.disabled = True

        with out:  # Capture the output within the 'out' widget
            # Call the function
            data = variables.data.copy()
            bins = json.loads(bins_fdm.value)
            figure_size = (figure_size_x_fdm.value, figure_size_y_fdm.value)
            try:
                # Use json.loads to convert the entered string to a list
                range_sequence = json.loads(range_sequence_fdm.value)
                range_mc = json.loads(range_mc_fdm.value)
                range_detx = json.loads(range_detx_fdm.value)
                range_dety = json.loads(range_dety_fdm.value)
                range_x = json.loads(range_x_fdm.value)
                range_y = json.loads(range_y_fdm.value)
                range_z = json.loads(range_z_fdm.value)
                range_vol = json.loads(range_vol_fdm.value)
                if range_sequence == [0, 0]:
                    range_sequence = []
                if range_mc == [0, 0]:
                    range_mc = []
                if range_detx == [0, 0] or range_dety == [0, 0]:
                    range_detx = []
                    range_dety = []

                if range_x == [0, 0] or range_y == [] or range_z == [0, 0]:
                    range_x = []
                    range_y = []
                    range_z = []
                if range_vol == [0, 0]:
                    range_vol = []
            except json.JSONDecodeError:
                # Handle invalid input
                print(f"Invalid range input")
            data_loadcrop.plot_crop_fdm(data['x_det (cm)'].to_numpy(), data['y_det (cm)'].to_numpy(), bins,
                                        frac_fdm_widget.value, axis_mode_fdm.value, figure_size,
                                        variables, range_sequence,
                                        range_mc, range_detx, range_dety, range_x, range_y, range_z, range_vol,
                                        False, False, 'circle',
                                        save_fdm_widget.value, figname_fdm_widget.value)

        # Enable the button when the code is finished
        plot_fdm_button.disabled = False

    #############
    figname_3d = widgets.Text(value='3d_plot')
    rotary_fig_save_p3 = widgets.Dropdown(options=[('False', False), ('True', True)])
    element_percentage_p3 = widgets.Textarea(value=element_percentage)
    opacity = widgets.FloatText(value=0.5, min=0, max=1, step=0.1)
    save_3d = widgets.Dropdown(options=[('True', True), ('False', False)], value=False)
    ions_individually_plots = widgets.Dropdown(options=[('True', True), ('False', False)], value=False)
    make_gif_p3 = widgets.Dropdown(options=[('True', True), ('False', False)], value=False)
    make_evap_3d = widgets.Dropdown(options=[('True', True), ('False', False)], value=False)
    plot_3d_button.on_click(lambda b: plot_3d(b, variables, out))
    range_sequence_3d = widgets.Textarea(value='[0,0]')
    range_detx_3d = widgets.Textarea(value='[0,0]')
    range_dety_3d = widgets.Textarea(value='[0,0]')
    range_mc_3d = widgets.Textarea(value='[0,0]')
    range_x_3d = widgets.Textarea(value='[0,0]')
    range_y_3d = widgets.Textarea(value='[0,0]')
    range_z_3d = widgets.Textarea(value='[0,0]')
    range_vol_3d = widgets.Textarea(value='[0,0]')

    def plot_3d(b, variables, out):
        plot_3d_button.disabled = True
        with out:
            try:
                # Use json.loads to convert the entered string to a list
                range_sequence = json.loads(range_sequence_3d.value)
                range_mc = json.loads(range_mc_3d.value)
                range_detx = json.loads(range_detx_3d.value)
                range_dety = json.loads(range_dety_3d.value)
                range_x = json.loads(range_x_3d.value)
                range_y = json.loads(range_y_3d.value)
                range_z = json.loads(range_z_3d.value)
                range_vol = json.loads(range_vol_3d.value)
                if range_sequence == [0, 0]:
                    range_sequence = []
                if range_mc == [0, 0]:
                    range_mc = []
                if range_detx == [0, 0] or range_dety == [0, 0]:
                    range_detx = []
                    range_dety = []

                if range_x == [0, 0] or range_y == [] or range_z == [0, 0]:
                    range_x = []
                    range_y = []
                    range_z = []
                if range_vol == [0, 0]:
                    range_vol = []
            except json.JSONDecodeError:
                # Handle invalid input
                print(f"Invalid range input")

            element_percentage_dic = ast.literal_eval(element_percentage_p3.value)
            # Iterate through the 'element' column
            element_percentage_list = []
            for row_elements in variables.range_data['element']:
                max_value = 0.1  # Default value if no matching element is found
                for element in row_elements:
                    if element in element_percentage_dic:
                        max_value = element_percentage_dic[element]
                element_percentage_list.append(max_value)

            reconstruction.reconstruction_plot(variables, element_percentage_list, opacity.value,
                                               rotary_fig_save_p3.value, figname_3d.value,
                                               save_3d.value, make_gif_p3.value, make_evap_3d.value, range_sequence,
                                               range_mc, range_detx, range_dety, range_x, range_y, range_z, range_vol,
                                               ions_individually_plots.value)

        plot_3d_button.disabled = False

    #############
    max_tof_mc_widget = widgets.FloatText(value=variables.max_tof)
    frac_mc_widget = widgets.FloatText(value=1.0)
    bins_x_mc = widgets.IntText(value=1200)
    bins_y_mc = widgets.IntText(value=800)
    figure_size_x_mc = widgets.FloatText(value=7.0)
    figure_size_y_mc = widgets.FloatText(value=3.0)
    draw_rect_mc_widget = widgets.fixed(False)
    data_crop_mc_widget = widgets.fixed(True)
    pulse_plot_mc_widget = widgets.Dropdown(options=[('False', False), ('True', True)], value=False)
    dc_plot_mc_widget = widgets.Dropdown(options=[('True', True), ('False', False)], value=True)
    pulse_mode_mc_widget = widgets.Dropdown(options=[('voltage', 'voltage'), ('laser', 'laser')], value='voltage')
    save_mc_widget = widgets.Dropdown(options=[('True', True), ('False', False)], value=False)
    figname_mc_widget = widgets.Text(value='hist_ini')

    plot_experiment_button.on_click(lambda b: plot_experimetns_hitstory(b, variables, out))

    def plot_experimetns_hitstory(b, variables, out):
        plot_experiment_button.disabled = True
        with out:
            data = variables.data.copy()
            variables = variables
            max_tof = max_tof_mc_widget.value
            frac = frac_mc_widget.value

            # Get the values from the editable widgets and create tuples
            bins = (bins_x_mc.value, bins_y_mc.value)
            figure_size = (figure_size_x_mc.value, figure_size_y_mc.value)

            draw_rect = draw_rect_mc_widget.value
            data_crop = data_crop_mc_widget.value
            pulse_plot = pulse_plot_mc_widget.value
            dc_plot = dc_plot_mc_widget.value
            pulse_mode = pulse_mode_mc_widget.value
            save = save_mc_widget.value
            figname = figname_mc_widget.value

            with out:  # Capture the output within the 'out' widget
                # Call the actual function with the obtained values
                data_loadcrop.plot_crop_experiment_history(data, variables, max_tof, frac, bins, figure_size,
                                                           draw_rect, data_crop, pulse_plot, dc_plot,
                                                           pulse_mode, save, figname)
        plot_experiment_button.disabled = False

    #############
    element_percentage_ph = widgets.Textarea(value=element_percentage)
    figname_heatmap = widgets.Text(value='hitmap')
    save_heatmap = widgets.Dropdown(options=[('True', True), ('False', False)], value=False)
    figure_mc_size_x_heatmap = widgets.FloatText(value=5.0)
    figure_mc_size_y_heatmap = widgets.FloatText(value=5.0)
    range_sequence_heatmap = widgets.Textarea(value='[0,0]')
    range_detx_heatmap = widgets.Textarea(value='[0,0]')
    range_dety_heatmap = widgets.Textarea(value='[0,0]')
    range_mc_heatmap = widgets.Textarea(value='[0,0]')
    range_x_heatmap = widgets.Textarea(value='[0,0]')
    range_y_heatmap = widgets.Textarea(value='[0,0]')
    range_z_heatmap = widgets.Textarea(value='[0,0]')
    range_vol_heatmap = widgets.Textarea(value='[0,0]')

    plot_heatmap_button.on_click(lambda b: plot_heatmap(b, variables, out))

    def plot_heatmap(b, variables, out):
        plot_heatmap_button.disabled = True
        figure_size = (figure_mc_size_x_heatmap.value, figure_mc_size_y_heatmap.value)
        with out:
            try:
                # Use json.loads to convert the entered string to a list
                range_sequence = json.loads(range_sequence_heatmap.value)
                range_mc = json.loads(range_mc_heatmap.value)
                range_detx = json.loads(range_detx_heatmap.value)
                range_dety = json.loads(range_dety_heatmap.value)
                range_x = json.loads(range_x_heatmap.value)
                range_y = json.loads(range_y_heatmap.value)
                range_z = json.loads(range_z_heatmap.value)
                range_vol = json.loads(range_vol_heatmap.value)
                if range_sequence == [0, 0]:
                    range_sequence = []
                if range_mc == [0, 0]:
                    range_mc = []
                if range_detx == [0, 0] or range_dety == [0, 0]:
                    range_detx = []
                    range_dety = []

                if range_x == [0, 0] or range_y == [] or range_z == [0, 0]:
                    range_x = []
                    range_y = []
                    range_z = []
                if range_vol == [0, 0]:
                    range_vol = []
            except json.JSONDecodeError:
                # Handle invalid input
                print(f"Invalid range input")
            element_percentage_dic = ast.literal_eval(element_percentage_p3.value)
            # Iterate through the 'element' column
            element_percentage_list = []
            for row_elements in variables.range_data['element']:
                max_value = 0.1  # Default value if no matching element is found
                for element in row_elements:
                    if element in element_percentage_dic and element_percentage_dic[element] > max_value:
                        max_value = element_percentage_dic[element]
                element_percentage_list.append(max_value)
            reconstruction.heatmap(variables, element_percentage_list, range_sequence, range_mc, range_detx,
                                   range_dety, range_x, range_y, range_z, range_vol,
                                   figname_heatmap.value, figure_sie=figure_size, save=save_heatmap.value)
        plot_heatmap_button.disabled = False

    #############

    points_per_frame_anim = widgets.IntText(value=50000)
    selected_area_temporally_anim = widgets.Dropdown(options=[('False', False), ('True', True)])
    selected_area_specially_anim = widgets.Dropdown(options=[('False', False), ('True', True)])
    figname_anim = widgets.Text(value='detector_animation')
    figure_size_x_anim = widgets.FloatText(value=5.0)
    figure_size_y_anim = widgets.FloatText(value=5.0)
    ranged_anim = widgets.Dropdown(options=[('False', False), ('True', True)])

    plot_animated_heatmap_button = widgets.Button(description="plot animated heatmap")

    def plot_animated_heatmap(b, variables, out):

        points_per_frame = points_per_frame_anim.value
        selected_area_temporally = selected_area_temporally_anim.value
        selected_area_specially = selected_area_specially_anim.value
        figure_name = figname_anim.value
        figure_size = (figure_size_x_anim.value, figure_size_y_anim.value)
        ranged = ranged_anim.value

        plot_animated_heatmap_button.disabled = True
        with out:
            reconstruction.detector_animation(variables, points_per_frame, ranged, selected_area_specially,
                                              selected_area_temporally, figure_name,
                                              figure_size, save=True)
            display(HTML(variables.animation_detector_html))
        plot_animated_heatmap_button.disabled = False

    plot_animated_heatmap_button.on_click(lambda b: plot_animated_heatmap(b, variables, out))

    #############

    element_percentage_pp = widgets.Textarea(value=element_percentage)
    x_or_y_pp = widgets.Dropdown(options=['x', 'y'], value='x')
    figname_p = widgets.Text(value='projection')
    save_projection = widgets.Dropdown(options=[('True', True), ('False', False)], value=False)
    figure_mc_size_x_projection = widgets.FloatText(value=5.0)
    figure_mc_size_y_projection = widgets.FloatText(value=5.0)
    range_sequence_pp = widgets.Textarea(value='[0,0]')
    range_detx_pp = widgets.Textarea(value='[0,0]')
    range_dety_pp = widgets.Textarea(value='[0,0]')
    range_mc_pp = widgets.Textarea(value='[0,0]')
    range_x_pp = widgets.Textarea(value='[0,0]')
    range_y_pp = widgets.Textarea(value='[0,0]')
    range_z_pp = widgets.Textarea(value='[0,0]')
    range_vol_pp = widgets.Textarea(value='[0,0]')

    plot_projection_button.on_click(lambda b: plot_projection(b, variables, out))

    def plot_projection(b, variables, out):
        plot_projection_button.disabled = True
        figure_size = (figure_mc_size_x_projection.value, figure_mc_size_y_projection.value)
        with out:
            try:
                # Use json.loads to convert the entered string to a list
                range_sequence = json.loads(range_sequence_pp.value)
                range_mc = json.loads(range_mc_pp.value)
                range_detx = json.loads(range_detx_pp.value)
                range_dety = json.loads(range_dety_pp.value)
                range_x = json.loads(range_x_pp.value)
                range_y = json.loads(range_y_pp.value)
                range_z = json.loads(range_z_pp.value)
                range_vol = json.loads(range_vol_pp.value)
                if range_sequence == [0, 0]:
                    range_sequence = []
                if range_mc == [0, 0]:
                    range_mc = []
                if range_detx == [0, 0] or range_dety == [0, 0]:
                    range_detx = []
                    range_dety = []

                if range_x == [0, 0] or range_y == [] or range_z == [0, 0]:
                    range_x = []
                    range_y = []
                    range_z = []
                if range_vol == [0, 0]:
                    range_vol = []
            except json.JSONDecodeError:
                # Handle invalid input
                print(f"Invalid range input")

            element_percentage_dic = ast.literal_eval(element_percentage_p3.value)
            # Iterate through the 'element' column
            element_percentage_list = []
            for row_elements in variables.range_data['element']:
                max_value = 0.1  # Default value if no matching element is found
                for element in row_elements:
                    if element in element_percentage_dic and element_percentage_dic[element] > max_value:
                        max_value = element_percentage_dic[element]
                element_percentage_list.append(max_value)

            reconstruction.projection(variables, element_percentage_list, range_sequence, range_mc, range_detx,
                                      range_dety, range_x, range_y, range_z, range_vol, x_or_y_pp.value,
                                      figname_p.value, figure_size, save_projection.value)
        plot_projection_button.disabled = False

    #############

    first_axis_d = widgets.Dropdown(options=['x', 'y', 'z'], value='x')
    second_axis_d = widgets.Dropdown(options=['x', 'y', 'z'], value='y')
    composition_d = widgets.Textarea(value="['Al']")
    roi_d = widgets.Textarea(value='[0,0,0]')
    bins2d_hist = widgets.Textarea(value='[0.5]')
    percentage_2d_hist = widgets.FloatText(value=1.0)
    save_2d_hist = widgets.Dropdown(options=[('True', True), ('False', False)], value=False)
    z_weight_2d_hist = widgets.Dropdown(options=[('False', False), ('True', True)], value=False)
    log_d = widgets.Dropdown(options=[('True', True), ('False', False)])
    axis_mode_d = widgets.Dropdown(options=['normal', 'scalebar'], value='normal')
    # plasma, viridis, inferno, cividis, coolwarm, twilight, cubehelix
    cmap_d = widgets.Dropdown(options=['plasma', 'viridis', 'inferno', 'cividis', 'coolwarm', 'twilight', 'cubehelix'])
    normalize_d = widgets.Dropdown(options=[('False', False), ('True', True)], value=False)
    normalize_axes_d = widgets.Dropdown(options=[('False', False), ('True', True)], value=False)
    figname_2d_hist = widgets.Text(value='density_map')
    figure_size_x_2d_hist = widgets.FloatText(value=5.0)
    figure_size_y_2d_hist = widgets.FloatText(value=4.0)
    range_sequence_2d_hist = widgets.Textarea(value='[0,0]')
    range_detx_2d_hist = widgets.Textarea(value='[0,0]')
    range_dety_2d_hist = widgets.Textarea(value='[0,0]')
    range_mc_2d_hist = widgets.Textarea(value='[0,0]')
    range_x_2d_hist = widgets.Textarea(value='[0,0]')
    range_y_2d_hist = widgets.Textarea(value='[0,0]')
    range_z_2d_hist = widgets.Textarea(value='[0,0]')
    range_vol_2d_hist = widgets.Textarea(value='[0,0]')

    density_map_button.on_click(lambda b: density_map_plot(b, variables, out))

    def density_map_plot(b, variables, out):
        density_map_button.disabled = True
        with out:  # Capture the output within the 'out' widget
            if first_axis_d.value == 'x' and second_axis_d.value == 'y':
                x = variables.x
                y = variables.y
                axis_d = ['x', 'y']
            elif first_axis_d.value == 'y' and second_axis_d.value == 'x':
                x = variables.y
                y = variables.x
                axis_d = ['y', 'x']
            elif first_axis_d.value == 'x' and second_axis_d.value == 'z':
                x = variables.x
                y = variables.z
                axis_d = ['x', 'z']
            elif first_axis_d.value == 'z' and second_axis_d.value == 'x':
                x = variables.z
                y = variables.x
                axis_d = ['z', 'x']
            elif first_axis_d.value == 'y' and second_axis_d.value == 'z':
                x = variables.y
                y = variables.z
                axis_d = ['y', 'z']
            elif first_axis_d.value == 'z' and second_axis_d.value == 'y':
                x = variables.z
                y = variables.y
                axis_d = ['z', 'y']
            else:
                raise ValueError('Invalid axis')

            bins = json.loads(bins2d_hist.value)
            percentage = percentage_2d_hist.value
            composition = ast.literal_eval(composition_d.value)
            roi = ast.literal_eval(roi_d.value)
            figure_name = figname_2d_hist.value + '_' + first_axis_d.value + '_' + second_axis_d.value
            save = save_2d_hist.value
            figure_size = (figure_size_x_2d_hist.value, figure_size_y_2d_hist.value)
            try:
                # Use json.loads to convert the entered string to a list
                range_sequence = json.loads(range_sequence_2d_hist.value)
                range_mc = json.loads(range_mc_2d_hist.value)
                range_detx = json.loads(range_detx_2d_hist.value)
                range_dety = json.loads(range_dety_2d_hist.value)
                range_x = json.loads(range_x_2d_hist.value)
                range_y = json.loads(range_y_2d_hist.value)
                range_z = json.loads(range_z_2d_hist.value)
                range_vol = json.loads(range_vol_2d_hist.value)
                if range_sequence == [0, 0]:
                    range_sequence = []
                if range_mc == [0, 0]:
                    range_mc = []
                if range_detx == [0, 0] or range_dety == [0, 0]:
                    range_detx = []
                    range_dety = []

                if range_x == [0, 0] or range_y == [] or range_z == [0, 0]:
                    range_x = []
                    range_y = []
                    range_z = []
                if range_vol == [0, 0]:
                    range_vol = []
            except json.JSONDecodeError:
                # Handle invalid input
                print(f"Invalid range input")
            density_map.plot_density_map(x, y, z_weight_2d_hist.value, log_d.value, bins, percentage, axis_mode_d.value,
                        figure_size, variables, composition, roi, range_sequence, range_mc, range_detx,
                        range_dety, range_x, range_y, range_z, range_vol, False, False, 'circle', axis_d, save,
                        figure_name, cmap_d.value, normalize_d.value, normalize_axes_d.value)
        density_map_button.disabled = False

    #############
    first_axis_sdm = widgets.Dropdown(options=['x', 'y', 'z'], value='z')
    second_axis_sdm = widgets.Dropdown(options=['x', 'y', 'z'], value='y')
    sdm_mode = widgets.Dropdown(options=['1D', '2D'], value='1D')
    normalized_sdm = widgets.Dropdown(options=[('False', False), ('True', True)], value=False)
    log_sdm = widgets.Dropdown(options=[('True', True), ('False', False)], value=True)
    roi_sdm = widgets.Textarea(value='[0,0,0.5]')
    unique_ions = variables.range_data['name'].unique()
    composition_sdm_list_i = []
    composition_sdm_list_j = []
    for ion in unique_ions:
        composition_sdm_list_i.append(widgets.Checkbox(value=False, description=ion))
        composition_sdm_list_j.append(widgets.Checkbox(value=False, description=ion))

    i_composition_sdm = composition_sdm_list_i
    j_composition_sdm = composition_sdm_list_j
    theta_x_sdm = widgets.FloatText(value=0.0)
    phi_y_sdm = widgets.FloatText(value=0.0)
    percentage_sdm = widgets.FloatText(value=1.0)
    range_sequence_sdm = widgets.Textarea(value='[0,0]')
    range_detx_sdm = widgets.Textarea(value='[0,0]')
    range_dety_sdm = widgets.Textarea(value='[0,0]')
    range_mc_sdm = widgets.Textarea(value='[0,0]')
    range_x_sdm = widgets.Textarea(value='[0,0]')
    range_y_sdm = widgets.Textarea(value='[0,0]')
    range_z_sdm = widgets.Textarea(value='[0,0]')
    range_vol_sdm = widgets.Textarea(value='[0,0]')
    z_cut_sdm = widgets.Dropdown(options=[('False', False), ('True', True)], value=True)
    plot_roi_sdm = widgets.Dropdown(options=[('False', False), ('True', True)], value=True)
    bins_size_sdm = widgets.FloatText(value=0.02)
    save_sdm = widgets.Dropdown(options=[('True', True), ('False', False)], value=False)
    plot_mode_sdm = widgets.Dropdown(options=[('line', 'line'), ('bar', 'bar')])
    figname_2d_sdm = widgets.Text(value='sdm')
    figure_size_x_2d_sdm = widgets.FloatText(value=10.0)
    figure_size_y_2d_sdm = widgets.FloatText(value=4.0)

    plot_sdm_button.on_click(lambda b: plot_sdm(b, variables, out))

    def plot_sdm(b, variables, out):
        try:
            # Use json.loads to convert the entered string to a list
            range_sequence = json.loads(range_sequence_sdm.value)
            range_mc = json.loads(range_mc_sdm.value)
            range_detx = json.loads(range_detx_sdm.value)
            range_dety = json.loads(range_dety_sdm.value)
            range_x = json.loads(range_x_sdm.value)
            range_y = json.loads(range_y_sdm.value)
            range_z = json.loads(range_z_sdm.value)
            range_vol = json.loads(range_vol_sdm.value)
            if range_sequence == [0, 0]:
                range_sequence = []
            if range_mc == [0, 0]:
                range_mc = []
            if range_detx == [0, 0] or range_dety == [0, 0]:
                range_detx = []
                range_dety = []

            if range_x == [0, 0] or range_y == [] or range_z == [0, 0]:
                range_x = []
                range_y = []
                range_z = []
            if range_vol == [0, 0]:
                range_vol = []
        except json.JSONDecodeError:
            # Handle invalid input
            print(f"Invalid range input")

        particles = np.vstack((variables.x, variables.y, variables.z)).T
        figure_size = (figure_size_x_2d_sdm.value, figure_size_y_2d_sdm.value)
        if sdm_mode.value == '1D':
            axis_s = [first_axis_sdm.value]
        elif sdm_mode.value == '2D':
            axis_s = [first_axis_sdm.value, second_axis_sdm.value]
        plot_sdm_button.disabled = True
        roi = ast.literal_eval(roi_sdm.value)
        with out:
            if 'x' in axis_s and 'y' in axis_s:
                axis_sdm = ['x', 'y']
            elif 'y' in axis_s and 'x' in axis_s:
                axis_sdm = ['y', 'x']
            elif 'x' in axis_s and 'z' in axis_s:
                axis_sdm = ['x', 'z']
            elif 'z' in axis_s and 'x' in axis_s:
                axis_sdm = ['z', 'x']
            elif 'y' in axis_s and 'z' in axis_s:
                axis_sdm = ['y', 'z']
            elif 'z' in axis_s and 'y' in axis_s:
                axis_sdm = ['z', 'y']
            elif 'x' in axis_s:
                axis_sdm = ['x']
            elif 'y' in axis_s:
                axis_sdm = ['y']
            elif 'z' in axis_s:
                axis_sdm = ['z']
            else:
                raise ValueError('Invalid axis')
            i_composition = [devom.value for devom in i_composition_sdm]
            j_composition = [devom.value for devom in j_composition_sdm]
            i_composition_filtered = [
                checkbox.description for checkbox, include in zip(i_composition_sdm, i_composition) if include
            ]
            j_composition_filtered = [
                checkbox.description for checkbox, include in zip(j_composition_sdm, j_composition) if include
            ]
            sdm.sdm(particles, bins_size_sdm.value, variables, roi, z_cut_sdm.value, normalized_sdm.value,
                    plot_mode_sdm.value, True, save_sdm.value, figure_size, figname_2d_sdm.value, sdm_mode.value,
                    axis_sdm, i_composition_filtered, j_composition_filtered, plot_roi_sdm.value, theta_x_sdm.value,
                    phi_y_sdm.value, log_sdm.value, percentage_sdm.value,
                    range_sequence, range_mc, range_detx, range_dety, range_x, range_y, range_z, range_vol)
        plot_sdm_button.disabled = False

    ##############
    normalized_rdf = widgets.Dropdown(options=[('True', True), ('False', False)], value=False)
    reference_point_rdf = widgets.Textarea(value='[0,0,0]')
    box_dimension_rdf = widgets.Textarea(value='[1,1,1]')
    dr_rdf = widgets.FloatText(value=0.1)
    rdf_cutoff = widgets.FloatText(value=0.9)
    save_rdf = widgets.Dropdown(options=[('True', True), ('False', False)], value=False)
    figname_2d_rdf = widgets.Text(value='rdf')
    figure_size_x_2d_rdf = widgets.FloatText(value=5.0)
    figure_size_y_2d_rdf = widgets.FloatText(value=4.0)
    range_sequence_rdf = widgets.Textarea(value='[0,0]')

    plot_rdf_button.on_click(lambda b: plot_rdf(b, variables, out))

    def plot_rdf(b, variables, out):
        try:
            # Use json.loads to convert the entered string to a list
            reference_point = json.loads(reference_point_rdf.value)
            box_dimension = json.loads(box_dimension_rdf.value)
            range_sequence = json.loads(range_sequence_rdf.value)
        except json.JSONDecodeError:
            # Handle invalid input
            print(f"Invalid range input")
        if range_sequence != [0, 0]:
            mask_sequence = np.zeros_like(variables.dld_x_det, dtype=bool)
            mask_sequence[range_sequence[0]:range_sequence[1]] = True
        else:
            mask_sequence = np.ones_like(variables.dld_x_det, dtype=bool)
        particles = np.vstack((variables.x[mask_sequence], variables.y[mask_sequence], variables.z[mask_sequence])).T
        figure_size_rdf = (figure_size_x_2d_rdf.value, figure_size_y_2d_rdf.value)
        plot_rdf_button.disabled = True
        with out:
            rdf.rdf(particles, dr_rdf.value, variables, rcutoff=rdf_cutoff.value, normalize=normalized_rdf.value,
                    reference_point=reference_point, box_dimensions=box_dimension, plot=True, save=save_rdf.value,
                    figure_size=figure_size_rdf, figname=figname_2d_rdf.value)

        plot_rdf_button.disabled = False

    #############
    figname_3d_iso = widgets.Text(value='3d_plot_iso')
    rotary_fig_save_p3_iso = widgets.Dropdown(options=[('False', False), ('True', True)])
    element_percentage_p3_iso = widgets.Textarea(value=element_percentage)
    opacity_iso = widgets.FloatText(value=0.5, min=0, max=1, step=0.1)
    save_3d_iso = widgets.Dropdown(options=[('True', True), ('False', False)], value=False)
    ions_individually_plots_iso = widgets.Dropdown(options=[('True', True), ('False', False)], value=False)
    make_gif_p3_iso = widgets.Dropdown(options=[('True', True), ('False', False)], value=False)
    plot_3d_button_iso.on_click(lambda b: plot_3d_iso(b, variables, out))
    isosurface_dic_p3_iso = widgets.Textarea(value="{Al: [3,3,3]}")
    detailed_isotope_charge_3d_iso = widgets.Dropdown(options=[('False', False), ('True', True)], value=False)
    only_iso_3d_iso = widgets.Dropdown(options=[('False', False), ('True', True)], value=False)
    range_sequence_3d_iso = widgets.Textarea(value='[0,0]')
    range_detx_3d_iso = widgets.Textarea(value='[0,0]')
    range_dety_3d_iso = widgets.Textarea(value='[0,0]')
    range_mc_3d_iso = widgets.Textarea(value='[0,0]')
    range_x_3d_iso = widgets.Textarea(value='[0,0]')
    range_y_3d_iso = widgets.Textarea(value='[0,0]')
    range_z_3d_iso = widgets.Textarea(value='[0,0]')
    range_vol_3d_iso = widgets.Textarea(value='[0,0]')

    def plot_3d_iso(b, variables, out):
        plot_3d_button_iso.disabled = True
        with out:
            try:
                # Use json.loads to convert the entered string to a list
                range_sequence = json.loads(range_sequence_3d_iso.value)
                range_mc = json.loads(range_mc_3d_iso.value)
                range_detx = json.loads(range_detx_3d_iso.value)
                range_dety = json.loads(range_dety_3d_iso.value)
                range_x = json.loads(range_x_3d_iso.value)
                range_y = json.loads(range_y_3d_iso.value)
                range_z = json.loads(range_z_3d_iso.value)
                range_vol = json.loads(range_vol_3d_iso.value)
                if range_sequence == [0, 0]:
                    range_sequence = []
                if range_mc == [0, 0]:
                    range_mc = []
                if range_detx == [0, 0] or range_dety == [0, 0]:
                    range_detx = []
                    range_dety = []

                if range_x == [0, 0] or range_y == [] or range_z == [0, 0]:
                    range_x = []
                    range_y = []
                    range_z = []
                if range_vol == [0, 0]:
                    range_vol = []
            except json.JSONDecodeError:
                # Handle invalid input
                print(f"Invalid range input")
            # Use regular expressions to add quotes around keys
            formatted_string = re.sub(r'(\w+):', r'"\1":', isosurface_dic_p3_iso.value)
            # Convert to dictionary
            isosurface_dic_p3_iso_value = ast.literal_eval(formatted_string)

            element_percentage_dic = ast.literal_eval(element_percentage_p3_iso.value)
            # Iterate through the 'element' column
            element_percentage_list_iso = []
            for row_elements in variables.range_data['element']:
                max_value = 0.1  # Default value if no matching element is found
                for element in row_elements:
                    if element in element_percentage_dic:
                        max_value = element_percentage_dic[element]
                element_percentage_list_iso.append(max_value)

            iso_surface.reconstruction_plot(variables, element_percentage_list_iso, opacity_iso.value,
                                               rotary_fig_save_p3_iso.value, figname_3d_iso.value,
                                               save_3d_iso.value, make_gif_p3_iso.value,
                                               range_sequence,
                                               range_mc, range_detx, range_dety, range_x, range_y, range_z, range_vol,
                                               ions_individually_plots_iso.value,
                                               max_num_ions=None, min_num_ions=None,
                                               isosurface_dic=isosurface_dic_p3_iso_value,
                                               detailed_isotope_charge=detailed_isotope_charge_3d_iso.value,
                                               only_iso=only_iso_3d_iso.value)

        plot_3d_button_iso.disabled = False

    #############
    row_index = widgets.IntText(value=0, description='index row:')
    color_picker = widgets.ColorPicker(description='Select a color:')
    change_color.on_click(lambda b: change_color_m(b, variables, out))

    show_color.on_click(lambda b: show_color_ions(b, variables, out))

    def show_color_ions(b, variables, output):
        with output:
            clear_output(True)
            display(variables.range_data.style.applymap(ion_selection.display_color, subset=['color']))

    #############
    def change_color_m(b, variables, output):
        with output:
            selected_color = mcolors.to_hex(color_picker.value)
            variables.range_data.at[row_index.value, 'color'] = selected_color
            clear_output(True)
            display(variables.range_data.style.applymap(ion_selection.display_color, subset=['color']))

    clear_button.on_click(lambda b: clear(b, out))

    def clear(b, out):
        with out:
            clear_output(True)
            print('')

    tab0 = widgets.HBox([
        widgets.VBox([
            widgets.HBox([widgets.Label(value="Target:", layout=label_layout), target_mode]),
            widgets.HBox([widgets.Label(value="Peak find:", layout=label_layout), peaks_find]),
            widgets.HBox([widgets.Label(value="Peak find plot:", layout=label_layout), peak_find_plot]),
            widgets.HBox([widgets.Label(value="Plot ranged colors:", layout=label_layout), plot_ranged_colors]),
            widgets.HBox([widgets.Label(value="Plot ranged peak:", layout=label_layout), plot_ranged_peak]),
            widgets.HBox([widgets.Label(value="Print info:", layout=label_layout), print_info]),
            widgets.HBox([widgets.Label(value="Bins size:", layout=label_layout), bin_size_pm]),
            widgets.HBox([widgets.Label(value="Limit mc:", layout=label_layout), lim_mc_pm]),
            widgets.HBox([widgets.Label(value="log:", layout=label_layout), log_widget]),
            widgets.HBox([widgets.Label(value="Normalize:", layout=label_layout), normalize]),
            widgets.HBox([widgets.Label(value="MRP all(1%, 10%, 50%):", layout=label_layout), mrp_all]),
            widgets.HBox([widgets.Label(value="MRP Percent:", layout=label_layout), percent]),
            widgets.HBox([widgets.Label(value="Legend mode:", layout=label_layout), legend_widget]),
            widgets.HBox([widgets.Label(value="Peak prominence:", layout=label_layout), prominence]),
            widgets.HBox([widgets.Label(value="Peak distance:", layout=label_layout), distance]),
            widgets.HBox([widgets.Label(value="Background:", layout=label_layout), background_mc]),
            widgets.HBox([widgets.Label(value="Grid:", layout=label_layout), grid_mc]),
            widgets.HBox([plot_mc_button, clear_button])
        ]),
        widgets.VBox([
            widgets.HBox([widgets.Label(value="sequence range:", layout=label_layout), range_sequence_mc]),
            widgets.HBox([widgets.Label(value="mc range:", layout=label_layout), range_mc_mc]),
            widgets.HBox([widgets.Label(value="detx range:", layout=label_layout), range_detx_mc]),
            widgets.HBox([widgets.Label(value="dety range:", layout=label_layout), range_dety_mc]),
            widgets.HBox([widgets.Label(value="x range:", layout=label_layout), range_x_mc]),
            widgets.HBox([widgets.Label(value="y range:", layout=label_layout), range_y_mc]),
            widgets.HBox([widgets.Label(value="z range:", layout=label_layout), range_z_mc]),
            widgets.HBox([widgets.Label(value="voltage range:", layout=label_layout), range_vol_mc]),
            widgets.HBox([widgets.Label(value="Fig name:", layout=label_layout), figname_mc]),
            widgets.HBox([widgets.Label(value="Save fig:", layout=label_layout), save_mc]),
            widgets.HBox([widgets.Label(value="Fig size:", layout=label_layout),
                          widgets.HBox([figure_mc_size_x_mc, figure_mc_size_y_mc]),]),
        ])
        ])

    tab1 = widgets.VBox([
        widgets.HBox([widgets.Label(value='Max TOF:', layout=label_layout), max_tof_mc_widget]),
        widgets.HBox([widgets.Label(value='Fraction:', layout=label_layout), frac_mc_widget]),
        widgets.HBox([widgets.Label(value='Bins:', layout=label_layout), widgets.HBox([bins_x_mc, bins_y_mc])]),
        widgets.HBox([widgets.Label(value='Pulse Plot:', layout=label_layout), pulse_plot_mc_widget]),
        widgets.HBox([widgets.Label(value='DC Plot:', layout=label_layout), dc_plot_mc_widget]),
        widgets.HBox([widgets.Label(value='Pulse Mode:', layout=label_layout), pulse_mode_mc_widget]),
        widgets.HBox([widgets.Label(value='Fig name:', layout=label_layout), figname_mc_widget]),
        widgets.HBox([widgets.Label(value='Save:', layout=label_layout), save_mc_widget]),
        widgets.HBox([widgets.Label(value='Fig size:', layout=label_layout),
                      widgets.HBox([figure_size_x_mc, figure_size_y_mc])]),
        widgets.HBox([plot_experiment_button, clear_button]),
    ])

    tab2 = (widgets.HBox([
        widgets.VBox([
            widgets.HBox([widgets.Label(value='Fraction:', layout=label_layout), frac_fdm_widget]),
            widgets.HBox([widgets.Label(value='Bins:', layout=label_layout), bins_fdm]),
            widgets.HBox([widgets.Label(value='Axis mode:', layout=label_layout), axis_mode_fdm]),
            widgets.HBox([widgets.Label(value='Fig name:', layout=label_layout), figname_fdm_widget]),
            widgets.HBox([widgets.Label(value='Save:', layout=label_layout), save_fdm_widget]),
            widgets.HBox([widgets.Label(value='Fig size:', layout=label_layout),
                          widgets.HBox([figure_size_x_fdm, figure_size_y_fdm])]),
            widgets.HBox([plot_fdm_button, clear_button]),
        ]),
        widgets.VBox([
            widgets.HBox([widgets.Label(value="sequence range:", layout=label_layout), range_sequence_fdm]),
            widgets.HBox([widgets.Label(value='mc range:', layout=label_layout), range_mc_fdm]),
            widgets.HBox([widgets.Label(value='detx range:', layout=label_layout), range_detx_fdm]),
            widgets.HBox([widgets.Label(value='dety range:', layout=label_layout), range_dety_fdm]),
            widgets.HBox([widgets.Label(value='x range:', layout=label_layout), range_x_fdm]),
            widgets.HBox([widgets.Label(value='y range:', layout=label_layout), range_y_fdm]),
            widgets.HBox([widgets.Label(value='z range:', layout=label_layout), range_z_fdm]),
            widgets.HBox([widgets.Label(value="voltage range:", layout=label_layout), range_vol_fdm]),
        ])
    ]))

    tab3 = (widgets.HBox([
        widgets.VBox([
            widgets.HBox([widgets.Label(value='Element percentage:', layout=label_layout), element_percentage_p3]),
            widgets.HBox([widgets.Label(value='Opacity:', layout=label_layout), opacity]),
            widgets.HBox(
                [widgets.Label(value='Ions individually plots:', layout=label_layout), ions_individually_plots]),
            widgets.HBox([widgets.Label(value='Fig name:', layout=label_layout), figname_3d]),
            widgets.HBox([widgets.Label(value='Rotary save:', layout=label_layout), rotary_fig_save_p3]),
            widgets.HBox([widgets.Label(value='Save GIF:', layout=label_layout), make_gif_p3]),
            widgets.HBox([widgets.Label(value='Save evaporation GIF:', layout=label_layout), make_evap_3d]),
            widgets.HBox([widgets.Label(value='Save fig:', layout=label_layout), save_3d]),
            widgets.HBox([plot_3d_button, clear_button]),
        ]),
        widgets.VBox([
            widgets.HBox([widgets.Label(value="sequence range:", layout=label_layout), range_sequence_3d]),
            widgets.HBox([widgets.Label(value='Range mc:', layout=label_layout), range_mc_3d]),
            widgets.HBox([widgets.Label(value='Range detx:', layout=label_layout), range_detx_3d]),
            widgets.HBox([widgets.Label(value='Range dety:', layout=label_layout), range_dety_3d]),
            widgets.HBox([widgets.Label(value='Range x:', layout=label_layout), range_x_3d]),
            widgets.HBox([widgets.Label(value='Range y:', layout=label_layout), range_y_3d]),
            widgets.HBox([widgets.Label(value='Range z:', layout=label_layout), range_z_3d]),
            widgets.HBox([widgets.Label(value="voltage range:", layout=label_layout), range_vol_3d]),
        ])
    ]))

    tab4 = (widgets.HBox([
        widgets.VBox([
            widgets.HBox([widgets.Label(value='Element percentage:', layout=label_layout), element_percentage_ph]),
            widgets.HBox([widgets.Label(value='Fig name:', layout=label_layout), figname_heatmap]),
            widgets.HBox([widgets.Label(value='Save fig:', layout=label_layout), save_heatmap]),
            widgets.HBox([widgets.Label(value='Fig size:', layout=label_layout),
                          widgets.HBox([figure_mc_size_x_heatmap, figure_mc_size_y_heatmap])]),
            widgets.HBox([plot_heatmap_button, clear_button]),
        ]),
        widgets.VBox([
            widgets.HBox([widgets.Label(value="sequence range:", layout=label_layout), range_sequence_heatmap]),
            widgets.HBox([widgets.Label(value='range mc:', layout=label_layout), range_mc_heatmap]),
            widgets.HBox([widgets.Label(value='range detx:', layout=label_layout), range_detx_heatmap]),
            widgets.HBox([widgets.Label(value='range dety:', layout=label_layout), range_dety_heatmap]),
            widgets.HBox([widgets.Label(value='range x:', layout=label_layout), range_x_heatmap]),
            widgets.HBox([widgets.Label(value='range y:', layout=label_layout), range_y_heatmap]),
            widgets.HBox([widgets.Label(value='range z:', layout=label_layout), range_z_heatmap]),
            widgets.HBox([widgets.Label(value="voltage range:", layout=label_layout), range_vol_heatmap]),
        ])
    ]))

    tab5 = widgets.VBox([
        widgets.HBox([widgets.Label(value='Points per frame:', layout=label_layout), points_per_frame_anim]),
        widgets.HBox([widgets.Label(value='Ranged:', layout=label_layout), ranged_anim]),
        widgets.HBox([widgets.Label(value='Fig name:', layout=label_layout), figname_anim]),
        widgets.HBox([widgets.Label(value='Fig size:', layout=label_layout),
                      widgets.HBox([figure_size_x_anim, figure_size_y_anim])]),
        widgets.HBox([plot_animated_heatmap_button, clear_button]),
    ])

    tab6 = (widgets.HBox([
        widgets.VBox([
            widgets.HBox([widgets.Label(value='Element percentage:', layout=label_layout), element_percentage_pp]),
            widgets.HBox([widgets.Label(value='X or Y:', layout=label_layout), x_or_y_pp]),
            widgets.HBox([widgets.Label(value='Fig name:', layout=label_layout), figname_p]),
            widgets.HBox([widgets.Label(value='Save fig:', layout=label_layout), save_projection]),
            widgets.HBox([widgets.Label(value='Fig size:', layout=label_layout),
                          widgets.HBox([figure_mc_size_x_projection, figure_mc_size_y_projection])]),
            widgets.HBox([plot_projection_button, clear_button])
        ]),
        widgets.VBox([
            widgets.HBox([widgets.Label(value="sequence range:", layout=label_layout), range_sequence_pp]),
            widgets.HBox([widgets.Label(value='range mc:', layout=label_layout), range_mc_pp]),
            widgets.HBox([widgets.Label(value='range detx:', layout=label_layout), range_detx_pp]),
            widgets.HBox([widgets.Label(value='range dety:', layout=label_layout), range_dety_pp]),
            widgets.HBox([widgets.Label(value='range x:', layout=label_layout), range_x_pp]),
            widgets.HBox([widgets.Label(value='range y:', layout=label_layout), range_y_pp]),
            widgets.HBox([widgets.Label(value='range z:', layout=label_layout), range_z_pp]),
            widgets.HBox([widgets.Label(value="voltage range:", layout=label_layout), range_vol_pp]),
        ])
    ]))

    tab7 = (widgets.HBox([
        widgets.VBox([
            widgets.HBox([widgets.Label(value='First axis:', layout=label_layout), first_axis_d]),
            widgets.HBox([widgets.Label(value='Second axis:', layout=label_layout), second_axis_d]),
            widgets.HBox([widgets.Label(value='Composition:', layout=label_layout), composition_d]),
            widgets.HBox([widgets.Label(value='z weight:', layout=label_layout), z_weight_2d_hist]),
            widgets.HBox([widgets.Label(value='Bin size (nm):', layout=label_layout), bins2d_hist]),
            widgets.HBox([widgets.Label(value='Log:', layout=label_layout), log_d]),
            widgets.HBox([widgets.Label(value='Axis mode:', layout=label_layout), axis_mode_d]),
            widgets.HBox([widgets.Label(value='Cmap:', layout=label_layout), cmap_d]),
            widgets.HBox([widgets.Label(value='Normalize:', layout=label_layout), normalize_d]),
            widgets.HBox([widgets.Label(value='Normalize axes:', layout=label_layout), normalize_axes_d]),
            widgets.HBox([widgets.Label(value='Fig name:', layout=label_layout), figname_2d_hist]),
            widgets.HBox([widgets.Label(value='Save:', layout=label_layout), save_2d_hist]),
            widgets.HBox([widgets.Label(value='Fig size:', layout=label_layout),
                          widgets.HBox([figure_size_x_2d_hist, figure_size_y_2d_hist])]),

            widgets.HBox([density_map_button, clear_button]),
        ]),
        widgets.VBox([
            widgets.HBox([widgets.Label(value='Percentage:', layout=label_layout), percentage_2d_hist]),
            widgets.HBox([widgets.Label(value="sequence range:", layout=label_layout), range_sequence_2d_hist]),
            widgets.HBox([widgets.Label(value='mc range:', layout=label_layout), range_mc_2d_hist]),
            widgets.HBox([widgets.Label(value='detx range:', layout=label_layout), range_detx_2d_hist]),
            widgets.HBox([widgets.Label(value='dety range:', layout=label_layout), range_dety_2d_hist]),
            widgets.HBox([widgets.Label(value='x range:', layout=label_layout), range_x_2d_hist]),
            widgets.HBox([widgets.Label(value='y range:', layout=label_layout), range_y_2d_hist]),
            widgets.HBox([widgets.Label(value='z range:', layout=label_layout), range_z_2d_hist]),
            widgets.HBox([widgets.Label(value="voltage range:", layout=label_layout), range_vol_2d_hist]),
        ])
    ]))
    tab8 = (widgets.HBox([
        widgets.VBox([
            widgets.HBox([widgets.Label(value='First axis:', layout=label_layout), first_axis_sdm]),
            widgets.HBox([widgets.Label(value='Second axis:', layout=label_layout), second_axis_sdm]),
            widgets.HBox([widgets.Label(value='Mode:', layout=label_layout), sdm_mode]),
            widgets.HBox([widgets.Label(value='ROI:', layout=label_layout), roi_sdm]),
            widgets.HBox([widgets.Label(value='Z cut 1 nm:', layout=label_layout), z_cut_sdm]),
            widgets.HBox([widgets.Label(value='Theta x:', layout=label_layout), theta_x_sdm]),
            widgets.HBox([widgets.Label(value='Phi y:', layout=label_layout), phi_y_sdm]),
            widgets.HBox([widgets.Label(value='Bins size (nm):', layout=label_layout), bins_size_sdm]),
            widgets.HBox([widgets.Label(value='Plot ROI:', layout=label_layout), plot_roi_sdm]),
            widgets.HBox([widgets.Label(value='Normalized:', layout=label_layout), normalized_sdm]),
            widgets.HBox([widgets.Label(value='Log:', layout=label_layout), log_sdm]),
            widgets.HBox([widgets.Label(value='Plot mode:', layout=label_layout), plot_mode_sdm]),
            widgets.HBox([widgets.Label(value='Fig name:', layout=label_layout), figname_2d_sdm]),
            widgets.HBox([widgets.Label(value='Save:', layout=label_layout), save_sdm]),
            widgets.HBox([widgets.Label(value='Fig size:', layout=label_layout),
                          widgets.HBox([figure_size_x_2d_sdm, figure_size_y_2d_sdm])]),
            widgets.HBox([plot_sdm_button, clear_button]),
        ]),
        widgets.VBox([
            widgets.Label(value="Ions I for SDM:"),
            *i_composition_sdm,
        ]),
        widgets.VBox([
            widgets.Label(value="Ions J for SDM:"),
            *j_composition_sdm,
        ]),
        widgets.VBox([
            widgets.HBox([widgets.Label(value='Percentage:', layout=label_layout), percentage_sdm]),
            widgets.HBox([widgets.Label(value='range sequence:', layout=label_layout), range_sequence_sdm]),
            widgets.HBox([widgets.Label(value='range mc:', layout=label_layout), range_mc_sdm]),
            widgets.HBox([widgets.Label(value='range detx:', layout=label_layout), range_detx_sdm]),
            widgets.HBox([widgets.Label(value='range dety:', layout=label_layout), range_dety_sdm]),
            widgets.HBox([widgets.Label(value='range x:', layout=label_layout), range_x_sdm]),
            widgets.HBox([widgets.Label(value='range y:', layout=label_layout), range_y_sdm]),
            widgets.HBox([widgets.Label(value='range z:', layout=label_layout), range_z_sdm]),
            widgets.HBox([widgets.Label(value='range vol:', layout=label_layout), range_vol_sdm]),
        ])
    ]))

    tab9 = (widgets.HBox([
        widgets.VBox([
            widgets.HBox([widgets.Label(value='dr:', layout=label_layout), dr_rdf]),
            widgets.HBox([widgets.Label(value='rdf cutoff:', layout=label_layout), rdf_cutoff]),
            widgets.HBox([widgets.Label(value='Normalized:', layout=label_layout), normalized_rdf]),
            widgets.HBox([widgets.Label(value='Reference point:', layout=label_layout), reference_point_rdf]),
            widgets.HBox([widgets.Label(value='Fig name:', layout=label_layout), figname_2d_rdf]),
            widgets.HBox([widgets.Label(value='Save:', layout=label_layout), save_rdf]),
            widgets.HBox([widgets.Label(value='Fig size:', layout=label_layout),
                          widgets.HBox([figure_size_x_2d_rdf, figure_size_y_2d_rdf])]),
            widgets.HBox([plot_rdf_button, clear_button]),
        ]),
        widgets.VBox([
            widgets.HBox([widgets.Label(value='range sequence:', layout=label_layout), range_sequence_rdf]),

        ])
    ]))

    tab10 = (widgets.HBox([
        widgets.VBox([
            widgets.HBox([widgets.Label(value='Element percentage:', layout=label_layout), element_percentage_p3_iso]),
            widgets.HBox([widgets.Label(value='Opacity:', layout=label_layout), opacity_iso]),
            widgets.HBox([widgets.Label(value='Ions individually plots:', layout=label_layout), ions_individually_plots_iso]),
            widgets.HBox([widgets.Label(value='Isosurface dic:', layout=label_layout), isosurface_dic_p3_iso]),
            widgets.HBox([widgets.Label(value='Detailed isotope charge:', layout=label_layout),
                            detailed_isotope_charge_3d_iso]),
            widgets.HBox([widgets.Label(value='Only iso:', layout=label_layout), only_iso_3d_iso]),
            widgets.HBox([widgets.Label(value='Make GIF:', layout=label_layout), make_gif_p3_iso]),
            widgets.HBox([widgets.Label(value='Fig name:', layout=label_layout), figname_3d_iso]),
            widgets.HBox([widgets.Label(value='Save:', layout=label_layout), save_3d_iso]),
            widgets.HBox([widgets.Label(value='Rotary save:', layout=label_layout), rotary_fig_save_p3_iso]),
            widgets.HBox([plot_3d_button_iso, clear_button]),
        ]),
        widgets.VBox([
            widgets.HBox([widgets.Label(value="sequence range:", layout=label_layout), range_sequence_3d_iso]),
            widgets.HBox([widgets.Label(value='range mc:', layout=label_layout), range_mc_3d_iso]),
            widgets.HBox([widgets.Label(value='range detx:', layout=label_layout), range_detx_3d_iso]),
            widgets.HBox([widgets.Label(value='range dety:', layout=label_layout), range_dety_3d_iso]),
            widgets.HBox([widgets.Label(value='range x:', layout=label_layout), range_x_3d_iso]),
            widgets.HBox([widgets.Label(value='range y:', layout=label_layout), range_y_3d_iso]),
            widgets.HBox([widgets.Label(value='range z:', layout=label_layout), range_z_3d_iso]),
            widgets.HBox([widgets.Label(value="voltage range:", layout=label_layout), range_vol_3d_iso]),
        ])
    ]))

    tab11 = widgets.VBox([
        widgets.HBox([clear_button]),
    ])

    tab12 = widgets.VBox([
        widgets.HBox([widgets.Label(value='Index row:', layout=label_layout), row_index]),
        widgets.HBox([widgets.Label(value='Color:', layout=label_layout), color_picker]),
        widgets.VBox([show_color, change_color, clear_button]),
    ])

    if not colab:
        tab = widgets.Tab([tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11,
                                    tab12])
        tab.set_title(0, 'mc')
        tab.set_title(1, 'Experiment history')
        tab.set_title(2, 'FDM')
        tab.set_title(3, '3D')
        tab.set_title(4, 'Hitmap')
        tab.set_title(5, 'Animated hitmap')
        tab.set_title(6, 'Projection')

        tab.set_title(7, 'Disparity map')
        tab.set_title(8, 'SDM')
        tab.set_title(9, 'RDF')
        tab.set_title(10, 'Iso surface')
        tab.set_title(11, 'Clustering')
        tab.set_title(12, 'Change Color')

        out = Output()

        display(widgets.VBox([tab, out]))

    else:
        # Define the content for each tab
        content = [tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11,
                                          tab12]

        # Create buttons for each tab
        buttons = [widgets.Button(description=title) for title in [
            'mc', 'Experiment history', 'FDM', '3D', 'Hitmap', 'Animated hitmap',
            'Projection', 'Disparity map', 'SDM', 'RDF', 'Iso surface', 'Clustering', 'Change Color'
        ]]

        # Output widget to display the content
        out = widgets.Output()

        def on_button_click(index):
            def handler(change):
                with out:
                    clear_output(wait=True)
                    display(content[index])

            return handler

        # Attach click handlers to each button
        for i, button in enumerate(buttons):
            button.on_click(on_button_click(i))

        # Display buttons and the output widget
        display(widgets.HBox(buttons))
        display(out)
