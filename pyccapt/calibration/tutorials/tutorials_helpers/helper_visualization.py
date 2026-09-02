import ast
import json
import re

import ipywidgets as widgets
import matplotlib.colors as mcolors
import numpy as np
from IPython.display import display, clear_output, HTML
from ipywidgets import Output

from pyccapt.calibration import clustering
from pyccapt.calibration.core import mc_plot, ion_selection
from pyccapt.calibration.core.mc_plot_peak_helpers import (
    _auto_mrp_window_from_array,
    gaussian_mrp_report,
)
from pyccapt.calibration.data_tools import data_loadcrop
from pyccapt.calibration.reconstructions import reconstruction, sdm, rdf, density_map, iso_surface, proxigram
from pyccapt.calibration.reconstructions.species_display import (
    default_element_controls,
    resolve_element_controls,
)
from pyccapt.calibration.tutorials.tutorials_helpers.helper_mc_tof_calculator import (
    build_mc_tof_calculator_panel,
)
from pyccapt.calibration.tutorials.tutorials_helpers.helper_concentration_profile import (
    build_concentration_profile_panel,
)
from pyccapt.calibration.tutorials.tutorials_helpers.helper_peak_spectral_analysis import \
    build_peak_spectral_analysis_panel

# Define a layout for labels to make them a fixed width
label_layout = widgets.Layout(width='200px')


def call_visualization(variables, colab=False):
    if getattr(variables, 'range_data', None) is not None and 'name' not in variables.range_data.columns:

        def _to_neutral_label(raw_value):
            text = str(raw_value).strip().replace('$', '')
            text = re.sub(r'_\{([^}]*)\}', r'\1', text)
            text = re.sub(r'\^\{[^}]*\}', '', text)
            text = text.replace('{', '').replace('}', '')
            text = re.sub(r'\s*\d*[+-]+\s*$', '', text)
            return text.strip()

        if 'ion' in variables.range_data.columns:
            variables.range_data['name'] = variables.range_data['ion'].apply(_to_neutral_label)
        elif 'ion_name' in variables.range_data.columns:
            variables.range_data['name'] = variables.range_data['ion_name'].apply(_to_neutral_label)
        else:
            variables.range_data['name'] = [f'range_{idx}' for idx in range(len(variables.range_data))]
        print("Range table did not include 'name'; generated it from legacy ion labels.")

    plot_mc_button = widgets.Button(description='Plot mc', button_style='primary')
    plot_3d_button = widgets.Button(description='Plot 3D', button_style='primary')
    plot_3d_button_iso = widgets.Button(description='Plot 3D iso surface', button_style='primary')
    plot_heatmap_button = widgets.Button(description='Plot heatmap', button_style='primary')
    plot_projection_button = widgets.Button(description='Plot projection', button_style='primary')
    clear_button = widgets.Button(description='Clear plots', button_style='warning')
    plot_fdm_button = widgets.Button(description="Plot FDM", button_style='primary')
    plot_experiment_button = widgets.Button(description="Plot experiment history", button_style='primary')
    density_map_button = widgets.Button(description="Plot density map", button_style='primary')
    plot_sdm_button = widgets.Button(description='Plot SDM', button_style='primary')
    plot_rdf_button = widgets.Button(description='Plot RDF', button_style='primary')
    show_color = widgets.Button(description='Show color')
    change_color = widgets.Button(description='Change color')

    element_percentage = str(default_element_controls(variables.range_data, 0.01))
    element_alpha = str(default_element_controls(variables.range_data, 0.9))

    #############
    peak_find_plot = widgets.Dropdown(options=[('True', True), ('False', False)])
    peaks_find = widgets.Dropdown(options=[('True', True), ('False', False)])
    plot_ranged_colors = widgets.Dropdown(options=[('False', False), ('True', True)])
    plot_ranged_peak = widgets.Dropdown(options=[('False', False), ('True', True)])
    range_overlay_note = widgets.HTML(
        value="<span style='color:#666;'>Ranged colors and ranged peak labels are mutually exclusive. "
        "If you enable one, the other will be turned off.</span>"
    )
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
    background_mc = widgets.Dropdown(
        options=[
            ('None', None),
            ('aspls', 'aspls'),
            ('fabc', 'fabc'),
            ('manual@4', 'manual@4'),
            ('manual@100', 'manual@100'),
            ('manual', 'manual'),
        ]
    )
    figname_mc = widgets.Text(value='mc')
    figure_mc_size_x_mc = widgets.FloatText(value=9.0)
    figure_mc_size_y_mc = widgets.FloatText(value=5.0)
    save_mc = widgets.Dropdown(options=[('True', True), ('False', False)], value=False)
    grid_mc = widgets.Dropdown(options=[('False', False), ('True', True)])
    mrp_left_mc = widgets.FloatText(value=0.0)
    mrp_right_mc = widgets.FloatText(value=0.0)
    load_mrp_window_mc_button = widgets.Button(description='Load selection')
    gaussian_mrp_mc_button = widgets.Button(description='MRP')
    range_sequence_mc = widgets.Textarea(value='[0,0]')
    range_detx_mc = widgets.Textarea(value='[0,0]')
    range_dety_mc = widgets.Textarea(value='[0,0]')
    range_mc_mc = widgets.Textarea(value='[0,0]')
    range_x_mc = widgets.Textarea(value='[0,0]')
    range_y_mc = widgets.Textarea(value='[0,0]')
    range_z_mc = widgets.Textarea(value='[0,0]')
    range_vol_mc = widgets.Textarea(value='[0,0]')

    plot_mc_button.on_click(lambda b: plot_mc(b, variables, out))

    def _sync_ranged_overlay(change, source):
        if change.get('name') != 'value' or change.get('new') is not True:
            return
        if source == 'colors' and plot_ranged_peak.value:
            plot_ranged_peak.value = False
        elif source == 'peak' and plot_ranged_colors.value:
            plot_ranged_colors.value = False

    plot_ranged_colors.observe(lambda change: _sync_ranged_overlay(change, 'colors'), names='value')
    plot_ranged_peak.observe(lambda change: _sync_ranged_overlay(change, 'peak'), names='value')

    def plot_mc(b, variables, out):
        plot_mc_button.disabled = True
        figure_size = (figure_mc_size_x_mc.value, figure_mc_size_y_mc.value)
        with out:
            range_sequence = []
            range_mc = []
            range_detx = []
            range_dety = []
            range_x = []
            range_y = []
            range_z = []
            range_vol = []
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
                plot_mc_button.disabled = False
                return
            effective_plot_ranged_colors = plot_ranged_colors.value
            effective_plot_ranged_peak = plot_ranged_peak.value
            if effective_plot_ranged_colors and effective_plot_ranged_peak:
                print('Both ranged colors and ranged peak labels were selected. Using ranged colors overlay.')
                effective_plot_ranged_peak = False
            mc_plot.hist_plot(
                variables,
                bin_size_pm.value,
                log=log_widget.value,
                target=target_mode.value,
                normalize=normalize.value,
                prominence=prominence.value,
                distance=distance.value,
                percent=percent.value,
                selector='rect',
                figname=figname_mc.value,
                lim=lim_mc_pm.value,
                peaks_find=peaks_find.value,
                peaks_find_plot=peak_find_plot.value,
                plot_ranged_colors=effective_plot_ranged_colors,
                plot_ranged_peak=effective_plot_ranged_peak,
                mrp_all=mrp_all.value,
                background=background_mc.value,
                grid=grid_mc.value,
                range_sequence=range_sequence,
                range_mc=range_mc,
                range_detx=range_detx,
                range_dety=range_dety,
                range_x=range_x,
                range_y=range_y,
                range_z=range_z,
                range_vol=range_vol,
                print_info=print_info.value,
                figure_size=figure_size,
                save_fig=save_mc.value,
                legend_mode=legend_widget.value,
            )

        plot_mc_button.disabled = False

    def _resolve_visualization_hist_array():
        if target_mode.value == 'mc_uc':
            return variables.data['mc_uc (Da)']
        if target_mode.value == 'tof_c':
            return variables.data['t_c (ns)']
        if target_mode.value == 'tof':
            return variables.data['t (ns)']
        return variables.data['mc (Da)']

    def _auto_visualization_mrp_window(half_width=0.8):
        """Centre a window on the tallest peak in the current histogram."""
        result = _auto_mrp_window_from_array(
            _resolve_visualization_hist_array(), half_width=half_width
        )
        if result is None:
            return None
        left, right, _center = result
        return left, right

    def _resolve_visualization_gaussian_window():
        left = float(mrp_left_mc.value)
        right = float(mrp_right_mc.value)
        if right > left:
            return left, right, 'manual'
        if getattr(variables, 'selected_x2', 0) > getattr(variables, 'selected_x1', 0):
            return float(variables.selected_x1), float(variables.selected_x2), 'selection'
        auto = _auto_visualization_mrp_window()
        if auto is not None:
            return auto[0], auto[1], 'auto'
        return None

    def _print_visualization_gaussian_report(result):
        print('=' * 60)
        print('PEAK PROFILE MRP REPORT')
        print('=' * 60)
        print(f'MRP model: {result["recommended_label"]}')
        print(f'MRP bin size used: {result["bin_size"]} ({result["num_bins"]} bins)')
        print(f'Peak position: {result["peak_position"]:.4f}')
        print(f'Ions in range: {result["num_ions"]:,}')
        print(f'Recommended FWHM MRP: {result["formatted_recommended_mrp"][0]}')
        if result["window_warning"]:
            print(result["window_warning"])
        print()
        print('Gaussian fit MRP:' if result['gaussian_ok'] else 'Gaussian fit FAILED')
        if result['gaussian_ok']:
            print(f'  MRP(0.5)  = {result["formatted_gaussian_mrp"][0]}')
            print(f'  MRP(0.1)  = {result["formatted_gaussian_mrp"][1]}')
            print(f'  MRP(0.01) = {result["formatted_gaussian_mrp"][2]}')
        print()
        print('Voigt fit MRP:' if result['voigt_ok'] else 'Voigt fit FAILED')
        if result['voigt_ok']:
            print(f'  MRP(0.5)  = {result["formatted_voigt_mrp"][0]}')
            print(f'  MRP(0.1)  = {result["formatted_voigt_mrp"][1]}')
            print(f'  MRP(0.01) = {result["formatted_voigt_mrp"][2]}')
            print(f'  Voigt FWHM = {result["voigt_fwhm"]:.6f}')
        print()
        print('Asymmetric (err*expDecay) fit MRP:' if result.get('asymmetric_ok') else 'Asymmetric (err*expDecay) fit FAILED')
        if result.get('asymmetric_ok'):
            print(f'  MRP(0.5)  = {result["formatted_asymmetric_mrp"][0]}')
            print(f'  MRP(0.1)  = {result["formatted_asymmetric_mrp"][1]}')
            print(f'  MRP(0.01) = {result["formatted_asymmetric_mrp"][2]}')
            asym_fwhm = result.get('asymmetric_fwhm', float('nan'))
            if np.isfinite(asym_fwhm):
                print(f'  Asymmetric FWHM = {asym_fwhm:.6f}')
        print()
        print('Histogram-based MRP:')
        print(f'  MRP(0.5)  = {result["formatted_histogram_mrp"][0]}')
        print(f'  MRP(0.1)  = {result["formatted_histogram_mrp"][1]}')
        print(f'  MRP(0.01) = {result["formatted_histogram_mrp"][2]}')
        print('=' * 60)

    def load_mc_gaussian_window(_):
        with out:
            window = _resolve_visualization_gaussian_window()
            if window is None:
                print('No active mass/charge selection is available yet.')
            else:
                left, right, source = window
                mrp_left_mc.value, mrp_right_mc.value = left, right
                tag = ' (auto-picked around tallest peak)' if source == 'auto' else ''
                print(f'Loaded Gaussian MRP window: ({left:.4f}, {right:.4f}){tag}')

    def run_mc_gaussian_mrp(_):
        gaussian_mrp_mc_button.disabled = True
        with out:
            window = _resolve_visualization_gaussian_window()
            if window is None:
                print('Set MRP left/right or draw a selection first.')
            else:
                left, right, source = window
                mrp_left_mc.value, mrp_right_mc.value = left, right
                if source == 'auto':
                    print(f'Auto-selected MRP window around tallest peak: ({left:.4f}, {right:.4f})')
                result = gaussian_mrp_report(
                    _resolve_visualization_hist_array(),
                    left,
                    right,
                    bin_size=0.001,
                )
                if result is None:
                    print('Gaussian MRP: insufficient data in selected range')
                else:
                    _print_visualization_gaussian_report(result)
        gaussian_mrp_mc_button.disabled = False

    load_mrp_window_mc_button.on_click(load_mc_gaussian_window)
    gaussian_mrp_mc_button.on_click(run_mc_gaussian_mrp)

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
            data_loadcrop.plot_crop_fdm(
                data['x_det (cm)'].to_numpy(),
                data['y_det (cm)'].to_numpy(),
                bins,
                frac_fdm_widget.value,
                axis_mode_fdm.value,
                figure_size,
                variables,
                range_sequence,
                range_mc,
                range_detx,
                range_dety,
                range_x,
                range_y,
                range_z,
                range_vol,
                False,
                False,
                'circle',
                save_fdm_widget.value,
                figname_fdm_widget.value,
            )

        # Enable the button when the code is finished
        plot_fdm_button.disabled = False

    #############
    figname_3d = widgets.Text(value='3d_plot')
    rotary_fig_save_p3 = widgets.Dropdown(options=[('False', False), ('True', True)])
    element_percentage_p3 = widgets.Textarea(value=element_percentage)
    element_alpha_p3 = widgets.Textarea(value=element_alpha)
    opacity = widgets.FloatText(value=0.5, min=0, max=1, step=0.1)
    save_3d = widgets.Dropdown(options=[('True', True), ('False', False)], value=False)
    ions_individually_plots = widgets.Dropdown(options=[('True', True), ('False', False)], value=False)
    make_gif_p3 = widgets.Dropdown(options=[('True', True), ('False', False)], value=False)
    make_evap_3d = widgets.Dropdown(options=[('True', True), ('False', False)], value=False)
    evaporation_gif_frames = widgets.BoundedIntText(value=120, min=2, max=1000, step=10)
    evaporation_gif_fps = widgets.BoundedIntText(value=15, min=1, max=60, step=1)
    cluster_precipitate_3d = widgets.Dropdown(options=[('False', False), ('True', True)], value=False)
    cluster_labels_3d = widgets.Text(value='', placeholder='Ni3Al, Al')
    cluster_method_3d = widgets.Dropdown(
        options=[
            ('MMax separation', 'maximum-separation'),
            ('HDBSCAN', 'hdbscan'),
            ('Comp-Seeded Support HDBSCAN', 'comp-seeded-support-hdbscan'),
            ('Composition GMM Voxel', 'composition-gmm-voxel'),
            ('CompSpace Agnostic + Seeded', 'compspace-agnostic-seeded'),
            ('Min-Max', 'min-max'),
        ],
        value='maximum-separation',
    )
    cluster_count_3d = widgets.BoundedIntText(value=2, min=2, max=12)
    cluster_dmax_3d = widgets.BoundedFloatText(value=1.0, min=0.0001, max=1_000_000.0, step=0.05)
    cluster_auto_dmax_3d = widgets.Dropdown(options=[('True', True), ('False', False)], value=True)
    cluster_kth_neighbor_3d = widgets.BoundedIntText(value=3, min=1, max=100)
    cluster_percentile_3d = widgets.BoundedFloatText(value=50.0, min=1.0, max=99.9, step=1.0)
    cluster_min_size_3d = widgets.BoundedIntText(value=25, min=2, max=1_000_000)
    plot_3d_button.on_click(lambda b: plot_3d(b, variables, out))
    range_sequence_3d = widgets.Textarea(value='[0,0]')
    range_detx_3d = widgets.Textarea(value='[0,0]')
    range_dety_3d = widgets.Textarea(value='[0,0]')
    range_mc_3d = widgets.Textarea(value='[0,0]')
    range_x_3d = widgets.Textarea(value='[0,0]')
    range_y_3d = widgets.Textarea(value='[0,0]')
    range_z_3d = widgets.Textarea(value='[0,0]')
    range_vol_3d = widgets.Textarea(value='[0,0]')

    def plot_3d(
        b,
        variables,
        out,
        cluster_display_mode='overlay',
        cluster_result_override=None,
        cluster_opacity_override=None,
    ):
        plot_3d_button.disabled = True
        try:
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

                element_percentage_list, unranged_fraction = resolve_element_controls(
                    variables.range_data, ast.literal_eval(element_percentage_p3.value), 0.1, 0.01
                )
                element_alpha_list, unranged_alpha = resolve_element_controls(
                    variables.range_data, ast.literal_eval(element_alpha_p3.value), 0.9
                )

                if cluster_result_override is not None:
                    cluster_result = cluster_result_override
                else:
                    cluster_result = _run_cluster_segmentation(
                        enabled=cluster_precipitate_3d.value,
                        selection_text=cluster_labels_3d.value,
                        method_value=cluster_method_3d.value,
                        cluster_count_value=cluster_count_3d.value,
                        d_max_value=cluster_dmax_3d.value,
                        auto_d_max_value=cluster_auto_dmax_3d.value,
                        kth_neighbor_value=cluster_kth_neighbor_3d.value,
                        percentile_value=cluster_percentile_3d.value,
                        n_min_value=cluster_min_size_3d.value,
                        context_label='3D clustering',
                    )

                reconstruction.reconstruction_plot(
                    variables,
                    element_percentage_list,
                    opacity.value,
                    rotary_fig_save_p3.value,
                    figname_3d.value,
                    save_3d.value,
                    make_gif_p3.value,
                    make_evap_3d.value,
                    range_sequence,
                    range_mc,
                    range_detx,
                    range_dety,
                    range_x,
                    range_y,
                    range_z,
                    range_vol,
                    ions_individually_plots.value,
                    cluster_result=cluster_result,
                    cluster_display_mode=cluster_display_mode,
                    cluster_opacity_override=cluster_opacity_override,
                    element_alpha=element_alpha_list,
                    unranged_fraction=unranged_fraction,
                    unranged_alpha=unranged_alpha,
                    evaporation_gif_frames=evaporation_gif_frames.value,
                    evaporation_gif_fps=evaporation_gif_fps.value,
                )
        finally:
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
                    pulse_mode,
                    save,
                    figname,
                )
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
            reconstruction.heatmap(
                variables,
                element_percentage_list,
                range_sequence,
                range_mc,
                range_detx,
                range_dety,
                range_x,
                range_y,
                range_z,
                range_vol,
                figname_heatmap.value,
                figure_sie=figure_size,
                save=save_heatmap.value,
            )
        plot_heatmap_button.disabled = False

    #############

    points_per_frame_anim = widgets.IntText(value=50000)
    selected_area_temporally_anim = widgets.Dropdown(options=[('False', False), ('True', True)])
    selected_area_specially_anim = widgets.Dropdown(options=[('False', False), ('True', True)])
    figname_anim = widgets.Text(value='detector_animation')
    figure_size_x_anim = widgets.FloatText(value=5.0)
    figure_size_y_anim = widgets.FloatText(value=5.0)
    ranged_anim = widgets.Dropdown(options=[('False', False), ('True', True)])

    plot_animated_heatmap_button = widgets.Button(description="Plot animated heatmap", button_style='primary')

    def plot_animated_heatmap(b, variables, out):

        points_per_frame = points_per_frame_anim.value
        selected_area_temporally = selected_area_temporally_anim.value
        selected_area_specially = selected_area_specially_anim.value
        figure_name = figname_anim.value
        figure_size = (figure_size_x_anim.value, figure_size_y_anim.value)
        ranged = ranged_anim.value

        plot_animated_heatmap_button.disabled = True
        with out:
            reconstruction.detector_animation(
                variables,
                points_per_frame,
                ranged,
                selected_area_specially,
                selected_area_temporally,
                figure_name,
                figure_size,
                save=True,
            )
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

            reconstruction.projection(
                variables,
                element_percentage_list,
                range_sequence,
                range_mc,
                range_detx,
                range_dety,
                range_x,
                range_y,
                range_z,
                range_vol,
                x_or_y_pp.value,
                figname_p.value,
                figure_size,
                save_projection.value,
            )
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
            density_map.plot_density_map(
                x,
                y,
                z_weight_2d_hist.value,
                log_d.value,
                bins,
                percentage,
                axis_mode_d.value,
                figure_size,
                variables,
                composition,
                roi,
                range_sequence,
                range_mc,
                range_detx,
                range_dety,
                range_x,
                range_y,
                range_z,
                range_vol,
                False,
                False,
                'circle',
                axis_d,
                save,
                figure_name,
                cmap_d.value,
                normalize_d.value,
                normalize_axes_d.value,
            )
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
            sdm.sdm(
                particles,
                bins_size_sdm.value,
                variables,
                roi,
                z_cut_sdm.value,
                normalized_sdm.value,
                plot_mode_sdm.value,
                True,
                save_sdm.value,
                figure_size,
                figname_2d_sdm.value,
                sdm_mode.value,
                axis_sdm,
                i_composition_filtered,
                j_composition_filtered,
                plot_roi_sdm.value,
                theta_x_sdm.value,
                phi_y_sdm.value,
                log_sdm.value,
                percentage_sdm.value,
                range_sequence,
                range_mc,
                range_detx,
                range_dety,
                range_x,
                range_y,
                range_z,
                range_vol,
            )
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
            mask_sequence[range_sequence[0] : range_sequence[1]] = True
        else:
            mask_sequence = np.ones_like(variables.dld_x_det, dtype=bool)
        particles = np.vstack((variables.x[mask_sequence], variables.y[mask_sequence], variables.z[mask_sequence])).T
        figure_size_rdf = (figure_size_x_2d_rdf.value, figure_size_y_2d_rdf.value)
        plot_rdf_button.disabled = True
        with out:
            rdf.rdf(
                particles,
                dr_rdf.value,
                variables,
                rcutoff=rdf_cutoff.value,
                normalize=normalized_rdf.value,
                reference_point=reference_point,
                box_dimensions=box_dimension,
                plot=True,
                save=save_rdf.value,
                figure_size=figure_size_rdf,
                figname=figname_2d_rdf.value,
            )

        plot_rdf_button.disabled = False

    def _parse_common_ranges(sequence_widget, mc_widget, detx_widget, dety_widget, x_widget, y_widget, z_widget, vol_widget):
        range_sequence = json.loads(sequence_widget.value)
        range_mc = json.loads(mc_widget.value)
        range_detx = json.loads(detx_widget.value)
        range_dety = json.loads(dety_widget.value)
        range_x = json.loads(x_widget.value)
        range_y = json.loads(y_widget.value)
        range_z = json.loads(z_widget.value)
        range_vol = json.loads(vol_widget.value)

        if range_sequence == [0, 0]:
            range_sequence = []
        if range_mc == [0, 0]:
            range_mc = []
        if range_detx == [0, 0] or range_dety == [0, 0]:
            range_detx = []
            range_dety = []
        if range_x == [0, 0] or range_y == [0, 0] or range_z == [0, 0]:
            range_x = []
            range_y = []
            range_z = []
        if range_vol == [0, 0]:
            range_vol = []
        return range_sequence, range_mc, range_detx, range_dety, range_x, range_y, range_z, range_vol

    def _parse_isosurface_dict(value, *, allow_multiple=True, field_name='Isosurface dictionary'):
        text = str(value).strip()
        if not text:
            if allow_multiple:
                raise ValueError(f'{field_name} must not be empty. Example: {{Al: [3,3,3], Ni: [4,4,4]}}')
            raise ValueError(f'{field_name} must not be empty. Example: {{Al: [3,3,3]}}')

        formatted_string = re.sub(r'([A-Za-z0-9_]+)\s*:', r'"\1":', text)
        try:
            parsed = ast.literal_eval(formatted_string)
        except (ValueError, SyntaxError) as exc:
            if allow_multiple:
                raise ValueError(f'{field_name} must look like {{Al: [3,3,3], Ni: [4,4,4]}}') from exc
            raise ValueError(f'{field_name} must look like {{Al: [3,3,3]}}') from exc

        if not isinstance(parsed, dict) or not parsed:
            raise ValueError(f'{field_name} must be a non-empty dictionary')
        if not allow_multiple and len(parsed) != 1:
            raise ValueError(f'{field_name} must contain exactly one entry for proxigram, for example {{Al: [3,3,3]}}')

        normalized = {}
        for key, raw_value in parsed.items():
            element_name = str(key).strip()
            if not element_name:
                raise ValueError(f'{field_name} contains an empty element name')
            if not isinstance(raw_value, (list, tuple)) or len(raw_value) != 3:
                raise ValueError(f'{field_name} entry for {element_name} must be a 3-value list like [3,3,3]')
            try:
                normalized[element_name] = [float(item) for item in raw_value]
            except (TypeError, ValueError) as exc:
                raise ValueError(f'{field_name} entry for {element_name} must contain only numeric values') from exc
        return normalized

    def _parse_element_list(value, field_name='Element list'):
        text = str(value).strip()
        if not text:
            raise ValueError(f'{field_name} must not be empty. Example: Al, Ni, Cr')

        parsed = None
        if text.startswith('[') or text.startswith('('):
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(
                    f'{field_name} must be comma-separated like Al, Ni, Cr or a list like ["Al", "Ni", "Cr"]'
                ) from exc

        if parsed is None:
            items = re.split(r'[,;\n]+', text)
        elif isinstance(parsed, str):
            items = [parsed]
        elif isinstance(parsed, (list, tuple, set)):
            items = list(parsed)
        else:
            raise ValueError(f'{field_name} must be comma-separated like Al, Ni, Cr or a list like ["Al", "Ni", "Cr"]')

        normalized = []
        seen = set()
        for item in items:
            item = str(item).strip()
            if not item or item in seen:
                continue
            normalized.append(item)
            seen.add(item)

        if not normalized:
            raise ValueError(f'{field_name} must contain at least one valid element')
        return normalized

    def _build_element_percentage_list(value):
        return resolve_element_controls(variables.range_data, ast.literal_eval(value), 0.1, 0.01)

    def _run_cluster_segmentation(
        *,
        enabled,
        selection_text,
        method_value,
        cluster_count_value,
        d_max_value,
        auto_d_max_value,
        kth_neighbor_value,
        percentile_value,
        n_min_value,
        context_label,
        method_kwargs=None,
    ):
        if not enabled:
            return None

        cluster_selection = clustering.parse_label_selection(selection_text)
        if not cluster_selection:
            print(f'{context_label} is enabled, but no ion or element labels were provided. Skipping segmentation.')
            return None

        try:
            method = clustering.normalize_clustering_method(method_value)
            if method_kwargs is None:
                method_kwargs = {}
            if method in ('min-max', 'composition-gmm-voxel', 'compspace-agnostic-seeded'):
                cluster_result = clustering.segment_ions(
                    variables,
                    cluster_selection,
                    method=method,
                    n_clusters=max(2, int(cluster_count_value)),
                    n_min=max(2, int(n_min_value)),
                    **method_kwargs,
                )
                print(f'{method} clustering counts:', cluster_result.counts)
                return cluster_result

            cluster_result = clustering.segment_ions(
                variables,
                cluster_selection,
                method=method,
                d_max=float(d_max_value),
                n_min=max(2, int(n_min_value)),
                auto_d_max=bool(auto_d_max_value),
                kth_neighbor=max(1, int(kth_neighbor_value)),
                percentile=float(percentile_value),
                **method_kwargs,
            )
            d_max_used = None
            if cluster_result.parameters is not None:
                d_max_used = float(cluster_result.parameters.get('d_max', 0.0))
            method_title = 'HDBSCAN' if method == 'hdbscan' else 'Maximum separation'
            if cluster_result.n_clusters == 0:
                if d_max_used is not None and d_max_used > 0:
                    print(
                        f'{method_title} found no clusters '
                        f'(d_max={d_max_used:.4f}, min cluster size={max(2, int(n_min_value))}).'
                    )
                else:
                    print(f'{method_title} found no clusters.')
            else:
                print(f'{method_title} counts:', cluster_result.counts)
                if d_max_used is not None and d_max_used > 0:
                    mode_name = 'auto-estimated' if auto_d_max_value else 'manual'
                    print(f'Using d_max={d_max_used:.4f} ({mode_name}), min cluster size={max(2, int(n_min_value))}.')
            return cluster_result
        except ValueError as exc:
            print(f'{context_label} error: {exc}')
            return None

    #############
    figname_3d_iso = widgets.Text(value='3d_plot_iso')
    rotary_fig_save_p3_iso = widgets.Dropdown(options=[('False', False), ('True', True)])
    element_percentage_p3_iso = widgets.Textarea(value=element_percentage)
    opacity_iso = widgets.FloatText(value=0.5, min=0, max=1, step=0.1)
    save_3d_iso = widgets.Dropdown(options=[('True', True), ('False', False)], value=False)
    ions_individually_plots_iso = widgets.Dropdown(options=[('True', True), ('False', False)], value=False)
    make_gif_p3_iso = widgets.Dropdown(options=[('True', True), ('False', False)], value=False)
    cluster_precipitate_iso = widgets.Dropdown(options=[('False', False), ('True', True)], value=False)
    cluster_labels_iso = widgets.Text(value='', placeholder='Ni3Al, Al')
    cluster_method_iso = widgets.Dropdown(
        options=[
            ('MMax separation', 'maximum-separation'),
            ('HDBSCAN', 'hdbscan'),
            ('Comp-Seeded Support HDBSCAN', 'comp-seeded-support-hdbscan'),
            ('Composition GMM Voxel', 'composition-gmm-voxel'),
            ('CompSpace Agnostic + Seeded', 'compspace-agnostic-seeded'),
            ('Min-Max', 'min-max'),
        ],
        value='maximum-separation',
    )
    cluster_count_iso = widgets.BoundedIntText(value=2, min=2, max=12)
    cluster_dmax_iso = widgets.BoundedFloatText(value=1.0, min=0.0001, max=1_000_000.0, step=0.05)
    cluster_auto_dmax_iso = widgets.Dropdown(options=[('True', True), ('False', False)], value=True)
    cluster_kth_neighbor_iso = widgets.BoundedIntText(value=3, min=1, max=100)
    cluster_percentile_iso = widgets.BoundedFloatText(value=50.0, min=1.0, max=99.9, step=1.0)
    cluster_min_size_iso = widgets.BoundedIntText(value=25, min=2, max=1_000_000)
    plot_3d_button_iso.on_click(lambda b: plot_3d_iso(b, variables, out))
    isosurface_dic_p3_iso = widgets.Textarea(
        value="{Al: [3,3,3]}",
        placeholder="{Al: [3,3,3], Ni: [4,4,4]}",
    )
    detailed_isotope_charge_3d_iso = widgets.Dropdown(options=[('False', False), ('True', True)], value=False)
    only_iso_3d_iso = widgets.Dropdown(options=[('False', False), ('True', True)], value=False)
    pure_element_only_iso = widgets.Dropdown(options=[('True', True), ('False', False)], value=True)
    smoothing_sigma_iso = widgets.FloatText(value=1.0)
    manual_iso_value_iso = widgets.FloatText(value=0.0)
    min_atoms_per_voxel_iso = widgets.IntText(value=10)
    min_vertices_iso = widgets.IntText(value=20)
    range_sequence_3d_iso = widgets.Textarea(value='[0,0]')
    range_detx_3d_iso = widgets.Textarea(value='[0,0]')
    range_dety_3d_iso = widgets.Textarea(value='[0,0]')
    range_mc_3d_iso = widgets.Textarea(value='[0,0]')
    range_x_3d_iso = widgets.Textarea(value='[0,0]')
    range_y_3d_iso = widgets.Textarea(value='[0,0]')
    range_z_3d_iso = widgets.Textarea(value='[0,0]')
    range_vol_3d_iso = widgets.Textarea(value='[0,0]')

    def _sync_cluster_method_controls(
        method_widget,
        count_widget,
        auto_dmax_widget,
        dmax_widget,
        kth_neighbor_widget,
        percentile_widget,
        min_size_widget,
        note_widget=None,
    ):
        method = clustering.normalize_clustering_method(method_widget.value)
        uses_fixed_count = method in ('min-max', 'composition-gmm-voxel', 'compspace-agnostic-seeded')
        uses_density_params = method in ('maximum-separation', 'hdbscan')
        auto_d_max = bool(auto_dmax_widget.value)

        count_widget.disabled = not uses_fixed_count
        auto_dmax_widget.disabled = not uses_density_params
        min_size_widget.disabled = not uses_density_params
        dmax_widget.disabled = (not uses_density_params) or auto_d_max
        kth_neighbor_widget.disabled = (not uses_density_params) or (not auto_d_max)
        percentile_widget.disabled = (not uses_density_params) or (not auto_d_max)

        if note_widget is not None:
            if uses_fixed_count:
                note_widget.value = "<i>This method uses the selected number of clusters and ignores d max settings.</i>"
            elif auto_d_max:
                note_widget.value = (
                    "<i>This method finds as many connected clusters as the data supports. Number of clusters is not used.</i>"
                )
            else:
                note_widget.value = "<i>This method uses the manual d max cutoff and ignores Number of clusters.</i>"

    def _update_all_cluster_control_states(*_args):
        _sync_cluster_method_controls(
            cluster_method_3d,
            cluster_count_3d,
            cluster_auto_dmax_3d,
            cluster_dmax_3d,
            cluster_kth_neighbor_3d,
            cluster_percentile_3d,
            cluster_min_size_3d,
        )
        _sync_cluster_method_controls(
            cluster_method_iso,
            cluster_count_iso,
            cluster_auto_dmax_iso,
            cluster_dmax_iso,
            cluster_kth_neighbor_iso,
            cluster_percentile_iso,
            cluster_min_size_iso,
        )

    for widget in (
        cluster_method_3d,
        cluster_auto_dmax_3d,
        cluster_method_iso,
        cluster_auto_dmax_iso,
    ):
        widget.observe(_update_all_cluster_control_states, names='value')

    _update_all_cluster_control_states()

    def plot_3d_iso(b, variables, out, cluster_display_mode='overlay', cluster_result_override=None):
        plot_3d_button_iso.disabled = True
        try:
            with out:
                try:
                    range_sequence, range_mc, range_detx, range_dety, range_x, range_y, range_z, range_vol = (
                        _parse_common_ranges(
                            range_sequence_3d_iso,
                            range_mc_3d_iso,
                            range_detx_3d_iso,
                            range_dety_3d_iso,
                            range_x_3d_iso,
                            range_y_3d_iso,
                            range_z_3d_iso,
                            range_vol_3d_iso,
                        )
                    )
                    isosurface_dic_p3_iso_value = _parse_isosurface_dict(
                        isosurface_dic_p3_iso.value,
                        allow_multiple=True,
                        field_name='Iso surface dictionary',
                    )
                    element_percentage_list_iso, unranged_fraction_iso = _build_element_percentage_list(
                        element_percentage_p3_iso.value
                    )
                except (json.JSONDecodeError, ValueError, SyntaxError) as exc:
                    print(f'Invalid iso plot input: {exc}')
                    return

                if cluster_result_override is not None:
                    cluster_result = cluster_result_override
                else:
                    cluster_result = _run_cluster_segmentation(
                        enabled=cluster_precipitate_iso.value,
                        selection_text=cluster_labels_iso.value,
                        method_value=cluster_method_iso.value,
                        cluster_count_value=cluster_count_iso.value,
                        d_max_value=cluster_dmax_iso.value,
                        auto_d_max_value=cluster_auto_dmax_iso.value,
                        kth_neighbor_value=cluster_kth_neighbor_iso.value,
                        percentile_value=cluster_percentile_iso.value,
                        n_min_value=cluster_min_size_iso.value,
                        context_label='Iso-surface clustering',
                    )

                iso_surface.reconstruction_plot(
                    variables,
                    element_percentage_list_iso,
                    opacity_iso.value,
                    rotary_fig_save_p3_iso.value,
                    figname_3d_iso.value,
                    save_3d_iso.value,
                    make_gif_p3_iso.value,
                    range_sequence,
                    range_mc,
                    range_detx,
                    range_dety,
                    range_x,
                    range_y,
                    range_z,
                    range_vol,
                    ions_individually_plots_iso.value,
                    max_num_ions=None,
                    min_num_ions=None,
                    isosurface_dic=isosurface_dic_p3_iso_value,
                    detailed_isotope_charge=detailed_isotope_charge_3d_iso.value,
                    only_iso=only_iso_3d_iso.value,
                    cluster_result=cluster_result,
                    smoothing_sigma=smoothing_sigma_iso.value,
                    manual_iso_value=(manual_iso_value_iso.value if manual_iso_value_iso.value > 0 else None),
                    min_atoms_per_voxel=min_atoms_per_voxel_iso.value,
                    min_isosurface_vertices=min_vertices_iso.value,
                    pure_element_only=pure_element_only_iso.value,
                    cluster_display_mode=cluster_display_mode,
                    unranged_fraction=unranged_fraction_iso,
                )
        finally:
            plot_3d_button_iso.disabled = False

    #############
    plot_proxigram_button = widgets.Button(description='Plot proxigram', button_style='primary')
    figname_proxigram = widgets.Text(value='proxigram')
    proxigram_isosurface_dic = widgets.Textarea(
        value="{Al: [3,3,3]}",
        placeholder="{Al: [3,3,3]}",
    )
    proxigram_elements = widgets.Text(
        value='Al',
        placeholder='Al, Ni, Ti or ["Al", "Ni", "Ti"]',
    )
    proxigram_bin_size = widgets.FloatText(value=0.1)
    proxigram_symmetric_range = widgets.FloatText(value=0.0)
    proxigram_flip_normals = widgets.Dropdown(options=[('False', False), ('True', True)], value=False)
    proxigram_save = widgets.Dropdown(options=[('True', True), ('False', False)], value=False)
    proxigram_pure_element_only = widgets.Dropdown(options=[('True', True), ('False', False)], value=True)
    proxigram_sigma = widgets.FloatText(value=1.0)
    proxigram_manual_iso_value = widgets.FloatText(value=0.0)
    proxigram_min_atoms = widgets.IntText(value=10)
    proxigram_min_vertices = widgets.IntText(value=20)
    range_sequence_prox = widgets.Textarea(value='[0,0]')
    range_detx_prox = widgets.Textarea(value='[0,0]')
    range_dety_prox = widgets.Textarea(value='[0,0]')
    range_mc_prox = widgets.Textarea(value='[0,0]')
    range_x_prox = widgets.Textarea(value='[0,0]')
    range_y_prox = widgets.Textarea(value='[0,0]')
    range_z_prox = widgets.Textarea(value='[0,0]')
    range_vol_prox = widgets.Textarea(value='[0,0]')

    def plot_proxigram_view(b, variables, out):
        plot_proxigram_button.disabled = True
        with out:
            try:
                range_sequence, range_mc, range_detx, range_dety, range_x, range_y, range_z, range_vol = _parse_common_ranges(
                    range_sequence_prox,
                    range_mc_prox,
                    range_detx_prox,
                    range_dety_prox,
                    range_x_prox,
                    range_y_prox,
                    range_z_prox,
                    range_vol_prox,
                )
                interface_dic = _parse_isosurface_dict(
                    proxigram_isosurface_dic.value,
                    allow_multiple=False,
                    field_name='Proxigram interface isosurface',
                )
                proxigram_elements_list = _parse_element_list(
                    proxigram_elements.value,
                    field_name='Proxigram elements',
                )
                symmetric_range = proxigram_symmetric_range.value if proxigram_symmetric_range.value > 0 else None
                proxigram.plot_proxigram(
                    variables,
                    interface_dic,
                    proxigram_elements_list,
                    figname=figname_proxigram.value,
                    save=proxigram_save.value,
                    bin_size=proxigram_bin_size.value,
                    symmetric_range=symmetric_range,
                    flip_normals=proxigram_flip_normals.value,
                    range_sequence=range_sequence,
                    range_mc=range_mc,
                    range_detx=range_detx,
                    range_dety=range_dety,
                    range_x=range_x,
                    range_y=range_y,
                    range_z=range_z,
                    range_vol=range_vol,
                    smoothing_sigma=proxigram_sigma.value,
                    min_atoms_per_voxel=proxigram_min_atoms.value,
                    min_isosurface_vertices=proxigram_min_vertices.value,
                    pure_only=proxigram_pure_element_only.value,
                    manual_iso_value=(proxigram_manual_iso_value.value if proxigram_manual_iso_value.value > 0 else None),
                )
            except (json.JSONDecodeError, ValueError, SyntaxError) as exc:
                print(f'Unable to plot proxigram: {exc}')
        plot_proxigram_button.disabled = False

    plot_proxigram_button.on_click(lambda b: plot_proxigram_view(b, variables, out))

    #############
    row_index = widgets.IntText(value=0, description='Index row:')
    color_picker = widgets.ColorPicker(description='Select a color:')
    change_color.on_click(lambda b: change_color_m(b, variables, out))

    show_color.on_click(lambda b: show_color_ions(b, variables, out))

    def show_color_ions(b, variables, output):
        with output:
            clear_output(True)
            display(variables.range_data.style.map(ion_selection.display_color, subset=['color']))

    #############
    def change_color_m(b, variables, output):
        with output:
            selected_color = mcolors.to_hex(color_picker.value)
            variables.range_data.at[row_index.value, 'color'] = selected_color
            clear_output(True)
            display(variables.range_data.style.map(ion_selection.display_color, subset=['color']))

    clear_button.on_click(lambda b: clear(b, out))

    def clear(b, out):
        with out:
            clear_output(True)
            print('')

    def _validate_cluster_plot_request(selection_text, context_label):
        if getattr(variables, 'range_data', None) is None or variables.range_data.empty:
            print(f'{context_label} requires range data. Load a range file first.')
            return False
        if not clustering.parse_label_selection(selection_text):
            print(f'{context_label}: enter one or more cluster ions/elements first.')
            return False
        return True

    def _apply_method_settings_to_3d_iso(selection_text, method_value, n_clusters, n_min, auto_d_max, d_max, kth, percentile):
        cluster_labels_3d.value = selection_text
        cluster_labels_iso.value = selection_text
        cluster_method_3d.value = method_value
        cluster_method_iso.value = method_value
        cluster_count_3d.value = n_clusters
        cluster_count_iso.value = n_clusters
        cluster_min_size_3d.value = n_min
        cluster_min_size_iso.value = n_min
        cluster_auto_dmax_3d.value = auto_d_max
        cluster_auto_dmax_iso.value = auto_d_max
        cluster_dmax_3d.value = d_max
        cluster_dmax_iso.value = d_max
        cluster_kth_neighbor_3d.value = kth
        cluster_kth_neighbor_iso.value = kth
        cluster_percentile_3d.value = percentile
        cluster_percentile_iso.value = percentile
        _update_all_cluster_control_states()

    def _build_clustering_method_panel(method_key, display_name, method_note, *, seed_placeholder=''):
        selection_widget = widgets.Text(value='', placeholder='Ni3Al, Al')
        n_clusters_widget = widgets.BoundedIntText(value=2, min=2, max=12)
        n_min_widget = widgets.BoundedIntText(value=25, min=2, max=1_000_000)
        auto_dmax_widget = widgets.Dropdown(options=[('True', True), ('False', False)], value=True)
        dmax_widget = widgets.BoundedFloatText(value=1.0, min=0.0001, max=1_000_000.0, step=0.05)
        kth_widget = widgets.BoundedIntText(value=3, min=1, max=100)
        percentile_widget = widgets.BoundedFloatText(value=50.0, min=1.0, max=99.9, step=1.0)
        voxel_size_widget = widgets.FloatText(value=1.0)
        seed_widget = widgets.Text(value='', placeholder=seed_placeholder)
        note_widget = widgets.HTML(value=f'<i>{method_note}</i>')

        calculate_button = widgets.Button(description='Calculate segmentation')
        plot_button = widgets.Button(description='Plot')
        plot_iso_button = widgets.Button(description='Plot iso')
        clear_local_button = widgets.Button(description='Clear plots')
        cache = {'result': None}

        def _set_disabled(disabled):
            calculate_button.disabled = disabled
            clear_local_button.disabled = disabled
            plot_button.disabled = disabled or cache.get('result') is None
            plot_iso_button.disabled = disabled or cache.get('result') is None

        def _sync_local_control_state(*_):
            _sync_cluster_method_controls(
                widgets.fixed(method_key),
                n_clusters_widget,
                auto_dmax_widget,
                dmax_widget,
                kth_widget,
                percentile_widget,
                n_min_widget,
                note_widget=note_widget,
            )

        # widgets.fixed doesn't trigger observers; call directly and also when auto mode changes
        auto_dmax_widget.observe(_sync_local_control_state, names='value')

        def _calculate(_):
            _set_disabled(True)
            try:
                with out:
                    out.clear_output()
                    if not _validate_cluster_plot_request(selection_widget.value, f'{display_name} segmentation'):
                        cache['result'] = None
                        return
                    method_kwargs = {}
                    if method_key == 'composition-gmm-voxel':
                        method_kwargs['voxel_size'] = float(voxel_size_widget.value)
                    elif method_key in ('compspace-agnostic-seeded', 'comp-seeded-support-hdbscan'):
                        method_kwargs['seed_labels'] = seed_widget.value
                    cluster_result = _run_cluster_segmentation(
                        enabled=True,
                        selection_text=selection_widget.value,
                        method_value=method_key,
                        cluster_count_value=n_clusters_widget.value,
                        d_max_value=dmax_widget.value,
                        auto_d_max_value=auto_dmax_widget.value,
                        kth_neighbor_value=kth_widget.value,
                        percentile_value=percentile_widget.value,
                        n_min_value=n_min_widget.value,
                        context_label=display_name,
                        method_kwargs=method_kwargs,
                    )
                    cache['result'] = cluster_result
                    if cluster_result is None:
                        print(f'{display_name}: segmentation did not produce clusters.')
                    else:
                        print(f'{display_name}: segmented {cluster_result.n_clusters} clusters.')
                        if method_key == 'composition-gmm-voxel':
                            print(f'Composition voxel size: {voxel_size_widget.value:.3f}')
                        if (
                            method_key in ('compspace-agnostic-seeded', 'comp-seeded-support-hdbscan')
                            and seed_widget.value.strip()
                        ):
                            print(f'Seed labels: {seed_widget.value.strip()}')
            finally:
                _set_disabled(False)

        def _plot_cluster(_):
            _set_disabled(True)
            try:
                result = cache.get('result')
                if result is None:
                    with out:
                        out.clear_output()
                        print(f'{display_name}: run Calculate segmentation first.')
                    return
                _apply_method_settings_to_3d_iso(
                    selection_widget.value,
                    method_key,
                    int(n_clusters_widget.value),
                    int(n_min_widget.value),
                    bool(auto_dmax_widget.value),
                    float(dmax_widget.value),
                    int(kth_widget.value),
                    float(percentile_widget.value),
                )
                original_opacity = float(opacity.value)
                cluster_precipitate_3d.value = True
                opacity.value = 0.3
                try:
                    plot_3d(
                        None,
                        variables,
                        out,
                        cluster_display_mode='overlay',
                        cluster_result_override=result,
                        cluster_opacity_override=1.0,
                    )
                finally:
                    opacity.value = original_opacity
            finally:
                _set_disabled(False)

        def _plot_iso(_):
            _set_disabled(True)
            try:
                result = cache.get('result')
                if result is None:
                    with out:
                        out.clear_output()
                        print(f'{display_name}: run Calculate segmentation first.')
                    return
                _apply_method_settings_to_3d_iso(
                    selection_widget.value,
                    method_key,
                    int(n_clusters_widget.value),
                    int(n_min_widget.value),
                    bool(auto_dmax_widget.value),
                    float(dmax_widget.value),
                    int(kth_widget.value),
                    float(percentile_widget.value),
                )
                cluster_precipitate_iso.value = True
                plot_3d_iso(None, variables, out, cluster_display_mode='clusters-only', cluster_result_override=result)
            finally:
                _set_disabled(False)

        def _clear_local(_):
            cache['result'] = None
            with out:
                clear_output(True)
            _set_disabled(False)

        calculate_button.on_click(_calculate)
        plot_button.on_click(_plot_cluster)
        plot_iso_button.on_click(_plot_iso)
        clear_local_button.on_click(_clear_local)
        _sync_local_control_state()
        _set_disabled(False)

        method_specific_rows = []
        if method_key == 'composition-gmm-voxel':
            method_specific_rows.append(
                widgets.HBox([widgets.Label(value='Voxel size:', layout=label_layout), voxel_size_widget])
            )
        if method_key in ('compspace-agnostic-seeded', 'comp-seeded-support-hdbscan'):
            method_specific_rows.append(widgets.HBox([widgets.Label(value='Seed labels:', layout=label_layout), seed_widget]))

        return widgets.VBox(
            [
                widgets.HBox([widgets.Label(value='Cluster ions/elements:', layout=label_layout), selection_widget]),
                widgets.HBox([widgets.Label(value='Number of clusters:', layout=label_layout), n_clusters_widget]),
                widgets.HBox([widgets.Label(value='Min atoms per cluster:', layout=label_layout), n_min_widget]),
                widgets.HBox([widgets.Label(value='Auto estimate d max:', layout=label_layout), auto_dmax_widget]),
                widgets.HBox([widgets.Label(value='D max:', layout=label_layout), dmax_widget]),
                widgets.HBox([widgets.Label(value='K-th NN:', layout=label_layout), kth_widget]),
                widgets.HBox([widgets.Label(value='NN percentile:', layout=label_layout), percentile_widget]),
                *method_specific_rows,
                note_widget,
                widgets.HBox([calculate_button, plot_button, plot_iso_button, clear_local_button]),
            ]
        )

    tab0 = widgets.HBox(
        [
            widgets.VBox(
                [
                    widgets.HBox([widgets.Label(value="Target:", layout=label_layout), target_mode]),
                    widgets.HBox([widgets.Label(value="Peak find:", layout=label_layout), peaks_find]),
                    widgets.HBox([widgets.Label(value="Peak find plot:", layout=label_layout), peak_find_plot]),
                    widgets.HBox([widgets.Label(value="Plot ranged colors:", layout=label_layout), plot_ranged_colors]),
                    widgets.HBox([widgets.Label(value="Plot ranged peak:", layout=label_layout), plot_ranged_peak]),
                    range_overlay_note,
                    widgets.HBox([widgets.Label(value="Print info:", layout=label_layout), print_info]),
                    widgets.HBox([widgets.Label(value="Bins size:", layout=label_layout), bin_size_pm]),
                    widgets.HBox([widgets.Label(value="Limit mc:", layout=label_layout), lim_mc_pm]),
                    widgets.HBox([widgets.Label(value="Log:", layout=label_layout), log_widget]),
                    widgets.HBox([widgets.Label(value="Normalize:", layout=label_layout), normalize]),
                    widgets.HBox([widgets.Label(value="MRP all(1%, 10%, 50%):", layout=label_layout), mrp_all]),
                    widgets.HBox([widgets.Label(value="MRP Percent:", layout=label_layout), percent]),
                    widgets.HBox([widgets.Label(value="Legend mode:", layout=label_layout), legend_widget]),
                    widgets.HBox([widgets.Label(value="Peak prominence:", layout=label_layout), prominence]),
                    widgets.HBox([widgets.Label(value="Peak distance:", layout=label_layout), distance]),
                    widgets.HBox([widgets.Label(value="Background:", layout=label_layout), background_mc]),
                    widgets.HBox([widgets.Label(value="Grid:", layout=label_layout), grid_mc]),
                    widgets.HBox(
                        [widgets.Label(value="MRP range:", layout=label_layout), widgets.HBox([mrp_left_mc, mrp_right_mc])]
                    ),
                    widgets.HBox([load_mrp_window_mc_button, gaussian_mrp_mc_button]),
                    widgets.HBox([plot_mc_button, clear_button]),
                ]
            ),
            widgets.VBox(
                [
                    widgets.HBox([widgets.Label(value="Sequence range:", layout=label_layout), range_sequence_mc]),
                    widgets.HBox([widgets.Label(value="Mc range:", layout=label_layout), range_mc_mc]),
                    widgets.HBox([widgets.Label(value="Detx range:", layout=label_layout), range_detx_mc]),
                    widgets.HBox([widgets.Label(value="Dety range:", layout=label_layout), range_dety_mc]),
                    widgets.HBox([widgets.Label(value="X range:", layout=label_layout), range_x_mc]),
                    widgets.HBox([widgets.Label(value="Y range:", layout=label_layout), range_y_mc]),
                    widgets.HBox([widgets.Label(value="Z range:", layout=label_layout), range_z_mc]),
                    widgets.HBox([widgets.Label(value="Voltage range:", layout=label_layout), range_vol_mc]),
                    widgets.HBox([widgets.Label(value="Fig name:", layout=label_layout), figname_mc]),
                    widgets.HBox([widgets.Label(value="Save fig:", layout=label_layout), save_mc]),
                    widgets.HBox(
                        [
                            widgets.Label(value="Fig size:", layout=label_layout),
                            widgets.HBox([figure_mc_size_x_mc, figure_mc_size_y_mc]),
                        ]
                    ),
                ]
            ),
        ]
    )

    tab1 = widgets.VBox(
        [
            widgets.HBox([widgets.Label(value='Max TOF:', layout=label_layout), max_tof_mc_widget]),
            widgets.HBox([widgets.Label(value='Fraction:', layout=label_layout), frac_mc_widget]),
            widgets.HBox([widgets.Label(value='Bins:', layout=label_layout), widgets.HBox([bins_x_mc, bins_y_mc])]),
            widgets.HBox([widgets.Label(value='Pulse Plot:', layout=label_layout), pulse_plot_mc_widget]),
            widgets.HBox([widgets.Label(value='DC Plot:', layout=label_layout), dc_plot_mc_widget]),
            widgets.HBox([widgets.Label(value='Pulse Mode:', layout=label_layout), pulse_mode_mc_widget]),
            widgets.HBox([widgets.Label(value='Fig name:', layout=label_layout), figname_mc_widget]),
            widgets.HBox([widgets.Label(value='Save:', layout=label_layout), save_mc_widget]),
            widgets.HBox(
                [widgets.Label(value='Fig size:', layout=label_layout), widgets.HBox([figure_size_x_mc, figure_size_y_mc])]
            ),
            widgets.HBox([plot_experiment_button, clear_button]),
        ]
    )

    tab2 = widgets.HBox(
        [
            widgets.VBox(
                [
                    widgets.HBox([widgets.Label(value='Fraction:', layout=label_layout), frac_fdm_widget]),
                    widgets.HBox([widgets.Label(value='Bins:', layout=label_layout), bins_fdm]),
                    widgets.HBox([widgets.Label(value='Axis mode:', layout=label_layout), axis_mode_fdm]),
                    widgets.HBox([widgets.Label(value='Fig name:', layout=label_layout), figname_fdm_widget]),
                    widgets.HBox([widgets.Label(value='Save:', layout=label_layout), save_fdm_widget]),
                    widgets.HBox(
                        [
                            widgets.Label(value='Fig size:', layout=label_layout),
                            widgets.HBox([figure_size_x_fdm, figure_size_y_fdm]),
                        ]
                    ),
                    widgets.HBox([plot_fdm_button, clear_button]),
                ]
            ),
            widgets.VBox(
                [
                    widgets.HBox([widgets.Label(value="Sequence range:", layout=label_layout), range_sequence_fdm]),
                    widgets.HBox([widgets.Label(value='Mc range:', layout=label_layout), range_mc_fdm]),
                    widgets.HBox([widgets.Label(value='Detx range:', layout=label_layout), range_detx_fdm]),
                    widgets.HBox([widgets.Label(value='Dety range:', layout=label_layout), range_dety_fdm]),
                    widgets.HBox([widgets.Label(value='X range:', layout=label_layout), range_x_fdm]),
                    widgets.HBox([widgets.Label(value='Y range:', layout=label_layout), range_y_fdm]),
                    widgets.HBox([widgets.Label(value='Z range:', layout=label_layout), range_z_fdm]),
                    widgets.HBox([widgets.Label(value="Voltage range:", layout=label_layout), range_vol_fdm]),
                ]
            ),
        ]
    )

    tab3 = widgets.HBox(
        [
            widgets.VBox(
                [
                    widgets.HBox([widgets.Label(value='Display fraction:', layout=label_layout), element_percentage_p3]),
                    widgets.HBox([widgets.Label(value='Element alphas:', layout=label_layout), element_alpha_p3]),
                    widgets.HBox([widgets.Label(value='Opacity:', layout=label_layout), opacity]),
                    widgets.HBox(
                        [widgets.Label(value='Ions individually plots:', layout=label_layout), ions_individually_plots]
                    ),
                    widgets.HBox([widgets.Label(value='Fig name:', layout=label_layout), figname_3d]),
                    widgets.HBox([widgets.Label(value='Rotary save:', layout=label_layout), rotary_fig_save_p3]),
                    widgets.HBox([widgets.Label(value='Save GIF:', layout=label_layout), make_gif_p3]),
                    widgets.HBox([widgets.Label(value='Save evaporation GIF:', layout=label_layout), make_evap_3d]),
                    widgets.HBox([widgets.Label(value='Evaporation GIF frames:', layout=label_layout), evaporation_gif_frames]),
                    widgets.HBox([widgets.Label(value='Evaporation GIF FPS:', layout=label_layout), evaporation_gif_fps]),
                    widgets.HBox([widgets.Label(value='Save fig:', layout=label_layout), save_3d]),
                    widgets.HBox([plot_3d_button, clear_button]),
                ]
            ),
            widgets.VBox(
                [
                    widgets.HBox([widgets.Label(value="Sequence range:", layout=label_layout), range_sequence_3d]),
                    widgets.HBox([widgets.Label(value='Range mc:', layout=label_layout), range_mc_3d]),
                    widgets.HBox([widgets.Label(value='Range detx:', layout=label_layout), range_detx_3d]),
                    widgets.HBox([widgets.Label(value='Range dety:', layout=label_layout), range_dety_3d]),
                    widgets.HBox([widgets.Label(value='Range x:', layout=label_layout), range_x_3d]),
                    widgets.HBox([widgets.Label(value='Range y:', layout=label_layout), range_y_3d]),
                    widgets.HBox([widgets.Label(value='Range z:', layout=label_layout), range_z_3d]),
                    widgets.HBox([widgets.Label(value="Voltage range:", layout=label_layout), range_vol_3d]),
                ]
            ),
        ]
    )

    tab4 = widgets.HBox(
        [
            widgets.VBox(
                [
                    widgets.HBox([widgets.Label(value='Element percentage:', layout=label_layout), element_percentage_ph]),
                    widgets.HBox([widgets.Label(value='Fig name:', layout=label_layout), figname_heatmap]),
                    widgets.HBox([widgets.Label(value='Save fig:', layout=label_layout), save_heatmap]),
                    widgets.HBox(
                        [
                            widgets.Label(value='Fig size:', layout=label_layout),
                            widgets.HBox([figure_mc_size_x_heatmap, figure_mc_size_y_heatmap]),
                        ]
                    ),
                    widgets.HBox([plot_heatmap_button, clear_button]),
                ]
            ),
            widgets.VBox(
                [
                    widgets.HBox([widgets.Label(value="Sequence range:", layout=label_layout), range_sequence_heatmap]),
                    widgets.HBox([widgets.Label(value='Range mc:', layout=label_layout), range_mc_heatmap]),
                    widgets.HBox([widgets.Label(value='Range detx:', layout=label_layout), range_detx_heatmap]),
                    widgets.HBox([widgets.Label(value='Range dety:', layout=label_layout), range_dety_heatmap]),
                    widgets.HBox([widgets.Label(value='Range x:', layout=label_layout), range_x_heatmap]),
                    widgets.HBox([widgets.Label(value='Range y:', layout=label_layout), range_y_heatmap]),
                    widgets.HBox([widgets.Label(value='Range z:', layout=label_layout), range_z_heatmap]),
                    widgets.HBox([widgets.Label(value="Voltage range:", layout=label_layout), range_vol_heatmap]),
                ]
            ),
        ]
    )

    tab5 = widgets.VBox(
        [
            widgets.HBox([widgets.Label(value='Points per frame:', layout=label_layout), points_per_frame_anim]),
            widgets.HBox([widgets.Label(value='Ranged:', layout=label_layout), ranged_anim]),
            widgets.HBox([widgets.Label(value='Fig name:', layout=label_layout), figname_anim]),
            widgets.HBox(
                [widgets.Label(value='Fig size:', layout=label_layout), widgets.HBox([figure_size_x_anim, figure_size_y_anim])]
            ),
            widgets.HBox([plot_animated_heatmap_button, clear_button]),
        ]
    )

    tab6 = widgets.HBox(
        [
            widgets.VBox(
                [
                    widgets.HBox([widgets.Label(value='Element percentage:', layout=label_layout), element_percentage_pp]),
                    widgets.HBox([widgets.Label(value='X or Y:', layout=label_layout), x_or_y_pp]),
                    widgets.HBox([widgets.Label(value='Fig name:', layout=label_layout), figname_p]),
                    widgets.HBox([widgets.Label(value='Save fig:', layout=label_layout), save_projection]),
                    widgets.HBox(
                        [
                            widgets.Label(value='Fig size:', layout=label_layout),
                            widgets.HBox([figure_mc_size_x_projection, figure_mc_size_y_projection]),
                        ]
                    ),
                    widgets.HBox([plot_projection_button, clear_button]),
                ]
            ),
            widgets.VBox(
                [
                    widgets.HBox([widgets.Label(value="Sequence range:", layout=label_layout), range_sequence_pp]),
                    widgets.HBox([widgets.Label(value='Range mc:', layout=label_layout), range_mc_pp]),
                    widgets.HBox([widgets.Label(value='Range detx:', layout=label_layout), range_detx_pp]),
                    widgets.HBox([widgets.Label(value='Range dety:', layout=label_layout), range_dety_pp]),
                    widgets.HBox([widgets.Label(value='Range x:', layout=label_layout), range_x_pp]),
                    widgets.HBox([widgets.Label(value='Range y:', layout=label_layout), range_y_pp]),
                    widgets.HBox([widgets.Label(value='Range z:', layout=label_layout), range_z_pp]),
                    widgets.HBox([widgets.Label(value="Voltage range:", layout=label_layout), range_vol_pp]),
                ]
            ),
        ]
    )

    tab7 = widgets.HBox(
        [
            widgets.VBox(
                [
                    widgets.HBox([widgets.Label(value='First axis:', layout=label_layout), first_axis_d]),
                    widgets.HBox([widgets.Label(value='Second axis:', layout=label_layout), second_axis_d]),
                    widgets.HBox([widgets.Label(value='Composition:', layout=label_layout), composition_d]),
                    widgets.HBox([widgets.Label(value='Z weight:', layout=label_layout), z_weight_2d_hist]),
                    widgets.HBox([widgets.Label(value='Bin size (nm):', layout=label_layout), bins2d_hist]),
                    widgets.HBox([widgets.Label(value='Log:', layout=label_layout), log_d]),
                    widgets.HBox([widgets.Label(value='Axis mode:', layout=label_layout), axis_mode_d]),
                    widgets.HBox([widgets.Label(value='Cmap:', layout=label_layout), cmap_d]),
                    widgets.HBox([widgets.Label(value='Normalize:', layout=label_layout), normalize_d]),
                    widgets.HBox([widgets.Label(value='Normalize axes:', layout=label_layout), normalize_axes_d]),
                    widgets.HBox([widgets.Label(value='Fig name:', layout=label_layout), figname_2d_hist]),
                    widgets.HBox([widgets.Label(value='Save:', layout=label_layout), save_2d_hist]),
                    widgets.HBox(
                        [
                            widgets.Label(value='Fig size:', layout=label_layout),
                            widgets.HBox([figure_size_x_2d_hist, figure_size_y_2d_hist]),
                        ]
                    ),
                    widgets.HBox([density_map_button, clear_button]),
                ]
            ),
            widgets.VBox(
                [
                    widgets.HBox([widgets.Label(value='Percentage:', layout=label_layout), percentage_2d_hist]),
                    widgets.HBox([widgets.Label(value="Sequence range:", layout=label_layout), range_sequence_2d_hist]),
                    widgets.HBox([widgets.Label(value='Mc range:', layout=label_layout), range_mc_2d_hist]),
                    widgets.HBox([widgets.Label(value='Detx range:', layout=label_layout), range_detx_2d_hist]),
                    widgets.HBox([widgets.Label(value='Dety range:', layout=label_layout), range_dety_2d_hist]),
                    widgets.HBox([widgets.Label(value='X range:', layout=label_layout), range_x_2d_hist]),
                    widgets.HBox([widgets.Label(value='Y range:', layout=label_layout), range_y_2d_hist]),
                    widgets.HBox([widgets.Label(value='Z range:', layout=label_layout), range_z_2d_hist]),
                    widgets.HBox([widgets.Label(value="Voltage range:", layout=label_layout), range_vol_2d_hist]),
                ]
            ),
        ]
    )
    tab8 = widgets.HBox(
        [
            widgets.VBox(
                [
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
                    widgets.HBox(
                        [
                            widgets.Label(value='Fig size:', layout=label_layout),
                            widgets.HBox([figure_size_x_2d_sdm, figure_size_y_2d_sdm]),
                        ]
                    ),
                    widgets.HBox([plot_sdm_button, clear_button]),
                ]
            ),
            widgets.VBox(
                [
                    widgets.Label(value="Ions I for SDM:"),
                    *i_composition_sdm,
                ]
            ),
            widgets.VBox(
                [
                    widgets.Label(value="Ions J for SDM:"),
                    *j_composition_sdm,
                ]
            ),
            widgets.VBox(
                [
                    widgets.HBox([widgets.Label(value='Percentage:', layout=label_layout), percentage_sdm]),
                    widgets.HBox([widgets.Label(value='Range sequence:', layout=label_layout), range_sequence_sdm]),
                    widgets.HBox([widgets.Label(value='Range mc:', layout=label_layout), range_mc_sdm]),
                    widgets.HBox([widgets.Label(value='Range detx:', layout=label_layout), range_detx_sdm]),
                    widgets.HBox([widgets.Label(value='Range dety:', layout=label_layout), range_dety_sdm]),
                    widgets.HBox([widgets.Label(value='Range x:', layout=label_layout), range_x_sdm]),
                    widgets.HBox([widgets.Label(value='Range y:', layout=label_layout), range_y_sdm]),
                    widgets.HBox([widgets.Label(value='Range z:', layout=label_layout), range_z_sdm]),
                    widgets.HBox([widgets.Label(value='Range vol:', layout=label_layout), range_vol_sdm]),
                ]
            ),
        ]
    )

    tab9 = widgets.HBox(
        [
            widgets.VBox(
                [
                    widgets.HBox([widgets.Label(value='Dr:', layout=label_layout), dr_rdf]),
                    widgets.HBox([widgets.Label(value='Rdf cutoff:', layout=label_layout), rdf_cutoff]),
                    widgets.HBox([widgets.Label(value='Normalized:', layout=label_layout), normalized_rdf]),
                    widgets.HBox([widgets.Label(value='Reference point:', layout=label_layout), reference_point_rdf]),
                    widgets.HBox([widgets.Label(value='Fig name:', layout=label_layout), figname_2d_rdf]),
                    widgets.HBox([widgets.Label(value='Save:', layout=label_layout), save_rdf]),
                    widgets.HBox(
                        [
                            widgets.Label(value='Fig size:', layout=label_layout),
                            widgets.HBox([figure_size_x_2d_rdf, figure_size_y_2d_rdf]),
                        ]
                    ),
                    widgets.HBox([plot_rdf_button, clear_button]),
                ]
            ),
            widgets.VBox(
                [
                    widgets.HBox([widgets.Label(value='Range sequence:', layout=label_layout), range_sequence_rdf]),
                ]
            ),
        ]
    )

    tab10 = widgets.HBox(
        [
            widgets.VBox(
                [
                    widgets.HTML(
                        value="<b>Iso surface dictionary</b>: one or more entries are allowed, for example "
                        "<code>{Al: [3,3,3], Ni: [4,4,4]}</code>."
                    ),
                    widgets.HBox([widgets.Label(value='Display fraction:', layout=label_layout), element_percentage_p3_iso]),
                    widgets.HBox([widgets.Label(value='Opacity:', layout=label_layout), opacity_iso]),
                    widgets.HBox(
                        [widgets.Label(value='Ions individually plots:', layout=label_layout), ions_individually_plots_iso]
                    ),
                    widgets.HBox([widgets.Label(value='Iso dict (multi):', layout=label_layout), isosurface_dic_p3_iso]),
                    widgets.HBox(
                        [widgets.Label(value='Detailed isotope charge:', layout=label_layout), detailed_isotope_charge_3d_iso]
                    ),
                    widgets.HBox([widgets.Label(value='Only iso:', layout=label_layout), only_iso_3d_iso]),
                    widgets.HBox([widgets.Label(value='Pure elements only:', layout=label_layout), pure_element_only_iso]),
                    widgets.HBox([widgets.Label(value='Smoothing sigma:', layout=label_layout), smoothing_sigma_iso]),
                    widgets.HBox(
                        [widgets.Label(value='Iso value override (0=auto):', layout=label_layout), manual_iso_value_iso]
                    ),
                    widgets.HBox([widgets.Label(value='Min atoms / voxel:', layout=label_layout), min_atoms_per_voxel_iso]),
                    widgets.HBox([widgets.Label(value='Min iso vertices:', layout=label_layout), min_vertices_iso]),
                    widgets.HBox([widgets.Label(value='Make GIF:', layout=label_layout), make_gif_p3_iso]),
                    widgets.HBox([widgets.Label(value='Fig name:', layout=label_layout), figname_3d_iso]),
                    widgets.HBox([widgets.Label(value='Save:', layout=label_layout), save_3d_iso]),
                    widgets.HBox([widgets.Label(value='Rotary save:', layout=label_layout), rotary_fig_save_p3_iso]),
                    widgets.HBox([plot_3d_button_iso, clear_button]),
                ]
            ),
            widgets.VBox(
                [
                    widgets.HBox([widgets.Label(value="Sequence range:", layout=label_layout), range_sequence_3d_iso]),
                    widgets.HBox([widgets.Label(value='Range mc:', layout=label_layout), range_mc_3d_iso]),
                    widgets.HBox([widgets.Label(value='Range detx:', layout=label_layout), range_detx_3d_iso]),
                    widgets.HBox([widgets.Label(value='Range dety:', layout=label_layout), range_dety_3d_iso]),
                    widgets.HBox([widgets.Label(value='Range x:', layout=label_layout), range_x_3d_iso]),
                    widgets.HBox([widgets.Label(value='Range y:', layout=label_layout), range_y_3d_iso]),
                    widgets.HBox([widgets.Label(value='Range z:', layout=label_layout), range_z_3d_iso]),
                    widgets.HBox([widgets.Label(value="Voltage range:", layout=label_layout), range_vol_3d_iso]),
                ]
            ),
        ]
    )

    tab11 = widgets.HBox(
        [
            widgets.VBox(
                [
                    widgets.HTML(
                        value="<b>Interface isosurface</b>: use one element entry, for example <code>{Al: [3,3,3]}</code>."
                    ),
                    widgets.HBox([widgets.Label(value='Interface iso dict:', layout=label_layout), proxigram_isosurface_dic]),
                    widgets.HTML(
                        value="<b>Proxigram elements</b>: multiple elements are allowed. "
                        "Examples: <code>Al, Ni, Ti</code> or <code>[\"Al\", \"Ni\", \"Ti\"]</code>."
                    ),
                    widgets.HBox([widgets.Label(value='Proxigram elements:', layout=label_layout), proxigram_elements]),
                    widgets.HBox([widgets.Label(value='Bin size (nm):', layout=label_layout), proxigram_bin_size]),
                    widgets.HBox(
                        [widgets.Label(value='Symmetric range (nm):', layout=label_layout), proxigram_symmetric_range]
                    ),
                    widgets.HBox([widgets.Label(value='Flip normals:', layout=label_layout), proxigram_flip_normals]),
                    widgets.HBox(
                        [widgets.Label(value='Pure elements only:', layout=label_layout), proxigram_pure_element_only]
                    ),
                    widgets.HBox([widgets.Label(value='Smoothing sigma:', layout=label_layout), proxigram_sigma]),
                    widgets.HBox(
                        [widgets.Label(value='Iso value override (0=auto):', layout=label_layout), proxigram_manual_iso_value]
                    ),
                    widgets.HBox([widgets.Label(value='Min atoms / voxel:', layout=label_layout), proxigram_min_atoms]),
                    widgets.HBox([widgets.Label(value='Min iso vertices:', layout=label_layout), proxigram_min_vertices]),
                    widgets.HBox([widgets.Label(value='Fig name:', layout=label_layout), figname_proxigram]),
                    widgets.HBox([widgets.Label(value='Save:', layout=label_layout), proxigram_save]),
                    widgets.HBox([plot_proxigram_button, clear_button]),
                ]
            ),
            widgets.VBox(
                [
                    widgets.HBox([widgets.Label(value="Sequence range:", layout=label_layout), range_sequence_prox]),
                    widgets.HBox([widgets.Label(value='Range mc:', layout=label_layout), range_mc_prox]),
                    widgets.HBox([widgets.Label(value='Range detx:', layout=label_layout), range_detx_prox]),
                    widgets.HBox([widgets.Label(value='Range dety:', layout=label_layout), range_dety_prox]),
                    widgets.HBox([widgets.Label(value='Range x:', layout=label_layout), range_x_prox]),
                    widgets.HBox([widgets.Label(value='Range y:', layout=label_layout), range_y_prox]),
                    widgets.HBox([widgets.Label(value='Range z:', layout=label_layout), range_z_prox]),
                    widgets.HBox([widgets.Label(value="Voltage range:", layout=label_layout), range_vol_prox]),
                ]
            ),
        ]
    )

    cluster_method_panels = [
        _build_clustering_method_panel(
            'maximum-separation',
            'MMax separation',
            'Uses connected-component clustering with d max and min atoms controls.',
        ),
        _build_clustering_method_panel(
            'hdbscan',
            'HDBSCAN',
            'Density-based mode. Uses the same d max / neighbor controls in this workflow.',
        ),
        _build_clustering_method_panel(
            'comp-seeded-support-hdbscan',
            'Comp-Seeded Support HDBSCAN',
            'Density-based mode with optional seed labels to guide clustering support.',
            seed_placeholder='Al, Ni',
        ),
        _build_clustering_method_panel(
            'composition-gmm-voxel',
            'Composition GMM Voxel',
            'Uses fixed cluster count with composition-voxel controls.',
        ),
        _build_clustering_method_panel(
            'compspace-agnostic-seeded',
            'CompSpace Agnostic + Seeded',
            'Uses fixed cluster count and optional seed labels.',
            seed_placeholder='Al, Ni',
        ),
    ]
    cluster_subtabs = widgets.Tab(children=cluster_method_panels)
    cluster_subtabs.set_title(0, 'MMax separation')
    cluster_subtabs.set_title(1, 'HDBSCAN')
    cluster_subtabs.set_title(2, 'Comp-Seeded Support HDBSCAN')
    cluster_subtabs.set_title(3, 'Composition GMM Voxel')
    cluster_subtabs.set_title(4, 'CompSpace Agnostic + Seeded')
    tab12 = widgets.VBox(
        [
            widgets.HTML(value='<b>Clustering methods</b>: calculate segmentation first, then plot cluster or iso.'),
            cluster_subtabs,
        ]
    )

    tab13 = build_peak_spectral_analysis_panel(variables, label_layout=label_layout)

    tab_mc_tof_calculator = build_mc_tof_calculator_panel(variables)
    tab_concentration_profile = build_concentration_profile_panel(
        variables,
        label_layout=label_layout,
    )

    tab14 = widgets.VBox(
        [
            widgets.HBox([widgets.Label(value='Index row:', layout=label_layout), row_index]),
            widgets.HBox([widgets.Label(value='Color:', layout=label_layout), color_picker]),
            widgets.VBox([show_color, change_color, clear_button]),
        ]
    )

    if not colab:
        tab = widgets.Tab(
            [
                tab0,
                tab_mc_tof_calculator,
                tab1,
                tab2,
                tab3,
                tab4,
                tab5,
                tab6,
                tab7,
                tab8,
                tab9,
                tab10,
                tab11,
                tab_concentration_profile,
                tab12,
                tab13,
                tab14,
            ]
        )
        tab.set_title(0, 'mc')
        tab.set_title(1, 'mc & tof calculator')
        tab.set_title(2, 'Experiment history')
        tab.set_title(3, 'FDM')
        tab.set_title(4, '3D')
        tab.set_title(5, 'Hitmap')
        tab.set_title(6, 'Animated hitmap')
        tab.set_title(7, 'Projection')

        tab.set_title(8, 'Disparity map')
        tab.set_title(9, 'SDM')
        tab.set_title(10, 'RDF')
        tab.set_title(11, 'Iso surface')
        tab.set_title(12, 'Proxigram')
        tab.set_title(13, 'Concentration profile')
        tab.set_title(14, 'Clustering')
        tab.set_title(15, 'Peak analysis')
        tab.set_title(16, 'Change Color')

        out = Output()

        display(widgets.VBox([tab, out]))

    else:
        # Define the content for each tab
        content = [
            tab0,
            tab_mc_tof_calculator,
            tab1,
            tab2,
            tab3,
            tab4,
            tab5,
            tab6,
            tab7,
            tab8,
            tab9,
            tab10,
            tab11,
            tab_concentration_profile,
            tab12,
            tab13,
            tab14,
        ]

        # Create buttons for each tab
        buttons = [
            widgets.Button(description=title)
            for title in [
                'mc',
                'mc & tof calculator',
                'Experiment history',
                'FDM',
                '3D',
                'Hitmap',
                'Animated hitmap',
                'Projection',
                'Disparity map',
                'SDM',
                'RDF',
                'Iso surface',
                'Proxigram',
                'Concentration profile',
                'Clustering',
                'Peak analysis',
                'Change Color',
            ]
        ]

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
