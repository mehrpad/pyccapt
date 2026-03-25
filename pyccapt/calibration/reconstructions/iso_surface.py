import pyvista as pv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import io
import plotly.graph_objects as go
from PIL import Image
from plotly.subplots import make_subplots
import plotly.io as pio


from pyccapt.calibration.clustering import build_cluster_scatter_traces
from pyccapt.calibration.reconstructions import reconstruction
from pyccapt.calibration.reconstructions.io_utils import (
    save_gif,
    save_plotly_animation,
    write_plotly_html,
    write_plotly_image,
)


def build_range_mask(variables, range_sequence=None, range_mc=None, range_detx=None, range_dety=None,
                     range_x=None, range_y=None, range_z=None, range_vol=None, verbose=False):
    """Build a boolean mask for the requested reconstruction sub-range."""
    range_sequence = range_sequence or []
    range_mc = range_mc or []
    range_detx = range_detx or []
    range_dety = range_dety or []
    range_x = range_x or []
    range_y = range_y or []
    range_z = range_z or []
    range_vol = range_vol or []

    if range_sequence or range_detx or range_dety or range_mc or range_x or range_y or range_z or range_vol:
        if range_sequence:
            mask_sequence = np.zeros_like(variables.dld_x_det, dtype=bool)
            start, stop = sorted((int(range_sequence[0]), int(range_sequence[1])))
            mask_sequence[start:stop] = True
        else:
            mask_sequence = np.ones_like(variables.dld_x_det, dtype=bool)

        if range_detx and range_dety:
            x_min, x_max = sorted((range_detx[0], range_detx[1]))
            y_min, y_max = sorted((range_dety[0], range_dety[1]))
            mask_det_x = (variables.dld_x_det < x_max) & (variables.dld_x_det > x_min)
            mask_det_y = (variables.dld_y_det < y_max) & (variables.dld_y_det > y_min)
            mask_det = mask_det_x & mask_det_y
        else:
            mask_det = np.ones(len(variables.dld_x_det), dtype=bool)

        if range_mc:
            mc_min, mc_max = sorted((range_mc[0], range_mc[1]))
            mask_mc = (variables.mc_uc < mc_max) & (variables.mc_uc > mc_min)
        else:
            mask_mc = np.ones(len(variables.mc), dtype=bool)

        if range_x and range_y and range_z:
            x_min, x_max = sorted((range_x[0], range_x[1]))
            y_min, y_max = sorted((range_y[0], range_y[1]))
            z_min, z_max = sorted((range_z[0], range_z[1]))
            mask_x = (variables.x < x_max) & (variables.x > x_min)
            mask_y = (variables.y < y_max) & (variables.y > y_min)
            mask_z = (variables.z < z_max) & (variables.z > z_min)
            mask_3d = mask_x & mask_y & mask_z
        else:
            mask_3d = np.ones(len(variables.x), dtype=bool)

        if range_vol:
            vol_min, vol_max = sorted((range_vol[0], range_vol[1]))
            mask_vol = (variables.volume < vol_max) & (variables.volume > vol_min)
        else:
            mask_vol = np.ones(len(variables.x), dtype=bool)

        mask_f = mask_sequence & mask_det & mask_mc & mask_3d & mask_vol
        if verbose:
            print('The number of data sequence:', len(mask_sequence[mask_sequence]))
            print('The number of data mc:', len(mask_mc[mask_mc]))
            print('The number of data det:', len(mask_det[mask_det]))
            print('The number of data 3d:', len(mask_3d[mask_3d]))
            print('The number of data after cropping:', len(mask_f[mask_f]))
        return mask_f

    return np.ones(len(variables.x), dtype=bool)


def build_element_mask(variables, element_name, base_mask=None):
    """Build a boolean mask for all ranged ions containing a given element."""
    if variables.range_data is None or variables.range_data.empty:
        raise ValueError('Range data must be available to build an element mask')

    matching_rows = variables.range_data[
        variables.range_data['element'].apply(lambda elements: element_name in elements)
    ]
    if matching_rows.empty:
        raise ValueError(f'{element_name} is not present in the range dataset')

    mask = np.zeros(len(variables.mc), dtype=bool)
    for _, row in matching_rows.iterrows():
        mask |= (variables.mc > row['mc_low']) & (variables.mc < row['mc_up'])

    if base_mask is not None:
        mask &= base_mask
    return mask


def filter_isosurface_by_size(isosurf, min_vertices=100, largest_n=None):
    """Remove small disconnected isosurface components."""
    if isosurf.n_points == 0:
        return isosurf

    try:
        components = isosurf.connectivity(largest=False)
        region_ids = components.point_data['RegionId']
        unique_regions = np.unique(region_ids)

        filtered_meshes = []
        for index, region_id in enumerate(unique_regions):
            mask = region_ids == region_id
            n_vertices = int(np.sum(mask))
            keep_region = n_vertices >= int(min_vertices)
            if largest_n is not None and index >= int(largest_n):
                keep_region = False
            if not keep_region:
                continue

            region_mesh = components.extract_points(mask, adjacent_cells=False)
            if isinstance(region_mesh, pv.UnstructuredGrid):
                region_mesh = region_mesh.extract_surface()
            filtered_meshes.append(region_mesh)

        if not filtered_meshes:
            return pv.PolyData()

        result = filtered_meshes[0]
        for mesh in filtered_meshes[1:]:
            result = result + mesh

        if isinstance(result, pv.UnstructuredGrid):
            result = result.extract_surface()
        return result.clean()
    except Exception as exc:
        print(f'Unable to filter disconnected isosurface regions: {exc}')
        return isosurf


def calculate_element_isosurface(variables, element_name, bin_values, base_mask=None, smoothing_sigma=1.0,
                                 min_atoms_per_voxel=10, min_vertices=20, fig_name=None):
    """Create a filtered isosurface mesh for a single interface element."""
    if base_mask is None:
        base_mask = np.ones(len(variables.x), dtype=bool)
    if not np.any(base_mask):
        raise ValueError('No ions are available inside the requested plotting range')

    element_mask = build_element_mask(variables, element_name, base_mask=base_mask)
    if not np.any(element_mask):
        raise ValueError(f'No {element_name} ions are available inside the requested plotting range')

    coords = np.column_stack([variables.x[base_mask], variables.y[base_mask], variables.z[base_mask]])
    species_mask = element_mask[base_mask]
    if len(coords) < 4 or np.sum(species_mask) < 4:
        raise ValueError(f'Not enough ions are available to build the {element_name} isosurface')

    bin_centers, _ = bin_vectors_from_distance(coords, bin_values, mode='distance')
    grid_vec = [bin_centers[0], bin_centers[1], bin_centers[2]]
    vox = pos_to_voxel(coords, grid_vec)
    vox_ion = pos_to_voxel(coords, grid_vec, species=species_mask)
    conc = np.divide(vox_ion, vox, out=np.zeros_like(vox_ion, dtype=float), where=vox != 0)
    conc[vox < max(1, int(min_atoms_per_voxel))] = 0

    if float(smoothing_sigma) > 0:
        conc_for_iso = gaussian_filter(conc, sigma=float(smoothing_sigma))
    else:
        conc_for_iso = conc

    iso_value = calculate_iso_value(
        conc_for_iso,
        save_path=variables.result_path,
        fig_name=fig_name,
    )
    isosurf = isosurface(grid_vec, conc_for_iso, isovalue=iso_value)
    isosurf = filter_isosurface_by_size(isosurf, min_vertices=max(1, int(min_vertices)))

    return {
        'mesh': isosurf,
        'iso_value': iso_value,
        'grid_vec': grid_vec,
        'coords': coords,
        'concentration': conc_for_iso,
        'voxel_counts': vox,
    }


def reconstruction_plot(variables, element_percentage, opacity, rotary_fig_save, figname, save, make_gif=False,
                        range_sequence=[], range_mc=[], range_detx=[], range_dety=[],
                        range_x=[], range_y=[], range_z=[], range_vol=[], ions_individually_plots=False,
                        max_num_ions=None, min_num_ions=None, isosurface_dic=None, detailed_isotope_charge=False,
                        only_iso=False, cluster_result=None, smoothing_sigma=1.0, min_atoms_per_voxel=10,
                        min_isosurface_vertices=20):
    """
    Generate a 3D plot for atom probe reconstruction data.

    Args:
        variables (DataFrame): variables object contains daraframe with the data.
        element_percentage (list): Percentage of each element to plot.
        opacity (float): Opacity of the ions.
        rotary_fig_save (bool): Whether to save the rotary figure.
        figname (str): Name of the figure.
        save (bool): Whether to save the figure.
        make_gif (bool): Whether to make a GIF.
        range_sequence (list): Sequence of the range data.
        range_mc (list): Mass-to-charge ratio of the range data.
        range_detx (list): Detector x-coordinate of the range data.
        range_dety (list): Detector y-coordinate of the range data.
        range_x (list): x-coordinate of the range data.
        range_y (list): y-coordinate of the range data.
        range_z (list): z-coordinate of the range data.
        range_vol (list): Volume of the range data.
        ions_individually_plots (bool): Whether to plot each ion individually.
        max_num_ions (int): Maximum number of ions to plot.
        min_num_ions (int): Minimum number of ions to plot.
        isosurface_dic (dic): Dictionary with the isosurface elements and their values.
        detailed_isotope_charge (bool): Whether to plot the range of each isotopes and charge state.
        only_iso (bool): Whether to plot only the isosurface.
        cluster_result: Optional Min-Max precipitate segmentation overlay.

    Returns:
        None
    """
    if isosurface_dic is not None:
        if type(isosurface_dic) is not dict:
            print('The isosurface_dic should be a dictionary')
            isosurface_dic = None
            print('The isosurface_dic is set to None')
    mask_f = build_range_mask(
        variables,
        range_sequence=range_sequence,
        range_mc=range_mc,
        range_detx=range_detx,
        range_dety=range_dety,
        range_x=range_x,
        range_y=range_y,
        range_z=range_z,
        range_vol=range_vol,
        verbose=True,
    )

    if isinstance(element_percentage, list):
        pass
    else:
        print('element_percentage should be a list')

    # Draw an edge of cube around the 3D plot
    x_range = [min(variables.x), max(variables.x)]
    y_range = [min(variables.y), max(variables.y)]
    z_range = [min(variables.z), max(variables.z)]
    range_cube = [x_range, y_range, z_range]

    def _safe_random_subset(mask_s, fraction):
        true_indices = np.flatnonzero(mask_s)
        if len(true_indices) == 0:
            return np.zeros_like(mask_s, dtype=bool)

        size = int(len(true_indices) * float(fraction))
        if min_num_ions is not None:
            size = max(size, int(min_num_ions))
        if max_num_ions is not None:
            size = min(size, int(max_num_ions))
        size = min(max(size, 1), len(true_indices))

        if size == len(true_indices):
            chosen = true_indices
        else:
            chosen = np.random.choice(true_indices, size=size, replace=False)

        sampled_mask = np.zeros_like(mask_s, dtype=bool)
        sampled_mask[chosen] = True
        return sampled_mask

    def _resolve_iso_element(ion_label):
        if isosurface_dic is None:
            return None
        for iso_element in isosurface_elements_list:
            if iso_element in ion_label:
                return iso_element
        return None

    indices_iso = []
    if isosurface_dic is not None:
        if variables.range_data is None:
            raise ValueError('Range data must be provided to plot isosurfaces')
        else:
            isosurface_elements_list = list(isosurface_dic.keys())
            # check if the data dataframe has the elements columns
            # # Flatten the lists in the 'element' column and find unique elements
            unique_elements = pd.Series([elem for sublist in variables.range_data['element'] for elem in sublist]).unique()
            unique_elements = list(unique_elements)
            for elem in isosurface_elements_list:
                if elem not in unique_elements:
                    raise ValueError(f'{elem} for isosurface is not in the range dataset')

            for iso_elem in isosurface_elements_list:
                # Apply a lambda function to check if target_element is in the list for each row
                indices_iso.append(variables.range_data.index[variables.range_data['element'].apply(lambda x: iso_elem in x)].tolist())
            # remove duplicates from the list
            indices_iso = list(set([item for sublist in indices_iso for item in sublist]))

    # Create a subplots with shared axes
    if variables.range_data is not None:

        colors = variables.range_data['color'].tolist()
        mc_low = variables.range_data['mc_low'].tolist()
        mc_up = variables.range_data['mc_up'].tolist()
        ion = variables.range_data['ion'].tolist()
        element = variables.range_data['element'].tolist()
        complex = variables.range_data['complex'].tolist()
        # add the noise color and name
        colors.append('#000000')
        ion.append('$noise$')
        mask_noise = np.full(len(variables.mc), False)

        if element_percentage is None:
            print('The element percentage is not provided, setting it to 0.01')
            element_percentage = [0.01] * len(ion)
            element_percentage[-1] = 0.0001  # add the noise percentage
        else:
            element_percentage.append(0.0001) # add the noise percentage

        if not detailed_isotope_charge:

            # Create the ion list
            ion_s = []
            for elems, comps in zip(element, complex):
                ion_slec = format_ion(elems, comps)
                ion_s.append(ion_slec)
            # Find duplicate indexes in the ions list
            ion_to_indexes = {}
            for idx, ion_e in enumerate(ion_s):
                if ion_e not in ion_to_indexes:
                    ion_to_indexes[ion_e] = []
                ion_to_indexes[ion_e].append(idx)

            ion_new = []
            colors_new = []
            mask_new = []
            element_percentage_new = []
            mask_noise = np.full(len(variables.mc), False)
            for ion_k, indexes in ion_to_indexes.items():
                ion_new.append(ion_k)
                colors_new.append(colors[indexes[0]])
                element_percentage_new.append(element_percentage[indexes[0]])
                mask_tmp = np.full(len(variables.mc), False)
                for idx in indexes:
                    mask_tmp = mask_tmp | ((variables.mc > mc_low[idx]) & (variables.mc < mc_up[idx]))
                mask_new.append(mask_tmp)
                mask_noise = mask_noise | mask_tmp
            ion_new.append('$noise$')
            colors_new.append('#000000')
            element_percentage_new.append(0.0001)
            mask_new.append(mask_noise)
            ion = ion_new
            colors = colors_new
            element_percentage = element_percentage_new
            if isosurface_dic is not None:
                indices_iso = [t for t, s in enumerate(ion_new) if any(t in s for t in isosurface_elements_list)]
            print('The ions with the same name are:', ion_new)


        print('The noise plot percentage is set to 0.0001')
        if ions_individually_plots:
            num_plots = len(ion)
            rows = (num_plots // 3) + (1 if num_plots % 3 != 0 else 0)
            cols = 3
            subplot_titles = ion
            # Generate the specs dictionary based on the number of rows and columns
            specs = [[{"type": "scatter3d", "rowspan": 1, "colspan": 1} for _ in range(cols)] for _ in range(rows)]

            fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles,
                                start_cell="top-left", specs=specs)
            for row in range(rows):
                for col in range(cols):
                    index = col + row * 3
                    if index == len(ion):
                        break
                    if detailed_isotope_charge:
                        if ion[index] == 'noise':
                            mask_s = mask_noise
                        else:
                            mask_s = (variables.mc > mc_low[index]) & (variables.mc < mc_up[index])
                            mask_noise = mask_noise | mask_s
                    else:
                        mask_s = mask_new[index]
                    new_mask = _safe_random_subset(mask_s, element_percentage[index])
                    mask = mask_s & new_mask & mask_f
                    if index in indices_iso:
                        iso_element = _resolve_iso_element(ion[index])
                        if iso_element is not None:
                            iso_result = calculate_element_isosurface(
                                variables,
                                iso_element,
                                isosurface_dic[iso_element],
                                base_mask=mask_f,
                                smoothing_sigma=smoothing_sigma,
                                min_atoms_per_voxel=min_atoms_per_voxel,
                                min_vertices=min_isosurface_vertices,
                                fig_name=f'{figname}_{iso_element}',
                            )
                            isosurf = iso_result['mesh']
                            if isosurf.n_points > 0 and isosurf.n_cells > 0:
                                vertices = isosurf.points
                                faces = isosurf.faces.reshape(-1, 4)[:, 1:]
                                ion_name = ion[index].rsplit('$', 1)[0]
                                ion_name = ion_name + '_{iso}~' + '(%s)' % (element_percentage[index]) + '$'
                                mesh = go.Mesh3d(
                                    x=vertices[:, 0],
                                    y=vertices[:, 1],
                                    z=vertices[:, 2],
                                    i=faces[:, 0],
                                    j=faces[:, 1],
                                    k=faces[:, 2],
                                    opacity=opacity,
                                    alphahull=5,
                                    color=colors[index],
                                    name=ion_name,
                                    showlegend=True
                                )
                                fig = reconstruction.draw_qube(fig, range_cube, col, row)
                                fig.add_trace(mesh, row=row + 1, col=col + 1)

                    ion_name = ion[index].rsplit('$', 1)[0]
                    ion_name = ion_name + '~' + '(%s)' % (element_percentage[index]) + '$'
                    scatter = go.Scatter3d(
                        x=variables.x[mask],
                        y=variables.y[mask],
                        z=variables.z[mask],
                        mode='markers',
                        name=ion_name,
                        showlegend=True,
                        marker=dict(
                            size=1,
                            color=colors[index],
                            opacity=opacity,
                        )
                    )
                    fig = reconstruction.draw_qube(fig, range_cube, col, row)

                    fig.add_trace(scatter, row=row + 1, col=col + 1)
            if cluster_result is not None:
                print('Cluster overlay is shown only in the combined 3D iso-surface plot mode.')
        else:
            fig = go.Figure()
            for index, elemen in enumerate(ion):
                if detailed_isotope_charge:
                    if ion[index] == 'noise':
                        mask_s = mask_noise
                    else:
                        mask_s = (variables.mc > mc_low[index]) & (variables.mc < mc_up[index])
                        mask_noise = mask_noise | mask_s
                else:
                    mask_s = mask_new[index]
                new_mask = _safe_random_subset(mask_s, element_percentage[index])
                mask = mask_s & new_mask & mask_f
                if index in indices_iso:
                    iso_element = _resolve_iso_element(ion[index])
                    if iso_element is not None:
                        iso_result = calculate_element_isosurface(
                            variables,
                            iso_element,
                            isosurface_dic[iso_element],
                            base_mask=mask_f,
                            smoothing_sigma=smoothing_sigma,
                            min_atoms_per_voxel=min_atoms_per_voxel,
                            min_vertices=min_isosurface_vertices,
                            fig_name=f'{figname}_{iso_element}',
                        )
                        isosurf = iso_result['mesh']
                        if isosurf.n_points > 0 and isosurf.n_cells > 0:
                            vertices = isosurf.points
                            faces = isosurf.faces.reshape(-1, 4)[:, 1:]
                            ion_name = ion[index].rsplit('$', 1)[0]
                            ion_name = ion_name + '_{iso}$'
                            mesh = go.Mesh3d(
                                x=vertices[:, 1],
                                y=vertices[:, 0],
                                z=vertices[:, 2],
                                i=faces[:, 0],
                                j=faces[:, 1],
                                k=faces[:, 2],
                                opacity=opacity,
                                alphahull=5,
                                color=colors[index],
                                name=ion_name,
                                showlegend=True
                            )
                            fig.add_trace(mesh)

                if not only_iso:
                    ion_name = ion[index].rsplit('$', 1)[0]
                    ion_name = ion_name + '~' + '(%s)' % (element_percentage[index]) + '$'
                    fig.add_trace(
                        go.Scatter3d(
                            x=variables.x[mask],
                            y=variables.y[mask],
                            z=variables.z[mask],
                            mode='markers',
                            name=ion_name,
                            showlegend=True,
                            marker=dict(
                                size=1,
                                color=colors[index],
                                opacity=opacity,
                            )
                        )
                    )

            if cluster_result is not None:
                for trace in build_cluster_scatter_traces(variables, cluster_result, opacity=min(1.0, opacity + 0.25)):
                    fig.add_trace(trace)

            fig = reconstruction.draw_qube(fig, range_cube)
    else:
        if max_num_ions is None:
            print('The maximum number of ions is not provided, setting it to 100,000')
            max_num_ions = 100_000
        sample_size = min(len(variables.x), int(max_num_ions))
        mask = np.random.choice(len(variables.x), size=sample_size, replace=False)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=variables.x[mask],
                y=variables.y[mask],
                z=variables.z[mask],
                mode='markers',
                name='ions' + ' ' + '(%s)' % (max_num_ions/len(variables.x)*100),
                showlegend=True,
                marker=dict(
                    size=1,
                    opacity=opacity,
                )
            )
        )

        fig = reconstruction.draw_qube(fig, range_cube)

    if rotary_fig_save or make_gif:
        rotary_fig(go.Figure(fig), rotary_fig_save, make_gif, figname)

    fig.update_layout(
    scene=dict(
        aspectmode='auto',
    ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.99
        )
    )

    config = dict(
        {
            'scrollZoom': True,
            'displayModeBar': True,
            'modeBarButtonsToAdd': [
                'drawline',
                'drawopenpath',
                'drawclosedpath',
                'drawcircle',
                'drawrect',
                'eraseshape'
            ]
        }
    )

    fig.show(config=config)
    # Set the renderer to 'browser'
    pio.renderers.default = 'browser'
    fig.show(config=config)

    if save:
        try:
            # fig1 = go.Figure(fig)
            fig.update_scenes(
                camera=dict(
                    eye=dict(x=4, y=4, z=4),  # Adjust the camera position for zooming
                )
            )
            write_plotly_html(fig, variables, f"{figname}_3d.html", include_mathjax='cdn')
            fig.update_layout(showlegend=False)
            layout = go.Layout(
                margin=go.layout.Margin(
                    l=0,  # left margin
                    r=0,  # right margin
                    b=0,  # bottom margin
                    t=0,  # top margin
                )
            )
            fig.update_layout(layout)
            write_plotly_image(fig, variables, f"{figname}_3d.png", scale=3, image_format='png')
            write_plotly_image(fig, variables, f"{figname}_3d.svg", scale=3, image_format='svg')
            fig.update_layout(showlegend=True)

            fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
            fig.update_layout(showlegend=False)
            write_plotly_image(fig, variables, f"{figname}_3d_o.png", scale=3, image_format='png')
            write_plotly_image(fig, variables, f"{figname}_3d_o.svg", scale=3, image_format='svg')
            fig.update_layout(showlegend=True)
            write_plotly_html(fig, variables, f"{figname}_3d_o.html", include_mathjax='cdn')
            fig.update_scenes(xaxis_visible=True, yaxis_visible=True, zaxis_visible=True)
        except Exception as e:
            print('The figure could not be saved')
            print(e)

def rotate_z(x, y, z, theta):
    """
    Rotate coordinates around the z-axis.

    Args:
        x (float): x-coordinate.
        y (float): y-coordinate.
        z (float): z-coordinate.
        theta (float): Rotation angle.

    Returns:
        tuple: Rotated coordinates (x, y, z).
    """
    w = x + 1j * y
    return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z


def plotly_fig2array(fig):
    """
    convert Plotly fig to  an array

    Args:
        fig (plotly.graph_objects.Figure): The base figure.

    Returns:
        array: The array representation of the figure.
    """
    # convert Plotly fig to  an array
    fig_bytes = pio.to_image(fig, format="jpeg", scale=5, engine="kaleido")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)


def rotary_fig(fig, variables, rotary_fig_save, make_gif, figname):
    """
    Generate a rotating figure using Plotly.

    Args:
        fig (plotly.graph_objects.Figure): The base figure.
        rotary_fig_save (bool): Whether to save the rotary figure.
        make_gif (bool): Whether to make a GIF.
        figname (str): The name of the figure.

    Returns:
        None
    """
    x_eye = -1.25
    y_eye = 2
    z_eye = 0.5
    fig = go.Figure(fig)

    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)

    if make_gif:
        fig.update_layout(showlegend=False)
        layout = go.Layout(
            margin=go.layout.Margin(
                l=0,  # left margin
                r=0,  # right margin
                b=0,  # bottom margin
                t=0,  # top margin
            )
        )
        fig.update_layout(layout)

        figures = []
        for t in np.arange(0, 4, 0.2):
            xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, t)
            rotated_fig = go.Figure(fig)
            rotated_fig.update_layout(scene_camera_eye=dict(x=xe, y=ye, z=ze))
            figures.append(rotated_fig)

        images = []
        print('Starting to process the frames for the GIF')
        print('The total number of frames is:', len(figures))
        for index, frame in enumerate(figures):
            images.append(plotly_fig2array(frame))
            print('frame', index, 'is being processed')
        print('The images are ready for the GIF')

        # Save the images as a GIF using imageio
        save_gif(images, variables, f"rota_{figname}.gif", fps=2)

        fig.update_layout(showlegend=True)

    if rotary_fig_save:
        fig.update_layout(
            scene_camera_eye=dict(x=x_eye, y=y_eye, z=z_eye),
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    y=1.2,
                    x=0.8,
                    xanchor='left',
                    yanchor='bottom',
                    pad=dict(t=45, r=10),
                    buttons=[
                        dict(
                            label='Play',
                            method='animate',
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=15, redraw=True),
                                    transition=dict(duration=0),
                                    fromcurrent=True,
                                    mode='immediate'
                                )
                            ]
                        )
                    ]
                )
            ]
        )

        frames = []

        for t in np.arange(0, 50, 0.1):
            xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
            frames.append(go.Frame(layout=dict(scene_camera_eye=dict(x=xe, y=ye, z=ze))))
        fig.frames = frames

        save_plotly_animation(
            fig,
            variables,
            filename=f"rota_{figname}.html",
            show_link=True,
            auto_open=False,
            include_mathjax='cdn',
        )


def format_ion(elements, complexities):
    ion_parts = []
    if elements == 'unranged':
        elements = ['unranged']
        complexities = [complexities]
    for el, comp in zip(elements, complexities):
        if comp > 1:
            ion_parts.append(f"{el}_{{{comp}}}")
        else:
            ion_parts.append(el)
    return "$" + "".join(ion_parts) + "$"

def bin_vectors_from_distance(dist, bin_values, mode='distance'):
    """
    Create a set of grid vectors to be used in nD binning. The bounds are calculated
    such that they don't go beyond the size of the dataset.

    Args:
        dist (numpy.ndarray): The distance variable to be binned. One column per dimension.
                              It is the generalized distance.
        bin_values (list or numpy.ndarray): The bin 'distance' per bin in either a distance metric or a count.
                                             Non-isometric bins are possible.
        mode (str): Mode can be 'distance' (constant distance) or 'count' (constant count). Default is 'distance'.

    Returns:
        tuple:
            - bin_centers (list of numpy.ndarray): The bin centers of each bin.
            - bin_edges (list of numpy.ndarray): The edges of each bin.
    """
    if mode not in ['distance', 'count']:
        raise ValueError("Mode must be 'distance' or 'count'.")

    is_constant_count = mode == 'count'
    is_constant_distance = mode == 'distance'
    num_dim = len(bin_values)
    # if dist is list of numpy arrays, convert to numpy array and reshape it
    if isinstance(dist, list):
        dist = np.array(dist).reshape(-1, num_dim)

    if dist.shape[1] != num_dim:
        raise ValueError("Dimensions of distance variable and bin variable must match.")
    if is_constant_count and num_dim != 1:
        raise ValueError("Constant count mode is only available for 1D binning.")

    bin_centers = []
    bin_edges = []

    # Constant bin distance interval
    if is_constant_distance:
        for dim in range(num_dim):
            # dmin = dist[:, dim].min()
            # dmax = dist[:, dim].max()
            # Generate raw bin vector
            bin_vector_raw = np.linspace(0, 10000 * bin_values[dim], 10001)
            bin_vector_raw = np.concatenate((-np.flip(bin_vector_raw[1:]), bin_vector_raw))

            # Filter bin centers within the distance range
            centers = bin_vector_raw[
                (bin_vector_raw >= dist[:, dim].min() - bin_values[dim]) &
                (bin_vector_raw <= dist[:, dim].max() + bin_values[dim])
            ]
            bin_centers.append(centers)

            # Calculate bin edges
            edges = (centers[1:] + centers[:-1]) / 2
            edges = np.concatenate((
                [centers[0] - (centers[1] - centers[0]) / 2],
                edges,
                [centers[-1] + (centers[-1] - centers[-2]) / 2]
            ))
            bin_edges.append(edges)

    # Constant bin count interval
    elif is_constant_count:
        dist = np.sort(dist.flatten())
        idx_edge = np.arange(0, len(dist), bin_values[0])

        # Handle remainder
        if idx_edge[-1] < len(dist):
            idx_edge = np.append(idx_edge, len(dist))

        idx_cent = np.round((idx_edge[1:] + idx_edge[:-1]) / 2).astype(int)
        centers = dist[idx_cent]
        edges = dist[idx_edge]

        # Adjust edges to avoid creating extra bins
        edges[0] -= 0.0001
        edges[-1] += 0.0001

        bin_centers.append(centers)
        bin_edges.append(edges)

    return bin_centers, bin_edges


def pos_to_voxel(data, grid_vec, species=None):
    """
    Creates a voxelization of the data in 'pos' based on the bin centers in 'grid_vec'
    for the atoms/ions in the specified species.

    Args:
        data (pyccapt DataFrame): The data to be voxelized. when input species is given, ranges must be allocated.
%          A decomposed DataFrame file is also possible. Use range_to_pyccapt to decompose the data.
        grid_vec (list of numpy.ndarray): Grid vectors for the voxel grid. These are the bin centers.
        species (list, str, or numpy.ndarray, optional): The species to filter by. Can be:
                                                         - List of species names (e.g., ['Fe', 'Mn']).
                                                         - Boolean array matching the length of `pos`.
                                                         - None, to include all atoms/ions.

    Returns:
        numpy.ndarray: A 3D array representing the voxelized data.
    """
    # Ensure `pos` is a numpy array
    if hasattr(data, "columns"):  # Assume pandas.DataFrame
        # pos_array = np.array([data["x (nm)"], data["y (nm)"], data["z (nm)"]]).T
        x = data["x (nm)"].to_numpy()
        y = data["y (nm)"].to_numpy()
        z = data["z (nm)"].to_numpy()
        pos_array = np.column_stack([x, y, z])
    elif isinstance(data, list):
            pos_array = np.array(data).T
    else:
        pos_array = data

    # Check for species filtering
    if species is not None:
        if isinstance(species, list):
            element_col = data.columns.get_loc("element") if "element" in data.columns else None
            species_mask = np.full(len(data), False)
            if element_col:
                for s in species:
                    mask_s = data['element'].apply(lambda x: s in x)
                    species_mask |= mask_s

            else:
                raise ValueError("Invalid species filter or table format.")
        elif isinstance(species, np.ndarray) and species.dtype == bool:
            species_mask = species
        else:
            raise ValueError("Species must be a list, boolean array, or None.")

        pos_array = pos_array[species_mask]

    # Calculate bin sizes and edge vectors
    bin_sizes = [
        grid_vec[d][1] - grid_vec[d][0] for d in range(3)
    ]
    edge_vec = [
        np.concatenate(([grid_vec[d][0] - bin_sizes[d] / 2],
                        grid_vec[d] + bin_sizes[d] / 2))
        for d in range(3)
    ]

    # Determine voxel indices
    loc = np.empty((pos_array.shape[0], 3), dtype=int)
    for d in range(3):
        loc[:, d] = np.digitize(pos_array[:, d], edge_vec[d]) - 1  # Adjust for 0-based indexing

    # Calculate the voxel grid size
    grid_size = np.maximum(np.max(loc, axis=0) + 1, [len(e) - 1 for e in edge_vec])

    # Count atoms in each voxel
    vox = np.zeros(grid_size, dtype=int)
    for i in range(loc.shape[0]):
        vox[tuple(loc[i])] += 1

    return vox.T

def isosurface(gridVec, data, isovalue):
    """
    Extract isosurface using pyvista for a custom 3D grid.

    Args:
        gridVec (list of np.ndarray): List of 3 arrays representing the grid points in x, y, and z.
        data (np.ndarray): 3D scalar field (same shape as the meshgrid defined by gridVec).
        isovalue (float): Scalar value to extract the isosurface.

    Returns:
        pyvista.PolyData: Isosurface with faces and vertices.
    """
    reordered_gridVec = [gridVec[0], gridVec[1], gridVec[2]]

    # Create a pyvista structured grid
    x, y, z = np.meshgrid(reordered_gridVec[0], reordered_gridVec[1], reordered_gridVec[2], indexing='ij')
    grid = pv.StructuredGrid(x, y, z)
    grid.point_data["values"] = data.flatten()

    # Extract the isosurface
    isosurf = grid.contour([isovalue])  # Pass isovalue as a list for compatibility
    return isosurf



def calculate_iso_value(conc, save_path=None, fig_name=None):
    """
    Calculate the optimal iso value from a 3D array and save the histogram plot.

    Args:
        conc (numpy.ndarray): 3D array of concentration values.
        save_path (str): Directory to save the histogram plot.

    Returns:
        float: Optimal iso value.
    """

    def find_first_trough_after_peak(hist, first_peak_idx):
        """Finds the index of the first trough after a given peak in a histogram."""
        for i in range(first_peak_idx + 1, len(hist)):
            if hist[i] < hist[i - 1]:  # Check if current value is less than the previous
                # Check if it's a local minimum (trough)
                if i + 1 < len(hist) and hist[i] < hist[i + 1]:
                    return i
        return None

    # Flatten the 3D array
    conc_flat = conc.flatten()

    # Define bin size and compute bin edges
    bin_size = 0.001
    min_val, max_val = np.min(conc_flat), np.max(conc_flat)
    bin_edges = np.arange(min_val, max_val + bin_size, bin_size)

    # Create the histogram
    hist, bin_edges = np.histogram(conc_flat, bins=bin_edges)

    # Find peaks in the histogram
    peaks, _ = find_peaks(hist, prominence=20)

    # Check if any peaks are found
    if len(peaks) == 0:
        print("No peaks found in the histogram to calculate the optimum iso value.")
        iso_val = (max_val + min_val) / 2
    else:
        # Identify the first peak
        first_peak_idx = peaks[0]

        # Find the first trough after the peak using the helper function
        trough_idx = find_first_trough_after_peak(hist, first_peak_idx)

        # Handle the case where no trough is found after the peak
        if trough_idx is None:
            print("No trough found after the first peak.")
            # Use a fallback strategy, e.g., the midpoint between the peak and max value
            iso_val = (bin_edges[first_peak_idx] + max_val) / 2

        else:
            # Calculate the corresponding isovalue
            iso_val = bin_edges[trough_idx]
    print(f"Default Iso Value: {iso_val} is set")

    if save_path is not None and fig_name is not None:
        if len(peaks) == 0:
            print("No peaks found in the histogram to calculate the optimum iso value.")
        else:
            # Plot the histogram with the identified peak and trough
            plt.figure(figsize=(10, 6))
            plt.hist(conc_flat, bins=bin_edges, color='blue', alpha=0.7, edgecolor='black', label="Histogram")
            plt.axvline(bin_edges[first_peak_idx], color='red', linestyle='--', label="First Peak")
            plt.axvline(iso_val, color='green', linestyle='--', label="Optimal Iso Value")
            plt.title("Histogram with Detected Iso Value", fontsize=16)
            plt.xlabel("Value", fontsize=14)
            plt.ylabel("Frequency", fontsize=14)
            plt.yscale('log')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)

            # Ensure the save directory exists
            plot_path = os.path.join(save_path, f"histogram_with_iso_value_{fig_name}.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()  # Close the plot to prevent it from displaying

    print(f"Optimal Iso Value: {iso_val}")

    return iso_val
