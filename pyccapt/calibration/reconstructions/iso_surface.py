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


from pyccapt.calibration.clustering import build_cluster_context_trace, build_cluster_scatter_traces
from pyccapt.calibration.reconstructions import reconstruction
from pyccapt.calibration.reconstructions.io_utils import (
    save_gif,
    save_plotly_animation,
    write_plotly_html,
    write_plotly_image,
)


def build_range_mask(
    variables,
    range_sequence=None,
    range_mc=None,
    range_detx=None,
    range_dety=None,
    range_x=None,
    range_y=None,
    range_z=None,
    range_vol=None,
    verbose=False,
):
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


def _normalize_element_list(elements):
    """Return a normalized list of element labels for one range row."""
    return [str(element).strip() for element in elements if str(element).strip()]


def _row_matches_element(elements, element_name, pure_only=False):
    """Return whether a range row matches an element selector."""
    normalized = _normalize_element_list(elements)
    if not normalized:
        return False
    if pure_only:
        return all(element == element_name for element in normalized)
    return element_name in normalized


def _match_isosurface_target(elements, targets, pure_only=False):
    """Return the first matching isosurface target for a row element list."""
    for target in targets:
        if _row_matches_element(elements, target, pure_only=pure_only):
            return target
    return None


def build_element_mask(variables, element_name, base_mask=None, pure_only=False):
    """Build a boolean mask for ranged ions matching a given element selector."""
    if variables.range_data is None or variables.range_data.empty:
        raise ValueError('Range data must be available to build an element mask')

    matching_rows = variables.range_data[
        variables.range_data['element'].apply(
            lambda elements: _row_matches_element(elements, element_name, pure_only=pure_only)
        )
    ]
    if matching_rows.empty:
        qualifier = ' as a pure-element species' if pure_only else ''
        raise ValueError(f'{element_name} is not present in the range dataset{qualifier}')

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
            region_mesh = _extract_surface_compat(region_mesh)
            filtered_meshes.append(region_mesh)

        if not filtered_meshes:
            return pv.PolyData()

        result = filtered_meshes[0]
        for mesh in filtered_meshes[1:]:
            result = result + mesh

        result = _extract_surface_compat(result)
        return result.clean()
    except Exception as exc:
        print(f'Unable to filter disconnected isosurface regions: {exc}')
        return isosurf


def _extract_surface_compat(mesh):
    """Extract a surface mesh across PyVista versions."""
    if not isinstance(mesh, pv.UnstructuredGrid):
        return mesh
    try:
        return mesh.extract_surface(algorithm='dataset_surface')
    except TypeError:
        return mesh.extract_surface()


def _select_points_inside_surface_compat(points, surface):
    """Mark query points inside a closed surface across PyVista versions."""
    if hasattr(points, 'select_interior_points'):
        selected = points.select_interior_points(surface)
        mask = selected.point_data.get('selected_points', np.array([]))
        return np.asarray(mask, dtype=bool)

    if hasattr(points, 'select_enclosed_points'):
        selected = points.select_enclosed_points(surface, check_surface=False)
        mask = selected.point_data.get('SelectedPoints', np.array([]))
        return np.asarray(mask, dtype=bool)

    if hasattr(points, 'compute_implicit_distance'):
        selected = points.compute_implicit_distance(surface)
        distances = selected.point_data.get('implicit_distance', np.array([]))
        return np.asarray(distances, dtype=float) <= 0

    raise AttributeError('No supported PyVista point-in-surface method is available')


def _structured_grid_from_volume(grid_vec, data, scalar_name):
    """Build a PyVista structured grid from a volume on the reconstruction grid."""
    x, y, z = np.meshgrid(grid_vec[0], grid_vec[1], grid_vec[2], indexing='ij')
    grid = pv.StructuredGrid(x, y, z)
    grid.point_data[scalar_name] = np.asarray(data, dtype=float).flatten()
    return grid


def _pad_grid_and_volume(grid_vec, data, pad_width=1):
    """Pad a voxel volume with empty space so extracted surfaces can close at the boundary."""
    padded_data = np.pad(np.asarray(data), int(pad_width), mode='constant', constant_values=0)
    padded_grid = []
    for axis_values in grid_vec:
        axis_values = np.asarray(axis_values, dtype=float)
        if axis_values.size > 1:
            step = float(np.median(np.diff(axis_values)))
        else:
            step = 1.0
        start = float(axis_values[0]) - step * int(pad_width)
        stop = float(axis_values[-1]) + step * int(pad_width)
        padded_grid.append(np.linspace(start, stop, axis_values.size + 2 * int(pad_width)))
    return padded_grid, padded_data


def _build_specimen_surface_from_voxels(grid_vec, voxel_counts, smoothing_sigma=1.0):
    """Build a closed specimen-envelope surface from occupied voxel counts."""
    occupancy = (np.asarray(voxel_counts) > 0).astype(float)
    if not np.any(occupancy):
        return pv.PolyData()

    envelope_sigma = 0.0 if float(smoothing_sigma) <= 0 else min(1.25, max(0.75, float(smoothing_sigma)))
    if envelope_sigma > 0:
        envelope = gaussian_filter(occupancy, sigma=envelope_sigma)
    else:
        envelope = occupancy

    positive_values = envelope[envelope > 0]
    if positive_values.size == 0:
        return pv.PolyData()

    threshold = max(0.05, min(0.35, 0.25 * float(np.max(positive_values))))
    padded_grid_vec, padded_envelope = _pad_grid_and_volume(grid_vec, envelope, pad_width=1)
    surface = isosurface(padded_grid_vec, padded_envelope, isovalue=threshold)
    if surface.n_points == 0 or surface.n_cells == 0:
        return pv.PolyData()
    surface = filter_isosurface_by_size(surface, min_vertices=50, largest_n=1)
    if surface.n_points == 0 or surface.n_cells == 0:
        return pv.PolyData()
    surface = surface.triangulate().clean()
    try:
        hole_size = float(np.linalg.norm(np.array(surface.bounds)[1::2] - np.array(surface.bounds)[::2]))
        surface = surface.fill_holes(hole_size).clean()
    except Exception:
        pass
    return surface


def _clip_isosurface_to_specimen_envelope(mesh, grid_vec, voxel_counts, smoothing_sigma=1.0):
    """Clip an isosurface so it stays inside the closed specimen surface."""
    if mesh is None or mesh.n_points == 0 or mesh.n_cells == 0:
        return mesh

    specimen_surface = _build_specimen_surface_from_voxels(
        grid_vec,
        voxel_counts,
        smoothing_sigma=smoothing_sigma,
    )
    if specimen_surface.n_points == 0 or specimen_surface.n_cells == 0:
        return mesh.triangulate().clean()

    tri_mesh = mesh.triangulate().clean()
    cell_centers = tri_mesh.cell_centers()
    cell_keep = _select_points_inside_surface_compat(cell_centers, specimen_surface)
    if cell_keep.size == 0:
        return tri_mesh
    if not np.any(cell_keep):
        # Every cell was clipped away -- the iso-surface fell entirely
        # outside the specimen envelope. Warn loudly: silently returning
        # an empty mesh makes a legitimate iso-surface disappear from the
        # plot with no diagnostic (common on small-cell-count specimens).
        print(
            '[_clip_isosurface_to_specimen_envelope] WARNING: all '
            f'{cell_keep.size} iso-surface cells fell outside the specimen '
            'envelope and were clipped; returning an empty mesh. Check the '
            'min_atoms_per_voxel / smoothing parameters or the iso value.'
        )
        return pv.PolyData()

    clipped = tri_mesh.extract_cells(np.flatnonzero(cell_keep))
    clipped = _extract_surface_compat(clipped)
    return clipped.clean()


def calculate_element_isosurface(
    variables,
    element_name,
    bin_values,
    base_mask=None,
    smoothing_sigma=1.0,
    min_atoms_per_voxel=10,
    min_vertices=20,
    fig_name=None,
    pure_only=False,
    manual_iso_value=None,
):
    """Create a filtered isosurface mesh for a single interface element."""
    if base_mask is None:
        base_mask = np.ones(len(variables.x), dtype=bool)
    # Drop partial-recovered rows from the base mask. Their (x, y, z) is
    # NaN because the detector position was incomplete -- including them
    # in ``coords`` poisons the voxel-binning / scipy isosurface step.
    finite_xyz = (
        np.isfinite(np.asarray(variables.x))
        & np.isfinite(np.asarray(variables.y))
        & np.isfinite(np.asarray(variables.z))
    )
    base_mask = base_mask & finite_xyz
    if not np.any(base_mask):
        raise ValueError('No ions are available inside the requested plotting range')

    element_mask = build_element_mask(variables, element_name, base_mask=base_mask, pure_only=pure_only)
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
    reliable_mask = vox >= max(1, int(min_atoms_per_voxel))
    if not np.any(reliable_mask):
        reliable_mask = vox > 0
    conc = np.divide(vox_ion, vox, out=np.zeros_like(vox_ion, dtype=float), where=vox != 0)
    conc[~reliable_mask] = 0

    if float(smoothing_sigma) > 0:
        conc_for_iso = gaussian_filter(conc, sigma=float(smoothing_sigma))
    else:
        conc_for_iso = conc.copy()

    if not np.any(conc_for_iso > 0):
        raise ValueError(
            f'No positive concentration voxels remain for {element_name}. Try lowering min atoms / voxel or smoothing sigma.'
        )

    if manual_iso_value is not None and float(manual_iso_value) > 0:
        iso_value = float(manual_iso_value)
    else:
        iso_value = calculate_iso_value(
            conc_for_iso,
            save_path=variables.result_path,
            fig_name=fig_name,
        )
    print(f'Iso value used for {element_name}: {iso_value:.6g}')
    isosurf = isosurface(grid_vec, conc_for_iso, isovalue=iso_value)
    isosurf = _clip_isosurface_to_specimen_envelope(
        isosurf,
        grid_vec,
        vox,
        smoothing_sigma=smoothing_sigma,
    )
    isosurf = filter_isosurface_by_size(isosurf, min_vertices=max(1, int(min_vertices)))

    return {
        'mesh': isosurf,
        'iso_value': iso_value,
        'grid_vec': grid_vec,
        'coords': coords,
        'concentration': conc_for_iso,
        'voxel_counts': vox,
    }


def reconstruction_plot(
    variables,
    element_percentage,
    opacity,
    rotary_fig_save,
    figname,
    save,
    make_gif=False,
    range_sequence=[],
    range_mc=[],
    range_detx=[],
    range_dety=[],
    range_x=[],
    range_y=[],
    range_z=[],
    range_vol=[],
    ions_individually_plots=False,
    max_num_ions=None,
    min_num_ions=None,
    isosurface_dic=None,
    detailed_isotope_charge=False,
    only_iso=False,
    cluster_result=None,
    smoothing_sigma=1.0,
    min_atoms_per_voxel=10,
    min_isosurface_vertices=20,
    pure_element_only=False,
    manual_iso_value=None,
    cluster_display_mode='overlay',
):
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
        cluster_display_mode (str): `overlay` or `clusters-only`.

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
        verbose=False,
    )
    cluster_only = cluster_result is not None and str(cluster_display_mode).strip().lower() == 'clusters-only'

    if isinstance(element_percentage, list):
        pass
    else:
        print('element_percentage should be a list')

    # Draw an edge of cube around the 3D plot. Use nanmin/nanmax so
    # partial-recovered rows (NaN x/y/z from undefined reconstruction)
    # don't poison the cube edges; Python's builtin min/max return NaN
    # if any element is NaN.
    x_range = [float(np.nanmin(variables.x)), float(np.nanmax(variables.x))]
    y_range = [float(np.nanmin(variables.y)), float(np.nanmax(variables.y))]
    z_range = [float(np.nanmin(variables.z)), float(np.nanmax(variables.z))]
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
            # Seed the local Generator from the input mask's length so
            # repeated plots of the same data give the same subsample
            # (the previous ``np.random.choice`` used the global state
            # and the same notebook re-run produced different scatter
            # points each time).
            _rng = np.random.default_rng(int(len(true_indices)))
            chosen = _rng.choice(true_indices, size=size, replace=False)

        sampled_mask = np.zeros_like(mask_s, dtype=bool)
        sampled_mask[chosen] = True
        return sampled_mask

    indices_iso = []
    ion_iso_targets = []
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

            row_iso_targets = [
                _match_isosurface_target(elements, isosurface_elements_list, pure_only=pure_element_only)
                for elements in variables.range_data['element']
            ]
            indices_iso = [index for index, target in enumerate(row_iso_targets) if target is not None]

    # Create a subplots with shared axes
    if variables.range_data is not None:
        colors = reconstruction._normalize_plotly_colors(variables.range_data['color'].tolist())
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
            element_percentage.append(0.0001)  # add the noise percentage

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
            ion_iso_targets_new = []
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
                ion_iso_target = None
                if isosurface_dic is not None:
                    for idx in indexes:
                        ion_iso_target = _match_isosurface_target(
                            element[idx], isosurface_elements_list, pure_only=pure_element_only
                        )
                        if ion_iso_target is not None:
                            break
                ion_iso_targets_new.append(ion_iso_target)
            ion_new.append('$noise$')
            colors_new.append('#000000')
            element_percentage_new.append(0.0001)
            mask_new.append(mask_noise)
            ion_iso_targets_new.append(None)
            ion = ion_new
            colors = colors_new
            element_percentage = element_percentage_new
            ion_iso_targets = ion_iso_targets_new
            if isosurface_dic is not None:
                indices_iso = [idx for idx, target in enumerate(ion_iso_targets) if target is not None]
        elif isosurface_dic is not None:
            ion_iso_targets = [
                _match_isosurface_target(elements_row, isosurface_elements_list, pure_only=pure_element_only)
                for elements_row in element
            ]
            ion_iso_targets.append(None)
            indices_iso = [idx for idx, target in enumerate(ion_iso_targets) if target is not None]

        if ions_individually_plots:
            num_plots = len(ion)
            rows = (num_plots // 3) + (1 if num_plots % 3 != 0 else 0)
            cols = 3
            subplot_titles = ion
            # Generate the specs dictionary based on the number of rows and columns
            specs = [[{"type": "scatter3d", "rowspan": 1, "colspan": 1} for _ in range(cols)] for _ in range(rows)]

            fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles, start_cell="top-left", specs=specs)
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
                        iso_element = ion_iso_targets[index]
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
                                pure_only=pure_element_only,
                                manual_iso_value=manual_iso_value,
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
                                    showlegend=True,
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
                        ),
                    )
                    fig = reconstruction.draw_qube(fig, range_cube, col, row)

                    fig.add_trace(scatter, row=row + 1, col=col + 1)
            if cluster_result is not None:
                print('Cluster overlay is shown only in the combined 3D iso-surface plot mode.')
        else:
            fig = go.Figure()
            drawn_iso_targets = set()
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
                    iso_element = ion_iso_targets[index]
                    if iso_element is not None and iso_element not in drawn_iso_targets:
                        iso_result = calculate_element_isosurface(
                            variables,
                            iso_element,
                            isosurface_dic[iso_element],
                            base_mask=mask_f,
                            smoothing_sigma=smoothing_sigma,
                            min_atoms_per_voxel=min_atoms_per_voxel,
                            min_vertices=min_isosurface_vertices,
                            fig_name=f'{figname}_{iso_element}',
                            pure_only=pure_element_only,
                            manual_iso_value=manual_iso_value,
                        )
                        isosurf = iso_result['mesh']
                        if isosurf.n_points > 0 and isosurf.n_cells > 0:
                            vertices = isosurf.points
                            faces = isosurf.faces.reshape(-1, 4)[:, 1:]
                            ion_name = ion[index].rsplit('$', 1)[0]
                            ion_name = ion_name + '_{iso}$'
                            # Axis order MUST match the per-ion-grid branch
                            # above (vertices[:, 0]=x, vertices[:, 1]=y);
                            # the previous swap reflected every iso-surface
                            # across y=x relative to the underlying scatter.
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
                                showlegend=True,
                            )
                            fig.add_trace(mesh)
                            drawn_iso_targets.add(iso_element)

                if not only_iso and not cluster_only:
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
                            ),
                        )
                    )

            if cluster_result is not None:
                if cluster_only:
                    background_trace = build_cluster_context_trace(
                        variables,
                        mask=mask_f & ~cluster_result.selected_mask,
                        name='Background specimen',
                        opacity=min(0.18, max(0.04, opacity * 0.3)),
                        marker_size=0.9,
                    )
                    if background_trace is not None:
                        fig.add_trace(background_trace)

                    unclustered_trace = build_cluster_context_trace(
                        variables,
                        mask=mask_f & cluster_result.selected_mask & (cluster_result.labels < 0),
                        name='Selected ions outside clusters',
                        color='rgba(120,120,120,0.85)',
                        opacity=min(0.35, max(0.10, opacity * 0.55)),
                        marker_size=1.4,
                    )
                    if unclustered_trace is not None:
                        fig.add_trace(unclustered_trace)

                for trace in build_cluster_scatter_traces(
                    variables,
                    cluster_result,
                    opacity=min(1.0, opacity + 0.25),
                    valid_mask=mask_f,
                ):
                    fig.add_trace(trace)

            fig = reconstruction.draw_qube(fig, range_cube)
    else:
        if max_num_ions is None:
            print('The maximum number of ions is not provided, setting it to 100,000')
            max_num_ions = 100_000
        sample_size = min(len(variables.x), int(max_num_ions))
        # Same reproducibility seed as _safe_random_subset above: keyed
        # on the dataset length, so re-plots are deterministic.
        _rng = np.random.default_rng(int(len(variables.x)))
        mask = _rng.choice(len(variables.x), size=sample_size, replace=False)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=variables.x[mask],
                y=variables.y[mask],
                z=variables.z[mask],
                mode='markers',
                name='ions' + ' ' + '(%s)' % (max_num_ions / len(variables.x) * 100),
                showlegend=True,
                marker=dict(
                    size=1,
                    opacity=opacity,
                ),
            )
        )

        fig = reconstruction.draw_qube(fig, range_cube)

    if rotary_fig_save or make_gif:
        rotary_fig(go.Figure(fig), rotary_fig_save, make_gif, figname)

    fig.update_layout(
        scene=dict(
            aspectmode='auto',
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.99),
    )

    config = dict(
        {
            'scrollZoom': True,
            'displayModeBar': True,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'],
        }
    )

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
        from tqdm.auto import tqdm

        fig.update_layout(showlegend=False)
        fig.update_layout(margin=go.layout.Margin(l=0, r=0, b=0, t=0))

        # Reuse a single figure and only swap the camera per frame — deep
        # copying the figure each iteration was the dominant cost when the
        # scene carries an iso-surface / large mesh.
        thetas = np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / 20.0)
        images = []
        for theta in tqdm(thetas, desc="Iso-surface GIF frames", unit="frame"):
            xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, theta)
            fig.update_layout(scene_camera_eye=dict(x=xe, y=ye, z=ze))
            images.append(plotly_fig2array(fig))

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
                                    mode='immediate',
                                ),
                            ],
                        )
                    ],
                )
            ],
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
            # Size the raw bin vector from the ACTUAL data span instead of
            # a fixed 10001-entry grid capped at 10000*bin. The old fixed
            # grid silently clipped any point beyond +/- 500 nm (e.g. a
            # 600 nm specimen with 0.05 nm bins), lumping out-of-range
            # ions into the end bin -- a silently wrong histogram -- while
            # also over-allocating 20001 entries for small specimens.
            bin = float(bin_values[dim])
            dmin = float(dist[:, dim].min())
            dmax = float(dist[:, dim].max())
            reach = max(abs(dmin), abs(dmax)) + bin
            n_steps = max(1, int(np.ceil(reach / bin)))
            bin_vector_raw = np.arange(0, (n_steps + 1)) * bin
            bin_vector_raw = np.concatenate((-np.flip(bin_vector_raw[1:]), bin_vector_raw))

            # Filter bin centers within the distance range
            centers = bin_vector_raw[
                (bin_vector_raw >= dist[:, dim].min() - bin_values[dim])
                & (bin_vector_raw <= dist[:, dim].max() + bin_values[dim])
            ]
            bin_centers.append(centers)

            # Calculate bin edges
            edges = (centers[1:] + centers[:-1]) / 2
            edges = np.concatenate(
                ([centers[0] - (centers[1] - centers[0]) / 2], edges, [centers[-1] + (centers[-1] - centers[-2]) / 2])
            )
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
    bin_sizes = [grid_vec[d][1] - grid_vec[d][0] for d in range(3)]
    edge_vec = [np.concatenate(([grid_vec[d][0] - bin_sizes[d] / 2], grid_vec[d] + bin_sizes[d] / 2)) for d in range(3)]

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

    # ``vox`` is indexed [ix, iy, iz] (matches np.meshgrid(..., indexing='ij'));
    # the downstream ``isosurface`` builder also assumes 'ij' ordering and
    # flattens the data with the same convention. Returning ``vox.T`` here
    # silently transposed to [iz, iy, ix] and put voxel concentration at the
    # wrong spatial coordinate. Mirror the leap-clustering copy in
    # clustering/isosurface.py and return ``vox`` directly.
    return vox


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

    # Flatten the 3D array and ignore empty background voxels when estimating the threshold.
    conc_flat = np.asarray(conc, dtype=float).ravel()
    conc_flat = conc_flat[np.isfinite(conc_flat)]
    positive_values = conc_flat[conc_flat > 0]
    if positive_values.size > 0:
        conc_flat = positive_values
    if conc_flat.size == 0:
        return 0.0

    # Define bin size and compute bin edges
    bin_size = 0.001
    min_val, max_val = np.min(conc_flat), np.max(conc_flat)
    if np.isclose(min_val, max_val):
        return float(max_val)
    num_bins = max(32, int(np.ceil((max_val - min_val) / bin_size)))
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)

    # Create the histogram
    hist, bin_edges = np.histogram(conc_flat, bins=bin_edges)

    # Find peaks in the histogram
    peaks, _ = find_peaks(hist, prominence=20)

    if len(peaks) == 0:
        iso_val = (max_val + min_val) / 2
    else:
        # Identify the first peak
        first_peak_idx = peaks[0]

        # Find the first trough after the peak using the helper function
        trough_idx = find_first_trough_after_peak(hist, first_peak_idx)

        # Handle the case where no trough is found after the peak
        if trough_idx is None:
            iso_val = (bin_edges[first_peak_idx] + max_val) / 2

        else:
            # Calculate the corresponding isovalue
            iso_val = bin_edges[trough_idx]

    if save_path is not None and fig_name is not None:
        if len(peaks) > 0:
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

    return iso_val
