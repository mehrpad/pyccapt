import os
from typing import Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from pyccapt.calibration.reconstructions import iso_surface


def _interface_vertices_and_normals(
    interface: Union[pv.PolyData, Dict[str, np.ndarray]], flip_normals: bool = False
):
    """Return (oriented_surface, vertices, point_normals).

    The oriented surface carries BOTH point and cell normals so the
    signed-distance computation can project onto the nearest FACE rather
    than only the nearest vertex (more accurate on coarse / clipped
    meshes). Vertices + point normals are still returned for the
    nearest-vertex fallback.
    """
    if isinstance(interface, pv.PolyData):
        surf = interface
    elif isinstance(interface, dict) and "vertices" in interface and "faces" in interface:
        vertices = np.asarray(interface["vertices"], dtype=np.float64)
        faces = np.asarray(interface["faces"])
        if faces.ndim == 2 and faces.shape[1] == 3:
            faces_pv = np.c_[np.full((faces.shape[0], 1), 3, dtype=int), faces].ravel()
        else:
            faces_pv = faces.ravel()
        surf = pv.PolyData(vertices, faces_pv)
    else:
        raise ValueError("interface must be a pyvista.PolyData or {'vertices','faces'} dict.")

    surf = surf.clean()
    try:
        normed = surf.compute_normals(
            cell_normals=True,
            point_normals=True,
            auto_orient_normals=True,
            inplace=False,
        )
    except TypeError:
        normed = surf.compute_normals(point_normals=True, inplace=False)

    vertices = normed.points.astype(np.float64, copy=False)
    normals = normed.point_normals.astype(np.float64, copy=False)
    if flip_normals:
        normals = -normals
        # Keep the cell normals consistent with the (flipped) point
        # normals so the face-based sign matches the vertex-based sign.
        try:
            if "Normals" in normed.cell_data:
                normed.cell_data["Normals"] = -np.asarray(normed.cell_data["Normals"])
        except Exception:
            pass
    return normed, vertices, normals


def _signed_distance_to_interface(points, surf, vertices, normals):
    """Signed perpendicular distance from each point to the interface.

    Prefers the FACE-based projection: ``find_closest_cell`` gives the
    closest point ON the surface (on a face, not just a vertex) plus the
    cell it lies on; the sign comes from the cell normal. Falls back to
    the nearest-vertex normal when the pyvista build doesn't support
    ``return_closest_point`` or cell normals are unavailable.
    """
    cell_normals = None
    try:
        cell_normals = surf.cell_normals
    except Exception:
        cell_normals = None

    if cell_normals is not None and surf.n_cells > 0:
        try:
            cell_ids, closest_pts = surf.find_closest_cell(points, return_closest_point=True)
            cell_ids = np.asarray(cell_ids).reshape(-1)
            closest_pts = np.asarray(closest_pts, dtype=np.float64).reshape(-1, 3)
            cn = np.asarray(cell_normals, dtype=np.float64)[cell_ids]
            return np.einsum("ij,ij->i", cn, points - closest_pts)
        except Exception:
            # Any incompatibility -> fall through to nearest-vertex.
            pass

    # Fallback: nearest-vertex normal (the original behaviour).
    tree = cKDTree(vertices)
    _, idx = tree.query(points, k=1)
    return np.einsum("ij,ij->i", normals[idx], points - vertices[idx])


def _stack_xyz(x, y, z) -> np.ndarray:
    """Stack (x, y, z) into an (N, 3) float64 array.

    Partial-recovered rows have NaN coordinates because their detector
    position was undefined. Proxigram distance computation uses
    ``cKDTree`` which rejects NaN inputs. Drop NaN rows here so the
    caller never has to know about the partials.
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    z = np.asarray(z).reshape(-1)
    if not (x.shape == y.shape == z.shape):
        raise ValueError("x, y, z must have same length.")
    coords = np.column_stack((x, y, z)).astype(np.float64, copy=False)
    nan_row_mask = np.isnan(coords).any(axis=1)
    if nan_row_mask.any():
        print(
            f'[proxigram._stack_xyz] Dropping {int(nan_row_mask.sum())} rows with '
            'NaN (x, y, z) (partial-recovered ions) before distance computation.'
        )
        coords = coords[~nan_row_mask]
    return coords


def _matlab_bin_centers(dist: np.ndarray, bin_nm: float) -> Tuple[np.ndarray, np.ndarray]:
    # Size the raw centre grid from the actual signed-distance span
    # rather than a fixed 10001-entry grid capped at 10000*bin_nm. The
    # fixed grid silently clipped distances beyond +/- 500 nm (with the
    # default sub-nm bin) into the end bin, producing a wrong proxigram
    # for large interfaces, while over-allocating for small ones.
    reach = max(abs(float(dist.min())), abs(float(dist.max()))) + bin_nm
    n_steps = max(1, int(np.ceil(reach / float(bin_nm))))
    centers_raw = np.arange(0, (n_steps + 1)) * float(bin_nm)
    centers_raw = np.concatenate((-centers_raw[1:][::-1], centers_raw))
    mask = (centers_raw >= dist.min()) & (centers_raw <= dist.max())
    centers = centers_raw[mask]
    if centers.size < 3:
        span = max(bin_nm, (dist.max() - dist.min()) / 2.0 + 1e-12)
        centers = np.array([-span, 0.0, span])

    steps = np.diff(centers)
    step = np.median(steps) if steps.size else bin_nm
    edges = np.concatenate(([centers[0] - step / 2.0], (centers[1:] + centers[:-1]) / 2.0, [centers[-1] + step / 2.0]))
    return centers, edges


def patch_create_proxigram_mask(
    pos_x,
    pos_y,
    pos_z,
    mask,
    interface: Union[pv.PolyData, Dict[str, np.ndarray]],
    bin_nm: float,
    flip_normals: bool = False,
    symmetric_range: float = None,
):
    """Calculate a proxigram concentration profile for one species mask."""
    points_all = _stack_xyz(pos_x, pos_y, pos_z)
    points_species = _stack_xyz(pos_x[mask], pos_y[mask], pos_z[mask])

    surf, vertices, normals = _interface_vertices_and_normals(interface, flip_normals=flip_normals)

    # Face-based signed distance (closest point on a face, sign from the
    # cell normal); falls back to the nearest-vertex normal internally.
    dist_all = _signed_distance_to_interface(points_all, surf, vertices, normals)
    dist_species = _signed_distance_to_interface(points_species, surf, vertices, normals)

    if symmetric_range is not None:
        dist_all = dist_all[np.abs(dist_all) <= symmetric_range]
        dist_species = dist_species[np.abs(dist_species) <= symmetric_range]

    bin_vector, edges = _matlab_bin_centers(dist_all, bin_nm)
    pos_hist, _ = np.histogram(dist_all, bins=edges)
    species_hist, _ = np.histogram(dist_species, bins=edges)

    with np.errstate(divide="ignore", invalid="ignore"):
        proxigram_fraction = np.divide(
            species_hist.astype(float),
            pos_hist,
            out=np.zeros_like(species_hist, dtype=float),
            where=pos_hist > 0,
        )

    return proxigram_fraction, bin_vector


def plot_proxigram(
    variables,
    isosurface_dic,
    proxigram_elements,
    figname='proxigram',
    save=False,
    bin_size=0.1,
    symmetric_range=None,
    flip_normals=False,
    range_sequence=None,
    range_mc=None,
    range_detx=None,
    range_dety=None,
    range_x=None,
    range_y=None,
    range_z=None,
    range_vol=None,
    smoothing_sigma=1.0,
    min_atoms_per_voxel=10,
    min_isosurface_vertices=20,
    pure_only=False,
    manual_iso_value=None,
):
    """Plot proxigram lines relative to a single interface isosurface."""
    if not isinstance(isosurface_dic, dict) or not isosurface_dic:
        raise ValueError('Isosurface definition must be a non-empty dictionary')
    if len(isosurface_dic) != 1:
        raise ValueError('Proxigram currently supports exactly one interface isosurface entry')

    interface_element, bin_values = next(iter(isosurface_dic.items()))
    if not proxigram_elements:
        raise ValueError('Please provide at least one proxigram element to plot')

    mask_f = iso_surface.build_range_mask(
        variables,
        range_sequence=range_sequence or [],
        range_mc=range_mc or [],
        range_detx=range_detx or [],
        range_dety=range_dety or [],
        range_x=range_x or [],
        range_y=range_y or [],
        range_z=range_z or [],
        range_vol=range_vol or [],
        verbose=False,
    )
    if not np.any(mask_f):
        raise ValueError('No ions remain after applying the requested plotting ranges')

    iso_result = iso_surface.calculate_element_isosurface(
        variables,
        interface_element,
        bin_values,
        base_mask=mask_f,
        smoothing_sigma=smoothing_sigma,
        min_atoms_per_voxel=min_atoms_per_voxel,
        min_vertices=min_isosurface_vertices,
        fig_name=f'{figname}_{interface_element}',
        pure_only=pure_only,
        manual_iso_value=manual_iso_value,
    )
    interface_mesh = iso_result['mesh']
    if interface_mesh.n_points == 0 or interface_mesh.n_cells == 0:
        raise ValueError(f'No valid isosurface could be extracted for {interface_element}')

    x = variables.x[mask_f]
    y = variables.y[mask_f]
    z = variables.z[mask_f]

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0
    for element_name in proxigram_elements:
        element_name = str(element_name).strip()
        if not element_name:
            continue

        try:
            species_mask = iso_surface.build_element_mask(variables, element_name, base_mask=mask_f, pure_only=pure_only)[
                mask_f
            ]
        except ValueError as exc:
            print(exc)
            continue

        if not np.any(species_mask):
            print(f'No {element_name} ions are available inside the requested plotting range')
            continue

        prox_frac, bin_vector = patch_create_proxigram_mask(
            x,
            y,
            z,
            species_mask,
            interface_mesh,
            bin_nm=float(bin_size),
            flip_normals=bool(flip_normals),
            symmetric_range=symmetric_range,
        )
        ax.plot(bin_vector, prox_frac * 100.0, linewidth=2, label=element_name)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        raise ValueError('No proxigram curves could be generated for the requested elements')

    ax.set_xlabel("Distance from interface [nm]", fontsize=12)
    ax.set_ylabel("Concentration [at.%]", fontsize=12)
    ax.set_ylim(0, 100)

    xlim_min, xlim_max = ax.get_xlim()
    if symmetric_range is not None:
        extent = float(abs(symmetric_range))
    else:
        extent = min(abs(xlim_min), abs(xlim_max))
    if extent > 0:
        ax.set_xlim(-extent, extent)

    ax.axvline(0, color='k', linestyle='--', linewidth=1, alpha=0.6, label='Interface')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='best')

    y_pos = ax.get_ylim()[1] * 0.95
    ax.text(-extent * 0.5 if extent > 0 else -0.5, y_pos, 'Precipitate', ha='center', va='top', fontsize=10, alpha=0.6)
    ax.text(extent * 0.5 if extent > 0 else 0.5, y_pos, 'Matrix', ha='center', va='top', fontsize=10, alpha=0.6)

    if extent > 0:
        ax.axvspan(-extent, 0, alpha=0.05, color='blue')
        ax.axvspan(0, extent, alpha=0.05, color='red')

    ax.set_title(f'Proxigram relative to {interface_element} isosurface')

    from pyccapt.calibration.core import plot_style
    plot_style.finalize_axes(ax)  # round-number axes with end ticks (paper styling)

    if save:
        os.makedirs(variables.result_path, exist_ok=True)
        fig.savefig(os.path.join(variables.result_path, f'proxigram_{figname}.png'), dpi=300, format='png')

    plt.show()
    return fig
