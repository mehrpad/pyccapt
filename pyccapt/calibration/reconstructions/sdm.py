from copy import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib import cm
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

from pyccapt.calibration.data_tools.merge_range import merge_by_range
from pyccapt.calibration.reconstructions.io_utils import save_matplotlib_figure


# Half-width of the SDM acceptance box, in nm. Pairs whose per-axis
# displacement exceeds this are discarded. Matches the hardcoded
# +/- 1 nm cut applied downstream in ``sdm``.
_SDM_BOX_HALF_WIDTH_NM = 1.0


def _compute_sdm_displacements(*, particles, mask_i, mask_j, axes, z_cut, theta, phi,
                               target_working_bytes=256_000_000):
    """Return box-cut 1-D pair displacements (dx, dy, dz) for the SDM.

    Dispatcher. With >=2 box-cut axes the surviving pairs lie in a small
    +/- 1 nm hyperbox, so a cKDTree narrows the candidate set from
    O(N_i x N_j) to O(N x neighbours) and the EXACT box cut is re-applied
    (bit-identical to the brute-force reference) -- ~40-60x faster on realistic
    2D/3D SDMs. With <=1 cut axis (e.g. the 1D z-only calibration SDM, or the
    z_cut=False keep-all case) selectivity is poor and the candidate lists get
    large, so the KDTree would REGRESS wall-clock; use the brute-force scan.
    """
    cut_cols = []
    if 'x' in axes:
        cut_cols.append(0)
    if 'y' in axes:
        cut_cols.append(1)
    if 'z' in axes and z_cut:
        cut_cols.append(2)
    if len(cut_cols) < 2:
        return _compute_sdm_displacements_bruteforce(
            particles=particles, mask_i=mask_i, mask_j=mask_j, axes=axes,
            z_cut=z_cut, theta=theta, phi=phi, target_working_bytes=target_working_bytes,
        )
    return _compute_sdm_displacements_kdtree(
        particles=particles, mask_i=mask_i, mask_j=mask_j, axes=axes,
        z_cut=z_cut, theta=theta, phi=phi, cut_cols=cut_cols,
    )


def _compute_sdm_displacements_kdtree(*, particles, mask_i, mask_j, axes, z_cut, theta, phi,
                                      cut_cols, chunk_points=8192):
    """cKDTree fast path for the box-cut SDM (>=1 cut axis).

    Builds a tree on the cut-axis projection of the j-particles, queries the
    enclosing ball (radius sqrt(n_cut) * box, the smallest sphere containing
    the [-box, box]^n_cut hyperbox, plus a tiny epsilon for FP safety), then
    re-applies the EXACT per-axis box cut from the brute-force path on the
    candidate pairs. Output displacements use the same original-coordinate
    delta + rotation formula as the brute force, so results are bit-identical.
    """
    want_x = 'x' in axes
    want_y = 'y' in axes
    want_z = 'z' in axes
    rotated = not (theta == 0 and phi == 0)
    box = _SDM_BOX_HALF_WIDTH_NM

    pts_i = particles[mask_i]
    pts_j = particles[mask_j]
    n_i = len(pts_i)
    n_j = len(pts_j)

    def _empty():
        e = np.empty(0, dtype=float)
        return (e.copy() if want_x else None,
                e.copy() if want_y else None,
                e.copy() if want_z else None)

    if n_i == 0 or n_j == 0:
        return _empty()

    if rotated:
        ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
        rot = np.array([[ct, st * sp, st * cp],
                        [0.0, cp, -sp],
                        [-st, ct * sp, ct * cp]])
        proj_i = (pts_i @ rot.T)[:, cut_cols]
        proj_j = (pts_j @ rot.T)[:, cut_cols]
    else:
        proj_i = pts_i[:, cut_cols]
        proj_j = pts_j[:, cut_cols]

    tree_j = cKDTree(np.ascontiguousarray(proj_j))
    radius = float(np.sqrt(len(cut_cols)) * box) + 1e-9

    jx, jy, jz = pts_j[:, 0], pts_j[:, 1], pts_j[:, 2]
    acc_x, acc_y, acc_z = [], [], []

    for start in range(0, n_i, chunk_points):
        stop = min(start + chunk_points, n_i)
        neigh = tree_j.query_ball_point(np.ascontiguousarray(proj_i[start:stop]), radius)
        counts = [len(nb) for nb in neigh]
        total = int(sum(counts))
        if total == 0:
            continue
        i_idx = np.repeat(np.arange(start, stop), counts)
        j_idx = np.fromiter((j for nb in neigh for j in nb), dtype=np.int64, count=total)

        dx = pts_i[i_idx, 0] - jx[j_idx]
        dy = pts_i[i_idx, 1] - jy[j_idx]
        dz = pts_i[i_idx, 2] - jz[j_idx]
        if rotated:
            cur_x = (ct * dx + st * sp * dy + st * cp * dz) if want_x else None
            cur_y = (cp * dy - sp * dz) if want_y else None
            cur_z = (-st * dx + ct * sp * dy + ct * cp * dz) if want_z else None
        else:
            cur_x = dx if want_x else None
            cur_y = dy if want_y else None
            cur_z = dz if want_z else None

        keep = None
        if want_x:
            keep = np.abs(cur_x) <= box
        if want_y:
            cond = np.abs(cur_y) <= box
            keep = cond if keep is None else (keep & cond)
        if want_z and z_cut:
            cond = np.abs(cur_z) <= box
            keep = cond if keep is None else (keep & cond)

        if want_x:
            acc_x.append(cur_x[keep])
        if want_y:
            acc_y.append(cur_y[keep])
        if want_z:
            acc_z.append(cur_z[keep])

    out_x = (np.concatenate(acc_x) if acc_x else np.empty(0, dtype=float)) if want_x else None
    out_y = (np.concatenate(acc_y) if acc_y else np.empty(0, dtype=float)) if want_y else None
    out_z = (np.concatenate(acc_z) if acc_z else np.empty(0, dtype=float)) if want_z else None
    return out_x, out_y, out_z


def _compute_sdm_displacements_bruteforce(*, particles, mask_i, mask_j, axes, z_cut, theta, phi,
                                          target_working_bytes=256_000_000):
    """Brute-force reference for :func:`_compute_sdm_displacements`.

    Iterates over the ``mask_i`` particles in adaptively-sized chunks, applies
    the SAME +/- 1 nm box cut per chunk, and accumulates only the surviving
    displacements. ``None`` is returned for any axis not in ``axes``.

    The box-cut rule: every active non-z axis is cut to [-1, 1]; the z axis is
    cut only when ``z_cut`` is True (so a z-only SDM with ``z_cut=False`` keeps
    every pair).
    """
    want_x = 'x' in axes
    want_y = 'y' in axes
    want_z = 'z' in axes
    rotated = not (theta == 0 and phi == 0)

    pts_i = particles[mask_i]
    pts_j = particles[mask_j]
    n_i = len(pts_i)
    n_j = len(pts_j)

    def _empty():
        e = np.empty(0, dtype=float)
        return (e.copy() if want_x else None,
                e.copy() if want_y else None,
                e.copy() if want_z else None)

    if n_i == 0 or n_j == 0:
        return _empty()

    jx, jy, jz = pts_j[:, 0], pts_j[:, 1], pts_j[:, 2]

    # Adaptive chunk: bound the per-chunk working set. The rotated path
    # materialises up to ~6 (C x n_j) arrays (3 deltas + 3 shifts); the
    # plain path up to 3. Use the larger factor so peak stays bounded.
    per_pair_arrays = 6 if rotated else 3
    bytes_per_row = max(1, n_j * 8 * per_pair_arrays)
    chunk = max(1, int(target_working_bytes // bytes_per_row))

    acc_x, acc_y, acc_z = [], [], []
    box = _SDM_BOX_HALF_WIDTH_NM

    for start in range(0, n_i, chunk):
        ci = pts_i[start:start + chunk]
        # Per-axis deltas for this chunk: (chunk_rows, n_j).
        if rotated:
            dxr = ci[:, 0][:, None] - jx[None, :]
            dyr = ci[:, 1][:, None] - jy[None, :]
            dzr = ci[:, 2][:, None] - jz[None, :]
            cur_x = (np.cos(theta) * dxr
                     + np.sin(theta) * np.sin(phi) * dyr
                     + np.sin(theta) * np.cos(phi) * dzr) if want_x else None
            cur_y = (np.cos(phi) * dyr - np.sin(phi) * dzr) if want_y else None
            cur_z = (-np.sin(theta) * dxr
                     + np.cos(theta) * np.sin(phi) * dyr
                     + np.cos(theta) * np.cos(phi) * dzr) if want_z else None
            del dxr, dyr, dzr
        else:
            cur_x = (ci[:, 0][:, None] - jx[None, :]) if want_x else None
            cur_y = (ci[:, 1][:, None] - jy[None, :]) if want_y else None
            cur_z = (ci[:, 2][:, None] - jz[None, :]) if want_z else None

        # Build the per-chunk keep mask using the exact downstream rule.
        keep = None

        def _and(mask, condition):
            return condition if mask is None else (mask & condition)

        if want_x:
            keep = _and(keep, np.abs(cur_x) <= box)
        if want_y:
            keep = _and(keep, np.abs(cur_y) <= box)
        if want_z and z_cut:
            keep = _and(keep, np.abs(cur_z) <= box)

        if keep is None:
            # Only reachable for the z-only / z_cut=False case: no cut,
            # keep every pair (matches original behaviour).
            if want_x:
                acc_x.append(np.ravel(cur_x))
            if want_y:
                acc_y.append(np.ravel(cur_y))
            if want_z:
                acc_z.append(np.ravel(cur_z))
            continue

        if want_x:
            acc_x.append(cur_x[keep])
        if want_y:
            acc_y.append(cur_y[keep])
        if want_z:
            acc_z.append(cur_z[keep])

    out_x = np.concatenate(acc_x) if want_x else None
    out_y = np.concatenate(acc_y) if want_y else None
    out_z = np.concatenate(acc_z) if want_z else None
    return out_x, out_y, out_z


def sdm(
    particles,
    bin_size,
    variables=None,
    roi=[0, 0, 0.5],
    z_cut=True,
    normalize=False,
    plot_mode='bar',
    plot=False,
    save=False,
    figure_size=(6, 6),
    figname='sdm',
    histogram_type='1d',
    axes=None,
    i_composition=None,
    j_composition=None,
    plot_roi=False,
    theta_x=0,
    phi_y=0,
    log=False,
    frac=1.0,
    range_sequence=[],
    range_mc=[],
    range_detx=[],
    range_dety=[],
    range_x=[],
    range_y=[],
    range_z=[],
    range_vol=[],
):
    """
        Computes 1D or 2D histograms for a set of particle coordinates.

        Parameters
        ----------
        particles : (N, 3) np.array
                Set of particle coordinates for which to compute the SDM.
        bin_size : float
                Bin size for each histogram.
        variables : variables object
        normalize : bool, optional
                Option to normalize the histograms. If True, the histogram values are normalized.
        roi : list, optional
            Region of interest for the SDM. Default is [0, 0, 1].
        z_cut : bool, optional
            Cut the z distances over 1 nm
        plot_mode : str, optional
                The plot mode for the histograms. Options are 'bar' or 'line'.
        plot : bool, optional
                Option to plot the histograms. If True, the histograms are plotted.
        save : bool, optional
                Option to save the histograms. If True, the histograms are saved.
        figure_size : (float, float), optional
                The size of the figure in inches.
        figname : str, optional
                The name of the figure.
        histogram_type : str, optional
                Type of histogram. Options are '1D' or '2D' or '3D'.
        axes : list or None, optional
                Specifies the axes for 1D or 2D histograms. For '1d', provide a list like ['x'], ['y'], or ['z'].
                For '2d', provide a list like ['x', 'y'], ['y', 'z'], or ['x', 'z'] or ['x', 'y', 'z'].
        i_composition : list, optional
            Composition of the first element in the SDM.
        j_composition : list, optional
            Composition of the second element in the SDM.
    plot_roi : bool, optional
        Option to plot the region of interest. If True, the region of interest is plotted.
    theta_x : float, optional
        Rotation angle around the x-axis.
    phi_y : float, optional
        Rotation angle around the y-axis.
    log : bool, optional
        Option to plot the SDM in log scale. If True, the SDM is plotted in log scale.
        frac : float, optional
            Fraction of the second element in the SDM.
        range_sequence : list, optional
            Sequence range for the SDM.
        range_mc : list, optional
            Mass-to-charge range for the SDM.
        range_detx : list, optional
            Detector x-coordinate range for the SDM.
    range_dety : list, optional
        Detector y-coordinate range for the SDM.
    range_x : list, optional
        X-coordinate range for the SDM.
    range_y : list, optional
        Y-coordinate range for the SDM.
    range_z : list, optional
        Z-coordinate range for the SDM.
    range_vol : list, optional
        Volume range for the SDM.

        Returns
        -------
        histograms : list of np.array
                List of 1D or 2D histograms based on user preferences.
        edges : list of np.array
                Bin edges for each histogram.
    """
    if axes is None:
        # Downstream `'x' in axes` checks raise a cryptic TypeError on None.
        # The axis choice is a deliberate analysis decision, so fail with an
        # actionable message rather than guessing a default.
        raise ValueError(
            "axes must be provided, e.g. ['z'] for a 1D SDM or ['x', 'y'] "
            "for a 2D SDM (got None)."
        )
    if range_sequence or range_mc or range_detx or range_dety or range_x or range_y or range_z or range_vol:
        if range_sequence:
            if range_sequence:
                mask_sequence = np.zeros(len(particles), dtype=bool)
                if range_sequence[0] < 1 and range_sequence[1] < 1:
                    mask_sequence[int(len(particles) * range_sequence[0]) : int(len(particles) * range_sequence[1])] = True
                else:
                    mask_sequence[range_sequence[0] : range_sequence[1]] = True
            else:
                mask_sequence = np.zeros(len(particles), dtype=bool)
                mask_sequence[: int(len(particles) * range_sequence)] = True

        else:
            mask_sequence = np.ones(len(particles), dtype=bool)
        if range_detx and range_dety:
            mask_det_x = (variables.dld_x_det < range_detx[1]) & (variables.dld_x_det > range_detx[0])
            mask_det_y = (variables.dld_y_det < range_dety[1]) & (variables.dld_y_det > range_dety[0])
            mask_det = mask_det_x & mask_det_y
        else:
            mask_det = np.ones(len(particles), dtype=bool)
        if range_mc:
            mask_mc = (variables.mc <= range_mc[1]) & (variables.mc >= range_mc[0])
        else:
            mask_mc = np.ones(len(particles), dtype=bool)
        if range_x and range_y and range_z:
            mask_x = (variables.x < range_x[1]) & (variables.x > range_x[0])
            mask_y = (variables.y < range_y[1]) & (variables.y > range_y[0])
            mask_z = (variables.z < range_z[1]) & (variables.z > range_z[0])
            mask_3d = mask_x & mask_y & mask_z
        else:
            mask_3d = np.ones(len(particles), dtype=bool)
        if range_vol:
            mask_vol = (variables.dld_high_voltage < range_vol[1]) & (variables.dld_high_voltage > range_vol[0])
        else:
            mask_vol = np.ones(len(particles), dtype=bool)
        mask = mask_sequence & mask_det & mask_mc & mask_3d & mask_vol
        if variables is not None:
            variables.mask = mask
        print('The number of data sequence:', len(mask_sequence[mask_sequence == True]))
        print('The number of data mc:', len(mask_mc[mask_mc == True]))
        print('The number of data det:', len(mask_det[mask_det == True]))
        print('The number of data 3d:', len(mask_3d[mask_3d == True]))
        print('The number of data vol:', len(mask_vol[mask_vol == True]))
    else:
        mask = np.ones(len(particles), dtype=bool)

    len_true_mask = len(mask[mask == True])

    if frac < 1:
        # set axis limits based on fraction of x and y data based on fraction
        true_indices = np.where(mask)[0]
        num_set_to_flase = int(len(true_indices) * (1 - frac))
        # Seeded RNG so re-plots are deterministic.
        indices_to_set_false = np.random.default_rng(int(len(true_indices))).choice(
            true_indices, num_set_to_flase, replace=False)
        mask[indices_to_set_false] = False
        print('The number of data cropping due to fraction:', len_true_mask - len(mask[mask == True]))

    if variables is not None:
        dist_temp = np.sqrt((variables.dld_x_det - roi[0]) ** 2 + (variables.dld_y_det - roi[1]) ** 2)
        mask_roi = dist_temp <= roi[2]
    else:
        print('Variables is None. ROI selection needs detector coordinates. Only use radius')
        dist_temp = np.sqrt((particles[:, 0]) ** 2 + (particles[:, 1]) ** 2)
        mask_roi = dist_temp <= roi[2]

    print('The number of data ROI:', len(mask_roi[mask_roi == True]))

    mask = mask & mask_roi

    print('The number of data after cropping:', len(mask[mask == True]))
    if variables is not None and i_composition and j_composition:
        if i_composition and isinstance(i_composition, list):
            if 'name' in variables.data.columns:
                data = variables.data
            else:
                if variables.range_data is None:
                    raise ValueError('Range data is not provided')
                data = merge_by_range(variables.data, variables.range_data, full=True)
            mask_i_comp = np.zeros(len(particles), dtype=bool)
            # Create a mask from the composition list of variables.data
            for comp in i_composition:
                mask_i_comp = mask_i_comp | data['name'].apply(lambda x: comp in str(x) if not pd.isna(x) else False)
        else:
            raise ValueError('No list of i composition is provided')
        if j_composition and isinstance(j_composition, list):
            if 'name' in variables.data.columns:
                pass
            else:
                if variables.range_data is None:
                    raise ValueError('Range data is not provided')
                data = merge_by_range(variables.data, variables.range_data, full=True)
            mask_j_comp = np.zeros(len(particles), dtype=bool)
            # Create a mask from the composition list of variables.data
            for comp in j_composition:
                mask_j_comp = mask_j_comp | data['name'].apply(lambda x: comp in str(x) if not pd.isna(x) else False)
        else:
            raise ValueError('No list of j composition is provided')

    if variables is not None:
        if i_composition and isinstance(i_composition, list) and j_composition and isinstance(j_composition, list):
            mask_i = mask & mask_i_comp
            mask_j = mask & mask_j_comp
        else:
            mask_i = mask
            mask_j = mask
    else:
        mask_i = mask
        mask_j = mask

    particles_backup = particles.copy()

    theta = np.radians(theta_x)
    phi = np.radians(phi_y)

    print('The number of ions in is:', len(particles[mask]))
    print('The number of ions in i composition is:', len(particles[mask_i]))
    print('The number of ions in j composition is:', len(particles[mask_j]))
    # Calculate relative pair displacements.
    #
    # MEMORY: the original code built full N_i x N_j matrices via
    # np.subtract.outer for each axis (and a dense (N_i, N_j, 3) array in
    # the rotated branch), then box-cut the result to +/- 1 nm. For a
    # normal-sized specimen (N ~ 1e5 per species) that is ~12 GB per axis
    # and reliably OOMs. Because only pairs inside the +/- 1 nm box
    # survive, we instead iterate over the i particles in adaptively
    # sized chunks and apply the EXACT same box-cut per chunk, keeping
    # only the surviving 1-D displacements. Peak memory is bounded to
    # ~chunk_i x N_j instead of N_i x N_j. The downstream histogram code
    # re-applies the (now idempotent) box-cut, so numerics are identical.
    dx, dy, dz = None, None, None
    histograms = []
    dx, dy, dz = _compute_sdm_displacements(
        particles=particles,
        mask_i=mask_i,
        mask_j=mask_j,
        axes=axes,
        z_cut=z_cut,
        theta=theta,
        phi=phi,
    )

    edges_list = []

    def _symmetric_edges(*arrays):
        finite_arrays = [np.asarray(arr, dtype=float).ravel() for arr in arrays if arr is not None]
        finite_arrays = [arr[np.isfinite(arr)] for arr in finite_arrays if arr.size > 0]
        if not finite_arrays:
            raise ValueError('No valid SDM distances are available to build histogram edges')
        max_abs = max(float(np.max(np.abs(arr))) for arr in finite_arrays)
        max_abs = max(max_abs, float(bin_size))
        return np.arange(-max_abs, max_abs + bin_size, bin_size)

    if 'x' in axes and 'y' in axes and 'z' in axes:
        dx = dx.flatten()
        dy = dy.flatten()
        dz = dz.flatten()
        mask = ((dx <= 1) & (dx >= -1)) & ((dy <= 1) & (dy >= -1))
        if z_cut:
            mask = mask & ((dz <= 1) & (dz >= -1))
        dx = dx[mask]
        dy = dy[mask]
        dz = dz[mask]
        edges = _symmetric_edges(dx, dy, dz)
    elif 'x' in axes and 'y' in axes:
        mask = ((dx <= 1) & (dx >= -1)) & ((dy <= 1) & (dy >= -1))
        dx = dx[mask]
        dy = dy[mask]
        edges = _symmetric_edges(dx, dy)
    elif 'y' in axes and 'z' in axes:
        dy = dy.flatten()
        dz = dz.flatten()
        mask = (dy <= 1) & (dy >= -1)
        if z_cut:
            mask = mask & ((dz <= 1) & (dz >= -1))
        dz = dz[mask]
        dy = dy[mask]
        edges = _symmetric_edges(dy, dz)
    elif 'x' in axes and 'z' in axes:
        dx = dx.flatten()
        dz = dz.flatten()
        mask = (dx <= 1) & (dx >= -1)
        if z_cut:
            mask = mask & ((dz <= 1) & (dz >= -1))
        dx = dx[mask]
        dz = dz[mask]
        edges = _symmetric_edges(dx, dz)
    elif 'x' in axes:
        dx = dx.flatten()
        mask = (dx <= 1) & (dx >= -1)
        dx = dx[mask]
        edges = _symmetric_edges(dx)
    elif 'y' in axes:
        dy = dy.flatten()
        mask = (dy <= 1) & (dy >= -1)
        dy = dy[mask]
        edges = _symmetric_edges(dy)
    elif 'z' in axes:
        dz = dz.flatten()
        if z_cut:
            mask = (dz <= 1) & (dz >= -1)
            dz = dz[mask]
        edges = _symmetric_edges(dz)

    if histogram_type == '1D':
        if 'x' in axes:
            dx = dx[dx != 0]
            histo_dx, bins_dx = np.histogram(dx, bins=edges)
            histograms.append(histo_dx)
            edges_list.append(bins_dx)
        elif 'y' in axes:
            dy = dy[dy != 0]
            histo_dy, bins_dy = np.histogram(dy, bins=edges)
            histograms.append(histo_dy)
            edges_list.append(bins_dy)
        elif 'z' in axes:
            dz = dz[dz != 0]
            histo_dz, bins_dz = np.histogram(dz, bins=edges)
            histograms.append(histo_dz)
            edges_list.append(bins_dz)
        else:
            raise ValueError("Invalid axes for 1D histogram. Choose from ['x'], ['y'], or ['z'].")
        if normalize:
            histograms[-1] = histograms[-1] / np.max(histograms[-1])

    if histogram_type == '2D':
        if 'x' in axes and 'y' in axes:
            mask = ~((dx == 0) & (dy == 0))
            dx = dx[mask]
            dy = dy[mask]
            hist2d, x_edges, y_edges = np.histogram2d(dx, dy, bins=[edges, edges])
            extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
            histograms.append(hist2d)
            edges_list.extend([x_edges, y_edges])
        elif 'y' in axes and 'z' in axes:
            mask = ~((dy == 0) & (dz == 0))
            dy = dy[mask]
            dz = dz[mask]
            hist2d, x_edges, y_edges = np.histogram2d(dy, dz, bins=[edges, edges])
            extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
            histograms.append(hist2d)
            edges_list.extend([x_edges, y_edges])
        elif 'x' in axes and 'z' in axes:
            mask = ~((dx == 0) & (dz == 0))
            dx = dx[mask]
            dz = dz[mask]
            hist2d, x_edges, y_edges = np.histogram2d(dx, dz, bins=[edges, edges])
            extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
            histograms.append(hist2d)
            edges_list.extend([x_edges, y_edges])
        else:
            raise ValueError("Invalid axes for 2D histogram. Choose from ['x', 'y'], ['y', 'z'], or ['x', 'z'].")

        if normalize:
            histograms[-1] = histograms[-1] / np.max(histograms[-1])

    if histogram_type == '3D':
        if 'x' in axes and 'y' in axes and 'z' in axes:
            mask = ~((dx == 0) & (dy == 0) & (dz == 0))
            dx = dx[mask]
            dy = dy[mask]
            dz = dz[mask]
            hist3d, edges = np.histogramdd((dx, dy, dz), bins=[edges, edges, edges])
            histograms.append(hist3d)
            edges_list.extend([edges])
        if normalize:
            histograms[-1] = histograms[-1] / np.max(histograms[-1])

    if plot or save:
        # Plot histograms
        if histogram_type == '1D':
            fig, ax = plt.subplots(figsize=figure_size)
            for i, hist in enumerate(histograms):
                if plot_mode == 'bar':
                    ax.bar(edges[:-1], hist, width=bin_size, align='edge')
                    ax.set_ylabel('Counts')
                    ax.set_xlabel(f'{axes[i]} (nm)')
                elif plot_mode == 'line':
                    ax.plot(edges[:-1], hist)
                    ax.set_ylabel('Counts')
                    ax.set_xlabel(f'{axes[i]} (nm)')
                if log:
                    ax.set_yscale('log')
                try:
                    # Detect peaks
                    peaks, _ = find_peaks(hist, height=0)
                    # Calculate distances between peaks
                    distances = np.diff(peaks)  # Horizontal distances (indices)
                    for i in range(len(peaks) - 1):
                        dx = edges[:-1]
                        # Get coordinates for the peaks
                        x1, x2 = dx[peaks[i]], dx[peaks[i + 1]]
                        y1, y2 = hist[peaks[i]], hist[peaks[i + 1]]

                        # Draw dashed line
                        yy = max(y1, y2)
                        plt.annotate('', xy=(x2, yy), xytext=(x1, yy), arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))

                        # Annotate with distance
                        plt.text((x1 + x2) / 2, max(y1, y2) + 0.1, f'{x2 - x1:.2f}', color='blue', ha='center', va='bottom')
                except Exception as e:
                    print('error:', e)
                    print('No peaks found in the histogram')

        elif histogram_type == '2D':
            if figure_size[0] - figure_size[1] > 2:
                print('The figure size is not appropriate for 2D histogram')
                figure_size = (5, 4)
                print('The figure size is changed to:', figure_size)
            fig, ax = plt.subplots(figsize=figure_size)
            img = ax.imshow(histograms[-1].T, origin='lower', extent=extent, aspect="auto")
            ax.set_ylabel(f'{axes[1]} (nm)')
            ax.set_xlabel(f'{axes[0]} (nm)')
            cmap = copy(plt.cm.plasma)
            cmap.set_bad(cmap(0))
            if log:
                pcm = ax.pcolormesh(x_edges, y_edges, histograms[-1].T, cmap=cmap, norm=colors.LogNorm(), rasterized=True)
            else:
                pcm = ax.pcolormesh(x_edges, y_edges, histograms[-1].T, cmap=cmap, rasterized=True)
            cbar = fig.colorbar(pcm, ax=ax, pad=0)
            cbar.set_label('Counts', fontsize=10)
        elif histogram_type == '3D':
            print('3D histogram is not supported yet.')

        if save and variables is not None:
            save_matplotlib_figure(fig, variables, stem=f"sdm_{figname}", formats=("png", "pdf"), dpi=600)

        if plot:
            plt.show()
        if plot_roi:
            x = particles_backup[:, 0]
            y = particles_backup[:, 1]
            edges_d = np.arange(min(np.min(x), np.min(y)), max(np.max(x), np.max(y)), 0.5)
            FDM, xedges, yedges = np.histogram2d(x, y, bins=(edges_d, edges_d))
            if normalize:
                FDM = FDM / np.max(FDM)
            cmap_instance = copy(cm.get_cmap('plasma'))
            cmap_instance.set_bad(cmap_instance(0))
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            if log and not normalize:
                FDM = np.log1p(FDM)
                pcm = ax1.pcolormesh(xedges, yedges, FDM.T, cmap=cmap_instance, norm=colors.LogNorm(), rasterized=True)
            else:
                pcm = ax1.pcolormesh(xedges, yedges, FDM.T, cmap=cmap_instance, rasterized=True)

            cbar = fig1.colorbar(pcm, ax=ax1, pad=0)
            cbar.set_label('Event Counts', fontsize=10)
            plt.xlabel('x (nm)')
            plt.ylabel('y (nm)')
            # draw a circle with radius roi[2]
            circle = plt.Circle((roi[0], roi[1]), roi[2], color='white', fill=False, linewidth=1.5)
            ax1.add_artist(circle)
            if save and variables is not None:
                save_matplotlib_figure(
                    fig1,
                    variables,
                    stem=f"disparity_roi_{figname}",
                    formats=("png", "pdf"),
                    dpi=600,
                )
            if plot:
                plt.show()

    return histograms, edges_list

# def sdm_background(res, limit):
#     """
#     Performs iterative smoothing of a 1D array with a convergence condition.
#
#     Parameters:
#     - res (np.ndarray): 2D array where the first column represents data points.
#     - limit (float): Convergence threshold for the deviation percentage.
#
#     Returns:
#     - np0 (np.ndarray): The original input data after smoothing.
#     - dev (np.ndarray): Array of deviation values over each iteration.
#     """
#     # Initialize arrays and variables
#     np0 = res[:, 0]  # Original data points
#     np1 = np.zeros_like(np0)  # Array for the new smoothed values
#     dev = np.zeros((1000, 3))  # Pre-allocate deviation array (size can be adjusted)
#
#     # Find the index of the maximum value in the first column
#     id_center = np.argmax(np0)
#
#     # Initialize deviation
#     dev[0, 0] = 0
#     dev[0, 1] = np.inf
#     dev[0, 2] = np.inf
#
#     i = 0
#     while dev[i, 2] >= limit:
#         # Smoothing step: update np1 based on the average of neighboring values
#         for j in range(1, len(np0) - 1):
#             np1[j] = min(np0[j], (np0[j + 1] + np0[j - 1]) / 2)
#
#         # Compute the deviation in the current iteration
#         dev[i + 1, 0] = i + 1
#         dev[i + 1, 1] = res[id_center, 0] - np.max(np1)  # Deviation from max value
#         dev[i + 1, 2] = np.abs(dev[i + 1, 1] - dev[i, 1]) * 100 / res[id_center, 0]  # Relative change in deviation
#
#         # Update the np0 array for the next iteration
#         np0 = np1.copy()
#         np1.fill(0)  # Reset np1 for the next iteration
#
#         i += 1
#
#     # Truncate dev array to the actual number of iterations
#     dev = dev[:i + 1, :]
#
#     return np0, dev
