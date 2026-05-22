import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial import cKDTree

from pyccapt.calibration.reconstructions.io_utils import save_matplotlib_figure

def rdf(
    particles,
    dr,
    variables=None,
    rho=None,
    rcutoff=0.9,
    eps=1e-15,
    normalize=True,
    reference_point=None,
    box_dimensions=None,
    plot=False,
    save=False,
    figure_size=(6, 6),
    figname='rdf',
):
    """
    Computes 2D or 3D radial distribution function g(r) of a set of particle
    coordinates of shape (N, d). Particle must be placed in a 2D or 3D cuboidal
    box of dimensions [width x height (x depth)].

    Parameters
    ----------
    particles : (N, d) np.array
            Set of particle from which to compute the radial distribution function
            g(r). Must be of shape (N, 2) or (N, 3) for 2D and 3D coordinates
            repsectively.
    dr : float
            Delta r. Determines the spacing between successive radii over which g(r)
            is computed.
    variables : variables object
    rho : float, optional
            Number density. If left as None, box dimensions will be inferred from
            the particles and the number density will be calculated accordingly.
    rcutoff : float
            radii cutoff value between 0 and 1. The default value of 0.9 means the
            independent variable (radius) over which the RDF is computed will range
            from 0 to 0.9*r_max. This removes the noise that occurs at r values
            close to r_max, due to fewer valid particles available to compute the
            RDF from at these r values.
    eps : float, optional
            Epsilon value used to find particles less than or equal to a distance
            in KDTree.
    normalize : bool, optional
            Option to normalize the RDF. If True, the RDF values are normalized.
    reference_point : (d,) np.array or list, optional
            The center of the box. If left as None, there is no data cropping and calculate the rdf for the whole data.
    box_dimensions :  (d,) np.array or list, optional
            The dimensions of the box. If left as None, the box dimensions will be inferred from the particles.
    plot : bool, optional
            Option to plot the RDF. If True, the RDF is plotted.
    save : bool, optional
            Option to save the RDF. If True, the RDF is saved.
    figure_size : (float, float), optional
            The size of the figure in inches.
    figname : str, optional
            The name of the figure.

    Returns
    -------
    g_r : (n_radii) np.array
            radial distribution function values g(r).
    radii : (n_radii) np.array
            radii over which g(r) is computed
    """
    coords = np.asarray(particles, dtype=float)
    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        raise ValueError("particles must be a (N,2) or (N,3) array")
    if coords.shape[0] < 2:
        raise ValueError("At least two particles are required to compute RDF")
    if reference_point is not None and box_dimensions is not None:
        reference_point = np.asarray(reference_point, dtype=float)
        box_dimensions = np.asarray(box_dimensions, dtype=float)
        if box_dimensions.size == 2 and coords.shape[1] == 3:
            box_dimensions = np.concatenate((box_dimensions, [coords[:, 2].ptp() or dr]))
        if box_dimensions.shape[0] != coords.shape[1]:
            raise ValueError("box_dimensions must match the particle dimensionality")
        box_min = reference_point - 0.5 * box_dimensions
        box_max = reference_point + 0.5 * box_dimensions
        inside_box = np.all((coords >= box_min) & (coords <= box_max), axis=1)
        coords = coords[inside_box]
    if coords.shape[0] < 2:
        raise ValueError("Not enough particles remain after RDF cropping")

    print('The number of ions is: ', len(coords))
    mins = np.min(coords, axis=0)
    maxs = np.max(coords, axis=0)
    coords = coords - mins
    dims = np.max(coords, axis=0)
    if np.any(dims <= 0):
        raise ValueError("RDF box dimensions must be positive")

    r_max = (np.min(dims) / 2.0) * float(rcutoff)
    radii = np.arange(float(dr), r_max, float(dr))
    if radii.size == 0:
        raise ValueError("RDF radius grid is empty; increase the box size or reduce dr")

    n_particles, n_dim = coords.shape
    if rho is None:
        rho = n_particles / np.prod(dims)

    tree = cKDTree(coords)
    g_r = np.zeros(len(radii), dtype=float)

    for r_idx, r in enumerate(radii):
        margin = r + dr
        valid_mask = np.bitwise_and.reduce(
            [(coords[:, axis] >= margin) & (coords[:, axis] <= dims[axis] - margin) for axis in range(n_dim)]
        )
        valid_particles = coords[valid_mask]
        if len(valid_particles) == 0:
            continue

        counts = 0.0
        for particle in valid_particles:
            outer = tree.query_ball_point(particle, r + dr - eps, return_length=True)
            inner = tree.query_ball_point(particle, r, return_length=True)
            counts += outer - inner

        if normalize:
            if n_dim == 3:
                shell_measure = (4.0 / 3.0) * np.pi * ((r + dr) ** 3 - r**3)
            else:
                shell_measure = np.pi * ((r + dr) ** 2 - r**2)
            g_r[r_idx] = counts / (len(valid_particles) * shell_measure * rho)
        else:
            g_r[r_idx] = counts

    if plot or save:
        fig, ax = plt.subplots(figsize=figure_size)
        ax.plot(radii, g_r)
        ax.set_xlabel('Distance (nm)')
        ax.set_ylabel('g(r)' if normalize else 'Counts')
        ax.grid(alpha=0.3, linestyle='-.', linewidth=0.4)
        if save and variables is not None:

            save_matplotlib_figure(fig, variables, stem=f"rdf_{figname}", formats=("png", "pdf"), dpi=600)
        if plot:
            plt.show()

    return g_r, radii
