import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from scipy.spatial import cKDTree
from pyccapt.calibration.reconstructions.io_utils import save_matplotlib_figure

def rdf(particles, dr, variables=None, rho=None, rcutoff=0.9, eps=1e-15, normalize=True, reference_point=None,
        box_dimensions=None, plot=False, save=False, figure_size=(6, 6), figname='rdf'):
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
	if reference_point is not None and box_dimensions is not None:
		if isinstance(reference_point, list):
			reference_point = np.array(reference_point)
		if isinstance(box_dimensions, list):
			box_dimensions = np.array(box_dimensions)
		# Ensure box_dimensions has at least 2 components
		assert len(box_dimensions) >= 2, "box_dimensions must have at least 2 components (x, y)."

		# If box_dimensions has 2 components, assume it's a 2D box (x, y)
		if len(box_dimensions) == 2:
			box_dimensions = np.concatenate((box_dimensions, [0]))

		# Calculate the bounds of the box
		box_min = reference_point - 0.5 * box_dimensions
		box_max = reference_point + 0.5 * box_dimensions

		# Crop particles within the specified box
		inside_box = np.all((particles >= box_min) & (particles <= box_max), axis=1)
		particles = particles[inside_box]

	print('The number of ions is: ', len(particles))
	mins = np.min(particles, axis=0)
	maxs = np.max(particles, axis=0)
	# translate particles such that the particle with min coords is at origin
	particles = particles - mins

	# dimensions of box
	dims = maxs - mins

	r_max = (np.min(dims) / 2) * rcutoff
	radii = np.arange(dr, r_max, dr)

	N, d = particles.shape
	if not rho:
		rho = N / np.prod(dims)  # number density

	# create a KDTree for fast nearest-neighbor lookup of particles
	tree = cKDTree(particles)

	g_r = np.zeros(shape=(len(radii)))
	for r_idx, r in enumerate(radii):
		# find all particles that are at least r + dr away from the edges of the box
		valid_idxs = np.bitwise_and.reduce(
			[(particles[:, i] - (r + dr) >= mins[i]) & (particles[:, i] + (r + dr) <= maxs[i]) for i in range(d)])
		valid_particles = particles[valid_idxs]

		# compute n_i(r) for valid particles.
		for particle in valid_particles:
			n = (tree.query_ball_point(particle, r + dr - eps, return_length=True) -
			     tree.query_ball_point(particle, r, return_length=True))
			g_r[r_idx] += n

		# normalize
		if normalize:
			n_valid = len(valid_particles)
			shell_vol = (4 / 3) * np.pi * ((r + dr) ** 3 - r ** 3) if d == 3 else np.pi * ((r + dr) ** 2 - r ** 2)
			g_r[r_idx] /= n_valid * shell_vol * rho

	if plot or save:
		# Plot RDF
		fig, ax = plt.subplots(figsize=figure_size)
		plt.plot(radii, g_r)
		plt.xlabel('Distance (nm)')
		plt.ylabel('Counts')
		if save and variables is not None:
			# Enable rendering for text elements
			rcParams['svg.fonttype'] = 'none'
			save_matplotlib_figure(fig, variables, stem=f"projection_{figname}", formats=("png", "svg"), dpi=600)

		if plot:
			plt.show()

	return g_r, radii
