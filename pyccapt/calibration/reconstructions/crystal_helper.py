import numpy as np
from pymatgen.core import Structure, Lattice
import math
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def filter_atoms_in_cone_and_hemisphere(structure, cone_r, cone_L, hemi_z_base):
    """
    Filters atoms that are inside a cone and inside/below a hemisphere.

    Parameters:
    structure (Structure): Pymatgen Structure object containing atom positions.
    cone_r (float): Radius of the cone's base.
    cone_L (float): Height of the cone along the z-axis.
    hemi_z_base (float): Height of the hemisphere's flat face along the z-axis.

    Returns:
    Structure: Filtered Structure object containing only atoms inside the cone or inside/below the hemisphere.
    numpy.ndarray: Filtered array of atom coordinates.
    """
    # Extract x, y, z positions from structure
    coords = structure.cart_coords
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]


    # Determine the center of the cone base
    x_center = (x.min() + x.max()) / 2
    y_center = (y.min() + y.max()) / 2
    # Cone filtering condition
    # Calculate radial distance from the center of the cone base
    radial_distance = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
    # Linear radius at each height z within the cone's height
    radius_at_z = (cone_L - z) / cone_L * cone_r / 2
    # Condition to check if atoms are inside the cone
    inside_cone = (z >= 0) & (z <= cone_L) & (radial_distance <= radius_at_z)

    # Calculate the radius of the hemisphere based on the cone's geometry at hemi_z_base
    if 0 <= hemi_z_base <= cone_L:
        hemi_r = (cone_L - hemi_z_base) / cone_L * cone_r / 2
    else:
        hemi_r = 0  # No intersection with cone, setting hemi_r to zero or handling as needed

    # Hemisphere filtering condition
    # Center of the hemisphere dome
    hemi_z_center = hemi_z_base + hemi_r
    # Calculate distance from the hemisphere's dome center
    distance_to_hemi_center = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2 + (z - hemi_z_center) ** 2)
    # Condition to check if atoms are inside or below the hemisphere
    inside_or_below_hemisphere = (z <= hemi_z_center) | (distance_to_hemi_center <= hemi_r)

    # Combine conditions (atoms that are inside either the cone or the hemisphere)
    inside_cone_or_hemisphere = inside_cone & inside_or_below_hemisphere

    # Filter the structure based on the conditions
    site_list = []
    coords_f = []
    for i, site in enumerate(structure.sites):
        if inside_cone_or_hemisphere[i]:
            site_list.append(site)
            coords_f.append(coords[i])

    # Create a new structure with the filtered sites
    structure_f = Structure.from_sites(site_list)

    # Return the filtered structure and the filtered coordinates
    return structure_f, np.array(coords_f)


def find_phi_and_theta(sdm_param, theta_min, theta_max, phi_min, phi_max):
    """
    Plots noise level based on theta and phi values.

    Parameters:
    - sdm_param (np.ndarray): Array of parameters, where columns 2, 3, 4 represent theta, phi, and noise respectively.
    - theta_min (float): Minimum theta for plot limits.
    - theta_max (float): Maximum theta for plot limits.
    - phi_min (float): Minimum phi for plot limits.
    - phi_max (float): Maximum phi for plot limits.

    Returns:
    - theta_value (float): Theta value with minimum noise.
    - phi_value (float): Phi value with minimum noise.
    """
    # Extract theta, phi, and noise data
    theta = sdm_param[:, 1]
    phi = sdm_param[:, 2]
    noise = sdm_param[:, 3]

    # Create grid for theta and phi
    theta_grid = np.linspace(min(theta), max(theta), 100)
    phi_grid = np.linspace(min(phi), max(phi), 100)
    X, Y = np.meshgrid(theta_grid, phi_grid)

    # Interpolate noise data onto grid
    Z = griddata((theta, phi), noise, (X, Y), method='cubic')

    # Plotting
    fig, ax = plt.subplots()
    surf = ax.pcolormesh(X, Y, Z, shading='auto', cmap='turbo', norm=plt.LogNorm())
    cbar = plt.colorbar(surf, ax=ax, orientation='vertical')
    cbar.set_label('Noise level')

    # Set plot properties
    ax.set_xlabel('Theta (°)')
    ax.set_ylabel('Phi (°)')
    ax.set_title('Background Noise Level')
    ax.set_xlim(theta_min, theta_max)
    ax.set_ylim(phi_min, phi_max)
    ax.set_aspect('equal')

    # Find minimum noise value and corresponding theta, phi
    min_noise_idx = np.unravel_index(np.nanargmin(Z), Z.shape)
    theta_value = X[min_noise_idx]
    phi_value = Y[min_noise_idx]

    # Display theta and phi with minimum noise
    print(f'Minimum noise at Theta: {theta_value}°, Phi: {phi_value}°')

    plt.show()

    return theta_value, phi_value

def rotate_structure(structure, theta_deg, phi_deg):
    """
    Rotates the atomic coordinates of a pymatgen Structure based on given angles
    theta (angle from the z-axis) and phi (angle from the x-axis in the xy-plane).

    Parameters:
    - structure (Structure): Pymatgen Structure object containing atomic positions.
    - theta_deg (float): Angle theta in degrees (from the z-axis).
    - phi_deg (float): Angle phi in degrees (from the x-axis in the xy-plane).

    Returns:
    - Structure: New pymatgen Structure with rotated atomic coordinates.
    """
    # Convert angles from degrees to radians
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)

    # Initialize list to store rotated sites
    rotated_sites = []

    # Apply the rotation to each site in the structure
    for site in structure.sites:
        # Original coordinates
        x, y, z = site.coords

        # Calculate shifted coordinates based on theta and phi
        new_x = np.cos(theta) * x + np.sin(theta) * np.sin(phi) * y + np.sin(theta) * np.cos(phi) * z
        new_y = np.cos(phi) * y - np.sin(phi) * z
        new_z = -np.sin(theta) * x + np.cos(theta) * np.sin(phi) * y + np.cos(theta) * np.cos(phi) * z

        # Append the rotated coordinates and species
        rotated_sites.append((site.specie, [new_x, new_y, new_z]))

    # Create a new structure with the rotated coordinates
    rotated_structure = Structure(structure.lattice, [s[0] for s in rotated_sites],
                                  [s[1] for s in rotated_sites], coords_are_cartesian=True)

    return rotated_structure


# def rotate_structure(structure, x, y):
#     """
#     Rotate the structure based on the stereographic projection coordinates (x, y)
#     and return the rotated structure.
#
#     Parameters:
#     structure (pymatgen.core.Structure): The input structure to be rotated.
#     x (float): The x-coordinate from the stereographic projection.
#     y (float): The y-coordinate from the stereographic projection.
#
#     Returns:
#     pymatgen.core.Structure: The rotated structure.
#     """
#     # Step 1: Calculate theta and phi from x and y
#     if x > 1 or x < -1 or y > 1 or y < -1:
#         raise ValueError("x and y must be between -1 and 1.")
#     theta = -math.asin(x)
#     phi = math.acos(y / math.sin(theta)) if math.sin(theta) != 0 else 0
#     print(f"theta: {theta}, phi: {phi}")
#     # Step 2: Define rotation matrices for x and y axes
#     # Create rotation matrices for each rotation
#     rotation_x = Rotation.from_euler('x', theta).as_matrix()  # Rotate around x-axis by theta
#     rotation_y = Rotation.from_euler('y', phi).as_matrix()    # Rotate around y-axis by phi
#
#     # Combine rotations: first apply rotation around x, then around y
#     combined_rotation = rotation_y @ rotation_x  # Matrix multiplication
#
#     # Rotate each site in the structure
#     rotated_sites = []
#     for site in structure.sites:
#         rotated_coords = np.dot(site.coords, combined_rotation.T)  # Rotate the coordinates
#         rotated_sites.append((site.specie, rotated_coords))  # Store rotated site with species
#
#     # Create a new structure with the rotated sites
#     rotated_structure = Structure(structure.lattice, [s[0] for s in rotated_sites],
#                                   [s[1] for s in rotated_sites], coords_are_cartesian=True)
#
#     return rotated_structure

def apply_noise_to_structure(structure, noise_levels=(5, 5, 2), noise_type='correlative'):
    """
    Applies random displacements to the atomic positions in a structure.

    Parameters:
    structure (Structure): The pymatgen Structure object containing atom positions.
    noise_levels (tuple of floats): Noise levels for the x, y, and z coordinates in Ångstroms.
                                    For example, (0.5, 0.3, 0.1) will add up to 0.5 Å noise in x,
                                    0.3 Å noise in y, and 0.1 Å noise in z.
    noise_type (str): Type of noise to apply. Options are 'correlative' or 'noncorrelative'.

    Returns:
    tuple:
        - Structure: A new Structure object with noise applied to each atomic position.
        - np.ndarray: The noise array applied to each atom (for debugging or visualization purposes).
    """
    # Create a copy of the structure to avoid modifying the original structure
    noisy_structure = structure.copy()

    # Initialize the noise array
    all_noise = []

    # Generate noise based on the type
    if noise_type == 'correlative':
        # Correlated-axis noise: each atom receives a normalized random direction
        # scaled by the requested per-axis amplitudes.
        for site in noisy_structure.sites:
            correlated_noise = np.random.normal(0, 1, size=(3,))
            correlated_noise = correlated_noise / np.linalg.norm(correlated_noise)
            displacement = correlated_noise * np.array(noise_levels)
            site.coords += displacement
            all_noise.append(displacement)
    elif noise_type == 'noncorrelative':
        # Uncorrelated noise: Independent displacement for each atom and axis
        for site in noisy_structure.sites:
            displacement = np.random.uniform(-1, 1, size=(3,)) * noise_levels
            site.coords += displacement
            all_noise.append(displacement)
    else:
        raise ValueError("Invalid noise_type. Choose 'correlative' or 'noncorrelative'.")

    # Convert the noise list to a NumPy array
    all_noise = np.array(all_noise)

    return noisy_structure, all_noise


def project_to_surface(structure):
    """
    Projects all atomic positions onto the surface

    Parameters:
    structure (Structure): A pymatgen Structure object with atomic positions.

    Returns:
    Structure: A new Structure object with all atomic positions projected onto the hemisphere surface.
    """
    # Copy the structure to avoid modifying the original
    projected_structure = structure.copy()

    # Extract atomic coordinates
    coords = projected_structure.cart_coords

    # # Find the midpoints of x and y
    min_x, max_x = np.min(coords[:, 0]), np.max(coords[:, 0])
    min_y, max_y = np.min(coords[:, 1]), np.max(coords[:, 1])

    mid_x = (max_x - min_x) / 2
    mid_y = (max_y - min_y) / 2

    coords[:, 0] = coords[:, 0] - mid_x
    coords[:, 1] = coords[:, 1] - mid_y

    # Normalize coordinates to lie on the sphere
    # current_radii = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2 + coords[:, 2]**2)
    current_radii = np.linalg.norm(coords, axis=1)

    normalized_coords = coords / current_radii[:, None]

    # normalized_coords[:, 0] = normalized_coords[:, 0] + mid_x
    # normalized_coords[:, 1] = normalized_coords[:, 1] + mid_x

    # Create a new structure with the projected coordinates
    projected_structure = Structure(structure.lattice, structure.species, normalized_coords)

    return projected_structure

def stereographic_projection(structure, d_z=0):
    """
    Projects all atomic positions onto the surface

    Parameters:
    structure (Structure): A pymatgen Structure object with atomic positions.
    d_z (float): The distance of the projection plane from the origin.

    Returns:
    Structure: A new Structure object with all atomic positions projected onto the hemisphere surface.
    """
    # Copy the structure to avoid modifying the original
    projected_structure = structure.copy()

    # Extract atomic coordinates
    coords = projected_structure.cart_coords

    # # Find the midpoints of x and y
    min_x, max_x = np.min(coords[:, 0]), np.max(coords[:, 0])
    min_y, max_y = np.min(coords[:, 1]), np.max(coords[:, 1])

    mid_x = (max_x - min_x) / 2
    mid_y = (max_y - min_y) / 2

    coords[:, 0] = coords[:, 0] - mid_x
    coords[:, 1] = coords[:, 1] - mid_y

    coords[:, 0] = coords[:, 0] / (1 - coords[:, 2] + 1e-6)
    coords[:, 1] = coords[:, 1] / (1 - coords[:, 2] +  1e-6)
    coords[:, 2] = 0
    print(coords.shape)
    print(np.max(coords[:, 0]), np.min(coords[:, 0]))
    print(np.max(coords[:, 1]), np.min(coords[:, 1]))
    print(np.max(coords[:, 2]), np.min(coords[:, 2]))


    # Create a new structure with the projected coordinates
    projected_structure = Structure(structure.lattice, structure.species, coords)

    return projected_structure


def pyccapt_to_pymatgen(data, range):
    """
    Converts data from pyccapt format to pymatgen format.

    Args:
        data: the data in pyccapt format
        range: the range of data IN pyccapt format to be converted

    Returns:
        pymatgen Structure object
    """
    #a simple identity lattice (1x1x1 unit cell)
    identity_lattice = Lattice.eye(3)  # This creates an identity 3x3 lattice

    # Extract the species and coordinates from the data
    species = [data['mc_uc'][i] for i in range]
    coords = data[['x', 'y', 'z']].to_numpy()
    # Create the structure using the identity lattice
    structure = Structure(identity_lattice, species, coords)

    return structure

    return structure

