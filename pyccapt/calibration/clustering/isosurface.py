import numpy as np
import pyvista as pv


def bin_vectors_from_distance(dist, bin_values, mode='distance'):
    """
    Create a set of grid vectors to be used in nD binning. The bounds are calculated
    such that they don't go beyond the size of the dataset.

    Parameters:
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
            # Generate raw bin vector
            bin_vector_raw = np.linspace(0, 10000 * bin_values[dim], 10001)
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


import numpy as np


def pos_to_voxel(data, grid_vec, species=None):
    """
        Creates a voxelization of the data in 'pos' based on the bin centers in 'grid_vec'
        for the atoms/ions in the specified species.

        Parameters:
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
        pos_array = np.array([data["x (nm)"], data["y (nm)"], data["z (nm)"]]).T
        element_col = data.columns.get_loc("element") if "element" in data.columns else None
    else:
        pos_array = np.array(data)

    # Check for species filtering
    if species is not None:
        if isinstance(species, list) and (element_col):
            if element_col:
                species_mask = data['element'].isin(species)
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

    return vox


def isosurface(gridVec, data, isovalue):
    """
    Extract isosurface using pyvista for a custom 3D grid.

    Parameters:
        gridVec (list of np.ndarray): List of 3 arrays representing the grid points in x, y, and z.
        data (np.ndarray): 3D scalar field (same shape as the meshgrid defined by gridVec).
        isovalue (float): Scalar value to extract the isosurface.

    Returns:
        pyvista.PolyData: Isosurface with faces and vertices.
    """
    # Create a pyvista structured grid
    x, y, z = np.meshgrid(gridVec[0], gridVec[1], gridVec[2], indexing='ij')
    grid = pv.StructuredGrid(x, y, z)
    grid.point_data["values"] = data.flatten()

    # Extract the isosurface
    isosurf = grid.contour([isovalue])  # Pass isovalue as a list for compatibility
    return isosurf


if __name__ == "__main__":
    import pandas as pd

    # Test the function
    def generate_atom_dataset(num_atoms):
        """
        Generate a dataset with atom positions and random element assignment.

        Parameters:
            num_atoms (int): Number of atoms to generate (100 to 10M).

        Returns:
            pd.DataFrame: A DataFrame with columns `x(nm)`, `y(nm)`, `z(nm)`, `element`.
        """
        # Generate random positions in nanometers (x, y, z)
        positions = np.random.uniform(0, 100, (num_atoms, 3))  # Example range: [0, 100] nm

        # Assign elements randomly with 80% Al and 20% Fe
        elements = np.random.choice(["Al", "Fe"], size=num_atoms, p=[0.8, 0.2])

        # Create the DataFrame
        df = pd.DataFrame(positions, columns=["x (nm)", "y (nm)", "z (nm)"])
        df["element"] = elements

        return df

    # Example usage
    num_atoms = 1_000  # Set the desired dataset size
    data = generate_atom_dataset(num_atoms)

    # make pandas dataframe
    bin_values = [1, 1, 1]  # nm
    bin_centers, bin_edges = bin_vectors_from_distance(
        [data['x (nm)'].to_numpy(), data['y (nm)'].to_numpy(), data['z (nm)'].to_numpy()], bin_values, mode='distance'
    )

    grid_vec = np.array(bin_centers)
    vox = pos_to_voxel(data, grid_vec)
    voxIon = pos_to_voxel(data, grid_vec, species=['Fe'])
    conc = np.divide(voxIon, vox, out=np.zeros_like(voxIon, dtype=float), where=vox != 0)

    print("Concentration value range:", conc.min(), conc.max())
    iso_value = (conc.max() + conc.min()) / 2
    isosurf = isosurface(grid_vec, conc, isovalue=iso_value)

    import matplotlib.pyplot as plt

    # # Visualize using pyvista
    # plotter = pv.Plotter()
    # plotter.add_mesh(isosurf, color="blue", opacity=0.6)
    # plotter.show()

    import plotly.graph_objects as go

    # Extract Al atom positions
    al_positions = data[data["element"] == "Al"][["x (nm)", "y (nm)", "z (nm)"]].to_numpy()

    # Extract vertices and faces from the isosurface
    vertices = isosurf.points
    faces = isosurf.faces.reshape(-1, 4)[:, 1:]  # Faces have a leading count

    # Create the scatter plot for Al atoms
    scatter = go.Scatter3d(
        x=al_positions[:, 0],
        y=al_positions[:, 1],
        z=al_positions[:, 2],
        mode="markers",
        marker=dict(size=2, color="red", opacity=0.5),
        name="Al Atoms",
    )

    # Create the mesh for the isosurface
    mesh = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=0.6,
        alphahull=5,
        color="blue",
        name="Fe Isosurface",
    )

    # Combine and plot
    fig = go.Figure(data=[scatter, mesh])
    fig.update_layout(
        scene=dict(xaxis_title="X (nm)", yaxis_title="Y (nm)", zaxis_title="Z (nm)"),
        title="Scatter Plot of Al Atoms with Fe Isosurface",
    )

    fig.show()
