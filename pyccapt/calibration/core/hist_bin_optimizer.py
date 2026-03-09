import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal


# Based on : Shimazaki H. and Shinomoto S., A method for selecting the bin size of a time histogram Neural
# Computation (2007) Vol. 19(6), 1503-1527
def bin_width_optimizer_1d(data, plot=False):
    """
    Calculates the optimal bin width for a 1-dimensional histogram.

    Args:
        data (array-like): Input data for which the histogram is calculated.
        plot (bool, optional): If True, a histogram plot will be displayed. Defaults to False.

    Returns:
        tuple: A tuple containing the optimal bin number and bin width.

    """

    # Calculate the maximum and minimum values in the data
    data_max = np.max(data)
    data_min = np.min(data)

    # Define the range of bin numbers
    n_min = 2
    n_max = 200

    # Define the number of shifts
    n_shift = 30

    # Generate an array of bin numbers
    N = np.arange(n_min, n_max)

    # Calculate the bin width for each bin number
    D = (data_max - data_min) / N

    # Initialize the array to store the cost values
    Cs = np.zeros((len(D), n_shift))

    # Iterate over bin numbers and bin widths
    for i, d in enumerate(D):
        # Generate a shift array
        shift = np.linspace(0, d, n_shift)
        for j, s in enumerate(shift):
            # Calculate the bin edges
            edges = np.linspace(data_min + s - d / 2, data_max + s - d / 2, N[i] + 1)

            # Assign data points to bins
            binindex = np.digitize(data, edges)

            # Count the number of points in each bin
            ki = np.bincount(binindex)[1:N[i] + 1]

            # Calculate the mean and variance of the counts
            k = np.mean(ki)
            v = np.sum((ki - k) ** 2) / N[i]

            # Calculate the cost function
            Cs[i, j] += (2 * k - v) / (d ** 2)

    # Calculate the mean cost values
    C = np.mean(Cs, axis=1)

    # Find the index of the minimum cost value
    idx = np.argmin(C)

    # Get the optimal bin width
    optD = D[idx]

    print('Optimal Bin Number:', N[idx])
    print('Optimal Bin Width:', optD)

    if plot:
        edges = np.linspace(data_min + shift[j] - D[idx] / 2, data_max + shift[j] - D[idx] / 2, N[idx] + 1)
        fig, ax = plt.subplots()
        ax.hist(data, edges)
        ax.set_title("Histogram")
        ax.set_ylabel("Event Counts")
        ax.set_xlabel("Value")
        plt.show()

    return N[idx], optD


def bin_width_optimizer_2d(x, y, plot=False):
    """
    Calculates the optimal bin width for a 2-dimensional histogram.

    Args:
        x (array-like): Input data for the x-axis.
        y (array-like): Input data for the y-axis.
        plot (bool, optional): If True, a 2D histogram plot will be displayed. Defaults to False.

    Returns:
        tuple: A tuple containing the optimal bin number for x and y axes.

    """

    # Calculate the maximum and minimum values for x and y
    x_max = np.max(x)
    x_min = np.min(x)
    y_max = np.max(y)
    y_min = np.min(y)

    # Define the range of bin numbers for x and y axes
    Nx_MIN = 1
    Nx_MAX = 100
    Ny_MIN = 1
    Ny_MAX = 100

    # Generate arrays of bin numbers for x and y axes
    Nx = np.arange(Nx_MIN, Nx_MAX)
    Ny = np.arange(Ny_MIN, Ny_MAX)

    # Calculate the bin width for x and y axes
    Dx = (x_max - x_min) / Nx
    Dy = (y_max - y_min) / Ny

    # Create a structured array to store bin widths for x and y axes
    Dxy = np.zeros((len(Dx), len(Dy)), dtype=[('x', float), ('y', float)])

    # Iterate over bin widths for x and y axes
    for i, dx in enumerate(Dx):
        for j, dy in enumerate(Dy):
            Dxy[i, j] = (dx, dy)

    # Create an array to store the cost values
    Cxy = np.zeros_like(Dxy, dtype=float)

    # Iterate over bin widths for x and y axes
    for i, dx in enumerate(Dx):
        for j, dy in enumerate(Dy):
            # Calculate the 2D histogram
            ki, _, _ = np.histogram2d(x, y, bins=(Nx[i], Ny[j]))

            # Calculate the mean and variance of the counts
            k = np.mean(ki)
            v = np.var(ki)

            # Calculate the cost function
            Cxy[i, j] = (2 * k - v) / ((dx * dy) ** 2)

    # Find the indices of the minimum cost value
    idx_min_Cxy = np.unravel_index(np.argmin(Cxy), Cxy.shape)

    # Get the minimum cost value and optimal bin widths
    Cxymin = np.min(Cxy)
    optDx = Dxy[idx_min_Cxy]['x']
    optDy = Dxy[idx_min_Cxy]['y']

    print('Nx:', Nx[idx_min_Cxy[0]], 'optDx:', optDx)
    print('Ny:', Ny[idx_min_Cxy[1]], 'optDy:', optDy)

    if plot:
        fig, ax = plt.subplots()
        H, xedges, yedges = np.histogram2d(x, y, bins=[Nx[idx_min_Cxy[0]], Ny[idx_min_Cxy[1]]])
        Hmasked = np.ma.masked_where(H == 0, H)
        im = ax.imshow(Hmasked.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], interpolation='nearest',
                       origin='lower', aspect='auto', cmap=plt.cm.Spectral)
        ax.set_ylabel("y")
        ax.set_xlabel("x")
        plt.colorbar(im).set_label('z')
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = Dxy['x']
        y = Dxy['y']
        z = Cxy.flatten()
        ax.scatter(x, y, z, c=z, marker='o')
        ax.set_xlabel('Dx')
        ax.set_ylabel('Dy')
        ax.set_zlabel('Cxy')
        ax.scatter([optDx], [optDy], [Cxymin], marker='v', s=150, c="red")
        ax.text(optDx, optDy, Cxymin, "Cxy min", color='red')
        plt.show()

    return Nx[idx_min_Cxy[0]], Ny[idx_min_Cxy[1]]


if __name__ == "__main__":
    # Generate 1-dimensional data
    data = normal(0, 1, 100000)
    bin_width_optimizer_1d(data, plot=True)

    # Generate 2-dimensional data
    x = normal(0, 100, 10000)
    y = normal(0, 100, 10000)
    bin_width_optimizer_2d(x, y, plot=True)
