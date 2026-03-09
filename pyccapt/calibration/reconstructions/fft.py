from copy import copy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, rcParams

from pyccapt.calibration.reconstructions.io_utils import save_matplotlib_figure


def fft(particles, d, variables=None, normalize=False, reference_point=None,
        box_dimensions=None, plot=False, save=False, figure_size=(6, 6), figname='fft', fft_type='1d', axes=None):
    """
    Calculate the 1D, 2D, or 3D FFT of the particle positions.

    Parameters
    ----------
    particles : (N, 3) np.array
        Set of particle coordinates for which to compute the SDM.
    d : float

    variables : variables object
    normalize : bool, optional
        Option to normalize the fft. If True, the fft values are normalized.
    reference_point : (3,) np.array or list, optional
        The center of the box. If left as None, there is no data cropping, and calculate the fft for the whole data.
    box_dimensions : (3,) np.array or list, optional
        The dimensions of the box. If left as None, the box dimensions will be inferred from the particles.
    plot : bool, optional
        Option to plot the histograms. If True, the fft are plotted.
    save : bool, optional
        Option to save the histograms. If True, the fft are saved.
    figure_size : (float, float), optional
        The size of the figure in inches.
    figname : str, optional
        The name of the figure.
    fft_type : str, optional
        Type of histogram. Options are '1d' or '2d' or '3d'.
    axes : list or None, optional
        Specifies the axes for 1D or 2D histograms. For '1d', provide a list like ['x'], ['y'], or ['z'].
        For '2d', provide a list like ['x', 'y'], ['y', 'z'], or ['x', 'z'] or ['x', 'y', 'z'].

    Returns
    -------
    fft : list of np.array
        List of 1D or 2D histograms based on user preferences.
    	"""

    if reference_point is not None and box_dimensions is not None:
        if isinstance(reference_point, list):
            reference_point = np.array(reference_point)
        if isinstance(box_dimensions, list):
            box_dimensions = np.array(box_dimensions)
        # Ensure box_dimensions has at least 3 components
        assert len(box_dimensions) == 3, "box_dimensions must have 3 components (x, y, z)."

        # Calculate the bounds of the box
        box_min = reference_point - 0.5 * box_dimensions
        box_max = reference_point + 0.5 * box_dimensions

        # Crop particles within the specified box
        inside_box = np.all((particles >= box_min) & (particles <= box_max), axis=1)
        particles = particles[inside_box]

    fft_list = []
    edges_list = []
    if fft_type == '1d':
        if 'x' in axes:
            fft = np.fft.fft(particles[:, 0])
            fft = np.fft.fftshift(fft)
            fft = np.abs(fft)
            fft_list.append(fft)
        elif 'y' in axes:
            fft = np.fft.fft(particles[:, 1])
            fft = np.fft.fftshift(fft)
            fft = np.abs(fft)
            fft_list.append(fft)
        elif 'z' in axes:
            fft = np.fft.fft(particles[:, 2])
            fft = np.fft.fftshift(fft)
            fft = np.abs(fft)
            fft_list.append(fft)
        else:
            raise ValueError("Invalid axes for 1D histogram. Choose from ['x'], ['y'], or ['z'].")
        if normalize:
            fft_list[-1] = fft_list[-1] / np.max(fft_list[-1])
    if fft_type == '2d':
        if 'x' in axes and 'y' in axes:
            x, y = np.meshgrid(particles[:, 0], particles[:, 1])
            fft2d = np.fft.fft2(x)
            # Shift zero frequency components to the center
            fft2d = np.fft.fftshift(fft2d)
            amplitude = 20 * np.log(np.abs(fft2d))
            fft_list.append(amplitude)
            x_edges = np.fft.fftfreq(len(particles[:, 0]))
            y_edges = np.fft.fftfreq(len(particles[:, 1]))
            edges_list.append([x_edges, y_edges])
        elif 'y' in axes and 'z' in axes:
            y, z = np.meshgrid(particles[:, 1], particles[:, 2])
            fft2d = np.fft.fft2(y)
            # Shift zero frequency components to the center
            fft2d = np.fft.fftshift(fft2d)
            amplitude = 20 * np.log(np.abs(fft2d))
            fft_list.append(amplitude)
            y_edges = np.fft.fftfreq(len(particles[:, 1]))
            z_edges = np.fft.fftfreq(len(particles[:, 2]))
            edges_list.append([y_edges, z_edges])
        elif 'x' in axes and 'z' in axes:
            x, z = np.meshgrid(particles[:, 0], particles[:, 2])
            fft2d = np.fft.fft2(z)
            # Shift zero frequency components to the center
            fft2d = np.fft.fftshift(fft2d)
            amplitude = 20 * np.log(np.abs(fft2d))
            fft_list.append(amplitude)
            x_edges = np.fft.fftfreq(len(particles[:, 0]))
            z_edges = np.fft.fftfreq(len(particles[:, 2]))
            edges_list.append([x_edges, z_edges])
        else:
            raise ValueError("Invalid axes for 2D histogram. Choose from ['x', 'y'], ['y', 'z'], or ['x', 'z'].")

        if normalize:
            fft_list[-1] = fft_list[-1] / np.max(fft_list[-1])

    if fft_type == '3d':
        if 'x' in axes and 'y' in axes and 'z' in axes:
            pass
        if normalize:
            pass
#             fft_list[-1] = fft_list[-1] / np.max(fft_list[-1])

    if plot or save:
        # Plot histograms
        if fft_type == '1d':
            fig, ax = plt.subplots(figsize=figure_size)
            for i, fft_i in enumerate(fft_list):
                plt.plot(fft_i)
                plt.xlabel(f'{axes[i]} Frequency (Hz)')
                plt.ylabel('Amplitude')
        elif fft_type == '2d':
            fig, ax = plt.subplots(figsize=figure_size)
            plt.imshow(fft_list[-1], extent=[-1, 1, -1, 1], origin='lower', aspect="auto")
            cmap = copy(plt.cm.plasma)
            cmap.set_bad(cmap(0))
            x_edges = edges_list[-1][0]
            y_edges = edges_list[-1][1]
            pcm = ax.pcolormesh(x_edges, y_edges, fft_list[-1], cmap=cmap, norm=colors.LogNorm(), rasterized=True)
            cbar = fig.colorbar(pcm, ax=ax, pad=0)
            cbar.set_label('Counts', fontsize=10)
            plt.xlabel(f'{axes[0]} Frequency (Hz)')
            plt.ylabel(f'{axes[1]} Frequency (Hz)')
        elif fft_type == '3d':
            pass

        if save and variables is not None:
            # Enable rendering for text elements
            rcParams['svg.fonttype'] = 'none'
            save_matplotlib_figure(fig, variables, stem=f"fft_{figname}", formats=("png", "svg"), dpi=600)

        if plot:
            plt.show()

    return fft_list
