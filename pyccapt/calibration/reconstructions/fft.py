from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, rcParams

from pyccapt.calibration.reconstructions.io_utils import save_matplotlib_figure


def _crop_particles(particles, reference_point=None, box_dimensions=None):
    coords = np.asarray(particles, dtype=float)
    if reference_point is not None and box_dimensions is not None:
        reference_point = np.asarray(reference_point, dtype=float)
        box_dimensions = np.asarray(box_dimensions, dtype=float)
        if box_dimensions.shape[0] != coords.shape[1]:
            raise ValueError("box_dimensions must match the particle dimensionality")
        box_min = reference_point - 0.5 * box_dimensions
        box_max = reference_point + 0.5 * box_dimensions
        inside_box = np.all((coords >= box_min) & (coords <= box_max), axis=1)
        coords = coords[inside_box]
    if len(coords) < 2:
        raise ValueError("At least two particles are required for FFT analysis")
    return coords


def _axis_index(axis_name):
    return {'x': 0, 'y': 1, 'z': 2}[axis_name]


def fft(particles, d, variables=None, normalize=False, reference_point=None,
        box_dimensions=None, plot=False, save=False, figure_size=(6, 6), figname='fft', fft_type='1d', axes=None):
    """Compute histogram-based FFT spectra for 1D or 2D coordinate profiles."""
    coords = _crop_particles(particles, reference_point=reference_point, box_dimensions=box_dimensions)
    if not axes:
        raise ValueError("axes must be provided for FFT analysis")
    spacing = float(d)
    if spacing <= 0:
        raise ValueError("d must be greater than 0")

    fft_list = []
    metadata = []

    if fft_type == '1d':
        axis_name = axes[0]
        values = coords[:, _axis_index(axis_name)]
        bins = np.arange(values.min(), values.max() + spacing, spacing)
        if len(bins) < 3:
            raise ValueError("Not enough bins are available for 1D FFT")
        hist, _ = np.histogram(values, bins=bins)
        signal = hist.astype(float) - np.mean(hist)
        spectrum = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), d=spacing)
        if len(freqs) > 1:
            spectrum = spectrum[1:]
            freqs = freqs[1:]
        if normalize and np.max(spectrum) > 0:
            spectrum = spectrum / np.max(spectrum)
        fft_list.append(spectrum)
        metadata.append((freqs, axis_name))

    elif fft_type == '2d':
        axis_x, axis_y = axes[:2]
        values_x = coords[:, _axis_index(axis_x)]
        values_y = coords[:, _axis_index(axis_y)]
        bins_x = np.arange(values_x.min(), values_x.max() + spacing, spacing)
        bins_y = np.arange(values_y.min(), values_y.max() + spacing, spacing)
        if len(bins_x) < 3 or len(bins_y) < 3:
            raise ValueError("Not enough bins are available for 2D FFT")
        hist2d, x_edges, y_edges = np.histogram2d(values_x, values_y, bins=[bins_x, bins_y])
        signal = hist2d.astype(float) - np.mean(hist2d)
        spectrum = np.abs(np.fft.fftshift(np.fft.fft2(signal)))
        if normalize and np.max(spectrum) > 0:
            spectrum = spectrum / np.max(spectrum)
        freq_x = np.fft.fftshift(np.fft.fftfreq(hist2d.shape[0], d=spacing))
        freq_y = np.fft.fftshift(np.fft.fftfreq(hist2d.shape[1], d=spacing))
        fft_list.append(spectrum)
        metadata.append((freq_x, freq_y, axis_x, axis_y))
    else:
        raise ValueError("fft_type must be '1d' or '2d'")

    if plot or save:
        if fft_type == '1d':
            fig, ax = plt.subplots(figsize=figure_size)
            freqs, axis_name = metadata[0]
            ax.plot(freqs, fft_list[0])
            ax.set_xlabel(f'{axis_name} spatial frequency (1/nm)')
            ax.set_ylabel('Normalized amplitude' if normalize else 'Amplitude')
            ax.grid(alpha=0.3, linestyle='-.', linewidth=0.4)
        else:
            fig, ax = plt.subplots(figsize=figure_size)
            freq_x, freq_y, axis_x, axis_y = metadata[0]
            cmap = copy(plt.cm.plasma)
            cmap.set_bad(cmap(0))
            spectrum = fft_list[0]
            if normalize:
                pcm = ax.pcolormesh(freq_x, freq_y, spectrum.T, cmap=cmap, rasterized=True)
            else:
                pcm = ax.pcolormesh(
                    freq_x,
                    freq_y,
                    np.maximum(spectrum.T, np.finfo(float).tiny),
                    cmap=cmap,
                    norm=colors.LogNorm(),
                    rasterized=True,
                )
            cbar = fig.colorbar(pcm, ax=ax, pad=0)
            cbar.set_label('Amplitude', fontsize=10)
            ax.set_xlabel(f'{axis_x} spatial frequency (1/nm)')
            ax.set_ylabel(f'{axis_y} spatial frequency (1/nm)')

        if save and variables is not None:
            rcParams['svg.fonttype'] = 'none'
            save_matplotlib_figure(fig, variables, stem=f"fft_{figname}", formats=("png", "svg"), dpi=600)
        if plot:
            plt.show()

    return fft_list
