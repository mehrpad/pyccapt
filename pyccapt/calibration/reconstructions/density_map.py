from copy import copy
from matplotlib import cm
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams, colors
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


from pyccapt.calibration.data_tools.data_loadcrop import elliptical_shape_selector
from pyccapt.calibration.data_tools.merge_range import merge_by_range


def plot_density_map(x, y, z_weigth=False, log=True, bins=(256, 256), frac=1.0, axis_mode='normal', figure_size=(5, 4),
                     variables=None, composition=None, roi=None,
                     range_sequence=[], range_mc=[], range_detx=[], range_dety=[], range_x=[], range_y=[], range_z=[],
                     range_vol=[], data_crop=False, draw_circle=False, mode_selector='circle', axis=['x', 'y'],
                     save=False, figname='disparity_map', cmap='plasma',
                     normalize=False, normalize_axes=False):
    """
    Plot and crop the density map with the option to select a region of interest.

    Args:
        x: x-axis data
        y: y-axis data
        z_weigth: z weight data
        log: Flag to choose whether to plot the log of the data
        bins: Number of bins for the histogram in tuple or bin size in nm as float
        frac: Fraction of the data to be plotted
        axis_mode: Flag to choose whether to plot axis or scalebar: 'normal' or 'scalebar'
        variables: Variables object
        composition: list of elements
        roi: Region of interest
        range_sequence: Range of sequence as list or percentage
        range_mc: Range of mc
        range_detx: Range of detx
        range_dety: Range of dety
        range_x: Range of x-axis
        range_y: Range of y-axis
        range_z: Range of z-axis
        range_vol: Range of voltage
        figure_size: Size of the plot
        draw_circle: Flag to enable circular region of interest selection
        mode_selector: Mode of selection (circle or ellipse)
        axis: Axes to be plotted
        save: Flag to choose whether to save the plot or not
        data_crop: Flag to control whether only the plot is shown or cropping functionality is enabled
        figname: Name of the figure
        cmap: Colormap for the plot
        normalize: Flag to normalize the histogram (default False).
        normalize_axes: Flag to normalize the axis limits (default False).

    Returns:
        None
    """
    if range_sequence or range_mc or range_detx or range_dety or range_x or range_y or range_z:
        if range_sequence:
            if range_sequence is list:
                mask_sequence = np.zeros_like(len(x), dtype=bool)
                if range_sequence[0] < 1 and range_sequence[1] < 1:
                    mask_sequence[int(len(x)*range_sequence[0]):int(len(x)*range_sequence[1])]=True
                else:
                    mask_sequence[range_sequence[0]:range_sequence[1]] = True
                mask_sequence[range_sequence[0]:range_sequence[1]] = True
            else:
                mask_sequence = np.zeros(len(x), dtype=bool)
                mask_sequence[:int(len(x)*range_sequence)] = True

        else:
            mask_sequence = np.ones(len(x), dtype=bool)
        if range_detx and range_dety:
            mask_det_x = (variables.dld_x_det < range_detx[1]) & (variables.dld_x_det > range_detx[0])
            mask_det_y = (variables.dld_y_det < range_dety[1]) & (variables.dld_y_det > range_dety[0])
            mask_det = mask_det_x & mask_det_y
        else:
            mask_det = np.ones(len(x), dtype=bool)
        if range_mc:
            mask_mc = (variables.mc <= range_mc[1]) & (variables.mc >= range_mc[0])
        else:
            mask_mc = np.ones(len(x), dtype=bool)
        if range_x and range_y and range_z:
            mask_x = (variables.x < range_x[1]) & (variables.x > range_x[0])
            mask_y = (variables.y < range_y[1]) & (variables.y > range_y[0])
            mask_z = (variables.z < range_z[1]) & (variables.z > range_z[0])
            mask_3d = mask_x & mask_y & mask_z
        else:
            mask_3d = np.ones(len(x), dtype=bool)
        if range_vol:
            mask_vol = (variables.dld_high_voltage < range_vol[1]) & (variables.dld_high_voltage > range_vol[0])
        else:
            mask_vol = np.ones(len(x), dtype=bool)
        mask = mask_sequence & mask_det & mask_mc & mask_3d & mask_vol
        if variables is not None:
            variables.mask = mask
        print('The number of data sequence:', len(mask_sequence[mask_sequence == True]))
        print('The number of data mc:', len(mask_mc[mask_mc == True]))
        print('The number of data det:', len(mask_det[mask_det == True]))
        print('The number of data 3d:', len(mask_3d[mask_3d == True]))
        print('The number of data after cropping:', len(mask[mask == True]))
    else:
        mask = np.ones(len(x), dtype=bool)

    if variables is not None:
        if composition and isinstance(composition, list):
            if 'element' in variables.data.columns:
                data = variables.data
            else:
                if variables.range_data is None:
                    raise ValueError('Range data is not provided')
                data = merge_by_range(variables.data, variables.range_data, full=True)
            mask_comp = np.zeros(len(x), dtype=bool)
            # Create a mask from the composition list of variables.data
            for comp in composition:
                mask_comp = mask_comp | data['element'].apply(lambda x: comp in x)
            if not mask_comp.any():
                print('No composition found - using all data')
                mask_comp = np.ones(len(x), dtype=bool)
        else:
            mask_comp = np.ones(len(x), dtype=bool)
    else:
        mask_comp = np.ones(len(x), dtype=bool)

    mask = mask & mask_comp

    fig1, ax1 = plt.subplots(figsize=figure_size, constrained_layout=True)

    if frac < 1:
        # set axis limits based on fraction of x and y data based on fraction
        true_indices = np.where(mask)[0]
        num_set_to_flase = int(len(true_indices) * (1 - frac))
        indices_to_set_false = np.random.choice(true_indices, num_set_to_flase, replace=False)
        mask[indices_to_set_false] = False
        x_t = np.copy(x)
        y_t = np.copy(y)

    x = x[mask]
    y = y[mask]
    # Check if the bin is a list
    if isinstance(bins, list):
        print('bins:', bins)
        if len(bins) == 1:
            x_edges = np.arange(x.min(), x.max() + bins, bins)
            y_edges = np.arange(y.min(), y.max() + bins, bins)
            bins = [x_edges, y_edges]
    else:
        raise ValueError("Bins should be a tuple")


    if z_weigth is False:
        FDM, xedges, yedges = np.histogram2d(x, y, bins=bins)
    else:
        if variables is None:
            raise ValueError("Variables object is required for z weighting")
        FDM, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=variables.z[mask])

    # Ensure that yedges are reversed for correct z direction
    yedges = yedges[::-1]

    # Normalize the histogram if requested
    if normalize:
        FDM = FDM / np.max(FDM)

    # Normalize the axes if requested
    if normalize_axes:
        xedges = (xedges - np.min(xedges)) / (np.max(xedges) - np.min(xedges))
        yedges = (yedges - np.min(yedges)) / (np.max(yedges) - np.min(yedges))

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    cmap_instance = copy(cm.get_cmap(cmap))
    cmap_instance.set_bad(cmap_instance(0))

    if log and not normalize:
        FDM = np.log1p(FDM)
        pcm = ax1.pcolormesh(xedges, yedges, FDM.T, cmap=cmap_instance, norm=colors.LogNorm(), rasterized=True)
    else:
        pcm = ax1.pcolormesh(xedges, yedges, FDM.T, cmap=cmap_instance, rasterized=True)

    cbar = fig1.colorbar(pcm, ax=ax1, pad=0)
    cbar.set_label('Event Counts', fontsize=10)

    if frac < 1:
        # extract tof
        x_lim = x_t
        y_lim = y_t
        ax1.set_xlim([min(x_lim), max(x_lim)])
        ax1.set_ylim([min(y_lim), max(y_lim)])

    if variables is not None:
        if data_crop:
            elliptical_shape_selector(ax1, fig1, variables, mode=mode_selector)
        if draw_circle:
            print('x:', variables.selected_x_fdm, 'y:', variables.selected_y_fdm, 'roi:', variables.roi_fdm)
            circ = Circle((variables.selected_x_fdm, variables.selected_y_fdm), variables.roi_fdm, fill=True,
                          alpha=0.3, color='green', linewidth=5)
            ax1.add_patch(circ)
    if axis_mode == 'scalebar':
        fontprops = fm.FontProperties(size=10)
        scalebar = AnchoredSizeBar(ax1.transData,
                                   1, '1 cm', 'lower left',
                                   pad=0.1,
                                   color='white',
                                   frameon=False,
                                   size_vertical=0.1,
                                   fontproperties=fontprops)

        ax1.add_artist(scalebar)
        plt.axis('off')  # Turn off both x and y axes
    elif axis_mode == 'normal':
        if 'x' in axis and 'y' in axis:
            ax1.set_xlabel(r"$x (nm)$", fontsize=10)
            ax1.set_ylabel(r"$y (nm)$", fontsize=10)
        elif 'y' in axis and 'z' in axis:
            ax1.set_xlabel(r"$y (nm)$", fontsize=10)
            ax1.set_ylabel(r"$z (nm)$", fontsize=10)
        elif 'x' in axis and 'z' in axis:
            ax1.set_xlabel(r"$x (nm)$", fontsize=10)
            ax1.set_ylabel(r"$z (nm)$", fontsize=10)

    if roi and roi!=[0,0,0]:
        if axis == ['x', 'y'] or axis == ['y', 'x']:
            # plot a circle at position roi[0], roi[1] with radius roi[2]
            circ = Circle((roi[0], roi[1]), roi[2], fill=False, color='white', linewidth=1)
            ax1.add_patch(circ)
        else:
            print('ROI is only supported for x-y axis')
    plt.show()
    if save and variables is not None:
        # Enable rendering for text elements
        rcParams['svg.fonttype'] = 'none'
        plt.savefig("%s.png" % (variables.result_path + figname), format="png", dpi=600)
        plt.savefig("%s.svg" % (variables.result_path + figname), format="svg", dpi=600)
