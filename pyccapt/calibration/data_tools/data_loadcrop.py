from copy import copy

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams, colors
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import RectangleSelector, EllipseSelector
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from numba.cpython.slicing import make_slice_from_constant

from pyccapt.calibration.data_tools import data_tools, selectors_data


def fetch_dataset_from_dld_grp(filename: str, extract_mode='dld') -> pd.DataFrame:
    """
    Fetches dataset from HDF5 file.

    Args:
        filename: Path to the HDF5 file.
        extract_mode: Mode of extraction.
                    dld: Extracts data from dld group.
                    tdc_sc: Extracts data from tdc for Surface Consept.
                    tdc_ro: Extracts data from tdc for Roentdek detector.

    Returns:
        DataFrame: Contains relevant information from the dld group.
    """
    if extract_mode == 'dld':
        try:
            hdf5Data = data_tools.read_hdf5(filename)
            if hdf5Data is None:
                raise FileNotFoundError
            dld_highVoltage = hdf5Data['dld/high_voltage'].to_numpy()
            if 'dld/pulse' in hdf5Data:
                dld_pulse_v = hdf5Data['dld/pulse'].to_numpy()
            elif 'dld/voltage_pulse' in hdf5Data:
                dld_pulse_v = hdf5Data['dld/voltage_pulse'].to_numpy()
            elif 'dld/pulse_voltage' in hdf5Data:
                dld_pulse_v = hdf5Data['dld/pulse_voltage'].to_numpy()
            else:
                dld_pulse_v = np.zeros(len(dld_highVoltage))
                dld_pulse_v = np.expand_dims(dld_pulse_v, axis=1)
            if 'dld/laser_intensity' in hdf5Data:
                dld_pulse_l = hdf5Data['dld/laser_intensity'].to_numpy()
            else:
                dld_pulse_l = np.zeros(len(dld_highVoltage))
                dld_pulse_l = np.expand_dims(dld_pulse_l, axis=1)
            if 'dld/start_counter' in hdf5Data:
                dld_startCounter = hdf5Data['dld/start_counter'].to_numpy()
            else:
                dld_startCounter = np.zeros(len(dld_highVoltage))
                dld_startCounter = np.expand_dims(dld_startCounter, axis=1)
            dld_t = hdf5Data['dld/t'].to_numpy()
            dld_x = hdf5Data['dld/x'].to_numpy()
            dld_y = hdf5Data['dld/y'].to_numpy()
            dldGroupStorage = np.concatenate(
                (dld_highVoltage, dld_pulse_v, dld_pulse_l, dld_startCounter, dld_t, dld_x, dld_y),
                                             axis=1)
            dld_group_storage = create_pandas_dataframe(dldGroupStorage, mode='dld')
            return dld_group_storage
        except KeyError as error:
            print(error)
            print("[*] Keys missing in the dataset")
        except FileNotFoundError as error:
            print(error)
            print("[*] HDF5 file not found")
    elif extract_mode == 'tdc_sc':
        try:
            hdf5Data = data_tools.read_hdf5(filename)
            if hdf5Data is None:
                raise FileNotFoundError
            channel = hdf5Data['tdc/channel'].to_numpy()
            start_counter = hdf5Data['tdc/start_counter'].to_numpy()
            high_voltage = hdf5Data['tdc/high_voltage'].to_numpy()
            if 'tdc/pulse' in hdf5Data:
                voltage_pulse = hdf5Data['tdc/pulse'].to_numpy()
            elif 'tdc/voltage_pulse' in hdf5Data:
                voltage_pulse = hdf5Data['tdc/voltage_pulse'].to_numpy()
            else:
                raise KeyError('Neither tdc/pulse nor tdc/voltage_pulse exists in the dataset')
            if 'tdc/laser_pulse' in hdf5Data:
                laser_pulse = hdf5Data['tdc/laser_pulse'].to_numpy()
            else:
                laser_pulse = np.zeros(len(channel))
            time_data = hdf5Data['tdc/time_data'].to_numpy()

            dldGroupStorage = np.concatenate((channel, start_counter, high_voltage, voltage_pulse,
                                              time_data), axis=1)
            dld_group_storage = create_pandas_dataframe(dldGroupStorage, mode='tdc_sc')
            return dld_group_storage
        except KeyError as error:
            print(error)
            print("[*] Keys missing in the dataset")
        except FileNotFoundError as error:
            print(error)
            print("[*] HDF5 file not found")
    elif extract_mode == 'tdc_ro':
        print('Not implemented yet')

def concatenate_dataframes_of_dld_grp(dataframeList: list) -> pd.DataFrame:
    """
    Concatenates dataframes into a single dataframe.

    Args:
        dataframeList: List of different information from dld group.

    Returns:
        DataFrame: Single concatenated dataframe containing all relevant information.
    """
    dld_masterDataframe = pd.concat(dataframeList, axis=1)
    return dld_masterDataframe


def plot_crop_experiment_history(data: pd.DataFrame, variables, max_tof, frac=1.0, bins=(1200, 800), figure_size=(8, 3),
                                 draw_rect=False, data_crop=True, pulse_plot=False, dc_plot=True, pulse_mode='voltage',
                                 save=True, figname=''):
    """
    Plots the experiment history.

    Args:
        dldGroupStorage: DataFrame containing info about the dld group.
        max_tof: The maximum tof to be plotted.
        frac: Fraction of the data to be plotted.
        figure_size: The size of the figure.
        data_crop: Flag to control if only the plot should be shown or cropping functionality should be enabled.
        draw_rect: Flag to draw  a rectangle over the selected area.
        pulse: Flag to choose whether to plot pulse.
        pulse_mode: Flag to choose whether to plot pulse voltage or pulse.
        dc_plot: Flag to choose whether to plot dc voltage.
        save: Flag to choose whether to save the plot or not.
        figname: Name of the figure to be saved.

    Returns:
        None.
    """

    if max_tof > 0:
        mask_1 = (data['t (ns)'].to_numpy() > max_tof)
        data.drop(np.where(mask_1)[0], inplace=True)
        data.reset_index(inplace=True, drop=True)
    if frac < 1:
        # set axis limits based on fraction of data
        dldGroupStorage = data.sample(frac=frac, random_state=42)
        dldGroupStorage.sort_index(inplace=True)
    else:
        dldGroupStorage = data

    fig1, ax1 = plt.subplots(figsize=figure_size, constrained_layout=True)

    # extract tof and high voltage from the data frame
    tof = dldGroupStorage['t (ns)'].to_numpy()
    high_voltage = data['high_voltage (V)'].to_numpy()
    high_voltage = high_voltage / 1000  # change to kV
    if pulse_mode == 'laser':
        pulse = dldGroupStorage['pulse_l (pJ)'].to_numpy()
    elif pulse_mode == 'voltage':
        pulse = dldGroupStorage['pulse_v (V)'].to_numpy()

    xaxis = np.arange(len(tof))

    # Check if the bin is a tuple
    if isinstance(bins, tuple):
        pass
    else:
        x_edges = np.arange(xaxis.min(), xaxis.max() + bins, bins)
        y_edges = np.arange(tof.min(), tof.max() + bins, bins)
        bins = [x_edges, y_edges]

    heatmap, xedges, yedges = np.histogram2d(xaxis, tof, bins=bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # Set x-axis label
    ax1.set_xlabel("Hit Sequence Number", fontsize=10)
    # Set y-axis label
    ax1.set_ylabel("Time of Flight [ns]", fontsize=10)
    img = plt.imshow(heatmap.T, extent=extent, origin='lower', aspect="auto")
    cmap = copy(plt.cm.plasma)
    cmap.set_bad(cmap(0))
    pcm = ax1.pcolormesh(xedges, yedges, heatmap.T, cmap=cmap, norm=colors.LogNorm(), rasterized=True)

    plt.ticklabel_format(style='sci', axis='x', scilimits=(6, 6))

    if dc_plot:
        ax2 = ax1.twinx()
        if not pulse_plot:
            ax2.spines.right.set_position(("axes", 1.13))
        else:
            ax2.spines.right.set_position(("axes", 1.29))
        # Plot high voltage curve
        xaxis2 = np.arange(len(high_voltage))
        dc_curve, = ax2.plot(xaxis2, high_voltage, color='red', linewidth=2)
        ax2.set_ylabel("DC Voltage [kV]", color="red", fontsize=10)
        ax2.set_ylim([min(high_voltage), max(high_voltage) + 0.5])
        ax2.spines['right'].set_color('red')  # Set Y-axis color to red
        ax2.yaxis.label.set_color('red')  # Set Y-axis label color to red
        ax2.tick_params(axis='y', colors='red')  # Set Y-axis tick labels color to red

    if pulse_plot:
        ax3 = ax1.twinx()
        ax3.spines.right.set_position(("axes", 1.13))
        if pulse_mode == 'laser':
            pulse_curve, = ax3.plot(xaxis, pulse, color='fuchsia', linewidth=2)
            ax3.set_ylabel("Pulse Energy [$pJ$]", color="fuchsia", fontsize=10)
            range = max(pulse) - min(pulse)
            ax3.set_ylim([min(pulse) - range * 0.1, max(pulse) + range * 0.1])
            ax3.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        elif pulse_mode == 'voltage':
            pulse = pulse / 1000
            pulse_curve, = ax3.plot(xaxis, pulse, color='fuchsia', linewidth=2)
            ax3.set_ylabel("Pulse Voltage [kV]", color="fuchsia", fontsize=10)
            ax3.set_ylim([min(pulse), max(pulse) + 0.5])
        ax3.spines['right'].set_color('fuchsia')  # Set Y-axis color to red
        ax3.yaxis.label.set_color('fuchsia')  # Set Y-axis label color to red
        ax3.tick_params(axis='y', colors='fuchsia')  # Set Y-axis tick labels color to red

    # if pulse_plot:
    #     pulse_curve.set_visible(False)
    # if dc_plot:
    #     dc_curve.set_visible(False)

    if dc_plot or pulse_plot:
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="2.5%", pad="0%")
        cbar = fig1.colorbar(pcm, cax=cax)
    else:
        cbar = fig1.colorbar(pcm, ax=ax1, pad=0)

    cbar.set_label('Event Counts', fontsize=10)

    if data_crop:
        if dc_plot and pulse_plot:
            rectangle_box_selector(ax3, variables)
        elif dc_plot and not pulse_plot:
            rectangle_box_selector(ax2, variables)
        elif not dc_plot and pulse_plot:
            rectangle_box_selector(ax3, variables)
        elif not pulse_plot and not dc_plot:
            rectangle_box_selector(ax1, variables)
        plt.connect('key_press_event', selectors_data.toggle_selector(variables))
    if draw_rect:
        left, bottom, width, height = (
            variables.selected_x1, 0, variables.selected_x2 - variables.selected_x1, np.max(tof))
        rect = Rectangle((left, bottom), width, height, fill=True, alpha=0.3, color="r", linewidth=5)
        ax1.add_patch(rect)

    if save:
        # Enable rendering for text elements
        rcParams['svg.fonttype'] = 'none'
        plt.savefig("%s.png" % (variables.result_path + figname), format="png", dpi=600, bbox_inches='tight')
        plt.savefig("%s.svg" % (variables.result_path + figname), format="svg", dpi=600)

    plt.show()


def plot_crop_fdm(x, y, bins=(256, 256), frac=1.0, axis_mode='normal', figure_size=(5, 4), variables=None,
                  range_sequence=[], range_mc=[], range_detx=[], range_dety=[], range_x=[], range_y=[], range_z=[],
                  range_vol=[], data_crop=False, draw_circle=False, mode_selector='circle', save=False, figname='FDM'):
    """
    Plot and crop the FDM with the option to select a region of interest.

    Args:
        x: x-axis data
        y: y-axis data
        bins: Number of bins for the histogram as a tuple or a single float as the bin size
        frac: Fraction of the data to be plotted
        axis_mode: Flag to choose whether to plot axis or scalebar: 'normal' or 'scalebar'
        variables: Variables object
        range_sequence: Range of sequence
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
        save: Flag to choose whether to save the plot or not
        data_crop: Flag to control whether only the plot is shown or cropping functionality is enabled
        figname: Name of the figure to be saved

    Returns:
        None
    """
    if range_sequence or range_mc or range_detx or range_dety or range_x or range_y or range_z:
        if range_sequence:
            mask_sequence = np.zeros_like(len(x), dtype=bool)
            mask_sequence[range_sequence[0]:range_sequence[1]] = True
        else:
            mask_sequence = np.ones(len(x), dtype=bool)
        if range_detx and range_dety:
            mask_det_x = (variables.dld_x_det < range_detx[1]) & (variables.dld_x_det > range_detx[0])
            mask_det_y = (variables.dld_y_det < range_dety[1]) & (variables.dld_y_det > range_dety[0])
            mask_det = mask_det_x & mask_det_y
        else:
            mask_det = np.ones(len(variables.dld_x_det), dtype=bool)
        if range_mc:
            mask_mc = (variables.mc <= range_mc[1]) & (variables.mc >= range_mc[0])
        else:
            mask_mc = np.ones(len(variables.mc), dtype=bool)
        if range_x and range_y and range_z:
            mask_x = (variables.x < range_x[1]) & (variables.x > range_x[0])
            mask_y = (variables.y < range_y[1]) & (variables.y > range_y[0])
            mask_z = (variables.z < range_z[1]) & (variables.z > range_z[0])
            mask_3d = mask_x & mask_y & mask_z
        else:
            mask_3d = np.ones(len(variables.x), dtype=bool)
        if range_vol:
            mask_vol = (variables.dld_high_voltage < range_vol[1]) & (variables.dld_high_voltage > range_vol[0])
        else:
            mask_vol = np.ones(len(variables.dld_high_voltage), dtype=bool)
        mask = mask_sequence & mask_det & mask_mc & mask_3d & mask_vol
        variables.mask = mask
        print('The number of data mc:', len(mask_mc[mask_mc == True]))
        print('The number of data det:', len(mask_det[mask_det == True]))
        print('The number of data 3d:', len(mask_3d[mask_3d == True]))
        print('The number of data after cropping:', len(mask[mask == True]))
    else:
        mask = np.ones(len(x), dtype=bool)

    x = x[mask]
    y = y[mask]

    if frac < 1:
        # set axis limits based on fraction of x and y data baded on fraction
        mask_fraq = np.random.choice(len(x), int(len(x) * frac), replace=False)
        x_t = np.copy(x)
        y_t = np.copy(y)
        x = x[mask_fraq]
        y = y[mask_fraq]


    fig1, ax1 = plt.subplots(figsize=figure_size, constrained_layout=True)


    # Check if the bin is a list
    if isinstance(bins, list):
        if len(bins) == 1:
            x_edges = np.arange(x.min(), x.max() + bins, bins)
            y_edges = np.arange(y.min(), y.max() + bins, bins)
            bins = [x_edges, y_edges]

    FDM, xedges, yedges = np.histogram2d(x, y, bins=bins)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    cmap = copy(plt.cm.plasma)
    cmap.set_bad(cmap(0))
    pcm = ax1.pcolormesh(xedges, yedges, FDM.T, cmap=cmap, norm=colors.LogNorm(), rasterized=True)
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
        ax1.set_xlabel(r"$X_{det} (cm)$", fontsize=10)
        ax1.set_ylabel(r"$Y_{det} (cm)$", fontsize=10)

    if save and variables is not None:
        # Enable rendering for text elements
        rcParams['svg.fonttype'] = 'none'
        plt.savefig("%s.png" % (variables.result_path + figname), format="png", dpi=600)
        plt.savefig("%s.svg" % (variables.result_path + figname), format="svg", dpi=600)
    plt.show()


def rectangle_box_selector(axisObject, variables):
    """
    Enable the creation of a rectangular box to select the region of interest.

    Args:
        axisObject: Object to create the rectangular box
        variables: Variables object

    Returns:
        None
    """
    selectors_data.toggle_selector.RS = RectangleSelector(axisObject,
                                                          lambda eclick, erelease: selectors_data.line_select_callback(
                                                              eclick, erelease, variables),
                                                          useblit=True,
                                                          button=[1, 3],
                                                          minspanx=1, minspany=1,
                                                          spancoords='pixels',
                                                          interactive=True)


def crop_dataset(dld_master_dataframe, variables):
    """
    Crop the dataset based on the selected region of interest.

    Args:
        dld_master_dataframe: Concatenated dataset
        variables: Variables object

    Returns:
        data_crop: Cropped dataset
    """
    data_crop = dld_master_dataframe.loc[int(variables.selected_x1):int(variables.selected_x2), :]
    data_crop.reset_index(inplace=True, drop=True)
    return data_crop


def elliptical_shape_selector(axisObject, figureObject, variables, mode='circle'):
    """
    Enable the creation of an elliptical box to select the region of interest.

    Args:
        axisObject: Object to create the axis of the plot
        figureObject: Object to create the figure
        variables: Variables object
        mode: Mode of selection (circle or ellipse)

    Returns:
        None
    """
    if mode == 'circle':
        selectors_data.toggle_selector.ES = selectors_data.CircleSelector(axisObject,
                                                                          lambda eclick,
                                                                                 erelease: selectors_data.onselect(
                                                                              eclick,
                                                                              erelease,
                                                                              variables),
                                                                          useblit=True,
                                                                          button=[1, 3],
                                                                          minspanx=1, minspany=1,
                                                                          spancoords='pixels',
                                                                          interactive=True)
    elif mode == 'ellipse':
        selectors_data.toggle_selector.ES = EllipseSelector(axisObject,
                                                            lambda eclick, erelease: selectors_data.onselect(eclick,
                                                                                                             erelease,
                                                                                                             variables),
                                                            useblit=True,
                                                            button=[1, 3],
                                                            minspanx=1, minspany=1,
                                                            spancoords='pixels',
                                                            interactive=True)

    figureObject.canvas.mpl_connect('key_press_event', selectors_data.toggle_selector)


def crop_data_after_selection(data_crop, variables):
    """
    Crop the dataset after the region of interest has been selected.

    Args:
        data_crop: Original dataset to be cropped
        variables: Variables object

    Returns:
        data_crop: Cropped dataset
    """
    x = data_crop['x_det (cm)'].to_numpy()
    y = data_crop['y_det (cm)'].to_numpy()
    detector_dist = np.sqrt((x - variables.selected_x_fdm) ** 2 + (y - variables.selected_y_fdm) ** 2)
    mask_fdm = (detector_dist > variables.roi_fdm)
    data_crop.drop(np.where(mask_fdm)[0], inplace=True)
    data_crop.reset_index(inplace=True, drop=True)
    return data_crop


def create_pandas_dataframe(data_crop, mode='dld'):
    """
    Create a pandas dataframe from the cropped data.

    Args:
        data_crop: Cropped dataset
        mode: Mode of extraction
                dld: Extracts data from dld group
                tdc_sc: Extracts data from tdc for Surface Consept
                tdc_ro: Extracts data from tdc for Roentdek detector

    Returns:
        hdf_dataframe: Dataframe to be inserted in the HDF file
    """
    if mode == 'dld':
        hdf_dataframe = pd.DataFrame(data=data_crop,
                                     columns=['high_voltage (V)', 'pulse_v (V)',
                                              'pulse_l (pJ)', 'start_counter', 't (ns)',
                                              'x_det (cm)', 'y_det (cm)'])

        hdf_dataframe['start_counter'] = hdf_dataframe['start_counter'].astype('uint32')
    elif mode == 'tdc_sc':
        hdf_dataframe = pd.DataFrame(data=data_crop,
                                     columns=['channel', 'start_counter', 'high_voltage (V)', 'pulse_v (V)',
                                              'pulse_l (pJ)', 'time_data'])

        hdf_dataframe['channel'] = hdf_dataframe['channel'].astype('uint32')
        hdf_dataframe['start_counter'] = hdf_dataframe['start_counter'].astype('uint32')
        hdf_dataframe['time_data'] = hdf_dataframe['time_data'].astype('uint32')
    elif mode == 'tdc_ro':
        print('Not implemented yet')
        hdf_dataframe = None

    return hdf_dataframe


def calculate_ppi_and_ipp(data, max_start_counter):
    """
    Calculate pulses since the last event pulse and ions per pulse.

    Args:
        data (dict): A dictionary containing the 'start_counter' data.
        max_start_counter (int): The maximum start counter value.

    Returns:
        tuple: A tuple containing two numpy arrays: delta_p and multi.

    Raises:
        IndexError: If the length of counter is less than 1.

    """

    counter = data['start_counter'].to_numpy()
    delta_p = np.zeros(len(counter))
    multi = np.zeros(len(counter))

    multi_hit_count = 1

    total_iterations = len(counter)
    twenty_percent = total_iterations // 5  # 20% of total iterations

    for i, current_counter in enumerate(counter):
        if i == 0:
            delta_p[i] = 0
            previous_counter = current_counter
        else:
            sc = current_counter - previous_counter
            if sc < 0:
                sc_a = max_start_counter - previous_counter
                sc_b = current_counter
                sc = sc_a + sc_b

            delta_p[i] = sc

            if current_counter == previous_counter:
                multi_hit_count += 1
            else:
                for j in range(multi_hit_count):
                    if i + j <= len(counter):
                        multi[i - j - 1] = multi_hit_count

                multi_hit_count = 1
                previous_counter = current_counter
        # for the last event
        if i == len(counter) - 1:
            multi[i] = multi_hit_count

        # Print progress at each 20% interval
        if i % twenty_percent == 0:
            progress_percent = int((i / total_iterations) * 100)
            print(f"Progress: {progress_percent}% complete")

    return delta_p, multi
