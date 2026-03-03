import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pybaselines
from adjustText import adjust_text
from matplotlib import rcParams
from pybaselines import Baseline
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths, peak_prominences

from pyccapt.calibration.calibration import intractive_point_identification
from pyccapt.calibration.calibration.background import fit_background
from pyccapt.calibration.data_tools import data_loadcrop, plot_vline_draw
from pyccapt.calibration.path_utils import save_figure



class AptHistPlotter:
    """
    This class plots the histogram of the mass-to-charge ratio (mc) or time of flight (tof) data.
    """

    def __init__(self, mc_tof, variables=None):
        """
        Initializes all the attributes of AptHistPlotter.

        Args:
            mc_tof (numpy.ndarray): Array for mc or tof data.
            variables (share_variables.Variables): The global experiment variables.
        """
        self.line_manager = None
        self.distance = None
        self.prominence = None
        self.percent = None
        self.rectangle = None
        self.bins = None
        self.plotted_circles = []
        self.plotted_lines = []
        self.plotted_labels = []
        self.original_x_limits = None
        self.bin_width = None
        self.fig = None
        self.ax = None
        self.mc_tof = mc_tof
        self.variables = variables
        self.x = None
        self.y = None
        self.peak_annotates = []
        self.annotates = []
        self.patches = None
        self.peaks = None
        self.properties = None
        self.peak_widths = None
        self.prominences = None
        self.mask_f = None
        self.plot_show = True
        self.legend_colors = []

    def plot_histogram(self, bin_width=0.1, normalize=False, label='mc', log=True, grid=False, steps='stepfilled',
                       fig_size=(9, 5), plot_show=True):
        """
        Plot the histogram of the mc or tof data.

        Args:
            bin_width (float): The width of the bins.
            normalize (bool): Whether to normalize the histogram.
            label (str): The label of the x-axis ('mc' or 'tof').
            log (bool): Whether to use log scale for the y-axis.
            grid (bool): Whether to show the grid.
            steps (str): The type of the histogram ('stepfilled' or 'bar').
            fig_size (tuple): The size of the figure.
            plot_show (bool): Whether to show the plot.

        Returns:
            tuple: A tuple of the y and x values of the histogram.

        """
        # Define the bins
        self.bin_width = bin_width
        self.plot_show = plot_show
        self.bins = np.linspace(np.min(self.mc_tof), np.max(self.mc_tof), round(np.max(self.mc_tof) / bin_width))

        # Plot the histogram directly
        self.fig, self.ax = plt.subplots(figsize=fig_size)

        if steps == 'bar':
            edgecolor = None
            alpha = 1
        else:
            edgecolor = 'k'
            alpha = 0.9

        if normalize:
            self.y, self.x, self.patches = self.ax.hist(self.mc_tof, bins=self.bins, alpha=alpha,
                                                        color='slategray', edgecolor=edgecolor, histtype=steps,
                                                        density=True)
        else:
            self.y, self.x, self.patches = self.ax.hist(self.mc_tof, bins=self.bins, alpha=alpha, color='slategray',
                                                        edgecolor=edgecolor, histtype=steps)
        self.ax.set_xlabel('Mass/Charge [Da]' if label == 'mc' else 'Time of Flight [ns]')
        self.ax.set_ylabel('Event Counts')
        self.ax.set_yscale('log' if log else 'linear')
        if grid:
            plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.4, alpha=0.3)
        if self.original_x_limits is None:
            self.original_x_limits = self.ax.get_xlim()  # Store the original x-axis limits
        plt.tight_layout()
        # Set the limits for both x and y axes using plt.ylim
        # plt.ylim(bottom=plt.yticks()[0][0], top=plt.yticks()[0][-1])
        # plt.xlim(left=plt.xticks()[0][0], right=plt.xticks()[0][-1])
        if plot_show:
            plt.show()
        else:
            plt.close()
        if self.variables is not None:
            self.variables.x_hist = self.x
            self.variables.y_hist = self.y
        return self.y, self.x

    def plot_line_hist(self):
        """
        Plot the histogram as a line plot.

        Args:
            None

        Returns:
            None
        """
        bin_centers = (self.bins[:-1] + self.bins[1:]) / 2  # Compute bin centers
        self.ax.plot(bin_centers, self.y, color='slategray')
        # Step 2: Remove the histogram patches (bars)
        for patch in self.patches:
            patch.set_visible(False)

    def plot_range(self, range_data, legend=True, legend_loc='upper right'):
        """
        Plot the range of the histogram.

        Args:
            range_data (data frame): The range data.
            legend (bool): Whether to show the legend.
            legend_loc (str): The location of the legend.

        Returns:
            None
        """
        if len(self.patches) == len(self.x) - 1:
            colors = range_data['color'].tolist()
            mc_low = range_data['mc_low'].tolist()
            mc_up = range_data['mc_up'].tolist()
            mc = range_data['mc'].tolist()
            ion = range_data['ion'].tolist()
            color_mask = np.full((len(self.x)), '#708090')  # default color is slategray
            for i in range(len(ion)):
                mask = np.logical_and(self.x >= mc_low[i], self.x <= mc_up[i])
                color_mask[mask] = colors[i]

            for i in range(len(self.x) - 1):
                if color_mask[i] != '#708090':
                    self.patches[i].set_facecolor(color_mask[i])

            for i in range(len(ion)):
                self.legend_colors.append((r'%s' % ion[i], plt.Rectangle((0, 0), 1, 1, fc=colors[i])))
                x_offset = 0.0  # Adjust this value as needed

                # Find the bin that contains the mc[i]
                bin_index = np.searchsorted(self.x, mc[i]) - 1
                if 0 <= bin_index < len(self.y):
                    # Define a small range around the bin to search for the local maximum
                    search_range = slice(max(0, bin_index - 1), min(len(self.y), bin_index + 2))
                    local_bins = self.y[search_range]
                    local_x = self.x[search_range.start:search_range.stop]

                    # Find the local maximum and its position
                    max_idx = np.argmax(local_bins)
                    peak_height = local_bins[max_idx]
                    peak_position = local_x[max_idx]

                    # Dynamic y_offset based on log scale
                    y_offset = peak_height * 0.05
                    if self.ax.get_yscale() == 'log':
                        y_offset = 10 ** (np.log10(peak_height) + 0.1) - peak_height

                    self.peak_annotates.append(plt.text(
                        peak_position + x_offset,
                        peak_height + y_offset,
                        r'%s' % ion[i],
                        color='black',
                        size=10,
                        alpha=1,
                        rotation=90
                    ))
                    self.annotates.append(str(i + 1))

            if legend:
                self.plot_color_legend(loc=legend_loc)
        else:
            print('plot_range only works in plot_histogram mode=bar')

    def change_peak_color(self, peak_loc, dx, color='red'):
        """
        Change the color of the peak.

        Args:
            peak_loc (float): The location of the peak.
            dx (float): The width of the peak.
            color (str): The color of the peak.

        Returns:
            None
        """
        bin_index = np.digitize([peak_loc], self.x) - 1
        try:
            self.ranged_line.remove()
        except AttributeError:
            pass
        # Ensure bin_index is within valid range
        if bin_index < 0 or bin_index >= len(self.y):
            raise IndexError(f"Bin index {bin_index} out of range for y array of length {len(self.y)}")

        # Get the scalar value for ymax
        ymax = float(self.y[bin_index])
        # Plot the vertical line with the specified color and properties
        self.ranged_line = plt.axvline(
            x=peak_loc,
            color=color,
            linestyle='dashdot',
            linewidth=2,
            ymax=ymax
        )

    def plot_peaks(self, range_data=None, mode='peaks'):
        """
        Plot the peaks of the histogram.

        Args:
            range_data (data frame): The range data.
            mode (str): The mode of the peaks ('peaks', 'range', or 'peaks_range').

        Returns:
            None
        """
        x_offset = 0.0  # Adjust this value as needed
        if range_data is not None:
            ion = range_data['ion'].tolist()
            mc = range_data['mc'].tolist()
            for i in range(len(ion)):
                # Find the bin that contains the mc[i]
                bin_index = np.searchsorted(self.x, mc[i]) - 1
                if 0 <= bin_index < len(self.y):
                    # Define a small range around the bin to search for the local maximum
                    search_range = slice(max(0, bin_index - 1), min(len(self.y), bin_index + 2))
                    local_bins = self.y[search_range]
                    local_x = self.x[search_range.start:search_range.stop]

                    # Find the local maximum and its position
                    max_idx = np.argmax(local_bins)
                    peak_height = local_bins[max_idx]
                    peak_position = local_x[max_idx]

                    # Dynamic y_offset based on log scale
                    y_offset = peak_height * 0.05
                    if self.ax.get_yscale() == 'log':
                        y_offset = 10 ** (np.log10(peak_height) + 0.1) - peak_height
                else:
                    peak_position = mc[i]
                    peak_height = self.y[np.searchsorted(self.x, mc[i]) - 1]
                if self.plot_show:
                    self.peak_annotates.append(plt.text(peak_position + x_offset, peak_height + y_offset,
                                                        r'%s' % ion[i], color='black', size=10, alpha=1,
                                                        rotation=90))
                    self.annotates.append(str(i + 1))
        else:
            y_offset = 0.0  # Adjust this value as needed
            if mode == 'peaks':
                for i in range(len(self.peaks)):
                    if self.plot_show:
                        # Dynamic y_offset based on log scale
                        peak_height = self.y[self.peaks][i]
                        y_offset = peak_height * 0.05
                        if self.ax.get_yscale() == 'log':
                            y_offset = 10 ** (np.log10(peak_height) + 0.1) - peak_height

                        self.peak_annotates.append(
                            plt.text(self.x[self.peaks][i] + x_offset, peak_height + y_offset,
                                     '%s' % '{:.2f}'.format(self.x[self.peaks][i]), color='black', size=10, alpha=1,
                                     rotation=90))

                        self.annotates.append(str(i + 1))

            elif mode == 'range':
                y_offset = 0.0  # Adjust this value as needed
                for i in range(len(self.variables.peaks_x_selected)):
                    # Find the bin that contains the mc[i]
                    bin_index = np.searchsorted(self.x, self.variables.peaks_x_selected[i])
                    peak_height = self.y[bin_index] * ((self.variables.peaks_x_selected[i] -
                                                        self.x[bin_index - 1]) / self.bin_width)
                    if self.plot_show:
                        self.peak_annotates.append(
                            plt.text(self.variables.peaks_x_selected[i] + x_offset, peak_height + y_offset,
                                     '%s' % '{:.2f}'.format(self.variables.peaks_x_selected[i]), color='black', size=10,
                                     alpha=1, rotation=90))

                        self.annotates.append(str(i + 1))

    def plot_color_legend(self, loc, detailed_isotope=False, detailed_charge=False):
        """
        Plot the color legend.

        Args:
            loc (str): The location of the legend.

        Returns:
            None
        """
        # make a copy of the legend colors
        legend_colors_edited = self.legend_colors.copy()
        if not detailed_isotope or not detailed_charge:
            # Regular expression pattern to remove isotope notation
            pattern = r"\$\{\}\^\{\d+\}([A-Za-z]+.*)\$"
            for i in range(len(legend_colors_edited)):
                legend_colors_edited[i] = (re.sub(pattern, r"$\1$", legend_colors_edited[i][0]), legend_colors_edited[i][1])
            # remove ununique labels
            unique_tuples = {}
            for key, value in legend_colors_edited:
                if key not in unique_tuples:
                    unique_tuples[key] = value

            # Convert the dictionary back to a list of tuples
            legend_colors_edited = list(unique_tuples.items())
        if not detailed_charge:
            # Regular expression pattern to remove isotope notation
            pattern_1 =  r"\^{\d+\}|\{\+|\{-\}|\{\d+[+-]?\}"
            # Regular expression pattern to remove the isotope notation, charge, and caret ^
            pattern_2 = r"\{\d+\}|[\^{}+-]"
            for i in range(len(legend_colors_edited)):
                legend_colors_edited[i] = (re.sub(pattern_1, "", legend_colors_edited[i][0]), legend_colors_edited[i][1])
                legend_colors_edited[i] = (re.sub(pattern_2, "", legend_colors_edited[i][0]), legend_colors_edited[i][1])
            # remove ununique labels
            # Using a set to track seen elements and filter out duplicates
            seen = set()
            unique_data = []

            for item in legend_colors_edited:
                if item[0] not in seen:
                    seen.add(item[0])
                    unique_data.append(item)
            legend_colors_edited = unique_data
        # Adjust the layout
        if len(legend_colors_edited) > 5:
            ncol = max(1, math.ceil(len(legend_colors_edited) / 8))
        else:
            ncol = 1
        self.ax.legend([label[1] for label in legend_colors_edited], [label[0] for label in legend_colors_edited],
                       loc=loc, ncol=ncol)

    def plot_hist_info_legend(self, label='mc', mrp_all=False, background=None, legend_mode='long',
                              loc='left'):
        """
        Plot the histogram information legend.

        Args:
            label (str): The label of the x-axis ('mc' or 'tof').
            mrp_all (bool): Whether to calculate the MRP for 0.1 and 0.01 and 0.5
            background (dict): The background data.
            legend_mode (str): long or short legend info
            loc (str): The location of the legend.

        Returns:
            None
        """
        index_peak_max = np.argmax(self.prominences[0])
        if label == 'mc' or label == 'mc_c':
            mrp = '{:.2f}'.format(
                self.x[self.peaks][index_peak_max] / (self.x[round(self.peak_widths[3][index_peak_max])] -
                                                      self.x[round(self.peak_widths[2][index_peak_max])]))
            if mrp_all:
                mrp_list, mrp_list_all_peak = self.mrp_calculation()
            if background is not None:
                if mrp_all:
                    if legend_mode == 'long':
                        txt = 'bin width: %s Da\nnum atoms: %.2f$e^6$\nbackG: %s ppm/Da\nMRP(0.5): %s\nMRP(0.1): %s\nMRP(0.01): %s' \
                              % (self.bin_width, len(self.mc_tof) / 1000000, round(self.background_ppm), mrp_list[0],
                                 mrp_list[1], mrp_list[2])
                    elif legend_mode == 'short':
                        txt = 'MRP(0.5): %s\nMRP(0.1): %s\nMRP(0.01): %s' % (mrp_list[0], mrp_list[1], mrp_list[2])

                else:
                    if legend_mode == 'long':
                        txt = 'bin width: %s Da\nnum atoms: %.2f$e^6$\nbackG: %s ppm/Da\nFW%d%%M: %s' \
                              % (self.bin_width, len(self.mc_tof) / 1000000, round(self.background_ppm), self.percent, mrp)
                    elif legend_mode == 'short':
                        txt = 'MRP(0.5): %s' % (mrp)
            else:
                # annotation with range stats

                if mrp_all:
                    if legend_mode == 'long':
                        txt = 'bin width: %s Da\nnum atoms: %.2f$e^6$\nMRP(0.5): %s\nMRP(0.1): %s\nMRP(0.01): %s' \
                              % (self.bin_width, len(self.mc_tof) / 1000000, mrp_list[0], mrp_list[1],
                                 mrp_list[2])
                    elif legend_mode == 'short':
                        txt = 'MRP(0.5): %s\nMRP(0.1): %s\nMRP(0.01): %s' % (mrp_list[0], mrp_list[1], mrp_list[2])
                else:
                    if legend_mode == 'long':
                        txt = 'bin width: %s Da\nnum atoms: %.2f$e^6$\nMRP(%s): %s' \
                              % (self.bin_width, (len(self.mc_tof) / 1000000), self.percent / 100, mrp)
                    elif legend_mode == 'short':
                        txt = 'MRP(0.5): %s' % (mrp)

        elif label == 'tof' or label == 'tof_c':
            mrp = '{:.2f}'.format(
                self.x[self.peaks[index_peak_max]] / (self.x[round(self.peak_widths[3][index_peak_max])] -
                                                      self.x[round(self.peak_widths[2][index_peak_max])]))
            if mrp_all:
                mrp_list, mrp_list_all_peak = self.mrp_calculation()
            if background is not None:
                if mrp_all:
                    if legend_mode == 'long':
                        txt = 'bin width: %s ns\nnum atoms: %.2f$e^6$\nbackG: %s ppm/ns\nMRP(0.5): %s\nMRP(0.1): %s\nMRP(0.01): %s' \
                              % (self.bin_width, len(self.mc_tof) / 1000000, round(self.background_ppm), mrp_list[0],
                                 mrp_list[1], mrp_list[2])
                    elif legend_mode == 'short':
                        txt = 'MRP(0.5): %s\nMRP(0.1): %s\nMRP(0.01): %s' % (mrp_list[0], mrp_list[1], mrp_list[2])
                else:
                    if legend_mode == 'long':
                        txt = 'bin width: %s ns\nnum atoms: %.2f$e^6$\nbackG: %s ppm/ns\nMRP(%s): %s' \
                              % (self.bin_width, len(self.mc_tof) / 1000000, round(self.background_ppm),
                                 self.percent / 100, mrp)
                    elif legend_mode == 'short':
                        txt = 'MRP(0.5): %s' % (mrp)
            else:
                if mrp_all:
                    if legend_mode == 'long':
                        txt = 'bin width: %s ns\nnum atoms: %.2f$e^6$\nMRP(0.5): %s\nMRP(0.1): %s\nMRP(0.01): %s' \
                              % (self.bin_width, len(self.mc_tof) / 1000000, mrp_list[0], mrp_list[1],
                                 mrp_list[2])
                    elif legend_mode == 'short':
                        txt = 'MRP(0.5): %s\nMRP(0.1): %s\nMRP(0.01): %s' % (mrp_list[0], mrp_list[1], mrp_list[2])
                else:
                    if legend_mode == 'long':
                        txt = 'bin width: %s ns\nnum atoms: %.2f$e^6$ \nMRP(%s): %s' \
                              % (self.bin_width, len(self.mc_tof) / 1000000, self.percent / 100, mrp)
                    elif legend_mode == 'short':
                        txt = 'MRP(0.5): %s' % (mrp)

        props = dict(boxstyle='round', facecolor='#CCCCCC', alpha=1)
        if loc == 'left':
            self.ax.text(.01, .95, txt, va='top', ma='left', transform=self.ax.transAxes, bbox=props, fontsize=10,
                         alpha=1,
                         horizontalalignment='left', verticalalignment='top')
        elif loc == 'right':
            self.ax.text(.98, .95, txt, va='top', ma='left', transform=self.ax.transAxes, bbox=props, fontsize=10,
                         alpha=1,
                         horizontalalignment='right', verticalalignment='top')

    def mrp_calculation(self):
        """
        Calculate the MRP.

        Args:
            None

        Returns:
            list: A list of the MRP values.
            dict: A dictionary contains the percentage of MRP and the MRP value
        """
        mrp_range = [0.5, 0.1, 0.01]
        mrp_peak = []
        mrp = {}
        for mrp_s in mrp_range:
            mrp_r = 1 - mrp_s
            try:
                peaks, properties = find_peaks(self.y, prominence=None, distance=None, height=0)
                peak_width = peak_widths(self.y, peaks, rel_height=mrp_r, prominence_data=None)
                prominences = peak_prominences(self.y, peaks, wlen=None)
            except ValueError:
                print('Peak finding failed for MRP(%s)' % mrp_r)

            index_peak_max = np.argmax(prominences[0])
            mrp_tmp = self.x[peaks][index_peak_max] / (self.x[round(peak_width[3][index_peak_max])] -
                                                       self.x[round(peak_width[2][index_peak_max])])
            mrp_peak.append(round(mrp_tmp, 2))

            try:
                peaks, properties = find_peaks(self.y, prominence=self.prominence, distance=self.distance, height=0)
                peak_width = peak_widths(self.y, peaks, rel_height=mrp_r, prominence_data=None)
                prominences = peak_prominences(self.y, peaks, wlen=None)
            except ValueError:
                print('Peak finding failed for MRP(%s)' % mrp_r)
            mrp_tmp = []
            peak_width_tmp = []
            for i in range(len(peaks)):
                mrp_tmp_2 = self.x[peaks][i] / (self.x[round(peak_width[3][i])] - self.x[round(peak_width[2][i])])
                mrp_tmp.append(round(mrp_tmp_2, 2))
                peak_width_tmp.append([self.x[round(peak_width[2][i])], self.x[round(peak_width[3][i])]])
                # dictionary contains the percentage of MRP and the MRP value
            mrp['MRP(%s)' % mrp_s] = mrp_tmp
            mrp['peak_sides(%s)' % mrp_s] = peak_width_tmp

        return mrp_peak, mrp

    def plot_horizontal_lines(self):
        """
        Plot the horizontal lines.

        Args:
            None

        Returns:
            None
        """
        for i in range(len(self.variables.h_line_pos)):
            if np.max(self.mc_tof) + 10 > self.variables.h_line_pos[i] > np.max(self.mc_tof) - 10:
                plt.axvline(x=self.variables.h_line_pos[i], color='b', linestyle='--', linewidth=2)

    def plot_background(self, mode, non_peaks=None, lam=1e6, tol=1e-1, max_iter=100, num_std=3.0, plot=True,
                        patch=True):
        """
        Plot the background of the histogram.

        Args:
            mode (str): The mode of the background ('aspls', 'fabc', 'manual@4', or 'manual@100').
            non_peaks (numpy.ndarray): The non-peaks data.
            lam (float): The lambda value for the background fitting.
            tol (float): The tolerance value for the background fitting.
            max_iter (int): The maximum number of iterations for the background fitting.
            num_std (int): The number of standard deviations for the background fitting.
            poly_order (int): The polynomial order for the background fitting.
            plot_no_back (bool): Whether to plot the background.
            plot (bool): Whether to plot the background fit.
            patch (bool): Whether to plot the patch.

        Returns:
            numpy.ndarray: The mask of the background.
        """
        if mode == 'aspls':
            baseline_fitter = Baseline(x_data=self.bins[:-1])
            fit_2, params_2 = baseline_fitter.aspls(self.y, lam=lam, tol=tol, max_iter=max_iter)

        if mode == 'fabc':
            fit_2, params_2 = pybaselines.classification.fabc(self.y, lam=lam,
                                                              num_std=num_std,
                                                              pad_kwargs='edges')

        if mode == 'manual@4':
            upperLim = 4.5  # Da
            lowerLim = 3.5  # Da
            mask = np.logical_and((self.x >= lowerLim), (self.x <= upperLim))
            BG4 = np.sum(self.y[np.array(mask[:-1])]) / (upperLim - lowerLim)
            self.background_ppm = BG4 / len(self.mc_tof) * 1E6
            self.background_ppm = round(self.background_ppm, 2)
            # print only the ppm of the noise up to two decimal places
            handles, labels = plt.gca().get_legend_handles_labels()
            handles.append(plt.Line2D([], [], linestyle='none'))
            labels.append('Noise ppm: ' + str(self.background_ppm))
            plt.legend(handles, labels, frameon=False, loc='upper left')

        if mode == 'manual@100':
            upperLim = 100.5
            lowerLim = 99.5
            mask = np.logical_and((self.x >= lowerLim), (self.x <= upperLim))
            BG100 = np.sum(self.y[np.array(mask[:-1])]) / (upperLim - lowerLim)
            self.background_ppm = BG100 / len(self.mc_tof) * 1E6
            self.background_ppm = round(self.background_ppm, 2)
            # print only the ppm of the noise up to two decimal places
            handles, labels = plt.gca().get_legend_handles_labels()
            handles.append(plt.Line2D([], [], linestyle='none'))
            labels.append('Noise ppm: ' + str(self.background_ppm))
            plt.legend(handles, labels, frameon=False, loc='upper left')

        if plot:
            if mode == 'fabc':
                keys = list(params_2.keys())
                if 'mask' in keys:
                    mask_2 = params_2['mask']
                    noise = 0
                    for i in range(len(mask_2)):
                        if mask_2[i]:
                            noise += self.y[i]
                    # print only the ppm of the noise up to two decimal places
                    handles, labels = plt.gca().get_legend_handles_labels()
                    handles.append(plt.Line2D([], [], linestyle='none'))
                    self.background_ppm = round(noise / len(self.mc_tof) * 1E6 / np.max(self.mc_tof), 2)
                    labels.append('Noise ppm: ' + str(self.background_ppm))
                    plt.legend(handles, labels, frameon=False, loc='upper left')

                    if patch:
                        self.ax.plot(self.bins[:-1][mask_2], self.y[mask_2], 'o', color='orange')[0]
            elif mode == 'aspls':
                # Calculate the effective height of each bin
                effective_heights = []
                for i in range(len(self.bins) - 1):
                    # Get the curve height at each bin's x position
                    background_height = fit_2[i]

                    # Bin height
                    bin_height = self.y[i]

                    # The effective height is the minimum of the bin height and the background line height
                    effective_height = min(bin_height, background_height)
                    effective_heights.append(effective_height)

                effective_heights = np.array(effective_heights)
                # print only the ppm of the noise up to two decimal places
                handles, labels = plt.gca().get_legend_handles_labels()
                handles.append(plt.Line2D([], [], linestyle='none'))
                self.background_ppm = round(np.sum(effective_heights) / len(self.mc_tof) * 1E6
                                            / np.max(self.mc_tof), 2)
                labels.append('Noise ppm: ' + str(self.background_ppm))
                plt.legend(handles, labels, frameon=False, loc='upper left')

        if plot and mode != 'manual@4' and mode != 'manual@100':
            self.ax.plot(self.bins[:-1], fit_2, label='class')
            ax3 = self.ax.twiny()
            ax3.axis("off")
            ax3.plot(fit_2, label='aspls', color='blue')

        return self.mask_f

    def exponential_decay_with_linear_and_dc(self, x, a, b, c, d):
        """Exponential decay function with a linear term and DC offset for fitting."""
        return a * np.exp(-b * x) + c * x + d

    def manual_background_fit(self, ):
        """
        Let the user interactively select points on the histogram plot with the left click and fit an exponential decay
        function with a linear term and a DC offset to the selected points using the right click. Points are removed
        after the line is plotted.
        """

        # Check if a histogram has been plotted
        if self.fig is None or self.ax is None:
            raise RuntimeError("No histogram plotted. Please run plot_histogram first.")

        selected_points = []
        point_markers = []  # List to store the plotted points for later removal

        def onclick(event):
            # Left click to select points
            if event.button == 1 and event.inaxes == self.ax:
                x, y = event.xdata, event.ydata
                selected_points.append((x, y))
                marker, = self.ax.plot(x, y, 'ro')  # Plot the selected point in red and store the marker
                point_markers.append(marker)
                self.fig.canvas.draw()

            # Right click to fit and plot exponential decay function with linear term and DC offset
            elif event.button == 3:
                self.fig.canvas.mpl_disconnect(cid)
                fit_and_plot_exponential_with_linear_and_dc()

        def fit_and_plot_exponential_with_linear_and_dc():

            if len(selected_points) < 2:
                print("At least 2 points are required to fit an exponential decay with a linear term and DC offset.")
                return

            # Extract x and y coordinates from the selected points
            x_points, y_points = zip(*selected_points)

            # Fit the exponential decay function with linear term and DC offset to the selected points
            self.popt, _ = curve_fit(self.exponential_decay_with_linear_and_dc, x_points, y_points, maxfev=10000)
            a, b, c, d = self.popt

            # Plot the fitted exponential decay function with linear term and DC offset
            x_vals = np.linspace(min(self.x), max(self.x), 500)
            y_vals = self.exponential_decay_with_linear_and_dc(x_vals, a, b, c, d)
            self.ax.plot(x_vals, y_vals, 'b-')

            # Remove the selected points from the plot
            for marker in point_markers:
                marker.remove()
            point_markers.clear()  # Clear the list of point markers

            # Redraw the plot with the fitted function
            self.fig.canvas.draw()

            self.calculate_noise(plot_without_noise=False)

        # Connect the click event to the onclick function
        cid = self.fig.canvas.mpl_connect('button_press_event', onclick)

        print("Left-click to select points on the plot. Right-click to fit the exponential decay with linear "
              "term and DC offset.")

    def calculate_noise(self, fig_size=(9, 5), plot_without_noise=False):
        """
        Create a mask by subtracting the fitted background from the original data,
        and calculate the noise ppm.
        """
        if self.popt is None:
            raise RuntimeError("No background fitted. Please fit the background first.")
        else:
            a, b, c, d = self.popt
            bin_edges = self.bins[:]

            effective_heights = []

            for i in range(len(bin_edges) - 1):
                # Get the curve height at bin edges
                y_left = self.exponential_decay_with_linear_and_dc(bin_edges[i], a, b, c, d)
                y_right = self.exponential_decay_with_linear_and_dc(bin_edges[i + 1], a, b, c, d)

                # Bin height
                bin_height = self.y[i]

                # The height of the bin covered by the curve (clamp to bin height if necessary)
                height_under_curve = min(max(y_left, y_right), bin_height)

                effective_heights.append(height_under_curve)

            # Convert effective_heights to numpy array
            effective_heights = np.array(effective_heights)

            # Subtract the effective heights (background) from the actual bin heights (self.y)
            y_noise_removed = self.y - effective_heights

            # print only the ppm of the noise up to two decimal places
            handles, labels = plt.gca().get_legend_handles_labels()
            handles.append(plt.Line2D([], [], linestyle='none'))
            self.background_ppm = round(np.sum(effective_heights) / len(self.mc_tof) * 1E6 /
                                        np.max(self.mc_tof), 2)
            labels.append('Noise ppm: ' + str(self.background_ppm))
            plt.legend(handles, labels, frameon=False, loc='upper left')

            if plot_without_noise:
                # Create a new figure and axis for the updated plot
                new_fig, new_ax = plt.subplots(figsize=fig_size)

                # Plot the histogram with the adjusted bin heights
                new_ax.hist(self.x[:-1], self.x, weights=y_noise_removed, alpha=0.9, color='slategray',
                            edgecolor='k', histtype='stepfilled')

                # Restore plot properties
                new_ax.set_xlabel(
                    'Mass/Charge [Da]' if self.ax.get_xlabel() == 'Mass/Charge [Da]' else 'Time of Flight [ns]')
                new_ax.set_ylabel('Event Counts')
                new_ax.set_yscale('log' if self.ax.get_yscale() == 'log' else 'linear')
                if self.original_x_limits is not None:
                    new_ax.set_xlim(self.original_x_limits)

                new_ax.legend()
                plt.tight_layout()
                plt.show()

    def plot_founded_range_loc(self, df, remove_lines=False):
        """
        Plot the founded range location.

        Args:
            df (data frame): The data frame of the founded range.
            remove_lines (bool): Whether to remove the lines.

        Returns:
            None
        """
        if remove_lines or self.plotted_lines:
            # Remove previously plotted lines,circles and labels
            for line, circle, label in zip(self.plotted_lines, self.plotted_circles, self.plotted_labels):
                line.remove()
                circle[0].remove()
                label.remove()

            # Clear the lists
            self.plotted_lines.clear()
            self.plotted_circles.clear()
            self.plotted_labels.clear()
        elif not remove_lines:
            ax1 = self.ax.twinx()
            ions = df['ion']
            abundances = df['abundance']
            mass = df['mass']

            # Define the scaling factor for the abundance to control the line height
            scaling_factor = 1.0  # Adjust as needed

            for ion, abundance, m in zip(ions, abundances, mass):
                # Calculate the height of the line based on abundance
                line_height = abundance * scaling_factor

                # Plot a vertical line at the position of 'mass' with the specified height
                line = ax1.vlines(x=m, ymin=0, ymax=line_height, color='red', linestyles='dashed')

                # Plot an empty circle marker at the top of the line
                circle = ax1.plot(m, line_height, marker='o', markersize=6, color='white', markeredgecolor='red')

                # Annotate the ion label (LaTeX formula) near the circle
                label = ax1.annotate(ion, xy=(m, line_height), xytext=(m, line_height), fontsize=10,
                                     color='blue', annotation_clip='clip_on', textcoords="offset points",
                                     xycoords="data")

                self.plotted_lines.append(line)  # Keep track of the plotted lines
                self.plotted_circles.append(circle)  # Keep track of the plotted circles
                self.plotted_labels.append(label)  # Keep track of the plotted labels
                # Remove the y-axis and labels
                ax1.get_yaxis().set_visible(False)
                # Set the y-axis to log scale
                ax1.set_yscale('log')

    def find_peaks_and_widths(self, prominence=None, distance=None, percent=50):
        """
        Find the peaks and widths of the histogram.

        Args:
            prominence (float): The minimum prominence of peaks.
            distance (float): The minimum horizontal distance in samples between peaks.
            percent (float): The percentage of the peak height to calculate the peak width.

        Returns:
            tuple: A tuple of the peaks, properties, peak widths, and prominences.
        """
        self.percent = percent
        percent = 100 - percent
        self.prominence = prominence
        self.distance = distance
        try:
            self.peaks, self.properties = find_peaks(self.y, prominence=self.prominence, distance=self.distance,
                                                     height=0)
            self.peak_widths = peak_widths(self.y, self.peaks, rel_height=(percent / 100), prominence_data=None)
            self.prominences = peak_prominences(self.y, self.peaks, wlen=None)

            x_peaks = self.x[self.peaks]
            y_peaks = self.y[self.peaks]
            self.variables.peak_x = x_peaks
            self.variables.peak_y = y_peaks
            index_max_ini = np.argmax(y_peaks)
            self.variables.max_peak = x_peaks[index_max_ini]
            self.variables.peak_widths = self.peak_widths

        except ValueError:
            print('Peak finding failed.')
            self.peaks = None
            self.properties = None
            self.peak_widths = None
            self.prominences = None
            self.variables.peak_x = None
            self.variables.peak_y = None
            self.variables.max_peak = None

        return self.peaks, self.properties, self.peak_widths, self.prominences

    def draw_rectangle(self, ):
        """
        Draw the rectangle and find .

        Args:
            None

        Returns:
            None
        """
        # Automatic left and right side of the max peak selection
        index_max_ini = np.argmax(self.prominences[0])
        self.variables.selected_x1 = self.x[round(self.peak_widths[2][index_max_ini])]
        self.variables.selected_x2 = self.x[round(self.peak_widths[3][index_max_ini])]
        self.variables.selected_y1 = 0
        self.variables.selected_y2 = self.prominences[0][index_max_ini]

        # Draw the rectangle
        self.rectangle = plt.Rectangle((self.variables.selected_x1, self.variables.selected_y1),
                                       self.variables.selected_x2 - self.variables.selected_x1,
                                       self.variables.selected_y2,
                                       edgecolor='g', facecolor=(0, 1, 0, 0.5), linewidth=1)
        self.ax.add_patch(self.rectangle)

    def selector(self, selector='rect'):
        """
        Connect the selector to the plot.

        Args:
            selector (str): The type of selector ('rect', 'peak', or 'range').

        Returns:
            None
        """
        if selector == 'rect':
            # Connect and initialize rectangle box selector
            data_loadcrop.rectangle_box_selector(self.ax, self.variables)
            plt.connect('key_press_event', lambda event: data_loadcrop.toggle_selector(event))
        elif selector == 'peak':
            # connect peak_x selector
            af = intractive_point_identification.AnnoteFinder(self.x[self.peaks], self.y[self.peaks], self.annotates,
                                                              self.variables, ax=self.ax)
            self.fig.canvas.mpl_connect('button_press_event', lambda event: af.annotates_plotter(event))

            zoom_manager = plot_vline_draw.HorizontalZoom(self.ax, self.fig)
            self.fig.canvas.mpl_connect('key_press_event', lambda event: zoom_manager.on_key_press(event))
            self.fig.canvas.mpl_connect('key_release_event', lambda event: zoom_manager.on_key_release(event))
            self.fig.canvas.mpl_connect('scroll_event', lambda event: zoom_manager.on_scroll(event))

        elif selector == 'range':
            # connect range selector
            self.line_manager = plot_vline_draw.VerticalLineManager(self.variables, self.ax, self.fig, [], [])

            self.fig.canvas.mpl_connect('button_press_event',
                                        lambda event: self.line_manager.on_press(event))
            self.fig.canvas.mpl_connect('button_release_event',
                                        lambda event: self.line_manager.on_release(event))
            self.fig.canvas.mpl_connect('motion_notify_event',
                                        lambda event: self.line_manager.on_motion(event))
            self.fig.canvas.mpl_connect('key_press_event',
                                        lambda event: self.line_manager.on_key_press(event))
            self.fig.canvas.mpl_connect('scroll_event', lambda event: self.line_manager.on_scroll(event))
            self.fig.canvas.mpl_connect('key_release_event',
                                        lambda event: self.line_manager.on_key_release(event))

    def zoom_to_x_range(self, x_min, x_max, reset=False):
        """
        Zoom the plot to a specific range of x-values.

        Args:
            x_min (float): Minimum x-value for the zoomed range.
            x_max (float): Maximum x-value for the zoomed range.
            reset (bool): If True, reset the zoom to the full range.

        Return:
            None
        """
        if reset:
            """Reset the plot to the original view."""
            if self.original_x_limits is not None:
                self.ax.set_xlim(self.original_x_limits)
                self.fig.canvas.draw()
        else:
            self.ax.set_xlim(x_min, x_max)
            self.fig.canvas.draw()

    def adjust_labels(self):
        """
        Adjust the labels.

        Args:
            None

        Returns:
            None
        """
        adjust_text(self.peak_annotates)

    def save_fig(self, label, fig_name):
        """
        Save the figure.

        Args:
            label (str): The label of the x-axis ('mc' or 'tof').
            fig_name (str): The name of the figure.

        Returns:
            None
        """
        rcParams['svg.fonttype'] = 'none'
        if label == 'mc' or label == 'mc_c':
            save_figure(
                self.fig,
                directory=self.variables.result_path,
                stem=f"mc_{fig_name}",
                formats=("svg", "png"),
                dpi=600,
            )
        elif label == 'tof' or label == 'tof_c':
            save_figure(
                self.fig,
                directory=self.variables.result_path,
                stem=f"tof_{fig_name}",
                formats=("svg", "png"),
                dpi=600,
            )


def hist_plot(variables, bin_size, log, target, normalize, prominence, distance, percent, selector, figname, lim,
              peaks_find=True, peaks_find_plot=False, plot_ranged_peak=False, plot_ranged_colors=False, mrp_all=False,
              background=None, grid=False, ranging_mode=False, range_sequence=[], range_mc=[], range_detx=[],
              range_dety=[], range_x=[], range_y=[], range_z=[], range_vol=[], save_fig=True, print_info=True,
              legend_mode='long', draw_calib_rect=False, figure_size=(9, 5), plot_show=True, fast_calibration=False):
    """
    Plot the mass spectrum or tof spectrum. It is helper function for tutorials.
    Args:
        variables (object): Variables object.
        bin_size (float): Bin size for the histogram.
        target (str): 'mc' for mass spectrum or 'tof' for tof spectrum.
        normalize (bool): Normalized the histogram.
        prominence (float): Prominence for the peak_x finding.
        distance (float): Distance for the peak_x finding.
        percent (float): Percent for the peak_x finding.
        selector (str): Selector for the peak_x finding.
        figname (str): Figure name.
        lim (float): Limit for the histogram.
        peaks_find (bool): Find the peaks.
        peaks_find_plot (bool): Plot the peaks.
        mrp_all (bool): Calculate the MRP for 0.1 and 0.01 and 0.5
        background (str): Background mode.
        grid (bool): Show the grid.
        ranging_mode (bool): Ranging mode.
        legend_mode (str): long or short legend info
        print_info (bool): Print the information about the peaks.
        range_sequence (list): Range sequence.
        range_mc: Range of mc
        range_detx: Range of detx
        range_dety: Range of dety
        range_x: Range of x-axis
        range_y: Range of y-axis
        range_z: Range of z-axis
        print_info: Print the information about the peaks.
        draw_calib_rect (bool): Draw the calibration rectangle.
        figure_size (tuple): Figure size.
        plot_show (bool): Show the plot.
        fast_calibration (bool): Fast calibration.
    Returns:
        None

    """
    assert target in ['mc', 'mc_uc', 'tof', 'tof_c', 'tof_calib', 'mc_calib'], 'Invalid target'
    assert selector in ['peak', 'rect', 'range'], 'Invalid selector'
    assert legend_mode in ['long', 'short'], 'Invalid legend mode'

    if target == 'mc':
        hist = variables.mc
        label = 'mc'
    elif target == 'mc_uc':
        hist = variables.mc_uc
        label = 'mc'
    elif target == 'tof':
        hist = variables.dld_t
        label = 'tof'
    elif target == 'tof_c':
        hist = variables.dld_t_c
        label = 'tof'
    elif target == 'tof_calib':
        hist = variables.dld_t_calib
        label = 'tof'
    elif target == 'mc_calib':
        hist = variables.mc_calib
        label = 'mc'
    if selector == 'peak':
        variables.peaks_x_selected = []
        variables.peak_widths = []
        variables.peaks_index_list = []

    if range_sequence or range_mc or range_detx or range_dety or range_x or range_y or range_z or range_vol:
        if range_sequence:
            mask_sequence = np.zeros_like(hist, dtype=bool)
            mask_sequence[range_sequence[0]:range_sequence[1]] = True
        else:
            mask_sequence = np.ones_like(hist, dtype=bool)
        if range_detx and range_dety:
            mask_det_x = (variables.dld_x_det < range_detx[1]) & (variables.dld_x_det > range_detx[0])
            mask_det_y = (variables.dld_y_det < range_dety[1]) & (variables.dld_y_det > range_dety[0])
            mask_det = mask_det_x & mask_det_y
        else:
            mask_det = np.ones(len(variables.dld_x_det), dtype=bool)
        if range_mc:
            mask_mc = (variables.mc < range_mc[1]) & (variables.mc > range_mc[0])
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
        print('The number of data sequence:', len(mask_sequence[mask_sequence == True]))
        print('The number of data mc:', len(mask_mc[mask_mc == True]))
        print('The number of data det:', len(mask_det[mask_det == True]))
        print('The number of data 3d:', len(mask_3d[mask_3d == True]))
        print('The number of data after cropping:', len(mask[mask == True]))
    else:
        mask = np.ones(len(variables.mc), dtype=bool)

    hist = hist[mask]

    # only calculate for 10 percent of the data
    if fast_calibration:
        hist = np.random.choice(hist, int(len(hist) * 0.1), replace=False)

    if plot_ranged_peak or plot_ranged_colors:
        steps = 'bar'
    else:
        steps = 'stepfilled'

    mc_hist = AptHistPlotter(hist[hist < lim], variables)
    y, x = mc_hist.plot_histogram(bin_width=bin_size, normalize=normalize, label=label, steps=steps, log=log,
                                    grid=grid, fig_size=figure_size, plot_show=plot_show)

    # copy the mc_hist to variables to use the methods of that class in other functions
    variables.AptHistPlotter = mc_hist

    if ranging_mode:
        mc_hist.plot_peaks(range_data=None, mode='range')
        peaks = None
        peak_widths = None
        prominences = None

    elif not normalize and peaks_find and not plot_ranged_peak and not plot_ranged_colors:
        peaks, properties, peak_widths, prominences = mc_hist.find_peaks_and_widths(prominence=prominence,
                                                                          distance=distance, percent=percent)
        if draw_calib_rect:
            mc_hist.draw_rectangle()
        if peaks_find_plot:
            mc_hist.plot_peaks()
        mc_hist.plot_hist_info_legend(label=label, mrp_all=mrp_all, background=None, legend_mode=legend_mode,
                                      loc='right')
    elif plot_ranged_colors and not plot_ranged_peak:
        mc_hist.plot_range(range_data=variables.range_data, legend=True, legend_loc='upper right')
        mc_hist.adjust_labels()
        peaks = None
        peak_widths = None
        prominences = None
    elif plot_ranged_peak and not plot_ranged_colors:
        mc_hist.plot_peaks(range_data=variables.range_data, mode='peaks')
        mc_hist.adjust_labels()
        peaks, properties, peak_widths, prominences = mc_hist.find_peaks_and_widths(prominence=prominence,
                                                                                    distance=distance, percent=percent)
        mc_hist.plot_hist_info_legend(label=label, mrp_all=mrp_all, background=None, legend_mode=legend_mode,
                                      loc='right')

    elif plot_ranged_colors and plot_ranged_peak:
        raise ValueError('Please select only one of the plot_ranged_peak and plot_ranged_colors')

    else:
        peaks = None
        peak_widths = None
        prominences = None

    mc_hist.selector(selector=selector)  # rect, peak_x, range

    if save_fig:
        mc_hist.save_fig(label=label, fig_name=figname)

    mrp_list, mrp_list_all_peak = mc_hist.mrp_calculation()

    if background is not None:
        if background in ['aspls', 'fabc', 'manual@4', 'manual@100']:
            mc_hist.plot_background(mode=background)
        elif background == 'user':
            mc_hist.manual_background_fit()
    if peaks is not None and print_info:
        index_max_ini = np.argmax(prominences[0])
        mrp = x[int(peaks[index_max_ini])] / (
                x[round(peak_widths[3][index_max_ini])] - x[round(peak_widths[2][index_max_ini])])
        print('Mass resolving power for the highest peak number %s (MRP --> m/m_2-m_1):' % int(index_max_ini + 1),
              mrp)
        print('-----------------------------------------------------------------------------')
        for i in range(len(peaks)):
            if not mrp_all:
                print('Peaks ', i + 1,
                      'is at location and height: ({:.2f}, {:.2f})'.format(x[int(peaks[i])], prominences[0][i]),
                      'peak_x window sides ({:.0f} percent-maximum) are: ({:.2f}, {:.2f})'.format(percent,
                                                                                                  x[round(
                                                                                                      peak_widths[2][
                                                                                                          i])],
                                                                                                  x[round(
                                                                                                      peak_widths[3][
                                                                                                          i])]),
                      '-> MRP: {:.2f}'.format(
                          x[round(peaks[i])] / (x[round(peak_widths[3][i])] - x[round(peak_widths[2][i])])))
            if mrp_all:

                for percent in [0.5, 0.1, 0.01]:
                    print('Peaks ', i + 1,
                          'is at location and height: ({:.2f}, {:.2f})'.format(x[int(peaks[i])], prominences[0][i]),
                          'peak_x window sides ({:.0f} percent-maximum) are: ({:.2f}, {:.2f})'.format(percent * 100,
                                                                                                      mrp_list_all_peak[
                                                                                                          'peak_sides(%s)' % percent][
                                                                                                          i][0],
                                                                                                      mrp_list_all_peak[
                                                                                                          'peak_sides(%s)' % percent][
                                                                                                          i][1]),
                          '-> MRP: {:.2f}'.format(mrp_list_all_peak['MRP(%s)' % percent][i]))

            print('------------------------------')

    return mrp_list

