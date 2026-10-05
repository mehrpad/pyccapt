"""Mass-spectrum plotting and peak handling (``AptHistPlotter``).

``AptHistPlotter`` builds the mass-to-charge (or time-of-flight)
histogram that drives peak finding, ranging, MRP estimation, and the
interactive Range tab. Two conventions matter for correctness and are
relied on throughout the calibration core:

- Histogram bin EDGES are anchored to the requested ``bin_width``
  (``np.arange(min, max + bin_width, bin_width)``); the bin count is
  derived from the data span, not the other way round.
- Peak LOCATIONS are reported at bin CENTERS, not left edges, so a
  detected peak is not biased low by half a bin.

The heavy peak/background/selector logic lives in the
``mc_plot_*_helpers`` modules and is imported here; this module is the
thin stateful front end used by the notebook helpers.
"""
import math
import re

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from matplotlib.ticker import FuncFormatter


def _safe_tight_layout(fig=None):
	"""``tight_layout`` that never crashes on a corrupted mathtext parser.

	Computing the layout renders the tick labels, and log-scale labels are
	mathtext (e.g. ``$\\mathdefault{10^{1}}$``). matplotlib's mathtext parser
	only resets pyparsing's global packrat cache *after a successful* parse, so
	a single earlier failed parse anywhere in the session can leave that cache
	poisoned and make every later label raise ``ValueError`` ("Expected end of
	text, found '$'"). Reset the cache and retry once; if it still fails, skip
	the (purely cosmetic) layout rather than break the caller.
	"""
	target = fig if fig is not None else plt
	try:
		target.tight_layout()
		return
	except ValueError:
		try:
			from pyparsing import ParserElement
			ParserElement.reset_cache()
		except Exception:
			pass
		try:
			target.tight_layout()
		except Exception:
			pass  # layout is cosmetic — never let it propagate

from pyccapt.calibration.path_utils import save_figure
from pyccapt.calibration.core.mc_plot_background_helpers import (
    calculate_noise as _calculate_noise,
    exponential_decay_with_linear_and_dc as _exp_decay,
    manual_background_fit as _manual_background_fit,
    plot_background as _plot_background,
)
from pyccapt.calibration.core.mc_plot_peak_helpers import (
    apply_hist_info_legend as _apply_hist_info_legend,
    calculate_mrp as _calculate_mrp,
    draw_rectangle as _draw_rectangle,
    find_peaks_and_widths as _find_peaks_and_widths,
)
from pyccapt.calibration.core.mc_plot_selector_helpers import (
    attach_selector as _attach_selector,
    zoom_to_x_range as _zoom_to_x_range,
)

def _normalize_range_colors(values):
    """Normalize stored range colors for matplotlib usage."""
    normalized = []
    for value in values:
        value = str(value).strip()
        if value and not value.startswith('#') and re.fullmatch(r'[A-Fa-f0-9]{6}', value):
            value = f'#{value}'
        normalized.append(value)
    return normalized

def _plain_range_label(value):
    """Convert stored ion/range labels into plain text safe for matplotlib."""
    text = str(value).strip()
    if not text:
        return text
    text = text.replace("$", "")
    text = re.sub(r"_\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\^\{([^}]*)\}", r" \1", text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("^", "").strip()
    return text

def _resolve_range_display_labels(range_data):
    """Return plain-text labels for ranged overlays and legends."""
    for column in ("name", "ion_name", "ion"):
        if column in range_data.columns:
            labels = [_plain_range_label(value) for value in range_data[column].tolist()]
            if any(label for label in labels):
                return labels
    return [_plain_range_label(value) for value in range(len(range_data))]

def _resolve_range_peak_labels(range_data):
    """Return peak annotation labels, preferring the raw ion column when available."""
    if "ion" in range_data.columns:
        labels = [str(value).strip() for value in range_data["ion"].tolist()]
        if any(label for label in labels):
            return labels
    for column in ("ion_name", "name"):
        if column in range_data.columns:
            labels = [str(value).strip() for value in range_data[column].tolist()]
            if any(label for label in labels):
                return labels
    return [str(value) for value in range(len(range_data))]

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
        self.normalize = False
        self.norm_factor = 1.0
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
        self.x_centers = None
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

    def plot_histogram(
        self,
        bin_width=0.1,
        normalize=False,
        label='mc',
        log=True,
        grid=False,
        steps='stepfilled',
        fig_size=(9, 5),
        plot_show=True,
        fast=False,
        x_lim=None,
        y_headroom=0.4,
    ):
        """
        Plot the histogram of the mc or tof data.

        Args:
            bin_width (float): The width of the bins.
            normalize (bool): Display-only y-axis normalization. The histogram
                data stays in raw counts (so peak finding, MRP, and background
                fits are unaffected); when True the y-axis ticks are relabeled
                to relative intensity (tallest bin = 1).
            label (str): The label of the x-axis ('mc' or 'tof').
            log (bool): Whether to use log scale for the y-axis.
            grid (bool): Whether to show the grid.
            steps (str): The type of the histogram ('stepfilled' or 'bar').
            fig_size (tuple): The size of the figure.
            plot_show (bool): Whether to show the plot.
            fast (bool): Use np.histogram + fill_between instead of ax.hist for speed.
            x_lim: Paper-style x-axis bound. ``None`` (default) hugs the data
                with a tiny pad so the spectrum doesn't trail off into a wide
                empty band (matplotlib's autoscale otherwise pads out to the
                next round tick, e.g. data ending at 100 Da but the axis drawn
                to 120). Pass a single number to force a sharp upper bound
                (e.g. ``100``) or a ``(min, max)`` pair to set both ends;
                ``None`` inside the pair keeps the hugged value for that end.
            y_headroom (float): Extra space above the tallest bin so the top
                peak and its rotated label aren't cramped against the frame.
                In log scale it's added in decades (0.4 ~= a factor of 2.5);
                in linear scale it's a multiplier (0.4 -> top = max * 1.4).
                Set to 0 to keep matplotlib's tight autoscale.

        Returns:
            tuple: A tuple of the y and x values of the histogram.

        """
        # Define the bins. Defensive: filter non-finite values so a
        # single NaN (e.g. from a partial-recovery row whose centred-axis
        # mc_uc failed) doesn't poison np.min / np.max and break the
        # histogram. Non-destructive: ``self.mc_tof`` is overwritten only
        # with the finite subset; the original array is not stored
        # elsewhere on the plotter.
        self.bin_width = bin_width
        self.plot_show = plot_show
        mc_tof_arr = np.asarray(self.mc_tof)
        finite_mask = np.isfinite(mc_tof_arr)
        if not finite_mask.all():
            n_dropped = int((~finite_mask).sum())
            print(f'[AptHistPlotter] Skipping {n_dropped} non-finite mc/tof values.')
            self.mc_tof = mc_tof_arr[finite_mask]
            if self.mc_tof.size == 0:
                raise ValueError("mc_tof has no finite values for histogram")
        # Build edges anchored to ``bin_width``; using ``np.linspace`` would
        # silently disagree with the requested width whenever min > 0 (the
        # actual width becomes (max-min)/(N-1), not bin_width). Use
        # ``arange`` so each bin has exactly the requested width, then add
        # one trailing edge so the last bin is closed.
        lo = float(np.min(self.mc_tof))
        hi = float(np.max(self.mc_tof))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo or bin_width <= 0:
            raise ValueError(
                "Cannot build histogram bins: invalid mc_tof range "
                f"[{lo}, {hi}] or bin_width={bin_width}."
            )
        self.bins = np.arange(lo, hi + bin_width, bin_width)
        if self.bins.size < 2:
            # Degenerate range narrower than one bin; fall back to two edges.
            self.bins = np.array([lo, lo + bin_width], dtype=float)

        # Plot the histogram directly
        self.fig, self.ax = plt.subplots(figsize=fig_size)

        # Always histogram in raw COUNTS so peak finding, MRP, and background
        # fits (which rely on count-based thresholds) keep working. ``normalize``
        # only rescales the y-axis *display* (handled after plotting); the
        # underlying data and every overlay stay in count space.
        if fast and steps != 'bar':
            self.y, self.x = np.histogram(self.mc_tof, bins=self.bins)
            self.x_centers = (self.x[:-1] + self.x[1:]) * 0.5
            self.ax.fill_between(self.x_centers, self.y, step='mid', alpha=0.9, color='slategray')
            self.ax.step(self.x_centers, self.y, where='mid', color='k', linewidth=0.5)
            self.patches = []
        else:
            if steps == 'bar':
                edgecolor = None
                alpha = 1
            else:
                edgecolor = 'k'
                alpha = 0.9

            self.y, self.x, self.patches = self.ax.hist(
                self.mc_tof, bins=self.bins, alpha=alpha, color='slategray', edgecolor=edgecolor, histtype=steps
            )
            self.x_centers = (self.x[:-1] + self.x[1:]) * 0.5

        self.ax.set_xlabel('Mass/Charge [Da]' if label == 'mc' else 'Time of Flight [ns]')
        self.ax.set_yscale('log' if log else 'linear')
        # Display-only normalization: the data above is in counts, so we relabel
        # the y-axis ticks to relative intensity (tallest bin = 1) instead of
        # rescaling the data. Peak/MRP/background analysis therefore keeps
        # running on real counts while the axis reads as normalized.
        self.normalize = normalize
        self.norm_factor = 1.0
        if normalize and self.y.size and float(np.max(self.y)) > 0:
            self.norm_factor = 1.0 / float(np.max(self.y))
            self.ax.yaxis.set_major_formatter(
                FuncFormatter(lambda value, _pos: f'{value * self.norm_factor:.2g}')
            )
            self.ax.set_ylabel('Normalized counts')
        else:
            self.ax.set_ylabel('Event Counts')
        if grid:
            plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.4, alpha=0.3)

        # --- Paper-style axis limits -----------------------------------------
        # matplotlib's autoscale leaves two cosmetic problems for these spectra:
        #   1) the x-axis pads out to the next round tick, leaving a wide empty
        #      band on the right (data ending ~100 Da but the axis drawn to 120);
        #   2) the y-axis top sits right at the tallest bin, so the peak and its
        #      rotated label are cramped against the frame.
        # Hug the x-axis to the data (or to a caller-supplied bound) and raise the
        # y-axis top by ``y_headroom``. Do this before caching ``original_x_limits``
        # so selector/background resets restore these tighter limits, not the
        # autoscaled ones.
        x_lo = float(self.x[0])
        x_hi = float(self.x[-1])
        if x_lim is not None:
            if np.isscalar(x_lim):
                x_hi = float(x_lim)
            else:
                lo_req, hi_req = x_lim
                if lo_req is not None:
                    x_lo = float(lo_req)
                if hi_req is not None:
                    x_hi = float(hi_req)
            self.ax.set_xlim(x_lo, x_hi)
        else:
            x_pad = (x_hi - x_lo) * 0.01
            self.ax.set_xlim(x_lo - x_pad, x_hi + x_pad)

        y_max = float(np.max(self.y)) if self.y.size else 0.0
        if y_max > 0 and y_headroom and y_headroom > 0:
            cur_bottom, _ = self.ax.get_ylim()
            if log:
                # Add headroom in decades; keep matplotlib's autoscaled bottom.
                self.ax.set_ylim(cur_bottom, 10 ** (np.log10(y_max) + y_headroom))
            else:
                self.ax.set_ylim(cur_bottom, y_max * (1.0 + y_headroom))

        if self.original_x_limits is None:
            self.original_x_limits = self.ax.get_xlim()  # Store the original x-axis limits
        _safe_tight_layout(self.fig)
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
            colors = _normalize_range_colors(range_data['color'].tolist())
            mc_low = range_data['mc_low'].tolist()
            mc_up = range_data['mc_up'].tolist()
            mc = range_data['mc'].tolist()
            labels = _resolve_range_display_labels(range_data)
            peak_labels = _resolve_range_peak_labels(range_data)
            color_mask = np.full((len(self.x)), '#708090')  # default color is slategray
            for i in range(len(labels)):
                mask = np.logical_and(self.x >= mc_low[i], self.x <= mc_up[i])
                color_mask[mask] = colors[i]

            for i in range(len(self.x) - 1):
                if color_mask[i] != '#708090':
                    self.patches[i].set_facecolor(color_mask[i])

            seen_legend_labels = set()
            for i in range(len(labels)):
                if labels[i] not in seen_legend_labels:
                    self.legend_colors.append((labels[i], plt.Rectangle((0, 0), 1, 1, fc=colors[i])))
                    seen_legend_labels.add(labels[i])
                x_offset = 0.0  # Adjust this value as needed

                # Find the bin that contains the mc[i]
                bin_index = np.searchsorted(self.x, mc[i]) - 1
                if 0 <= bin_index < len(self.y):
                    # Define a small range around the bin to search for the local maximum
                    search_range = slice(max(0, bin_index - 1), min(len(self.y), bin_index + 2))
                    local_bins = self.y[search_range]
                    local_x = self.x[search_range.start : search_range.stop]

                    # Find the local maximum and its position
                    max_idx = np.argmax(local_bins)
                    peak_height = local_bins[max_idx]
                    peak_position = local_x[max_idx]

                    # Dynamic y_offset based on log scale
                    y_offset = peak_height * 0.05
                    if self.ax.get_yscale() == 'log':
                        y_offset = 10 ** (np.log10(peak_height) + 0.1) - peak_height

                    self.peak_annotates.append(
                        plt.text(
                            peak_position + x_offset,
                            peak_height + y_offset,
                            peak_labels[i],
                            color='black',
                            size=10,
                            alpha=1,
                            rotation=90,
                        )
                    )
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
        # np.digitize returns an ndarray when given an array input; cast
        # to a scalar int so the bounds check (and any future arithmetic)
        # behaves like a normal Python int. The previous
        # ``ndarray < int or ndarray >= int`` raised numpy's
        # DeprecationWarning and will fail outright under future numpy.
        bin_index = int(np.digitize([peak_loc], self.x)[0]) - 1
        try:
            self.ranged_line.remove()
        except AttributeError:
            pass
        # Ensure bin_index is within valid range
        if bin_index < 0 or bin_index >= len(self.y):
            raise IndexError(f"Bin index {bin_index} out of range for y array of length {len(self.y)}")

        # Plot the vertical line on the plotter's own axes so the marker lands
        # on the currently displayed figure even when matplotlib's pyplot state
        # has drifted to a different figure (common under %matplotlib ipympl).
        # NOTE: ``ymax`` for ``Axes.axvline`` is in AXES-FRACTION space
        # ([0, 1]); passing a raw event count (e.g. 8000) silently clips
        # to 1.0 and makes the marker span the full y axis every time --
        # which is the same as just omitting the kwarg. Drop it so the
        # behaviour is explicit.
        self.ranged_line = self.ax.axvline(
            x=peak_loc,
            color=color,
            linestyle='dashdot',
            linewidth=2,
        )
        if self.fig is not None and self.fig.canvas is not None:
            self.fig.canvas.draw_idle()

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
            labels = _resolve_range_peak_labels(range_data)
            mc = range_data['mc'].tolist()
            for i in range(len(labels)):
                if self.y is None or len(self.y) == 0 or self.x is None or len(self.x) == 0:
                    continue
                # Find the bin that contains the mc[i]
                bin_index = np.searchsorted(self.x, mc[i]) - 1
                clamped_index = min(max(int(bin_index), 0), len(self.y) - 1)
                if 0 <= bin_index < len(self.y):
                    # Define a small range around the bin to search for the local maximum
                    search_range = slice(max(0, bin_index - 1), min(len(self.y), bin_index + 2))
                    local_bins = self.y[search_range]
                    local_x = self.x[search_range.start : search_range.stop]

                    # Find the local maximum and its position
                    max_idx = np.argmax(local_bins)
                    peak_height = local_bins[max_idx]
                    peak_position = local_x[max_idx]

                    # Dynamic y_offset based on log scale
                    y_offset = peak_height * 0.05
                    if self.ax.get_yscale() == 'log':
                        y_offset = 10 ** (np.log10(peak_height) + 0.1) - peak_height
                else:
                    peak_position = float(np.clip(mc[i], self.x[0], self.x[-1]))
                    peak_height = float(self.y[clamped_index])
                    y_offset = peak_height * 0.05
                    if self.ax.get_yscale() == 'log' and peak_height > 0:
                        y_offset = 10 ** (np.log10(peak_height) + 0.1) - peak_height
                if self.plot_show:
                    self.peak_annotates.append(
                        plt.text(
                            peak_position + x_offset,
                            peak_height + y_offset,
                            labels[i],
                            color='black',
                            size=10,
                            alpha=1,
                            rotation=90,
                        )
                    )
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
                            plt.text(
                                self.x[self.peaks][i] + x_offset,
                                peak_height + y_offset,
                                '%s' % '{:.2f}'.format(self.x[self.peaks][i]),
                                color='black',
                                size=10,
                                alpha=1,
                                rotation=90,
                            )
                        )

                        self.annotates.append(str(i + 1))

            elif mode == 'range':
                y_offset = 0.0  # Adjust this value as needed
                for i in range(len(self.variables.peaks_x_selected)):
                    # Find the bin that CONTAINS the selected mass. self.x are
                    # bin EDGES (len = len(self.y)+1), so use the same
                    # ``searchsorted - 1`` + clamp convention as plot_range /
                    # plot_peaks(mode='peaks') above. Without the -1 and the
                    # clamp, a selection at/beyond the last edge raised
                    # IndexError on self.y, and in-range selections read the
                    # count of the bin to the RIGHT of the one containing it.
                    sel = self.variables.peaks_x_selected[i]
                    bin_index = int(np.searchsorted(self.x, sel)) - 1
                    bin_index = min(max(bin_index, 0), len(self.y) - 1)
                    peak_height = self.y[bin_index] * (
                        (sel - self.x[bin_index]) / self.bin_width
                    )
                    if self.plot_show:
                        self.peak_annotates.append(
                            plt.text(
                                self.variables.peaks_x_selected[i] + x_offset,
                                peak_height + y_offset,
                                '%s' % '{:.2f}'.format(self.variables.peaks_x_selected[i]),
                                color='black',
                                size=10,
                                alpha=1,
                                rotation=90,
                            )
                        )

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
            pattern_1 = r"\^{\d+\}|\{\+|\{-\}|\{\d+[+-]?\}"
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
        self.ax.legend(
            [label[1] for label in legend_colors_edited], [label[0] for label in legend_colors_edited], loc=loc, ncol=ncol
        )

    def plot_hist_info_legend(self, label='mc', mrp_all=False, background=None, legend_mode='long', loc='left'):
        """Plot summary legend info for histogram quality metrics."""
        return _apply_hist_info_legend(
            self, label=label, mrp_all=mrp_all, background=background, legend_mode=legend_mode, loc=loc
        )

    def mrp_calculation(self):
        """Calculate MRP metrics for current histogram peaks."""
        return _calculate_mrp(self)

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

    def plot_background(self, mode, non_peaks=None, lam=1e6, tol=1e-1, max_iter=100, num_std=3.0, plot=True, patch=True):
        """Fit and plot histogram background."""
        return _plot_background(
            self, mode, non_peaks=non_peaks, lam=lam, tol=tol, max_iter=max_iter, num_std=num_std, plot=plot, patch=patch
        )

    def exponential_decay_with_linear_and_dc(self, x, a, b, c, d):
        """Exponential decay helper retained for compatibility."""
        return _exp_decay(x, a, b, c, d)

    def manual_background_fit(
        self,
    ):
        """Interactive manual background fitting."""
        return _manual_background_fit(self)

    def calculate_noise(self, fig_size=(9, 5), plot_without_noise=False):
        """Calculate noise after fitted background subtraction."""
        return _calculate_noise(self, fig_size=fig_size, plot_without_noise=plot_without_noise)

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
                label = ax1.annotate(
                    ion,
                    xy=(m, line_height),
                    xytext=(m, line_height),
                    fontsize=10,
                    color='blue',
                    annotation_clip='clip_on',
                    textcoords="offset points",
                    xycoords="data",
                )

                self.plotted_lines.append(line)  # Keep track of the plotted lines
                self.plotted_circles.append(circle)  # Keep track of the plotted circles
                self.plotted_labels.append(label)  # Keep track of the plotted labels
                # Remove the y-axis and labels
                ax1.get_yaxis().set_visible(False)
                # Set the y-axis to log scale
                ax1.set_yscale('log')

    def find_peaks_and_widths(self, prominence=None, distance=None, percent=50):
        """Find peaks and widths and update shared variables."""
        return _find_peaks_and_widths(self, prominence=prominence, distance=distance, percent=percent)

    def draw_rectangle(self, initial=False):
        """Draw auto-selected peak rectangle."""
        return _draw_rectangle(self, initial=initial)

    def selector(self, selector='rect'):
        """Attach interaction selector handlers."""
        return _attach_selector(self, selector=selector)

    def zoom_to_x_range(self, x_min, x_max, reset=False):
        """Zoom the histogram to a selected x-range or reset view."""
        return _zoom_to_x_range(self, x_min, x_max, reset=reset)

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

        if label == 'mc' or label == 'mc_c':
            save_figure(
                self.fig,
                directory=self.variables.result_path,
                stem=f"mc_{fig_name}",
                formats=("pdf", "png"),
                dpi=600,
            )
        elif label == 'tof' or label == 'tof_c':
            save_figure(
                self.fig,
                directory=self.variables.result_path,
                stem=f"tof_{fig_name}",
                formats=("pdf", "png"),
                dpi=600,
            )

def hist_plot(
    variables,
    bin_size,
    log,
    target,
    normalize,
    prominence,
    distance,
    percent,
    selector,
    figname,
    lim,
    peaks_find=True,
    peaks_find_plot=False,
    plot_ranged_peak=False,
    plot_ranged_colors=False,
    mrp_all=False,
    background=None,
    grid=False,
    ranging_mode=False,
    range_sequence=[],
    range_mc=[],
    range_detx=[],
    range_dety=[],
    range_x=[],
    range_y=[],
    range_z=[],
    range_vol=[],
    save_fig=True,
    print_info=True,
    legend_mode='long',
    draw_calib_rect=False,
    figure_size=(9, 5),
    plot_show=True,
    fast_calibration=False,
    fast_histogram=True,
    initial_peak_selection=False,
    compute_mrp=True,
):
    """Backward-compatible wrapper delegating to :mod:`mc_plot_api`."""
    from pyccapt.calibration.core.mc_plot_api import hist_plot as _hist_plot

    return _hist_plot(
        variables,
        bin_size,
        log,
        target,
        normalize,
        prominence,
        distance,
        percent,
        selector,
        figname,
        lim,
        peaks_find=peaks_find,
        peaks_find_plot=peaks_find_plot,
        plot_ranged_peak=plot_ranged_peak,
        plot_ranged_colors=plot_ranged_colors,
        mrp_all=mrp_all,
        background=background,
        grid=grid,
        ranging_mode=ranging_mode,
        range_sequence=range_sequence,
        range_mc=range_mc,
        range_detx=range_detx,
        range_dety=range_dety,
        range_x=range_x,
        range_y=range_y,
        range_z=range_z,
        range_vol=range_vol,
        save_fig=save_fig,
        print_info=print_info,
        legend_mode=legend_mode,
        draw_calib_rect=draw_calib_rect,
        figure_size=figure_size,
        plot_show=plot_show,
        fast_calibration=fast_calibration,
        fast_histogram=fast_histogram,
        initial_peak_selection=initial_peak_selection,
        compute_mrp=compute_mrp,
    )
