import matplotlib.pyplot as plt
import numpy as np
import pybaselines
import re
from adjustText import adjust_text

from pybaselines import Baseline
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths

from pyccapt.calibration.core import interactive_point_identification
from pyccapt.calibration.core.background import fit_background
from pyccapt.calibration.core.exceptions import CalibrationInputError
from pyccapt.calibration.core.validation import ensure_positive
from pyccapt.calibration.data_tools import data_loadcrop, plot_vline_draw, selectors_data
from pyccapt.calibration.path_utils import save_figure

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

def hist_plot(
    mc_tof,
    variables,
    bin,
    label,
    range_data=None,
    adjust_label=False,
    ranging=False,
    hist_color_range=False,
    log=True,
    mode='count',
    percent=50,
    peaks_find=True,
    peaks_find_plot=False,
    plot=False,
    prominence=50,
    distance=None,
    h_line=False,
    selector='None',
    fast_hist=True,
    fig_name=None,
    text_loc='right',
    fig_size=(9, 5),
    background={'calculation': False},
):
    """
    Generate a histogram plot with optional peak_x finding and background calculation.

    Args:
        mc_tof (array-like): Input array of time-of-flight values.
        bin (float): Bin width for the histogram.
        label (str): Label type ('mc' or 'tof').
        range_data (optional, array-like): Range data.
        adjust_label (bool): Flag to adjust overlapping peak_x labels.
        ranging (bool): Flag to enable ranging.
        hist_color_range (bool): Flag to enable histogram color ranging.
        log (bool): Flag to enable logarithmic y-axis scale.
        mode (str): Mode for histogram calculation ('count' or 'normalised').
        percent (int): Percentage value for peak_x width calculation.
        peaks_find (bool): Flag to enable peak_x finding.
        peaks_find_plot (bool): Flag to plot peak_x finding results.
        plot (bool): Flag to enable plotting.
        prominence (float): Minimum prominence value for peak_x finding.
        distance (optional, float): Minimum horizontal distance between peaks for peak_x finding.
        h_line (bool): Flag to draw horizontal lines for peak_x width.
        selector (str): Selector mode for interactive selection ('None', 'rect', or 'peak_x').
        fast_hist (bool): Flag to enable fast histogram calculation.
        fig_name (optional, str): Name of the figure file to save.
        text_loc (str): Location of the text annotation ('left' or 'right').
        fig_size (tuple): Size of the figure.
        background (dict): Background calculation options.

    Returns:
        tuple: Tuple containing x_peaks, y_peaks, peaks_widths, and mask.

    Raises:
        ValueError: If an invalid mode or selector is provided.
    """

    mc_tof = np.asarray(mc_tof)
    if mc_tof.size == 0:
        raise CalibrationInputError("mc_tof cannot be empty")
    bin = ensure_positive(bin, field_name="bin")

    # Defensive NaN/inf handling. After ``merge_partial_tdc`` every row
    # should have a finite mc/tof (partials get a centred-axis estimate),
    # but a buggy upstream calibration step could still produce non-finite
    # values. ``np.min`` / ``np.max`` would propagate NaN and break the
    # histogram. Filter to the finite subset with a one-line notice.
    finite_mask = np.isfinite(mc_tof)
    if not finite_mask.all():
        n_dropped = int((~finite_mask).sum())
        print(f'[hist_plot] Skipping {n_dropped} non-finite mc/tof values.')
        mc_tof = mc_tof[finite_mask]
        if mc_tof.size == 0:
            raise CalibrationInputError("mc_tof has no finite values")

    bins = np.linspace(np.min(mc_tof), np.max(mc_tof), round(np.max(mc_tof) / bin))

    if fast_hist:
        steps = 'stepfilled'
    else:
        steps = 'bar'

    if mode == 'count':
        y, x = np.histogram(mc_tof, bins=bins)
        # y = np.log(y)
    elif mode == 'normalised':
        # counts/(Da * totalCts) so spectra with different total counts are
        # comparable. Normalise the COUNTS (y), not the data (x). The previous
        # code divided mc_tof itself by (bin*N) and then histogrammed it against
        # the raw-data-scale ``bins``, collapsing every value below bins[0] and
        # yielding an empty/meaningless spectrum.
        y, x = np.histogram(mc_tof, bins=bins)
        y = y.astype(float) / (bin * len(mc_tof))
    else:
        raise CalibrationInputError("Unsupported histogram mode: %r" % mode)

    try:
        if peaks_find:
            peaks, properties = find_peaks(y, prominence=prominence, distance=distance, height=0)
            index_peak_max = np.argmax(properties['peak_heights'])
            # find peak_x width
            peak_widths_p = peak_widths(y, peaks, rel_height=(percent / 100), prominence_data=None)
    except ValueError:
        print('Peak finding failed.')
        peaks_find = False

    if plot:
        fig1, ax1 = plt.subplots(figsize=fig_size)
        if ranging and hist_color_range:
            colors = _normalize_range_colors(range_data['color'].tolist())
            mc_low = range_data['mc_low'].tolist()
            mc_up = range_data['mc_up'].tolist()
            labels = _resolve_range_display_labels(range_data)
            mask_all = np.full(len(mc_tof), False)

            # Loop runs ``len(labels) + 1`` times: one pass per ranged
            # species, then a final pass that draws the leftover (unranged)
            # events as ``mc_tof[~mask_all]``. The previous code compared
            # ``i`` to a non-existent ``len(ion)`` and recomputed
            # ``mask_all`` from a stale ``mask`` — both bugs (the former
            # raised NameError; the latter was a no-op since every per-label
            # iteration already ORed its mask into ``mask_all``).
            for i in range(len(labels) + 1):
                if i < len(labels):
                    mask = np.logical_and((mc_tof < mc_up[i]), mc_tof > mc_low[i])
                    mask_all = np.logical_or(mask_all, mask)

                    name_element = 'unranged' if labels[i] == 'unranged' else labels[i]
                    y, x, _ = plt.hist(
                        mc_tof[mask],
                        bins=bins,
                        log=log,
                        histtype=steps,
                        color=colors[i],
                        label=name_element,
                    )
                else:
                    # Final iteration: i == len(labels). Plot the events not
                    # captured by any range as a slate-gray "unranged" baseline.
                    y, x, _ = plt.hist(
                        mc_tof[~mask_all],
                        bins=bins,
                        log=log,
                        histtype=steps,
                        color='slategray',
                    )
        else:
            y, x, _ = plt.hist(mc_tof, bins=bins, log=log, histtype=steps, color='slategray')
        # calculate the background
        if background['calculation']:
            if background['mode'] == 'aspls':
                baseline_fitter = Baseline(x_data=bins[:-1])
                fit_1, params_1 = baseline_fitter.aspls(y, lam=5e10, tol=1e-1, max_iter=100)

            if background['mode'] == 'fabc':
                fit_2, params_2 = pybaselines.classification.fabc(
                    y, lam=background['lam'], num_std=background['num_std'], pad_kwargs='edges'
                )
            if background['mode'] == 'dietrich':
                fit_2, params_2 = pybaselines.classification.dietrich(y, num_std=background['num_std'])
            if background['mode'] == 'cwt_br':
                fit_2, params_2 = pybaselines.classification.cwt_br(
                    y, poly_order=background['poly_order'], num_std=background['num_std'], tol=background['tol']
                )
            if background['mode'] == 'selective_mask_t':
                p = np.poly1d(np.polyfit(background['non_mask'][:, 0], background['non_mask'][:, 1], 5))
                baseline_handle = ax1.plot(x, p(x), '--')
            if background['mode'] == 'selective_mask_mc':
                fitresult, _ = curve_fit(fit_background, background['non_mask'][:, 0], background['non_mask'][:, 1])
                yy = fit_background(x, *fitresult)
                ax1.plot(x, yy, '--')

            if background['plot_no_back']:
                mask_2 = params_2['mask']
                mask_f = np.full((len(mc_tof)), False)
                for i in range(len(mask_2)):
                    if mask_2[i]:
                        step_loc = np.min(mc_tof) + bin * i
                        mask_t = np.logical_and((mc_tof < step_loc + bin), (mc_tof > step_loc))
                        mask_f = np.logical_or(mask_f, mask_t)
                background_ppm = (len(mask_f[mask_f == True]) * 1e6 / len(mask_f)) / np.max(mc_tof)

            if background['plot_no_back']:
                if background['plot']:
                    ax1.plot(bins[:-1], fit_2, label='class', color='r')
                    ax3 = ax1.twiny()
                    ax3.axis("off")
                    ax3.plot(fit_1, label='aspls', color='black')

                mask_2 = params_2['mask']
                if background['patch']:
                    ax1.plot(bins[:-1][mask_2], y[mask_2], 'o', color='orange')[0]
        if peaks_find:
            ax1.set_ylabel("Event Counts", fontsize=14)
            if label == 'mc':
                ax1.set_xlabel("Mass/Charge [Da]", fontsize=14)
            elif label == 'tof':
                ax1.set_xlabel("Time of Flight [ns]", fontsize=14)
            print("The peak_x index for MRP calculation is:", index_peak_max)
            if label == 'mc':
                mrp = '{:.2f}'.format(
                    x[peaks[index_peak_max]]
                    / (x[int(peak_widths_p[3][index_peak_max])] - x[int(peak_widths_p[2][index_peak_max])])
                )
                if background['calculation'] and background['plot_no_back']:
                    txt = 'bin width: %s Da\nnum atoms: %.2f$e^6$\nbackG: %s ppm/Da\nMRP(FWHM): %s' % (
                        bin,
                        len(mc_tof) / 1000000,
                        int(background_ppm),
                        mrp,
                    )
                else:
                    # annotation with range stats
                    upperLim = 4.5  # Da
                    lowerLim = 3.5  # Da
                    mask = np.logical_and((x >= lowerLim), (x <= upperLim))
                    BG4 = np.sum(y[np.array(mask[:-1])]) / (upperLim - lowerLim)
                    BG4 = BG4 / len(mc_tof) * 1e6

                    txt = 'bin width: %s Da\nnum atoms: %.2f$e^6$\nBG@4: %s ppm/Da\nMRP(FWHM): %s' % (
                        bin,
                        (len(mc_tof) / 1000000),
                        int(BG4),
                        mrp,
                    )

            elif label == 'tof':
                mrp = '{:.2f}'.format(
                    x[peaks[index_peak_max]]
                    / (x[int(peak_widths_p[3][index_peak_max])] - x[int(peak_widths_p[2][index_peak_max])])
                )
                if background['calculation'] and background['plot_no_back']:
                    txt = 'bin width: %s ns\nnum atoms: %.2f$e^6$\nbackG: %s ppm/ns\nMRP(FWHM): %s' % (
                        bin,
                        len(mc_tof) / 1000000,
                        int(background_ppm),
                        mrp,
                    )
                else:
                    # annotation with range stats
                    upperLim = 50.5  # ns
                    lowerLim = 49.5  # ns
                    mask = np.logical_and((x >= lowerLim), (x <= upperLim))
                    BG50 = np.sum(y[np.array(mask[:-1])]) / (upperLim - lowerLim)
                    BG50 = BG50 / len(mc_tof) * 1e6
                    txt = 'bin width: %s ns\nnum atoms: %.2f$e^6$ \nBG@50: %s ppm/ns\nMRP(FWHM): %s' % (
                        bin,
                        len(mc_tof) / 1000000,
                        int(BG50),
                        mrp,
                    )

            props = dict(boxstyle='round', facecolor='wheat', alpha=1)
            if text_loc == 'left':
                ax1.text(
                    0.01,
                    0.95,
                    txt,
                    va='top',
                    ma='left',
                    transform=ax1.transAxes,
                    bbox=props,
                    fontsize=10,
                    alpha=1,
                    horizontalalignment='left',
                    verticalalignment='top',
                )
            elif text_loc == 'right':
                ax1.text(
                    0.98,
                    0.95,
                    txt,
                    va='top',
                    ma='left',
                    transform=ax1.transAxes,
                    bbox=props,
                    fontsize=10,
                    alpha=1,
                    horizontalalignment='right',
                    verticalalignment='top',
                )

            ax1.tick_params(axis='both', which='major', labelsize=10)
            ax1.tick_params(axis='both', which='minor', labelsize=10)

            annotes = []
            texts = []
            if peaks_find_plot:
                if ranging:
                    ion = range_data['ion'].tolist()
                    x_peak_loc = range_data['mc'].tolist()
                    y_peak_loc = range_data['peak_count'].tolist()
                    for i in range(len(ion)):
                        texts.append(plt.text(x_peak_loc[i], y_peak_loc[i], r'%s' % ion[i], color='black', size=10, alpha=1))
                        annotes.append(str(i + 1))
                else:
                    for i in range(len(peaks)):
                        if selector == 'range':
                            if i in variables.peaks_x_selected:
                                texts.append(
                                    plt.text(
                                        x[peaks][i],
                                        y[peaks][i],
                                        '%s' % '{:.2f}'.format(x[peaks][i]),
                                        color='black',
                                        size=10,
                                        alpha=1,
                                    )
                                )
                        else:
                            texts.append(
                                plt.text(
                                    x[peaks][i],
                                    y[peaks][i],
                                    '%s' % '{:.2f}'.format(x[peaks][i]),
                                    color='black',
                                    size=10,
                                    alpha=1,
                                )
                            )

                        if h_line:
                            for i in range(len(variables.h_line_pos)):
                                if (
                                    variables.h_line_pos[i] < np.max(mc_tof) + 10
                                    and variables.h_line_pos[i] > np.max(mc_tof) - 10
                                ):
                                    plt.axvline(x=variables.h_line_pos[i], color='b', linestyle='--', linewidth=2)
                        annotes.append(str(i + 1))
            if adjust_label:
                adjust_text(texts, arrowprops=dict(arrowstyle='-', color='red', lw=0.5))
            if selector == 'rect':
                # Connect and initialize rectangle box selector
                data_loadcrop.rectangle_box_selector(ax1, variables)
                plt.connect('key_press_event', selectors_data.toggle_selector(variables))
            elif selector == 'peak_x':
                # connect peak_x selector
                af = interactive_point_identification.AnnoteFinder(x[peaks], y[peaks], annotes, variables, ax=ax1)
                fig1.canvas.mpl_connect('button_press_event', lambda event: af.annotates_plotter(event))
                zoom_manager = plot_vline_draw.HorizontalZoom(ax1, fig1)
                fig1.canvas.mpl_connect('key_press_event', lambda event: zoom_manager.on_key_press(event))
                fig1.canvas.mpl_connect('key_release_event', lambda event: zoom_manager.on_key_release(event))
                fig1.canvas.mpl_connect('scroll_event', lambda event: zoom_manager.on_scroll(event))
            elif selector == 'range':
                # connect range selector
                line_manager = plot_vline_draw.VerticalLineManager(variables, ax1, fig1, [], [])
                fig1.canvas.mpl_connect('button_press_event', lambda event: line_manager.on_press(event))
                fig1.canvas.mpl_connect('button_release_event', lambda event: line_manager.on_release(event))
                fig1.canvas.mpl_connect('motion_notify_event', lambda event: line_manager.on_motion(event))
                fig1.canvas.mpl_connect('key_press_event', lambda event: line_manager.on_key_press(event))
                fig1.canvas.mpl_connect('scroll_event', lambda event: line_manager.on_scroll(event))
                fig1.canvas.mpl_connect('key_release_event', lambda event: line_manager.on_key_release(event))

        else:
            if selector == 'range':
                # connect range selector
                line_manager = plot_vline_draw.VerticalLineManager(variables, ax1, fig1, [], [])
                texts = []
                for i in range(len(variables.peak_x)):
                    if i in variables.peaks_x_selected:
                        texts.append(
                            plt.text(
                                variables.peak_x[i],
                                variables.peak_y[i],
                                '%s' % '{:.2f}'.format(variables.peak_x[i]),
                                color='black',
                                size=10,
                                alpha=1,
                            )
                        )
        plt.tight_layout()
        if fig_name is not None:
            if label == 'mc':
                save_figure(fig1, directory=variables.result_path, stem=f"mc_{fig_name}", formats=("pdf", "png"), dpi=300)
            elif label == 'tof':
                save_figure(
                    fig1,
                    directory=variables.result_path,
                    stem=f"tof_{fig_name}",
                    formats=("pdf", "png"),
                    dpi=300,
                )
        if ranging and hist_color_range:
            plt.legend(loc='center right')

        plt.show()

    if peaks_find:
        peak_widths_f = []
        for i in range(len(peaks)):
            peak_widths_f.append([y[int(peak_widths_p[2][i])], x[int(peak_widths_p[2][i])], x[int(peak_widths_p[3][i])]])

        if background['calculation'] and background['plot_no_back']:
            x_peaks = x[peaks]
            y_peaks = y[peaks]
            peaks_widths = peak_widths_f
            mask = mask_f
        else:
            x_peaks = x[peaks]
            y_peaks = y[peaks]
            peaks_widths = peak_widths_f
            mask = None
        index_max_ini = np.argmax(y_peaks)
        variables.max_peak = x_peaks[index_max_ini]
        variables.peak_x = x_peaks
        variables.peak_y = y_peaks
    else:
        x_peaks = None
        y_peaks = None
        peaks_widths = None
        mask = None
    return x_peaks, y_peaks, peaks_widths, mask

def mc_hist_plot(variables, bin_size, mode, prominence, distance, percent, selector, plot, figname, lim, peaks_find_plot):
    """
    Plot the mass spectrum or tof spectrum. It is helper function for tutorials.
    Args:
        variables (object): Variables object.
        bin_size (float): Bin size for the histogram.
        mode (str): 'mc' for mass spectrum or 'tof' for tof spectrum.
        prominence (float): Prominence for the peak_x finding.
        distance (float): Distance for the peak_x finding.
        percent (float): Percent for the peak_x finding.
        selector (str): Selector for the peak_x finding.
        plot (bool): Plot the histogram.
        figname (str): Figure name.
        lim (float): Limit for the histogram.
        peaks_find_plot (bool): Plot the peaks.
    Returns:
        None

    """
    mode_map = {
        'mc': (variables.mc_calib, 'mc'),
        'mc_c': (variables.mc_uc, 'mc'),
        'tof': (variables.dld_t_calib, 'tof'),
        'tof_c': (variables.dld_t_c, 'tof'),
    }
    if mode not in mode_map:
        raise CalibrationInputError(f"Unsupported mode: {mode!r}")
    hist, label = mode_map[mode]
    if selector == 'peak_x':
        variables.peaks_x_selected = []
    peaks_ini, peaks_y_ini, peak_widths_p_ini, _ = hist_plot(
        hist[hist < lim],
        variables,
        bin_size,
        label=label,
        distance=distance,
        percent=percent,
        prominence=prominence,
        selector=selector,
        plot=plot,
        fig_name=figname,
        peaks_find_plot=peaks_find_plot,
    )
    if peaks_ini is not None:
        index_max_ini = np.argmax(peaks_y_ini)
        mrp = peaks_ini[index_max_ini] / (peak_widths_p_ini[index_max_ini][2] - peak_widths_p_ini[index_max_ini][1])
        print('Mass resolving power for the highest peak_x at peak_x index %a (MRP --> m/m_2-m_1):' % index_max_ini, mrp)
        for i in range(len(peaks_ini)):
            print(
                'Peaks ',
                i,
                'is at location and height: ({:.2f}, {:.2f})'.format(peaks_ini[i], peaks_y_ini[i]),
                'peak_x window sides ({:.1f}-maximum) are: ({:.2f}, {:.2f})'.format(
                    percent, peak_widths_p_ini[i][1], peak_widths_p_ini[i][2]
                ),
                '-> {:.2f}'.format(peak_widths_p_ini[i][2] - peak_widths_p_ini[i][1]),
            )
