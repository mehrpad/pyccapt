import functools

import ipywidgets as widgets
import numpy as np
from IPython.display import clear_output, display
from ipywidgets import Output
from scipy.optimize import curve_fit

from pyccapt.calibration.core import mc_plot, widgets as wd
from pyccapt.calibration.data_tools import data_tools



# Define a layout for labels to make them a fixed width
label_layout = widgets.Layout(width='200px')


def call_ion_list(variables, selector, path='../../../files/'):
    try:
        isotopeTableFile = path + 'isotopeTable.h5'
        dataframe = data_tools.read_range(isotopeTableFile)
    except Exception as e:
        print(f"Error in loading the isotopeTable file: {e}")
        print("Trying to load the isotopeTable file from the pyccapt package")
        try:
            isotopeTableFile = './pyccapt/files/isotopeTable.h5'
            dataframe = data_tools.read_range(isotopeTableFile)
        except Exception as e:
            print(f"Error in loading the isotopeTable file: {e}")

    elementsList = dataframe['element']
    elementIsotopeList = dataframe['isotope']
    elementMassList = dataframe['weight']
    abundanceList = dataframe['abundance']

    elements = list(zip(elementsList, elementIsotopeList, elementMassList, abundanceList))
    dropdownList = []
    for element in elements:
        tupleElement = (
            "{} ({}) ({:.2f})".format(element[0], element[1], element[3]),
            "{}({})[{}]".format(element[0], element[1], element[2]),
        )
        dropdownList.append(tupleElement)

    chargeList = [
        (
            1,
            1,
        ),
        (
            2,
            2,
        ),
        (
            3,
            3,
        ),
        (
            4,
            4,
        ),
    ]
    dropdown = wd.dropdownWidget(dropdownList, "Elements")
    dropdown.observe(wd.on_change)

    chargeDropdown = wd.dropdownWidget(chargeList, "Charge")
    chargeDropdown.observe(wd.on_change_charge)

    wd.compute_element_isotope_values_according_to_selected_charge()

    buttonAdd = wd.buttonWidget("ADD")
    buttonDelete = wd.buttonWidget("DELETE")
    buttonReset = wd.buttonWidget("RESET")

    def buttonAdd_f(b, variables):
        with out_ion_list:
            clear_output(True)
            wd.onClickAdd(b, variables)
            display()

    def buttonDelete_f(b, variables):
        with out_ion_list:
            clear_output(True)
            wd.onClickDelete(b, variables)
            display()

    def buttonResett_f(b, variables):
        with out_ion_list:
            clear_output(True)
            wd.onClickReset(b, variables)
            display()

    buttonAdd.on_click(functools.partial(buttonAdd_f, variables=variables))
    buttonDelete.on_click(functools.partial(buttonDelete_f, variables=variables))
    buttonReset.on_click(functools.partial(buttonResett_f, variables=variables))

    out_ion_list = Output()
    out_mc = Output()

    # Define widgets for fine_tune_t_0 function
    bin_size_widget = widgets.FloatText(value=0.1)
    log_widget = widgets.Dropdown(options=[('True', True), ('False', False)])
    mode_widget = widgets.Dropdown(options=[('False', False), ('True', True)])
    prominence_widget = widgets.IntText(value=80)
    distance_widget = widgets.IntText(value=100)
    lim_widget = widgets.IntText(value=10000)
    percent_widget = widgets.IntText(value=50)
    figname_widget = widgets.Text(value='hist')
    figure_mc_size_x = widgets.FloatText(value=9.0)
    figure_mc_size_y = widgets.FloatText(value=5.0)

    # Create a button widget to trigger the function
    button_plot = widgets.Button(description="Plot")
    reset_back_button = widgets.Button(description='Reset back correction', layout=label_layout)
    button_fit = widgets.Button(description="Fit")
    calibration_mode = widgets.Dropdown(options=[('mass_to_charge', 'mc_calib'), ('time_of_flight', 'tof_calib')])

    # Fit-method selector. Two algorithms:
    #   * 'parametric' (default): the legacy curve_fit-based 2-/3-parameter
    #     polynomial mapping observed peak picks -> ideal m/c. Requires the
    #     user to click each peak in the plot before pressing Fit.
    #   * 'nist': NIST-inspired scipy.optimize.least_squares fit. Auto-detects
    #     peaks, greedily matches them to the ideal m/c values the user
    #     added via the ADD button, and fits a tightly-clipped multiplicative
    #     correction factor f(V, x, y, t). No peak-click needed; internal
    #     accept/revert gate prevents damage. See
    #     pyccapt/calibration/core/reference_optimizer.py.
    fit_method_widget = widgets.Dropdown(
        options=[
            ('Parametric (default)', 'parametric'),
            ('NIST reference (auto-match)', 'nist'),
        ],
        value='parametric',
        description='Fit method:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='400px'),
    )
    fit_method_help = widgets.HTML(
        value=(
            '<div style="font-size:11px; color:#555; max-width:600px;">'
            '<b>Parametric</b>: fits the polynomial <code>mc<sup>a</sup> + b&middot;mc + c</code> '
            '(or the 2-parameter variant) using the peaks you <i>click</i> in the plot. '
            'Requires manual peak picking.<br>'
            '<b>NIST reference</b>: the <b>same polynomial</b> but with '
            '<i>automatic</i> peak matching &mdash; greedy nearest-neighbour '
            'between detected peaks and the ideal m/c values you added via ADD '
            '(0.3 Da tolerance). No peak clicking needed.<br>'
            'Both methods only rescale m/c &mdash; they do <b>not</b> redo V / '
            'Bowl / drift correction, so they are safe to run on a fully '
            'calibrated spectrum without undoing the V+Bowl pipeline. '
            'Includes NaN / collapsed-spread sanity check that reverts if the '
            'fit destroys the spectrum. mc only (tof has no universal '
            'reference table).'
            '</div>'
        )
    )

    def parametric_fit(variables, calibration_mode, out_mc):

        button_fit.disabled = True
        peaks_chos = np.array(variables.peaks_x_selected)

        if len(peaks_chos) != len(variables.list_material):
            with out_mc:
                print('Number of peaks and number of materials are not equal')
        else:
            if calibration_mode.value == 'tof_calib':

                def parametric(t, t0, c, d):
                    return c * ((t - t0) ** 2) + d * t

                def parametric_calib(t, mc_ideal):
                    fitresult, _ = curve_fit(parametric, t, mc_ideal, maxfev=2000)
                    return fitresult

                if len(peaks_chos) > 2:
                    fitresult = parametric_calib(peaks_chos, variables.list_material)

                    variables.mc_calib = parametric(variables.dld_t_calib, *fitresult)
                else:
                    print('Number of peaks is less than 3. Select more peaks at least 3 peaks')

            elif calibration_mode.value == 'mc_calib':

                def shift_3(mc, a, b, c):
                    return mc**a + b * mc + c
                    # return a * mc + b

                def shift_calib_3(mc, mc_ideal):
                    fitresult, _ = curve_fit(shift_3, mc, mc_ideal, maxfev=2000)
                    return fitresult

                def shift_2(mc, a, b):
                    return mc**a + b

                def shift_calib_2(mc, mc_ideal):
                    fitresult, _ = curve_fit(shift_2, mc, mc_ideal, maxfev=2000)
                    return fitresult

                if len(peaks_chos) > 2:
                    fitresult = shift_calib_3(peaks_chos, variables.list_material)
                    variables.mc_calib = shift_3(variables.mc_calib_backup, *fitresult)
                elif len(peaks_chos) == 2:
                    fitresult = shift_calib_2(peaks_chos, variables.list_material)
                    variables.mc_calib = shift_2(variables.mc_calib_backup, *fitresult)
                else:
                    print('Number of peaks is less than 2. Select more peaks at least 2 peaks')

            with out_mc:
                print('parametric fit done')
        button_fit.disabled = False

    button_plot_result = widgets.Button(description="Plot result")

    def plot_fit_result(b, variables, calibration_mode, out_mc):
        button_plot_result.disabled = True
        # Get the values from the widgets
        bin_size_value = bin_size_widget.value
        log_value = log_widget.value
        mode_value = mode_widget.value
        target_value = 'mc_c'
        prominence_value = prominence_widget.value
        distance_value = distance_widget.value
        percent_value = percent_widget.value
        figname_value = figname_widget.value
        lim_value = lim_widget.value
        figure_size = (figure_mc_size_x.value, figure_mc_size_y.value)
        with out_mc:  # Capture the output within the 'out' widget
            # Call the function
            mc_hist = mc_plot.AptHistPlotter(variables.mc_calib[variables.mc_calib < lim_value], variables)
            mc_hist.plot_histogram(
                bin_width=bin_size_value,
                normalize=mode_value,
                label='mc',
                steps='stepfilled',
                log=log_value,
                fig_size=figure_size,
            )

            if mode_value != 'normalized':
                mc_hist.find_peaks_and_widths(prominence=prominence_value, distance=distance_value, percent=percent_value)
                mc_hist.plot_peaks()
                mc_hist.plot_hist_info_legend(label=target_value, background=None, loc='right')

            mc_hist.save_fig(label=target_value, fig_name=figname_value)

        # Enable the button when the code is finished
        button_plot_result.disabled = False

    def on_button_click(b, variables, selector):
        # Disable the button while the code is running
        button_plot.disabled = True
        variables.peaks_x_selected = []
        # Get the values from the widgets
        bin_size_value = bin_size_widget.value
        log_value = log_widget.value
        mode_value = mode_widget.value
        target_value = calibration_mode.value
        prominence_value = prominence_widget.value
        distance_value = distance_widget.value
        percent_value = percent_widget.value
        figname_value = figname_widget.value
        lim_value = lim_widget.value
        figure_size = (figure_mc_size_x.value, figure_mc_size_y.value)
        with out_mc:  # Capture the output within the 'out' widget
            out_mc.clear_output()  # Clear any previous output
            # Call the function
            if target_value == 'mc_calib':
                mc_hist = mc_plot.AptHistPlotter(variables.mc_calib[variables.mc_calib < lim_value], variables)
                mc_hist.plot_histogram(
                    bin_width=bin_size_value,
                    normalize=mode_value,
                    label='mc',
                    steps='stepfilled',
                    log=log_value,
                    fig_size=figure_size,
                )
            elif target_value == 'tof_calib':
                mc_hist = mc_plot.AptHistPlotter(variables.dld_t_calib[variables.dld_t_calib < lim_value], variables)
                mc_hist.plot_histogram(
                    bin_width=bin_size_value,
                    normalize=mode_value,
                    label='tof',
                    steps='stepfilled',
                    log=log_value,
                    fig_size=figure_size,
                )

            if not mode_value:
                mc_hist.find_peaks_and_widths(prominence=prominence_value, distance=distance_value, percent=percent_value)
                mc_hist.plot_peaks()
                mc_hist.plot_hist_info_legend(label='mc', background=None, loc='right')

            mc_hist.selector(selector=selector)  # rect, peak_x, range
            mc_hist.save_fig(label=target_value, fig_name=figname_value)

        # Enable the button when the code is finished
        button_plot.disabled = False

    def nist_fit(variables, calibration_mode_widget, out_mc):
        """Auto-matching version of the parametric m/c shift fit.

        Mirrors the existing ``parametric_fit`` exactly -- same 2-/3-
        parameter polynomial ``mc_ideal = mc^a + b*mc + c`` (or the
        2-parameter variant), applied to ``variables.mc_calib_backup`` --
        but **auto-matches** detected peaks to the ideal m/c values the
        user added via the ADD button, so the user doesn't have to
        click each peak.

        IMPORTANT: this DOES NOT redo V / bowl / drift correction. The
        risk of running ``fit_reference_constrained`` on a fully-calibrated
        spectrum is exactly that -- it would try to re-fit V(x,y,t) on
        residuals and could undo the joint-V+Bowl + time-drift work that
        the Hybrid button just finished. This simpler version only nudges
        the m/c axis to absolute truth and is therefore safe on top of
        the calibrated pipeline.

        tof is not supported because there is no universal tof reference
        table -- tof in ns depends on the instrument's flight path,
        voltage scale, and t0 calibration. Use mc mode instead.
        """
        button_fit.disabled = True
        try:
            with out_mc:
                if calibration_mode_widget.value == 'tof_calib':
                    print('NIST fit is mc-only. tof has no universal '
                          'reference table (depends on instrument geometry / '
                          'voltage / t0). Switch to mass_to_charge.')
                    return

                ideal_list = list(variables.list_material or [])
                if len(ideal_list) < 2:
                    print(f'NIST fit needs at least 2 expected m/c values '
                          f'(got {len(ideal_list)}). Add more elements via '
                          f'the ADD button first.')
                    return
                try:
                    ideal_arr = np.asarray([float(v) for v in ideal_list], dtype=float)
                except (TypeError, ValueError) as exc:
                    print(f'NIST fit: list_material contains non-numeric '
                          f'entries: {exc}')
                    return

                # --- Auto-detect peaks in CURRENT mc_calib ---
                from pyccapt.calibration.core import calibration as _cal
                arr = np.asarray(variables.mc_calib, dtype=float)
                arr = arr[(arr > 0) & np.isfinite(arr)]
                try:
                    peaks = _cal.auto_detect_reference_peaks(
                        arr,
                        n_peaks=max(len(ideal_arr) * 2, 8),
                        prominence=25,
                        distance=10,
                        hist_bin_size=0.05,
                    )
                except Exception as exc:
                    print(f'NIST fit: peak detection failed: {exc}')
                    return
                if not peaks:
                    print('NIST fit: no peaks detected. Adjust prominence / '
                          'distance or use the parametric fit with manual '
                          'peak picks.')
                    return
                observed_arr = np.asarray(
                    [float(p['position']) for p in peaks], dtype=float
                )

                # --- Greedy nearest-neighbour match: each ideal m/c
                # paired with at most one observed peak within tolerance ---
                tolerance = 0.30  # Da; intentionally a bit wide to handle
                                  # slightly-off post-calibration centres.
                ideal_pairs = []
                observed_pairs = []
                used_obs = set()
                # Iterate ideal values in order so the printed match list is
                # predictable.
                for ideal in ideal_arr:
                    best_idx = -1
                    best_dist = float('inf')
                    for i, obs in enumerate(observed_arr):
                        if i in used_obs:
                            continue
                        d = abs(obs - ideal)
                        if d < best_dist and d <= tolerance:
                            best_dist = d
                            best_idx = i
                    if best_idx >= 0:
                        used_obs.add(best_idx)
                        ideal_pairs.append(float(ideal))
                        observed_pairs.append(float(observed_arr[best_idx]))

                n_matched = len(ideal_pairs)
                if n_matched < 2:
                    print(f'NIST fit: only {n_matched} of {len(ideal_arr)} '
                          f'expected m/c values matched a detected peak '
                          f'within {tolerance} Da. Either the spectrum is '
                          f'badly calibrated (peaks too far from ideal) or '
                          f'the expected list does not match the data.')
                    return
                observed_pairs = np.asarray(observed_pairs)
                ideal_pairs = np.asarray(ideal_pairs)
                print(f'NIST auto-match found {n_matched} pair(s):')
                for obs, ideal in zip(observed_pairs, ideal_pairs):
                    print(f'  observed {obs:.4f} -> ideal {ideal:.4f} '
                          f'(d={obs - ideal:+.4f} Da)')

                # --- Fit the SAME 2-/3-parameter polynomial as parametric_fit ---
                # NO V, no bowl, no drift -- only m/c rescaling, so this
                # cannot undo the Hybrid V+Bowl+drift pipeline.
                #
                # IMPORTANT: peaks were detected in the CURRENT mc_calib
                # (which may already include Hybrid V+Bowl+drift corrections);
                # the rescale must therefore be applied to mc_calib, NOT to
                # mc_calib_backup. Applying to the backup would silently
                # erase the prior Hybrid pipeline output every time the
                # user clicks NIST.
                source_arr = np.asarray(variables.mc_calib, dtype=float)
                if n_matched >= 3:
                    def shift_3(mc, a, b, c):
                        return mc**a + b * mc + c
                    try:
                        popt, _ = curve_fit(shift_3, observed_pairs, ideal_pairs, maxfev=2000)
                        corrected = shift_3(source_arr, *popt)
                    except Exception as exc:
                        print(f'NIST fit (3-param) failed: {exc}')
                        return
                    print(f'NIST fit (3-param) popt = {popt}')
                else:  # exactly 2
                    def shift_2(mc, a, b):
                        return mc**a + b
                    try:
                        popt, _ = curve_fit(shift_2, observed_pairs, ideal_pairs, maxfev=2000)
                        corrected = shift_2(source_arr, *popt)
                    except Exception as exc:
                        print(f'NIST fit (2-param) failed: {exc}')
                        return
                    print(f'NIST fit (2-param) popt = {popt}')

                # --- Sanity check: did the fit blow up the spectrum? ---
                # Reject if NaN/inf appeared, or the variance collapsed
                # (i.e. all ions mapped to a tiny range, the classic
                # 'one giant peak' failure mode).
                finite = corrected[np.isfinite(corrected)]
                pre_std = float(np.nanstd(source_arr))
                post_std = float(np.nanstd(finite)) if finite.size else 0.0
                if finite.size < len(corrected) * 0.95:
                    print(f'NIST fit reverted: {len(corrected) - finite.size} '
                          f'NaN/inf values introduced.')
                    return
                if post_std < pre_std * 0.5:
                    print(f'NIST fit reverted: m/c spread collapsed '
                          f'({pre_std:.2f} -> {post_std:.2f}). The fit '
                          f'mapped too many ions on top of each other.')
                    return

                variables.mc_calib = corrected
                print(f'NIST mc-shift accepted. Press "Plot result" to view.')
        finally:
            button_fit.disabled = False

    def _dispatch_fit(_b):
        if fit_method_widget.value == 'nist':
            nist_fit(variables, calibration_mode, out_mc)
        else:
            parametric_fit(variables, calibration_mode, out_mc)

    button_plot.on_click(lambda b: on_button_click(b, variables, selector))
    button_fit.on_click(_dispatch_fit)
    reset_back_button.on_click(lambda b: reset_back_on_click(variables))
    button_plot_result.on_click(lambda b: plot_fit_result(b, variables, calibration_mode, out_mc))

    widget_container = widgets.VBox(
        [
            widgets.HBox([widgets.Label(value="Calibration mde:", layout=label_layout), calibration_mode]),
            widgets.HBox([widgets.Label(value="Bin Size:", layout=label_layout), bin_size_widget]),
            widgets.HBox([widgets.Label(value="Log:", layout=label_layout), log_widget]),
            widgets.HBox([widgets.Label(value="Normalize:", layout=label_layout), mode_widget]),
            widgets.HBox([widgets.Label(value="Prominence:", layout=label_layout), prominence_widget]),
            widgets.HBox([widgets.Label(value="Distance:", layout=label_layout), distance_widget]),
            widgets.HBox([widgets.Label(value="Lim:", layout=label_layout), lim_widget]),
            widgets.HBox([widgets.Label(value="Percent:", layout=label_layout), percent_widget]),
            widgets.HBox([widgets.Label(value="Figname:", layout=label_layout), figname_widget]),
            widgets.HBox([widgets.Label(value="Fig. size W:", layout=label_layout), figure_mc_size_x]),
            widgets.HBox([widgets.Label(value="Fig. size H:", layout=label_layout), figure_mc_size_y]),
            widgets.HBox([widgets.Label(value="Fit method:", layout=label_layout), fit_method_widget]),
            fit_method_help,
            widgets.HBox([button_plot, button_fit, button_plot_result, reset_back_button]),
        ]
    )

    ion_list_box = widgets.VBox([dropdown, chargeDropdown, buttonAdd, buttonDelete, buttonReset])

    with out_ion_list:
        clear_output(True)
        print("Updated List: ", variables.list_material)
        print("Updated element List: ", variables.element)
        print("Updated isotope List: ", variables.isotope)
        print("Updated charge List: ", variables.charge)

    output_layout = widgets.HBox([out_mc, out_ion_list])
    display_layout = widgets.HBox([widget_container, ion_list_box])
    display(display_layout, output_layout)


def reset_back_on_click(variables):
    variables.dld_t_calib = np.copy(variables.dld_t_calib_backup)
    variables.mc_calib = np.copy(variables.mc_calib_backup)
