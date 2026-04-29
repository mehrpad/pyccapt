"""Pure helpers extracted from :mod:`helper_calibration`.

These were previously top-level public utilities or inner closures of
``call_voltage_bowl_calibration``. They are now module-level so the host file
can stay under the calibration module-length policy. They are intentionally
free of widget / variables-state captures so they can be unit-tested in
isolation.
"""

from __future__ import annotations

import ipywidgets as widgets
import numpy as np

label_layout = widgets.Layout(width='300px')


def reset_on_click(variables, calibration_mode):
    """Restore the active calibration array to the dataframe's untouched values."""
    if calibration_mode.value == 'tof_calib':
        variables.dld_t_calib = variables.data['t (ns)'].to_numpy()
    elif calibration_mode.value == 'mc_calib':
        variables.mc_calib = variables.data['mc_uc (Da)'].to_numpy()
    variables.clear_calibration_selection_mask()
    variables.clear_calibration_peak_range()


def reset_back_on_click(variables, calibration_mode):
    """Restore the active calibration array from the most recent backup."""
    if calibration_mode.value == 'tof_calib':
        variables.dld_t_calib = np.copy(variables.dld_t_calib_backup)
    elif calibration_mode.value == 'mc_calib':
        variables.mc_calib = np.copy(variables.mc_calib_backup)
    variables.clear_calibration_selection_mask()
    variables.clear_calibration_peak_range()


def save_on_click(variables, calibration_mode):
    """Snapshot the current calibration array into the backup slot."""
    if calibration_mode.value == 'tof_calib':
        variables.dld_t_calib_backup = np.copy(variables.dld_t_calib)
    elif calibration_mode.value == 'mc_calib':
        variables.mc_calib_backup = np.copy(variables.mc_calib)


def clear_plot_on_click(out, out_status):
    """Clear both diagnostic outputs and any inline display."""
    from IPython.display import clear_output

    with out:
        out.clear_output()
    with out_status:
        out_status.clear_output()
    clear_output(wait=True)


def peaks_overlap(first_peak, second_peak):
    """Return True iff the two peak windows overlap on the x axis."""
    return not (first_peak['x2'] <= second_peak['x1'] or second_peak['x2'] <= first_peak['x1'])


def score_improved(candidate, best):
    """Decide whether ``candidate`` is meaningfully better than ``best``."""
    if not np.isfinite(candidate):
        return False
    if not np.isfinite(best):
        return True
    return candidate > best + max(0.1, abs(best) * 0.002)


def score_not_worse(candidate, best, tolerance_ratio=0.01):
    """Return True iff ``candidate`` is within tolerance of ``best`` or better."""
    if not np.isfinite(best):
        return True
    if not np.isfinite(candidate):
        return False
    return candidate >= best - max(0.1, abs(best) * tolerance_ratio)
