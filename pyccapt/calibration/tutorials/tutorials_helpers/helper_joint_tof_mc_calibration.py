"""
Jupyter helper for the Joint ToF + m/c Iterative Calibration.

Provides :func:`call_joint_tof_mc_calibration` which creates an
interactive ipywidgets panel with a push button that runs the
*Physics-Constrained Iterative Co-Calibration* algorithm directly
from Jupyter notebook data-processing workflows.
"""

from __future__ import annotations

import time

import ipywidgets as widgets
import numpy as np
from IPython.display import clear_output, display
from ipywidgets import Output

from pyccapt.calibration.core.joint_tof_mc_calibration import (
    joint_tof_mc_calibration,
)

_LABEL_LAYOUT = widgets.Layout(width='300px')
_WIDE_LAYOUT = widgets.Layout(width='620px')


def call_joint_tof_mc_calibration(variables, det_diam, flight_path_length, pulse_mode, t0=0.0):
    """
    Create and display a Jupyter widget panel for Joint ToF + m/c Calibration.

    Parameters
    ----------
    variables : Variables
        Shared calibration state (must already have data loaded).
    det_diam : float
        Detector diameter in mm.
    flight_path_length : float
        Nominal flight-path length in mm.
    pulse_mode : str
        ``'voltage'`` or ``'laser'``.
    t0 : float
        Time-zero offset in ns (default 0).
    """
    out = Output()
    out_status = Output()

    # ---- widgets -----------------------------------------------------------
    run_button = widgets.Button(
        description='Run Joint ToF+m/c Calibration',
        layout=_WIDE_LAYOUT,
        button_style='success',
        tooltip='Run the Physics-Constrained Iterative Co-Calibration',
        icon='play',
    )
    reset_button = widgets.Button(
        description='Reset to uncalibrated',
        layout=_LABEL_LAYOUT,
        tooltip='Restore original uncalibrated arrays',
    )
    save_button = widgets.Button(
        description='Save calibration',
        layout=_LABEL_LAYOUT,
        tooltip='Save current calibration as backup',
    )
    clear_button = widgets.Button(
        description='Clear output',
        layout=_LABEL_LAYOUT,
    )

    n_peaks_w = widgets.IntText(value=6, description='Max peaks:', layout=_LABEL_LAYOUT)
    prominence_w = widgets.IntText(value=100, description='Prominence:', layout=_LABEL_LAYOUT)
    distance_w = widgets.IntText(value=500, description='Peak distance:', layout=_LABEL_LAYOUT)
    bin_mc_w = widgets.FloatText(value=0.1, description='Bin size m/c:', layout=_LABEL_LAYOUT)
    bin_tof_w = widgets.FloatText(value=1.0, description='Bin size ToF:', layout=_LABEL_LAYOUT)
    max_iter_w = widgets.IntText(value=10, description='Max iterations:', layout=_LABEL_LAYOUT)
    conv_tol_w = widgets.FloatText(value=1e-4, description='Conv. tol:', layout=_LABEL_LAYOUT)
    tof_weight_w = widgets.FloatSlider(
        value=0.7, min=0.0, max=1.0, step=0.05,
        description='ToF weight:',
        layout=_LABEL_LAYOUT,
        readout_format='.2f',
    )
    mc_weight_w = widgets.FloatSlider(
        value=0.3, min=0.0, max=1.0, step=0.05,
        description='m/c weight:',
        layout=_LABEL_LAYOUT,
        readout_format='.2f',
    )
    t0_w = widgets.FloatText(value=float(t0), description='t0 (ns):', layout=_LABEL_LAYOUT)
    verbose_w = widgets.Dropdown(
        options=[('True', True), ('False', False)],
        description='Verbose:',
        layout=_LABEL_LAYOUT,
    )

    status_html = widgets.HTML(value='<b>Ready</b>', layout=_LABEL_LAYOUT)

    # Link ToF/m/c weight sliders (guard against circular updates)
    _syncing = False

    def _sync_mc_weight(change):
        nonlocal _syncing
        if _syncing:
            return
        _syncing = True
        mc_weight_w.value = round(1.0 - change['new'], 2)
        _syncing = False

    def _sync_tof_weight(change):
        nonlocal _syncing
        if _syncing:
            return
        _syncing = True
        tof_weight_w.value = round(1.0 - change['new'], 2)
        _syncing = False

    tof_weight_w.observe(_sync_mc_weight, names='value')
    mc_weight_w.observe(_sync_tof_weight, names='value')

    # ---- callbacks ---------------------------------------------------------
    def _on_run(_):
        run_button.disabled = True
        status_html.value = '<b style="color:orange;">Running...</b>'
        with out_status:
            out_status.clear_output()
            print('=' * 60)
            print('JOINT ToF + m/c ITERATIVE CALIBRATION')
            print('=' * 60)
            try:
                start = time.perf_counter()
                result = joint_tof_mc_calibration(
                    variables,
                    flight_path_length=flight_path_length,
                    t0=t0_w.value,
                    det_diam=det_diam,
                    pulse_mode=pulse_mode,
                    n_peaks=n_peaks_w.value,
                    prominence=prominence_w.value,
                    distance=distance_w.value,
                    bin_size_mc=bin_mc_w.value,
                    bin_size_tof=bin_tof_w.value,
                    max_iterations=max_iter_w.value,
                    convergence_tol=conv_tol_w.value,
                    tof_weight=tof_weight_w.value,
                    mc_weight=mc_weight_w.value,
                    verbose=verbose_w.value,
                )
                elapsed = time.perf_counter() - start
                print()
                print('--- Results ---')
                print(f'Matched peaks: {result["n_matched_peaks"]}')
                print(f'Iterations: {result["n_iterations"]}')
                print(f'Final loss: {result["final_loss"]:.6f}')
                print(f'Runtime: {elapsed:.2f} s')
                print()
                print('Peak summary:')
                for pk in result['matched_peaks']:
                    print(
                        f'  m/c={pk["mc_position"]:.2f} Da  '
                        f'ToF={pk["tof_position"]:.1f} ns  '
                        f'ions={pk["n_ions"]:,}'
                    )
                print()
                print('Loss history:', [f'{v:.6f}' for v in result['loss_history']])
                print('=' * 60)
                print('Joint ToF+m/c calibration complete!')
                status_html.value = '<b style="color:green;">Done ✓</b>'
            except Exception as exc:
                print(f'Calibration failed: {exc}')
                print('=' * 60)
                status_html.value = f'<b style="color:red;">Error: {exc}</b>'
        run_button.disabled = False

    def _on_reset(_):
        with out_status:
            out_status.clear_output()
            try:
                if variables.data is None:
                    raise RuntimeError("No dataset loaded in variables.data")
                for col in ('t (ns)', 'mc_uc (Da)'):
                    if col not in variables.data.columns:
                        raise KeyError(f"Expected column '{col}' not found in dataset")
                variables.dld_t_calib = variables.data['t (ns)'].to_numpy()
                variables.mc_calib = variables.data['mc_uc (Da)'].to_numpy()
                print('Reset to uncalibrated arrays.')
                status_html.value = '<b>Reset</b>'
            except Exception as exc:
                print(f'Reset failed: {exc}')

    def _on_save(_):
        with out_status:
            out_status.clear_output()
            variables.dld_t_calib_backup = np.copy(variables.dld_t_calib)
            variables.mc_calib_backup = np.copy(variables.mc_calib)
            print('Calibration saved as backup.')
            status_html.value = '<b style="color:blue;">Saved</b>'

    def _on_clear(_):
        with out:
            out.clear_output()
        with out_status:
            out_status.clear_output()
        status_html.value = '<b>Ready</b>'

    run_button.on_click(_on_run)
    reset_button.on_click(_on_reset)
    save_button.on_click(_on_save)
    clear_button.on_click(_on_clear)

    # ---- layout ------------------------------------------------------------
    col_params = widgets.VBox([
        n_peaks_w,
        prominence_w,
        distance_w,
        bin_mc_w,
        bin_tof_w,
    ])
    col_optim = widgets.VBox([
        max_iter_w,
        conv_tol_w,
        tof_weight_w,
        mc_weight_w,
        t0_w,
        verbose_w,
    ])
    col_actions = widgets.VBox([
        save_button,
        reset_button,
        clear_button,
        status_html,
    ])

    params_row = widgets.HBox([col_params, col_optim, col_actions])
    button_row = widgets.HBox([run_button])
    layout = widgets.VBox([params_row, button_row])

    display(layout)
    display(widgets.VBox([out, out_status]))
