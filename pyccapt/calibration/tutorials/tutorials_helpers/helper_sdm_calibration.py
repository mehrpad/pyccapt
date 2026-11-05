"""Interactive notebook GUI for SDM-based ICF/kf reconstruction calibration.

Public entry point: :func:`call_sdm_icf_kf_calibration`. Builds a two-tab
ipywidgets panel on top of the widget-free algorithms in
:mod:`pyccapt.calibration.reconstructions.sdm_calibration`:

* **Manual (spreadsheet-style)** -- show the normal + "best"/sharp FDM, pick a
  pole ROI, then iterate: reconstruct with the current (kf, icf), rotate the
  pole to +z, view the z-SDM, read off the peak spacing, compare to the
  theoretical d-spacing (from the lattice + Miller index) and apply the
  suggested kf. The ROI is a detector disk, so it stays constant across
  iterations.
* **Ratio suggestion** -- near-axis q = ICF/kf profile at an independently
  indexed spacing, validated across acquisition bands. Applying one compatible
  family member requires an explicitly chosen ICF anchor.

For the z-SDM to show clean peaks the structure is first rotated so the chosen
pole faces +z (planes become perpendicular to z).
"""

from __future__ import annotations

import numpy as np
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import Output

from pyccapt.calibration.core import plot_style
from pyccapt.calibration.reconstructions import sdm_calibration as sc
from pyccapt.calibration.reconstructions import ratio_calibration as rc
from pyccapt.calibration.core.exceptions import CalibrationInputError, CalibrationStateError
from pyccapt.calibration.tutorials.tutorials_helpers.guided_jobs import NotebookJob

_label = widgets.Layout(width='190px')
_narrow = widgets.Layout(width='90px')


def call_sdm_icf_kf_calibration(variables, flight_path_length, element_selected, colab=False, *,
                              display_panel=True, on_parameters=None, parameter_widgets=None):
    import matplotlib.pyplot as plt

    existing = getattr(variables, '_reconstruction_panel', None)
    if display_panel and existing is not None:
        existing.reconstruction_tabs.selected_index = 1
        print('Optional ICF/kf calibration is in the second tab of the reconstruction panel above.')
        return None

    avg_dens0 = element_selected.value[1]
    field_evap0 = element_selected.value[2]
    saved = getattr(variables, 'reconstruction_parameters', {})
    shared = parameter_widgets or {}
    ratio_job = NotebookJob()

    # ---- shared reconstruction parameters --------------------------------
    kf_w = shared['kf'] if 'kf' in shared else widgets.FloatText(value=saved.get('kf', 3.3), step=0.05, layout=_narrow)
    icf_w = shared['icf'] if 'icf' in shared else widgets.FloatText(value=saved.get('icf', 1.65), step=0.05, layout=_narrow)
    det_eff_w = shared['det_eff'] if 'det_eff' in shared else widgets.FloatText(value=saved.get('det_eff', 0.7), step=0.05, layout=_narrow)
    field_evap_w = shared['field_evap'] if 'field_evap' in shared else widgets.FloatText(value=saved.get('field_evap', field_evap0), layout=_narrow)
    avg_dens_w = shared['avg_dens'] if 'avg_dens' in shared else widgets.FloatText(value=saved.get('avg_dens', avg_dens0), layout=_narrow)
    mode_w = shared['mode'] if 'mode' in shared else widgets.Dropdown(options=[('Geiser', 'Geiser'), ('Bas', 'Bas')], value=saved.get('mode', saved.get('projection', 'Geiser')), layout=_narrow)

    def _parameters():
        return dict(kf=kf_w.value, icf=icf_w.value, det_eff=det_eff_w.value,
                    field_evap=field_evap_w.value, avg_dens=avg_dens_w.value,
                    mode=mode_w.value, flight_path_length=flight_path_length.value)

    def _save_parameters():
        if ratio_job.running:
            raise CalibrationStateError('Wait for or cancel the ratio preview before changing accepted parameters')
        variables.reconstruction_parameters = _parameters()
        variables.reconstruction_ratio_result = {}
        if on_parameters:
            on_parameters(dict(variables.reconstruction_parameters))
        print('Parameters transferred to the reconstruction tab.')

    # ---- SDM / FDM parameters --------------------------------------------
    bin_w = widgets.FloatText(value=0.01, step=0.005, layout=_narrow)
    zmax_w = widgets.FloatText(value=2.0, step=0.5, layout=_narrow)
    lat_w = widgets.FloatText(value=1.5, step=0.25, layout=_narrow)
    maxatoms_w = widgets.IntText(value=20000, layout=_narrow)
    fdm_bins_w = widgets.IntText(value=200, layout=_narrow)
    fdm_smooth_w = widgets.FloatText(value=1.4, step=0.2, layout=_narrow)

    # ---- lattice (for theoretical d-spacing) -----------------------------
    a_w = widgets.FloatText(value=0.405, step=0.001, layout=_narrow)
    b_w = widgets.FloatText(value=0.405, step=0.001, layout=_narrow)
    c_w = widgets.FloatText(value=0.405, step=0.001, layout=_narrow)
    al_w = widgets.FloatText(value=90.0, layout=_narrow)
    be_w = widgets.FloatText(value=90.0, layout=_narrow)
    ga_w = widgets.FloatText(value=90.0, layout=_narrow)
    h_w = widgets.IntText(value=0, layout=_narrow)
    k_w = widgets.IntText(value=0, layout=_narrow)
    l_w = widgets.IntText(value=2, layout=_narrow)

    # ---- ROI (detector disk, cm) -- constant across iterations -----------
    cx_w = widgets.FloatText(value=0.0, step=0.05, layout=_narrow)
    cy_w = widgets.FloatText(value=0.0, step=0.05, layout=_narrow)
    r_w = widgets.FloatText(value=0.05, step=0.01, layout=_narrow)
    pole_dd = widgets.Dropdown(options=[('(show FDMs first)', None)], layout=widgets.Layout(width='280px'))

    d_obs_w = widgets.FloatText(value=0.0, step=0.001, layout=_narrow)
    theo_lbl = widgets.HTML(value='Theoretical d: —')
    suggest_lbl = widgets.HTML(value='Suggested kf: —')

    out_fdm = Output()
    out_sdm = Output()
    out_hist = Output()
    out_auto = Output()
    out_status = Output()

    history = []

    def _theo_d():
        return sc.d_spacing((h_w.value, k_w.value, l_w.value),
                            a_w.value, b_w.value, c_w.value,
                            al_w.value, be_w.value, ga_w.value)

    def _ratio_inputs():
        fixed = {key: value for key, value in _parameters().items() if key not in ('kf', 'icf')}
        return (fixed, cx_w.value, cy_w.value, r_w.value, _theo_d(),
                q_lo_w.value, q_hi_w.value, q_n_w.value, icf_lo_w.value, icf_hi_w.value,
                full_source_w.value, spacing_confirmed_w.value, id(getattr(variables, 'dataset', None)),
                rc.selection_hash(variables))

    # ---- show normal + sharp FDM, detect pole candidates -----------------
    def on_show_fdm(_):
        with out_status:
            out_status.clear_output()
            print('Building FDMs...')
        detx = np.asarray(variables.dld_x_det, dtype=float)
        dety = np.asarray(variables.dld_y_det, dtype=float)
        active = sc.active_selection_mask(variables)
        if active is not None:
            detx, dety = detx[active], dety[active]
        density, contrast, xe, ye = sc.sharp_fdm_map(
            detx, dety, bins=int(fdm_bins_w.value), smooth_sigma=fdm_smooth_w.value)
        cands = sc.detect_pole_candidates(contrast, xe, ye, n=8)
        opts = [('(manual cx/cy below)', None)]
        for i, (px, py, s) in enumerate(cands):
            opts.append((f'unindexed ROI {i + 1}: ({px:.2f}, {py:.2f}) cm  contrast {s:.1f}', (px, py)))
        pole_dd.options = opts
        pole_dd.value = None  # Density contrast does not identify a crystallographic pole.
        ext = [xe[0], xe[-1], ye[0], ye[-1]]
        with out_fdm:
            out_fdm.clear_output(wait=True)
            fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
            d = density.T.copy()
            d[d <= 0] = np.nan
            ax[0].imshow(np.log10(d), origin='lower', extent=ext, cmap='viridis', aspect='equal')
            ax[0].set_title('Normal FDM (log hit density)')
            ax[1].imshow(contrast.T, origin='lower', extent=ext, cmap='magma', aspect='equal')
            ax[1].set_title('Detrended FDM (unindexed density features)')
            for a in ax:
                a.set_xlabel('det x (cm)'); a.set_ylabel('det y (cm)')
            for i, (px, py, s) in enumerate(cands):
                ax[1].plot(px, py, 'c+', ms=10, mew=2)
                ax[1].annotate(str(i + 1), (px, py), color='cyan', fontsize=9)
            # draw current ROI
            th = np.linspace(0, 2 * np.pi, 80)
            ax[1].plot(cx_w.value + r_w.value * np.cos(th),
                       cy_w.value + r_w.value * np.sin(th), 'w-', lw=1.2)
            plt.show()
        with out_status:
            out_status.clear_output()
            print(f'Found {len(cands)} unindexed density feature(s). Current ROI has not changed. '
                  'Review a feature or set cx/cy/radius, then inspect its z-SDM. '
                  'Neither a density feature nor detector center identifies a plane; confirm the spacing independently.')

    def on_pole_select(change):
        if change['new'] is not None:
            cx_w.value, cy_w.value = round(change['new'][0], 3), round(change['new'][1], 3)

    pole_dd.observe(on_pole_select, names='value')

    # ---- reconstruct current (kf,icf) -> rotate pole->z -> z-SDM ---------
    def _measure(plot=True):
        cx, cy, r = cx_w.value, cy_w.value, r_w.value
        mask = sc.detector_roi_mask(variables, cx, cy, r)
        n_roi = int(mask.sum())
        if n_roi < 50:
            with out_status:
                out_status.clear_output()
                print(f'ROI has only {n_roi} ions — widen the radius or move the centre.')
            return None
        x, y, z = sc.reconstruct_xyz(variables, kf_w.value, icf_w.value, det_eff_w.value,
                                     field_evap_w.value, avg_dens_w.value,
                                     flight_path_length.value, mode=mode_w.value)
        pole_dir = sc.pole_axis_from_detector(cx, cy, flight_path_length.value, icf_w.value, mode=mode_w.value)
        xr, yr, zr, _ = sc.align_pole_to_z(x[mask], y[mask], z[mask], pole_dir)
        centers, counts = sc.z_sdm(xr, yr, zr, bin_size=bin_w.value, z_max=zmax_w.value,
                                   lateral_radius=lat_w.value, max_atoms=int(maxatoms_w.value))
        pk = sc.find_sdm_peaks(centers, counts, min_spacing=0.05)
        strength = sc.sdm_peakiness(centers, counts)
        if plot:
            with out_sdm:
                out_sdm.clear_output(wait=True)
                fig, ax = plt.subplots(figsize=(9, 3.6), constrained_layout=True)
                ax.plot(centers, counts, '-', lw=1)
                for p in pk['peak_positions']:
                    ax.axvline(p, color='r', ls='--', lw=0.8)
                ax.set_xlabel('Δz (nm)'); ax.set_ylabel('pair counts')
                ax.set_title(f'z-SDM  ROI={n_roi} ions  kf={kf_w.value:.3f} icf={icf_w.value:.3f}  '
                             f'spacing={pk["spacing"]:.3f} nm  peakiness={strength:.1f}')
                plot_style.finalize_axes(ax)  # round-number axes with end ticks
                plt.show()
        return {'spacing': pk['spacing'], 'strength': strength, 'n_roi': n_roi,
                'n_peaks': pk['n_peaks']}

    def on_reconstruct_sdm(_):
        res = _measure(plot=True)
        if res is None:
            return
        d_obs_w.value = round(res['spacing'], 4) if np.isfinite(res['spacing']) else 0.0
        theo = _theo_d()
        theo_lbl.value = f'Theoretical d ({h_w.value}{k_w.value}{l_w.value}): <b>{theo:.4f} nm</b>'
        sug = sc.corrected_kf(kf_w.value, theo, res['spacing'])
        suggest_lbl.value = (f'Suggested kf = kf·√(d_theo/d_obs) = <b>{sug:.4f}</b>'
                             if np.isfinite(sug) else 'Suggested kf: —')
        history.append(dict(it=len(history) + 1, kf=kf_w.value, icf=icf_w.value,
                            d_obs=res['spacing'], d_theo=theo, strength=res['strength'],
                            suggest_kf=sug))
        _render_history()

    def _render_history():
        with out_hist:
            out_hist.clear_output()
            print(f'{"it":>2} {"kf":>7} {"icf":>6} {"d_obs":>7} {"d_theo":>7} '
                  f'{"ratio":>6} {"peaky":>7} {"kf*":>7}')
            for row in history:
                ratio = (row['d_theo'] / row['d_obs']) if row['d_obs'] else float('nan')
                print(f'{row["it"]:>2} {row["kf"]:>7.3f} {row["icf"]:>6.3f} '
                      f'{row["d_obs"]:>7.3f} {row["d_theo"]:>7.3f} {ratio:>6.3f} '
                      f'{row["strength"]:>7.1f} {row["suggest_kf"]:>7.3f}')

    def on_apply_kf(_):
        if ratio_job.running:
            return
        if history and np.isfinite(history[-1]['suggest_kf']):
            kf_w.value = round(history[-1]['suggest_kf'], 4)
            with out_status:
                out_status.clear_output()
                print(f'kf set to {kf_w.value}. Adjust icf if needed and re-run "Reconstruct + z-SDM".')
                _save_parameters()

    # ---- identifiable near-axis ratio, with an explicitly anchored family ---
    q0 = icf_w.value / kf_w.value
    q_lo_w = widgets.FloatText(value=.75*q0, layout=_narrow)
    q_hi_w = widgets.FloatText(value=1.25*q0, layout=_narrow)
    q_n_w = widgets.IntText(value=201, layout=_narrow)
    icf_lo_w = widgets.FloatText(value=1.1, step=.05, layout=_narrow)
    icf_hi_w = widgets.FloatText(value=1.9, step=.05, layout=_narrow)
    full_source_w = widgets.Checkbox(value=False, description='Full acquisition loaded from event zero', indent=False)
    spacing_confirmed_w = widgets.Checkbox(value=False, description='Indexed spacing independently known', indent=False)
    anchor_confirmed_w = widgets.Checkbox(value=False, description='ICF above is my chosen independent/manual anchor', indent=False)
    icf_w.observe(lambda _: setattr(anchor_confirmed_w, 'value', False), names='value')
    def reference_changed(_):
        spacing_confirmed_w.value = False
    for reference in (cx_w, cy_w, r_w, a_w, b_w, c_w, al_w, be_w, ga_w, h_w, k_w, l_w):
        reference.observe(reference_changed, names='value')
    auto_best = {}
    ratio_progress = widgets.FloatProgress(min=0, max=1, description='Ratio profile:')
    ratio_progress_text = widgets.Label()

    def _profile_error(error):
        auto_best.clear()
        ratio_progress.bar_style = 'warning'
        with out_status:
            out_status.clear_output()
            print(f'Ratio unavailable: {error}. Manual reconstruction remains available.')

    def _display_profile(res, inputs, source_hash):
        if inputs != _ratio_inputs() or source_hash != rc._hash_source(variables):
            raise CalibrationStateError('Inputs or acquisition arrays changed while profiling; stale result discarded')
        if res['status'] == 'supported':
            if res['metadata']['source_hash'] != source_hash:
                raise CalibrationStateError('Acquisition arrays changed during profiling; stale result discarded')
            auto_best.update(result=res, inputs=inputs)
        with out_status:
            out_status.clear_output()
            print(f"{res['status']}: {res['message']}")
            band_evidence = res.get('diagnostics', {}).get('band_evidence', {})
            if band_evidence:
                print('Band diagnostics (scanned maxima are exploratory, not accepted estimates):')
                for band in band_evidence['bands']:
                    print(f"  {band['role']} band {band['band']+1}: n={band['events']}, "
                          f"coherence at fixed training choice={band['coherence_at_training_choice']:.3f}, "
                          f"best scanned={band['exploratory_peak_coherence']:.3f}, gate={band['coherence_floor']:.3f}")
            for step in rc.ratio_next_steps(res):
                print(f'Next: {step}')
            if res['q'] is not None:
                print(f"q = ICF/kf = {res['q']:.5f}; descriptive profile region {res['supported_interval']}.")
                print('Compatible family: kf = ICF/q. The ratio does not determine a unique ICF or kf.')
                for row in rc.compatible_parameter_family(res, np.linspace(*res['metadata']['icf_bounds'], 5)):
                    print(f"ICF {row['icf']:.3f} -> kf {row['kf']:.3f}; profile range {row['kf_profile_interval']}")
        with out_auto:
            out_auto.clear_output(wait=True)
            if res.get('training_profile'):
                fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
                ax.plot(res['q_grid'], res['training_profile'], label='training acquisition bands')
                ax.plot(res['q_grid'], res['heldout_profile'], label='held-out acquisition bands')
                if res['supported_interval'] is not None:
                    ax.axvspan(*res['supported_interval'], alpha=.15, label='descriptive support region')
                ax.set_xlabel('q = ICF/kf (dimensionless)')
                ax.set_ylabel('absolute indexed-period coherence')
                floors = res.get('diagnostics', {}).get('coherence_floor')
                if floors:
                    ax.axhline(min(floors), color='gray', ls=':', label='smallest individual-band evidence gate')
                ax.legend()
                plt.show()
        ratio_progress.value = 1.
        ratio_progress.bar_style = 'success' if res['status'] == 'supported' else 'warning'
        ratio_progress_text.value = 'Preview complete; accepted parameters have not changed.'

    def on_auto_run(_):
        if ratio_job.running:
            return
        auto_best.clear()
        try:
            if not spacing_confirmed_w.value:
                raise CalibrationInputError('Confirm the lattice and indexed plane spacing; defaults are not evidence')
            inputs = _ratio_inputs()  # Snapshot widget values before starting the worker.
            positional = ((cx_w.value, cy_w.value), r_w.value, _theo_d(),
                          np.linspace(q_lo_w.value, q_hi_w.value, int(q_n_w.value)))
            options = dict(det_eff=det_eff_w.value, field_evap_v_nm=field_evap_w.value,
                atomic_density_nm3=avg_dens_w.value, flight_path_mm=flight_path_length.value,
                full_acquisition_confirmed=full_source_w.value,
                icf_bounds=(icf_lo_w.value, icf_hi_w.value), mode=mode_w.value)
        except (CalibrationInputError, CalibrationStateError, ValueError) as error:
            _profile_error(error)
            return
        previous_disabled = [control.disabled for control in profile_controls]
        for control in profile_controls:
            control.disabled = True
        btn_cancel.disabled = False
        ratio_progress.value, ratio_progress.bar_style = 0., ''
        ratio_progress_text.value = 'Preparing the full acquisition depth prefix...'

        def operation(cancel, report):
            if cancel.is_set():
                raise CalibrationStateError('Ratio profiling cancelled')
            source_hash = rc._hash_source(variables)
            def progress(done, total):
                if cancel.is_set():
                    raise CalibrationStateError('Ratio profiling cancelled')
                if done == total or done % max(1, total//100) == 0:
                    report(done/total, f'Profile point {done}/{total}')
            result = rc.profile_icf_kf_ratio(variables, *positional, **options, progress=progress)
            return result, inputs, source_hash

        def progress(fraction, text):
            ratio_progress.value, ratio_progress_text.value = fraction, text

        def finished():
            for control, disabled in zip(profile_controls, previous_disabled):
                control.disabled = disabled
            btn_cancel.disabled = True

        try:
            ratio_job.start(operation, on_result=lambda payload: _display_profile(*payload),
                            on_error=_profile_error, on_progress=progress, on_finished=finished)
        except Exception as error:
            finished()
            _profile_error(error)

    def on_auto_apply(_):
        if ratio_job.running:
            return
        with out_status:
            out_status.clear_output()
            if not auto_best:
                print('No supported ratio to apply. Profile first or use manual reconstruction.')
                return
            if auto_best['inputs'] != _ratio_inputs():
                print('Geometry, spacing, source or profile inputs changed. Run the ratio profile again.')
                return
            if not anchor_confirmed_w.value:
                print('Confirm the current ICF as your independent/manual anchor. The profile has not estimated it.')
                return
            try:
                parameters = rc.accept_ratio_family(variables, auto_best['result'], icf_anchor=icf_w.value)
                kf_w.value = parameters['kf']
                print(f"Accepted q={auto_best['result']['q']:.5f} with chosen ICF={icf_w.value}; kf={kf_w.value:.5f}.")
                if on_parameters:
                    on_parameters(dict(variables.reconstruction_parameters))
                from pyccapt.calibration.core.calibration_metadata import record_step
                record_step(variables, 'accept_icf_kf_family', parameters=parameters,
                            model=auto_best['result'])
                print('Accepted parameters transferred to the reconstruction tab.')
            except (CalibrationInputError, CalibrationStateError) as error:
                print(str(error))

    # ---- buttons ---------------------------------------------------------
    btn_fdm = widgets.Button(description='Show FDMs', button_style='primary')
    btn_sdm = widgets.Button(description='Reconstruct + z-SDM', button_style='info')
    btn_apply_kf = widgets.Button(description='Apply suggested kf')
    btn_auto = widgets.Button(description='Profile ICF/kf ratio', button_style='info')
    btn_auto_apply = widgets.Button(description='Accept anchored family', layout=widgets.Layout(width='190px'))
    btn_cancel = widgets.Button(description='Cancel ratio profile', disabled=True)
    profile_controls = [btn_fdm, btn_sdm, btn_apply_kf, btn_auto, btn_auto_apply,
        kf_w, icf_w, det_eff_w, field_evap_w, avg_dens_w, mode_w,
        cx_w, cy_w, r_w, pole_dd, a_w, b_w, c_w, al_w, be_w, ga_w, h_w, k_w, l_w,
        q_lo_w, q_hi_w, q_n_w, icf_lo_w, icf_hi_w, full_source_w, spacing_confirmed_w, anchor_confirmed_w]
    btn_fdm.on_click(on_show_fdm)
    btn_sdm.on_click(on_reconstruct_sdm)
    btn_apply_kf.on_click(on_apply_kf)
    btn_auto.on_click(on_auto_run)
    btn_auto_apply.on_click(on_auto_apply)
    btn_cancel.on_click(lambda _: ratio_job.cancel())

    def _row(lbl, *w):
        return widgets.HBox([widgets.Label(lbl, layout=_label), *w])

    recon_box = widgets.VBox([
        widgets.HTML('<b>Shared reconstruction parameters</b>' if shared else '<b>Reconstruction parameters</b>'),
        _row('kf / icf:', kf_w, icf_w),
        _row('det_eff:', det_eff_w),
        _row('field_evap / avg_dens:', field_evap_w, avg_dens_w),
        _row('mode:', mode_w),
    ])
    sdm_box = widgets.VBox([
        widgets.HTML('<b>SDM / FDM settings</b>'),
        _row('z-SDM bin / z_max (nm):', bin_w, zmax_w),
        _row('lateral radius (nm):', lat_w),
        _row('max atoms (SDM):', maxatoms_w),
        _row('FDM bins / smooth:', fdm_bins_w, fdm_smooth_w),
    ])
    lattice_box = widgets.VBox([
        widgets.HTML('<b>Lattice + pole (theoretical d)</b>'),
        _row('a / b / c (nm):', a_w, b_w, c_w),
        _row('α / β / γ (deg):', al_w, be_w, ga_w),
        _row('Miller h / k / l:', h_w, k_w, l_w),
    ])
    roi_box = widgets.VBox([
        widgets.HTML('<b>Review a detector ROI (cm)</b><br>Density features are unindexed candidates. '
                     'Selecting one requires independent spacing confirmation. Crops select events; the full depth prefix is retained.'),
        btn_fdm,
        _row('review candidate:', pole_dd),
        _row('cx / cy / radius (cm):', cx_w, cy_w, r_w),
        out_fdm,
    ])

    manual_tab = widgets.VBox([
        sdm_box,
        widgets.HBox([btn_sdm, btn_apply_kf]),
        out_sdm,
        widgets.HBox([widgets.Label('Measured d_obs (nm):', layout=_label), d_obs_w, theo_lbl]),
        suggest_lbl,
        widgets.HTML('<b>Iteration history</b>'),
        out_hist,
    ])

    auto_tab = widgets.VBox([
        widgets.HTML('<b>Near-axis ratio profile: q = ICF/kf</b><br>'
                     'Uses indexed plane spacing and held-out acquisition bands. '
                     'Off-axis, weak, boundary or ambiguous evidence abstains. '
                     'The compatible family requires a separately chosen ICF anchor.'),
        _row('q lo / hi / grid points:', q_lo_w, q_hi_w, q_n_w),
        _row('Allowed ICF family:', icf_lo_w, icf_hi_w),
        full_source_w, spacing_confirmed_w, anchor_confirmed_w,
        widgets.HBox([btn_auto, btn_auto_apply, btn_cancel]), ratio_progress, ratio_progress_text, out_auto,
    ])
    tabs = widgets.Tab(children=[auto_tab, manual_tab])
    tabs.set_title(0, 'ICF/kf ratio suggestion')
    tabs.set_title(1, 'Manual SDM tuning')
    panel = widgets.VBox([widgets.HBox([recon_box, lattice_box]), roi_box, tabs,
                          widgets.HTML('<b>Status</b>'), out_status])
    panel.parameter_widgets = dict(kf=kf_w, icf=icf_w, det_eff=det_eff_w,
                                  field_evap=field_evap_w, avg_dens=avg_dens_w, mode=mode_w)
    if display_panel:
        display(panel)
        return None
    return panel
