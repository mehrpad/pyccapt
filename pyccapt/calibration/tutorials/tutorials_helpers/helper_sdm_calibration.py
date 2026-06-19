""Interactive notebook GUI for SDM-based ICF/kf reconstruction calibration.

Public entry point: :func:`call_sdm_icf_kf_calibration`. Builds a two-tab
ipywidgets panel on top of the widget-free algorithms in
:mod:`pyccapt.calibration.reconstructions.sdm_calibration`:

* **Manual (spreadsheet-style)** -- show the normal + "best"/sharp FDM, pick a
  pole ROI, then iterate: reconstruct with the current (kf, icf), rotate the
  pole to +z, view the z-SDM, read off the peak spacing, compare to the
  theoretical d-spacing (from the lattice + Miller index) and apply the
  suggested kf. The ROI is a detector disk, so it stays constant across
  iterations. ICF is calibrated separately from inter-pole angles (the
  Day & Breen / Gault 2009 angle method): theoretical interplanar angle over
  the angle the poles subtend on the detector, averaged over pole pairs.
* **Automatic** -- grid-sweep (kf, icf) maximising z-SDM peakiness in the same
  fixed ROI, show the strength heat-map, and apply the best pair.

Both tabs can render the resulting 3-D reconstruction and write the calibrated
(x, y, z) back onto ``variables`` so the downstream cells use it.

For the z-SDM to show clean peaks the structure is first rotated so the chosen
pole faces +z (planes become perpendicular to z).
"""

from __future__ import annotations

import numpy as np
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import Output

from pyccapt.calibration.reconstructions import reconstruction
from pyccapt.calibration.reconstructions import sdm_calibration as sc
from __future__ import annotations

import ipywidgets as widgets
import numpy as np
from IPython.display import display
from ipywidgets import Output

from pyccapt.calibration.reconstructions import reconstruction, sdm_calibration as sc

_label = widgets.Layout(width='190px')
_narrow = widgets.Layout(width='90px')
_tiny = widgets.Layout(width='60px')


def call_sdm_icf_kf_calibration(variables, flight_path_length, element_selected, colab=False):
	avg_dens0 = element_selected.value[1]
    field_evap0 = element_selected.value[2]

    # ---- shared reconstruction parameters --------------------------------
    KF0, ICF0 = 3.3, 1.65
    kf_w = widgets.FloatText(value=KF0, step=0.05, layout=_narrow)
    icf_w = widgets.FloatText(value=ICF0, step=0.05, layout=_narrow)
    det_eff_w = widgets.FloatText(value=0.7, step=0.05, layout=_narrow)
    field_evap_w = widgets.FloatText(value=field_evap0, layout=_narrow)
    avg_dens_w = widgets.FloatText(value=avg_dens0, layout=_narrow)
    mode_w = widgets.Dropdown(options=[('Geiser', 'Geiser'), ('Bas', 'Bas')], layout=_narrow)

    # ---- SDM / FDM parameters --------------------------------------------
    bin_w = widgets.FloatText(value=0.01, step=0.005, layout=_narrow)
    zmax_w = widgets.FloatText(value=2.0, step=0.5, layout=_narrow)
    lat_w = widgets.FloatText(value=1.5, step=0.25, layout=_narrow)
    maxatoms_w = widgets.IntText(value=20000, layout=_narrow)
    fdm_bins_w = widgets.IntText(value=200, layout=_narrow)
    fdm_smooth_w = widgets.FloatText(value=1.4, step=0.2, layout=_narrow)

    # ---- lattice (for theoretical d-spacing & inter-pole angles) ---------
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
    r_w = widgets.FloatText(value=0.30, step=0.05, layout=_narrow)
    pole_dd = widgets.Dropdown(options=[('(show FDMs first)', None)], layout=widgets.Layout(width='280px'))

    d_obs_w = widgets.FloatText(value=0.0, step=0.001, layout=_narrow)
    theo_lbl = widgets.HTML(value='Theoretical d: —')
    suggest_lbl = widgets.HTML(value='Suggested kf: —')

    # ---- ICF-from-angles pole table (detector mm + Miller index) ---------
    n_poles = 4
    pole_x = [widgets.FloatText(value=0.0, step=0.1, layout=_narrow) for _ in range(n_poles)]
    pole_y = [widgets.FloatText(value=0.0, step=0.1, layout=_narrow) for _ in range(n_poles)]
    pole_h = [widgets.IntText(value=0, layout=_tiny) for _ in range(n_poles)]
    pole_k = [widgets.IntText(value=0, layout=_tiny) for _ in range(n_poles)]
    pole_l = [widgets.IntText(value=0, layout=_tiny) for _ in range(n_poles)]
    icf_angle_lbl = widgets.HTML(value='ICF (angles): —')

    out_fdm = Output()
    out_sdm = Output()
    out_hist = Output()
    out_icf = Output()
    out_recon_m = Output()
    out_auto = Output()
    out_auto_recon = Output()
    out_status = Output()

    history = []
    last_cands = []  # FDM pole candidates (cm), for autofill
    manual_cal = {'kf': None, 'icf': None}
    auto_best = {}

    def _status(msg):
	    with out_status:
		    out_status.clear_output()
		    print(msg)

    def _theo_d():
        return sc.d_spacing((h_w.value, k_w.value, l_w.value),
                            a_w.value, b_w.value, c_w.value,
                            al_w.value, be_w.value, ga_w.value)

    # ---- shared: full reconstruction + plotly 3-D view -------------------
    def _reconstruct_full(kf, icf):
	    return sc.reconstruct_xyz(variables, kf, icf, det_eff_w.value,
	                              field_evap_w.value, avg_dens_w.value,
	                              flight_path_length.value, mode=mode_w.value)

    def _element_percentage_list():
	    """Per-range plot fraction, mirroring helper_3d_reconstruction."""
	    rd = getattr(variables, 'range_data', None)
	    if rd is None or rd.empty or rd['ion'].iloc[0] == 'unranged':
		    return [0.01]
	    known = {}
	    for element_list in rd['element']:
		    for element in element_list:
			    known.setdefault(element, 0.01)
	    epl = []
	    for row_elements in rd['element']:
		    value = 0.1
		    for element in row_elements:
			    if element in known:
				    value = known[element]
		    epl.append(value)
	    return epl

    def _show_recon_plotly(kf, icf, out, figname):
	    """Full plotly 3-D reconstruction (opens in a browser window), like the
		data_processing ``call_x_y_z_calculation`` cell. Also writes
		variables.x / y / z as a side effect of the reconstruction.
		"""
	    with out:
		    out.clear_output(wait=True)
		    reconstruction.x_y_z_calculation_and_plot(
			    variables=variables,
			    element_percentage=_element_percentage_list(),
			    kf=kf, det_eff=det_eff_w.value, icf=icf,
			    field_evap=field_evap_w.value, avg_dens=avg_dens_w.value,
			    flight_path_length=flight_path_length.value,
			    rotary_fig_save=False, mode=mode_w.value, opacity=0.5,
			    figname=figname, save=False, colab=colab, cluster_result=None,
		    )

    # ---- show normal + sharp FDM, detect pole candidates -----------------
    def on_show_fdm(_):
	    _status('Building FDMs...')
        detx = np.asarray(variables.dld_x_det, dtype=float)
        dety = np.asarray(variables.dld_y_det, dtype=float)
        density, contrast, xe, ye = sc.sharp_fdm_map(
            detx, dety, bins=int(fdm_bins_w.value), smooth_sigma=fdm_smooth_w.value)
        cands = sc.detect_pole_candidates(contrast, xe, ye, n=8)
	    last_cands.clear()
	    last_cands.extend(cands)
        opts = [('(manual cx/cy below)', None)]
        for i, (px, py, s) in enumerate(cands):
            opts.append((f'pole {i + 1}: ({px:.2f}, {py:.2f}) cm  {s:.1f}σ', (px, py)))
        pole_dd.options = opts
        if cands:
            cx_w.value, cy_w.value = round(cands[0][0], 3), round(cands[0][1], 3)
        ext = [xe[0], xe[-1], ye[0], ye[-1]]
        with out_fdm:
            out_fdm.clear_output(wait=True)
            fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
            d = density.T.copy()
            d[d <= 0] = np.nan
            ax[0].imshow(np.log10(d), origin='lower', extent=ext, cmap='viridis', aspect='equal')
            ax[0].set_title('Normal FDM (log hit density)')
            ax[1].imshow(contrast.T, origin='lower', extent=ext, cmap='magma', aspect='equal')
            ax[1].set_title('Best/sharp FDM (pole contrast, σ)')
            for a in ax:
                a.set_xlabel('det x (cm)'); a.set_ylabel('det y (cm)')
            for i, (px, py, s) in enumerate(cands):
                ax[1].plot(px, py, 'c+', ms=10, mew=2)
                ax[1].annotate(str(i + 1), (px, py), color='cyan', fontsize=9)
            th = np.linspace(0, 2 * np.pi, 80)
            ax[1].plot(cx_w.value + r_w.value * np.cos(th),
                       cy_w.value + r_w.value * np.sin(th), 'w-', lw=1.2)
            plt.show()
	    _status(f'Detected {len(cands)} pole candidate(s). Pick one for the z-SDM ROI, or use '
	            '"Fill X/Y from poles" + Miller indices for the angle-based ICF.')

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
	        _status(f'ROI has only {n_roi} ions — widen the radius or move the centre.')
            return None
        x, y, z = sc.reconstruct_xyz(variables, kf_w.value, icf_w.value, det_eff_w.value,
                                     field_evap_w.value, avg_dens_w.value,
                                     flight_path_length.value, mode=mode_w.value)
        pole_dir = sc.pole_axis_from_detector(cx, cy, flight_path_length.value, icf_w.value)
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
        if np.isfinite(sug):
	        manual_cal['kf'] = round(sug, 4)
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

    # ---- ICF from inter-pole angles (spreadsheet / Gault 2009) -----------
    def on_fill_poles(_):
	    if not last_cands:
		    _status('Show the FDMs first — then the detected pole positions fill the table.')
		    return
	    for i in range(n_poles):
		    if i < len(last_cands):
			    px, py, _s = last_cands[i]
			    pole_x[i].value = round(px * 10.0, 3)  # cm -> mm
			    pole_y[i].value = round(py * 10.0, 3)
	    _status('Filled detector X/Y (mm) from detected poles. Enter each pole\'s Miller index, '
	            'then "Compute ICF (angles)".')

    def on_compute_icf(_):
	    poles = []
	    for i in range(n_poles):
		    hkl = (pole_h[i].value, pole_k[i].value, pole_l[i].value)
		    if all(v == 0 for v in hkl):
			    continue
		    poles.append({'xy': (pole_x[i].value, pole_y[i].value), 'hkl': hkl})
	    if len(poles) < 2:
		    _status('Need at least two poles with a non-zero Miller index for the angle ICF.')
		    return
	    res = sc.icf_from_pole_angles(poles, a_w.value, b_w.value, c_w.value,
	                                  al_w.value, be_w.value, ga_w.value,
	                                  flight_path_length.value)
	    with out_icf:
		    out_icf.clear_output()
		    print(f'{"pair":>6} {"theo°":>8} {"obs°":>8} {"theo/obs":>9}')
		    for (i, j, theo, obs, ratio) in res['pairs']:
			    print(f'{i + 1}-{j + 1:<4} {theo:>8.3f} {obs:>8.3f} {ratio:>9.4f}')
		    print(f'{"avg ICF":>6} {"":>8} {"":>8} {res["icf"]:>9.4f}')
	    if np.isfinite(res['icf']):
		    icf_w.value = round(res['icf'], 4)
		    manual_cal['icf'] = round(res['icf'], 4)
		    icf_angle_lbl.value = f'ICF (angles) = <b>{res["icf"]:.4f}</b>  (applied)'
		    _status(f'ICF from {len(res["pairs"])} pole pair(s) = {res["icf"]:.4f} (set as icf).')
	    else:
		    icf_angle_lbl.value = 'ICF (angles): —'
		    _status('Could not compute ICF — check pole positions and indices.')

    # ---- apply / show recon / reset (manual) -----------------------------
    def on_apply_cal(_):
	    applied = []
	    if manual_cal.get('kf') and np.isfinite(manual_cal['kf']):
		    kf_w.value = manual_cal['kf'];
		    applied.append(f'kf={kf_w.value}')
	    if manual_cal.get('icf') and np.isfinite(manual_cal['icf']):
		    icf_w.value = manual_cal['icf'];
		    applied.append(f'icf={icf_w.value}')
	    if applied:
		    _status('Applied calibrated ' + ', '.join(applied) +
		            '. Use "Show recon (calibrated)" to view, or re-run the z-SDM to iterate.')
	    else:
		    _status('Nothing to apply yet — run "Reconstruct + z-SDM" (kf) and/or '
		            '"Compute ICF (angles)" (icf) first.')

    def on_show_recon_manual(_):
	    _status(f'Reconstructing all ions (plotly, opens in a browser window) with '
	            f'kf={kf_w.value}, icf={icf_w.value}...')
	    _show_recon_plotly(kf_w.value, icf_w.value, out_recon_m, '3d_calibration_manual')
	    _status('Opened the 3-D reconstruction in a browser window. variables.x/y/z updated '
	            f'(kf={kf_w.value}, icf={icf_w.value}).')

    def on_reset_manual(_):
	    kf_w.value, icf_w.value = KF0, ICF0
	    history.clear()
	    manual_cal.update({'kf': None, 'icf': None})
	    d_obs_w.value = 0.0
	    theo_lbl.value = 'Theoretical d: —'
	    suggest_lbl.value = 'Suggested kf: —'
	    icf_angle_lbl.value = 'ICF (angles): —'
	    for out in (out_sdm, out_hist, out_icf, out_recon_m):
		    out.clear_output()
	    _status(f'Manual tab reset (kf={KF0}, icf={ICF0}).')

    # ---- automatic grid sweep --------------------------------------------
    kf_lo_w = widgets.FloatText(value=2.8, step=0.1, layout=_narrow)
    kf_hi_w = widgets.FloatText(value=3.8, step=0.1, layout=_narrow)
    kf_n_w = widgets.IntText(value=9, layout=_narrow)
    icf_lo_w = widgets.FloatText(value=1.4, step=0.05, layout=_narrow)
    icf_hi_w = widgets.FloatText(value=1.9, step=0.05, layout=_narrow)
    icf_n_w = widgets.IntText(value=9, layout=_narrow)

    def on_auto_run(_):
        cx, cy, r = cx_w.value, cy_w.value, r_w.value
        if int(sc.detector_roi_mask(variables, cx, cy, r).sum()) < 50:
	        _status('Set a valid ROI (Manual tab) first.')
            return
        kf_vals = np.linspace(kf_lo_w.value, kf_hi_w.value, int(kf_n_w.value))
        icf_vals = np.linspace(icf_lo_w.value, icf_hi_w.value, int(icf_n_w.value))
        prog = widgets.IntProgress(value=0, min=0, max=kf_vals.size * icf_vals.size,
                                   description='sweep')
        with out_status:
            out_status.clear_output(); display(prog)

        def _p(done, total):
            prog.value = done

        res = sc.grid_sweep_kf_icf(
            variables, (cx, cy), r, det_eff_w.value, field_evap_w.value, avg_dens_w.value,
            flight_path_length.value, kf_vals, icf_vals, mode=mode_w.value,
            bin_size=bin_w.value, z_max=zmax_w.value, lateral_radius=lat_w.value,
            max_atoms=int(maxatoms_w.value), progress=_p)
        auto_best.clear(); auto_best.update(res['best'])
        with out_auto:
            out_auto.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
            im = ax.imshow(res['strength'].T, origin='lower', aspect='auto', cmap='viridis',
                           extent=[kf_vals[0], kf_vals[-1], icf_vals[0], icf_vals[-1]])
            ax.plot(res['best']['kf'], res['best']['icf'], 'r*', ms=16)
            ax.set_xlabel('kf'); ax.set_ylabel('icf')
            ax.set_title('z-SDM peakiness (higher = sharper)')
            fig.colorbar(im, ax=ax, label='peakiness')
            plt.show()
        b = res['best']
        _status(f'Best: kf={b["kf"]:.3f}, icf={b["icf"]:.3f}  '
                f'(peakiness={b["strength"]:.1f}, spacing={b["spacing"]:.3f} nm, '
                f'ROI={res["roi_atoms"]} ions). "Show 3D recon (best)" or "Apply best & save".')

    def on_auto_show(_):
	    if not auto_best:
		    _status('Run the auto sweep first.')
		    return
	    _status(f'Reconstructing all ions (plotly, opens in a browser window) with best '
	            f'kf={auto_best["kf"]:.3f}, icf={auto_best["icf"]:.3f}...')
	    _show_recon_plotly(auto_best['kf'], auto_best['icf'], out_auto_recon, '3d_calibration_best')
	    _status(f'Opened the best 3-D reconstruction in a browser window '
	            f'(kf={auto_best["kf"]:.3f}, icf={auto_best["icf"]:.3f}). variables.x/y/z updated. '
	            'Use "Apply best & save → x/y/z" to also set the kf/icf fields.')

    def on_auto_apply(_):
	    if not auto_best:
		    _status('Run the auto sweep first.')
		    return
	    kf_w.value = round(auto_best['kf'], 4)
	    icf_w.value = round(auto_best['icf'], 4)
	    _status(f'Applied best kf={kf_w.value}, icf={icf_w.value}. Saving x/y/z...')
	    x, y, z = _reconstruct_full(kf_w.value, icf_w.value)
	    variables.x, variables.y, variables.z = x, y, z
	    _status(f'Applied best kf={kf_w.value}, icf={icf_w.value} and saved '
	            f'{x.size} reconstructed ions to variables.x/y/z.')

    def on_auto_reset(_):
	    auto_best.clear()
	    kf_w.value, icf_w.value = KF0, ICF0
	    for out in (out_auto, out_auto_recon):
		    out.clear_output()
	    _status(f'Automatic tab reset (kf={KF0}, icf={ICF0}).')

    # ---- buttons ---------------------------------------------------------
    btn_fdm = widgets.Button(description='Show FDMs', button_style='primary')
    btn_sdm = widgets.Button(description='Reconstruct + z-SDM', button_style='info')
    btn_fill = widgets.Button(description='Fill X/Y from poles')
    btn_icf = widgets.Button(description='Compute ICF (angles)', button_style='info')
    btn_apply = widgets.Button(description='Apply calibrated (kf, icf)', button_style='success')
    btn_recon_m = widgets.Button(description='Show recon (calibrated)')
    btn_reset_m = widgets.Button(description='Reset', button_style='warning')

    btn_auto = widgets.Button(description='Run auto sweep', button_style='info')
    btn_auto_show = widgets.Button(description='Show 3D recon (best)')
    btn_auto_apply = widgets.Button(description='Apply best & save → x/y/z', button_style='success')
    btn_auto_reset = widgets.Button(description='Reset', button_style='warning')

    btn_fdm.on_click(on_show_fdm)
    btn_sdm.on_click(on_reconstruct_sdm)
    btn_fill.on_click(on_fill_poles)
    btn_icf.on_click(on_compute_icf)
    btn_apply.on_click(on_apply_cal)
    btn_recon_m.on_click(on_show_recon_manual)
    btn_reset_m.on_click(on_reset_manual)
    btn_auto.on_click(on_auto_run)
    btn_auto_show.on_click(on_auto_show)
    btn_auto_apply.on_click(on_auto_apply)
    btn_auto_reset.on_click(on_auto_reset)

    def _row(lbl, *w):
        return widgets.HBox([widgets.Label(lbl, layout=_label), *w])

    recon_box = widgets.VBox([
        widgets.HTML('<b>Reconstruction parameters</b>'),
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
        widgets.HTML('<b>Pole ROI (detector, cm) — constant over iterations</b>'),
        _row('pick pole:', pole_dd),
        _row('cx / cy / radius (cm):', cx_w, cy_w, r_w),
    ])

    def _pole_row(i):
	    return widgets.HBox([widgets.Label(f'pole {i + 1}', layout=_tiny),
	                         pole_x[i], pole_y[i], pole_h[i], pole_k[i], pole_l[i]])

    icf_box = widgets.VBox([
	    widgets.HTML('<b>ICF from inter-pole angles (Gault 2009 / Day-Breen spreadsheet)</b><br>'
	                 '<i>Enter ≥2 poles: detector X, Y in mm and their Miller index. '
	                 'ICF = mean(theoretical angle / observed angle).</i>'),
	    widgets.HBox([widgets.Label('', layout=_tiny),
	                  widgets.Label('X (mm)', layout=_narrow), widgets.Label('Y (mm)', layout=_narrow),
	                  widgets.Label('h', layout=_tiny), widgets.Label('k', layout=_tiny),
	                  widgets.Label('l', layout=_tiny)]),
	    *[_pole_row(i) for i in range(n_poles)],
	    widgets.HBox([btn_fill, btn_icf, icf_angle_lbl]),
	    out_icf,
    ])

    manual_tab = widgets.VBox([
        widgets.HBox([recon_box, sdm_box]),
        widgets.HBox([lattice_box, roi_box]),
	    widgets.HBox([btn_fdm, btn_sdm]),
        out_fdm,
        out_sdm,
        widgets.HBox([widgets.Label('Measured d_obs (nm):', layout=_label), d_obs_w, theo_lbl]),
        suggest_lbl,
        widgets.HTML('<b>Iteration history</b>'),
        out_hist,
	    widgets.HTML('<hr>'),
	    icf_box,
	    widgets.HTML('<hr>'),
	    widgets.HBox([btn_apply, btn_recon_m, btn_reset_m]),
	    out_recon_m,
    ])

    auto_tab = widgets.VBox([
        widgets.HTML('<b>Automatic (kf, icf) grid sweep — maximises z-SDM peakiness '
                     'in the ROI set on the Manual tab</b>'),
        _row('kf  lo / hi / n:', kf_lo_w, kf_hi_w, kf_n_w),
        _row('icf lo / hi / n:', icf_lo_w, icf_hi_w, icf_n_w),
	    widgets.HBox([btn_auto, btn_auto_show, btn_auto_apply, btn_auto_reset]),
        out_auto,
	    out_auto_recon,
    ])

    tabs = widgets.Tab(children=[manual_tab, auto_tab])
    tabs.set_title(0, 'Manual (ICF/kf iterate)')
    tabs.set_title(1, 'Automatic sweep')
    display(widgets.VBox([tabs, widgets.HTML('<b>Status</b>'), out_status]))
