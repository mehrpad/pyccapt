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
* **Automatic** -- grid-sweep (kf, icf) maximising z-SDM peakiness in the same
  fixed ROI, show the strength heat-map, and apply the best pair.

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

_label = widgets.Layout(width='190px')
_narrow = widgets.Layout(width='90px')


def call_sdm_icf_kf_calibration(variables, flight_path_length, element_selected, colab=False):
    import matplotlib.pyplot as plt

    avg_dens0 = element_selected.value[1]
    field_evap0 = element_selected.value[2]

    # ---- shared reconstruction parameters --------------------------------
    kf_w = widgets.FloatText(value=3.3, step=0.05, layout=_narrow)
    icf_w = widgets.FloatText(value=1.65, step=0.05, layout=_narrow)
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
    r_w = widgets.FloatText(value=0.30, step=0.05, layout=_narrow)
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

    # ---- show normal + sharp FDM, detect pole candidates -----------------
    def on_show_fdm(_):
        with out_status:
            out_status.clear_output()
            print('Building FDMs...')
        detx = np.asarray(variables.dld_x_det, dtype=float)
        dety = np.asarray(variables.dld_y_det, dtype=float)
        density, contrast, xe, ye = sc.sharp_fdm_map(
            detx, dety, bins=int(fdm_bins_w.value), smooth_sigma=fdm_smooth_w.value)
        cands = sc.detect_pole_candidates(contrast, xe, ye, n=8)
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
            # draw current ROI
            th = np.linspace(0, 2 * np.pi, 80)
            ax[1].plot(cx_w.value + r_w.value * np.cos(th),
                       cy_w.value + r_w.value * np.sin(th), 'w-', lw=1.2)
            plt.show()
        with out_status:
            out_status.clear_output()
            print(f'Detected {len(cands)} pole candidate(s). Pick one or set cx/cy/radius, '
                  'then "Reconstruct + z-SDM".')

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
        if history and np.isfinite(history[-1]['suggest_kf']):
            kf_w.value = round(history[-1]['suggest_kf'], 4)
            with out_status:
                out_status.clear_output()
                print(f'kf set to {kf_w.value}. Adjust icf if needed and re-run "Reconstruct + z-SDM".')

    # ---- automatic grid sweep --------------------------------------------
    kf_lo_w = widgets.FloatText(value=2.8, step=0.1, layout=_narrow)
    kf_hi_w = widgets.FloatText(value=6.0, step=0.1, layout=_narrow)
    kf_n_w = widgets.IntText(value=9, layout=_narrow)
    icf_lo_w = widgets.FloatText(value=1.1, step=0.05, layout=_narrow)
    icf_hi_w = widgets.FloatText(value=1.9, step=0.05, layout=_narrow)
    icf_n_w = widgets.IntText(value=9, layout=_narrow)
    auto_best = {}

    def on_auto_run(_):
        cx, cy, r = cx_w.value, cy_w.value, r_w.value
        if int(sc.detector_roi_mask(variables, cx, cy, r).sum()) < 50:
            with out_status:
                out_status.clear_output(); print('Set a valid ROI (Manual tab) first.')
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
        with out_status:
            out_status.clear_output()
            b = res['best']
            print(f'Best: kf={b["kf"]:.3f}, icf={b["icf"]:.3f}  '
                  f'(peakiness={b["strength"]:.1f}, spacing={b["spacing"]:.3f} nm, '
                  f'ROI={res["roi_atoms"]} ions)')

    def on_auto_apply(_):
        if auto_best:
            kf_w.value = round(auto_best['kf'], 4)
            icf_w.value = round(auto_best['icf'], 4)
            with out_status:
                out_status.clear_output()
                print(f'Applied best kf={kf_w.value}, icf={icf_w.value}.')

    # ---- buttons ---------------------------------------------------------
    btn_fdm = widgets.Button(description='Show FDMs', button_style='primary')
    btn_sdm = widgets.Button(description='Reconstruct + z-SDM', button_style='info')
    btn_apply_kf = widgets.Button(description='Apply suggested kf')
    btn_auto = widgets.Button(description='Run auto sweep', button_style='info')
    btn_auto_apply = widgets.Button(description='Apply best (kf, icf)')
    btn_fdm.on_click(on_show_fdm)
    btn_sdm.on_click(on_reconstruct_sdm)
    btn_apply_kf.on_click(on_apply_kf)
    btn_auto.on_click(on_auto_run)
    btn_auto_apply.on_click(on_auto_apply)

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

    manual_tab = widgets.VBox([
        widgets.HBox([recon_box, sdm_box]),
        widgets.HBox([lattice_box, roi_box]),
        widgets.HBox([btn_fdm, btn_sdm, btn_apply_kf]),
        out_fdm,
        out_sdm,
        widgets.HBox([widgets.Label('Measured d_obs (nm):', layout=_label), d_obs_w, theo_lbl]),
        suggest_lbl,
        widgets.HTML('<b>Iteration history</b>'),
        out_hist,
    ])

    auto_tab = widgets.VBox([
        widgets.HTML('<b>Automatic (kf, icf) grid sweep — maximises z-SDM peakiness '
                     'in the ROI set on the Manual tab</b>'),
        _row('kf  lo / hi / n:', kf_lo_w, kf_hi_w, kf_n_w),
        _row('icf lo / hi / n:', icf_lo_w, icf_hi_w, icf_n_w),
        widgets.HBox([btn_auto, btn_auto_apply]),
        out_auto,
    ])

    tabs = widgets.Tab(children=[auto_tab, manual_tab])
    tabs.set_title(0, 'Automatic sweep')
    tabs.set_title(1, 'Manual (ICF/kf iterate)')
    display(widgets.VBox([tabs, widgets.HTML('<b>Status</b>'), out_status]))
