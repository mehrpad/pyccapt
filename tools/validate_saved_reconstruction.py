r"""Replay a saved Geiser reconstruction and compare conditional suggestion objectives.

Example (all physical assumptions are diagnostic, not measured/indexed truth)::

    python tools/validate_saved_reconstruction.py --data acquisition.h5 \
        --output-directory analysis_outputs/reconstruction --flight-mm 110 \
        --spacing-nm 0.1795 --field-v-nm 35 --efficiency 0.7

Requires a PyCCAPT/PyTables fixed-format /df with float block0 containing detector
coordinates, voltage, pulse voltage, and saved x/y/z. Fits effective ICF, kf*field,
and density*efficiency products. No physical reference or source-coverage claim is
inferred from the numerical replay. The source is always opened read-only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import h5py
import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pyccapt.calibration.data_tools.analysis_session import software_identity
from pyccapt.calibration.core.exceptions import CalibrationInputError
from pyccapt.calibration.reconstructions import ratio_calibration as rc
from pyccapt.calibration.reconstructions import sdm_calibration as sc

REPORT_NAME = 'saved_reconstruction_audit.json'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, required=True, help='Read-only PyCCAPT /df HDF5 source')
    parser.add_argument('--output-directory', type=Path, required=True, help='Directory for JSON and PNG diagnostic outputs')
    parser.add_argument('--flight-mm', type=float, required=True, help='Declared flight length in mm; not fitted')
    parser.add_argument('--spacing-nm', type=float, default=.1795,
        help='Assumed plane spacing for conditional profile (default .1795 nm); no plane identity is inferred')
    parser.add_argument('--field-v-nm', type=float, default=35., help='Replay gauge field, NOT a measured value (default 35 V/nm)')
    parser.add_argument('--efficiency', type=float, default=.7, help='Replay gauge efficiency, NOT measured (default .7)')
    parser.add_argument('--roi-x-cm', type=float, default=0., help='Conditional ROI center x, no automatic indexing (default 0 cm)')
    parser.add_argument('--roi-y-cm', type=float, default=0., help='Conditional ROI center y, no automatic indexing (default 0 cm)')
    parser.add_argument('--roi-radius-cm', type=float, default=.1, help='Conditional ROI radius (default .1 cm)')
    parser.add_argument('--icf-min', type=float, default=1.1, help='Conditional profile family lower bound (default 1.1)')
    parser.add_argument('--icf-max', type=float, default=1.9, help='Conditional profile family upper bound (default 1.9)')
    parser.add_argument('--q-relative-min', type=float, default=.75, help='q lower bound / replay q (default .75)')
    parser.add_argument('--q-relative-max', type=float, default=1.25, help='q upper bound / replay q (default 1.25)')
    parser.add_argument('--grid-budget-seconds', type=float, default=45., help='Production-grid time budget, checked between candidates')
    args = parser.parse_args(argv)
    numeric = [getattr(args, key) for key in ('flight_mm', 'spacing_nm', 'field_v_nm', 'efficiency',
        'roi_radius_cm', 'icf_min', 'icf_max', 'q_relative_min', 'q_relative_max', 'grid_budget_seconds')]
    if not np.isfinite(numeric).all() or min(numeric) <= 0 or args.efficiency > 1:
        parser.error('Physical scales and time budget must be positive finite; efficiency must be <=1')
    if not np.isfinite([args.roi_x_cm, args.roi_y_cm]).all():
        parser.error('ROI center coordinates must be finite')
    if args.icf_max < args.icf_min or args.q_relative_max <= args.q_relative_min:
        parser.error('ICF and q bounds must be ordered')
    args.data = args.data.resolve()
    args.output_directory = args.output_directory.resolve()
    args.output = args.output_directory/REPORT_NAME
    for output in (args.output, args.output.with_suffix('.png')):
        if output.resolve() == args.data or (output.exists() and args.data.exists() and output.samefile(args.data)):
            parser.error('An output path resolves to the input data; select another output directory')
    if not args.data.is_file():
        parser.error('The input data file does not exist')
    if args.output_directory.exists() and not args.output_directory.is_dir():
        parser.error('The output directory points to an existing file')
    return args


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, bytes):
        return value.decode(errors='replace')
    if isinstance(value, h5py.Empty):
        return {'hdf5_empty_dtype': str(value.dtype)}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(report, output):
    output.write_text(json.dumps(clean(report), indent=2, allow_nan=False), encoding='utf8')


def file_sha256(path):
    """Hash without materializing a large acquisition; compatible with Python 3.10."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(source):
    """Load four event columns and only a bounded coordinate-replay sample."""
    with h5py.File(source, 'r') as store:
        group = store['df']
        names = [v.decode() if isinstance(v, bytes) else str(v) for v in group['block0_items'][:]]
        dataset = group['block0_values']
        n = len(dataset)
        required = ['x_det (cm)', 'y_det (cm)', 'high_voltage (V)', 'pulse_v (V)']
        saved_coordinates = ['x (nm)', 'y (nm)', 'z (nm)']
        missing = set(required+saved_coordinates)-set(names)
        if missing:
            raise ValueError(f'Required columns missing from /df/block0: {sorted(missing)}')
        if n < 100:
            raise ValueError('At least 100 events are required for this numerical diagnostic')
        xyz_columns = [names.index(name) for name in saved_coordinates]
        data = {key: np.empty(n) for key in required}
        # Regular sample is ONLY for numerical recipe fitting, never periodic scoring.
        sample = np.unique(np.linspace(0, n-1, 8000, dtype=int))
        xyz = np.empty((len(sample), 3))
        for lo in range(0, n, 250000):
            hi = min(n, lo+250000)
            block = dataset[lo:hi]
            for name in required:
                data[name][lo:hi] = block[:, names.index(name)]
            hit = (sample >= lo) & (sample < hi)
            xyz[hit] = block[np.ix_(sample[hit]-lo, xyz_columns)]
        metadata = dict(root_groups=list(store.keys()),
                        group_attrs={key: clean(value) for key, value in group.attrs.items()},
                        rows=n, columns=names, dtypes=str(dataset.dtype))
    variables = SimpleNamespace(dld_x_det=data['x_det (cm)'], dld_y_det=data['y_det (cm)'],
        dld_high_voltage=data['high_voltage (V)'], dld_pulse_v=data['pulse_v (V)'],
        pulse_mode='voltage', data_backup=None)
    return variables, sample, xyz, metadata


def fit_recipe(variables, sample, xyz, hv, convention, flight_mm):
    radius_mm = np.hypot(variables.dld_x_det[sample], variables.dld_y_det[sample])*10.
    observed = np.arctan(radius_mm/flight_mm)
    saved_radius = np.hypot(xyz[:, 0], xyz[:, 1])
    voltage = hv[sample]
    good = np.isfinite(saved_radius) & np.isfinite(radius_mm) & (radius_mm > .1)
    if np.count_nonzero(good) < 50 or not np.isfinite(xyz).all() or not np.isfinite(hv).all() or np.any(hv <= 0):
        raise ValueError('Recipe replay requires finite saved coordinates, positive voltage and >=50 off-axis fitting events')
    # The audit uses a bounded forward-hemisphere Geiser family, not an invalid asin.
    max_observed = float(np.max(observed[good]))
    fit_upper = min(2.5, 1.+np.cos(max_observed)/np.sin(max_observed))
    if fit_upper <= 1.01:
        raise ValueError('Flight length/detector units do not permit the audit Geiser ICF fitting domain')

    def scale(icf):
        theta = observed + np.arcsin((icf-1)*np.sin(observed))
        predictor = voltage*np.sin(theta)
        inverse_kf_field = np.dot(predictor[good], saved_radius[good])/np.dot(predictor[good], predictor[good])
        mse = np.mean((predictor[good]*inverse_kf_field-saved_radius[good])**2)
        return mse, inverse_kf_field, theta

    fit = minimize_scalar(lambda icf: scale(icf)[0], bounds=(1.01, fit_upper), method='bounded',
                          options={'xatol': 1e-13})
    mse, inverse_kf_field, theta = scale(fit.x)
    kf_field = 1./inverse_kf_field
    prefix = np.cumsum(1./hv**2)
    cap = voltage*inverse_kf_field*(1.-np.cos(theta))
    design = np.column_stack((prefix[sample], np.ones(len(sample))))
    depth_coefficient, depth_offset = np.linalg.lstsq(design, xyz[:, 2]-cap, rcond=None)[0]
    z_residual = design @ [depth_coefficient, depth_offset] + cap - xyz[:, 2]
    area = np.pi*np.nanmax(np.hypot(variables.dld_x_det*10., variables.dld_y_det*10.))**2
    density_efficiency = flight_mm**2*kf_field**2/(area*fit.x**2*depth_coefficient)
    if not fit.success or not np.isfinite(density_efficiency) or density_efficiency <= 0:
        raise ValueError('Saved coordinates do not yield a finite positive Geiser recipe in the declared domain')
    # The Bas lateral model identifies only ICF/(kf*field), independent of ICF separately.
    bas_predictor = voltage*radius_mm/flight_mm
    bas_scale = np.dot(bas_predictor[good], saved_radius[good])/np.dot(bas_predictor[good], bas_predictor[good])
    residual = bas_predictor*bas_scale-saved_radius
    xy_cross = xyz[:, 0]*variables.dld_y_det[sample]-xyz[:, 1]*variables.dld_x_det[sample]
    return dict(hv_convention=convention, fitting_rows=len(sample), flight_path_mm=flight_mm,
        fitted_effective_icf=float(fit.x), fitted_effective_kf_times_field_V_nm=kf_field,
        fitted_effective_density_times_efficiency_nm3=density_efficiency,
        fitted_depth_coefficient_nm_V2=depth_coefficient, fitted_depth_origin_nm=depth_offset,
        geiser_radial_rmse_nm=np.sqrt(mse), geiser_depth_rmse_nm=np.sqrt(np.mean(z_residual**2)),
        max_depth_error_nm=np.max(np.abs(z_residual)),
        bas_best_lateral_rmse_nm=np.sqrt(np.mean(residual[good]**2)),
        saved_xy_vs_detector_azimuth_cross_max_nm_cm=np.max(np.abs(xy_cross)),
        fitting_icf_bounds=[1.01, fit_upper],
        inference='Effective recipe fit conditional on declared flight length, Geiser model and whole-file order; not measured settings or physical truth'), prefix


def legacy_grid_from_full_prefix(variables, prefix, parameters, kf_values, icf_values, center, radius):
    """Exact factorization of current Geiser grid geometry; keep original SDM objective.

    All events contribute to prefix and detector area. Only selected ROI coordinates
    are materialized, which is algebraically identical to reconstructing all then
    cropping and avoids allocating full x/y/z for every candidate.
    """
    flight_mm = parameters['flight_path_mm']
    roi = sc.detector_roi_mask(variables, *center, radius)
    indices = np.flatnonzero(roi)
    hv = sc.build_hv(variables)[indices]
    xx, yy = variables.dld_x_det[indices], variables.dld_y_det[indices]
    detector_area = np.pi*np.nanmax(np.hypot(variables.dld_x_det*10., variables.dld_y_det*10.))**2
    observed = np.arctan(np.hypot(xx, yy)*10./flight_mm)
    azimuth = np.arctan2(yy, xx)
    strengths, spacings = np.zeros((3, 3)), np.full((3, 3), np.nan)
    start = perf_counter()
    for i, kf in enumerate(kf_values):
        for j, icf in enumerate(icf_values):
            theta = observed + np.arcsin((icf-1)*np.sin(observed))
            cap_radius = hv/(kf*parameters['field_evap_v_nm'])
            lateral = cap_radius*np.sin(theta)
            x, y = lateral*np.cos(azimuth), lateral*np.sin(azimuth)
            coefficient = (flight_mm**2*(kf*parameters['field_evap_v_nm'])**2 /
                (parameters['atomic_density_nm3']*parameters['det_eff']*detector_area*icf**2))
            z = coefficient*prefix[indices] + cap_radius*(1.-np.cos(theta))
            x, y, z, _ = sc.align_pole_to_z(x, y, z, sc.pole_axis_from_detector(*center, flight_mm, icf))
            centers, counts = sc.z_sdm(x, y, z, bin_size=.01, z_max=2., lateral_radius=1.5, max_atoms=1000)
            strengths[i, j] = sc.sdm_peakiness(centers, counts)
            spacings[i, j] = sc.find_sdm_peaks(centers, counts)['spacing']
    i, j = np.unravel_index(np.argmax(strengths), strengths.shape)
    return dict(scope='Bounded legacy peakiness objective replay with exact full-prefix ROI factorization; historical grid settings unavailable',
        elapsed_seconds=perf_counter()-start, roi_events=len(indices),
        settings=dict(kf_values=kf_values, icf_values=icf_values, roi_center_cm=center, roi_radius_cm=radius,
            bin_size_nm=.01, z_max_nm=2., lateral_radius_nm=1.5, max_atoms=1000, sample_seed=0),
        strength_grid=strengths, measured_spacing_grid_nm=spacings,
        unconditional_argmax=dict(kf=kf_values[i], icf=icf_values[j], q=icf_values[j]/kf_values[i],
            strength=strengths[i,j], measured_spacing_nm=spacings[i,j]),
        current_guard_status='not_verified',
        expected_spacing_used_for_selection=False)


def main(argv=None):
    args = parse_args(argv)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    variables, sample, xyz, metadata = load(args.data)
    load_seconds = perf_counter()-start
    ratio = variables.dld_pulse_v/variables.dld_high_voltage
    dc_fit, dc_prefix = fit_recipe(variables, sample, xyz, variables.dld_high_voltage, 'DC only', args.flight_mm)
    del dc_prefix
    voltage = sc.build_hv(variables)
    vp_fit, full_prefix = fit_recipe(variables, sample, xyz, voltage, 'DC plus pulse voltage', args.flight_mm)
    del voltage
    file_hash = file_sha256(args.data)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), source=str(args.data),
        source_file_sha256=file_hash, source_metadata=metadata, load_seconds=load_seconds,
        software=software_identity(), effective_recipe_fits=[dc_fit, vp_fit],
        command_arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        pulse_to_dc_ratio_min=float(np.min(ratio)), pulse_to_dc_ratio_max=float(np.max(ratio)),
        saved_xyz_sample_bounds_nm=[np.min(xyz, axis=0), np.max(xyz, axis=0)],
        limitations=[
            'Full FILE order is available; original acquisition event-zero/completeness and any pre-export crop are unverified.',
            'Saved reconstruction settings, indexed pole, phase orientation, independently measured density/efficiency/field are unavailable.',
            'An effective saved-coordinate recipe fit is a numerical replay and does not validate reconstruction accuracy.',
            'ROI and spacing are explicitly conditional diagnostic assumptions; no plane identity is inferred by this tool.',
            'A constant pulse/DC ratio makes voltage convention degenerate with the field-factor product; compare recorded residuals for this source.',
            'No suggested parameters are accepted or written to source/shared state.'])
    del ratio
    # Choose a gauge solely to replay the saved recipe in the current voltage mode.
    # Neither factor is claimed independently measured.
    parameters = dict(det_eff=args.efficiency, field_evap_v_nm=args.field_v_nm,
        atomic_density_nm3=vp_fit['fitted_effective_density_times_efficiency_nm3']/args.efficiency,
        flight_path_mm=args.flight_mm)
    kf_replay = vp_fit['fitted_effective_kf_times_field_V_nm']/args.field_v_nm
    icf_replay = vp_fit['fitted_effective_icf']
    q_replay = icf_replay/kf_replay
    q_values = np.linspace(q_replay*args.q_relative_min, q_replay*args.q_relative_max, 251)
    spacing, radius = args.spacing_nm, args.roi_radius_cm
    center, icf_bounds = [args.roi_x_cm, args.roi_y_cm], [args.icf_min, args.icf_max]
    report.update(parameters=parameters, parameter_choice='Declared field and efficiency gauges with fitted remaining products, solely for saved-recipe replay; no factor independently measured',
        reconstructed_recipe_replay=dict(kf=kf_replay, icf=icf_replay, q=q_replay),
        q_convention='ICF/kf', spacing_nm=spacing, icf_bounds=icf_bounds,
        requested_q_grid=[q_values[0], q_values[-1], len(q_values)], roi_center_cm=center, roi_radius_cm=radius)
    report['default_unconfirmed_coverage_result'] = rc.profile_icf_kf_ratio(
        variables, center, radius, spacing, q_values, **parameters)
    write(report, args.output)
    print(json.dumps(clean({'effective_recipe_fits': report['effective_recipe_fits'],
        'source_file_sha256': file_hash, 'q_replay': q_replay}), indent=2), flush=True)
    start = perf_counter()
    full_x, full_y, full_z = sc.reconstruct_xyz(variables, kf_replay, icf_replay,
        parameters['det_eff'], parameters['field_evap_v_nm'], parameters['atomic_density_nm3'],
        args.flight_mm, mode='Geiser')
    residual = np.column_stack([full_x[sample], full_y[sample], full_z[sample]])-xyz
    report['public_reconstruction_replay'] = dict(api='sdm_calibration.reconstruct_xyz',
        reconstructed_events=len(full_x), checked_saved_xyz_events=len(sample),
        elapsed_seconds=perf_counter()-start, max_abs_error_nm=np.max(np.abs(residual), axis=0),
        rmse_nm=np.sqrt(np.mean(residual**2, axis=0)),
        note='Full event reconstruction through production API, compared against saved coordinate sample; not an independent physical validation')
    del full_x, full_y, full_z
    start = perf_counter()
    geometry = rc.prepare_ratio_geometry(variables, center, radius, full_acquisition_confirmed=True,
        sample_seed=0, **parameters)
    report['conditional_ratio_profile'] = rc.profile_ratio_geometry(geometry, spacing, q_values,
        icf_bounds=icf_bounds, mode='Geiser')
    report['conditional_ratio_seconds'] = perf_counter()-start
    report['conditional_coverage_override'] = 'Complete file numerical probe only; does not establish original acquisition completeness'
    write(report, args.output)
    kf_values = (np.array([.85, 1., 1.15])*kf_replay).tolist()
    icf_values = (np.array([.85, 1., 1.15])*icf_replay).tolist()
    report['legacy_grid'] = legacy_grid_from_full_prefix(variables, full_prefix, parameters, kf_values, icf_values, center, radius)
    start = perf_counter()
    def bounded_progress(done, total):
        if perf_counter()-start > args.grid_budget_seconds:
            raise TimeoutError(f'Production grid exceeded {args.grid_budget_seconds:g}-second budget at {done}/{total}')
    try:
        public_grid = sc.grid_sweep_kf_icf(variables, center, radius,
            parameters['det_eff'], parameters['field_evap_v_nm'], parameters['atomic_density_nm3'],
            args.flight_mm, kf_values, icf_values, mode='Geiser', bin_size=.01, z_max=2.,
            lateral_radius=1.5, max_atoms=1000, progress=bounded_progress)
        report['legacy_grid']['production_api_verification'] = dict(
            api='sdm_calibration.grid_sweep_kf_icf', status=public_grid['status'],
            best=public_grid['best'], elapsed_seconds=perf_counter()-start,
            max_abs_strength_error=float(np.max(np.abs(public_grid['strength']-report['legacy_grid']['strength_grid']))),
            max_abs_spacing_error_nm=float(np.nanmax(np.abs(public_grid['spacing']-report['legacy_grid']['measured_spacing_grid_nm']))),
            full_acquisition_events_per_candidate=len(full_prefix))
        report['legacy_grid']['current_guard_status'] = public_grid['status']
    except (TimeoutError, CalibrationInputError) as exc:
        report['legacy_grid']['production_api_verification'] = dict(
            status='time_budget_reached' if isinstance(exc, TimeoutError) else 'invalid_grid', message=str(exc))
    write(report, args.output)
    make_figure(report, args.output.with_suffix('.png'))
    print(json.dumps(clean({'report': str(args.output), 'ratio_status': report['conditional_ratio_profile']['status'],
        'ratio_diagnostics': report['conditional_ratio_profile']['diagnostics'],
        'legacy_grid': report['legacy_grid']}), indent=2), flush=True)


def make_figure(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    profile = report['conditional_ratio_profile']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    q = profile['q_grid']
    if profile['training_profile']:
        axes[0].plot(q, profile['training_profile'], label='Training-band median')
        axes[0].plot(q, profile['heldout_profile'], label='Held-out-band median', alpha=.8)
        axes[0].axhline(np.min(profile['diagnostics']['coherence_floor']), color='black', ls=':', label='Smallest band evidence gate')
    axes[0].axvline(report['reconstructed_recipe_replay']['q'], color='gray', ls='--', label='Stored-recipe replay q (conditional gauge)')
    axes[0].set(xlabel='q = ICF/kf', ylabel='Exact-period coherence', title=f"Ratio profile: {profile['status']}")
    axes[0].legend(fontsize=7, loc='upper left')
    grid = report['legacy_grid']
    image = axes[1].imshow(grid['strength_grid'], origin='lower', aspect='auto', cmap='viridis')
    axes[1].set_xticks(range(3), [f'{v:.2f}' for v in grid['settings']['icf_values']])
    axes[1].set_yticks(range(3), [f'{v:.2f}' for v in grid['settings']['kf_values']])
    axes[1].set(xlabel='ICF', ylabel='kf', title=f"Legacy peakiness grid: {grid['current_guard_status']}")
    fig.colorbar(image, ax=axes[1], label='FFT peakiness')
    fig.suptitle('Conditional diagnostic: no indexed-pole or physical ground truth', fontsize=11)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
