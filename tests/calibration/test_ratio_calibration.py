"""Independent periodic-depth truth, abstention and full-prefix invariants."""
from types import SimpleNamespace

import numpy as np
import pytest

from pyccapt.calibration.core.exceptions import CalibrationInputError, CalibrationStateError
from pyccapt.calibration.reconstructions import ratio_calibration as rc
from pyccapt.calibration.reconstructions import reconstruction


def periodic_acquisition(q=.5, n=90000, random_roi=False, changing_ratio=False):
    rng = np.random.default_rng(19)
    indices = np.arange(n)
    voltage = 5000. + 500.*np.sin(indices/n*2*np.pi)
    area = np.pi*10.**2  # All non-ROI hits extend to radius 1 cm.
    coefficient = 110.**2*30.**2/(60.*.7*area)
    inverse_prefix = np.cumsum(1./voltage**2)
    depth = coefficient*inverse_prefix/q**2
    if changing_ratio:
        band = indices*6//n
        depth = np.where(band % 2 == 1, coefficient*inverse_prefix/(q*1.05)**2, depth)
    distance_to_plane = np.abs((depth+.1) % .2 - .1)
    selected = distance_to_plane < .012
    if random_roi:
        selected = rng.random(n) < .12
    x, y = np.ones(n), np.zeros(n)
    x[selected] = rng.uniform(-.005, .005, selected.sum())
    y[selected] = rng.uniform(-.005, .005, selected.sum())
    variables = SimpleNamespace(dld_x_det=x, dld_y_det=y, dld_high_voltage=voltage,
                                pulse_mode='laser', data_backup=None, reconstruction_parameters={})
    return variables, inverse_prefix, selected


def prepare(variables, **kwargs):
    return rc.prepare_ratio_geometry(variables, (0., 0.), .01, det_eff=.7, field_evap_v_nm=30.,
                                     atomic_density_nm3=60., flight_path_mm=110.,
                                     full_acquisition_confirmed=True, **kwargs)


def profile(variables, **kwargs):
    return rc.profile_ratio_geometry(prepare(variables), .2, np.linspace(.4, .6, 161), **kwargs)


def test_recovers_user_ratio_and_family_without_claiming_unique_pair():
    variables, _, _ = periodic_acquisition()
    result = profile(variables)
    assert result['status'] == 'supported', result
    assert result['q'] == pytest.approx(.5, abs=.0015)
    assert result['metadata']['q_convention'] == 'ICF/kf'
    assert result['supported_interval'][0] <= .5 <= result['supported_interval'][1]
    family = rc.compatible_parameter_family(result, [1.3, 1.6, 1.9])
    np.testing.assert_allclose([row['kf'] for row in family], [2.6, 3.2, 3.8], atol=.012)
    assert not variables.reconstruction_parameters
    accepted = rc.accept_ratio_family(variables, result, icf_anchor=1.6)
    assert accepted['icf'] == 1.6 and accepted['kf'] == pytest.approx(3.2, abs=.012)
    assert 'not inferred' in variables.reconstruction_ratio_result['anchor_source']
    assert result['diagnostics']['training_band_bootstrap']['repeats'] == 200


def test_full_prefix_precedes_roi_species_and_sampling_and_matches_reconstruction_depth():
    variables, expected_prefix, selected = periodic_acquisition()
    # An unavailable detector position still contributes its acquisition event to depth.
    variables.dld_x_det[1] = np.nan
    species = np.arange(len(selected)) % 2 == 0
    geometry = prepare(variables, event_mask=species, max_events_per_band=200)
    np.testing.assert_allclose(geometry.cumulative_inverse_voltage2, expected_prefix, rtol=0, atol=0)
    assert len(geometry.selected_indices) <= 1200
    assert np.all(species[geometry.selected_indices])
    assert geometry.metadata['source_events'] == len(expected_prefix)
    q, icf = .5, 1.6
    x, y, z = reconstruction.atom_probe_recons_from_detector_Geiser_et_al(
        variables.dld_x_det, variables.dld_y_det, variables.dld_high_voltage, 110., icf/q, .7, icf, 30., 60.)
    radius_nm = variables.dld_high_voltage/(icf/q*30.)
    cap = radius_nm-np.sqrt(radius_nm**2-x**2-y**2)
    np.testing.assert_allclose((z-cap)[geometry.selected_indices], geometry.base_depth_nm/q**2, rtol=2e-12, atol=1e-11)
    bad_prefix = np.cumsum(1./variables.dld_high_voltage[geometry.selected_indices]**2)
    assert expected_prefix[geometry.selected_indices][-1] > 20*bad_prefix[-1]


def test_missing_full_source_declaration_and_detected_crop_abstain():
    variables, _, _ = periodic_acquisition(n=1000)
    kwargs = dict(det_eff=.7, field_evap_v_nm=30., atomic_density_nm3=60., flight_path_mm=110.)
    result = rc.profile_icf_kf_ratio(variables, (0., 0.), .01, .2, np.linspace(.4, .6, 41), **kwargs)
    assert result['q'] is None and 'event zero' in result['message']
    variables.data_backup = np.empty(2000)
    result = rc.profile_icf_kf_ratio(variables, (0., 0.), .01, .2, np.linspace(.4, .6, 41),
                                    full_acquisition_confirmed=True, **kwargs)
    assert result['q'] is None and 'cropped' in result['message']


def test_uncrystalline_data_has_no_ratio_suggestion():
    variables, _, _ = periodic_acquisition(random_roi=True)
    result = profile(variables)
    assert result['q'] is None and result['status'] == 'no_periodic_evidence'
    evidence = result['diagnostics']['band_evidence']
    assert evidence['bands_with_exploratory_signal'] == 0
    assert len(evidence['bands']) == 6
    assert all(row['exploratory_peak_coherence'] < row['coherence_floor'] for row in evidence['bands'])
    assert any('independently indexed' in step for step in rc.ratio_next_steps(result))


def test_active_crop_selects_events_and_band_span_without_restarting_depth():
    variables, expected_prefix, _ = periodic_acquisition(n=180000)
    active = np.zeros(len(expected_prefix), dtype=bool)
    active[30000:150000] = True
    variables.reconstruction_selection_mask = active
    geometry = prepare(variables)
    np.testing.assert_array_equal(geometry.cumulative_inverse_voltage2, expected_prefix)
    assert geometry.metadata['source_events'] == 180000
    assert geometry.metadata['active_selection_events'] == 120000
    assert geometry.metadata['analysis_source_index_span'] == [30000, 150000]
    assert np.all(active[geometry.selected_indices])
    assert len(np.unique(geometry.bands)) == 6
    result = rc.profile_ratio_geometry(geometry, .2, np.linspace(.4, .6, 161))
    assert result['status'] == 'supported' and result['q'] == pytest.approx(.5, abs=.002)
    # In-place crop edits must invalidate acceptance even though source arrays are unchanged.
    variables.reconstruction_selection_mask[30000] = False
    with pytest.raises(CalibrationStateError, match='selection changed'):
        rc.accept_ratio_family(variables, result, icf_anchor=1.6)


def test_recorded_cropped_export_cannot_override_missing_prefix_with_checkbox():
    import pandas as pd
    variables, _, _ = periodic_acquisition(n=1000)
    variables.data = pd.DataFrame({'event_id': np.arange(1000, 2000)})
    variables.data.attrs['pyccapt_selection'] = dict(source_events=2000, selected_events=1000, preserves_full_source=False)
    kwargs = dict(det_eff=.7, field_evap_v_nm=30., atomic_density_nm3=60., flight_path_mm=110.)
    result = rc.profile_icf_kf_ratio(variables, (0.,0.), .01, .2, np.linspace(.4,.6,41),
                                    full_acquisition_confirmed=True, **kwargs)
    assert result['q'] is None and 'cropped export' in result['message']
    from pyccapt.calibration.reconstructions.sdm_calibration import reconstruct_xyz
    with pytest.raises(CalibrationInputError, match='cropped export'):
        reconstruct_xyz(variables, 3., 1.5, .7, 30., 60., 110.)


def test_band_diagnostics_reveal_signal_that_does_not_transfer_without_accepting_it():
    variables, _, _ = periodic_acquisition(changing_ratio=True)
    result = profile(variables)
    rows = result['diagnostics']['band_evidence']['bands']
    assert result['q'] is None and result['status'] == 'heldout_disagreement'
    assert all(row['exploratory_peak_passes_gates'] for row in rows)
    np.testing.assert_allclose([row['exploratory_peak_q'] for row in rows[::2]], .5, atol=.002)
    np.testing.assert_allclose([row['exploratory_peak_q'] for row in rows[1::2]], .525, atol=.002)


def test_off_axis_data_abstains_with_actionable_geometry_message():
    variables, _, _ = periodic_acquisition()
    geometry = rc.prepare_ratio_geometry(variables, (1., 0.), .01, det_eff=.7, field_evap_v_nm=30.,
                                          atomic_density_nm3=60., flight_path_mm=110., full_acquisition_confirmed=True)
    result = rc.profile_ratio_geometry(geometry, .2, np.linspace(.4, .6, 81))
    assert result['q'] is None and result['status'] == 'geometry_unsupported'
    assert 'near-axis' in result['message']


@pytest.mark.parametrize('mode,outer_radius_cm', [('Bas', 5.9), ('Geiser', 20.)])
def test_central_roi_cannot_certify_invalid_full_detector_icf_family(mode, outer_radius_cm):
    variables, _, _ = periodic_acquisition()
    variables.dld_x_det[1] = outer_radius_cm
    geometry = prepare(variables)
    assert geometry.metadata['full_detector_radius_cm'] == pytest.approx(outer_radius_cm)
    assert np.max(np.abs(geometry.detector_x_cm)) < .01  # The offending event was not sampled.
    result = rc.profile_ratio_geometry(geometry, .2, np.linspace(.4, .6, 161),
                                      icf_bounds=(1.85, 1.9), mode=mode)
    assert result['status'] == 'geometry_unsupported' and result['q'] is None
    assert 'full-detector' in result['message']
    if mode == 'Bas':
        with pytest.raises(CalibrationInputError, match='domain'):
            reconstruction.atom_probe_recons_Bas_et_al(
                variables.dld_x_det, variables.dld_y_det, variables.dld_high_voltage,
                110., 1.9/.5, .7, 1.9, 30., 60.)


@pytest.mark.parametrize('mode', ['Geiser', 'Bas'])
def test_fixed_origin_ratio_geometry_matches_public_reconstruction_with_variable_radius(mode):
    rng = np.random.default_rng(401)
    n = 1200
    variables = SimpleNamespace(dld_x_det=rng.uniform(.16, .24, n),
        dld_y_det=rng.uniform(.06, .14, n), dld_high_voltage=np.linspace(4000., 6500., n),
        dld_pulse_v=np.linspace(800., 1300., n), pulse_mode='voltage', data_backup=None)
    geometry = rc.prepare_ratio_geometry(variables, (.2, .1), .05, det_eff=.7, field_evap_v_nm=30.,
        atomic_density_nm3=60., flight_path_mm=110., full_acquisition_confirmed=True, max_events_per_band=100)
    fn = (reconstruction.atom_probe_recons_Bas_et_al if mode == 'Bas'
          else reconstruction.atom_probe_recons_from_detector_Geiser_et_al)
    q, icf = .5, 1.6
    xyz = np.column_stack(fn(variables.dld_x_det, variables.dld_y_det,
        variables.dld_high_voltage+variables.dld_pulse_v, 110., icf/q, .7, icf, 30., 60.))
    # Independent normal from the projection's detector-angle relation.
    radius_mm = np.hypot(.2, .1)*10.
    theta = (np.arcsin(icf*radius_mm/110.) if mode == 'Bas' else
             np.arctan(radius_mm/110.) + np.arcsin((icf-1)*np.sin(np.arctan(radius_mm/110.))))
    phi = np.arctan2(.1, .2)
    pole = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    normal = pole*np.array([-1., -1., 1.])
    projected = xyz[geometry.selected_indices]@normal
    direction = rc._launch(geometry.detector_x_cm, geometry.detector_y_cm, icf, 110., mode)
    radius_nm = geometry.voltage_V*q/(icf*30.)
    exact = geometry.base_depth_nm/q**2*pole[2] + radius_nm*(pole[2]-direction@pole)
    np.testing.assert_allclose(projected, exact, rtol=2e-12, atol=1e-10)
    moving_origin = geometry.base_depth_nm/q**2*pole[2] + radius_nm*(1.-direction@pole)
    assert np.ptp(moving_origin-projected) > .001  # The omitted origin term is not a constant offset.


def test_source_changes_after_profile_cannot_be_applied():
    variables, _, _ = periodic_acquisition()
    result = profile(variables)
    variables.dld_high_voltage[0] += 1.
    with pytest.raises(CalibrationStateError, match='changed'):
        rc.accept_ratio_family(variables, result, icf_anchor=1.6)
    assert not variables.reconstruction_parameters


def test_changing_depth_ratio_fails_heldout_transfer():
    variables, _, _ = periodic_acquisition(changing_ratio=True)
    result = profile(variables)
    assert result['status'] == 'heldout_disagreement' and result['q'] is None


def test_ratio_at_boundary_is_not_supported():
    variables, _, _ = periodic_acquisition(q=.4)
    result = profile(variables)
    assert result['status'] == 'boundary' and result['q'] is None


def test_wide_search_exposes_harmonic_branches_instead_of_unique_answer():
    variables, _, _ = periodic_acquisition()
    result = rc.profile_ratio_geometry(prepare(variables), .2, np.linspace(.34, .6, 521), profile_drop=.15)
    assert result['status'] == 'ambiguous' and result['q'] is None


def test_invalid_hv_is_not_dropped_from_prefix():
    variables, _, _ = periodic_acquisition(n=1000)
    variables.dld_high_voltage[10] = np.nan
    with pytest.raises(CalibrationInputError, match='Every event'):
        prepare(variables)


def test_profile_regions_are_descriptive_not_probabilities():
    q = np.linspace(.4, .6, 101)
    _, _, status = rc._profile_peak(q, np.ones(len(q)), .25)
    assert status in ('flat', 'boundary')
    _, _, status = rc._profile_peak(q, np.maximum(1-1000*(q-.45)**2, 1-1000*(q-.55)**2), .25)
    assert status == 'ambiguous'


def test_underresolved_grid_refines_or_abstains_instead_of_missing_true_period():
    variables, _, _ = periodic_acquisition()
    geometry = prepare(variables)
    result = rc.profile_ratio_geometry(geometry, .2, np.linspace(.4, .6, 9))
    assert len(result['q_grid']) > 9
    assert result['status'] == 'supported'
    assert result['q'] == pytest.approx(.5, abs=.004)
    limited = rc.profile_ratio_geometry(geometry, .2, np.linspace(.4, .6, 9), max_grid_points=9)
    assert limited['status'] == 'grid_underresolved' and limited['q'] is None


def test_ratio_progress_callback_can_cancel_before_any_coherence_evaluation(monkeypatch):
    variables, _, _ = periodic_acquisition()
    geometry = prepare(variables)
    calls = []
    monkeypatch.setattr(rc, 'periodic_band_evidence', lambda *args: calls.append(args))
    def cancel(done, total):
        assert done == 0 and total > 0
        raise CalibrationStateError('cancelled by user')
    with pytest.raises(CalibrationStateError, match='cancelled'):
        rc.profile_ratio_geometry(geometry, .2, np.linspace(.4, .6, 161), progress=cancel)
    assert calls == []


def test_random_subsampling_does_not_invent_planes_from_uniform_event_spacing():
    # With systematic linspace sampling this noncrystalline sequence creates
    # an almost perfect 50-point comb per band and falsely estimates q=.5.
    n, samples = 30000, 50
    x = np.zeros(n)
    x[-1] = 1.  # Full detector extent is independent of the central ROI.
    coefficient = 110.**2*30.**2/(60.*.7*np.pi*100.)
    stride = (n//6-1)/(samples-1)
    voltage = np.sqrt(coefficient*stride/(.2*.5**2))
    variables = SimpleNamespace(dld_x_det=x, dld_y_det=np.zeros(n),
        dld_high_voltage=np.full(n, voltage), pulse_mode='laser', data_backup=None)
    geometry = prepare(variables, max_events_per_band=samples, sample_seed=17)
    same = prepare(variables, max_events_per_band=samples, sample_seed=17)
    other = prepare(variables, max_events_per_band=samples, sample_seed=18)
    np.testing.assert_array_equal(geometry.selected_indices, same.selected_indices)
    assert not np.array_equal(geometry.selected_indices, other.selected_indices)
    assert np.all(np.diff(geometry.selected_indices) > 0)
    assert geometry.metadata['sample_seed'] == 17
    result = rc.profile_ratio_geometry(geometry, .2, np.linspace(.48, .52, 201), minimum_events_per_band=50)
    assert result['q'] is None and result['status'] == 'no_periodic_evidence'
    # Independent full-sequence Fourier amplitude confirms the absence of the
    # aliased period; this is not an assertion mirroring the sampling method.
    full_depth = coefficient*np.arange(1, n//6+1)/voltage**2/.5**2
    assert abs(np.mean(np.exp(2j*np.pi*full_depth/.2))) < .001
