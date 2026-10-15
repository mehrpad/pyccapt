"""Regression cases with independent isotope, range and geometry expectations."""

from types import SimpleNamespace
from threading import Event

import ipywidgets as widgets
import numpy as np
import pandas as pd
import pytest

from pyccapt.calibration.core import ion_selection
from pyccapt.calibration.core.concentration_profile import _assign_species, _range_species
from pyccapt.calibration.core.exceptions import CalibrationInputError
from pyccapt.calibration.core.share_variables import Variables
from pyccapt.calibration.reconstructions import reconstruction, sdm_calibration as sc
from pyccapt.calibration.reconstructions import ratio_calibration as rc
from pyccapt.calibration.reconstructions.species_display import range_row_masks_and_unranged
from pyccapt.calibration.tutorials.tutorials_helpers import helper_3d_reconstruction as h3
from pyccapt.calibration.tutorials.tutorials_helpers import helper_sdm_calibration as hs
from pyccapt.calibration.tutorials.tutorials_helpers.guided_jobs import NotebookJob


def test_manual_isobar_lookup_uses_element_and_isotope(monkeypatch):
    table = pd.DataFrame(dict(element=['Ar', 'K', 'Ca', 'O'], isotope=[40, 40, 40, 16],
                              weight=[39.9624, 39.9640, 39.9626, 15.9949],
                              abundance=[99.6, .0117, 96.941, 99.757]))
    monkeypatch.setattr(ion_selection, '_load_bundled_table', lambda *_: table)
    result = ion_selection.molecule_manual('{40}Ca{16}O2', charge=2).iloc[0]
    assert result['mass'] == pytest.approx(round((39.9626 + 2 * 15.9949) / 2, 4))
    assert result['abundance'] == pytest.approx(round(.96941 * .99757**2, 4))
    assert result['element'] == ['Ca', 'O']
    assert result['complex'] == [1, 2]
    with pytest.raises(CalibrationInputError, match='Ca-41'):
        ion_selection.molecule_manual('{41}Ca1', charge=1)


def test_mixed_isotopes_canonicalize_independent_of_permutation():
    expected = (['C', 'C', 'O'], [12, 13, 16], [1, 2, 1])
    assert ion_selection.transform_combination_and_isotopes(['C', 'C', 'O', 'C'], [12, 13, 16, 13]) == expected
    assert ion_selection.transform_combination_and_isotopes(['O', 'C', 'C', 'C'], [16, 13, 13, 12]) == expected


@pytest.mark.parametrize('overlap', [False, True])
def test_display_and_composition_assign_identical_half_open_ranges(overlap):
    ranges = pd.DataFrame(dict(ion=['Fe', 'Ni'], element=[['Fe'], ['Ni']], complex=[[1], [1]],
                               mc_low=[10., 11. if overlap else 12.], mc_up=[12., 13.]))
    masses = np.array([9., 10., 11., 11.5, 12., 13., np.nan, np.inf])
    assigned = _assign_species(masses, _range_species(ranges))
    expected = [-1, 0, 0, 0, 1, -1, -1, -1]
    np.testing.assert_array_equal(assigned, expected)
    masks, unranged, _ = range_row_masks_and_unranged(masses, ranges)
    np.testing.assert_array_equal(masks[0], assigned == 0)
    np.testing.assert_array_equal(masks[1], assigned == 1)
    np.testing.assert_array_equal(unranged, assigned == -1)
    assert not np.any(masks[0] & masks[1])


def test_invalid_composition_row_cannot_steal_events_from_valid_species():
    ranges = pd.DataFrame(dict(ion=['bad', 'Fe'], element=[['Ni'], "['Fe']"], complex=[[0], '[1]'],
                               mc_low=[10., 10.], mc_up=[12., 12.]))
    masks, unranged, valid = range_row_masks_and_unranged(np.array([10., 11.]), ranges)
    species = _range_species(ranges)
    assert valid == [1] and species[0].index == 1
    assert not masks[0].any() and masks[1].all() and not unranged.any()


@pytest.mark.parametrize('mode', ['Geiser', 'Bas'])
def test_pole_normal_flattens_independently_constructed_cap_tangent(mode):
    # A sphere in canonical reconstruction coordinates has z=H+R-sqrt(R²-x²-y²).
    # Its tangent has dz/dx=ux/uz and dz/dy=uy/uz; this fixes the lateral signs.
    cx, cy, length, icf = 2.0, 1.0, 110., 1.65
    radial = np.hypot(cx, cy) * 10
    observed = np.arctan(radial / length)
    theta = (observed + np.arcsin((icf-1)*np.sin(observed)) if mode == 'Geiser'
             else np.arcsin(icf*radial/length))
    phi = np.arctan2(cy, cx)
    px, py, pz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
    x, y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
    x, y = x.ravel(), y.ravel()
    z = (px*x + py*y) / pz
    normal = sc.pole_axis_from_detector(cx, cy, length, icf, mode=mode)
    _, _, rotated, _ = sc.align_pole_to_z(x, y, z, normal)
    assert np.ptp(rotated) < 1e-12


def test_off_axis_pole_angle_is_angle_between_rays():
    assert sc.observed_pole_angle((30, 0), (-30, 0), 100) == pytest.approx(2*np.degrees(np.arctan(.3)))
    assert sc.observed_pole_angle((0, 0), (30, 0), 100) == pytest.approx(np.degrees(np.arctan(.3)))


@pytest.mark.parametrize('fn', [reconstruction.atom_probe_recons_Bas_et_al,
                               reconstruction.atom_probe_recons_from_detector_Geiser_et_al])
@pytest.mark.parametrize('bad_voltage', [np.nan, np.inf, 0., -1.])
def test_missing_voltage_rejected_before_cumulative_depth(fn, bad_voltage):
    with pytest.raises(CalibrationInputError, match='Every event needs positive finite voltage'):
        fn(np.array([.1, .2, .3]), np.array([.1, .2, .3]), np.array([5000., bad_voltage, 5000.]),
           110., 3.3, .7, 1.65, 30., 60.)


@pytest.mark.parametrize('fn', [reconstruction.atom_probe_recons_Bas_et_al,
                               reconstruction.atom_probe_recons_from_detector_Geiser_et_al])
def test_missing_detector_axis_keeps_later_volume_contribution(fn):
    args = (110., 3.3, .7, 1.65, 30., 60.)
    hv = np.full(3, 5000.)
    x, y, z = fn(np.array([.1, np.nan, .3]), np.array([.1, .2, .3]), hv, *args)
    baseline = fn(np.array([.1, .2, .3]), np.array([.1, .2, .3]), hv, *args)[2]
    assert np.isnan(z[1])
    assert z[2] == pytest.approx(baseline[2])


def test_sweep_without_resolvable_pairs_has_no_best():
    variables = SimpleNamespace(dld_x_det=np.full(10, .1), dld_y_det=np.full(10, .1),
                                dld_high_voltage=np.full(10, 5000.), pulse_mode='laser')
    result = sc.grid_sweep_kf_icf(variables, (.1, .1), .2, .7, 30., 60., 110.,
                                  [3., 3.5, 4.], [1.4, 1.5, 1.6],
                                  bin_size=1e-8, z_max=1e-6, lateral_radius=1e-5)
    assert result['best'] is None
    assert result['status'] == 'no_evidence'


@pytest.mark.parametrize('case,status', [('tie', 'ambiguous'), ('edge', 'boundary'), ('spacing', 'no_evidence'),
                                       ('interior', 'candidate')])
def test_sweep_only_exposes_supported_interior_candidate(monkeypatch, case, status):
    variables = SimpleNamespace(dld_x_det=np.array([.1, .1]), dld_y_det=np.array([.1, .1]))
    monkeypatch.setattr(sc, 'reconstruct_xyz', lambda *a, **k: (np.zeros(2), np.zeros(2), np.zeros(2)))
    monkeypatch.setattr(sc, 'z_sdm', lambda *a, **k: (np.arange(10), np.ones(10)))
    monkeypatch.setattr(sc, 'find_sdm_peaks', lambda *a, **k: {'spacing': .2})
    scores = np.ones(9)
    if case != 'tie':
        scores[0 if case == 'edge' else 4] = 10.
    iterator = iter(scores)
    monkeypatch.setattr(sc, 'sdm_peakiness', lambda *a, **k: next(iterator))
    result = sc.grid_sweep_kf_icf(variables, (.1, .1), .2, .7, 30., 60., 110.,
                                [3., 3.5, 4.], [1.4, 1.5, 1.6],
                                expected_spacing=.4 if case == 'spacing' else .2)
    assert result['status'] == status
    if case == 'interior':
        assert result['best']['kf'] == 3.5 and result['best']['icf'] == 1.5
    else:
        assert result['best'] is None


def _widgets(root):
    yield root
    for child in getattr(root, 'children', ()):
        yield from _widgets(child)


def _button(panel, label):
    return next(w for w in _widgets(panel) if isinstance(w, widgets.Button) and w.description == label)


def _queued_ratio_jobs(monkeypatch):
    jobs, callbacks = [], []
    def factory():
        job = NotebookJob(dispatch=callbacks.append)
        jobs.append(job)
        return job
    monkeypatch.setattr(hs, 'NotebookJob', factory)
    return jobs, callbacks


def _finish_ratio_job(jobs, callbacks):
    jobs[0].thread.join(timeout=5)
    assert not jobs[0].thread.is_alive()
    while callbacks:
        callbacks.pop(0)()


def test_accepted_sdm_parameters_reach_actual_reconstruction(monkeypatch):
    variables = Variables()
    variables.dld_x_det = np.zeros(60)
    variables.dld_y_det = np.zeros(60)
    variables.dld_high_voltage = np.full(60, 5000.)
    flight = widgets.FloatText(value=110.)
    element = SimpleNamespace(value=('Al', 60., 30.))
    monkeypatch.setattr(hs, 'display', lambda *_: None)
    monkeypatch.setattr(h3, 'display', lambda *_: None)
    monkeypatch.setattr(reconstruction.plt, 'show', lambda: None)
    jobs, callbacks = _queued_ratio_jobs(monkeypatch)
    monkeypatch.setattr(rc, 'profile_icf_kf_ratio', lambda *a, **k: dict(
        status='supported', message='synthetic profile', q=.4, supported_interval=[.39, .41],
        training_profile=[], heldout_profile=[], metadata=dict(
            source_hash=rc._hash_source(variables), icf_bounds=[1.1, 1.9], det_eff=.7,
            field_evap=30., avg_dens=60., flight_path_length=110., mode='Geiser')))
    old_panel = h3.call_x_y_z_calculation(variables, flight, element, display_panel=False)
    panel = old_panel.controls['tuning']
    old_panel.controls['parameters']['icf'].value = 1.65  # Explicitly choose the anchor shared by both tabs.
    for widget in _widgets(panel):
        if isinstance(widget, widgets.Checkbox):
            widget.value = True
    _button(panel, 'Profile ICF/kf ratio').click()
    _finish_ratio_job(jobs, callbacks)
    assert not variables.reconstruction_parameters
    _button(panel, 'Accept anchored family').click()
    assert variables.reconstruction_parameters['kf'] == pytest.approx(4.125)
    assert variables.reconstruction_parameters['icf'] == 1.65  # Explicit preserved anchor.
    calls = []
    monkeypatch.setattr(reconstruction, 'x_y_z_calculation_and_plot', lambda **kw: calls.append(kw))
    _button(old_panel, 'Load accepted parameters').click()
    _button(old_panel, 'Reconstruct & plot').click()
    assert calls[-1]['kf'] == pytest.approx(4.125) and calls[-1]['icf'] == 1.65
    assert old_panel.reconstruction_tabs.selected_index == 0
    new_panel = h3.call_x_y_z_calculation(variables, flight, element, display_panel=False)
    _button(new_panel, 'Reconstruct & plot').click()
    assert calls[-1]['kf'] == pytest.approx(4.125) and calls[-1]['icf'] == 1.65


def _background_ratio_panel(monkeypatch, profiler):
    variables = Variables()
    variables.dld_x_det = np.zeros(60)
    variables.dld_y_det = np.zeros(60)
    variables.dld_high_voltage = np.full(60, 5000.)
    flight = widgets.FloatText(value=110.)
    monkeypatch.setattr(hs, 'display', lambda *_: None)
    monkeypatch.setattr(rc, 'profile_icf_kf_ratio', profiler)
    jobs, callbacks = _queued_ratio_jobs(monkeypatch)
    panel = hs.call_sdm_icf_kf_calibration(variables, flight, SimpleNamespace(value=('Al', 60., 30.)), display_panel=False)
    for widget in _widgets(panel):
        if isinstance(widget, widgets.Checkbox):
            widget.value = True
    return variables, flight, panel, jobs, callbacks


def _supported_ratio(variables, *args, **kwargs):
    kwargs['progress'](1, 4)
    return dict(status='supported', message='synthetic profile', q=.4, supported_interval=[.39, .41],
        training_profile=[], heldout_profile=[], metadata=dict(source_hash=rc._hash_source(variables),
            icf_bounds=[1.1, 1.9], det_eff=.7, field_evap=30., avg_dens=60., flight_path_length=110., mode='Geiser'))


def test_ratio_background_cancel_interrupts_progress_without_accepting_state(monkeypatch):
    started, release = Event(), Event()
    def profiler(*args, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        return _supported_ratio(*args, **kwargs)
    variables, _, panel, jobs, callbacks = _background_ratio_panel(monkeypatch, profiler)
    _button(panel, 'Profile ICF/kf ratio').click()
    assert started.wait(timeout=5) and jobs[0].running
    assert _button(panel, 'Accept anchored family').disabled
    assert _button(panel, 'Apply suggested kf').disabled
    assert not _button(panel, 'Cancel ratio profile').disabled
    _button(panel, 'Accept anchored family').click()  # Also guarded if invoked programmatically.
    _button(panel, 'Cancel ratio profile').click()
    release.set()
    _finish_ratio_job(jobs, callbacks)
    _button(panel, 'Accept anchored family').click()
    assert not variables.reconstruction_parameters and not variables.reconstruction_ratio_result
    assert not _button(panel, 'Profile ICF/kf ratio').disabled
    assert not _button(panel, 'Apply suggested kf').disabled
    assert _button(panel, 'Cancel ratio profile').disabled


@pytest.mark.parametrize('change', ['cancel', 'flight', 'source', 'selection'])
def test_completed_ratio_result_is_discarded_if_cancelled_or_stale(monkeypatch, change):
    variables, flight, panel, jobs, callbacks = _background_ratio_panel(monkeypatch, _supported_ratio)
    _button(panel, 'Profile ICF/kf ratio').click()
    jobs[0].thread.join(timeout=5)
    assert jobs[0].running  # Completion is queued for the notebook UI thread.
    if change == 'cancel':
        _button(panel, 'Cancel ratio profile').click()
    elif change == 'flight':
        flight.value = 120.
    elif change == 'selection':
        variables.reconstruction_selection_mask = np.ones(60, dtype=bool)
        variables.reconstruction_selection_mask[0] = False
    else:
        variables.dld_high_voltage[0] += 10.
    _finish_ratio_job(jobs, callbacks)
    _button(panel, 'Accept anchored family').click()
    assert not variables.reconstruction_parameters and not variables.reconstruction_ratio_result
    assert not _button(panel, 'Profile ICF/kf ratio').disabled
    assert any(isinstance(widget, widgets.FloatProgress) and widget.value == .25 for widget in _widgets(panel))


def test_showing_density_candidates_preserves_roi_until_review_and_resets_spacing_confirmation(monkeypatch):
    variables = Variables()
    variables.dld_x_det = np.ones(60)
    variables.dld_y_det = np.zeros(60)
    variables.dld_high_voltage = np.full(60, 5000.)
    monkeypatch.setattr(hs, 'display', lambda *_: None)
    monkeypatch.setattr(reconstruction.plt, 'show', lambda: None)
    monkeypatch.setattr(hs.sc, 'sharp_fdm_map', lambda *args, **kwargs: (
        np.ones((2,2)), np.ones((2,2)), np.array([-1.,0.,1.]), np.array([-1.,0.,1.])))
    monkeypatch.setattr(hs.sc, 'detect_pole_candidates', lambda *args, **kwargs: [(.2,.3,5.)])
    panel = hs.call_sdm_icf_kf_calibration(variables, widgets.FloatText(value=110.), SimpleNamespace(value=('Al',60.,30.)), display_panel=False)
    confirmation = next(w for w in _widgets(panel) if isinstance(w, widgets.Checkbox) and 'Indexed spacing' in w.description)
    confirmation.value = True
    _button(panel, 'Show FDMs').click()
    candidate = next(w for w in _widgets(panel) if isinstance(w, widgets.Dropdown) and any('unindexed ROI' in str(opt) for opt in w.options))
    assert candidate.value is None and confirmation.value  # Map inspection does not alter the reference.
    candidate.value = (.2,.3)
    assert not confirmation.value  # A different ROI needs a newly confirmed plane assignment.
    reconstruction.plt.close('all')


def test_reconstruct_button_recovers_after_downstream_failure(monkeypatch):
    monkeypatch.setattr(h3, 'display', lambda *_: None)
    panel = h3.call_x_y_z_calculation(Variables(), widgets.FloatText(value=110.),
                                     SimpleNamespace(value=('Al', 60., 30.)), display_panel=False)
    def fail(**kwargs):
        raise CalibrationInputError('invalid input')
    monkeypatch.setattr(reconstruction, 'x_y_z_calculation_and_plot', fail)
    button = _button(panel, 'Reconstruct & plot')
    button.click()  # ipywidgets captures callback exceptions.
    assert not button.disabled


def _plot_variables():
    variables = Variables()
    variables.mc = np.array([10., 11., 12., 13.])
    variables.mc_uc = variables.mc.copy()
    variables.mc_calib = variables.mc.copy()
    variables.dld_t = np.arange(4.)
    variables.x = np.array([1., 2., 3., 4.])
    variables.y = variables.x.copy()
    variables.z = variables.x.copy()
    variables.dld_x_det = variables.x.copy()
    variables.dld_y_det = variables.x.copy()
    variables.volume = np.zeros(4)
    variables.range_data = pd.DataFrame(dict(ion=['A', 'B'], color=['red', 'blue'],
                                             element=[['A'], ['B']], mc_low=[10., 11.], mc_up=[12., 14.]))
    return variables


def test_projection_retains_all_species_after_crop(monkeypatch):
    variables = _plot_variables()
    monkeypatch.setattr(reconstruction.plt, 'show', lambda: None)
    reconstruction.projection(variables, [1., 1.], range_mc=[9., 14.])
    collections = reconstruction.plt.gca().collections
    np.testing.assert_array_equal(collections[0].get_offsets()[:, 0], [1., 2.])
    np.testing.assert_array_equal(collections[1].get_offsets()[:, 0], [3., 4.])
    reconstruction.plt.close('all')


def test_animation_masks_follow_filtered_events_and_honor_frame_size(monkeypatch):
    variables = _plot_variables()
    variables.selected_x1, variables.selected_x2 = 2., 4.
    variables.selected_y1, variables.selected_y2 = 2., 4.
    variables.selected_z1, variables.selected_z2 = 2., 4.
    observed = []
    class FakeAnimation:
        def __init__(self, fig, callback, frames, interval):
            assert frames == 2  # 3 selected events, including the final one-event frame.
            for frame in range(frames):
                callback(frame)
                observed.append([c.get_offsets().copy() for c in fig.axes[0].collections])
        def to_jshtml(self):
            return '<animation>'
    monkeypatch.setattr(reconstruction, 'FuncAnimation', FakeAnimation)
    reconstruction.detector_animation(variables, 2, True, True, False, 'test', (3, 3), False)
    np.testing.assert_array_equal(observed[0][0][:, 0], [2.])
    np.testing.assert_array_equal(observed[0][1][:, 0], [3.])
    np.testing.assert_array_equal(observed[1][1][:, 0], [4.])


def test_heatmap_crop_without_sequence_uses_canonical_assignments(monkeypatch):
    variables = _plot_variables()
    monkeypatch.setattr(reconstruction.plt, 'show', lambda: None)
    reconstruction.heatmap(variables, [1., 1.], range_mc=[9., 14.])
    collections = reconstruction.plt.gca().collections
    # Existing plot positions scale cm to mm; canonical assignment avoids overlap duplicates.
    np.testing.assert_array_equal(collections[0].get_offsets()[:, 0], [10., 20.])
    np.testing.assert_array_equal(collections[1].get_offsets()[:, 0], [30., 40.])
    reconstruction.plt.close('all')
