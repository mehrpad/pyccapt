"""Known-truth asymmetric spectra and explicit ambiguity/identity boundaries."""
import json

import numpy as np
import pytest

from pyccapt.calibration.core.exceptions import CalibrationInputError
from pyccapt.calibration.core.isotope_family import (
    EmpiricalTemplate, build_empirical_template, fit_isotope_family,
)


def _template():
    offsets = np.linspace(-.4, .8, 601)
    shape = np.exp(-.5*(offsets/.045)**2)+.45*np.exp(-np.maximum(offsets, 0.)/.15)*(offsets >= 0)
    return build_empirical_template(20.+offsets, shape*1000+2., reference_mass=20., background=2.)


def _spectrum(template, mass, isotope_masses, areas, shift, background):
    midpoint = (mass[1:]+mass[:-1])/2
    edges = np.r_[mass[0]-(midpoint[0]-mass[0]), midpoint, mass[-1]+(mass[-1]-midpoint[-1])]
    columns = np.diff(template.integral(edges[:, None]-np.asarray(isotope_masses)[None, :]-shift), axis=0)
    return columns@np.asarray(areas)+background*np.diff(edges)


def test_asymmetric_family_overlap_recovers_shared_offset_and_areas():
    template = _template()
    mass = np.linspace(19.4, 22.0, 1301)
    isotopes = [20., 20.21, 21.02]
    areas = [12000., 8000., 4000.]
    expected = _spectrum(template, mass, isotopes, areas, .023, 25.)
    counts = np.random.default_rng(31415).poisson(expected)
    original = counts.copy()
    result = fit_isotope_family(mass, counts, isotope_masses=isotopes, template=template,
                                shift_bounds=(-.06, .06))
    assert result.mass_offset == pytest.approx(.023, abs=.002)
    np.testing.assert_allclose(result.amplitudes, areas, rtol=.12)
    np.testing.assert_array_equal(result.fitted_centres, np.asarray(isotopes)+result.mass_offset)
    np.testing.assert_array_equal(counts, original)
    assert result.diagnostics["validation"]["bins"] == 260
    assert result.diagnostics["validation"]["poisson_deviance_per_bin"] < 1.5
    assert result.uncertainty["mass_offset_std"] > 0
    assert result.provenance["abundance_prior"] is None
    json.dumps(result.summary(), allow_nan=False)


def test_noiseless_asymmetric_truth_and_nonuniform_bins():
    template = _template()
    mass = np.r_[np.linspace(19.4, 20.4, 501), np.linspace(20.402, 22.0, 400)]
    counts = _spectrum(template, mass, [20., 21.], [10000., 4000.], -.017, 100.)
    result = fit_isotope_family(mass, counts, isotope_masses=[20., 21.], template=template)
    assert result.mass_offset == pytest.approx(-.017, abs=1e-6)
    np.testing.assert_allclose(result.amplitudes, [10000., 4000.], rtol=1e-5)
    assert result.background_density == pytest.approx(100., rel=1e-5)
    assert result.diagnostics["validation"]["rmse_counts"] < 1e-3


def test_nearly_coincident_candidates_require_review_not_confident_identity():
    template = _template()
    mass = np.linspace(19.4, 21., 801)
    counts = _spectrum(template, mass, [20., 20.0001], [5000., 5000.], .01, 10.)
    result = fit_isotope_family(mass, counts, isotope_masses=[20., 20.0001], template=template)
    assert result.diagnostics["ambiguous"]
    assert "strongly_overlapping_component_shapes" in result.diagnostics["flags"]
    assert result.diagnostics["review_required"]
    assert "probability" not in result.summary()


def test_enriched_family_has_no_implicit_natural_abundance_constraint():
    template = _template()
    mass = np.linspace(19.4, 22.0, 1001)
    counts = _spectrum(template, mass, [20., 21.], [1000., 10000.], .01, 100.)
    free = fit_isotope_family(mass, counts, isotope_masses=[20., 21.], template=template)
    disabled = fit_isotope_family(mass, counts, isotope_masses=[20., 21.], template=template,
                                  abundance_prior=[.9, .1], prior_strength=0.)
    np.testing.assert_array_equal(free.amplitudes, disabled.amplitudes)
    assert free.amplitudes[1]/free.amplitudes[0] == pytest.approx(10., rel=1e-4)
    regularized = fit_isotope_family(mass, counts, isotope_masses=[20., 21.], template=template,
                                     abundance_prior=[.9, .1], prior_strength=1e7)
    assert regularized.amplitudes[1]/regularized.amplitudes[0] < free.amplitudes[1]/free.amplitudes[0]
    assert regularized.provenance["prior_strength"] == 1e7


def test_holdout_values_never_affect_parameter_fit():
    template = _template()
    mass = np.linspace(19.4, 22., 1001)
    counts = _spectrum(template, mass, [20., 21.], [10000., 4000.], .012, 100.)
    validation = np.arange(len(mass)) % 7 == 0
    first = fit_isotope_family(mass, counts, isotope_masses=[20., 21.], template=template,
                               validation_mask=validation)
    modified = counts.copy()
    modified[validation] += 100
    second = fit_isotope_family(mass, modified, isotope_masses=[20., 21.], template=template,
                                validation_mask=validation)
    assert first.mass_offset == second.mass_offset
    np.testing.assert_array_equal(first.amplitudes, second.amplitudes)
    assert second.diagnostics["validation"]["rmse_counts"] > first.diagnostics["validation"]["rmse_counts"]


def test_bound_solution_and_unseen_component_are_flagged():
    template = _template()
    mass = np.linspace(19.4, 21., 801)
    counts = _spectrum(template, mass, [20.], [10000.], .08, 100.)
    result = fit_isotope_family(mass, counts, isotope_masses=[20., 25.], template=template,
                                shift_bounds=(-.02, .02))
    assert "mass_offset_at_search_boundary" in result.diagnostics["flags"]
    assert "template_support_truncated" in result.diagnostics["flags"]
    assert "local_parameters_not_identifiable" in result.diagnostics["flags"]
    assert result.uncertainty["mass_offset_std"] is None
    json.dumps(result.summary(), allow_nan=False)


@pytest.mark.parametrize("bad", [[0., np.nan, 1.], [0., 0., 0.], [-1., 2., 3.]])
def test_invalid_reference_counts_rejected(bad):
    with pytest.raises(CalibrationInputError):
        build_empirical_template([19.9, 20., 20.1], bad, reference_mass=20.)


def test_template_owns_reference_arrays_and_rejects_invalid_family():
    offsets = np.array([-.1, 0., .1])
    density = np.array([0., 10., 0.])
    template = EmpiricalTemplate(offsets, density, 20.)
    density[:] = 99
    assert template.density[0] == 0.
    with pytest.raises(ValueError):
        template.density[0] = 5
    mass = np.linspace(19., 21., 100)
    with pytest.raises(CalibrationInputError, match="distinct"):
        fit_isotope_family(mass, np.ones(100), isotope_masses=[20., 20.], template=template)
    with pytest.raises(CalibrationInputError, match="explicit abundance"):
        fit_isotope_family(mass, np.ones(100), isotope_masses=[20.], template=template, prior_strength=1.)


@pytest.mark.parametrize('bad', ['invalid', [1., 2.], np.nan])
def test_invalid_scalar_options_use_calibration_exceptions(bad):
    with pytest.raises(CalibrationInputError):
        build_empirical_template([19.9, 20., 20.1], [0., 10., 0.], reference_mass=bad)
    with pytest.raises(CalibrationInputError):
        fit_isotope_family(np.linspace(19., 21., 100), np.ones(100), isotope_masses=[20.],
                           template=_template(), prior_strength=bad)


def test_omitted_holdout_and_empty_count_holdout_report_limited_evidence():
    template = _template()
    mass = np.linspace(19.4, 21., 501)
    counts = _spectrum(template, mass, [20.], [5000.], .01, 10.)
    no_holdout = fit_isotope_family(mass, counts, isotope_masses=[20.], template=template,
                                    validation_mask=np.zeros(len(mass), dtype=bool))
    assert no_holdout.diagnostics['validation'] is None
    assert 'no_held_out_bins' in no_holdout.diagnostics['flags']
    validation = np.arange(len(mass)) % 5 == 0
    counts[validation] = 0.
    zero_holdout = fit_isotope_family(mass, counts, isotope_masses=[20.], template=template,
                                      validation_mask=validation)
    assert zero_holdout.diagnostics['validation']['observed_counts'] == 0.
    assert 'held_out_bins_have_no_observed_counts' in zero_holdout.diagnostics['flags']
    assert zero_holdout.diagnostics['validation']['poisson_deviance_per_bin'] > 0.


def test_no_supported_component_cannot_claim_identifiability():
    result = fit_isotope_family(np.linspace(19., 21., 100), np.ones(100),
                                isotope_masses=[100.], template=_template())
    assert result.diagnostics['review_required']
    assert 'template_support_truncated' in result.diagnostics['flags']
    assert 'local_parameters_not_identifiable' in result.diagnostics['flags']
    assert result.uncertainty['mass_offset_std'] is None
    assert result.uncertainty['amplitude_std'] == [None]
    np.testing.assert_array_equal(result.components, 0.)
    json.dumps(result.summary(), allow_nan=False)


def test_positive_counts_only_in_holdout_cannot_fit_family():
    validation = np.arange(100) % 5 == 0
    with pytest.raises(CalibrationInputError, match='Training bins require positive'):
        fit_isotope_family(np.linspace(19., 21., 100), validation.astype(float),
                           isotope_masses=[20.], template=_template(), validation_mask=validation)
