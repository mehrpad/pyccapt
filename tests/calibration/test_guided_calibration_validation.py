"""Counterexamples for balanced fitting and conservative calibration replacement."""
import numpy as np
import pandas as pd
import pytest

from pyccapt.calibration.core.exceptions import CalibrationStateError
from pyccapt.calibration.core.guided_calibration import GuidedCalibrationSession, ValidationPolicy
from pyccapt.calibration.core.multireference_tof import MultiReferenceToFModel, ToFModelConfig
from pyccapt.calibration.core.share_variables import Variables
from tests.calibration.test_guided_calibration import known_data, session_for


def test_reference_replication_does_not_change_weighted_robust_fit():
    mass = np.repeat([16., 64.], 50)
    residual = np.r_[np.tile([0., 0., 0., 0., -6.], 10), np.tile([0., 0., 0., 0., 8.], 10)]
    features = np.column_stack((np.full(100, 5000.), np.zeros((100, 2)),
                                np.linspace(0., 1., 100), np.full(100, 100.)))
    target = np.column_stack((20.+100.*np.sqrt(mass)+residual, mass))
    config = ToFModelConfig(voltage_term=False, spatial_terms=False)
    balanced = MultiReferenceToFModel(config).fit(features, target)
    indices = np.r_[np.tile(np.arange(50), 20), np.arange(50, 100)]
    replicated = MultiReferenceToFModel(config).fit(features[indices], target[indices])
    # Repeating one reference changes abundance, not the information or the
    # intended equal-reference objective, even for non-Gaussian tails.
    assert replicated.t0_ns == pytest.approx(balanced.t0_ns, abs=2e-5)
    np.testing.assert_allclose(replicated.predict_factor(features), balanced.predict_factor(features), atol=1e-8)
    assert replicated.fit_diagnostics["robust_cutoff_ns"] == 1.


def test_within_absolute_limits_but_worse_baseline_cannot_reach_confirmation():
    variables, refs, roles, truth = known_data()
    variables.dld_t[roles["selection"]] += .2
    before = variables.mc_calib.copy()
    session = session_for(variables, refs, roles, baseline_mass_da=truth)
    session.fit()
    result = session.validate_selection()
    assert all(row["passed_absolute_limits"] for row in result["groups"])
    assert not result["passed"] and session.status == "selection_failed"
    assert all(not row["baseline_comparison"]["passed"] for row in result["groups"])
    with pytest.raises(CalibrationStateError, match="Selection validation must pass"):
        session.confirm()
    assert "confirmation" not in session.diagnostics
    np.testing.assert_array_equal(variables.mc_calib, before)


def test_temporal_bias_cancellation_cannot_hide_in_global_validation_average():
    variables, refs, roles, _ = known_data()
    event = np.linspace(0., 1., len(variables.dld_t))
    # Remove the known block step, then inject equal/opposite selection-only
    # mass errors. They cancel globally and pass the old aggregate RMSE gate.
    variables.dld_t[:] = 17.4+(variables.dld_t-17.4)/np.exp(.006*(event >= .5))
    error = np.where(event < .5, .008, -.008)
    selected = roles["selection"]
    variables.dld_t[selected] = 17.4+(variables.dld_t[selected]-17.4)*np.sqrt(1.+error[selected])
    config = ToFModelConfig(linear_drift=True)
    global_only = GuidedCalibrationSession(variables, refs, roles, mode="mc", flight_length_mm=110.,
        model_config=config, policy=ValidationPolicy(validation_time_bins=1))
    global_only.fit()
    assert global_only.validate_selection()["passed"]
    session = GuidedCalibrationSession(variables, refs, roles, mode="mc", flight_length_mm=110., model_config=config)
    session.fit()
    result = session.validate_selection()
    assert not result["passed"]
    assert all(not row["passed_absolute_limits"] for row in result["groups"])
    assert len(result["groups"]) == 12


def test_insufficient_time_band_is_reported_without_consuming_confirmation():
    variables, refs, roles, _ = known_data()
    # Overall declared-model-block counts remain adequate; one smaller
    # validation band has no held-out coverage.
    roles["selection"][refs[13.5] & (np.arange(len(variables.dld_t)) >= 900) &
                       (np.arange(len(variables.dld_t)) < 1350)] = False
    session = session_for(variables, refs, roles)
    session.fit()
    failed = session.validate_selection()
    assert not failed["passed"]
    assert any("insufficient events" in row.get("reason", "") for row in failed["groups"])
    assert "confirmation" not in session.diagnostics


def test_recorded_calibrated_mass_is_baseline_not_default_uncorrected_active_array():
    variables, refs, roles, truth = known_data()
    frame = variables.data.copy()
    frame["mc (Da)"] = truth
    frame["mc_uc (Da)"] = truth*1.1
    variables.sync_from_data(frame)
    assert not np.array_equal(variables.mc_calib, truth)
    session = session_for(variables, refs, roles)
    np.testing.assert_array_equal(session.baseline_mass, truth)
    assert session.preview()["baseline_comparison"] == "recorded mc (Da) column"
    assert not session.baseline_mass.flags.writeable


def test_placeholder_mass_is_not_treated_as_a_comparison_baseline():
    variables, refs, roles, _ = known_data()
    frame = variables.data.copy()
    frame["mc (Da)"] = 0.
    variables.sync_from_data(frame)
    session = session_for(variables, refs, roles)
    session.fit()
    result = session.validate_selection()
    assert result["passed"] and session.baseline_mass is None
    assert all(not row["baseline_comparison"]["available"] for row in result["groups"])


def test_nonpositive_heldout_elapsed_time_is_a_failed_validation_not_an_exception():
    variables, refs, roles, _ = known_data()
    variables.dld_t[np.flatnonzero(roles["selection"])[0]] = 16.
    session = session_for(variables, refs, roles)
    session.fit()
    result = session.validate_selection()
    assert not result["passed"]
    assert sum(row["nonpositive_elapsed"] for row in result["groups"]) == 1
    assert session.status == "selection_failed"


def test_seeded_fit_sampling_avoids_regular_stride_alias_of_periodic_tail():
    index = np.arange(6006)
    mass = np.where(index % 2 == 0, 16., 64.)
    roles = {name: (index//2) % 3 == i for i, name in enumerate(("train", "selection", "confirmation"))}
    tail = np.where((index//6) % 10 == 0, 5., 0.)
    raw = 17.4+np.sqrt(mass*1.66e-27/(2*1.6e-19*5000.))*.110*1e9+tail
    variables = Variables()
    variables.sync_from_data(pd.DataFrame({"t (ns)": raw, "high_voltage (V)": 5000., "pulse_v (V)": 0.,
        "x_det (cm)": 0., "y_det (cm)": 0., "mc (Da)": mass*1.1, "mc_uc (Da)": mass*1.1}))
    refs = {value: mass == value for value in (16., 64.)}
    config = ToFModelConfig(voltage_term=False, spatial_terms=False)
    session = GuidedCalibrationSession(variables, refs, roles, mode="mc", flight_length_mm=110.,
        model_config=config, max_fit_events_per_group=101, fit_seed=2)
    session.fit()
    assert session.validate_selection()["passed"]
    previous_indices = np.concatenate([np.flatnonzero(roles["train"] & mask)[np.linspace(0, 1000, 101, dtype=int)]
                                       for mask in refs.values()])
    aliased = MultiReferenceToFModel(config).fit(session.features[previous_indices],
        np.column_stack((raw[previous_indices], mass[previous_indices])))
    selected = roles["selection"] & refs[16.]
    old_error = (aliased.predict_mass_da(session.features[selected], raw[selected])-16.)/16.
    assert abs(np.median(old_error)) > .02
    original_hash = session.diagnostics["training"]["fit_event_hash"]
    session.fit()
    assert session.diagnostics["training"]["fit_event_hash"] == original_hash


@pytest.mark.parametrize("linear_drift", [False, True])
def test_contiguous_holdout_checks_stationarity_but_never_extrapolates_fitted_drift(linear_drift):
    variables, refs, _, truth = known_data()
    event = np.linspace(0., 1., len(truth))
    # Make a truly stationary specimen, independently removing known synthetic
    # event-dependent terms; a stationary model can be tested on later events.
    variables.dld_t[:] = 17.4+(variables.dld_t-17.4)/np.exp(.004*event+.006*(event >= .5))
    roles = dict(train=event < .6, selection=(event >= .6) & (event < .8), confirmation=event >= .8)
    session = GuidedCalibrationSession(variables, refs, roles, mode="mc", flight_length_mm=110.,
        model_config=ToFModelConfig(linear_drift=linear_drift))
    session.fit()
    selected = session.validate_selection()
    assert selected["evaluated_time_bands"] == [2, 3]
    if linear_drift:
        assert not selected["passed"]
        assert any(row["outside_domain"] for row in selected["groups"])
    else:
        assert selected["passed"]
        confirmed = session.confirm()
        assert confirmed["passed"] and confirmed["evaluated_time_bands"] == [3]


def test_unseen_declared_acquisition_block_remains_outside_direct_model_domain():
    mass = np.tile([16., 64.], 50)
    features = np.column_stack((np.full(100, 5000.), np.zeros((100, 2)),
                                np.linspace(0., .4, 100), np.full(100, 100.)))
    target = np.column_stack((20.+100.*np.sqrt(mass), mass))
    model = MultiReferenceToFModel(ToFModelConfig(voltage_term=False, spatial_terms=False, block_edges=(.5,)))
    model.fit(features, target)
    later = features[:1].copy()
    later[:, 3] = .8
    assert not model.domain_mask(later).any()


def test_application_selection_cannot_erase_heldout_validation_evidence():
    variables, refs, roles, _ = known_data(confirmation_error_ns=15.)
    # It is legitimate to apply to a subset, but this cannot turn a bad
    # confirmation population into a successful zero-group evaluation.
    session = session_for(variables, refs, roles, apply_mask=roles["train"])
    session.fit()
    selected = session.validate_selection()
    assert selected["passed"] and len(selected["groups"]) == 12
    confirmed = session.confirm()
    assert not confirmed["passed"] and len(confirmed["groups"]) == 12
    assert session.status == "confirmation_failed"
    with pytest.raises(CalibrationStateError, match="confirmation"):
        session.accept()
