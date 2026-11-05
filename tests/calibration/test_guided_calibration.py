"""Known-truth and state-boundary tests for optional physical ToF calibration."""
import json

import numpy as np
import pandas as pd
import pytest

from pyccapt.calibration.core.exceptions import CalibrationInputError, CalibrationStateError
from pyccapt.calibration.core.guided_calibration import GuidedCalibrationSession
from pyccapt.calibration.core.multireference_tof import ToFModelConfig, MultiReferenceToFModel
from pyccapt.calibration.core.share_variables import Variables
from pyccapt.calibration.mc.mc_tools import tof2mc


def known_data(n=1800, *, confirmation_error_ns=0., outside_domain=False, missing=False):
    rng = np.random.default_rng(541)
    index = np.arange(n)
    mass = np.array([13.5, 27., 54.])[index % 3]
    voltage = rng.uniform(4200, 5800, n)
    x, y = rng.uniform(-2, 2, (2, n))
    event = index / (n - 1)
    roles = {name: (index // 3) % 3 == i for i, name in enumerate(("train", "selection", "confirmation"))}
    if outside_domain:
        voltage[np.flatnonzero(roles["confirmation"])[0]] = 9000.
    # Independent generator from electrostatic energy, flight geometry and declared drift.
    flight_m = np.sqrt(.110**2 + (.01*x)**2 + (.01*y)**2)
    ideal_ns = np.sqrt(mass * 1.66e-27 / (2 * 1.6e-19 * voltage)) * flight_m * 1e9
    log_scale = (.015 + .00002*(voltage-5000) + .001*x + .0015*y + .0002*x*x
                 - .0001*x*y + .0003*y*y + .004*event + .006*(event >= .5))
    raw_t = 17.4 + ideal_ns * np.exp(log_scale)
    raw_t[roles["confirmation"]] += confirmation_error_ns
    references = {value: mass == value for value in np.unique(mass)}
    if missing:
        x[0] = np.nan
        for mask in references.values():
            mask[0] = False
    variables = Variables()
    variables.sync_from_data(pd.DataFrame({"t (ns)": raw_t, "high_voltage (V)": voltage,
                                           "pulse_v (V)": np.zeros(n),
                                           "x_det (cm)": x, "y_det (cm)": y,
                                           "mc_uc (Da)": mass * 1.1, "mc (Da)": mass * 1.1}))
    return variables, references, roles, mass


def session_for(variables, references, roles, mode="mc", **kwargs):
    return GuidedCalibrationSession(variables, references, roles, mode=mode, flight_length_mm=110.,
                                    model_config=ToFModelConfig(linear_drift=True, block_edges=(.5,)), **kwargs)


def accepted(session):
    session.fit()
    assert session.validate_selection()["passed"]
    assert session.confirm()["passed"]
    session.accept()
    return session


@pytest.mark.parametrize("mode", ["mc", "tof"])
def test_known_offset_scale_drift_recovery_and_no_change_until_accept(mode):
    variables, references, roles, mass = known_data()
    original = variables.get_calibration_array(mode).copy()
    raw = variables.dld_t.copy()
    session = session_for(variables, references, roles, mode)
    assert session.preview()["events"] == len(mass)
    session.fit()
    assert session.model.t0_ns == pytest.approx(17.4, abs=1e-6)
    with pytest.raises(CalibrationStateError, match="acceptance"):
        session.apply(active_mode=mode)
    np.testing.assert_array_equal(variables.get_calibration_array(mode), original)
    assert session.validate_selection()["passed"]
    assert session.confirm()["passed"]
    session.accept()
    np.testing.assert_array_equal(variables.get_calibration_array(mode), original)
    output = session.apply(active_mode=mode, chunk_size=17)
    result = output if mode == "mc" else session.convert_applied_tof_to_mass()
    np.testing.assert_allclose(result, mass, rtol=1e-9)
    np.testing.assert_array_equal(variables.dld_t, raw)
    np.testing.assert_array_equal(getattr(variables, variables._CALIBRATION_ATTR[mode] + "_backup"), original)
    assert variables.guided_calibration_receipt["output"]["excluded_policy"] == "NaN"
    assert "not coverage probabilities" in session.model.fit_diagnostics["uncertainty_kind"]


def test_normalized_tof_public_conversion_uses_reference_geometry_and_zero_offset():
    variables, refs, roles, mass = known_data()
    session = accepted(session_for(variables, refs, roles, "tof", reference_voltage_V=6000.,
                                   reference_flight_length_mm=125.))
    tof = session.apply(active_mode="tof")
    zeros = np.zeros(len(tof))
    converted = tof2mc(tof, 0., np.full(len(tof), 6000.), zeros, zeros, 125., zeros, mode="laser")
    np.testing.assert_allclose(converted, mass, rtol=1e-9)
    np.testing.assert_allclose(session.mass_from_normalized_tof(tof), converted)
    # Reusing event voltage, coordinates and old t0 would produce a different mass.
    wrong = tof2mc(tof, 17.4, variables.dld_high_voltage, variables.dld_x_det,
                    variables.dld_y_det, 110., zeros, mode="laser")
    assert np.mean(np.abs(wrong-mass)) > 1.


def test_confirmation_cannot_be_reused_to_fit_after_failure():
    variables, refs, roles, _ = known_data(confirmation_error_ns=15.)
    session = session_for(variables, refs, roles)
    session.fit()
    assert session.model.t0_ns == pytest.approx(17.4, abs=1e-6)
    assert session.validate_selection()["passed"]
    assert not session.confirm()["passed"]
    for action in (session.fit, session.confirm, session.accept):
        with pytest.raises(CalibrationStateError):
            action()


def test_heldout_outside_domain_fails_confirmation():
    variables, refs, roles, _ = known_data(outside_domain=True)
    session = session_for(variables, refs, roles)
    session.fit()
    assert session.validate_selection()["passed"]
    result = session.confirm()
    assert not result["passed"]
    assert sum(row["outside_domain"] for row in result["groups"]) == 1


def test_apply_domain_failure_is_atomic_even_after_first_chunk():
    variables, refs, roles, _ = known_data()
    # An event with unknown identity remains eligible for application, outside model coverage.
    variables.dld_high_voltage[-1] = 10000.
    for mask in refs.values():
        mask[-1] = False
    session = accepted(session_for(variables, refs, roles))
    before = variables.mc_calib.copy()
    with pytest.raises(CalibrationInputError, match="outside the training domain"):
        session.apply(active_mode="mc", chunk_size=50)
    np.testing.assert_array_equal(variables.mc_calib, before)
    assert session.status == "accepted"


@pytest.mark.parametrize("mode", ["tof", "mc"])
def test_chunk_parity_excluded_rows_and_mode_gate(mode):
    variables, refs, roles, mass = known_data(missing=True)
    session = accepted(session_for(variables, refs, roles, mode))
    with pytest.raises(CalibrationStateError, match="Active ToF/mass mode"):
        session.apply(active_mode="mc" if mode == "tof" else "tof")
    output = session.apply(active_mode=mode, chunk_size=7)
    assert np.isnan(output[0])
    assert session.receipt()["excluded_events"] == 1
    actual = output if mode == "mc" else session.mass_from_normalized_tof(output)
    np.testing.assert_allclose(actual[1:], mass[1:], rtol=1e-9)
    variables2, refs2, roles2, _ = known_data(missing=True)
    other = accepted(session_for(variables2, refs2, roles2, mode))
    np.testing.assert_allclose(other.apply(active_mode=mode, chunk_size=5000), output, equal_nan=True)


@pytest.mark.parametrize("change", ["source", "active", "reload", "model"])
def test_stale_session_invalidation(change):
    variables, refs, roles, _ = known_data()
    session = accepted(session_for(variables, refs, roles))
    if change == "source":
        variables.dld_t[0] += 1.
    elif change == "active":
        variables.mc_calib[0] += 1.
    elif change == "reload":
        variables.sync_from_data(variables.data)
    else:
        session.model.coefficients[0] += .01
    with pytest.raises(CalibrationStateError, match="changed"):
        session.apply(active_mode="mc")
    assert session.status == "invalidated"


def test_masks_are_immutable_roles_disjoint_and_receipt_serializable(tmp_path):
    variables, refs, roles, _ = known_data()
    session = session_for(variables, refs, roles)
    with pytest.raises(ValueError):
        session.roles["train"][0] = False
    refs[13.5][:] = False  # Caller mutation cannot change a snapshot.
    assert np.any(session.labels == 13.5)
    accepted(session)
    destination = session.save_receipt(tmp_path / "receipt.json")
    receipt = json.loads(destination.read_text())
    assert receipt["source_hash"] == session.source_hash
    assert receipt["diagnostics"]["confirmation"]["passed"]
    variables, refs, roles, _ = known_data()
    roles["selection"] = roles["train"]
    with pytest.raises(CalibrationInputError, match="disjoint"):
        session_for(variables, refs, roles)


def test_missing_reference_block_coverage_rejected_before_fit():
    variables, refs, roles, _ = known_data()
    roles["confirmation"][900:] = False
    with pytest.raises(CalibrationInputError, match="insufficient reference"):
        session_for(variables, refs, roles)


def test_references_with_identical_physical_arrival_time_do_not_identify_offset_and_scale():
    mass = np.tile([13.5, 27.], 20)
    features = np.column_stack((np.full(len(mass), 5000.), np.zeros((len(mass), 2)),
                                 np.linspace(0, 1, len(mass)), 100. / np.sqrt(mass)))
    model = MultiReferenceToFModel(ToFModelConfig(voltage_term=False, spatial_terms=False))
    with pytest.raises(CalibrationInputError, match="unidentifiable"):
        model.fit(features, np.column_stack((np.full(len(mass), 117.4), mass)))


def test_notebook_helper_does_not_fit_until_clicked_and_exposes_full_path(monkeypatch):
    from pyccapt.calibration.tutorials.tutorials_helpers import helper_guided_calibration as helper
    import ipywidgets as widgets
    variables, refs, roles, mass = known_data()
    session = session_for(variables, refs, roles)
    monkeypatch.setattr(helper, 'display', lambda *_: None)
    panel = helper.show_guided_calibration(session)
    buttons = {button.description: button for row in panel.children if isinstance(row, widgets.HBox)
               for button in row.children}
    assert session.status == "preview" and buttons["Apply accepted"].disabled
    for name in ("Fit training", "Validate selection", "Confirm once", "Accept model", "Apply accepted"):
        assert not buttons[name].disabled
        buttons[name].click()
    assert session.status == "applied"
    np.testing.assert_allclose(variables.mc_calib, mass, rtol=1e-9)


def test_voltage_pulse_model_matches_existing_effective_voltage_convention():
    variables, refs, roles, mass = known_data()
    pulse = np.full(len(mass), 1000.)
    effective = 1.015 * (variables.dld_high_voltage + .7*pulse)
    variables.dld_t[:] = 17.4 + (variables.dld_t - 17.4) * np.sqrt(variables.dld_high_voltage/effective)
    variables.dld_pulse_v = pulse
    session = accepted(session_for(variables, refs, roles, pulse_mode="voltage"))
    np.testing.assert_allclose(session.apply(active_mode="mc"), mass, rtol=1e-9)


def test_reference_proposals_keep_identity_unconfirmed_and_do_not_mutate():
    from pyccapt.calibration.tutorials.tutorials_helpers.helper_guided_calibration import propose_reference_windows
    variables = Variables()
    rng = np.random.default_rng(6)
    variables.mc_calib = np.r_[rng.normal(27., .04, 4000), rng.normal(54., .04, 4000)]
    before = variables.mc_calib.copy()
    proposals = propose_reference_windows(variables, n_peaks=2, prominence=20, distance=10, hist_bin_size=.01)
    assert len(proposals) == 2
    assert all(p['identity'] == 'unconfirmed' and p['event_mask'].dtype == bool for p in proposals)
    np.testing.assert_allclose(sorted(p['position'] for p in proposals), [27., 54.], atol=.02)
    np.testing.assert_array_equal(variables.mc_calib, before)


@pytest.mark.parametrize('attribute,value', [('reference_voltage_V', 7000.), ('reference_flight_length_mm', 150.)])
def test_reference_normalization_changes_invalidate_conversion(attribute, value):
    variables, refs, roles, _ = known_data()
    session = accepted(session_for(variables, refs, roles, 'tof'))
    output = session.apply(active_mode='tof')
    setattr(session, attribute, value)
    with pytest.raises(CalibrationStateError, match='protocol'):
        session.mass_from_normalized_tof(output)
    assert session.status == 'invalidated'


def test_direct_model_failed_refit_cannot_predict_with_stale_coefficients():
    variables, refs, roles, _ = known_data()
    session = session_for(variables, refs, roles)
    session.fit()
    model = session.model
    with pytest.raises(CalibrationInputError, match='coverage'):
        model.fit(session.features[:3], np.column_stack((session.raw[0][:3], session.labels[:3])))
    with pytest.raises(CalibrationStateError, match='not been fitted'):
        model.predict_factor(session.features[:3])
    assert model.valid_domain == {}


@pytest.mark.parametrize("column", ["t (ns)", "high_voltage (V)", "x_det (cm)", "y_det (cm)", "pulse_v (V)"])
def test_direct_session_rejects_fields_marked_unavailable_even_when_finite(column):
    variables, refs, roles, _ = known_data()
    frame = variables.data.copy()
    frame.attrs["pyccapt_unavailable_columns"] = [column]
    variables.sync_from_data(frame)
    with pytest.raises(CalibrationInputError, match="unavailable"):
        session_for(variables, refs, roles, pulse_mode="voltage")


def test_laser_session_does_not_require_pulse_measurement_and_measured_zero_is_valid():
    variables, refs, roles, _ = known_data()
    frame = variables.data.drop(columns="pulse_v (V)")
    frame.loc[0, ["x_det (cm)", "y_det (cm)"]] = 0.
    variables.sync_from_data(frame)
    session = session_for(variables, refs, roles, pulse_mode="laser")
    assert session.apply_mask[0]


def test_direct_voltage_session_checks_availability_of_legacy_pulse_alias():
    variables, refs, roles, _ = known_data()
    frame = variables.data.rename(columns={"pulse_v (V)": "pulse"})
    variables.sync_from_data(frame)
    assert session_for(variables, refs, roles, pulse_mode="voltage").apply_mask.all()
    frame.attrs["pyccapt_unavailable_columns"] = ["pulse"]
    variables.sync_from_data(frame)
    with pytest.raises(CalibrationInputError, match="unavailable: pulse"):
        session_for(variables, refs, roles, pulse_mode="voltage")
