"""Reference-equivalence checks for bounded-memory calibration operations."""

import numpy as np
import pytest
from scipy import sparse

from pyccapt.calibration.core import adaptive_residual_calibration as adaptive
from pyccapt.calibration.core import calibration_advanced as advanced
from pyccapt.calibration.core import ml_calibration as ml
from pyccapt.calibration.core.calibration_sampling import _voltage_segments


@pytest.mark.parametrize("step", [0.1, 5, 10000])
def test_voltage_grouping_preserves_exact_boundaries_and_original_order(step):
    rng = np.random.default_rng(1)
    boundary = 3000 + step * np.arange(20)
    voltage = np.concatenate([boundary, np.nextafter(boundary, -np.inf),
                              np.nextafter(boundary, np.inf), rng.uniform(3000, 3010, 300)])
    rng.shuffle(voltage)
    values = np.arange(len(voltage))
    count = max(1, int(np.ceil((voltage.max() - voltage.min()) / step)))
    expected = []
    for i in range(count):
        low, high = voltage.min() + i * step, voltage.min() + (i + 1) * step
        mask = voltage >= low if i == count - 1 else (voltage >= low) & (voltage < high)
        expected.append((voltage[mask], values[mask]))
    grouped = _voltage_segments(voltage, values, step)
    assert len(grouped) == len(expected)
    for actual, reference in zip(grouped, expected):
        np.testing.assert_array_equal(actual[0], reference[0])
        np.testing.assert_array_equal(actual[1], reference[1])


def test_voltage_grouping_keeps_constant_voltage_population():
    voltage = np.full(40, 5000.0)
    groups = _voltage_segments(voltage, np.arange(40), 5)
    assert len(groups) == 1
    np.testing.assert_array_equal(groups[0][1], np.arange(40))


@pytest.mark.parametrize("size,fast", [(1, False), (100, False), (100, True)])
def test_histogram_peak_of_constant_population_is_exact(size, fast):
    from pyccapt.calibration.core.calibration_sampling import _resolve_peak_location
    assert _resolve_peak_location(np.full(size, 28.), "histogram", .01, fast_calibration=fast) == 28.


@pytest.mark.parametrize("workspace_bytes", [1, 30000, 1000000])
def test_blocked_template_likelihood_matches_dense_temporal_and_spatial(workspace_bytes):
    rng = np.random.default_rng(6)
    values = rng.normal(28.0, 0.09, 400)
    peak = {"position": 28.0, "x1": 27.7, "x2": 28.3}
    template = adaptive._build_peak_template(values, peak, 0.01)
    shifts = np.linspace(-0.2, 0.2, 43)
    centered = values[:, None] - template.center - shifts[None, :]
    logp = np.interp(centered.ravel(), template.offsets, template.logpdf,
                     left=template.log_floor, right=template.log_floor).reshape(len(values), len(shifts))
    expected = np.sum(logp, axis=0)
    actual = adaptive._template_shift_scores(values, template, shifts, workspace_bytes=workspace_bytes)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-10)
    assert np.argmax(actual) == np.argmax(expected)
    cells = rng.integers(0, 17, len(values))
    onehot = sparse.csr_matrix((np.ones(len(values)), (cells, np.arange(len(values)))), shape=(17, len(values)))
    expected_spatial = onehot @ logp
    actual_spatial = adaptive._template_shift_scores(values, template, shifts, onehot,
                                                     workspace_bytes=workspace_bytes)
    np.testing.assert_array_equal(actual_spatial, expected_spatial)
    np.testing.assert_array_equal(np.argmax(actual_spatial, axis=1), np.argmax(expected_spatial, axis=1))


def test_chunked_ml_predictions_equal_single_batch():
    rng = np.random.default_rng(4)
    voltage, x, y, seq = rng.normal(size=(4, 241))
    X, _ = ml._build_feature_matrix(voltage, x, y, seq)
    model, _ = ml._train_gbcs(X, 1 + 0.01 * x, n_estimators=5, compute_cv=False)
    expected = ml._predict_correction(model, voltage, x, y, seq, (0, 1, 1), chunk_size=1000)
    actual = ml._predict_correction(model, voltage, x, y, seq, (0, 1, 1), chunk_size=7)
    np.testing.assert_array_equal(actual, expected)


def test_chunked_joint_predictions_equal_dense_design_and_preserve_partials():
    rng = np.random.default_rng(5)
    voltage, x, y = rng.normal(size=(3, 241))
    params = rng.normal(0, 0.01, 12)
    params[0] = 1
    expected = advanced._joint_feature_matrix(voltage, x, y, 0, 1, 1) @ params
    actual = advanced._predict_joint_factors(voltage, x, y, params, 0, 1, 1, chunk_size=7)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    x[0], y[8], voltage[9] = np.nan, np.inf, np.nan
    actual = advanced._predict_joint_factors(voltage, x, y, params, 0, 1, 1, chunk_size=7)
    np.testing.assert_array_equal(actual[[0, 8, 9]], 1.0)


def test_ml_verbose_does_not_trigger_extra_cv_models(monkeypatch):
    from pyccapt.calibration.core import calibration
    from pyccapt.calibration.core.share_variables import Variables
    v = Variables()
    rng = np.random.default_rng(7)
    v.mc_calib = np.repeat([28.0, 58.0], 200) + rng.normal(0, 0.01, 400)
    v.dld_high_voltage = rng.normal(5000, 100, 400)
    v.dld_x_det, v.dld_y_det = rng.normal(size=(2, 400))
    peaks = [{"position": p, "x1": p - 0.3, "x2": p + 0.3} for p in [28, 58]]
    monkeypatch.setattr(calibration, "auto_detect_reference_peaks", lambda *args, **kwargs: peaks)
    monkeypatch.setattr(ml, "_evaluate_peaks", lambda *args: 100.0)
    calls = []
    monkeypatch.setattr(ml, "cross_val_score", lambda *args, **kwargs: calls.append(True) or np.array([0.5, 0.5]))
    without = ml.ml_calibration(v, n_estimators=3, verbose=True)
    assert not calls
    with_cv = ml.ml_calibration(v, n_estimators=3, verbose=False, diagnostic_cv=True)
    assert calls == [True]
    assert without["accepted"] == with_cv["accepted"]
    np.testing.assert_array_equal(without["model"].feature_importances_, with_cv["model"].feature_importances_)
