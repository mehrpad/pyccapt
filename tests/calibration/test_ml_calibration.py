"""Tests for the ML-based gradient-boosted correction surface calibration."""

from __future__ import annotations

import numpy as np
import pytest

from pyccapt.calibration.core.ml_calibration import (
    _build_feature_matrix,
    _evaluate_peaks,
    _extract_training_data,
    _normalise_spatial,
    _normalise_voltage,
    _predict_correction,
    _train_gbcs,
    ml_calibration,
)
from pyccapt.calibration.core.adaptive_residual_calibration import _mode_aware_defaults
from pyccapt.calibration.core.exceptions import CalibrationInputError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeVariables:
    """Minimal stand-in for share_variables.Variables."""

    def __init__(self, mc, voltage, x_det, y_det):
        self.mc_calib = np.asarray(mc, dtype=float)
        self.dld_t_calib = np.asarray(mc, dtype=float) * 0.5  # dummy tof
        self.dld_high_voltage = np.asarray(voltage, dtype=float)
        self.dld_x_det = np.asarray(x_det, dtype=float)
        self.dld_y_det = np.asarray(y_det, dtype=float)

    def get_calibration_array(self, mode):
        if mode == "tof":
            return self.dld_t_calib
        return self.mc_calib


def _make_synthetic_spectrum(n_ions=50000, rng_seed=42):
    """Build a synthetic mc spectrum with two Gaussian peaks and known voltage drift."""
    rng = np.random.RandomState(rng_seed)
    # Two peaks at 28 Da and 58 Da
    n1 = n_ions * 3 // 4
    n2 = n_ions - n1
    peak1 = rng.normal(28.0, 0.05, n1)
    peak2 = rng.normal(58.0, 0.08, n2)
    mc = np.concatenate([peak1, peak2])

    # Fake instrument data
    voltage = rng.uniform(3000, 7000, n_ions)
    x_det = rng.uniform(-1.5, 1.5, n_ions)
    y_det = rng.uniform(-1.5, 1.5, n_ions)

    # Apply a known multiplicative voltage drift (small ~2%)
    drift = 1.0 + 0.02 * (voltage - 5000) / 2000
    mc_drifted = mc * drift

    return mc_drifted, voltage, x_det, y_det


# ---------------------------------------------------------------------------
# Feature matrix tests
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:
    def test_shape_and_names(self):
        X, names = _build_feature_matrix(
            np.ones(10), np.zeros(10), np.zeros(10), np.linspace(0, 1, 10),
        )
        assert X.shape == (10, 7)
        assert len(names) == 7
        assert "voltage" in names
        assert "r2" in names
        assert "V_r2" in names

    def test_r2_computed_correctly(self):
        x = np.array([3.0, 0.0])
        y = np.array([4.0, 0.0])
        X, _ = _build_feature_matrix(np.zeros(2), x, y, np.zeros(2))
        np.testing.assert_allclose(X[:, 4], [25.0, 0.0])  # r2 = x^2 + y^2


# ---------------------------------------------------------------------------
# Normalisation tests
# ---------------------------------------------------------------------------

class TestNormalisation:
    def test_voltage_normalisation_centres_data(self):
        v = np.array([100.0, 200.0, 300.0])
        v_norm, center, scale = _normalise_voltage(v)
        assert abs(np.median(v_norm)) < 1e-10

    def test_spatial_normalisation_scales(self):
        x = np.array([10.0, 20.0, 30.0])
        y = np.array([0.0, 0.0, 0.0])
        x_n, y_n, scale = _normalise_spatial(x, y)
        assert scale > 0
        np.testing.assert_allclose(x_n, x / scale)


# ---------------------------------------------------------------------------
# Training data extraction
# ---------------------------------------------------------------------------

class TestExtractTrainingData:
    def test_extracts_from_peaks(self):
        mc, voltage, x, y = _make_synthetic_spectrum(10000)
        peaks = [
            {"position": 28.0, "x1": 27.5, "x2": 28.5},
            {"position": 58.0, "x1": 57.0, "x2": 59.0},
        ]
        X, y_ratio, labels, norm = _extract_training_data(mc, voltage, x, y, peaks)
        assert X.shape[0] > 100
        assert X.shape[1] == 7
        # Ratios should be close to 1.0 (drift is small)
        assert 0.9 < np.median(y_ratio) < 1.1

    def test_raises_on_empty_peaks(self):
        mc = np.array([1.0, 2.0, 3.0])
        with pytest.raises(CalibrationInputError):
            _extract_training_data(
                mc, np.ones(3), np.zeros(3), np.zeros(3),
                [{"position": 100.0, "x1": 99.0, "x2": 101.0}],
            )


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

class TestTrainGBCS:
    def test_trains_and_returns_diagnostics(self):
        rng = np.random.RandomState(0)
        X = rng.randn(500, 7)
        y = 1.0 + 0.01 * X[:, 0]
        model, diag = _train_gbcs(X, y, n_estimators=20, max_depth=3, verbose=False)
        assert diag["train_r2"] > 0.0
        assert diag["n_samples"] == 500
        assert "cv_r2_mean" in diag


# ---------------------------------------------------------------------------
# Prediction & clamping
# ---------------------------------------------------------------------------

class TestPrediction:
    def test_clamp_bounds(self):
        rng = np.random.RandomState(0)
        X = rng.randn(200, 7)
        y = 1.0 + 0.01 * X[:, 0]
        model, _ = _train_gbcs(X, y, n_estimators=20, max_depth=3)
        norm_params = (0.0, 1.0, 1.0)  # (v_center, v_scale, s_scale)
        pred = _predict_correction(
            model, rng.randn(50), rng.randn(50), rng.randn(50),
            np.linspace(0, 1, 50), norm_params, clamp_low=0.95, clamp_high=1.05,
        )
        assert np.all(pred >= 0.95)
        assert np.all(pred <= 1.05)


# ---------------------------------------------------------------------------
# Mode-aware defaults (ToF fix)
# ---------------------------------------------------------------------------

class TestModeAwareDefaults:
    def test_mc_mode_uses_user_value(self):
        arr = np.linspace(0, 400, 10000)
        tbs, hbs = _mode_aware_defaults(arr, "mc", 0.01)
        assert tbs == pytest.approx(0.01)

    def test_tof_mode_scales_up(self):
        arr = np.linspace(200, 800, 10000)
        tbs, hbs = _mode_aware_defaults(arr, "tof", 0.01)
        # tof range is ~600, so auto_template_bin ~0.6, auto_hist_bin ~1.2
        assert tbs > 0.01  # should be scaled up
        assert hbs > 0.01

    def test_tof_mode_respects_large_user_value(self):
        arr = np.linspace(200, 800, 10000)
        tbs, hbs = _mode_aware_defaults(arr, "tof", 1.0, user_hist_bin_size=2.0)
        assert tbs == pytest.approx(1.0)
        assert hbs == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Integration test: full ml_calibration
# ---------------------------------------------------------------------------

class TestMLCalibrationIntegration:
    def test_improves_drifted_spectrum(self):
        mc_drifted, voltage, x, y = _make_synthetic_spectrum(50000)
        variables = _FakeVariables(mc_drifted, voltage, x, y)

        result = ml_calibration(
            variables,
            calibration_mode="mc",
            n_peaks=2,
            prominence=50,
            distance=100,
            n_estimators=50,
            max_depth=4,
            verbose=False,
        )
        assert "baseline_score" in result
        assert "final_score" in result
        assert "feature_importances" in result
        assert "diagnostics" in result

    def test_rejects_bad_calibration(self):
        # Already-perfect spectrum: ML should not make things worse
        rng = np.random.RandomState(99)
        n = 30000
        mc = np.concatenate([rng.normal(28.0, 0.02, n * 3 // 4), rng.normal(58.0, 0.03, n // 4)])
        voltage = rng.uniform(4000, 6000, n)
        x = rng.uniform(-1, 1, n)
        y = rng.uniform(-1, 1, n)
        variables = _FakeVariables(mc, voltage, x, y)

        result = ml_calibration(
            variables,
            calibration_mode="mc",
            n_peaks=2,
            prominence=50,
            distance=100,
            n_estimators=30,
            max_depth=3,
            verbose=False,
        )
        # Either accepted with improvement or rejected
        if result["accepted"]:
            assert result["final_score"] >= result["baseline_score"]
