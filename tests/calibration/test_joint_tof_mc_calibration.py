"""Tests for the Joint ToF + m/c Iterative Calibration module."""

import numpy as np
import pytest

from pyccapt.calibration.core.exceptions import CalibrationInputError, CalibrationStateError
from pyccapt.calibration.core.share_variables import Variables
from pyccapt.calibration.core.joint_tof_mc_calibration import (
    _detect_peaks_1d,
    _effective_flight_path,
    _histogram_peak_center,
    _peak_fwhm,
    _tof_to_mc,
    _mc_to_tof,
    _build_correction_feature_matrix,
    dual_space_peak_detection,
    joint_tof_mc_calibration,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_data(n_ions=5000, flight_path_mm=100.0, seed=42):
    """Create a synthetic dataset with two clear peaks for testing."""
    rng = np.random.RandomState(seed)

    # Physical constants
    e = 1.6e-19
    amu = 1.66e-27
    l_m = flight_path_mm * 1e-3

    voltages = 3000.0 + rng.randn(n_ions) * 50.0
    x_det_cm = rng.randn(n_ions) * 0.5
    y_det_cm = rng.randn(n_ions) * 0.5

    # Create two peaks: one at m/c ~27 (Al+) and one at m/c ~56 (Fe+)
    # Use wider peaks to be more realistic and detectable with standard params
    n_peak1 = n_ions // 2
    n_peak2 = n_ions - n_peak1

    mc_true = np.empty(n_ions, dtype=float)
    mc_true[:n_peak1] = 27.0 + rng.randn(n_peak1) * 0.15
    mc_true[n_peak1:] = 56.0 + rng.randn(n_peak2) * 0.25

    # Compute corresponding ToF from m/c
    fp_m = np.sqrt((x_det_cm * 1e-2) ** 2 + (y_det_cm * 1e-2) ** 2 + l_m ** 2)
    tof_ns = np.sqrt(mc_true * amu * fp_m ** 2 / (2.0 * e * voltages)) * 1e9

    # Add small noise to ToF
    tof_ns += rng.randn(n_ions) * 0.1

    return {
        'tof': tof_ns,
        'voltage': voltages,
        'x_det': x_det_cm,
        'y_det': y_det_cm,
        'mc_true': mc_true,
        'flight_path_mm': flight_path_mm,
    }


def _make_variables(data):
    """Populate a Variables instance from synthetic data."""
    v = Variables()
    v.dld_t = data['tof']
    v.dld_high_voltage = data['voltage']
    v.dld_x_det = data['x_det']
    v.dld_y_det = data['y_det']
    v.dld_pulse_v = np.zeros_like(data['voltage'])
    # Also set calibration arrays to initial values
    v.dld_t_calib = np.copy(data['tof'])
    v.mc_calib = np.copy(data['mc_true'])
    return v


# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------

class TestEffectiveFlightPath:
    def test_zero_detector_offset(self):
        result = _effective_flight_path(0.0, 0.0, 100.0)
        assert np.isclose(result, 0.1, atol=1e-6)

    def test_nonzero_offset(self):
        result = _effective_flight_path(1.0, 1.0, 100.0)
        x_m = 0.01
        y_m = 0.01
        l_m = 0.1
        expected = np.sqrt(x_m**2 + y_m**2 + l_m**2)
        assert np.isclose(result, expected, atol=1e-8)

    def test_array_input(self):
        result = _effective_flight_path(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            100.0,
        )
        assert result.shape == (2,)


class TestTofMcConversions:
    def test_roundtrip(self):
        mc = 28.0
        v = 3000.0
        fp = 0.1
        tof = _mc_to_tof(mc, v, fp)
        mc_back = _tof_to_mc(tof, v, fp)
        assert np.isclose(mc_back, mc, rtol=1e-6)

    def test_array_roundtrip(self):
        mc = np.array([14.0, 28.0, 56.0])
        v = np.array([3000.0, 3000.0, 3000.0])
        fp = np.array([0.1, 0.1, 0.1])
        tof = _mc_to_tof(mc, v, fp)
        mc_back = _tof_to_mc(tof, v, fp)
        np.testing.assert_allclose(mc_back, mc, rtol=1e-6)


class TestHistogramPeakCenter:
    def test_unimodal(self):
        rng = np.random.RandomState(0)
        data = rng.normal(50.0, 1.0, 1000)
        center = _histogram_peak_center(data, 0.5)
        assert abs(center - 50.0) < 1.5

    def test_too_few_points(self):
        result = _histogram_peak_center(np.array([1.0, 2.0]), 0.5)
        assert np.isnan(result)


class TestPeakFwhm:
    def test_narrow_peak(self):
        rng = np.random.RandomState(0)
        data = rng.normal(100.0, 0.5, 5000)
        fwhm, center = _peak_fwhm(data, 0.1)
        assert fwhm > 0
        assert abs(center - 100.0) < 2.0
        # FWHM of Gaussian ≈ 2.355 * sigma ≈ 1.18
        assert fwhm < 3.0

    def test_too_few_points(self):
        fwhm, center = _peak_fwhm(np.array([1.0]), 0.5)
        assert np.isnan(fwhm)


class TestDetectPeaks1d:
    def test_two_peaks(self):
        rng = np.random.RandomState(0)
        peak1 = rng.normal(27.0, 0.15, 2000)
        peak2 = rng.normal(56.0, 0.25, 2000)
        data = np.concatenate([peak1, peak2])
        peaks = _detect_peaks_1d(data, n_peaks=5, prominence_threshold=20,
                                 distance=50, bin_size=0.02)
        assert len(peaks) >= 2
        positions = sorted([p['position'] for p in peaks])
        assert abs(positions[0] - 27.0) < 1.0
        assert abs(positions[1] - 56.0) < 1.0

    def test_empty_input(self):
        result = _detect_peaks_1d(np.array([]), 3, 100, 500, 0.1)
        assert result == []


class TestDualSpacePeakDetection:
    def test_finds_matched_peaks(self):
        data = _make_synthetic_data(n_ions=10000)
        fp_m = _effective_flight_path(data['x_det'], data['y_det'], data['flight_path_mm'])
        peaks = dual_space_peak_detection(
            data['tof'], data['mc_true'], data['voltage'], fp_m,
            n_peaks=4, prominence=20, distance=50,
            bin_size_tof=0.5, bin_size_mc=0.02,
        )
        assert len(peaks) >= 2
        mc_positions = sorted([p['mc_position'] for p in peaks])
        # Should find peaks near 27 and 56
        assert any(abs(p - 27.0) < 2.0 for p in mc_positions)
        assert any(abs(p - 56.0) < 2.0 for p in mc_positions)


class TestBuildCorrectionFeatureMatrix:
    def test_shape(self):
        n = 100
        v = np.ones(n) * 3000.0
        x = np.zeros(n)
        y = np.zeros(n)
        fm = _build_correction_feature_matrix(v, x, y, 3000.0, 1.0, 1.0)
        assert fm.shape == (n, 12)

    def test_bias_column(self):
        n = 50
        fm = _build_correction_feature_matrix(
            np.ones(n), np.zeros(n), np.zeros(n), 1.0, 1.0, 1.0)
        np.testing.assert_allclose(fm[:, 0], 1.0)


# ---------------------------------------------------------------------------
# Integration test for the main function
# ---------------------------------------------------------------------------

class TestJointTofMcCalibration:
    def test_runs_on_synthetic_data(self):
        data = _make_synthetic_data(n_ions=10000)
        v = _make_variables(data)

        result = joint_tof_mc_calibration(
            v,
            flight_path_length=data['flight_path_mm'],
            t0=0.0,
            det_diam=50.0,
            pulse_mode='laser',
            n_peaks=4,
            prominence=20,
            distance=50,
            bin_size_mc=0.02,
            bin_size_tof=0.5,
            max_iterations=3,
            tof_weight=0.7,
            mc_weight=0.3,
            verbose=False,
        )

        assert 'matched_peaks' in result
        assert 'parameters' in result
        assert 'loss_history' in result
        assert len(result['parameters']) == 12
        assert result['n_matched_peaks'] >= 2
        assert result['final_loss'] < 1e12

        # Verify variables were updated
        assert v.mc_calib is not None
        assert v.dld_t_calib is not None
        assert len(v.mc_calib) == len(data['tof'])

    def test_empty_data_raises(self):
        v = Variables()
        v.dld_t = np.array([])
        v.dld_high_voltage = np.array([])
        v.dld_x_det = np.array([])
        v.dld_y_det = np.array([])
        with pytest.raises(CalibrationInputError):
            joint_tof_mc_calibration(v, flight_path_length=100.0, verbose=False)

    def test_mismatched_lengths_raises(self):
        v = Variables()
        v.dld_t = np.ones(100)
        v.dld_high_voltage = np.ones(50)
        v.dld_x_det = np.ones(100)
        v.dld_y_det = np.ones(100)
        with pytest.raises(CalibrationInputError):
            joint_tof_mc_calibration(v, flight_path_length=100.0, verbose=False)

    def test_negative_weights_raises(self):
        data = _make_synthetic_data()
        v = _make_variables(data)
        with pytest.raises(CalibrationInputError):
            joint_tof_mc_calibration(
                v, flight_path_length=100.0, tof_weight=-1.0, mc_weight=0.5,
                verbose=False,
            )

    def test_loss_decreases(self):
        data = _make_synthetic_data(n_ions=10000)
        v = _make_variables(data)
        result = joint_tof_mc_calibration(
            v,
            flight_path_length=data['flight_path_mm'],
            pulse_mode='laser',
            n_peaks=4,
            prominence=20,
            distance=50,
            bin_size_mc=0.02,
            bin_size_tof=0.5,
            max_iterations=5,
            verbose=False,
        )
        history = result['loss_history']
        # Loss should not increase (allowing for small numerical noise)
        if len(history) > 1:
            assert history[-1] <= history[0] + 1e-6
