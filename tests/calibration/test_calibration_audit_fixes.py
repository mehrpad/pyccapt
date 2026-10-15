"""Synthetic physical recovery and unavailable-input regression tests."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from pyccapt.calibration.core import diagnostics, ml_calibration as ml, reference_optimizer as ref
from pyccapt.calibration.core import new_methods
from pyccapt.calibration.core.exceptions import CalibrationInputError
from pyccapt.calibration.core.share_variables import Variables


@pytest.mark.parametrize("train_before,holdout_before,holdout_after", [
    (np.nan, 100, 110), (100, np.nan, 110), (100, 100, np.nan), (100, np.nan, np.nan),
])
def test_adaptive_candidate_requires_measurable_training_and_declared_holdout(
        train_before, holdout_before, holdout_after):
    from pyccapt.calibration.core import adaptive_residual_calibration as adaptive
    baseline = {"train_score": train_before, "holdout_score": holdout_before, "holdout_required": True}
    candidate = {"quality": {"train_score": 110.0, "holdout_score": holdout_after}}
    assert adaptive._best_candidate([candidate], baseline) is None


def test_adaptive_absent_holdout_does_not_prevent_measured_training_improvement(monkeypatch):
    from pyccapt.calibration.core import adaptive_residual_calibration as adaptive
    monkeypatch.setattr(adaptive, "_evaluate_peak_group", lambda values, peaks, **kwargs: 100 if peaks else np.nan)
    baseline = adaptive._evaluate_quality(np.ones(100), {"train": [{}], "holdout": []})
    candidate = {"quality": {"train_score": 110.0, "holdout_score": np.nan}}
    assert not baseline["holdout_required"]
    assert adaptive._best_candidate([candidate], baseline) is candidate


def test_initial_geometry_and_voltage_recover_a_constant_time():
    length = 100.0
    radius = np.array([0.0, 20.0, 40.0, 60.0])
    voltage = np.array([4000.0, 5000.0, 6000.0, 7000.0])
    distance = np.sqrt(length**2 + radius**2)
    # Straight flight at a fixed mass: time is proportional to path / sqrt(V).
    times = distance / np.sqrt(voltage)
    data = pd.DataFrame({"high_voltage (V)": voltage, "t (ns)": times,
                         "x_det (cm)": radius / 10, "y_det (cm)": np.zeros(4)})
    output = diagnostics.initial_calibration(data, length)
    np.testing.assert_allclose(output, np.sqrt(np.mean(distance**2) / np.mean(voltage)), atol=1e-12)


def test_initial_bad_voltages_invalidate_only_their_rows():
    data = pd.DataFrame({"high_voltage (V)": [5000.0, np.nan, 0.0, -1.0, np.inf],
                         "t (ns)": np.full(5, 100.0), "x_det (cm)": np.zeros(5),
                         "y_det (cm)": np.zeros(5)})
    output = diagnostics.initial_calibration(data, 100.0)
    assert output[0] == 100.0
    assert np.isnan(output[1:]).all()
    with pytest.raises(CalibrationInputError, match="positive voltage"):
        diagnostics.initial_calibration(data.iloc[1:], 100.0)


@pytest.mark.parametrize("missing", [np.nan, np.inf])
def test_initial_partial_positions_preserve_time_without_spatial_factor(missing):
    data = pd.DataFrame({"high_voltage (V)": [5000.0, 5000.0], "t (ns)": [100.0, 101.0],
                         "x_det (cm)": [0.0, missing], "y_det (cm)": [0.0, 0.0]})
    np.testing.assert_array_equal(diagnostics.initial_calibration(data, 100.0), [100.0, 101.0])


def _reference_data(mode="mc", count=3):
    rng = np.random.default_rng(43)
    centres = np.linspace(50.0, 200.0, count)
    labels = np.repeat(centres, 4000)
    v = Variables()
    measured = labels * 1.005 + rng.normal(0, 0.12, labels.size)
    v.mc_calib = measured.copy()
    v.dld_t_calib = measured.copy()
    v.dld_high_voltage = np.linspace(4000, 6000, labels.size)
    v.dld_x_det = rng.uniform(-1, 1, labels.size)
    v.dld_y_det = rng.uniform(-1, 1, labels.size)
    v.max_tof = 250.0
    references = [ref.ReferenceLine(label=str(c), mz=c, tolerance=1.5) for c in centres]
    return v, labels, references


@pytest.mark.parametrize("mode", ["mc", "tof"])
def test_reference_fit_recovers_absolute_line_centres(mode):
    v, labels, references = _reference_data(mode)
    result = ref.fit_reference_constrained(v, calibration_mode=mode, reference_lines=references, verbose=False)
    assert result["ok"], result
    for centre in np.unique(labels):
        assert abs(np.mean(v.get_calibration_array(mode)[labels == centre]) - centre) < 0.01


def test_reference_spatial_fit_handles_partial_and_invalid_voltage_rows():
    v, _, references = _reference_data(count=12)
    v.dld_x_det[0] = np.nan
    v.dld_y_det[1] = np.inf
    v.dld_high_voltage[2] = np.nan
    original = v.mc_calib.copy()
    result = ref.fit_reference_constrained(v, reference_lines=references, verbose=False)
    assert "lsq_error" not in result.get("reason", ""), result
    assert result["n_matches"] >= 8  # actually exercise the spatial model
    assert result["ok"], result
    assert np.isfinite(v.mc_calib).all()
    np.testing.assert_array_equal(v.mc_calib[:3], original[:3])


def test_reference_missing_quality_cannot_accept(monkeypatch):
    v, _, references = _reference_data()
    original = v.mc_calib.copy()
    monkeypatch.setattr(ref, "_audit_simple", lambda *args: {
        "n_peaks": 3, "ref_hit": 3, "geomean_mrp": float("nan")})
    result = ref.fit_reference_constrained(v, reference_lines=references, verbose=False)
    assert not result["ok"]
    np.testing.assert_array_equal(v.mc_calib, original)


def _ml_data():
    rng = np.random.default_rng(8)
    v = Variables()
    v.mc_calib = np.repeat([28.0, 58.0, 100.0], 400) + rng.normal(0, 0.02, 1200)
    v.dld_high_voltage = rng.uniform(4000, 6000, 1200)
    v.dld_x_det = rng.uniform(-1, 1, 1200)
    v.dld_y_det = rng.uniform(-1, 1, 1200)
    peaks = [{"position": p, "x1": p - 0.3, "x2": p + 0.3} for p in [28, 58, 100]]
    return v, peaks


def test_ml_prediction_preserves_unavailable_features():
    v, peaks = _ml_data()
    v.dld_x_det[0] = np.nan
    v.dld_y_det[1] = np.inf
    v.dld_high_voltage[2] = np.nan
    X, y, _, norm = ml._extract_training_data(v.mc_calib, v.dld_high_voltage,
                                             v.dld_x_det, v.dld_y_det, peaks)
    assert np.isfinite(X).all()
    model, _ = ml._train_gbcs(X, y, n_estimators=3, compute_cv=False)
    factors = ml._predict_correction(model, v.dld_high_voltage, v.dld_x_det,
                                    v.dld_y_det, np.linspace(0, 1, 1200), norm)
    assert np.isfinite(factors).all()
    np.testing.assert_array_equal(factors[:3], 1.0)


def test_ml_unmeasurable_holdout_rejects_without_mutating(monkeypatch):
    from pyccapt.calibration.core import calibration
    v, peaks = _ml_data()
    original = v.mc_calib.copy()
    monkeypatch.setattr(calibration, "auto_detect_reference_peaks", lambda *args, **kwargs: peaks)
    scores = iter([100.0, 100.0, 110.0, float("nan")])
    monkeypatch.setattr(ml, "_evaluate_peaks", lambda *args: next(scores))
    result = ml.ml_calibration(v, n_estimators=3, verbose=False)
    assert not result["accepted"]
    np.testing.assert_array_equal(v.mc_calib, original)


def test_experimental_preview_does_not_mutate_legacy_warm_start():
    voltage = np.repeat(np.linspace(4000, 6000, 30), 100)
    v = Variables()
    v.mc_calib = 100 * (1 + (voltage - 5000) * 0.00001)
    v.set_calibration_peak_range("mc", 98, 102)
    original = v.mc_calib.copy()
    candidate = new_methods.voltage_corr_time_dependent(
        voltage, v, sample_size=100, apply=False, gate_min_ions=10000)
    np.testing.assert_array_equal(v.mc_calib, original)
    assert np.std(candidate) < np.std(original) * 0.05


@pytest.mark.parametrize("mode", ["mc", "tof"])
def test_experimental_joint_voltage_time_fit_recovers_known_drift(mode):
    rng = np.random.default_rng(91)
    voltage = rng.uniform(4000, 6000, 10000)
    measured = 100 * (1 + (voltage - 5000) * 0.00001) + rng.normal(0, 0.01, len(voltage))
    v = Variables()
    v.mc_calib = measured.copy()
    v.dld_t_calib = measured.copy()
    v.set_calibration_peak_range(mode, 98, 102)
    result = new_methods.voltage_corr_v_t_joint(voltage, v, calibration_mode=mode, apply=False)
    assert np.std(result) < 0.02  # remove systematic spread down to the planted noise
    np.testing.assert_array_equal(v.get_calibration_array(mode), measured)


def test_self_similarity_unmeasurable_quality_rejects(monkeypatch):
    from pyccapt.calibration.core import calibration
    v = Variables()
    v.mc_calib = np.linspace(10, 100, 3000)
    original = v.mc_calib.copy()
    monkeypatch.setattr(calibration, "auto_detect_reference_peaks", lambda *args, **kwargs: [])
    result = new_methods.drift_correction_self_similarity(v, n_chunks=4, n_grid=5, verbose=False)
    assert not result["ok"]
    np.testing.assert_array_equal(v.mc_calib, original)


@pytest.mark.parametrize("mode", ["mc", "tof"])
def test_experimental_spline_recovers_known_spatial_drift_and_preserves_partials(mode):
    rng = np.random.default_rng(92)
    x, y = rng.uniform(-2, 2, (2, 20000))  # detector cm
    measured = 100 * (1 + 0.005 * x) + rng.normal(0, 0.01, len(x))
    x[0] = np.nan
    v = Variables()
    v.mc_calib = measured.copy()
    v.dld_t_calib = measured.copy()
    v.set_calibration_peak_range(mode, 98, 102)
    result = new_methods.bowl_correction_spline(
        x, y, np.full(len(x), 5000.0), v, det_diam=80, sample_size=2,
        calibration_mode=mode, apply=False)
    assert np.std(result[1:]) < 0.05
    assert result[0] == measured[0]
    np.testing.assert_array_equal(v.get_calibration_array(mode), measured)


def test_advanced_api_imports_first_in_a_fresh_process():
    result = subprocess.run([sys.executable, "-c", "from pyccapt.calibration.core.calibration_advanced "
                             "import joint_voltage_bowl_corr_main"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_notebook_presets_match_display_and_disable_legacy_pruning(monkeypatch):
    from pyccapt.calibration.tutorials.tutorials_helpers import helper_calibration as helper
    import ipywidgets as widgets

    displayed = []
    monkeypatch.setattr(helper, "display", lambda *objects: displayed.extend(objects))
    monkeypatch.setattr(helper.mc_plot, "hist_plot", lambda *args, **kwargs: None)
    v = Variables()
    v.max_tof = 1000
    helper.call_voltage_bowl_calibration(v, det_diam=80, flight_path_length=100, pulse_mode="laser")

    def walk(widget):
        yield widget
        for child in getattr(widget, "children", ()):
            yield from walk(child)

    preset = next(widget for root in displayed for widget in walk(root)
                  if isinstance(widget, widgets.Dropdown) and widget.description == "Config preset:")
    assert preset.value == v.calibration_profile == "new"
    assert v.residual_coarse_to_fine_top_k == 1
    preset.value = "old"
    assert v.calibration_profile == "old"
    assert getattr(v, "residual_coarse_to_fine_top_k", None) is None
    preset.value = "best"
    assert v.use_time_dep_v and v.residual_coarse_to_fine_top_k == 1
