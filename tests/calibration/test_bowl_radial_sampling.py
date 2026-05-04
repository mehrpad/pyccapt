import numpy as np

from pyccapt.calibration.core import calibration
from pyccapt.calibration.core.correction_models import _predict_bowl_model


class _Variables:
    result_path = ""


def _sample_uniform_disk(rng, n_points, radius_mm):
    theta = rng.uniform(0.0, 2.0 * np.pi, n_points)
    radius = radius_mm * np.sqrt(rng.uniform(0.0, 1.0, n_points))
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return x, y


def test_bowl_correction_uses_radial_model_with_polar_sampling():
    rng = np.random.default_rng(7)
    x_mm, y_mm = _sample_uniform_disk(rng, n_points=5000, radius_mm=20.0)
    r2 = x_mm ** 2 + y_mm ** 2

    peak_location = 25.0
    true_factor = 1.0 + 2.0e-4 * r2 + 2.0e-7 * (r2 ** 2)
    dld_t = peak_location * true_factor + rng.normal(0.0, 0.02, size=len(r2))

    model = calibration.bowl_correction(
        x_mm,
        y_mm,
        dld_t,
        _Variables(),
        det_diam=40.0,
        maximum_location=peak_location,
        sample_range_max="mean",
        sample_size=4,
        calibration_mode="tof",
        fit_mode="curve_fit",
        index_fig=0,
        plot=False,
        save=False,
        bin_size=0.02,
    )

    assert model["model"] == "radial_curve_fit"
    predicted = _predict_bowl_model("curve_fit", model, x_mm, y_mm)
    assert np.isfinite(predicted).all()

    mae = np.mean(np.abs(predicted - true_factor))
    assert mae < 0.05


def test_bowl_correction_supports_cartesian_override():
    rng = np.random.default_rng(9)
    x_mm, y_mm = _sample_uniform_disk(rng, n_points=2500, radius_mm=20.0)
    r2 = x_mm ** 2 + y_mm ** 2
    peak_location = 25.0
    true_factor = 1.0 + 2.0e-4 * r2
    dld_t = peak_location * true_factor + rng.normal(0.0, 0.03, size=len(r2))

    model = calibration.bowl_correction(
        x_mm,
        y_mm,
        dld_t,
        _Variables(),
        det_diam=40.0,
        maximum_location=peak_location,
        sample_range_max="mean",
        sample_size=4,
        calibration_mode="tof",
        fit_mode="curve_fit",
        index_fig=0,
        plot=False,
        save=False,
        bin_size=0.02,
        sampling_mode="cartesian",
    )

    # Cartesian mode keeps the legacy polynomial surface model (6 coefficients).
    assert not isinstance(model, dict)
    assert len(model) == 6


