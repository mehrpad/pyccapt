import numpy as np

from pyccapt.calibration.core import calibration
from pyccapt.calibration.core.share_variables import Variables


def test_auto_detect_reference_peaks_and_joint_refinement_run_on_synthetic_data():
    rng = np.random.default_rng(1)
    variables = Variables()

    true_peaks = np.repeat([10.0, 20.0, 30.0, 40.0], 800)
    voltage = np.tile(np.linspace(4000.0, 6000.0, 800), 4)
    x_det = rng.uniform(-1.8, 1.8, len(true_peaks)) / 10.0
    y_det = rng.uniform(-1.8, 1.8, len(true_peaks)) / 10.0
    v_norm = (voltage - voltage.mean()) / voltage.std()
    r2_norm = ((x_det * 10.0) ** 2 + (y_det * 10.0) ** 2) / 6.0
    measured = true_peaks * (1.0 + 0.02 * v_norm + 0.015 * r2_norm) + rng.normal(0.0, 0.02, len(true_peaks))

    variables.mc_calib = measured.copy()
    variables.dld_t_calib = measured.copy()

    peaks = calibration.auto_detect_reference_peaks(
        variables.mc_calib,
        n_peaks=4,
        prominence=5,
        distance=10,
        hist_bin_size=0.05,
    )

    assert len(peaks) >= 3

    model, used_peaks = calibration.joint_voltage_bowl_corr_main(
        x_det,
        y_det,
        voltage,
        variables,
        det_diam=40,
        calibration_mode="mc",
        sample_size=1,
        bin_size=0.05,
        n_peaks=4,
        prominence=5,
        distance=10,
    )

    assert len(used_peaks) >= 3
    assert len(model["parameters"]) == 12
    assert np.isfinite(variables.mc_calib).all()
