"""Known-truth recovery across fitting and application, in both domains."""
import numpy as np
import pytest

from pyccapt.calibration.core import calibration
from pyccapt.calibration.core.share_variables import Variables


@pytest.mark.parametrize("mode", ["tof", "mc"])
def test_voltage_fit_removes_known_multiplicative_drift(mode):
    voltage = np.repeat(np.linspace(4000.0, 6000.0, 30), 100)
    measured = 100.0 * (1.0 + 0.0001 * (voltage - 5000.0))
    variables = Variables()
    variables.dld_t_calib = measured.copy()
    variables.mc_calib = measured.copy()
    variables.set_calibration_peak_range(mode, 89.0, 111.0)
    calibration.voltage_corr_main(
        voltage, variables, sample_size=100, mode="ion_seq",
        calibration_mode=mode, index_fig=0, plot=False, save=False,
        model="curve_fit", maximum_sample_method="mean",
        maximum_cal_method="mean", peak_maximum=100.0,
    )
    np.testing.assert_allclose(variables.get_calibration_array(mode), 100.0, atol=1e-6)


@pytest.mark.parametrize("mode", ["tof", "mc"])
def test_multi_peak_voltage_fit_removes_known_drift(mode):
    voltage = np.tile(np.linspace(4000.0, 6000.0, 10000), 3)
    labels = np.repeat([28.0, 58.0, 100.0], 10000)
    measured = labels * (1.0 + 0.002 * (voltage - 5000.0) / 1000.0)
    variables = Variables()
    variables.dld_t_calib = measured.copy()
    variables.mc_calib = measured.copy()
    calibration.multi_peak_voltage_corr_main(
        voltage, variables, calibration_mode=mode, model="curve_fit",
        bin_size=0.01, n_peaks=3, prominence=10, distance=10,
    )
    corrected = variables.get_calibration_array(mode)
    # Histogram centres can change global scale; the fitted voltage dependence
    # must nevertheless disappear in each known reference population.
    for target in np.unique(labels):
        selected = labels == target
        assert np.std(corrected[selected]) < np.std(measured[selected]) * 0.01
