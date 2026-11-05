"""Known-truth recovery across fitting and application, in both domains."""
import numpy as np
import pandas as pd
import pytest

from pyccapt.calibration.core import calibration
from pyccapt.calibration.core.share_variables import Variables


@pytest.mark.parametrize("mode", ["tof", "mc"])
def test_voltage_fit_removes_known_multiplicative_drift(mode):
    voltage = np.repeat(np.linspace(4000.0, 6000.0, 30), 100)
    measured = 100.0 * (1.0 + 0.0001 * (voltage - 5000.0))
    variables = Variables()
    variables.sync_from_data(pd.DataFrame({'t (ns)': measured, 'high_voltage (V)': voltage}))
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
    step = variables.calibration_metadata.state['steps'][-1]
    assert step['operation'] == 'voltage_correction' and step['status'] == 'working'
    assert step['parameters']['calibration_mode'] == mode and len(step['model']) == 3


@pytest.mark.parametrize("mode", ["tof", "mc"])
def test_multi_peak_voltage_fit_removes_known_drift(mode):
    # Peaks need an intrinsic width: with drift-only (flat-topped) peaks the
    # histogram maximum is arbitrary, and detection at the requested 0.01 bins
    # (rather than the old hard-coded 0.1, which found only one peak here)
    # resolves those plateaus.
    rng = np.random.default_rng(0)
    voltage = np.tile(np.linspace(4000.0, 6000.0, 20000), 3)
    labels = np.repeat([28.0, 58.0, 100.0], 20000)
    intrinsic = 0.0005
    measured = labels * (1.0 + 0.002 * (voltage - 5000.0) / 1000.0) * (1.0 + rng.normal(0, intrinsic, labels.size))
    variables = Variables()
    variables.dld_t_calib = measured.copy()
    variables.mc_calib = measured.copy()
    _, peaks = calibration.multi_peak_voltage_corr_main(
        voltage, variables, calibration_mode=mode, model="curve_fit",
        bin_size=0.01, n_peaks=3, prominence=10, distance=10,
    )
    assert sorted(round(p["position"]) for p in peaks) == [28, 58, 100]
    corrected = variables.get_calibration_array(mode)
    # Histogram centres can change global scale; the fitted voltage dependence
    # must nevertheless disappear in each known reference population, leaving
    # only the intrinsic peak width.
    for target in np.unique(labels):
        selected = labels == target
        slope_before = np.polyfit(voltage[selected], measured[selected], 1)[0]
        slope_after = np.polyfit(voltage[selected], corrected[selected], 1)[0]
        assert abs(slope_after) < abs(slope_before) * 0.05
        assert np.std(corrected[selected]) < target * intrinsic * 1.1
