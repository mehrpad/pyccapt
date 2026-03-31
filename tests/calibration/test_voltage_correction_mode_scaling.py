import numpy as np

from pyccapt.calibration.core import calibration


class _DummyVariables:
    def __init__(self):
        self.selected_x1 = 9.5
        self.selected_x2 = 11.5
        self.dld_t_calib = np.array([8.0, 10.0, 12.0, 14.0], dtype=float)
        self.mc_calib = np.array([80.0, 100.0, 120.0, 140.0], dtype=float)

    def build_calibration_mask(self, calibration_mode):
        return np.array([False, True, True, False], dtype=bool)

    def get_calibration_array(self, calibration_mode):
        if calibration_mode == "tof":
            return self.dld_t_calib
        return self.mc_calib


def test_voltage_corr_main_uses_sqrt_factor_for_tof_and_linear_for_mc(monkeypatch):
    dld_high_voltage = np.array([1000.0, 1100.0, 1200.0, 1300.0], dtype=float)

    monkeypatch.setattr(
        calibration,
        "voltage_correction",
        lambda *args, **kwargs: np.array([1.0, 0.0, 0.0], dtype=float),
    )
    monkeypatch.setattr(
        calibration,
        "_predict_voltage_model",
        lambda model, fitresult, voltage_values: np.full(len(voltage_values), 4.0, dtype=float),
    )

    variables_tof = _DummyVariables()
    calibration.voltage_corr_main(
        dld_high_voltage,
        variables_tof,
        sample_size=1,
        mode="ion_seq",
        calibration_mode="tof",
        index_fig=0,
        plot=False,
        save=False,
        model="curve_fit",
    )

    variables_mc = _DummyVariables()
    calibration.voltage_corr_main(
        dld_high_voltage,
        variables_mc,
        sample_size=1,
        mode="ion_seq",
        calibration_mode="mc",
        index_fig=0,
        plot=False,
        save=False,
        model="curve_fit",
    )

    np.testing.assert_allclose(variables_tof.dld_t_calib, np.array([4.0, 5.0, 6.0, 7.0]))
    np.testing.assert_allclose(variables_mc.mc_calib, np.array([20.0, 25.0, 30.0, 35.0]))

