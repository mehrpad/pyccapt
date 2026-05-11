import numpy as np
import pandas as pd

from pyccapt.calibration.core import calibration
from pyccapt.calibration.core import diagnostics
from pyccapt.calibration.core import mc_plot
from pyccapt.calibration.reconstructions import reconstruction


def test_calibration_reexports_voltage_model_function():
    result = calibration.voltage_corr(np.array([0.0, 1.0]), 1.0, 2.0, 3.0)
    assert result.tolist() == [1.0, 6.0]


def test_calibration_reexports_initial_calibration_function():
    data = pd.DataFrame(
        {
            "high_voltage (V)": [1000.0, 1200.0],
            "t (ns)": [50.0, 60.0],
            "x_det (cm)": [0.1, 0.2],
            "y_det (cm)": [0.2, 0.1],
        }
    )

    from_calibration = calibration.initial_calibration(data, flight_path_length=100.0)
    from_diagnostics = diagnostics.initial_calibration(data, flight_path_length=100.0)

    assert from_calibration.shape == (2,)
    assert np.allclose(from_calibration, from_diagnostics)


def test_mc_plot_hist_plot_wrapper_delegates(monkeypatch):
    captured = {}

    def fake_hist_plot(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "delegated"

    monkeypatch.setattr("pyccapt.calibration.core.mc_plot_api.hist_plot", fake_hist_plot)

    result = mc_plot.hist_plot(
        "vars",
        0.1,
        False,
        "mc",
        False,
        10,
        1,
        50,
        "peak",
        "fig",
        100,
    )

    assert result == "delegated"
    assert captured["args"][0] == "vars"
    assert captured["args"][3] == "mc"


def test_reconstruction_reexports_rotate_z():
    x_rot, y_rot, z_rot = reconstruction.rotate_z(1.0, 0.0, 2.0, np.pi / 2)
    assert np.isclose(x_rot, 0.0)
    assert np.isclose(y_rot, 1.0)
    assert z_rot == 2.0
