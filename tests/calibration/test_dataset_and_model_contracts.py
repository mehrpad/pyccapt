from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pyccapt.calibration.core.correction_models import PolynomialCorrectionModel
from pyccapt.calibration.core.dataset import CalibrationDataset
from pyccapt.calibration.core.share_variables import Variables


def test_calibration_dataset_publishes_units_masks_and_stable_hash():
    frame = pd.DataFrame(
        {
            "t (ns)": [1.0, 2.0],
            "x_det (cm)": [0.1, np.nan],
            "y_det (cm)": [0.2, 0.3],
            "mc (Da)": [10.0, 11.0],
        }
    )
    first = CalibrationDataset.from_frame(frame)
    second = CalibrationDataset.from_frame(frame.copy())

    assert first.row_count == 2
    assert first.units["t (ns)"] == "ns"
    assert first.finite_mask("detector_positions").tolist() == [True, False]
    assert first.source_hash == second.source_hash


def test_polynomial_model_contract_round_trips_provenance():
    voltage = np.linspace(1000.0, 2000.0, 40)
    factor = 1.0 + 2e-4 * voltage + 3e-8 * voltage**2
    model = PolynomialCorrectionModel("voltage").fit(voltage, factor)

    assert np.allclose(model.predict_factor(voltage), factor)
    payload = model.provenance()
    assert payload["valid_domain"]["voltage"] == [1000.0, 2000.0]
    json.dumps(payload)


def test_voltage_model_honors_requested_polynomial_degree():
    voltage = np.linspace(-2.0, 2.0, 9)
    factor = 2.0 - 0.5 * voltage + 0.25 * voltage**2 + 0.1 * voltage**3
    model = PolynomialCorrectionModel(kind="voltage", degree=3).fit(voltage, factor)

    np.testing.assert_allclose(model.predict_factor(voltage), factor, atol=1e-12)
    assert len(model.provenance()["coefficients"]) == 4


def test_working_arrays_are_writable_and_detached_from_the_loaded_frame():
    """The calibration steps mutate the working arrays in place.

    ``Series.to_numpy()`` returns a read-only view under pandas copy-on-write,
    and a writable view would write through into the loaded dataframe the
    source hash was computed over, so both accessors must copy.
    """
    frame = pd.DataFrame(
        {
            "t (ns)": [1.0, 2.0, 3.0],
            "high_voltage (V)": [4000.0, 4100.0, 4200.0],
            "pulse_v (V)": [0.0, 0.0, 0.0],
            "x_det (cm)": [0.1, 0.2, 0.3],
            "y_det (cm)": [0.4, 0.5, 0.6],
            "mc (Da)": [10.0, 11.0, 12.0],
            "mc_uc (Da)": [10.0, 11.0, 12.0],
        }
    )
    dataset = CalibrationDataset.from_frame(frame)
    column = dataset.values("t (ns)")
    column[0] = -1.0
    assert dataset.frame["t (ns)"].iloc[0] == 1.0

    variables = Variables()
    variables.sync_from_data(frame)
    source_hash = variables.dataset.source_hash
    for name in ("dld_t", "dld_high_voltage", "dld_pulse_v", "dld_x_det",
                 "dld_y_det", "mc", "mc_uc"):
        array = getattr(variables, name)
        assert array.flags.writeable, name
        array[0] += 1.0
    assert variables.data["t (ns)"].iloc[0] == 1.0
    assert variables.dataset.source_hash == source_hash
