from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pyccapt.calibration.core.correction_models import PolynomialCorrectionModel
from pyccapt.calibration.core.dataset import CalibrationDataset


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
