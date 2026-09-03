from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pyccapt.calibration.clustering.isosurface import pos_to_voxel
from pyccapt.calibration.core.correction_models import PolynomialCorrectionModel
from pyccapt.calibration.core.dataset import CalibrationDataset
from pyccapt.calibration.data_tools._raw_workflow_surface_concept import (
    summarize_surface_concept_raw_sequences,
    summarize_surface_concept_sequences,
)


def test_random_detector_sequences_preserve_count_invariants():
    rng = np.random.default_rng(20260903)
    for _ in range(100):
        records = []
        expected_timestamps = 0
        for _ in range(int(rng.integers(0, 30))):
            size = int(rng.integers(0, 25))
            channels = rng.integers(0, 4, size=size).tolist()
            records.append(
                {
                    "channels": channels,
                    "time_data": rng.integers(1, 100_000, size=size).tolist(),
                    "valid_event": [False] * int(np.ceil(size / 4)),
                }
            )
            expected_timestamps += size

        raw = summarize_surface_concept_raw_sequences(records)
        recovered = summarize_surface_concept_sequences(records)
        assert raw["total_timestamps"] == expected_timestamps
        assert sum(raw["channel_timestamp_totals"].values()) == expected_timestamps
        assert sum(length * count for length, count in recovered["total"].items()) == expected_timestamps
        assert all(count >= 0 for group in recovered.values() for count in group.values())


def test_partial_hit_nan_masks_are_aligned_and_stable():
    frame = pd.DataFrame(
        {
            "x_det (cm)": [0.1, np.nan, 0.3, np.inf],
            "y_det (cm)": [0.2, 0.2, np.nan, 0.4],
            "t (ns)": [10.0, np.nan, 12.0, 13.0],
            "mc (Da)": [1.0, 2.0, np.nan, 4.0],
        }
    )
    dataset = CalibrationDataset.from_frame(frame)
    np.testing.assert_array_equal(dataset.finite_mask("detector_positions"), [True, False, False, False])
    np.testing.assert_array_equal(dataset.finite_mask("time_of_flight"), [True, False, True, True])
    assert dataset.row_count == len(frame)
    assert dataset.source_hash == CalibrationDataset.from_frame(frame.copy()).source_hash


def test_golden_calibration_and_voxel_reconstruction():
    fixture_path = Path(__file__).with_name("data") / "golden_scientific_dataset.json"
    golden = json.loads(fixture_path.read_text(encoding="utf-8"))

    model = PolynomialCorrectionModel(kind="voltage").fit(
        golden["voltage"], golden["correction_factor"]
    )
    np.testing.assert_allclose(model.coefficients, golden["expected_coefficients"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(model.predict_factor(golden["voltage"]), golden["correction_factor"], atol=1e-12)

    voxels = pos_to_voxel(
        np.asarray(golden["positions_nm"], dtype=float),
        [np.asarray(axis, dtype=float) for axis in golden["grid_centers_nm"]],
    )
    np.testing.assert_array_equal(voxels, golden["expected_voxels"])
