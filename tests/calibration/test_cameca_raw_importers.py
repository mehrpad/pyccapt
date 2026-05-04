import numpy as np
import pandas as pd
import pytest

from pyccapt.calibration.leap_tools.cameca_raw import (
    rhit_apply_calibration,
    rhit_to_ccapt,
    str_calculate_positions,
    str_to_ccapt,
)


def test_rhit_apply_calibration_uses_saved_polynomial():
    hits = pd.DataFrame(
        {
            "detx": [0.0, 1.0],
            "dety": [0.0, 2.0],
            "tof": [10.0, 12.0],
            "VDC": [1000.0, 1000.0],
        }
    )
    calibration = {"t_offset": 1.0, "C_poly": [1.0, 0.5, 0.1]}

    calibrated = rhit_apply_calibration(hits, calibration)

    assert calibrated["mc"].tolist() == pytest.approx([81000.0, 726000.0])


def test_rhit_to_ccapt_builds_processed_dataset_columns():
    hits = pd.DataFrame(
        {
            "mc": [1.0, 2.0],
            "VDC": [1000.0, 1100.0],
            "tof": [5.0, 6.0],
            "detx": [1.0, 2.0],
            "dety": [3.0, 4.0],
            "pulse": [10.0, 20.0],
        }
    )

    dataset = rhit_to_ccapt(hits)

    assert list(dataset.columns[:5]) == ["x (nm)", "y (nm)", "z (nm)", "mc (Da)", "mc_uc (Da)"]
    assert dataset["x_det (cm)"].tolist() == pytest.approx([0.1, 0.2])
    assert dataset["pulse_v (V)"].tolist() == pytest.approx([10.0, 20.0])


def test_str_calculate_positions_recovers_partial_hits():
    hits = pd.DataFrame(
        {
            "ionIdx": [1, 2, 3, 4],
            "detxt1": [100.0, 110.0, np.nan, 120.0],
            "detxt2": [80.0, 90.0, np.nan, 100.0],
            "detyt1": [200.0, 210.0, 220.0, np.nan],
            "detyt2": [180.0, 190.0, 200.0, np.nan],
            "detwt1": [314.1421, np.nan, 334.1421, 344.1421],
            "detwt2": [285.8579, np.nan, 305.8579, 315.8579],
            "quality": [0.0, 0.0, 0.0, 0.0],
        }
    )

    result = str_calculate_positions(hits)

    assert result["detxRaw"].tolist() == pytest.approx([20.0, 20.0, 20.0, 20.0], abs=1e-3)
    assert result["detyRaw"].tolist() == pytest.approx([20.0, 20.0, 20.0, 20.0], abs=1e-3)
    assert result["hitType"].tolist() == [3, 2, 2, 2]
    assert result["tof"].notna().all()


def test_str_to_ccapt_requires_calibration_columns():
    hits = pd.DataFrame({"tof": [1.0], "detxRaw": [2.0], "detyRaw": [3.0], "hitType": [2]})

    with pytest.raises(ValueError):
        str_to_ccapt(hits)


def test_str_to_ccapt_uses_unit_multi_not_hit_type():
    hits = pd.DataFrame(
        {
            "mc": [10.0, 11.0],
            "tof_ns": [1.0, 2.0],
            "VDC": [1000.0, 1001.0],
            "detx": [1.0, 2.0],
            "dety": [3.0, 4.0],
            "hitType": [2, 3],
            "ionIdx": [1, 2],
        }
    )

    dataset = str_to_ccapt(hits)

    assert dataset["multi"].tolist() == [1, 1]
