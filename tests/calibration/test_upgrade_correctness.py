from __future__ import annotations

import numpy as np
import pandas as pd

from pyccapt.calibration.core.share_variables import Variables
from pyccapt.calibration.data_tools import data_tools, raw_data_surface_concept


def test_surface_concept_incomplete_repeated_group_terminates_and_is_invalid():
    channels, _, _, flags = raw_data_surface_concept._normalize_sequence(
        [7, 7, 7, 7, 7], [0, 1, 2, 0, 1], [10, 11, 12, 13, 14]
    )
    assert len(channels) == 5
    assert flags == [False, False]


def test_surface_concept_empty_input_has_no_synthetic_event():
    groups = raw_data_surface_concept.find_consecutive_sequences_seperatly([], [], [], [], [])
    assert all(group == [] for group in groups)


def test_extract_data_clears_stale_positions_and_marks_missing_detector_capability():
    variables = Variables()
    variables.x = np.array([99.0])
    frame = pd.DataFrame({"t (ns)": [10.0, 11.0], "high_voltage (V)": [1000.0, 1001.0]})

    data_tools.extract_data(frame, variables, flightPathLength_d=100.0, max_mc=100.0)

    assert np.isnan(variables.dld_x_det).all()
    assert np.isnan(variables.x).all()
    assert variables.has_detector_positions is False
    assert variables.has_reconstruction is False
