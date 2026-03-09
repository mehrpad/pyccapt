from __future__ import annotations

import numpy as np
import pandas as pd

from pyccapt.calibration.clustering import min_max_clustering, segment_ions_by_min_max
from pyccapt.calibration.core.share_variables import Variables


def test_min_max_clustering_splits_two_point_clouds():
    rng = np.random.default_rng(42)
    left = rng.normal(loc=(-5.0, 0.0, 0.0), scale=0.15, size=(20, 3))
    right = rng.normal(loc=(5.0, 0.0, 0.0), scale=0.15, size=(20, 3))
    points = np.vstack((left, right))

    labels, centers = min_max_clustering(points, n_clusters=2)

    assert set(labels) == {0, 1}
    assert centers.shape == (2, 3)
    assert centers[0, 0] < 0
    assert centers[1, 0] > 0


def test_segment_ions_by_min_max_writes_cluster_labels_to_variables_data():
    rng = np.random.default_rng(7)
    variables = Variables()
    cloud_a = rng.normal(loc=(0.0, 0.0, 0.0), scale=0.1, size=(12, 3))
    cloud_b = rng.normal(loc=(3.0, 3.0, 3.0), scale=0.1, size=(12, 3))
    xyz = np.vstack((cloud_a, cloud_b))
    variables.x = xyz[:, 0]
    variables.y = xyz[:, 1]
    variables.z = xyz[:, 2]
    variables.mc = np.full(len(xyz), 27.0)
    variables.data = pd.DataFrame({"mc (Da)": variables.mc})
    variables.range_data = pd.DataFrame(
        {
            "ion": ["Ni3Al"],
            "mc_low": [26.0],
            "mc_up": [28.0],
            "element": [["Ni", "Al"]],
            "color": ["#ff0000"],
        }
    )

    result = segment_ions_by_min_max(variables, ["Ni3Al"], n_clusters=2)

    assert result.counts == (12, 12)
    assert "cluster_minmax" in variables.data.columns
    assert set(variables.data["cluster_minmax"]) == {0, 1}
