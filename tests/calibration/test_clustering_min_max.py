from __future__ import annotations

import numpy as np
import pandas as pd

from pyccapt.calibration.clustering import (
    MinMaxClusterResult,
    build_cluster_context_trace,
    build_cluster_scatter_traces,
    estimate_maximum_separation_distance,
    maximum_separation_clustering,
    min_max_clustering,
    segment_ions_by_maximum_separation,
    segment_ions_by_min_max,
)
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


def test_maximum_separation_clustering_finds_connected_components_and_noise():
    rng = np.random.default_rng(11)
    cloud_a = rng.normal(loc=(0.0, 0.0, 0.0), scale=0.03, size=(15, 3))
    cloud_b = rng.normal(loc=(2.0, 2.0, 2.0), scale=0.03, size=(14, 3))
    noise = np.array([[6.0, 6.0, 6.0]])
    points = np.vstack((cloud_a, cloud_b, noise))

    labels, centers = maximum_separation_clustering(points, d_max=0.18, n_min=5)

    assert centers.shape == (2, 3)
    assert tuple(np.bincount(labels[labels >= 0])) == (15, 14)
    assert labels[-1] == -1


def test_segment_ions_by_maximum_separation_auto_estimates_distance_and_writes_labels():
    rng = np.random.default_rng(19)
    variables = Variables()
    cloud_a = rng.normal(loc=(-1.0, 0.0, 0.0), scale=0.03, size=(18, 3))
    cloud_b = rng.normal(loc=(1.2, 0.1, 0.1), scale=0.03, size=(16, 3))
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

    result = segment_ions_by_maximum_separation(
        variables,
        ["Ni3Al"],
        auto_d_max=True,
        kth_neighbor=3,
        percentile=90.0,
        n_min=5,
    )

    assert result.algorithm == "maximum-separation"
    assert result.n_clusters == 2
    assert sum(result.counts) >= 30
    assert "cluster_maxsep" in variables.data.columns
    assert {0, 1}.issubset(set(variables.data["cluster_maxsep"]))
    assert result.parameters is not None
    assert result.parameters["d_max"] > 0


def test_estimate_maximum_separation_distance_returns_positive_value():
    rng = np.random.default_rng(23)
    points = rng.normal(loc=(0.0, 0.0, 0.0), scale=0.1, size=(30, 3))

    d_max = estimate_maximum_separation_distance(points, kth_neighbor=3, percentile=50.0)

    assert np.isfinite(d_max)
    assert d_max > 0


def test_cluster_plot_traces_are_named_by_cluster_and_respect_masks():
    variables = Variables()
    variables.x = np.array([0.0, 1.0, 2.0, 3.0])
    variables.y = np.array([0.0, 0.0, 1.0, 1.0])
    variables.z = np.array([0.0, 0.5, 1.0, 1.5])

    cluster_result = MinMaxClusterResult(
        labels=np.array([0, 0, 1, -1]),
        selected_mask=np.array([True, True, True, True]),
        selected_indices=np.array([0, 1, 2, 3]),
        centers=np.array([[0.5, 0.0, 0.25], [2.0, 1.0, 1.0]]),
        ion_names=("Ni3Al",),
        cluster_column="cluster_minmax",
    )

    traces = build_cluster_scatter_traces(
        variables,
        cluster_result,
        valid_mask=np.array([True, False, True, True]),
    )

    assert [trace.name for trace in traces] == ["Cluster 1 (n=1)", "Cluster 2 (n=1)"]
    assert len(traces[0].x) == 1
    assert len(traces[1].x) == 1

    background = build_cluster_context_trace(
        variables,
        mask=np.array([False, False, False, True]),
        name="Background specimen",
    )
    assert background is not None
    assert background.name == "Background specimen"
