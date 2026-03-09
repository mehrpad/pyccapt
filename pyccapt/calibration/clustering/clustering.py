"""Clustering helpers for calibrated APT datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import plotly.graph_objects as go


DEFAULT_CLUSTER_COLORS = (
    "#EF553B",
    "#00CC96",
    "#636EFA",
    "#AB63FA",
    "#FFA15A",
)


@dataclass(frozen=True)
class MinMaxClusterResult:
    """Result of a Min-Max segmentation on a selected ion population."""

    labels: np.ndarray
    selected_mask: np.ndarray
    selected_indices: np.ndarray
    centers: np.ndarray
    ion_names: tuple[str, ...]
    cluster_column: str
    algorithm: str = "min-max"

    @property
    def counts(self) -> tuple[int, ...]:
        return tuple(int(np.count_nonzero(self.labels == idx)) for idx in range(len(self.centers)))

    @property
    def n_clusters(self) -> int:
        return int(len(self.centers))


def parse_label_selection(selection: str | Iterable[str]) -> tuple[str, ...]:
    """Normalize a comma-separated label selection."""
    if isinstance(selection, str):
        labels = [item.strip() for item in selection.split(",")]
    else:
        labels = [str(item).strip() for item in selection]
    labels = [label for label in labels if label]
    return tuple(dict.fromkeys(labels))


def _resolve_xyz(variables) -> np.ndarray:
    x = np.asarray(getattr(variables, "x", np.zeros(0)))
    y = np.asarray(getattr(variables, "y", np.zeros(0)))
    z = np.asarray(getattr(variables, "z", np.zeros(0)))
    if len(x) == 0 or len(y) == 0 or len(z) == 0:
        raise ValueError("Reconstruction coordinates are empty. Run the reconstruction first.")
    if not (len(x) == len(y) == len(z)):
        raise ValueError("Reconstruction coordinates must have the same length.")
    return np.column_stack((x, y, z))


def _resolve_mc(variables) -> np.ndarray:
    mc = np.asarray(getattr(variables, "mc", np.zeros(0)))
    if len(mc) == 0 and getattr(variables, "data", None) is not None and "mc (Da)" in variables.data.columns:
        mc = variables.data["mc (Da)"].to_numpy()
    if len(mc) == 0:
        raise ValueError("Mass-to-charge data is empty. Load or extract calibrated data first.")
    return mc


def _build_selection_mask(variables, ion_names: Sequence[str]) -> np.ndarray:
    if getattr(variables, "range_data", None) is None or variables.range_data.empty:
        raise ValueError("Range data is required for precipitate clustering.")

    labels = {label.strip() for label in ion_names if str(label).strip()}
    if not labels:
        raise ValueError("Provide at least one ion or element label for clustering.")

    mc = _resolve_mc(variables)
    selection_mask = np.zeros(len(mc), dtype=bool)
    matched_labels: set[str] = set()

    for _, row in variables.range_data.iterrows():
        row_ion = str(row.get("ion", "")).strip()
        row_elements = row.get("element", [])
        if not isinstance(row_elements, (list, tuple, np.ndarray)):
            row_elements = [row_elements]
        row_elements = {str(item).strip() for item in row_elements if str(item).strip()}

        if row_ion in labels or labels.intersection(row_elements):
            row_mask = (mc > float(row["mc_low"])) & (mc < float(row["mc_up"]))
            selection_mask |= row_mask
            if row_ion in labels:
                matched_labels.add(row_ion)
            matched_labels.update(labels.intersection(row_elements))

    if not np.any(selection_mask):
        joined = ", ".join(sorted(labels))
        raise ValueError(f"No ions matched the requested cluster selection: {joined}")
    if not matched_labels:
        joined = ", ".join(sorted(labels))
        raise ValueError(f"None of the requested labels were found in the range data: {joined}")

    return selection_mask


def min_max_clustering(points: np.ndarray, n_clusters: int = 2, max_iter: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Segment points with a deterministic Min-Max initialization plus centroid refinement."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be a (N, 3) array")
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2")
    if len(points) < n_clusters:
        raise ValueError("Not enough points for the requested number of clusters")

    centroid = points.mean(axis=0)
    first_index = int(np.argmax(np.linalg.norm(points - centroid, axis=1)))
    centers = [points[first_index]]

    while len(centers) < n_clusters:
        distances = np.stack([np.linalg.norm(points - center, axis=1) for center in centers], axis=1)
        candidate_index = int(np.argmax(np.min(distances, axis=1)))
        centers.append(points[candidate_index])

    centers = np.asarray(centers, dtype=float)
    labels = np.zeros(len(points), dtype=int)

    for _ in range(max_iter):
        distances = np.stack([np.linalg.norm(points - center, axis=1) for center in centers], axis=1)
        new_labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for idx in range(n_clusters):
            cluster_points = points[new_labels == idx]
            if len(cluster_points) > 0:
                new_centers[idx] = cluster_points.mean(axis=0)
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            labels = new_labels
            centers = new_centers
            break
        labels = new_labels
        centers = new_centers

    order = np.argsort(centers[:, 0], kind="stable")
    remap = {int(old): int(new) for new, old in enumerate(order)}
    labels = np.array([remap[int(label)] for label in labels], dtype=int)
    centers = centers[order]
    return labels, centers


def segment_ions_by_min_max(
    variables,
    ion_names: Sequence[str] | str,
    *,
    n_clusters: int = 2,
    cluster_column: str = "cluster_minmax",
) -> MinMaxClusterResult:
    """Cluster a selected ion population into `n_clusters` precipitate segments."""
    ion_names_tuple = parse_label_selection(ion_names)
    xyz = _resolve_xyz(variables)
    selection_mask = _build_selection_mask(variables, ion_names_tuple)
    selected_indices = np.flatnonzero(selection_mask)
    labels, centers = min_max_clustering(xyz[selection_mask], n_clusters=n_clusters)

    full_labels = np.full(len(xyz), -1, dtype=int)
    full_labels[selected_indices] = labels

    if getattr(variables, "data", None) is not None and len(variables.data) == len(full_labels):
        variables.data[cluster_column] = full_labels
    setattr(variables, cluster_column, full_labels)

    return MinMaxClusterResult(
        labels=full_labels,
        selected_mask=selection_mask,
        selected_indices=selected_indices,
        centers=centers,
        ion_names=ion_names_tuple,
        cluster_column=cluster_column,
    )


def build_cluster_scatter_traces(
    variables,
    cluster_result: MinMaxClusterResult,
    *,
    opacity: float = 0.9,
    marker_size: float = 2.5,
) -> list[go.Scatter3d]:
    """Build Plotly traces for clustered precipitate segments."""
    traces: list[go.Scatter3d] = []
    for label_index in range(cluster_result.n_clusters):
        mask = cluster_result.labels == label_index
        if not np.any(mask):
            continue
        traces.append(
            go.Scatter3d(
                x=np.asarray(variables.x)[mask],
                y=np.asarray(variables.y)[mask],
                z=np.asarray(variables.z)[mask],
                mode="markers",
                name=f"Segment {label_index + 1} ({', '.join(cluster_result.ion_names)})",
                showlegend=True,
                marker=dict(
                    size=marker_size,
                    color=DEFAULT_CLUSTER_COLORS[label_index % len(DEFAULT_CLUSTER_COLORS)],
                    opacity=opacity,
                ),
            )
        )
    return traces


__all__ = [
    "MinMaxClusterResult",
    "build_cluster_scatter_traces",
    "min_max_clustering",
    "parse_label_selection",
    "segment_ions_by_min_max",
]
