"""Clustering helpers for calibrated APT datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import plotly.graph_objects as go
from scipy.spatial import cKDTree


DEFAULT_CLUSTER_COLORS = (
    "#EF553B",
    "#00CC96",
    "#636EFA",
    "#AB63FA",
    "#FFA15A",
)

SUPPORTED_CLUSTERING_METHODS = ("min-max", "maximum-separation")


@dataclass(frozen=True)
class MinMaxClusterResult:
    """Result of a clustering pass on a selected ion population."""

    labels: np.ndarray
    selected_mask: np.ndarray
    selected_indices: np.ndarray
    centers: np.ndarray
    ion_names: tuple[str, ...]
    cluster_column: str
    algorithm: str = "min-max"
    parameters: dict[str, float | int | bool] | None = None

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


def normalize_clustering_method(method: str) -> str:
    """Return the canonical clustering method identifier."""
    normalized = str(method).strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "min-max": "min-max",
        "minmax": "min-max",
        "maximum-separation": "maximum-separation",
        "max-separation": "maximum-separation",
        "maximumseparation": "maximum-separation",
        "maximum-seperation": "maximum-separation",
        "max-seperation": "maximum-separation",
    }
    resolved = aliases.get(normalized)
    if resolved is None:
        supported = ", ".join(SUPPORTED_CLUSTERING_METHODS)
        raise ValueError(f"Unsupported clustering method {method!r}. Choose one of: {supported}")
    return resolved


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


def _write_cluster_labels(variables, cluster_column: str, labels: np.ndarray) -> None:
    """Persist clustering labels onto shared variables and dataframe when possible."""
    if getattr(variables, "data", None) is not None and len(variables.data) == len(labels):
        variables.data[cluster_column] = labels
    setattr(variables, cluster_column, labels)


def _build_cluster_result(
    variables,
    *,
    ion_names: tuple[str, ...],
    selection_mask: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    cluster_column: str,
    algorithm: str,
    parameters: dict[str, float | int | bool] | None = None,
) -> MinMaxClusterResult:
    """Expand selected-ion labels back to the full reconstruction length."""
    full_labels = np.full(len(selection_mask), -1, dtype=int)
    selected_indices = np.flatnonzero(selection_mask)
    full_labels[selected_indices] = labels
    _write_cluster_labels(variables, cluster_column, full_labels)
    return MinMaxClusterResult(
        labels=full_labels,
        selected_mask=selection_mask,
        selected_indices=selected_indices,
        centers=np.asarray(centers, dtype=float).reshape((-1, 3)) if len(centers) else np.empty((0, 3), dtype=float),
        ion_names=ion_names,
        cluster_column=cluster_column,
        algorithm=algorithm,
        parameters=parameters,
    )


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


def estimate_maximum_separation_distance(
    points: np.ndarray,
    *,
    kth_neighbor: int = 3,
    percentile: float = 50.0,
) -> float:
    """Estimate `d_max` from the kth-nearest-neighbor distance distribution."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be a (N, 3) array")
    if len(points) < 2:
        raise ValueError("At least two points are required to estimate d_max")
    kth_neighbor = int(kth_neighbor)
    if kth_neighbor < 1:
        raise ValueError("kth_neighbor must be at least 1")
    percentile = float(percentile)
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100")

    effective_neighbor = min(kth_neighbor, len(points) - 1)
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=effective_neighbor + 1)
    kth_distances = np.asarray(distances[:, effective_neighbor], dtype=float)
    d_max = float(np.percentile(kth_distances, percentile))
    if not np.isfinite(d_max) or d_max <= 0:
        raise ValueError("Estimated d_max must be a positive finite number")
    return d_max


def maximum_separation_clustering(
    points: np.ndarray,
    *,
    d_max: float,
    n_min: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster points by connected components with maximum edge length `d_max`."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be a (N, 3) array")
    if len(points) == 0:
        raise ValueError("points cannot be empty")
    d_max = float(d_max)
    if not np.isfinite(d_max) or d_max <= 0:
        raise ValueError("d_max must be a positive finite number")
    n_min = int(n_min)
    if n_min < 2:
        raise ValueError("n_min must be at least 2")

    n_points = len(points)
    labels = np.full(n_points, -1, dtype=int)
    if n_points < n_min:
        return labels, np.empty((0, 3), dtype=float)

    tree = cKDTree(points)
    try:
        pairs = tree.query_pairs(d_max, output_type="ndarray")
    except TypeError:
        pairs = np.asarray(sorted(tree.query_pairs(d_max)), dtype=int)

    if pairs.size == 0:
        return labels, np.empty((0, 3), dtype=float)

    parent = np.arange(n_points, dtype=int)
    size = np.ones(n_points, dtype=int)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        if size[root_left] < size[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        size[root_left] += size[root_right]

    for left, right in np.asarray(pairs, dtype=int):
        union(int(left), int(right))

    roots = np.fromiter((find(index) for index in range(n_points)), count=n_points, dtype=int)
    unique_roots, inverse, counts = np.unique(roots, return_inverse=True, return_counts=True)
    valid_mask = counts >= n_min
    if not np.any(valid_mask):
        return labels, np.empty((0, 3), dtype=float)

    centers = []
    cluster_sizes = []
    old_to_new: dict[int, int] = {}
    valid_root_ids = np.flatnonzero(valid_mask)
    for local_root_id in valid_root_ids:
        member_mask = inverse == local_root_id
        centers.append(points[member_mask].mean(axis=0))
        cluster_sizes.append(int(np.count_nonzero(member_mask)))
        old_to_new[int(local_root_id)] = len(centers) - 1

    centers = np.asarray(centers, dtype=float)
    cluster_sizes = np.asarray(cluster_sizes, dtype=int)
    order = np.lexsort((centers[:, 2], centers[:, 1], centers[:, 0], -cluster_sizes))

    remap = {int(old_idx): int(new_idx) for new_idx, old_idx in enumerate(order)}
    for point_index, local_root_id in enumerate(inverse):
        if not valid_mask[local_root_id]:
            continue
        labels[point_index] = remap[old_to_new[int(local_root_id)]]

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
    labels, centers = min_max_clustering(xyz[selection_mask], n_clusters=n_clusters)
    return _build_cluster_result(
        variables,
        ion_names=ion_names_tuple,
        selection_mask=selection_mask,
        labels=labels,
        centers=centers,
        cluster_column=cluster_column,
        algorithm="min-max",
        parameters={"n_clusters": int(n_clusters)},
    )


def segment_ions_by_maximum_separation(
    variables,
    ion_names: Sequence[str] | str,
    *,
    d_max: float | None = None,
    n_min: int = 25,
    auto_d_max: bool = True,
    kth_neighbor: int = 3,
    percentile: float = 50.0,
    cluster_column: str = "cluster_maxsep",
) -> MinMaxClusterResult:
    """Cluster a selected ion population with a fast maximum-separation rule."""
    ion_names_tuple = parse_label_selection(ion_names)
    xyz = _resolve_xyz(variables)
    selection_mask = _build_selection_mask(variables, ion_names_tuple)
    selected_points = xyz[selection_mask]

    if auto_d_max or d_max is None:
        d_max_value = estimate_maximum_separation_distance(
            selected_points,
            kth_neighbor=kth_neighbor,
            percentile=percentile,
        )
    else:
        d_max_value = float(d_max)

    labels, centers = maximum_separation_clustering(
        selected_points,
        d_max=d_max_value,
        n_min=n_min,
    )
    return _build_cluster_result(
        variables,
        ion_names=ion_names_tuple,
        selection_mask=selection_mask,
        labels=labels,
        centers=centers,
        cluster_column=cluster_column,
        algorithm="maximum-separation",
        parameters={
            "d_max": float(d_max_value),
            "n_min": int(n_min),
            "kth_neighbor": int(kth_neighbor),
            "percentile": float(percentile),
            "auto_d_max": bool(auto_d_max),
        },
    )


def segment_ions(
    variables,
    ion_names: Sequence[str] | str,
    *,
    method: str = "min-max",
    n_clusters: int = 2,
    d_max: float | None = None,
    n_min: int = 25,
    auto_d_max: bool = True,
    kth_neighbor: int = 3,
    percentile: float = 50.0,
    cluster_column: str | None = None,
) -> MinMaxClusterResult:
    """Cluster a selected ion population with the requested algorithm."""
    normalized_method = normalize_clustering_method(method)
    if normalized_method == "min-max":
        return segment_ions_by_min_max(
            variables,
            ion_names,
            n_clusters=n_clusters,
            cluster_column=cluster_column or "cluster_minmax",
        )
    return segment_ions_by_maximum_separation(
        variables,
        ion_names,
        d_max=d_max,
        n_min=n_min,
        auto_d_max=auto_d_max,
        kth_neighbor=kth_neighbor,
        percentile=percentile,
        cluster_column=cluster_column or "cluster_maxsep",
    )


def build_cluster_scatter_traces(
    variables,
    cluster_result: MinMaxClusterResult,
    *,
    opacity: float = 0.9,
    marker_size: float = 2.5,
    valid_mask: np.ndarray | None = None,
) -> list[go.Scatter3d]:
    """Build Plotly traces for clustered precipitate segments."""
    if valid_mask is None:
        mask_limit = np.ones(len(cluster_result.labels), dtype=bool)
    else:
        mask_limit = np.asarray(valid_mask, dtype=bool)
        if mask_limit.shape != cluster_result.labels.shape:
            raise ValueError("valid_mask must match the cluster label array length")

    traces: list[go.Scatter3d] = []
    for label_index in range(cluster_result.n_clusters):
        mask = (cluster_result.labels == label_index) & mask_limit
        if not np.any(mask):
            continue
        cluster_size = int(np.count_nonzero(mask))
        traces.append(
            go.Scatter3d(
                x=np.asarray(variables.x)[mask],
                y=np.asarray(variables.y)[mask],
                z=np.asarray(variables.z)[mask],
                mode="markers",
                name=f"Cluster {label_index + 1} (n={cluster_size})",
                showlegend=True,
                legendgroup="clusters",
                legendrank=10 + label_index,
                marker=dict(
                    size=marker_size,
                    color=DEFAULT_CLUSTER_COLORS[label_index % len(DEFAULT_CLUSTER_COLORS)],
                    opacity=opacity,
                ),
            )
        )
    return traces


def build_cluster_context_trace(
    variables,
    *,
    mask: np.ndarray,
    name: str,
    color: str = "rgba(160,160,160,0.55)",
    opacity: float = 0.12,
    marker_size: float = 1.0,
    showlegend: bool = True,
) -> go.Scatter3d | None:
    """Build a faint context trace to show specimen geometry around clusters."""
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0 or not np.any(mask):
        return None
    return go.Scatter3d(
        x=np.asarray(variables.x)[mask],
        y=np.asarray(variables.y)[mask],
        z=np.asarray(variables.z)[mask],
        mode="markers",
        name=name,
        showlegend=showlegend,
        legendgroup="cluster-context",
        legendrank=1,
        marker=dict(
            size=marker_size,
            color=color,
            opacity=opacity,
        ),
    )


__all__ = [
    "MinMaxClusterResult",
    "SUPPORTED_CLUSTERING_METHODS",
    "build_cluster_context_trace",
    "build_cluster_scatter_traces",
    "estimate_maximum_separation_distance",
    "min_max_clustering",
    "maximum_separation_clustering",
    "normalize_clustering_method",
    "parse_label_selection",
    "segment_ions",
    "segment_ions_by_maximum_separation",
    "segment_ions_by_min_max",
]
