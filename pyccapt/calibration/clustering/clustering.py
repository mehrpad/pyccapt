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

SUPPORTED_CLUSTERING_METHODS = (
    "min-max",
    "maximum-separation",
    "hdbscan",
    "comp-seeded-support-hdbscan",
    "composition-gmm-voxel",
    "compspace-agnostic-seeded",
)


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
    normalized = str(method).strip().lower().replace("_", "-").replace("+", "-").replace(" ", "-")
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    aliases = {
        "min-max": "min-max",
        "minmax": "min-max",
        "composition-gmm-voxel": "composition-gmm-voxel",
        "composition-gmm": "composition-gmm-voxel",
        "gmm-voxel": "composition-gmm-voxel",
        "compspace-agnostic-seeded": "compspace-agnostic-seeded",
        "comp-space-agnostic-seeded": "compspace-agnostic-seeded",
        "agnostic-seeded": "compspace-agnostic-seeded",
        "maximum-separation": "maximum-separation",
        "max-separation": "maximum-separation",
        "maximumseparation": "maximum-separation",
        "maximum-seperation": "maximum-separation",
        "max-seperation": "maximum-separation",
        "mmax-separation": "maximum-separation",
        "mmax-seperation": "maximum-separation",
        "mmax-sepretion": "maximum-separation",
        "hdbscan": "hdbscan",
        "comp-seeded-support-hdbscan": "comp-seeded-support-hdbscan",
        "comp-seeded-hdbscan": "comp-seeded-support-hdbscan",
        "comp-seeded-support": "comp-seeded-support-hdbscan",
        "comp-seeded": "comp-seeded-support-hdbscan",
    }
    resolved = aliases.get(normalized)
    if resolved is None:
        supported = ", ".join(SUPPORTED_CLUSTERING_METHODS)
        raise ValueError(f"Unsupported clustering method {method!r}. Choose one of: {supported}")
    return resolved


def _resolve_xyz(variables) -> np.ndarray:
    """Return the (N, 3) reconstruction coordinates.

    Partial-recovered rows have NaN x/y/z because their detector position
    was incomplete. ``np.column_stack`` happily carries the NaN through,
    but downstream clustering (DBSCAN / HDBSCAN / MinMax) raises on NaN
    inputs. The mask is preserved at full length here so downstream
    consumers like the colour / mc array stay aligned; callers that
    actually pass the array into clustering must drop NaN rows with
    ``np.isnan(coords).any(axis=1)``. We surface a clear notice the first
    time so the omission is obvious.
    """
    x = np.asarray(getattr(variables, "x", np.zeros(0)))
    y = np.asarray(getattr(variables, "y", np.zeros(0)))
    z = np.asarray(getattr(variables, "z", np.zeros(0)))
    if len(x) == 0 or len(y) == 0 or len(z) == 0:
        raise ValueError("Reconstruction coordinates are empty. Run the reconstruction first.")
    if not (len(x) == len(y) == len(z)):
        raise ValueError("Reconstruction coordinates must have the same length.")
    coords = np.column_stack((x, y, z))
    n_nan = int(np.isnan(coords).any(axis=1).sum())
    if n_nan > 0:
        print(
            f'[clustering._resolve_xyz] Note: {n_nan} rows have NaN (x, y, z) '
            '(partial-recovered ions with undefined detector position). '
            'Downstream clustering should mask them out with '
            '``~np.isnan(coords).any(axis=1)``.'
        )
    return coords


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


def _centers_from_labeled_points(points: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute cluster centers for non-negative labels."""
    labels = np.asarray(labels, dtype=int)
    valid = labels >= 0
    if not np.any(valid):
        return np.empty((0, 3), dtype=float)
    centers = []
    for label in sorted(np.unique(labels[valid]).tolist()):
        centers.append(np.asarray(points[labels == label], dtype=float).mean(axis=0))
    return np.asarray(centers, dtype=float)


def _drop_small_clusters(labels: np.ndarray, *, n_min: int) -> np.ndarray:
    """Mark clusters smaller than n_min as noise and compact surviving labels."""
    labels = np.asarray(labels, dtype=int).copy()
    if n_min <= 1:
        return labels
    valid = labels >= 0
    if not np.any(valid):
        return labels
    unique, counts = np.unique(labels[valid], return_counts=True)
    keep = {int(label) for label, count in zip(unique, counts) if int(count) >= int(n_min)}
    labels = np.array([label if int(label) in keep else -1 for label in labels], dtype=int)
    valid = labels >= 0
    if not np.any(valid):
        return labels
    remap = {int(old): idx for idx, old in enumerate(sorted(np.unique(labels[valid]).tolist()))}
    return np.array([remap[int(label)] if label >= 0 else -1 for label in labels], dtype=int)


def min_max_clustering(
    points: np.ndarray,
    n_clusters: int = 2,
    max_iter: int = 50,
    n_min: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment points with a deterministic Min-Max initialization plus centroid refinement.

    Parameters
    ----------
    n_min : optional int
        When provided, clusters with fewer than ``n_min`` members are
        relabelled as noise (-1) and the surviving labels compacted --
        consistent with the HDBSCAN / DBSCAN / maximum-separation
        algorithms in this module, which all drop tiny clusters. The
        default (None) preserves the legacy behaviour of returning
        exactly ``n_clusters`` partitions.

    Notes
    -----
    NaN-coordinate rows (partial-recovered ions) are dropped before
    clustering and re-inserted afterwards with label -1; previously the
    NaN values flowed through ``np.linalg.norm`` / ``argmin`` and
    silently produced garbage labels.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be a (N, 3) array")
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2")

    nan_row_mask = np.isnan(points).any(axis=1)
    finite_points = points[~nan_row_mask]
    if len(finite_points) < n_clusters:
        raise ValueError("Not enough (finite) points for the requested number of clusters")

    centroid = finite_points.mean(axis=0)
    first_index = int(np.argmax(np.linalg.norm(finite_points - centroid, axis=1)))
    centers = [finite_points[first_index]]

    while len(centers) < n_clusters:
        distances = np.stack([np.linalg.norm(finite_points - center, axis=1) for center in centers], axis=1)
        candidate_index = int(np.argmax(np.min(distances, axis=1)))
        centers.append(finite_points[candidate_index])

    centers = np.asarray(centers, dtype=float)
    labels = np.zeros(len(finite_points), dtype=int)

    for _ in range(max_iter):
        distances = np.stack([np.linalg.norm(finite_points - center, axis=1) for center in centers], axis=1)
        new_labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for idx in range(n_clusters):
            cluster_points = finite_points[new_labels == idx]
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

    if n_min is not None:
        labels = _drop_small_clusters(labels, n_min=int(n_min))
        centers = _centers_from_labeled_points(finite_points, labels)

    if nan_row_mask.any():
        # Re-expand to the original input length so callers that index by
        # selection mask stay aligned; NaN rows are noise (-1).
        full = np.full(len(points), -1, dtype=int)
        full[~nan_row_mask] = labels
        labels = full

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


def _expand_maxsep_labels(filtered_labels: np.ndarray, finite_idx: np.ndarray, input_len: int) -> np.ndarray:
    """Re-expand labels computed on the NaN-filtered subset to full length.

    Dropped (NaN-coordinate) rows become -1 (noise) so callers that index
    by the original selection mask stay aligned.
    """
    if len(filtered_labels) == input_len:
        return filtered_labels
    full = np.full(input_len, -1, dtype=int)
    full[finite_idx] = filtered_labels
    return full


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

    # NaN-coordinate rows (partial-recovered ions) would break cKDTree;
    # drop them up front and re-expand labels to the original length with
    # -1 (noise) afterwards, consistent with hdbscan_clustering and
    # min_max_clustering. ``input_len`` / ``finite_idx`` carry the mapping.
    input_len = len(points)
    nan_row_mask = np.isnan(points).any(axis=1)
    finite_idx = np.flatnonzero(~nan_row_mask)
    if nan_row_mask.any():
        print(
            f'[maximum_separation_clustering] Dropping {int(nan_row_mask.sum())} '
            'rows with NaN (x, y, z) (partial-recovered ions) before clustering.'
        )
    points = points[finite_idx]

    n_points = len(points)
    labels = np.full(n_points, -1, dtype=int)
    if n_points < n_min:
        return _expand_maxsep_labels(labels, finite_idx, input_len), np.empty((0, 3), dtype=float)

    tree = cKDTree(points)
    try:
        pairs = tree.query_pairs(d_max, output_type="ndarray")
    except TypeError:
        pairs = np.asarray(sorted(tree.query_pairs(d_max)), dtype=int)

    if pairs.size == 0:
        return _expand_maxsep_labels(labels, finite_idx, input_len), np.empty((0, 3), dtype=float)

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
    # numpy 2.0 briefly returned ``inverse`` with the input's shape
    # (i.e. (n, 1) here); ravel so the ``enumerate(inverse)`` loop below
    # always yields scalar local_root_id values.
    inverse = np.ravel(inverse)
    valid_mask = counts >= n_min
    if not np.any(valid_mask):
        return _expand_maxsep_labels(labels, finite_idx, input_len), np.empty((0, 3), dtype=float)

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
        local_root_id = int(local_root_id)
        if not valid_mask[local_root_id]:
            continue
        # Defensive .get: old_to_new only holds valid roots, and the
        # valid_mask guard above already gates entry, but a guard here
        # makes the relabel robust to any future index-space drift.
        center_idx = old_to_new.get(local_root_id)
        if center_idx is None:
            continue
        labels[point_index] = remap[center_idx]

    centers = centers[order]
    return _expand_maxsep_labels(labels, finite_idx, input_len), centers


def hdbscan_clustering(
    points: np.ndarray,
    *,
    n_min: int,
    min_samples: int = 3,
    d_max: float | None = None,
    auto_d_max: bool = True,
    percentile: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | bool]]:
    """Cluster points using HDBSCAN when available, with DBSCAN fallback."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be a (N, 3) array")
    if len(points) == 0:
        raise ValueError("points cannot be empty")

    # Both HDBSCAN and sklearn DBSCAN raise on NaN coordinates. Drop
    # partial-recovered rows with undefined (x, y, z) before clustering
    # rather than letting the backend explode. ``labels`` is returned at
    # FULL input length: NaN rows get label -1 (noise) so the caller's
    # ``full_labels[selected_indices] = labels`` assignment continues to
    # line up with ``selected_indices``. Previously the returned labels
    # had the filtered length and broadcasting to a longer index slice
    # crashed with ValueError (or silently misaligned in pre-numpy 1.20
    # behaviour, putting every label after the first NaN row on the
    # wrong ion).
    nan_row_mask = np.isnan(points).any(axis=1)
    finite_points = points[~nan_row_mask]
    if nan_row_mask.any():
        print(
            f'[hdbscan_clustering] Dropping {int(nan_row_mask.sum())} rows with NaN '
            '(x, y, z) (partial-recovered ions) before clustering.'
        )
        if len(finite_points) == 0:
            raise ValueError("points has no finite rows after dropping NaN coordinates")

    n_min = max(2, int(n_min))
    min_samples = max(1, int(min_samples))
    parameters: dict[str, float | int | bool] = {
        "n_min": int(n_min),
        "min_samples": int(min_samples),
    }

    def _expand_labels(filtered_labels: np.ndarray) -> np.ndarray:
        """Expand labels back to full input length; NaN rows get -1 (noise)."""
        full = np.full(len(points), -1, dtype=int)
        full[~nan_row_mask] = filtered_labels
        return full

    try:
        import hdbscan as _hdbscan

        clusterer = _hdbscan.HDBSCAN(min_cluster_size=n_min, min_samples=min_samples)
        filtered_labels = np.asarray(clusterer.fit_predict(finite_points), dtype=int)
        filtered_labels = _drop_small_clusters(filtered_labels, n_min=n_min)
        # Compute centers from the filtered (NaN-free) points so the
        # centroid calculation isn't poisoned by undefined coords.
        centers = _centers_from_labeled_points(finite_points, filtered_labels)
        labels = _expand_labels(filtered_labels)
        parameters["backend"] = True
        return labels, centers, parameters
    except Exception:
        # Fallback for environments without hdbscan: density clustering with DBSCAN.
        from sklearn.cluster import DBSCAN

        if auto_d_max or d_max is None:
            eps = estimate_maximum_separation_distance(
                finite_points,
                kth_neighbor=max(1, min_samples),
                percentile=float(percentile),
            )
        else:
            eps = float(d_max)
        filtered_labels = np.asarray(
            DBSCAN(eps=eps, min_samples=n_min).fit_predict(finite_points), dtype=int
        )
        filtered_labels = _drop_small_clusters(filtered_labels, n_min=n_min)
        centers = _centers_from_labeled_points(finite_points, filtered_labels)
        labels = _expand_labels(filtered_labels)
        parameters.update(
            {"backend": False, "d_max": float(eps), "auto_d_max": bool(auto_d_max), "percentile": float(percentile)}
        )
        return labels, centers, parameters


def composition_gmm_voxel_clustering(
    points: np.ndarray,
    *,
    n_clusters: int,
    voxel_size: float = 1.0,
    n_min: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | bool]]:
    """Cluster points by fitting a Gaussian mixture over voxelized spatial features."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be a (N, 3) array")
    n_clusters = max(2, int(n_clusters))
    voxel_size = float(voxel_size)
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be a positive finite number")

    voxel_index = np.floor(points / voxel_size).astype(int)
    unique_voxels, inverse, counts = np.unique(voxel_index, axis=0, return_inverse=True, return_counts=True)
    voxel_centers = (unique_voxels.astype(float) + 0.5) * voxel_size
    # Include normalized occupancy as a weak composition-like signal per voxel.
    occupancy = counts.astype(float) / max(float(np.max(counts)), 1.0)
    features = np.column_stack((voxel_centers, occupancy))

    from sklearn.mixture import GaussianMixture

    n_components = min(n_clusters, len(features))
    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=0)
    voxel_labels = np.asarray(gmm.fit_predict(features), dtype=int)
    point_labels = voxel_labels[inverse]
    point_labels = _drop_small_clusters(point_labels, n_min=max(2, int(n_min)))
    centers = _centers_from_labeled_points(points, point_labels)
    params = {"n_clusters": int(n_clusters), "voxel_size": float(voxel_size), "n_min": int(max(2, int(n_min)))}
    return point_labels, centers, params


def _seed_points_from_selection(
    variables,
    seed_labels: Sequence[str] | str,
) -> np.ndarray:
    labels = parse_label_selection(seed_labels)
    if not labels:
        return np.empty((0, 3), dtype=float)
    xyz = _resolve_xyz(variables)
    seed_mask = _build_selection_mask(variables, labels)
    return xyz[seed_mask]


def compspace_agnostic_seeded_clustering(
    points: np.ndarray,
    *,
    n_clusters: int,
    seed_points: np.ndarray | None = None,
    n_min: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | bool]]:
    """Cluster points with optional seeded initialization, agnostic to composition features."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be a (N, 3) array")
    n_clusters = max(2, int(n_clusters))

    from sklearn.cluster import KMeans

    if seed_points is None:
        seed_points = np.empty((0, 3), dtype=float)
    seed_points = np.asarray(seed_points, dtype=float)
    if seed_points.ndim != 2:
        seed_points = np.empty((0, 3), dtype=float)
    if seed_points.size and seed_points.shape[1] != 3:
        seed_points = np.empty((0, 3), dtype=float)

    init = "k-means++"
    n_init: int | str = "auto"
    if len(seed_points) > 0:
        if len(seed_points) >= n_clusters:
            init = seed_points[:n_clusters]
        else:
            remaining = n_clusters - len(seed_points)
            # Add farthest points for missing seeds.
            centroid = points.mean(axis=0)
            distances = np.linalg.norm(points - centroid, axis=1)
            extra = points[np.argsort(distances)[-remaining:]]
            init = np.vstack((seed_points, extra))
        n_init = 1

    model = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, random_state=0)
    labels = np.asarray(model.fit_predict(points), dtype=int)
    labels = _drop_small_clusters(labels, n_min=max(2, int(n_min)))
    centers = _centers_from_labeled_points(points, labels)
    params = {"n_clusters": int(n_clusters), "n_min": int(max(2, int(n_min))), "seeded": bool(len(seed_points) > 0)}
    return labels, centers, params


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
    voxel_size: float = 1.0,
    seed_labels: Sequence[str] | str | None = None,
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

    ion_names_tuple = parse_label_selection(ion_names)
    xyz = _resolve_xyz(variables)
    selection_mask = _build_selection_mask(variables, ion_names_tuple)
    selected_points = xyz[selection_mask]

    if normalized_method == "maximum-separation":
        d_max_value = (
            estimate_maximum_separation_distance(selected_points, kth_neighbor=kth_neighbor, percentile=percentile)
            if auto_d_max or d_max is None
            else float(d_max)
        )
        labels, centers = maximum_separation_clustering(
            selected_points,
            d_max=d_max_value,
            n_min=n_min,
        )
        params = {
            "d_max": float(d_max_value),
            "n_min": int(n_min),
            "kth_neighbor": int(kth_neighbor),
            "percentile": float(percentile),
            "auto_d_max": bool(auto_d_max),
        }
        return _build_cluster_result(
            variables,
            ion_names=ion_names_tuple,
            selection_mask=selection_mask,
            labels=labels,
            centers=centers,
            cluster_column=cluster_column or "cluster_maxsep",
            algorithm="maximum-separation",
            parameters=params,
        )

    if normalized_method == "hdbscan":
        labels, centers, params = hdbscan_clustering(
            selected_points,
            n_min=n_min,
            min_samples=kth_neighbor,
            d_max=d_max,
            auto_d_max=auto_d_max,
            percentile=percentile,
        )
        return _build_cluster_result(
            variables,
            ion_names=ion_names_tuple,
            selection_mask=selection_mask,
            labels=labels,
            centers=centers,
            cluster_column=cluster_column or "cluster_hdbscan",
            algorithm="hdbscan",
            parameters=params,
        )

    if normalized_method == "comp-seeded-support-hdbscan":
        seed_points = _seed_points_from_selection(variables, seed_labels or ())
        seeded_support = len(seed_points) >= 2
        d_max_seed = d_max
        auto_from_seed = auto_d_max
        if seeded_support and (auto_d_max or d_max is None):
            d_max_seed = estimate_maximum_separation_distance(
                seed_points,
                kth_neighbor=max(1, int(kth_neighbor)),
                percentile=float(percentile),
            )
            auto_from_seed = False
        labels, centers, params = hdbscan_clustering(
            selected_points,
            n_min=n_min,
            min_samples=kth_neighbor,
            d_max=d_max_seed,
            auto_d_max=auto_from_seed,
            percentile=percentile,
        )
        params.update({"seeded_support": bool(seeded_support)})
        return _build_cluster_result(
            variables,
            ion_names=ion_names_tuple,
            selection_mask=selection_mask,
            labels=labels,
            centers=centers,
            cluster_column=cluster_column or "cluster_comp_seed_hdbscan",
            algorithm="comp-seeded-support-hdbscan",
            parameters=params,
        )

    if normalized_method == "composition-gmm-voxel":
        labels, centers, params = composition_gmm_voxel_clustering(
            selected_points,
            n_clusters=n_clusters,
            voxel_size=voxel_size,
            n_min=n_min,
        )
        return _build_cluster_result(
            variables,
            ion_names=ion_names_tuple,
            selection_mask=selection_mask,
            labels=labels,
            centers=centers,
            cluster_column=cluster_column or "cluster_comp_gmm_voxel",
            algorithm="composition-gmm-voxel",
            parameters=params,
        )

    seed_points = _seed_points_from_selection(variables, seed_labels or ())
    labels, centers, params = compspace_agnostic_seeded_clustering(
        selected_points,
        n_clusters=n_clusters,
        seed_points=seed_points,
        n_min=n_min,
    )
    return _build_cluster_result(
        variables,
        ion_names=ion_names_tuple,
        selection_mask=selection_mask,
        labels=labels,
        centers=centers,
        cluster_column=cluster_column or "cluster_comp_seeded",
        algorithm="compspace-agnostic-seeded",
        parameters=params,
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
