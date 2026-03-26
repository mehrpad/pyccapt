"""Reflectron detector-distortion correction based on FAU MATLAB workflows."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from pyccapt.calibration.data_tools import data_tools
from pyccapt.calibration.leap_tools import ccapt_tools, leap_tools
from pyccapt.calibration.path_utils import build_output_path


DATA_DIR = Path(__file__).resolve().parent / "data" / "presets"


@dataclass(frozen=True)
class ReflectronPreset:
    """Bundled reflectron correction preset exported from the MATLAB toolbox."""

    key: str
    display_name: str
    source_mat_file: str
    intersections_csv: str
    triangles_csv: str


@dataclass(frozen=True)
class ReflectronMesh:
    """Reflectron correction mesh for affine detector-to-grid remapping."""

    preset: ReflectronPreset
    intersections: pd.DataFrame
    triangles: np.ndarray

    @property
    def detector_vertices(self) -> np.ndarray:
        return self.intersections[["detectorY", "detectorX"]].to_numpy(dtype=float, copy=False)

    @property
    def grid_vertices(self) -> np.ndarray:
        return self.intersections[["gridX", "gridY"]].to_numpy(dtype=float, copy=False)


BUILTIN_REFLECTRON_PRESETS: dict[str, ReflectronPreset] = {
    "3000xhr_leoben_21_14093": ReflectronPreset(
        key="3000xhr_leoben_21_14093",
        display_name="3000 XHR LEAP - Leoben 21_14093",
        source_mat_file="3000XHR_Leoben_21_14093_intersections.mat",
        intersections_csv="3000XHR_Leoben_21_14093_intersections.csv",
        triangles_csv="3000XHR_Leoben_21_14093_triangles.csv",
    ),
    "4000xhr_erlangen_56_4833": ReflectronPreset(
        key="4000xhr_erlangen_56_4833",
        display_name="4000 XHR LEAP - Erlangen 56_4833",
        source_mat_file="4000XHR_Erlangen_56_4833_intersections.mat",
        intersections_csv="4000XHR_Erlangen_56_4833_intersections.csv",
        triangles_csv="4000XHR_Erlangen_56_4833_triangles.csv",
    ),
    "5000xr_leoben_5124_368": ReflectronPreset(
        key="5000xr_leoben_5124_368",
        display_name="5000 XR LEAP - Leoben 5124_368",
        source_mat_file="5000XR_Leoben_5124_368_intersections.mat",
        intersections_csv="5000XR_Leoben_5124_368_intersections.csv",
        triangles_csv="5000XR_Leoben_5124_368_triangles.csv",
    ),
    "5000xr_oxford_5083_23091": ReflectronPreset(
        key="5000xr_oxford_5083_23091",
        display_name="5000 XR LEAP - Oxford 5083_23091",
        source_mat_file="5000XR_Oxford_5083_23091_intersections.mat",
        intersections_csv="5000XR_Oxford_5083_23091_intersections.csv",
        triangles_csv="5000XR_Oxford_5083_23091_triangles.csv",
    ),
    "6000xr_chalmers_6002_770": ReflectronPreset(
        key="6000xr_chalmers_6002_770",
        display_name="6000 XHR LEAP - Chalmers 6002_770",
        source_mat_file="6000XR_Chalmers_6002_770_intersections.mat",
        intersections_csv="6000XR_Chalmers_6002_770_intersections.csv",
        triangles_csv="6000XR_Chalmers_6002_770_triangles.csv",
    ),
}


def list_builtin_presets() -> dict[str, str]:
    """Return built-in preset labels keyed by stable preset id."""
    return {key: preset.display_name for key, preset in BUILTIN_REFLECTRON_PRESETS.items()}


def _normalize_preset_key(preset_key: str) -> str:
    key = str(preset_key).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {preset.display_name.lower().replace(" ", "_").replace("-", "_"): name
               for name, preset in BUILTIN_REFLECTRON_PRESETS.items()}
    aliases.update({name: name for name in BUILTIN_REFLECTRON_PRESETS})
    if key not in aliases:
        supported = ", ".join(sorted(BUILTIN_REFLECTRON_PRESETS))
        raise ValueError(f"Unknown reflectron preset {preset_key!r}. Choose one of: {supported}")
    return aliases[key]


@lru_cache(maxsize=None)
def load_builtin_preset(preset_key: str) -> ReflectronMesh:
    """Load one bundled reflectron mesh preset."""
    normalized_key = _normalize_preset_key(preset_key)
    preset = BUILTIN_REFLECTRON_PRESETS[normalized_key]
    intersections_path = DATA_DIR / preset.intersections_csv
    triangles_path = DATA_DIR / preset.triangles_csv
    if not intersections_path.is_file() or not triangles_path.is_file():
        raise FileNotFoundError(
            f"Reflectron preset files are missing for {preset.display_name}. "
            f"Expected {intersections_path} and {triangles_path}."
        )

    intersections = pd.read_csv(intersections_path)
    triangles = pd.read_csv(triangles_path, header=None).to_numpy(dtype=int, copy=False) - 1
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"Invalid triangle mesh shape for {preset.display_name}: {triangles.shape}")
    return ReflectronMesh(preset=preset, intersections=intersections, triangles=triangles)


def _extract_detector_mm(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, str]:
    """Return detector coordinates in millimeters and the dataframe layout type."""
    if {"det_x (mm)", "det_y (mm)"} <= set(data.columns):
        return (
            data["det_x (mm)"].to_numpy(dtype=float, copy=False),
            data["det_y (mm)"].to_numpy(dtype=float, copy=False),
            "epos",
        )
    if {"x_det (cm)", "y_det (cm)"} <= set(data.columns):
        return (
            data["x_det (cm)"].to_numpy(dtype=float, copy=False) * 10.0,
            data["y_det (cm)"].to_numpy(dtype=float, copy=False) * 10.0,
            "ccapt",
        )
    if {"detx", "dety"} <= set(data.columns):
        return (
            data["detx"].to_numpy(dtype=float, copy=False),
            data["dety"].to_numpy(dtype=float, copy=False),
            "matlab_table",
        )
    raise ValueError(
        "Reflectron correction requires detector coordinates in "
        "`det_x (mm)`/`det_y (mm)`, `x_det (cm)`/`y_det (cm)`, or `detx`/`dety` columns."
    )


def _assign_detector_mm(data: pd.DataFrame, detx_mm: np.ndarray, dety_mm: np.ndarray, layout: str) -> pd.DataFrame:
    """Write corrected detector coordinates back to a copied dataframe."""
    corrected = data.copy()
    if layout == "epos":
        corrected["det_x (mm)"] = np.asarray(detx_mm, dtype=float)
        corrected["det_y (mm)"] = np.asarray(dety_mm, dtype=float)
    elif layout == "ccapt":
        corrected["x_det (cm)"] = np.asarray(detx_mm, dtype=float) / 10.0
        corrected["y_det (cm)"] = np.asarray(dety_mm, dtype=float) / 10.0
    elif layout == "matlab_table":
        corrected["detx"] = np.asarray(detx_mm, dtype=float)
        corrected["dety"] = np.asarray(dety_mm, dtype=float)
    else:
        raise ValueError(f"Unsupported detector layout {layout!r}")
    return corrected


def _first_triangle_per_vertex(triangles: np.ndarray, n_vertices: int) -> np.ndarray:
    """Return the first triangle index using each vertex, matching MATLAB's `find(..., 1)` semantics."""
    mapping = np.full(n_vertices, -1, dtype=int)
    for triangle_index, row in enumerate(np.asarray(triangles, dtype=int)):
        for vertex_index in row:
            if 0 <= int(vertex_index) < n_vertices and mapping[int(vertex_index)] == -1:
                mapping[int(vertex_index)] = triangle_index
    return mapping


def _triangle_finder(mesh: ReflectronMesh):
    triangulation = mtri.Triangulation(
        mesh.intersections["detectorY"].to_numpy(dtype=float, copy=False),
        mesh.intersections["detectorX"].to_numpy(dtype=float, copy=False),
        triangles=mesh.triangles,
    )
    return triangulation.get_trifinder()


def _barycentric_coordinates(triangle_vertices: np.ndarray, query_points: np.ndarray) -> np.ndarray:
    """Compute barycentric coordinates for 2D query points inside a triangle."""
    a = triangle_vertices[0]
    b = triangle_vertices[1]
    c = triangle_vertices[2]
    v0 = b - a
    v1 = c - a
    den = v0[0] * v1[1] - v1[0] * v0[1]
    if np.isclose(den, 0.0):
        raise ValueError("Degenerate reflectron triangle encountered in preset mesh")

    v2 = query_points - a
    w1 = (v2[:, 0] * v1[1] - v1[0] * v2[:, 1]) / den
    w2 = (v0[0] * v2[:, 1] - v2[:, 0] * v0[1]) / den
    w0 = 1.0 - w1 - w2
    return np.column_stack([w0, w1, w2])


def correct_detector_coordinates(
    detx_mm: np.ndarray | pd.Series,
    dety_mm: np.ndarray | pd.Series,
    mesh: ReflectronMesh,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the MATLAB reflectron affine mesh correction to detector coordinates in millimeters."""
    detx = np.asarray(detx_mm, dtype=float).reshape(-1)
    dety = np.asarray(dety_mm, dtype=float).reshape(-1)
    if detx.shape != dety.shape:
        raise ValueError("detx and dety must have the same shape")

    query_points = np.column_stack([detx, dety])
    detector_vertices = mesh.detector_vertices
    grid_vertices = mesh.grid_vertices
    triangles = np.asarray(mesh.triangles, dtype=int)

    tri_finder = _triangle_finder(mesh)
    triangle_number = np.asarray(tri_finder(detx, dety), dtype=int)
    outside_mask = triangle_number < 0
    if np.any(outside_mask):
        nearest_tree = cKDTree(detector_vertices)
        closest_vertex = nearest_tree.query(query_points[outside_mask], k=1)[1]
        fallback_triangles = _first_triangle_per_vertex(triangles, len(detector_vertices))[closest_vertex]
        unresolved = fallback_triangles < 0
        if np.any(unresolved):
            raise ValueError("Could not resolve a fallback triangle for points outside the reflectron mesh")
        triangle_number[outside_mask] = fallback_triangles

    corrected_coords = np.empty((len(query_points), 2), dtype=float)
    for triangle_index in np.unique(triangle_number):
        point_mask = triangle_number == triangle_index
        vertex_ids = triangles[int(triangle_index)]
        bary = _barycentric_coordinates(detector_vertices[vertex_ids], query_points[point_mask])
        corrected_coords[point_mask] = bary @ grid_vertices[vertex_ids]

    corrected_detx = corrected_coords[:, 0]
    corrected_dety = corrected_coords[:, 1]
    return corrected_detx, corrected_dety


def apply_reflectron_correction_to_epos(epos_table: pd.DataFrame, mesh: ReflectronMesh) -> pd.DataFrame:
    """Apply reflectron correction to an EPOS-format dataframe."""
    detx_mm, dety_mm, layout = _extract_detector_mm(epos_table)
    corrected_detx, corrected_dety = correct_detector_coordinates(detx_mm, dety_mm, mesh)
    return _assign_detector_mm(epos_table, corrected_detx, corrected_dety, layout=layout)


def apply_reflectron_correction_to_ccapt(data: pd.DataFrame, mesh: ReflectronMesh) -> pd.DataFrame:
    """Apply reflectron correction to a PyCCAPT-format dataframe."""
    detx_mm, dety_mm, layout = _extract_detector_mm(data)
    corrected_detx, corrected_dety = correct_detector_coordinates(detx_mm, dety_mm, mesh)
    return _assign_detector_mm(data, corrected_detx, corrected_dety, layout=layout)


def reflectron_image_transform(data: pd.DataFrame, mesh: ReflectronMesh) -> pd.DataFrame:
    """Apply the alternate `reflectronImageTransform.m` sign/swizzle convention."""
    detx_mm, dety_mm, layout = _extract_detector_mm(data)
    corrected_detx, corrected_dety = correct_detector_coordinates(detx_mm, dety_mm, mesh)
    transformed_detx = -corrected_dety
    transformed_dety = -corrected_detx
    return _assign_detector_mm(data, transformed_detx, transformed_dety, layout=layout)


def correct_epos_file(
    epos_path: str | Path,
    preset_key: str,
    *,
    output_epos_path: str | Path | None = None,
    output_h5_path: str | Path | None = None,
) -> dict[str, str | pd.DataFrame]:
    """Correct one EPOS file with a bundled preset and optionally save the outputs."""
    epos_path = Path(epos_path).expanduser()
    mesh = load_builtin_preset(preset_key)
    corrected_ccapt = apply_reflectron_correction_to_ccapt(
        ccapt_tools.epos_to_ccapt(str(epos_path)),
        mesh,
    )

    outputs: dict[str, str | pd.DataFrame] = {
        "corrected_ccapt": corrected_ccapt,
        "preset": mesh.preset.display_name,
    }

    if output_epos_path is not None:
        output_epos = Path(output_epos_path).expanduser()
    else:
        output_epos = epos_path.with_name(f"{epos_path.stem}_corrected.epos")

    ccapt_tools.ccapt_to_epos(
        corrected_ccapt,
        path=str(output_epos.parent) + "\\",
        name=output_epos.name,
    )
    outputs["epos_path"] = str(output_epos)

    if output_h5_path is not None:
        output_h5 = Path(output_h5_path).expanduser()
        data_tools.store_df_to_hdf(corrected_ccapt, "df", output_h5)
        outputs["h5_path"] = str(output_h5)

    return outputs


def _detector_histogram(data: pd.DataFrame, bins: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    detx_mm, dety_mm, _ = _extract_detector_mm(data)
    histogram, x_edges, y_edges = np.histogram2d(detx_mm, dety_mm, bins=int(bins))
    return histogram.T, x_edges, y_edges


def plot_detector_maps(
    raw_data: pd.DataFrame,
    corrected_data: pd.DataFrame,
    *,
    bins: int = 256,
    title_prefix: str = "Reflectron correction",
):
    """Plot detector hit maps before and after correction."""
    raw_hist, raw_x_edges, raw_y_edges = _detector_histogram(raw_data, bins=bins)
    corrected_hist, corrected_x_edges, corrected_y_edges = _detector_histogram(corrected_data, bins=bins)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, hist, x_edges, y_edges, title in [
        (axes[0], raw_hist, raw_x_edges, raw_y_edges, "Before correction"),
        (axes[1], corrected_hist, corrected_x_edges, corrected_y_edges, "After correction"),
    ]:
        image = ax.pcolormesh(x_edges, y_edges, hist, shading="auto")
        ax.set_aspect("equal")
        ax.set_xlabel("detector x [mm]")
        ax.set_ylabel("detector y [mm]")
        ax.set_title(title)
        figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    figure.suptitle(title_prefix)
    return figure


def save_corrected_outputs(
    corrected_ccapt: pd.DataFrame,
    *,
    output_directory: str | Path,
    output_stem: str,
    save_epos: bool = True,
    save_h5: bool = True,
) -> dict[str, str]:
    """Save corrected datasets in one or more formats."""
    output_directory = Path(output_directory).expanduser()
    outputs: dict[str, str] = {}
    if save_epos:
        output_epos = build_output_path(output_directory, f"{output_stem}.epos")
        ccapt_tools.ccapt_to_epos(
            corrected_ccapt,
            path=str(output_epos.parent) + "\\",
            name=output_epos.name,
        )
        outputs["epos"] = str(output_epos)
    if save_h5:
        output_h5 = build_output_path(output_directory, f"{output_stem}.h5")
        data_tools.store_df_to_hdf(corrected_ccapt, "df", output_h5)
        outputs["h5"] = str(output_h5)
    return outputs


def load_epos_for_reflectron_correction(epos_path: str | Path) -> pd.DataFrame:
    """Load an EPOS file into the PyCCAPT dataframe convention for correction."""
    return ccapt_tools.epos_to_ccapt(str(Path(epos_path).expanduser()))


__all__ = [
    "BUILTIN_REFLECTRON_PRESETS",
    "ReflectronMesh",
    "ReflectronPreset",
    "apply_reflectron_correction_to_ccapt",
    "apply_reflectron_correction_to_epos",
    "correct_detector_coordinates",
    "correct_epos_file",
    "list_builtin_presets",
    "load_builtin_preset",
    "load_epos_for_reflectron_correction",
    "plot_detector_maps",
    "reflectron_image_transform",
    "save_corrected_outputs",
]
