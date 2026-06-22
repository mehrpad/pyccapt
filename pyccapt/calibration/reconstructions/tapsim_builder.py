"""Build TAPSim-ready synthetic specimens and node files."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import Element as PMGElement
from scipy.spatial import cKDTree


PRESET_CONFIGS = {
    "Al": {
        "mode": "single-phase",
        "geometry_mode": "bulk",
        "single_phase_composition": {"Al": 1.0},
        "a_gamma": 4.0495,
        "a_gprime": 4.0495,
        "supercell": (18, 18, 24),
        "n_precip": 0,
        "cone_diameter_angstrom": 120.0,
        "cone_length_angstrom": 500.0,
        "hemisphere_base_angstrom": 250.0,
    },
    "Pd": {
        "mode": "single-phase",
        "geometry_mode": "bulk",
        "single_phase_composition": {"Pd": 1.0},
        "a_gamma": 3.889,
        "a_gprime": 3.889,
        "supercell": (18, 18, 24),
        "n_precip": 0,
        "cone_diameter_angstrom": 120.0,
        "cone_length_angstrom": 500.0,
        "hemisphere_base_angstrom": 250.0,
    },
    "Nimonic 90": {
        "mode": "gamma-gamma-prime",
        "geometry_mode": "tapered-tip",
        "single_phase_composition": {"Ni": 0.58, "Co": 0.18, "Cr": 0.19, "Al": 0.02, "Ti": 0.02, "Fe": 0.01},
        "a_gamma": 3.54,
        "a_gprime": 3.57,
        "supercell": (24, 24, 36),
        "wt_spec": {
            "Cr": ["range", 18.0, 21.0],
            "Co": ["range", 15.0, 21.0],
            "Ti": ["range", 2.0, 3.0],
            "Al": ["range", 1.0, 2.0],
            "Fe": ["max", 1.5],
            "Cu": ["max", 0.2],
            "Mn": ["max", 1.0],
            "Si": ["max", 1.0],
            "Zr": ["max", 0.15],
        },
        "gamma_partition": {
            "Al": 0.30,
            "Ti": 0.25,
            "Cr": 1.00,
            "Co": 1.00,
            "Fe": 1.00,
            "Cu": 1.00,
            "Mn": 1.00,
            "Si": 1.00,
            "Zr": 1.00,
            "Ni": 1.00,
        },
        "gprime_a_site_ratio": {"Al": 0.6, "Ti": 0.4},
        "n_precip": 10,
        "precipitate_r_min_angstrom": 6.0,
        "precipitate_r_max_angstrom": 18.0,
        "target_gprime_fraction": 0.18,
        "cone_diameter_angstrom": 140.0,
        "cone_length_angstrom": 600.0,
        "hemisphere_base_angstrom": 220.0,
        "surf_tol_angstrom": 2.0,
        "bottom_tol_angstrom": 2.0,
    },
}

ELEMENT_DEFAULTS = {
    "Al": {"evap_field": 19.0, "charge": 1},
    "Co": {"evap_field": 37.0, "charge": 2},
    "Cr": {"evap_field": 27.0, "charge": 1},
    "Cu": {"evap_field": 30.0, "charge": 1},
    "Fe": {"evap_field": 33.0, "charge": 2},
    "Mn": {"evap_field": 30.0, "charge": 1},
    "Ni": {"evap_field": 35.0, "charge": 1},
    "Pd": {"evap_field": 32.0, "charge": 1},
    "Si": {"evap_field": 33.0, "charge": 2},
    "Ti": {"evap_field": 26.0, "charge": 2},
    "Zr": {"evap_field": 28.0, "charge": 2},
}

DEFAULT_PHASE_SCALE = {"gamma": 1.0, "gprime": 1.15}


@dataclass
class TAPSimBuildResult:
    """In-memory result of a TAPSim specimen build."""

    preset: str
    mode: str
    geometry_mode: str
    base_structure: Structure
    phase_structure: Structure
    shaped_structure: Structure
    precipitates: list[tuple[np.ndarray, float]]
    atoms_df: pd.DataFrame
    species_map_df: pd.DataFrame
    summary: dict[str, object]
    export_paths: dict[str, str] = field(default_factory=dict)


def available_presets() -> list[str]:
    """Return built-in workflow presets."""
    return list(PRESET_CONFIGS.keys())


def get_preset_config(name: str) -> dict[str, object]:
    """Return a deep copy of a built-in preset configuration."""
    if name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset {name!r}. Available presets: {available_presets()}")
    return deepcopy(PRESET_CONFIGS[name])


def _normalize_composition(composition: dict[str, float]) -> dict[str, float]:
    cleaned = {str(key): float(value) for key, value in composition.items() if float(value) > 0}
    total = float(sum(cleaned.values()))
    if total <= 0:
        raise ValueError("Composition must contain at least one positive fraction")
    return {key: value / total for key, value in cleaned.items()}


def _tag_structure_phase(structure: Structure, phase: str) -> Structure:
    tagged = Structure(structure.lattice, [], [])
    for site in structure:
        props = dict(site.properties) if site.properties else {}
        props["phase"] = phase
        tagged.append(site.specie, site.coords, coords_are_cartesian=True, properties=props)
    return tagged


def choose_bulk_wt(spec: dict[str, list[object]], strategy: str = "mid", max_frac: float = 0.2) -> dict[str, float]:
    """Pick a concrete weight-percent composition from a range/max specification."""
    wt = {}
    for element, info in spec.items():
        kind = str(info[0]).strip().lower()
        if kind == "range":
            low = float(info[1])
            high = float(info[2])
            if strategy == "low":
                value = low
            elif strategy == "high":
                value = high
            else:
                value = 0.5 * (low + high)
        elif kind == "max":
            value = float(info[1]) * float(max_frac)
        else:
            value = float(info[1])
        wt[str(element)] = float(value)
    return wt


def wt_to_atomic_fractions(wt_dict: dict[str, float]) -> dict[str, float]:
    """Convert a weight-percent composition to atomic fractions, using Ni as the balance.

    Ni is the balance to 100 wt%: ``Ni = 100 - sum(other elements)``. Any Ni
    value supplied in ``wt_dict`` is therefore replaced by the balance.
    Previously an input Ni value was summed AND then subtracted from itself,
    double-counting Ni and driving its balance toward 0.
    """
    symbols = set(wt_dict.keys()) | {"Ni"}
    atomic_weights = {symbol: float(PMGElement(symbol).atomic_mass) for symbol in symbols}
    wt_full = {str(key): float(value) for key, value in wt_dict.items() if str(key) != "Ni"}
    wt_full["Ni"] = max(0.0, 100.0 - sum(wt_full.values()))
    molar = {symbol: wt_full[symbol] / atomic_weights[symbol] for symbol in wt_full}
    total = float(sum(molar.values()))
    return {symbol: value / total for symbol, value in molar.items()}


def make_fcc_structure(lattice_constant: float, base_element: str = "Ni") -> Structure:
    """Create a conventional FCC unit cell."""
    lattice = Lattice.cubic(float(lattice_constant))
    species = [base_element] * 4
    frac_coords = [
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 0.0],
    ]
    return Structure(lattice, species, frac_coords)


def make_l12_gamma_prime_structure(lattice_constant: float) -> Structure:
    """Create a simple L1_2 gamma-prime unit cell."""
    lattice = Lattice.cubic(float(lattice_constant))
    species = ["Al", "Ni", "Ni", "Ni"]
    frac_coords = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ]
    return Structure(lattice, species, frac_coords)


def supercell(structure: Structure, size: int | tuple[int, int, int]) -> Structure:
    """Return a copied supercell with either isotropic or per-axis replication."""
    result = structure.copy()
    if isinstance(size, int):
        if size < 1:
            raise ValueError("Supercell size must be positive")
        result.make_supercell([size, size, size])
        return result
    reps = tuple(int(value) for value in size)
    if len(reps) != 3 or any(value < 1 for value in reps):
        raise ValueError("Supercell size must be a positive (x, y, z) tuple")
    result.make_supercell([[reps[0], 0, 0], [0, reps[1], 0], [0, 0, reps[2]]])
    return result


def randomly_substitute_structure(structure: Structure, composition: dict[str, float], seed: int | None = None) -> Structure:
    """Randomly substitute species according to the requested composition."""
    normalized = _normalize_composition(composition)
    rng = np.random.default_rng(seed)
    species = rng.choice(list(normalized.keys()), size=len(structure), p=list(normalized.values()))
    return Structure(structure.lattice, list(species), structure.frac_coords)


def assign_a_sublattice_species(structure: Structure, a_site_ratio: dict[str, float], seed: int | None = None) -> Structure:
    """Assign A-sublattice species for the L1_2 gamma-prime cell."""
    normalized = _normalize_composition(a_site_ratio)
    result = structure.copy()
    a_site_indices = [index for index, site in enumerate(result) if site.specie.symbol == "Al"]
    if not a_site_indices:
        return result

    rng = np.random.default_rng(seed)
    counts = {}
    running_total = 0
    items = list(normalized.items())
    for element, fraction in items[:-1]:
        count = int(round(len(a_site_indices) * fraction))
        counts[element] = count
        running_total += count
    counts[items[-1][0]] = max(0, len(a_site_indices) - running_total)

    shuffled_indices = rng.permutation(a_site_indices)
    cursor = 0
    for element, count in counts.items():
        for local_index in shuffled_indices[cursor : cursor + count]:
            result[int(local_index)] = element
        cursor += count
    return result


def generate_nonoverlapping_precipitates(
    n_precip: int,
    box_lengths_angstrom: np.ndarray,
    r_min_angstrom: float,
    r_max_angstrom: float,
    *,
    clearance_angstrom: float = 0.0,
    margin_angstrom: float = 0.0,
    max_attempts: int = 100000,
    seed: int | None = None,
    periodic: bool = False,
    radius_dist: str = "uniform",
    lognorm_sigma: float = 0.35,
) -> list[tuple[np.ndarray, float]]:
    """Generate non-overlapping spherical precipitates inside a rectangular box."""
    n_precip = int(n_precip)
    if n_precip <= 0:
        return []
    box_lengths = np.asarray(box_lengths_angstrom, dtype=float)
    if box_lengths.shape != (3,):
        raise ValueError("box_lengths_angstrom must be a length-3 vector")
    r_min = float(r_min_angstrom)
    r_max = float(r_max_angstrom)
    if r_min <= 0 or r_max <= 0 or r_max < r_min:
        raise ValueError("Precipitate radius range is invalid")
    rng = np.random.default_rng(seed)

    if radius_dist == "uniform":
        radii = rng.uniform(r_min, r_max, size=n_precip)
    elif radius_dist == "lognormal":
        mu = np.log((r_min + r_max) / 2.0) - 0.5 * (float(lognorm_sigma) ** 2)
        radii = np.clip(rng.lognormal(mu, float(lognorm_sigma), size=n_precip), r_min, r_max)
    else:
        raise ValueError("radius_dist must be either 'uniform' or 'lognormal'")
    radii = radii[np.argsort(-radii)]

    centers: list[np.ndarray] = []
    placed_radii: list[float] = []

    def is_valid(center: np.ndarray, radius: float) -> bool:
        for other_center, other_radius in zip(centers, placed_radii):
            delta = np.abs(center - other_center)
            if periodic:
                delta = np.minimum(delta, box_lengths - delta)
            if np.linalg.norm(delta) < (radius + other_radius + float(clearance_angstrom)):
                return False
        return True

    attempts = 0
    for radius in radii:
        low = float(margin_angstrom) + radius
        high = box_lengths - (float(margin_angstrom) + radius)
        if np.any(high - low <= 0):
            continue
        placed = False
        while attempts < int(max_attempts):
            attempts += 1
            center = rng.uniform(low=low, high=high)
            if is_valid(center, float(radius)):
                centers.append(center)
                placed_radii.append(float(radius))
                placed = True
                break
        if not placed:
            continue
        if len(centers) == n_precip:
            break

    return [(np.asarray(center, dtype=float), float(radius)) for center, radius in zip(centers, placed_radii)]


def carve_spherical_precipitates(
    gamma_structure: Structure,
    gprime_structure: Structure,
    precipitates: list[tuple[np.ndarray, float]],
    *,
    remove_cut_angstrom: float = 2.3,
) -> Structure:
    """Embed gamma-prime spheres into a gamma matrix."""
    if not precipitates:
        return _tag_structure_phase(gamma_structure, "gamma")

    gamma_cart = np.asarray([site.coords for site in gamma_structure], dtype=float)
    gprime_cart = np.asarray([site.coords for site in gprime_structure], dtype=float)
    keep_gamma = np.ones(len(gamma_structure), dtype=bool)
    keep_gprime = np.zeros(len(gprime_structure), dtype=bool)

    for center, radius in precipitates:
        keep_gprime |= np.linalg.norm(gprime_cart - center, axis=1) <= float(radius)
        keep_gamma &= np.linalg.norm(gamma_cart - center, axis=1) > float(radius)

    gprime_points = gprime_cart[keep_gprime]
    if gprime_points.size:
        tree = cKDTree(gprime_points)
        distances, _ = tree.query(gamma_cart, k=1)
        keep_gamma &= distances > float(remove_cut_angstrom)

    combined = Structure(gamma_structure.lattice, [], [])
    for index, site in enumerate(gamma_structure):
        if keep_gamma[index]:
            combined.append(site.specie, site.frac_coords, coords_are_cartesian=False, properties={"phase": "gamma"})
    for index, site in enumerate(gprime_structure):
        if keep_gprime[index]:
            combined.append(site.specie, gprime_cart[index], coords_are_cartesian=True, properties={"phase": "gprime"})
    return combined


def _phase_fraction(structure: Structure, phase: str) -> float:
    phases = [str(site.properties.get("phase", "gamma")) for site in structure]
    if not phases:
        return 0.0
    return float(np.mean(np.asarray(phases) == phase))


def fit_target_gprime_fraction(
    gamma_structure: Structure,
    gprime_structure: Structure,
    precipitates: list[tuple[np.ndarray, float]],
    target_fraction: float,
    *,
    tolerance: float = 0.003,
    max_iter: int = 12,
    remove_cut_angstrom: float = 2.3,
) -> tuple[Structure, list[tuple[np.ndarray, float]], float]:
    """Scale precipitate radii until the requested gamma-prime atom fraction is reached approximately."""
    target = float(target_fraction)
    if target <= 0:
        combined = carve_spherical_precipitates(
            gamma_structure,
            gprime_structure,
            precipitates,
            remove_cut_angstrom=remove_cut_angstrom,
        )
        return combined, precipitates, _phase_fraction(combined, "gprime")

    low = 0.5
    high = 2.5
    best_structure = carve_spherical_precipitates(
        gamma_structure,
        gprime_structure,
        precipitates,
        remove_cut_angstrom=remove_cut_angstrom,
    )
    best_precipitates = precipitates
    best_fraction = _phase_fraction(best_structure, "gprime")

    for _ in range(int(max_iter)):
        scale = 0.5 * (low + high)
        scaled = [(center, float(radius) * scale) for center, radius in precipitates]
        combined = carve_spherical_precipitates(
            gamma_structure,
            gprime_structure,
            scaled,
            remove_cut_angstrom=remove_cut_angstrom,
        )
        fraction = _phase_fraction(combined, "gprime")
        best_structure = combined
        best_precipitates = scaled
        best_fraction = fraction
        if abs(fraction - target) <= float(tolerance):
            break
        if fraction < target:
            low = scale
        else:
            high = scale
    return best_structure, best_precipitates, best_fraction


def shape_tip_with_surface_labels(
    structure: Structure,
    *,
    cone_diameter_angstrom: float,
    cone_length_angstrom: float,
    hemisphere_base_angstrom: float,
    cylinder: bool = False,
    surf_tol_angstrom: float = 2.0,
    bottom_tol_angstrom: float = 2.0,
) -> Structure:
    """Filter a structure to a tapered tip or cylinder with a hemispherical cap and label its boundaries."""
    coords = np.asarray(structure.cart_coords, dtype=float)
    if coords.size == 0:
        raise ValueError("Structure is empty")
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    x_center = 0.5 * (x.min() + x.max())
    y_center = 0.5 * (y.min() + y.max())
    radial = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
    base_radius = float(cone_diameter_angstrom) / 2.0
    if cylinder:
        radius_at_z = np.full_like(z, base_radius)
    else:
        radius_at_z = (float(cone_length_angstrom) - z) / float(cone_length_angstrom) * base_radius

    inside_cone = (z >= 0.0) & (z <= float(cone_length_angstrom)) & (radial <= radius_at_z + 1e-12)
    if 0.0 <= float(hemisphere_base_angstrom) <= float(cone_length_angstrom):
        if cylinder:
            hemisphere_radius = base_radius
        else:
            hemisphere_radius = (
                (float(cone_length_angstrom) - float(hemisphere_base_angstrom)) / float(cone_length_angstrom) * base_radius
            )
    else:
        hemisphere_radius = 0.0

    hemisphere_center_z = float(hemisphere_base_angstrom) + hemisphere_radius
    d_sphere = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2 + (z - hemisphere_center_z) ** 2)
    inside_or_below_hemisphere = (z <= hemisphere_center_z + 1e-12) | (d_sphere <= hemisphere_radius + 1e-12)
    inside_shape = inside_cone & inside_or_below_hemisphere
    on_cone_surface = inside_shape & (np.abs(radial - radius_at_z) <= float(surf_tol_angstrom))
    on_sphere_surface = (
        inside_shape & (np.abs(d_sphere - hemisphere_radius) <= float(surf_tol_angstrom)) & (z >= hemisphere_center_z)
    )
    on_bottom = inside_shape & (z <= float(bottom_tol_angstrom)) & (radial <= base_radius + float(surf_tol_angstrom))

    shaped = Structure(structure.lattice, [], [])
    for index, site in enumerate(structure):
        if not inside_shape[index]:
            continue
        props = dict(site.properties) if site.properties else {}
        props["is_surface"] = bool(on_cone_surface[index] or on_sphere_surface[index])
        props["surface_part"] = "cone" if on_cone_surface[index] else ("sphere" if on_sphere_surface[index] else None)
        props["is_bottom"] = bool(on_bottom[index])
        shaped.append(site.specie, coords[index], coords_are_cartesian=True, properties=props)

    if len(shaped) == 0:
        raise ValueError("No atoms remain after applying the requested tip geometry")
    return shaped


def _element_defaults(symbol: str) -> dict[str, float]:
    defaults = ELEMENT_DEFAULTS.get(symbol, {"evap_field": 30.0, "charge": 1})
    return {"evap_field": float(defaults["evap_field"]), "charge": int(defaults["charge"])}


def build_tapsim_tables(
    structure: Structure,
    *,
    include_boundaries: bool = True,
    elem_start_id: int = 10,
    max_id: int = 31,
    separate_phase_ids: bool = True,
    phase_scale: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert a shaped structure into TAPSim atom and species tables."""
    if len(structure) == 0:
        raise ValueError("Structure is empty")
    phase_scale = DEFAULT_PHASE_SCALE if phase_scale is None else {str(k): float(v) for k, v in phase_scale.items()}

    present: dict[str, dict[str, bool]] = {}
    for site in structure:
        props = site.properties or {}
        if include_boundaries and (bool(props.get("is_surface", False)) or bool(props.get("is_bottom", False))):
            continue
        symbol = site.specie.symbol
        phase = "gprime" if str(props.get("phase", "gamma")) == "gprime" else "gamma"
        present.setdefault(symbol, {"gamma": False, "gprime": False})[phase] = True

    element_to_ids: dict[str, dict[str, int | None]] = {}
    next_id = int(elem_start_id)
    for symbol in sorted(present.keys()):
        if next_id > int(max_id):
            raise RuntimeError(f"Ran out of TAPSim IDs at {next_id}")
        gamma_id = next_id
        next_id += 1
        gprime_id = None
        if separate_phase_ids and present[symbol].get("gprime", False):
            if next_id > int(max_id):
                raise RuntimeError(f"Need an extra TAPSim ID for {symbol} gamma-prime, but reached {max_id}")
            gprime_id = next_id
            next_id += 1
        element_to_ids[symbol] = {"gamma": gamma_id, "gprime": gprime_id}

    a_len, b_len, _ = structure.lattice.abc
    x_center = float(a_len) / 2.0
    y_center = float(b_len) / 2.0
    rows = []
    for site in structure:
        props = site.properties or {}
        symbol = site.specie.symbol
        phase = str(props.get("phase", "gamma"))
        is_surface = bool(props.get("is_surface", False))
        is_bottom = bool(props.get("is_bottom", False))
        if include_boundaries and is_bottom:
            element_idx = 2
            phase_name = "bottom"
            name = "bottom"
        elif include_boundaries and is_surface:
            element_idx = 0
            phase_name = "vacuum"
            name = "vacuum"
        else:
            phase_key = "gprime" if phase == "gprime" and element_to_ids.get(symbol, {}).get("gprime") is not None else "gamma"
            element_idx = int(element_to_ids[symbol][phase_key])
            phase_name = phase_key
            name = f"{symbol}_{phase_key}" if separate_phase_ids and phase_key == "gprime" else symbol
        rows.append(
            {
                "x": float(site.coords[0] - x_center),
                "y": float(site.coords[1] - y_center),
                "z": float(site.coords[2]),
                "elements": symbol if phase_name not in {"vacuum", "bottom"} else phase_name,
                "element_idx": int(element_idx),
                "phase": phase_name,
                "name": name,
            }
        )

    atoms_df = pd.DataFrame(rows, columns=["x", "y", "z", "elements", "element_idx", "phase", "name"])
    total_atoms = max(1, len(atoms_df))
    map_rows = []
    for element_idx in sorted(atoms_df["element_idx"].unique()):
        subset = atoms_df[atoms_df["element_idx"] == element_idx]
        first = subset.iloc[0]
        if int(element_idx) in {0, 2}:
            map_rows.append(
                {
                    "ID": int(element_idx),
                    "name": str(first["name"]),
                    "element": str(first["elements"]),
                    "mass": 0.0,
                    "charge": 0,
                    "evap_field": 0.0,
                    "percent": 100.0 * len(subset) / total_atoms,
                }
            )
            continue
        symbol = str(first["elements"])
        defaults = _element_defaults(symbol)
        phase = str(first["phase"])
        map_rows.append(
            {
                "ID": int(element_idx),
                "name": str(first["name"]),
                "element": symbol,
                "mass": float(PMGElement(symbol).atomic_mass),
                "charge": int(defaults["charge"]),
                "evap_field": float(defaults["evap_field"]) * float(phase_scale.get(phase, 1.0)),
                "percent": 100.0 * len(subset) / total_atoms,
            }
        )
    species_map_df = pd.DataFrame(map_rows).sort_values("ID").reset_index(drop=True)
    return atoms_df, species_map_df


def species_map_to_range_data(species_map_df: pd.DataFrame, default_half_width: float = 0.2) -> pd.DataFrame:
    """Create a minimal PyCCAPT range dataframe from a TAPSim species map."""
    rows = []
    for record in species_map_df.to_dict(orient="records"):
        name = str(record["name"])
        element = str(record.get("element", name))
        mass = float(record.get("mass", 0.0))
        charge = int(record.get("charge", 0))
        rows.append(
            {
                "name": name,
                "ion": name,
                "mass": mass,
                "mc": mass,
                "mc_low": max(0.0, mass - float(default_half_width)),
                "mc_up": mass + float(default_half_width),
                "color": "#808080" if element in {"vacuum", "bottom"} else "#1f77b4",
                "element": [element],
                "complex": [[1]],
                "isotope": [[0]],
                "charge": charge,
            }
        )
    return pd.DataFrame(rows)


def write_datapart_text_minimal(
    atoms_df: pd.DataFrame,
    path: str | Path,
    *,
    unit_scale: float = 1.0,
    precision: int = 5,
) -> str:
    """Write the TAPSim ASCII datapart format without potentials."""
    required = {"x", "y", "z", "element_idx"}
    missing = required - set(atoms_df.columns)
    if missing:
        raise ValueError(f"atoms_df is missing required columns: {sorted(missing)}")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = f"{{:.{int(precision)}e}}\t{{:.{int(precision)}e}}\t{{:.{int(precision)}e}}\t{{:d}}\n"
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"ASCII {len(atoms_df)} 0 0\n")
        for row in atoms_df.itertuples(index=False):
            handle.write(
                fmt.format(
                    float(row.x) * float(unit_scale),
                    float(row.y) * float(unit_scale),
                    float(row.z) * float(unit_scale),
                    int(row.element_idx),
                )
            )
    return str(output_path)


def write_datapart_text_with_potential(
    atoms_df: pd.DataFrame,
    path: str | Path,
    *,
    unit_scale: float = 1.0,
    coord_precision: int = 5,
    potential_start: float = 1.0e3,
    potential_end: float = 2.0e3,
    potential_precision: int = 2,
) -> str:
    """Write the TAPSim ASCII datapart format with a linearly ramped potential column."""
    required = {"x", "y", "z", "element_idx"}
    missing = required - set(atoms_df.columns)
    if missing:
        raise ValueError(f"atoms_df is missing required columns: {sorted(missing)}")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    potentials = (
        np.linspace(float(potential_start), float(potential_end), len(atoms_df))
        if len(atoms_df) > 1
        else np.asarray([float(potential_start)])
    )
    coord_fmt = f"{{:.{int(coord_precision)}e}}\t{{:.{int(coord_precision)}e}}\t{{:.{int(coord_precision)}e}}"
    pot_fmt = f"{{:.{int(potential_precision)}e}}"
    row_fmt = coord_fmt + "\t{:d}\t" + pot_fmt + "\n"
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"ASCII {len(atoms_df)} 0 1\n")
        for row, potential in zip(atoms_df.itertuples(index=False), potentials):
            handle.write(
                row_fmt.format(
                    float(row.x) * float(unit_scale),
                    float(row.y) * float(unit_scale),
                    float(row.z) * float(unit_scale),
                    int(row.element_idx),
                    float(potential),
                )
            )
    return str(output_path)


def _build_single_phase_structure(
    composition: dict[str, float],
    lattice_constant: float,
    supercell_size: tuple[int, int, int],
    *,
    seed: int | None = None,
) -> Structure:
    normalized = _normalize_composition(composition)
    base_element = max(normalized, key=normalized.get)
    structure = supercell(make_fcc_structure(lattice_constant, base_element=base_element), supercell_size)
    structure = randomly_substitute_structure(structure, normalized, seed=seed)
    return _tag_structure_phase(structure, "gamma")


def _build_gamma_gprime_structure(
    *,
    a_gamma: float,
    a_gprime: float,
    supercell_size: tuple[int, int, int],
    wt_spec: dict[str, list[object]],
    gamma_partition: dict[str, float],
    gprime_a_site_ratio: dict[str, float],
    n_precip: int,
    r_min_angstrom: float,
    r_max_angstrom: float,
    target_gprime_fraction: float | None,
    clearance_angstrom: float,
    margin_angstrom: float,
    periodic: bool,
    radius_dist: str,
    lognorm_sigma: float,
    remove_cut_angstrom: float,
    seed: int | None,
) -> tuple[Structure, list[tuple[np.ndarray, float]], float]:
    wt_bulk = choose_bulk_wt(wt_spec, strategy="mid", max_frac=0.2)
    atomic_bulk = wt_to_atomic_fractions(wt_bulk)
    matrix_fraction = {}
    for element, fraction in atomic_bulk.items():
        matrix_fraction[element] = float(fraction) * float(gamma_partition.get(element, 1.0))
    matrix_fraction = _normalize_composition(matrix_fraction)

    gamma = supercell(make_fcc_structure(a_gamma), supercell_size)
    gamma = randomly_substitute_structure(gamma, matrix_fraction, seed=seed)

    gamma_box = np.asarray(supercell_size, dtype=float) * float(a_gamma)
    gprime_supercell = tuple((np.ceil(gamma_box / float(a_gprime)).astype(int) + 2).tolist())
    gprime = supercell(make_l12_gamma_prime_structure(a_gprime), gprime_supercell)
    gprime = assign_a_sublattice_species(gprime, gprime_a_site_ratio, seed=seed)

    precipitates = generate_nonoverlapping_precipitates(
        n_precip,
        gamma_box,
        r_min_angstrom,
        r_max_angstrom,
        clearance_angstrom=clearance_angstrom,
        margin_angstrom=margin_angstrom,
        max_attempts=200000,
        seed=seed,
        periodic=periodic,
        radius_dist=radius_dist,
        lognorm_sigma=lognorm_sigma,
    )
    if target_gprime_fraction is not None and float(target_gprime_fraction) > 0:
        combined, precipitates, achieved_fraction = fit_target_gprime_fraction(
            gamma,
            gprime,
            precipitates,
            float(target_gprime_fraction),
            remove_cut_angstrom=remove_cut_angstrom,
        )
    else:
        combined = carve_spherical_precipitates(
            gamma,
            gprime,
            precipitates,
            remove_cut_angstrom=remove_cut_angstrom,
        )
        achieved_fraction = _phase_fraction(combined, "gprime")
    return combined, precipitates, achieved_fraction


def build_tapsim_specimen(
    *,
    preset: str = "Nimonic 90",
    mode: str = "gamma-gamma-prime",
    geometry_mode: str = "tapered-tip",
    single_phase_composition: dict[str, float] | None = None,
    a_gamma: float = 3.54,
    a_gprime: float = 3.57,
    supercell_size: tuple[int, int, int] = (24, 24, 36),
    wt_spec: dict[str, list[object]] | None = None,
    gamma_partition: dict[str, float] | None = None,
    gprime_a_site_ratio: dict[str, float] | None = None,
    n_precip: int = 10,
    precipitate_r_min_angstrom: float = 6.0,
    precipitate_r_max_angstrom: float = 18.0,
    target_gprime_fraction: float | None = None,
    clearance_angstrom: float = 1.0,
    margin_angstrom: float = 2.0,
    periodic_precipitates: bool = True,
    radius_dist: str = "uniform",
    lognorm_sigma: float = 0.35,
    remove_cut_angstrom: float = 2.3,
    cone_diameter_angstrom: float = 140.0,
    cone_length_angstrom: float = 600.0,
    hemisphere_base_angstrom: float = 220.0,
    surf_tol_angstrom: float = 2.0,
    bottom_tol_angstrom: float = 2.0,
    elem_start_id: int = 10,
    max_id: int = 31,
    separate_phase_ids: bool = True,
    phase_scale: dict[str, float] | None = None,
    seed: int | None = 42,
) -> TAPSimBuildResult:
    """Build a synthetic specimen and convert it into TAPSim-ready atom/species tables."""
    mode = str(mode).strip().lower()
    geometry_mode = str(geometry_mode).strip().lower()
    if geometry_mode not in {"bulk", "tapered-tip", "cylinder-hemisphere"}:
        raise ValueError("geometry_mode must be one of: bulk, tapered-tip, cylinder-hemisphere")
    supercell_size = tuple(int(value) for value in supercell_size)
    if len(supercell_size) != 3 or any(value < 1 for value in supercell_size):
        raise ValueError("supercell_size must be a positive (x, y, z) tuple")

    if mode == "single-phase":
        phase_structure = _build_single_phase_structure(
            single_phase_composition or {"Ni": 1.0},
            float(a_gamma),
            supercell_size,
            seed=seed,
        )
        precipitates = []
        achieved_fraction = 0.0
    elif mode == "gamma-gamma-prime":
        if wt_spec is None or gamma_partition is None or gprime_a_site_ratio is None:
            raise ValueError("wt_spec, gamma_partition, and gprime_a_site_ratio are required for gamma-gamma-prime mode")
        phase_structure, precipitates, achieved_fraction = _build_gamma_gprime_structure(
            a_gamma=float(a_gamma),
            a_gprime=float(a_gprime),
            supercell_size=supercell_size,
            wt_spec=wt_spec,
            gamma_partition=gamma_partition,
            gprime_a_site_ratio=gprime_a_site_ratio,
            n_precip=int(n_precip),
            r_min_angstrom=float(precipitate_r_min_angstrom),
            r_max_angstrom=float(precipitate_r_max_angstrom),
            target_gprime_fraction=target_gprime_fraction,
            clearance_angstrom=float(clearance_angstrom),
            margin_angstrom=float(margin_angstrom),
            periodic=bool(periodic_precipitates),
            radius_dist=radius_dist,
            lognorm_sigma=float(lognorm_sigma),
            remove_cut_angstrom=float(remove_cut_angstrom),
            seed=seed,
        )
    else:
        raise ValueError("mode must be either 'single-phase' or 'gamma-gamma-prime'")

    base_structure = phase_structure.copy()
    if geometry_mode == "bulk":
        shaped_structure = phase_structure.copy()
    else:
        shaped_structure = shape_tip_with_surface_labels(
            phase_structure,
            cone_diameter_angstrom=float(cone_diameter_angstrom),
            cone_length_angstrom=float(cone_length_angstrom),
            hemisphere_base_angstrom=float(hemisphere_base_angstrom),
            cylinder=geometry_mode == "cylinder-hemisphere",
            surf_tol_angstrom=float(surf_tol_angstrom),
            bottom_tol_angstrom=float(bottom_tol_angstrom),
        )

    atoms_df, species_map_df = build_tapsim_tables(
        shaped_structure,
        include_boundaries=geometry_mode != "bulk",
        elem_start_id=int(elem_start_id),
        max_id=int(max_id),
        separate_phase_ids=bool(separate_phase_ids),
        phase_scale=phase_scale,
    )

    phase_counts = {}
    for site in phase_structure:
        phase = str(site.properties.get("phase", "gamma"))
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    shaped_phase_counts = atoms_df["phase"].value_counts().to_dict()
    summary = {
        "total_atoms_phase_structure": int(len(phase_structure)),
        "total_atoms_shaped": int(len(shaped_structure)),
        "phase_counts": phase_counts,
        "shaped_phase_counts": {str(k): int(v) for k, v in shaped_phase_counts.items()},
        "element_counts": {str(k): int(v) for k, v in atoms_df["elements"].value_counts().to_dict().items()},
        "precipitate_count": int(len(precipitates)),
        "achieved_gprime_fraction": float(achieved_fraction),
        "used_ids": [int(value) for value in species_map_df["ID"].tolist()],
        "bbox_angstrom": {
            "x": [float(atoms_df["x"].min()), float(atoms_df["x"].max())],
            "y": [float(atoms_df["y"].min()), float(atoms_df["y"].max())],
            "z": [float(atoms_df["z"].min()), float(atoms_df["z"].max())],
        },
    }
    return TAPSimBuildResult(
        preset=preset,
        mode=mode,
        geometry_mode=geometry_mode,
        base_structure=base_structure,
        phase_structure=phase_structure,
        shaped_structure=shaped_structure,
        precipitates=precipitates,
        atoms_df=atoms_df,
        species_map_df=species_map_df,
        summary=summary,
    )


def plot_precipitate_layout(result: TAPSimBuildResult) -> plt.Figure:
    """Plot the precipitate positions in XY and XZ."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if not result.precipitates:
        for axis in axes:
            axis.text(0.5, 0.5, "No precipitates", ha="center", va="center", transform=axis.transAxes)
            axis.set_axis_off()
        fig.suptitle("Precipitate Layout")
        fig.tight_layout()
        return fig

    for center, radius in result.precipitates:
        xy_circle = plt.Circle((center[0], center[1]), radius, fill=False, alpha=0.7, color="tab:orange")
        xz_circle = plt.Circle((center[0], center[2]), radius, fill=False, alpha=0.7, color="tab:orange")
        axes[0].add_patch(xy_circle)
        axes[1].add_patch(xz_circle)
    axes[0].set_title("XY precipitate layout")
    axes[1].set_title("XZ precipitate layout")
    axes[0].set_xlabel("x (A)")
    axes[0].set_ylabel("y (A)")
    axes[1].set_xlabel("x (A)")
    axes[1].set_ylabel("z (A)")
    for axis in axes:
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_structure_preview(result: TAPSimBuildResult, max_points: int = 12000) -> plt.Figure:
    """Plot sampled XZ and YZ views of the shaped specimen."""
    data = result.atoms_df
    if len(data) > int(max_points):
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(data), size=int(max_points), replace=False)
        data = data.iloc[np.sort(sample_idx)]

    color_map = {"gamma": "#1f77b4", "gprime": "#ff7f0e", "vacuum": "#bdbdbd", "bottom": "#111111"}
    colors = [color_map.get(str(phase), "#2ca02c") for phase in data["phase"]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(data["x"], data["z"], s=2, c=colors, alpha=0.7)
    axes[1].scatter(data["y"], data["z"], s=2, c=colors, alpha=0.7)
    axes[0].set_title("XZ preview")
    axes[1].set_title("YZ preview")
    axes[0].set_xlabel("x (A)")
    axes[1].set_xlabel("y (A)")
    axes[0].set_ylabel("z (A)")
    axes[1].set_ylabel("z (A)")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_species_summary(result: TAPSimBuildResult) -> plt.Figure:
    """Plot species counts by TAPSim ID."""
    counts = result.atoms_df["element_idx"].value_counts().sort_index()
    fig, axis = plt.subplots(figsize=(9, 4))
    axis.bar(counts.index.astype(str), counts.values, color="#4c78a8")
    axis.set_title("TAPSim ID counts")
    axis.set_xlabel("element_idx")
    axis.set_ylabel("count")
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def export_tapsim_bundle(
    result: TAPSimBuildResult,
    *,
    output_dir: str,
    dataset_name: str,
    include_potential: bool = False,
    potential_start: float = 1.0e3,
    potential_end: float = 2.0e3,
    write_csv: bool = True,
    write_hdf: bool = True,
    write_range_hdf: bool = True,
    unit_scale: float = 1.0e-10,
) -> dict[str, str]:
    """Export TAPSim datapart text, species map, and optional companion files."""
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    safe_name = str(dataset_name).strip()
    if not safe_name:
        raise ValueError("dataset_name cannot be empty")

    export_paths: dict[str, str] = {}
    species_map_path = outdir / f"{safe_name}_species_map.csv"
    result.species_map_df.to_csv(species_map_path, index=False)
    export_paths["species_map_csv"] = str(species_map_path)

    if write_csv:
        csv_path = outdir / f"{safe_name}.csv"
        result.atoms_df.to_csv(csv_path, index=False)
        export_paths["atoms_csv"] = str(csv_path)
    if write_hdf:
        hdf_path = outdir / f"{safe_name}.h5"
        result.atoms_df.to_hdf(hdf_path, key="filtered_structure", mode="w")
        export_paths["atoms_hdf"] = str(hdf_path)
    if write_range_hdf:
        range_path = outdir / f"range_{safe_name}.h5"
        species_map_to_range_data(result.species_map_df).to_hdf(range_path, key="df", mode="w")
        export_paths["range_hdf"] = str(range_path)

    txt_path = outdir / f"{safe_name}.txt"
    if include_potential:
        export_paths["datapart_txt"] = write_datapart_text_with_potential(
            result.atoms_df,
            txt_path,
            unit_scale=unit_scale,
            potential_start=potential_start,
            potential_end=potential_end,
        )
    else:
        export_paths["datapart_txt"] = write_datapart_text_minimal(
            result.atoms_df,
            txt_path,
            unit_scale=unit_scale,
        )
    result.export_paths = export_paths.copy()
    return export_paths


__all__ = [
    "ELEMENT_DEFAULTS",
    "PRESET_CONFIGS",
    "TAPSimBuildResult",
    "available_presets",
    "build_tapsim_specimen",
    "export_tapsim_bundle",
    "get_preset_config",
    "plot_precipitate_layout",
    "plot_species_summary",
    "plot_structure_preview",
    "species_map_to_range_data",
    "write_datapart_text_minimal",
    "write_datapart_text_with_potential",
]
