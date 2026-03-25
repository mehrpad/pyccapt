"""User-friendly specimen-like dataset generation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import Element as PMGElement

from pyccapt.calibration.data_tools import data_tools
from pyccapt.calibration.mc import tof_tools
from pyccapt.calibration.reconstructions import crystal_helper


PRESET_MATERIALS = {
    "Al": {
        "base_element": "Al",
        "lattice_constant": 4.0495,
        "composition": {"Al": 1.0},
    },
    "Pd": {
        "base_element": "Pd",
        "lattice_constant": 3.889,
        "composition": {"Pd": 1.0},
    },
    "Nimonic 90": {
        "base_element": "Ni",
        "lattice_constant": 3.58,
        "composition": {
            "Ni": 0.58,
            "Co": 0.18,
            "Cr": 0.19,
            "Al": 0.02,
            "Ti": 0.02,
            "Fe": 0.01,
        },
    },
}


def available_presets() -> list[str]:
    """Return the names of built-in specimen presets."""
    return list(PRESET_MATERIALS.keys())


def _fcc_structure(element: str, lattice_constant: float) -> Structure:
    """Create a conventional FCC unit cell."""
    lattice = Lattice.cubic(float(lattice_constant))
    frac_coords = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ]
    return Structure(lattice, [element] * 4, frac_coords)


def _normalize_composition(composition: dict[str, float]) -> dict[str, float]:
    total = float(sum(composition.values()))
    if total <= 0:
        raise ValueError("Composition fractions must sum to a positive value")
    return {element: float(value) / total for element, value in composition.items() if float(value) > 0}


def _apply_random_composition(structure: Structure, composition: dict[str, float], seed: int | None = None) -> Structure:
    composition = _normalize_composition(composition)
    rng = np.random.default_rng(seed)
    species = rng.choice(list(composition.keys()), size=len(structure), p=list(composition.values()))
    return Structure(structure.lattice, list(species), structure.frac_coords)


def structure_to_ccapt_dataset(
    structure: Structure,
    *,
    detector_radius_cm: float = 2.5,
    high_voltage: float = 5000.0,
    flight_path_length_mm: float = 110.0,
) -> pd.DataFrame:
    """Convert a pymatgen structure into a pyccapt-like processed dataframe."""
    coords_nm = np.asarray(structure.cart_coords, dtype=float) / 10.0
    coords_nm[:, 0] -= np.mean(coords_nm[:, 0])
    coords_nm[:, 1] -= np.mean(coords_nm[:, 1])
    coords_nm[:, 2] -= np.min(coords_nm[:, 2])

    radial = np.sqrt(coords_nm[:, 0] ** 2 + coords_nm[:, 1] ** 2)
    max_radial = float(np.max(radial)) if len(radial) else 1.0
    scale = detector_radius_cm / max(max_radial, 1e-6)
    x_det = coords_nm[:, 0] * scale
    y_det = coords_nm[:, 1] * scale

    masses = np.array([float(PMGElement(str(site.specie)).atomic_mass) for site in structure.sites], dtype=float)
    tof = np.array(
        [
            tof_tools.mc2tof(mc=mass, V=high_voltage, xDet=xd, yDet=yd, flightPathLength=flight_path_length_mm)
            for mass, xd, yd in zip(masses, x_det, y_det)
        ],
        dtype=float,
    )

    data = pd.DataFrame(
        {
            "x (nm)": coords_nm[:, 0],
            "y (nm)": coords_nm[:, 1],
            "z (nm)": coords_nm[:, 2],
            "mc (Da)": masses,
            "mc_uc (Da)": masses,
            "high_voltage (V)": np.full(len(structure), float(high_voltage)),
            "pulse_v (V)": np.zeros(len(structure)),
            "pulse_l (pJ)": np.zeros(len(structure)),
            "t (ns)": tof,
            "t_c (ns)": tof.copy(),
            "x_det (cm)": x_det,
            "y_det (cm)": y_det,
            "delta_p": np.zeros(len(structure)),
            "multi": np.zeros(len(structure)),
            "start_counter": np.arange(len(structure), dtype=np.int64),
            "element": [str(site.specie) for site in structure.sites],
        }
    )
    return data


def build_specimen_dataset(
    preset: str = "Al",
    *,
    supercell: tuple[int, int, int] = (20, 20, 20),
    cone_diameter_angstrom: float = 120.0,
    cone_length_angstrom: float = 600.0,
    hemisphere_base_angstrom: float = 200.0,
    noise_levels_angstrom: tuple[float, float, float] = (0.0, 0.0, 0.0),
    noise_type: str = "correlative",
    seed: int | None = None,
    detector_radius_cm: float = 2.5,
    high_voltage: float = 5000.0,
    flight_path_length_mm: float = 110.0,
    save_path: str | None = None,
) -> pd.DataFrame:
    """Generate a specimen-like pyccapt dataset from a built-in preset."""
    if preset not in PRESET_MATERIALS:
        raise ValueError(f"Unknown preset {preset!r}. Available presets: {available_presets()}")

    preset_cfg = PRESET_MATERIALS[preset]
    structure = _fcc_structure(preset_cfg["base_element"], preset_cfg["lattice_constant"])
    structure.make_supercell(list(supercell))
    structure = _apply_random_composition(structure, preset_cfg["composition"], seed=seed)

    filtered_structure, _ = crystal_helper.filter_atoms_in_cone_and_hemisphere(
        structure,
        cone_diameter_angstrom,
        cone_length_angstrom,
        hemisphere_base_angstrom,
    )

    if any(float(level) > 0 for level in noise_levels_angstrom):
        filtered_structure, _ = crystal_helper.apply_noise_to_structure(
            filtered_structure,
            noise_levels=noise_levels_angstrom,
            noise_type=noise_type,
        )

    dataset = structure_to_ccapt_dataset(
        filtered_structure,
        detector_radius_cm=detector_radius_cm,
        high_voltage=high_voltage,
        flight_path_length_mm=flight_path_length_mm,
    )
    if save_path:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data_tools.store_df_to_hdf(dataset, "df", str(output_path))
    return dataset
