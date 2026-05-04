from __future__ import annotations

from pathlib import Path

from pyccapt.calibration.reconstructions import tapsim_builder


def test_single_phase_bulk_build_creates_atoms_and_species_map():
    result = tapsim_builder.build_tapsim_specimen(
        preset="Al",
        mode="single-phase",
        geometry_mode="bulk",
        single_phase_composition={"Al": 1.0},
        a_gamma=4.0495,
        supercell_size=(4, 4, 4),
        n_precip=0,
        seed=7,
    )

    assert len(result.atoms_df) > 0
    assert set(result.atoms_df["phase"]) == {"gamma"}
    assert result.species_map_df["ID"].tolist() == [10]
    assert result.species_map_df["element"].tolist() == ["Al"]


def test_gamma_gprime_tip_build_and_export_creates_tapsim_files(tmp_path: Path):
    result = tapsim_builder.build_tapsim_specimen(
        preset="Nimonic 90",
        mode="gamma-gamma-prime",
        geometry_mode="tapered-tip",
        a_gamma=3.54,
        a_gprime=3.57,
        supercell_size=(8, 8, 10),
        wt_spec={
            "Cr": ["range", 18.0, 21.0],
            "Co": ["range", 15.0, 21.0],
            "Ti": ["range", 2.0, 3.0],
            "Al": ["range", 1.0, 2.0],
            "Fe": ["max", 1.5],
        },
        gamma_partition={"Al": 0.3, "Ti": 0.25, "Cr": 1.0, "Co": 1.0, "Fe": 1.0, "Ni": 1.0},
        gprime_a_site_ratio={"Al": 0.6, "Ti": 0.4},
        n_precip=1,
        precipitate_r_min_angstrom=3.0,
        precipitate_r_max_angstrom=4.0,
        cone_diameter_angstrom=18.0,
        cone_length_angstrom=28.0,
        hemisphere_base_angstrom=10.0,
        seed=11,
    )

    assert len(result.atoms_df) > 0
    assert 0 in set(result.species_map_df["ID"])
    assert 2 in set(result.species_map_df["ID"])
    assert result.summary["precipitate_count"] >= 1

    export_paths = tapsim_builder.export_tapsim_bundle(
        result,
        output_dir=str(tmp_path),
        dataset_name="sample_tip",
        include_potential=False,
        write_csv=True,
        write_hdf=False,
        write_range_hdf=False,
    )

    txt_path = Path(export_paths["datapart_txt"])
    species_map_path = Path(export_paths["species_map_csv"])
    atoms_csv_path = Path(export_paths["atoms_csv"])

    assert txt_path.is_file()
    assert species_map_path.is_file()
    assert atoms_csv_path.is_file()
    assert txt_path.read_text(encoding="utf-8").splitlines()[0].startswith("ASCII ")
