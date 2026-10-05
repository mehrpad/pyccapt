import numpy as np
import pandas as pd
from pathlib import Path

from pyccapt.calibration.core.concentration_profile import (
    calculate_concentration_profile,
    plot_concentration_profile,
    profile_species_options,
)


def _ranges():
    return pd.DataFrame(
        {
            "name": ["H", "H2", "O"],
            "mc_low": [0.5, 1.5, 15.5],
            "mc_up": [1.5, 2.5, 16.5],
            "element": [["H"], ["H"], ["O"]],
            "complex": [[1], [2], [1]],
        }
    )


def test_profile_uses_all_events_as_denominator_and_stoichiometry():
    mc = np.array([1.0, 2.0, 16.0, 99.0, 16.0, 16.1])
    profile = calculate_concentration_profile(
        mc,
        _ranges(),
        ["element:H", "ion:1"],
        window_size=4,
    )

    assert profile["sequence_start"].tolist() == [1, 5]
    assert profile["sequence_end"].tolist() == [4, 6]
    # First window: H contributes 1 atom, H2 contributes 2, O contributes 1,
    # and the unranged event contributes one unknown atom-equivalent.
    assert profile["H (element)"].tolist() == [60.0, 0.0]
    assert profile["H2 (ion)"].tolist() == [40.0, 0.0]
    assert profile["ranged_atoms"].tolist() == [4.0, 2.0]
    assert profile["unranged_events"].tolist() == [1, 0]
    assert profile["total_atom_equivalents"].tolist() == [5.0, 2.0]
    assert np.isclose(profile.attrs["overall_percentages"]["H (element)"], 300 / 7)
    assert np.isclose(profile.attrs["overall_percentages"]["H2 (ion)"], 200 / 7)


def test_plot_legend_reports_only_selected_overall_percentages():
    profile = calculate_concentration_profile(
        np.array([1.0, 2.0, 16.0, 16.1]),
        _ranges(),
        ["element:H"],
        window_size=2,
    )
    fig, axis = plot_concentration_profile(profile)
    labels = axis.get_legend_handles_labels()[1]
    assert labels == ["H: 60.00 at.%"]
    fig.clear()


def test_profile_can_drop_final_partial_window():
    profile = calculate_concentration_profile(
        np.array([1.0, 2.0, 16.0, 16.1, 1.1]),
        _ranges(),
        ["H"],
        window_size=4,
        include_partial_window=False,
    )
    assert len(profile) == 1
    assert profile.iloc[0]["detected_events"] == 4


def test_overlapping_ranges_use_first_range_without_double_counting():
    ranges = pd.DataFrame(
        {
            "name": ["first", "second"],
            "mc_low": [1.0, 1.5],
            "mc_up": [2.0, 2.5],
            "element": [["H"], ["O"]],
            "complex": [[1], [1]],
        }
    )
    profile = calculate_concentration_profile(
        np.array([1.75]), ranges, ["element:H", "element:O"], window_size=1
    )
    assert profile.iloc[0]["ranged_atoms"] == 1
    assert profile.iloc[0]["H (element)"] == 100
    assert profile.iloc[0]["O (element)"] == 0


def test_species_options_include_elements_and_individual_ions():
    options = dict(profile_species_options(_ranges()))
    assert options["Element: H"] == "element:H"
    assert options["Ion: H2"] == "ion:1"
    assert options["Unranged"] == "unranged"


def test_multiple_selected_series_and_unranged_share_the_full_denominator():
    ranges = pd.DataFrame(
        {
            "name": ["H", "D"],
            "mc_low": [0.5, 1.5],
            "mc_up": [1.5, 2.5],
            "element": [["H"], ["D"]],
            "complex": [[1], [1]],
        }
    )
    profile = calculate_concentration_profile(
        np.array([1.0, 2.0, 9.0, 10.0]),
        ranges,
        ["element:H", "element:D", "unranged"],
        window_size=4,
    )
    assert profile.iloc[0]["H (element)"] == 25.0
    assert profile.iloc[0]["D (element)"] == 25.0
    assert profile.iloc[0]["Unranged"] == 50.0
    assert sum(profile.iloc[0][name] for name in ("H (element)", "D (element)", "Unranged")) == 100.0

    fig, axis = plot_concentration_profile(profile)
    assert len(axis.lines) == 3
    assert axis.get_legend_handles_labels()[1] == [
        "H: 25.00 at.%", "D: 25.00 at.%", "Unranged: 50.00 at.%"
    ]
    fig.clear()


def test_unranged_profile_works_without_any_real_ranges():
    placeholder = pd.DataFrame(
        {
            "name": ["unranged"],
            "mc_low": [0.0],
            "mc_up": [400.0],
            "element": [["unranged"]],
            "complex": [[0]],
        }
    )
    profile = calculate_concentration_profile(
        np.array([1.0, 2.0, 3.0]), placeholder, ["unranged"], window_size=2
    )
    assert profile["Unranged"].tolist() == [100.0, 100.0]


def test_plot_y_axis_expands_around_a_constant_concentration():
    profile = pd.DataFrame({"sequence": [1.0, 2.0], "H (element)": [25.0, 25.0]})
    profile.attrs["overall_percentages"] = {"H (element)": 25.0}
    fig, axis = plot_concentration_profile(profile)
    lower, upper = axis.get_ylim()
    assert 0.0 < lower < 25.0 < upper < 100.0
    fig.clear()


def test_bundled_isotope_table_contains_deuterium_alias():
    table_path = Path(__file__).resolve().parents[2] / "pyccapt" / "files" / "isotopeTable.h5"
    isotopes = pd.read_hdf(table_path, key="isotope")
    deuterium = isotopes[(isotopes["element"] == "D") & (isotopes["isotope"] == 2)]
    assert len(deuterium) == 1
    assert deuterium.iloc[0]["weight"] == 2.01

    colors = pd.read_hdf(table_path.with_name("color_scheme.h5"), key="df")
    assert colors["ion"].astype(str).eq("D").sum() == 1
