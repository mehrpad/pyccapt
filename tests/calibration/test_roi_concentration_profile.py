import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from pyccapt.calibration.core.concentration_profile import (
    calculate_roi_concentration_profile,
    plot_roi_concentration_profile,
)
from pyccapt.calibration.tutorials.tutorials_helpers.helper_roi_concentration_profile import (
    build_roi_concentration_profile_panel,
)


def _ranges():
    return pd.DataFrame(
        {
            "name": ["A", "B"],
            "mc_low": [0.5, 1.5],
            "mc_up": [1.5, 2.5],
            "element": [["A"], ["B"]],
            "complex": [[1], [1]],
        }
    )


def test_roi_profile_filters_transverse_box_and_bins_along_x():
    profile = calculate_roi_concentration_profile(
        x_values=[0.1, 0.6, 1.1, 1.6, 2.1, 2.4],
        y_values=[0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
        z_values=[0.0] * 6,
        mc_values=[1.0, 2.0, 1.0, 2.0, 2.0, 99.0],
        range_data=_ranges(),
        selected_species=["element:A", "element:B", "unranged"],
        axis="x",
        transverse_center=(0.0, 0.0),
        transverse_size=(1.0, 1.0),
        bin_width=1.0,
        profile_start=0.0,
        profile_end=2.5,
    )

    assert profile["detected_events"].tolist() == [1, 2, 2]
    assert profile["A (element)"].tolist() == [100.0, 50.0, 0.0]
    assert profile["B (element)"].tolist() == [0.0, 50.0, 50.0]
    assert profile["Unranged"].tolist() == [0.0, 0.0, 50.0]
    assert profile.attrs["axis"] == "x"
    assert profile.attrs["roi_event_count"] == 5


def test_roi_profile_requires_a_valid_axis_and_nonempty_roi():
    arguments = dict(
        x_values=[0.0], y_values=[0.0], z_values=[0.0], mc_values=[1.0],
        range_data=_ranges(), selected_species=["element:A"],
        transverse_center=(0.0, 0.0), transverse_size=(1.0, 1.0),
    )
    with pytest.raises(ValueError, match="one of: x, y, z"):
        calculate_roi_concentration_profile(axis="time", **arguments)
    with pytest.raises(ValueError, match="contains no ions"):
        calculate_roi_concentration_profile(
            axis="x", transverse_center=(4.0, 4.0),
            transverse_size=arguments["transverse_size"],
            x_values=arguments["x_values"], y_values=arguments["y_values"], z_values=arguments["z_values"],
            mc_values=arguments["mc_values"], range_data=arguments["range_data"],
            selected_species=arguments["selected_species"],
        )


def test_roi_profile_plot_uses_spatial_axis_label():
    profile = calculate_roi_concentration_profile(
        x_values=np.array([0.1, 0.6]), y_values=np.zeros(2), z_values=np.zeros(2),
        mc_values=np.array([1.0, 2.0]), range_data=_ranges(), selected_species=["element:A"],
        axis="x", transverse_center=(0.0, 0.0), transverse_size=(1.0, 1.0),
        bin_width=1.0, profile_start=0.0, profile_end=1.0,
    )
    fig, axis = plot_roi_concentration_profile(profile)
    assert axis.get_xlabel() == "x position [nm]"
    assert axis.get_legend_handles_labels()[1] == ["A: 50.00 at.%"]
    fig.clear()


def test_roi_profile_panel_builds_with_reconstruction_data():
    variables = SimpleNamespace(
        x=np.array([0.0, 1.0]),
        y=np.zeros(2),
        z=np.zeros(2),
        mc=np.array([1.0, 2.0]),
        range_data=_ranges(),
        result_path="",
    )
    panel = build_roi_concentration_profile_panel(variables)
    assert "ROI concentration profile" in panel.children[0].value
    assert len(panel.children) == 15
