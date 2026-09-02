from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pyccapt.calibration.reconstructions import reconstruction
from pyccapt.calibration.reconstructions.species_display import (
    default_element_controls,
    range_row_masks_and_unranged,
    resolve_element_controls,
)


def _range_data():
    return pd.DataFrame(
        {
            "ion": ["H+", "D+"],
            "element": [["H"], ["D"]],
            "mc_low": [1.0, 3.0],
            "mc_up": [2.0, 4.0],
            "color": ["ff0000", "0000ff"],
        }
    )


def test_controls_always_include_unranged():
    ranges = _range_data()
    assert default_element_controls(ranges, 0.01) == {"H": 0.01, "D": 0.01, "unranged": 0.01}
    rows, unranged = resolve_element_controls(ranges, {"H": 0.2, "D": 0.3, "unranged": 0.4}, 0.1)
    assert rows == [0.2, 0.3]
    assert unranged == 0.4


def test_unranged_mask_is_outside_union_of_mass_ranges():
    mc = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    masks, unranged, valid = range_row_masks_and_unranged(mc, _range_data())
    assert valid == [0, 1]
    np.testing.assert_array_equal(masks[0], [False, True, False, False, False])
    np.testing.assert_array_equal(masks[1], [False, False, False, True, False])
    np.testing.assert_array_equal(unranged, [True, False, True, False, True])


def test_placeholder_range_makes_every_event_unranged():
    ranges = pd.DataFrame(
        {"ion": ["unranged"], "element": [["unranged"]], "mc_low": [0.0], "mc_up": [400.0]}
    )
    masks, unranged, valid = range_row_masks_and_unranged(np.array([1.0, 200.0]), ranges)
    assert valid == []
    assert not masks[0].any()
    assert unranged.all()


def test_reconstruction_adds_configurable_unranged_trace(monkeypatch):
    mc = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    variables = SimpleNamespace(
        mc=mc,
        x=np.arange(5.0),
        y=np.arange(5.0),
        z=np.arange(5.0),
        range_data=_range_data(),
    )
    monkeypatch.setattr(reconstruction.go, "FigureWidget", lambda figure: figure)
    monkeypatch.setattr(reconstruction.go.Figure, "show", lambda *args, **kwargs: None)

    reconstruction.reconstruction_plot(
        variables,
        [1.0, 1.0],
        opacity=0.5,
        rotary_fig_save=False,
        figname="test",
        save=False,
        colab=True,
        element_alpha=[0.2, 0.3],
        unranged_fraction=1.0,
        unranged_alpha=0.4,
    )

    traces = [trace for trace in variables.plotly_3d_reconstruction.data if trace.name is not None]
    assert [trace.name for trace in traces] == ["H+", "D+", "unranged"]
    assert list(traces[-1].x) == [0.0, 2.0, 4.0]
    assert traces[-1].marker.opacity == pytest.approx(0.4)


def test_evaporation_gif_uses_smooth_cumulative_frames_and_requested_fps(monkeypatch):
    mc = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    variables = SimpleNamespace(
        mc=mc,
        x=np.arange(5.0),
        y=np.arange(5.0),
        z=np.arange(5.0),
        dld_t=np.arange(5.0),
        range_data=_range_data(),
    )
    frame_sizes = []
    saved = {}

    def _capture_frame(fig):
        frame_sizes.append(sum(len(trace.x) for trace in fig.data if trace.name is not None))
        return np.zeros((1, 1, 3), dtype=np.uint8)

    def _capture_gif(images, _variables, _filename, *, fps):
        saved.update(frame_count=len(images), fps=fps)

    monkeypatch.setattr(reconstruction.go, "FigureWidget", lambda figure: figure)
    monkeypatch.setattr(reconstruction.go.Figure, "show", lambda *args, **kwargs: None)
    monkeypatch.setattr(reconstruction, "plotly_fig2array", _capture_frame)
    monkeypatch.setattr(reconstruction, "save_gif", _capture_gif)

    reconstruction.reconstruction_plot(
        variables,
        [1.0, 1.0],
        opacity=0.5,
        rotary_fig_save=False,
        figname="test",
        save=False,
        make_evaporation_gif=True,
        colab=True,
        unranged_fraction=1.0,
        evaporation_gif_frames=4,
        evaporation_gif_fps=10,
    )

    assert frame_sizes == [2, 3, 4, 5]
    assert saved == {"frame_count": 4, "fps": 10}
