"""Drawing uses original row positions and draft-only true-circle callbacks."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from matplotlib.backend_bases import KeyEvent, MouseEvent

from pyccapt.calibration.core.exceptions import CalibrationInputError
from pyccapt.calibration.tutorials.tutorials_helpers.helper_guided_crop import (
    circle_from_drag, create_crop_drawing, detector_density, interval_from_drag,
)


def source():
    frame = pd.DataFrame({"high_voltage (V)": np.linspace(500., 5000., 100),
        "x_det (cm)": np.tile(np.linspace(-3., 3., 10), 10),
        "y_det (cm)": np.repeat(np.linspace(-2., 2., 10), 10),
        "start_counter": np.tile(np.arange(10), 10)}, index=np.arange(100)+9000)
    return SimpleNamespace(n=len(frame), source=frame)


@pytest.mark.parametrize("left,right,expected", [(10.8, 3.2, (3, 11)), (-7., 200., (0, 100)),
    (3., 9., (3, 9)), (4., 4., None), (101., 200., None), (None, 5., None), (np.nan, 5., None)])
def test_interval_preserves_legacy_half_open_source_positions(left, right, expected):
    assert interval_from_drag(left, right, 100) == expected


@pytest.mark.parametrize("drag,expected", [((3., 2., -1., -4.), (1., -1., 2.)),
    ((0., 0., 2., 2.), (1., 1., 1.)), ((0., 0., 0., 1.), None),
    ((None, 0., 1., 1.), None), ((0., 0., np.inf, 1.), None)])
def test_circle_uses_center_and_shorter_diameter_for_reversed_drags(drag, expected):
    assert circle_from_drag(*drag) == expected


def test_density_exactly_counts_original_interval_with_nans_and_edge_values():
    workflow = source()
    workflow.source.iloc[24, workflow.source.columns.get_loc("x_det (cm)")] = np.nan
    workflow.source.iloc[25, workflow.source.columns.get_loc("y_det (cm)")] = np.inf
    before = workflow.source.copy(deep=True)
    counts, xedges, yedges, count = detector_density(workflow, 20, 51, bins=9, chunk_size=7)
    x, y = (before[name].iloc[20:51].to_numpy() for name in ("x_det (cm)", "y_det (cm)"))
    valid = np.isfinite(x) & np.isfinite(y)
    expected = np.histogram2d(x[valid], y[valid], bins=(xedges, yedges))[0]
    np.testing.assert_array_equal(counts, expected)
    assert count == counts.sum() == 29
    pd.testing.assert_frame_equal(workflow.source, before)


def test_density_constant_coordinates_and_unavailable_inputs():
    workflow = source()
    workflow.source[["x_det (cm)", "y_det (cm)"]] = 0.
    counts, xedges, yedges, count = detector_density(workflow, 7, 19, chunk_size=3)
    assert count == counts.sum() == 12
    assert xedges[0] < 0 < xedges[-1] and yedges[0] < 0 < yedges[-1]
    workflow.source.attrs["pyccapt_unavailable_columns"] = ["x_det (cm)"]
    with pytest.raises(CalibrationInputError, match="unavailable"):
        detector_density(workflow)


def _mouse(drawing, axis, name, x, y):
    px, py = axis.transData.transform((x, y))
    event = MouseEvent(name, drawing.canvas, px, py, button=1)
    drawing.canvas.callbacks.process(name, event)


def test_actual_rectangle_and_circle_events_only_change_draft_callbacks():
    workflow = source()
    before = workflow.source.copy(deep=True)
    intervals, circles = [], []
    draft = [0, workflow.n]
    drawing = create_crop_drawing(workflow, on_interval=lambda *value: intervals.append(value),
        on_circle=lambda *value: circles.append(value), interval_provider=lambda: tuple(draft), interactive=False)
    try:
        drawing.canvas.draw()
        time_axis, detector_axis = drawing.axes
        _mouse(drawing, time_axis, "button_press_event", 30.2, 4000.)
        _mouse(drawing, time_axis, "motion_notify_event", 10.8, 1000.)
        _mouse(drawing, time_axis, "button_release_event", 10.8, 1000.)
        assert intervals == [(10, 31)]
        assert drawing.density_interval == (0, 100)
        draft[:] = intervals[-1]
        drawing.refresh_detector()
        assert drawing.density_interval == (10, 31) and drawing.density_count == 21
        # Return to full detector extents for a reversed non-square drag.
        draft[:] = [0, 100]
        drawing.refresh_detector()
        drawing.canvas.draw()
        _mouse(drawing, detector_axis, "button_press_event", 2., 1.)
        _mouse(drawing, detector_axis, "motion_notify_event", -2., -1.)
        _mouse(drawing, detector_axis, "button_release_event", -2., -1.)
        np.testing.assert_allclose(circles[-1], [0., 0., 1.], atol=.03)
        patch = drawing.circle_selector._selection_artist
        assert patch.width == pytest.approx(patch.height)
        np.testing.assert_allclose(patch.center, circles[-1][:2])
        assert patch.radius == pytest.approx(circles[-1][2])
        assert detector_axis.get_aspect() == 1.
        pd.testing.assert_frame_equal(workflow.source, before)
        assert not hasattr(workflow, "accepted")
    finally:
        drawing.close()


def test_escape_disabled_and_closed_selectors_never_publish_a_drag():
    intervals, circles = [], []
    drawing = create_crop_drawing(source(), on_interval=lambda *value: intervals.append(value),
        on_circle=lambda *value: circles.append(value), interactive=False)
    drawing.canvas.draw()
    axis = drawing.axes[0]
    _mouse(drawing, axis, "button_press_event", 10., 1000.)
    _mouse(drawing, axis, "motion_notify_event", 30., 4000.)
    event = KeyEvent("key_press_event", drawing.canvas, key="escape")
    drawing.canvas.callbacks.process("key_press_event", event)
    _mouse(drawing, axis, "button_release_event", 30., 4000.)
    assert not intervals
    press, release = SimpleNamespace(xdata=1., ydata=1.), SimpleNamespace(xdata=5., ydata=5.)
    drawing.set_enabled(False)
    drawing.cancelled = False
    drawing._rectangle_selected(press, release)
    drawing._circle_selected(press, release)
    assert not intervals and not circles
    drawing.set_enabled(True)
    drawing._rectangle_selected(press, release)
    assert intervals == [(1, 5)]
    drawing.close()
    drawing.set_enabled(True)
    drawing._rectangle_selected(press, release)
    drawing._circle_selected(press, release)
    assert intervals == [(1, 5)] and not circles


def test_real_widget_canvas_does_not_switch_global_backend():
    import ipywidgets
    import matplotlib
    pytest.importorskip("ipympl")
    backend = matplotlib.get_backend()
    drawing = create_crop_drawing(source(), on_interval=lambda *_: None, on_circle=lambda *_: None)
    try:
        assert isinstance(drawing.canvas, ipywidgets.Widget)
        assert drawing.canvas in drawing.widget.children
        assert matplotlib.get_backend() == backend
        assert drawing.density_count == 100
    finally:
        drawing.close()


def test_missing_detector_positions_do_not_block_temporal_drawing():
    workflow = source()
    workflow.source.loc[:, "x_det (cm)"] = np.nan
    intervals = []
    drawing = create_crop_drawing(workflow, on_interval=lambda *value: intervals.append(value),
        on_circle=lambda *_: pytest.fail("Unavailable detector coordinates must not publish a circle"), interactive=False)
    try:
        assert drawing.rectangle_selector.active and not drawing.circle_selector.active
        press, release = SimpleNamespace(xdata=1., ydata=1.), SimpleNamespace(xdata=5., ydata=5.)
        drawing._rectangle_selected(press, release)
        drawing._circle_selected(press, release)
        assert intervals == [(1, 5)]
        workflow.source.loc[:, "x_det (cm)"] = 0.
        drawing.refresh_detector()
        assert drawing.circle_selector.active and drawing.density_count == 100
    finally:
        drawing.close()


def test_draft_overlay_sync_is_exact_silent_and_does_not_recompute_density(monkeypatch):
    drawing = create_crop_drawing(source(), on_interval=lambda *_: pytest.fail("Overlay emitted interval"),
        on_circle=lambda *_: pytest.fail("Overlay emitted circle"), interactive=False)
    monkeypatch.setattr(drawing, "refresh_detector", lambda: pytest.fail("Overlay recomputed density"))
    try:
        drawing.set_draft((7, 41), (.25, -.5, .75))
        np.testing.assert_allclose(drawing.rectangle_selector.extents, (7, 41, *drawing.axes[0].get_ylim()))
        patch = drawing.circle_selector._selection_artist
        np.testing.assert_allclose(patch.center, [.25, -.5])
        assert patch.radius == pytest.approx(.75) and patch.get_visible()
        assert drawing.density_interval == (0, 100)
        drawing.set_draft((15, 30), None)
        assert not patch.get_visible()
        assert drawing.rectangle_selector._selection_artist.get_visible()
        drawing.set_draft((np.nan, 30), (0., 0., -1.))
        assert not drawing.rectangle_selector._selection_artist.get_visible()
        assert not patch.get_visible()
        drawing.set_draft((1.5, 10), (0., 0., np.inf))
        assert not drawing.rectangle_selector._selection_artist.get_visible()
        drawing.set_draft((0, 100), (0., 0., 1.))
        assert patch.get_visible()
        drawing.close()
        drawing.set_draft((1, 2), None)
    finally:
        drawing.close()


def test_starting_circle_encloses_all_finite_hits_and_keeps_handles_visible():
    workflow = source()
    drawing = create_crop_drawing(workflow, on_interval=lambda *_: None, on_circle=lambda *_: None,
                                  interactive=False)
    try:
        circle = drawing.initial_circle()
        cx, cy, radius = circle
        distance = np.hypot(workflow.source["x_det (cm)"]-cx, workflow.source["y_det (cm)"]-cy)
        assert np.all(distance <= radius)
        drawing.set_draft((0, workflow.n), circle)
        x0, x1 = drawing.axes[1].get_xlim()
        y0, y1 = drawing.axes[1].get_ylim()
        assert x0 < cx-radius < cx+radius < x1
        assert y0 < cy-radius < cy+radius < y1
    finally:
        drawing.close()


def test_existing_circle_can_grow_shrink_and_move_using_actual_handle_events():
    circles = []
    drawing = create_crop_drawing(source(), on_interval=lambda *_: None,
        on_circle=lambda *circle: circles.append(circle), interactive=False)
    try:
        axis = drawing.axes[1]
        for original, target in ((.7, 1.1), (1.1, .45)):
            drawing.set_draft((0, 100), (.25, -.2, original))
            drawing.canvas.draw()
            _mouse(drawing, axis, "button_press_event", .25+original, -.2)
            _mouse(drawing, axis, "motion_notify_event", .25+target, -.2)
            _mouse(drawing, axis, "button_release_event", .25+target, -.2)
            np.testing.assert_allclose(circles[-1], (.25, -.2, target), atol=.025)
            assert drawing.circle_selector._selection_artist.width == pytest.approx(
                drawing.circle_selector._selection_artist.height)
        drawing.set_draft((0, 100), (.25, -.2, .45))
        drawing.canvas.draw()
        _mouse(drawing, axis, "button_press_event", .25, -.2)
        _mouse(drawing, axis, "motion_notify_event", .7, .3)
        _mouse(drawing, axis, "button_release_event", .7, .3)
        np.testing.assert_allclose(circles[-1], (.7, .3, .45), atol=.025)
    finally:
        drawing.close()
