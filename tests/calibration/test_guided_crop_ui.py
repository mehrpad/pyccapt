"""Drawing edits drafts; preview/accept and source lifetimes remain explicit."""
from types import SimpleNamespace

import ipywidgets as widgets
import numpy as np
import pytest

from pyccapt.calibration.tutorials.tutorials_helpers import helper_guided_crop as drawing_helper
from pyccapt.calibration.tutorials.tutorials_helpers import helper_guided_workflow as helper
from tests.calibration.test_guided_notebook_ui import ImmediateJob
from tests.calibration.test_guided_workflow_review import workflow


@pytest.fixture
def drawing_panel(monkeypatch):
    def create(session, *, on_interval, on_circle, interval_provider):
        drawing = SimpleNamespace(widget=widgets.VBox(), closed=False, enabled=True, intervals=[],
                                  interval=on_interval, circle=on_circle)
        drawing.close = lambda: setattr(drawing, "closed", True)
        drawing.set_enabled = lambda value: setattr(drawing, "enabled", value)
        drawing.set_draft = lambda interval, circle: setattr(drawing, "draft", (interval, circle))
        drawing.initial_circle = lambda: None
        drawing.refresh_detector = lambda: drawing.intervals.append(interval_provider())
        return drawing
    monkeypatch.setattr(drawing_helper, "create_crop_drawing", create)
    monkeypatch.setattr(helper, "display", lambda *_: None)
    monkeypatch.setattr(helper, "NotebookJob", ImmediateJob)
    return helper.show_guided_workflow(workflow(), display_panel=False)


def test_drawing_requires_preview_and_accept_and_keeps_sequence_semantics(drawing_panel):
    panel = drawing_panel
    session = panel.controller["workflow"]
    panel.controls["Draw crop regions"].click()
    drawing = panel.controller["crop_drawing"]
    drawing.interval(100, 900)
    drawing.circle(0., 0., .2)
    assert drawing.intervals[-1] == (100, 900)
    assert not panel.controls["crop_voltage"].value
    assert panel.controls["crop_detector"].value
    assert session.active_mask.all() and "crop" not in session.preview
    assert panel.controls['Accept crop'].disabled
    assert 'Preview crop' in panel.controls['crop_state'].value
    panel.controls["Preview crop"].click()
    expected = np.zeros(session.n, dtype=bool)
    expected[100:900] = True
    expected &= np.hypot(session.source["x_det (cm)"], session.source["y_det (cm)"]) <= .2
    np.testing.assert_array_equal(session.preview["crop"]["mask"], expected)
    assert session.active_mask.all()
    assert not panel.controls['Accept crop'].disabled
    assert 'Preview ready' in panel.controls['crop_state'].value
    panel.controls["Accept crop"].click()
    np.testing.assert_array_equal(session.active_mask, expected)
    assert panel.controls['Accept crop'].disabled
    assert 'Crop accepted' in panel.controls['crop_state'].value


@pytest.mark.parametrize("field,value", [("crop_start", 101), ("crop_stop", 901),
    ("crop_detector", True), ("crop_radius", .3), ("crop_tof", True),
    ("crop_voltage_min", 2000.)])
def test_editing_crop_invalidates_old_preview(drawing_panel, field, value):
    panel = drawing_panel
    session = panel.controller["workflow"]
    panel.controls["Preview crop"].click()
    assert not panel.controls["Accept crop"].disabled
    panel.controls[field].value = value
    assert "crop" not in session.preview
    assert panel.controls["Accept crop"].disabled
    assert 'Draft selection' in panel.controls['crop_state'].value
    assert session.active_mask.all()


def test_failed_preview_shows_local_reason_and_can_be_corrected(drawing_panel):
    panel = drawing_panel
    session = panel.controller['workflow']
    panel.controls['crop_start'].value = 900
    panel.controls['crop_stop'].value = 100
    panel.controls['Preview crop'].click()
    assert panel.controls['Accept crop'].disabled and 'crop' not in session.preview
    assert 'Preview failed' in panel.controls['crop_state'].value
    assert 'start &lt; stop' in panel.controls['crop_state'].value
    assert session.active_mask.all()
    panel.controls['crop_start'].value = 100
    panel.controls['crop_stop'].value = 900
    panel.controls['Preview crop'].click()
    assert not panel.controls['Accept crop'].disabled
    assert 'Preview ready' in panel.controls['crop_state'].value


def test_failed_plot_does_not_leave_an_unreviewed_crop_accept_enabled(drawing_panel, monkeypatch):
    panel = drawing_panel
    panel.controls['Preview crop'].click()
    assert not panel.controls['Accept crop'].disabled
    def fail(*a, **k):
        raise RuntimeError('Preview plot unavailable')
    monkeypatch.setattr(drawing_helper, 'source_overview', fail)
    panel.controls['Preview crop'].click()
    assert panel.controls['Accept crop'].disabled
    assert 'crop' not in panel.controller['workflow'].preview
    assert 'Preview plot unavailable' in panel.controls['crop_state'].value


def test_undo_restores_fields_and_disconnects_obsolete_drawing(drawing_panel):
    panel = drawing_panel
    session = panel.controller["workflow"]
    panel.controls["Draw crop regions"].click()
    old = panel.controller["crop_drawing"]
    old.interval(100, 900)
    panel.controls["Preview crop"].click()
    panel.controls["Accept crop"].click()
    panel.controls["Undo accepted change"].click()
    assert old.closed and panel.controller["crop_drawing"] is None
    assert panel.controls["crop_start"].value == 0
    assert panel.controls["crop_stop"].value == session.n
    old.interval(200, 800)
    old.circle(1., 1., .5)
    assert panel.controls["crop_start"].value == 0
    assert not panel.controls["crop_detector"].value
    assert session.active_mask.all()


def test_replaced_and_busy_drawing_callbacks_cannot_change_fields(drawing_panel):
    panel = drawing_panel
    panel.controls["Draw crop regions"].click()
    old = panel.controller["crop_drawing"]
    panel.controls["Draw crop regions"].click()
    current = panel.controller["crop_drawing"]
    assert old.closed
    old.interval(200, 800)
    assert panel.controls["crop_start"].value == 0
    panel.controls["job"].running = True
    panel.controls["refresh"]()
    assert not current.enabled
    current.interval(200, 800)
    current.circle(1., 1., .5)
    assert panel.controls["crop_start"].value == 0
    assert not panel.controls["crop_detector"].value


def test_preview_all_resets_visible_draft_and_resume_preserves_pending_crop(drawing_panel, monkeypatch):
    panel = drawing_panel
    session = panel.controller["workflow"]
    panel.controls["crop_start"].value = 100
    panel.controls["crop_stop"].value = 900
    panel.controls["Preview crop"].click()
    resumed = helper.show_guided_workflow(session, display_panel=False)
    assert resumed.controls["crop_start"].value == 100
    assert resumed.controls["crop_stop"].value == 900
    assert not resumed.controls["Accept crop"].disabled
    resumed.controls["Preview all events"].click()
    assert resumed.controls["crop_start"].value == 0
    assert resumed.controls["crop_stop"].value == session.n
    assert not resumed.controls["crop_detector"].value
    assert session.preview["crop"]["mask"].all()


def test_entering_crop_tab_opens_real_selectors_with_conservative_starting_circle(monkeypatch):
    original = drawing_helper.create_crop_drawing
    monkeypatch.setattr(drawing_helper, "create_crop_drawing", lambda *args, **kwargs:
                        original(*args, **kwargs, interactive=False))
    session = workflow()
    panel = helper.show_guided_workflow(session, display_panel=False)
    assert panel.controller["crop_drawing"] is None
    panel.controls["tabs"].selected_index = 2
    drawing = panel.controller["crop_drawing"]
    try:
        assert drawing is not None and drawing.rectangle_selector._selection_artist.get_visible()
        assert drawing.circle_selector._selection_artist.get_visible()
        assert panel.controls["crop_detector"].value
        assert session.active_mask.all() and session.accepted["crop"] is None and not session.preview
        panel.controls["Preview crop"].click()
        assert session.preview["crop"]["mask"].all()
        # Reentering the tab keeps the user's geometry instead of resetting it.
        panel.controls["crop_radius"].value /= 2
        radius = panel.controls["crop_radius"].value
        panel.controls["tabs"].selected_index = 1
        panel.controls["tabs"].selected_index = 2
        assert panel.controller["crop_drawing"] is drawing
        assert panel.controls["crop_radius"].value == radius
    finally:
        drawing.close()


def test_entering_crop_preserves_accepted_event_only_selection(monkeypatch):
    original = drawing_helper.create_crop_drawing
    monkeypatch.setattr(drawing_helper, "create_crop_drawing", lambda *args, **kwargs:
                        original(*args, **kwargs, interactive=False))
    session = workflow()
    session.preview_crop(event_start=100, event_stop=800)
    session.accept("crop")
    panel = helper.show_guided_workflow(session, display_panel=False)
    panel.controls["tabs"].selected_index = 2
    drawing = panel.controller["crop_drawing"]
    try:
        assert not panel.controls["crop_detector"].value
        assert (panel.controls["crop_start"].value, panel.controls["crop_stop"].value) == (100, 800)
        assert session.active_mask.sum() == 700
    finally:
        drawing.close()
