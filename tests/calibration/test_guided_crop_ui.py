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
    assert drawing.intervals == [(100, 900)]
    assert not panel.controls["crop_voltage"].value
    assert panel.controls["crop_detector"].value
    assert session.active_mask.all() and "crop" not in session.preview
    panel.controls["Preview crop"].click()
    expected = np.zeros(session.n, dtype=bool)
    expected[100:900] = True
    expected &= np.hypot(session.source["x_det (cm)"], session.source["y_det (cm)"]) <= .2
    np.testing.assert_array_equal(session.preview["crop"]["mask"], expected)
    assert session.active_mask.all()
    panel.controls["Accept crop"].click()
    np.testing.assert_array_equal(session.active_mask, expected)


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
    assert session.active_mask.all()


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
