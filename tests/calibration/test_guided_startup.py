"""An updated notebook must not silently render cached, incompatible controls."""
import sys

import pytest

from pyccapt.calibration.core import guided_workflow as core
from pyccapt.calibration.tutorials.tutorials_helpers import helper_guided_startup as startup
from pyccapt.calibration.tutorials.tutorials_helpers import helper_guided_workflow as helper
from tests.calibration.test_guided_notebook_ui import ImmediateJob
from tests.calibration.test_guided_workflow_review import workflow


def test_current_runtime_reports_paths_and_has_crop_before_calibration(capsys):
    factory = startup.load_guided_workflow()
    panel = factory(display_panel=False)
    tabs = panel.controls["tabs"]
    assert tuple(tabs.get_title(i) for i in range(len(tabs.children))) == startup.EXPECTED_STAGES
    assert panel.controller["workflow"] is None
    report = capsys.readouterr().out
    assert "Crop → Calibrate" in report
    assert sys.executable in report
    assert "PyCCAPT source:" in report


@pytest.mark.parametrize("stale", ["core_stages", "helper_stages", "ui_version", "class"])
def test_cached_old_or_mixed_runtime_requires_restart_without_reloading(monkeypatch, stale):
    old_factory = helper.show_guided_workflow
    if stale == "core_stages":
        monkeypatch.setattr(core, "STAGES", tuple(stage for stage in core.STAGES if stage != "Crop"))
    elif stale == "helper_stages":
        monkeypatch.setattr(helper, "STAGES", ("Import", "Calibrate"))
    elif stale == "ui_version":
        monkeypatch.delattr(helper, "GUIDED_UI_API_VERSION")
    else:
        monkeypatch.setattr(helper, "GuidedWorkflow", object())
    with pytest.raises(RuntimeError, match="Restart Kernel and Run All"):
        startup.load_guided_workflow()
    assert helper.show_guided_workflow is old_factory


def test_other_checkout_requires_restart_and_reports_loaded_path(tmp_path):
    with pytest.raises(RuntimeError, match="This kernel loaded PyCCAPT from") as caught:
        startup.load_guided_workflow(repository_root=tmp_path)
    assert str(tmp_path) in str(caught.value)
    assert "Restart Kernel and Run All" in str(caught.value)


def test_job_completion_does_not_still_say_converting(monkeypatch):
    monkeypatch.setattr(helper, "NotebookJob", ImmediateJob)
    monkeypatch.setattr(helper, "display", lambda *_: None)
    session = workflow()
    monkeypatch.setattr(core.GuidedWorkflow, "import_file", lambda *_, **__: session)
    panel = helper.show_guided_workflow(display_panel=False)
    panel.controls["Import dataset"].click()
    assert panel.controller["workflow"] is session
    assert not panel.controls["job"].running
    assert panel.controls["progress_text"].value == "Operation complete; review the result below."
    assert panel.controls["progress"].bar_style == "success"


def test_failed_job_reports_stopped_and_keeps_existing_source(monkeypatch):
    monkeypatch.setattr(helper, "NotebookJob", ImmediateJob)
    def fail(*_, **__):
        raise ValueError("Invalid input")
    monkeypatch.setattr(core.GuidedWorkflow, "import_file", fail)
    session = workflow()
    panel = helper.show_guided_workflow(session, display_panel=False)
    panel.controls["Import dataset"].click()
    assert panel.controller["workflow"] is session
    assert not panel.controls["job"].running
    assert panel.controls["progress_text"].value == "Operation stopped; see the message below."
    assert panel.controls["progress"].bar_style == "warning"
