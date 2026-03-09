"""Tests for control runtime/bootstrap helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pyccapt.control.core import runtime


def test_find_project_root_contains_config_and_package():
    root = runtime.find_project_root(Path(__file__))

    assert (root / "config.toml").exists()
    assert (root / "control").exists()


def test_load_project_config_without_chdir():
    conf, root = runtime.load_project_config(change_cwd=False)

    assert isinstance(conf, dict)
    assert "COM_PORT_cryo" in conf
    assert root.name == "pyccapt"


def test_load_project_config_rejects_explicit_json(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"legacy": True}), encoding="utf-8")

    with pytest.raises(ValueError):
        runtime.load_project_config(
            config_name="config.json",
            root=tmp_path,
            change_cwd=False,
        )


def test_create_shared_context_has_queues_and_variables():
    conf, _ = runtime.load_project_config(change_cwd=False)
    shared = runtime.create_shared_context(conf)

    try:
        assert shared.variables.COM_PORT_cryo == conf["COM_PORT_cryo"]
        assert shared.x_plot is not None
        assert shared.y_plot is not None
        assert shared.t_plot is not None
        assert shared.main_v_dc_plot is not None
    finally:
        shared.manager.shutdown()


def test_load_project_config_requires_toml(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"fallback": True}), encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        runtime.load_project_config(root=tmp_path, change_cwd=False)

