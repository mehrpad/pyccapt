"""Tests for control config file loading helpers."""

from __future__ import annotations

import json

import pytest

from pyccapt.control.core import read_files


def test_read_json_file_returns_dict(tmp_path):
    json_path = tmp_path / "config.json"
    payload = {"a": 1, "b": "x"}
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    assert read_files.read_json_file(json_path) == payload


def test_read_json_file_rejects_non_object(tmp_path):
    json_path = tmp_path / "config.json"
    json_path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ValueError):
        read_files.read_json_file(json_path)


def test_read_json_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_files.read_json_file(tmp_path / "missing.json")


def test_read_toml_file_returns_dict(tmp_path):
    toml_path = tmp_path / "config.toml"
    toml_path.write_text("value = 12\nflag = true\n", encoding="utf-8")

    assert read_files.read_toml_file(toml_path) == {"value": 12, "flag": True}


def test_load_config_file_supports_toml(tmp_path):
    toml_path = tmp_path / "config.toml"
    toml_path.write_text('format = "toml"\n', encoding="utf-8")

    assert read_files.load_config_file(toml_path)["format"] == "toml"


def test_load_config_file_rejects_json(tmp_path):
    json_path = tmp_path / "config.json"
    json_path.write_text(json.dumps({"format": "json"}), encoding="utf-8")
    with pytest.raises(ValueError):
        read_files.load_config_file(json_path)


def test_load_config_file_rejects_unsupported_suffix(tmp_path):
    ini_path = tmp_path / "config.ini"
    ini_path.write_text("[x]\na=1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_files.load_config_file(ini_path)

