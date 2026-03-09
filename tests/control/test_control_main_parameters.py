"""Tests for main GUI parameter helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from pyccapt.control.gui import main_parameters


@pytest.fixture
def conf():
    return {
        "max_vdc": 10000,
        "min_vp": 100,
        "max_vp": 1000,
        "pulse_fraction_max": 20,
        "max_voltage_pulse_frequency": 200000,
        "min_voltage_pulse_frequency": 1,
        "max_laser_pulse_frequency": 10000,
        "min_laser_pulse_frequency": 1,
    }


def test_parse_textline_experiments_requires_keys():
    with pytest.raises(main_parameters.ParameterError):
        main_parameters.parse_textline_experiments("{ex_user=a}")


def test_parse_textline_experiments_parses_numeric_and_bool():
    block = (
        "{ex_user=u;ex_name=e;electrode=el;ex_time=10;max_ions=20;ex_freq=5;"
        "vdc_min=100;vdc_max=500;vdc_steps_up=0.5;vdc_steps_down=0.7;control_algorithm=PID;"
        "pulse_mode=Voltage;vp_min=200;vp_max=900;pulse_fraction=10;pulse_frequency=200;"
        "detection_rate_init=1.5;hit_displayed=1000;email=a@b.com;counter_source=TDC;"
        "criteria_time=true;criteria_ions=false;criteria_vdc=true}"
    )
    items = main_parameters.parse_textline_experiments(block)
    assert items[0]["vdc_steps_up"] == 0.5
    assert items[0]["criteria_time"] is True


def test_apply_form_values_returns_corrections(conf):
    variables = SimpleNamespace()
    errors: list[str] = []

    values = main_parameters.FormValues(
        user_name="u",
        ex_name="exp",
        electrode="el",
        ex_time="10",
        ex_freq="2",
        max_ions="100",
        vdc_min="100",
        vdc_max="12000",
        detection_rate_init="1.0",
        pulse_fraction="22",
        email="a@b.com",
        vdc_steps_up="1",
        vdc_steps_down="1",
        control_algorithm="PID",
        pulse_mode="Voltage",
        vp_min="120",
        vp_max="4000",
        pulse_frequency="300000",
        counter_source="TDC",
        criteria_time=True,
        criteria_ions=False,
        criteria_vdc=True,
    )

    corrections = main_parameters.apply_form_values(
        variables,
        conf,
        values,
        lambda msg: errors.append(msg),
    )

    assert "pulse_fraction" in corrections
    assert "vp_max" in corrections
    assert "vdc_max" in corrections
    assert "pulse_frequency" in corrections
    assert errors


def test_load_electrode_items_from_toml(tmp_path):
    toml_path = tmp_path / "electrode.toml"
    toml_path.write_text(
        (
            "# Electrode config\n"
            "[electrodes]\n"
            "names = [\n"
            '  "E1",\n'
            '  "E2",\n'
            "]\n"
        ),
        encoding="utf-8",
    )

    assert main_parameters.load_electrode_items(str(toml_path)) == ["E1", "E2"]


def test_load_electrode_items_falls_back_to_legacy_json(tmp_path):
    toml_path = tmp_path / "electrode.toml"
    json_path = tmp_path / "electrode.json"
    json_path.write_text(
        json.dumps({"2": "E2", "1": "E1", "3": ""}),
        encoding="utf-8",
    )

    assert main_parameters.load_electrode_items(str(toml_path)) == ["E1", "E2"]
