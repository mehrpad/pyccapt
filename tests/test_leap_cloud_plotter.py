from pathlib import Path

import pandas as pd
import pytest

import pyccapt.calibration.leap_tools.cloud_plotter as cloud_plotter


def _example_deconvolved_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x (nm)": [1.0, 2.0],
            "y (nm)": [1.5, 2.5],
            "z (nm)": [3.0, 4.0],
            "colour": ["#ff0000", "#00ff00"],
        },
        index=["Al", "Cu"],
    )


def test_pre_data_rejects_unsupported_input_type():
    with pytest.raises(ValueError):
        cloud_plotter.pre_data("data.pos", "data.rrng", input_type="csv")


def test_pre_data_dispatches_to_pos_reader(monkeypatch):
    calls = {}

    def fake_read_rrng(path):
        calls["rrng"] = path
        return None, "ranges"

    def fake_read_pos(path):
        calls["pos"] = path
        return "pos_data"

    def fake_label_ions(data, ranges):
        calls["label"] = (data, ranges)
        return "labeled"

    def fake_deconvolve(data):
        calls["deconvolve"] = data
        return {"result": "ok"}

    monkeypatch.setattr(cloud_plotter.leap_tools, "read_rrng", fake_read_rrng)
    monkeypatch.setattr(cloud_plotter.leap_tools, "read_pos", fake_read_pos)
    monkeypatch.setattr(cloud_plotter.leap_tools, "label_ions", fake_label_ions)
    monkeypatch.setattr(cloud_plotter.leap_tools, "deconvolve", fake_deconvolve)

    result = cloud_plotter.pre_data("input.pos", "ranges.rrng", input_type="pos")
    assert result == {"result": "ok"}
    assert Path(calls["rrng"]).name == "ranges.rrng"
    assert Path(calls["pos"]).name == "input.pos"
    assert calls["label"] == ("pos_data", "ranges")
    assert calls["deconvolve"] == "labeled"


def test_cloud_plotter_projection_writes_png(tmp_path: Path):
    data = _example_deconvolved_dataframe()
    cloud_plotter.cloud_plotter(
        data,
        phases=["Al"],
        result_path=tmp_path,
        filename="projection_demo",
        plot_type="projection",
    )
    assert (tmp_path / "output_projection_demo.png").exists()


def test_cloud_plotter_cloud_mode_returns_figure_dict():
    data = _example_deconvolved_dataframe()
    figure = cloud_plotter.cloud_plotter(
        data,
        phases=["Al", "Cu"],
        result_path=".",
        filename="cloud_demo",
        plot_type="cloud",
    )
    assert isinstance(figure, dict)
    assert "data" in figure
    assert figure["layout"]["title"] == "APT 3D Point Cloud"


def test_cloud_plotter_rejects_invalid_plot_type():
    data = _example_deconvolved_dataframe()
    with pytest.raises(ValueError):
        cloud_plotter.cloud_plotter(
            data,
            phases=["Al"],
            result_path=".",
            filename="invalid",
            plot_type="invalid-plot",
        )
