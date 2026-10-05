from pathlib import Path
from unittest.mock import patch

from pyccapt.calibration.reconstructions import io_utils


class _DummyVariables:
    def __init__(self, result_path=""):
        self.result_path = result_path


class _ResolverVariables:
    def __init__(self):
        self.calls = []

    def resolve_result_file(self, filename):
        self.calls.append(filename)
        return f"/tmp/{filename}"


def test_resolve_result_file_prefers_shared_variable_resolver():
    variables = _ResolverVariables()
    resolved = io_utils.resolve_result_file(variables, "demo.png")
    assert resolved == "/tmp/demo.png"
    assert variables.calls == ["demo.png"]


def test_resolve_result_file_falls_back_to_result_path(tmp_path: Path):
    variables = _DummyVariables(result_path=str(tmp_path))
    resolved = io_utils.resolve_result_file(variables, "demo.png")
    assert Path(resolved) == tmp_path / "demo.png"


def test_save_gif_writes_file_to_resolved_path(tmp_path: Path):
    import numpy as np
    variables = _DummyVariables(result_path=str(tmp_path))
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    io_utils.save_gif([frame, frame], variables, "movie.gif", fps=5)
    assert (tmp_path / "movie.gif").exists()


def test_save_gif_rejects_empty_frames(tmp_path: Path):
    variables = _DummyVariables(result_path=str(tmp_path))
    import pytest

    with pytest.raises(ValueError, match="without frames"):
        io_utils.save_gif([], variables, "movie.gif")


def test_save_plotly_animation_uses_resolved_output_path(tmp_path: Path):
    variables = _DummyVariables(result_path=str(tmp_path))
    fig = object()
    with patch("pyccapt.calibration.reconstructions.io_utils.plotly.offline.plot") as mock_plot:
        io_utils.save_plotly_animation(fig, variables, filename="rotation.html", show_link=False, auto_open=False)
    mock_plot.assert_called_once()
    kwargs = mock_plot.call_args.kwargs
    assert Path(kwargs["filename"]) == tmp_path / "rotation.html"
    assert kwargs["show_link"] is False
    assert kwargs["auto_open"] is False


def test_camera_gif_exporter_is_added_to_saved_html(tmp_path: Path):
    output = tmp_path / "rotation.html"
    output.write_text("<html><body><div class='plotly-graph-div'></div></body></html>", encoding="utf-8")

    io_utils._add_camera_gif_exporter(output, "rota_demo.html")

    html = output.read_text(encoding="utf-8")
    assert "Save rotating GIF" in html
    assert 'const baseName = "rota_demo"' in html
    assert "showlegend: false" in html
    assert "createWorkerScriptUrl" in html
    assert "workerScript: workerScriptUrl" in html


def test_save_plotly_animation_can_add_camera_gif_exporter(tmp_path: Path, monkeypatch):
    variables = _DummyVariables(result_path=str(tmp_path))

    def _write_html(_fig, *, filename, **_kwargs):
        Path(filename).write_text("<html><body></body></html>", encoding="utf-8")

    monkeypatch.setattr(io_utils.plotly.offline, "plot", _write_html)
    io_utils.save_plotly_animation(
        object(), variables, filename="rotation.html", show_link=False, add_camera_gif_exporter=True
    )

    assert "Save rotating GIF" in (tmp_path / "rotation.html").read_text(encoding="utf-8")


def test_write_plotly_html_uses_resolved_output_path(tmp_path: Path):
    variables = _DummyVariables(result_path=str(tmp_path))
    fig = object()
    with patch("pyccapt.calibration.reconstructions.io_utils.pio.write_html") as mock_write:
        io_utils.write_plotly_html(fig, variables, "plot.html")
    mock_write.assert_called_once()
    assert Path(mock_write.call_args.args[1]) == tmp_path / "plot.html"


def test_write_plotly_html_can_add_camera_gif_exporter(tmp_path: Path, monkeypatch):
    variables = _DummyVariables(result_path=str(tmp_path))

    def _write_html(_fig, filename, **_kwargs):
        Path(filename).write_text("<html><body></body></html>", encoding="utf-8")

    monkeypatch.setattr(io_utils.pio, "write_html", _write_html)
    io_utils.write_plotly_html(object(), variables, "plot.html", add_camera_gif_exporter=True)

    assert "Save rotating GIF" in (tmp_path / "plot.html").read_text(encoding="utf-8")


def test_write_plotly_image_infers_format_from_filename(tmp_path: Path):
    variables = _DummyVariables(result_path=str(tmp_path))
    fig = object()
    with patch("pyccapt.calibration.reconstructions.io_utils.pio.write_image") as mock_write:
        io_utils.write_plotly_image(fig, variables, "plot.svg", scale=2)
    mock_write.assert_called_once()
    kwargs = mock_write.call_args.kwargs
    assert kwargs["scale"] == 2
    assert kwargs["format"] == "svg"
