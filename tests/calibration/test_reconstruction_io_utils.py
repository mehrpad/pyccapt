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


def test_save_gif_uses_resolved_output_path(tmp_path: Path):
    variables = _DummyVariables(result_path=str(tmp_path))
    with patch("pyccapt.calibration.reconstructions.io_utils.imageio.mimsave") as mock_save:
        io_utils.save_gif([object()], variables, "movie.gif", fps=5)
    mock_save.assert_called_once()
    call_args = mock_save.call_args
    assert Path(call_args.args[0]) == tmp_path / "movie.gif"
    assert call_args.kwargs["fps"] == 5


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


def test_write_plotly_html_uses_resolved_output_path(tmp_path: Path):
    variables = _DummyVariables(result_path=str(tmp_path))
    fig = object()
    with patch("pyccapt.calibration.reconstructions.io_utils.pio.write_html") as mock_write:
        io_utils.write_plotly_html(fig, variables, "plot.html")
    mock_write.assert_called_once()
    assert Path(mock_write.call_args.args[1]) == tmp_path / "plot.html"


def test_write_plotly_image_infers_format_from_filename(tmp_path: Path):
    variables = _DummyVariables(result_path=str(tmp_path))
    fig = object()
    with patch("pyccapt.calibration.reconstructions.io_utils.pio.write_image") as mock_write:
        io_utils.write_plotly_image(fig, variables, "plot.svg", scale=2)
    mock_write.assert_called_once()
    kwargs = mock_write.call_args.kwargs
    assert kwargs["scale"] == 2
    assert kwargs["format"] == "svg"
