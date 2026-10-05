from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go

from pyccapt.calibration.reconstructions import rotation_tools


def test_gif_figure_downsamples_large_scatter_trace():
    points = np.arange(100, dtype=float)
    figure = go.Figure(go.Scatter3d(x=points, y=points, z=points, mode="markers"))

    gif_figure, original_count, sampled_count = rotation_tools._gif_figure(figure, max_scatter_points=20)

    assert original_count == 100
    assert sampled_count == 20
    assert len(gif_figure.data[0].x) == 20


def test_rotation_gif_failure_returns_without_saving(monkeypatch, capsys):
    figure = go.Figure(go.Scatter3d(x=[0, 1], y=[0, 1], z=[0, 1], mode="markers"))
    variables = SimpleNamespace(result_path="")

    def _fail(_figure):
        raise rotation_tools.GifFrameExportError("renderer timed out")

    monkeypatch.setattr(rotation_tools, "plotly_fig2array", _fail)
    monkeypatch.setattr(rotation_tools, "save_gif", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()))

    rotation_tools.rotary_fig(figure, variables, rotary_fig_save=False, make_gif=True, figname="demo")

    assert "Rotation GIF was cancelled" in capsys.readouterr().out
