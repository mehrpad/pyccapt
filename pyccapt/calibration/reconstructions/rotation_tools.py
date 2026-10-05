"""Rotation and frame-export helpers for 3D reconstruction figures."""

from __future__ import annotations

import io
import os
import signal
import subprocess
import sys

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image

from pyccapt.calibration.reconstructions.io_utils import save_gif, save_plotly_animation


_GIF_FRAME_TIMEOUT_SECONDS = 45
_GIF_MAX_SCATTER_POINTS = 75_000
_PLOTLY_RENDER_SCRIPT = """
import sys
import plotly.io as pio

figure = pio.from_json(sys.stdin.buffer.read().decode('utf-8'))
png = pio.to_image(figure, format='png', scale=int(sys.argv[1]), engine='kaleido')
sys.stdout.buffer.write(png)
"""


class GifFrameExportError(RuntimeError):
    """Raised when the isolated Plotly/Kaleido renderer cannot make a GIF frame."""


def _stop_process_tree(process: subprocess.Popen) -> None:
    """Stop a timed-out renderer and its Kaleido child processes."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        os.killpg(process.pid, signal.SIGKILL)


def _render_plotly_png(fig, *, scale: int, timeout_seconds: float) -> bytes:
    """Render one frame in a killable child process with a hard timeout."""
    if timeout_seconds <= 0:
        raise ValueError("GIF frame timeout must be greater than zero")
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen(
        [sys.executable, "-c", _PLOTLY_RENDER_SCRIPT, str(scale)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=os.name != "nt",
        creationflags=creationflags,
    )
    try:
        png, error = process.communicate(fig.to_json().encode("utf-8"), timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        _stop_process_tree(process)
        process.communicate()
        raise GifFrameExportError(
            f"Kaleido did not render a frame within {timeout_seconds:g} seconds"
        ) from None
    if process.returncode or not png:
        detail = error.decode("utf-8", errors="replace").strip()
        raise GifFrameExportError(detail or "Kaleido returned no image data")
    return png


def _sample_values(values, indices: np.ndarray, expected_size: int):
    """Return indexed trace data only when it is one value per plotted point."""
    if values is None or isinstance(values, str):
        return values
    try:
        if len(values) == expected_size:
            return [values[index] for index in indices]
    except TypeError:
        pass
    return values


def _gif_figure(fig, *, max_scatter_points: int = _GIF_MAX_SCATTER_POINTS) -> tuple[go.Figure, int, int]:
    """Clone and deterministically thin 3D scatter traces for responsive GIF export."""
    gif_figure = go.Figure(fig)
    scatter_traces = [trace for trace in gif_figure.data if isinstance(trace, go.Scatter3d)]
    counts = [len(trace.x) if trace.x is not None else 0 for trace in scatter_traces]
    total = sum(counts)
    if total <= max_scatter_points:
        return gif_figure, total, total

    remaining = max_scatter_points
    for position, (trace, count) in enumerate(zip(scatter_traces, counts)):
        if not count:
            continue
        keep = min(count, max(1, round(max_scatter_points * count / total)))
        if position == len(scatter_traces) - 1:
            keep = min(count, max(1, remaining))
        remaining -= keep
        indices = np.linspace(0, count - 1, keep, dtype=int)
        for field in ("x", "y", "z", "text", "hovertext", "customdata", "ids"):
            setattr(trace, field, _sample_values(getattr(trace, field, None), indices, count))
        for field in ("color", "size", "symbol", "opacity"):
            setattr(trace.marker, field, _sample_values(getattr(trace.marker, field, None), indices, count))
    return gif_figure, total, sum(len(trace.x) if trace.x is not None else 0 for trace in scatter_traces)


def rotate_z(x, y, z, theta):
    """Rotate coordinates around the z-axis."""
    w_values = x + 1j * y
    return np.real(np.exp(1j * theta) * w_values), np.imag(np.exp(1j * theta) * w_values), z


def plotly_fig2array(fig, *, scale: int = 1, timeout_seconds: float = _GIF_FRAME_TIMEOUT_SECONDS):
    """Convert a Plotly figure into a GIF frame without allowing Kaleido to hang the notebook."""
    fig_bytes = _render_plotly_png(fig, scale=scale, timeout_seconds=timeout_seconds)
    buffer = io.BytesIO(fig_bytes)
    with Image.open(buffer) as image:
        return np.asarray(image.convert("RGB"))


def rotary_fig(fig, variables, rotary_fig_save, make_gif, figname):
    """Generate and optionally save rotating 3D Plotly figure variants."""
    x_eye = -1.25
    y_eye = 2
    z_eye = 0.5
    fig = go.Figure(fig)

    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)

    if make_gif:
        from tqdm.auto import tqdm

        gif_figure, original_points, gif_points = _gif_figure(fig)
        gif_figure.update_layout(showlegend=False)
        gif_figure.update_layout(margin=go.layout.Margin(l=0, r=0, b=0, t=0))
        if gif_points < original_points:
            print(f"Rotation GIF uses {gif_points:,} of {original_points:,} scatter points for responsive export.")

        thetas = np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / 20.0)
        images = []
        try:
            for frame_index, theta in enumerate(tqdm(thetas, desc="Rotation GIF frames", unit="frame"), start=1):
                xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, theta)
                gif_figure.update_layout(scene_camera_eye=dict(x=xe, y=ye, z=ze))
                try:
                    images.append(plotly_fig2array(gif_figure))
                except GifFrameExportError as exc:
                    raise GifFrameExportError(f"frame {frame_index}/{len(thetas)} failed: {exc}") from exc
            save_gif(images, variables, f"rota_{figname}.gif", fps=2)
            print(f"Saved rotation GIF with {len(images)} frames.")
        except (GifFrameExportError, OSError, ValueError) as exc:
            print(f"Rotation GIF was cancelled; the 3D plot is still available. {exc}")
        finally:
            images.clear()

    if rotary_fig_save:
        fig.update_layout(
            scene_camera_eye=dict(x=x_eye, y=y_eye, z=z_eye),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=1.2,
                    x=0.8,
                    xanchor="left",
                    yanchor="bottom",
                    pad=dict(t=45, r=10),
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=15, redraw=True),
                                    transition=dict(duration=0),
                                    fromcurrent=True,
                                    mode="immediate",
                                ),
                            ],
                        )
                    ],
                )
            ],
        )

        frames = []
        for theta in np.arange(0, 50, 0.1):
            xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -theta)
            frames.append(go.Frame(layout=dict(scene_camera_eye=dict(x=xe, y=ye, z=ze))))
        fig.frames = frames

        save_plotly_animation(
            fig,
            variables,
            filename=f"rota_{figname}.html",
            show_link=True,
            auto_open=False,
            include_mathjax="cdn",
            add_camera_gif_exporter=True,
        )


__all__ = ["rotate_z", "plotly_fig2array", "rotary_fig"]
