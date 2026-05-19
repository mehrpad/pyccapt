"""Rotation and frame-export helpers for 3D reconstruction figures."""

from __future__ import annotations

import io

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image

from pyccapt.calibration.reconstructions.io_utils import save_gif, save_plotly_animation


def rotate_z(x, y, z, theta):
    """Rotate coordinates around the z-axis."""
    w_values = x + 1j * y
    return np.real(np.exp(1j * theta) * w_values), np.imag(np.exp(1j * theta) * w_values), z


def plotly_fig2array(fig):
    """Convert a Plotly figure into an ndarray frame."""
    fig_bytes = pio.to_image(fig, format="jpeg", scale=5, engine="kaleido")
    buffer = io.BytesIO(fig_bytes)
    image = Image.open(buffer)
    return np.asarray(image)


def rotary_fig(fig, variables, rotary_fig_save, make_gif, figname):
    """Generate and optionally save rotating 3D Plotly figure variants."""
    x_eye = -1.25
    y_eye = 2
    z_eye = 0.5
    fig = go.Figure(fig)

    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)

    if make_gif:
        from tqdm.auto import tqdm

        fig.update_layout(showlegend=False)
        fig.update_layout(margin=go.layout.Margin(l=0, r=0, b=0, t=0))

        # Full 2π turn in 20 frames. Reuse a single figure and only swap the
        # camera eye each iteration — deep-copying the figure per frame was
        # the dominant cost when the scene has millions of points.
        thetas = np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / 20.0)
        images = []
        for theta in tqdm(thetas, desc="Rotation GIF frames", unit="frame"):
            xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, theta)
            fig.update_layout(scene_camera_eye=dict(x=xe, y=ye, z=ze))
            images.append(plotly_fig2array(fig))

        save_gif(images, variables, f"rota_{figname}.gif", fps=2)
        fig.update_layout(showlegend=True)

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
        )


__all__ = ["rotate_z", "plotly_fig2array", "rotary_fig"]
