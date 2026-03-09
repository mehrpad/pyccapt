"""Shared I/O helpers for reconstruction plotting outputs."""

from __future__ import annotations

from pathlib import Path

import imageio
import plotly
import plotly.io as pio

from pyccapt.calibration.path_utils import build_output_path, save_figure


def resolve_result_file(variables, filename: str) -> str:
    """Resolve an output filename against the reconstruction result directory."""
    resolver = getattr(variables, "resolve_result_file", None)
    if callable(resolver):
        return resolver(filename)

    result_path = str(getattr(variables, "result_path", "")).strip()
    if result_path:
        return str(build_output_path(result_path, filename))
    return str(Path(filename))


def save_matplotlib_figure(
    fig,
    variables,
    *,
    stem: str,
    formats: tuple[str, ...] = ("png", "svg"),
    dpi: int = 600,
    **savefig_kwargs,
) -> list[Path]:
    """Save a matplotlib figure to the reconstruction result directory."""
    output_dir = getattr(variables, "result_path", "") or "."
    return save_figure(fig, directory=output_dir, stem=stem, formats=formats, dpi=dpi, **savefig_kwargs)


def write_plotly_html(fig, variables, filename: str, *, include_mathjax: str = "cdn") -> None:
    """Write a Plotly figure to HTML inside the reconstruction result directory."""
    pio.write_html(fig, resolve_result_file(variables, filename), include_mathjax=include_mathjax)


def write_plotly_image(
    fig,
    variables,
    filename: str,
    *,
    scale: int = 3,
    image_format: str | None = None,
) -> None:
    """Write a Plotly static image inside the reconstruction result directory."""
    resolved = resolve_result_file(variables, filename)
    if image_format is None:
        image_format = Path(filename).suffix.lstrip(".")
    pio.write_image(fig, resolved, scale=scale, format=image_format)


def save_gif(images, variables, filename: str, *, fps: int = 2) -> None:
    """Save a sequence of frames to GIF inside the reconstruction result directory."""
    imageio.mimsave(resolve_result_file(variables, filename), images, fps=fps)


def save_plotly_animation(
    fig,
    variables,
    *,
    filename: str,
    show_link: bool = True,
    auto_open: bool = False,
    include_mathjax: str = "cdn",
) -> None:
    """Save an interactive Plotly animation HTML file to the result directory."""
    plotly.offline.plot(
        fig,
        filename=resolve_result_file(variables, filename),
        show_link=show_link,
        auto_open=auto_open,
        include_mathjax=include_mathjax,
    )
