"""Cross-platform filesystem helpers used by calibration modules."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Iterable

import matplotlib.text as mtext
from matplotlib import rcParams


# Font conventions for PDF output: editable TrueType, Arial family, 7pt.
# PNG output is left untouched so on-screen / rasterised plots keep the
# font size chosen by the plotting code.
PDF_FONT_FAMILY = "Arial"
PDF_FONT_SIZE = 7


def ensure_directory(path_value: str | Path) -> Path:
    """Return `path_value` as a created directory path."""
    directory = Path(path_value).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def build_output_path(directory: str | Path, filename: str) -> Path:
    """Build an output file path inside `directory`."""
    if not filename or not str(filename).strip():
        raise ValueError("filename must be a non-empty string")
    return ensure_directory(directory) / filename


@contextlib.contextmanager
def _pdf_font_context(figure):
    """Temporarily restyle every Text artist in *figure* for a PDF save."""
    text_artists = list(figure.findobj(mtext.Text))
    snapshot = [(artist, artist.get_fontproperties().copy()) for artist in text_artists]
    saved_pdf_fonttype = rcParams.get("pdf.fonttype", 3)
    saved_pdf_use14 = rcParams.get("pdf.use14corefonts", False)
    try:
        # 42 = TrueType: text stays editable in the PDF (not flattened to paths).
        rcParams["pdf.fonttype"] = 42
        rcParams["pdf.use14corefonts"] = False
        for artist in text_artists:
            artist.set_fontfamily(PDF_FONT_FAMILY)
            artist.set_fontsize(PDF_FONT_SIZE)
        yield
    finally:
        for artist, props in snapshot:
            artist.set_fontproperties(props)
        rcParams["pdf.fonttype"] = saved_pdf_fonttype
        rcParams["pdf.use14corefonts"] = saved_pdf_use14


def save_figure(
    figure,
    *,
    directory: str | Path,
    stem: str,
    formats: Iterable[str] = ("png", "pdf"),
    dpi: int = 600,
    tidy_axes: bool = True,
    **savefig_kwargs,
) -> list[Path]:
    """Save a matplotlib figure in one or more formats and return output paths.

    PDF outputs are rendered with Arial 7pt fonts (text stays editable via
    TrueType embedding); PNG and other formats use whatever fonts the figure
    was built with.

    When ``tidy_axes`` is True (default), linear plot axes are snapped to
    round-number limits with ticks at both ends (paper styling) before saving;
    map/image, equal-aspect and log axes are left untouched. Pass
    ``tidy_axes=False`` to save the figure exactly as drawn.
    """
    if not stem or not stem.strip():
        raise ValueError("stem must be a non-empty string")

    if tidy_axes:
        try:
            from pyccapt.calibration.core import plot_style
            plot_style.finalize_figure(figure)
        except Exception:
            # Styling is cosmetic and must never prevent a save.
            pass

    output_paths: list[Path] = []
    for extension in formats:
        ext = str(extension).lstrip(".").lower()
        output_path = build_output_path(directory, f"{stem}.{ext}")
        if ext == "pdf":
            with _pdf_font_context(figure):
                figure.savefig(output_path, format=ext, dpi=dpi, **savefig_kwargs)
        else:
            figure.savefig(output_path, format=ext, dpi=dpi, **savefig_kwargs)
        output_paths.append(output_path)
    return output_paths
