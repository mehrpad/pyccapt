"""Cross-platform filesystem helpers used by calibration modules."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


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


def save_figure(
    figure,
    *,
    directory: str | Path,
    stem: str,
    formats: Iterable[str] = ("png", "svg"),
    dpi: int = 600,
    **savefig_kwargs,
) -> list[Path]:
    """Save a matplotlib figure in one or more formats and return output paths."""
    if not stem or not stem.strip():
        raise ValueError("stem must be a non-empty string")

    output_paths: list[Path] = []
    for extension in formats:
        ext = str(extension).lstrip(".")
        output_path = build_output_path(directory, f"{stem}.{ext}")
        figure.savefig(output_path, format=ext, dpi=dpi, **savefig_kwargs)
        output_paths.append(output_path)
    return output_paths
