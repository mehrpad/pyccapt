"""Notebook-safe helpers for the existing Qt file picker."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def resolve_initial_directory(current_value: str | None = None, fallback_directory: str | None = None) -> str:
    """Return the best directory to open in the Qt file dialog."""
    for candidate in (current_value, fallback_directory):
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file():
            return str(path.parent)
        if path.is_dir():
            return str(path)
    return str(Path.cwd())


def choose_file_path(initial_directory: str | None = None, file_kind: str = "any") -> str | None:
    """Open the existing Qt picker in a subprocess and return the chosen file."""
    start_directory = resolve_initial_directory(initial_directory)
    script_path = Path(__file__).with_name("run_dataset_path_qt.py")
    result = subprocess.run(
        [sys.executable, str(script_path), start_directory, str(file_kind)],
        capture_output=True,
        text=True,
        check=False,
    )
    selected_path = (result.stdout or "").strip()
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(stderr or "The file chooser exited with a non-zero status")
    if not selected_path or selected_path == "No file chosen":
        return None
    return selected_path


def choose_directory_path(initial_directory: str | None = None) -> str | None:
    """Open a Qt directory picker in a subprocess and return the chosen directory."""
    start_directory = resolve_initial_directory(initial_directory)
    script_path = Path(__file__).with_name("run_directory_path_qt.py")
    result = subprocess.run(
        [sys.executable, str(script_path), start_directory],
        capture_output=True,
        text=True,
        check=False,
    )
    selected_path = (result.stdout or "").strip()
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(stderr or "The directory chooser exited with a non-zero status")
    if not selected_path or selected_path == "No directory chosen":
        return None
    return selected_path


__all__ = ["choose_file_path", "choose_directory_path", "resolve_initial_directory"]
