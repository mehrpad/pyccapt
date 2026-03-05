"""File-loading helpers for the control package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for older Python
    import tomli as tomllib  # type: ignore[no-redef]


def read_json_file(json_file_path: str | Path) -> dict[str, Any]:
    """Read a JSON file and return its content as a dictionary.

    Args:
        json_file_path: Absolute or relative path to the JSON file.

    Returns:
        Parsed JSON content.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file does not contain a JSON object.
        json.JSONDecodeError: If the file content is invalid JSON.
    """
    path = Path(json_file_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(data).__name__}.")

    return data


def read_toml_file(toml_file_path: str | Path) -> dict[str, Any]:
    """Read a TOML file and return its content as a dictionary.

    Args:
        toml_file_path: Absolute or relative path to the TOML file.

    Returns:
        Parsed TOML content.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file does not contain a TOML object.
        tomllib.TOMLDecodeError: If the file content is invalid TOML.
    """
    path = Path(toml_file_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"TOML file not found: {path}")

    with path.open("rb") as file_handle:
        data = tomllib.load(file_handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a TOML object in {path}, got {type(data).__name__}.")

    return data


def load_config_file(config_file_path: str | Path) -> dict[str, Any]:
    """Load a control configuration file from `.toml`.

    Args:
        config_file_path: Absolute or relative path to the config file.

    Returns:
        Parsed config dictionary.

    Raises:
        ValueError: If file extension is unsupported.
    """
    path = Path(config_file_path).expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix == ".toml":
        return read_toml_file(path)
    if suffix == ".json":
        raise ValueError(
            f"JSON config is no longer supported for control runtime: {path}. "
            "Please migrate to config.toml."
        )
    raise ValueError(f"Unsupported config format for {path}. Use a .toml configuration file.")
