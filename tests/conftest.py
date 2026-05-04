"""Shared pytest configuration for domain-separated test suites."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

# Force matplotlib to a non-interactive backend BEFORE any test (or any module
# they import) calls `matplotlib.pyplot`. Without this, helpers that internally
# call `plt.show()` open real windows during pytest and the user has to close
# them manually. Setting MPLBACKEND in the environment is the most reliable
# way: it takes effect even for matplotlib imports that happen inside imported
# project modules before the test body runs.
os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as _plt  # noqa: F401  (force backend init)

    _plt.ioff()
except Exception:
    # If matplotlib isn't available the project tests can't run anyway.
    pass

import pytest


CONTROL_SENTINEL_MODULES = ("serial", "pyvisa", "simple_pid", "PyQt6")
CALIBRATION_SENTINEL_MODULES = ("adjustText", "pybaselines")


def _to_path(pathlike: object) -> Path:
    return Path(str(pathlike))


def _path_has_group(pathlike: object, group: str) -> bool:
    parts = _to_path(pathlike).parts
    return "tests" in parts and group in parts


def _missing_modules(module_names: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for module_name in module_names:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    return missing


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("pyccapt")
    group.addoption(
        "--run-control",
        action="store_true",
        default=False,
        help="Run only tests under tests/control.",
    )
    group.addoption(
        "--run-calibration",
        action="store_true",
        default=False,
        help="Run only tests under tests/calibration.",
    )


def pytest_ignore_collect(collection_path: object, config: pytest.Config) -> bool:
    run_control = config.getoption("--run-control")
    run_calibration = config.getoption("--run-calibration")

    if _path_has_group(collection_path, "control"):
        if run_calibration and not run_control:
            return True
        if run_control:
            return False
        return bool(_missing_modules(CONTROL_SENTINEL_MODULES))

    if _path_has_group(collection_path, "calibration"):
        if run_control and not run_calibration:
            return True
        if run_calibration:
            return False
        return bool(_missing_modules(CALIBRATION_SENTINEL_MODULES))

    return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        path = _to_path(item.path)
        if _path_has_group(path, "control"):
            item.add_marker(pytest.mark.control)
        elif _path_has_group(path, "calibration"):
            item.add_marker(pytest.mark.calibration)
