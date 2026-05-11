"""Tests for ``pyccapt.calibration.core.parallel``.

The helper has three failure modes that matter:

- It must produce *identical* results across the serial / thread / process
  backends (otherwise calibration outputs would silently depend on the
  worker count).
- It must honor ``PYCCAPT_PARALLEL_WORKERS=1`` so CI runs deterministically.
- It must degrade gracefully when ``fn`` isn't picklable (the common
  Windows / Jupyter trap when a closure leaks into a process pool).
"""

from __future__ import annotations

import os

import pytest

from pyccapt.calibration.core import parallel


def _square(x: int) -> int:
    """Module-level worker -- picklable for the process backend."""
    return x * x


def test_parallel_map_serial_matches_python():
    result = parallel.parallel_map(
        _square,
        range(100),
        config=parallel.ParallelConfig(backend="serial"),
    )
    assert result == [i * i for i in range(100)]


def test_parallel_map_thread_matches_serial():
    items = list(range(500))
    thread = parallel.parallel_map(_square, items, config=parallel.ParallelConfig(backend="thread", workers=4))
    serial = parallel.parallel_map(_square, items, config=parallel.ParallelConfig(backend="serial"))
    assert thread == serial


def test_parallel_map_process_matches_serial():
    items = list(range(parallel.DEFAULT_PROCESS_MIN_ITEMS + 50))
    process = parallel.parallel_map(
        _square,
        items,
        config=parallel.ParallelConfig(backend="process", workers=2),
        gil_releasing=False,
    )
    serial = parallel.parallel_map(_square, items, config=parallel.ParallelConfig(backend="serial"))
    assert process == serial


def test_parallel_map_empty_input():
    assert parallel.parallel_map(_square, []) == []


def test_below_threshold_runs_serial(monkeypatch):
    # Far below the auto thread threshold of 32 -> serial.
    choice = parallel.describe_runtime_choice(parallel.ParallelConfig(), n_items=4, gil_releasing=True)
    assert choice["backend"] == "serial"


def test_env_var_workers_one_forces_serial(monkeypatch):
    monkeypatch.setenv(parallel.WORKERS_ENV_VAR, "1")
    choice = parallel.describe_runtime_choice(parallel.ParallelConfig(), n_items=10_000, gil_releasing=True)
    assert choice["backend"] == "serial"


def test_env_var_backend_override(monkeypatch):
    monkeypatch.delenv(parallel.WORKERS_ENV_VAR, raising=False)
    monkeypatch.setenv(parallel.BACKEND_ENV_VAR, "thread")
    choice = parallel.describe_runtime_choice(parallel.ParallelConfig(), n_items=10_000, gil_releasing=False)
    assert choice["backend"] == "thread"


def test_env_var_workers_zero_treated_as_serial(monkeypatch):
    monkeypatch.setenv(parallel.WORKERS_ENV_VAR, "0")
    out = parallel.parallel_map(_square, range(10_000))
    assert out[:5] == [0, 1, 4, 9, 16]


def test_unpicklable_closure_degrades_to_thread_under_auto():
    """A closure can't be pickled; auto must silently downgrade to threads."""
    multiplier = 3

    def times_n(x):
        return x * multiplier

    out = parallel.parallel_map(
        times_n,
        list(range(parallel.DEFAULT_PROCESS_MIN_ITEMS + 50)),
        config=parallel.ParallelConfig(backend="auto"),
        gil_releasing=False,  # would have asked for the process backend
    )
    expected = [i * multiplier for i in range(parallel.DEFAULT_PROCESS_MIN_ITEMS + 50)]
    assert out == expected


def test_explicit_process_backend_with_unpicklable_raises(monkeypatch):
    """Explicit process backend should *not* silently fall back -- the
    caller wants to know their fn isn't actually parallelizable."""
    monkeypatch.delenv(parallel.WORKERS_ENV_VAR, raising=False)
    monkeypatch.delenv(parallel.BACKEND_ENV_VAR, raising=False)
    multiplier = 5

    def times_n(x):
        return x * multiplier

    with pytest.raises(Exception):
        parallel.parallel_map(
            times_n,
            list(range(parallel.DEFAULT_PROCESS_MIN_ITEMS + 50)),
            config=parallel.ParallelConfig(backend="process", workers=2),
            gil_releasing=False,
        )


def test_describe_runtime_choice_reports_env(monkeypatch):
    monkeypatch.setenv(parallel.WORKERS_ENV_VAR, "2")
    monkeypatch.setenv(parallel.BACKEND_ENV_VAR, "thread")
    choice = parallel.describe_runtime_choice(None, n_items=100, gil_releasing=True)
    assert choice["workers"] == 2
    assert choice["backend"] == "thread"
    assert choice["workers_env"] == "2"
    assert choice["backend_env"] == "thread"
