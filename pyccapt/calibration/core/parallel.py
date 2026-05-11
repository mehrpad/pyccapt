"""Shared parallelism helper for calibration hot paths.

Calibration steps like bowl correction, voltage correction, and raw-data
analysis loop over many independent work items (cells in a detector grid,
voltage sub-segments, sequence records). The inner kernels are either
NumPy/SciPy (which release the GIL) or pure Python (which doesn't). This
module provides a single :func:`parallel_map` that:

- Picks the right executor automatically (threads for GIL-releasing kernels,
  processes for pure Python).
- Falls back to plain serial execution when the work item count is too small
  for the executor's startup cost to pay off.
- Honors ``PYCCAPT_PARALLEL_WORKERS`` and ``PYCCAPT_PARALLEL_BACKEND``
  environment variables so tests and CI can pin behavior without modifying
  callers.
- Degrades gracefully when ``fn`` / ``items`` aren't picklable -- a common
  trap when callers pass closures into a ProcessPoolExecutor on Windows.

Design choices
--------------

- Backward-compatible default: when called with no config and no env vars on
  a small input, behavior is identical to the previous serial code.
- The caller declares whether the inner work is ``gil_releasing``. NumPy /
  SciPy / scikit-image calls are; pure-Python loops aren't. This is the
  single most important signal for picking the backend correctly.
- Worker count defaults to ``cpu_count - 1`` capped at 8, so a calibration
  run still leaves room for the notebook UI / Jupyter kernel to remain
  responsive on small machines.
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Callable, Iterable, TypeVar


T = TypeVar("T")
R = TypeVar("R")

_VALID_BACKENDS = ("auto", "thread", "process", "serial")

WORKERS_ENV_VAR = "PYCCAPT_PARALLEL_WORKERS"
BACKEND_ENV_VAR = "PYCCAPT_PARALLEL_BACKEND"

# Default fallback thresholds. Below these, parallel_map runs in-process to
# avoid paying the executor's startup overhead for a tiny workload. Threads
# spin up in microseconds, so threading is worthwhile from ~32 items; spawned
# processes on Windows cost ~200 ms each, so process pools only pay off for
# substantially larger work batches.
DEFAULT_THREAD_MIN_ITEMS = 32
DEFAULT_PROCESS_MIN_ITEMS = 200

# Maximum number of workers we'll auto-pick. Higher than this rarely helps
# for NumPy-bound calibration work because each worker reads from shared
# memory anyway and there's diminishing return past ~8 cores. The user can
# always override via ``PYCCAPT_PARALLEL_WORKERS``.
DEFAULT_MAX_AUTO_WORKERS = 8


@dataclass(frozen=True)
class ParallelConfig:
    """Caller-side knobs that env vars can override.

    Attributes:
        workers: Worker count. ``None`` = auto (``cpu_count - 1`` capped at
            :data:`DEFAULT_MAX_AUTO_WORKERS`). ``0`` or ``1`` forces serial.
        backend: One of ``'auto'``, ``'thread'``, ``'process'``, ``'serial'``.
        min_items: Below this many items, run in serial regardless of
            backend (avoids executor startup overhead on tiny workloads).
            ``None`` (the default) means "pick the backend's natural
            threshold": :data:`DEFAULT_THREAD_MIN_ITEMS` for threads,
            :data:`DEFAULT_PROCESS_MIN_ITEMS` for processes. Pass an
            explicit integer to override (callers that batch the work at a
            higher level know better than the generic floor).
    """

    workers: int | None = None
    backend: str = "auto"
    min_items: int | None = None


def _default_worker_count() -> int:
    """Pick a sensible auto-workers count."""
    try:
        n_cpus = os.process_cpu_count()  # py3.13+
    except AttributeError:
        n_cpus = None
    if n_cpus is None:
        try:
            n_cpus = multiprocessing.cpu_count()
        except NotImplementedError:
            n_cpus = 2
    return max(1, min(DEFAULT_MAX_AUTO_WORKERS, n_cpus - 1))


def _resolve_config(config: ParallelConfig | None) -> ParallelConfig:
    """Apply env-var overrides on top of the supplied config (or default)."""
    cfg = config or ParallelConfig()
    overrides: dict = {}
    workers_env = os.environ.get(WORKERS_ENV_VAR)
    if workers_env is not None:
        try:
            workers = int(workers_env)
        except ValueError:
            workers = None
        if workers is not None:
            overrides["workers"] = max(0, workers)
    backend_env = os.environ.get(BACKEND_ENV_VAR)
    if backend_env is not None:
        backend = backend_env.strip().lower()
        if backend in _VALID_BACKENDS:
            overrides["backend"] = backend
    return replace(cfg, **overrides) if overrides else cfg


def _pick_backend(cfg: ParallelConfig, n_items: int, gil_releasing: bool) -> tuple[str, int]:
    """Resolve final backend choice and worker count for one invocation."""
    if cfg.backend == "serial":
        return "serial", 1
    if cfg.workers is not None and cfg.workers <= 1:
        return "serial", 1
    n_workers = cfg.workers if cfg.workers else _default_worker_count()
    if cfg.backend == "auto":
        backend = "thread" if gil_releasing else "process"
    elif cfg.backend in _VALID_BACKENDS:
        backend = cfg.backend
    else:
        backend = "thread"
    if cfg.min_items is not None:
        threshold = cfg.min_items
    elif backend == "thread":
        threshold = DEFAULT_THREAD_MIN_ITEMS
    elif backend == "process":
        threshold = DEFAULT_PROCESS_MIN_ITEMS
    else:
        threshold = DEFAULT_THREAD_MIN_ITEMS
    if n_items < threshold:
        return "serial", 1
    return backend, min(n_workers, max(1, n_items))


def parallel_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    config: ParallelConfig | None = None,
    gil_releasing: bool = True,
    chunksize: int | None = None,
) -> list[R]:
    """Apply ``fn`` to each item, returning results in input order.

    Parameters
    ----------
    fn:
        Callable applied to each item. For the process backend, ``fn`` and
        all items must be picklable (so no closures or local functions).
    items:
        Materialized via ``list(items)`` so the resulting list can be ordered
        deterministically.
    config:
        :class:`ParallelConfig` overrides. ``None`` uses defaults. Env vars
        ``PYCCAPT_PARALLEL_WORKERS`` and ``PYCCAPT_PARALLEL_BACKEND`` override
        on top of whatever ``config`` provides.
    gil_releasing:
        ``True`` (default) when ``fn`` spends its time inside NumPy / SciPy /
        scikit-image / pandas. ``False`` when it's pure Python; that flips
        the auto backend to processes so the GIL can't serialize threads.
    chunksize:
        Only relevant for the process backend. ``None`` = pick automatically.
    """
    items_list = list(items)
    n_items = len(items_list)
    if n_items == 0:
        return []

    cfg = _resolve_config(config)
    backend, n_workers = _pick_backend(cfg, n_items, gil_releasing)

    if backend == "serial":
        return [fn(item) for item in items_list]

    if backend == "thread":
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            return list(ex.map(fn, items_list))

    if chunksize is None:
        chunksize = max(1, n_items // (n_workers * 4))

    try:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            return list(ex.map(fn, items_list, chunksize=chunksize))
    except (pickle.PicklingError, AttributeError, OSError):
        # Common Windows / Jupyter trap: closures or notebook-defined
        # functions can't be pickled. If the caller explicitly asked for the
        # process backend we re-raise; otherwise fall back to threads (which
        # may be slower for pure-Python kernels, but at least returns a
        # correct result).
        if cfg.backend == "process":
            raise
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            return list(ex.map(fn, items_list))


def describe_runtime_choice(cfg: ParallelConfig | None, n_items: int, gil_releasing: bool) -> dict:
    """Return the executor choice ``parallel_map`` would make.

    Useful for logging / debugging in the calibration notebooks.
    """
    resolved = _resolve_config(cfg)
    backend, workers = _pick_backend(resolved, n_items, gil_releasing)
    return {
        "backend": backend,
        "workers": workers,
        "n_items": n_items,
        "gil_releasing": gil_releasing,
        "workers_env": os.environ.get(WORKERS_ENV_VAR),
        "backend_env": os.environ.get(BACKEND_ENV_VAR),
    }


__all__ = [
    "BACKEND_ENV_VAR",
    "DEFAULT_MAX_AUTO_WORKERS",
    "DEFAULT_PROCESS_MIN_ITEMS",
    "DEFAULT_THREAD_MIN_ITEMS",
    "ParallelConfig",
    "WORKERS_ENV_VAR",
    "describe_runtime_choice",
    "parallel_map",
]
