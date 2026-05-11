"""Runtime helpers for control GUI entrypoints and experiment bootstrap."""

from __future__ import annotations

import multiprocessing
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyccapt.control.core import read_files, share_variables
from pyccapt.control.core.shared_ring_buffer import SharedRingBuffer

# Each plot ring buffer holds 1 M float32 samples = 4 MB.  At 1 kHz ion
# rate that's ~17 minutes of buffered history, far more than the
# visualization process ever falls behind.  Bump if needed.
_PLOT_BUFFER_CAPACITY = 1 << 20  # 1 048 576


@dataclass
class SharedContext:
    """Container for manager-backed shared objects used by control processes."""

    manager: Any
    namespace: Any
    variables: share_variables.Variables
    # Plot data: SPSC ring buffers in shared memory.  Producer is the TDC
    # subprocess (one writer per buffer); consumer is the visualization
    # subprocess (one reader per buffer).
    x_plot: SharedRingBuffer
    y_plot: SharedRingBuffer
    t_plot: SharedRingBuffer
    main_v_dc_plot: SharedRingBuffer


def find_project_root(start: Path | None = None) -> Path:
    """Locate the project root containing control config.

    Args:
        start: Optional search start path. Defaults to this module location.

    Returns:
        Absolute project root path.

    Raises:
        FileNotFoundError: If no suitable root can be found.
    """
    start_path = (start or Path(__file__)).resolve()
    if start_path.is_file():
        start_path = start_path.parent

    for candidate in (start_path, *start_path.parents):
        for root in (candidate, candidate / "pyccapt"):
            if (root / "config.toml").exists() and (root / "control").exists():
                return root

    raise FileNotFoundError(
        f"Could not locate project root from {start_path}. Expected config.toml."
    )


def project_path(*parts: str, root: Path | None = None) -> Path:
    """Build a path under the project root."""
    base = root or find_project_root()
    return base.joinpath(*parts)


def load_project_config(
    config_name: str | None = None,
    *,
    root: Path | None = None,
    change_cwd: bool = True,
) -> tuple[dict[str, Any], Path]:
    """Load control configuration and optionally change CWD to project root.

    Priority:
    1) Explicit `config_name` if provided
    2) `config.toml`
    """
    project_root = root or find_project_root()

    if config_name is not None:
        config_path = project_root / config_name
        if config_path.suffix.lower() != ".toml":
            raise ValueError(
                f"Unsupported config file {config_path.name}. Control config must be TOML (config.toml)."
            )
    else:
        toml_path = project_root / "config.toml"
        if toml_path.exists():
            config_path = toml_path
        else:
            raise FileNotFoundError(
                f"No config file found in {project_root}. Expected config.toml."
            )

    conf = read_files.load_config_file(config_path)
    if change_cwd:
        os.chdir(project_root)
    return conf, project_root


def create_shared_context(conf: dict[str, Any]) -> SharedContext:
	"""Create manager namespace, shared variables, and plot ring buffers.

	The ring buffers replace the old multiprocessing.Queue plot pipeline:
	bounded memory (4 MB / signal), zero-copy reads, no pickling per
	chunk.  Owner is the parent process; child processes attach to the
	same shared blocks by name.
	"""
    manager = multiprocessing.Manager()
    namespace = manager.Namespace()
    variables = share_variables.Variables(conf, namespace)

	# Unique per-launch suffix so we never collide with a stale block
	# left behind by a crashed previous run.
	suffix = uuid.uuid4().hex[:8]
	x_plot = SharedRingBuffer.create(f"pyccapt_xplot_{suffix}",
	                                 _PLOT_BUFFER_CAPACITY, "float32")
	y_plot = SharedRingBuffer.create(f"pyccapt_yplot_{suffix}",
	                                 _PLOT_BUFFER_CAPACITY, "float32")
	t_plot = SharedRingBuffer.create(f"pyccapt_tplot_{suffix}",
	                                 _PLOT_BUFFER_CAPACITY, "float32")
	main_v_dc_plot = SharedRingBuffer.create(f"pyccapt_vplot_{suffix}",
	                                         _PLOT_BUFFER_CAPACITY, "float32")

    return SharedContext(
        manager=manager,
        namespace=namespace,
        variables=variables,
	    x_plot=x_plot,
	    y_plot=y_plot,
	    t_plot=t_plot,
	    main_v_dc_plot=main_v_dc_plot,
    )


def release_shared_context(context: SharedContext) -> None:
	"""Tear down ring buffers and the manager.  Call on full shutdown."""
	for buf in (context.x_plot, context.y_plot, context.t_plot,
	            context.main_v_dc_plot):
		try:
			buf.unlink()
		except Exception:
			pass
	try:
		context.manager.shutdown()
	except Exception:
		pass


def ensure_counter_file(root: Path | None = None) -> Path:
    """Ensure the experiment counter file exists and return its path."""
    counter_path = project_path("files", "counter_experiments.txt", root=root)
    if not counter_path.exists():
        counter_path.write_text("1", encoding="utf-8")
    return counter_path

