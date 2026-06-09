"""Application and per-experiment logging for the pyccapt control package.

There are two layers:

1. Application logging — configured once at GUI startup via
   :func:`setup_application_logging`. This installs:

   - A rotating per-day file handler under
     ``<project_root>/files/logs/gui/gui_<YYYY-MM-DD>.log``.
     Files rotate at 5 MiB, with 10 backups kept. The intent is that this log
     captures everything the operator needs to diagnose what happened in a
     given day, without growing without bound.
   - A console handler that mirrors records to stderr at INFO level (so the
     operator still sees the same kind of output that ``print`` was producing
     before).
   - ``logging.captureWarnings(True)`` so Python warnings land in the log.
   - A ``sys.excepthook`` replacement so otherwise-uncaught exceptions are
     written to the log with a full traceback before propagating.
   - An optional ``sys.stdout`` / ``sys.stderr`` mirror: every ``print`` call
     in the GUI process is duplicated into the log without losing console
     output.

2. Per-experiment logging — :func:`logger_creator` returns a logger whose
   own dedicated file lives in the experiment's ``meta_data`` folder
   (``<exp>/meta_data/apt.log`` by default). Because the per-experiment
   logger inherits from the root logger configured in step 1, every
   experiment record is also captured in the daily GUI log, giving a single
   chronological view across multiple experiments while keeping each
   experiment's own folder self-contained for archival.

Both functions are idempotent: calling them multiple times will not produce
duplicate handlers or duplicate log lines.
"""

from __future__ import annotations

import logging
import logging.handlers
import threading
import os
import platform
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# A verbose default format. Filename + line number are very valuable when
# diagnosing issues from a log file alone.
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-22s | %(filename)s:%(lineno)d | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_DEFAULT_LOG_SUBPATH = ("files", "logs", "gui")
_MAX_BYTES = 5 * 1024 * 1024  # 5 MiB
_BACKUP_COUNT = 10

# Module-level guards so re-imports / repeated init do not cause duplicate
# handlers or repeated startup banners.
_app_logging_initialised = False
_excepthook_installed = False
_stdio_mirrored = False


def _build_formatter() -> logging.Formatter:
    return logging.Formatter(_LOG_FORMAT, _DATE_FORMAT)


def _resolve_gui_log_dir(project_root: os.PathLike[str] | str) -> Path:
    """Return the GUI log directory, creating it if missing."""
    log_dir = Path(project_root).joinpath(*_DEFAULT_LOG_SUBPATH)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _has_handler(logger: logging.Logger, predicate) -> bool:
    return any(predicate(h) for h in logger.handlers)


def setup_application_logging(
    project_root: os.PathLike[str] | str,
    *,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    mirror_stdio: bool = True,
) -> logging.Logger:
    """Configure the root logger for the running pyccapt process.

    Args:
            project_root: Project root directory (the folder that contains
                    ``config.toml`` and ``files/``). The GUI log file is created under
                    ``<project_root>/files/logs/gui/``.
            console_level: Verbosity for the console mirror. The file always
                    records ``file_level`` and below regardless of console level.
            file_level: Verbosity for the rotating file. Defaults to DEBUG so the
                    log file always contains the most detail available.
            mirror_stdio: When True (the default), ``print`` output is also
                    recorded in the log. Console output is preserved.

    Returns:
            The ``pyccapt.gui`` logger. Returning a child logger here, rather than
            the root logger, gives downstream code a sensible default name for
            records that do not specify their own logger.
    """
    global _app_logging_initialised, _excepthook_installed, _stdio_mirrored

    try:
        log_dir = _resolve_gui_log_dir(project_root)
    except Exception as exc:
        # Last-ditch fallback: a stream-only configuration is still better
        # than crashing during startup.
        logging.basicConfig(level=console_level, format=_LOG_FORMAT, datefmt=_DATE_FORMAT)
        logging.getLogger("pyccapt").error(
            "Could not create GUI log directory under %r: %s. Continuing with console-only logging.",
            project_root,
            exc,
        )
        return logging.getLogger("pyccapt.gui")

    log_file = log_dir / f"gui_{datetime.now().strftime('%Y-%m-%d')}.log"

    root = logging.getLogger()
    # Let handlers do the level filtering. Setting the root to DEBUG ensures
    # that records can reach the file handler without being filtered out
    # upstream.
    if root.level == logging.WARNING or root.level == 0:
        root.setLevel(logging.DEBUG)

    formatter = _build_formatter()

    # Rotating file handler — only attach if not already attached for this path.
    target = str(log_file)

    def _is_our_file_handler(h: logging.Handler) -> bool:
        return isinstance(h, logging.handlers.RotatingFileHandler) and getattr(h, "baseFilename", "") == target

    if not _has_handler(root, _is_our_file_handler):
        try:
            fh = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=_MAX_BYTES,
                backupCount=_BACKUP_COUNT,
                encoding="utf-8",
            )
            fh.setLevel(file_level)
            fh.setFormatter(formatter)
            root.addHandler(fh)
        except Exception as exc:
            # Console-only fallback rather than crashing the GUI.
            print(f"[loggi] Could not open log file {log_file}: {exc}", file=sys.__stderr__ or sys.stderr)

    # Console handler — at most one. We deliberately use sys.__stderr__ so
    # that subsequent stdio mirroring (below) does not feed records back into
    # the logger, which would otherwise cause unbounded recursion.
    def _is_our_console_handler(h: logging.Handler) -> bool:
        return (
            isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
            and getattr(h, "_pyccapt_console", False)
        )

    if not _has_handler(root, _is_our_console_handler):
        console_stream = sys.__stderr__ or sys.stderr
        sh = logging.StreamHandler(console_stream)
        sh.setLevel(console_level)
        sh.setFormatter(formatter)
        # Mark so we can detect it on subsequent calls.
        setattr(sh, "_pyccapt_console", True)
        root.addHandler(sh)

    # One-time wiring — captureWarnings, excepthook, stdio mirror.
    if not _excepthook_installed:
        _install_excepthook()
        _excepthook_installed = True
    logging.captureWarnings(True)
    if mirror_stdio and not _stdio_mirrored:
        _mirror_stdio()
        _stdio_mirrored = True

    gui_logger = logging.getLogger("pyccapt.gui")

    if not _app_logging_initialised:
        _log_session_banner(gui_logger, project_root, log_file)
        _app_logging_initialised = True
    else:
        gui_logger.debug("Application logging re-initialised (no-op).")

    return gui_logger


def _install_excepthook() -> None:
    """Make uncaught exceptions land in the log with a full traceback."""
    prev_hook = sys.excepthook

    def _hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            # Preserve normal Ctrl-C behaviour.
            prev_hook(exc_type, exc, tb)
            return
        logging.getLogger("pyccapt").critical(
            "Uncaught exception",
            exc_info=(exc_type, exc, tb),
        )
        prev_hook(exc_type, exc, tb)

    sys.excepthook = _hook


class _StreamToLogger:
    """File-like proxy that mirrors writes to a logger and an original stream.

    Buffer until newline so multi-line ``print`` output is grouped per line
    in the log, instead of producing one record per write fragment.
    """

    def __init__(self, logger: logging.Logger, level: int, original) -> None:
        self._logger = logger
        self._level = level
        self._original = original
        self._buffer = ""
        # ``sys.stdout`` is shared across all threads in the process
        # (TDC drain, GUI, pump polling, baking, ...), so concurrent
        # writes through this wrapper race on ``self._buffer +=`` and
        # ``split``. Python's logging module is thread-safe at the
        # logger.log() boundary, but the buffering layer here is not.
        # Guard with a lock.
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        if self._original is not None:
            try:
                self._original.write(text)
            except Exception:
                pass
        if not text:
            return 0
        with self._lock:
            self._buffer += text
            lines_to_log = []
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                line = line.rstrip("\r")
                if line:
                    lines_to_log.append(line)
        # Log outside the lock to avoid holding it across the
        # potentially-slow logger.log call (which can itself acquire
        # other locks inside logging).
        for line in lines_to_log:
            self._logger.log(self._level, line)
        return len(text)

    def flush(self) -> None:
        if self._original is not None:
            try:
                self._original.flush()
            except Exception:
                pass
        with self._lock:
            tail = self._buffer.strip()
            self._buffer = ""
        if tail:
            self._logger.log(self._level, tail)

    def isatty(self) -> bool:
        try:
            return bool(self._original and self._original.isatty())
        except Exception:
            return False

    def fileno(self) -> int:
        if self._original is None or not hasattr(self._original, "fileno"):
            raise OSError("no fileno")
        return self._original.fileno()


def _mirror_stdio() -> None:
    """Re-route stdout/stderr through loggers without losing console output."""
    if not isinstance(sys.stdout, _StreamToLogger):
        sys.stdout = _StreamToLogger(
            logging.getLogger("pyccapt.stdout"),
            logging.INFO,
            sys.__stdout__,
        )
    if not isinstance(sys.stderr, _StreamToLogger):
        sys.stderr = _StreamToLogger(
            logging.getLogger("pyccapt.stderr"),
            logging.ERROR,
            sys.__stderr__,
        )


def _try_get_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("pyccapt")
        except PackageNotFoundError:
            return "unknown"
    except Exception:
        return "unknown"


def _enumerate_serial_ports() -> list[str]:
    """Return detected serial port names; never raises."""
    try:
        import serial.tools.list_ports

        return sorted(port.device for port in serial.tools.list_ports.comports() if getattr(port, "device", ""))
    except Exception:
        return []


def _log_session_banner(
    logger: logging.Logger,
    project_root: os.PathLike[str] | str,
    log_file: Path,
) -> None:
    """Write a one-time block of environment metadata to the log."""
    ports = _enumerate_serial_ports()
    logger.info("=" * 88)
    logger.info("pyccapt control session start")
    logger.info("  version       : %s", _try_get_version())
    logger.info("  project root  : %s", project_root)
    logger.info("  log file      : %s", log_file)
    logger.info("  python        : %s", sys.version.split()[0])
    logger.info("  platform      : %s %s", platform.system(), platform.release())
    logger.info("  host          : %s", socket.gethostname())
    logger.info(
        "  user          : %s",
        os.environ.get("USERNAME") or os.environ.get("USER") or "unknown",
    )
    logger.info("  pid           : %s", os.getpid())
    logger.info(
        "  com ports     : %s",
        ", ".join(ports) if ports else "none detected",
    )
    logger.info("=" * 88)


def log_configuration_snapshot(
    logger: logging.Logger,
    conf: dict[str, Any],
    variables: Any | None = None,
) -> None:
    """Log a stable subset of config keys so we can compare runs after the fact.

    This is informational only and never raises.
    """
    device_keys = (
        "v_dc",
        "v_p",
        "tdc",
        "cryo",
        "laser",
        "signal_generator",
        "pump_ll",
        "pump_cll",
        "gauges",
    )
    port_keys = (
        "COM_PORT_V_dc",
        "COM_PORT_V_p",
        "COM_PORT_cryo",
        "COM_PORT_laser",
        "COM_PORT_gauge_mc",
        "COM_PORT_gauge_bc",
        "COM_PORT_gauge_ll",
        "COM_PORT_gauge_cll",
        "COM_PORT_signal_generator",
    )
    try:
        logger.info("Device toggles in config.toml:")
        for key in device_keys:
            logger.info("  %-20s = %s", key, conf.get(key, "(not set)"))
        logger.info("Configured port assignments:")
        for key in port_keys:
            logger.info("  %-28s = %s", key, conf.get(key, "(not set)"))
        if variables is not None:
            pulse_mode = getattr(variables, "pulse_mode", None)
            if pulse_mode is not None:
                logger.info("Active pulse_mode: %s", pulse_mode)
    except Exception:
        # Logging must never crash startup. Swallow.
        logger.debug("log_configuration_snapshot failed", exc_info=True)


def logger_creator(
    script_name: str,
    variables: Any,
    log_name: str,
    path: os.PathLike[str] | str,
) -> logging.Logger:
    """Create or fetch a per-experiment logger writing to ``<path>/<log_name>``.

    This signature matches the historical API and is used by
    ``apt_exp_control.py`` to attach a dedicated log file to each experiment's
    ``meta_data`` folder.

    Behaviours:

    - Inherits from the root logger, so the same records also appear in the
      GUI session log configured by :func:`setup_application_logging`.
    - Has its own dedicated file handler in the experiment folder.
    - Is idempotent: a duplicate file handler is not created on repeat calls.
    - Always returns a working logger object. If the file handler cannot be
      attached (permissions, missing folder, etc.) the returned logger still
      accepts ``.info`` / ``.warning`` / ``.exception`` calls — they just
      won't be persisted to the experiment file. This avoids cascading
      ``AttributeError: 'NoneType' object has no attribute 'info'`` failures
      later in the experiment hot path.

    Args:
            script_name: Logger name. Conventional value is ``"apt"``.
            variables: Shared variables namespace (currently unused, kept for
                    backwards compatibility with the previous signature).
            log_name: File name (e.g. ``"apt.log"``).
            path: Directory in which to create the file. Created if missing.

    Returns:
            A configured ``logging.Logger``.
    """
    del variables  # accepted for backwards compatibility; not used here

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.propagate = True  # let root handlers see these too

    try:
        log_dir = Path(path)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / log_name
    except Exception as exc:
        logging.getLogger("pyccapt").warning(
            "Could not prepare experiment log directory %r: %s — per-experiment file logging is disabled for this run.",
            path,
            exc,
        )
        return logger

    target = str(log_path)

    def _is_our_handler(h: logging.Handler) -> bool:
        return isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == target

    if _has_handler(logger, _is_our_handler):
        # Already wired up — just hand back the existing logger without
        # re-emitting an activation banner.
        return logger

    try:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(_build_formatter())
        logger.addHandler(fh)
    except Exception as exc:
        logging.getLogger("pyccapt").warning(
            "Could not attach experiment log file %s: %s",
            log_path,
            exc,
        )
        return logger

    logger.info("-" * 60)
    logger.info("Experiment logger '%s' active. File: %s", script_name, log_path)
    logger.info("-" * 60)
    return logger
