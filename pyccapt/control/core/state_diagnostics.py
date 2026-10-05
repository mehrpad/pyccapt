"""Read-only state diagnostics and final dataset snapshots."""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import time
from pathlib import Path


def collect_control_states(variables, now=None) -> dict:
    getter = getattr(variables, "control_states", None)
    if not callable(getter):
        return {}
    try:
        states = getter()
        # The snapshot may contain an update made while the Manager read was
        # in flight. Sample the clock afterwards so a fresh update is not
        # misclassified as a future/stale observation.
        checked_at = time.monotonic() if now is None else now
        return {name: state.to_dict(checked_at) for name, state in states.items()}
    except Exception:
        logging.getLogger("pyccapt.state").exception("Could not collect control state diagnostics")
        return {}


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def save_control_states(variables, metadata_directory) -> Path | None:
    """Atomic JSON snapshot in an already-created dataset, never another run.

    Snapshot/logging failures cannot prevent safe-off or completion reporting.
    This is a final snapshot, not a replacement for alignment event journals.
    """
    if metadata_directory is None:
        return None
    directory = Path(metadata_directory)
    if not directory.is_dir():
        return None
    snapshot = collect_control_states(variables)
    if not snapshot:
        return None
    target = directory / "control_states.json"
    temporary = directory / f".control_states.{os.getpid()}.tmp"
    try:
        record = {"schema_version": 1, "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                  "states": snapshot,
                  "notes": "Commanded output/gate states are not independent physical confirmation. "
                           "Monotonic timestamps belong to this computer's runtime clock."}
        temporary.write_text(json.dumps(_json_safe(record), indent=2, allow_nan=False, default=str)+"\n",
                             encoding="utf-8")
        os.replace(temporary, target)
        return target
    except Exception:
        logging.getLogger("pyccapt.state").exception("Could not save final control states to %s", target)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        return None
