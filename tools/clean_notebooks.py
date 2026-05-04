"""Repo-level CLI wrapper for notebook cleaning."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyccapt.calibration.tutorials.notebook_cleaner import main


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
