"""Control subpackage."""

from __future__ import annotations

import sys

from . import core

# Backward-compatible import alias:
#   pyccapt.control.control -> pyccapt.control.core
sys.modules.setdefault(__name__ + ".control", core)

__all__ = ["core"]
