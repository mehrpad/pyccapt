"""Calibration subpackage for data preparation, calibration, and reconstruction."""

from __future__ import annotations

import sys

from . import core

# Backward-compatible import alias:
#   pyccapt.calibration.calibration -> pyccapt.calibration.core
sys.modules.setdefault(__name__ + ".calibration", core)

__all__ = ["core"]
