"""Shared background-fitting helpers for histogram modules."""

from __future__ import annotations

import numpy as np


def fit_background(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Return an inverse-sqrt background fit for x."""
    return (a / (2 * np.sqrt(b))) * 1 / np.sqrt(x)
