"""Synthetic mass-spectrum generation utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def gaussian_peak(x: np.ndarray, center: float, sigma: float, amplitude: float) -> np.ndarray:
    """Return a Gaussian peak profile."""
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError("sigma must be greater than 0")
    return float(amplitude) * np.exp(-0.5 * ((x - float(center)) / sigma) ** 2)


def exponential_background(x: np.ndarray, amplitude: float = 0.0, decay: float = 0.0) -> np.ndarray:
    """Return a simple exponential background profile."""
    amplitude = float(amplitude)
    decay = float(decay)
    if amplitude <= 0 or decay <= 0:
        return np.zeros_like(x, dtype=float)
    shifted = x - np.min(x)
    return amplitude * np.exp(-decay * shifted)


def simulate_mass_spectrum(
    peaks: Iterable[dict],
    x_min: float | None = None,
    x_max: float | None = None,
    bin_width: float = 0.01,
    background_amplitude: float = 0.0,
    background_decay: float = 0.0,
    poisson_noise: bool = True,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Simulate a 1D mass spectrum from Gaussian peaks plus optional exponential background.

    Each peak dictionary may contain:
    - ``mass`` or ``center``: peak position in Da
    - ``sigma``: peak width in Da
    - ``intensity`` or ``amplitude``: peak amplitude
    """
    peak_list = list(peaks)
    if not peak_list:
        raise ValueError("At least one peak must be provided")
    if bin_width <= 0:
        raise ValueError("bin_width must be greater than 0")

    centers = np.array([float(peak.get("mass", peak.get("center"))) for peak in peak_list], dtype=float)
    sigmas = np.array([float(peak.get("sigma", 0.02)) for peak in peak_list], dtype=float)
    amplitudes = np.array(
        [float(peak.get("intensity", peak.get("amplitude", 1.0))) for peak in peak_list],
        dtype=float,
    )
    if np.any(sigmas <= 0):
        raise ValueError("All peak sigmas must be greater than 0")
    if np.any(amplitudes < 0):
        raise ValueError("Peak amplitudes must be non-negative")

    lower = float(np.min(centers - 6.0 * sigmas)) if x_min is None else float(x_min)
    upper = float(np.max(centers + 6.0 * sigmas)) if x_max is None else float(x_max)
    if upper <= lower:
        raise ValueError("x_max must be greater than x_min")

    x = np.arange(lower, upper + bin_width, float(bin_width), dtype=float)
    signal = np.zeros_like(x)
    for center, sigma, amplitude in zip(centers, sigmas, amplitudes):
        signal += gaussian_peak(x, center, sigma, amplitude)
    background = exponential_background(x, amplitude=background_amplitude, decay=background_decay)
    expected = np.clip(signal + background, 0.0, None)

    if poisson_noise:
        rng = np.random.default_rng(seed)
        counts = rng.poisson(expected)
    else:
        counts = expected

    return pd.DataFrame(
        {
            "mc (Da)": x,
            "signal": signal,
            "background": background,
            "expected_counts": expected,
            "counts": counts,
        }
    )
