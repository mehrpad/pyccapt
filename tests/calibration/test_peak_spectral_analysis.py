import numpy as np
import pandas as pd

from pyccapt.calibration.core.peak_spectral_analysis import (
    estimate_background_ppm,
    fit_local_linear_background,
    fit_peak_deconvolution,
    resolve_deconvolution_components,
)


def test_local_linear_background_recovers_peak_counts():
    rng = np.random.default_rng(12)
    background = rng.uniform(49.0, 51.0, 900)
    peak = rng.normal(50.0, 0.045, 550)
    values = np.concatenate([background, peak])
    result = fit_local_linear_background(
        values,
        bin_width=0.01,
        peak_range=(49.85, 50.15),
        bg_left_range=(49.2, 49.6),
        bg_right_range=(50.4, 50.8),
    )
    assert result["corrected_counts"] > 350
    assert result["background_peak_counts"] >= 0
    assert result["background_ppm_per_da"] > 0


def test_background_ppm_estimation_reports_expected_counts():
    rng = np.random.default_rng(13)
    values = np.concatenate([rng.uniform(3.5, 4.5, 220), rng.uniform(48.0, 52.0, 1000)])
    result = estimate_background_ppm(values, (3.5, 4.5), (49.8, 50.2))
    assert result["background_counts"] == 220
    assert result["ppm_per_da"] > 0
    assert result["expected_background_counts"] > 0
    assert result["expected_background_counts_ci"][1] >= result["expected_background_counts"]


def test_overlap_deconvolution_recovers_two_component_ratio():
    rng = np.random.default_rng(14)
    component_a = rng.normal(52.00, 0.025, 700)
    component_b = rng.normal(52.18, 0.025, 300)
    background = rng.uniform(51.8, 52.35, 160)
    values = np.concatenate([component_a, component_b, background])
    components = pd.DataFrame(
        {
            "label": ["Cr", "Mo"],
            "center": [52.00, 52.18],
            "row_index": [0, 1],
        }
    )
    result = fit_peak_deconvolution(
        values,
        bin_width=0.01,
        window_range=(51.85, 52.30),
        components=components,
        shape="pseudo_voigt",
        background_mode="linear",
        initial_fwhm=0.08,
    )
    grouped = result["grouped_components"].set_index("label")
    ratio = grouped.loc["Cr", "fitted_counts"] / grouped["fitted_counts"].sum()
    assert result["success"]
    assert 0.55 < ratio < 0.85


def test_resolve_deconvolution_components_uses_range_rows_and_manual_tokens():
    range_data = pd.DataFrame(
        {
            "name": ["Cr", "Mo", "Ni"],
            "mc": [52.00, 52.18, 58.70],
            "mc_low": [51.95, 52.12, 58.6],
            "mc_up": [52.05, 52.24, 58.8],
        }
    )
    resolved = resolve_deconvolution_components(range_data, (51.9, 52.25), "Cr, Mo")
    assert set(resolved["label"]) == {"Cr", "Mo"}
    manual = resolve_deconvolution_components(range_data, (51.9, 52.25), "Custom@52.08")
    assert list(manual["label"]) == ["Custom"]
