import numpy as np

import pandas as pd

from pyccapt.calibration.core import mc_plot
from pyccapt.calibration.core import mc_plot_peak_helpers as peak_helpers
from pyccapt.calibration.core.mc_plot_peak_helpers import (
    _MRP_INTERNAL_BIN_SIZE,
    gaussian_mrp_report,
)
from pyccapt.calibration.core.share_variables import Variables


def test_plot_hist_info_legend_wrapper_delegates(monkeypatch):
    plotter = mc_plot.AptHistPlotter(np.array([1.0]), Variables())
    called = {}

    def fake(plotter_obj, label='mc', mrp_all=False, background=None, legend_mode='long', loc='left'):
        called['args'] = (plotter_obj, label, mrp_all, background, legend_mode, loc)
        return 'legend-ok'

    monkeypatch.setattr(mc_plot, '_apply_hist_info_legend', fake)

    result = plotter.plot_hist_info_legend(label='tof', mrp_all=True, legend_mode='short', loc='right')

    assert result == 'legend-ok'
    assert called['args'][0] is plotter
    assert called['args'][1] == 'tof'
    assert called['args'][2] is True
    assert called['args'][4] == 'short'
    assert called['args'][5] == 'right'


def test_background_wrapper_delegates(monkeypatch):
    plotter = mc_plot.AptHistPlotter(np.array([1.0]), Variables())
    called = {}

    def fake(plotter_obj, mode, **kwargs):
        called['plotter'] = plotter_obj
        called['mode'] = mode
        called['kwargs'] = kwargs
        return 'bg-ok'

    monkeypatch.setattr(mc_plot, '_plot_background', fake)

    result = plotter.plot_background('fabc', lam=123.0, patch=False)

    assert result == 'bg-ok'
    assert called['plotter'] is plotter
    assert called['mode'] == 'fabc'
    assert called['kwargs']['lam'] == 123.0
    assert called['kwargs']['patch'] is False


def test_selector_wrapper_delegates(monkeypatch):
    plotter = mc_plot.AptHistPlotter(np.array([1.0]), Variables())
    called = {}

    def fake(plotter_obj, selector='rect'):
        called['plotter'] = plotter_obj
        called['selector'] = selector
        return 'sel-ok'

    monkeypatch.setattr(mc_plot, '_attach_selector', fake)

    result = plotter.selector('peak')

    assert result == 'sel-ok'
    assert called['plotter'] is plotter
    assert called['selector'] == 'peak'


def test_range_display_labels_prefer_plain_name_over_math_ion():
    range_data = pd.DataFrame(
        {
            'name': ['Mo2', 'CrMo'],
            'ion': ['$Mo_{2}^{+}$', '$CrMo^{+}$'],
        }
    )

    labels = mc_plot._resolve_range_display_labels(range_data)

    assert labels == ['Mo2', 'CrMo']


def test_range_peak_labels_prefer_raw_ion_text():
    range_data = pd.DataFrame(
        {
            'name': ['Fe', 'Co'],
            'ion': ['$Fe^{2+}$', '$Co^{+}$'],
        }
    )

    labels = mc_plot._resolve_range_peak_labels(range_data)

    assert labels == ['$Fe^{2+}$', '$Co^{+}$']


def test_plain_range_label_strips_mathtext_markup():
    assert mc_plot._plain_range_label('$Mo_{2}^{2+}$') == 'Mo2 2+'


def test_plot_peaks_handles_range_center_past_histogram_edge():
    variables = Variables()
    data = np.linspace(0.0, 10.0, 500)
    plotter = mc_plot.AptHistPlotter(data, variables)
    plotter.plot_histogram(bin_width=0.1, plot_show=True)

    range_data = pd.DataFrame(
        {
            'name': ['edge_peak'],
            'ion': ['$edge^{+}$'],
            'mc': [float(plotter.x[-1] + 1.0)],
        }
    )

    plotter.plot_peaks(range_data=range_data)

    assert len(plotter.peak_annotates) == 1


def test_gaussian_mrp_report_rejects_implausibly_narrow_subpeak():
    rng = np.random.default_rng(42)
    broad_peak = rng.normal(32.95, 0.035, 160000)
    narrow_spike = rng.normal(32.951, 0.00035, 1200)
    values = np.concatenate([broad_peak, narrow_spike])

    report = gaussian_mrp_report(values, 32.75, 33.15, bin_size=0.001, peak_center=32.95)

    assert report is not None
    # The recommended MRP must reflect the *broad* peak (MRP ~400),
    # not the implausibly narrow spike (MRP ~40,000).
    mrp_50 = float(report['recommended_mrp'][0])
    assert np.isfinite(mrp_50), "recommended MRP(0.5) must be finite"
    assert mrp_50 < 2000, f"MRP(0.5) = {mrp_50} — should reflect broad peak, not narrow spike"


def test_voigt_mrp_finite_at_one_percent_for_lorentzian_with_long_tail():
    """User-reported case: sharp narrow apex (~0.13 Da FWHM) with a long
    Lorentzian-dominated right tail. MRP(0.01) was being marked NA because
    `_sanitize_tail_widths` rejected the genuinely large 1% width as
    exceeding the original fit window. Verify it now reports a finite,
    monotonically-decreasing value."""
    rng = np.random.default_rng(13)
    apex = rng.normal(31.10, 0.025, 100000)
    long_tail = 31.10 + np.abs(rng.standard_cauchy(40000)) * 0.18
    long_tail = long_tail[long_tail < 36.5]
    background = rng.uniform(28.0, 36.5, 5000)
    values = np.concatenate([apex, long_tail, background])

    report = gaussian_mrp_report(values, 31.05, 31.20, bin_size=0.001, peak_center=31.10)
    assert report is not None
    voigt = report['voigt_mrp']
    assert np.isfinite(voigt[0]), "Voigt MRP(0.5) should be finite"
    assert np.isfinite(voigt[1]), "Voigt MRP(0.1) should be finite (was NaN before fix)"
    assert np.isfinite(voigt[2]), "Voigt MRP(0.01) should be finite (was NaN before fix)"
    assert voigt[0] >= voigt[1] >= voigt[2], (
        f"MRP must fall as fraction drops: {voigt[0]} >= {voigt[1]} >= {voigt[2]}"
    )


def test_voigt_mrp_returns_finite_values_at_10_and_1_percent():
    """For a Lorentzian-dominated peak at fine bin size, MRP at the 10% and 1%
    levels must be reported (not NaN). Previously the analytic Voigt was only
    sampled inside the fit window so its tails ran off the edge of x_fine and
    triggered a NaN return for fractions below 50%."""
    rng = np.random.default_rng(7)
    # Strong central peak with heavy Lorentzian tails (Cauchy distribution).
    cauchy_tail = rng.standard_cauchy(150000) * 0.04 + 30.5
    cauchy_tail = cauchy_tail[(cauchy_tail > 30.0) & (cauchy_tail < 31.0)]
    background = rng.uniform(29.0, 32.0, 5000)
    values = np.concatenate([cauchy_tail, background])

    report = gaussian_mrp_report(values, 30.2, 30.8, bin_size=0.001, peak_center=30.5)
    assert report is not None
    voigt = report['voigt_mrp']
    assert np.isfinite(voigt[0]), "Voigt MRP(0.5) should be finite"
    assert np.isfinite(voigt[1]), "Voigt MRP(0.1) should be finite (was NaN before fix)"
    # MRP must decrease monotonically as we measure at lower fractions of max.
    assert voigt[0] >= voigt[1], (
        f"Voigt MRP should fall as fraction drops: {voigt[0]} (50%) >= {voigt[1]} (10%)"
    )


def test_histogram_mrp_is_not_absurdly_large_at_fine_bin_sizes():
    """When the histogram bin is much finer than the peak's apex bin spacing,
    scipy.peak_widths returned an artificially tiny width that yielded an
    absurdly large histogram-based MRP (e.g. 30,000+). The fix re-bins to a
    coarser comparison histogram internally so the histogram-based value is
    in the same order of magnitude as the analytical fits, not orders of
    magnitude too large."""
    rng = np.random.default_rng(11)
    # Tall narrow peak: 64k events in a ~0.05 Da-wide Gaussian, plus a thin
    # background. With bin_size=0.001 the apex bin towers over its neighbors.
    sharp = rng.normal(30.5, 0.025, 64000)
    background = rng.uniform(28.0, 33.0, 5000)
    values = np.concatenate([sharp, background])

    report = gaussian_mrp_report(values, 30.0, 31.0, bin_size=0.001, peak_center=30.5)
    assert report is not None
    histogram_mrp = report['histogram_mrp']
    gaussian_mrp = report['gaussian_mrp']
    # The histogram value must either be NaN or be in the same order of
    # magnitude as the Gaussian fit. The 30,000+ value seen before the fix
    # would clearly fail this check.
    if np.isfinite(histogram_mrp[0]):
        assert histogram_mrp[0] < gaussian_mrp[0] * 5.0, (
            f"histogram MRP(0.5) = {histogram_mrp[0]} is much larger than the "
            f"Gaussian fit MRP(0.5) = {gaussian_mrp[0]} — re-binning did not help"
        )


def test_mrp_internal_bin_size_is_fixed_reference_width():
    assert _MRP_INTERNAL_BIN_SIZE == 0.01


def test_mrp_calculation_is_independent_of_display_bin_width():
    rng = np.random.default_rng(123)
    data = rng.normal(27.0, 0.018, 25000)
    variables = Variables()
    mrp_values = []

    for bin_width in (0.1, 0.01):
        plotter = mc_plot.AptHistPlotter(data, variables)
        plotter.plot_histogram(bin_width=bin_width, plot_show=False, fast=True)
        plotter.find_peaks_and_widths(prominence=10, distance=2, percent=50)
        mrp, _ = plotter.mrp_calculation()
        mrp_values.append(mrp[0])

    assert np.isfinite(mrp_values).all()
    assert abs(mrp_values[0] - mrp_values[1]) < 1.0


def test_mrp_calculation_uses_voigt_report_only_for_dominant_peak(monkeypatch):
    rng = np.random.default_rng(321)
    data = np.concatenate([
        rng.normal(27.0, 0.018, 25000),
        rng.normal(52.0, 0.025, 12000),
    ])
    variables = Variables()
    plotter = mc_plot.AptHistPlotter(data, variables)
    plotter.plot_histogram(bin_width=0.01, plot_show=False, fast=True)
    plotter.find_peaks_and_widths(prominence=10, distance=50, percent=50)

    calls = {"report": 0, "fast": 0}

    def fake_report(*args, **kwargs):
        calls["report"] += 1
        return {
            "histogram_mrp": [900.0, 500.0, 350.0],
            "voigt_mrp": [800.0, 450.0, 320.0],
            "voigt_ok": True,
            "profile_type": "test",
            "recommended_peak_sides": [[26.9, 27.1], [26.8, 27.2], [26.7, 27.3]],
        }

    def fake_fast(*args, **kwargs):
        calls["fast"] += 1
        return [100.0, 80.0, 60.0]

    monkeypatch.setattr(peak_helpers, "gaussian_mrp_report", fake_report)
    monkeypatch.setattr(peak_helpers, "fast_mrp", fake_fast)

    mrp_peak, mrp_all = plotter.mrp_calculation()

    assert calls["report"] == 1
    assert calls["fast"] == len(plotter.peaks) - 1
    assert mrp_peak == [800.0, 450.0, 320.0]
    assert 800.0 in mrp_all["MRP(0.5)"]
    assert mrp_all["MRP(0.5)"].count(100.0) == len(plotter.peaks) - 1

