import numpy as np

import pandas as pd

from pyccapt.calibration.core import mc_plot
from pyccapt.calibration.core.mc_plot_peak_helpers import gaussian_mrp_report
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
    assert report['recommended_label'].startswith('Histogram')
    assert float(report['recommended_mrp'][0]) < float(report['gaussian_mrp'][0])

