import numpy as np

from pyccapt.calibration.core import mc_plot
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

