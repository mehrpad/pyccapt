"""Notebook display and transfer controls must not alter reconstruction mathematics."""
from types import SimpleNamespace

import ipywidgets as widgets
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import pytest

from pyccapt.calibration.core.exceptions import CalibrationInputError
from pyccapt.calibration.reconstructions import reconstruction
from pyccapt.calibration.reconstructions.display_limits import limit_display
from pyccapt.calibration.tutorials.tutorials_helpers import helper_3d_reconstruction as helper
from pyccapt.calibration.tutorials.tutorials_helpers import helper_sdm_calibration as tuning
from tests.calibration.test_ranging_reconstruction_regressions import _plot_variables


def test_helpers_display_once_and_legacy_sdm_cell_routes_to_embedded_tab(monkeypatch):
    displays = []
    monkeypatch.setattr(helper, 'display', displays.append)
    monkeypatch.setattr(tuning, 'display', displays.append)
    variables = _plot_variables()
    args = variables, widgets.FloatText(value=110.), SimpleNamespace(value=('Al', 60., 30.))
    assert helper.call_x_y_z_calculation(*args) is None
    assert len(displays) == 1
    panel = displays[0]
    assert panel.reconstruction_tabs.get_title(0) == '1. Find / tune ICF and kf'
    assert panel.reconstruction_tabs.get_title(1) == '2. Reconstruct and plot'
    assert panel.controls['load'].disabled
    assert 'No accepted parameter set' in panel.controls['status'].value
    panel.reconstruction_tabs.selected_index = 1
    assert tuning.call_sdm_icf_kf_calibration(*args) is None
    assert len(displays) == 1 and panel.reconstruction_tabs.selected_index == 0
    variables.reconstruction_parameters = dict(kf=4.2, icf=1.7, det_eff=.6, field_evap=30., avg_dens=60., projection='Bas', flight_path_length=120.)
    panel.controls['load'].click()
    assert panel.controls['parameters']['kf'].value == 4.2
    assert panel.controls['parameters']['mode'].value == 'Bas'
    assert 'loaded' in panel.controls['status'].value and not panel.controls['load'].disabled


@pytest.mark.parametrize('mode,renderer', [('inline', 'plotly_mimetype'), ('browser', 'browser')])
def test_compact_plot_uses_selected_renderer_caps_display_and_keeps_full_coordinates(monkeypatch, mode, renderer):
    variables = _plot_variables()
    variables.reconstruction_selection_mask = np.array([False, True, True, True])
    original = variables.x.copy()
    figures = []
    default = pio.renderers.default
    monkeypatch.setattr(go.Figure, 'show', lambda self, **kw: figures.append((self, kw)))
    monkeypatch.setattr(go, 'FigureWidget', lambda figure: figure)
    reconstruction.reconstruction_plot(variables, [1., 1.], .5, False, 'test', False,
                                       display_mode=mode, max_display_points=2)
    figure, options = figures[0]
    assert options['renderer'] == renderer and pio.renderers.default == default
    traces = [trace for trace in figure.data if trace.type == 'scatter3d' and trace.mode == 'markers']
    shown = np.concatenate([np.asarray(trace.x) for trace in traces])
    assert len(shown) <= 2 and 1. not in shown
    assert figure.layout.height == 480 and figure.layout.width == 760
    assert variables.plotly_3d_reconstruction.layout.height == 480
    np.testing.assert_array_equal(variables.x, original)


def test_sparse_cluster_overlay_is_included_in_total_display_limit():
    mask = np.ones(100, dtype=bool)
    species = np.zeros(100, dtype=bool)
    species[:2] = True
    cluster = SimpleNamespace(labels=np.r_[np.full(50, -1), np.zeros(50)])
    limited, sampled, plotted = limit_display(mask, [species], cluster, False, 10)
    assert plotted.sum() == 10 and limited.sum() == 10
    assert (np.flatnonzero(plotted) >= 50).any()
    assert not np.any(sampled[0] & ~limited)
    with pytest.raises(CalibrationInputError, match='positive integer'):
        limit_display(mask, [species], None, False, 1.5)
