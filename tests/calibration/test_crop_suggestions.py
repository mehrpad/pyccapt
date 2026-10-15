"""Proposals must follow measured structure and remain reviewable before mutation."""
import numpy as np
import pandas as pd
import pytest

from pyccapt.calibration.core.crop_suggestions import suggest_detector_circle, suggest_temporal_crop
from pyccapt.calibration.core.share_variables import Variables
from pyccapt.calibration.tutorials.tutorials_helpers import helper_temporal_crop as temporal
from pyccapt.calibration.tutorials.tutorials_helpers import helper_special_crop as spatial


def dense_detector(seed=9):
    rng = np.random.default_rng(seed)
    angle = rng.uniform(-np.pi, np.pi, 80000)
    radius = 2*np.sqrt(rng.random(len(angle)))
    x, y = radius*np.cos(angle), radius*np.sin(angle)
    # An edge-connected bite should move/shrink the proposal away from this edge.
    keep = ~((x > 1.) & (np.abs(y) < .8))
    angle = rng.uniform(-np.pi, np.pi, 7000)
    radius = rng.uniform(2.2, 3.8, len(angle))
    return pd.DataFrame({'x_det (cm)': np.r_[x[keep], radius*np.cos(angle)],
                         'y_det (cm)': np.r_[y[keep], radius*np.sin(angle)]})


def test_temporal_suggestion_excludes_startup_and_late_voltage_reset_without_mutating():
    voltage = np.r_[np.tile([0., 4000.], 1000), np.linspace(3000, 5000, 16000),
                    np.linspace(2000, 2400, 2000)]
    frame = pd.DataFrame({'high_voltage (V)': voltage})
    original = frame.copy(deep=True)
    report = suggest_temporal_crop(frame)
    assert report['supported'] and 2000 <= report['event_start'] < 3000
    assert 17000 < report['event_stop'] <= 18000
    pd.testing.assert_frame_equal(frame, original)
    assert report == suggest_temporal_crop(frame)


def test_clean_ramp_and_unavailable_voltage_do_not_invent_crop_boundaries():
    frame = pd.DataFrame({'high_voltage (V)': np.linspace(2000, 5000, 10000)})
    report = suggest_temporal_crop(frame)
    assert report['supported'] and (report['event_start'], report['event_stop']) == (0, len(frame))
    frame.attrs['pyccapt_unavailable_columns'] = ['high_voltage (V)']
    assert not suggest_temporal_crop(frame)['supported']
    assert not suggest_temporal_crop(frame.iloc[:20])['supported']


def test_density_circle_avoids_halo_and_edge_bite_and_is_deterministic():
    frame = dense_detector()
    original = frame.copy(deep=True)
    report = suggest_detector_circle(frame)
    assert report['supported'] and report == suggest_detector_circle(frame)
    cx, cy, radius = report['circle_cm']
    assert 1.1 < radius < 1.9 and cx < -.1 and abs(cy) < .6
    assert cx + radius < 1.3  # The bite begins at x=1; histogram smoothing gives finite resolution.
    assert report['diagnostics']['finite_events'] == len(frame)
    pd.testing.assert_frame_equal(frame, original)


def test_degenerate_or_sparse_detector_abstains():
    assert not suggest_detector_circle(pd.DataFrame({'x_det (cm)': np.zeros(4000),
                                                     'y_det (cm)': np.zeros(4000)}))['supported']
    assert not suggest_detector_circle(dense_detector().iloc[:100])['supported']


def test_legacy_plot_seeds_shapes_but_only_apply_changes_data(monkeypatch):
    frame = dense_detector()
    frame['high_voltage (V)'] = np.linspace(2000, 5000, len(frame))
    frame['t (ns)'] = 100.
    variables = Variables()
    variables.sync_from_data(frame)
    original_ids = variables.data.event_id.to_numpy(copy=True)
    captured = {}
    monkeypatch.setattr(temporal.data_loadcrop, 'plot_crop_experiment_history',
                        lambda *a: captured.update(rectangle=a[6]))
    monkeypatch.setattr(spatial.data_loadcrop, 'plot_crop_fdm', lambda *a, **k: captured.update(circle=k['draw_circle']))
    interval = temporal.call_plot_crop_experiment(variables, 'voltage', display_panel=False)
    circle = spatial.call_plot_crop_fdm(variables, display_panel=False)
    interval.controls['Plot'].click()
    circle.controls['Plot'].click()
    assert captured == {'rectangle': True, 'circle': True}
    np.testing.assert_array_equal(variables.data.event_id, original_ids)
    assert variables.selected_x2 == len(frame)-1  # Legacy end is inclusive.
    circle.controls['manual'].value = True
    circle.controls['x'].value, circle.controls['y'].value = 0., 0.
    circle.controls['radius'].value = .75
    circle.controls['Plot'].click()
    assert variables.roi_fdm == .75  # Manual values win over the proposal.
    circle.controls['Apply crop'].click()
    assert 0 < len(variables.data) < len(frame)
    assert (np.hypot(variables.data['x_det (cm)'], variables.data['y_det (cm)']) <= .75).all()
    assert variables.calibration_metadata.state['steps'][-1]['operation'] == 'detector_crop'


def test_legacy_seeded_circle_handles_resize_and_move_update_applied_fields(monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import MouseEvent
    from pyccapt.calibration.data_tools import data_loadcrop, selectors_data
    variables = Variables()
    frame = dense_detector().iloc[:15000].copy()
    frame['t (ns)'] = 100.
    variables.sync_from_data(frame)
    variables.selected_x_fdm, variables.selected_y_fdm, variables.roi_fdm = -.3, .2, 1.
    monkeypatch.setattr(plt, 'show', lambda: None)
    data_loadcrop.plot_crop_fdm(frame['x_det (cm)'].to_numpy(), frame['y_det (cm)'].to_numpy(),
        (80, 80), 1., variables=variables, data_crop=True, draw_circle=True)
    selector = selectors_data.toggle_selector.ES
    canvas, axis = selector.ax.figure.canvas, selector.ax
    def drag(start, end):
        canvas.draw()
        for name, point in [('button_press_event', start), ('motion_notify_event', end), ('button_release_event', end)]:
            x, y = axis.transData.transform(point)
            canvas.callbacks.process(name, MouseEvent(name, canvas, x, y, button=1))
    try:
        assert selector._selection_artist.get_visible()
        drag((.7, .2), (1.2, .2))
        assert variables.roi_fdm == pytest.approx(1.5, abs=.02)
        assert variables.selected_x_fdm == pytest.approx(-.3, abs=.02)
        drag((-.3, .2), (.2, -.3))
        assert variables.selected_x_fdm == pytest.approx(.2, abs=.02)
        assert variables.selected_y_fdm == pytest.approx(-.3, abs=.02)
        assert variables.roi_fdm == pytest.approx(1.5, abs=.02)
    finally:
        plt.close(axis.figure)
