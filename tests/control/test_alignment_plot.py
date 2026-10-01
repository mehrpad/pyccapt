"""Display history limits and units; no hardware or OpenGL rendering required."""
import numpy as np

from pyccapt.control.gui.alignment_plot import AlignmentHitmapHistory, AlignmentPlotHistory


def snapshot(t=0, sample=1, sequence='one', **changes):
    return dict(time=t, sample=sample, sequence_id=sequence, origin_m=(.01, .02, .03),
                position_m=(.010001, .019998, .03), rate_percent=.3, target_percent=1.,
                voltage=1500., phase='coarse', **changes)


def test_units_bounded_history_and_sequence_isolation():
    history = AlignmentPlotHistory(capacity=3)
    for i in range(6):
        assert history.append(snapshot(i))
    xyz, records = history.coordinates(1)
    np.testing.assert_allclose(xyz, [[1, -2, .3]]*3)
    assert [r['time'] for r in records] == [3, 4, 5]
    assert not history.append(snapshot(5))
    assert history.append(snapshot(6, sample=2))
    assert len(history.samples[1]) == 3
    assert history.append(snapshot(7, sequence='new'))
    assert set(history.samples) == {1}
    assert len(history.samples[1]) == 1


def test_invalid_plot_reading_is_ignored():
    history = AlignmentPlotHistory()
    invalid = snapshot()
    invalid['rate_percent'] = float('nan')
    assert not history.append(invalid)
    assert not history.append({})
    assert not history.samples


def test_alignment_hitmap_uses_only_new_paired_detector_events():
    history = AlignmentHitmapHistory(capacity=3)
    assert history.append({'epoch': 'a', 'sequence': 2, 'points_mm': [[1, 2], [3, 4]]})
    assert not history.append({'epoch': 'a', 'sequence': 2, 'points_mm': [[1, 2], [3, 4]]})
    assert history.append({'epoch': 'a', 'sequence': 3, 'points_mm': [[1, 2], [3, 4], [5, 6]]})
    np.testing.assert_allclose(history.coordinates(), [[1, 2], [3, 4], [5, 6]])
    history.reset()
    assert len(history.coordinates()) == 0
    assert not history.append({'epoch': 'a', 'sequence': 3, 'points_mm': [[1, 2], [3, 4], [5, 6]]})
    assert history.append({'epoch': 'a', 'sequence': 4, 'points_mm': [[3, 4], [5, 6], [7, 8]]})
    np.testing.assert_allclose(history.coordinates(), [[7, 8]])
    assert history.append({'epoch': 'b', 'sequence': 1, 'points_mm': [[9, 10]]})
    np.testing.assert_allclose(history.coordinates(), [[9, 10]])
    assert history.prepare_epoch('c')
    assert len(history.coordinates()) == 0
    assert not history.prepare_epoch('c')
    assert not history.append({'epoch': 'b', 'sequence': 2, 'points_mm': [[float('nan'), 1]]})
