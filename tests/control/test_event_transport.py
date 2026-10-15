"""Overflow, process transfer and retained-history regression tests."""
import multiprocessing
import threading
import uuid

import numpy as np
import pytest

from pyccapt.control.core.shared_ring_buffer import SharedEventBuffer, SharedRingBuffer, publish_plot_events
from pyccapt.control.core.plot_history import EventHistory


def _spawn_publish(buffer):
    ids = np.arange(23)
    buffer.write_events(ids, ids + 100, ids + 200, ids + 300)


@pytest.fixture
def events():
    buffer = SharedEventBuffer.create('events_' + uuid.uuid4().hex, 4)
    try:
        yield buffer
    finally:
        buffer.unlink()


def test_complete_events_stay_paired_across_overflow(events):
    for batch in (np.arange(4), np.array([4]), np.array([5])):
        publish_plot_events(events, events, events, events, batch, batch + 100, batch + 200, batch + 300)
    columns = events.read_events()
    for offset, column in zip((0, 100, 200, 300), columns):
        np.testing.assert_array_equal(column, np.arange(2, 6) + offset)
    assert events.dropped == 2
    assert events.pending() == 0


def test_oversized_batch_counts_incoming_and_unread_losses(events):
    events.write_events(*([np.arange(3)] * 4))
    events.write_events(*([np.arange(10, 20)] * 4))
    assert events.dropped == 9  # 13 produced, freshest 4 retained
    assert int(events._idx[0]) == 13
    np.testing.assert_array_equal(events.read_events()[0], np.arange(16, 20))
    events.write_events(*([np.array([20])] * 4))
    assert events.dropped == 9


def test_misaligned_columns_rejected_before_publication(events):
    with pytest.raises(ValueError, match='equal lengths'):
        events.write_events([1, 2], [1], [1], [1])
    assert events.pending() == 0


def test_spawn_uses_same_memory_and_commit_lock(events):
    process = multiprocessing.get_context('spawn').Process(target=_spawn_publish, args=(events,))
    process.start()
    process.join(15)
    try:
        assert process.exitcode == 0
        x, y, t, v = events.read_events()
        np.testing.assert_array_equal(x, np.arange(19, 23))
        np.testing.assert_array_equal(y, x + 100)
        np.testing.assert_array_equal(t, x + 200)
        np.testing.assert_array_equal(v, x + 300)
        assert events.dropped == 19
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)


def test_concurrent_wraps_never_return_mixed_events(events):
    done = threading.Event()
    def publish():
        try:
            for i in range(2000):
                events.write_events([i], [i + 10], [i + 20], [i + 30])
        finally:
            done.set()
    writer = threading.Thread(target=publish)
    writer.start()
    previous = -1
    while not done.is_set() or events.pending():
        x, y, t, v = events.read_events()
        if not len(x):
            continue
        assert x[0] > previous
        previous = x[-1]
        np.testing.assert_array_equal(y, x + 10)
        np.testing.assert_array_equal(t, x + 20)
        np.testing.assert_array_equal(v, x + 30)
    writer.join(5)
    assert previous == 1999


def test_scalar_legacy_ring_counts_only_new_losses():
    buffer = SharedRingBuffer.create('scalar_' + uuid.uuid4().hex, 4)
    try:
        for batch in (range(4), [4], [5]):
            buffer.write(batch)
        assert buffer.dropped == 2
        np.testing.assert_array_equal(buffer.read_all(), [2, 3, 4, 5])
    finally:
        buffer.unlink()


def test_abandoned_plot_lock_does_not_block_acquisition_or_expose_partial_events(events):
    class UnavailableLock:
        def acquire(self, *, timeout):
            assert 0 < timeout <= .05
            return False
    events._lock = UnavailableLock()
    assert events.write_events([1], [2], [3], [4]) is False
    assert all(len(column) == 0 for column in events.read_events())
    events.reset()
    assert events.transport_error
    assert int(events._idx[0]) == 0


def test_history_matches_concatenation_last_n_for_varied_batches():
    history = EventHistory(capacity=7)
    expected = np.empty((0, 4))
    for ids in (np.arange(3), np.arange(3, 11), np.array([]), np.arange(11, 13), np.arange(13, 17)):
        batch = np.column_stack([ids + j for j in range(4)])
        history.append(*batch.T)
        expected = np.concatenate((expected, batch))[-7:]
        np.testing.assert_array_equal(history.snapshot(), expected)
        np.testing.assert_array_equal(history.snapshot(3), expected[-3:])


def test_history_snapshot_survives_append_and_reset():
    history = EventHistory(capacity=3, columns=2)
    history.append([1, 2], [11, 12])
    snapshot = history.snapshot()
    assert history.snapshot() is snapshot
    history.append([3, 4, 5], [13, 14, 15])
    history.clear()
    np.testing.assert_array_equal(snapshot, [[1, 11], [2, 12]])
    assert not snapshot.flags.writeable
    assert history.snapshot().shape == (0, 2)


def test_unfiltered_hitmap_is_bounded_when_all_tof_events_rejected():
    from pyccapt.control.gui.gui_visualization import Ui_Visualization
    ui = Ui_Visualization.__new__(Ui_Visualization)
    ui._event_history = EventHistory(capacity=4)
    ui._hitmap_history = EventHistory(capacity=4, columns=2)
    for i in range(20):
        ui._hitmap_history.append([i], [-i])
        ui._event_history.append([], [], [], [])
    np.testing.assert_array_equal(ui.last_100_thousand_det_x_heatmap, [16, 17, 18, 19])
    assert len(ui.last_100_thousand_t) == 0
