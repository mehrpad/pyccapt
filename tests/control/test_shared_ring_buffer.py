"""Tests for the SPSC shared-memory ring buffer (CC1).

Previously the producer's ``write`` advanced the CONSUMER's read index
when it lapped the consumer, making the producer a second writer of the
read index and racing the consumer's own update in ``read_all`` (a
lost-update bug). The fix keeps the read index strictly single-writer:
lapping is resolved entirely on the consumer side via clamping. These
tests pin both the normal round-trip and the lapping behaviour.
"""
import numpy as np
import pytest

from pyccapt.control.core.shared_ring_buffer import SharedRingBuffer


@pytest.fixture()
def ring():
    rb = SharedRingBuffer.create("test_ring_cc1", capacity=8, dtype="float32")
    try:
        yield rb
    finally:
        rb.unlink()


def test_basic_write_read_roundtrip(ring):
    ring.write(np.array([1, 2, 3], dtype="float32"))
    assert ring.read_all().tolist() == [1.0, 2.0, 3.0]
    # Nothing left after consuming.
    assert ring.read_all().size == 0
    assert ring.pending() == 0


def test_producer_does_not_touch_read_index(ring):
    # Write without reading; the read index (idx[1]) must stay at 0 --
    # only the consumer may advance it.
    ring.write(np.array([1, 2, 3], dtype="float32"))
    assert int(ring._idx[1]) == 0, "producer must not advance the read index"
    assert ring.pending() == 3


def test_lapping_returns_freshest_capacity_samples(ring):
    # Capacity is 8. Write 3, then 10 more without reading -> the producer
    # has lapped the consumer. The consumer must still get only the
    # freshest 8 samples, with no crash / negative count.
    ring.write(np.array([1, 2, 3], dtype="float32"))
    ring.write(np.arange(10, 20, dtype="float32"))  # 10..19

    out = ring.read_all()
    assert out.size == 8, f"expected the freshest 8 samples, got {out.size}"
    # The 8 freshest of the total written (..., 12..19) are 12..19.
    assert out.tolist() == [12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
    assert ring.pending() == 0


def test_pending_is_clamped_to_capacity(ring):
    ring.write(np.arange(0, 30, dtype="float32"))  # far more than capacity
    assert ring.pending() == ring.capacity


def test_reset_drops_queued_data(ring):
    ring.write(np.array([1, 2, 3], dtype="float32"))
    ring.reset()
    assert ring.pending() == 0
    assert ring.read_all().size == 0
