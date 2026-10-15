"""Bounded shared-memory buffers for live plot delivery.

The runtime uses ``SharedEventBuffer``: all four fields of an event share one
copy/publication lock, including during wraparound. ``SharedRingBuffer`` is
retained for legacy scalar callers; its lock-free copy cannot guarantee a
coherent snapshot when a producer laps an active reader. Do not use separate
scalar rings to transport associated event fields.

Legacy scalar layout
--------------------
* One ``multiprocessing.shared_memory.SharedMemory`` block holds the data
  ring as a 1-D numpy array of fixed dtype + capacity.
* A second small block holds two ``int64`` indices: ``write_idx`` and
  ``read_idx`` (monotonically increasing, modulo `capacity` for ring
  arithmetic).  On x86-64 aligned 8-byte writes are atomic and store
  order is preserved (TSO), which is what this SPSC design relies on.
  STRICT SPSC discipline: ``write_idx`` is written ONLY by the producer
  and ``read_idx`` ONLY by the consumer. The producer never advances the
  read index (lapping is resolved on the consumer side via clamping);
  this avoids a lost-update race on the shared index. On weakly-ordered
  architectures (e.g. some ARM) a fence would be required between the
  data store and the index publish -- not added here because the control
  rig is x86-64.
* Producer (TDC): :meth:`write` copies samples into the ring at
  ``write_idx % capacity`` and advances ``write_idx``.  If the producer
  laps the consumer the *oldest* samples in the lapped window are
  overwritten silently - sane behaviour for a live plot whose consumer
  already discards old data.
* Consumer (visualization): :meth:`read_all` returns a freshly-allocated
  numpy copy of every sample produced since its last call, up to the
  buffer capacity, then advances ``read_idx``.

Lifecycle
---------
The *parent* process creates a buffer with :meth:`create`; child
processes attach with :meth:`attach` using the same ``name``.  The
parent must call :meth:`unlink` on shutdown to release the OS-level
memory; children only call :meth:`close`.
"""

from __future__ import annotations

import multiprocessing
import logging
from multiprocessing import shared_memory

import numpy as np

_INDEX_DTYPE = np.int64
_INDEX_BYTES = 24  # three int64s: [write_idx, read_idx, dropped]


class SharedRingBuffer:
    """SPSC ring buffer over multiprocessing shared memory.

    Use :meth:`create` from the parent process and :meth:`attach` from
    children.  The two share a common ``name`` and ``capacity``.
    """

    def __init__(
        self,
        name: str,
        capacity: int,
        dtype: np.dtype,
        data_shm: shared_memory.SharedMemory,
        index_shm: shared_memory.SharedMemory,
        owner: bool,
    ):
        self.name = name
        self.capacity = int(capacity)
        self.dtype = np.dtype(dtype)
        self._data_shm = data_shm
        self._index_shm = index_shm
        self._owner = owner
        # Numpy views over the shared blocks - no copy.
        self._data = np.ndarray((self.capacity,), dtype=self.dtype, buffer=data_shm.buf)
        self._idx = np.ndarray((3,), dtype=_INDEX_DTYPE, buffer=index_shm.buf)

    # ------------------------------------------------------------------ ctors

    @classmethod
    def create(cls, name: str, capacity: int, dtype="float32") -> "SharedRingBuffer":
        """Allocate the shared blocks (parent / owner only)."""
        if int(capacity) <= 0:
            raise ValueError("Ring capacity must be positive")
        dtype = np.dtype(dtype)
        data_shm = shared_memory.SharedMemory(
            name=f"{name}__data",
            create=True,
            size=capacity * dtype.itemsize,
        )
        index_shm = shared_memory.SharedMemory(
            name=f"{name}__idx",
            create=True,
            size=_INDEX_BYTES,
        )
        # Initialise both indices to 0 in the shared block.
        np.ndarray((3,), dtype=_INDEX_DTYPE, buffer=index_shm.buf)[:] = 0
        return cls(name, capacity, dtype, data_shm, index_shm, owner=True)

    @classmethod
    def attach(cls, name: str, capacity: int, dtype="float32") -> "SharedRingBuffer":
        """Attach to an existing pair of shared blocks (child / consumer)."""
        dtype = np.dtype(dtype)
        data_shm = shared_memory.SharedMemory(name=f"{name}__data")
        index_shm = shared_memory.SharedMemory(name=f"{name}__idx")
        return cls(name, capacity, dtype, data_shm, index_shm, owner=False)

    # ------------------------------------------------------------------ pickle

    def __reduce__(self):
        """Send only the metadata across process boundaries.

        The receiving process re-attaches via :meth:`attach`, never as
        owner - so unlinking the OS blocks remains the parent's job.
        """
        return (_attach_for_unpickle, (self.name, self.capacity, self.dtype))

    # ------------------------------------------------------------------ I/O

    def write(self, samples) -> None:
        """Append samples to the ring.  Oldest are silently overwritten if full."""
        arr = np.asarray(samples, dtype=self.dtype).ravel()
        n = arr.size
        produced = n
        if n == 0:
            return
        # If a single chunk exceeds the ring, only the last `capacity`
        # samples can survive; trimming up front avoids needless wrap-and-
        # overwrite work and keeps the layout aligned for the reader.
        if n > self.capacity:
            arr = arr[-self.capacity :]
            n = arr.size

        write_idx = int(self._idx[0])
        unread = max(0, min(self.capacity, write_idx - int(self._idx[1])))
        overwritten = max(0, unread + produced - self.capacity)
        if overwritten:
            self._idx[2] = int(self._idx[2]) + overwritten
        # Sequence indices include incoming records discarded by truncation.
        start = (write_idx + produced - n) % self.capacity
        end = start + n
        if end <= self.capacity:
            self._data[start:end] = arr
        else:
            split = self.capacity - start
            self._data[start:] = arr[:split]
            self._data[: end - self.capacity] = arr[split:]
        new_write = write_idx + produced
        # Publish the new write index LAST, after the data stores above,
        # so a consumer that observes the advanced index also sees the
        # data (store-order is preserved on x86-64 TSO; see module note).
        self._idx[0] = new_write
        # NOTE: the producer deliberately does NOT touch ``self._idx[1]``
        # (the read index). Doing so made the producer a second writer of
        # the consumer's index, racing the consumer's own update in
        # ``read_all`` -- a lost-update bug that could make the consumer
        # re-read consumed data or skip fresh data. Lapping is handled
        # entirely on the consumer side: ``read_all`` and ``pending``
        # clamp to ``capacity`` when ``write_idx - read_idx`` exceeds it,
        # so the consumer still sees only the freshest ``capacity``
        # samples. This keeps the read index strictly single-writer.

    def read_all(self) -> np.ndarray:
        """Return a copy of every sample produced since the last call."""
        write_idx = int(self._idx[0])
        read_idx = int(self._idx[1])
        n = write_idx - read_idx
        if n <= 0:
            return np.empty(0, dtype=self.dtype)
        if n > self.capacity:
            # Producer lapped us between the two index loads above; clamp.
            n = self.capacity
            read_idx = write_idx - n
        out = np.empty(n, dtype=self.dtype)
        start = read_idx % self.capacity
        end = start + n
        if end <= self.capacity:
            out[:] = self._data[start:end]
        else:
            split = self.capacity - start
            out[:split] = self._data[start:]
            out[split:] = self._data[: end - self.capacity]
        self._idx[1] = write_idx
        return out

    def reset(self) -> None:
        """Set both write and read indices to zero (drops all queued data)."""
        self._idx[0] = 0
        self._idx[1] = 0
        self._idx[2] = 0

    def pending(self) -> int:
        """Number of samples waiting to be read (capped at capacity)."""
        n = int(self._idx[0]) - int(self._idx[1])
        return max(0, min(n, self.capacity))

    @property
    def dropped(self) -> int:
        """Total samples overwritten before the consumer observed them."""
        return max(0, int(self._idx[2]))

    # ------------------------------------------------------------------ teardown

    def close(self) -> None:
        """Detach this process from the shared blocks (no unlink)."""
        try:
            self._data_shm.close()
        except Exception:
            pass
        try:
            self._index_shm.close()
        except Exception:
            pass

    def unlink(self) -> None:
        """Release the OS-level shared memory.  Owner only, on shutdown."""
        if not self._owner:
            return
        for shm in (self._data_shm, self._index_shm):
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except Exception:
                pass


def _attach_for_unpickle(name: str, capacity: int, dtype: str) -> "SharedRingBuffer":
    """Module-level reconstructor invoked by ``__reduce__`` on unpickle."""
    return SharedRingBuffer.attach(name, capacity, dtype)


_EVENT_DTYPE = np.dtype([(name, "float32") for name in ("x", "y", "t", "v")])


class SharedEventBuffer(SharedRingBuffer):
    """Bounded complete-event transport with one cross-process commit lock.

    The lock protects both copying and publication: a producer cannot overwrite
    an event while a reader copies it. Only plot delivery uses this lock; saved
    acquisition data and feedback control remain independent. Spawn transfers
    the shared lock and memory names, never the event payload.
    """

    @classmethod
    def create(cls, name, capacity, dtype=None):
        buffer = super().create(name, capacity, _EVENT_DTYPE)
        buffer._lock = multiprocessing.get_context("spawn").RLock()
        buffer.transport_error = ''
        return buffer

    def __reduce__(self):
        return (_attach_events, (self.name, self.capacity, self._lock))

    def write_events(self, x, y, t, v):
        columns = [np.asarray(values, dtype="float32").reshape(-1) for values in (x, y, t, v)]
        lengths = {len(values) for values in columns}
        if len(lengths) != 1:
            raise ValueError("Plot event columns must have equal lengths")
        events = np.empty(len(columns[0]), dtype=_EVENT_DTYPE)
        for name, values in zip(_EVENT_DTYPE.names, columns):
            events[name] = values
        if not self._acquire():
            return False
        try:
            super().write(events)
        finally:
            self._lock.release()
        return True

    def read_events(self):
        if not self._acquire():
            return tuple(np.empty(0, dtype='float32') for _ in range(4))
        try:
            events = super().read_all()
        finally:
            self._lock.release()
        return tuple(events[name] for name in _EVENT_DTYPE.names)

    def reset(self):
        if not self._acquire():
            return
        try:
            super().reset()
        finally:
            self._lock.release()

    def _acquire(self):
        # A terminated process can abandon a Windows semaphore. Plot-only
        # delivery must never strand acquisition/Stop. Fail closed locally:
        # don't copy possibly half-written events and don't mutate the ring.
        if self.transport_error:
            return False
        if self._lock.acquire(timeout=.05):
            return True
        self.transport_error = 'Plot event transport unavailable; restart the visualization context'
        logging.getLogger('apt').error(self.transport_error)
        return False


def _attach_events(name, capacity, lock):
    buffer = SharedEventBuffer.attach(name, capacity, _EVENT_DTYPE)
    buffer._lock = lock
    buffer.transport_error = ''
    return buffer


def publish_plot_events(x_plot, y_plot, t_plot, v_plot, x, y, t, v):
    """Publish one event batch; retain legacy scalar sinks for external callers."""
    if isinstance(x_plot, SharedEventBuffer):
        x_plot.write_events(x, y, t, v)
    else:
        for sink, values in zip((x_plot, y_plot, t_plot, v_plot), (x, y, t, v)):
            if sink is not None:
                sink.write(values)
