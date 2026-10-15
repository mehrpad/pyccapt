"""Fixed-size event histories for live plots; no full-history copy on append."""
from __future__ import annotations

import numpy as np


class EventHistory:
    """Retain the last capacity rows, with lazily materialized ordered snapshots.

    The owner supplies synchronization when readers run on another thread.
    Snapshots own their data and remain valid after subsequent appends/resets.
    """

    def __init__(self, capacity=100_000, columns=4):
        if capacity <= 0 or columns <= 0:
            raise ValueError("History dimensions must be positive")
        self.capacity = int(capacity)
        self._data = np.empty((self.capacity, columns), dtype=np.float64)
        self.clear()

    def clear(self):
        self._end = self._count = 0
        self._snapshot = None

    def append(self, *columns):
        arrays = [np.asarray(column).reshape(-1) for column in columns]
        if len(arrays) != self._data.shape[1] or len({len(a) for a in arrays}) != 1:
            raise ValueError("History requires aligned columns")
        n = min(len(arrays[0]), self.capacity)
        if not n:
            return
        start = self._end if len(arrays[0]) < self.capacity else 0
        first = min(n, self.capacity - start)
        for i, values in enumerate(arrays):
            values = values[-n:]
            self._data[start:start + first, i] = values[:first]
            self._data[:n - first, i] = values[first:]
        self._end = (start + n) % self.capacity
        self._count = min(self.capacity, self._count + n)
        self._snapshot = None

    def snapshot(self, limit=None):
        n = self._count if limit is None else min(self._count, max(0, int(limit)))
        if n == self._count and self._snapshot is not None:
            return self._snapshot
        start = (self._end - n) % self.capacity
        first = min(n, self.capacity - start)
        result = np.empty((n, self._data.shape[1]), dtype=self._data.dtype)
        result[:first] = self._data[start:start + first]
        result[first:] = self._data[:n - first]
        result.flags.writeable = False
        if n == self._count:
            self._snapshot = result
        return result
