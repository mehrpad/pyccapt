"""Lock-free single-producer / single-consumer shared-memory ring buffer.

Used by the live-data pipeline so the detector subprocess (TDC) can hand
plot samples to the visualization subprocess without going through a
multiprocessing.Queue (which pickles every chunk and grows without bound
if the consumer falls behind).

Design
------
* One ``multiprocessing.shared_memory.SharedMemory`` block holds the data
  ring as a 1-D numpy array of fixed dtype + capacity.
* A second small block holds two ``int64`` indices: ``write_idx`` and
  ``read_idx`` (monotonically increasing, modulo `capacity` for ring
  arithmetic).  On x86-64 aligned 8-byte writes are atomic, which is all
  SPSC needs.
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

from multiprocessing import shared_memory

import numpy as np

_INDEX_DTYPE = np.int64
_INDEX_BYTES = 16  # two int64s: [write_idx, read_idx]


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
		self._idx = np.ndarray((2,), dtype=_INDEX_DTYPE, buffer=index_shm.buf)

	# ------------------------------------------------------------------ ctors

	@classmethod
	def create(cls, name: str, capacity: int, dtype="float32") -> "SharedRingBuffer":
		"""Allocate the shared blocks (parent / owner only)."""
		dtype = np.dtype(dtype)
		data_shm = shared_memory.SharedMemory(
			name=f"{name}__data", create=True, size=capacity * dtype.itemsize,
		)
		index_shm = shared_memory.SharedMemory(
			name=f"{name}__idx", create=True, size=_INDEX_BYTES,
		)
		# Initialise both indices to 0 in the shared block.
		np.ndarray((2,), dtype=_INDEX_DTYPE, buffer=index_shm.buf)[:] = 0
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
		return (_attach_for_unpickle, (self.name, self.capacity, str(self.dtype)))

	# ------------------------------------------------------------------ I/O

	def write(self, samples) -> None:
		"""Append samples to the ring.  Oldest are silently overwritten if full."""
		arr = np.asarray(samples, dtype=self.dtype).ravel()
		n = arr.size
		if n == 0:
			return
		# If a single chunk exceeds the ring, only the last `capacity`
		# samples can survive; trimming up front avoids needless wrap-and-
		# overwrite work and keeps the layout aligned for the reader.
		if n > self.capacity:
			arr = arr[-self.capacity:]
			n = arr.size

		write_idx = int(self._idx[0])
		start = write_idx % self.capacity
		end = start + n
		if end <= self.capacity:
			self._data[start:end] = arr
		else:
			split = self.capacity - start
			self._data[start:] = arr[:split]
			self._data[: end - self.capacity] = arr[split:]
		new_write = write_idx + n
		self._idx[0] = new_write
		# If we lapped the consumer, drag its read pointer forward so it
		# sees only the freshest `capacity` samples on the next read.
		if new_write - int(self._idx[1]) > self.capacity:
			self._idx[1] = new_write - self.capacity

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

	def pending(self) -> int:
		"""Number of samples waiting to be read (capped at capacity)."""
		n = int(self._idx[0]) - int(self._idx[1])
		return max(0, min(n, self.capacity))

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
