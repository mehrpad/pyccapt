"""Bounded typed acquisition blocks and supervised background disk writes.

Only completed immutable NumPy blocks cross the writer queue. This avoids
growing Manager lists and scalar pickling while retaining the chunk manifest
and atomic publication contract used by acquisition recovery.
"""
from __future__ import annotations

from collections import deque
import multiprocessing
from pathlib import Path
from queue import Empty, Full
import time

import numpy as np

from pyccapt.control.core.chunk_store import atomic_write_chunk_group


def _write_chunks(directory, queue, errors):
    owner = multiprocessing.parent_process()
    try:
        while True:
            # A forcibly terminated detector cannot send its sentinel. Do not
            # leave an orphan I/O worker waiting forever for another packet.
            if owner is not None and not owner.is_alive():
                return
            try:
                task = queue.get(timeout=.1)
            except Empty:
                continue
            if task is None:
                return
            stream_name, chunk_id, arrays = task
            atomic_write_chunk_group(directory, stream_name=stream_name,
                                     chunk_id=chunk_id, arrays=arrays)
    except Exception as exc:
        errors.send(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        errors.close()


class ChunkWriter:
    """One bounded I/O worker; failures are observed by the acquisition owner."""

    def __init__(self, directory, *, capacity=8, timeout_s=30., stop_event=None):
        self.directory = Path(directory)
        self.timeout_s = float(timeout_s)
        self.stop_event = stop_event
        self.queue = multiprocessing.Queue(maxsize=capacity)
        self.errors, sender = multiprocessing.Pipe(duplex=False)
        self.closed = False
        self.process = multiprocessing.Process(target=_write_chunks,
                                               args=(self.directory, self.queue, sender),
                                               name="detector-chunk-writer")
        self.process.start()
        sender.close()

    def check(self):
        try:
            if self.errors.poll():
                message = self.errors.recv()
                raise RuntimeError(f"Detector chunk writer failed: {message}")
        except (EOFError, BrokenPipeError):
            # Windows reports a normally closed sender from poll(), whereas
            # POSIX reports EOF from recv(). Process status distinguishes a
            # successful shutdown from a worker that died without a message.
            pass
        if self.process.exitcode not in (None, 0):
            raise RuntimeError(f"Detector chunk writer exited with code {self.process.exitcode}")
        if not self.closed and not self.process.is_alive():
            raise RuntimeError("Detector chunk writer exited unexpectedly")

    def _put(self, task):
        deadline = time.monotonic() + self.timeout_s
        stop_seen = False
        while True:
            self.check()
            if not stop_seen and self.stop_event is not None and self.stop_event.is_set():
                # Finish already acquired blocks, but never make Stop wait
                # indefinitely behind a full queue or failed disk.
                stop_seen = True
                deadline = min(deadline, time.monotonic() + min(5., self.timeout_s))
            try:
                self.queue.put(task, timeout=min(.05, max(0., deadline-time.monotonic())))
                return
            except Full:
                if time.monotonic() >= deadline:
                    stopping = self.stop_event is not None and self.stop_event.is_set()
                    raise TimeoutError("Detector chunk writer did not drain its queue"
                                       + (" during shutdown" if stopping else ""))

    def submit(self, stream_name, chunk_id, arrays):
        if self.closed:
            raise RuntimeError("Detector chunk writer is already closed")
        self._put((stream_name, chunk_id, arrays))

    def close(self):
        if self.closed:
            return
        try:
            self._put(None)
            self.closed = True
            self.process.join(self.timeout_s)
            if self.process.is_alive():
                raise TimeoutError("Detector chunk writer did not finish before the shutdown deadline")
            self.check()
        finally:
            self.closed = True
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(2.)
            # Do not block process exit flushing tasks after a writer failure.
            self.queue.cancel_join_thread()
            self.queue.close()
            self.errors.close()


class ChunkStream:
    """Aligned columns buffered in array blocks, with every full chunk drained."""

    def __init__(self, writer, name, dtypes, *, chunk_size=100_000):
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        self.writer = writer
        self.name = name
        self.dtypes = dict(dtypes)
        self.chunk_size = int(chunk_size)
        self.blocks = deque()
        self.rows = 0
        self.chunk_id = 0

    def append(self, arrays):
        if arrays.keys() != self.dtypes.keys():
            raise ValueError(f"Unexpected columns in detector stream {self.name}")
        # The queue's feeder serializes asynchronously and SDK buffers may be
        # reused immediately after append returns. Own each typed column once;
        # subsequent chunk slicing never mutates these arrays.
        normalized = {key: np.array(arrays[key], dtype=dtype, copy=True).reshape(-1)
                      for key, dtype in self.dtypes.items()}
        lengths = {len(values) for values in normalized.values()}
        if len(lengths) != 1:
            raise ValueError(f"Mismatched detector columns in {self.name}")
        size = next(iter(lengths))
        if size:
            self.blocks.append(normalized)
            self.rows += size
        while self.rows >= self.chunk_size:
            self._flush(self.chunk_size)

    def _flush(self, count):
        pieces = {key: [] for key in self.dtypes}
        remaining = count
        # Build slices without changing the buffer until submission succeeds.
        for block in self.blocks:
            take = min(remaining, len(next(iter(block.values()))))
            for key in pieces:
                pieces[key].append(block[key][:take])
            remaining -= take
            if not remaining:
                break
        arrays = {key: parts[0] if len(parts) == 1 else np.concatenate(parts)
                  for key, parts in pieces.items()}
        self.writer.submit(self.name, self.chunk_id + 1, arrays)
        self.chunk_id += 1
        remaining = count
        while remaining:
            block = self.blocks.popleft()
            size = len(next(iter(block.values())))
            if size > remaining:
                self.blocks.appendleft({key: values[remaining:] for key, values in block.items()})
            remaining -= min(size, remaining)
        self.rows -= count

    def finish(self):
        if self.rows:
            self._flush(self.rows)


def read_event_metadata(variables, energy_reader, *, attempts=3):
    """Retry one packet's shared metadata; never silently discard that packet."""
    for attempt in range(attempts):
        try:
            return (float(variables.specimen_voltage), float(variables.pulse_voltage),
                    float(energy_reader(variables)))
        except Exception as exc:
            if attempt + 1 == attempts:
                raise RuntimeError("Cannot associate detector packet with voltage/laser metadata") from exc
            time.sleep(.01)


def preserve_unattributed_packet(directory, stream_name, arrays):
    """Keep SDK evidence separately when metadata cannot be read; not a valid chunk."""
    directory = Path(directory) / "unattributed"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stream_name}-{time.time_ns()}.npz"
    np.savez(path, **{key: np.asarray(value) for key, value in arrays.items()})
    return path
