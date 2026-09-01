"""Compact, multi-resolution history storage for the vacuum live plot."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable


VACUUM_CHANNELS = ("main", "buffer", "ll", "cll")


@dataclass(frozen=True)
class VacuumSample:
    timestamp: datetime
    main: float
    buffer: float
    ll: float
    cll: float


class VacuumHistory:
    """Keep recent samples at full resolution and older samples compactly.

    The most recent hour is retained as sampled.  One representative sample
    per 30 seconds is retained for the rest of the week, keeping the GUI's
    memory use bounded while preserving useful long-term trends.
    """

    def __init__(self, raw_seconds: int = 3600, archive_seconds: int = 30, retention_seconds: int = 7 * 86400):
        self.raw_seconds = raw_seconds
        self.archive_seconds = archive_seconds
        self.retention_seconds = retention_seconds
        self._raw: deque[VacuumSample] = deque()
        self._archive: deque[VacuumSample] = deque()
        self._archive_bucket: int | None = None

    def add(self, timestamp: datetime, values: Iterable[float]) -> None:
        sample = VacuumSample(timestamp, *(float(value) for value in values))
        self._raw.append(sample)
        raw_cutoff = timestamp - timedelta(seconds=self.raw_seconds)
        while self._raw and self._raw[0].timestamp < raw_cutoff:
            old = self._raw.popleft()
            bucket = int(old.timestamp.timestamp()) // self.archive_seconds
            if bucket != self._archive_bucket:
                self._archive.append(old)
                self._archive_bucket = bucket

        retention_cutoff = timestamp - timedelta(seconds=self.retention_seconds)
        while self._archive and self._archive[0].timestamp < retention_cutoff:
            self._archive.popleft()

    def window(self, seconds: int, max_points: int = 1500) -> list[VacuumSample]:
        if not self._raw:
            return []
        cutoff = self._raw[-1].timestamp - timedelta(seconds=seconds)
        samples = [sample for sample in self._archive if sample.timestamp >= cutoff]
        samples.extend(sample for sample in self._raw if sample.timestamp >= cutoff)
        if len(samples) <= max_points:
            return samples
        stride = (len(samples) + max_points - 1) // max_points
        reduced = samples[::stride]
        if reduced[-1] is not samples[-1]:
            reduced.append(samples[-1])
        return reduced
