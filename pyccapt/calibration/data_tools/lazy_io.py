"""Column-oriented lazy I/O for big APT datasets.

This module gives the rest of the calibration code a single, backend-agnostic
way to read multi-gigabyte EPOS / POS / pyccapt-raw HDF5 files without paying
the cost of materializing every column into RAM up front.

Why this exists
---------------

The eager paths in :mod:`pyccapt.calibration.leap_tools.leap_tools` and
:mod:`pyccapt.calibration.data_tools.data_tools` make several full copies of
each dataset during load -- e.g. :func:`read_epos` ran 11 ``np.asarray`` casts
on the memmap (one per field) before the DataFrame was even built, so a 2 GB
EPOS briefly pinned ~7 GB of RAM. That blocks 8 GB machines from using the
calibration tutorials at all.

The classes here let downstream code work on a column-major *view* that reads
on demand:

- :class:`LazyColumn` wraps either a NumPy memmap field (EPOS / POS) or an
  ``h5py.Dataset`` (pyccapt-raw HDF5) and exposes a NumPy-compatible API.
  Slicing reads exactly the requested bytes from disk; the ``__array__``
  protocol means most NumPy / SciPy / matplotlib functions accept it
  unchanged.
- :class:`LazyTable` is a column dict that knows its row count and can stream
  itself in chunks via :meth:`LazyTable.iter_chunks`.
- :func:`open_epos`, :func:`open_pos`, :func:`open_pyccapt_raw_hdf5` are the
  three loaders.
- :func:`chunked_histogram`, :func:`chunked_min_max`, :func:`chunked_apply`
  are the building blocks the analysis helpers need so they can compute
  statistics over a big file in fixed-size chunks.

Backward compatibility
----------------------

Nothing in this module replaces existing eager call sites. The existing
``read_epos`` / ``read_pos`` / ``read_hdf5`` keep their signatures; callers
that want the new lazy behavior import :func:`open_epos` etc. directly.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable, Sequence
from warnings import warn

import numpy as np
import pandas as pd


# Default chunk size for streaming reads. 1<<20 = ~1M rows. EPOS rows are 40
# bytes so a chunk is ~40 MB; a pyccapt-raw HDF5 row is similar. This sits in
# L3 cache on most workstations and is small enough to not pressure an 8 GB
# machine while big enough to amortize per-call overhead.
DEFAULT_CHUNK_SIZE = 1 << 20


# ---------------------------------------------------------------------------
# Column view
# ---------------------------------------------------------------------------


class LazyColumn:
    """A NumPy-compatible view of one column in a :class:`LazyTable`.

    The wrapped source is either a memmap field (a strided view into a NumPy
    structured array) or an ``h5py.Dataset``. Both support row-range slicing
    cheaply, which is the only access pattern this class promises.

    The :meth:`__array__` protocol means ``np.asarray(col)``,
    ``np.histogram(col, ...)``, ``col.mean()`` etc. all work -- they trigger
    exactly one full read of just this column.
    """

    __slots__ = ("_source", "_native_dtype", "_name", "_length")

    def __init__(self, source, *, name: str, native_dtype: np.dtype | None = None):
        self._source = source
        self._name = name
        self._length = len(source)
        if native_dtype is None:
            native_dtype = _native_view_dtype(np.dtype(source.dtype))
        self._native_dtype = np.dtype(native_dtype)

    # --- introspection -----------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def dtype(self) -> np.dtype:
        """The *native* dtype callers see; conversion happens on read."""
        return self._native_dtype

    @property
    def source_dtype(self) -> np.dtype:
        """The on-disk dtype before any byte-order conversion."""
        return np.dtype(self._source.dtype)

    @property
    def itemsize(self) -> int:
        return self._native_dtype.itemsize

    @property
    def shape(self) -> tuple[int, ...]:
        return (self._length,)

    def __len__(self) -> int:
        return self._length

    def __repr__(self) -> str:
        return f"LazyColumn(name={self._name!r}, len={self._length}, dtype={self._native_dtype})"

    # --- access ------------------------------------------------------------

    def __getitem__(self, item) -> np.ndarray:
        raw = self._source[item]
        if isinstance(raw, np.ndarray):
            if raw.dtype != self._native_dtype:
                return raw.astype(self._native_dtype, copy=False)
            return raw
        # Scalar fetch (single int index).
        return self._native_dtype.type(raw)

    def __array__(self, dtype=None) -> np.ndarray:
        # NumPy / pandas / matplotlib pull the whole column through this hook.
        full = self[:]
        if dtype is not None:
            full = full.astype(dtype, copy=False)
        return full

    def to_numpy(self, dtype: np.dtype | None = None, copy: bool = False) -> np.ndarray:
        """Return the full column as a contiguous ``ndarray``.

        Materializes the whole column, so prefer slicing or :func:`chunked_*`
        helpers for the big-RAM case.
        """
        out = self[:]
        if dtype is not None and np.dtype(dtype) != out.dtype:
            out = out.astype(dtype, copy=False)
        if copy:
            out = np.ascontiguousarray(out).copy()
        else:
            out = np.ascontiguousarray(out)
        return out


# ---------------------------------------------------------------------------
# Table view
# ---------------------------------------------------------------------------


class LazyTable:
    """A lazy column-store over an APT-style file.

    The table is column-oriented: looking up a column returns a
    :class:`LazyColumn` and the underlying bytes are *not* read until the
    column is sliced or iterated. The :meth:`iter_chunks` helper is the
    primary way analysis code should consume a :class:`LazyTable` -- a chunk
    fits in L3 cache and the worst-case resident memory is bounded by
    ``chunk_size`` regardless of how big the file is.
    """

    def __init__(
        self,
        columns: dict[str, LazyColumn],
        *,
        n_rows: int | None = None,
        source_path: Path | None = None,
        close: Callable[[], None] | None = None,
    ):
        if not columns:
            raise ValueError("LazyTable needs at least one column")
        self._columns: dict[str, LazyColumn] = dict(columns)
        column_lengths = [len(c) for c in self._columns.values()]
        if n_rows is None:
            n_rows = max(column_lengths)
        # We *allow* mismatched column lengths (pyccapt-raw HDF5 has /dld and
        # /tdc groups with independent row counts). The headline ``n_rows`` is
        # the longest column; iteration helpers use each column's true length.
        if n_rows < 0:
            raise ValueError("n_rows must be non-negative")
        self._n_rows = int(n_rows)
        self._source_path = Path(source_path) if source_path is not None else None
        self._close = close
        self._closed = False

    # --- introspection -----------------------------------------------------

    @property
    def columns(self) -> list[str]:
        return list(self._columns)

    @property
    def n_rows(self) -> int:
        return self._n_rows

    def __len__(self) -> int:
        return self._n_rows

    @property
    def source_path(self) -> Path | None:
        return self._source_path

    def __repr__(self) -> str:
        return (
            f"LazyTable(rows={self._n_rows}, columns={self.columns}, "
            f"source={self._source_path})"
        )

    # --- access ------------------------------------------------------------

    def __getitem__(self, name: str) -> LazyColumn:
        try:
            return self._columns[name]
        except KeyError:
            available = ", ".join(self._columns)
            raise KeyError(f"No column {name!r} in LazyTable. Available: {available}") from None

    def __contains__(self, name: str) -> bool:
        return name in self._columns

    def __iter__(self):
        return iter(self._columns)

    # --- chunked reads ----------------------------------------------------

    def iter_chunks(
        self,
        columns: Sequence[str] | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield ``{column: ndarray}`` slices of at most ``chunk_size`` rows.

        The iteration extent is the *minimum* of the selected columns' true
        lengths, so per-group iteration on a mixed-length HDF5 table reads
        only the rows that exist in every selected column (no NaN padding).
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        wanted = list(columns) if columns is not None else self.columns
        unknown = [c for c in wanted if c not in self._columns]
        if unknown:
            raise KeyError(f"Unknown columns: {unknown}")
        extent = min(len(self._columns[name]) for name in wanted)
        for start in range(0, extent, chunk_size):
            stop = min(start + chunk_size, extent)
            yield {name: self._columns[name][start:stop] for name in wanted}

    def iter_chunk_frames(
        self,
        columns: Sequence[str] | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> Iterator[pd.DataFrame]:
        """Same as :meth:`iter_chunks` but yields DataFrames."""
        for chunk in self.iter_chunks(columns, chunk_size):
            yield pd.DataFrame(chunk)

    def head(self, n: int = 20, columns: Sequence[str] | None = None) -> pd.DataFrame:
        n = max(0, int(n))
        wanted = list(columns) if columns is not None else self.columns
        return pd.DataFrame({name: self._columns[name][:n] for name in wanted})

    def to_dataframe(
        self,
        columns: Sequence[str] | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> pd.DataFrame:
        """Materialize the whole table into a DataFrame.

        Built chunk-by-chunk so the peak temporary memory is bounded by
        ``chunk_size`` rather than the full file size during construction.
        """
        wanted = list(columns) if columns is not None else self.columns
        lengths = {name: len(self._columns[name]) for name in wanted}
        if not lengths or max(lengths.values()) == 0:
            return pd.DataFrame({name: np.empty(0, dtype=self._columns[name].dtype) for name in wanted})
        # Preallocate the destination per-column. Once the destination buffers
        # exist we never hold a second full copy of any column.
        buffers = {
            name: np.empty(lengths[name], dtype=self._columns[name].dtype) for name in wanted
        }
        extent = max(lengths.values())
        for start in range(0, extent, chunk_size):
            for name in wanted:
                stop = min(start + chunk_size, lengths[name])
                if start >= stop:
                    continue
                buffers[name][start:stop] = self._columns[name][start:stop]
        if len(set(lengths.values())) == 1:
            return pd.DataFrame(buffers, copy=False)
        # Mixed lengths: build per-column Series and let pandas align/pad with NaN.
        return pd.DataFrame({name: pd.Series(buffers[name]) for name in wanted})

    # --- resource management ---------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._close is not None:
            try:
                self._close()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                warn(f"LazyTable.close() raised: {exc}", RuntimeWarning)

    def __enter__(self) -> "LazyTable":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):  # pragma: no cover - GC fallback
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------


def _native_view_dtype(dtype: np.dtype) -> np.dtype:
    """Return the native-byte-order equivalent of ``dtype`` (no copy implied)."""
    if dtype.byteorder in ("=", "|"):
        return dtype
    return dtype.newbyteorder("=")


class _StructuredFieldView:
    """Adapter that lets a memmap field expose the same ``[slice]`` API as
    :class:`h5py.Dataset` and supports ``np.asarray`` cheaply."""

    __slots__ = ("_memmap", "_field")

    def __init__(self, structured_memmap: np.memmap, field: str):
        self._memmap = structured_memmap
        self._field = field

    @property
    def dtype(self) -> np.dtype:
        return self._memmap.dtype[self._field]

    def __len__(self) -> int:
        return len(self._memmap)

    def __getitem__(self, item):
        return self._memmap[self._field][item]


class _PlainMemmapColumn:
    """Adapter for a 2-D memmap when each row is a fixed-length array (POS)."""

    __slots__ = ("_memmap", "_col_index")

    def __init__(self, memmap_2d: np.memmap, col_index: int):
        self._memmap = memmap_2d
        self._col_index = col_index

    @property
    def dtype(self) -> np.dtype:
        return self._memmap.dtype

    def __len__(self) -> int:
        return self._memmap.shape[0]

    def __getitem__(self, item):
        return self._memmap[item, self._col_index]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


EPOS_RECORD_DTYPE = np.dtype([
    ("x", ">f4"),
    ("y", ">f4"),
    ("z", ">f4"),
    ("mc", ">f4"),
    ("tof", ">f4"),
    ("hv", ">f4"),
    ("pulse", ">f4"),
    ("det_x", ">f4"),
    ("det_y", ">f4"),
    ("pslep", ">u4"),
    ("ipp", ">u4"),
])

# Column rename map matches the eager ``read_epos`` output so callers can
# swap between the two without renaming downstream references.
EPOS_COLUMN_NAMES = {
    "x": "x (nm)",
    "y": "y (nm)",
    "z": "z (nm)",
    "mc": "m/n (Da)",
    "tof": "TOF (ns)",
    "hv": "HV_DC (V)",
    "pulse": "pulse (V)",
    "det_x": "det_x (mm)",
    "det_y": "det_y (mm)",
    "pslep": "pslep",
    "ipp": "ipp",
}


def open_epos(file_path: str | Path) -> LazyTable:
    """Open an APT ``.epos`` file as a :class:`LazyTable` (memmap-backed)."""
    file_path = Path(file_path)
    record_size = EPOS_RECORD_DTYPE.itemsize
    file_size = os.path.getsize(file_path)
    record_count, remainder = divmod(file_size, record_size)
    if remainder:
        warn(
            f"The .epos file size ({file_size} bytes) is not an exact multiple of "
            f"{record_size}. Ignoring the final {remainder} trailing bytes.",
            RuntimeWarning,
        )
    if record_count == 0:
        return _empty_lazy_table(EPOS_COLUMN_NAMES.values(), source_path=file_path)

    records = np.memmap(file_path, dtype=EPOS_RECORD_DTYPE, mode="r", shape=(record_count,))
    columns = {
        EPOS_COLUMN_NAMES[field]: LazyColumn(
            _StructuredFieldView(records, field),
            name=EPOS_COLUMN_NAMES[field],
        )
        for field, _ in EPOS_RECORD_DTYPE.fields.items()
    }

    def _close():
        # Release the memmap so the file handle drops on Windows.
        try:
            records._mmap.close()
        except Exception:
            pass

    return LazyTable(columns, n_rows=record_count, source_path=file_path, close=_close)


POS_COLUMN_NAMES = ("x (nm)", "y (nm)", "z (nm)", "m/n (Da)")


def open_pos(file_path: str | Path) -> LazyTable:
    """Open an APT ``.pos`` file as a :class:`LazyTable` (memmap-backed)."""
    file_path = Path(file_path)
    record_size = 16  # 4 columns x float32 big-endian
    file_size = os.path.getsize(file_path)
    record_count, remainder = divmod(file_size, record_size)
    if remainder:
        warn(
            f"The .pos file size ({file_size} bytes) is not an exact multiple of "
            f"{record_size}. Ignoring the final {remainder} trailing bytes.",
            RuntimeWarning,
        )
    if record_count == 0:
        return _empty_lazy_table(POS_COLUMN_NAMES, source_path=file_path)

    records = np.memmap(file_path, dtype=">f4", mode="r", shape=(record_count, 4))
    columns = {
        name: LazyColumn(_PlainMemmapColumn(records, idx), name=name)
        for idx, name in enumerate(POS_COLUMN_NAMES)
    }

    def _close():
        try:
            records._mmap.close()
        except Exception:
            pass

    return LazyTable(columns, n_rows=record_count, source_path=file_path, close=_close)


def open_pyccapt_raw_hdf5(file_path: str | Path) -> LazyTable:
    """Open a pyccapt-raw HDF5 file as a :class:`LazyTable` (h5py-backed).

    All datasets in ``/dld/`` and ``/tdc/`` become columns named like
    ``dld/high_voltage``. The file handle stays open for the lifetime of the
    table and is closed via :meth:`LazyTable.close` (or the context manager).
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ImportError("h5py is required for open_pyccapt_raw_hdf5()") from exc

    file_path = Path(file_path)
    handle = h5py.File(file_path, "r")

    columns: dict[str, LazyColumn] = {}
    n_rows: int | None = None

    def visit(name, obj):
        nonlocal n_rows
        if not isinstance(obj, h5py.Dataset):
            return
        if not name.startswith(("dld/", "tdc/")):
            return
        # Flatten any (N, 1) shaped dataset to 1-D.
        if obj.ndim == 2 and obj.shape[1] == 1:
            wrapped = _H5SqueezeWrapper(obj)
        else:
            wrapped = obj
        col = LazyColumn(wrapped, name=name)
        columns[name] = col
        if n_rows is None:
            n_rows = len(col)

    handle.visititems(visit)

    if not columns:
        handle.close()
        raise ValueError(
            f"{file_path} contains no /dld/* or /tdc/* datasets -- is this a pyccapt-raw HDF5?"
        )

    return LazyTable(columns, source_path=file_path, close=handle.close)


class _H5SqueezeWrapper:
    """Make an ``h5py.Dataset`` of shape ``(N, 1)`` look 1-D."""

    __slots__ = ("_ds",)

    def __init__(self, dataset):
        self._ds = dataset

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._ds.dtype)

    def __len__(self) -> int:
        return int(self._ds.shape[0])

    def __getitem__(self, item):
        data = self._ds[item]
        if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[-1] == 1:
            return data[:, 0]
        return data


# ---------------------------------------------------------------------------
# Chunked accumulators
# ---------------------------------------------------------------------------


def chunked_min_max(column: LazyColumn, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> tuple[float, float]:
    """Single-pass min/max over a :class:`LazyColumn` without holding it in RAM."""
    lo: float | None = None
    hi: float | None = None
    for start in range(0, len(column), chunk_size):
        chunk = column[start:start + chunk_size]
        if chunk.size == 0:
            continue
        c_lo = float(np.min(chunk))
        c_hi = float(np.max(chunk))
        lo = c_lo if lo is None else min(lo, c_lo)
        hi = c_hi if hi is None else max(hi, c_hi)
    if lo is None:
        raise ValueError("chunked_min_max() called on an empty column")
    return lo, hi


def chunked_histogram(
    column: LazyColumn,
    bins: int | np.ndarray,
    *,
    value_range: tuple[float, float] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-pass histogram of a :class:`LazyColumn`.

    Returns ``(counts, edges)`` with the same shape contract as
    :func:`numpy.histogram`. When ``bins`` is an int and ``value_range`` is
    omitted, a separate min/max pass is run first.
    """
    if isinstance(bins, int):
        if value_range is None:
            value_range = chunked_min_max(column, chunk_size=chunk_size)
        lo, hi = value_range
        if lo == hi:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, bins + 1)
    else:
        edges = np.asarray(bins, dtype=float)
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    for start in range(0, len(column), chunk_size):
        chunk = column[start:start + chunk_size]
        if chunk.size == 0:
            continue
        c_counts, _ = np.histogram(chunk, bins=edges)
        counts += c_counts.astype(np.int64, copy=False)
    return counts, edges


def chunked_apply(
    table: LazyTable,
    fn: Callable[[dict[str, np.ndarray]], Any],
    *,
    columns: Sequence[str] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[Any]:
    """Apply ``fn`` to each chunk of ``table`` and yield each result.

    A thin wrapper around :meth:`LazyTable.iter_chunks` for cases where the
    analysis code wants to drive its own accumulation loop but prefers the
    function-call shape.
    """
    for chunk in table.iter_chunks(columns=columns, chunk_size=chunk_size):
        yield fn(chunk)


# ---------------------------------------------------------------------------
# Empty-table helper
# ---------------------------------------------------------------------------


def _empty_lazy_table(column_names, *, source_path: Path | None = None) -> LazyTable:
    columns: dict[str, LazyColumn] = {}
    empty = np.empty(0, dtype=np.float32)
    for name in column_names:
        columns[name] = LazyColumn(_PlainNdarraySource(empty), name=name)
    return LazyTable(columns, n_rows=0, source_path=source_path)


class _PlainNdarraySource:
    """Used by :func:`_empty_lazy_table` -- a column-shaped ndarray adapter."""

    __slots__ = ("_arr",)

    def __init__(self, arr: np.ndarray):
        self._arr = arr

    @property
    def dtype(self) -> np.dtype:
        return self._arr.dtype

    def __len__(self) -> int:
        return int(self._arr.shape[0])

    def __getitem__(self, item):
        return self._arr[item]


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "EPOS_COLUMN_NAMES",
    "EPOS_RECORD_DTYPE",
    "LazyColumn",
    "LazyTable",
    "POS_COLUMN_NAMES",
    "chunked_apply",
    "chunked_histogram",
    "chunked_min_max",
    "open_epos",
    "open_pos",
    "open_pyccapt_raw_hdf5",
]
