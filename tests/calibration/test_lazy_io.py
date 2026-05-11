"""Tests for the lazy I/O abstraction in pyccapt.calibration.data_tools.lazy_io.

These tests build small synthetic EPOS / POS / pyccapt-raw HDF5 files in a
temporary directory and verify that:

- :class:`LazyColumn` returns the right values via slicing and ``__array__``.
- :class:`LazyTable` materializes the same DataFrame as the eager
  :func:`read_epos` / :func:`read_pos` paths.
- :meth:`LazyTable.iter_chunks` covers every row exactly once.
- :func:`chunked_histogram` / :func:`chunked_min_max` agree with NumPy on
  the eager array.
- The lazy ``read_hdf5(lazy=True)`` mode walks ``/dld/*`` and ``/tdc/*``.
"""

from __future__ import annotations

import os
import struct
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyccapt.calibration.data_tools import data_tools, lazy_io
from pyccapt.calibration.leap_tools import leap_tools


# ---------------------------------------------------------------------------
# EPOS round-trip
# ---------------------------------------------------------------------------


def _write_synthetic_epos(path: Path, n: int) -> dict[str, np.ndarray]:
    """Write ``n`` records following the LEAP big-endian EPOS layout.

    Returns the source arrays in *native* dtype so tests can compare against
    them after the lazy loader does the byte-order swap.
    """
    rng = np.random.default_rng(0)
    floats = {
        "x": rng.uniform(-5, 5, n).astype(np.float32),
        "y": rng.uniform(-5, 5, n).astype(np.float32),
        "z": rng.uniform(0, 100, n).astype(np.float32),
        "mc": rng.uniform(0, 150, n).astype(np.float32),
        "tof": rng.uniform(100, 10000, n).astype(np.float32),
        "hv": rng.uniform(1000, 12000, n).astype(np.float32),
        "pulse": rng.uniform(0, 500, n).astype(np.float32),
        "det_x": rng.uniform(-30, 30, n).astype(np.float32),
        "det_y": rng.uniform(-30, 30, n).astype(np.float32),
    }
    uints = {
        "pslep": rng.integers(0, 5000, n, dtype=np.uint32),
        "ipp": rng.integers(0, 10, n, dtype=np.uint32),
    }
    record = np.zeros(n, dtype=lazy_io.EPOS_RECORD_DTYPE)
    for field, arr in {**floats, **uints}.items():
        record[field] = arr
    with open(path, "wb") as fh:
        # tobytes() preserves the structured dtype's big-endian fields.
        fh.write(record.tobytes())
    return {**floats, **uints}


@pytest.fixture
def epos_file(tmp_path: Path):
    path = tmp_path / "synthetic.epos"
    source = _write_synthetic_epos(path, n=2048)
    return path, source


def test_open_epos_basic(epos_file):
    path, source = epos_file
    with lazy_io.open_epos(path) as table:
        assert table.n_rows == 2048
        assert "x (nm)" in table
        assert "ipp" in table
        col_x = table["x (nm)"]
        # __array__ pulls the whole column, byteswapped to native.
        assert np.allclose(np.asarray(col_x), source["x"])
        # Slicing reads only the requested range.
        assert np.allclose(col_x[10:25], source["x"][10:25])
        # Scalar indexing returns a scalar (not a 0-d array).
        scalar = col_x[7]
        assert np.shape(scalar) == ()


def test_open_epos_dataframe_matches_eager(epos_file):
    path, source = epos_file
    eager = leap_tools.read_epos(path)
    with lazy_io.open_epos(path) as table:
        lazy_df = table.to_dataframe()
    assert list(eager.columns) == list(lazy_df.columns)
    for col in eager.columns:
        np.testing.assert_array_equal(eager[col].to_numpy(), lazy_df[col].to_numpy())


def test_iter_chunks_covers_every_row(epos_file):
    path, source = epos_file
    seen_x: list[np.ndarray] = []
    with lazy_io.open_epos(path) as table:
        for chunk in table.iter_chunks(["x (nm)"], chunk_size=300):
            seen_x.append(chunk["x (nm)"])
    joined = np.concatenate(seen_x)
    assert joined.shape == (2048,)
    np.testing.assert_array_equal(joined, source["x"])


def test_chunked_histogram_matches_numpy(epos_file):
    path, source = epos_file
    with lazy_io.open_epos(path) as table:
        counts, edges = lazy_io.chunked_histogram(table["m/n (Da)"], bins=32, chunk_size=257)
    expected_counts, expected_edges = np.histogram(source["mc"], bins=32)
    np.testing.assert_array_equal(counts, expected_counts)
    np.testing.assert_allclose(edges, expected_edges)


def test_chunked_min_max_matches_numpy(epos_file):
    path, source = epos_file
    with lazy_io.open_epos(path) as table:
        lo, hi = lazy_io.chunked_min_max(table["TOF (ns)"], chunk_size=128)
    assert lo == pytest.approx(float(np.min(source["tof"])))
    assert hi == pytest.approx(float(np.max(source["tof"])))


def test_head_returns_dataframe(epos_file):
    path, _ = epos_file
    with lazy_io.open_epos(path) as table:
        head = table.head(5)
    assert isinstance(head, pd.DataFrame)
    assert len(head) == 5
    assert set(head.columns) == set(lazy_io.EPOS_COLUMN_NAMES.values())


# ---------------------------------------------------------------------------
# POS
# ---------------------------------------------------------------------------


def test_open_pos_round_trip(tmp_path: Path):
    n = 100
    rng = np.random.default_rng(1)
    payload = rng.uniform(-10, 10, (n, 4)).astype(">f4")
    path = tmp_path / "synth.pos"
    path.write_bytes(payload.tobytes())
    eager = leap_tools.read_pos(path)
    with lazy_io.open_pos(path) as table:
        lazy_df = table.to_dataframe()
    pd.testing.assert_frame_equal(eager, lazy_df)


def test_open_pos_empty(tmp_path: Path):
    path = tmp_path / "empty.pos"
    path.write_bytes(b"")
    with lazy_io.open_pos(path) as table:
        assert table.n_rows == 0
        df = table.to_dataframe()
    assert list(df.columns) == list(lazy_io.POS_COLUMN_NAMES)
    assert len(df) == 0


# ---------------------------------------------------------------------------
# pyccapt-raw HDF5 round-trip
# ---------------------------------------------------------------------------


def _write_synthetic_raw_hdf5(path: Path):
    import h5py

    n_dld = 5000
    n_tdc = 8000
    rng = np.random.default_rng(2)
    with h5py.File(path, "w") as fh:
        dld = fh.create_group("dld")
        dld.create_dataset("high_voltage", data=rng.uniform(1000, 8000, (n_dld, 1)).astype(np.float32))
        dld.create_dataset("t", data=rng.uniform(100, 10000, (n_dld, 1)).astype(np.float32))
        dld.create_dataset("x", data=rng.uniform(-2, 2, (n_dld, 1)).astype(np.float32))
        dld.create_dataset("y", data=rng.uniform(-2, 2, (n_dld, 1)).astype(np.float32))

        tdc = fh.create_group("tdc")
        tdc.create_dataset("channel", data=rng.integers(0, 8, n_tdc, dtype=np.int32))
        tdc.create_dataset("start_counter", data=rng.integers(0, 1000, n_tdc, dtype=np.int32))
        tdc.create_dataset("time_data", data=rng.uniform(0, 1e5, n_tdc).astype(np.float64))
    return n_dld, n_tdc


def test_lazy_read_hdf5_basic(tmp_path: Path):
    path = tmp_path / "synth-raw.h5"
    n_dld, n_tdc = _write_synthetic_raw_hdf5(path)
    table = data_tools.read_hdf5(path, lazy=True)
    try:
        assert isinstance(table, lazy_io.LazyTable)
        cols = set(table.columns)
        assert {"dld/high_voltage", "dld/t", "dld/x", "dld/y"} <= cols
        assert {"tdc/channel", "tdc/start_counter", "tdc/time_data"} <= cols
        # iter_chunks works with mixed-length groups when we select one group.
        for chunk in table.iter_chunks(["dld/t"], chunk_size=1024):
            assert chunk["dld/t"].dtype == np.float32
    finally:
        table.close()


def test_lazy_read_hdf5_chunked_histogram(tmp_path: Path):
    path = tmp_path / "synth-raw.h5"
    _write_synthetic_raw_hdf5(path)
    with data_tools.read_hdf5(path, lazy=True) as table:
        counts, edges = lazy_io.chunked_histogram(table["dld/t"], bins=20, chunk_size=512)
        # The lazy histogram should agree with numpy on the same column.
        expected = np.asarray(table["dld/t"])
        ref_counts, ref_edges = np.histogram(expected, bins=20)
    np.testing.assert_array_equal(counts, ref_counts)
    np.testing.assert_allclose(edges, ref_edges)


# ---------------------------------------------------------------------------
# Read paths preserve old behavior
# ---------------------------------------------------------------------------


def test_read_epos_returns_same_dtype_as_before(epos_file):
    path, _ = epos_file
    df = leap_tools.read_epos(path)
    # Float32 / uint32 contract preserved.
    assert df["x (nm)"].dtype == np.float32
    assert df["pslep"].dtype == np.uint32
