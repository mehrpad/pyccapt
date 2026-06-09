"""Tests for hdf5_creator._write_chunked_dataset (CC4).

The chunked writer assembles per-acquisition .npy chunk files into one
HDF5 dataset. The CC4 change caches chunk sizes (no double file open),
streams each chunk via mmap, and asserts the chunk dtype matches the
destination so a mismatched chunk (e.g. an interrupted run restarted
under new code) raises instead of being silently cast.
"""
import h5py
import numpy as np
import pytest

from pyccapt.control.core import hdf5_creator


def _write_chunk(tmp_path, idx, array):
    p = tmp_path / f"chunk_{idx}.npy"
    np.save(p, array)
    return p


def test_chunked_dataset_concatenates_in_order(tmp_path):
    chunks = [
        np.arange(0, 5, dtype=np.float32),
        np.arange(5, 12, dtype=np.float32),
        np.arange(12, 14, dtype=np.float32),
    ]
    files = [_write_chunk(tmp_path, i, c) for i, c in enumerate(chunks)]
    expected = np.concatenate(chunks)

    out_path = tmp_path / "out.h5"
    with h5py.File(out_path, "w") as f:
        hdf5_creator._write_chunked_dataset(f, "data", files, np.float32)

    with h5py.File(out_path, "r") as f:
        got = f["data"][...]
    assert got.shape == expected.shape
    assert np.array_equal(got, expected)


def test_chunked_dataset_rejects_dtype_mismatch(tmp_path):
    # A chunk saved as float64 while the destination dataset is float32
    # must raise rather than silently down-cast acquisition data.
    files = [
        _write_chunk(tmp_path, 0, np.arange(0, 4, dtype=np.float32)),
        _write_chunk(tmp_path, 1, np.arange(4, 8, dtype=np.float64)),  # wrong dtype
    ]
    out_path = tmp_path / "out.h5"
    with h5py.File(out_path, "w") as f:
        with pytest.raises(ValueError, match="dtype"):
            hdf5_creator._write_chunked_dataset(f, "data", files, np.float32)


def test_chunked_dataset_empty_chunk_list(tmp_path):
    out_path = tmp_path / "out.h5"
    with h5py.File(out_path, "w") as f:
        hdf5_creator._write_chunked_dataset(f, "data", [], np.float32)
        assert f["data"].shape == (0,)
