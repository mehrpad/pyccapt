"""Tests for the HDF5 schema source-of-truth + shape-robust reader (IO2).

The reader previously assembled per-event columns with
``np.concatenate(axis=1)``, which only works if every column is 2-D
(N, 1). hdf5_schema.stack_columns normalises each column to 1-D first,
so the reader now loads files whose /dld or /tdc datasets were written
either 1-D (N,) or 2-D (N, 1). These tests pin both the helper and an
end-to-end load of a 1-D-dataset file.
"""
from pathlib import Path

import h5py
import numpy as np
import pytest

from pyccapt.calibration.data_tools import hdf5_schema
from pyccapt.calibration.data_tools import data_loadcrop


def test_stack_columns_handles_1d_and_2d_identically():
    a1 = np.arange(6, dtype=float)
    b1 = np.arange(6, dtype=float) * 2
    r1 = hdf5_schema.stack_columns((a1, b1))
    r2 = hdf5_schema.stack_columns((a1.reshape(-1, 1), b1.reshape(-1, 1)))
    assert r1.shape == (6, 2)
    assert np.array_equal(r1, r2)


def test_stack_columns_rejects_length_mismatch():
    with pytest.raises(ValueError, match="inconsistent"):
        hdf5_schema.stack_columns((np.arange(6), np.arange(5)))


def test_schema_column_order_matches_reader():
    # The schema constants must mirror create_pandas_dataframe's column
    # order so the documented contract stays truthful.
    assert hdf5_schema.DLD_COLUMNS == (
        "high_voltage (V)", "pulse_v (V)", "pulse_l (pJ)", "start_counter",
        "t (ns)", "x_det (cm)", "y_det (cm)",
    )
    assert hdf5_schema.TDC_COLUMNS == (
        "channel", "start_counter", "high_voltage (V)", "pulse_v (V)",
        "pulse_l (pJ)", "time_data",
    )


def test_reader_loads_1d_dld_datasets(tmp_path: Path):
    # Write a /dld group with 1-D datasets (shape (N,), not (N, 1)). The
    # old np.concatenate(axis=1) would raise AxisError on these; the
    # schema-based stack must load them cleanly.
    path = tmp_path / "dld_1d.h5"
    n = 5
    with h5py.File(path, "w") as hdf:
        dld = hdf.create_group("dld")
        dld.create_dataset("high_voltage", data=np.full(n, 1000.0))      # 1-D
        dld.create_dataset("voltage_pulse", data=np.full(n, 200.0))      # 1-D
        dld.create_dataset("laser_intensity", data=np.zeros(n))          # 1-D
        dld.create_dataset("start_counter", data=np.arange(n, dtype=np.int64))
        dld.create_dataset("t", data=np.linspace(100.0, 200.0, n))
        dld.create_dataset("x", data=np.full(n, 0.5))
        dld.create_dataset("y", data=np.full(n, -0.5))

    df = data_loadcrop.fetch_dataset_from_dld_grp(str(path), extract_mode="dld")
    assert df is not None
    assert len(df) == n
    # Column order + values survive the 1-D load.
    assert list(df.columns)[:7] == list(hdf5_schema.DLD_COLUMNS)
    assert np.allclose(df["high_voltage (V)"].to_numpy(), 1000.0)
    assert np.allclose(df["t (ns)"].to_numpy(), np.linspace(100.0, 200.0, n))


def test_reader_loads_canonical_dld_laser_pulse_alias(tmp_path: Path):
    path = tmp_path / "dld_laser_pulse.h5"
    n = 3
    with h5py.File(path, "w") as hdf:
        dld = hdf.create_group("dld")
        dld.create_dataset("high_voltage", data=np.full(n, 1000.0))
        dld.create_dataset("voltage_pulse", data=np.full(n, 200.0))
        dld.create_dataset("laser_pulse", data=np.array([1.5, 2.5, 3.5]))
        dld.create_dataset("start_counter", data=np.arange(n, dtype=np.uint64))
        dld.create_dataset("t", data=np.linspace(100.0, 200.0, n))
        dld.create_dataset("x", data=np.full(n, 0.5))
        dld.create_dataset("y", data=np.full(n, -0.5))

    df = data_loadcrop.fetch_dataset_from_dld_grp(str(path), extract_mode="dld")

    assert np.allclose(df["pulse_l (pJ)"].to_numpy(), [1.5, 2.5, 3.5])


def test_reader_preserves_uint64_counters_and_tdc_times(tmp_path: Path):
    path = tmp_path / "uint64_raw.h5"
    big = np.uint64(2**60 + 7)
    with h5py.File(path, "w") as hdf:
        dld = hdf.create_group("dld")
        dld.create_dataset("high_voltage", data=np.array([1000.0]))
        dld.create_dataset("voltage_pulse", data=np.array([200.0]))
        dld.create_dataset("laser_pulse", data=np.array([1.0]))
        dld.create_dataset("start_counter", data=np.array([big], dtype=np.uint64))
        dld.create_dataset("t", data=np.array([100.0]))
        dld.create_dataset("x", data=np.array([0.5]))
        dld.create_dataset("y", data=np.array([-0.5]))

        tdc = hdf.create_group("tdc")
        tdc.create_dataset("channel", data=np.array([1], dtype=np.uint32))
        tdc.create_dataset("start_counter", data=np.array([big], dtype=np.uint64))
        tdc.create_dataset("high_voltage", data=np.array([1000.0]))
        tdc.create_dataset("voltage_pulse", data=np.array([200.0]))
        tdc.create_dataset("laser_pulse", data=np.array([1.0]))
        tdc.create_dataset("time_data", data=np.array([big], dtype=np.uint64))

    dld_df = data_loadcrop.fetch_dataset_from_dld_grp(str(path), extract_mode="dld")
    tdc_df = data_loadcrop.fetch_dataset_from_dld_grp(str(path), extract_mode="tdc_sc")

    assert int(dld_df["start_counter"].iloc[0]) == int(big)
    assert int(tdc_df["start_counter"].iloc[0]) == int(big)
    assert int(tdc_df["time_data"].iloc[0]) == int(big)
    assert str(dld_df["start_counter"].dtype) == "uint64"
    assert str(tdc_df["time_data"].dtype) == "uint64"
