"""Tests for the LazyTable-aware stat helpers in ``_raw_workflow_common``.

Verifies that ``summarize_processed_dataset``, ``compute_tof_segment_drift``
and ``plot_processed_dataset_overview`` produce the same results (within
documented streaming-approximation tolerances) whether they're called on a
materialized DataFrame or on a memory-mapped :class:`LazyTable`. Also covers
the new ``fetch_dataset_from_dld_grp(..., lazy=True)`` path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # avoid blocking on a display when running headless

import numpy as np
import pandas as pd
import pytest

from pyccapt.calibration.data_tools import data_loadcrop, lazy_io, raw_data_workflow
from pyccapt.calibration.data_tools._raw_workflow_common import (
    compute_tof_segment_drift,
    plot_processed_dataset_overview,
    summarize_processed_dataset,
)


def _make_processed_dataframe(n: int = 8000, seed: int = 11) -> pd.DataFrame:
    """Synthetic processed dataset with the columns the stat helpers expect."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            'mc (Da)': rng.normal(40.0, 10.0, n).astype(np.float32),
            't (ns)': rng.normal(800.0, 200.0, n).astype(np.float32),
            'high_voltage (V)': rng.uniform(2000, 8000, n).astype(np.float32),
            'pulse_v (V)': rng.uniform(50, 200, n).astype(np.float32),
            'x_det (cm)': rng.uniform(-3, 3, n).astype(np.float32),
            'y_det (cm)': rng.uniform(-3, 3, n).astype(np.float32),
            'multi': rng.integers(0, 4, n).astype(np.int32),
            'delta_p': rng.integers(0, 200, n).astype(np.int32),
        }
    )


def _dataframe_to_lazy(df: pd.DataFrame) -> lazy_io.LazyTable:
    """Wrap an in-memory DataFrame as a LazyTable for the stat-equivalence checks."""

    class _ArraySource:
        __slots__ = ("_arr",)

        def __init__(self, arr):
            self._arr = arr

        @property
        def dtype(self):
            return self._arr.dtype

        def __len__(self):
            return int(self._arr.size)

        def __getitem__(self, item):
            return self._arr[item]

    columns = {name: lazy_io.LazyColumn(_ArraySource(df[name].to_numpy()), name=name) for name in df.columns}
    return lazy_io.LazyTable(columns)


# ---------------------------------------------------------------------------
# summarize_processed_dataset
# ---------------------------------------------------------------------------


def test_summarize_processed_dataset_dataframe_unchanged():
    df = _make_processed_dataframe()
    eager = summarize_processed_dataset(df)
    assert eager['num_rows'] == len(df)
    assert eager['mc (Da)_min'] == pytest.approx(float(df['mc (Da)'].min()))
    assert eager['mc (Da)_max'] == pytest.approx(float(df['mc (Da)'].max()))
    assert eager['mc (Da)_median'] == pytest.approx(float(df['mc (Da)'].median()))


def test_summarize_processed_dataset_lazy_min_max_match():
    df = _make_processed_dataframe()
    eager = summarize_processed_dataset(df)
    lazy = summarize_processed_dataset(_dataframe_to_lazy(df), chunk_size=512)
    for column in ['mc (Da)', 't (ns)', 'high_voltage (V)']:
        assert lazy[f'{column}_min'] == pytest.approx(eager[f'{column}_min'])
        assert lazy[f'{column}_max'] == pytest.approx(eager[f'{column}_max'])
    # The streaming median is an approximation; check it's within bin width.
    assert abs(lazy['mc (Da)_median'] - eager['mc (Da)_median']) < 1.0


# ---------------------------------------------------------------------------
# compute_tof_segment_drift
# ---------------------------------------------------------------------------


def test_tof_drift_dataframe_and_lazy_agree():
    df = _make_processed_dataframe(n=12000)
    windows = [{'label': 'peak', 'min': 600, 'max': 1000}]
    eager = compute_tof_segment_drift(df, windows=windows, num_segments=8, max_value=2000.0)
    lazy = compute_tof_segment_drift(
        _dataframe_to_lazy(df), windows=windows, num_segments=8, max_value=2000.0, chunk_size=1024
    )
    assert list(eager.columns) == list(lazy.columns)
    # Same segments are visited and peak detection is deterministic on the
    # same data -- DataFrame and LazyTable must produce identical rows.
    pd.testing.assert_frame_equal(eager.reset_index(drop=True), lazy.reset_index(drop=True))


def test_tof_drift_lazy_requires_explicit_windows():
    df = _make_processed_dataframe(n=200)
    with pytest.raises(ValueError):
        compute_tof_segment_drift(_dataframe_to_lazy(df), num_segments=4)


# ---------------------------------------------------------------------------
# plot_processed_dataset_overview
# ---------------------------------------------------------------------------


def test_plot_overview_runs_on_lazy_table():
    df = _make_processed_dataframe(n=20000)
    lazy = _dataframe_to_lazy(df)
    fig = plot_processed_dataset_overview(
        lazy,
        mc_max=80.0,
        tof_max=2000.0,
        bin_size=1.0,
        chunk_size=4096,
    )
    assert fig is not None
    # Each panel must have actually been drawn into.
    for ax in fig.axes:
        assert ax.has_data() or ax.lines or ax.collections or ax.patches
    matplotlib.pyplot.close(fig)


# ---------------------------------------------------------------------------
# fetch_dataset_from_dld_grp(lazy=True)
# ---------------------------------------------------------------------------


def test_fetch_dataset_from_dld_grp_lazy(tmp_path: Path):
    import h5py

    path = tmp_path / 'synth-raw.h5'
    n = 4000
    rng = np.random.default_rng(3)
    with h5py.File(path, 'w') as fh:
        dld = fh.create_group('dld')
        dld.create_dataset('high_voltage', data=rng.uniform(2000, 8000, (n, 1)).astype(np.float32))
        dld.create_dataset('t', data=rng.uniform(100, 2000, (n, 1)).astype(np.float32))
        dld.create_dataset('x', data=rng.uniform(-3, 3, (n, 1)).astype(np.float32))
        dld.create_dataset('y', data=rng.uniform(-3, 3, (n, 1)).astype(np.float32))
        dld.create_dataset('pulse', data=rng.uniform(50, 200, (n, 1)).astype(np.float32))

    table = data_loadcrop.fetch_dataset_from_dld_grp(str(path), extract_mode='dld', lazy=True)
    try:
        assert isinstance(table, lazy_io.LazyTable)
        # The friendly calibration column names should be present.
        for name in ('high_voltage (V)', 't (ns)', 'x_det (cm)', 'y_det (cm)', 'pulse_v (V)'):
            assert name in table.columns
        sample = table['t (ns)'][:10]
        assert sample.shape == (10,)
    finally:
        table.close()
