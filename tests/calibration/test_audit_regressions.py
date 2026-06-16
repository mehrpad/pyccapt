"""Regression tests for the May-2026 audit fixes.

Each test pins down a specific bug that was identified and fixed during
the comprehensive audit pass; the comment on each test cites the
commit that introduced the fix so future bisection can find the
context quickly.
"""
import numpy as np
import pandas as pd
import pytest

from pyccapt.calibration.core import mc_plot, calibration as cal
from pyccapt.calibration.core.share_variables import Variables
from pyccapt.calibration.clustering import clustering as clst


# ---------------------------------------------------------------------------
# C1: AptHistPlotter.plot_histogram bin width
# Commit 4a2ddcc -- previously np.linspace(min, max, round(max/bin_width))
# ignored ``min`` and used N samples, so the actual bin width was
# (max-min)/(N-1) rather than bin_width whenever min > 0.
# ---------------------------------------------------------------------------
def test_apt_hist_plotter_bin_width_is_exact_for_offset_data():
    rng = np.random.default_rng(7)
    # Data centred around 100 Da: well above zero, so the old buggy
    # formula would give a wildly different bin width than requested.
    data = rng.normal(100.0, 0.2, 20000)
    variables = Variables()

    bin_width = 0.05
    plotter = mc_plot.AptHistPlotter(data, variables)
    plotter.plot_histogram(bin_width=bin_width, plot_show=False, fast=True)

    diffs = np.diff(plotter.bins)
    assert np.allclose(diffs, bin_width, rtol=0, atol=1e-12), (
        f"Histogram bin width {diffs.mean():.6f} disagrees with the "
        f"requested {bin_width}."
    )


def test_apt_hist_plotter_rejects_degenerate_range():
    """Pre-fix: linspace silently produced a 0-length or 1-length edges
    array, then peak finding crashed downstream. Now we raise loudly."""
    variables = Variables()
    plotter = mc_plot.AptHistPlotter(np.array([5.0, 5.0, 5.0]), variables)
    with pytest.raises(ValueError):
        plotter.plot_histogram(bin_width=0.01, plot_show=False, fast=True)


# ---------------------------------------------------------------------------
# C2: bin centers (not edges) for histogram peak readings
# Commit 4a2ddcc -- bins[peaks[i]] returned the LEFT edge, biasing every
# reference peak reading low by half a bin width.
# ---------------------------------------------------------------------------
def test_resolve_peak_location_returns_bin_center_not_edge():
    """For a tight peak centered at 27.5, with bin_size 0.5, the previous
    bug returned 27.0 (left edge of the maximum bin). The fix returns
    27.25 ± half a bin."""
    rng = np.random.default_rng(11)
    data = rng.normal(27.5, 0.05, 20000)

    found = cal._resolve_peak_location(data, method='histogram', bin_size=0.5)

    # The TRUE peak is at 27.5. The fixed centre estimate must land
    # closer than half a bin (0.25) on the high side; the legacy
    # edge-based estimate would land at 27.0 (0.5 below truth), which
    # this assertion catches.
    assert abs(found - 27.5) <= 0.30, (
        f"Histogram peak estimate {found:.3f} too far from truth 27.5; "
        f"is the bin-center (not bin-edge) being returned?"
    )


# ---------------------------------------------------------------------------
# C4: sync_from_data resets initial_calibration_done_{mc,tof} flags
# Commit 1b7e256 -- flags persisted across datasets, causing the auto-cal
# helpers to skip initial calibration on the new dataset.
# ---------------------------------------------------------------------------
def test_sync_from_data_resets_initial_calibration_flags():
    variables = Variables()
    variables.initial_calibration_done_mc = True
    variables.initial_calibration_done_tof = True

    # Build a minimal dataframe that sync_from_data is happy to consume.
    df = pd.DataFrame({
        "high_voltage (V)": np.array([1000.0, 1100.0, 1200.0]),
        "pulse_v (V)": np.array([100.0, 110.0, 120.0]),
        "t (ns)": np.array([100.0, 200.0, 300.0]),
        "mc (Da)": np.array([10.0, 20.0, 30.0]),
        "x_det (cm)": np.array([0.0, 0.1, -0.1]),
        "y_det (cm)": np.array([0.0, -0.1, 0.1]),
        "x (nm)": np.array([0.0, 1.0, -1.0]),
        "y (nm)": np.array([0.0, -1.0, 1.0]),
        "z (nm)": np.array([0.0, 1.0, -1.0]),
    })

    variables.sync_from_data(df, update_backups=True)

    assert variables.initial_calibration_done_mc is False, (
        "sync_from_data must reset the per-mode initial-calibration flag; "
        "stale True forces downstream code to skip initial calibration "
        "after a dataset reload."
    )
    assert variables.initial_calibration_done_tof is False


# ---------------------------------------------------------------------------
# H10: _resolve_peak_location fast_calibration subsample is deterministic
# Commit 1b7e256 -- the previous np.random.choice (global RNG) made
# fast-mode calibrations non-reproducible across runs.
# ---------------------------------------------------------------------------
def test_fast_calibration_subsample_is_reproducible_across_calls():
    rng = np.random.default_rng(13)
    data = rng.normal(50.0, 0.1, 5000)

    a = cal._resolve_peak_location(
        data, method='histogram', bin_size=0.05, fast_calibration=True
    )
    b = cal._resolve_peak_location(
        data, method='histogram', bin_size=0.05, fast_calibration=True
    )

    assert a == b, (
        f"Fast-mode peak location {a} != {b}; "
        f"_resolve_peak_location is using a non-seeded RNG."
    )


# ---------------------------------------------------------------------------
# C6: hdbscan_clustering returns full-length labels even when NaN rows drop
# Commit 474c179 -- previously the returned labels had the FILTERED length,
# which broke ``full_labels[selected_indices] = labels`` for callers with
# any partial-recovered (NaN x/y/z) ions.
# ---------------------------------------------------------------------------
def test_hdbscan_clustering_labels_align_with_input_when_nan_rows_present():
    # Mix of dense-cluster points + a few NaN-coord partial-recovered rows.
    rng = np.random.default_rng(17)
    cluster_a = rng.normal(loc=(0.0, 0.0, 0.0), scale=0.1, size=(30, 3))
    cluster_b = rng.normal(loc=(5.0, 5.0, 5.0), scale=0.1, size=(30, 3))
    nan_rows = np.full((4, 3), np.nan)
    points = np.vstack([cluster_a, nan_rows, cluster_b])

    labels, centers, params = clst.hdbscan_clustering(
        points,
        n_min=5,
        min_samples=3,
        auto_d_max=True,
    )

    assert len(labels) == len(points), (
        f"hdbscan_clustering returned {len(labels)} labels for "
        f"{len(points)} input rows; must match the input length so "
        f"callers can index by selection mask."
    )
    # NaN rows must be flagged as noise (label = -1).
    assert (labels[30:34] == -1).all(), (
        "NaN-coord rows must receive label -1 (noise) so the rest of "
        "the labels line up with the original input order."
    )


# ---------------------------------------------------------------------------
# C5: pos_to_voxel returns vox in [ix, iy, iz] order (not transposed)
# Commit 65cbaef -- the transposed return mis-aligned voxel concentration
# values with the world coordinates, so iso-surfaces appeared in the
# wrong place relative to the underlying scatter.
# ---------------------------------------------------------------------------
def test_pos_to_voxel_returns_ij_axis_order():
    from pyccapt.calibration.reconstructions import iso_surface as iso

    # Three points sitting in distinguishable bins along x, y, z.
    # Build a synthetic DataFrame so the species-mask branch is exercised.
    data = pd.DataFrame({
        "x (nm)": [0.5, 5.5, 0.5],
        "y (nm)": [0.5, 0.5, 5.5],
        "z (nm)": [0.5, 0.5, 0.5],
    })
    bin = 1.0
    grid_vec = [
        np.arange(0.5, 10.5, bin),  # 10 bins along x
        np.arange(0.5, 10.5, bin),
        np.arange(0.5, 10.5, bin),
    ]
    vox = iso.pos_to_voxel(data, grid_vec)

    # The voxel grid is indexed [ix, iy, iz]; we expect counts at
    # (0, 0, 0), (5, 0, 0), (0, 5, 0).
    assert vox[0, 0, 0] == 1, "voxel (0, 0, 0) should hold one ion"
    assert vox[5, 0, 0] == 1, "voxel (5, 0, 0) should hold one ion"
    assert vox[0, 5, 0] == 1, "voxel (0, 5, 0) should hold one ion"
    # Confirm the previously-transposed slot is NOT hit (would catch a
    # future regression that re-introduces ``return vox.T``).
    assert vox[0, 0, 5] == 0, (
        "voxel (0, 0, 5) must be empty; if non-zero the axis order has "
        "been re-transposed."
    )
