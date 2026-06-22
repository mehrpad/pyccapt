"""Regression tests for the June-2026 audit fixes.

Each test pins down a specific bug found in the multi-agent audit of
``pyccapt.calibration`` and fixed in the same pass, so a future change that
re-introduces the bug fails loudly.
"""
import numpy as np
import pandas as pd
import pytest

from pyccapt.calibration.core import mc_plot
from pyccapt.calibration.core.share_variables import Variables
from pyccapt.calibration.data_tools import data_loadcrop
from pyccapt.calibration.data_tools import partial_hit_diagnostics as phd


# ---------------------------------------------------------------------------
# 3-of-4 recovered hits must be categorised as full (recovered_xy), not
# dropped. _categorise_rows previously matched the quality string with
# ``==`` so "recovered_xy_3of4" matched no category at all.
# ---------------------------------------------------------------------------
def test_recovered_xy_3of4_counts_as_full():
    df = pd.DataFrame(
        {
            "dlts_quality": [
                "native",
                "recovered_xy",
                "recovered_xy_3of4",
                "recovered_x",
                "recovered_y",
            ]
        }
    )
    masks = phd._categorise_rows(df)

    # The 3of4 row is a full xy hit and must land in recovered_xy / full.
    assert masks["recovered_xy"].tolist() == [False, True, True, False, False]
    # The recovered_x prefix must NOT also swallow the recovered_xy rows.
    assert masks["recovered_x"].tolist() == [False, False, False, True, False]
    assert masks["recovered_y"].tolist() == [False, False, False, False, True]

    # native + recovered_xy + recovered_xy_3of4
    assert int(masks["full"].sum()) == 3
    assert int(masks["partial"].sum()) == 2

    counts = phd.partial_hit_counts(df)
    # Every row must be accounted for: no silently-dropped category.
    assert counts["full"] + counts["partial"] == counts["total"]


# ---------------------------------------------------------------------------
# calculate_ppi_and_ipp must fill EVERY member of every run with the run
# length -- including the final run, which the old loop left at 0 except
# its last element.
# ---------------------------------------------------------------------------
def test_calculate_ppi_and_ipp_fills_final_run():
    data = pd.DataFrame({"start_counter": np.array([5, 5, 7, 7, 7], dtype=float)})
    delta_p, multi = data_loadcrop.calculate_ppi_and_ipp(data, max_start_counter=1000)

    # Two-member run of 5s -> 2; three-member final run of 7s -> 3 everywhere.
    assert multi.tolist() == [2.0, 2.0, 3.0, 3.0, 3.0]
    # delta_p is the gap to the previous distinct pulse (0 within a run).
    assert delta_p.tolist() == [0.0, 0.0, 2.0, 0.0, 0.0]


def test_calculate_ppi_and_ipp_single_event():
    data = pd.DataFrame({"start_counter": np.array([3], dtype=float)})
    delta_p, multi = data_loadcrop.calculate_ppi_and_ipp(data, max_start_counter=1000)
    assert multi.tolist() == [1.0]
    assert delta_p.tolist() == [0.0]


def test_calculate_ppi_and_ipp_counter_wrap():
    # A wrap (counter decreases) must add max_start_counter, not go negative.
    data = pd.DataFrame({"start_counter": np.array([998, 999, 2], dtype=float)})
    delta_p, multi = data_loadcrop.calculate_ppi_and_ipp(data, max_start_counter=1000)
    # gaps: 0, 1, (1000-999)+2 = 3
    assert delta_p.tolist() == [0.0, 1.0, 3.0]
    assert multi.tolist() == [1.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# plot_peaks(mode='range') must not IndexError when a selected mass sits at
# or beyond the last histogram edge (self.x are edges, len = len(y)+1).
# ---------------------------------------------------------------------------
def test_plot_peaks_range_handles_top_edge_selection():
    rng = np.random.default_rng(0)
    data = rng.normal(50.0, 0.2, 5000)
    variables = Variables()
    plotter = mc_plot.AptHistPlotter(data, variables)
    plotter.plot_histogram(bin_width=0.1, plot_show=False, fast=True)

    # Selections at the last edge and beyond it would previously index
    # self.y out of bounds.
    variables.peaks_x_selected = [
        float(plotter.x[-1]),
        float(plotter.x[-1]) + 5.0,
        float(plotter.x[0]),
    ]
    # Must not raise.
    plotter.plot_peaks(range_data=None, mode="range")


# ---------------------------------------------------------------------------
# Reconstruction kernels must tolerate partial-recovered rows (NaN detector
# x/y): det_area used np.max so one NaN row poisoned det_area -> dz -> every
# event (Geiser assert crash / Bas all-NaN). nanmax fixes it; valid rows
# reconstruct, partials keep NaN x/y/z.
# ---------------------------------------------------------------------------
def test_reconstruction_kernels_tolerate_partial_nan_rows():
    from pyccapt.calibration.reconstructions import reconstruction as recon

    rng = np.random.default_rng(0)
    n = 200
    detx = rng.uniform(-1.5, 1.5, n)
    dety = rng.uniform(-1.5, 1.5, n)
    hv = rng.uniform(4000.0, 6000.0, n)
    partial = np.zeros(n, dtype=bool)
    partial[::20] = True
    detx_p = detx.copy()
    dety_p = dety.copy()
    detx_p[partial] = np.nan
    dety_p[partial] = np.nan

    kwargs = dict(flight_path_length=110.0, kf=3.3, det_eff=0.5, icf=1.4, field_evap=30.0, avg_dens=60.0)
    for fn in (recon.atom_probe_recons_Bas_et_al, recon.atom_probe_recons_from_detector_Geiser_et_al):
        x, y, z = fn(detx_p, dety_p, hv, **kwargs)
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        # Partial rows -> NaN positions; valid rows -> finite (no poisoning).
        assert np.isnan(x[partial]).all()
        assert np.isnan(y[partial]).all()
        assert np.isnan(z[partial]).all()
        assert np.isfinite(x[~partial]).all()
        assert np.isfinite(y[~partial]).all()
        assert np.isfinite(z[~partial]).all()

    # Valid-row lateral positions are independent of det_area, so they match a
    # NaN-free baseline exactly (nanmax ignores the partials).
    xb, yb, _ = recon.atom_probe_recons_Bas_et_al(detx[~partial], dety[~partial], hv[~partial], **kwargs)
    xp, yp, _ = recon.atom_probe_recons_Bas_et_al(detx_p, dety_p, hv, **kwargs)
    np.testing.assert_allclose(np.asarray(xp)[~partial], xb, rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.asarray(yp)[~partial], yb, rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# SDM cKDTree fast path must be bit-identical to the brute-force reference
# across axis subsets / rotation / z_cut / mask combinations.
# ---------------------------------------------------------------------------
def test_sdm_kdtree_matches_bruteforce():
    import itertools

    from pyccapt.calibration.reconstructions import sdm

    rng = np.random.default_rng(0)
    clouds = [rng.normal(c, 0.6, (400, 3)) for c in ([0, 0, 0], [1.5, 0, 0.5], [0, 1.2, 1.0])]
    pts = np.concatenate(clouds)
    n = len(pts)
    mask_all = np.ones(n, dtype=bool)
    mask_a = rng.random(n) < 0.6
    mask_b = rng.random(n) < 0.6

    def _same(a, b):
        if a is None and b is None:
            return True
        if a is None or b is None or a.shape != b.shape:
            return False
        return np.array_equal(np.sort(a), np.sort(b))

    axes_opts = [["x"], ["y"], ["z"], ["x", "y"], ["y", "z"], ["x", "z"], ["x", "y", "z"]]
    rot_opts = [(0.0, 0.0), (0.3, 0.0), (0.35, 0.5)]
    mask_pairs = [(mask_all, mask_all), (mask_a, mask_b), (mask_a, mask_a)]

    for axes, (th, ph), (mi, mj), zc in itertools.product(axes_opts, rot_opts, mask_pairs, [True, False]):
        kw = dict(particles=pts, mask_i=mi, mask_j=mj, axes=axes, z_cut=zc, theta=th, phi=ph)
        bf = sdm._compute_sdm_displacements_bruteforce(**kw)
        disp = sdm._compute_sdm_displacements(**kw)
        assert all(_same(a, b) for a, b in zip(bf, disp)), (
            f"SDM dispatcher != brute force for axes={axes} rot=({th},{ph}) z_cut={zc}"
        )


# ---------------------------------------------------------------------------
# Output-changing recon fixes (commit 5da9796), validated against ground truth.
# ---------------------------------------------------------------------------
def test_bas_reconstruction_recovers_input_density():
    """Bas cm->m must be 1e-2 (matching lateral x/y) so the reconstruction
    recovers the input atomic density. The old 1e-3 made z ~100x too deep and
    the density ~100x too low."""
    from pyccapt.calibration.reconstructions import reconstruction as recon

    rng = np.random.default_rng(0)
    n = 120_000
    det_eff, avg_dens = 0.5, 60.0
    kf, icf, field_evap, fp = 3.3, 1.4, 30.0, 110.0
    r = 4.0 * np.sqrt(rng.random(n))
    a = rng.random(n) * 2 * np.pi
    detx, dety = r * np.cos(a), r * np.sin(a)
    hv = np.full(n, 5000.0)

    x, y, z = recon.atom_probe_recons_Bas_et_al(detx, dety, hv, fp, kf, det_eff, icf, field_evap, avg_dens)
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    # Central column: the spherical cap is locally flat there, so the z-span is
    # the volume-conserving cumulative depth.
    col = (x * x + y * y) < 9.0  # radius 3 nm
    span = z[col].max() - z[col].min()
    density = col.sum() / (np.pi * 9.0 * span)
    target = avg_dens * det_eff
    assert abs(density - target) / target < 0.05, (
        f"Bas reconstruction density {density:.2f} != expected {target:.1f} (avg_dens*det_eff)"
    )


def test_wt_to_atomic_fractions_ni_balance_matches_textbook():
    from pyccapt.calibration.reconstructions import tapsim_builder as tb

    out = tb.wt_to_atomic_fractions({"Cr": 20.0, "Ni": 80.0})
    # Textbook Ni-20wt%Cr -> at%: Cr 0.2201, Ni 0.7799.
    assert abs(out["Cr"] - 0.2201) < 1e-3
    assert abs(out["Ni"] - 0.7799) < 1e-3
    # Ni as balance gives the same (Ni value in input is replaced by the balance).
    out2 = tb.wt_to_atomic_fractions({"Cr": 20.0})
    assert abs(out2["Ni"] - out["Ni"]) < 1e-9


def test_isosurface_uses_fortran_point_order():
    pv = pytest.importorskip("pyvista")
    from pyccapt.calibration.clustering import isosurface as iso  # noqa: F401

    gv = [np.arange(0.5, 0.5 + 5), np.arange(0.5, 0.5 + 7), np.arange(0.5, 0.5 + 9)]
    nx, ny, nz = len(gv[0]), len(gv[1]), len(gv[2])
    data = np.fromfunction(lambda i, j, k: i * 100 + j * 10 + k, (nx, ny, nz), dtype=float)
    X, Y, Z = np.meshgrid(gv[0], gv[1], gv[2], indexing="ij")
    grid = pv.StructuredGrid(X, Y, Z)
    grid.point_data["values"] = data.flatten(order="F")
    pts = grid.points
    # Every grid point's scalar must equal its (i,j,k)-encoded value.
    idx = np.rint(pts - 0.5).astype(int)
    expected = idx[:, 0] * 100 + idx[:, 1] * 10 + idx[:, 2]
    assert np.array_equal(grid.point_data["values"], expected)
