"""Tests for paper-style round-number axis finalisation."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyccapt.calibration.core import plot_style

pytestmark = pytest.mark.calibration


def test_nice_bounds_examples():
    lo, hi, ticks = plot_style.nice_bounds(1, 9)
    assert (lo, hi) == (0.0, 10.0)
    assert ticks[0] == 0.0 and ticks[-1] == 10.0

    lo, hi, _ = plot_style.nice_bounds(0.013, 0.087)
    assert (lo, hi) == (0.0, 0.1)

    lo, hi, _ = plot_style.nice_bounds(-19.1, 23.1)
    assert lo <= -19.1 and hi >= 23.1
    assert lo == -20.0 and hi == 30.0


def test_nice_bounds_degenerate():
    assert plot_style.nice_bounds(5, 5) is None
    assert plot_style.nice_bounds(np.nan, 1) is None


def test_finalize_axes_line_plot_gets_end_ticks():
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, 10), np.arange(1, 10))
    plot_style.finalize_axes(ax)
    x0, x1 = ax.get_xlim()
    xt = ax.get_xticks()
    assert (x0, x1) == (0.0, 10.0)
    # first and last ticks sit exactly on the axis ends
    assert xt[0] == pytest.approx(x0)
    assert xt[-1] == pytest.approx(x1)
    plt.close(fig)


def test_finalize_skips_image_axes():
    fig, ax = plt.subplots()
    ax.imshow(np.random.rand(20, 20), extent=[0.1, 9.9, 0.1, 9.9])
    before = (ax.get_xlim(), ax.get_ylim())
    plot_style.finalize_axes(ax)
    assert (ax.get_xlim(), ax.get_ylim()) == before  # imshow axes untouched
    plt.close(fig)


def test_finalize_skips_log_axis():
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, 10), np.arange(1, 10))
    ax.set_yscale('log')
    y_before = ax.get_ylim()
    plot_style.finalize_axes(ax)
    assert ax.get_ylim() == y_before          # log y untouched
    assert ax.get_xlim() == (0.0, 10.0)        # linear x still snapped
    plt.close(fig)


def test_global_install_styles_show_and_savefig(tmp_path):
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    orig_show, orig_savefig = plt.show, Figure.savefig
    try:
        plot_style.install_global_paper_style()
        assert plt.show is not orig_show
        fig, ax = plt.subplots()
        ax.plot(np.arange(1, 10), np.arange(1, 10))
        fig.savefig(tmp_path / "g.png")          # wrapped savefig finalizes first
        assert ax.get_xlim() == (0.0, 10.0)
        plt.close(fig)
    finally:
        plot_style.uninstall_global_paper_style()
        assert plt.show is orig_show
        assert Figure.savefig is orig_savefig


def test_finalize_does_not_change_data():
    fig, ax = plt.subplots()
    counts, edges, _ = ax.hist(np.arange(1, 10), bins=9)
    plot_style.finalize_axes(ax)
    counts2 = [p.get_height() for p in ax.patches]
    assert list(counts) == counts2  # bar heights (data) unchanged by styling
    plt.close(fig)
