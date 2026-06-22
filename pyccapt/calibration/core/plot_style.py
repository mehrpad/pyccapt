"""Paper-quality axis styling.

The goal: every linear plot axis should span "round" numbers with the **first
and last tick labelled at the corners** (e.g. data 1..9 -> axis 0..10 with ticks
0, 2, 4, 6, 8, 10), the way figures are usually drawn for publication.

The main entry points are :func:`finalize_axes` / :func:`finalize_figure`, which
snap the current view limits to round numbers and place ticks at both ends.
:func:`save_figure` (``pyccapt.calibration.path_utils``) calls
:func:`finalize_figure` on every figure it writes, so saved/paper figures get
this automatically.

Safety: snapping is **purely cosmetic** (only view limits + tick locations -- the
underlying data, histogram counts and peak positions are untouched) and is
skipped for axes where changing the limits would be wrong:

* log-scaled axes (a round *linear* grid is meaningless),
* axes containing an image / ``imshow`` (FDM, density maps -- snapping would crop),
* equal-aspect / fixed-aspect axes (maps -- snapping would distort).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "nice_bounds",
    "finalize_axes",
    "finalize_figure",
    "apply_paper_style",
    "install_global_paper_style",
    "uninstall_global_paper_style",
]

_INSTALLED = False
_ORIG = {}


def nice_bounds(vmin, vmax, max_ticks=6):
    """Round (lo, hi, ticks) bracketing [vmin, vmax] with ticks at both ends.

    Returns ``None`` when the range is degenerate (non-finite or zero width),
    signalling "leave this axis alone".
    """
    from matplotlib.ticker import MaxNLocator

    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmax <= vmin:
        return None
    locator = MaxNLocator(nbins=max_ticks, steps=[1, 2, 2.5, 5, 10])
    ticks = locator.tick_values(vmin, vmax)
    if len(ticks) < 2:
        return None
    step = ticks[1] - ticks[0]
    if step <= 0:
        return None
    lo = np.floor(vmin / step) * step
    hi = np.ceil(vmax / step) * step
    n = int(round((hi - lo) / step)) + 1
    ticks = lo + step * np.arange(n)
    return float(lo), float(hi), ticks


def _axis_is_map_like(ax):
    """True for image / equal-aspect axes that must not have their limits snapped."""
    if ax.images:
        return True
    try:
        if ax.get_aspect() != 'auto':
            return True
    except Exception:
        pass
    return False


def _data_interval(ax, which):
    """(min, max) of the actually-plotted data along ``which`` axis, or None.

    Uses ``ax.dataLim`` (the bounding box matplotlib maintains over every
    artist added to the axes), so it reflects where the real data is -- not the
    view limits, which the caller may have set wider (a fixed max_tof, autoscale
    padding, ...). Returns ``None`` when there is no finite data extent.
    """
    try:
        iv = ax.dataLim.intervalx if which == 'x' else ax.dataLim.intervaly
        lo, hi = float(iv[0]), float(iv[1])
    except Exception:
        return None
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return None
    return lo, hi


def _effective_interval(view, data):
    """Tighter of the view and data intervals.

    When the view is WIDER than the data (the "round number sitting in empty
    space" case), hug the data; when the caller has zoomed in TIGHTER than the
    data, respect the zoom. Falls back to the view when there is no data extent
    or the two do not overlap.
    """
    v_lo, v_hi = (view[0], view[1]) if view[0] <= view[1] else (view[1], view[0])
    if data is None:
        return v_lo, v_hi
    d_lo, d_hi = data
    lo = max(v_lo, d_lo)
    hi = min(v_hi, d_hi)
    if hi <= lo:
        return v_lo, v_hi
    return lo, hi


def _snap_axis(ax, which, max_ticks):
    """Snap one linear axis to round bounds that hug the data, preserving any
    inverted orientation."""
    view = ax.get_xlim() if which == 'x' else ax.get_ylim()
    inverted = view[0] > view[1]
    lo0, hi0 = _effective_interval(view, _data_interval(ax, which))
    bounds = nice_bounds(lo0, hi0, max_ticks=max_ticks)
    if bounds is None:
        return
    lo, hi, ticks = bounds
    limits = (hi, lo) if inverted else (lo, hi)
    if which == 'x':
        ax.set_xlim(*limits)
        ax.set_xticks(ticks)
    else:
        ax.set_ylim(*limits)
        ax.set_yticks(ticks)


def finalize_axes(ax=None, x=True, y=True, max_ticks=6):
    """Snap an axes' linear x/y limits to round numbers with end ticks.

    The round bounds hug the actual data extent (``ax.dataLim``), so a forced
    round last tick never floats in a large empty region when the view is wider
    than the data. No-op for map-like axes; per-axis no-op for log scales.
    Returns ``ax``.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()
    if _axis_is_map_like(ax):
        return ax
    if x and ax.get_xscale() == 'linear':
        _snap_axis(ax, 'x', max_ticks)
    if y and ax.get_yscale() == 'linear':
        _snap_axis(ax, 'y', max_ticks)
    return ax


def finalize_figure(fig=None, **kwargs):
    """Apply :func:`finalize_axes` to every axes of a figure (guarded)."""
    import matplotlib.pyplot as plt

    if fig is None:
        fig = plt.gcf()
    for ax in fig.get_axes():
        try:
            finalize_axes(ax, **kwargs)
        except Exception:
            # Cosmetic styling must never break the caller (e.g. a save).
            pass
    return fig


def apply_paper_style():
    """Set baseline rcParams so on-screen auto-scaled axes prefer round limits.

    This nudges *auto-scaled* axes toward round-number limits with end ticks
    (snapping to the data extremes). For the full "0..10 bracket" behaviour on a
    specific figure use :func:`finalize_figure`; saved figures get it for free
    via ``save_figure``.
    """
    import matplotlib as mpl

    mpl.rcParams['axes.autolimit_mode'] = 'round_numbers'
    mpl.rcParams['axes.xmargin'] = 0.0
    mpl.rcParams['axes.ymargin'] = 0.0


def install_global_paper_style():
    """Apply round-number axes to **every** figure, on-screen and saved.

    Wraps ``pyplot.show`` and ``Figure.savefig`` so that linear axes are snapped
    once at display/save time (not on every redraw, so zoom/pan still work). Call
    it once at the top of a notebook or script. Idempotent; set the environment
    variable ``PYCCAPT_NO_PAPER_STYLE`` to opt out entirely.
    """
    global _INSTALLED
    import os

    if _INSTALLED or os.environ.get("PYCCAPT_NO_PAPER_STYLE"):
        return
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    apply_paper_style()
    _ORIG['show'] = plt.show
    _ORIG['savefig'] = Figure.savefig

    def _show(*args, **kwargs):
        try:
            for num in plt.get_fignums():
                finalize_figure(plt.figure(num))
        except Exception:
            pass
        return _ORIG['show'](*args, **kwargs)

    def _savefig(self, *args, **kwargs):
        try:
            finalize_figure(self)
        except Exception:
            pass
        return _ORIG['savefig'](self, *args, **kwargs)

    plt.show = _show
    Figure.savefig = _savefig
    _INSTALLED = True


def uninstall_global_paper_style():
    """Undo :func:`install_global_paper_style` (restore matplotlib's show/savefig)."""
    global _INSTALLED
    if not _INSTALLED:
        return
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    plt.show = _ORIG['show']
    Figure.savefig = _ORIG['savefig']
    _INSTALLED = False
