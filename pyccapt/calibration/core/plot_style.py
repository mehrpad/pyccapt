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
    "set_paper_style",
    "paper_style_enabled",
]

_INSTALLED = False
_ORIG = {}
_ORIG_RCPARAMS = {}

# Master switch. When False, ``finalize_axes``/``finalize_figure`` are no-ops
# (so even ``save_figure`` leaves matplotlib's normal axes alone) and
# ``set_paper_style(False)`` also removes the global show/savefig hooks.
_PAPER_ENABLED = True


def set_paper_style(mode=True):
    """Switch plotting between paper style (round-number axes) and normal.

    Call once near the top of a notebook/script::

        from pyccapt.calibration.core import plot_style
        plot_style.set_paper_style('paper')    # or True  -> round-number axes
        plot_style.set_paper_style('normal')   # or False -> matplotlib defaults

    Affects both inline (``plt.show``) and saved figures. Returns the resolved
    boolean.
    """
    global _PAPER_ENABLED
    if isinstance(mode, str):
        enabled = mode.strip().lower() in ("paper", "on", "true", "1", "yes")
    else:
        enabled = bool(mode)
    _PAPER_ENABLED = enabled
    if enabled:
        install_global_paper_style()
    else:
        uninstall_global_paper_style()
    return enabled


def paper_style_enabled():
    """Return whether paper style is currently enabled."""
    return _PAPER_ENABLED


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
    """True for axes whose limits must NOT be snapped to round numbers.

    Covers: colorbar axes (snapping garbles the colorbar tick labels -- the
    "weird numbers below the colorbar"); image (``imshow``) axes; 2-D mesh axes
    (``pcolormesh`` / ``hist2d`` heatmaps, whose ``dataLim`` is the full bin
    grid incl. empty cells, so snapping leaves empty bands); and equal/fixed
    -aspect axes (maps -- snapping would distort).
    """
    # Colorbar axes (matplotlib labels them '<colorbar>' and/or sets _colorbar).
    if getattr(ax, "_colorbar", None) is not None:
        return True
    try:
        if ax.get_label() == "<colorbar>":
            return True
    except Exception:
        pass
    if ax.images:
        return True
    try:
        from matplotlib.collections import QuadMesh

        if any(isinstance(c, QuadMesh) for c in ax.collections):
            return True
    except Exception:
        pass
    try:
        if ax.get_aspect() != 'auto':
            return True
    except Exception:
        pass
    return False


def _shares_with_map(ax, which):
    """True if ``ax`` shares its ``which`` axis with a map-like sibling.

    A twin axis (``twinx``/``twiny``, e.g. the HV/DC-voltage overlay) shares one
    axis with the heatmap underneath it. Snapping the twin's shared axis would
    move the heatmap's limits too, reintroducing the empty bands -- so skip it.
    """
    try:
        grp = ax.get_shared_x_axes() if which == 'x' else ax.get_shared_y_axes()
        siblings = list(grp.get_siblings(ax))
    except Exception:
        return False
    return any(s is not ax and _axis_is_map_like(s) for s in siblings)


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
    if not _PAPER_ENABLED:
        return ax
    if _axis_is_map_like(ax):
        return ax
    if x and ax.get_xscale() == 'linear' and not _shares_with_map(ax, 'x'):
        _snap_axis(ax, 'x', max_ticks)
    if y and ax.get_yscale() == 'linear' and not _shares_with_map(ax, 'y'):
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
    """Set baseline rcParams for tight, paper-style axes.

    Only zeroes the auto-scale margins so axes hug the data (no padding band).
    It deliberately does NOT set ``axes.autolimit_mode='round_numbers'``: that
    rounds the autoscale UP to a coarse round number (1500 -> 2000, 24e6 ->
    25e6), which is exactly the large empty band the user sees -- and it affects
    even axes that :func:`finalize_axes` skips (heatmaps, colorbars). Round-number
    *ticks* are added by :func:`finalize_axes`, which hugs the real data extent.
    """
    import matplotlib as mpl

    for key in ('axes.xmargin', 'axes.ymargin'):
        _ORIG_RCPARAMS.setdefault(key, mpl.rcParams[key])
    mpl.rcParams['axes.xmargin'] = 0.0
    mpl.rcParams['axes.ymargin'] = 0.0


def _restore_rcparams():
    """Restore the rcParams that :func:`apply_paper_style` changed."""
    import matplotlib as mpl

    for key, value in _ORIG_RCPARAMS.items():
        mpl.rcParams[key] = value


def _is_paper_wrapper(fn):
    """True if ``fn`` is one of our show/savefig wrappers (this or a prior
    version -- matched by tag or by nested qualname)."""
    if getattr(fn, "_pyccapt_wrapper", False):
        return True
    return "install_global_paper_style.<locals>" in getattr(fn, "__qualname__", "")


def install_global_paper_style():
    """Apply round-number axes to **every** figure, on-screen and saved.

    Wraps ``pyplot.show`` and ``Figure.savefig`` so that linear axes are snapped
    once at display/save time (not on every redraw, so zoom/pan still work). Call
    it once at the top of a notebook or script. Idempotent; set the environment
    variable ``PYCCAPT_NO_PAPER_STYLE`` to opt out entirely.
    """
    global _INSTALLED
    import os

    if os.environ.get("PYCCAPT_NO_PAPER_STYLE"):
        return
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    # Stash the genuine originals on STABLE matplotlib attributes (the pyplot
    # module / Figure class). These survive reloads of THIS module -- which
    # reset _ORIG and _INSTALLED -- so the originals are never lost. The older
    # design kept them only in _ORIG; a reload wiped it while the wrappers
    # stayed bound, then every render raised ``KeyError: 'show'/'savefig'``.
    if not _is_paper_wrapper(plt.show) and not hasattr(plt, "_pyccapt_orig_show"):
        plt._pyccapt_orig_show = plt.show
    if not _is_paper_wrapper(Figure.savefig) and not hasattr(Figure, "_pyccapt_orig_savefig"):
        Figure._pyccapt_orig_savefig = Figure.savefig

    orig_show = getattr(plt, "_pyccapt_orig_show", None)
    orig_savefig = getattr(Figure, "_pyccapt_orig_savefig", None)
    if orig_show is None or orig_savefig is None:
        import warnings
        warnings.warn(
            "pyccapt paper-style: matplotlib show/savefig were left wrapped by a "
            "previous (pre-fix) session and the originals can't be recovered. "
            "Restart the kernel to enable paper style.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    apply_paper_style()

    if _INSTALLED and _is_paper_wrapper(plt.show) and _is_paper_wrapper(Figure.savefig):
        return  # already installed and intact

    def _show(*args, **kwargs):
        try:
            for num in plt.get_fignums():
                finalize_figure(plt.figure(num))
        except Exception:
            pass
        return orig_show(*args, **kwargs)

    def _savefig(self, *args, **kwargs):
        try:
            finalize_figure(self)
        except Exception:
            pass
        return orig_savefig(self, *args, **kwargs)

    for fn, orig in ((_show, orig_show), (_savefig, orig_savefig)):
        fn._pyccapt_wrapper = True
        fn._pyccapt_orig = orig
    _ORIG['show'] = orig_show
    _ORIG['savefig'] = orig_savefig
    plt.show = _show
    Figure.savefig = _savefig
    _INSTALLED = True


def uninstall_global_paper_style():
    """Undo :func:`install_global_paper_style` (restore matplotlib's show/savefig)."""
    global _INSTALLED
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    show = getattr(plt, "_pyccapt_orig_show", None) or getattr(plt.show, "_pyccapt_orig", _ORIG.get("show"))
    savefig = getattr(Figure, "_pyccapt_orig_savefig", None) or getattr(
        Figure.savefig, "_pyccapt_orig", _ORIG.get("savefig")
    )
    if show is not None:
        plt.show = show
    if savefig is not None:
        Figure.savefig = savefig
    _restore_rcparams()
    _INSTALLED = False
