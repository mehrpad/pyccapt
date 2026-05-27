"""Diagnostic plots and statistics for partial-hit recovery.

After the data-processing workflow runs with ``merge_partial_tdc=True``
the DLD dataframe carries a ``dlts`` column (4 = full, 2 = single-axis
partial) and a ``dlts_quality`` column (``native``, ``recovered_xy``,
``recovered_x``, ``recovered_y``). The raw TDC frame on
``variables.data_tdc`` retains every channel tick, with the
``has_dld_match`` and ``event_group_id`` columns linking each tick to
its parent pulse.

This module turns those tags into physics-flavoured diagnostics that
help answer the questions a microscope operator usually asks about
partial hits:

* **How many of each kind?** Count breakdown of native-full,
  recovered-full, x-only, y-only.
* **Where on the detector?** 2-D maps of full vs partial; per-axis
  hit-density profiles.
* **What's the bias?** Are partials concentrated near the centre,
  edges, or a "dead zone"? Compare radial CDFs of full and partial.
* **How do they correlate with the running voltage?** HV/Vp scatter
  and rolling fraction.
* **Where in mass / ToF?** Mass-spectrum overlay split by ``dlts``
  class, so you see which species are partial-heavy.
* **Are multi-hit pulses more likely to be partial?** Per-pulse tick
  counts vs partial-rate.

All plots accept either the in-memory ``variables.data`` /
``variables.data_tdc`` directly, or any DataFrame that carries the
expected columns; outputs are matplotlib figures returned for
embedding in a notebook (and optionally written to disk via the
caller).
"""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DLTS_COL = "dlts"
DLTS_QUALITY_COL = "dlts_quality"
EVENT_GROUP_ID_COL = "event_group_id"
TDC_HAS_DLD_MATCH_COL = "has_dld_match"


# ---------------------------------------------------------------------------
# Lightweight categorisation helpers
# ---------------------------------------------------------------------------


def _categorise_rows(dld_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return boolean masks for the categories used across the diagnostics.

    Categories:
        ``native``        — full 4-DLTS hit emitted by the hardware
        ``recovered_xy``  — full 4-DLTS hit recovered by the merge step
        ``recovered_x``   — 2-DLTS partial with x_det recovered, y NaN
        ``recovered_y``   — 2-DLTS partial with y_det recovered, x NaN
        ``full``          — union of ``native`` and ``recovered_xy``
        ``partial``       — union of ``recovered_x`` and ``recovered_y``
    """
    n = len(dld_df)
    out: dict[str, np.ndarray] = {}
    if DLTS_QUALITY_COL in dld_df.columns:
        q = dld_df[DLTS_QUALITY_COL].astype(str).to_numpy()
    else:
        q = np.array(["native"] * n)
    out["native"] = q == "native"
    out["recovered_xy"] = q == "recovered_xy"
    out["recovered_x"] = q == "recovered_x"
    out["recovered_y"] = q == "recovered_y"
    out["full"] = out["native"] | out["recovered_xy"]
    out["partial"] = out["recovered_x"] | out["recovered_y"]
    return out


def partial_hit_counts(dld_df: pd.DataFrame) -> dict[str, int]:
    """Return per-category row counts plus useful ratios.

    The ratios mirror what users typically want at a glance: how many
    hits the partial recovery added, and what fraction of the final
    spectrum each class accounts for.
    """
    masks = _categorise_rows(dld_df)
    counts = {name: int(m.sum()) for name, m in masks.items()}
    total = int(len(dld_df))
    counts["total"] = total
    if total > 0:
        counts["partial_fraction"] = counts["partial"] / total
        counts["recovered_fraction"] = (counts["recovered_xy"] + counts["partial"]) / total
    else:
        counts["partial_fraction"] = 0.0
        counts["recovered_fraction"] = 0.0
    return counts


# ---------------------------------------------------------------------------
# Plot 1 — high-level breakdown
# ---------------------------------------------------------------------------


def plot_partial_hit_breakdown(dld_df: pd.DataFrame, figsize: tuple[float, float] = (7.0, 3.6)) -> plt.Figure:
    """Bar chart + pie chart of the four hit categories.

    A single figure with two panels: counts (log y) and proportional
    breakdown. Both make the same point but the bar plot keeps the
    absolute numbers visible while the pie chart gives an at-a-glance
    fraction.
    """
    counts = partial_hit_counts(dld_df)
    labels = ["native", "recovered_xy", "recovered_x", "recovered_y"]
    values = [counts[label] for label in labels]
    colors = ["#2563eb", "#22c55e", "#f59e0b", "#d97706"]

    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    ax_bar.bar(labels, values, color=colors)
    ax_bar.set_ylabel("Hit count")
    if max(values, default=0) > 0:
        ax_bar.set_yscale("log")
    ax_bar.tick_params(axis="x", rotation=20)
    ax_bar.set_title("Hit count by category")

    nonzero = [(lab, val, col) for lab, val, col in zip(labels, values, colors) if val > 0]
    if nonzero:
        labs, vals, cols = zip(*nonzero)
        ax_pie.pie(vals, labels=labs, autopct="%1.1f%%", colors=cols, startangle=90)
    ax_pie.set_title(
        f"Composition (n_total = {counts['total']:,}, "
        f"partials = {counts['partial']:,}, "
        f"recovered share = {counts['recovered_fraction']:.1%})"
    )
    return fig


# ---------------------------------------------------------------------------
# Plot 2 — detector-space distribution
# ---------------------------------------------------------------------------


def plot_partial_hit_detector_distribution(
    dld_df: pd.DataFrame,
    *,
    detector_radius_cm: float = 4.0,
    bins: int = 96,
    figsize: tuple[float, float] = (10.0, 6.5),
) -> plt.Figure:
    """Three FDM panels: full hits, recovered x-partials, recovered y-partials.

    The fourth panel shows the radial-cumulative profile of full vs
    partial hits, which surfaces whether partials concentrate at
    specific radii (e.g. near a known dead zone or at the detector
    edge where one delay line can drop counts).

    Recovered y-only partials have NaN x_det by construction; the
    panel uses ``y_det`` against ``y_det`` (degenerate) so they show
    up as a 1-D rug — which is the honest visualisation: their x
    position is unknown.
    """
    masks = _categorise_rows(dld_df)
    x = dld_df["x_det (cm)"].to_numpy(dtype=float) if "x_det (cm)" in dld_df.columns else np.full(len(dld_df), np.nan)
    y = dld_df["y_det (cm)"].to_numpy(dtype=float) if "y_det (cm)" in dld_df.columns else np.full(len(dld_df), np.nan)

    edges = np.linspace(-detector_radius_cm, detector_radius_cm, bins + 1)

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    ax_full, ax_x, ax_y, ax_radial = axes.ravel()

    def _panel(ax, sel_mask, title):
        sel_x = x[sel_mask]
        sel_y = y[sel_mask]
        finite = np.isfinite(sel_x) & np.isfinite(sel_y)
        if finite.any():
            counts, _, _, im = ax.hist2d(sel_x[finite], sel_y[finite], bins=[edges, edges], cmap="viridis")
            plt.colorbar(im, ax=ax, label="counts")
        ax.set_xlim(-detector_radius_cm, detector_radius_cm)
        ax.set_ylim(-detector_radius_cm, detector_radius_cm)
        ax.set_xlabel("x_det (cm)")
        ax.set_ylabel("y_det (cm)")
        ax.set_aspect("equal")
        ax.set_title(f"{title} (n={int(sel_mask.sum()):,})")

    _panel(ax_full, masks["full"], "Full hits (native + recovered_xy)")
    # For x-only partials, y_det is NaN -- show a 1-D x histogram along
    # the y=0 line instead of a degenerate 2-D map.
    x_only = x[masks["recovered_x"]]
    x_only_finite = x_only[np.isfinite(x_only)]
    if x_only_finite.size:
        ax_x.hist(x_only_finite, bins=edges, color="#f59e0b", edgecolor="none")
    ax_x.set_xlim(-detector_radius_cm, detector_radius_cm)
    ax_x.set_xlabel("x_det (cm)")
    ax_x.set_ylabel("counts (y unknown)")
    ax_x.set_title(f"x-only partials (n={int(masks['recovered_x'].sum()):,})")

    y_only = y[masks["recovered_y"]]
    y_only_finite = y_only[np.isfinite(y_only)]
    if y_only_finite.size:
        ax_y.hist(y_only_finite, bins=edges, color="#d97706", edgecolor="none", orientation="horizontal")
    ax_y.set_ylim(-detector_radius_cm, detector_radius_cm)
    ax_y.set_xlabel("counts (x unknown)")
    ax_y.set_ylabel("y_det (cm)")
    ax_y.set_title(f"y-only partials (n={int(masks['recovered_y'].sum()):,})")

    # Radial CDF: a partial population that follows the full distribution
    # would give parallel CDFs; deviations point to spatially-biased loss
    # mechanisms (edge effects, dead zone, off-center plasma plume).
    r_full = np.hypot(x[masks["full"]], y[masks["full"]])
    r_full = r_full[np.isfinite(r_full)]
    r_x_part = np.abs(x[masks["recovered_x"]])
    r_x_part = r_x_part[np.isfinite(r_x_part)]
    r_y_part = np.abs(y[masks["recovered_y"]])
    r_y_part = r_y_part[np.isfinite(r_y_part)]
    radii = np.linspace(0.0, detector_radius_cm, 200)
    for label, arr, color in [
        ("full |r|", r_full, "#2563eb"),
        ("partial |x| (x-only)", r_x_part, "#f59e0b"),
        ("partial |y| (y-only)", r_y_part, "#d97706"),
    ]:
        if arr.size == 0:
            continue
        cdf = np.searchsorted(np.sort(arr), radii, side="right") / arr.size
        ax_radial.plot(radii, cdf, label=f"{label} (n={arr.size:,})", color=color)
    ax_radial.set_xlim(0.0, detector_radius_cm)
    ax_radial.set_ylim(0.0, 1.0)
    ax_radial.set_xlabel("radial distance (cm)")
    ax_radial.set_ylabel("cumulative fraction")
    ax_radial.set_title("Radial CDF: full vs partial")
    ax_radial.legend(loc="lower right", fontsize=8)

    return fig


# ---------------------------------------------------------------------------
# Plot 3 — voltage / time correlation
# ---------------------------------------------------------------------------


def plot_partial_hit_voltage_correlation(
    dld_df: pd.DataFrame,
    *,
    window_ions: int = 5000,
    figsize: tuple[float, float] = (8.0, 5.0),
) -> plt.Figure:
    """Rolling partial-hit fraction over the ion index and HV histogram.

    The top panel shows the partial-hit fraction in sliding windows of
    ``window_ions`` ions; spikes correlate with periods of high
    multi-hit rate, plasma transients, or detector mis-behaviour during
    the experiment. The bottom panel histograms HV separately for full
    and partial hits — if partials are enriched at extreme voltages
    you'll see a clear shape difference.
    """
    masks = _categorise_rows(dld_df)
    n = len(dld_df)
    if n == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Empty dataframe", ha="center", va="center")
        return fig

    fig, (ax_roll, ax_hv) = plt.subplots(2, 1, figsize=figsize, constrained_layout=True)

    partial_indicator = masks["partial"].astype(float)
    w = max(50, int(window_ions))
    if n >= w:
        kernel = np.ones(w) / w
        rolling = np.convolve(partial_indicator, kernel, mode="valid")
        x_axis = np.arange(rolling.size) + w // 2
        ax_roll.plot(x_axis, rolling, color="#d97706", linewidth=1.0)
        ax_roll.set_xlabel("Ion index (sequence order)")
        ax_roll.set_ylabel(f"Partial fraction (window = {w} ions)")
        ax_roll.set_title("Partial-hit fraction over time-in-experiment")
        ax_roll.set_ylim(0.0, max(0.05, float(rolling.max()) * 1.1))
    else:
        ax_roll.text(0.5, 0.5, "Fewer rows than window size", ha="center", va="center", transform=ax_roll.transAxes)

    hv_col = "high_voltage (V)"
    if hv_col in dld_df.columns:
        hv = dld_df[hv_col].to_numpy(dtype=float) / 1000.0  # kV
        bins = np.linspace(np.nanmin(hv), np.nanmax(hv), 80) if np.any(np.isfinite(hv)) else 20
        ax_hv.hist(hv[masks["full"]], bins=bins, histtype="step", color="#2563eb", label="full", linewidth=1.4)
        ax_hv.hist(hv[masks["partial"]], bins=bins, histtype="step", color="#d97706", label="partial", linewidth=1.4)
        ax_hv.set_xlabel("High voltage (kV)")
        ax_hv.set_ylabel("Counts")
        ax_hv.set_yscale("log")
        ax_hv.legend(loc="upper right")
        ax_hv.set_title("HV distribution: full vs partial")
    else:
        ax_hv.text(0.5, 0.5, "high_voltage (V) column missing", ha="center", va="center", transform=ax_hv.transAxes)
    return fig


# ---------------------------------------------------------------------------
# Plot 4 — mass / ToF spectrum overlay by dlts class
# ---------------------------------------------------------------------------


def plot_partial_hit_signal_overlay(
    dld_df: pd.DataFrame,
    *,
    signal_kind: str = "mc",
    max_value: float | None = None,
    bin_width: float = 0.1,
    figsize: tuple[float, float] = (8.0, 4.0),
) -> plt.Figure:
    """Stacked / overlayed histogram of mc (or tof) split by hit category.

    Surfaces which mass peaks are over-represented in the partial-hit
    population. A peak that gains a large fraction of its counts only
    after partial recovery may be detector-edge dominated; one that is
    largely unaffected is a "core" species.
    """
    if signal_kind == "mc":
        col = "mc (Da)"
        if col not in dld_df.columns or (dld_df[col] == 0).all():
            col = "mc_uc (Da)"
        xlabel = "Mass-to-charge (Da)"
    elif signal_kind == "tof":
        col = "t (ns)"
        xlabel = "Time of flight (ns)"
    else:
        raise ValueError(f"signal_kind must be 'mc' or 'tof', got {signal_kind!r}")

    if col not in dld_df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"Column {col!r} missing", ha="center", va="center")
        return fig

    values = dld_df[col].to_numpy(dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"Column {col!r} has no finite values", ha="center", va="center")
        return fig

    upper = float(max_value) if max_value is not None else float(np.nanmax(values[finite]))
    upper = max(upper, bin_width * 2)
    edges = np.arange(0.0, upper + bin_width, bin_width)

    masks = _categorise_rows(dld_df)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    series = [
        ("native", masks["native"], "#2563eb"),
        ("recovered_xy", masks["recovered_xy"], "#22c55e"),
        ("recovered_x", masks["recovered_x"], "#f59e0b"),
        ("recovered_y", masks["recovered_y"], "#d97706"),
    ]
    for label, mask, color in series:
        sub = values[mask & finite]
        if sub.size == 0:
            continue
        ax.hist(sub, bins=edges, histtype="step", linewidth=1.2, label=f"{label} (n={sub.size:,})", color=color)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts")
    ax.set_title(f"{signal_kind.upper()} spectrum split by hit category")
    ax.legend(loc="upper right", fontsize=8)
    return fig


# ---------------------------------------------------------------------------
# Plot 5 — multi-hit-vs-partial correlation (per-pulse)
# ---------------------------------------------------------------------------


def _per_pulse_aggregate(
    dld_df: pd.DataFrame,
    tdc_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build a per-pulse aggregate: tick-count, dld-count, partial-count.

    The grouping uses ``event_group_id`` when present (consistent
    between dld and tdc thanks to the partial-recovery merge); each
    row in the returned frame corresponds to one pulse.
    """
    if EVENT_GROUP_ID_COL not in dld_df.columns:
        return pd.DataFrame()
    masks = _categorise_rows(dld_df)
    dld_pulse_stats = dld_df.assign(
        _is_partial=masks["partial"].astype(int),
        _is_full=masks["full"].astype(int),
    ).groupby(EVENT_GROUP_ID_COL).agg(
        n_dld_rows=(EVENT_GROUP_ID_COL, "size"),
        n_partial=("_is_partial", "sum"),
        n_full=("_is_full", "sum"),
    ).reset_index()

    if tdc_df is not None and EVENT_GROUP_ID_COL in tdc_df.columns:
        tdc_counts = tdc_df.groupby(EVENT_GROUP_ID_COL).size().rename("n_tdc_ticks").reset_index()
        merged = dld_pulse_stats.merge(tdc_counts, on=EVENT_GROUP_ID_COL, how="left")
        merged["n_tdc_ticks"] = merged["n_tdc_ticks"].fillna(0).astype(int)
        return merged
    dld_pulse_stats["n_tdc_ticks"] = np.nan
    return dld_pulse_stats


def plot_partial_hit_multihit_correlation(
    dld_df: pd.DataFrame,
    tdc_df: pd.DataFrame | None = None,
    *,
    figsize: tuple[float, float] = (9.0, 4.5),
) -> plt.Figure:
    """Partial-hit rate as a function of pulse multiplicity.

    Two panels:

    * **Left** — partial fraction vs ``n_tdc_ticks`` (the raw number
      of channel firings in the pulse). The hypothesis "multi-hit
      pulses are more likely to produce partials" maps to a rising
      curve here. Confirmation would mean the SC firmware's coincidence
      logic struggles to pair channels when many ions arrive within
      one pulse window.
    * **Right** — partial fraction vs ``n_dld_rows`` (the number of
      ions emitted from that pulse, including any recovered partials).
      Tests the same hypothesis from the dld side.
    """
    pulse_stats = _per_pulse_aggregate(dld_df, tdc_df)
    fig, (ax_tdc, ax_dld) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    if pulse_stats.empty:
        for ax in (ax_tdc, ax_dld):
            ax.text(0.5, 0.5, "event_group_id missing", ha="center", va="center", transform=ax.transAxes)
        return fig

    # Tick-count panel ----------------------------------------------------
    if pulse_stats["n_tdc_ticks"].notna().any():
        max_ticks = int(min(20, pulse_stats["n_tdc_ticks"].max() or 0))
        if max_ticks > 0:
            grouped = pulse_stats.groupby("n_tdc_ticks").agg(
                pulses=("n_tdc_ticks", "size"),
                partials=("n_partial", "sum"),
                ions=("n_dld_rows", "sum"),
            )
            grouped = grouped.loc[grouped.index <= max_ticks]
            with np.errstate(invalid="ignore", divide="ignore"):
                fraction = grouped["partials"] / grouped["ions"].clip(lower=1)
            ax_tdc.bar(grouped.index.astype(int), fraction, color="#d97706")
            ax_tdc.set_xlabel("TDC ticks in pulse")
            ax_tdc.set_ylabel("Partial fraction of dld rows")
            ax_tdc.set_title("Partial rate vs pulse multiplicity (TDC side)")
            ax_tdc.set_xticks(np.arange(0, max_ticks + 1))
            ax_tdc.set_ylim(0.0, max(0.05, float(fraction.max()) * 1.15))
    else:
        ax_tdc.text(0.5, 0.5, "tdc_df not supplied", ha="center", va="center", transform=ax_tdc.transAxes)

    # DLD-count panel -----------------------------------------------------
    max_dld = int(min(15, pulse_stats["n_dld_rows"].max() or 0))
    if max_dld > 0:
        grouped = pulse_stats.groupby("n_dld_rows").agg(
            pulses=("n_dld_rows", "size"),
            partials=("n_partial", "sum"),
            ions=("n_dld_rows", "sum"),
        )
        grouped = grouped.loc[grouped.index <= max_dld]
        with np.errstate(invalid="ignore", divide="ignore"):
            fraction = grouped["partials"] / grouped["ions"].clip(lower=1)
        ax_dld.bar(grouped.index.astype(int), fraction, color="#2563eb")
        ax_dld.set_xlabel("DLD rows per pulse")
        ax_dld.set_ylabel("Partial fraction of dld rows")
        ax_dld.set_title("Partial rate vs DLD multiplicity")
        ax_dld.set_xticks(np.arange(1, max_dld + 1))
        ax_dld.set_ylim(0.0, max(0.05, float(fraction.max()) * 1.15))

    return fig


# ---------------------------------------------------------------------------
# Plot 6 — axis-channel bias
# ---------------------------------------------------------------------------


def plot_partial_hit_axis_bias(
    dld_df: pd.DataFrame,
    figsize: tuple[float, float] = (6.0, 3.6),
) -> plt.Figure:
    """Bar chart comparing the population of x-only vs y-only partials.

    A healthy SC delay-line detector should produce both axes with
    roughly equal partial-hit rates. A large asymmetry points to a
    failing or noisy channel on one axis (ch0/ch1 for x, ch2/ch3 for
    y). This single chart catches the issue at a glance.
    """
    masks = _categorise_rows(dld_df)
    n_x = int(masks["recovered_x"].sum())
    n_y = int(masks["recovered_y"].sum())
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.bar(["x-only", "y-only"], [n_x, n_y], color=["#f59e0b", "#d97706"])
    ax.set_ylabel("Recovered partial count")
    total = n_x + n_y
    if total > 0:
        ratio = n_x / total
        ax.set_title(f"Per-axis partial bias  |  x/(x+y) = {ratio:.2%}  |  ideal ≈ 50 %")
    else:
        ax.set_title("Per-axis partial bias (no partials)")
    return fig


# ---------------------------------------------------------------------------
# Combined dashboard
# ---------------------------------------------------------------------------


def run_partial_hit_diagnostics(
    variables,
    *,
    detector_radius_cm: float = 4.0,
    save_dir: str | None = None,
) -> dict[str, plt.Figure]:
    """Build every diagnostic plot in one call.

    Convenience entry-point for the data-processing or
    raw-data-analysis notebook: pass ``variables`` and get a dict of
    named figures back. Optionally writes them to ``save_dir``.
    """
    dld_df = getattr(variables, "data", None)
    tdc_df = getattr(variables, "data_tdc", None)
    if dld_df is None or len(dld_df) == 0:
        raise ValueError("variables.data is empty -- load a dataset first.")

    counts = partial_hit_counts(dld_df)
    figures: dict[str, plt.Figure] = {
        "breakdown": plot_partial_hit_breakdown(dld_df),
        "detector_distribution": plot_partial_hit_detector_distribution(
            dld_df,
            detector_radius_cm=detector_radius_cm,
        ),
        "voltage_correlation": plot_partial_hit_voltage_correlation(dld_df),
        "axis_bias": plot_partial_hit_axis_bias(dld_df),
        "mc_overlay": plot_partial_hit_signal_overlay(dld_df, signal_kind="mc"),
        "tof_overlay": plot_partial_hit_signal_overlay(dld_df, signal_kind="tof"),
        "multihit_correlation": plot_partial_hit_multihit_correlation(dld_df, tdc_df),
    }

    if save_dir:
        from pathlib import Path

        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            fig.savefig(out / f"partial_hit_{name}.png", dpi=150)

    return figures
