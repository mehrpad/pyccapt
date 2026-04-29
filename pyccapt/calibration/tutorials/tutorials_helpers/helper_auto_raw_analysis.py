"""Auto-driven raw-data analysis for h5 files that bundle calibrated dld,
linked raw tdc, and an optional range table.

The notebook entry point is :func:`call_auto_raw_data_analysis`. It renders a
two-tab UI (``From range file`` and ``Manual ranges``); each tab has a Run
button that executes :func:`run_analysis` with a species list derived from
either the loaded range table or the user-typed peak windows. All sections
emit a Markdown summary inline beneath the matching plot.

The analyses follow the two ``raw_data_analysis_*-Copy1`` reference notebooks:

- DLTS-per-pulse breakdown (counts of raw rows per linked dld pulse trigger).
- TOF and mass/charge histograms with peak overlays.
- Field desorption map (overall + per-species sub-panels).
- Multi-hit / dead-zone diagnostics from ``delta_p`` / ``multi``.
- Per-species ion counts and percentages (replaces the manual TOF/mc masks
  used in the legacy notebooks).
"""

from __future__ import annotations

from typing import Iterable

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from ipywidgets import Output


# ---------------------------------------------------------------------------
# Detector kind detection
# ---------------------------------------------------------------------------


def detect_detector_kind(tdc_df: pd.DataFrame | None) -> str:
    """Return ``'surface_concept'`` (4 DLTS) or ``'roentdek'`` (6 DLTS).

    Decision is based on the maximum value seen in the ``channel`` column.
    Surface Concept TDCs use channels 0-3 (two delay lines, four signals).
    RoentDek hexanodes use 0-5 (three delay lines, six signals).
    """
    if tdc_df is None or "channel" not in tdc_df.columns or len(tdc_df) == 0:
        return "unknown"
    max_channel = int(np.max(tdc_df["channel"].to_numpy()))
    if max_channel <= 3:
        return "surface_concept"
    if max_channel <= 5:
        return "roentdek"
    return "unknown"


def expected_dlts_full(detector_kind: str) -> int:
    """Number of DLTS that constitute a complete event for the detector."""
    if detector_kind == "surface_concept":
        return 4
    if detector_kind == "roentdek":
        return 6
    return 0


# ---------------------------------------------------------------------------
# Species table builders
# ---------------------------------------------------------------------------


def species_from_range(range_df: pd.DataFrame | None) -> list[dict]:
    """Convert a saved range table into the species schema used here.

    Each entry is ``{label, mc_low, mc_up, color}``. Rows that look like the
    placeholder ``unranged`` row are skipped.
    """
    if range_df is None or len(range_df) == 0:
        return []
    species: list[dict] = []
    for _, row in range_df.iterrows():
        name = str(row.get("name", row.get("ion", ""))).strip()
        if not name or name.lower().startswith("unranged"):
            continue
        try:
            mc_low = float(row["mc_low"])
            mc_up = float(row["mc_up"])
        except (KeyError, TypeError, ValueError):
            continue
        if mc_up <= mc_low:
            continue
        species.append({
            "label": name,
            "mc_low": mc_low,
            "mc_up": mc_up,
            "color": str(row.get("color", "#1f77b4")),
        })
    return species


def species_from_manual(rows: Iterable[tuple[widgets.Text, widgets.FloatText, widgets.FloatText]]) -> list[dict]:
    """Convert manual-input widget rows to the species schema."""
    species: list[dict] = []
    for index, (label_widget, low_widget, high_widget) in enumerate(rows, start=1):
        label = (label_widget.value or "").strip() or f"Peak {index}"
        low = float(low_widget.value)
        high = float(high_widget.value)
        if low == 0 and high == 0:
            continue
        if high <= low:
            raise ValueError(f"Peak {label!r}: max must be greater than min")
        species.append({"label": label, "mc_low": low, "mc_up": high, "color": "#1f77b4"})
    return species


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _close_after(fig):
    plt.close(fig)


def _md(text: str) -> None:
    display(Markdown(text))


def _format_pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0 (0.00%)"
    return f"{numerator:,} ({100.0 * numerator / denominator:.2f}%)"


def plot_dlts_per_pulse(tdc_df: pd.DataFrame, detector_kind: str) -> None:
    """Stacked histogram of DLTS-per-pulse, broken down by linked vs orphan."""
    if tdc_df is None or len(tdc_df) == 0 or "event_group_id" not in tdc_df.columns:
        _md("_No raw tdc loaded with linking — skipping DLTS breakdown._")
        return

    matched = tdc_df[tdc_df["has_dld_match"]]
    orphans = tdc_df[~tdc_df["has_dld_match"]]

    matched_counts = (
        matched.groupby("event_group_id").size().to_numpy()
        if len(matched) else np.array([], dtype=int)
    )
    # Orphan groups: each orphan run shares no group id (-1), so count consecutive
    # equal start_counter values in the orphan slice instead.
    if len(orphans) > 0:
        sc = orphans["start_counter"].to_numpy()
        run_starts = np.r_[0, np.where(np.diff(sc) != 0)[0] + 1, sc.size]
        orphan_counts = np.diff(run_starts)
    else:
        orphan_counts = np.array([], dtype=int)

    if matched_counts.size == 0 and orphan_counts.size == 0:
        return

    max_n = int(max(
        matched_counts.max() if matched_counts.size else 0,
        orphan_counts.max() if orphan_counts.size else 0,
    ))
    bins = np.arange(0.5, max_n + 1.5)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    matched_hist, _ = np.histogram(matched_counts, bins=bins)
    orphan_hist, _ = np.histogram(orphan_counts, bins=bins)
    centers = np.arange(1, max_n + 1)
    ax.bar(centers, matched_hist, label="linked to dld", color="#2ca02c")
    ax.bar(centers, orphan_hist, bottom=matched_hist, label="orphan (no dld)", color="#d62728")
    ax.set_yscale("log")
    ax.set_xlabel("DLTS per pulse trigger")
    ax.set_ylabel("Count")
    ax.set_title(f"DLTS-per-pulse distribution ({detector_kind})")
    ax.set_xticks(centers)
    ax.legend()
    fig.tight_layout()
    plt.show()
    _close_after(fig)

    n_full = expected_dlts_full(detector_kind)
    total_pulses = matched_counts.size + orphan_counts.size
    full_hits = int((matched_counts == n_full).sum() + (orphan_counts == n_full).sum())
    matched_total = int(matched_counts.size)

    lines = [
        "**DLTS-per-pulse breakdown**",
        "",
        f"- Detector kind: `{detector_kind}` (full event = {n_full} DLTS)" if n_full else f"- Detector kind: `{detector_kind}`",
        f"- Total pulse triggers seen in tdc: {total_pulses:,}",
        f"- Linked to a dld row:               {_format_pct(matched_total, total_pulses)}",
        f"- Orphan (no dld counterpart):       {_format_pct(orphan_counts.size, total_pulses)}",
    ]
    if n_full:
        lines.append(f"- Pulses with the full {n_full} DLTS: {_format_pct(full_hits, total_pulses)}")
    _md("\n".join(lines))


def plot_tof_with_peaks(dld_df: pd.DataFrame, species: list[dict]) -> None:
    """Histogram of calibrated TOF with peak windows shaded."""
    tof_col = "t_c (ns)" if "t_c (ns)" in dld_df.columns and (dld_df["t_c (ns)"] != 0).any() else "t (ns)"
    if tof_col not in dld_df.columns:
        return
    tof = dld_df[tof_col].to_numpy()
    if tof.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(tof, bins=400, color="#1f77b4", log=True)
    ax.set_xlabel(tof_col)
    ax.set_ylabel("Count (log)")
    ax.set_title("Time-of-flight histogram")
    fig.tight_layout()
    plt.show()
    _close_after(fig)


def plot_mc_with_peaks(dld_df: pd.DataFrame, species: list[dict]) -> None:
    """Histogram of calibrated mc with shaded species windows + per-peak MRP table."""
    if "mc (Da)" not in dld_df.columns:
        return
    mc = dld_df["mc (Da)"].to_numpy()
    if mc.size == 0:
        return
    upper = float(np.percentile(mc, 99.5)) if mc.size else 0.0
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(mc, bins=400, range=(0, max(upper, 50.0)), color="#555", log=True)
    for sp in species:
        ax.axvspan(sp["mc_low"], sp["mc_up"], color=sp.get("color", "#1f77b4"), alpha=0.25, label=sp["label"])
    if species:
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("mc (Da)")
    ax.set_ylabel("Count (log)")
    ax.set_title("Mass/charge histogram")
    fig.tight_layout()
    plt.show()
    _close_after(fig)

    if not species:
        _md("_No species defined — skipping per-peak MRP table._")
        return

    rows = []
    total = int(mc.size)
    for sp in species:
        in_window = (mc >= sp["mc_low"]) & (mc <= sp["mc_up"])
        count = int(in_window.sum())
        if count > 0:
            mrp = compute_mrp_half(mc[in_window])
        else:
            mrp = float("nan")
        rows.append((sp["label"], sp["mc_low"], sp["mc_up"], count, count / total * 100 if total else 0.0, mrp))

    md = ["**Per-peak counts and MRP(0.5)**", "", "| Peak | mc_low | mc_up | Count | % of all | MRP(0.5) |", "| --- | --- | --- | --- | --- | --- |"]
    for label, lo, hi, count, pct, mrp in rows:
        mrp_str = f"{mrp:.0f}" if np.isfinite(mrp) else "n/a"
        md.append(f"| {label} | {lo:.3f} | {hi:.3f} | {count:,} | {pct:.2f}% | {mrp_str} |")
    _md("\n".join(md))


def compute_mrp_half(mc_window: np.ndarray) -> float:
    """Approximate MRP(0.5) = m / FWHM from a vector of mc values inside one peak."""
    if mc_window.size < 50:
        return float("nan")
    counts, edges = np.histogram(mc_window, bins=100)
    peak_index = int(np.argmax(counts))
    peak_value = (edges[peak_index] + edges[peak_index + 1]) / 2
    half = counts.max() / 2.0
    above = np.where(counts >= half)[0]
    if above.size < 2:
        return float("nan")
    fwhm = edges[above[-1] + 1] - edges[above[0]]
    if fwhm <= 0:
        return float("nan")
    return float(peak_value / fwhm)


def plot_fdm(dld_df: pd.DataFrame, species: list[dict]) -> None:
    """Field desorption map: overall plus one panel per species."""
    if not {"x_det (cm)", "y_det (cm)"}.issubset(dld_df.columns):
        return
    x = dld_df["x_det (cm)"].to_numpy()
    y = dld_df["y_det (cm)"].to_numpy()
    if x.size == 0:
        return

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    h = ax.hist2d(x, y, bins=150, cmap="viridis", norm=plt.matplotlib.colors.LogNorm())
    ax.set_xlabel("x_det (cm)")
    ax.set_ylabel("y_det (cm)")
    ax.set_aspect("equal")
    ax.set_title("FDM (all events)")
    fig.colorbar(h[3], ax=ax, label="Count")
    fig.tight_layout()
    plt.show()
    _close_after(fig)

    if not species or "mc (Da)" not in dld_df.columns:
        return

    mc = dld_df["mc (Da)"].to_numpy()
    n = len(species)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows), squeeze=False)
    for idx, sp in enumerate(species):
        ax = axes[idx // cols][idx % cols]
        mask = (mc >= sp["mc_low"]) & (mc <= sp["mc_up"])
        if mask.any():
            ax.hist2d(x[mask], y[mask], bins=120, cmap="viridis", norm=plt.matplotlib.colors.LogNorm())
        ax.set_title(f"FDM: {sp['label']} ({int(mask.sum()):,})")
        ax.set_aspect("equal")
        ax.set_xlabel("x_det (cm)")
        ax.set_ylabel("y_det (cm)")
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.tight_layout()
    plt.show()
    _close_after(fig)


def plot_multihit_and_deadzone(dld_df: pd.DataFrame) -> None:
    """Multi-hit fraction + delta_p (pulses since previous event) histogram."""
    if "multi" not in dld_df.columns or "delta_p" not in dld_df.columns:
        _md("_`multi` / `delta_p` columns not present — skipping multi-hit diagnostics._")
        return
    multi = dld_df["multi"].to_numpy()
    delta_p = dld_df["delta_p"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist(multi[multi > 0], bins=np.arange(0.5, max(int(multi.max()) + 1, 2) + 0.5))
    axes[0].set_yscale("log")
    axes[0].set_xlabel("multi (ions per pulse)")
    axes[0].set_ylabel("Count (log)")
    axes[0].set_title("Multi-hit distribution")

    delta_p_pos = delta_p[delta_p > 0]
    if delta_p_pos.size:
        upper = int(np.percentile(delta_p_pos, 99))
        axes[1].hist(delta_p_pos, bins=80, range=(0, max(upper, 10)))
    axes[1].set_yscale("log")
    axes[1].set_xlabel("delta_p (pulses since previous event)")
    axes[1].set_ylabel("Count (log)")
    axes[1].set_title("Pulse-to-pulse interval")
    fig.tight_layout()
    plt.show()
    _close_after(fig)

    n_total = int(multi.size)
    n_multi = int((multi > 1).sum())
    _md(
        "**Multi-hit summary**\n\n"
        f"- Total events: {n_total:,}\n"
        f"- Events with multi > 1: {_format_pct(n_multi, n_total)}\n"
        f"- delta_p median: {int(np.median(delta_p_pos)) if delta_p_pos.size else 'n/a'}\n"
    )


# ---------------------------------------------------------------------------
# Top-level analysis runner
# ---------------------------------------------------------------------------


def run_analysis(variables, species: list[dict]) -> None:
    """Render every analysis section against ``variables.data`` and ``variables.data_tdc``."""
    dld_df = getattr(variables, "data", None)
    tdc_df = getattr(variables, "data_tdc", None)
    if dld_df is None or len(dld_df) == 0:
        _md("_No dld data is loaded._")
        return

    detector_kind = detect_detector_kind(tdc_df)
    name = getattr(variables, "dataset_name", "(unnamed dataset)")
    _md(
        f"# Raw-data analysis — `{name}`\n"
        f"- dld rows: {len(dld_df):,}\n"
        f"- tdc rows: {0 if tdc_df is None else len(tdc_df):,}\n"
        f"- Detector kind (auto-detected): `{detector_kind}`\n"
        f"- Species supplied: {len(species)}\n"
    )

    _md("## DLTS-per-pulse")
    plot_dlts_per_pulse(tdc_df, detector_kind)

    _md("## Time-of-flight")
    plot_tof_with_peaks(dld_df, species)

    _md("## Mass/charge")
    plot_mc_with_peaks(dld_df, species)

    _md("## Field desorption map")
    plot_fdm(dld_df, species)

    _md("## Multi-hit / dead-zone")
    plot_multihit_and_deadzone(dld_df)


# ---------------------------------------------------------------------------
# UI: tabbed widget
# ---------------------------------------------------------------------------


def _build_manual_rows() -> list[tuple[widgets.Text, widgets.FloatText, widgets.FloatText]]:
    rows = []
    for index in range(1, 7):
        label = widgets.Text(value=f"Peak {index}", description=f"Peak {index}:", layout=widgets.Layout(width="220px"))
        low = widgets.FloatText(value=0.0, description="mc_low:", layout=widgets.Layout(width="160px"))
        high = widgets.FloatText(value=0.0, description="mc_up:", layout=widgets.Layout(width="160px"))
        rows.append((label, low, high))
    return rows


def call_auto_raw_data_analysis(variables) -> None:
    """Display the two-tab UI and wire Run buttons to :func:`run_analysis`.

    Tabs:

    - **From range file** — uses ``variables.range_data`` to derive the species
      list. Click *Run* to render the analysis.
    - **Manual ranges** — type peak windows directly. Empty rows are ignored.
    """
    range_df = getattr(variables, "range_data", None)
    range_species = species_from_range(range_df)

    range_out = Output()
    manual_out = Output()

    range_summary = widgets.HTML(
        value=(
            f"Range table loaded with <b>{len(range_species)}</b> usable rows."
            if range_species
            else "<i>No range table loaded — switch to the Manual tab or load a range file first.</i>"
        )
    )
    range_run = widgets.Button(description="Run analysis", button_style="primary")

    def _on_range_run(_):
        range_out.clear_output()
        with range_out:
            if not range_species:
                _md("_Range table is empty; nothing to analyze in this tab._")
                return
            run_analysis(variables, range_species)

    range_run.on_click(_on_range_run)
    range_tab = widgets.VBox([range_summary, range_run, range_out])

    manual_rows = _build_manual_rows()
    manual_run = widgets.Button(description="Run analysis", button_style="primary")
    manual_help = widgets.HTML(
        value="<i>Type any number of peak windows (rows with both fields = 0 are skipped).</i>"
    )

    def _on_manual_run(_):
        manual_out.clear_output()
        with manual_out:
            try:
                manual_species = species_from_manual(manual_rows)
            except ValueError as exc:
                _md(f"**Input error:** {exc}")
                return
            if not manual_species:
                _md("_All peak windows are empty — nothing to analyze._")
                return
            run_analysis(variables, manual_species)

    manual_run.on_click(_on_manual_run)
    manual_grid = widgets.VBox([
        widgets.HBox([label, low, high]) for label, low, high in manual_rows
    ])
    manual_tab = widgets.VBox([manual_help, manual_grid, manual_run, manual_out])

    tabs = widgets.Tab(children=[range_tab, manual_tab])
    tabs.set_title(0, "From range file")
    tabs.set_title(1, "Manual ranges")
    display(tabs)
