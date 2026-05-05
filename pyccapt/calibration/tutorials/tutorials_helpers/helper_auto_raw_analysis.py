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

from pathlib import Path
from typing import Iterable

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from ipywidgets import Output

from pyccapt.calibration.path_utils import ensure_directory, save_figure


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


def _resolve_dataset_path(variables) -> Path | None:
    raw_path = str(getattr(variables, "path", "") or "").strip()
    if raw_path:
        dataset_path = Path(raw_path).expanduser()
        if dataset_path.is_file():
            return dataset_path
    return None


def _analysis_save_directory(variables, enabled: bool) -> Path | None:
    if not enabled:
        return None
    dataset_path = _resolve_dataset_path(variables)
    if dataset_path is None:
        return None
    return ensure_directory(dataset_path.parent / f"{dataset_path.stem}_raw_analysis_plots")


def _show_figure(fig, *, save_dir: str | Path | None = None, stem: str | None = None) -> None:
    if fig is None:
        return
    if save_dir is not None and stem:
        save_figure(
            fig,
            directory=save_dir,
            stem=stem,
            formats=("svg", "png"),
            dpi=300,
            bbox_inches="tight",
        )
    display(fig)
    _close_after(fig)


def _md(text: str) -> None:
    display(Markdown(text))


def _format_pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0 (0.00%)"
    return f"{numerator:,} ({100.0 * numerator / denominator:.2f}%)"


def _classify_pulse_groups(df: pd.DataFrame, group_col: str):
    """Classify each pulse group by which TDC channels are present.

    Surface Concept has 4 channels (0,1 = x delay line; 2,3 = y delay line).

    Returns three numpy arrays — one entry per pulse group:
      counts     : total number of TDC signals in the group
      complete   : True when all 4 channels (0,1,2,3) are present
                   → both delay-line directions resolved → "4 DLTS"
      partial    : True when only x-pair (0+1) OR only y-pair (2+3) present
                   → one direction missing → "2 DLTS"

    A group can be both complete=False and partial=False (e.g. noise burst
    with only channel 0).  A group can never be both True simultaneously.
    """
    if len(df) == 0:
        empty = np.array([], dtype=int)
        return empty, empty.astype(bool), empty.astype(bool)

    grp = df.groupby(group_col)["channel"]
    counts = grp.count().to_numpy()

    # Boolean: does channel N appear at least once in this group?
    ch = df["channel"]
    idx = df[group_col]
    has = {c: (ch == c).groupby(idx).any().to_numpy() for c in range(4)}

    complete = has[0] & has[1] & has[2] & has[3]
    has_x    = has[0] & has[1]
    has_y    = has[2] & has[3]
    partial  = (has_x & ~has_y) | (has_y & ~has_x)

    return counts, complete, partial


def plot_dlts_per_pulse(
    tdc_df: pd.DataFrame,
    detector_kind: str,
    *,
    save_dir: str | Path | None = None,
    save_stem: str | None = None,
) -> None:
    """DLTS-per-pulse histogram with channel-based 4-DLTS / 2-DLTS classification.

    Each x-position (number of TDC signals per pulse) shows up to three bars:

    • **Gray**   – total frequency (all pulses at that DLTS count)
    • **Orange** – "2 DLTS" partial pulses: only one delay-line direction fired
                   (channels 0+1 for x  OR  channels 2+3 for y, not both)
    • **Blue**   – "4 DLTS" complete pulses: all four channels present
                   (channels 0+1+2+3 → both x and y resolved)

    This reproduces the style and physics of the legacy reference notebook.
    """
    if tdc_df is None or len(tdc_df) == 0 or "event_group_id" not in tdc_df.columns:
        _md("_No raw tdc loaded with linking — skipping DLTS breakdown._")
        return
    if "channel" not in tdc_df.columns:
        _md("_No `channel` column in tdc data — cannot classify DLTS groups._")
        return

    matched = tdc_df[tdc_df["has_dld_match"]]
    orphans = tdc_df[~tdc_df["has_dld_match"]]

    m_counts, m_complete, m_partial = _classify_pulse_groups(matched, "event_group_id")

    # Orphan rows share event_group_id = -1; group them by start_counter instead.
    o_counts, o_complete, o_partial = _classify_pulse_groups(orphans, "start_counter")

    if m_counts.size == 0 and o_counts.size == 0:
        return

    all_counts   = np.concatenate([m_counts,   o_counts])   if o_counts.size   else m_counts
    all_complete = np.concatenate([m_complete,  o_complete]) if o_complete.size else m_complete
    all_partial  = np.concatenate([m_partial,   o_partial])  if o_partial.size  else m_partial

    max_n = int(all_counts.max())
    bins    = np.arange(0.5, max_n + 1.5)
    centers = np.arange(1, max_n + 1)

    freq_hist,     _ = np.histogram(all_counts,                   bins=bins)
    complete_hist, _ = np.histogram(all_counts[all_complete],     bins=bins)
    partial_hist,  _ = np.histogram(all_counts[all_partial],      bins=bins)

    fig, ax = plt.subplots(figsize=(9, 4))
    # Exact same layout as the reference notebook (raw_data_analysis_surface_concept):
    #
    #   gray  (Frequency) — WIDE bar (width=0.4) centred AT x, semi-transparent
    #                        (alpha=0.5) so orange/blue bars drawn on top show through
    #   orange (2 DLTS)   — NARROW bar (width=0.2) shifted 0.1 LEFT of centre
    #                        → covers the left half of the gray bar
    #   blue   (4 DLTS)   — NARROW bar (width=0.2) shifted 0.1 RIGHT of centre
    #                        → covers the right half of the gray bar
    #
    # Result: at each x tick you see all three colours simultaneously.
    w = 0.2
    ax.bar(centers,           freq_hist,     width=w * 2, label="Frequency", alpha=0.5, color="gray")
    ax.bar(centers - 0.5 * w, partial_hist,  width=w,     label="2 DLTS",              color="orange")
    ax.bar(centers + 0.5 * w, complete_hist, width=w,     label="4 DLTS",              color="blue")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Delayline Time Stamps per Pulse")
    ax.set_ylabel("Count")
    ax.set_title(f"DLTS-per-pulse distribution ({detector_kind})")
    readable_limit = min(max_n, 20)
    ax.set_xticks(np.arange(1, readable_limit + 1))
    ax.set_xlim(0.5, readable_limit + 0.5)
    ax.legend()
    fig.tight_layout()
    _show_figure(fig, save_dir=save_dir, stem=save_stem)

    n_full         = expected_dlts_full(detector_kind)
    total_pulses   = int(all_counts.size)
    matched_total  = int(m_counts.size)
    n_complete_tot = int(all_complete.sum())
    n_partial_tot  = int(all_partial.sum())

    lines = [
        "**DLTS-per-pulse breakdown**",
        "",
        f"- Detector kind: `{detector_kind}` (full event = {n_full} DLTS)" if n_full else f"- Detector kind: `{detector_kind}`",
        f"- Total pulse triggers seen in tdc: {total_pulses:,}",
        f"- Linked to a dld row (has position): {_format_pct(matched_total, total_pulses)}",
        f"- Orphan (no dld counterpart):        {_format_pct(total_pulses - matched_total, total_pulses)}",
        f"- **4 DLTS** — complete (all 4 channels present): {_format_pct(n_complete_tot, total_pulses)}",
        f"- **2 DLTS** — partial  (one delay-line only):    {_format_pct(n_partial_tot,  total_pulses)}",
    ]
    _md("\n".join(lines))


def plot_tof_with_peaks(
    dld_df: pd.DataFrame,
    species: list[dict],
    *,
    save_dir: str | Path | None = None,
    save_stem: str | None = None,
) -> None:
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
    _show_figure(fig, save_dir=save_dir, stem=save_stem)


def _pick_mc_col(dld_df: pd.DataFrame) -> str | None:
    """Return the best available mass/charge column name.

    Prefers ``mc (Da)`` when it contains non-zero values (i.e. the file was
    calibrated).  Falls back to ``mc_uc (Da)`` (uncalibrated, computed from
    the raw TOF) when ``mc (Da)`` is all-zero or absent — which is the common
    case for pure-raw acquisition files where calibration has not yet been
    applied.
    """
    for col in ("mc (Da)", "mc_uc (Da)"):
        if col in dld_df.columns and (dld_df[col] != 0).any():
            return col
    return None


def plot_mc_with_peaks(
    dld_df: pd.DataFrame,
    species: list[dict],
    *,
    save_dir: str | Path | None = None,
    save_stem: str | None = None,
) -> None:
    """Histogram of calibrated mc with shaded species windows + per-peak MRP table."""
    mc_col = _pick_mc_col(dld_df)
    if mc_col is None:
        _md(
            "_**Mass/charge skipped** — both `mc (Da)` and `mc_uc (Da)` are either absent "
            "or all-zero in this file. The file may not have been through calibration yet, "
            "or the column names differ._"
        )
        return
    mc = dld_df[mc_col].to_numpy()
    if mc.size == 0:
        return
    upper = float(np.percentile(mc, 99.5)) if mc.size else 0.0
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(mc, bins=400, range=(0, max(upper, 50.0)), color="#555", log=True)
    for sp in species:
        ax.axvspan(sp["mc_low"], sp["mc_up"], color=sp.get("color", "#1f77b4"), alpha=0.25, label=sp["label"])
    if species:
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel(mc_col)
    ax.set_ylabel("Count (log)")
    ax.set_title(f"Mass/charge histogram ({mc_col})")
    fig.tight_layout()
    _show_figure(fig, save_dir=save_dir, stem=save_stem)

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


def _surface_concept_length_breakdown_markdown(sequence_stats: dict[str, dict[int, int]], *, max_bins: int = 20) -> str:
    total = sequence_stats.get('total', {})
    dld2 = sequence_stats.get('dld2', {})
    dld4 = sequence_stats.get('dld4', {})
    invalid = sequence_stats.get('invalid', {})
    lines = ["**Per-length recovery counts**", ""]
    for n in range(1, max_bins + 1):
        total_count = int(total.get(n, 0))
        if total_count == 0:
            continue
        possible_events = total_count * (((n - 1) // 4) + 1)
        recovered_4 = int(dld4.get(n, 0))
        recovered_2 = int(dld2.get(n, 0))
        unrecoverable = int(invalid.get(n, 0))
        lines.append(
            f"- For number {n}: frequency = {total_count:,}; possible events = {possible_events:,}; "
            f"4 DLTS + 2 DLTS + unrecoverable = {recovered_4 + recovered_2 + unrecoverable:,}"
        )
    return "\n".join(lines)


def _surface_concept_raw_summary_markdown(raw_summary: dict[str, object], recovery_stats: dict[str, int]) -> str:
    return "\n".join(
        [
            "**Surface Concept raw summary**",
            "",
            f"- Total grouped pulses: {int(raw_summary.get('total_sequences', 0)):,}",
            f"- Total delay-line timestamps: {int(raw_summary.get('total_timestamps', 0)):,}",
            f"- Valid 4-channel groups: {int(raw_summary.get('valid_four_channel_groups', 0)):,}",
            f"- Invalid 4-channel groups: {int(raw_summary.get('invalid_four_channel_groups', 0)):,}",
            f"- 3-channel groups: {int(raw_summary.get('length_three_groups', 0)):,}",
            f"- 2-channel groups: {int(raw_summary.get('length_two_groups', 0)):,}",
            f"- 1-channel groups: {int(raw_summary.get('length_one_groups', 0)):,}",
            f"- Multi-hit groups with length multiple of 4: {int(raw_summary.get('multi_hit_groups_of_four', 0)):,}",
            f"- Multi-hit irregular groups: {int(raw_summary.get('multi_hit_irregular', 0)):,}",
            f"- Recovered 4 DLTS hits in detector: {int(recovery_stats.get('two_d_in_detector', 0)):,}",
            f"- Recovered 2 DLTS hits in detector: {int(recovery_stats.get('one_d_in_detector', 0)):,}",
            f"- Recovered hits outside detector: {int(recovery_stats.get('outside_detector_hits', 0)):,}",
            f"- Unrecoverable chunks: {int(recovery_stats.get('unrecoverable_chunks', 0)):,}",
        ]
    )


def _species_to_windows(species: list[dict]) -> list[dict]:
    windows = []
    for index, sp in enumerate(species, start=1):
        windows.append(
            {
                "label": str(sp.get("label", f"Peak {index}")),
                "min": float(sp["mc_low"]),
                "max": float(sp["mc_up"]),
            }
        )
    return windows


def _surface_concept_peak_ratio_markdown(ratio_table: pd.DataFrame) -> str:
    if ratio_table.empty:
        return "_No peak-window ratio table could be built._"
    rows = ["**Peak-window 2 DLTS / 4 DLTS summary**", ""]
    for _, row in ratio_table.iterrows():
        ratio_value = row["Two/Four DLTS"]
        ratio_text = "n/a" if not np.isfinite(ratio_value) else f"{ratio_value:.3f}"
        rows.append(
            f"- {row['Peak']}: 2 DLTS = {int(row['Two DLTS count']):,} ({row['Two DLTS %']:.2f}%), "
            f"4 DLTS = {int(row['Four DLTS count']):,} ({row['Four DLTS %']:.2f}%), "
            f"2/4 ratio = {ratio_text}"
        )
    return "\n".join(rows)


def _same_pulse_pair_summary_markdown(summary: dict[str, float | int], *, title: str) -> str:
    if not summary or int(summary.get("pair_count", 0)) == 0:
        return f"_No same-pulse detector pairs were available for {title}._"

    def _fmt(value: float | int | None) -> str:
        if value is None:
            return "n/a"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "n/a"
        if not np.isfinite(numeric):
            return "n/a"
        return f"{numeric:.4f}"

    return "\n".join(
        [
            f"**{title}**",
            "",
            f"- Pulse groups with pairs: {int(summary.get('groups_with_pairs', 0)):,}",
            f"- Pair count: {int(summary.get('pair_count', 0)):,}",
            f"- Min dx: {_fmt(summary.get('min_dx'))} cm",
            f"- Min dy: {_fmt(summary.get('min_dy'))} cm",
            f"- Min dr: {_fmt(summary.get('min_dr'))} cm",
            f"- Median dr: {_fmt(summary.get('median_dr'))} cm",
        ]
    )


def plot_fdm(
    dld_df: pd.DataFrame,
    species: list[dict],
    *,
    save_dir: str | Path | None = None,
    all_stem: str | None = None,
    species_stem: str | None = None,
) -> None:
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
    _show_figure(fig, save_dir=save_dir, stem=all_stem)

    mc_col = _pick_mc_col(dld_df)
    if not species or mc_col is None:
        return

    mc = dld_df[mc_col].to_numpy()
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
    _show_figure(fig, save_dir=save_dir, stem=species_stem)


def plot_multihit_and_deadzone(
    dld_df: pd.DataFrame,
    *,
    save_dir: str | Path | None = None,
    save_stem: str | None = None,
) -> None:
    """Multi-hit fraction + delta_p (pulses since previous event) histogram.

    Two ``multi`` encoding conventions are handled automatically:

    - **Convention A** (processed files): single hit = 1, two ions = 2, ...
      Multi-hit events satisfy ``multi > 1``.
    - **Convention B** (some raw files): single hit = 0, two ions = 1, ...
      Multi-hit events satisfy ``multi > 0``.

    The convention is auto-detected from ``multi.min()``.
    """
    if "multi" not in dld_df.columns or "delta_p" not in dld_df.columns:
        _md("_`multi` / `delta_p` columns not present — skipping multi-hit diagnostics._")
        return
    multi   = dld_df["multi"].to_numpy()
    delta_p = dld_df["delta_p"].to_numpy()

    # ── Validate multi column ──────────────────────────────────────────────
    # Some files store multi as a float placeholder (all 0.0) or as a column
    # that was never computed.  Detect this by checking:
    #   • all values equal (no variation) AND value is 0
    multi_is_valid = not (multi.max() == multi.min() == 0)

    # ── Detect encoding convention (only when column is valid) ─────────────
    # • Convention A (processed files): single hit = 1, two ions = 2, …
    # • Convention B (some raw files) : single hit = 0, two ions = 1, …
    if multi_is_valid:
        multi_min    = int(multi.min())
        single_val   = multi_min       # 0 (conv B) or 1 (conv A)
        multi_for_plot = multi[multi >= single_val]
        multi_max    = int(multi_for_plot.max()) if multi_for_plot.size else single_val
    else:
        single_val   = 0
        multi_for_plot = np.array([], dtype=int)
        multi_max    = 0

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # Multi-hit distribution
    if multi_is_valid and multi_for_plot.size > 0:
        bins = np.arange(single_val - 0.5, multi_max + 1.5)
        axes[0].hist(multi_for_plot, bins=bins)
    else:
        axes[0].text(0.5, 0.5,
                     "multi column is all-zero\n(not populated in this file)",
                     ha="center", va="center", transform=axes[0].transAxes, color="gray")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("multi (ions per pulse)")
    axes[0].set_ylabel("Count (log)")
    axes[0].set_title("Multi-hit distribution")

    # Pulse-to-pulse interval (delta_p)
    delta_p_pos = delta_p[delta_p > 0]
    if delta_p_pos.size:
        # Use 95th percentile as the upper limit so that rare counter-reset
        # outliers (e.g. wrap-around at 2^32) do not compress the main
        # distribution into a single invisible bar on the left.
        upper = int(np.percentile(delta_p_pos, 95))
        axes[1].hist(delta_p_pos, bins=80, range=(0, max(upper, 10)))
    else:
        axes[1].text(0.5, 0.5, "delta_p is all-zero\nin this file",
                     ha="center", va="center", transform=axes[1].transAxes, color="gray")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("delta_p (pulses since previous event)")
    axes[1].set_ylabel("Count (log)")
    axes[1].set_title("Pulse-to-pulse interval")
    fig.tight_layout()
    _show_figure(fig, save_dir=save_dir, stem=save_stem)

    n_total = int(multi.size)
    if multi_is_valid:
        n_multi   = int((multi > single_val).sum())
        conv_note = f"(encoding: single-hit = {single_val})"
        multi_line = f"- Events with multi > {single_val}: {_format_pct(n_multi, n_total)} {conv_note}"
    else:
        multi_line = "- Multi-hit: _column not populated in this file (all-zero)_"

    _md(
        "**Multi-hit summary**\n\n"
        f"- Total events: {n_total:,}\n"
        f"{multi_line}\n"
        f"- delta_p median: {int(np.median(delta_p_pos)) if delta_p_pos.size else 'n/a (all-zero in this file)'}\n"
    )


# ---------------------------------------------------------------------------
# Top-level analysis runner
# ---------------------------------------------------------------------------


def run_analysis(variables, species: list[dict], *, save_plots: bool = False) -> None:
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

    save_dir = _analysis_save_directory(variables, save_plots)
    if save_plots:
        if save_dir is not None:
            _md(f"_Saving plots to:_ `{save_dir}`")
        else:
            _md("_Save plots was enabled, but the dataset path could not be resolved. Skipping plot export._")

    plot_df = dld_df
    if detector_kind == "surface_concept" and tdc_df is not None and len(tdc_df) > 0:
        from pyccapt.calibration.data_tools.raw_data_workflow import (
            analyze_surface_concept_tdc_frame,
            compute_same_pulse_detector_separations,
            plot_detector_dead_zone_and_neighbors,
            plot_detector_overview,
            plot_partial_hit_efficiency_maps,
            plot_same_pulse_detector_separations,
            plot_signal_overlay_by_dlts,
            plot_surface_concept_peak_breakdown,
            plot_surface_concept_peak_ratio_table,
            plot_surface_concept_recovery_summary,
            plot_surface_concept_recovery_yield,
            plot_surface_concept_sequence_statistics,
            summarize_surface_concept_peak_windows,
            surface_concept_hits_to_processed_dataframe,
        )

        flight_path_length = float(getattr(variables, "flight_path_length", 110.0) or 110.0)
        pulse_mode = str(getattr(variables, "pulse_mode", "voltage") or "voltage")
        analysis = analyze_surface_concept_tdc_frame(
            tdc_df,
            flight_path_length_mm=flight_path_length,
            pulse_mode=pulse_mode,
            t0=0.0,
            show_progress=True,
        )

        if not analysis["hit_table"].empty:
            plot_df = surface_concept_hits_to_processed_dataframe(
                analysis["hit_table"],
                pulse_mode=pulse_mode,
            )

        _md("## DLTS-per-pulse")
        _show_figure(
            plot_surface_concept_sequence_statistics(analysis["sequence_stats"]),
            save_dir=save_dir,
            stem="dlts_per_pulse",
        )
        _md(_surface_concept_raw_summary_markdown(analysis["raw_summary"], analysis["recovery_stats"]))
        _md(_surface_concept_length_breakdown_markdown(analysis["sequence_stats"]))

        _md("## Recovery summary")
        _show_figure(
            plot_surface_concept_recovery_summary(analysis["recovery_stats"]),
            save_dir=save_dir,
            stem="surface_concept_recovery_summary",
        )
        _show_figure(
            plot_surface_concept_recovery_yield(analysis["recovery_diagnostics"]),
            save_dir=save_dir,
            stem="surface_concept_recovery_yield",
        )
        _show_figure(
            plot_partial_hit_efficiency_maps(analysis["recovery_diagnostics"]),
            save_dir=save_dir,
            stem="surface_concept_partial_hit_efficiency",
        )

        if not analysis["hit_table"].empty:
            windows = _species_to_windows(species)
            _md("## Time-of-flight overlay by recovered DLTS class")
            _show_figure(
                plot_signal_overlay_by_dlts(
                    analysis["hit_table"],
                    signal_kind="tof",
                    max_value=1000.0,
                    bin_size=0.1,
                    only_in_detector=True,
                    title="Recovered Surface Concept TOF overlay",
                ),
                save_dir=save_dir,
                stem="surface_concept_tof_overlay",
            )

            _md("## Mass/charge overlay by recovered DLTS class")
            _show_figure(
                plot_signal_overlay_by_dlts(
                    analysis["hit_table"],
                    signal_kind="mc",
                    max_value=40.0,
                    bin_size=0.1,
                    only_in_detector=True,
                    title="Recovered Surface Concept mass/charge overlay",
                ),
                save_dir=save_dir,
                stem="surface_concept_mc_overlay",
            )

            _md("## Recovered detector maps")
            _show_figure(
                plot_detector_overview(
                    analysis["hit_table"],
                    detector_limit_cm=4.0,
                    only_in_detector=True,
                    title_prefix="Recovered Surface Concept",
                ),
                save_dir=save_dir,
                stem="surface_concept_detector_overview",
            )

            _md("## Peak-window recovery breakdown")
            peak_summary = summarize_surface_concept_peak_windows(
                analysis["hit_table"],
                analysis["recovery_diagnostics"],
                windows,
                signal_kind="mc",
                only_in_detector=True,
            )
            _show_figure(
                plot_surface_concept_peak_breakdown(peak_summary),
                save_dir=save_dir,
                stem="surface_concept_peak_breakdown",
            )
            ratio_table = peak_summary["ratios"]
            if isinstance(ratio_table, pd.DataFrame) and not ratio_table.empty:
                _show_figure(
                    plot_surface_concept_peak_ratio_table(ratio_table),
                    save_dir=save_dir,
                    stem="surface_concept_peak_ratio_table",
                )
                _md(_surface_concept_peak_ratio_markdown(ratio_table))
                display(ratio_table)

            _md("## Dead-zone / nearest-neighbor diagnostics")
            _show_figure(
                plot_detector_dead_zone_and_neighbors(analysis["hit_table"]),
                save_dir=save_dir,
                stem="surface_concept_dead_zone_neighbors",
            )

            pair_table, pair_summary = compute_same_pulse_detector_separations(
                analysis["hit_table"],
                only_in_detector=True,
                dlts_values=[4],
                show_progress=True,
            )
            _md(_same_pulse_pair_summary_markdown(pair_summary, title="Same-pulse pairwise separations (4 DLTS only)"))
            _show_figure(
                plot_same_pulse_detector_separations(
                    pair_table,
                    bin_size=0.1,
                    title_prefix="Same-pulse separations",
                ),
                save_dir=save_dir,
                stem="surface_concept_same_pulse_separations",
            )

    if detector_kind != "surface_concept" or tdc_df is None or len(tdc_df) == 0:
        _md("## DLTS-per-pulse")
        plot_dlts_per_pulse(tdc_df, detector_kind, save_dir=save_dir, save_stem="dlts_per_pulse")

    _md("## Time-of-flight")
    plot_tof_with_peaks(plot_df, species, save_dir=save_dir, save_stem="tof_histogram")

    _md("## Mass/charge")
    plot_mc_with_peaks(plot_df, species, save_dir=save_dir, save_stem="mc_histogram")

    _md("## Field desorption map")
    plot_fdm(plot_df, species, save_dir=save_dir, all_stem="fdm_all", species_stem="fdm_species")

    _md("## Multi-hit / dead-zone")
    plot_multihit_and_deadzone(plot_df, save_dir=save_dir, save_stem="multihit_deadzone")


# ---------------------------------------------------------------------------
# UI: single panel with peak-source dropdown
# ---------------------------------------------------------------------------


def _build_manual_rows() -> list[tuple[widgets.Text, widgets.FloatText, widgets.FloatText]]:
    rows = []
    for index in range(1, 7):
        label = widgets.Text(value=f"Peak {index}", description=f"Peak {index}:", layout=widgets.Layout(width="220px"))
        low = widgets.FloatText(value=0.0, description="tof/mc_low:", layout=widgets.Layout(width="190px"))
        high = widgets.FloatText(value=0.0, description="tof/mc_up:", layout=widgets.Layout(width="190px"))
        rows.append((label, low, high))
    return rows


def _set_rows_disabled(rows, disabled: bool) -> None:
    for label, low, high in rows:
        label.disabled = disabled
        low.disabled = disabled
        high.disabled = disabled


def call_auto_raw_data_analysis(variables) -> None:
    """Display a single-panel analysis UI driven by a peak-source dropdown.

    The dropdown selects either:

    - **Manual peak windows** — the user types up to six ``(label, tof/mc_low,
      tof/mc_up)`` triples below; rows left at 0/0 are skipped.
    - **From range file** — the species list is derived from
      ``variables.range_data``; the manual rows are disabled.

    Either way, clicking *Run analysis* renders the same set of plots
    (DLTS-per-pulse, TOF, M/C, FDM, multi-hit) plus an inline Markdown
    summary beneath each section.
    """
    range_df = getattr(variables, "range_data", None)
    range_species = species_from_range(range_df)
    has_range = bool(range_species)

    out = Output()
    summary = widgets.HTML()

    def _refresh_summary(*_):
        if peak_source.value == "range":
            if has_range:
                summary.value = (
                    f"Range table loaded with <b>{len(range_species)}</b> usable rows. "
                    "Click <i>Run analysis</i> to plot."
                )
            else:
                summary.value = (
                    "<span style='color:#b91c1c;'>No range table is loaded. "
                    "Switch to <b>Manual peak windows</b> or load a range table first.</span>"
                )
        else:
            summary.value = (
                "<i>Type peak windows below. Rows with both tof/mc fields = 0 are skipped.</i>"
            )

    peak_source = widgets.Dropdown(
        options=[("Manual peak windows", "manual"), ("From range file", "range")],
        value="range" if has_range else "manual",
        description="Peak source:",
        layout=widgets.Layout(width="320px"),
    )

    manual_rows = _build_manual_rows()
    manual_grid = widgets.VBox([
        widgets.HBox([label, low, high]) for label, low, high in manual_rows
    ])

    save_plots = widgets.Checkbox(
        value=False,
        description="Save plots",
        indent=False,
        tooltip="When enabled, save every figure beside the dataset as SVG and PNG (300 dpi).",
    )
    run_button = widgets.Button(description="Run analysis", button_style="primary")

    def _on_source_change(_change):
        _set_rows_disabled(manual_rows, peak_source.value == "range")
        _refresh_summary()

    def _on_run(_):
        out.clear_output()
        with out:
            if peak_source.value == "range":
                if not range_species:
                    _md("**Range table is empty** — switch to *Manual peak windows*.")
                    return
                species = range_species
            else:
                try:
                    species = species_from_manual(manual_rows)
                except ValueError as exc:
                    _md(f"**Input error:** {exc}")
                    return
                if not species:
                    _md("_All peak windows are empty — nothing to analyze._")
                    return
            run_analysis(variables, species, save_plots=bool(save_plots.value))

    peak_source.observe(_on_source_change, names="value")
    run_button.on_click(_on_run)

    # Initialize disabled state to match the dropdown's starting value.
    _set_rows_disabled(manual_rows, peak_source.value == "range")
    _refresh_summary()

    panel = widgets.VBox([
        peak_source,
        summary,
        save_plots,
        manual_grid,
        run_button,
        out,
    ])
    display(panel)
