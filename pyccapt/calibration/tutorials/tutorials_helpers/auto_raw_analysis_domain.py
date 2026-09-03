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
    """Return the detector family inferred from the channel range in ``tdc_df``.

    - ``'single_delay_line'`` — only channels 0-1 ever appear (one delay line,
      two channels). A complete event is "2 DLTS".
    - ``'surface_concept'`` — channels go up to 3 (two delay lines, four
      channels). A complete event is "4 DLTS".
    - ``'roentdek'`` — channels go up to 5 (three delay lines / hexanode, six
      channels). A complete event is "6 DLTS".
    - ``'unknown'`` — anything outside that.
    """
    if tdc_df is None or "channel" not in tdc_df.columns or len(tdc_df) == 0:
        return "unknown"
    max_channel = int(np.max(tdc_df["channel"].to_numpy()))
    if max_channel <= 1:
        return "single_delay_line"
    if max_channel <= 3:
        return "surface_concept"
    if max_channel <= 5:
        return "roentdek"
    return "unknown"

def _delay_line_pairs(detector_kind: str) -> list[tuple[int, int]]:
    """Return the list of ``(low_channel, high_channel)`` pairs, one per delay line.

    A "complete event" needs every pair to fire. A "partial" hit fires only a
    subset of pairs. The number of channels per complete event is therefore
    ``2 * len(pairs)`` (= 2 for 1-DL, 4 for SC, 6 for RoentDek).
    """
    if detector_kind == "single_delay_line":
        return [(0, 1)]
    if detector_kind == "surface_concept":
        return [(0, 1), (2, 3)]
    if detector_kind == "roentdek":
        return [(0, 1), (2, 3), (4, 5)]
    return []

def expected_dlts_full(detector_kind: str) -> int:
    """Number of DLTS that constitute a complete event for the detector."""
    pairs = _delay_line_pairs(detector_kind)
    return 2 * len(pairs)

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
        species.append(
            {
                "label": name,
                "mc_low": mc_low,
                "mc_up": mc_up,
                "color": str(row.get("color", "#1f77b4")),
            }
        )
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
    """Close ``fig`` only when the active matplotlib backend renders the
    figure as a static image.

    With ``%matplotlib inline`` (the default in pure Jupyter Notebook) the
    figure is rendered to a PNG by the time ``display(fig)`` returns, so
    closing immediately frees memory and avoids "<Figure size … with 0 Axes>"
    duplicate prints.

    With an interactive backend like ``%matplotlib widget`` (``ipympl``)
    the displayed figure stays a live canvas — closing it here would freeze
    the canvas and disable zoom / pan / resize / save. So we just skip the
    close in that case and let ipympl manage the lifecycle.
    """
    backend = plt.get_backend().lower()
    if 'ipympl' in backend or 'nbagg' in backend or 'widget' in backend or 'qt' in backend or 'tk' in backend:
        return
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
            formats=("pdf", "png"),
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

def _classify_pulse_chunks(df: pd.DataFrame, group_col: str, detector_kind: str) -> dict:
    """Walk every pulse group and classify each chunk-of-N as complete / midtier / partial.

    Per pulse the channels are sorted, then split into ``ceil(L/N)``
    consecutive non-overlapping chunks (``N = 2 * len(delay-line pairs)``).
    Each chunk emits **one** classification, so a length-8 SC pulse that
    contains two valid 4-DLTS sub-events contributes 2 to the "complete" bar
    at x=8 — same semantics as the legacy
    ``raw_data_surface_concept.find_consecutive_sequences``.

    Categories:

    - ``complete`` — every delay-line pair fires inside the chunk.
    - ``midtier`` — RoentDek-only intermediate ("4 DLTS"): exactly two of the
      three pairs fully present in the chunk.
    - ``partial`` — at least one pair fully present and the chunk is *not*
      complete/midtier. Length-2 chunks like ``[0, 1]`` count: physically a
      ``(0, 1)`` pair is reconstructible in x even when only those two
      timestamps fire, so we count them as 2-DLTS hits rather than throwing
      them away as noise. (The legacy notebook's ``len(chs) > 2`` guard
      *excluded* these length-2 chunks; we deliberately diverge from that
      behaviour because the more permissive interpretation is the
      physics-correct one.)

    The return value is a dict of arrays, each entry being the **pulse length**
    of the parent pulse:

    - ``frequency`` : one entry per pulse (drives the gray "Frequency" bar).
    - ``complete``  : one entry per complete chunk.
    - ``midtier``   : one entry per midtier chunk (always empty for non-RD).
    - ``partial``   : one entry per partial chunk.

    .. note::
        This classifier sorts by channel before chunking. For *position
        reconstruction* (det_x / det_y) sort order is suboptimal because
        timestamps that are physically adjacent in the TDC sequence are more
        likely to come from the same physical event; a future smarter
        grouping would walk the sequence and try alternative partitions,
        dropping any candidate event that lands outside the detector. For
        the histogram-only classification used here, sort-then-chunk is
        adequate.
    """
    pairs = _delay_line_pairs(detector_kind)
    empty = {
        "frequency": np.array([], dtype=int),
        "complete": np.array([], dtype=int),
        "midtier": np.array([], dtype=int),
        "partial": np.array([], dtype=int),
    }
    if not pairs or len(df) == 0:
        return empty
    n_per_chunk = 2 * len(pairs)
    n_pairs = len(pairs)

    freq_lengths: list[int] = []
    complete_lengths: list[int] = []
    midtier_lengths: list[int] = []
    partial_lengths: list[int] = []

    for _, sub in df.groupby(group_col, sort=False):
        channels_sorted = np.sort(sub["channel"].to_numpy())
        pulse_len = int(channels_sorted.size)
        if pulse_len == 0:
            continue
        freq_lengths.append(pulse_len)

        # ceil(L / N) chunks — each iteration emits one classification.
        # Trailing chunks shorter than N still emit a label as long as they
        # contain at least one fully-fired delay-line pair.
        n_iter = (pulse_len + n_per_chunk - 1) // n_per_chunk
        for i in range(n_iter):
            chunk = channels_sorted[i * n_per_chunk : (i + 1) * n_per_chunk]
            chunk_len = chunk.size
            chunk_set = set(int(c) for c in chunk)
            pairs_present = sum(1 for lo, hi in pairs if lo in chunk_set and hi in chunk_set)

            if chunk_len == n_per_chunk and pairs_present == n_pairs:
                complete_lengths.append(pulse_len)
            elif n_pairs == 3 and pairs_present == 2:
                # RoentDek "4 DLTS" — two of three delay lines fired.
                midtier_lengths.append(pulse_len)
            elif pairs_present >= 1:
                # At least one pair fully present → reconstructible in one
                # direction → 2-DLTS partial. No length guard: a length-2
                # chunk with channels [0, 1] (or [2, 3]) is a valid partial
                # hit, not noise.
                partial_lengths.append(pulse_len)
            # else: no full pair anywhere in the chunk → drops to "noise".

    return {
        "frequency": np.array(freq_lengths, dtype=int),
        "complete": np.array(complete_lengths, dtype=int),
        "midtier": np.array(midtier_lengths, dtype=int),
        "partial": np.array(partial_lengths, dtype=int),
    }

def plot_dlts_per_pulse(
    tdc_df: pd.DataFrame,
    detector_kind: str,
    *,
    save_dir: str | Path | None = None,
    save_stem: str | None = None,
) -> None:
    """DLTS-per-pulse histogram with chunked, detector-aware classification.

    The bars at each x-position (= number of TDC signals per pulse) are:

    • **Gray**  – total frequency (all pulses at that DLTS count, **per pulse**).
    • **Orange** – "2 DLTS": one delay-line pair fired (legacy partial).
    • **Blue**  – Surface Concept "4 DLTS" / RoentDek "4 DLTS" (intermediate).
    • **Green** – RoentDek "6 DLTS": all three delay-line pairs fired.
    • For 1-delay-line systems the "2 DLTS" complete event is shown in blue.

    This reproduces Figure 9A (Surface Concept) and 9B (RoentDek) of the
    PyCCAPT paper. Each chunk of N timestamps inside a pulse contributes one
    entry, so a length-8 pulse with two valid events shows up twice in the
    blue bar at x=8 — matching the legacy ``find_consecutive_sequences``
    behaviour.
    """
    if tdc_df is None or len(tdc_df) == 0 or "event_group_id" not in tdc_df.columns:
        _md("_No raw tdc loaded with linking — skipping DLTS breakdown._")
        return
    if "channel" not in tdc_df.columns:
        _md("_No `channel` column in tdc data — cannot classify DLTS groups._")
        return
    pairs = _delay_line_pairs(detector_kind)
    if not pairs:
        _md(f"_Detector kind `{detector_kind}` is not supported for DLTS classification — skipping the per-pulse breakdown._")
        return

    matched = tdc_df[tdc_df["has_dld_match"]]
    orphans = tdc_df[~tdc_df["has_dld_match"]]

    m = _classify_pulse_chunks(matched, "event_group_id", detector_kind)
    # Orphan rows share event_group_id = -1; group them by start_counter instead.
    o = _classify_pulse_chunks(orphans, "start_counter", detector_kind)

    def _join(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.size and b.size:
            return np.concatenate([a, b])
        return a if a.size else b

    freq_arr = _join(m["frequency"], o["frequency"])
    complete_arr = _join(m["complete"], o["complete"])
    midtier_arr = _join(m["midtier"], o["midtier"])
    partial_arr = _join(m["partial"], o["partial"])

    if freq_arr.size == 0:
        return

    max_n = int(freq_arr.max())
    bins = np.arange(0.5, max_n + 1.5)
    centers = np.arange(1, max_n + 1)

    freq_hist = np.histogram(freq_arr, bins=bins)[0]
    complete_hist = np.histogram(complete_arr, bins=bins)[0]
    midtier_hist = np.histogram(midtier_arr, bins=bins)[0]
    partial_hist = np.histogram(partial_arr, bins=bins)[0]

    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.2
    ax.bar(centers, freq_hist, width=w * 2, label="Frequency", alpha=0.5, color="gray")

    if detector_kind == "roentdek":
        # 3 narrow bars centred on x, offset by -w / 0 / +w (Figure 9B style).
        ax.bar(centers - w, partial_hist, width=w, label="2 DLTS", color="orange")
        ax.bar(centers, midtier_hist, width=w, label="4 DLTS", color="blue")
        ax.bar(centers + w, complete_hist, width=w, label="6 DLTS", color="green")
    elif detector_kind == "surface_concept":
        # 2 narrow bars (Figure 9A style).
        ax.bar(centers - 0.5 * w, partial_hist, width=w, label="2 DLTS", color="orange")
        ax.bar(centers + 0.5 * w, complete_hist, width=w, label="4 DLTS", color="blue")
    else:  # single_delay_line: a "complete" event is a 2-DLTS hit.
        ax.bar(centers, complete_hist, width=w, label="2 DLTS", color="blue")

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

    n_full = expected_dlts_full(detector_kind)
    total_pulses = int(freq_arr.size)
    matched_total = int(m["frequency"].size)
    n_complete_tot = int(complete_arr.size)
    n_midtier_tot = int(midtier_arr.size)
    n_partial_tot = int(partial_arr.size)

    lines = [
        "**DLTS-per-pulse breakdown**",
        "",
        f"- Detector kind: `{detector_kind}`" + (f" (complete event = {n_full} DLTS)" if n_full else ""),
        f"- Total pulse triggers: {total_pulses:,}",
        f"- Linked to a dld row (has position): {_format_pct(matched_total, total_pulses)}",
        f"- Orphan (no dld counterpart):        {_format_pct(total_pulses - matched_total, total_pulses)}",
    ]
    if detector_kind == "roentdek":
        lines.extend(
            [
                f"- **6 DLTS** — complete (all 3 delay lines fired): {_format_pct(n_complete_tot, total_pulses)}",
                f"- **4 DLTS** — two of three delay lines fired:    {_format_pct(n_midtier_tot, total_pulses)}",
                f"- **2 DLTS** — one delay line fired:              {_format_pct(n_partial_tot, total_pulses)}",
            ]
        )
    elif detector_kind == "surface_concept":
        lines.extend(
            [
                f"- **4 DLTS** — complete (both delay lines fired): {_format_pct(n_complete_tot, total_pulses)}",
                f"- **2 DLTS** — one delay line fired:              {_format_pct(n_partial_tot, total_pulses)}",
            ]
        )
    else:  # single_delay_line
        lines.append(
            f"- **2 DLTS** — complete (single delay line):      {_format_pct(n_complete_tot, total_pulses)}",
        )
    _md("\n".join(lines))

def _pick_tof_col(dld_df: pd.DataFrame) -> str | None:
    """Return the best available TOF column name.

    Prefers ``t_c (ns)`` (calibrated) when it has non-zero values; otherwise
    ``t (ns)`` (raw). Returns ``None`` when neither is usable.
    """
    if "t_c (ns)" in dld_df.columns and (dld_df["t_c (ns)"] != 0).any():
        return "t_c (ns)"
    if "t (ns)" in dld_df.columns:
        return "t (ns)"
    return None

def plot_tof_with_peaks(
    dld_df: pd.DataFrame,
    species: list[dict],
    *,
    peak_units: str = "tof",
    save_dir: str | Path | None = None,
    save_stem: str | None = None,
) -> None:
    """TOF histogram. When ``peak_units == 'tof'`` the species windows are
    overlaid as shaded vertical bands (just like the legacy notebook does for
    its TOF peaks). When ``peak_units == 'mc'`` the histogram is plotted
    without overlays — the species ranges are in mc units and would not
    correspond to TOF positions."""
    tof_col = _pick_tof_col(dld_df)
    if tof_col is None:
        return
    tof = dld_df[tof_col].to_numpy()
    if tof.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(tof, bins=400, color="#1f77b4", log=True)
    if peak_units == "tof":
        for sp in species:
            ax.axvspan(sp["mc_low"], sp["mc_up"], color=sp.get("color", "#1f77b4"), alpha=0.25, label=sp["label"])
        if species:
            ax.legend(loc="upper right", fontsize=8)
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

def plot_signal_overview(
    dld_df: pd.DataFrame,
    *,
    save_dir: str | Path | None = None,
    save_stem: str | None = None,
) -> None:
    """Render a quick preview of every available signal column on the dld frame.

    For each of ``t (ns)``, ``t_c (ns)``, ``mc (Da)``, ``mc_uc (Da)`` that is
    present and has at least one non-zero value, draw a log-y histogram in a
    grid. When a column is all-zero (e.g. ``t_c (ns)`` on a never-calibrated
    file, or ``mc_uc (Da)`` when the raw inputs were missing) it is reported
    as "not available" instead of plotted.

    This helper is meant to drive a *preview cell* placed before the analysis
    cell in the raw-data-analysis notebook, so the user can decide which
    column to enter peak windows in (TOF vs mc) before clicking *Run analysis*.
    """
    if dld_df is None or len(dld_df) == 0:
        _md("_No dld data is loaded — nothing to preview._")
        return

    candidates = ("t (ns)", "t_c (ns)", "mc (Da)", "mc_uc (Da)")
    available: list[tuple[str, np.ndarray]] = []
    missing_or_zero: list[str] = []
    for col in candidates:
        if col not in dld_df.columns:
            missing_or_zero.append(f"`{col}` (column missing)")
            continue
        arr = dld_df[col].to_numpy()
        if arr.size == 0 or not (arr != 0).any():
            missing_or_zero.append(f"`{col}` (all zero)")
            continue
        available.append((col, arr))

    if not available:
        _md("_None of `t (ns)`, `t_c (ns)`, `mc (Da)`, `mc_uc (Da)` carry usable values — cannot render the preview._")
        return

    n = len(available)
    cols_layout = min(2, n)
    rows_layout = (n + cols_layout - 1) // cols_layout
    fig, axes = plt.subplots(rows_layout, cols_layout, figsize=(6.0 * cols_layout, 3.2 * rows_layout), squeeze=False)
    for index, (col, arr) in enumerate(available):
        ax = axes[index // cols_layout][index % cols_layout]
        if col.endswith("(ns)"):
            upper = float(np.percentile(arr, 99.5))
            ax.hist(arr, bins=400, range=(0, max(upper, 1.0)), color="#1f77b4", log=True)
        else:
            upper = float(np.percentile(arr, 99.5))
            ax.hist(arr, bins=400, range=(0, max(upper, 50.0)), color="#555", log=True)
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Count (log)")
    for index in range(n, rows_layout * cols_layout):
        axes[index // cols_layout][index % cols_layout].axis("off")
    fig.tight_layout()
    _show_figure(fig, save_dir=save_dir, stem=save_stem)

    summary_lines = ["**Signal preview**", ""]
    for col, arr in available:
        finite = arr[np.isfinite(arr)]
        summary_lines.append(
            f"- `{col}`: {arr.size:,} values, "
            f"min = {float(finite.min()):.3f}, "
            f"max = {float(finite.max()):.3f}, "
            f"median = {float(np.median(finite)):.3f}"
        )
    if missing_or_zero:
        summary_lines.append("")
        summary_lines.append("_Skipped: " + ", ".join(missing_or_zero) + "._")
    _md("\n".join(summary_lines))

def plot_full_spectrum(
    dld_df: pd.DataFrame,
    *,
    save_dir: str | Path | None = None,
    save_stem: str | None = None,
) -> None:
    """Render the full mass spectrum + full time-of-flight on one figure.

    No peak shading, no DLTS-class overlay — just the unfiltered log-y
    histograms covering every event in the dataset including noise, the
    way a "normal APT mass spectrum" view typically looks. The TOF panel
    uses the calibrated ``t_c (ns)`` column when present, otherwise falls
    back to ``t (ns)``; the M/C panel uses ``mc (Da)`` when populated and
    falls back to ``mc_uc (Da)`` otherwise. Panels with no usable column
    are silently dropped, so a pure raw acquisition file (no ``t_c``, no
    calibrated mc) gets ``t (ns)`` + ``mc_uc (Da)`` instead.
    """
    if dld_df is None or len(dld_df) == 0:
        _md("_No dld data is loaded — skipping full-spectrum plot._")
        return
    tof_col = _pick_tof_col(dld_df)
    mc_col = _pick_mc_col(dld_df)
    if tof_col is None and mc_col is None:
        _md("_No usable TOF or mass/charge column — skipping full-spectrum plot._")
        return

    n_panels = (1 if tof_col else 0) + (1 if mc_col else 0)
    # Cap each dimension well below matplotlib's 2^16-pixel hard limit so a
    # downstream ``bbox_inches='tight'`` save / inline render can never blow
    # up into the multi-million-pixel range that triggered
    # "Image size too large" on user input with many peaks.
    height_inches = min(max(3.4 * n_panels, 3.6), 20.0)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(10.0, height_inches),
        squeeze=False,
    )
    panel_index = 0

    if tof_col:
        ax = axes[panel_index][0]
        tof = dld_df[tof_col].to_numpy()
        positive = tof[np.isfinite(tof) & (tof > 0)]
        upper = float(np.percentile(positive, 99.5)) if positive.size else 1000.0
        ax.hist(tof, bins=600, range=(0, max(upper, 1.0)), color="#1f77b4", log=True, histtype="stepfilled")
        ax.set_xlabel(tof_col)
        ax.set_ylabel("Count (log)")
        ax.set_title(f"Full time-of-flight spectrum ({tof_col})")
        panel_index += 1

    if mc_col:
        ax = axes[panel_index][0]
        mc = dld_df[mc_col].to_numpy()
        positive = mc[np.isfinite(mc) & (mc > 0)]
        upper = float(np.percentile(positive, 99.5)) if positive.size else 50.0
        ax.hist(mc, bins=600, range=(0, max(upper, 50.0)), color="#444444", log=True, histtype="stepfilled")
        ax.set_xlabel(mc_col)
        ax.set_ylabel("Count (log)")
        ax.set_title(f"Full mass spectrum ({mc_col})")
        panel_index += 1

    fig.tight_layout()
    _show_figure(fig, save_dir=save_dir, stem=save_stem)

def plot_mc_with_peaks(
    dld_df: pd.DataFrame,
    species: list[dict],
    *,
    peak_units: str = "tof",
    save_dir: str | Path | None = None,
    save_stem: str | None = None,
) -> None:
    """Histogram of calibrated mc. Species windows are overlaid only when
    ``peak_units == 'mc'`` (otherwise the user-typed values are TOF, not mc,
    and shading them on this axis would be misleading)."""
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
    if peak_units == "mc":
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
    if peak_units != "mc":
        # The user's species values are TOF windows — the per-peak MRP table
        # would be meaningless against the mc axis. Skip it; the per-peak
        # breakdown for TOF mode is rendered by the SC / RoentDek workflow
        # against the chosen signal kind.
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

    md = [
        "**Per-peak counts and MRP(0.5)**",
        "",
        "| Peak | mc_low | mc_up | Count | % of all | MRP(0.5) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
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

def _roentdek_raw_summary_markdown(raw_summary: dict[str, object]) -> str:
    channel_totals = raw_summary.get('channel_timestamp_totals', {})
    channel_text = ", ".join(f"ch{channel}={int(channel_totals.get(channel, 0)):,}" for channel in range(1, 7))
    return "\n".join(
        [
            "**RoentDek raw summary**",
            "",
            f"- Total grouped pulses: {int(raw_summary.get('total_events', 0)):,}",
            f"- Total delay-line timestamps: {int(raw_summary.get('total_timestamps', 0)):,}",
            f"- Events with recovered patterns: {int(raw_summary.get('matched_pattern_events', 0)):,}",
            f"- Invalid pattern events: {int(raw_summary.get('invalid_pattern_events', 0)):,}",
            f"- Events with leftover unmatched timestamps: {int(raw_summary.get('unmatched_pattern_events', 0)):,}",
            f"- Multi-hit events: {int(raw_summary.get('multi_hit_events', 0)):,}",
            f"- Channel timestamp totals: {channel_text}",
        ]
    )

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
    multi = dld_df["multi"].to_numpy()
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
        multi_min = int(multi.min())
        single_val = multi_min  # 0 (conv B) or 1 (conv A)
        multi_for_plot = multi[multi >= single_val]
        multi_max = int(multi_for_plot.max()) if multi_for_plot.size else single_val
    else:
        single_val = 0
        multi_for_plot = np.array([], dtype=int)
        multi_max = 0

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # Multi-hit distribution
    if multi_is_valid and multi_for_plot.size > 0:
        bins = np.arange(single_val - 0.5, multi_max + 1.5)
        axes[0].hist(multi_for_plot, bins=bins)
    else:
        axes[0].text(
            0.5,
            0.5,
            "multi column is all-zero\n(not populated in this file)",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
            color="gray",
        )
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
        axes[1].text(
            0.5, 0.5, "delta_p is all-zero\nin this file", ha="center", va="center", transform=axes[1].transAxes, color="gray"
        )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("delta_p (pulses since previous event)")
    axes[1].set_ylabel("Count (log)")
    axes[1].set_title("Pulse-to-pulse interval")
    fig.tight_layout()
    _show_figure(fig, save_dir=save_dir, stem=save_stem)

    n_total = int(multi.size)
    if multi_is_valid:
        n_multi = int((multi > single_val).sum())
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
