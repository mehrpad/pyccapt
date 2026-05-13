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

# ---------------------------------------------------------------------------
# Top-level analysis runner
# ---------------------------------------------------------------------------

def run_analysis(
    variables,
    species: list[dict],
    *,
    save_plots: bool = False,
    peak_units: str = "tof",
    recovery_mode: str = "exhaustive",
    pair_sum_tolerance_bins: float = 200.0,
) -> None:
    """Render every analysis section against ``variables.data`` and ``variables.data_tdc``.

    ``peak_units`` controls how the user-typed peak windows are interpreted:

    - ``"tof"`` (default) — windows are TOF intervals in nanoseconds. The
      Surface Concept / RoentDek per-peak ratio table is computed against
      ``tof (ns)``. The TOF histogram overlays each window as a shaded band.
    - ``"mc"`` — windows are mass/charge intervals in Daltons. The per-peak
      table is computed against ``mc (Da)`` and the mc histogram overlays
      each window.

    ``recovery_mode`` selects the Surface Concept combinatorial per-pulse
    hit-recovery algorithm:

    - ``"exhaustive"`` (default) — picks the *maximum* index-disjoint subset
      of valid candidates via branch-and-bound. Slow but optimal; falls
      back to greedy when a single pulse generates more than ~80 candidates
      so the worst case stays bounded.
    - ``"greedy"`` — fast O(N²) variant that ranks valid candidates by
      complete-vs-partial first then by signal distance to the nearest peak
      centre, picking the highest-ranked candidate whose timestamps are
      still free at each step. Use when the analysis must run on a tight
      time budget.

    Both modes enumerate every candidate ``(0, 1)`` x-pair, every ``(2, 3)``
    y-pair, and every complete-event quadruple whose x-pair-sum matches the
    y-pair-sum within ``pair_sum_tolerance_bins`` (default ±200 TDC bins
    ≈ 1.4 ns at 6.86 ps/bin). Validity = in-detector AND in any peak window.
    Selection is two-stage: completes lock their timestamps first, then
    partials run on the remaining indices.

    ``pair_sum_tolerance_bins`` controls the coincidence window for forming
    complete-event candidates from individual delay-line pairs.
    """
    peak_units = (peak_units or "tof").lower()
    if peak_units not in {"tof", "mc"}:
        peak_units = "tof"

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
        f"- Species supplied: {len(species)} (interpreted as **{peak_units.upper()}** windows)\n"
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
            analyze_surface_concept_tdc_frame_combinatorial,
            compute_same_pulse_detector_separations,
            load_detector_constants,
            plot_detector_dead_zone_and_neighbors,
            plot_detector_overview,
            plot_partial_hit_efficiency_maps,
            plot_peak_chunk_length_distribution,
            plot_peak_detector_diagnostics,
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

        # Resolve detector geometry from config (with fallback to historical
        # SC defaults). The detector_limit_cm gate is applied to *every* hit
        # — full and partial — inside build_surface_concept_recovery_diagnostics.
        detector_constants = load_detector_constants("surface_concept", getattr(variables, "conf", None))
        detector_limit_cm = float(detector_constants["detector_limit_cm"])
        windows_for_recovery = _species_to_windows(species)

        recovery_mode_value = (recovery_mode or "exhaustive").lower()
        if recovery_mode_value not in {"greedy", "exhaustive"}:
            recovery_mode_value = "exhaustive"

        _md(
            f"_Surface Concept partial recovery mode: **{recovery_mode_value}** "
            f"(combinatorial; pair-sum tolerance ±{int(pair_sum_tolerance_bins)} bins)._"
        )
        # Recovery validity is geometric only — every hit whose tof is in
        # [0, max_tof_ns] AND whose reconstructed position lands inside the
        # detector is emitted. Peak windows are applied DOWNSTREAM, in the
        # per-peak yield table and per-peak diagnostic plots, against the
        # full hit table. This is what keeps the "Full spectrum" view
        # honest — noise events between peaks are still in the hit table.
        max_tof_ns = float(getattr(variables, "max_tof_ns", 5000.0) or 5000.0)
        combinatorial_analysis = analyze_surface_concept_tdc_frame_combinatorial(
            tdc_df,
            peak_windows=windows_for_recovery,
            signal_kind=peak_units,
            detector_limit_cm=detector_limit_cm,
            max_tof_ns=max_tof_ns,
            mode=recovery_mode_value,
            pair_sum_tolerance_bins=float(pair_sum_tolerance_bins),
            t0=0.0,
            flight_path_length_mm=flight_path_length,
            pulse_mode=pulse_mode,
            show_progress=True,
        )
        # The combinatorial path replaces only the partial-recovery branch.
        # ``analyze_surface_concept_tdc_frame`` still produces sequence_stats,
        # raw_summary, and recovery_diagnostics that the rest of the page
        # consumes (DLTS-per-pulse plot, recovery summary, etc.) so we run it
        # too and then swap in the combinatorial hit_table for the parts of
        # the workflow that drive peak yields and FDM.
        analysis = analyze_surface_concept_tdc_frame(
            tdc_df,
            detector_limit_cm=detector_limit_cm,
            flight_path_length_mm=flight_path_length,
            pulse_mode=pulse_mode,
            t0=0.0,
            show_progress=False,
        )
        if not combinatorial_analysis["hit_table"].empty:
            analysis["hit_table"] = combinatorial_analysis["hit_table"]
        counts = combinatorial_analysis["candidate_counts"]
        _md(
            "**Combinatorial recovery candidates:**\n\n"
            f"- Considered: {counts['total']:,}\n"
            f"- Passed geometry (in detector AND tof in [0, {int(max_tof_ns)}] ns): {counts['valid']:,}\n"
            f"- Of those, also in a user peak window (informational): {counts.get('in_peak', 0):,}\n"
            f"- Emitted (after index-disjoint two-stage selection): {counts['emitted']:,}\n"
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
                    detector_limit_cm=detector_limit_cm,
                    only_in_detector=True,
                    title_prefix="Recovered Surface Concept",
                ),
                save_dir=save_dir,
                stem="surface_concept_detector_overview",
            )

            if windows:
                _md(f"## Peak-window recovery breakdown ({peak_units.upper()} windows)")
                peak_summary = summarize_surface_concept_peak_windows(
                    analysis["hit_table"],
                    analysis["recovery_diagnostics"],
                    windows,
                    signal_kind=peak_units,
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

                # Per-peak diagnostics: where are the partial hits coming from
                # (parent pulse length distribution) and where on the detector
                # do they land (2D + 1D x/y maps).
                _md(f"## Per-peak diagnostics ({peak_units.upper()} windows)")
                chunk_len_fig = plot_peak_chunk_length_distribution(
                    analysis["hit_table"],
                    windows,
                    signal_kind=peak_units,
                    only_in_detector=True,
                )
                if chunk_len_fig is not None:
                    _show_figure(
                        chunk_len_fig,
                        save_dir=save_dir,
                        stem="surface_concept_peak_chunk_lengths",
                    )
                det_fig = plot_peak_detector_diagnostics(
                    analysis["hit_table"],
                    windows,
                    signal_kind=peak_units,
                    only_in_detector=True,
                    detector_limit_cm=detector_limit_cm,
                )
                if det_fig is not None:
                    _show_figure(
                        det_fig,
                        save_dir=save_dir,
                        stem="surface_concept_peak_detector_diagnostics",
                    )
            else:
                _md(
                    "_Skipping the per-peak recovery breakdown and per-peak "
                    "detector diagnostics — no peak windows were supplied._"
                )

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

    elif detector_kind == "roentdek" and tdc_df is not None and len(tdc_df) > 0:
        from pyccapt.calibration.data_tools.raw_data_workflow import (
            analyze_roentdek_tdc_frame,
            compute_same_pulse_detector_separations,
            load_detector_constants,
            plot_detector_dead_zone_and_neighbors,
            plot_detector_overview,
            plot_roentdek_statistics,
            plot_same_pulse_detector_separations,
            plot_signal_overlay_by_dlts,
            plot_signal_window_breakdown,
            plot_tof_segment_drift,
            roentdek_processed_to_hit_table,
        )

        # Resolve RoentDek detector geometry from config (with fallback to
        # the historical 80 mm / 4 cm hardcoded defaults).
        detector_constants_ro = load_detector_constants("roentdek", getattr(variables, "conf", None))
        detector_limit_cm_ro = float(detector_constants_ro["detector_limit_cm"])

        windows = _species_to_windows(species)
        analysis = analyze_roentdek_tdc_frame(tdc_df, show_progress=True)
        roentdek_hit_table = roentdek_processed_to_hit_table(dld_df, analysis)
        # Drop hits whose pre-computed detector coordinates are outside the
        # physical detector — same gate that's enforced for SC.
        if not roentdek_hit_table.empty:
            in_det_mask = (roentdek_hit_table["x_det (cm)"].abs() <= detector_limit_cm_ro) & (
                roentdek_hit_table["y_det (cm)"].abs() <= detector_limit_cm_ro
            )
            roentdek_hit_table = roentdek_hit_table[in_det_mask].reset_index(drop=True)

        _md("## DLTS-per-pulse")
        _show_figure(
            plot_roentdek_statistics(analysis["counters"]),
            save_dir=save_dir,
            stem="roentdek_dlts_per_pulse",
        )
        _md(_roentdek_raw_summary_markdown(analysis["raw_summary"]))

        if not roentdek_hit_table.empty:
            _md("## Time-of-flight overlay by recovered DLTS class")
            _show_figure(
                plot_signal_overlay_by_dlts(
                    roentdek_hit_table,
                    signal_kind="tof",
                    max_value=1000.0,
                    bin_size=0.1,
                    only_in_detector=False,
                    title="Recovered RoentDek TOF overlay",
                ),
                save_dir=save_dir,
                stem="roentdek_tof_overlay",
            )

            _md("## Mass/charge overlay by recovered DLTS class")
            _show_figure(
                plot_signal_overlay_by_dlts(
                    roentdek_hit_table,
                    signal_kind="mc",
                    max_value=40.0,
                    bin_size=0.1,
                    only_in_detector=False,
                    title="Recovered RoentDek mass/charge overlay",
                ),
                save_dir=save_dir,
                stem="roentdek_mc_overlay",
            )

            _md("## Recovered detector maps")
            _show_figure(
                plot_detector_overview(
                    roentdek_hit_table,
                    only_in_detector=False,
                    title_prefix="Recovered RoentDek",
                ),
                save_dir=save_dir,
                stem="roentdek_detector_overview",
            )

            if windows:
                _md(f"## Peak-window recovery breakdown ({peak_units.upper()} windows)")
                _show_figure(
                    plot_signal_window_breakdown(
                        roentdek_hit_table,
                        windows,
                        signal_kind=peak_units,
                        only_in_detector=False,
                        title=f"RoentDek peak-window counts ({peak_units.upper()})",
                    ),
                    save_dir=save_dir,
                    stem="roentdek_peak_window_breakdown",
                )
            else:
                _md("_Skipping the RoentDek per-peak window breakdown — no peak windows were supplied._")

            _md("## TOF drift by segment")
            _show_figure(
                plot_tof_segment_drift(
                    roentdek_hit_table,
                    windows=None,
                    num_segments=20,
                    max_value=1000.0,
                ),
                save_dir=save_dir,
                stem="roentdek_tof_drift",
            )

            _md("## Dead-zone / nearest-neighbor diagnostics")
            _show_figure(
                plot_detector_dead_zone_and_neighbors(roentdek_hit_table),
                save_dir=save_dir,
                stem="roentdek_dead_zone_neighbors",
            )

            pair_table, pair_summary = compute_same_pulse_detector_separations(
                roentdek_hit_table,
                group_column="event_group_id",
                only_in_detector=False,
                dlts_values=[6, 4, 2],
                show_progress=True,
            )
            _md(_same_pulse_pair_summary_markdown(pair_summary, title="Same-pulse pairwise separations (RoentDek)"))
            _show_figure(
                plot_same_pulse_detector_separations(
                    pair_table,
                    bin_size=0.1,
                    title_prefix="RoentDek same-pulse separations",
                ),
                save_dir=save_dir,
                stem="roentdek_same_pulse_separations",
            )

    if detector_kind not in {"surface_concept", "roentdek"} or tdc_df is None or len(tdc_df) == 0:
        _md("## DLTS-per-pulse")
        plot_dlts_per_pulse(tdc_df, detector_kind, save_dir=save_dir, save_stem="dlts_per_pulse")

    # Full-spectrum view: clean log-y TOF + M/C histograms covering every
    # event including noise, with no peak shading. Independent of whether
    # any peak windows were supplied — this is the "normal APT mass spectrum"
    # the user wants beside the per-peak plots.
    _md("## Full spectrum (all events, including noise)")
    plot_full_spectrum(plot_df, save_dir=save_dir, save_stem="full_spectrum")

    if species:
        _md("## Time-of-flight (with peak windows)")
        plot_tof_with_peaks(
            plot_df,
            species,
            peak_units=peak_units,
            save_dir=save_dir,
            save_stem="tof_histogram",
        )

        _md("## Mass/charge (with peak windows)")
        plot_mc_with_peaks(
            plot_df,
            species,
            peak_units=peak_units,
            save_dir=save_dir,
            save_stem="mc_histogram",
        )
    else:
        _md(
            "_Skipping the per-peak TOF / mass-charge plots — no peak windows "
            "were supplied. The full spectrum above already shows every event._"
        )

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
    """Display a single-panel analysis UI driven by three dropdowns.

    Dropdowns:

    - **Peak source** — *Manual peak windows* (type up to six rows) or
      *From range file* (use ``variables.range_data``). Manual rows are
      disabled when *From range file* is selected.
    - **Peak units** — *TOF (ns)* (default) or *Mass/charge (Da)*. The
      user-typed window values are interpreted in the chosen unit; the
      per-peak ratio table is computed against that signal column. With
      *TOF (ns)* the TOF histogram overlays the species windows; with
      *Mass/charge (Da)* the mc histogram overlays them.
    - **Save plots** — *No* (default) or *Yes*. When enabled, every figure
      is also saved beside the dataset as SVG + PNG (300 dpi).

    Clicking *Run analysis* renders the full set of plots (DLTS-per-pulse,
    TOF, M/C, FDM, multi-hit) plus an inline Markdown summary beneath each
    section, with all peak-window math driven by the units dropdown.
    """
    range_df = getattr(variables, "range_data", None)
    range_species = species_from_range(range_df)
    has_range = bool(range_species)

    out = Output()
    summary = widgets.HTML()

    def _refresh_summary(*_):
        unit_label = "TOF (ns)" if peak_units.value == "tof" else "Mass/charge (Da)"
        if peak_source.value == "range":
            if has_range:
                summary.value = (
                    f"Range table loaded with <b>{len(range_species)}</b> usable rows. "
                    f"Windows are <b>mc</b> ranges from the table; the dropdown "
                    f"selection (<i>{unit_label}</i>) controls only how the "
                    f"per-peak ratio table is binned. Click <i>Run analysis</i>."
                )
            else:
                summary.value = (
                    "<span style='color:#b91c1c;'>No range table is loaded. "
                    "Switch to <b>Manual peak windows</b> or load a range table first.</span>"
                )
        else:
            summary.value = f"<i>Type peak windows below in <b>{unit_label}</b>. Rows left at 0/0 are skipped.</i>"

    peak_source = widgets.Dropdown(
        options=[("Manual peak windows", "manual"), ("From range file", "range")],
        value="range" if has_range else "manual",
        description="Peak source:",
        layout=widgets.Layout(width="320px"),
    )
    peak_units = widgets.Dropdown(
        options=[("TOF (ns)", "tof"), ("Mass/charge (Da)", "mc")],
        value="tof",
        description="Peak units:",
        layout=widgets.Layout(width="320px"),
    )
    save_plots = widgets.Dropdown(
        options=[("No", False), ("Yes", True)],
        value=False,
        description="Save plots:",
        layout=widgets.Layout(width="320px"),
    )
    recovery_mode = widgets.Dropdown(
        options=[
            ("Exhaustive (combinatorial, slow — default)", "exhaustive"),
            ("Greedy (combinatorial, fast)", "greedy"),
        ],
        value="exhaustive",
        description="Recovery:",
        layout=widgets.Layout(width="380px"),
    )
    pair_sum_tol = widgets.IntText(
        value=200,
        description="Pair-sum tol (bins):",
        layout=widgets.Layout(width="380px"),
        style={"description_width": "140px"},
    )

    manual_rows = _build_manual_rows()
    manual_grid = widgets.VBox([widgets.HBox([label, low, high]) for label, low, high in manual_rows])

    run_button = widgets.Button(description="Run analysis", button_style="primary")

    def _set_panel_busy(busy: bool) -> None:
        """Disable every interactive control while the analysis is running so
        the user can't change inputs / re-trigger the run mid-flight, and
        flips the run button label to give a visible busy indication."""
        for control in (
            peak_source,
            peak_units,
            save_plots,
            recovery_mode,
            pair_sum_tol,
        ):
            control.disabled = busy
        # Manual rows: while busy, force-disabled. Otherwise, restore the
        # source-driven disabled state (rows are disabled in "From range" mode).
        rows_disabled = busy or peak_source.value == "range"
        _set_rows_disabled(manual_rows, rows_disabled)
        run_button.disabled = busy
        run_button.description = "Processing…" if busy else "Run analysis"

    def _on_source_change(_change):
        _set_rows_disabled(manual_rows, peak_source.value == "range")
        _refresh_summary()

    def _on_units_change(_change):
        _refresh_summary()

    def _on_run(_):
        out.clear_output()
        _set_panel_busy(True)
        try:
            with out:
                if peak_source.value == "range":
                    if not range_species:
                        _md(
                            "**Range table is empty.** Running global analyses "
                            "(DLTS-per-pulse, full TOF/MC spectra, FDM, multi-hit) "
                            "without per-peak yields. Switch to *Manual peak windows* "
                            "to type peak ranges and unlock the per-peak sections."
                        )
                        species = []
                    else:
                        species = range_species
                else:
                    try:
                        species = species_from_manual(manual_rows)
                    except ValueError as exc:
                        _md(f"**Input error:** {exc}")
                        return
                    if not species:
                        _md(
                            "_No peak windows supplied._ Running global analyses "
                            "(DLTS-per-pulse, full TOF/MC spectra, FDM, multi-hit) "
                            "without per-peak yields."
                        )
                run_analysis(
                    variables,
                    species,
                    save_plots=bool(save_plots.value),
                    peak_units=str(peak_units.value),
                    recovery_mode=str(recovery_mode.value),
                    pair_sum_tolerance_bins=float(pair_sum_tol.value),
                )
        finally:
            _set_panel_busy(False)

    peak_source.observe(_on_source_change, names="value")
    peak_units.observe(_on_units_change, names="value")
    run_button.on_click(_on_run)

    # Initialize disabled state to match the dropdown's starting value.
    _set_rows_disabled(manual_rows, peak_source.value == "range")
    _refresh_summary()

    panel = widgets.VBox(
        [
            peak_source,
            peak_units,
            save_plots,
            recovery_mode,
            pair_sum_tol,
            summary,
            manual_grid,
            run_button,
            out,
        ]
    )
    display(panel)

def call_signal_preview(variables) -> None:
    """Render the preview panel — a processing-workflow-style histogram
    explorer for the raw-data analysis notebook.

    Mirrors the *Tab 1* layout of :func:`helper_visualization.call_visualization`:
    a row of label/widget pairs feeding :func:`pyccapt.calibration.core.mc_plot.hist_plot`.
    The user picks a target column (``tof``, ``tof_c``, ``mc``, ``mc_uc``),
    bin size, axis limit (``Max TOF`` / ``Max m/c`` depending on the target),
    peak-find toggle and prominence / distance, log scale, save-fig flag,
    figname, and figure size — then clicks *Plot* to render.

    Targets that are absent or all-zero in ``variables.data`` are dropped from
    the *Target* dropdown automatically, so e.g. a pure-raw acquisition file
    won't show ``mc`` (calibrated) or ``tof_c`` if those columns are zero.
    """
    from pyccapt.calibration.core import mc_plot

    dld_df = getattr(variables, "data", None)
    if dld_df is None or len(dld_df) == 0:
        _md("_No dld data is loaded — nothing to preview._")
        return

    label_layout = widgets.Layout(width="160px")
    field_layout = widgets.Layout(width="220px")

    target_columns = {
        "tof": "t (ns)",
        "tof_c": "t_c (ns)",
        "mc": "mc (Da)",
        "mc_uc": "mc_uc (Da)",
    }

    def _has_signal(col_name: str) -> bool:
        if col_name not in dld_df.columns:
            return False
        arr = dld_df[col_name].to_numpy()
        return arr.size > 0 and bool((arr != 0).any())

    target_options: list[tuple[str, str]] = []
    for tgt, col in target_columns.items():
        if _has_signal(col):
            target_options.append((f"{tgt}  ({col})", tgt))
    if not target_options:
        _md("_None of `t (ns)`, `t_c (ns)`, `mc (Da)`, `mc_uc (Da)` carry usable values — cannot render the preview._")
        return

    initial_target = target_options[0][1]
    is_tof_target = initial_target in {"tof", "tof_c"}

    target_mode = widgets.Dropdown(
        options=target_options,
        value=initial_target,
        layout=field_layout,
    )
    bin_size_widget = widgets.FloatText(value=0.1, layout=field_layout)
    lim_widget = widgets.IntText(value=100 if not is_tof_target else 5000, layout=field_layout)
    log_widget = widgets.Dropdown(
        options=[("True", True), ("False", False)],
        value=True,
        layout=field_layout,
    )
    peaks_find = widgets.Dropdown(
        options=[("False", False), ("True", True)],
        value=False,
        layout=field_layout,
    )
    prominence = widgets.IntText(value=50, layout=field_layout)
    distance = widgets.IntText(value=50, layout=field_layout)
    figname_widget = widgets.Text(value=f"preview_{initial_target}", layout=field_layout)
    save_widget = widgets.Dropdown(
        options=[("False", False), ("True", True)],
        value=False,
        layout=field_layout,
    )
    fig_size_x = widgets.FloatText(value=9.0, layout=widgets.Layout(width="105px"))
    fig_size_y = widgets.FloatText(value=5.0, layout=widgets.Layout(width="105px"))

    plot_button = widgets.Button(description="Plot", button_style="primary")
    clear_button = widgets.Button(description="Clear")

    out = Output()

    def _on_target_change(change):
        if change.get("name") != "value":
            return
        new_target = change.get("new")
        new_is_tof = new_target in {"tof", "tof_c"}
        bin_size_widget.value = 0.1
        lim_widget.value = 5000 if new_is_tof else 100
        figname_widget.value = f"preview_{new_target}"

    preview_controls = (
        target_mode,
        bin_size_widget,
        lim_widget,
        log_widget,
        peaks_find,
        prominence,
        distance,
        figname_widget,
        save_widget,
        fig_size_x,
        fig_size_y,
        clear_button,
    )

    def _set_preview_busy(busy: bool) -> None:
        """Disable every interactive control while the preview is rendering
        so the user can't change parameters mid-flight, and flips the Plot
        button label to give a visible busy indication."""
        for control in preview_controls:
            control.disabled = busy
        plot_button.disabled = busy
        plot_button.description = "Plotting…" if busy else "Plot"

    def _on_plot(_):
        out.clear_output()
        _set_preview_busy(True)
        try:
            with out:
                try:
                    # When the user enables "Peak find", we automatically tie
                    # ``peaks_find_plot`` and ``print_info`` to True. That way
                    # one switch produces:
                    #   - peaks marked on the histogram (overlay), and
                    #   - peak locations + left/right window edges + MRP
                    #     printed beneath the figure (same text format as
                    #     helper_visualization.call_visualization).
                    find_peaks = bool(peaks_find.value)
                    mc_plot.hist_plot(
                        variables,
                        bin_size_widget.value,
                        log=log_widget.value,
                        target=target_mode.value,
                        normalize=False,
                        prominence=prominence.value,
                        distance=distance.value,
                        percent=50,
                        selector="rect",
                        figname=figname_widget.value,
                        lim=lim_widget.value,
                        peaks_find=find_peaks,
                        peaks_find_plot=find_peaks,
                        plot_ranged_peak=False,
                        plot_ranged_colors=False,
                        mrp_all=False,
                        background=None,
                        grid=False,
                        save_fig=save_widget.value,
                        print_info=find_peaks,
                        figure_size=(fig_size_x.value, fig_size_y.value),
                    )
                except Exception as exc:  # pragma: no cover - widget runtime path
                    _md(f"**Plot failed:** `{type(exc).__name__}: {exc}`")
        finally:
            _set_preview_busy(False)

    def _on_clear(_):
        out.clear_output()

    target_mode.observe(_on_target_change, names="value")
    plot_button.on_click(_on_plot)
    clear_button.on_click(_on_clear)

    def _row(label_text: str, w) -> widgets.HBox:
        return widgets.HBox([widgets.Label(value=label_text, layout=label_layout), w])

    panel = widgets.VBox(
        [
            _row("Target:", target_mode),
            _row("Bin size:", bin_size_widget),
            _row("Max TOF / m/c:", lim_widget),
            _row("Log:", log_widget),
            _row("Peak find:", peaks_find),
            _row("Peak prominence:", prominence),
            _row("Peak distance:", distance),
            _row("Fig name:", figname_widget),
            _row("Save fig:", save_widget),
            widgets.HBox(
                [
                    widgets.Label(value="Fig size:", layout=label_layout),
                    widgets.HBox([fig_size_x, fig_size_y]),
                ]
            ),
            widgets.HBox([plot_button, clear_button]),
            out,
        ]
    )
    display(panel)
