"""Peak-window analysis and persistence conversion for Surface Concept data."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from pyccapt.calibration.data_tools._raw_workflow_common import (
    DLTS_COLORS, _calculate_delta_p_and_multi, _normalize_signal_kind,
    normalize_signal_windows, summarize_signal_windows,
)
from pyccapt.calibration.data_tools._raw_workflow_surface_concept import analyze_surface_concept_dataset

def summarize_surface_concept_peak_windows(
    hit_table: pd.DataFrame,
    recovery_diagnostics: pd.DataFrame,
    windows: Sequence[dict] | Sequence[Sequence] | None,
    *,
    signal_kind: str = 'mc',
    only_in_detector: bool = True,
) -> dict[str, pd.DataFrame | int]:
    """Summarize user-defined peak windows across 2-DLTS and 4-DLTS recovery classes."""
    normalized_windows = normalize_signal_windows(windows)
    if not normalized_windows:
        empty = pd.DataFrame(columns=['label', 'two_dlts_count', 'four_dlts_count'])
        empty_ratios = pd.DataFrame(
            columns=['Peak', 'Two DLTS count', 'Four DLTS count', 'Two DLTS %', 'Four DLTS %', 'Two/Four DLTS']
        )
        empty_bars = pd.DataFrame(columns=['label', 'count', 'color'])
        return {
            'counts': empty,
            'ratios': empty_ratios,
            'bars': empty_bars,
            'outside_detector_count': 0,
            'unrecoverable_count': 0,
            'total_in_detector': 0,
        }

    signal_value = _normalize_signal_kind(signal_kind)
    summary = summarize_signal_windows(
        hit_table,
        normalized_windows,
        signal_kind=signal_value,
        only_in_detector=only_in_detector,
    )

    labels = [str(window['label']) for window in normalized_windows]
    noise_label = 'Noise'
    ordered_labels = [*labels, noise_label]
    if summary.empty:
        summary = pd.DataFrame(
            {'label': ordered_labels, 'dlts': np.zeros(len(ordered_labels)), 'count': np.zeros(len(ordered_labels))}
        )

    def _count_for(label: str, dlts: int) -> int:
        matches = summary[(summary['label'] == label) & (summary['dlts'] == dlts)]
        if matches.empty:
            return 0
        return int(matches['count'].sum())

    total_in_detector = int(
        np.count_nonzero(
            (hit_table['dlts'].isin([2, 4]).to_numpy())
            & (
                hit_table['in_detector'].to_numpy()
                if 'in_detector' in hit_table.columns
                else np.ones(len(hit_table), dtype=bool)
            )
        )
    )
    outside_detector_count = int(
        np.count_nonzero(recovery_diagnostics['status'].isin(['2 DLTS outside detector', '4 DLTS outside detector']))
    )
    unrecoverable_count = int(np.count_nonzero(recovery_diagnostics['status'] == 'unrecoverable'))

    count_rows = []
    ratio_rows = []
    bar_rows = []
    for label in labels:
        two_count = _count_for(label, 2)
        four_count = _count_for(label, 4)
        count_rows.append(
            {
                'label': label,
                'two_dlts_count': two_count,
                'four_dlts_count': four_count,
            }
        )
        ratio_rows.append(
            {
                'Peak': label,
                'Two DLTS count': two_count,
                'Four DLTS count': four_count,
                'Two DLTS %': (100.0 * two_count / total_in_detector) if total_in_detector else 0.0,
                'Four DLTS %': (100.0 * four_count / total_in_detector) if total_in_detector else 0.0,
                'Two/Four DLTS': (two_count / four_count) if four_count else np.nan,
            }
        )
        bar_rows.append({'label': f'{label} 4 DLTS', 'count': four_count, 'color': '#f59e0b'})

    noise_four = _count_for(noise_label, 4)
    bar_rows.append({'label': f'{noise_label} 4 DLTS', 'count': noise_four, 'color': '#f59e0b'})

    for label in labels:
        two_count = _count_for(label, 2)
        bar_rows.append({'label': f'{label} 2 DLTS', 'count': two_count, 'color': '#10b981'})

    noise_two = _count_for(noise_label, 2)
    bar_rows.append({'label': f'{noise_label} 2 DLTS', 'count': noise_two, 'color': '#10b981'})
    bar_rows.append({'label': 'Outside detector', 'count': outside_detector_count, 'color': '#ef4444'})
    bar_rows.append({'label': 'Unrecoverable', 'count': unrecoverable_count, 'color': '#ef4444'})

    return {
        'counts': pd.DataFrame(count_rows),
        'ratios': pd.DataFrame(ratio_rows),
        'bars': pd.DataFrame(bar_rows),
        'outside_detector_count': outside_detector_count,
        'unrecoverable_count': unrecoverable_count,
        'total_in_detector': total_in_detector,
    }


def plot_surface_concept_peak_breakdown(
    peak_summary: dict[str, pd.DataFrame | int],
    *,
    title: str = 'Surface Concept peak-window breakdown',
) -> plt.Figure | None:
    """Plot a peak-by-peak 2-DLTS / 4-DLTS bar chart plus rejected-event counts."""
    bars = peak_summary.get('bars')
    if not isinstance(bars, pd.DataFrame) or bars.empty:
        return None

    fig, ax = plt.subplots(figsize=(max(7.5, len(bars) * 0.7), 4.0))
    ax.bar(bars['label'], bars['count'], color=bars['color'], edgecolor='#4b5563', linewidth=0.6)
    ax.set_ylabel('Counts')
    ax.set_title(title, fontsize=11, fontweight='semibold')
    if np.nanmax(bars['count'].to_numpy(dtype=float)) > 20:
        ax.set_yscale('log')
    ax.tick_params(axis='x', rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#e5e7eb', linewidth=0.8, alpha=0.7)
    fig.tight_layout()
    return fig


def plot_surface_concept_peak_ratio_table(
    ratio_table: pd.DataFrame,
    *,
    title: str = 'Two delay line (Surface Concept)',
) -> plt.Figure | None:
    """Render a compact percentage/ratio table for the user-defined peak windows."""
    if ratio_table.empty:
        return None

    formatted = ratio_table.copy()
    formatted['Two DLTS %'] = formatted['Two DLTS %'].map(lambda value: f'{value:.2f}%')
    formatted['Four DLTS %'] = formatted['Four DLTS %'].map(lambda value: f'{value:.2f}%')
    formatted['Two/Four DLTS'] = formatted['Two/Four DLTS'].map(
        lambda value: 'n/a' if not np.isfinite(value) else f'{value:.3f}'
    )

    figure_width = max(6.8, 0.9 + 1.25 * len(formatted.columns))
    figure_height = max(2.6, 1.55 + 0.62 * len(formatted))
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    ax.axis('off')
    title_band_height = 0.17
    ax.add_patch(
        Rectangle(
            (0.0, 1.0 - title_band_height),
            1.0,
            title_band_height,
            transform=ax.transAxes,
            facecolor='#f3f4f6',
            edgecolor='#b6b8bb',
            linewidth=0.8,
        )
    )
    ax.text(
        0.03,
        1.0 - title_band_height / 2.0,
        title,
        transform=ax.transAxes,
        ha='left',
        va='center',
        fontsize=11,
        fontweight='semibold',
        color='black',
    )
    table = ax.table(
        cellText=formatted[['Peak', 'Two DLTS %', 'Four DLTS %', 'Two/Four DLTS']].to_numpy(),
        colLabels=['Ion', 'Two DLTS', 'Four DLTS', 'Two/four DLTS'],
        cellLoc='center',
        loc='center',
        colWidths=[0.17, 0.23, 0.23, 0.27],
        bbox=[0.0, 0.0, 1.0, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.35)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#b6b8bb')
        cell.set_linewidth(0.75)
        if row == 0:
            cell.set_facecolor('#f7f7f7')
            cell.set_text_props(weight='semibold', color='black')
            cell.set_height(cell.get_height() * 1.08)
        else:
            cell.set_facecolor('white')
            if col == 0:
                cell.set_text_props(ha='left')

    fig.tight_layout()
    return fig


def surface_concept_hits_to_processed_dataframe(
    hit_table: pd.DataFrame,
    *,
    pulse_mode: str = 'voltage',
    max_start_counter: int = 20000,
) -> pd.DataFrame:
    """Convert recovered Surface Concept hits into a PyCCAPT-style processed dataframe."""
    frame = hit_table.copy()
    if 'in_detector' in frame.columns:
        frame = frame[frame['in_detector']]
    if frame.empty:
        raise ValueError('No in-detector Surface Concept hits are available to build a processed dataframe')
    if 'mc (Da)' not in frame.columns:
        raise ValueError("The hit table must contain 'mc (Da)' before converting to a processed dataframe")

    pulse_v = frame['pulse'].to_numpy() if pulse_mode == 'voltage' else np.zeros(len(frame))
    pulse_l = frame['pulse'].to_numpy() if pulse_mode == 'laser' else np.zeros(len(frame))
    delta_p, multi = _calculate_delta_p_and_multi(frame['start_counter'].to_numpy(), max_start_counter=max_start_counter)

    processed = pd.DataFrame(
        {
            'x (nm)': np.zeros(len(frame)),
            'y (nm)': np.zeros(len(frame)),
            'z (nm)': np.zeros(len(frame)),
            'mc (Da)': frame['mc (Da)'].to_numpy(),
            'mc_uc (Da)': frame['mc (Da)'].to_numpy().copy(),
            'high_voltage (V)': frame['high_voltage (V)'].to_numpy(),
            'pulse_v (V)': pulse_v,
            'pulse_l (pJ)': pulse_l,
            't (ns)': frame['tof (ns)'].to_numpy(),
            't_c (ns)': frame['tof (ns)'].to_numpy().copy(),
            'x_det (cm)': frame['x_det (cm)'].to_numpy(),
            'y_det (cm)': frame['y_det (cm)'].to_numpy(),
            'delta_p': delta_p,
            'multi': multi,
            'start_counter': frame['start_counter'].to_numpy(dtype=np.uint32),
        }
    )
    return processed


def reconstruct_surface_concept_dataset(
    hdf5_path: str,
    *,
    flight_path_length_mm: float = 110.0,
    pulse_mode: str = 'voltage',
    t0: float = 0.0,
    detector_limit_cm: float = 4.0,
) -> pd.DataFrame:
    """Reconstruct a processed PyCCAPT-style dataset from a Surface Concept raw HDF5 file."""
    analysis = analyze_surface_concept_dataset(
        hdf5_path,
        detector_limit_cm=detector_limit_cm,
        t0=t0,
        flight_path_length_mm=flight_path_length_mm,
        pulse_mode=pulse_mode,
    )
    return surface_concept_hits_to_processed_dataframe(analysis['hit_table'], pulse_mode=pulse_mode)


# ---------------------------------------------------------------------------
# Per-peak diagnostics: chunk-length distribution + detector-position plots
# ---------------------------------------------------------------------------


def _resolve_peak_signal_column(hit_table: pd.DataFrame, signal_kind: str) -> str:
    signal_kind = _normalize_signal_kind(signal_kind)
    if signal_kind == 'mc':
        return 'mc (Da)'
    return 'tof (ns)'


def filter_peak_hits(
    hit_table: pd.DataFrame,
    window: dict,
    *,
    signal_kind: str = 'tof',
    only_in_detector: bool = True,
) -> pd.DataFrame:
    """Return the rows of ``hit_table`` whose chosen signal lies in ``window``.

    ``window`` is a dict with ``min``/``max`` keys (the format produced by
    :func:`pyccapt.calibration.data_tools._raw_workflow_common.normalize_signal_windows`).
    Setting ``only_in_detector=True`` (default) drops hits that failed the
    per-axis detector-area check from
    :func:`build_surface_concept_recovery_diagnostics`.
    """
    if hit_table is None or hit_table.empty:
        return hit_table.iloc[0:0] if hit_table is not None else pd.DataFrame()
    column = _resolve_peak_signal_column(hit_table, signal_kind)
    if column not in hit_table.columns:
        return hit_table.iloc[0:0]

    frame = hit_table
    if only_in_detector and 'in_detector' in frame.columns:
        frame = frame[frame['in_detector']]
    values = pd.to_numeric(frame[column], errors='coerce').to_numpy()
    lower = float(window['min'])
    upper = float(window['max'])
    mask = (values >= lower) & (values <= upper)
    return frame.loc[mask].copy()


def plot_peak_chunk_length_distribution(
    hit_table: pd.DataFrame,
    windows: Sequence[dict],
    *,
    signal_kind: str = 'tof',
    only_in_detector: bool = True,
    max_length: int = 20,
) -> plt.Figure | None:
    """For every user peak window, render two side-by-side bars per parent
    chunk length: how many *2-DLTS partial* hits and *4-DLTS full* hits in
    that peak came from a parent pulse of that length.

    ``parent_pulse_length`` must be present on ``hit_table`` (it is added by
    :func:`build_surface_concept_recovery_diagnostics`).  Returns ``None`` if
    the column is missing or no peak windows are supplied.
    """
    if hit_table is None or hit_table.empty or 'parent_pulse_length' not in hit_table.columns:
        return None
    if not windows:
        return None

    n_peaks = len(windows)
    # Cap the figure height well below matplotlib's 2^16-pixel hard limit so
    # IPython's ``bbox_inches='tight'`` PNG render can't blow the dimensions
    # into the multi-million-pixel range that triggered "Image size too large"
    # on user input.  20 in × 300 dpi = 6 000 px — comfortably below 65 535.
    height_inches = min(max(2.4 * n_peaks, 3.0), 20.0)
    fig, axes = plt.subplots(
        n_peaks,
        1,
        figsize=(8.0, height_inches),
        squeeze=False,
    )
    bins = np.arange(0.5, max_length + 1.5)
    centers = np.arange(1, max_length + 1)

    for index, window in enumerate(windows):
        ax = axes[index][0]
        peak = filter_peak_hits(hit_table, window, signal_kind=signal_kind, only_in_detector=only_in_detector)
        if peak.empty:
            ax.text(0.5, 0.5, "no hits in this peak window", ha='center', va='center', transform=ax.transAxes, color='gray')
            ax.set_xlim(0.5, max_length + 0.5)
            ax.set_xticks(centers)
            ax.set_title(f"{window.get('label', f'Peak {index + 1}')}")
            ax.set_xlabel("Parent pulse length (DLTS per pulse)")
            ax.set_ylabel("Hit count")
            continue

        partial = peak[peak['dlts'] == 2]['parent_pulse_length'].to_numpy()
        full = peak[peak['dlts'] == 4]['parent_pulse_length'].to_numpy()
        partial_hist = np.histogram(partial, bins=bins)[0]
        full_hist = np.histogram(full, bins=bins)[0]

        w = 0.4
        ax.bar(centers - 0.5 * w, partial_hist, width=w, color=DLTS_COLORS.get(2, '#f59e0b'), label='2 DLTS')
        ax.bar(centers + 0.5 * w, full_hist, width=w, color=DLTS_COLORS.get(4, '#1f77b4'), label='4 DLTS')
        ax.set_yscale('log')
        ax.set_xlim(0.5, max_length + 0.5)
        ax.set_xticks(centers)
        ax.set_xlabel("Parent pulse length (DLTS per pulse)")
        ax.set_ylabel("Hit count (log)")
        ax.set_title(
            f"{window.get('label', f'Peak {index + 1}')} — partial = {int(partial_hist.sum()):,}; "
            f"full = {int(full_hist.sum()):,}"
        )
        ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    return fig
