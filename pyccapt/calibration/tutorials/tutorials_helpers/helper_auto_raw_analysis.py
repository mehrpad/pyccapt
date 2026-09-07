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
from pyccapt.calibration.tutorials.tutorials_helpers import auto_raw_analysis_domain as _domain

# ---------------------------------------------------------------------------
# Detector kind detection
# ---------------------------------------------------------------------------


# Domain and plotting helpers are isolated from the notebook controller.
from pyccapt.calibration.tutorials.tutorials_helpers.auto_raw_analysis_domain import (
    detect_detector_kind,
    _delay_line_pairs,
    expected_dlts_full,
    species_from_range,
    species_from_manual,
    _close_after,
    _resolve_dataset_path,
    _analysis_save_directory,
    _format_pct,
    _classify_pulse_chunks,
    plot_dlts_per_pulse,
    _pick_tof_col,
    plot_tof_with_peaks,
    _pick_mc_col,
    plot_signal_overview,
    plot_mc_with_peaks,
    compute_mrp_half,
    _surface_concept_length_breakdown_markdown,
    _surface_concept_raw_summary_markdown,
    _species_to_windows,
    _surface_concept_peak_ratio_markdown,
    _roentdek_raw_summary_markdown,
    _same_pulse_pair_summary_markdown,
    plot_fdm,
    plot_multihit_and_deadzone,
)


def _show_figure(*args, **kwargs):
    """UI adapter that preserves notebook display monkeypatching/customization."""
    _domain.display = display
    return _domain._show_figure(*args, **kwargs)


def _md(*args, **kwargs):
    _domain.display = display
    return _domain._md(*args, **kwargs)


def plot_full_spectrum(*args, **kwargs):
    _domain.display = display
    return _domain.plot_full_spectrum(*args, **kwargs)

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
        # All stats (sequence_stats, raw_summary, recovery_diagnostics,
        # recovery_stats) were computed inside the combinatorial function
        # while sequence_records was still alive, and sequence_records was
        # freed immediately after to avoid keeping ~20 GB of Python objects
        # in memory alongside the downstream DataFrames.
        analysis = {
            'sequence_stats': combinatorial_analysis['sequence_stats'],
            'raw_summary': combinatorial_analysis['raw_summary'],
            'recovery_diagnostics': combinatorial_analysis['recovery_diagnostics'],
            'recovery_stats': combinatorial_analysis['recovery_stats'],
            'hit_table': combinatorial_analysis['hit_table'],
        }
        counts = combinatorial_analysis["candidate_counts"]
        _md(
            "**Combinatorial recovery candidates:**\n\n"
            f"- Considered: {counts['total']:,}\n"
            f"- Passed geometry (in detector AND tof in [0, {int(max_tof_ns)}] ns): {counts['valid']:,}\n"
            f"- Of those, also in a user peak window (informational): {counts.get('in_peak', 0):,}\n"
            f"- Emitted (after index-disjoint two-stage selection): {counts['emitted']:,}\n"
        )

        if not analysis["hit_table"].empty:
            # Two modes:
            # * raw load (no calibrated ``mc (Da)`` in ``dld_df``) — replace
            #   plot_df with the hit-table-derived processed frame so the
            #   downstream FDM / TOF / MC histograms have populated mc
            #   columns to draw from.
            # * calibrated-bundle load (``mc (Da)`` already populated by a
            #   previous calibration run) — KEEP the calibrated frame as
            #   plot_df. Throwing it away here would silently substitute
            #   the centred-axis raw mc for the actual calibration result
            #   in every downstream plot (full-spectrum, FDM, multi-hit).
            #   The SC-specific recovery overlays below still use
            #   ``analysis['hit_table']`` directly, so they're unaffected.
            # ``_pick_mc_col`` returns the column with non-zero values,
            # preferring ``mc (Da)``. A return of ``'mc (Da)'`` therefore
            # means the dataframe was calibrated (column is non-zero);
            # ``'mc_uc (Da)'`` or ``None`` means raw-only.
            has_calibrated_mc = _pick_mc_col(dld_df) == "mc (Da)"
            if not has_calibrated_mc:
                plot_df = surface_concept_hits_to_processed_dataframe(
                    analysis["hit_table"],
                    pulse_mode=pulse_mode,
                )
            else:
                _md(
                    "_Keeping the calibrated `/df` frame as the plot input — "
                    "the SC hit-table is shown via the dedicated recovery "
                    "overlays only._"
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
        _rd = analysis.get("recovery_diagnostics")
        if _rd is not None and not (hasattr(_rd, "empty") and _rd.empty):
            _show_figure(
                plot_surface_concept_recovery_yield(_rd),
                save_dir=save_dir,
                stem="surface_concept_recovery_yield",
            )
            _show_figure(
                plot_partial_hit_efficiency_maps(_rd),
                save_dir=save_dir,
                stem="surface_concept_partial_hit_efficiency",
            )
        else:
            _md("_Recovery yield and partial-hit efficiency maps skipped (not computed in memory-safe mode)._")

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
                _rd2 = analysis.get("recovery_diagnostics")
                _has_rd = _rd2 is not None and not (hasattr(_rd2, "empty") and _rd2.empty)
                peak_summary = summarize_surface_concept_peak_windows(
                    analysis["hit_table"],
                    _rd2 if _has_rd else pd.DataFrame(),
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

    # ----------------------------------------------------------------------
    # Partial-hit diagnostics: only meaningful when the merged DLD frame
    # carries the ``dlts_quality`` annotation. The data-processing workflow
    # writes that column when ``merge_partial_tdc=True`` was used (or any
    # downstream caller of ``partial_recovery.merge_partial_tdc_into_dld``);
    # pure-raw files without the merge will lack it and we skip silently.
    # The same set of figures is produced regardless of detector kind so
    # SC and RoentDek users see the same view.
    # ----------------------------------------------------------------------
    if "dlts_quality" in dld_df.columns:
        from pyccapt.calibration.data_tools.partial_hit_diagnostics import (
            partial_hit_counts,
            plot_partial_hit_axis_bias,
            plot_partial_hit_breakdown,
            plot_partial_hit_detector_distribution,
            plot_partial_hit_multihit_correlation,
            plot_partial_hit_signal_overlay,
            plot_partial_hit_voltage_correlation,
        )

        counts = partial_hit_counts(dld_df)
        if counts["partial"] > 0 or counts["recovered_xy"] > 0:
            _md("## Partial-hit diagnostics")
            _md(
                f"_Recovered partial-hit summary:_ **{counts['partial']:,} partial** "
                f"({counts['recovered_x']:,} x-only, {counts['recovered_y']:,} y-only) "
                f"+ **{counts['recovered_xy']:,} recovered_xy** out of "
                f"**{counts['total']:,}** total — "
                f"partial fraction `{counts['partial_fraction']:.2%}`, "
                f"recovery share `{counts['recovered_fraction']:.2%}`."
            )

            # Resolve the detector radius for the spatial-distribution panel
            # so RoentDek and Surface Concept users both get correctly-sized
            # 2-D histograms. Default to 4 cm if no detector kind is known.
            try:
                from pyccapt.calibration.data_tools._raw_workflow_common import (
                    load_detector_constants,
                )

                _kind = detector_kind if detector_kind in {"surface_concept", "roentdek"} else "surface_concept"
                _dconst = load_detector_constants(_kind, getattr(variables, "conf", None))
                _detector_radius_cm = float(_dconst["detector_limit_cm"])
            except Exception:
                _detector_radius_cm = 4.0

            _md("### Category breakdown")
            _show_figure(
                plot_partial_hit_breakdown(dld_df),
                save_dir=save_dir,
                stem="partial_hit_breakdown",
            )
            _md("### Detector-space distribution (full vs partial)")
            _show_figure(
                plot_partial_hit_detector_distribution(
                    dld_df,
                    detector_radius_cm=_detector_radius_cm,
                ),
                save_dir=save_dir,
                stem="partial_hit_detector_distribution",
            )
            _md("### Time-domain / HV correlation")
            _show_figure(
                plot_partial_hit_voltage_correlation(dld_df),
                save_dir=save_dir,
                stem="partial_hit_voltage_correlation",
            )
            _md("### Mass-spectrum split by hit category")
            _show_figure(
                plot_partial_hit_signal_overlay(dld_df, signal_kind="mc"),
                save_dir=save_dir,
                stem="partial_hit_mc_overlay",
            )
            _md("### Time-of-flight split by hit category")
            _show_figure(
                plot_partial_hit_signal_overlay(dld_df, signal_kind="tof"),
                save_dir=save_dir,
                stem="partial_hit_tof_overlay",
            )
            _md("### Multi-hit correlation (do partials concentrate in busy pulses?)")
            _show_figure(
                plot_partial_hit_multihit_correlation(dld_df, tdc_df),
                save_dir=save_dir,
                stem="partial_hit_multihit_correlation",
            )
            _md("### Per-axis bias (ideal ≈ 50 % each)")
            _show_figure(
                plot_partial_hit_axis_bias(dld_df),
                save_dir=save_dir,
                stem="partial_hit_axis_bias",
            )
        else:
            _md(
                "_Partial-hit diagnostics: the ``dlts_quality`` column is "
                "present but no recovered partials were found in this "
                "dataset — every row is a native 4-DLTS hit._"
            )
    else:
        _md(
            "_Partial-hit diagnostics skipped — the dataframe lacks the "
            "``dlts_quality`` column (run ``load_data(..., load_tdc_raw=True, "
            "merge_partial_tdc=True)`` during data-processing to enable)._"
        )

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
