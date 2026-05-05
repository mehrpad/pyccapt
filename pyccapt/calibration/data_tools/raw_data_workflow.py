"""Reusable raw-data analysis helpers extracted from tutorial notebooks.

This module is a thin facade that re-exports the public surface from three
internal sibling submodules so existing imports keep working unchanged:

- :mod:`._raw_workflow_common` — constants, shared validators, generic plots,
  drift / dead-zone diagnostics, and processed-dataset I/O.
- :mod:`._raw_workflow_roentdek` — RoentDek (3-delay-line / hexanode) pipeline.
- :mod:`._raw_workflow_surface_concept` — Surface Concept (2-delay-line) pipeline.

The split was performed to keep each file under the calibration module-length
policy enforced by ``tests/calibration/test_calibration_module_lengths.py``.
"""

from __future__ import annotations

from pyccapt.calibration.data_tools._raw_workflow_common import (
    BINNING_FACTOR,
    DEFAULT_ROENTDEK_DETX_COLUMNS,
    DEFAULT_ROENTDEK_DETY_COLUMNS,
    DEFAULT_ROENTDEK_SIGNAL_COLUMNS,
    DETBINS,
    DLTS_COLORS,
    TOF_FACTOR_NS,
    TOF_FACTOR_NS_1D,
    XY_BIN_SHIFT,
    XY_FACTOR,
    _auto_peak_windows,
    _binned_status_fraction,
    _calculate_delta_p_and_multi,
    _compute_histogram_bins,
    _normalize_index_positions,
    _normalize_signal_kind,
    _signal_column_and_label,
    _validate_numeric_table_columns,
    compute_tof_segment_drift,
    compute_same_pulse_detector_separations,
    load_numeric_text_table,
    normalize_signal_windows,
    plot_detector_dead_zone_and_neighbors,
    plot_detector_overview,
    plot_processed_dataset_overview,
    plot_same_pulse_detector_separations,
    plot_signal_overlay_by_dlts,
    plot_signal_window_breakdown,
    plot_tof_segment_drift,
    save_processed_raw_dataset,
    summarize_processed_dataset,
    summarize_signal_windows,
)
from pyccapt.calibration.data_tools._raw_workflow_roentdek import (
    _extract_max_dld_patterns,
    _extract_roentdek_pattern_details,
    _initialize_roentdek_counters,
    analyze_roentdek_dataset,
    attach_roentdek_measurements,
    classify_roentdek_events,
    parse_roentdek_events,
    plot_roentdek_statistics,
    roentdek_hits_to_dataframe,
    summarize_roentdek_raw_events,
)
from pyccapt.calibration.data_tools._raw_workflow_surface_concept import (
    _recover_surface_concept_partial_hits,
    _surface_concept_hit_from_time_data,
    _surface_concept_position_from_pair,
    analyze_surface_concept_dataset,
    analyze_surface_concept_tdc_frame,
    build_surface_concept_recovery_diagnostics,
    extract_surface_concept_hits,
    plot_partial_hit_efficiency_maps,
    plot_surface_concept_peak_breakdown,
    plot_surface_concept_peak_ratio_table,
    plot_surface_concept_recovery_summary,
    plot_surface_concept_recovery_yield,
    plot_surface_concept_sequence_statistics,
    reconstruct_surface_concept_dataset,
    summarize_surface_concept_peak_windows,
    summarize_surface_concept_raw_sequences,
    summarize_surface_concept_sequences,
    surface_concept_hits_to_processed_dataframe,
)

__all__ = [
    # constants
    "BINNING_FACTOR",
    "DEFAULT_ROENTDEK_DETX_COLUMNS",
    "DEFAULT_ROENTDEK_DETY_COLUMNS",
    "DEFAULT_ROENTDEK_SIGNAL_COLUMNS",
    "DETBINS",
    "DLTS_COLORS",
    "TOF_FACTOR_NS",
    "TOF_FACTOR_NS_1D",
    "XY_BIN_SHIFT",
    "XY_FACTOR",
    # common helpers
    "compute_tof_segment_drift",
    "compute_same_pulse_detector_separations",
    "load_numeric_text_table",
    "normalize_signal_windows",
    "plot_detector_dead_zone_and_neighbors",
    "plot_detector_overview",
    "plot_processed_dataset_overview",
    "plot_same_pulse_detector_separations",
    "plot_signal_overlay_by_dlts",
    "plot_signal_window_breakdown",
    "plot_tof_segment_drift",
    "save_processed_raw_dataset",
    "summarize_processed_dataset",
    "summarize_signal_windows",
    # roentdek
    "analyze_roentdek_dataset",
    "attach_roentdek_measurements",
    "classify_roentdek_events",
    "parse_roentdek_events",
    "plot_roentdek_statistics",
    "roentdek_hits_to_dataframe",
    "summarize_roentdek_raw_events",
    # surface concept
    "analyze_surface_concept_dataset",
    "analyze_surface_concept_tdc_frame",
    "build_surface_concept_recovery_diagnostics",
    "extract_surface_concept_hits",
    "plot_partial_hit_efficiency_maps",
    "plot_surface_concept_peak_breakdown",
    "plot_surface_concept_peak_ratio_table",
    "plot_surface_concept_recovery_summary",
    "plot_surface_concept_recovery_yield",
    "plot_surface_concept_sequence_statistics",
    "reconstruct_surface_concept_dataset",
    "summarize_surface_concept_peak_windows",
    "summarize_surface_concept_raw_sequences",
    "summarize_surface_concept_sequences",
    "surface_concept_hits_to_processed_dataframe",
]
