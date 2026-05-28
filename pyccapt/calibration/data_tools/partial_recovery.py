"""Merge partial-hit recovery into the main DLD frame.

When ``load_tdc_raw=True`` is requested in the data-processing workflow,
the raw ``/tdc`` group contains every channel tick the detector emitted,
including pulses where only 1, 2, or 3 of the 4 delay-line channels fired
("orphan" pulses, ``has_dld_match == False``). The hardware/driver
discarded these because they could not be paired into a full
4-channel DLD event in real time. Many of them are still physical: a 2-
channel pulse on (ch0, ch1) gives a 1-D ``x`` position and a 1-D ToF;
the missing ``y`` is the only thing lost.

This module enumerates physically-valid pair combinations from orphan
TDC ticks (gated by the detector-surface constraint), converts each
recovered candidate into a row matching the calibration DLD schema with
``NaN`` for the axis that was not recovered, assigns fresh
``event_group_id`` values that do not collide with the existing DLD
GIDs, and concatenates the new rows into ``variables.data`` sorted by
``start_counter`` so the merged frame stays chronological. The
corresponding orphan TDC rows are updated in place
(``event_group_id`` ← new GID, ``has_dld_match`` ← ``True``) so the
``filter_tdc_by_dld`` save logic remains symmetric.

Two extra columns are written into the DLD frame:

* ``dlts`` (``int8``) — ``4`` for native or fully-recovered hits with
  both ``x_det`` and ``y_det``; ``2`` for partial hits where one axis
  is ``NaN``.
* ``dlts_quality`` (``str``) — ``'native'`` for the original DLD rows
  that came straight from the hardware-paired ``/dld`` group;
  ``'recovered'`` for rows produced by this module.

Downstream code (FDM, spatial crop, 3-D reconstruction) that needs both
detector axes must filter ``dlts == 4`` (or equivalently drop rows with
``NaN`` in either ``x_det (cm)`` or ``y_det (cm)``). ToF, mass-spectrum,
and 1-D detector-projection plots can include the partials directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pyccapt.calibration.data_tools._raw_workflow_common import (
    TOF_FACTOR_NS_1D,
    load_detector_constants,
)
from pyccapt.calibration.data_tools._raw_workflow_surface_concept import (
    _recover_surface_concept_partial_hits,
    _surface_concept_position_from_pair,
)
from pyccapt.calibration.data_tools.data_loadcrop import (
    EVENT_GROUP_ID_COLUMN,
    TDC_HAS_DLD_MATCH_COLUMN,
)

DLTS_COLUMN = "dlts"
DLTS_QUALITY_COLUMN = "dlts_quality"


def _annotate_native_rows(dld_df: pd.DataFrame) -> pd.DataFrame:
    """Tag rows that already exist in ``dld_df`` as native 4-DLTS hits."""
    if DLTS_COLUMN not in dld_df.columns:
        dld_df[DLTS_COLUMN] = np.int8(4)
    else:
        # Preserve any pre-existing dlts annotation (e.g. from a previous
        # merge that's being re-run) by only filling unset entries.
        mask = dld_df[DLTS_COLUMN].isna()
        dld_df.loc[mask, DLTS_COLUMN] = np.int8(4)
        dld_df[DLTS_COLUMN] = dld_df[DLTS_COLUMN].astype(np.int8)
    if DLTS_QUALITY_COLUMN not in dld_df.columns:
        dld_df[DLTS_QUALITY_COLUMN] = "native"
    else:
        mask = dld_df[DLTS_QUALITY_COLUMN].isna() | (dld_df[DLTS_QUALITY_COLUMN] == "")
        dld_df.loc[mask, DLTS_QUALITY_COLUMN] = "native"
    return dld_df


def _build_recovered_row(
    *,
    start_counter: int,
    high_voltage: float,
    pulse_v: float,
    pulse_l: float,
    recovered_hit: dict,
    new_gid: int,
    dld_columns: list[str],
    flight_path_length_mm: float = 110.0,
) -> dict:
    """Construct a recovered DLD row aligned with the processed schema.

    Missing axes (e.g. ``y_det`` for an x-only partial) are stored as
    ``NaN`` so downstream consumers can detect partial hits by
    ``data['y_det (cm)'].isna()`` instead of having to look at the
    ``dlts`` column.

    ``mc_uc (Da)`` is computed for every recovered row, including
    1-DLTS partials, using the centred-axis approximation: the missing
    detector coordinate is treated as 0 (centre of detector). Since
    ``|det_x|, |det_y| <= 4 cm`` while the flight path is ~110 mm, the
    spatial term in ``sqrt(x_det^2 + y_det^2 + L^2)`` contributes
    ~0.1 % to the flight-path length, so this approximation gives a
    usable mc estimate for the mass spectrum. Partials therefore show
    up at roughly the right mass position (rather than clustering at
    0), and downstream code that needs exact mc (calibration, ranging)
    filters them via ``dlts == 4`` or the NaN detector mask.
    """
    from pyccapt.calibration.mc import mc_tools

    detector_axis = recovered_hit.get("detector_axis", "xy")
    if detector_axis == "xy":
        dlts_value = 4
        # A 3-of-4 time-sum recovery is a full xy position (dlts 4) but
        # derived with one reconstructed delay-line end; label it
        # distinctly so it can be told apart from native / 4-stop xy.
        if recovered_hit.get("recovery_method") == "3of4":
            quality = "recovered_xy_3of4"
        else:
            quality = "recovered_xy"
    else:
        dlts_value = 2
        quality = f"recovered_{detector_axis}"

    x_det_recovered = float(recovered_hit["x_det (cm)"])  # NaN for y-only
    y_det_recovered = float(recovered_hit["y_det (cm)"])  # NaN for x-only
    tof_ns = float(recovered_hit["tof (ns)"])

    # Centred-axis approximation for mc_uc: use the recovered axis at
    # its real value, set the missing axis to 0. This produces a
    # reasonable mc estimate so partials are visible in the mass
    # spectrum near their true mass instead of piling up at zero.
    x_for_mc = x_det_recovered if np.isfinite(x_det_recovered) else 0.0
    y_for_mc = y_det_recovered if np.isfinite(y_det_recovered) else 0.0
    try:
        mc_uc_value = float(
            mc_tools.tof2mc(
                t=np.array([tof_ns], dtype=float),
                t0=0,
                V=np.array([float(high_voltage)], dtype=float),
                xDet=np.array([x_for_mc], dtype=float),
                yDet=np.array([y_for_mc], dtype=float),
                flightPathLength=float(flight_path_length_mm),
                V_pulse=np.array([0.0], dtype=float),
                mode="voltage",
            )[0]
        )
    except Exception:
        mc_uc_value = float("nan")

    row: dict[str, float | int | str] = {column: 0.0 for column in dld_columns}
    row["high_voltage (V)"] = float(high_voltage)
    row["pulse_v (V)"] = float(pulse_v)
    row["pulse_l (pJ)"] = float(pulse_l)
    row["t (ns)"] = tof_ns
    row["x_det (cm)"] = x_det_recovered
    row["y_det (cm)"] = y_det_recovered
    row["start_counter"] = int(start_counter)
    row[EVENT_GROUP_ID_COLUMN] = int(new_gid)
    row[DLTS_COLUMN] = np.int8(dlts_value)
    row[DLTS_QUALITY_COLUMN] = quality
    # mc and mc_uc carry the centred-axis estimate. Downstream
    # calibration steps will overwrite these for fully-paired ions;
    # partials retain the estimate so they remain visible in the mass
    # spectrum view, marked by dlts<4 / dlts_quality=='recovered_*'.
    if "mc (Da)" in row:
        row["mc (Da)"] = mc_uc_value
    if "mc_uc (Da)" in row:
        row["mc_uc (Da)"] = mc_uc_value
    return row


def _subtract_firmware_stops(
    channels: np.ndarray,
    times: np.ndarray,
    native_events,
    match_tol_cm: float,
    axis_consistency_ns: float | None = 5.0,
    tof_match_tol_ns: float = 10.0,
):
    """Remove the stops the firmware used for its DLD event(s) from a matched
    multi-hit pulse and return the residual ``(channels, times)``.

    The Surface Concept quadrupel finder keeps only one hit per pulse, so a
    multi-hit pulse (more stops than the firmware used) can carry a second
    ion the firmware discarded. ``native_events`` is a list of
    ``(det_x, det_y, tof_ns)`` for the pulse's DLD event(s). For each we
    look for a COHERENT quadruplet -- a ``(ch0, ch1)`` pair within
    ``match_tol_cm`` of ``det_x`` and a ``(ch2, ch3)`` pair within
    ``match_tol_cm`` of ``det_y`` -- subject to two ToF gates:

    * the x-pair and y-pair ToFs agree within ``axis_consistency_ns`` (the
      per-axis time-sum coincidence ``t0 + t1 = t2 + t3``), and
    * the quadruplet ToF matches the firmware event's recorded ``tof_ns``
      within ``tof_match_tol_ns``.

    Both gates are needed because a detector COORDINATE depends only on the
    time DIFFERENCE of an axis, so a second ion at a similar position (but a
    very different flight time) would otherwise be matched and removed,
    fabricating a garbage residual or double-counting. The ToF match keys
    on the firmware's own recorded flight time, which differs between ions.

    The four matched stops are dropped; the residual is re-run through the
    normal recovery. Returns ``None`` if any native event has no coherent
    quadruplet within tolerance -- the caller then skips that pulse. (A
    firmware hit that was itself a 3-channel reconstruction has no full
    4-stop set in the raw stream, so it safely skips here.)
    """
    n = len(channels)
    used = [False] * n
    idx_by_channel = {c: [i for i in range(n) if int(channels[i]) == c] for c in (0, 1, 2, 3)}

    def _candidate_pairs(a_list, b_list, target):
        # (pos_err, i, j, tof) for unused (a, b) pairs within match_tol_cm.
        out = []
        for i in a_list:
            if used[i]:
                continue
            ti = float(times[i])
            for j in b_list:
                if used[j]:
                    continue
                tj = float(times[j])
                pos = _surface_concept_position_from_pair(ti, tj)
                err = abs(pos - target)
                if err <= match_tol_cm:
                    out.append((err, i, j, (ti + tj) * TOF_FACTOR_NS_1D))
        return out

    for event in native_events:
        det_x, det_y = float(event[0]), float(event[1])
        native_tof = float(event[2]) if len(event) > 2 else float("nan")
        cand_x = _candidate_pairs(idx_by_channel[0], idx_by_channel[1], det_x)
        cand_y = _candidate_pairs(idx_by_channel[2], idx_by_channel[3], det_y)
        if not cand_x or not cand_y:
            return None
        best = None  # (score, xi, xj, yi, yj)
        for ex, xi, xj, tof_x in cand_x:
            for ey, yi, yj, tof_y in cand_y:
                # (1) per-axis time-sum coincidence (same ion's two axes).
                if axis_consistency_ns is not None and abs(tof_x - tof_y) > axis_consistency_ns:
                    continue
                # (2) match the firmware event's recorded flight time, which
                # separates ions that share a similar position.
                quad_tof = 0.5 * (tof_x + tof_y)
                if np.isfinite(native_tof) and abs(quad_tof - native_tof) > tof_match_tol_ns:
                    continue
                score = ex + ey
                if best is None or score < best[0]:
                    best = (score, xi, xj, yi, yj)
        if best is None:
            return None
        _, xi, xj, yi, yj = best
        used[xi] = used[xj] = used[yi] = used[yj] = True

    residual = [i for i in range(n) if not used[i]]
    if not residual:
        return None
    return (
        np.asarray([channels[i] for i in residual], dtype=np.int64),
        np.asarray([times[i] for i in residual], dtype=np.int64),
    )


def _recover_for_pulse(
    pulse_channels: np.ndarray,
    pulse_times: np.ndarray,
    detector_radius_cm: float,
    max_tof_ns: float | None,
    axis_consistency_ns: float | None,
) -> list[dict]:
    """Wrapper that drops the empty / single-tick degenerate cases.

    The underlying recovery enumerates every (ch0, ch1) and (ch2, ch3)
    combination, applies the detector-surface gate, runs bipartite
    matching to pick non-conflicting per-axis pairs, then cross-matches
    the two axes by ToF agreement. A 20-tick pulse can yield multiple
    rows (e.g. 3 full xy + 2 partials) — exactly the multi-hit case
    the data-processing workflow needs to recover.
    """
    if pulse_channels.size < 2:
        return []
    return _recover_surface_concept_partial_hits(
        pulse_channels,
        pulse_times,
        detector_radius_cm=detector_radius_cm,
        max_tof_ns=max_tof_ns,
        axis_consistency_ns=axis_consistency_ns,
    )


def merge_partial_tdc_into_dld(
    variables,
    *,
    detector_kind: str = "surface_concept",
    conf=None,
    max_tof_ns: float | None = None,
    axis_consistency_ns: float | None = 5.0,
    include_one_d_partials: bool = False,
    recover_from_matched_multihit: bool = False,
    multihit_match_tol_cm: float = 0.1,
    verbose: bool = True,
) -> int:
    """Recover partial-hit events from orphan TDC ticks and merge into DLD.

    Order preservation
    ------------------
    Recovered atoms are inserted into ``variables.data`` at their correct
    ACQUISITION position, not by sorting on ``start_counter`` (which is a
    periodic counter that resets many times during a run, so sorting on it
    scrambles the event sequence and destroys the reconstructed z/depth).
    Each recovered atom is placed immediately after the native DLD rows of
    the most recent matched pulse that precedes its orphan pulse in the raw
    TDC stream; native rows keep their original order exactly.

    ``include_one_d_partials`` (default False) controls whether single-axis
    2-DLTS partials (one detector coordinate is ``NaN``) are merged. They
    are excluded by default because a NaN detector coordinate cannot be
    placed in the 3-D reconstruction; only full (x, y) hits -- native-style
    4-channel and 3-of-4 time-sum recoveries -- are added. Set it True to
    also append the 1-D partials (visible in the mass spectrum via their
    centred-axis mc estimate, but dropped by the 3-D reconstruction).

    ``recover_from_matched_multihit`` (default False) additionally recovers
    ions the firmware discarded from MATCHED multi-hit pulses. The Surface
    Concept quadrupel finder keeps one hit per pulse, so a pulse that fired
    more stops than its DLD event(s) used may contain a second ion. For
    each such pulse the stops the firmware used are inverse-matched to its
    DLD event(s) (the (ch0,ch1) pair closest to det_x and the (ch2,ch3)
    pair closest to det_y, within ``multihit_match_tol_cm``) and removed;
    the residual stops are then run through the normal recovery. It is
    opt-in because the inverse match is heuristic -- if the firmware's
    stops cannot be matched within tolerance the pulse is skipped, to avoid
    re-reconstructing (double-counting) the firmware's own hit. Recovered
    rows are tagged ``recovered_xy`` / ``recovered_xy_3of4`` like the orphan
    path.

    Parameters
    ----------
    variables :
        The shared-variables object produced by the data-processing
        notebook. Must already have ``variables.data`` (the DLD frame
        with an ``event_group_id`` column) and ``variables.data_tdc``
        (the raw TDC frame with ``event_group_id`` and ``has_dld_match``
        columns). Both are populated by ``load_data(..., load_tdc_raw=True)``.
    detector_kind : str, default ``'surface_concept'``
        Selects the detector-geometry constants used to gate the
        recovered positions against the active detector area.
    conf : optional
        ``pyccapt`` configuration mapping; passed to
        :func:`load_detector_constants` to honour per-rig overrides.
    max_tof_ns : float, optional
        If supplied, reject candidate pairs whose 1-D ToF lies outside
        ``(0, max_tof_ns]``. When ``None`` and
        ``variables.max_tof`` is set, the latter is used.
    axis_consistency_ns : float or None, default 5.0
        Forwarded to the underlying recovery. Maximum allowed
        ``|tof_x - tof_y|`` when promoting an (x-pair, y-pair) match
        to a full 4-DLTS xy hit. ``None`` skips the agreement check.
    verbose : bool, default True
        Print a summary of how many rows were merged.

    Returns
    -------
    int
        Number of recovered partial-hit rows appended to ``variables.data``.
    """
    if variables.data is None:
        raise ValueError("variables.data is None; load a dataset first.")
    if getattr(variables, "data_tdc", None) is None:
        raise ValueError(
            "variables.data_tdc is None; partial-hit recovery requires "
            "load_tdc_raw=True so the raw /tdc group is available."
        )
    if EVENT_GROUP_ID_COLUMN not in variables.data.columns:
        raise ValueError(
            f"variables.data is missing the {EVENT_GROUP_ID_COLUMN!r} column; "
            "reload the dataset with load_tdc_raw=True."
        )
    if (
        EVENT_GROUP_ID_COLUMN not in variables.data_tdc.columns
        or TDC_HAS_DLD_MATCH_COLUMN not in variables.data_tdc.columns
    ):
        raise ValueError(
            "variables.data_tdc is missing the event_group_id / "
            "has_dld_match columns; reload via fetch_dataset_with_tdc."
        )

    constants = load_detector_constants(detector_kind, conf=conf)
    detector_radius_cm = float(constants["detector_limit_cm"])

    if max_tof_ns is None:
        max_tof_ns = getattr(variables, "max_tof", None)
        if max_tof_ns is not None:
            max_tof_ns = float(max_tof_ns)

    # Flight-path length used for the centred-axis mc_uc estimate on
    # recovered rows; falls back to the pyccapt default if not set.
    flight_path_length_mm = float(getattr(variables, "flight_path_length", 110.0) or 110.0)

    tdc_df = variables.data_tdc
    dld_df = variables.data

    # Annotate the native rows so the merged frame has a consistent
    # ``dlts`` / ``dlts_quality`` schema regardless of recovery results.
    dld_df = _annotate_native_rows(dld_df.copy())

    orphan_mask = ~tdc_df[TDC_HAS_DLD_MATCH_COLUMN].to_numpy(dtype=bool)
    orphans = tdc_df.loc[orphan_mask]
    if orphans.empty and not recover_from_matched_multihit:
        # No orphan pulses to recover, and matched-multihit residual
        # recovery is off -- nothing to do.
        if verbose:
            print("[partial_recovery] No orphan tdc rows; nothing to recover.")
        variables.data = dld_df.reset_index(drop=True)
        variables.sync_from_data(update_backups=True)
        return 0

    # Group orphan ticks by pulse trigger.
    #
    # IMPORTANT: ``start_counter`` is a uint32 hardware counter that
    # WRAPS roughly every ~71 minutes at typical rates. Sorting by
    # start_counter (the previous approach) lumped two physically
    # distinct pulses that happen to share a counter value (one before
    # and one after a wrap) into a single "mega-pulse", producing
    # spurious paired hits separated by hours.
    #
    # Instead we rely on the fact that, within the raw TDC stream, all
    # ticks belonging to one pulse are CONTIGUOUS and the orphan subset
    # (a boolean mask over tdc_df) preserves that acquisition order. So
    # we keep acquisition order and run-length-encode by a CHANGE in
    # start_counter: consecutive pulses always differ (the counter
    # increments, and even at a wrap MAX->low it still changes), while a
    # wrapped duplicate value reappears as a separate contiguous run
    # rather than being merged with its earlier namesake.
    orphan_channels = orphans["channel"].to_numpy()
    orphan_times = orphans["time_data"].to_numpy()
    orphan_sc = orphans["start_counter"].to_numpy()
    orphan_hv = orphans["high_voltage (V)"].to_numpy()
    orphan_pv = orphans["pulse_v (V)"].to_numpy()
    orphan_pl = (
        orphans["pulse_l (pJ)"].to_numpy()
        if "pulse_l (pJ)" in orphans.columns
        else np.zeros(len(orphans))
    )
    orphan_original_index = orphans.index.to_numpy()

    # Defensive: ensure strict acquisition order (in case the frame's
    # index was not a sorted RangeIndex). A stable sort by the original
    # index preserves per-pulse tick contiguity.
    acq_order = np.argsort(orphan_original_index, kind="stable")
    orphan_channels = orphan_channels[acq_order]
    orphan_times = orphan_times[acq_order]
    orphan_sc = orphan_sc[acq_order]
    orphan_hv = orphan_hv[acq_order]
    orphan_pv = orphan_pv[acq_order]
    orphan_pl = orphan_pl[acq_order]
    orphan_original_index = orphan_original_index[acq_order]

    # Run-length encode by a CHANGE in start_counter in acquisition
    # order; each contiguous equal-counter run is one pulse (wrap-safe).
    run_starts = np.r_[0, np.where(np.diff(orphan_sc) != 0)[0] + 1, orphan_sc.size]

    existing_max_gid = int(dld_df[EVENT_GROUP_ID_COLUMN].max()) if len(dld_df) > 0 else -1
    next_gid = existing_max_gid + 1

    dld_columns = list(dld_df.columns)
    if DLTS_COLUMN not in dld_columns:
        dld_columns.append(DLTS_COLUMN)
    if DLTS_QUALITY_COLUMN not in dld_columns:
        dld_columns.append(DLTS_QUALITY_COLUMN)

    # --- Acquisition-order placement (never sort on start_counter) -------
    # Native DLD rows are already in acquisition order. To insert each
    # recovered atom at the right place, map every native event_group_id to
    # the last native DLD row carrying it, and compute a running "last
    # matched gid" along the full TDC stream (matched gids increase with
    # acquisition, so a running max gives the last matched gid seen at each
    # row). An orphan pulse is then placed right after the native rows of
    # the matched pulse that most recently preceded it.
    native_gid = dld_df[EVENT_GROUP_ID_COLUMN].to_numpy()
    last_dld_row_of_gid: dict[int, int] = {}
    for _row_idx, _g in enumerate(native_gid):
        last_dld_row_of_gid[int(_g)] = _row_idx
    tdc_gid_full = tdc_df[EVENT_GROUP_ID_COLUMN].to_numpy()
    last_matched_gid_at_row = np.maximum.accumulate(
        np.where(tdc_gid_full >= 0, tdc_gid_full, -1)
    )

    new_rows: list[dict] = []
    # Native DLD row index AFTER which each recovered atom should be
    # inserted (parallel to ``new_rows``); -1 means "before all native".
    new_row_insert_after: list[int] = []
    # Map original-tdc-index -> new GID, so we can flip has_dld_match
    # and event_group_id on the TDC orphan rows that contributed.
    tdc_index_to_new_gid: dict[int, int] = {}

    for i in range(run_starts.size - 1):
        a, b = int(run_starts[i]), int(run_starts[i + 1])
        pulse_channels = orphan_channels[a:b]
        pulse_times = orphan_times[a:b]
        recovered = _recover_for_pulse(
            pulse_channels,
            pulse_times,
            detector_radius_cm=detector_radius_cm,
            max_tof_ns=max_tof_ns,
            axis_consistency_ns=axis_consistency_ns,
        )
        if not recovered:
            continue

        # By default merge only full (x, y) hits; a single-axis 2-DLTS
        # partial has a NaN detector coordinate that cannot be placed in
        # the 3-D reconstruction.
        if not include_one_d_partials:
            recovered = [h for h in recovered if h.get("detector_axis") == "xy"]
            if not recovered:
                continue

        pulse_sc = int(orphan_sc[a])
        pulse_hv = float(orphan_hv[a])
        pulse_pv = float(orphan_pv[a])
        pulse_pl = float(orphan_pl[a])

        # Insert position: right after the native rows of the matched pulse
        # most recently preceding this orphan pulse in the TDC stream.
        preceding_gid = int(last_matched_gid_at_row[int(orphan_original_index[a])])
        insert_after_row = last_dld_row_of_gid.get(preceding_gid, -1)

        # One GID per pulse — all recovered candidates from this pulse
        # share it so the dld<->tdc link stays per-pulse, not per-row.
        new_gid = next_gid
        next_gid += 1

        for hit in recovered:
            new_rows.append(
                _build_recovered_row(
                    start_counter=pulse_sc,
                    high_voltage=pulse_hv,
                    pulse_v=pulse_pv,
                    pulse_l=pulse_pl,
                    recovered_hit=hit,
                    new_gid=new_gid,
                    dld_columns=dld_columns,
                    flight_path_length_mm=flight_path_length_mm,
                )
            )
            new_row_insert_after.append(insert_after_row)

        # Tag every orphan tdc tick for this pulse with the new GID.
        for original_idx in orphan_original_index[a:b]:
            tdc_index_to_new_gid[int(original_idx)] = new_gid

    # --- Residual recovery from MATCHED multi-hit pulses -----------------
    # The firmware keeps one hit per pulse, so a matched pulse with more
    # stops than its DLD event(s) used may hold a second ion. Subtract the
    # firmware's stops (inverse-matched to its DLD events) and recover from
    # the residual. Opt-in; runs on the full TDC stream's matched pulses.
    if recover_from_matched_multihit and len(tdc_df) > 0:
        sc_full = tdc_df["start_counter"].to_numpy()
        ch_full = tdc_df["channel"].to_numpy()
        t_full = tdc_df["time_data"].to_numpy()
        hv_full = (
            tdc_df["high_voltage (V)"].to_numpy()
            if "high_voltage (V)" in tdc_df.columns
            else np.zeros(len(sc_full))
        )
        pv_full = (
            tdc_df["pulse_v (V)"].to_numpy()
            if "pulse_v (V)" in tdc_df.columns
            else np.zeros(len(sc_full))
        )
        pl_full = (
            tdc_df["pulse_l (pJ)"].to_numpy()
            if "pulse_l (pJ)" in tdc_df.columns
            else np.zeros(len(sc_full))
        )
        # Wrap-safe full-stream pulse boundaries.
        bnd = np.empty(len(sc_full), dtype=bool)
        bnd[0] = True
        bnd[1:] = sc_full[1:] != sc_full[:-1]
        pulse_starts = np.flatnonzero(bnd)
        pulse_ends = np.r_[pulse_starts[1:], len(sc_full)]
        pulse_gid = tdc_gid_full[pulse_starts]
        pulse_tick = pulse_ends - pulse_starts
        # K (native DLD events) per gid, and candidate pulses where the raw
        # pulse has more stops than the firmware's 4*K.
        k_series = dld_df[EVENT_GROUP_ID_COLUMN].value_counts()
        k_per_pulse = pd.Series(pulse_gid).map(k_series).fillna(0).to_numpy()
        candidates = np.flatnonzero((pulse_gid >= 0) & (pulse_tick > 4 * k_per_pulse))
        if candidates.size:
            cand_gids = {int(pulse_gid[p]) for p in candidates}
            x_det_native = dld_df["x_det (cm)"].to_numpy()
            y_det_native = dld_df["y_det (cm)"].to_numpy()
            tof_native = (
                dld_df["t (ns)"].to_numpy()
                if "t (ns)" in dld_df.columns
                else np.full(len(dld_df), np.nan)
            )
            events_by_gid: dict[int, list] = {}
            for row_idx, g in enumerate(native_gid):
                gi = int(g)
                if gi in cand_gids:
                    events_by_gid.setdefault(gi, []).append(
                        (x_det_native[row_idx], y_det_native[row_idx], tof_native[row_idx])
                    )
            for p in candidates:
                gid = int(pulse_gid[p])
                events = events_by_gid.get(gid)
                if not events:
                    continue
                s, e = int(pulse_starts[p]), int(pulse_ends[p])
                residual = _subtract_firmware_stops(
                    ch_full[s:e], t_full[s:e], events,
                    match_tol_cm=multihit_match_tol_cm,
                    axis_consistency_ns=axis_consistency_ns,
                )
                if residual is None:
                    continue
                res_channels, res_times = residual
                if res_channels.size < 2:
                    continue
                recovered = _recover_for_pulse(
                    res_channels,
                    res_times,
                    detector_radius_cm=detector_radius_cm,
                    max_tof_ns=max_tof_ns,
                    axis_consistency_ns=axis_consistency_ns,
                )
                if not recovered:
                    continue
                if not include_one_d_partials:
                    recovered = [h for h in recovered if h.get("detector_axis") == "xy"]
                    if not recovered:
                        continue
                insert_after_row = last_dld_row_of_gid.get(gid, -1)
                pulse_sc = int(sc_full[s])
                pulse_hv = float(hv_full[s])
                pulse_pv = float(pv_full[s])
                pulse_pl = float(pl_full[s])
                new_gid = next_gid
                next_gid += 1
                for hit in recovered:
                    new_rows.append(
                        _build_recovered_row(
                            start_counter=pulse_sc,
                            high_voltage=pulse_hv,
                            pulse_v=pulse_pv,
                            pulse_l=pulse_pl,
                            recovered_hit=hit,
                            new_gid=new_gid,
                            dld_columns=dld_columns,
                            flight_path_length_mm=flight_path_length_mm,
                        )
                    )
                    new_row_insert_after.append(insert_after_row)

    if not new_rows:
        if verbose:
            print(
                "[partial_recovery] No physically-valid partial hits could be "
                "recovered from the orphan tdc rows."
            )
        variables.data = dld_df.reset_index(drop=True)
        variables.sync_from_data(update_backups=True)
        return 0

    recovered_frame = pd.DataFrame(new_rows, columns=dld_columns)

    # Interleave the recovered atoms into the native sequence at their
    # acquisition position. The sort key for a native row is its own row
    # index; for a recovered atom it is the native row it should follow.
    # A secondary "is recovered" flag breaks ties so a recovered atom with
    # insert_after_row == R lands AFTER native row R and before native row
    # R+1; a tertiary append-order key keeps multiple recovered atoms (and
    # multi-hit atoms) in acquisition order. This NEVER sorts on
    # start_counter, which is a periodic counter that would scramble depth.
    n_native = len(dld_df)
    n_rec = len(new_rows)
    insert_after = np.asarray(new_row_insert_after, dtype=np.int64)
    primary = np.concatenate([np.arange(n_native, dtype=np.int64), insert_after])
    is_recovered = np.concatenate(
        [np.zeros(n_native, dtype=np.int8), np.ones(n_rec, dtype=np.int8)]
    )
    tiebreak = np.concatenate(
        [np.arange(n_native, dtype=np.int64), np.arange(n_rec, dtype=np.int64)]
    )
    order = np.lexsort((tiebreak, is_recovered, primary))
    merged = (
        pd.concat([dld_df, recovered_frame], ignore_index=True)
        .iloc[order]
        .reset_index(drop=True)
    )

    # Update the orphan tdc rows in place: flip has_dld_match and write
    # the new GIDs so filter_tdc_by_dld preserves them after cropping.
    updated_tdc = tdc_df.copy()
    if tdc_index_to_new_gid:
        idx_array = np.array(list(tdc_index_to_new_gid.keys()), dtype=np.int64)
        gid_array = np.array(list(tdc_index_to_new_gid.values()), dtype=np.int64)
        updated_tdc.loc[idx_array, EVENT_GROUP_ID_COLUMN] = gid_array
        updated_tdc.loc[idx_array, TDC_HAS_DLD_MATCH_COLUMN] = True

    variables.data = merged
    variables.data_tdc = updated_tdc
    # Refresh BOTH backups so reset / crop operations restore the
    # post-merge state rather than the pre-merge state.
    variables.data_backup = merged.copy()
    variables.data_tdc_backup = updated_tdc.copy()
    variables.sync_from_data(update_backups=True)

    n_added = len(recovered_frame)
    if verbose:
        n_xy = int((recovered_frame[DLTS_COLUMN] == 4).sum())
        n_2dlts = int((recovered_frame[DLTS_COLUMN] == 2).sum())
        print(
            f"[partial_recovery] Merged {n_added} recovered rows "
            f"({n_xy} full xy, {n_2dlts} single-axis partial) into the DLD frame. "
            f"Detector radius gate: {detector_radius_cm:.2f} cm. "
            f"New GID range: {existing_max_gid + 1}..{next_gid - 1}."
        )
    return n_added
