from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from pyccapt.calibration.data_tools import data_tools
from pyccapt.calibration.mc import tof_tools


def _loader_progress(*, total: int, enabled: bool, desc: str):
    return tqdm(
        total=total,
        desc=desc,
        unit="stage",
        dynamic_ncols=True,
        disable=not enabled,
        leave=True,
    )


def load_data(
    dataset_path,
    max_mc,
    flightPathLength,
    pulse_mode,
    tdc,
    variables,
    processing_mode=True,
    load_tdc_raw=False,
    merge_partial_tdc=None,
    recover_from_matched_multihit=False,
    multihit_match_tol_cm=0.1,
    detector_kind='surface_concept',
    show_progress=True,
):
    """Load a calibration dataset and stash it on ``variables``.

    Parameters
    ----------
    load_tdc_raw : bool, default False
            If True and ``tdc == 'pyccapt'``, also load the raw ``/tdc`` group from
            the h5 file. The dld and tdc dataframes are linked via a shared
            ``event_group_id`` column so dld filtering decisions can later be
            propagated to tdc at save time. Stored on ``variables.data_tdc``.
    merge_partial_tdc : bool or None, default None
            Whether to run partial-hit recovery after the main DLD frame is
            loaded. ``None`` (default) means "follow ``load_tdc_raw``": if
            the raw ``/tdc`` group is loaded for a pyccapt h5 dataset,
            recovery runs automatically and the recovered atoms (full xy,
            3-of-4 time-sum, and 1-D partials) are appended to
            ``variables.data`` -- so enabling "Load raw tdc" actually adds
            the recovered events and increases ``len(variables.data)``.
            Pass ``True``/``False`` to force it on/off.
            When True (and ``load_tdc_raw`` is True on a pyccapt h5 dataset),
            after the main DLD frame is loaded, run the partial-hit recovery
            pipeline: orphan TDC ticks (pulses with 1-3 channels fired) are
            paired axis-by-axis, gated by the detector-surface constraint,
            and the physically-valid candidates are merged into
            ``variables.data`` as additional rows. Partial rows have
            ``NaN`` on the axis that could not be reconstructed and a
            ``dlts`` column (``2`` for single-axis, ``4`` for native or
            recovered xy). Downstream code that needs both detector axes
            must filter rows with NaN in ``x_det (cm)`` or ``y_det (cm)``.
    recover_from_matched_multihit : bool, default False
            Forwarded to :func:`partial_recovery.merge_partial_tdc_into_dld`.
            When True (and recovery runs), also mine MATCHED multi-hit
            pulses for a second ion the firmware discarded (it keeps one
            hit per pulse): the firmware's stops are inverse-matched to its
            DLD event(s) and removed, and the residual stops are recovered.
            Opt-in because the inverse match is heuristic (a pulse whose
            firmware stops cannot be matched within ``multihit_match_tol_cm``
            is skipped, so the firmware's own hit is never double-counted).
    multihit_match_tol_cm : float, default 0.1
            Position tolerance (cm) for inverse-matching the firmware's
            stops to its DLD event in matched multi-hit recovery.
    detector_kind : str, default ``'surface_concept'``
            Detector-geometry key forwarded to the partial-hit recovery.
            Use ``'roentdek'`` for RoentDek hardware. Ignored when
            ``merge_partial_tdc=False``.
    """
    if tdc == 'pyccapt':
        # Check that the dataset is a valid pyccapt dataset with .h5 extension
        if not dataset_path.endswith(('.h5', '.H5')):
            raise ValueError('The dataset should be a valid pyccapt dataset with .h5 extension')
    elif tdc == 'leap_epos':
        # Check that the dataset is a valid leap_epos dataset with .epos extension
        if not dataset_path.endswith(('.epos', '.EPOS')):
            raise ValueError('The dataset should be a valid leap_epos dataset with .epos extension')
    elif tdc in {'pos', 'leap_pos'}:
        # Check that the dataset is a valid pos dataset with .pos extension
        if not dataset_path.endswith(('.pos', '.POS')):
            raise ValueError('The dataset should be a valid pos dataset with .pos extension')
    elif tdc == 'leap_apt':
        # Check that the dataset is a valid leap_apt dataset with .apt extension
        if not dataset_path.endswith(('.apt', '.APT')):
            raise ValueError('The dataset should be a valid leap_apt dataset with .apt extension')
    elif tdc == 'ato_v6':
        # Check that the dataset is a valid ato_v6 dataset with .ato extension
        if not dataset_path.endswith(('.ato', '.ATO')):
            raise ValueError('The dataset should be a valid ato_v6 dataset with .ato extension')

    # Resolve the "follow load_tdc_raw" default: when the caller leaves
    # merge_partial_tdc unset (None), recover partial hits automatically
    # whenever the raw /tdc group is loaded for a pyccapt h5 dataset. This
    # makes "Load raw tdc" actually add the recovered atoms to
    # variables.data instead of only loading the raw stops. Pass an
    # explicit True/False to override.
    if merge_partial_tdc is None:
        merge_partial_tdc = bool(load_tdc_raw) and tdc == 'pyccapt'

    # Staged progress bar over the three coarse phases of a load: reading
    # the file (the slow, opaque HDF5/POS/EPOS read), synchronising the
    # shared arrays, and the optional partial-hit recovery. The single
    # file read can't be sub-divided cheaply, so the bar is staged rather
    # than per-row. Works the same in raw-tdc and normal modes.
    progress = _loader_progress(total=3, enabled=show_progress, desc="Loading dataset")
    progress.set_postfix_str("reading dataset file")

    if processing_mode:
        # Calculate the maximum possible time of flight (TOF)
        max_tof = int(tof_tools.mc2tof(max_mc, 1000, 0, 0, flightPathLength))
        print('The maximum possible TOF is:', max_tof, 'ns')
        print('=============================')
        variables.pulse_mode = pulse_mode
        dataset_path_obj = Path(dataset_path)
        variables.path = str(dataset_path_obj)
        variables.dataset_name = dataset_path_obj.stem
        output_dir = dataset_path_obj.parent / variables.dataset_name / 'data_processing'
        variables.set_result_data_directory(output_dir)
        variables.result_data_name = variables.dataset_name
        variables.set_result_directory(output_dir)

        print('The data will be saved on the path:', variables.result_data_path)
        print('=============================')
        print('The dataset name after saving is:', variables.result_data_name)
        print('=============================')
        print('The figures will be saved on the path:', variables.result_path)
        print('=============================')

        # Create data frame out of hdf5 file dataset
        tdc_df = None
        if tdc == 'pyccapt':
            try:
                loaded = data_tools.load_data(dataset_path, tdc, mode='raw', load_tdc=load_tdc_raw)
                if isinstance(loaded, tuple):
                    dld_group_storage, tdc_df = loaded
                else:
                    dld_group_storage = loaded
                print('The data is loaded in raw mode' + (' (with raw tdc)' if tdc_df is not None else ''))
                mode = 'raw'
            except Exception:
                loaded = data_tools.load_data(dataset_path, tdc, mode='processed', load_tdc=load_tdc_raw)
                if isinstance(loaded, tuple):
                    dld_group_storage, tdc_df = loaded
                else:
                    dld_group_storage = loaded
                print('The data is loaded in processed mode' + (' (with raw tdc)' if tdc_df is not None else ''))
                if load_tdc_raw and tdc_df is None:
                    print(
                        'WARNING: cannot load raw TDC -- the dataset is already processed '
                        'or does not contain a raw /tdc group. Raw TDC is only available '
                        'on pyccapt h5 files saved in raw acquisition mode.'
                    )
                if 'x (nm)' not in dld_group_storage:
                    mode = 'raw'
                else:
                    mode = 'processed'
        else:
            if load_tdc_raw:
                _ext_map = {
                    'leap_epos': '.epos',
                    'pos': '.pos',
                    'leap_pos': '.pos',
                    'leap_apt': '.apt',
                    'ato_v6': '.ato',
                }
                _fmt = _ext_map.get(tdc, tdc)
                print(
                    f'WARNING: cannot load raw TDC -- {_fmt} files do not contain raw '
                    'TDC data. Raw TDC is only available on pyccapt h5 files saved in '
                    'raw acquisition mode. Ignoring load_tdc_raw=True.'
                )
            load_data_type = 'leap_pos' if tdc in {'pos', 'leap_pos'} else tdc
            dld_group_storage = data_tools.load_data(dataset_path, load_data_type)

        if tdc == 'pyccapt' and mode == 'raw':
            data = data_tools.remove_invalid_data(dld_group_storage, max_tof)
            data = data_tools.pyccapt_raw_to_processed(data)
        else:
            data = dld_group_storage

    elif not processing_mode:
        tdc_df = None
        max_tof = int(tof_tools.mc2tof(max_mc, 1000, 0, 0, flightPathLength))
        variables.pulse_mode = pulse_mode
        dataset_path_obj = Path(dataset_path)
        variables.path = str(dataset_path_obj)
        variables.dataset_name = dataset_path_obj.stem
        output_dir = dataset_path_obj.parent / variables.dataset_name / 'visualization'
        variables.set_result_data_directory(output_dir)
        variables.result_data_name = variables.dataset_name
        variables.set_result_directory(output_dir)

        print('The data will be saved on the path:', variables.result_data_path)
        print('=============================')
        print('The dataset name after saving is:', variables.result_data_name)
        print('=============================')
        print('The figures will be saved on the path:', variables.result_path)
        print('=============================')

        # Create data frame out of hdf5 file dataset
        if tdc == 'pyccapt':
            loaded = data_tools.load_data(dataset_path, tdc, mode='processed', load_tdc=load_tdc_raw)
            if isinstance(loaded, tuple):
                data, tdc_df = loaded
            else:
                data = loaded
            if load_tdc_raw and tdc_df is None:
                print(
                    'WARNING: cannot load raw TDC -- the dataset is already processed '
                    'or does not contain a raw /tdc group. Raw TDC is only available '
                    'on pyccapt h5 files saved in raw acquisition mode.'
                )
        else:
            if load_tdc_raw:
                _ext_map = {
                    'leap_epos': '.epos',
                    'pos': '.pos',
                    'leap_pos': '.pos',
                    'leap_apt': '.apt',
                    'ato_v6': '.ato',
                }
                _fmt = _ext_map.get(tdc, tdc)
                print(
                    f'WARNING: cannot load raw TDC -- {_fmt} files do not contain raw '
                    'TDC data. Raw TDC is only available on pyccapt h5 files saved in '
                    'raw acquisition mode. Ignoring load_tdc_raw=True.'
                )
            load_data_type = 'leap_pos' if tdc in {'pos', 'leap_pos'} else tdc
            data = data_tools.load_data(dataset_path, load_data_type)

    progress.update(1)  # phase 1: dataset read
    progress.set_postfix_str("synchronizing shared variables")

    print('Total number of Ions:', len(data))
    if tdc_df is not None:
        print('Loaded raw tdc rows:', len(tdc_df))

    variables.data = data
    variables.data_tdc = tdc_df
    variables.data_tdc_backup = tdc_df.copy() if tdc_df is not None else None
    variables.max_mc = max_mc
    variables.max_tof = max_tof
    variables.flight_path_length = flightPathLength
    variables.pulse_mode = pulse_mode
    variables.sync_from_data(update_backups=True)
    progress.update(1)  # phase 2: shared variables synchronized

    if merge_partial_tdc:
        progress.set_postfix_str("recovering partial hits")
        if tdc != 'pyccapt':
            print(
                'WARNING: merge_partial_tdc=True requires a pyccapt h5 dataset; '
                f'ignoring (tdc={tdc!r}).'
            )
        elif not load_tdc_raw or tdc_df is None:
            print(
                'WARNING: merge_partial_tdc=True requires load_tdc_raw=True with '
                'a successful raw /tdc load; ignoring.'
            )
        else:
            from pyccapt.calibration.data_tools.partial_recovery import (
                merge_partial_tdc_into_dld,
            )

            try:
                merge_partial_tdc_into_dld(
                    variables,
                    detector_kind=detector_kind,
                    max_tof_ns=float(max_tof),
                    recover_from_matched_multihit=bool(recover_from_matched_multihit),
                    multihit_match_tol_cm=float(multihit_match_tol_cm),
                )
            except Exception as exc:
                print(f'WARNING: merge_partial_tdc failed: {exc}')

    progress.update(1)  # phase 3: partial-hit recovery (or skipped)
    progress.set_postfix_str("done")
    progress.close()


def summarize_loaded_events(variables, *, print_summary=True):
    """Report complete vs partial event counts for the loaded dataset.

    Combines two views of the just-loaded data:

    * DLD side (``variables.data``): how many reconstructed events the file
      contains. When partial-hit recovery has run (the ``dlts`` column is
      present) the count is split into complete (native + recovered xy) and
      partial (single-axis recovered) hits.
    * Raw TDC side (``variables.data_tdc``, present only when the dataset
      was loaded with raw tdc): how many pulses fired the full delay-line
      channel set (complete) versus only part of it (partial), via
      :func:`partial_hit_diagnostics.tdc_pulse_completeness`.

    Returns the counts as a dict and, by default, prints a short summary.
    """
    from pyccapt.calibration.data_tools.partial_hit_diagnostics import (
        matched_dld_tdc_residuals,
        tdc_pulse_completeness,
    )

    summary: dict[str, int] = {}
    data = getattr(variables, "data", None)

    if data is not None and "dlts" in getattr(data, "columns", []):
        # ``dlts`` carries the per-row delay-line-timestamp count: 4 for a
        # native or fully-recovered two-axis hit, 2 for a single-axis
        # partial recovered by merge_partial_tdc.
        dlts = np.asarray(data["dlts"].to_numpy())
        summary["dld_total"] = int(len(data))
        summary["dld_complete"] = int((dlts == 4).sum())
        summary["dld_partial"] = int((dlts == 2).sum())
    elif data is not None:
        # No recovery column: every DLD row is a complete reconstructed event.
        summary["dld_total"] = int(len(data))
        summary["dld_complete"] = int(len(data))
        summary["dld_partial"] = 0
    else:
        summary["dld_total"] = summary["dld_complete"] = summary["dld_partial"] = 0

    # DLD events that have no raw-TDC pulse behind them. At load time
    # build_event_group_mapping links each dld row to its raw /tdc stops by
    # start_counter; a dld run that never pairs with a tdc run is tagged with
    # a NEGATIVE event_group_id (data_loadcrop.build_event_group_mapping).
    # The column exists only when the file was loaded with load_tdc_raw=True,
    # so the count is "known" only then. Recovered atoms get positive gids,
    # so a negative gid is always an unmatched *native* dld event.
    summary["dld_without_match_known"] = False
    summary["dld_without_match"] = 0
    if data is not None and "event_group_id" in getattr(data, "columns", []):
        try:
            gid = np.asarray(data["event_group_id"].to_numpy(), dtype=np.float64)
            summary["dld_without_match"] = int(np.sum(np.isfinite(gid) & (gid < 0)))
            summary["dld_without_match_known"] = True
        except (TypeError, ValueError):
            # Non-numeric event_group_id (unexpected) -- leave as unknown.
            summary["dld_without_match_known"] = False

    tdc = getattr(variables, "data_tdc", None)
    tdc_stats = tdc_pulse_completeness(tdc)
    summary["tdc_loaded"] = tdc is not None
    summary["tdc_total_pulses"] = tdc_stats["total_pulses"]
    summary["tdc_complete"] = tdc_stats["complete"]   # fired all channels
    summary["tdc_partial"] = tdc_stats["partial"]     # fired some-but-not-all
    summary["tdc_with_dld_match"] = tdc_stats["with_dld_match"]
    summary["tdc_channels_required"] = tdc_stats["channels_required"]
    summary["tdc_channel_histogram"] = tdc_stats["channel_histogram"]

    # Match-quality cross-check: re-derive det_x/det_y/tof from the raw stops
    # of cleanly-matched single-hit pulses and compare to the firmware-written
    # DLD values. Small residuals confirm the event_group_id link (and the
    # reconstruction formula) are correct.
    if tdc is not None and data is not None:
        match_res = matched_dld_tdc_residuals(data, tdc)
    else:
        match_res = {"n_compared": 0, "reason": "raw tdc not loaded"}
    summary["match_check"] = match_res

    if print_summary:
        print("Event statistics")
        print("================")
        print(f"  Complete events in DLD (atoms)          : {summary['dld_complete']:,}")
        if summary["dld_partial"]:
            print(f"  Partial events recovered into DLD       : {summary['dld_partial']:,}")
        if summary["dld_without_match_known"]:
            n_unmatched = summary["dld_without_match"]
            total = max(summary["dld_total"], 1)
            print(
                f"  DLD events with no raw-TDC match        : {n_unmatched:,} "
                f"({n_unmatched / total:.3%})"
            )
            if n_unmatched:
                # Why: build_event_group_mapping pairs dld<->tdc by contiguous
                # start_counter runs; a dld run with no tdc counterpart keeps a
                # negative event_group_id. This happens when the raw /tdc stops
                # for those pulses were never written -- an acquisition stop or
                # crash flushed /dld but not /tdc, or a partial-write recovery
                # truncated the tail. The dld atoms are valid and kept.
                print(
                    "      (reason: no raw /tdc stops recorded for these pulses -- "
                    "dld/tdc start_counter runs do not align,"
                )
                print(
                    "       usually an acquisition stop/crash or a truncated tail; "
                    "tagged with a negative event_group_id and kept.)"
                )
        if summary["tdc_loaded"]:
            req = summary["tdc_channels_required"]
            print(f"  Raw TDC pulses (triggers)               : {summary['tdc_total_pulses']:,}")
            # Per-channel-count histogram (descending), so nothing is hidden.
            hist = summary["tdc_channel_histogram"]
            for k in sorted(hist, reverse=True):
                if k == 0:
                    continue
                tag = " (complete)" if k == req else ""
                print(f"      fired {k} channel(s){tag:<12}: {hist[k]:,}")
            print(f"  Complete pulses in raw TDC ({req} ch)      : {summary['tdc_complete']:,}")
            print(f"  Partial pulses in raw TDC (<{req} ch)      : {summary['tdc_partial']:,}")
            # The authoritative 'produced an atom' count. It can differ
            # slightly from the complete-pulse count because the raw stop
            # stream and the reconstructed DLD stream are captured
            # separately (see tdc_pulse_completeness docstring).
            print(f"  Raw TDC pulses linked to a DLD event    : {summary['tdc_with_dld_match']:,}")

            # Match-quality cross-check (TDC vs DLD). Re-derives the detector
            # position and ToF from the raw stops of cleanly-matched single-hit
            # pulses and compares to the firmware-recorded DLD values; near-zero
            # residuals confirm the match (event_group_id link + formula).
            mc = summary["match_check"]
            print("  Match check (TDC-derived vs DLD recorded, clean 4-ch single-hit pulses):")
            if mc.get("n_compared", 0) > 0:
                print(f"      pulses cross-checked                : {mc['n_compared']:,}")
                print(
                    f"      |dx_det| median / max (cm)          : "
                    f"{mc['dx_median_cm']:.2e} / {mc['dx_max_cm']:.2e}"
                )
                print(
                    f"      |dy_det| median / max (cm)          : "
                    f"{mc['dy_median_cm']:.2e} / {mc['dy_max_cm']:.2e}"
                )
                print(
                    f"      |dtof|   median / max (ns)          : "
                    f"{mc['dtof_median_ns']:.2e} / {mc['dtof_max_ns']:.2e}"
                )
                print(
                    f"      within tol (x/y={mc['pos_tol_cm']:.2e} cm, "
                    f"tof={mc['tof_tol_ns']:.2e} ns)  : "
                    f"{mc['frac_x_within_tol']:.1%} / "
                    f"{mc['frac_y_within_tol']:.1%} / "
                    f"{mc['frac_tof_within_tol']:.1%}"
                )
            else:
                print(f"      (not available: {mc.get('reason', 'no comparable pulses')})")
        else:
            print("  (raw TDC not loaded; enable 'Load raw tdc' for TDC-side counts)")

    return summary


def add_columns(variables, max_mc):

    if 'x (nm)' not in variables.data:
        variables.data.insert(0, 'x (nm)', np.zeros(len(variables.dld_t)))
    if 'y (nm)' not in variables.data:
        variables.data.insert(1, 'y (nm)', np.zeros(len(variables.dld_t)))
    if 'z (nm)' not in variables.data:
        variables.data.insert(2, 'z (nm)', np.zeros(len(variables.dld_t)))
    if 'mc (Da)' not in variables.data:
        variables.data.insert(4, 'mc (Da)', np.zeros(len(variables.dld_t)))
    if 'mc_uc (Da)' not in variables.data:
        variables.data.insert(5, 'mc_uc (Da)', variables.mc_uc)
    else:
        variables.data['mc_uc (Da)'] = variables.mc_uc
    if 't_c (ns)' not in variables.data:
        variables.data.insert(8, 't_c (ns)', np.zeros(len(variables.dld_t)))

    # Remove the data with mc biger than max mc
    mask = variables.data['mc (Da)'].to_numpy() > max_mc.value
    print('The number of data over max_mc:', len(mask[mask == True]))
    variables.data.drop(np.where(mask)[0], inplace=True)
    variables.data.reset_index(inplace=True, drop=True)

    # Remove the data with x,y,t = 0
    mask1 = variables.data['x (nm)'].to_numpy() == 0
    mask2 = variables.data['y (nm)'].to_numpy() == 0
    mask3 = variables.data['t (ns)'].to_numpy() == 0
    mask = np.logical_and(mask1, mask2)
    mask = np.logical_and(mask, mask3)
    print('The number of data with having t, x, and y equal to zero is:', len(mask[mask == True]))
    variables.data.drop(np.where(mask)[0], inplace=True)
    variables.data.reset_index(inplace=True, drop=True)
    variables.data_backup = variables.data.copy()


def load_calibrated_h5(dataset_path, variables, *, range_path=None, show_progress=True):
    """Load a PyCCAPT ``.h5`` (or LEAP ``.RHIT``) file for raw-data analysis.

    Three file layouts are supported, dispatched by file extension:

    1. **Calibrated PyCCAPT bundle** (``.h5``, preferred) — produced by
       ``data_tools.save_data(..., save_tdc=True, save_range=True)``. Contains
       ``/df`` (calibrated dld), and optionally ``/tdc`` (raw delay-line
       timestamps linked via ``event_group_id``) and ``/range`` (identified
       ion windows).
    2. **Pure raw PyCCAPT acquisition** (``.h5``) — the file as written by
       the control software: ``/dld`` and ``/tdc`` groups (no ``/df``,
       no ``/range``). The dld records are converted to the processed
       dataframe schema in memory, and the linked raw tdc rows are kept
       on ``variables.data_tdc``.
    3. **LEAP CAMECA RHIT** (``.rhit``) — a Cameca-LEAP ROOT bundle. Decoded
       via :func:`pyccapt.calibration.leap_tools.cameca_raw.rhit_load` and
       converted to the processed dataframe schema with
       :func:`...rhit_to_ccapt`. RHIT files have no raw delay-line tdc data
       (the mass/charge and detector positions are already calibrated by the
       instrument), so ``variables.data_tdc`` is set to ``None`` and the
       downstream analyses that require ``/tdc`` (DLTS-per-pulse, combinatorial
       recovery) are skipped automatically.

    Older datasets that store the range table in a separate
    ``<dataset>_range.h5`` file are also supported via ``range_path``.

    Populates ``variables.data``, ``variables.data_tdc``, ``variables.range_data``,
    and the standard backup fields.
    """
    import pandas as pd  # local import to keep optional Jupyter loaders fast

    from pyccapt.calibration.data_tools import data_loadcrop, data_tools

    progress = _loader_progress(total=5, enabled=show_progress, desc="Loading PyCCAPT HDF5")
    try:
        progress.set_postfix_str("validating dataset path")
        dataset_path_obj = Path(dataset_path)
        if not dataset_path_obj.is_file():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        variables.path = str(dataset_path_obj)
        variables.dataset_name = dataset_path_obj.stem
        output_dir = dataset_path_obj.parent / variables.dataset_name / 'raw_analysis'
        variables.set_result_data_directory(output_dir)
        variables.set_result_directory(output_dir)
        variables.result_data_name = variables.dataset_name
        progress.update(1)

        suffix = dataset_path_obj.suffix.lower()

        dld_df = None
        tdc_df = None
        # LEAP RHIT path: the file is a Cameca ROOT bundle, not HDF5. Trying
        # pd.read_hdf on it raises HDF5ExtError ("file signature not found")
        # because it isn't HDF5 at all — so we MUST dispatch on extension
        # before the calibrated-bundle attempt.
        if suffix == ".rhit":
            progress.set_postfix_str("reading LEAP RHIT raw data")
            from pyccapt.calibration.leap_tools.cameca_raw import (
                rhit_load,
                rhit_to_ccapt,
            )

            hits, _histograms, _metadata = rhit_load(str(dataset_path_obj))
            dld_df = rhit_to_ccapt(hits)
            tdc_df = None
            source = "leap_rhit"
            progress.update(1)  # step 2: read
            progress.update(1)  # step 3: normalize (already processed-schema)
        elif suffix in {".str", ".hits"}:
            # LEAP STR/HITS path: same story as RHIT — these are Cameca raw
            # bundles, not HDF5, so pd.read_hdf would raise HDF5ExtError.
            progress.set_postfix_str("reading LEAP STR/HITS raw data")
            from pyccapt.calibration.leap_tools.cameca_raw import (
                str_calculate_positions,
                str_load,
                str_to_ccapt,
            )

            hits, _metadata = str_load(str(dataset_path_obj))
            hits = str_calculate_positions(hits)
            dld_df = str_to_ccapt(hits)
            tdc_df = None
            source = "leap_str"
            progress.update(1)  # step 2: read
            progress.update(1)  # step 3: normalize (already processed-schema)
        else:
            # First try the calibrated bundle layout (/df). If that fails
            # because the file only has the raw acquisition groups, fall back
            # to raw /dld + /tdc, then to raw /dld only.
            source = "calibrated"
            progress.set_postfix_str("reading bundled /df and optional /tdc")
            try:
                loaded = data_tools.load_data(str(dataset_path_obj), 'pyccapt', mode='processed', load_tdc=True)
                if isinstance(loaded, tuple):
                    dld_df, tdc_df = loaded
                else:
                    dld_df, tdc_df = loaded, None
            except (KeyError, ValueError) as calibrated_error:
                progress.set_postfix_str("falling back to raw /dld + /tdc groups")
                try:
                    dld_df, tdc_df = data_loadcrop.fetch_dataset_with_tdc(str(dataset_path_obj))
                    source = "raw"
                except Exception as paired_error:
                    # Bundled /tdc either is missing or its start_counter does
                    # not align with /dld (e.g. our pyccapt-raw RHIT export
                    # has a single-channel tdc mirror that is just redundant
                    # with dld; pyccapt-raw STR has many tdc rows per event
                    # and that alignment may also fail on filtered exports).
                    # For the rest of the calibration workflow, /dld alone is
                    # sufficient. Try dld-only before giving up.
                    progress.set_postfix_str("falling back to /dld only (no tdc)")
                    dld_only = data_loadcrop.fetch_dataset_from_dld_grp(str(dataset_path_obj), extract_mode='dld')
                    if dld_only is None:
                        # Inspect the file to give a targeted error. The most
                        # common cause is a pyccapt-raw STR export that was
                        # saved before calibration was applied: that file has
                        # a /tdc group (raw delay-line counts) but no /dld
                        # group (which only exists once VDC / tof_ns / detx /
                        # dety have been calibrated against a matching RHIT).
                        import h5py

                        has_dld = False
                        has_tdc = False
                        try:
                            with h5py.File(str(dataset_path_obj), 'r') as _hf:
                                top_keys = list(_hf.keys())
                                has_dld = 'dld' in top_keys
                                has_tdc = 'tdc' in top_keys
                        except OSError:
                            top_keys = []
                        if has_tdc and not has_dld:
                            raise ValueError(
                                f"{dataset_path!r} only has a /tdc group (raw delay-line "
                                "counts) — no /dld group, so it cannot be loaded into the "
                                "calibration workflow. This typically happens when a STR "
                                "file was exported as pyccapt-raw HDF5 *before* clicking "
                                "'Calibrate from RHIT' in the cameca raw import workflow. "
                                "Re-export it from cameca raw import after running calibration, "
                                "or load it via the raw-data-analysis workflow which uses /tdc "
                                "directly."
                            ) from paired_error
                        raise ValueError(
                            f"Could not load {dataset_path!r}: not a calibrated bundle "
                            f"({calibrated_error}); /dld + /tdc paired load failed "
                            f"({paired_error}); /dld-only load also failed. "
                            f"File top-level groups: {top_keys}."
                        ) from paired_error
                    dld_df = dld_only
                    tdc_df = None
                    source = "raw_dld_only"
                    print(
                        "Loaded /dld group only -- /tdc was missing or its "
                        "start_counter does not align with /dld. The rest of "
                        "the calibration workflow does not require /tdc; only "
                        "DLTS-per-pulse / combinatorial recovery diagnostics "
                        "will be unavailable."
                    )
            progress.update(1)

            progress.set_postfix_str("normalizing loaded data")
            if source in ("raw", "raw_dld_only"):
                # Apply the standard raw -> processed pipeline so downstream analyses
                # see the same dataframe schema as a calibrated load. The
                # invalid-row filter is intentionally skipped here: raw-data
                # analysis is supposed to expose every recorded event, including
                # the empty (x=y=t=0) buffer entries — and a previously-merged
                # partial-hit recovery uses NaN on one axis, which the filter
                # leaves alone but which downstream code now also tolerates.
                # Keeping the row count untouched also means "raw load" and
                # "calibrated-bundle load" produce equivalent inputs to the
                # raw_data_analysis workflow.
                dld_df = data_tools.pyccapt_raw_to_processed(dld_df)
            progress.update(1)

        variables.data = dld_df
        variables.data_backup = dld_df.copy()
        variables.data_tdc = tdc_df
        variables.data_tdc_backup = tdc_df.copy() if tdc_df is not None else None
        # Populate the geometry / pulse fields that raw_data_analysis helpers
        # read via ``getattr(variables, ..., default)``. Without these set
        # explicitly, ``load_calibrated_h5`` users hit the 110 mm / 5000 ns /
        # 'voltage' defaults baked into the helpers. Honour any value already
        # on the object so a caller can override before invoking the loader.
        if not getattr(variables, "flight_path_length", None):
            variables.flight_path_length = 110.0
        if not getattr(variables, "pulse_mode", None):
            variables.pulse_mode = "voltage"
        # Estimate ``max_tof`` from the data so the combinatorial recovery
        # gate matches the file rather than the 5000 ns default. Add 5 % head
        # room so the histogram tail isn't clipped.
        if dld_df is not None and "t (ns)" in dld_df.columns and len(dld_df) > 0:
            _t_arr = dld_df["t (ns)"].to_numpy(dtype=float)
            _t_finite = _t_arr[np.isfinite(_t_arr)]
            if _t_finite.size:
                estimated_max_tof = float(np.max(_t_finite)) * 1.05
                if not getattr(variables, "max_tof", None):
                    variables.max_tof = estimated_max_tof
                if not getattr(variables, "max_tof_ns", None):
                    variables.max_tof_ns = estimated_max_tof

        # Range table: prefer /range in the same h5, then explicit range_path,
        # then fall back to <dataset>_range.h5 next to the file.
        progress.set_postfix_str("loading range table")
        range_df = None
        # Only PyCCAPT HDF5 files can carry an embedded /range group; trying
        # pd.read_hdf on a Cameca RHIT raises HDF5ExtError ("file signature
        # not found") which is NEITHER KeyError NOR ValueError, so it would
        # escape the except below. Skip the embedded read for non-HDF5 files.
        if suffix in {".h5", ".hdf5", ".hdf"}:
            try:
                range_df = pd.read_hdf(str(dataset_path_obj), key='range', mode='r')
            except (KeyError, ValueError):
                range_df = None

        if range_df is None and range_path:
            range_df = data_tools.read_range(range_path)
        if range_df is None:
            sibling = dataset_path_obj.with_name(f"{dataset_path_obj.stem}_range.h5")
            if sibling.is_file():
                range_df = data_tools.read_range(str(sibling))

        if range_df is not None:
            variables.range_data = range_df
            variables.range_data_backup = range_df.copy()
        progress.update(1)

        progress.set_postfix_str("synchronizing shared variables")
        variables.sync_from_data(update_backups=True)
        progress.update(1)
        progress.set_postfix_str("done")

        print(f"Loaded {source} dld rows: {len(dld_df)}")
        if tdc_df is not None:
            print(f"Loaded raw tdc rows:        {len(tdc_df)}")
        if range_df is not None:
            print(f"Loaded range rows:          {len(range_df)}")
        return dld_df, tdc_df, range_df
    finally:
        progress.close()
