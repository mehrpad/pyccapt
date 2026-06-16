from copy import copy

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import RectangleSelector, EllipseSelector
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from pyccapt.calibration.data_tools import data_tools, hdf5_schema, selectors_data
from pyccapt.calibration.path_utils import save_figure

EXTRACT_MODE_ALIASES = {
    "dld": "dld",
    "surface_concept": "dld",
    "tdc_sc": "tdc_sc",
    "roentdec": "tdc_ro",
    "tdc_ro": "tdc_ro",
}

def _normalize_extract_mode(extract_mode: str) -> str:
    """Normalize legacy and modern extract mode names."""
    mode = EXTRACT_MODE_ALIASES.get(extract_mode)
    if mode is None:
        supported = ", ".join(sorted(EXTRACT_MODE_ALIASES))
        raise ValueError(f"Unsupported extract_mode {extract_mode!r}. Supported modes: {supported}")
    return mode

EVENT_GROUP_ID_COLUMN = "event_group_id"
TDC_HAS_DLD_MATCH_COLUMN = "has_dld_match"


def _first_hdf5_column(hdf5_data, aliases, *, default_length=None):
    """Return the first present HDF5 dataset for a list of aliases."""
    for alias in aliases:
        if alias in hdf5_data:
            return hdf5_data[alias].to_numpy()
    if default_length is None:
        raise KeyError(f"None of the HDF5 aliases exist: {aliases}")
    return np.zeros((default_length, 1))


def _run_starts(values: np.ndarray) -> np.ndarray:
    """Index boundaries of consecutive equal-value runs.

    Returns an array of length n_runs+1 where slice [run_starts[i]:run_starts[i+1]]
    selects the i-th run.
    """
    if values.size == 0:
        return np.array([0], dtype=np.int64)
    return np.r_[0, np.where(np.diff(values) != 0)[0] + 1, values.size].astype(np.int64)

def build_event_group_mapping(
    dld_start_counter: np.ndarray,
    tdc_start_counter: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign shared event-group ids linking dld rows to their tdc rows.

    The start_counter wraps; values are not unique by themselves. But within each
    pulse trigger, all related tdc rows are consecutive in time order, and so are
    any dld rows for that pulse. A two-pointer sweep over the consecutive runs is
    enough to pair them: each tdc run either matches the next unmatched dld run
    (same counter value), or it is an "orphan" pulse that did not produce a
    reconstructible dld event.

    Returns
    -------
    dld_gid : int64 array, parallel to ``dld_start_counter``
        Group id assigned to each dld row.
    tdc_gid : int64 array, parallel to ``tdc_start_counter``
        Group id for matched tdc rows; ``-1`` for orphan tdc rows.
    tdc_has_match : bool array, parallel to ``tdc_start_counter``
        True iff the tdc row's pulse trigger has at least one dld row.
    """
    dld_sc = np.asarray(dld_start_counter).reshape(-1)
    tdc_sc = np.asarray(tdc_start_counter).reshape(-1)

    dld_gid = np.full(dld_sc.size, -1, dtype=np.int64)
    tdc_gid = np.full(tdc_sc.size, -1, dtype=np.int64)
    tdc_has_match = np.zeros(tdc_sc.size, dtype=bool)

    if dld_sc.size == 0 or tdc_sc.size == 0:
        return dld_gid, tdc_gid, tdc_has_match

    dld_runs = _run_starts(dld_sc)
    tdc_runs = _run_starts(tdc_sc)
    n_dld_runs = dld_runs.size - 1
    n_tdc_runs = tdc_runs.size - 1

    next_gid = 0
    j = 0
    for i in range(n_tdc_runs):
        t_start, t_end = tdc_runs[i], tdc_runs[i + 1]
        if j < n_dld_runs:
            d_start, d_end = dld_runs[j], dld_runs[j + 1]
            if tdc_sc[t_start] == dld_sc[d_start]:
                dld_gid[d_start:d_end] = next_gid
                tdc_gid[t_start:t_end] = next_gid
                tdc_has_match[t_start:t_end] = True
                next_gid += 1
                j += 1
                continue
        # Orphan tdc run: keep gid = -1, has_match = False.
    if j < n_dld_runs:
        # Acquisition crashes mid-run routinely produce a few dld events
        # whose corresponding tdc rows were never flushed (or were
        # already truncated by a partial-write recovery). Previously this
        # raised ValueError and the file became completely unloadable --
        # which is exactly the situation partial_recovery.py was added
        # for, but it can't help if the loader fails first. Demote to a
        # warning and assign the orphan dld rows a fresh NEGATIVE GID
        # range so they stay distinguishable; downstream filters can
        # drop them by predicate (gid < 0) if needed.
        unmatched_total = int(n_dld_runs - j)
        first_orphan_start = int(dld_runs[j])
        first_orphan_size = int(dld_runs[j + 1] - dld_runs[j])
        print(
            f"[build_event_group_mapping] WARNING: {unmatched_total} dld run(s) "
            f"have no matching tdc start_counter. First orphan starts at dld "
            f"index {first_orphan_start} (size {first_orphan_size}). Loading "
            f"with negative event_group_id for the orphan rows; check that "
            f"the dld and tdc datasets are from the same acquisition."
        )
        # Assign each orphan dld run a unique negative gid: -1, -2, -3, ...
        for orphan_idx, k in enumerate(range(j, n_dld_runs)):
            d_start, d_end = int(dld_runs[k]), int(dld_runs[k + 1])
            dld_gid[d_start:d_end] = -(orphan_idx + 1)
    return dld_gid, tdc_gid, tdc_has_match

def fetch_dataset_with_tdc(
    filename: str,
    tdc_extract_mode: str = "tdc_sc",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both the dld and tdc dataframes and add shared ``event_group_id``.

    The dld dataframe gets a single new integer column ``event_group_id`` that
    survives ``drop``/``iloc``/``reset_index``. The tdc dataframe gets the same
    column plus ``has_dld_match`` (bool) so orphan tdc rows can be preserved at
    save time even after dld filtering.
    """
    dld_df = fetch_dataset_from_dld_grp(filename, extract_mode="dld")
    if dld_df is None:
        raise FileNotFoundError(f"Failed to load dld group from {filename!r}")

    tdc_df = fetch_dataset_from_dld_grp(filename, extract_mode=tdc_extract_mode)
    if tdc_df is None:
        raise FileNotFoundError(f"Failed to load tdc group from {filename!r} with mode {tdc_extract_mode!r}")

    dld_gid, tdc_gid, tdc_has_match = build_event_group_mapping(
        dld_df["start_counter"].to_numpy(),
        tdc_df["start_counter"].to_numpy(),
    )
    dld_df[EVENT_GROUP_ID_COLUMN] = dld_gid
    tdc_df[EVENT_GROUP_ID_COLUMN] = tdc_gid
    tdc_df[TDC_HAS_DLD_MATCH_COLUMN] = tdc_has_match
    return dld_df, tdc_df

def filter_tdc_by_dld(dld_df: pd.DataFrame, tdc_df: pd.DataFrame) -> pd.DataFrame:
    """Return tdc rows whose dld counterpart still exists, plus all orphan rows.

    Rule: drop a tdc row only when its linked dld row was deleted. Orphan tdc
    rows (pulses that never produced a dld event) are always preserved.
    """
    if EVENT_GROUP_ID_COLUMN not in tdc_df.columns or TDC_HAS_DLD_MATCH_COLUMN not in tdc_df.columns:
        raise ValueError("tdc_df is missing the event_group_id/has_dld_match columns; load it via fetch_dataset_with_tdc().")
    if EVENT_GROUP_ID_COLUMN not in dld_df.columns:
        raise ValueError("dld_df is missing the event_group_id column; load it via fetch_dataset_with_tdc().")

    surviving_groups = pd.unique(dld_df[EVENT_GROUP_ID_COLUMN].to_numpy())
    keep = (~tdc_df[TDC_HAS_DLD_MATCH_COLUMN].to_numpy()) | (tdc_df[EVENT_GROUP_ID_COLUMN].isin(surviving_groups).to_numpy())
    return tdc_df.loc[keep].reset_index(drop=True).copy()

def fetch_dataset_from_dld_grp(filename: str, extract_mode='dld', *, lazy: bool = False):
    """
    Fetches dataset from HDF5 file.

    Args:
        filename: Path to the HDF5 file.
        extract_mode: Mode of extraction.
                    dld: Extracts data from dld group.
                    tdc_sc: Extracts data from tdc for Surface Consept.
                    tdc_ro: Extracts data from tdc for Roentdek detector.
        lazy: When ``True``, return a
            :class:`pyccapt.calibration.data_tools.lazy_io.LazyTable` view of
            the requested group (``/dld`` or ``/tdc``) instead of a
            materialized DataFrame. Use this on small-RAM machines: the
            file is opened with h5py and downstream analyses can iterate
            chunks via :meth:`LazyTable.iter_chunks`. The caller is
            responsible for closing the table (preferably via a ``with``
            block).

    Returns:
        DataFrame | LazyTable: Contains relevant information from the
        requested group, materialized or lazy depending on ``lazy``.
    """
    extract_mode = _normalize_extract_mode(extract_mode)
    if lazy:
        return _fetch_dataset_from_dld_grp_lazy(filename, extract_mode)
    flag_old_pyccpat_data = False
    if extract_mode == 'dld':
        try:
            hdf5_data = data_tools.read_hdf5(filename)
            if hdf5_data is None:
                raise FileNotFoundError
            dld_high_voltage = hdf5_data['dld/high_voltage'].to_numpy()
            if 'dld/pulse' in hdf5_data:
                dld_pulse_v = hdf5_data['dld/pulse'].to_numpy()
            elif 'dld/voltage_pulse' in hdf5_data:
                dld_pulse_v = hdf5_data['dld/voltage_pulse'].to_numpy()
                print('The data is loaded in dld mode')
            elif 'dld/pulse_voltage' in hdf5_data:
                dld_pulse_v = hdf5_data['dld/pulse_voltage'].to_numpy()
                flag_old_pyccpat_data = True
            else:
                dld_pulse_v = np.zeros(len(dld_high_voltage))
                dld_pulse_v = np.expand_dims(dld_pulse_v, axis=1)
            dld_pulse_l = _first_hdf5_column(
                hdf5_data,
                hdf5_schema.DLD_GROUP_ALIASES["pulse_l (pJ)"],
                default_length=len(dld_high_voltage),
            )
            if 'dld/start_counter' in hdf5_data:
                dld_start_counter = hdf5_data['dld/start_counter'].to_numpy()
            else:
                dld_start_counter = np.zeros(len(dld_high_voltage))
                dld_start_counter = np.expand_dims(dld_start_counter, axis=1)
            dld_t = hdf5_data['dld/t'].to_numpy()
            dld_x = hdf5_data['dld/x'].to_numpy()
            dld_y = hdf5_data['dld/y'].to_numpy()
            # Assemble via the schema helper, which normalises each column
            # to 1-D before stacking. This works whether the writer emitted
            # (N,) or (N, 1) datasets; the previous np.concatenate(axis=1)
            # raised AxisError if any column came back 1-D. Column order
            # MUST match hdf5_schema.DLD_COLUMNS / create_pandas_dataframe.
            dld_group_array = hdf5_schema.stack_columns(
                (dld_high_voltage, dld_pulse_v, dld_pulse_l, dld_start_counter, dld_t, dld_x, dld_y)
            )
            dld_group_storage = create_pandas_dataframe(
                dld_group_array, mode='dld', flag_old_pyccpat_data=flag_old_pyccpat_data
            )
            return dld_group_storage
        except KeyError as error:
            print(error)
            print("[*] Keys missing in the dataset")
        except FileNotFoundError as error:
            print(error)
            print("[*] HDF5 file not found")
    elif extract_mode in {'tdc_sc', 'tdc_ro'}:
        try:
            hdf5_data = data_tools.read_hdf5(filename)
            if hdf5_data is None:
                raise FileNotFoundError
            channel = hdf5_data['tdc/channel'].to_numpy()
            start_counter = hdf5_data['tdc/start_counter'].to_numpy()
            high_voltage = hdf5_data['tdc/high_voltage'].to_numpy()
            if 'tdc/pulse' in hdf5_data:
                voltage_pulse = hdf5_data['tdc/pulse'].to_numpy()
            elif 'tdc/voltage_pulse' in hdf5_data:
                voltage_pulse = hdf5_data['tdc/voltage_pulse'].to_numpy()
            else:
                raise KeyError('Neither tdc/pulse nor tdc/voltage_pulse exists in the dataset')
            if 'tdc/laser_pulse' in hdf5_data:
                laser_pulse = hdf5_data['tdc/laser_pulse'].to_numpy()
            else:
                laser_pulse = np.zeros((len(channel), 1))
            if laser_pulse.ndim == 1:
                laser_pulse = laser_pulse.reshape(-1, 1)
            time_data = hdf5_data['tdc/time_data'].to_numpy()

            # Shape-robust stack (see DLD branch). Column order MUST match
            # hdf5_schema.TDC_COLUMNS / create_pandas_dataframe('tdc_*').
            dld_group_array = hdf5_schema.stack_columns(
                (channel, start_counter, high_voltage, voltage_pulse, laser_pulse, time_data)
            )
            dld_group_storage = create_pandas_dataframe(dld_group_array, mode=extract_mode)
            return dld_group_storage
        except KeyError as error:
            print(error)
            print("[*] Keys missing in the dataset")
        except FileNotFoundError as error:
            print(error)
            print("[*] HDF5 file not found")
    return None

def _fetch_dataset_from_dld_grp_lazy(filename: str, extract_mode: str):
    """Return a memory-mapped :class:`LazyTable` view of one group.

    Opens ``filename`` once with h5py and exposes only the columns of the
    requested group (``/dld`` or ``/tdc``), so the heavyweight conversion to a
    PyCCAPT DataFrame is skipped entirely. The caller drives the streaming
    analyses via :meth:`LazyTable.iter_chunks`.
    """
    from pyccapt.calibration.data_tools import lazy_io

    group = 'dld' if extract_mode == 'dld' else 'tdc'
    raw = lazy_io.open_pyccapt_raw_hdf5(filename)
    # Filter to one group + rename to the calibration column names so callers
    # can use the same names whether they got a DataFrame or a LazyTable.
    rename_map = (
        {
            'dld/high_voltage': 'high_voltage (V)',
            'dld/pulse': 'pulse_v (V)',
            'dld/voltage_pulse': 'pulse_v (V)',
            'dld/pulse_voltage': 'pulse_v (V)',
            'dld/laser_pulse': 'pulse_l (pJ)',
            'dld/laser_intensity': 'pulse_l (pJ)',
            'dld/start_counter': 'start_counter',
            'dld/t': 't (ns)',
            'dld/x': 'x_det (cm)',
            'dld/y': 'y_det (cm)',
        }
        if group == 'dld'
        else {
            'tdc/channel': 'channel',
            'tdc/start_counter': 'start_counter',
            'tdc/high_voltage': 'high_voltage (V)',
            'tdc/pulse': 'pulse_v (V)',
            'tdc/voltage_pulse': 'pulse_v (V)',
            'tdc/laser_pulse': 'pulse_l (pJ)',
            'tdc/time_data': 'time_data',
        }
    )
    selected: dict = {}
    for raw_name, friendly in rename_map.items():
        if raw_name in raw.columns and friendly not in selected:
            selected[friendly] = raw[raw_name]

    if not selected:
        raw.close()
        raise ValueError(f"No /{group}/* columns found in {filename}. Is this a pyccapt-raw HDF5?")

    from pyccapt.calibration.data_tools.lazy_io import LazyTable

    return LazyTable(
        selected,
        source_path=raw.source_path,
        close=raw.close,
    )

def concatenate_dataframes_of_dld_grp(dataframe_list: list) -> pd.DataFrame:
    """
    Concatenates dataframes into a single dataframe.

    Args:
        dataframe_list: List of different information from dld group.

    Returns:
        DataFrame: Single concatenated dataframe containing all relevant information.
    """
    dld_master_dataframe = pd.concat(dataframe_list, axis=1)
    return dld_master_dataframe

def plot_crop_experiment_history(
    data: pd.DataFrame,
    variables,
    max_tof,
    frac=1.0,
    bins=(1200, 800),
    figure_size=(8, 3),
    draw_rect=False,
    data_crop=True,
    pulse_plot=False,
    dc_plot=True,
    pulse_mode='voltage',
    save=True,
    figname='',
):
    """
    Plots the experiment history.

    Args:
        dldGroupStorage: DataFrame containing info about the dld group.
        max_tof: The maximum tof to be plotted.
        frac: Fraction of the data to be plotted.
        figure_size: The size of the figure.
        data_crop: Flag to control if only the plot should be shown or cropping functionality should be enabled.
        draw_rect: Flag to draw  a rectangle over the selected area.
        pulse: Flag to choose whether to plot pulse.
        pulse_mode: Flag to choose whether to plot pulse voltage or pulse.
        dc_plot: Flag to choose whether to plot dc voltage.
        save: Flag to choose whether to save the plot or not.
        figname: Name of the figure to be saved.

    Returns:
        None.
    """

    if max_tof > 0:
        mask_1 = data['t (ns)'].to_numpy() > max_tof
        data.drop(np.where(mask_1)[0], inplace=True)
        data.reset_index(inplace=True, drop=True)
    if frac < 1:
        # set axis limits based on fraction of data
        dldGroupStorage = data.sample(frac=frac, random_state=42)
        dldGroupStorage.sort_index(inplace=True)
    else:
        dldGroupStorage = data

    fig1, ax1 = plt.subplots(figsize=figure_size, constrained_layout=True)

    # extract tof and high voltage from the data frame
    tof = dldGroupStorage['t (ns)'].to_numpy()
    high_voltage = data['high_voltage (V)'].to_numpy()
    high_voltage = high_voltage / 1000  # change to kV
    if pulse_mode == 'laser':
        pulse = dldGroupStorage['pulse_l (pJ)'].to_numpy()
    elif pulse_mode == 'voltage':
        pulse = dldGroupStorage['pulse_v (V)'].to_numpy()

    xaxis = np.arange(len(tof))

    # Check if the bin is a tuple
    if isinstance(bins, tuple):
        pass
    else:
        x_edges = np.arange(xaxis.min(), xaxis.max() + bins, bins)
        y_edges = np.arange(tof.min(), tof.max() + bins, bins)
        bins = [x_edges, y_edges]

    heatmap, xedges, yedges = np.histogram2d(xaxis, tof, bins=bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # Set x-axis label
    ax1.set_xlabel("Hit Sequence Number", fontsize=10)
    # Set y-axis label
    ax1.set_ylabel("Time of Flight [ns]", fontsize=10)
    img = plt.imshow(heatmap.T, extent=extent, origin='lower', aspect="auto")
    cmap = copy(plt.cm.plasma)
    cmap.set_bad(cmap(0))
    pcm = ax1.pcolormesh(xedges, yedges, heatmap.T, cmap=cmap, norm=colors.LogNorm(), rasterized=True)

    plt.ticklabel_format(style='sci', axis='x', scilimits=(6, 6))

    if dc_plot:
        ax2 = ax1.twinx()
        if not pulse_plot:
            ax2.spines.right.set_position(("axes", 1.13))
        else:
            ax2.spines.right.set_position(("axes", 1.29))
        # Plot high voltage curve
        xaxis2 = np.arange(len(high_voltage))
        (dc_curve,) = ax2.plot(xaxis2, high_voltage, color='red', linewidth=2)
        ax2.set_ylabel("DC Voltage [kV]", color="red", fontsize=10)
        ax2.set_ylim([min(high_voltage), max(high_voltage) + 0.5])
        ax2.spines['right'].set_color('red')  # Set Y-axis color to red
        ax2.yaxis.label.set_color('red')  # Set Y-axis label color to red
        ax2.tick_params(axis='y', colors='red')  # Set Y-axis tick labels color to red

    if pulse_plot:
        ax3 = ax1.twinx()
        ax3.spines.right.set_position(("axes", 1.13))
        if pulse_mode == 'laser':
            (pulse_curve,) = ax3.plot(xaxis, pulse, color='fuchsia', linewidth=2)
            ax3.set_ylabel("Pulse Energy [$pJ$]", color="fuchsia", fontsize=10)
            range = max(pulse) - min(pulse)
            ax3.set_ylim([min(pulse) - range * 0.1, max(pulse) + range * 0.1])
            ax3.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        elif pulse_mode == 'voltage':
            pulse = pulse / 1000
            (pulse_curve,) = ax3.plot(xaxis, pulse, color='fuchsia', linewidth=2)
            ax3.set_ylabel("Pulse Voltage [kV]", color="fuchsia", fontsize=10)
            ax3.set_ylim([min(pulse), max(pulse) + 0.5])
        ax3.spines['right'].set_color('fuchsia')  # Set Y-axis color to red
        ax3.yaxis.label.set_color('fuchsia')  # Set Y-axis label color to red
        ax3.tick_params(axis='y', colors='fuchsia')  # Set Y-axis tick labels color to red

    # if pulse_plot:
    #     pulse_curve.set_visible(False)
    # if dc_plot:
    #     dc_curve.set_visible(False)

    if dc_plot or pulse_plot:
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="2.5%", pad="0%")
        cbar = fig1.colorbar(pcm, cax=cax)
    else:
        cbar = fig1.colorbar(pcm, ax=ax1, pad=0)

    cbar.set_label('Event Counts', fontsize=10)

    if data_crop:
        if dc_plot and pulse_plot:
            rectangle_box_selector(ax3, variables)
        elif dc_plot and not pulse_plot:
            rectangle_box_selector(ax2, variables)
        elif not dc_plot and pulse_plot:
            rectangle_box_selector(ax3, variables)
        elif not pulse_plot and not dc_plot:
            rectangle_box_selector(ax1, variables)
        plt.connect('key_press_event', selectors_data.toggle_selector(variables))
    if draw_rect:
        left, bottom, width, height = (variables.selected_x1, 0, variables.selected_x2 - variables.selected_x1, np.max(tof))
        rect = Rectangle((left, bottom), width, height, fill=True, alpha=0.3, color="r", linewidth=5)
        ax1.add_patch(rect)

    if save:
        save_figure(
            fig1,
            directory=variables.result_path,
            stem=figname or "experiment_history",
            formats=("png", "pdf"),
            dpi=600,
            bbox_inches='tight',
        )

    plt.show()

def plot_crop_fdm(
    x,
    y,
    bins=(256, 256),
    frac=1.0,
    axis_mode='normal',
    figure_size=(5, 4),
    variables=None,
    range_sequence=[],
    range_mc=[],
    range_detx=[],
    range_dety=[],
    range_x=[],
    range_y=[],
    range_z=[],
    range_vol=[],
    data_crop=False,
    draw_circle=False,
    mode_selector='circle',
    save=False,
    figname='FDM',
):
    """
    Plot and crop the FDM with the option to select a region of interest.

    Args:
        x: x-axis data
        y: y-axis data
        bins: Number of bins for the histogram as a tuple or a single float as the bin size
        frac: Fraction of the data to be plotted
        axis_mode: Flag to choose whether to plot axis or scalebar: 'normal' or 'scalebar'
        variables: Variables object
        range_sequence: Range of sequence
        range_mc: Range of mc
        range_detx: Range of detx
        range_dety: Range of dety
        range_x: Range of x-axis
        range_y: Range of y-axis
        range_z: Range of z-axis
        range_vol: Range of voltage
        figure_size: Size of the plot
        draw_circle: Flag to enable circular region of interest selection
        mode_selector: Mode of selection (circle or ellipse)
        save: Flag to choose whether to save the plot or not
        data_crop: Flag to control whether only the plot is shown or cropping functionality is enabled
        figname: Name of the figure to be saved

    Returns:
        None
    """
    # When the dataset has been merged with recovered partial-hit rows
    # (``merge_partial_tdc=True``), partials carry NaN on x_det or y_det
    # because only one delay-line axis was reconstructable. NaNs cannot be
    # binned into a 2-D histogram (matplotlib silently drops them, but the
    # length-mismatch with ``mask`` below would then misalign rows). Filter
    # them up front and emit a one-line notice so the user knows the FDM
    # was built from the position-capable subset only.
    x = np.asarray(x)
    y = np.asarray(y)
    nan_mask = np.isnan(x) | np.isnan(y)
    if nan_mask.any():
        n_dropped = int(nan_mask.sum())
        print(
            f'[plot_crop_fdm] Skipping {n_dropped} partial-recovered rows '
            'with NaN on x_det or y_det (the 2-D histogram requires both axes).'
        )
        x = x[~nan_mask]
        y = y[~nan_mask]

    if range_sequence or range_mc or range_detx or range_dety or range_x or range_y or range_z:
        if range_sequence:
            mask_sequence = np.zeros(len(x), dtype=bool)
            mask_sequence[range_sequence[0] : range_sequence[1]] = True
        else:
            mask_sequence = np.ones(len(x), dtype=bool)
        if range_detx and range_dety:
            mask_det_x = (variables.dld_x_det < range_detx[1]) & (variables.dld_x_det > range_detx[0])
            mask_det_y = (variables.dld_y_det < range_dety[1]) & (variables.dld_y_det > range_dety[0])
            mask_det = mask_det_x & mask_det_y
        else:
            mask_det = np.ones(len(variables.dld_x_det), dtype=bool)
        if range_mc:
            mask_mc = (variables.mc <= range_mc[1]) & (variables.mc >= range_mc[0])
        else:
            mask_mc = np.ones(len(variables.mc), dtype=bool)
        if range_x and range_y and range_z:
            mask_x = (variables.x < range_x[1]) & (variables.x > range_x[0])
            mask_y = (variables.y < range_y[1]) & (variables.y > range_y[0])
            mask_z = (variables.z < range_z[1]) & (variables.z > range_z[0])
            mask_3d = mask_x & mask_y & mask_z
        else:
            mask_3d = np.ones(len(variables.x), dtype=bool)
        if range_vol:
            mask_vol = (variables.dld_high_voltage < range_vol[1]) & (variables.dld_high_voltage > range_vol[0])
        else:
            mask_vol = np.ones(len(variables.dld_high_voltage), dtype=bool)
        mask = mask_sequence & mask_det & mask_mc & mask_3d & mask_vol
        variables.mask = mask
        print('The number of data mc:', len(mask_mc[mask_mc == True]))
        print('The number of data det:', len(mask_det[mask_det == True]))
        print('The number of data 3d:', len(mask_3d[mask_3d == True]))
        print('The number of data after cropping:', len(mask[mask == True]))
    else:
        mask = np.ones(len(x), dtype=bool)

    x = x[mask]
    y = y[mask]

    if frac < 1:
        # set axis limits based on fraction of x and y data baded on fraction
        mask_fraq = np.random.choice(len(x), int(len(x) * frac), replace=False)
        x_t = np.copy(x)
        y_t = np.copy(y)
        x = x[mask_fraq]
        y = y[mask_fraq]

    fig1, ax1 = plt.subplots(figsize=figure_size, constrained_layout=True)

    # Check if the bin is a list
    if isinstance(bins, list):
        if len(bins) == 1:
            x_edges = np.arange(x.min(), x.max() + bins, bins)
            y_edges = np.arange(y.min(), y.max() + bins, bins)
            bins = [x_edges, y_edges]

    FDM, xedges, yedges = np.histogram2d(x, y, bins=bins)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    cmap = copy(plt.cm.plasma)
    cmap.set_bad(cmap(0))
    pcm = ax1.pcolormesh(xedges, yedges, FDM.T, cmap=cmap, norm=colors.LogNorm(), rasterized=True)
    cbar = fig1.colorbar(pcm, ax=ax1, pad=0)
    cbar.set_label('Event Counts', fontsize=10)

    if frac < 1:
        # extract tof
        x_lim = x_t
        y_lim = y_t

        ax1.set_xlim([min(x_lim), max(x_lim)])
        ax1.set_ylim([min(y_lim), max(y_lim)])

    if variables is not None:
        if data_crop:
            elliptical_shape_selector(ax1, fig1, variables, mode=mode_selector)
        if draw_circle:
            print('x:', variables.selected_x_fdm, 'y:', variables.selected_y_fdm, 'roi:', variables.roi_fdm)
            circ = Circle(
                (variables.selected_x_fdm, variables.selected_y_fdm),
                variables.roi_fdm,
                fill=True,
                alpha=0.3,
                color='green',
                linewidth=5,
            )
            ax1.add_patch(circ)
    if axis_mode == 'scalebar':
        fontprops = fm.FontProperties(size=10)
        scalebar = AnchoredSizeBar(
            ax1.transData,
            1,
            '1 cm',
            'lower left',
            pad=0.1,
            color='white',
            frameon=False,
            size_vertical=0.1,
            fontproperties=fontprops,
        )

        ax1.add_artist(scalebar)
        plt.axis('off')  # Turn off both x and y axes
    elif axis_mode == 'normal':
        ax1.set_xlabel(r"$X_{det} (cm)$", fontsize=10)
        ax1.set_ylabel(r"$Y_{det} (cm)$", fontsize=10)

    if save and variables is not None:
        save_figure(fig1, directory=variables.result_path, stem=figname or "FDM", formats=("png", "pdf"), dpi=600)
    plt.show()

def _legacy_extract_xy(data) -> tuple[np.ndarray, np.ndarray]:
    """Extract detector x/y from legacy ndarray inputs."""
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D array for legacy FDM plotting")
    if arr.shape[1] >= 7:
        return arr[:, 5], arr[:, 6]
    if arr.shape[1] >= 2:
        return arr[:, 0], arr[:, 1]
    raise ValueError("Expected at least 2 columns for legacy FDM plotting")

def plot_crop_FDM(ax, fig, data, bins=(256, 256), save_name=None):
    """
    Backward-compatible wrapper for legacy camelCase API.

    Notes:
        The `ax` and `fig` arguments are kept for compatibility with older call
        sites and are not used directly because the modern API creates its own figure.
    """
    x, y = _legacy_extract_xy(data)
    # Lightweight adapter for older UI callbacks.
    variables = type(
        "LegacyVariables",
        (),
        {"selected_x_fdm": 0, "selected_y_fdm": 0, "roi_fdm": 0, "result_path": "."},
    )()
    plot_crop_fdm(
        x,
        y,
        bins=bins,
        variables=variables,
        data_crop=True,
        mode_selector="circle",
        save=bool(save_name),
        figname=save_name or "FDM",
    )

def plot_FDM(ax, fig, data, bins=(256, 256), save_name=None):
    """Backward-compatible wrapper for legacy `plot_FDM` API."""
    x, y = _legacy_extract_xy(data)
    fdm, xedges, yedges = np.histogram2d(x, y, bins=bins)
    plt.imshow(fdm.T, origin="lower", aspect="auto", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    if save_name:
        plt.savefig(f"{save_name}.png", format="png", dpi=600)
    plt.show()

def rectangle_box_selector(axisObject, variables):
    """
    Enable the creation of a rectangular box to select the region of interest.

    Args:
        axisObject: Object to create the rectangular box
        variables: Variables object

    Returns:
        None
    """
    selectors_data.toggle_selector.RS = RectangleSelector(
        axisObject,
        lambda eclick, erelease: selectors_data.line_select_callback(eclick, erelease, variables),
        useblit=True,
        button=[1, 3],
        minspanx=1,
        minspany=1,
        spancoords='pixels',
        interactive=True,
    )

def crop_dataset(dld_master_dataframe, variables):
    """
    Crop the dataset based on the selected region of interest.

    Args:
        dld_master_dataframe: Concatenated dataset
        variables: Variables object

    Returns:
        data_crop: Cropped dataset
    """
    if dld_master_dataframe is None or dld_master_dataframe.empty:
        raise ValueError('Dataset is empty, temporal crop cannot be applied')

    left = int(np.floor(min(variables.selected_x1, variables.selected_x2)))
    right = int(np.ceil(max(variables.selected_x1, variables.selected_x2)))
    last_index = len(dld_master_dataframe) - 1

    if left < 0 or right < 0:
        raise ValueError('Crop indices must be non-negative')
    if left > last_index or right > last_index:
        raise ValueError(f'Crop indices must stay within [0, {last_index}]')
    if right < left:
        raise ValueError('End index must be greater than or equal to start index')

    data_crop = dld_master_dataframe.iloc[left : right + 1, :].copy()
    data_crop.reset_index(inplace=True, drop=True)
    return data_crop

def elliptical_shape_selector(axisObject, figureObject, variables, mode='circle'):
    """
    Enable the creation of an elliptical box to select the region of interest.

    Args:
        axisObject: Object to create the axis of the plot
        figureObject: Object to create the figure
        variables: Variables object
        mode: Mode of selection (circle or ellipse)

    Returns:
        None
    """
    if mode == 'circle':
        selectors_data.toggle_selector.ES = selectors_data.CircleSelector(
            axisObject,
            lambda eclick, erelease: selectors_data.onselect(eclick, erelease, variables),
            useblit=True,
            button=[1, 3],
            minspanx=1,
            minspany=1,
            spancoords='pixels',
            interactive=True,
        )
    elif mode == 'ellipse':
        selectors_data.toggle_selector.ES = EllipseSelector(
            axisObject,
            lambda eclick, erelease: selectors_data.onselect(eclick, erelease, variables),
            useblit=True,
            button=[1, 3],
            minspanx=1,
            minspany=1,
            spancoords='pixels',
            interactive=True,
        )

    figureObject.canvas.mpl_connect('key_press_event', selectors_data.toggle_selector)

def crop_data_after_selection(data_crop, variables):
    """
    Crop the dataset after the region of interest has been selected.

    Args:
        data_crop: Original dataset to be cropped
        variables: Variables object

    Returns:
        data_crop: Cropped dataset
    """
    if data_crop is None or data_crop.empty:
        raise ValueError('Dataset is empty, spatial crop cannot be applied')

    center_x = float(variables.selected_x_fdm)
    center_y = float(variables.selected_y_fdm)
    radius = float(variables.roi_fdm)

    if not np.isfinite(center_x) or not np.isfinite(center_y):
        raise ValueError('Crop center must be numeric')
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Crop radius must be greater than 0')

    x = data_crop['x_det (cm)'].to_numpy()
    y = data_crop['y_det (cm)'].to_numpy()

    # Partial-recovered rows (merge_partial_tdc=True) carry NaN on the
    # delay-line axis that could not be reconstructed. Their detector
    # position is *unknown*, not "outside the ROI" -- silently dropping
    # them would throw away ToF / mass / pulse info that is still valid.
    # Build the in/out mask only from rows where both axes are finite,
    # and force partials to pass through unconditionally so the spatial
    # crop is non-destructive to them.
    finite_mask = np.isfinite(x) & np.isfinite(y)
    partial_mask = ~finite_mask
    n_partials = int(partial_mask.sum())

    x_finite = x[finite_mask]
    y_finite = y[finite_mask]
    if x_finite.size == 0:
        raise ValueError('Spatial crop has no rows with both detector axes (only partials present)')

    x_min, x_max = float(np.min(x_finite)), float(np.max(x_finite))
    y_min, y_max = float(np.min(y_finite)), float(np.max(y_finite))
    if center_x < x_min or center_x > x_max or center_y < y_min or center_y > y_max:
        raise ValueError(
            f'Crop center must stay inside detector bounds: x in [{x_min:.4f}, {x_max:.4f}], y in [{y_min:.4f}, {y_max:.4f}]'
        )

    detector_dist = np.full(len(data_crop), np.inf, dtype=np.float64)
    detector_dist[finite_mask] = np.sqrt((x_finite - center_x) ** 2 + (y_finite - center_y) ** 2)
    mask_fdm = (detector_dist <= radius) | partial_mask  # keep partials always
    if not np.any(mask_fdm & finite_mask):
        raise ValueError('Spatial crop does not contain any position-capable detector hits')

    if n_partials > 0:
        print(
            f'[crop_data_after_selection] Keeping {n_partials} partial-recovered rows '
            '(NaN x_det/y_det) unconditionally; spatial-only analyses must still filter them.'
        )

    cropped = data_crop.loc[mask_fdm].copy()
    cropped.reset_index(inplace=True, drop=True)
    return cropped

def create_pandas_dataframe(data_crop, mode='dld', flag_old_pyccpat_data=False):
    """
    Create a pandas dataframe from the cropped data.

    Args:
        data_crop: Cropped dataset
        mode: Mode of extraction
                dld: Extracts data from dld group
                tdc_sc: Extracts data from tdc for Surface Consept
                tdc_ro: Extracts data from tdc for RoentDek detector
        flag_old_pyccpat_data: Flag to determine if data is already convert from bin to ns and mm (old pyccapt datas)

    Returns:
        hdf_dataframe: Dataframe to be inserted in the HDF file
    """
    mode = _normalize_extract_mode(mode)
    if mode == 'dld':
        if flag_old_pyccpat_data:
            TOFFACTOR = 27.432 / (1000 * 4)  # 27.432 ps/bin, tof in ns, data is TDC time sum
            DETBINS = 4900
            BINNINGFAC = 2
            XYFACTOR = 80 / DETBINS * BINNINGFAC  # XXX mm/bin
            XYBINSHIFT = DETBINS / BINNINGFAC / 2  # to center detector

            data_crop[:, 4] = data_crop[:, 4] * TOFFACTOR  # in ns
            data_crop[:, 5] = ((data_crop[:, 5] - XYBINSHIFT) * XYFACTOR) * 0.1  # from mm to in cm by dividing by 10
            data_crop[:, 6] = ((data_crop[:, 6] - XYBINSHIFT) * XYFACTOR) * 0.1  # from mm to in cm by dividing by 10

        hdf_dataframe = pd.DataFrame(
            data=data_crop,
            columns=['high_voltage (V)', 'pulse_v (V)', 'pulse_l (pJ)', 'start_counter', 't (ns)', 'x_det (cm)', 'y_det (cm)'],
        )

        for column in ['high_voltage (V)', 'pulse_v (V)', 'pulse_l (pJ)', 't (ns)', 'x_det (cm)', 'y_det (cm)']:
            hdf_dataframe[column] = hdf_dataframe[column].astype('float64')
        hdf_dataframe['start_counter'] = hdf_dataframe['start_counter'].astype('uint64')
    elif mode == 'tdc_sc':
        hdf_dataframe = pd.DataFrame(
            data=data_crop,
            columns=['channel', 'start_counter', 'high_voltage (V)', 'pulse_v (V)', 'pulse_l (pJ)', 'time_data'],
        )

        for column in ['high_voltage (V)', 'pulse_v (V)', 'pulse_l (pJ)']:
            hdf_dataframe[column] = hdf_dataframe[column].astype('float64')
        hdf_dataframe['channel'] = hdf_dataframe['channel'].astype('uint32')
        hdf_dataframe['start_counter'] = hdf_dataframe['start_counter'].astype('uint64')
        hdf_dataframe['time_data'] = hdf_dataframe['time_data'].astype('uint64')
    elif mode == 'tdc_ro':
        hdf_dataframe = pd.DataFrame(
            data=data_crop,
            columns=['channel', 'start_counter', 'high_voltage (V)', 'pulse_v (V)', 'pulse_l (pJ)', 'time_data'],
        )

        for column in ['high_voltage (V)', 'pulse_v (V)', 'pulse_l (pJ)']:
            hdf_dataframe[column] = hdf_dataframe[column].astype('float64')
        hdf_dataframe['channel'] = hdf_dataframe['channel'].astype('uint32')
        hdf_dataframe['start_counter'] = hdf_dataframe['start_counter'].astype('uint64')
        hdf_dataframe['time_data'] = hdf_dataframe['time_data'].astype('uint64')
    else:
        raise ValueError(f"Unsupported mode: {mode!r}")

    return hdf_dataframe

def calculate_ppi_and_ipp(data, max_start_counter):
    """
    Calculate pulses since the last event pulse and ions per pulse.

    Args:
        data (dict): A dictionary containing the 'start_counter' data.
        max_start_counter (int): The maximum start counter value.

    Returns:
        tuple: A tuple containing two numpy arrays: delta_p and multi.

    Raises:
        IndexError: If the length of counter is less than 1.

    """

    counter = data['start_counter'].to_numpy()
    delta_p = np.zeros(len(counter))
    multi = np.zeros(len(counter))

    multi_hit_count = 1

    total_iterations = len(counter)
    if total_iterations == 0:
        return delta_p, multi
    progress_step = max(1, total_iterations // 5)

    for i, current_counter in enumerate(counter):
        if i == 0:
            delta_p[i] = 0
            previous_counter = current_counter
        else:
            sc = current_counter - previous_counter
            if sc < 0:
                sc_a = max_start_counter - previous_counter
                sc_b = current_counter
                sc = sc_a + sc_b

            delta_p[i] = sc

            if current_counter == previous_counter:
                multi_hit_count += 1
            else:
                for j in range(multi_hit_count):
                    if i + j <= len(counter):
                        multi[i - j - 1] = multi_hit_count

                multi_hit_count = 1
                previous_counter = current_counter
        # for the last event
        if i == len(counter) - 1:
            multi[i] = multi_hit_count

        # Print progress at each ~20% interval
        if i % progress_step == 0:
            progress_percent = int((i / total_iterations) * 100)
            print(f"Progress: {progress_percent}% complete")

    return delta_p, multi
