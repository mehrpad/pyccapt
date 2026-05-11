"""Data loading and persistence helpers for calibration workflows."""

from __future__ import annotations

import os
from pathlib import Path
import warnings
import ast
import re

import h5py
import numpy as np
import pandas as pd
import scipy.io

from pyccapt.calibration.data_tools import ato_tools
from pyccapt.calibration.leap_tools import ccapt_tools, leap_tools
from pyccapt.calibration.mc import tof_tools


def _as_path(path_value: str | Path) -> Path:
    """Return `path_value` as a normalized filesystem path."""
    return Path(path_value).expanduser()


def _resolve_variable_output_file(variables, *, filename: str, data_directory: bool) -> str:
    """Build an output file path from shared variables with compatibility fallbacks."""
    resolver_name = "resolve_result_data_file" if data_directory else "resolve_result_file"
    resolver = getattr(variables, resolver_name, None)
    if callable(resolver):
        return resolver(filename)

    base_attr = "result_data_path" if data_directory else "result_path"
    base_path = str(getattr(variables, base_attr, "")).strip()
    if base_path:
        return str(_as_path(base_path) / filename)
    return filename


def read_hdf5(filename: str | Path, *, lazy: bool = False):
    """
    Read non-pandas HDF5 content into a dictionary of dataframes.

    Parameters:
        filename: HDF5 file path.
        lazy: When ``True``, return a
            :class:`pyccapt.calibration.data_tools.lazy_io.LazyTable` view that
            reads each ``/group/dataset`` on demand. Use this on small-RAM
            machines: a 2-3 GB pyccapt-raw file opens with essentially zero
            resident memory and downstream analyses can call
            :meth:`LazyTable.iter_chunks` to stream the rows. The caller is
            responsible for closing the table (preferably with a
            ``with`` block).

    Returns:
        dict[str, pd.DataFrame] | LazyTable | None
    """
    file_path = _as_path(filename)
    if lazy:
        from pyccapt.calibration.data_tools import lazy_io
        try:
            return lazy_io.open_pyccapt_raw_hdf5(file_path)
        except FileNotFoundError:
            print("[*] HDF5 File could not be found")
        except ValueError as exc:
            # Group structure incompatible with the lazy reader's contract.
            print(f"[*] Lazy HDF5 open failed: {exc}")
        return None

    try:
        dataframe_storage: dict[str, pd.DataFrame] = {}
        group_dict: dict[str, list[str]] = {}

        with h5py.File(file_path, "r") as hdf:
            groups = list(hdf.keys())
            for item in groups:
                group_dict[item] = list(hdf[item].keys())

            for key, value in group_dict.items():
                for item in value:
                    dataset = pd.DataFrame(np.array(hdf[f"{key}/{item}"]), columns=["values"])
                    dataframe_storage[f"{key}/{item}"] = dataset

        return dataframe_storage
    except FileNotFoundError:
        print("[*] HDF5 File could not be found")
    except IndexError:
        print("[*] No Group keys could be found in HDF5 File")
    return None


def read_range(filename: str | Path) -> pd.DataFrame:
    """Read saved range definitions from `.h5`, `.rrng`, or legacy `.rng` files."""
    file_path = _as_path(filename)
    try:
        suffix = file_path.suffix.lower()
        if suffix == ".h5":
            return _normalize_range_dataframe(pd.read_hdf(file_path, mode="r"))
        if suffix == ".rrng":
            return _normalize_range_dataframe(leap_tools.read_rrng(str(file_path)))
        if suffix == ".rng":
            return _normalize_range_dataframe(leap_tools.read_rng(str(file_path)))
        raise ValueError(f"Unsupported range file extension: {file_path.suffix!r}")
    except FileNotFoundError as error:
        print("[*] Range file could not be found")
        print(error)
        raise


def _normalize_range_dataframe(range_df: pd.DataFrame) -> pd.DataFrame:
    """Backfill legacy range schema fields required by modern visualization helpers."""
    if not isinstance(range_df, pd.DataFrame):
        return range_df

    def _coerce_list(value):
        if isinstance(value, (list, tuple, np.ndarray)):
            return list(value)
        text = str(value).strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple, np.ndarray)):
                    return [str(item).strip() for item in parsed]
            except (ValueError, SyntaxError):
                pass
        return [text]

    def _ion_to_neutral_formula(ion_label):
        text = str(ion_label).strip()
        if not text:
            return ""
        text = text.replace("$", "")
        text = re.sub(r"_\{([^}]*)\}", r"\1", text)
        text = re.sub(r"\^\{[^}]*\}", "", text)
        text = text.replace("{", "").replace("}", "")
        text = re.sub(r"\s*[+-]+\s*$", "", text)
        text = re.sub(r"\s*\d*[+-]+\s*$", "", text)
        return text.strip()

    def _derive_neutral_name(row):
        elements = _coerce_list(row.get("element", []))
        complexes = _coerce_list(row.get("complex", []))
        if elements:
            formula_parts = []
            for index, element in enumerate(elements):
                stoich = 1
                if index < len(complexes):
                    try:
                        stoich = int(float(complexes[index]))
                    except (TypeError, ValueError):
                        stoich = 1
                suffix = "" if stoich <= 1 else str(stoich)
                formula_parts.append(f"{element}{suffix}")
            formula = "".join(formula_parts).strip()
            if formula:
                return formula
        if "ion" in row:
            neutral = _ion_to_neutral_formula(row.get("ion", ""))
            if neutral:
                return neutral
        if "ion_name" in row:
            neutral = _ion_to_neutral_formula(row.get("ion_name", ""))
            if neutral:
                return neutral
        return ""

    normalized = range_df.copy()
    derived_names = normalized.apply(_derive_neutral_name, axis=1)
    if "name" not in normalized.columns:
        normalized["name"] = derived_names
    else:
        raw_names = normalized["name"].astype(str).replace("", np.nan)
        looks_charged = raw_names.fillna("").str.contains(r"[\^$]|\+|-", regex=True)
        normalized["name"] = raw_names
        normalized.loc[looks_charged, "name"] = derived_names[looks_charged]
        normalized["name"] = normalized["name"].fillna(derived_names)

    normalized["name"] = normalized["name"].astype(str).replace("", np.nan)
    normalized.loc[normalized["name"].str.lower() == "nan", "name"] = np.nan
    placeholder_names = pd.Series(
        [f"range_{index}" for index in range(len(normalized))],
        index=normalized.index,
    )
    normalized["name"] = normalized["name"].fillna(placeholder_names)
    return normalized


def read_mat_files(filename: str | Path):
    """Read a `.mat` file and return its dictionary contents."""
    file_path = _as_path(filename)
    try:
        return scipy.io.loadmat(file_path)
    except FileNotFoundError:
        print("[*] Mat File could not be found")
    return None


def convert_mat_to_df(hdf5_file_response: dict) -> pd.DataFrame:
    """Convert loaded `.mat` content to a dataframe and persist it as HDF5."""
    if hdf5_file_response is None or "None" not in hdf5_file_response:
        raise ValueError("Expected key 'None' in .mat content")
    pd_dataframe = pd.DataFrame(hdf5_file_response["None"])
    store_df_to_hdf(pd_dataframe, "dataframe/isotope", "isotopeTable.h5")
    return pd_dataframe


def store_df_to_hdf(dataframe, key, filename, *, format: str = "fixed"):
    """
    Store dataframe to HDF5.

    Args:
        dataframe: Pandas DataFrame to serialize.
        key: HDF5 key (e.g. ``"df"``).
        filename: Destination ``.h5`` path.
        format: Pytables format. ``"fixed"`` (default, fast full-file reads,
            single-write) or ``"table"`` (slightly slower but supports
            ``pd.read_hdf(..., iterator=True, chunksize=...)`` and is more
            permissive about mixed-dtype columns). Use ``"table"`` for big raw
            outputs you'll later want to stream back without loading the
            whole file into RAM.

    Supports both modern argument order ``(dataframe, key, filename)`` and the
    legacy order ``(filename, dataframe, key)`` for backwards compatibility.
    """
    if isinstance(dataframe, (str, Path)):
        # Legacy order: (filename, dataframe, key)
        filename, dataframe, key = dataframe, key, filename

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas.DataFrame")

    file_path = _as_path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_hdf(file_path, key=str(key), mode="w", format=format)


def store_df_to_csv(data: pd.DataFrame, path: str | Path) -> None:
    """Store dataframe to CSV with project-default encoding and separator."""
    file_path = _as_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(file_path, encoding="utf-8", index=False, sep=";")


def remove_invalid_data(dld_group_storage: pd.DataFrame, max_tof: float) -> pd.DataFrame:
    """Remove invalid TOF and detector rows and return the cleaned dataframe."""
    mask_1 = dld_group_storage["t (ns)"].to_numpy() > max_tof
    mask_2 = dld_group_storage["t (ns)"].to_numpy() < 50
    mask_3 = (
        (dld_group_storage["x_det (cm)"].to_numpy() == 0)
        & (dld_group_storage["y_det (cm)"].to_numpy() == 0)
        & (dld_group_storage["t (ns)"].to_numpy() == 0)
    )
    mask_4 = dld_group_storage["high_voltage (V)"].to_numpy() < 0
    mask_5 = (
        (dld_group_storage["x_det (cm)"].to_numpy() == 0)
        & (dld_group_storage["y_det (cm)"].to_numpy() == 0)
    )

    mask_f_1 = np.logical_or(mask_1, mask_2)
    mask_f_2 = np.logical_or(mask_3, mask_4)
    mask_f_2 = np.logical_or(mask_f_2, mask_5)
    mask = np.logical_or(mask_f_1, mask_f_2)

    num_over_max_tof = int(np.count_nonzero(mask))
    dld_group_storage.drop(np.where(mask)[0], inplace=True)
    dld_group_storage.reset_index(inplace=True, drop=True)
    print("The number of data that is removed:", num_over_max_tof)
    return dld_group_storage


def _slice_data_for_export(
    data: pd.DataFrame,
    start_index: int | None = None,
    end_index: int | None = None,
) -> pd.DataFrame:
    """Return an index-sliced copy for export using inclusive end semantics."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas.DataFrame")

    if start_index in (None, "") and end_index in (None, ""):
        return data

    total_rows = len(data)
    if total_rows == 0:
        raise ValueError("Cannot slice an empty dataset for export")

    start = 0 if start_index in (None, "") else int(start_index)
    end = (total_rows - 1) if end_index in (None, "", -1) else int(end_index)

    if start < 0 or start >= total_rows:
        raise ValueError(f"start_index must be between 0 and {total_rows - 1}")
    if end == -1:
        end = total_rows - 1
    if end < 0 or end >= total_rows:
        raise ValueError(f"end_index must be between 0 and {total_rows - 1}, or -1 for the last row")
    if end < start:
        raise ValueError("end_index must be greater than or equal to start_index")

    return data.iloc[start:end + 1].reset_index(drop=True).copy()


def save_data(
    data,
    variables,
    name=None,
    hdf=True,
    epos=False,
    pos=False,
    ato_6v=False,
    csv=False,
    temp=False,
    start_index=0,
    end_index=-1,
    save_tdc=False,
    save_range=False,
):
    """Persist data in one or more supported export formats.

    Parameters
    ----------
    save_tdc : bool, default False
        If True and ``hdf=True`` and ``variables.data_tdc`` was loaded, also
        write a filtered ``/tdc`` group into the same HDF5 file. The tdc rows
        are filtered so that only those whose linked dld row still exists in
        ``data`` are kept; "orphan" tdc rows (pulses that never produced a dld
        event in the first place) are always preserved.
    save_range : bool, default False
        If True and ``hdf=True`` and ``variables.range_data`` is non-empty,
        also write the current range table under ``/range`` so the calibrated,
        raw, and ranging information all live in a single h5 file.
    """
    if name is not None:
        data_name = name
    elif temp:
        data_name = f"{variables.result_data_name}_temp"
    else:
        data_name = variables.result_data_name

    export_data = _slice_data_for_export(data, start_index=start_index, end_index=end_index)
    if not any([hdf, epos, pos, ato_6v, csv]):
        print("No export format was selected.")
        return None

    saved_outputs: dict[str, str] = {}

    if hdf:
        output_h5 = _resolve_variable_output_file(variables, filename=f"{data_name}.h5", data_directory=True)
        store_df_to_hdf(export_data, "df", output_h5)
        saved_outputs["hdf"] = output_h5
        if save_tdc:
            tdc_full = getattr(variables, "data_tdc", None)
            if tdc_full is None:
                warnings.warn(
                    "save_tdc=True but variables.data_tdc is None; no raw tdc to save.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            elif "event_group_id" not in export_data.columns:
                warnings.warn(
                    "save_tdc=True but exported dld has no event_group_id column; "
                    "raw tdc was not loaded with the linking flag enabled.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                from pyccapt.calibration.data_tools.data_loadcrop import filter_tdc_by_dld

                tdc_filtered = filter_tdc_by_dld(export_data, tdc_full)
                # Append /tdc to the same h5 file (mode='a' to preserve /df).
                tdc_filtered.to_hdf(_as_path(output_h5), key="tdc", mode="a")
                saved_outputs["hdf_tdc"] = f"{output_h5} (/tdc, {len(tdc_filtered)} rows)"
        if save_range:
            range_table = getattr(variables, "range_data", None)
            if range_table is None or len(range_table) == 0:
                warnings.warn(
                    "save_range=True but variables.range_data is empty.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                range_table.to_hdf(_as_path(output_h5), key="range", mode="a")
                saved_outputs["hdf_range"] = f"{output_h5} (/range, {len(range_table)} rows)"
    if epos:
        output_epos = _resolve_variable_output_file(variables, filename=f"{data_name}.epos", data_directory=True)
        output_epos_path = _as_path(output_epos)
        output_epos_path.parent.mkdir(parents=True, exist_ok=True)
        ccapt_tools.ccapt_to_epos(
            export_data,
            path=str(output_epos_path.parent) + os.sep,
            name=output_epos_path.name,
        )
        saved_outputs["epos"] = str(output_epos_path)
    if pos:
        output_pos = _resolve_variable_output_file(variables, filename=f"{data_name}.pos", data_directory=True)
        output_pos_path = _as_path(output_pos)
        output_pos_path.parent.mkdir(parents=True, exist_ok=True)
        ccapt_tools.ccapt_to_pos(
            export_data,
            path=str(output_pos_path.parent) + os.sep,
            name=output_pos_path.name,
        )
        saved_outputs["pos"] = str(output_pos_path)
    if ato_6v:
        output_ato = _resolve_variable_output_file(variables, filename=f"{data_name}.ato", data_directory=True)
        output_ato_path = _as_path(output_ato)
        output_ato_path.parent.mkdir(parents=True, exist_ok=True)
        ato_tools.ccapt_to_ato(
            export_data,
            path=str(output_ato_path.parent),
            name=output_ato_path.name,
        )
        saved_outputs["ato"] = str(output_ato_path)
    if csv:
        output_csv = _resolve_variable_output_file(
            variables, filename=f"{data_name}.csv", data_directory=True
        )
        store_df_to_csv(export_data, output_csv)
        saved_outputs["csv"] = output_csv

    for format_name, output_path in saved_outputs.items():
        print(f"Saved {format_name.upper()}: {output_path}")

    return None


def load_data(dataset_path, data_type, mode="processed", *, load_tdc=False, tdc_extract_mode="tdc_sc"):
    """Load supported dataset formats into the pyccapt dataframe convention.

    When ``load_tdc=True`` (only supported for ``data_type='pyccapt'`` in
    ``mode='raw'``), also reads the ``/tdc`` group and returns
    ``(dld_df, tdc_df)``. Both dataframes carry a shared ``event_group_id``
    column so the dld→tdc link survives downstream cropping.
    """
    if data_type in {"leap_pos", "leap_epos"}:
        if load_tdc:
            raise ValueError("load_tdc is only supported for pyccapt h5 datasets")
        if data_type == "leap_epos":
            return ccapt_tools.epos_to_ccapt(dataset_path)
        print("The dataset should contains at least epos information to use all possible analysis")
        return ccapt_tools.pos_to_ccapt(dataset_path)

    if data_type == "leap_apt":
        if load_tdc:
            raise ValueError("load_tdc is only supported for pyccapt h5 datasets")
        return ccapt_tools.apt_to_ccapt(dataset_path)

    if data_type == "ato_v6":
        if load_tdc:
            raise ValueError("load_tdc is only supported for pyccapt h5 datasets")
        return ato_tools.ato_to_ccapt(dataset_path, mode="pyccapt")

    if data_type == "pyccapt":
        if mode == "raw":
            from pyccapt.calibration.data_tools import data_loadcrop

            if load_tdc:
                primary_mode = tdc_extract_mode
                fallback_modes = [mode_name for mode_name in ("tdc_sc", "tdc_ro") if mode_name != primary_mode]
                last_error = None
                for mode_name in [primary_mode, *fallback_modes]:
                    try:
                        return data_loadcrop.fetch_dataset_with_tdc(
                            dataset_path, tdc_extract_mode=mode_name
                        )
                    except Exception as error:
                        last_error = error
                raise last_error
            return data_loadcrop.fetch_dataset_from_dld_grp(dataset_path)
        if mode == "processed":
            data = pd.read_hdf(dataset_path, key="df", mode="r")
            if "pulse_v (V)" not in data.columns:
                if "pulse" in data.columns:
                    insert_at = data.columns.get_loc("pulse") + 1
                    data.insert(insert_at, "pulse_v (V)", data["pulse"])
                    data.drop("pulse", axis=1, inplace=True)
                else:
                    warnings.warn(
                        "Main dataframe does not have 'pulse' or 'pulse_v (V)'; using zeros for pulse_v (V).",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    pulse_v = np.zeros(len(data), dtype=float)
                    insert_at = data.columns.get_loc("high_voltage (V)") + 1 if "high_voltage (V)" in data.columns else len(data.columns)
                    data.insert(insert_at, "pulse_v (V)", pulse_v)
            if "pulse_l (pJ)" not in data.columns:
                pulse_l = np.zeros(len(data), dtype=float)
                data.insert(data.columns.get_loc("pulse_v (V)") + 1, "pulse_l (pJ)", pulse_l)
            if load_tdc:
                try:
                    tdc_df = pd.read_hdf(dataset_path, key="tdc", mode="r")
                except (KeyError, ValueError):
                    warnings.warn(
                        f"load_tdc=True but {dataset_path!r} has no /tdc group. "
                        "Returning dld only.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    return data
                return data, tdc_df
            return data
        raise ValueError(f"Unsupported mode for data_type='pyccapt': {mode!r}")

    raise ValueError(f"Unsupported data_type: {data_type!r}")


def extract_data(data, variables, flightPathLength_d, max_mc):
    """Extract common calibrated arrays and metadata into shared variables."""
    def _resolve_column(candidates):
        for column in candidates:
            if column in data.columns:
                return column
        return None

    high_voltage_aliases = (
        "high_voltage (V)",
        "high_voltage",
        "HV_DC (V)",
        "Voltage",
        "VDC",
        "Vref",
    )
    voltage_column = next((column for column in high_voltage_aliases if column in data.columns), None)
    if voltage_column is None:
        warnings.warn(
            "Main dataframe does not have a high-voltage column; using zeros for 'high_voltage (V)'.",
            RuntimeWarning,
            stacklevel=2,
        )
        variables.dld_high_voltage = np.zeros(len(data), dtype=float)
    else:
        variables.dld_high_voltage = data[voltage_column].to_numpy()
    if "pulse_v (V)" in data.columns:
        variables.dld_pulse_v = data["pulse_v (V)"].to_numpy()
    elif "pulse" in data.columns:
        variables.dld_pulse_v = data["pulse"].to_numpy()
    else:
        warnings.warn(
            "Main dataframe does not have 'pulse' or 'pulse_v (V)'; using zeros for pulse values.",
            RuntimeWarning,
            stacklevel=2,
        )
        variables.dld_pulse_v = np.zeros_like(variables.dld_high_voltage)
    if "pulse_l (pJ)" not in data.columns:
        variables.dld_pulse_l = np.zeros_like(variables.dld_high_voltage)
    else:
        variables.dld_pulse_l = data["pulse_l (pJ)"].to_numpy()

    tof_aliases = ("t (ns)", "TOF (ns)", "Epos ToF", "tof_ns", "tof", "tofc", "t_c (ns)")
    tof_column = _resolve_column(tof_aliases)
    if tof_column is None:
        raise KeyError(
            "No TOF column found. Expected one of: "
            + ", ".join(tof_aliases)
            + f". Available columns: {list(data.columns)}"
        )
    if tof_column != "t (ns)":
        warnings.warn(
            f"Using legacy TOF column '{tof_column}' as 't (ns)'.",
            RuntimeWarning,
            stacklevel=2,
        )
    variables.dld_t = data[tof_column].to_numpy()

    x_det_cm_column = _resolve_column(("x_det (cm)", "x_det", "det_x (cm)", "XDet_cm"))
    y_det_cm_column = _resolve_column(("y_det (cm)", "y_det", "det_y (cm)", "YDet_cm"))
    x_det_mm_column = _resolve_column(("det_x (mm)", "XDet_mm", "det_x"))
    y_det_mm_column = _resolve_column(("det_y (mm)", "YDet_mm", "det_y"))

    if x_det_cm_column is not None:
        variables.dld_x_det = data[x_det_cm_column].to_numpy()
    elif x_det_mm_column is not None:
        warnings.warn(
            f"Using mm detector column '{x_det_mm_column}' and converting to cm for x_det.",
            RuntimeWarning,
            stacklevel=2,
        )
        variables.dld_x_det = data[x_det_mm_column].to_numpy() / 10.0
    else:
        warnings.warn(
            "No x detector column found; using zeros for x_det (cm).",
            RuntimeWarning,
            stacklevel=2,
        )
        variables.dld_x_det = np.zeros(len(data), dtype=float)

    if y_det_cm_column is not None:
        variables.dld_y_det = data[y_det_cm_column].to_numpy()
    elif y_det_mm_column is not None:
        warnings.warn(
            f"Using mm detector column '{y_det_mm_column}' and converting to cm for y_det.",
            RuntimeWarning,
            stacklevel=2,
        )
        variables.dld_y_det = data[y_det_mm_column].to_numpy() / 10.0
    else:
        warnings.warn(
            "No y detector column found; using zeros for y_det (cm).",
            RuntimeWarning,
            stacklevel=2,
        )
        variables.dld_y_det = np.zeros(len(data), dtype=float)

    if "mc (Da)" in data.columns:
        variables.mc = data["mc (Da)"].to_numpy()
    if "t_c (ns)" in data.columns:
        variables.dld_t_c = data["t_c (ns)"].to_numpy()
    if "mc_uc (Da)" in data.columns:
        variables.mc_calib = data["mc_uc (Da)"].to_numpy()
        variables.mc_calib_backup = data["mc_uc (Da)"].to_numpy()
        variables.mc_uc = data["mc_uc (Da)"].to_numpy()

    variables.max_tof = int(tof_tools.mc2tof(max_mc, 1000, 0, 0, flightPathLength_d))
    variables.dld_t_calib = variables.dld_t.copy()
    variables.dld_t_calib_backup = variables.dld_t.copy()

    if {"x (nm)", "y (nm)", "z (nm)"} <= set(data.columns):
        variables.x = data["x (nm)"].to_numpy()
        variables.y = data["y (nm)"].to_numpy()
        variables.z = data["z (nm)"].to_numpy()
    print("The maximum possible time of flight is:", variables.max_tof)
    return variables


def pyccapt_raw_to_processed(data):
    """Convert a raw pyccapt dataframe to the processed schema.

    Existing calibrated columns are preserved when present:

    - If the input already has ``mc (Da)`` (e.g. it came from a partly-processed
      bundle), it is copied through untouched.
    - If the input already has ``mc_uc (Da)``, that is also copied through.
    - Otherwise, when the inputs needed for the uncalibrated mc formula are all
      present (``t (ns)``, ``high_voltage (V)``, ``x_det (cm)``, ``y_det (cm)``),
      ``mc_uc (Da)`` is computed on the fly using
      ``tof2mc(t0=0, V_pulse=0, flightPathLength=110, mode='voltage')`` — the
      same uncalibrated formula the legacy raw-data notebook used to produce
      Figure 6A in the PyCCAPT paper. This makes raw acquisition files (which
      have never been through calibration) usable in downstream M/C plots.

    Columns that have no obvious raw equivalent (``x/y/z (nm)``, ``t_c (ns)``,
    ``delta_p``, ``multi``) are zero-initialized as before.
    """
    # Local import to avoid a circular dependency at module load (mc_tools is in
    # a sibling subpackage that itself imports from data_tools).
    from pyccapt.calibration.mc import mc_tools

    n = len(data)
    data_processed = pd.DataFrame()
    data_processed["x (nm)"] = np.zeros(n)
    data_processed["y (nm)"] = np.zeros(n)
    data_processed["z (nm)"] = np.zeros(n)

    if "mc (Da)" in data.columns:
        data_processed["mc (Da)"] = data["mc (Da)"].to_numpy()
    else:
        data_processed["mc (Da)"] = np.zeros(n)

    if "mc_uc (Da)" in data.columns:
        data_processed["mc_uc (Da)"] = data["mc_uc (Da)"].to_numpy()
    else:
        required = {"t (ns)", "high_voltage (V)", "x_det (cm)", "y_det (cm)"}
        if n > 0 and required.issubset(data.columns):
            data_processed["mc_uc (Da)"] = mc_tools.tof2mc(
                t=data["t (ns)"].to_numpy().astype(float),
                t0=0,
                V=data["high_voltage (V)"].to_numpy().astype(float),
                xDet=data["x_det (cm)"].to_numpy().astype(float),
                yDet=data["y_det (cm)"].to_numpy().astype(float),
                flightPathLength=110,
                V_pulse=np.zeros(n),
                mode="voltage",
            )
        else:
            data_processed["mc_uc (Da)"] = np.zeros(n)

    data_processed["high_voltage (V)"] = data["high_voltage (V)"].to_numpy()
    data_processed["pulse_v (V)"] = data["pulse_v (V)"].to_numpy()
    data_processed["pulse_l (pJ)"] = data["pulse_l (pJ)"].to_numpy()
    data_processed["t (ns)"] = data["t (ns)"].to_numpy()
    data_processed["t_c (ns)"] = (
        data["t_c (ns)"].to_numpy() if "t_c (ns)" in data.columns else np.zeros(n)
    )
    data_processed["x_det (cm)"] = data["x_det (cm)"].to_numpy()
    data_processed["y_det (cm)"] = data["y_det (cm)"].to_numpy()
    data_processed["delta_p"] = (
        data["delta_p"].to_numpy() if "delta_p" in data.columns else np.zeros(n)
    )
    data_processed["multi"] = (
        data["multi"].to_numpy() if "multi" in data.columns else np.zeros(n)
    )
    data_processed["start_counter"] = data["start_counter"].to_numpy()
    if "event_group_id" in data.columns:
        data_processed["event_group_id"] = data["event_group_id"].to_numpy()
    return data_processed


def save_range(variables) -> None:
    """Save range data as both HDF5 and CSV."""
    if getattr(variables, "range_data", None) is None or variables.range_data.empty:
        print("No range table is loaded, so nothing was saved.")
        return
    h5_file = _resolve_variable_output_file(
        variables,
        filename=f"{variables.dataset_name}_range.h5",
        data_directory=True,
    )
    csv_file = _resolve_variable_output_file(
        variables,
        filename=f"{variables.dataset_name}_range.csv",
        data_directory=True,
    )
    store_df_to_hdf(variables.range_data, "df", h5_file)
    store_df_to_csv(variables.range_data, csv_file)
    print(f"Saved range HDF: {h5_file}")
    print(f"Saved range CSV: {csv_file}")
