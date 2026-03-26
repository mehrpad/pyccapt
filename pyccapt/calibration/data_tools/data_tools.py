"""Data loading and persistence helpers for calibration workflows."""

from __future__ import annotations

import os
from pathlib import Path

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


def read_hdf5(filename: str | Path):
    """
    Read non-pandas HDF5 content into a dictionary of dataframes.

    Returns:
        dict[str, pd.DataFrame] | None
    """
    file_path = _as_path(filename)
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
            return pd.read_hdf(file_path, mode="r")
        if suffix == ".rrng":
            return leap_tools.read_rrng(str(file_path))
        if suffix == ".rng":
            return leap_tools.read_rng(str(file_path))
        raise ValueError(f"Unsupported range file extension: {file_path.suffix!r}")
    except FileNotFoundError as error:
        print("[*] Range file could not be found")
        print(error)
        raise


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


def store_df_to_hdf(dataframe, key, filename):
    """
    Store dataframe to HDF5.

    Supports both modern argument order `(dataframe, key, filename)` and the
    legacy order `(filename, dataframe, key)` for backwards compatibility.
    """
    if isinstance(dataframe, (str, Path)):
        # Legacy order: (filename, dataframe, key)
        filename, dataframe, key = dataframe, key, filename

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas.DataFrame")

    file_path = _as_path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_hdf(file_path, str(key), mode="w")


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
):
    """Persist data in one or more supported export formats."""
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


def load_data(dataset_path, data_type, mode="processed"):
    """Load supported dataset formats into the pyccapt dataframe convention."""
    if data_type in {"leap_pos", "leap_epos"}:
        if data_type == "leap_epos":
            return ccapt_tools.epos_to_ccapt(dataset_path)
        print("The dataset should contains at least epos information to use all possible analysis")
        return ccapt_tools.pos_to_ccapt(dataset_path)

    if data_type == "leap_apt":
        return ccapt_tools.apt_to_ccapt(dataset_path)

    if data_type == "ato_v6":
        return ato_tools.ato_to_ccapt(dataset_path, mode="pyccapt")

    if data_type == "pyccapt":
        if mode == "raw":
            from pyccapt.calibration.data_tools import data_loadcrop

            return data_loadcrop.fetch_dataset_from_dld_grp(dataset_path)
        if mode == "processed":
            data = pd.read_hdf(dataset_path, mode="r")
            if "pulse_v (V)" not in data.columns:
                data.insert(data.columns.get_loc("pulse") + 1, "pulse_v (V)", data["pulse"])
                data.drop("pulse", axis=1, inplace=True)
            if "pulse_l (pJ)" not in data.columns:
                pulse_l = np.zeros_like(data["high_voltage (V)"])
                data.insert(data.columns.get_loc("pulse_v (V)") + 1, "pulse_l (pJ)", pulse_l)
            return data
        raise ValueError(f"Unsupported mode for data_type='pyccapt': {mode!r}")

    raise ValueError(f"Unsupported data_type: {data_type!r}")


def extract_data(data, variables, flightPathLength_d, max_mc):
    """Extract common calibrated arrays and metadata into shared variables."""
    variables.dld_high_voltage = data["high_voltage (V)"].to_numpy()
    if "pulse_v (V)" not in data.columns:
        variables.dld_pulse_v = data["pulse"].to_numpy()
    else:
        variables.dld_pulse_v = data["pulse_v (V)"].to_numpy()
    if "pulse_l (pJ)" not in data.columns:
        variables.dld_pulse_l = np.zeros_like(variables.dld_high_voltage)
    else:
        variables.dld_pulse_l = data["pulse_l (pJ)"].to_numpy()

    variables.dld_t = data["t (ns)"].to_numpy()
    variables.dld_x_det = data["x_det (cm)"].to_numpy()
    variables.dld_y_det = data["y_det (cm)"].to_numpy()

    if "mc (Da)" in data.columns:
        variables.mc = data["mc (Da)"].to_numpy()
    if "t_c (ns)" in data.columns:
        variables.dld_t_c = data["t_c (ns)"].to_numpy()
    if "mc_uc (Da)" in data.columns:
        variables.mc_calib = data["mc_uc (Da)"].to_numpy()
        variables.mc_calib_backup = data["mc_uc (Da)"].to_numpy()
        variables.mc_uc = data["mc_uc (Da)"].to_numpy()

    variables.max_tof = int(tof_tools.mc2tof(max_mc, 1000, 0, 0, flightPathLength_d))
    variables.dld_t_calib = data["t (ns)"].to_numpy()
    variables.dld_t_calib_backup = data["t (ns)"].to_numpy()

    if {"x (nm)", "y (nm)", "z (nm)"} <= set(data.columns):
        variables.x = data["x (nm)"].to_numpy()
        variables.y = data["y (nm)"].to_numpy()
        variables.z = data["z (nm)"].to_numpy()
    print("The maximum possible time of flight is:", variables.max_tof)
    return variables


def pyccapt_raw_to_processed(data):
    """Convert a raw pyccapt dataframe to the processed schema."""
    data_processed = pd.DataFrame()
    data_processed["x (nm)"] = np.zeros(len(data))
    data_processed["y (nm)"] = np.zeros(len(data))
    data_processed["z (nm)"] = np.zeros(len(data))
    data_processed["mc (Da)"] = np.zeros(len(data))
    data_processed["mc_uc (Da)"] = np.zeros(len(data))
    data_processed["high_voltage (V)"] = data["high_voltage (V)"].to_numpy()
    data_processed["pulse_v (V)"] = data["pulse_v (V)"].to_numpy()
    data_processed["pulse_l (pJ)"] = data["pulse_l (pJ)"].to_numpy()
    data_processed["t (ns)"] = data["t (ns)"].to_numpy()
    data_processed["t_c (ns)"] = np.zeros(len(data))
    data_processed["x_det (cm)"] = data["x_det (cm)"].to_numpy()
    data_processed["y_det (cm)"] = data["y_det (cm)"].to_numpy()
    data_processed["delta_p"] = np.zeros(len(data))
    data_processed["multi"] = np.zeros(len(data))
    data_processed["start_counter"] = data["start_counter"].to_numpy()
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
