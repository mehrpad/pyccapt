"""Point cloud plotting helpers for LEAP data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import plotly

from pyccapt.calibration.leap_tools import leap_tools

_INPUT_TYPES = {"pos", "epos"}
_PLOT_TYPES = {"projection", "cloud"}


def _ensure_choice(value: str, *, field_name: str, allowed: set[str]) -> str:
    normalized = str(value).strip().lower()
    if normalized not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"Invalid {field_name!r}: {value!r}. Allowed values are: {choices}")
    return normalized


def _resolve_output_dir(result_path: str | Path) -> Path:
    output_dir = Path(result_path).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _require_phases(phases) -> list[str]:
    if phases is None:
        raise ValueError("'phases' must be a non-empty sequence")
    phase_list = [str(item) for item in phases if str(item).strip()]
    if not phase_list:
        raise ValueError("'phases' must contain at least one non-empty phase label")
    return phase_list


def pre_data(pos_file: str | Path, rrng_file: str | Path, input_type: str):
    """
    Read POS/EPOS and RRNG files, label ions, and return deconvolved data.
    """
    normalized_input = _ensure_choice(input_type, field_name="input_type", allowed=_INPUT_TYPES)
    pos_file = str(Path(pos_file).expanduser())
    rrng_file = str(Path(rrng_file).expanduser())

    _, ranges = leap_tools.read_rrng(rrng_file)
    if normalized_input == "pos":
        position_data = leap_tools.read_pos(pos_file)
    else:
        position_data = leap_tools.read_epos(pos_file)
    labeled = leap_tools.label_ions(position_data, ranges)
    return leap_tools.deconvolve(labeled)


def decompose(data, element: str):
    """
    Extract x/y/z coordinates and display color for one element label.
    """
    required_columns = {"x (nm)", "y (nm)", "z (nm)", "colour"}
    missing = required_columns.difference(set(data.columns))
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise KeyError(f"Missing required data columns: {missing_text}")
    if element not in data.index:
        raise KeyError(f"Element {element!r} not found in data index")

    selection = data.loc[element]
    if hasattr(selection, "to_frame"):
        selection = selection.to_frame().T if getattr(selection, "ndim", 1) == 1 else selection
    positions = selection[["x (nm)", "y (nm)", "z (nm)"]].to_numpy()
    color = selection["colour"].iloc[0]
    px, py, pz = positions[:, 0], positions[:, 1], positions[:, 2]
    return px, py, pz, color


def cloud_plotter(
    data,
    phases,
    result_path: str | Path,
    filename: str,
    plot_type: str = "cloud",
    open_new_window: bool = False,
):
    """
    Build either a projection image or a Plotly point-cloud figure.
    """
    normalized_plot_type = _ensure_choice(plot_type, field_name="plot_type", allowed=_PLOT_TYPES)
    phase_list = _require_phases(phases)

    if normalized_plot_type == "projection":
        output_dir = _resolve_output_dir(result_path)
        figure, axis = plt.subplots()
        for element in phase_list:
            _, py, pz, _ = decompose(data, element)
            axis.scatter(py, pz, s=2, label=element)
        axis.xaxis.tick_top()
        axis.invert_yaxis()
        axis.set_xlabel("Y")
        axis.xaxis.set_label_position("top")
        axis.set_ylabel("Z")
        axis.legend()
        figure.savefig(output_dir / f"output_{filename}.png")
        plt.close(figure)
        return None

    plotly_data = []
    for element in phase_list:
        px, py, pz, color = decompose(data, element)
        plotly_data.append(
            {
                "mode": "markers",
                "name": element,
                "type": "scatter3d",
                "x": px,
                "y": py,
                "z": pz,
                "opacity": 0.2,
                "marker": {"size": 2, "color": color},
            }
        )

    layout = {
        "title": "APT 3D Point Cloud",
        "scene": {
            "xaxis": {"zeroline": False, "title": "x (nm)"},
            "yaxis": {"zeroline": False, "title": "y (nm)"},
            "zaxis": {"zeroline": False, "title": "z (nm)", "autorange": "reversed"},
        },
    }
    figure = {"data": plotly_data, "layout": layout}

    if open_new_window:
        output_dir = _resolve_output_dir(result_path)
        plotly.offline.plot(figure, filename=str(output_dir / f"{filename}.html"), show_link=False)
        return None
    return figure

