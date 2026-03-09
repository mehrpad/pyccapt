"""Parameter parsing and validation helpers for the main control GUI."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyccapt.control.core import read_files


class ParameterError(ValueError):
    """Raised when GUI parameter blocks cannot be parsed or validated."""


TEXTLINE_REQUIRED_KEYS = (
    "ex_user",
    "ex_name",
    "electrode",
    "ex_time",
    "max_ions",
    "ex_freq",
    "vdc_min",
    "vdc_max",
    "vdc_steps_up",
    "vdc_steps_down",
    "control_algorithm",
    "pulse_mode",
    "vp_min",
    "vp_max",
    "pulse_fraction",
    "pulse_frequency",
    "detection_rate_init",
    "hit_displayed",
    "email",
    "counter_source",
    "criteria_time",
    "criteria_ions",
    "criteria_vdc",
)


@dataclass
class FormValues:
    user_name: str
    ex_name: str
    electrode: str
    ex_time: str
    ex_freq: str
    max_ions: str
    vdc_min: str
    vdc_max: str
    detection_rate_init: str
    pulse_fraction: str
    email: str
    vdc_steps_up: str
    vdc_steps_down: str
    control_algorithm: str
    pulse_mode: str
    vp_min: str
    vp_max: str
    pulse_frequency: str
    counter_source: str
    criteria_time: bool
    criteria_ions: bool
    criteria_vdc: bool


def _electrode_names_from_mapping(data: Mapping[str, Any]) -> list[str]:
    """Extract electrode names from mapping data.

    Supports both:
    - numeric-key mappings (legacy JSON style)
    - generic string-key mappings
    """
    sortable_items: list[tuple[str, Any]] = list(data.items())
    if not sortable_items:
        return []

    def sort_key(item: tuple[str, Any]) -> tuple[int, int | str]:
        key = str(item[0])
        try:
            return (0, int(key))
        except ValueError:
            return (1, key)

    names = [str(value).strip() for _, value in sorted(sortable_items, key=sort_key) if str(value).strip()]
    return names


def _electrode_names_from_toml_data(data: Mapping[str, Any]) -> list[str]:
    """Extract electrode names from TOML config structure."""
    if "electrodes" in data and isinstance(data["electrodes"], Mapping):
        electrode_block = data["electrodes"]
        names = electrode_block.get("names")
        if isinstance(names, list):
            return [str(item).strip() for item in names if str(item).strip()]
        return _electrode_names_from_mapping(electrode_block)

    names = data.get("names")
    if isinstance(names, list):
        return [str(item).strip() for item in names if str(item).strip()]
    return _electrode_names_from_mapping(data)


def load_electrode_items(file_path: str) -> list[str]:
    """Load electrode names from TOML config, with legacy JSON fallback.

    Preferred format:
        [electrodes]
        names = ["...", "..."]
    """
    path = Path(file_path).expanduser().resolve()
    candidates = [path]
    if path.suffix.lower() == ".toml":
        candidates.append(path.with_suffix(".json"))

    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix.lower() == ".toml":
            data = read_files.read_toml_file(candidate)
            names = _electrode_names_from_toml_data(data)
        elif candidate.suffix.lower() == ".json":
            data = read_files.read_json_file(candidate)
            names = _electrode_names_from_mapping(data)
        else:
            continue

        if names:
            return names

    raise FileNotFoundError(
        f"Could not load electrode configuration from {path} "
        f"(or legacy fallback {path.with_suffix('.json')})."
    )


def _convert_value(raw_value: str) -> Any:
    value = raw_value.strip()
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def parse_textline_experiments(lines: str) -> list[dict[str, Any]]:
    """Parse text-line experiment blocks from GUI text input."""
    matches = re.findall(r"{(.*?)}", lines, re.DOTALL)
    if not matches:
        raise ParameterError("No parameter blocks found in TextLine input.")

    parsed_items: list[dict[str, Any]] = []
    for block in matches:
        item: dict[str, Any] = {}
        for element in block.split(";"):
            element = element.strip()
            if not element:
                continue
            if "=" not in element:
                raise ParameterError(f"Invalid key/value pair: {element!r}")
            key, value = element.split("=", 1)
            item[key.strip()] = _convert_value(value)

        missing = [key for key in TEXTLINE_REQUIRED_KEYS if key not in item]
        if missing:
            raise ParameterError(f"Missing keys in dictionary: {missing}")
        parsed_items.append(item)

    return parsed_items


def apply_textline_item(
    variables: Any,
    conf: Mapping[str, Any],
    item: Mapping[str, Any],
    emit_error: Callable[[str], None],
) -> None:
    """Apply one parsed text-line experiment definition to shared variables."""
    variables.ex_user = str(item["ex_user"])
    variables.ex_name = str(item["ex_name"])
    variables.electrode = str(item["electrode"])
    variables.ex_time = int(item["ex_time"])
    variables.max_ions = int(item["max_ions"])
    variables.ex_freq = int(item["ex_freq"])
    variables.vdc_min = int(item["vdc_min"])

    vdc_max = int(item["vdc_max"])
    if vdc_max < conf["max_vdc"]:
        variables.vdc_max = vdc_max
    else:
        emit_error(f"Maximum possible Vdc is {conf['max_vdc']}")

    variables.vdc_steps_up = float(item["vdc_steps_up"])
    variables.vdc_steps_down = float(item["vdc_steps_down"])
    variables.control_algorithm = str(item["control_algorithm"])
    variables.pulse_mode = str(item["pulse_mode"])

    vp_min = int(item["vp_min"])
    if vp_min < conf["min_vp"]:
        variables.vp_min = conf["min_vp"]
        emit_error(f"Minimum possible V_p is {conf['min_vp']}")
    elif vp_min > conf["max_vp"]:
        emit_error(f"Maximum possible V_p is {conf['max_vp']}")
    else:
        variables.vp_min = vp_min

    vp_max = int(item["vp_max"])
    if vp_max < conf["max_vp"]:
        variables.vp_max = vp_max
    else:
        emit_error(f"Maximum possible V_p is {conf['max_vp']}")

    variables.pulse_fraction = int(item["pulse_fraction"])
    variables.pulse_frequency = int(item["pulse_frequency"])
    variables.detection_rate_init = float(item["detection_rate_init"])
    variables.hit_display = int(item["hit_displayed"])
    variables.email = str(item["email"])
    variables.counter_source = str(item["counter_source"])
    variables.criteria_time = bool(item["criteria_time"])
    variables.criteria_ions = bool(item["criteria_ions"])
    variables.criteria_vdc = bool(item["criteria_vdc"])


def _bounded_pulse_frequency(value: int, pulse_mode: str, conf: Mapping[str, Any]) -> tuple[int, str | None]:
    if pulse_mode == "Voltage":
        minimum = int(conf["min_voltage_pulse_frequency"])
        maximum = int(conf["max_voltage_pulse_frequency"])
    elif pulse_mode == "Laser":
        minimum = int(conf["min_laser_pulse_frequency"])
        maximum = int(conf["max_laser_pulse_frequency"])
    else:
        minimum = max(int(conf["min_voltage_pulse_frequency"]), int(conf["min_laser_pulse_frequency"]))
        maximum = min(int(conf["max_voltage_pulse_frequency"]), int(conf["max_laser_pulse_frequency"]))

    if value < minimum:
        return minimum, f"Minimum possible number is {minimum}"
    if value > maximum:
        return maximum, f"Maximum possible number is {maximum}"
    return value, None


def apply_form_values(
    variables: Any,
    conf: Mapping[str, Any],
    values: FormValues,
    emit_error: Callable[[str], None],
) -> dict[str, str]:
    """Apply textbox parameters and return widget text corrections."""
    corrections: dict[str, str] = {}

    variables.user_name = values.user_name
    variables.ex_name = values.ex_name
    variables.electrode = values.electrode
    variables.ex_time = int(float(values.ex_time))
    variables.ex_freq = int(float(values.ex_freq))
    variables.max_ions = int(float(values.max_ions))
    variables.vdc_min = int(float(values.vdc_min))
    variables.detection_rate = float(values.detection_rate_init)
    variables.hdf5_data_name = values.ex_name
    variables.email = values.email
    variables.vdc_step_up = float(values.vdc_steps_up)
    variables.vdc_step_down = float(values.vdc_steps_down)
    variables.control_algorithm = values.control_algorithm
    variables.pulse_mode = values.pulse_mode
    variables.v_p_min = int(float(values.vp_min))
    variables.v_p_max = int(float(values.vp_max))
    variables.counter_source = values.counter_source

    pulse_fraction_value = int(float(values.pulse_fraction))
    if pulse_fraction_value > conf["pulse_fraction_max"]:
        emit_error(f"Maximum possible number is {conf['pulse_fraction_max']}")
        pulse_fraction_value = int(conf["pulse_fraction_max"])
        corrections["pulse_fraction"] = str(pulse_fraction_value)
    variables.pulse_fraction = pulse_fraction_value

    vp_max_value = int(float(values.vp_max))
    if vp_max_value > conf["max_vp"]:
        emit_error(f"Maximum possible number is {conf['max_vp']}")
        vp_max_value = int(conf["max_vp"])
        corrections["vp_max"] = str(vp_max_value)
    variables.v_p_max = vp_max_value

    vdc_max_value = int(float(values.vdc_max))
    if vdc_max_value > conf["max_vdc"]:
        emit_error(f"Maximum possible number is {conf['max_vdc']}")
        vdc_max_value = int(conf["max_vdc"])
        corrections["vdc_max"] = str(vdc_max_value)
    variables.vdc_max = vdc_max_value

    pulse_frequency_value = int(values.pulse_frequency)
    pulse_frequency_value, frequency_message = _bounded_pulse_frequency(
        pulse_frequency_value,
        values.pulse_mode,
        conf,
    )
    if frequency_message:
        emit_error(frequency_message)
        corrections["pulse_frequency"] = str(pulse_frequency_value)
    variables.pulse_frequency = pulse_frequency_value

    variables.criteria_time = values.criteria_time
    variables.criteria_ions = values.criteria_ions
    variables.criteria_vdc = values.criteria_vdc

    return corrections

