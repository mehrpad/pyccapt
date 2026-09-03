"""Parameter parsing and validation helpers for the main control GUI."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyccapt.control.apt.detector_models import normalize_counter_source
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
    # Optional so older callers / TextLine blocks that don't supply it
    # still construct a valid FormValues.
    criteria_email: bool = False
    email_interval_events: str = "1000000"


@dataclass(frozen=True)
class RunConfig:
    """Validated immutable snapshot consumed when a process is started."""

    ex_freq: float
    ex_time: float
    max_ions: int
    vdc_min: float
    vdc_max: float
    v_p_min: float
    v_p_max: float
    pulse_fraction: float
    pulse_frequency: float
    detection_rate: float
    pulse_amp_per_supply_voltage: float
    counter_source: str
    pulse_mode: str

    @classmethod
    def from_variables(cls, variables: Any) -> "RunConfig":
        return cls(
            ex_freq=float(variables.ex_freq), ex_time=float(variables.ex_time),
            max_ions=int(variables.max_ions), vdc_min=float(variables.vdc_min),
            vdc_max=float(variables.vdc_max), v_p_min=float(variables.v_p_min),
            v_p_max=float(variables.v_p_max), pulse_fraction=float(variables.pulse_fraction),
            pulse_frequency=float(variables.pulse_frequency), detection_rate=float(variables.detection_rate),
            pulse_amp_per_supply_voltage=float(getattr(variables, "pulse_amp_per_supply_voltage", 1.0)),
            counter_source=normalize_counter_source(variables.counter_source),
            pulse_mode=str(getattr(variables, "pulse_mode", "Voltage")).strip(),
        )


def validate_run_parameters(variables: Any, conf: Mapping[str, Any]) -> RunConfig:
    """Validate cross-field and hardware limits before creating a worker."""
    try:
        run = RunConfig.from_variables(variables)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ParameterError(f"Experiment parameters are incomplete or non-numeric: {exc}") from exc
    numeric_values = (
        run.ex_freq, run.ex_time, run.vdc_min, run.vdc_max, run.v_p_min, run.v_p_max,
        run.pulse_fraction, run.pulse_frequency, run.detection_rate, run.pulse_amp_per_supply_voltage,
    )
    if not all(math.isfinite(value) for value in numeric_values):
        raise ParameterError("Experiment parameters must be finite")
    if run.ex_freq <= 0:
        raise ParameterError("Experiment frequency must be greater than zero")
    if not 0 <= run.vdc_min < run.vdc_max <= float(conf["max_vdc"]):
        raise ParameterError("Vdc must satisfy 0 <= minimum < maximum <= configured hardware maximum")
    if run.pulse_mode == "Voltage" and not (
        float(conf["min_vp"]) <= run.v_p_min < run.v_p_max <= float(conf["max_vp"])
    ):
        raise ParameterError("Pulse voltage must satisfy configured minimum <= minimum < maximum <= maximum")
    if not 0 <= run.pulse_fraction <= float(conf["pulse_fraction_max"]):
        raise ParameterError("Pulse fraction is outside the configured safe range")
    if run.pulse_frequency <= 0 or run.pulse_amp_per_supply_voltage <= 0:
        raise ParameterError("Pulse frequency and amplifier divisor must be greater than zero")
    if not 0 <= run.detection_rate <= 100:
        raise ParameterError("Detection rate must be between 0 and 100 percent")
    if bool(getattr(variables, "criteria_time", False)) and run.ex_time <= 0:
        raise ParameterError("Experiment time must be positive when the time criterion is enabled")
    if bool(getattr(variables, "criteria_ions", False)) and run.max_ions <= 0:
        raise ParameterError("Maximum ions must be positive when the ion criterion is enabled")
    variables.counter_source = run.counter_source
    return run


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
        f"Could not load electrode configuration from {path} (or legacy fallback {path.with_suffix('.json')})."
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
    apply_form_values(
        variables,
        conf,
        FormValues(
            user_name=str(item["ex_user"]), ex_name=str(item["ex_name"]), electrode=str(item["electrode"]),
            ex_time=str(item["ex_time"]), ex_freq=str(item["ex_freq"]), max_ions=str(item["max_ions"]),
            vdc_min=str(item["vdc_min"]), vdc_max=str(item["vdc_max"]),
            detection_rate_init=str(item["detection_rate_init"]), pulse_fraction=str(item["pulse_fraction"]),
            email=str(item["email"]), vdc_steps_up=str(item["vdc_steps_up"]),
            vdc_steps_down=str(item["vdc_steps_down"]), control_algorithm=str(item["control_algorithm"]),
            pulse_mode=str(item["pulse_mode"]), vp_min=str(item["vp_min"]), vp_max=str(item["vp_max"]),
            pulse_frequency=str(item["pulse_frequency"]), counter_source=str(item["counter_source"]),
            criteria_time=bool(item["criteria_time"]), criteria_ions=bool(item["criteria_ions"]),
            criteria_vdc=bool(item["criteria_vdc"]), criteria_email=bool(item.get("criteria_email", False)),
            email_interval_events=str(item.get("email_interval_events", 1000000)),
        ),
        emit_error,
    )
    variables.hit_display = int(item["hit_displayed"])
    validate_run_parameters(variables, conf)


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
    variables.counter_source = normalize_counter_source(values.counter_source)

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
    variables.criteria_email = values.criteria_email

    try:
        variables.email_interval_events = int(float(values.email_interval_events))
    except (TypeError, ValueError):
        emit_error("Email interval must be a whole number of ions")

    if getattr(variables, "email_interval_events", 0) <= 0:
        raise ParameterError("Email interval must be greater than zero")

    return corrections
