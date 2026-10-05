"""Interactive ideal mass-to-charge / time-of-flight calculator."""

from __future__ import annotations

import html

import ipywidgets as widgets
import numpy as np

from pyccapt.calibration.core.exceptions import CalibrationInputError
from pyccapt.calibration.core.validation import ensure_positive
from pyccapt.calibration.mc import mc_tools, tof_tools


DEFAULT_FLIGHT_PATH_MM = 110.0
DEFAULT_VOLTAGE_V = 1000.0
_MAX_VOLTAGE_SAMPLES = 100_000


def _finite_value(value: float, *, field_name: str, allow_zero: bool = False) -> float:
    """Return a validated finite numeric value."""
    numeric = ensure_positive(value, field_name=field_name, allow_zero=allow_zero)
    if not np.isfinite(numeric):
        raise CalibrationInputError(f"{field_name!r} must be finite, got {numeric}")
    return numeric


def calculate_tof_ns(
    mass_to_charge_da: float,
    voltage_v: float,
    flight_path_length_mm: float,
) -> float:
    """Convert ideal mass-to-charge (Da) to TOF (ns) at detector centre."""
    mass_to_charge_da = _finite_value(
        mass_to_charge_da,
        field_name="Mass-to-charge",
        allow_zero=True,
    )
    voltage_v = _finite_value(voltage_v, field_name="Voltage")
    flight_path_length_mm = _finite_value(
        flight_path_length_mm,
        field_name="Flight path",
    )

    tof_ns = tof_tools.mc2tof(
        mass_to_charge_da,
        voltage_v,
        xDet=0.0,
        yDet=0.0,
        flightPathLength=flight_path_length_mm,
    )
    return float(tof_ns)


def calculate_mass_to_charge_da(
    tof_ns: float,
    voltage_v: float,
    flight_path_length_mm: float,
) -> float:
    """Convert ideal TOF (ns) to mass-to-charge (Da) at detector centre."""
    tof_ns = _finite_value(tof_ns, field_name="Time of flight", allow_zero=True)
    voltage_v = _finite_value(voltage_v, field_name="Voltage")
    flight_path_length_mm = _finite_value(
        flight_path_length_mm,
        field_name="Flight path",
    )

    mass_to_charge_da = mc_tools.tof2mcSimple(
        tof_ns,
        t0=0.0,
        V=voltage_v,
        xDet=0.0,
        yDet=0.0,
        flightPathLength=flight_path_length_mm,
    )
    return float(mass_to_charge_da)


def _default_flight_path_mm(variables) -> float:
    value = getattr(variables, "flight_path_length", None)
    try:
        return _finite_value(value, field_name="Flight path")
    except CalibrationInputError:
        return DEFAULT_FLIGHT_PATH_MM


def _default_voltage_v(variables) -> float:
    """Use a representative dataset voltage without scanning huge arrays."""
    raw_voltage = getattr(variables, "dld_high_voltage", None)
    try:
        voltage = np.asarray(raw_voltage, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return DEFAULT_VOLTAGE_V

    if voltage.size > _MAX_VOLTAGE_SAMPLES:
        sample_indices = np.linspace(
            0,
            voltage.size - 1,
            num=_MAX_VOLTAGE_SAMPLES,
            dtype=np.intp,
        )
        voltage = voltage[sample_indices]

    valid_voltage = voltage[np.isfinite(voltage) & (voltage > 0)]
    if valid_voltage.size == 0:
        return DEFAULT_VOLTAGE_V
    return float(np.median(valid_voltage))


def _format_result(value: float) -> str:
    return f"{value:.9g}"


def build_mc_tof_calculator_panel(variables) -> widgets.VBox:
    """Build a live calculator panel using the dataset flight-path default."""
    description_style = {"description_width": "initial"}
    input_layout = widgets.Layout(width="245px")
    result_layout = widgets.Layout(width="265px")
    section_layout = widgets.Layout(
        border="1px solid #d9d9d9",
        padding="10px",
        width="48%",
    )

    voltage = widgets.FloatText(
        value=_default_voltage_v(variables),
        description="Voltage (V):",
        style=description_style,
        layout=input_layout,
    )
    flight_path = widgets.FloatText(
        value=_default_flight_path_mm(variables),
        description="Flight path (mm):",
        style=description_style,
        layout=input_layout,
    )
    mass_to_charge = widgets.FloatText(
        value=1.0,
        description="m/c (Da):",
        style=description_style,
        layout=input_layout,
    )
    tof = widgets.FloatText(
        value=100.0,
        description="TOF (ns):",
        style=description_style,
        layout=input_layout,
    )
    tof_result = widgets.Text(
        description="Calculated TOF (ns):",
        disabled=True,
        style=description_style,
        layout=result_layout,
    )
    mass_to_charge_result = widgets.Text(
        description="Calculated m/c (Da):",
        disabled=True,
        style=description_style,
        layout=result_layout,
    )
    status = widgets.HTML()

    def update_results(_change=None):
        errors = []
        try:
            tof_result.value = _format_result(
                calculate_tof_ns(
                    mass_to_charge.value,
                    voltage.value,
                    flight_path.value,
                )
            )
        except (CalibrationInputError, ArithmeticError) as exc:
            tof_result.value = "--"
            errors.append(str(exc))

        try:
            mass_to_charge_result.value = _format_result(
                calculate_mass_to_charge_da(
                    tof.value,
                    voltage.value,
                    flight_path.value,
                )
            )
        except (CalibrationInputError, ArithmeticError) as exc:
            mass_to_charge_result.value = "--"
            errors.append(str(exc))

        unique_errors = list(dict.fromkeys(errors))
        if unique_errors:
            message = "<br>".join(html.escape(error) for error in unique_errors)
            status.value = f'<span style="color:#b00020">{message}</span>'
        else:
            status.value = ""

    for input_widget in (voltage, flight_path, mass_to_charge, tof):
        input_widget.observe(update_results, names="value")

    update_results()

    conditions = widgets.VBox(
        [
            widgets.HTML("<b>Shared calculation conditions</b>"),
            widgets.HBox([voltage, flight_path]),
            widgets.HTML(
                "<small>Flight path starts from the value selected above in the "
                "notebook. Voltage starts from the positive dataset median when "
                "available (otherwise 1000 V).</small>"
            ),
        ]
    )
    mass_to_tof = widgets.VBox(
        [
            widgets.HTML("<b>m/c &rarr; TOF</b>"),
            mass_to_charge,
            tof_result,
        ],
        layout=section_layout,
    )
    tof_to_mass = widgets.VBox(
        [
            widgets.HTML("<b>TOF &rarr; m/c</b>"),
            tof,
            mass_to_charge_result,
        ],
        layout=section_layout,
    )

    return widgets.VBox(
        [
            widgets.HTML("<h4>Ideal m/c &amp; TOF calculator</h4>"),
            conditions,
            widgets.HBox(
                [mass_to_tof, tof_to_mass],
                layout=widgets.Layout(justify_content="space-between", width="100%"),
            ),
            status,
            widgets.HTML(
                "<small>Uses the repository's ideal electrostatic conversion at "
                "the detector centre (x = y = 0), with t0 = 0 and no pulse-voltage "
                "correction.</small>"
            ),
        ],
        layout=widgets.Layout(width="100%"),
    )


__all__ = [
    "build_mc_tof_calculator_panel",
    "calculate_mass_to_charge_da",
    "calculate_tof_ns",
]
