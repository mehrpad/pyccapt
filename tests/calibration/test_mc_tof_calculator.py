from types import SimpleNamespace

import ipywidgets as widgets
import numpy as np
import pytest

from pyccapt.calibration.core.exceptions import CalibrationInputError
from pyccapt.calibration.mc import mc_tools, tof_tools
from pyccapt.calibration.tutorials.tutorials_helpers.helper_mc_tof_calculator import (
    build_mc_tof_calculator_panel,
    calculate_mass_to_charge_da,
    calculate_tof_ns,
)


def _widgets_by_description(root):
    found = {}
    stack = [root]
    while stack:
        widget = stack.pop()
        description = getattr(widget, "description", "")
        if description:
            found[description] = widget
        stack.extend(getattr(widget, "children", ()))
    return found


@pytest.mark.parametrize("mass_to_charge_da", [0.0, 1.0, 2.0, 27.5])
def test_ideal_conversions_use_repository_formula_pair(mass_to_charge_da):
    voltage_v = 5000.0
    flight_path_mm = 110.0

    tof_ns = calculate_tof_ns(mass_to_charge_da, voltage_v, flight_path_mm)
    expected_tof_ns = tof_tools.mc2tof(
        mass_to_charge_da,
        voltage_v,
        xDet=0.0,
        yDet=0.0,
        flightPathLength=flight_path_mm,
    )
    assert tof_ns == pytest.approx(expected_tof_ns)

    mass_round_trip = calculate_mass_to_charge_da(
        tof_ns,
        voltage_v,
        flight_path_mm,
    )
    expected_mass = mc_tools.tof2mcSimple(
        tof_ns,
        t0=0.0,
        V=voltage_v,
        xDet=0.0,
        yDet=0.0,
        flightPathLength=flight_path_mm,
    )
    assert mass_round_trip == pytest.approx(expected_mass)
    assert mass_round_trip == pytest.approx(mass_to_charge_da)


@pytest.mark.parametrize(
    ("converter", "args"),
    [
        (calculate_tof_ns, (-1.0, 5000.0, 110.0)),
        (calculate_tof_ns, (1.0, 0.0, 110.0)),
        (calculate_tof_ns, (1.0, 5000.0, np.nan)),
        (calculate_mass_to_charge_da, (-1.0, 5000.0, 110.0)),
        (calculate_mass_to_charge_da, (100.0, np.inf, 110.0)),
        (calculate_mass_to_charge_da, (100.0, 5000.0, 0.0)),
    ],
)
def test_ideal_conversions_reject_invalid_inputs(converter, args):
    with pytest.raises(CalibrationInputError):
        converter(*args)


def test_panel_uses_shared_defaults_and_recalculates_live():
    variables = SimpleNamespace(
        flight_path_length=123.4,
        dld_high_voltage=np.array([1000.0, np.nan, -20.0, 3000.0]),
    )
    panel = build_mc_tof_calculator_panel(variables)
    fields = _widgets_by_description(panel)

    assert isinstance(fields["Flight path (mm):"], widgets.FloatText)
    assert fields["Flight path (mm):"].value == pytest.approx(123.4)
    assert fields["Voltage (V):"].value == pytest.approx(2000.0)

    fields["Voltage (V):"].value = 5000.0
    fields["Flight path (mm):"].value = 110.0
    fields["m/c (Da):"].value = 2.0
    fields["TOF (ns):"].value = 100.0

    assert float(fields["Calculated TOF (ns):"].value) == pytest.approx(
        calculate_tof_ns(2.0, 5000.0, 110.0),
        rel=1e-8,
    )
    assert float(fields["Calculated m/c (Da):"].value) == pytest.approx(
        calculate_mass_to_charge_da(100.0, 5000.0, 110.0),
        rel=1e-8,
    )


def test_panel_falls_back_when_shared_defaults_are_unavailable():
    variables = SimpleNamespace(
        flight_path_length=None,
        dld_high_voltage=np.array([0.0, np.nan]),
    )
    fields = _widgets_by_description(build_mc_tof_calculator_panel(variables))

    assert fields["Flight path (mm):"].value == pytest.approx(110.0)
    assert fields["Voltage (V):"].value == pytest.approx(1000.0)
