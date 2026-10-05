from __future__ import annotations

import numpy as np


def tof2mcSimple(t: int, t0: int, V: float, xDet: int, yDet: int, flightPathLength: int) -> float:
    """
    Calculate m/c based on idealized geometry and electrostatics using the formula:
    m/c = 2eV(t/L)^2

    Args:
        t: Time (unit: ns)
        t0: Initial time (unit: ns)
        V: Voltage (unit: volts)
        xDet: Distance along the x-axis (unit: cm)
        yDet: Distance along the y-axis (unit: cm)
        flightPathLength: Length of the flight path (unit: mm)

    Returns:
        mc: Mass-to-charge ratio (unit: Dalton)
    """

    try:
        t = t - t0  # t0 correction
        t = t * 1e-9  # tof from ns to s
        xDet = xDet * 1e-2  # xDet from cm to m
        yDet = yDet * 1e-2  # yDet from cm to m
        flightPathLength = flightPathLength * 1e-3  # flightPathLength from mm to m

        e = 1.6e-19  # coulombs per electron
        amu = 1.66e-27  # conversion from kg to Dalton

        flightPathLength = xDet**2 + yDet**2 + flightPathLength**2

        mc = 2 * e * V * (t**2) / flightPathLength
        mc = mc / amu  # conversion from kg/C to Da (6.022E23 g/mol, 1.6E-19C/ec)

        return mc
    except TypeError as error:
        print(error)
        return None


def tof2mc(
    t: int, t0: int, V: float, xDet: int, yDet: int, flightPathLength: int, V_pulse: float, mode: str = 'voltage'
) -> np.ndarray | None:
    """
    Calculate m/c based on idealized geometry and electrostatics using the formula:
    m/c = 2eα(V + βV_pulse)(t/L)^2

    Args:
        t: Time (unit: ns)
        t0: Initial time (unit: ns)
        V: Voltage (unit: volts)
        V_pulse: Voltage pulse (unit: volts)
        xDet: Distance along the x-axis (unit: cm)
        yDet: Distance along the y-axis (unit: cm)
        flightPathLength: Length of the flight path (unit: mm)
        mode: Type of mode ('voltage' or 'laser'). NOTE: the 'laser' branch
            drops the alpha/beta pulse-voltage terms used by 'voltage'.

    Returns:
        mc: Mass-to-charge ratio (unit: Dalton), or None on a TypeError /
            unknown ``mode`` (the latter leaves mc unbound and is caught).
    """
    # check to see that the input are arrays
    assert isinstance(t, np.ndarray), "t must be a NumPy array"
    assert isinstance(V, np.ndarray), "V must be a NumPy array"
    assert isinstance(V_pulse, np.ndarray), "V_pulse must be a NumPy array"
    assert isinstance(xDet, np.ndarray), "xDet must be a NumPy array"
    assert isinstance(yDet, np.ndarray), "yDet must be a NumPy array"

    try:
        # The value of α is greater than one, accounting for the fact that The value of a
        # is slightly greater than one, accounting for the fact that the
        # evaporation pulse is slightly amplified due to reflections and
        # impedance mismatches along the pulse transmission line.
        alpha = 1.015
        # cThe value of b is less
        # than one, accounting for the fact that the ions field evaporate,
        # on average, not at the peak_x of the evaporation pulse, but
        # during the ascending and descending edges of the incoming
        # evaporation pulse.
        beta = 0.7

        t = t - t0  # t0 correction
        t = t * 1e-9  # tof from ns to s
        xDet = xDet * 1e-2  # xDet from cm to m
        yDet = yDet * 1e-2  # yDet from cm to m
        flightPathLength = flightPathLength * 1e-3  # flightPathLength from mm to m

        e = 1.6e-19  # coulombs per electron
        amu = 1.66e-27  # conversion from kg to Dalton

        flightPathLength = np.sqrt(xDet**2 + yDet**2 + flightPathLength**2)

        if mode == 'laser':
            mc = 2 * V * e * (t / flightPathLength) ** 2
        elif mode == 'voltage':
            mc = 2 * alpha * (V + beta * V_pulse) * e * (t / flightPathLength) ** 2
            # mc = 2 * alpha * (V) * e * (t / flightPathLength) ** 2

        mc = mc / amu  # conversion from kg/C to Da (6.022E23 g/mol, 1.6E-19C/ec)

        return mc
    except TypeError as error:
        print(error)
        return None
    except UnboundLocalError as error:
        print(error)
        return None
