import numpy as np


def mc2tof(mc: float, V: float, xDet: float, yDet: float, flightPathLength: float) -> float:
    """
    Calculate time of flight (tof) based on idealized geometry and electrostatics using the formula:
    t = sqrt((mc * amu * (flightPathLength^2)) / (2eV)) * 1E9

    Args:
        mc: Mass-to-charge ratio (unit: Dalton)
        V: Voltage (unit: volts)
        xDet: Distance along the x-axis (unit: cm)
        yDet: Distance along the y-axis (unit: cm)
        flightPathLength: Length of the flight path (unit: mm)

    Returns:
        t: Time of flight (unit: ns)
    """
    xDet = xDet * 1e-2  # xDet from cm to m
    yDet = yDet * 1e-2  # yDet from cm to m
    flightPathLength = flightPathLength * 1e-3  # flightPathLength from mm to m
    e = 1.6e-19  # coulombs per electron
    amu = 1.66e-27  # conversion from kg to Dalton

    flightPathLength = np.sqrt(xDet**2 + yDet**2 + flightPathLength**2)

    t = np.sqrt((mc * amu * (flightPathLength) ** 2) / (2 * e * V))
    t = t * 1e9  # tof from s to ns

    return t


# print(mc2tof(1, 2000, 1, 1, 100))
