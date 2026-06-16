from types import SimpleNamespace

import numpy as np

from pyccapt.calibration.reconstructions import plot_bounds


def test_range_cube_uses_finite_plotted_mask_not_unplotted_outlier():
    variables = SimpleNamespace(
        x=np.array([0.0, 1.0, 2.0, 10_000.0]),
        y=np.array([0.0, 1.0, 2.0, -10_000.0]),
        z=np.array([0.0, 1.0, 2.0, 50_000.0]),
    )
    plotted_mask = np.array([True, True, True, False])

    cube = plot_bounds.range_cube_from_mask(variables, plotted_mask)

    assert cube[0][1] < 10.0
    assert cube[1][0] > -10.0
    assert cube[2][1] < 10.0


def test_sample_mask_treats_fraction_as_display_fraction():
    mask = np.ones(100, dtype=bool)

    sampled = plot_bounds.sample_mask(mask, 0.9, len(mask))

    assert np.count_nonzero(sampled) == 90
