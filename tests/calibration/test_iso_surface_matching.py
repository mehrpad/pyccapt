import numpy as np
import pandas as pd
import pyvista as pv

from pyccapt.calibration.core.share_variables import Variables
from pyccapt.calibration.reconstructions import iso_surface


def test_build_element_mask_supports_pure_element_only():
    variables = Variables()
    variables.mc = np.array([10.0, 15.0, 20.0, 25.0])
    variables.range_data = pd.DataFrame(
        {
            'mc_low': [9.0, 19.0],
            'mc_up': [16.0, 26.0],
            'element': [['Al'], ['Al', 'O']],
        }
    )

    broad_mask = iso_surface.build_element_mask(variables, 'Al')
    pure_mask = iso_surface.build_element_mask(variables, 'Al', pure_only=True)

    assert broad_mask.tolist() == [True, True, True, True]
    assert pure_mask.tolist() == [True, True, False, False]


def test_calculate_iso_value_ignores_zero_background():
    conc = np.zeros((10, 10, 10), dtype=float)
    conc[3:7, 3:7, 3:7] = 0.2
    conc[4:6, 4:6, 4:6] = 0.5

    iso_value = iso_surface.calculate_iso_value(conc)

    assert 0.0 < iso_value <= 0.5


def test_specimen_envelope_clip_removes_mesh_outside_sample():
    grid_vec = [np.arange(5, dtype=float), np.arange(5, dtype=float), np.arange(5, dtype=float)]
    vox = np.zeros((5, 5, 5), dtype=int)
    vox[1:3, 1:3, 1:3] = 1
    outside_mesh = pv.Cube(bounds=(4.2, 4.8, 4.2, 4.8, 4.2, 4.8)).triangulate()
    specimen_surface = iso_surface._build_specimen_surface_from_voxels(grid_vec, vox, smoothing_sigma=1.0)

    assert specimen_surface.n_points > 0
    clipped = iso_surface._clip_isosurface_to_specimen_envelope(
        outside_mesh,
        grid_vec,
        vox,
        smoothing_sigma=1.0,
    )

    assert clipped.n_points == 0
