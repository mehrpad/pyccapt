import numpy as np
import pandas as pd
import pytest

from pyccapt.calibration.reflectron_correction import core as reflectron_core


@pytest.mark.calibration
@pytest.mark.parametrize("preset_key", sorted(reflectron_core.BUILTIN_REFLECTRON_PRESETS))
def test_builtin_reflectron_presets_match_matlab_reference(preset_key: str):
    mesh = reflectron_core.load_builtin_preset(preset_key)
    reference_filename = mesh.preset.source_mat_file.replace("_intersections.mat", "_reference.csv")
    reference_path = reflectron_core.DATA_DIR / reference_filename
    assert reference_path.is_file(), f"Missing MATLAB reference export: {reference_path}"

    reference = pd.read_csv(reference_path)
    corrected_detx, corrected_dety = reflectron_core.correct_detector_coordinates(
        reference["input_detx"].to_numpy(),
        reference["input_dety"].to_numpy(),
        mesh,
    )

    np.testing.assert_allclose(
        corrected_detx,
        reference["corrected_detx"].to_numpy(),
        rtol=0.0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        corrected_dety,
        reference["corrected_dety"].to_numpy(),
        rtol=0.0,
        atol=1e-9,
    )
