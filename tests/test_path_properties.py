from __future__ import annotations

import random
import string

import pytest

from pyccapt.calibration.path_utils import build_output_path
from pyccapt.control.core.hdf5_creator import _sanitize_for_path


def test_random_experiment_names_never_retain_windows_illegal_characters():
    rng = random.Random(20260903)
    alphabet = string.ascii_letters + string.digits + '<>:"/\\|?* .'
    for _ in range(500):
        value = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 80)))
        sanitized = _sanitize_for_path(value)
        assert sanitized
        assert not any(character in sanitized for character in '<>:"/\\|?*')
        assert not sanitized.endswith((" ", "."))


def test_output_path_rejects_blank_names(tmp_path):
    for value in ("", " ", "\t"):
        with pytest.raises(ValueError):
            build_output_path(tmp_path, value)
