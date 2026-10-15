"""Read-only source guarantees of the optional real-data validation runner."""
import pytest

from tools.validate_guided_notebook_ui import validate


@pytest.mark.parametrize("collision", ["session.h5", "accepted.h5"])
def test_validation_runner_refuses_input_output_path_collision(tmp_path, collision):
    data = tmp_path / collision
    ranges = tmp_path / "reviewed_ranges.h5"
    data.write_bytes(b"source must remain unchanged")
    ranges.write_bytes(b"range source must remain unchanged")
    with pytest.raises(ValueError, match="overwrite an input"):
        validate(data, ranges, tmp_path)
    assert data.read_bytes() == b"source must remain unchanged"
    assert ranges.read_bytes() == b"range source must remain unchanged"
    assert not (tmp_path / "ui_transcript.txt").exists()
