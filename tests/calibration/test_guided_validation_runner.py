"""Read-only source guarantees of the optional real-data validation runner."""
import numpy as np
import pandas as pd
import pytest

from tools.validate_guided_notebook_ui import InProcessJob, validate


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


@pytest.mark.parametrize("collision", ["session.h5", "accepted.h5", "ui_transcript.txt", "summary.json"])
def test_validation_runner_refuses_hardlink_input_alias(tmp_path, collision):
    data = tmp_path / "original.h5"
    ranges = tmp_path / "reviewed_ranges.h5"
    output = tmp_path / "outputs"
    output.mkdir()
    data.write_bytes(b"source must remain unchanged")
    ranges.write_bytes(b"range source must remain unchanged")
    (output / collision).hardlink_to(data)
    with pytest.raises(ValueError, match="overwrite an input"):
        validate(data, ranges, output)
    assert data.read_bytes() == b"source must remain unchanged"
    assert ranges.read_bytes() == b"range source must remain unchanged"


@pytest.mark.parametrize("bounds", [{"mass_min": float("nan")}, {"mass_max": float("inf")},
                                    {"mass_min": 65., "mass_max": 0.}, {"mass_min": 1., "mass_max": 1.}])
def test_validation_runner_refuses_invalid_detection_window_before_reading_sources(tmp_path, bounds):
    output = tmp_path / "outputs"
    with pytest.raises(ValueError, match="Detection mass"):
        validate(tmp_path / "not-read.h5", tmp_path / "not-read-ranges.h5", output, **bounds)
    assert not output.exists()


def test_in_process_job_retains_original_error_even_when_widget_displays_it():
    job = InProcessJob()
    displayed = []
    failure = ValueError("Too many detected peaks; narrow the window")

    def operation(cancel, progress):
        raise failure

    with pytest.raises(ValueError, match="Too many detected peaks"):
        job.start(operation, on_result=lambda value: None, on_error=displayed.append)
    assert job.error is failure and displayed == [failure] and not job.running
    job.start(lambda cancel, progress: 3, on_result=lambda value: None, on_error=displayed.append)
    assert job.error is None


def test_validation_runner_retains_outside_events_and_reports_identity_tolerance_mismatch(tmp_path):
    rng = np.random.default_rng(726)
    mass = np.r_[rng.normal(26.97, .015, 400), rng.normal(28.3, .015, 400), np.full(40, 100.)]
    tof = np.sqrt(mass)*20
    tof[0] = 0.  # A recorded finite zero must survive beside positive measurements.
    frame = pd.DataFrame({"mc (Da)": mass, "mc_uc (Da)": mass, "t (ns)": tof,
        "t_c (ns)": tof, "high_voltage (V)": np.full(len(mass), 5000.),
        "x_det (cm)": np.zeros(len(mass)), "y_det (cm)": np.zeros(len(mass)),
        "event_id": np.arange(len(mass), dtype=np.uint64)+2**60})
    historical = pd.DataFrame({"ion": ["54Fe2+", "56Fe2+"], "mc": [26.97, 27.97],
        "mc_low": [26.85, 28.2], "mc_up": [27.1, 28.4], "element": [["Fe"], ["Fe"]],
        "isotope": [[54], [56]], "complex": [[1], [1]], "charge": [2, 2]})
    data, ranges = tmp_path / "data.h5", tmp_path / "ranges.h5"
    frame.to_hdf(data, key="df")
    historical.to_hdf(ranges, key="df")
    report = validate(data, ranges, tmp_path / "outputs", mass_min=26., mass_max=29.)
    assert report["events"] == 840 and report["events_in_detection_window"] == 800
    assert report["events_outside_detection_window_or_nonfinite"] == 40
    assert report["proposal_count"] == 2 and report["replayed_range_count"] == 1
    assert report["replayed_ranged_events"] == 400 and report["historical_ranged_events"] == 800
    assert report["imported_tof_bitwise_unchanged"] and report["accepted_tof_export_and_resume_exact"]
    assert report["source_files_unchanged"]
    missed = report["unmatched_historical_ranges"][0]
    assert "outside the candidate mass tolerance" in missed["reason"]
    assert missed["historical_row"] == 1 and len(missed["available_proposed_centers_Da"]) == 1
    assert .3 < missed["nearest_absolute_mass_residual_Da"] < .35
