"""Real notebook raw-data callback replay, without fabricated calibration inputs."""
import h5py
import numpy as np
import pytest

from tools.validate_raw_guided_workflow import OUTPUTS, file_sha256, validate


def raw_source(path, n=240):
    i = np.arange(n, dtype=float)
    with h5py.File(path, "w") as handle:
        for name, values in {
            "t": 350.+i*.1, "x": .1*np.sin(i), "y": .2*np.cos(i),
            "high_voltage": 5000.+i, "voltage_pulse": 600.+i*.01,
            "laser_pulse": np.zeros(n),
            # Values above float64 integer precision must survive exactly.
            "start_counter": np.arange(n, dtype=np.uint64)+2**60,
        }.items():
            handle.create_dataset("dld/"+name, data=values)
        handle.create_dataset("tdc/time_data", data=np.zeros(n*4))


def test_raw_runner_no_mass_or_instrument_inputs_preserves_full_source_and_crop(tmp_path):
    source, output = tmp_path / "raw.h5", tmp_path / "results"
    raw_source(source)
    digest = file_sha256(source)
    report = validate(source, output, event_start=10, event_stop=230)
    assert report["status"] == "passed" and report["source_file_unchanged"]
    assert report["source_events"] == 240 and report["selected_events"] == 220
    assert report["session_resume_verified"] and report["full_loaded_source_retained"]
    assert report["imported_mass_disabled"] and report["unconfirmed_physical_conversion_refused"]
    assert not report["physical_conversion_requested"] and not (output / "accepted.h5").exists()
    assert report["source_measurements"]["columns"]["start_counter"]["source_dtype"] == "uint64"
    assert all(row["all_rows_equal"] for row in report["source_measurements"]["columns"].values())
    assert file_sha256(source) == digest


@pytest.mark.parametrize("mode", ["laser", "voltage"])
def test_raw_runner_optional_explicit_physical_conversion_and_embedded_resume(tmp_path, mode):
    source, output = tmp_path / "raw.h5", tmp_path / "results"
    raw_source(source)
    with h5py.File(source, "r+") as handle:
        handle["dld/t"][6:9] = [38., 20., np.nan]
    report = validate(source, output, event_start=5, event_stop=235, flight_mm=110., t0_ns=38., pulse_mode=mode)
    assert report["physical_conversion"]["public_converter_exact_parity"]
    assert report["physical_conversion"]["finite_mass_events"] == 227
    assert report["physical_conversion"]["selected_events_without_valid_mass"] == 3
    assert report["physical_conversion"]["selected_recorded_tof_at_or_below_t0"] == 2
    assert report["physical_conversion"]["selected_nonfinite_tof"] == 1
    assert report["embedded_session_verified"] and report["session_resume_verified"]
    assert report["export_selection"]["preserves_full_source"] is False
    assert report["export_selection"]["source_events"] == 240
    # Independent energy/flight-path expression checks actual accepted output.
    from pyccapt.calibration.core.guided_workflow import GuidedWorkflow
    restored = GuidedWorkflow.load(output / "session.h5")
    frame = restored.source
    voltage = frame["high_voltage (V)"].to_numpy()
    if mode == "voltage":
        voltage = 1.015*(voltage+.7*frame["pulse_v (V)"].to_numpy())
    length_squared = .110**2+(frame["x_det (cm)"].to_numpy()*.01)**2+(frame["y_det (cm)"].to_numpy()*.01)**2
    expected = (2*1.6e-19/1.66e-27)*voltage*((frame["t (ns)"].to_numpy()-38.)*1e-9)**2/length_squared
    expected[6:9] = np.nan
    np.testing.assert_allclose(restored.variables.mc[5:235], expected[5:235], rtol=1e-14)
    assert np.isnan(restored.variables.mc[:5]).all() and np.isnan(restored.variables.mc[235:]).all()


@pytest.mark.parametrize("settings", [{"flight_mm": 110.}, {"t0_ns": 38.}, {"pulse_mode": "laser"}])
def test_raw_runner_refuses_partial_settings_before_io(tmp_path, settings):
    with pytest.raises(ValueError, match="all of"):
        validate(tmp_path / "unread.h5", tmp_path / "outputs", **settings)
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("name", OUTPUTS)
@pytest.mark.parametrize("hardlink", [False, True])
def test_raw_runner_refuses_input_collision_before_any_write(tmp_path, name, hardlink):
    source = tmp_path / ("original.h5" if hardlink else name)
    source.write_bytes(b"untouched acquisition")
    if hardlink:
        (tmp_path / name).hardlink_to(source)
    with pytest.raises(ValueError, match="overwrite an input"):
        validate(source, tmp_path)
    assert source.read_bytes() == b"untouched acquisition"


@pytest.mark.parametrize("name", OUTPUTS)
def test_raw_runner_refuses_stale_outputs_before_any_callback_or_write(tmp_path, name):
    source, output = tmp_path / "raw.h5", tmp_path / "results"
    raw_source(source)
    output.mkdir()
    stale = output / name
    stale.write_bytes(b"old result must not count as current success")
    digest = file_sha256(source)
    with pytest.raises(ValueError, match="fresh output directory"):
        validate(source, output)
    assert stale.read_bytes() == b"old result must not count as current success"
    assert len(list(output.iterdir())) == 1
    assert file_sha256(source) == digest
