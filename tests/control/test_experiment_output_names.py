"""Experiment output names exclude electrode details kept in metadata."""

from types import SimpleNamespace

from pyccapt.control.apt import experiment_state


def test_output_folder_and_data_name_exclude_electrode(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setattr(experiment_state.runtime, "project_path", lambda *parts: data_root)
    variables = SimpleNamespace(counter=7, electrode="NiC1", hdf5_data_name="test run")

    data_path, meta_path = experiment_state.prepare_experiment_output_paths(variables)

    assert variables.exp_name.startswith("7_")
    assert variables.exp_name.endswith("_test_run")
    assert "NiC1" not in variables.exp_name
    assert data_path.name == variables.exp_name
    assert meta_path == data_path / "meta_data"
    assert variables.electrode == "NiC1"
