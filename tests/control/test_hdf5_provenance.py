from __future__ import annotations

import json
from types import SimpleNamespace

import h5py

from pyccapt.control.core.data_integrity import validate_hdf5
from pyccapt.control.core.hdf5_creator import hdf_creator


def test_hdf5_records_schema_units_hashes_and_model_provenance(tmp_path):
    variables = SimpleNamespace(
        path=str(tmp_path),
        exp_name="provenance:test",
        main_counter=[1, 2],
        main_raw_counter=[4, 8],
        main_temperature=[45.0, 46.0],
        main_chamber_vacuum=[1e-8, 2e-8],
        excluded_row_count=3,
        dataset=SimpleNamespace(source_hash="a" * 64),
        calibration_model_provenance={"model_type": "golden", "parameters": [1, 2]},
    )
    conf = {"tdc": "off", "v_dc": "off"}
    hdf_creator(variables, conf, [0, 1], [0.0, 0.1])
    output = tmp_path / "provenance_test.h5"

    with h5py.File(output, "r") as handle:
        provenance = handle["provenance"].attrs
        assert provenance["schema_version"] == "2.0"
        assert len(provenance["control_config_sha256"]) == 64
        assert provenance["calibration_input_sha256"] == "a" * 64
        assert provenance["excluded_row_count"] == 3
        assert json.loads(provenance["model_provenance_json"])["model_type"] == "golden"
        assert handle["apt/temperature"].attrs["units"] == "K"
        assert handle["apt/id"].attrs["units"] == "1"

    result = validate_hdf5(output)
    assert result["valid"], result["issues"]
    assert result["schema_version"] == "2.0"
