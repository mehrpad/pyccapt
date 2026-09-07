from __future__ import annotations

import json

import numpy as np

from pyccapt.control.core import chunk_store
from pyccapt.control.core.chunk_store import (
    atomic_write_chunk_group,
    validate_manifest_records,
)


def test_atomic_chunk_manifest_has_checksum_and_rows(tmp_path):
    record = atomic_write_chunk_group(
        tmp_path,
        stream_name="dld",
        chunk_id=1,
        arrays={"x": np.array([1.0, 2.0]), "y": np.array([3.0, 4.0])},
    )
    valid, invalid = validate_manifest_records(tmp_path)

    assert not invalid
    assert valid == [record]
    assert record["status"] == "complete"
    assert record["rows"] == 2
    assert len(record["fields"]["x"]["sha256"]) == 64


def test_corrupt_chunk_is_preserved_in_quarantine(tmp_path):
    record = atomic_write_chunk_group(
        tmp_path,
        stream_name="dld",
        chunk_id=7,
        arrays={"x": np.array([1.0, 2.0])},
    )
    (tmp_path / record["fields"]["x"]["file"]).write_bytes(b"corrupt")

    valid, invalid = validate_manifest_records(tmp_path, quarantine=True)

    assert not valid
    assert len(invalid) == 1
    quarantine = tmp_path / "quarantine" / "dld-7"
    assert (quarantine / "x_chunk_7.npy").is_file()
    assert (quarantine / "reason.json").is_file()


def test_legacy_atomic_manifest_remains_recoverable(tmp_path):
    values = np.arange(4, dtype=np.float64)
    np.save(tmp_path / "x_chunk_7.npy", values)
    legacy = {
        "chunk_id": 7,
        "fields": {"x": {"length": 4, "dtype": "float64"}},
    }
    (tmp_path / "manifest.jsonl").write_text(json.dumps(legacy) + "\n", encoding="utf-8")

    valid, invalid = validate_manifest_records(tmp_path)

    assert not invalid
    assert valid[0]["manifest_version"] == 0
    assert chunk_store.validated_files_for_stem(tmp_path, "x") == [tmp_path / "x_chunk_7.npy"]
