"""End-to-end test for the streaming reflectron correction path.

Verifies that :func:`pyccapt.calibration.reflectron_correction.core
.correct_epos_streaming` produces an HDF5 file whose contents match what the
eager
:func:`pyccapt.calibration.reflectron_correction.core
.apply_reflectron_correction_to_ccapt` pipeline would have produced for the
same input. The streaming path is what the reflectron batch CLI uses when
``--save-epos`` is not requested, so this is the calibration round-trip that
8 GB RAM users will rely on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyccapt.calibration.data_tools import lazy_io
from pyccapt.calibration.leap_tools import ccapt_tools, leap_tools
from pyccapt.calibration.reflectron_correction import core as reflectron_core


def _write_synthetic_epos(path: Path, n: int) -> None:
    """Write a small synthetic EPOS file that exercises the same code path
    as a real LEAP export. Detector coordinates stay inside the bundled
    Erlangen mesh so the correction doesn't fall back to the KDTree edge."""
    rng = np.random.default_rng(7)
    record = np.zeros(n, dtype=lazy_io.EPOS_RECORD_DTYPE)
    record["x"] = rng.uniform(-2, 2, n).astype(np.float32)
    record["y"] = rng.uniform(-2, 2, n).astype(np.float32)
    record["z"] = rng.uniform(0, 50, n).astype(np.float32)
    record["mc"] = rng.uniform(0, 80, n).astype(np.float32)
    record["tof"] = rng.uniform(100, 2000, n).astype(np.float32)
    record["hv"] = rng.uniform(2000, 8000, n).astype(np.float32)
    record["pulse"] = rng.uniform(0, 200, n).astype(np.float32)
    # Detector coordinates in mm; correction expects mm in EPOS.
    record["det_x"] = rng.uniform(-15, 15, n).astype(np.float32)
    record["det_y"] = rng.uniform(-15, 15, n).astype(np.float32)
    record["pslep"] = rng.integers(0, 1000, n, dtype=np.uint32)
    record["ipp"] = rng.integers(0, 4, n, dtype=np.uint32)
    path.write_bytes(record.tobytes())


@pytest.fixture
def synthetic_dataset(tmp_path: Path):
    epos = tmp_path / "synth.epos"
    _write_synthetic_epos(epos, n=4096)
    mesh = reflectron_core.load_builtin_preset("4000xhr_erlangen_56_4833")
    return epos, mesh


def test_streaming_matches_eager(synthetic_dataset, tmp_path):
    epos_path, mesh = synthetic_dataset

    # Eager reference.
    raw = reflectron_core.load_epos_for_reflectron_correction(epos_path)
    eager = reflectron_core.apply_reflectron_correction_to_ccapt(raw, mesh)
    eager_sorted = eager.reset_index(drop=True)

    # Streaming output. Choose a chunk size that forces multiple passes.
    out_path = tmp_path / "corrected.h5"
    result = reflectron_core.correct_epos_streaming(epos_path, mesh, out_path, chunk_size=512)
    assert result["rows"] == len(raw)
    assert Path(result["h5"]) == out_path

    # Re-read the streamed HDF5 with the same key the CLI uses.
    streamed = pd.read_hdf(out_path, key="df")
    streamed = streamed.reset_index(drop=True)

    assert list(streamed.columns) == list(eager_sorted.columns)
    for col in eager_sorted.columns:
        np.testing.assert_allclose(
            streamed[col].to_numpy(),
            eager_sorted[col].to_numpy(),
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"Mismatch in column {col!r}",
        )


def test_streaming_supports_chunked_read_back(synthetic_dataset, tmp_path):
    """The ``format='table'`` write enables iterator-mode reads -- check that
    the round-trip-via-iterator concatenates back to the same data."""
    epos_path, mesh = synthetic_dataset
    out_path = tmp_path / "corrected.h5"
    reflectron_core.correct_epos_streaming(epos_path, mesh, out_path, chunk_size=300)

    pieces = []
    with pd.HDFStore(str(out_path), mode="r") as store:
        for chunk in store.select("df", iterator=True, chunksize=500):
            pieces.append(chunk)
    streamed_iter = pd.concat(pieces, ignore_index=True)
    streamed_direct = pd.read_hdf(out_path, key="df").reset_index(drop=True)

    for col in streamed_direct.columns:
        np.testing.assert_array_equal(streamed_iter[col].to_numpy(), streamed_direct[col].to_numpy())


def test_plain_streaming_h5_matches_eager_epos_to_ccapt(synthetic_dataset, tmp_path):
    """The plain (uncorrected) streaming converter must produce an HDF5 whose
    contents match the eager ``epos_to_ccapt`` -- this is the ``--plain-h5``
    output of the batch CLI, the uncorrected sibling of the corrected stream."""
    epos_path, _ = synthetic_dataset

    eager = ccapt_tools.epos_to_ccapt(str(epos_path)).reset_index(drop=True)

    out_path = tmp_path / "plain.h5"
    result = ccapt_tools.epos_to_ccapt_h5_streaming(epos_path, out_path, chunk_size=512)
    assert result["rows"] == len(eager)
    assert Path(result["h5"]) == out_path

    streamed = pd.read_hdf(out_path, key="df").reset_index(drop=True)
    assert list(streamed.columns) == list(eager.columns)
    for col in eager.columns:
        np.testing.assert_allclose(
            streamed[col].to_numpy(),
            eager[col].to_numpy(),
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"Mismatch in column {col!r}",
        )


def test_epos_lazy_to_ccapt_chunks_matches_eager(synthetic_dataset):
    """The streaming epos_to_ccapt helper must produce the same DataFrame
    (column-by-column, dtypes intact) as the eager :func:`epos_to_ccapt`."""
    epos_path, _ = synthetic_dataset
    eager = ccapt_tools.epos_to_ccapt(str(epos_path))
    with leap_tools.read_epos_lazy(epos_path) as table:
        chunks = list(ccapt_tools.epos_lazy_to_ccapt_chunks(table, chunk_size=777))
    streamed = pd.concat(chunks, ignore_index=True)
    assert list(streamed.columns) == list(eager.columns)
    for col in eager.columns:
        assert streamed[col].dtype == eager[col].dtype, col
        np.testing.assert_array_equal(streamed[col].to_numpy(), eager[col].to_numpy())
