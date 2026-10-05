from __future__ import annotations

import os
import time
import tracemalloc

import numpy as np
import pytest

from pyccapt.calibration.core.correction_models import PolynomialCorrectionModel


pytestmark = pytest.mark.performance


@pytest.mark.skipif(
    os.environ.get("PYCCAPT_RUN_BENCHMARKS") != "1",
    reason="set PYCCAPT_RUN_BENCHMARKS=1 to enforce multi-million-ion thresholds",
)
def test_two_million_ion_correction_cpu_and_memory_regression():
    count = 2_000_000
    voltage = np.linspace(2_000.0, 12_000.0, count, dtype=np.float64)
    model = PolynomialCorrectionModel(kind="voltage").fit(
        voltage[::10_000], 1.0 + voltage[::10_000] * 1e-5
    )

    tracemalloc.start()
    started = time.perf_counter()
    factors = model.predict_factor(voltage)
    elapsed = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert factors.shape == (count,)
    assert np.isfinite(factors).all()
    assert elapsed < 8.0, f"2M correction took {elapsed:.2f}s (threshold 8s)"
    assert peak_bytes < 350 * 1024 * 1024, f"peak allocation {peak_bytes / 1024**2:.1f} MiB (threshold 350 MiB)"
