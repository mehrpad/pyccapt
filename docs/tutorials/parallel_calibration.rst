Parallel Execution in Calibration (``PYCCAPT_PARALLEL_WORKERS``)
================================================================

PyCCAPT now opportunistically parallelizes a handful of hot loops in the
calibration and raw-data analysis paths. The default is automatic: when a
loop has enough independent work to amortize executor startup, it runs on
multiple threads (or processes, for pure-Python kernels); otherwise it falls
back to the serial code that's always been there.

Two environment variables let you pin the behavior without changing any code
-- useful for benchmarking, reproducible runs, CI, and small-RAM machines
that can't afford a process pool.

Environment variables
---------------------

``PYCCAPT_PARALLEL_WORKERS``
    Integer. ``0`` or ``1`` forces strictly serial execution (identical to
    the pre-parallel behavior, used by the test suite). Any larger integer
    caps the worker count. Unset = auto (``cpu_count - 1`` capped at 8).

``PYCCAPT_PARALLEL_BACKEND``
    One of ``auto``, ``thread``, ``process``, ``serial``. Unset = auto.
    ``thread`` is right for NumPy/SciPy-bound kernels (GIL-releasing);
    ``process`` for pure-Python kernels; ``serial`` for debugging.

What's parallelized today
-------------------------

These are the loops where measurement has shown a real wall-clock win:

============================================================  ===========================
Hot path                                                      Measured speedup
============================================================  ===========================
``bowl_correction._collect_spatial_samples`` (polar)          ~**3x** on 5M ions / 2.6k cells
``voltage_correction`` segment loop (``ion_seq``)             ~**2.2x** on 5M ions / 500 segs
``build_surface_concept_recovery_diagnostics``                ~**1.5x** on >=1M sequences
============================================================  ===========================

Loops where parallelism was tried and *rejected* (the per-task work is too
cheap, so executor dispatch / pickling dominates):

- ``bowl_correction`` cartesian path (cell work is sub-millisecond on
  pre-sorted slices).
- Smaller Surface Concept datasets (<500 k sequence records) -- pickle
  overhead on the dict-of-lists data structure exceeds the inner compute.
- Adaptive residual calibration's outer iterations (iterations depend on
  each other; can't parallelize).
- Reflectron per-chunk correction (already vectorized; chunk parallelism
  contends with the streaming HDF5 writer).

Each of these paths has a comment in the source explaining the measurement
that led to keeping it serial. When you re-measure on different hardware
(e.g. a Linux server with fork-based multiprocessing) you may find the
break-even shifts -- the env-var override is the right knob to flip.

Usage
-----

Pin behavior in tests / CI (the test suite already does this):

.. code-block:: bash

   export PYCCAPT_PARALLEL_WORKERS=1   # serial, deterministic
   pytest tests/

Run with explicit worker count:

.. code-block:: bash

   export PYCCAPT_PARALLEL_WORKERS=8
   jupyter notebook

Force a specific backend (useful when debugging worker-vs-serial divergence):

.. code-block:: bash

   export PYCCAPT_PARALLEL_BACKEND=thread
   pytest tests/calibration/test_calibration_state.py

Python-side overrides
---------------------

Library code that wants finer control passes a :class:`ParallelConfig`:

.. code-block:: python

   from pyccapt.calibration.core.parallel import ParallelConfig, parallel_map

   results = parallel_map(
       my_kernel,
       items,
       config=ParallelConfig(backend="thread", workers=4, min_items=8),
       gil_releasing=True,
   )

``ParallelConfig.min_items`` overrides the per-backend default threshold
(``32`` for threads, ``200`` for processes). Set it explicitly when you've
already done coarse-grained batching at the caller level so each work item
represents real CPU work.

Design notes
------------

The parallelism layer lives in
``pyccapt.calibration.core.parallel``. Key invariants:

- **Backward-compat:** every caller keeps its old signature. Below the
  configured ``min_items``, ``parallel_map`` runs in serial -- identical
  results, no executor created.
- **Determinism:** results are returned in input order regardless of
  backend. Tests pin ``PYCCAPT_PARALLEL_WORKERS=1`` to reproduce serial
  output bit-for-bit.
- **Graceful degradation:** when a closure or notebook-defined function is
  handed to the auto backend with a pure-Python kernel, ``parallel_map``
  silently downgrades from processes to threads so the caller still gets a
  correct answer. Setting ``backend='process'`` explicitly raises instead --
  the caller wanted parallelism and should know it failed.
- **No nested parallelism:** if you're already inside a parallel worker,
  ``parallel_map`` still applies its thresholds; nested-pool deadlock is
  prevented by the auto-serial fallback for small workloads, not by
  explicit detection.

For the matching low-RAM / memory-mapped I/O work see
:doc:`low_memory_mode`.
