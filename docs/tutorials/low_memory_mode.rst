Working with Big Datasets on Small RAM (Memory-Mapped I/O)
==========================================================

PyCCAPT now ships a shared, memory-mapped I/O layer that lets the calibration
workflows handle multi-gigabyte EPOS, POS, and pyccapt-raw HDF5 files on
machines with as little as 8 GB of RAM. The original code paths
``.to_numpy()``-ed every column into RAM up front -- a 2 GB EPOS would briefly
pin ~7 GB before the DataFrame was even built. The lazy abstraction below
slices and streams those columns on demand instead.

The shared abstraction: ``LazyTable`` / ``LazyColumn``
------------------------------------------------------

Module: ``pyccapt.calibration.data_tools.lazy_io``.

- :class:`LazyColumn` wraps either a NumPy ``memmap`` field view (for binary
  EPOS / POS records) or an ``h5py.Dataset`` (for pyccapt-raw HDF5). It
  exposes a NumPy-compatible API (``__len__``, slicing, ``__array__``,
  ``to_numpy``, ``.dtype``) so any function that today accepts an ``ndarray``
  also accepts a ``LazyColumn``.
- :class:`LazyTable` is a column-oriented view with two iteration helpers:

  - ``table.iter_chunks(columns, chunk_size)`` -- yields ``{name: ndarray}``
    dicts of at most ``chunk_size`` rows. The peak resident memory is bounded
    by ``chunk_size`` no matter how big the file is.
  - ``table.to_dataframe(columns)`` -- materialize the whole table when you
    really do want a DataFrame; the construction is chunked so the peak is
    bounded too.
- Chunked accumulators ``chunked_histogram(column, bins, ...)``,
  ``chunked_min_max(column)``, and ``chunked_apply(table, fn, columns, ...)``
  are drop-in replacements for the eager NumPy equivalents.

Loaders for the three main raw formats:

.. code-block:: python

   from pyccapt.calibration.data_tools import lazy_io

   table = lazy_io.open_epos("path/to/dataset.epos")        # memmap-backed
   table = lazy_io.open_pos("path/to/dataset.pos")          # memmap-backed
   table = lazy_io.open_pyccapt_raw_hdf5("dataset_pyccapt-raw.h5")  # h5py-backed

Always prefer ``with ... as table: ...`` so the underlying memmap or HDF5
file handle is released promptly on Windows.

Backwards-compatible entry points
---------------------------------

The eager APIs every tutorial uses today still return the same materialized
DataFrames as before. Opt into the lazy path explicitly:

.. code-block:: python

   from pyccapt.calibration.leap_tools import leap_tools
   from pyccapt.calibration.data_tools import data_tools, data_loadcrop

   # EPOS / POS
   df    = leap_tools.read_epos(path)            # eager (DataFrame)
   table = leap_tools.read_epos_lazy(path)       # lazy   (LazyTable)
   table = leap_tools.read_pos_lazy(path)

   # Generic pyccapt-raw HDF5
   data  = data_tools.read_hdf5(path)            # eager
   table = data_tools.read_hdf5(path, lazy=True) # lazy

   # /dld or /tdc group of a pyccapt-raw HDF5
   df    = data_loadcrop.fetch_dataset_from_dld_grp(path, extract_mode="dld")
   table = data_loadcrop.fetch_dataset_from_dld_grp(
       path, extract_mode="dld", lazy=True,
   )

The eager ``read_epos`` was also refactored to stop the per-column
``np.asarray`` cascade that used to materialize 11 byte-swap copies of the
file in series; even users who never touch the lazy API see a halved peak
working set for free.

Streaming reflectron correction
-------------------------------

The reflectron batch CLI now applies its mesh correction one chunk at a time
when ``--save-epos`` is not requested. The corrected output is written to
HDF5 via ``pd.HDFStore.append`` with ``format='table'`` so peak RAM during
correction is bounded by one chunk regardless of the EPOS size.

.. code-block:: bash

   # 2 GB EPOS, 8 GB RAM box: streams comfortably
   python -m pyccapt.calibration.reflectron_correction.batch_cli \
       E:/path/to/dataset/ --instrument 4000xhr_erlangen_56_4833 --recursive

The same ``format='table'`` output lets the calibration tutorials read the
file back in chunks later:

.. code-block:: python

   import pandas as pd
   with pd.HDFStore("R56_xxx_corrected.h5", mode="r") as store:
       for chunk in store.select("df", iterator=True, chunksize=500_000):
           ...  # process chunk

If you also need a corrected ``.epos`` (``--save-epos``), the path falls back
to the eager loader because the binary EPOS writer needs the whole frame in
memory.

Programmatic call:

.. code-block:: python

   from pyccapt.calibration.reflectron_correction import core as rcore
   mesh = rcore.load_builtin_preset("4000xhr_erlangen_56_4833")
   rcore.correct_epos_streaming(
       epos_path="big.epos",
       mesh=mesh,
       h5_output_path="big_corrected.h5",
       chunk_size=1 << 20,   # ~1M rows per chunk (~60 MB)
   )

Raw-data analysis: the "Low memory" toggle
------------------------------------------

The raw-data widget exposes a **Low memory** checkbox in the *Plot settings*
row (notebook:
``pyccapt/calibration/tutorials/jupyter_files/raw_data_analysis.ipynb``).
When ticked, the 3DL pipeline opens the pyccapt-raw HDF5 lazily via
``fetch_dataset_from_dld_grp(..., lazy=True)`` and the diagnostic helpers
route through the streaming code paths:

- ``summarize_processed_dataset(...)`` -- chunked min/max plus a
  weighted-median-of-per-chunk-medians estimator.
- ``compute_tof_segment_drift(...)`` -- reads only the current segment's TOF
  column slice from disk.
- ``plot_processed_dataset_overview(...)`` -- TOF and mass-spectrum
  histograms accumulate chunk-by-chunk; detector heatmap is a streaming 2-D
  histogram; experiment-history line plots are evenly down-sampled
  (default 100k points).

When **Low memory** is enabled, ``compute_tof_segment_drift`` requires an
explicit ``windows=...`` list (auto-peak detection would need a separate
full pre-pass over the column).

Measured impact
---------------

End-to-end benchmarks on real files. Peak Python heap measured with
``tracemalloc``; OS file-cache pages from the memmap are excluded because
they are reclaimable.

==================================================  ===========================  =================
Operation                                           Eager (before)               Lazy (now)
==================================================  ===========================  =================
``read_epos`` of a 1.2 GB EPOS (no other ops)       ~7 GB peak heap              ~1.5 GB peak heap
``chunked_histogram`` over a 1.2 GB EPOS column     n/a (eager: ~5 GB)           **8 MB** peak
Reflectron correction of a 1.2 GB EPOS              ~3+ GB peak, OOM at 8 GB     **268 MB** peak, 30 s
``summarize_processed_dataset`` of 5 M rows         ~200 MB DataFrame + copies   **29 MB** peak
``plot_processed_dataset_overview`` of 5 M rows     ~1 GB during plotting        **83 MB** peak
==================================================  ===========================  =================

Tuning tips
-----------

- ``chunk_size`` defaults to ``1 << 20`` rows (~1 M rows, ~60 MB for EPOS).
  That fits in L3 cache on most workstations. If your machine has less L3
  cache or you want even lower RAM, drop it to ``1 << 18`` (~256 k rows).
- Lazy tables hold an open file handle. On Windows, prefer
  ``with table: ...`` so the handle is released eagerly -- otherwise the
  file can stay locked until the next garbage-collection pass.
- After lazy correction, the ``format='table'`` HDF5 outputs are slightly
  slower to bulk-read than the older ``format='fixed'`` outputs (~5%). The
  trade is worth it because the same file is now streamable; if you have a
  workflow that always wants the whole frame, pass ``format='fixed'`` to
  ``data_tools.store_df_to_hdf``.

Extending the lazy paths
------------------------

To add lazy support to another helper, follow the pattern from
``_raw_workflow_common.py``:

.. code-block:: python

   from pyccapt.calibration.data_tools import lazy_io

   def my_stat(source):
       # Accept DataFrame or LazyTable.
       if isinstance(source, lazy_io.LazyTable):
           # Use iter_chunks([...]) to bound RAM.
           for chunk in source.iter_chunks(["mc (Da)"], chunk_size=1 << 20):
               accumulate(chunk["mc (Da)"])
       else:
           accumulate(source["mc (Da)"].to_numpy(dtype=float))

Or expose just a single column and use ``chunked_histogram`` /
``chunked_min_max`` directly.
