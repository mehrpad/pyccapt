Reflectron Batch Correction (CLI)
=================================

A command-line companion to the ``reflectron_correction.ipynb`` widget
workflow. It scans a folder for ``.epos`` files, applies the affine
reflectron-mesh correction for a chosen instrument preset to each one, and
writes a corrected ``.h5`` (and optionally ``.epos``) next to every input
file.

Source: ``pyccapt/calibration/reflectron_correction/batch_cli.py``

When to use this instead of the notebook
----------------------------------------

- You have many EPOS files in one folder (or a tree of folders) and want
  to correct them all with the same instrument preset in one pass.
- You want corrected outputs sitting next to the raw EPOS, ready to be
  loaded by the calibration tutorials as ``.h5``.

For one-off interactive exploration with detector-map previews, use the
notebook workflow at
``pyccapt/calibration/tutorials/jupyter_files/reflectron_correction.ipynb``.

Usage
-----

List the available instrument presets:

.. code-block:: bash

   python -m pyccapt.calibration.reflectron_correction.batch_cli --list-instruments

Correct every EPOS in a folder (non-recursive):

.. code-block:: bash

   python -m pyccapt.calibration.reflectron_correction.batch_cli \
       "E:/datasets/run_42" \
       --instrument 5000xr_oxford_5083_23091

Recurse into subdirectories, also write a corrected ``.epos`` next to the
``.h5``, and overwrite existing outputs:

.. code-block:: bash

   python -m pyccapt.calibration.reflectron_correction.batch_cli \
       "E:/datasets" \
       --instrument 3000xhr_leoben_21_14093 \
       --recursive --save-epos --overwrite

Remove every previously written ``*_corrected.h5`` and ``*_corrected.epos``
under a folder. Preview with ``--dry-run`` first:

.. code-block:: bash

   python -m pyccapt.calibration.reflectron_correction.batch_cli "E:/datasets" --clean --recursive --dry-run
   python -m pyccapt.calibration.reflectron_correction.batch_cli "E:/datasets" --clean --recursive

Arguments
---------

``folder``
    Folder to scan for ``.epos`` files. Omit when using
    ``--list-instruments``.

``-i, --instrument``
    Instrument preset key (e.g. ``5000xr_oxford_5083_23091``) or its
    human-readable display name (case-, space- and dash-insensitive). Run
    with ``--list-instruments`` to see the full list.

``-r, --recursive``
    Recurse into subdirectories when searching for ``.epos`` files.

``--save-epos``
    Also write a corrected ``.epos`` file next to the ``.h5`` output.

``--overwrite``
    Overwrite existing ``_corrected`` outputs (skipped by default).

``--list-instruments``
    Print the available instrument presets and exit.

``--clean``
    Delete every ``*_corrected.h5`` and ``*_corrected.epos`` file under the
    folder (respects ``--recursive``). No reflectron correction is run. Combine
    with ``--dry-run`` to preview the deletions first.

``--dry-run``
    With ``--clean``, list the files that would be removed without deleting
    them.

Running on small-RAM machines
-----------------------------

When ``--save-epos`` is *not* set, the CLI applies the reflectron correction
chunk-by-chunk via memory-mapped reads and streams the result into the
output HDF5 with ``format='table'``. Peak Python heap during correction is
bounded by one chunk (~270 MB for a 2 GB EPOS) instead of the full file,
which makes 8 GB RAM machines comfortable. The same machinery is used by
the lazy raw-data workflow; see
:doc:`low_memory_mode` for the shared design and the lazy entry points you
can call from your own scripts.

Output layout
-------------

For each input ``<name>.epos`` the script writes, in the same directory:

- ``<name>_corrected.h5`` (always)
- ``<name>_corrected.epos`` (when ``--save-epos`` is passed)

Existing outputs are skipped unless ``--overwrite`` is supplied. The
process exits with a non-zero status if any file failed.
