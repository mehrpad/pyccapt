Import Ranging from MATLAB Atom-Probe-Toolbox ``.fig``
======================================================

The MATLAB Atom-Probe-Toolbox saves its ranging by drawing each range as a
filled area on a mass spectrum and saving the whole figure as a ``.fig`` file.
This CLI walks one or more such figures and writes a PyCCAPT-compatible
``<stem>_range.h5`` (plus a ``.csv``) next to each input — ready to be loaded
by the calibration tutorials via ``data_tools.read_range`` or picked up
automatically as a ``<dataset>_range.h5`` sibling.

Source: ``pyccapt/calibration/leap_tools/matlab_fig_range.py``

What gets extracted
-------------------

From every range area plot in the figure:

- ``XData(1)`` / ``XData(end)`` → ``mc_low`` / ``mc_up``
- ``FaceColor`` (RGB 0..1) → ``color`` (``#RRGGBB``)
- ``UserData.chargeState`` → ``charge``
- ``DisplayName`` (e.g. ``"56Fe +"`` or ``"12C 16O2 +2"``) →
  ``element`` / ``isotope`` / ``complex`` / ``name`` / ``raw_comp``

The detection rule is structural: a range is any ``specgraph.areaseries``
whose ``UserData`` exposes a ``chargeState`` field. The toolbox's
``UserData.plotType`` and ``UserData.ion`` fields are MATLAB MCOS opaque
objects and are intentionally **not** decoded; the same information is
available in plain text from ``DisplayName``.

Usage
-----

Convert a single figure (writes ``<stem>_range.h5`` next to the input):

.. code-block:: bash

   python -m pyccapt.calibration.leap_tools.matlab_fig_range \
       "E:/datasets/A718_08676/Massspectrum.fig"

Walk a folder of datasets, converting every ``.fig`` found:

.. code-block:: bash

   python -m pyccapt.calibration.leap_tools.matlab_fig_range \
       "E:/datasets" --recursive

Restrict to a specific filename (e.g. only the "pretty" figures):

.. code-block:: bash

   python -m pyccapt.calibration.leap_tools.matlab_fig_range \
       "E:/datasets" --recursive --pattern "Massspectrum_pretty.fig"

Rename / relocate the output for a single figure so it sits next to your
processed dataset:

.. code-block:: bash

   python -m pyccapt.calibration.leap_tools.matlab_fig_range \
       "E:/datasets/A718_08676/Massspectrum.fig" \
       --output "E:/datasets/A718_08676/R56_08676_range.h5"

Arguments
---------

``path``
    Path to a single ``.fig`` file or a folder containing them.

``-r, --recursive``
    Recurse into subdirectories when ``path`` is a folder.

``--pattern``
    Glob pattern for figures when ``path`` is a folder (default ``*.fig``).

``-o, --output``
    Explicit output ``.h5`` path. Only valid when ``path`` is a single
    ``.fig`` file. When omitted, writes ``<stem>_range.h5`` next to each
    input.

``--no-csv``
    Skip writing the sibling ``.csv`` companion.

``--overwrite``
    Overwrite existing ``<stem>_range.h5`` outputs (default: skip if present).

``-v, --verbose``
    Print full Python tracebacks for any figure that fails to convert
    (default: one-line error).

Limitations
-----------

- Reads MATLAB ``.fig`` files saved with the default v5/v6/v7 (binary MAT)
  format. Figures saved as ``-v7.3`` (HDF5-backed) are not yet supported; if
  you have one, resave it from MATLAB with ``saveas(gcf, 'name.fig')`` or
  ``savefig('name.fig', '-v7')`` first.
- ``vol`` is set to ``0.0`` (the toolbox stores ``0`` by default too); fill
  it in from your dataset's range editor if you need it for concentration
  calculations.
