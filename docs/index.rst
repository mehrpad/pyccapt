PyCCAPT: A Modular, FAIR Open-Source Python Package for Atom Probe Control and Calibration
=============================================================================================

Historically, atom probe tomography (APT) detection systems have often relied on compiled
software with tight hardware-software co-design to handle high detector data rates.
With continued advances in compute hardware, higher-level programming approaches are now
practical for many control and analysis workflows [1].

PyCCAPT is an open-source Python package for APT experiment control, calibration, and data
processing. The package stores data in a FAIR (findable, accessible, interoperable, reusable)
HDF5-based format that can include full experiment context and detector raw data.

Documentation
=============

This documentation includes installation instructions, control and calibration modules,
configuration guidance, tutorials, and API references.

Most PyCCAPT outputs are represented as
`Pandas DataFrames <https://pandas.pydata.org/pandas-docs/stable/reference/frame.html>`_,
which helps interoperability and downstream analysis.

Control and calibration data structures are documented here:

- `Control data structure <https://github.com/mmonajem/pyccapt/blob/main/pyccapt/control/DATA_STRUCTURE.md>`_
- `Calibration data structure <https://github.com/mmonajem/pyccapt/blob/main/pyccapt/calibration/DATA_STRUCTURE.md>`_

HDF5 is used as the primary storage format because it is widely supported across languages
and can store large, heterogeneous datasets together with metadata.

Contents
========

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   control_index
   configuration
   calibration_index
   tutorials
   modules
   license
   bibliography


Bibliography
============

1. B. Gault et al., *Atom probe tomography*. Nat Rev Methods Primers 1, 52 (2021).
2. D. W. Saxey, *Correlated ion analysis and the interpretation of atom probe mass spectra*.
   Ultramicroscopy 111, 473-479 (2011).

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
