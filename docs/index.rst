PyCCAPT: A Modular, FAIR Open-Source Python Package for Atom Probe Control and Calibration
=============================================================================================

PyCCAPT is an open-source Python package for atom probe tomography (APT) experiment control,
calibration, reconstruction, and data processing.

The project follows FAIR data principles and uses an HDF5-based storage model that can include
experiment context, detector streams, and calibration outputs in a single interoperable format.

Documentation
=============

This documentation covers:

- installation and environment setup
- control runtime architecture and configuration
- calibration workflows and data structures
- tutorials for common processing pipelines
- API reference pages

Most PyCCAPT tabular outputs are represented as
`Pandas DataFrames <https://pandas.pydata.org/pandas-docs/stable/reference/frame.html>`_,
which supports integration with scientific Python workflows.

Control and calibration data structures are documented here:

- `Control data structure <https://github.com/mmonajem/pyccapt/blob/main/pyccapt/control/DATA_STRUCTURE.md>`_
- `Calibration data structure <https://github.com/mmonajem/pyccapt/blob/main/pyccapt/calibration/DATA_STRUCTURE.md>`_

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

Citation
========

If you use PyCCAPT in your work, please cite:

.. code-block:: bibtex

   @article{monajem2025pyccapt,
     title={PyCCAPT: A Python Package for Open-Source Atom Probe Instrument Control and Data Calibration},
     author={Monajem, Mehrpad and Ott, Benedict and Heimerl, Jonas and Meier, Stefan and Hommelhoff, Peter and Felfer, Peter},
     journal={Microscopy Research and Technique},
     volume={88},
     number={12},
     pages={3199--3210},
     year={2025},
     publisher={Wiley Online Library}
   }

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
