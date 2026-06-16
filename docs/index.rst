PyCCAPT: A Modular, FAIR Open-Source Python Package for Atom Probe Control and Calibration
=============================================================================================

PyCCAPT is an open-source Python package for atom probe tomography (APT) experiment control,
calibration, reconstruction, and data processing.

The project follows FAIR data principles and uses an HDF5-based storage model that can include
experiment context, detector streams, and calibration outputs in a single interoperable format.

Recommended setup starts with a conda environment and a pip install:

.. code-block:: bash

   conda create -n pyccapt python=3.11
   conda activate pyccapt
   python -m pip install --upgrade pip
   pip install "pyccapt[full]"

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

.. toctree::
   :hidden:

   README
   appendix


Citation
========

If you use PyCCAPT in your work, please cite the paper :cite:`monajem2025pyccapt`.
The complete list of references is on the :doc:`bibliography` page, and
machine-readable citation metadata is provided in ``CITATION.cff`` at the
repository root.

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
