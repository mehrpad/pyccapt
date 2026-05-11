Tutorials for APT Data Processing
=================================

This section provides step-by-step tutorials for common PyCCAPT calibration and
analysis workflows, including data processing, propagation delay correction,
visualization, and 3D reconstruction.

Additional widget-based Jupyter workflows are available under
``pyccapt/calibration/tutorials/jupyter_files`` for:

- raw detector analysis (RoentDek, Surface Concept, and LEAP/Cameca imports)
- `t0` and flight-path estimation
- Cameca RHIT/STR/HITS import
- reflectron detector correction for LEAP EPOS datasets
- TAPSim node/specimen generation

Google Colab support is currently provided for the
``data_processing.ipynb`` and ``visualization.ipynb`` notebooks.

Dataset Download
----------------

.. toctree::
   :maxdepth: 1

   download_tutorial_data.md

Tutorial Guides
---------------

.. toctree::
   :maxdepth: 2

   tutorials/data_processing
   tutorials/propagation_delay_calculation
   tutorials/visualization
   tutorials/3d_reconstruction
   tutorials/reflectron_batch_cli
   tutorials/matlab_fig_range_import
   tutorials/low_memory_mode
   tutorials/parallel_calibration
