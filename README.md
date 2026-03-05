# PyCCAPT (APT_PyControl)

# A modular, FAIR open-source Python package for atom probe tomography control and data calibration

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10210507.svg)](https://doi.org/10.5281/zenodo.10210507)
[![Documentation Status](https://readthedocs.org/projects/pyccapt/badge/?version=latest)](https://pyccapt.readthedocs.io/en/latest/?badge=latest)
<!--[![coverage report](https://gitlab.com/jesseds/apav/badges/master/coverage.svg)](https://gitlab.com/jesseds/apav/commits/master)
[![pipeline status](https://gitlab.com/jesseds/apav/badges/master/pipeline.svg)](https://gitlab.com/jesseds/apav/-/commits/master)-->

<img align="right" src="https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/logo2.png?raw=True" alt="PyCCAPT logo" width="300" height="300">

PyCCAPT is an open-source software package for controlling atom probe systems and calibrating APT data.
It is modular and adaptable to a wide range of devices used in atom probe instrumentation. The package
currently supports data acquisition from Surface Concept and RoentDek TDC systems.

The calibration module includes key workflows such as t<sub>0</sub> and flight-path estimation, region of
interest (ROI) selection, voltage and bowl calibration, and 3D reconstruction.

----------

# Overview

PyCCAPT was initially developed and tested on the OXCART atom probe, an in-house instrument at the
Department of Materials Science and Engineering, University of Erlangen-Nuremberg. OXCART uses a
titanium-based chamber designed for ultra-low hydrogen vacuum conditions and a detector with
approximately 80% detection efficiency. Although originally developed for OXCART, PyCCAPT is designed
to be adaptable to other atom probe systems.

![](https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/readme_images/oxcart.jpg?raw=True)

The package architecture is intentionally modular, making integration with new instruments easier.
Current integrations include devices such as Pfeiffer gauges, Fug power supplies, Siglent signal
generators, and both Surface Concept and RoentDek TDC systems.

The PyCCAPT package forms the foundation of a fully FAIR atom probe data collection and processing chain. This
repository includes the graphical user interface (GUI) and control program, which enable experiment control,
visualization, and data acquisition. The following images provide an overview of the user interface:

![](https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/readme_images/main_gui.png?raw=True)

The calibration module is another component of the PyCCAPT, providing essential tools for data calibration and
interpretation. This module includes functionalities such as t<sub>0</sub> and flight path calculation, region of
interest (ROI) selection, voltage and bowl calibration, and 3D reconstruction techniques. 

![](https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/readme_images/visualization_gif.gif?raw=True)

Some calibration features are shown below.

FDM and detector hitmap GIFs for an aluminum sample:

![](https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/readme_images/hist.png?raw=True)

<div align="center">
  <img width = "37%" src="https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/readme_images/fdm.png?raw=True">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img width = "33%" src="https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/readme_images/detector.gif?raw=True">
</div>

Bowl and voltage calibration:


<div align="center">
  <img width="30%" src="https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/readme_images/vol_corr.png?raw=True">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img width="30%" src="https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/readme_images/bowl_corr.png?raw=True">
</div>
<div align="center">
  <img width = "30%" src="https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/readme_images/tof_V_corr.png?raw=True">
  &nbsp;&nbsp;&nbsp;&nbsp;  
  <img width = "30%" src="https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/readme_images/tof_bowl_corr_y_det.png?raw=True">
</div>


A ranged mass spectrum for a Nimonic 90 sample:

<div align="center">
  <img width = "90%" src="https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/readme_images/mc.png?raw=True">
</div>


The following HTML link shows a 3D reconstruction of a Nimonic 90 sample:
[Nimonic 90 3D reconstruction](https://rawcdn.githack.com/mmonajem/pyccapt/52835bc47735ef12bffcf7e18ce90b556b07d12f/pyccapt/files/readme_images/3d_o.html)

The 3D reconstruction of Nimonic 90 and precipitates is shown in the GIFs below:


<div align="center">
  <img width = "40%" src="https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/readme_images/roto.gif?raw=True">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img width = "40%" src="https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/readme_images/iso.gif?raw=True">
</div>

The PyCCAPT package is a tool for controlling atom probe systems and
calibrating data.

 ---------------------

# Directory structure
```text
pyccapt/
├── pyproject.toml
├── setup.py
├── README.md
├── pyccapt/
│   ├── __init__.py
│   ├── config.toml
│   ├── files/
│   ├── control/
│   │   ├── __main__.py
│   │   └── core/            # canonical control runtime modules
│   └── calibration/
│       └── core/            # canonical calibration modules
├── docs/
└── tests/
```
Control configuration is provided via `config.toml` (comment-friendly). `config.json` is no longer accepted.

Control GUI electrode options are configured in `pyccapt/control/electrode.toml`:

```toml
[electrodes]
names = [
  "NiC1", # Example comment
  "CuC1",
]
```

You can edit this file and keep inline comments.

---------------------

# Installation

## 1) Create and activate an environment

```bash
conda create -n apt_env python=3.11
conda activate apt_env
python -m pip install --upgrade pip
```

## 2) Install from PyPI (online)

Default install (calibration profile):

```bash
pip install pyccapt
```

Install calibration + control dependencies (full installation):

```bash
pip install "pyccapt[full]"
```

Install calibration profile explicitly (same as default):

```bash
pip install "pyccapt[calibration]"
```

Add control dependencies on top of default calibration:

```bash
pip install "pyccapt[control]"
```

Note: pip extras are additive. With one package, `control` is added to the default profile, so it results in calibration + control.

## 3) Install with Conda

If PyCCAPT is available in your conda channel (for example conda-forge):

```bash
conda install -c conda-forge pyccapt
```

Local conda package build + install from this repository:

```bash
conda install -c conda-forge conda-build
conda build conda-recipe
conda install --use-local pyccapt
```

If you need full dependencies in a conda environment:

```bash
pip install "pyccapt[full]"
```

Or create ready-to-use conda environments:

```bash
conda env create -f environment.yml
conda env create -f environment.full.yml
```

## 4) Local development install

From the repository root:

```bash
pip install -e ".[full]"
```

Or module-focused local installs:

```bash
pip install -e ".[control]"
pip install -e ".[calibration]"
```

## 5) Run PyCCAPT control GUI

```bash
pyccapt
```

If the console script is not available:

```bash
python -m pyccapt.control
```

## 6) Run tutorials

```bash
jupyter lab
```

Then open notebooks under `pyccapt/calibration/tutorials`.

--------------

# Documentation

The latest documentation is available on [ReadTheDocs](https://pyccapt.readthedocs.io/).
It includes feature descriptions, tutorials, and configuration guidance.


---------------------
# Using PyCCAPT

For the control part of the package, follow the steps in the
[configuration documentation](https://pyccapt.readthedocs.io/en/latest/configuration.html).

For calibration, review the [tutorial](https://pyccapt.readthedocs.io/en/latest/tutorials.html) to understand package
features.

To try tutorials on Google Colab, use the following links:
[data processing](https://colab.research.google.com/github/mmonajem/pyccapt/blob/main/pyccapt/calibration/tutorials/colab/data_processing.ipynb), 
[data visualization](https://colab.research.google.com/github/mmonajem/pyccapt/blob/main/pyccapt/calibration/tutorials/colab/visualization.ipynb), and
[t<sub>0</sub> and flight path calculation](https://colab.research.google.com/github/mmonajem/pyccapt/blob/main/pyccapt/calibration/tutorials/colab/L_and_t0_determination.ipynb).

---------------------
# Data structure

For the control module data structure, see [pyccapt/control/DATA_STRUCTURE.md](pyccapt/control/DATA_STRUCTURE.md).
For the calibration module data structure, see [pyccapt/calibration/DATA_STRUCTURE.md](pyccapt/calibration/DATA_STRUCTURE.md).

---------------------
# Test data

To get started with the calibration package, you can use the test data (pure aluminum) available at
the link below. It includes a raw dataset collected from the OXCART atom probe, calibration outputs,
reconstruction outputs, and a range file (HDF5) generated by the calibration workflow.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14673955.svg)](https://doi.org/10.5281/zenodo.14673955)

------------------
# Bug reports

Report bugs, request help, or provide feedback in the
[GitHub Issues](https://github.com/mmonajem/pyccapt/issues).

Questions/comments:
- Mehrpad Monajem, mehrpad.monajem@fau.de

-----------

# Citing

-----------

If you use PyCCAPT in your work, please cite the software DOI shown at the top of this README.
Citation metadata is available in [CITATION.cff](CITATION.cff).

# Contributing

Contributions to PyCCAPT are always welcome, and they are greatly appreciated! Our contribution
policy can be found [here](CONTRIBUTING.md).

------------

# License

This project is licensed under the GNU General Public License v3.0. See
the [LICENSE](LICENSE) file for details.

