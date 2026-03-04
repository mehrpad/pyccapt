# PyCCAPT (APT_PyControl)

# A modular, FAIR open-source python atom probe tomography software package for experiment control and data calibration

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10210507.svg)](https://doi.org/10.5281/zenodo.10210507)
[![Documentation Status](https://readthedocs.org/projects/pyccapt/badge/?version=latest)](https://pyccapt.readthedocs.io/en/latest/?badge=latest)
<!--[![coverage report](https://gitlab.com/jesseds/apav/badges/master/coverage.svg)](https://gitlab.com/jesseds/apav/commits/master)
[![pipeline status](https://gitlab.com/jesseds/apav/badges/master/pipeline.svg)](https://gitlab.com/jesseds/apav/-/commits/master)-->

<img align="right" src="https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/logo2.png?raw=True)" alt="Alt Text" width="300" height="300">

This package aims to provide open-source software for controlling atom probe systems and calibrating data.
The package is modular and adaptable for a wide range of devices in atom probe instrument. So far it is
capable of collecting data from Surface consept and ReoenDek TDC systems. The package is developed in Python 3.9.
Additionally, the package encompasses a comprehensive calibration module that provides various functionalities,
including t<sub>0</sub> and flight path calculation, region of interest (ROI) selection, voltage and bowl calibration,
as well as
3D reconstruction techniques.
This module enhances the capabilities of PyCCAPT by providing essential tools for accurate data interpretation and
analysis.

----------

# Overview

PyCCAPT was initially developed and tested on the OXCART atom probe, an in-house atom probe system situated
within the Department of Materials Science & Engineering at the University of Erlangen-NÃ¼rnberg. The OXCART atom probe boasts a titanium-based measuring chamber that facilitates an ultra-low hydrogen 
vacuum environment. This system also features a state-of-the-art detector with a high detection
efficiency of approximately 80%. While tailor-made for the OXCART, the PyCCAPT package offers versatility, extending its
capabilities to effectively control atom probe systems.

![](https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/readme_images/oxcart.jpg?raw=True)

The package is designed with modularity in mind, making it highly adaptable to new instruments. This
adaptability extends to instruments like Pfeifer gauges, Fug power supplies, and Siglent signal generators. Notably, the
PyCCAPT package has already demonstrated its proficiency in collecting data from Surface Concept and ReoenDek TDC
systems.

The PyCCAPT package forms the foundation of a fully FAIR atom probe data collection and processing chain. This
repository includes the graphical user interface (GUI) and control program, which enable experiment control,
visualization, and data acquisition. The following images provide an overview of the user interface:

![](https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/readme_images/main_gui.png?raw=True)

The calibration module is another component of the PyCCAPT, providing essential tools for data calibration and
interpretation. This module includes functionalities such as t<sub>0</sub> and flight path calculation, region of
interest (ROI) selection, voltage and bowl calibration, and 3D reconstruction techniques. 

![](https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/readme_images/visualization_gif.gif?raw=True)

You can see some of the features of the calibration module in the following images.

The FDM and detector hitmap as Gif images for an Aluminium sample:

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


A ranged mass spectrum for a NimonicÂ® 90 sample:

<div align="center">
  <img width = "90%" src="https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/readme_images/mc.png?raw=True">
</div>


Html link below can be used to show a 3d reconstruction of NimonicÂ® 90 sample: [NimonicÂ® 3D reconstruction](https://rawcdn.githack.com/mmonajem/pyccapt/52835bc47735ef12bffcf7e18ce90b556b07d12f/pyccapt/files/readme_images/3d_o.html)

The 3d reconstruction of NimonicÂ® 90 and precipitates can be seen in the following Gifs:


<div align="center">
  <img width = "40%" src="https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/readme_images/roto.gif?raw=True">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img width = "40%" src="https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/readme_images/iso.gif?raw=True">
</div>

The PyCCAPT package is a tool for controlling atom probe systems and
calibrating data.

 ---------------------

# Directory structure
```
pyccapt/
â”œâ”€â”€ pyccapt/
â”‚   â”œâ”€â”€ __init__.py
â”‚   â”œâ”€â”€ config.toml   
â”‚   â”œâ”€â”€ calibration/
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â””â”€â”€ module_folders   
â”‚   â”œâ”€â”€ control/
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â””â”€â”€ module_folders
â”‚   â””â”€â”€ files/
â”œâ”€â”€ docs/
â”œâ”€â”€ setup.py
â”œâ”€â”€ README.md
â”œâ”€â”€ CONTRIBUTION.md
â”œâ”€â”€ MANIFEST.in
â”œâ”€â”€ Licence
â””â”€â”€ tox.ini
â””â”€â”€ tests/
    â”œâ”€â”€ __init__.py
    â”œâ”€â”€ data/
    â””â”€â”€ tests
```
Control configuration is provided via `config.toml` (comment-friendly). `config.json` is no longer accepted.
 ---------------------

# Installation

1. Create a virtual environment using Anaconda:

   ```bash
   conda create -n apt_env python=3.9
    ```

2. Activate the virtual environment:

   ```bash
   conda activate apt_env
   ```

3. Clone or download this repository, unzip it, and in the project directory run:

   ```bash
   pip install -e .
   ```

After installation, you can run the control GUI by entering the following command in the console:

   ```bash
   pyccapt
   ```

or if the above command does not work, you can run the following command:

   ```bash
   python -m pyccapt.control
   ```


To start the tutorial, please follow these steps:

1- Open your terminal.

2- Navigate to the project folder using the cd command.

3- Once you're inside the project folder, go to the tutorial folder by running the following command:

   ```bash
   Jupyter lab
   ```

--------------

# Documentation

The latest documentation is available on
[ReadTheDocs](https://pyccapt.readthedocs.io/) page. It provides feature descriptions, tutorials, and
valuable information.


---------------------
# Using PyCCAPT

For control part of the package you can follow the steps
on [documentation](https://pyccapt.readthedocs.io/en/latest/configuration.html).

For calibration, review the [tutorial](https://pyccapt.readthedocs.io/en/latest/tutorials.html) to understand package
features.

To test the code on Google colab, you can use the following links:
[data processing](https://colab.research.google.com/github/mmonajem/pyccapt/blob/main/pyccapt/calibration/tutorials/colab/data_processing.ipynb), 
[data visualization](https://colab.research.google.com/github/mmonajem/pyccapt/blob/main/pyccapt/calibration/tutorials/colab/visualization.ipynb), and
[t<sub>0</sub> and flight path calculation](https://colab.research.google.com/github/mmonajem/pyccapt/blob/main/pyccapt/calibration/tutorials/colab/L_and_t0_determination.ipynb).

---------------------
# Data structure

For checking the data structure of the control module you can check [here](pyccapt/control/DATA_STRUCTURE.md). For the 
calibration module, you can check [here](pyccapt/calibration/DATA_STRUCTURE.md).

---------------------
# Test data

For start using the calibration package, you can use the test data (pure Aluminium) provided in the following link. The
link contains the raw dataset that is collected from the OXCART atom probe from a pure Aluminium sample. It also
contains
the output file from the calibration module, which contains the calibrated data as well as the reconstructed data. The
link also contain the range file (HDF5) that is calculated by the calibration module.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14673955.svg)](https://doi.org/10.5281/zenodo.14673955)

------------------
# Bug reports

Report bugs, issues, ask for help, or provide feedback on
the [Github section](https://github.com/mmonajem/pyccapt/issues).

Qestions/comments:
  - Mehrpad Monajem, mehrpad.monajem@fau.de

-----------

# Citing

-----------

# Contributing

Contributions to PyCCAPT are always welcome, and they are greatly appreciated! Our contribution
policy can be found [here](CONTRIBUTING.md).

------------

# License

This project is licensed under the GNU General Public License v3.0. See
the [LICENSE](LICENSE) file for details.

