Installation
===============================

PyPI Installation (Online)
--------------------------

Default installation (calibration profile):

``pip install pyccapt``

Full installation (control + calibration dependencies):

``pip install "pyccapt[full]"``

Calibration profile explicitly (same as default):

``pip install "pyccapt[calibration]"``

Add control dependencies on top of default calibration:

``pip install "pyccapt[control]"``

Note: pip extras are additive, so ``[control]`` adds control deps to the
default calibration profile.

Local Development Installation
------------------------------

Clone/download this repository and install from the project root:

``pip install -e ".[full]"``

For module-focused local installs:

``pip install -e ".[control]"``

``pip install -e ".[calibration]"``

Conda Installation
------------------

If PyCCAPT is available in your conda channel (for example, conda-forge):

``conda install -c conda-forge pyccapt``

Local conda build + install from this repository:

``conda install -c conda-forge conda-build``

``conda build conda-recipe``

``conda install --use-local pyccapt``

If you also need control dependencies in that conda environment, add:

``pip install "pyccapt[full]"``

Or create pre-defined conda environments:

``conda env create -f environment.yml``

``conda env create -f environment.full.yml``

Running PyCCAPT Control GUI
---------------------------

After installation:

``pyccapt``

or:

``python -m pyccapt.control``

Running PyCCAPT Tutorials
-------------------------

Run JupyterLab:

``jupyter lab``

Then open notebooks under ``pyccapt/calibration/tutorials``.

Testing
-------

Run tests from the project root:

``pytest -q``
