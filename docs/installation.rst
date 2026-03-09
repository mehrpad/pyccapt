Installation
============

PyCCAPT requires Python ``>=3.9``.

Recommended Conda Workflow
--------------------------

For most users, the smoothest setup is a conda environment with a pip install:

.. code-block:: bash

   conda create -n pyccapt python=3.11
   conda activate pyccapt
   python -m pip install --upgrade pip
   pip install "pyccapt[full]"

For a local checkout of this repository:

.. code-block:: bash

   git clone https://github.com/mmonajem/pyccapt.git
   cd pyccapt
   conda activate pyccapt
   pip install -e ".[full]"

PyPI Installation (Online)
--------------------------

Default installation (core dependencies only):

.. code-block:: bash

   pip install pyccapt

Install optional dependency groups:

.. code-block:: bash

   pip install "pyccapt[calibration]"
   pip install "pyccapt[control]"
   pip install "pyccapt[full]"

``[full]`` installs both control and calibration dependency sets.

Local Development Installation
------------------------------

From the project root:

.. code-block:: bash

   pip install -e ".[full]"

Module-specific editable installs:

.. code-block:: bash

   pip install -e ".[control]"
   pip install -e ".[calibration]"

Conda Installation
------------------

If PyCCAPT is available on your selected conda channel:

.. code-block:: bash

   conda install -c conda-forge pyccapt

Build and install locally from this repository:

.. code-block:: bash

   conda install -c conda-forge conda-build
   conda build conda-recipe
   conda install --use-local pyccapt

If you need both optional dependency sets in that environment:

.. code-block:: bash

   pip install "pyccapt[full]"

You can also create predefined environments:

.. code-block:: bash

   conda env create -f environment.yml
   conda env create -f environment.full.yml

Running PyCCAPT Control
-----------------------

After installation:

.. code-block:: bash

   pyccapt

Alternative entrypoint:

.. code-block:: bash

   python -m pyccapt.control

Running Tutorials
-----------------

Start JupyterLab:

.. code-block:: bash

   jupyter lab

Then open notebooks under `pyccapt/calibration/tutorials`.

Testing
-------

Run module-specific test suites:

.. code-block:: bash

   pytest -q --run-calibration
   pytest -q --run-control

Run all discoverable tests with currently installed optional dependencies:

.. code-block:: bash

   pytest -q
