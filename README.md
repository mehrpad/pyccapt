# PyCCAPT

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10210507.svg)](https://doi.org/10.5281/zenodo.10210507)
[![Documentation Status](https://readthedocs.org/projects/pyccapt/badge/?version=latest)](https://pyccapt.readthedocs.io/en/latest/?badge=latest)

PyCCAPT is a modular, FAIR-oriented Python package for atom probe tomography (APT) instrument control, data calibration, and reconstruction workflows.

It provides:

- experiment control and acquisition for APT hardware
- calibration workflows (for example, `t0` and flight-path estimation, ROI selection, voltage/bowl correction)
- reconstruction and visualization tooling
- HDF5-based data handling for interoperable downstream analysis

<img align="right" src="https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/logo2.png?raw=True" alt="PyCCAPT logo" width="220" height="220">

## Project Scope

PyCCAPT was developed and validated on the OXCART atom probe platform and is designed to be adaptable to other APT systems through device-specific modules. Current integrations include detector backends such as Surface Concept and RoentDek, together with modular support for common laboratory hardware.

![Main GUI](https://github.com/mmonajem/pyccapt/blob/main/pyccapt/files/readme_images/main_gui.png?raw=True)

## Installation

PyCCAPT requires Python `>=3.9`.

1. Create and activate an environment:

```bash
conda create -n apt_env python=3.11
conda activate apt_env
python -m pip install --upgrade pip
```

2. Install from PyPI:

```bash
pip install pyccapt
```

Optional dependency groups:

```bash
pip install "pyccapt[calibration]"
pip install "pyccapt[control]"
pip install "pyccapt[full]"
```

3. Local development install (from repository root):

```bash
pip install -e ".[full]"
```

Or module-specific editable installs:

```bash
pip install -e ".[control]"
pip install -e ".[calibration]"
```

## Running PyCCAPT

Start the control application:

```bash
pyccapt
```

Fallback entrypoint:

```bash
python -m pyccapt.control
```

Run tests:

```bash
pytest -q --run-calibration
pytest -q --run-control
pytest -q
```

Run calibration tutorials:

```bash
jupyter lab
```

Then open notebooks under `pyccapt/calibration/tutorials`.

## Configuration

Control runtime configuration is stored in `pyccapt/config.toml` (comment-friendly). `config.json` is not supported.

Control GUI electrode labels are stored in `pyccapt/control/electrode.toml`:

```toml
[electrodes]
names = [
  "NiC1", # Nickel capillary
  "CuC1",
]
```

## Documentation

- Full documentation: [Read the Docs](https://pyccapt.readthedocs.io/)
- Control guide: [docs/configuration](https://pyccapt.readthedocs.io/en/latest/configuration.html)
- Calibration tutorials: [docs/tutorials](https://pyccapt.readthedocs.io/en/latest/tutorials.html)

Google Colab notebooks:

- [Data processing](https://colab.research.google.com/github/mmonajem/pyccapt/blob/main/pyccapt/calibration/tutorials/colab/data_processing.ipynb)
- [Visualization](https://colab.research.google.com/github/mmonajem/pyccapt/blob/main/pyccapt/calibration/tutorials/colab/visualization.ipynb)
- [`t0` and flight path estimation](https://colab.research.google.com/github/mmonajem/pyccapt/blob/main/pyccapt/calibration/tutorials/colab/L_and_t0_determination.ipynb)

## Data Structures

- Control data model: [pyccapt/control/DATA_STRUCTURE.md](pyccapt/control/DATA_STRUCTURE.md)
- Calibration data model: [pyccapt/calibration/DATA_STRUCTURE.md](pyccapt/calibration/DATA_STRUCTURE.md)

## Tutorial Dataset

Calibration tutorial data (pure aluminum), including raw and processed outputs, is available on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14673955.svg)](https://doi.org/10.5281/zenodo.14673955)

## Citation

If you use PyCCAPT in your work, please cite:

```bibtex
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
```

Citation metadata is also available in [CITATION.cff](CITATION.cff).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow and pull-request guidance.

## Support

- Issues and bug reports: [GitHub Issues](https://github.com/mmonajem/pyccapt/issues)
- Contact: Mehrpad Monajem (`mehrpad.monajem@fau.de`)

## License

PyCCAPT is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE).
