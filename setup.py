from __future__ import annotations

import re
from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent.resolve()
README_PATH = ROOT / "README.md"
INIT_PATH = ROOT / "pyccapt" / "__init__.py"


def read_version() -> str:
    content = INIT_PATH.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if match is None:
        raise RuntimeError("Could not find __version__ in pyccapt/__init__.py")
    return match.group(1)


def unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


common_deps = [
    "deepdiff",
    "h5py",
    "matplotlib",
    "numba",
    "numpy",
    "pandas",
    "requests",
    "scipy",
    "tables",
    "tomli; python_version < '3.11'",
    "wget",
]

control_deps = [
    "mcculw; platform_system == 'Windows'",
    "networkx",
    "nidaqmx; platform_system == 'Windows'",
    "opencv-python",
    "PyQt6",
    "pyqtgraph",
    "pypylon; platform_system == 'Windows'",
    "pyserial",
    "pyvisa",
    "simple-pid",
    # SmarAct MCS2 stage controller SDK  (smaract.ctl).
    # NOT on PyPI – install manually from the SmarAct SDK zip bundled with the
    # MCS2 software installer:
    #   pip install "C:\SmarAct\MCS2\SDK\Python\packages\smaract.ctl-<version>.zip"
    # Tested with: smaract.ctl-1.5.2
    # Uncomment the line below if/when SmarAct publishes the package to PyPI:
    # "smaract.ctl",
]

calibration_deps = [
    "adjustText",
    # anywidget backs Plotly's go.FigureWidget (required since Plotly 6); without
    # it reconstruction_plot() raises ImportError when building the 3D widget.
    "anywidget",
    "ase",
    "faker",
    "fast-histogram",
    "imageio",
    "ipympl",
    "ipywidgets",
    "jupyterlab",
    "kaleido",
    "nglview",
    "plotly",
    "pybaselines",
    "pymatgen",
    "pyvista",
    "scikit-learn",
    "tqdm",
    # uproot reads Cameca LEAP RHIT files (ROOT-format bundles) inside
    # ``pyccapt.calibration.leap_tools.cameca_raw.rhit_load``. The helper
    # raises an ImportError pointing here when missing, so without this
    # dependency the raw-data analysis notebook fails on .RHIT input.
    "uproot",
    "vispy",
]

package_data = {
    "pyccapt": [
        "config.toml",
        "control/electrode.toml",
        "files/*.h5",
        "files/*.png",
        "files/*.jpg",
        "files/*.txt",
        "files/PyQt6_UI/*.ui",
        "files/PyQt6_UI/*.md",
        "calibration/reflectron_correction/data/presets/*.csv",
        "control/*/*.dll",
        "control/*/*.lib",
        "control/*/*.exp",
        "control/*/*.ocx",
        "control/*/*.chm",
        "control/*/*.cfg",
        "control/*/*.ini",
        "control/*/*.bit",
        "control/*/*.exe",
        "control/*/*.lmf",
    ],
}


setup(
    name="pyccapt",
    version=read_version(),
    author="Mehrpad Monajem",
    author_email="mehrpad.monajem@fau.de",
    description="A Python package for atom probe control and data calibration.",
    long_description=README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else "",
    long_description_content_type="text/markdown",
    url="https://github.com/mmonajem/pyccapt",
    packages=find_packages(
        include=("pyccapt", "pyccapt.*"),
        exclude=(
            "tests",
            "tests.*",
            "pyccapt.calibration.tutorials",
            "pyccapt.calibration.tutorials.*",
            "pyccapt.calibration.leap_tools.tutorials",
            "pyccapt.calibration.leap_tools.tutorials.*",
            "pyccapt.control.devices_test",
            "pyccapt.control.devices_test.*",
        ),
    ),
    include_package_data=False,
    package_data=package_data,
    entry_points={"console_scripts": ["pyccapt=pyccapt.control.__main__:main"]},
    python_requires=">=3.9",
    install_requires=unique(common_deps),
    extras_require={
        "calibration": unique(calibration_deps),
        "control": unique(control_deps),
        "full": unique(calibration_deps + control_deps),
        "all": unique(calibration_deps + control_deps),
        "dev": ["build", "pytest", "pytest-mock", "twine"],
    },
    license="GPL-3.0-or-later",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    keywords=["atom probe", "apt", "calibration", "instrument control"],
)
