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
]

calibration_deps = [
    "adjustText",
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
    "vispy",
]


setup(
    name="pyccapt",
    version=read_version(),
    author="Mehrpad Monajem",
    author_email="mehrpad.monajem@fau.de",
    description="A Python package for atom probe control and data calibration.",
    long_description=README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else "",
    long_description_content_type="text/markdown",
    url="https://github.com/mmonajem/pyccapt",
    packages=find_packages(include=("pyccapt", "pyccapt.*"), exclude=("tests", "tests.*")),
    include_package_data=True,
    entry_points={"console_scripts": ["pyccapt=pyccapt.control.__main__:main"]},
    python_requires=">=3.9",
    install_requires=unique(common_deps + calibration_deps),
    extras_require={
        "calibration": [],
        "control": unique(control_deps),
        "full": unique(control_deps),
        "all": unique(control_deps),
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
