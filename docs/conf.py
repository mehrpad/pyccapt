# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

import os
from pathlib import Path
import sys

DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.append(str(DOCS_DIR / "extensions"))

project = 'PyCCAPT'
copyright = '2022, Mehrpad Monajem'
author = 'Mehrpad Monajem'

# The full version, including alpha/beta/rc tags
release = '0.0.34'

env_flag = os.getenv("status_flag")

# set device name
if env_flag != None:
    status_flag = 'HTML'
else:
    status_flag = 'HTML'  # default device name


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'nbsphinx',
    'myst_parser',
    'sphinx_markdown_tables',
    'sphinxcontrib.bibtex',
    ]

# The html index document.
bibtex_bibfiles = ['references.bib']
bibtex_default_style = 'unsrt'
bibtex_reference_style = 'label'
autodoc_member_order = 'groupwise'
autoclass_content = 'both'
nbsphinx_timeout = -1
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

myst_heading_anchors = 3  # Allows linking to headings in Markdown files

autodoc_mock_imports = [
    # Hardware deps (Windows-only or vendor SDKs)
    'mcculw',
    'mcculw.enums',
    'mcculw.ul',
    'nidaqmx',
    'pypylon',
    'pypylon.genicam',
    'pypylon.pylon',
    # Optional control-extra deps that the docs CI environment doesn't install
    'serial',
    'serial.tools',
    'serial.tools.list_ports',
    'cv2',
    'pyvisa',
    'PyQt6',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'pyqtgraph',
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_logo = "../pyccapt/files/logo2.png"

html_css_files = [
    'widestyle.css',
]

html_context = {
  'display_github': True,
  'github_user': 'mmonajem',
  'github_repo': 'pyccapt',
  'github_version': 'master/docs/',
}

html_theme_options = {
    'logo_only': False,
    'titles_only': False,
}

nbsphinx_allow_errors = True

# Enable Markdown cross-references
myst_enable_extensions = ["dollarmath", "amsmath", "deflist", "html_admonition", "html_image"]

# Keep the RTD build resilient even when optional GUI and hardware types
# are unavailable in the docs environment.
nitpicky = False
nitpick_ignore = []

# Add mappings for intersphinx to handle references to external libraries
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}
