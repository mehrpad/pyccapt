"""Top-level package for PyCCAPT."""

import warnings

# SciPy ships a runtime guard that complains when it's loaded against a
# newer NumPy than the one it was built for. Our lab environment runs
# NumPy 1.26 against an older SciPy build, and the bound is advisory
# (the SciPy routines we use work fine). Filter the specific message
# here, before any submodule pulls in scipy, so startup logs stay clean.
warnings.filterwarnings(
    "ignore",
    message=r"A NumPy version >=.* and <.* is required for this version of SciPy.*",
    category=UserWarning,
)

__version__ = "0.2.3"
