"""Check notebook/kernel compatibility before rendering the guided controls."""
from importlib import import_module
from pathlib import Path
import sys


EXPECTED_STAGES = ("Import", "Inspect", "Crop", "Calibrate", "Range", "Reconstruct", "Analyze", "Export")
GUIDED_UI_API_VERSION = 4


def load_guided_workflow(*, repository_root=None):
    """Return the current panel factory or explain how to recover a stale kernel.

    Reloading scientific modules around live sessions can mix incompatible class
    definitions. Require an explicit kernel restart rather than doing that here.
    """
    package = import_module("pyccapt")
    package_directory = Path(package.__file__).resolve().parent
    restart = "Save any needed session, then choose Kernel > Restart Kernel and Run All."
    if repository_root is not None:
        expected = Path(repository_root).resolve() / "pyccapt"
        if package_directory != expected:
            raise RuntimeError(
                f"This kernel loaded PyCCAPT from {package_directory}, but this notebook uses {expected}. "
                f"{restart} Select the environment installed for this checkout if the paths still differ.")
    core = import_module("pyccapt.calibration.core.guided_workflow")
    helper = import_module("pyccapt.calibration.tutorials.tutorials_helpers.helper_guided_workflow")
    if (getattr(helper, "GUIDED_UI_API_VERSION", None) != GUIDED_UI_API_VERSION or
            tuple(getattr(core, "STAGES", ())) != EXPECTED_STAGES or
            tuple(getattr(helper, "STAGES", ())) != EXPECTED_STAGES or
            getattr(helper, "GuidedWorkflow", None) is not core.GuidedWorkflow):
        raise RuntimeError(
            "Jupyter is still using an older guided workflow. The current panel must include the Crop tab. "
            f"{restart} Rerunning an import cell alone does not refresh cached Python modules.")
    print(f"Guided workflow: {' → '.join(EXPECTED_STAGES)}")
    print(f"PyCCAPT source: {package_directory}")
    print(f"Python kernel: {sys.executable}")
    return helper.show_guided_workflow
