from pathlib import Path
from unittest.mock import patch

import numpy as np

from pyccapt.calibration.calibration import calibration
from pyccapt.calibration.calibration.share_variables import Variables


def test_plot_fdm_uses_shared_save_helper(tmp_path: Path):
    variables = Variables()
    variables.set_result_directory(tmp_path)
    x = np.array([0.1, 0.2, 0.3, 0.4])
    y = np.array([0.2, 0.1, 0.4, 0.3])

    with patch("pyccapt.calibration.calibration.diagnostics.save_figure") as mock_save:
        with patch("pyccapt.calibration.calibration.diagnostics.plt.show"):
            calibration.plot_fdm(x, y, variables, save=True, bins_s=4, index_fig=7)

    mock_save.assert_called_once()
    kwargs = mock_save.call_args.kwargs
    assert kwargs["directory"] == variables.result_path
    assert kwargs["stem"] == "fdm_7"
    assert kwargs["formats"] == ("png", "svg")

