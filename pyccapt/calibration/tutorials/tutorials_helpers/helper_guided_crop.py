"""Small source/selection plots that also work before a mass spectrum exists."""
import matplotlib.pyplot as plt
import numpy as np


def source_overview(workflow, selection=None):
    """Plot a bounded display sample; all selection counts use the complete data."""
    frame = workflow.source
    selected = workflow.active_mask if selection is None else selection
    indices = np.unique(np.linspace(0, workflow.n-1, min(workflow.n, 40000), dtype=int))
    unavailable = frame.attrs.get("pyccapt_unavailable_columns", [])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3), constrained_layout=True)
    for axis, column in zip(axes[:2], ("t (ns)", "high_voltage (V)")):
        if column in frame and column not in unavailable:
            values = frame[column].to_numpy(dtype=float)[indices]
            axis.scatter(indices, values, s=1, color="lightgray", rasterized=True)
            axis.scatter(indices[selected[indices]], values[selected[indices]], s=1, rasterized=True)
            axis.set(xlabel="Source event position", ylabel=column)
        else:
            axis.text(.5, .5, f"{column} unavailable", ha="center", transform=axis.transAxes)
    if all(name in frame and name not in unavailable for name in ("x_det (cm)", "y_det (cm)")):
        x, y = (frame[name].to_numpy(dtype=float)[indices] for name in ("x_det (cm)", "y_det (cm)"))
        axes[2].scatter(x, y, s=1, color="lightgray", rasterized=True)
        axes[2].scatter(x[selected[indices]], y[selected[indices]], s=1, rasterized=True)
        axes[2].set(xlabel="Detector x (cm)", ylabel="Detector y (cm)", aspect="equal")
    else:
        axes[2].text(.5, .5, "Detector coordinates unavailable", ha="center", transform=axes[2].transAxes)
    fig.suptitle(f"{int(selected.sum()):,} of {workflow.n:,} events selected; plots show at most 40,000 events")
    return fig
