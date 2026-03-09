"""Diagnostic plotting helpers for calibration workflows."""

from __future__ import annotations

from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, rcParams

from pyccapt.calibration.core.exceptions import CalibrationInputError
from pyccapt.calibration.core.validation import ensure_choice, ensure_positive
from pyccapt.calibration.path_utils import save_figure


def plot_fdm(x, y, variables, save, bins_s, index_fig, figure_size=(5, 4)):
    """Plot a field desorption map from detector hit coordinates."""
    fig1, ax1 = plt.subplots(figsize=figure_size, constrained_layout=True)

    fdm, xedges, yedges = np.histogram2d(x, y, bins=bins_s)

    ax1.set_xlabel(r"$X_{det} (cm)$", fontsize=10)
    ax1.set_ylabel(r"$Y_{det} (cm)$", fontsize=10)

    cmap = copy(plt.cm.plasma)
    cmap.set_bad(cmap(0))
    pcm = ax1.pcolormesh(xedges, yedges, fdm.T, cmap=cmap, norm=colors.LogNorm(), rasterized=True)
    fig1.colorbar(pcm, ax=ax1, pad=0)

    if save:
        rcParams["svg.fonttype"] = "none"
        save_figure(
            fig1,
            directory=variables.result_path,
            stem=f"fdm_{index_fig}",
            formats=("png", "svg"),
            dpi=600,
        )
    plt.show()


def initial_calibration(data, flight_path_length):
    """Compute initial time-of-flight calibration factors."""
    v_dc = data["high_voltage (V)"].to_numpy()
    t_values = data["t (ns)"].to_numpy()
    x_det = data["x_det (cm)"].to_numpy() * 10
    y_det = data["y_det (cm)"].to_numpy() * 10
    d_values = x_det ** 2 + y_det ** 2 + flight_path_length ** 2

    init_flight_path_factor = np.mean(d_values) / d_values
    init_voltage_factor = np.sqrt(v_dc / np.mean(v_dc))
    return t_values * init_flight_path_factor * init_voltage_factor


def _plot_scatter(x_values, y_values, x_label, y_label, variables, save, stem, fig_size):
    """Plot and optionally save a detector-vs-signal scatter view."""
    fig, axis = plt.subplots(figsize=fig_size, constrained_layout=True)
    plt.scatter(x_values, y_values, color="blue", label=r"$t$", s=1)
    axis.set_xlabel(x_label, fontsize=10)
    axis.set_ylabel(y_label, fontsize=10)
    plt.grid(alpha=0.3, linestyle="-.", linewidth=0.4)
    if save:
        rcParams["svg.fonttype"] = "none"
        save_figure(
            fig,
            directory=variables.result_path,
            stem=stem,
            formats=("png", "svg"),
            dpi=600,
        )
    plt.show()


def plot_selected_statistic(variables, bin_fdm, index_fig, calibration_mode, save, fig_size=(5, 4)):
    """Plot detector and voltage statistics for the currently selected peak range."""
    ensure_choice(calibration_mode, field_name="calibration_mode", allowed=["tof", "mc"])
    ensure_positive(bin_fdm, field_name="bin_fdm")

    print("Selected tof are: (%s, %s)" % (variables.selected_x1, variables.selected_x2))
    mask_temporal = variables.build_calibration_mask(calibration_mode)

    x = variables.dld_x_det[mask_temporal]
    y = variables.dld_y_det[mask_temporal]
    dld_high_voltage = variables.dld_high_voltage[mask_temporal]
    t_values = variables.get_calibration_array(calibration_mode)[mask_temporal]

    if len(x) == 0:
        raise CalibrationInputError("Selected peak contains no detector points")

    bins = [bin_fdm, bin_fdm]
    plot_fdm(x, y, variables, save, bins, index_fig)

    sample_count = min(len(x), 1000)
    mask = np.random.randint(0, len(x), sample_count)
    y_label = "tof" if calibration_mode == "tof" else "mc"

    _plot_scatter(
        dld_high_voltage[mask],
        t_values[mask],
        x_label="Voltage (V)",
        y_label=y_label,
        variables=variables,
        save=save,
        stem=f"v_t_{index_fig}",
        fig_size=fig_size,
    )
    _plot_scatter(
        x[mask],
        t_values[mask],
        x_label=r"$X_{det} (cm)$",
        y_label=y_label,
        variables=variables,
        save=save,
        stem=f"x_t_{index_fig}",
        fig_size=fig_size,
    )
    _plot_scatter(
        y[mask],
        t_values[mask],
        x_label=r"$Y_{det} (cm)$",
        y_label=y_label,
        variables=variables,
        save=save,
        stem=f"y_t_{index_fig}",
        fig_size=fig_size,
    )


__all__ = ["plot_fdm", "initial_calibration", "plot_selected_statistic"]

