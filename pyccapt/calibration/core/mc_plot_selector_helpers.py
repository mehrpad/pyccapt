"""Selector wiring helpers for :mod:`mc_plot`."""

from __future__ import annotations

import matplotlib.pyplot as plt

from pyccapt.calibration.core import interactive_point_identification
from pyccapt.calibration.data_tools import data_loadcrop, plot_vline_draw


def attach_selector(plotter, selector='rect'):
    """Attach selection/zoom interactions for the histogram figure."""
    if selector == 'rect':
        data_loadcrop.rectangle_box_selector(plotter.ax, plotter.variables)
        plt.connect('key_press_event', lambda event: data_loadcrop.toggle_selector(event))
    elif selector == 'peak':
        annotation_finder = interactive_point_identification.AnnoteFinder(
            plotter.x[plotter.peaks],
            plotter.y[plotter.peaks],
            plotter.annotates,
            plotter.variables,
            ax=plotter.ax,
        )
        plotter.fig.canvas.mpl_connect('button_press_event', lambda event: annotation_finder.annotates_plotter(event))

        zoom_manager = plot_vline_draw.HorizontalZoom(plotter.ax, plotter.fig)
        plotter.fig.canvas.mpl_connect('key_press_event', lambda event: zoom_manager.on_key_press(event))
        plotter.fig.canvas.mpl_connect('key_release_event', lambda event: zoom_manager.on_key_release(event))
        plotter.fig.canvas.mpl_connect('scroll_event', lambda event: zoom_manager.on_scroll(event))

    elif selector == 'range':
        plotter.line_manager = plot_vline_draw.VerticalLineManager(plotter.variables, plotter.ax, plotter.fig, [], [])
        plotter.fig.canvas.mpl_connect('button_press_event', lambda event: plotter.line_manager.on_press(event))
        plotter.fig.canvas.mpl_connect('button_release_event', lambda event: plotter.line_manager.on_release(event))
        plotter.fig.canvas.mpl_connect('motion_notify_event', lambda event: plotter.line_manager.on_motion(event))
        plotter.fig.canvas.mpl_connect('key_press_event', lambda event: plotter.line_manager.on_key_press(event))
        plotter.fig.canvas.mpl_connect('scroll_event', lambda event: plotter.line_manager.on_scroll(event))
        plotter.fig.canvas.mpl_connect('key_release_event', lambda event: plotter.line_manager.on_key_release(event))


def zoom_to_x_range(plotter, x_min, x_max, reset=False):
    """Zoom the histogram x-limits to a specified range."""
    if reset:
        if plotter.original_x_limits is not None:
            plotter.ax.set_xlim(plotter.original_x_limits)
            plotter.fig.canvas.draw()
    else:
        plotter.ax.set_xlim(x_min, x_max)
        plotter.fig.canvas.draw()


__all__ = ['attach_selector', 'zoom_to_x_range']

