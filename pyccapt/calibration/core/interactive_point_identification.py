"""Interactive peak annotation helpers for matplotlib plots."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt


def distance(x1: float, x2: float, y1: float, y2: float) -> float:
    """Return Euclidean distance between two 2D points."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def distances(x1: float, x2: float, y1: float, y2: float) -> float:
    """Backward-compatible alias for :func:`distance`."""
    return distance(x1, x2, y1, y2)


class AnnotationFinder:
    """
    Matplotlib callback that selects and deselects nearest annotated peaks.

    Left click selects the nearest point within tolerance.
    Right click deselects it.
    """

    def __init__(self, xdata, ydata, annotations, variables, ax=None, xtol=None, ytol=None):
        if len(xdata) != len(ydata) or len(xdata) != len(annotations):
            raise ValueError("xdata, ydata, and annotations must have matching lengths")
        if len(xdata) == 0:
            raise ValueError("xdata cannot be empty")

        self.data = list(zip(xdata, ydata, annotations))
        if xtol is None:
            xtol = ((max(xdata) - min(xdata)) / float(len(xdata))) / 2
        if ytol is None:
            ytol = ((max(ydata) - min(ydata)) / float(len(ydata))) / 2
        self.xtol = xtol
        self.ytol = ytol
        self.ax = plt.gca() if ax is None else ax
        self.drawn_annotations = {}
        self.links = []
        self.variables = variables

    @staticmethod
    def _annotation_to_index(annotation) -> int:
        return int(annotation) - 1

    def annotate_plotter(self, event) -> None:
        """Handle matplotlib click events and update selected annotations."""
        if not event.inaxes:
            return
        click_x = event.xdata
        click_y = event.ydata
        if (self.ax is not None) and (self.ax is not event.inaxes):
            return

        candidates = []
        for x, y, annotation in self.data:
            if ((click_x - self.xtol < x < click_x + self.xtol) and
                    (click_y - self.ytol < y < click_y + self.ytol)):
                candidates.append((distance(x, click_x, y, click_y), x, y, annotation))
        if not candidates:
            return

        _, x, y, annotation = min(candidates, key=lambda item: item[0])
        if event.button == 3:
            self.deselect_point(event.inaxes, x, y, annotation)
        else:
            self.draw_annotation(event.inaxes, x, y, annotation)
        for linked in self.links:
            linked.draw_specific_annotation(annotation)

    def draw_annotation(self, ax, x, y, annotation) -> None:
        """Draw one annotation and register it in shared state."""
        if (x, y) in self.drawn_annotations:
            return

        annotation_text = ax.text(x - 0.8, y, str(annotation), ha="right", va="center")
        marker = ax.scatter([x], [y], marker="H", c="r", zorder=100)
        self.drawn_annotations[(x, y)] = (annotation_text, marker)
        self.ax.figure.canvas.draw_idle()

        point_index = self._annotation_to_index(annotation)
        if x not in self.variables.peaks_x_selected:
            self.variables.peaks_x_selected.append(x)
            self.variables.peaks_x_selected.sort()
        if point_index not in self.variables.peaks_index_list:
            self.variables.peaks_index_list.append(point_index)
            self.variables.peaks_index_list.sort()

    def deselect_point(self, ax, x, y, annotation) -> None:
        """Hide one annotation and remove it from shared state if present."""
        if (x, y) in self.drawn_annotations:
            markers = self.drawn_annotations[(x, y)]
            for marker in markers:
                marker.set_visible(not marker.get_visible())
            self.ax.figure.canvas.draw_idle()

        point_index = self._annotation_to_index(annotation)
        if x in self.variables.peaks_x_selected:
            self.variables.peaks_x_selected.remove(x)
            self.variables.peaks_x_selected.sort()
        if point_index in self.variables.peaks_index_list:
            self.variables.peaks_index_list.remove(point_index)
            self.variables.peaks_index_list.sort()

    def draw_specific_annotation(self, annotation) -> None:
        """Draw annotation for every matching point label."""
        annotations_to_draw = [(x, y, label) for x, y, label in self.data if label == annotation]
        for x, y, label in annotations_to_draw:
            self.draw_annotation(self.ax, x, y, label)

    def annotates_plotter(self, event) -> None:
        """Backward-compatible wrapper for legacy method name."""
        self.annotate_plotter(event)

    def drawAnnote(self, ax, x, y, annotation) -> None:
        """Backward-compatible wrapper for legacy method name."""
        self.draw_annotation(ax, x, y, annotation)

    def deselectPoint(self, ax, x, y, annotation) -> None:
        """Backward-compatible wrapper for legacy method name."""
        self.deselect_point(ax, x, y, annotation)

    def drawSpecificAnnote(self, annotation) -> None:
        """Backward-compatible wrapper for legacy method name."""
        self.draw_specific_annotation(annotation)


class AnnoteFinder(AnnotationFinder):
    """Backward-compatible class alias with legacy name."""

