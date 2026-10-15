"""Small source/selection plots that also work before a mass spectrum exists."""
import matplotlib.pyplot as plt
import numpy as np

from pyccapt.calibration.core.exceptions import CalibrationInputError, CalibrationStateError
from pyccapt.calibration.data_tools.selectors_data import CropCircleSelector


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


def interval_from_drag(left, right, event_count):
    """Convert plot coordinates to original row positions [floor(left), ceil(right))."""
    if left is None or right is None or not np.isfinite([left, right]).all():
        return None
    lower, upper = sorted((float(left), float(right)))
    if lower == upper:
        return None
    start = max(0, min(event_count, int(np.floor(lower))))
    stop = max(0, min(event_count, int(np.ceil(upper))))
    return (start, stop) if start < stop else None


def circle_from_drag(x0, y0, x1, y1):
    """Return the true circle inscribed in the dragged box, in detector cm."""
    if any(value is None for value in (x0, y0, x1, y1)) or not np.isfinite([x0, y0, x1, y1]).all():
        return None
    radius = min(abs(x1-x0), abs(y1-y0)) / 2.
    if radius <= 0:
        return None
    return (float((x0+x1)/2.), float((y0+y1)/2.), float(radius))


def _available(frame, columns):
    unavailable = frame.attrs.get("pyccapt_unavailable_columns", [])
    return all(name in frame and name not in unavailable for name in columns)


def detector_density(workflow, start=0, stop=None, *, bins=180, chunk_size=250000):
    """Exact count map of a draft temporal interval, using bounded temporary arrays.

    Detector coordinates are already in cm. Frame labels and start-counter resets
    do not alter the original acquisition row positions used for the interval.
    """
    stop = workflow.n if stop is None else stop
    if (not isinstance(start, (int, np.integer)) or not isinstance(stop, (int, np.integer)) or
            not 0 <= start < stop <= workflow.n):
        raise CalibrationInputError("Detector display requires 0 <= start < stop <= source event count")
    if not isinstance(bins, int) or bins < 2 or not isinstance(chunk_size, int) or chunk_size < 1:
        raise CalibrationInputError("Density bins and chunk size must be positive integers (at least two bins)")
    names = ("x_det (cm)", "y_det (cm)")
    if not _available(workflow.source, names):
        raise CalibrationInputError("Detector coordinates are unavailable for this source")

    def chunks():
        for first in range(start, stop, chunk_size):
            last = min(stop, first+chunk_size)
            x, y = (workflow.source[name].iloc[first:last].to_numpy(dtype=float, copy=False) for name in names)
            valid = np.isfinite(x) & np.isfinite(y)
            yield x[valid], y[valid]

    limits = np.array([[np.inf, -np.inf], [np.inf, -np.inf]])
    count = 0
    for x, y in chunks():
        if x.size:
            limits[:, 0] = np.minimum(limits[:, 0], [x.min(), y.min()])
            limits[:, 1] = np.maximum(limits[:, 1], [x.max(), y.max()])
            count += x.size
    if not count:
        raise CalibrationInputError("The draft time interval has no measured finite detector positions")
    for bounds in limits:
        if bounds[0] == bounds[1]:
            bounds += [-.5, .5]
    edges = [np.linspace(lower, upper, bins+1) for lower, upper in limits]
    counts = np.zeros((bins, bins), dtype=np.int64)
    for x, y in chunks():
        counts += np.histogram2d(x, y, bins=edges)[0].astype(np.int64)
    return counts, edges[0], edges[1], count


class CropDrawing:
    """Retained selectors whose callbacks only update the caller's draft fields.

    The caller must retain this object and call ``close`` on source replacement.
    ``interactive=False`` supplies an Agg canvas for headless event tests. The
    normal canvas is an ipympl widget, independent of pyplot's global backend.
    """
    def __init__(self, workflow, *, on_interval, on_circle, interval_provider=None, interactive=True):
        import ipywidgets as widgets
        from matplotlib.figure import Figure
        from matplotlib.widgets import RectangleSelector

        self.workflow = workflow
        self.on_interval, self.on_circle = on_interval, on_circle
        self.interval_provider = interval_provider or (lambda: (0, workflow.n))
        self.closed, self.enabled, self.cancelled = False, True, False
        self.figure = Figure(figsize=(12, 4), constrained_layout=True)
        if interactive:
            try:
                from ipympl.backend_nbagg import Canvas, FigureManager
            except ImportError as error:
                raise CalibrationStateError("Interactive drawing requires the notebook ipympl widget backend") from error
            self.canvas = Canvas(self.figure)
            self.manager = FigureManager(self.canvas, 0)
            self.canvas.toolbar_position = "bottom"
        else:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            self.canvas = FigureCanvasAgg(self.figure)
            self.manager = None
        self.axes = self.figure.subplots(1, 2)
        self.status = widgets.HTML()
        instructions = widgets.HTML(
            "<b>Adjust the starting regions:</b> drag the orange rectangle's left/right handles to choose the smooth part of the DC-voltage curve. "
            "Drag the cyan circle's center to move it, or its edge/corner handles to make it smaller or larger. "
            "The starting circle covers the finite detector hits; adjust it to exclude unwanted edges and mouse bites. "
            "Only the rectangle's horizontal event interval is used. Detector distances are in cm. "
            "Escape cancels the current drag. Preview and accept the crop separately.")
        children = [instructions, self.status]
        if interactive:
            children.append(self.canvas)
        self.widget = widgets.VBox(children)
        self._children = children
        self._event_ids = [self.canvas.mpl_connect("button_press_event", self._start_drag),
                           self.canvas.mpl_connect("key_press_event", self._key_press)]
        voltage_axis, detector_axis = self.axes
        self.voltage_available = _available(workflow.source, ("high_voltage (V)",))
        if self.voltage_available:
            index = np.unique(np.linspace(0, workflow.n-1, min(workflow.n, 40000), dtype=int))
            voltage = workflow.source["high_voltage (V)"].iloc[index].to_numpy(dtype=float)
            voltage_axis.plot(index, voltage, linewidth=.7)
            voltage_axis.set(xlim=(-.025*workflow.n, 1.025*workflow.n), xlabel="Original source event position", ylabel="DC voltage (V)",
                title=f"DC voltage\nDisplay at most 40,000 of {workflow.n:,} events")
            voltage_axis.title.set_fontsize(10)
        else:
            voltage_axis.text(.5, .5, "DC voltage unavailable", ha="center", transform=voltage_axis.transAxes)
        self.rectangle_selector = RectangleSelector(voltage_axis, self._rectangle_selected, button=[1],
            interactive=True, useblit=False, drag_from_anywhere=True, grab_range=12,
            handle_props={"markersize": 8}, props={"facecolor": "tab:orange", "edgecolor": "tab:orange", "alpha": .25})
        self.circle_selector = CropCircleSelector(detector_axis, self._circle_selected, button=[1],
            interactive=True, useblit=False, drag_from_anywhere=True, grab_range=12, use_data_coordinates=True,
            handle_props={"markersize": 8}, props={"facecolor": "none", "edgecolor": "cyan", "linewidth": 2., "zorder": 2})
        self.detector_available = _available(workflow.source, ("x_det (cm)", "y_det (cm)"))
        self.detector_ready = False
        self._density_artist, self._density_colorbar = None, None
        self._density_message = None
        try:
            self.refresh_detector()
        except CalibrationInputError as error:
            self.status.value = str(error)
        self.set_enabled(True)

    def _start_drag(self, event):
        if self.enabled and not self.closed and event.inaxes in self.axes:
            self.cancelled = False

    def _key_press(self, event):
        if event.key == "escape":
            self.cancelled = True
            self.status.value = "Current drag cancelled; previous draft fields are unchanged."

    def _rectangle_selected(self, press, release):
        if self.closed or not self.enabled or self.cancelled or not self.voltage_available:
            return
        interval = interval_from_drag(press.xdata, release.xdata, self.workflow.n)
        if interval is not None:
            self.on_interval(*interval)
            self.status.value = f"Draft event interval [{interval[0]:,}, {interval[1]:,}); voltage height is not a filter. Review the detector map for this interval."

    def _circle_selected(self, press, release):
        if self.closed or not self.enabled or self.cancelled or not self.detector_ready:
            return
        if circle_from_drag(press.xdata, press.ydata, release.xdata, release.ydata) is not None:
            # The selector may have resized or moved an existing circle. Publish
            # its actual geometry rather than rederiving it from bounding edges.
            patch = self.circle_selector._selection_artist
            circle = (*map(float, patch.center), float(patch.radius))
            if not np.isfinite(circle).all() or circle[2] <= 0:
                return
            self.on_circle(*circle)
            self.status.value = f"Draft circle: center ({circle[0]:.4g}, {circle[1]:.4g}) cm; radius {circle[2]:.4g} cm."

    def refresh_detector(self):
        """Refresh exact density for draft time bounds, retaining the circle overlay."""
        if self.closed:
            return
        axis = self.axes[1]
        if not self.detector_available:
            axis.text(.5, .5, "Detector coordinates unavailable", ha="center", transform=axis.transAxes)
            self.canvas.draw_idle()
            return
        from matplotlib.colors import LogNorm
        start, stop = self.interval_provider()
        try:
            counts, xedges, yedges, count = detector_density(self.workflow, start, stop)
        except CalibrationInputError:
            self.detector_ready = False
            self.circle_selector.set_active(False)
            if self._density_artist is not None:
                self._density_artist.set_visible(False)
            if self._density_message is None:
                self._density_message = axis.text(.5, .5, "No detector density for the current draft interval",
                    ha="center", transform=axis.transAxes)
            self._density_message.set_visible(True)
            self.canvas.draw_idle()
            raise
        if self._density_artist is not None:
            self._density_artist.remove()
        if self._density_message is not None:
            self._density_message.set_visible(False)
        self._density_artist = axis.pcolormesh(xedges, yedges, np.ma.masked_equal(counts.T, 0),
            norm=LogNorm(vmin=1, vmax=max(2, counts.max())), cmap="inferno", shading="auto", rasterized=True)
        if self._density_colorbar is None:
            self._density_colorbar = self.figure.colorbar(self._density_artist, ax=axis, label="Detected ions per bin")
        else:
            self._density_colorbar.update_normal(self._density_artist)
        axis.set(xlim=(xedges[0], xedges[-1]), ylim=(yedges[0], yedges[-1]), aspect="equal",
            xlabel="Detector x (cm)", ylabel="Detector y (cm)",
            title=f"Detector density · {count:,} finite hits\nEvents [{start:,}, {stop:,})")
        axis.title.set_fontsize(10)
        self.density_count, self.density_interval = count, (start, stop)
        self.detector_bounds = (xedges[0], xedges[-1], yedges[0], yedges[-1])
        self.detector_ready = True
        self.circle_selector.set_active(self.enabled)
        self.canvas.draw_idle()

    def initial_circle(self):
        """Enclose the finite map bounds without choosing a scientific crop."""
        if not self.detector_ready:
            return None
        x0, x1, y0, y1 = self.detector_bounds
        radius = np.nextafter(np.hypot(x1-x0, y1-y0)/2., np.inf)
        return (float((x0+x1)/2.), float((y0+y1)/2.), float(radius))

    def set_enabled(self, enabled):
        self.enabled = bool(enabled) and not self.closed
        self.rectangle_selector.set_active(self.enabled and self.voltage_available)
        self.circle_selector.set_active(self.enabled and self.detector_ready)

    def set_draft(self, interval, circle=None):
        """Synchronize overlays from draft fields without callbacks or density work.

        Invalid/disabled draft selections hide their overlay. The interval spans
        the displayed voltage height because its y coordinates are not a filter.
        A disabled detector-circle checkbox is represented by ``circle=None``.
        """
        if self.closed:
            return

        def numbers(value, size):
            try:
                result = np.asarray(value, dtype=float)
            except (TypeError, ValueError, OverflowError):
                return None
            return result if result.shape == (size,) and np.isfinite(result).all() else None

        def overlay(selector, extents):
            if extents is None:
                if selector._selection_artist.get_visible():
                    selector.clear()
            elif not selector._selection_artist.get_visible() or not np.array_equal(selector.extents, extents):
                selector.set_visible(True)
                selector.extents = extents

        bounds = numbers(interval, 2)
        valid_interval = (self.voltage_available and bounds is not None and
            0 <= bounds[0] < bounds[1] <= self.workflow.n and np.equal(bounds, np.floor(bounds)).all())
        y0, y1 = self.axes[0].get_ylim()
        overlay(self.rectangle_selector, (*bounds, y0, y1) if valid_interval else None)
        geometry = numbers(circle, 3)
        valid_circle = self.detector_ready and geometry is not None and geometry[2] > 0
        if valid_circle:
            x, y, radius = geometry
            # Keep the entire initial circle and its handles inside the view.
            axis = self.axes[1]
            x0, x1 = axis.get_xlim()
            y0, y1 = axis.get_ylim()
            margin = .06*radius
            if x-radius < x0 or x+radius > x1 or y-radius < y0 or y+radius > y1:
                axis.set(xlim=(min(x0, x-radius-margin), max(x1, x+radius+margin)),
                         ylim=(min(y0, y-radius-margin), max(y1, y+radius+margin)))
            overlay(self.circle_selector, (x-radius, x+radius, y-radius, y+radius))
        else:
            overlay(self.circle_selector, None)

    def close(self):
        if self.closed:
            return
        self.set_enabled(False)
        self.closed = True
        for selector in (self.rectangle_selector, self.circle_selector):
            selector.disconnect_events()
        for connection in self._event_ids:
            self.canvas.mpl_disconnect(connection)
        if self.manager is not None:
            self.manager.destroy()
        for child in self._children:
            if hasattr(child, "close"):
                child.close()
        self.widget.close()
        plt.close(self.figure)


def create_crop_drawing(workflow, *, on_interval, on_circle, interval_provider=None, interactive=True):
    """Build retained, draft-only rectangle and true-circle drawing controls."""
    return CropDrawing(workflow, on_interval=on_interval, on_circle=on_circle,
                       interval_provider=interval_provider, interactive=interactive)
