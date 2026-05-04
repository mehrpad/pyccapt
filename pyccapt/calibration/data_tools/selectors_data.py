from matplotlib.patches import Circle
from matplotlib.widgets import RectangleSelector


class CircleSelector(RectangleSelector):
    """
    Select a circular region of an Axes.

    For the cursor to remain responsive you must keep a reference to it.

    Press and release events triggered at the same coordinates outside the
    selection will clear the selector, except when
    `ignore_event_outside=True`.

    Examples
    --------
    :doc:`/gallery/widgets/rectangle_selector`
    """

    def _init_shape(self, **props):
        return Circle((0, 0), 0, visible=False, **props)

    def _draw_shape(self, extents):
        x0, x1, y0, y1 = extents
        xmin, xmax = sorted([x0, x1])
        ymin, ymax = sorted([y0, y1])
        center = [x0 + (x1 - x0) / 2., y0 + (y1 - y0) / 2.]
        radius = min((xmax - xmin) / 2., (ymax - ymin) / 2.)

        self._selection_artist.center = center
        self._selection_artist.radius = radius

    @property
    def _rect_bbox(self):
        x, y = self._selection_artist.center
        radius = self._selection_artist.radius
        return x - radius, y - radius, 2 * radius, 2 * radius

def onselect(eclick, erelease, variables):
    """
    Callback function for the selection event.

    Args:
        eclick (MouseEvent): Event object representing the click event.
        erelease (MouseEvent): Event object representing the release event.
        variables (object): Object containing the variables.

    """
    if any(value is None for value in (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata)):
        return

    x0, x1 = sorted((float(eclick.xdata), float(erelease.xdata)))
    y0, y1 = sorted((float(eclick.ydata), float(erelease.ydata)))

    variables.selected_x_fdm = x0 + (x1 - x0) / 2
    variables.selected_y_fdm = y0 + (y1 - y0) / 2
    variables.roi_fdm = min(x1 - x0, y1 - y0) / 2

def line_select_callback(eclick, erelease, variables):
    """
    Callback function for line selection event.

    Args:
        eclick (MouseEvent): Event object representing the press event.
        erelease (MouseEvent): Event object representing the release event.
        variables (object): Object containing the variables.

    """
    if any(value is None for value in (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata)):
        return

    x1, x2 = sorted((float(eclick.xdata), float(erelease.xdata)))
    y1, y2 = sorted((float(eclick.ydata), float(erelease.ydata)))

    variables.selected_x1 = x1
    variables.selected_x2 = x2
    variables.selected_y1 = y1
    variables.selected_y2 = y2
    variables.selected_calculated = False


def toggle_selector(event,):
    """
    Toggles the rectangle selector based on the key press event.

    Args:
        event (KeyEvent): Event object representing the key press event.

    """

    try:
        if event.key in ['Q', 'q'] and toggle_selector.RS.active:
            toggle_selector.RS.set_active(False)
        if event.key in ['A', 'a'] and not toggle_selector.RS.active:
            toggle_selector.RS.set_active(True)

    except AttributeError:
        pass
