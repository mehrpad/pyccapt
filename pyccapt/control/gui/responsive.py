"""Screen-aware window sizing without scaling instrument controls or fonts."""
from PyQt6 import QtCore, QtWidgets


class ResponsiveWindow(QtCore.QObject):
    """Keep the existing layout; scroll only below its readable minimum size.

    Sizes are Qt logical pixels, so the operating system's DPI scaling continues
    to apply. The controller never changes fonts, control widths or signals.
    """

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self._first_show = True
        self._screen_connected = False
        self._observed_screen = None
        self._pending = False
        self._content_pending = False
        preferred_size = window.size()
        self.scroll = QtWidgets.QScrollArea(window)
        self.scroll.setObjectName('responsive_scroll_area')
        self.scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        # Qt's automatic sizing treats a plot's preferred height as a minimum
        # when another widget in the layout wraps text. Size the content from
        # its actual layout minimum instead, so plots can shrink normally.
        self.scroll.setWidgetResizable(False)
        self.scroll.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                  QtWidgets.QSizePolicy.Policy.Expanding)
        if isinstance(window, QtWidgets.QMainWindow):
            self.content = window.takeCentralWidget()
            window.setCentralWidget(self.scroll)
        else:
            self.content = QtWidgets.QWidget()
            # QWidget.setLayout transfers the old layout and its children.
            self.content.setLayout(window.layout())
            outer = QtWidgets.QVBoxLayout(window)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetNoConstraint)
            outer.addWidget(self.scroll)
        self.content.setObjectName(self.content.objectName() or 'responsive_content')
        self.content.layout().setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
        self.scroll.setWidget(self.content)
        self.scroll.viewport().installEventFilter(self)
        self.content.installEventFilter(self)
        # Only the viewport gets smaller; the content keeps its layout minimum.
        window.setMinimumSize(320, 200)
        window.resize(preferred_size)
        window.installEventFilter(self)

    def eventFilter(self, watched, event):
        if ((watched is self.scroll.viewport() and event.type() == QtCore.QEvent.Type.Resize)
                or (watched is self.content and event.type() == QtCore.QEvent.Type.LayoutRequest)):
            if not self._content_pending:
                self._content_pending = True
                QtCore.QTimer.singleShot(0, self._resize_content)
        if watched is self.window and event.type() == QtCore.QEvent.Type.Show:
            handle = self.window.windowHandle()
            if handle is not None and not self._screen_connected:
                handle.screenChanged.connect(self._screen_changed)
                self._screen_connected = True
            self._watch_screen(self.window.screen())
            self._schedule_fit()
        return super().eventFilter(watched, event)

    def _resize_content(self):
        self._content_pending = False
        self.content.layout().activate()
        minimum = self.content.minimumSizeHint().expandedTo(self.content.minimumSize())
        self.content.resize(self.scroll.viewport().size().expandedTo(minimum))

    def _schedule_fit(self, *_args):
        if not self._pending:
            self._pending = True
            QtCore.QTimer.singleShot(0, self._fit_current_screen)

    def _watch_screen(self, screen):
        if screen is self._observed_screen:
            return
        if self._observed_screen is not None:
            try:
                self._observed_screen.availableGeometryChanged.disconnect(self._schedule_fit)
            except (RuntimeError, TypeError):
                pass  # The previous monitor may have been disconnected.
        self._observed_screen = screen
        if screen is not None:
            screen.availableGeometryChanged.connect(self._schedule_fit)

    def _screen_changed(self, screen):
        self._watch_screen(screen)
        self._schedule_fit()

    def _fit_current_screen(self):
        self._pending = False
        if self.window.isVisible() and self.window.isWindow():
            self.fit_to_screen(self.window.screen().availableGeometry())

    def fit_to_screen(self, available):
        """Fit the frame inside a monitor's work area, including taskbar space."""
        window = self.window
        if window.isMaximized() or window.isFullScreen():
            self._first_show = False
            return
        self.content.layout().activate()
        area = available.adjusted(8, 8, -8, -8)
        decoration = window.frameGeometry().size() - window.size()
        chrome = window.size() - self.scroll.viewport().size()
        desired = window.size()
        if self._first_show:
            desired = desired.expandedTo(self.content.minimumSizeHint() + chrome)
        maximum = area.size() - decoration
        # Tiny work areas must not inherit a minimum larger than the screen.
        window.setMinimumSize(min(320, maximum.width()), min(200, maximum.height()))
        window.resize(desired.boundedTo(maximum))
        frame = window.frameGeometry()
        x = min(max(frame.left(), area.left()), area.right() - frame.width() + 1)
        y = min(max(frame.top(), area.top()), area.bottom() - frame.height() + 1)
        window.move(x, y)
        self._resize_content()
        self._first_show = False


def make_window_responsive(window):
    """Install once. Embedded controls use their containing window's viewport."""
    if not window.isWindow() or hasattr(window, '_responsive_window'):
        return
    window._responsive_window = ResponsiveWindow(window)
