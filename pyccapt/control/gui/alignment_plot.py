"""Bounded, read-only live alignment monitor. No motor or detector commands."""
from collections import deque
import math
import time

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets


PHASE_COLOURS = {
    'ramp': (.7, .5, 1., 1.), 'moving': (.6, .65, .7, 1.),
    'confirming': (.7, .5, 1., 1.),
    'coarse': (.2, .7, 1., 1.), 'fine': (1., .7, .15, 1.),
    'aligned': (.2, .9, .4, 1.),
}


class AlignmentPlotHistory:
    """Display samples, not acquisition data; a fixed memory cap per sample."""
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.sequence_id = None
        self.samples = {}
        self.last_time = -math.inf

    def append(self, snapshot):
        try:
            sample = int(snapshot['sample'])
            position = tuple(float(v) for v in snapshot['position_m'])
            origin = tuple(float(v) for v in snapshot['origin_m'])
            values = [float(snapshot[k]) for k in ('time', 'rate_percent', 'target_percent', 'voltage')]
            if (sample not in (1, 2, 3) or len(position) != 3 or len(origin) != 3
                    or not all(math.isfinite(v) for v in (*position, *origin, *values))
                    or values[1] < 0 or values[2] <= 0):
                return False
            sequence = snapshot['sequence_id']
        except (KeyError, TypeError, ValueError):
            return False
        if sequence != self.sequence_id:
            self.sequence_id = sequence
            self.samples.clear()
            self.last_time = -math.inf
        if values[0] <= self.last_time:
            return False
        self.last_time = values[0]
        self.samples.setdefault(sample, deque(maxlen=self.capacity)).append(dict(snapshot))
        return True

    def coordinates(self, sample):
        records = list(self.samples.get(sample, ()))
        if not records:
            return np.empty((0, 3)), records
        # Relative XY retains sub-micrometre detail at millimetre stage offsets.
        origin = np.asarray(records[0]['origin_m'])[:2]
        xy = (np.asarray([r['position_m'][:2] for r in records])-origin)*1e6
        return np.column_stack((xy, [r['rate_percent'] for r in records])), records


class AlignmentHitmapHistory:
    """Retain only new, paired detector hits from the alignment event window."""
    def __init__(self, capacity=2000):
        self.points = deque(maxlen=capacity)
        self.epoch = None
        self.sequence = None

    def reset(self):
        self.points.clear()

    def prepare_epoch(self, epoch):
        if epoch != self.epoch:
            self.epoch = epoch
            self.sequence = None
            self.points.clear()
            return True
        return False

    def append(self, window):
        try:
            epoch = window['epoch']
            sequence = int(window['sequence'])
            points = np.asarray(window['points_mm'], dtype=float)
            if (not epoch or sequence < 0 or points.ndim != 2 or points.shape[1] != 2
                    or not np.isfinite(points).all()):
                return False
            points = points[-self.points.maxlen:]
        except (KeyError, TypeError, ValueError):
            return False
        if epoch != self.epoch or self.sequence is None or sequence < self.sequence:
            self.epoch = epoch
            self.sequence = sequence
            self.points.clear()
            self.points.extend(map(tuple, points))
            return True
        if sequence == self.sequence:
            return False
        count = min(sequence-self.sequence, len(points))
        self.sequence = sequence
        self.points.extend(map(tuple, points[-count:]))
        return True

    def coordinates(self):
        return np.asarray(self.points, dtype=float).reshape(-1, 2)


class AlignmentPlotWindow(QtWidgets.QWidget):
    def __init__(self, variables, parent=None, detector_radius_mm=None):
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self.variables = variables
        self.history = AlignmentPlotHistory()
        self.hitmap_history = AlignmentHitmapHistory()
        self.monitoring = False
        self.current_sample = None
        self.detector_radius_mm = (float(detector_radius_mm) if detector_radius_mm is not None
                                   else float(getattr(self.variables, 'alignment_settings', {}).get(
                                       'detector_radius_mm', 40.)))
        self.setWindowTitle('Automatic Alignment — live stage / detection rate')
        self.resize(660, 350)
        layout = QtWidgets.QVBoxLayout(self)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel('Sample'))
        self.sample_selector = QtWidgets.QComboBox()
        controls.addWidget(self.sample_selector)
        controls.addStretch()
        reset = QtWidgets.QPushButton('Reset view')
        controls.addWidget(reset)
        layout.addLayout(controls)
        self.status = QtWidgets.QLabel('Waiting for an automatic alignment experiment.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        plots = QtWidgets.QHBoxLayout()
        layout.addLayout(plots, 1)
        self.view = None
        try:
            import pyqtgraph.opengl as gl
        except ImportError:
            message = QtWidgets.QLabel(
                '3D plotting needs PyOpenGL. Install in the Python environment used to run PyCCAPT:\n'
                'python -m pip install PyOpenGL\nThen restart PyCCAPT.')
            message.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            message.setWordWrap(True)
            plots.addWidget(message, 3)
        else:
            self.view = gl.GLViewWidget()
            self.view.setBackgroundColor('#17212b')
            plots.addWidget(self.view, 3)
            grid = gl.GLGridItem()
            grid.setSize(20, 20)
            grid.setSpacing(2, 2)
            self.view.addItem(grid)
            axes = gl.GLLinePlotItem(pos=np.array([
                [-10, 10, 0], [10, 10, 0], [10, -10, 0], [10, 10, 0],
                [-10, -10, 0], [-10, -10, 12]], dtype=float),
                mode='lines', color=(.8, .85, .9, 1), width=1)
            self.view.addItem(axes)
            self.trace = gl.GLLinePlotItem(mode='line_strip', color=(.5, .6, .7, .5), width=1)
            self.points = gl.GLScatterPlotItem(size=5, pxMode=True)
            self.latest = gl.GLScatterPlotItem(size=12, color=(1., 1., 1., 1.), pxMode=True)
            for item in (self.trace, self.points, self.latest):
                self.view.addItem(item)
            self.labels = [gl.GLTextItem(color='white', font=QtGui.QFont('Arial', 10)) for _ in range(12)]
            for label in self.labels:
                self.view.addItem(label)
            self.reset_view()
        self._setup_hitmap(plots)
        help_label = QtWidgets.QLabel(
            '3D: X/Y offset (µm), height = detection rate (%). Drag to rotate; wheel to zoom.\n'
            'Detector: last 2000 hits. Reset clears this display only.')
        help_label.setWordWrap(True)
        help_label.setToolTip(
            'Blue: coarse; amber: fine; green: aligned; purple: ramp/confirming; grey: moving; white: latest.\n'
            'The 3D view retains 10,000 readings per sample and updates up to 5 Hz.\n'
            'Detector hits come from the current alignment observation.')
        layout.addWidget(help_label)
        self.sample_selector.currentIndexChanged.connect(self.redraw)
        reset.clicked.connect(self.reset_view)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.refresh)

    def _setup_hitmap(self, plots):
        import pyqtgraph as pg
        panel = QtWidgets.QWidget(self)
        column = QtWidgets.QVBoxLayout(panel)
        column.setContentsMargins(0, 0, 0, 0)
        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel('Detector'))
        self.hitmap_count = QtWidgets.QLineEdit('0')
        self.hitmap_count.setReadOnly(True)
        self.hitmap_count.setFixedWidth(90)
        header.addWidget(self.hitmap_count)
        header.addStretch()
        column.addLayout(header)
        self.detector_hitmap = pg.PlotWidget()
        self.detector_hitmap.setBackground('w')
        self.detector_hitmap.setLabel('left', 'X_det', units='mm', color='r')
        self.detector_hitmap.setLabel('bottom', 'Y_det', units='mm', color='r')
        radius = self.detector_radius_mm
        self.detector_hitmap.setXRange(-radius*1.08, radius*1.08, padding=0)
        self.detector_hitmap.setYRange(-radius*1.08, radius*1.08, padding=0)
        self.detector_hitmap.getViewBox().setAspectLocked(True)
        self.detector_scatter = pg.ScatterPlotItem(size=1., brush='black', pen=None)
        self.detector_hitmap.addItem(self.detector_scatter)
        self.detector_circle = QtWidgets.QGraphicsEllipseItem(-radius, -radius, 2*radius, 2*radius)
        self.detector_circle.setPen(pg.mkPen(color=(255, 0, 0), width=2))
        self.detector_hitmap.addItem(self.detector_circle)
        self.footprint_circle = QtWidgets.QGraphicsEllipseItem()
        self.footprint_circle.setPen(pg.mkPen(color=(0, 170, 60), width=2))
        self.footprint_circle.setVisible(False)
        self.detector_hitmap.addItem(self.footprint_circle)
        column.addWidget(self.detector_hitmap, 1)
        controls = QtWidgets.QHBoxLayout()
        self.hitmap_reset = QtWidgets.QPushButton('Reset')
        self.hitmap_reset.clicked.connect(self.reset_hitmap)
        controls.addWidget(self.hitmap_reset)
        self.hitmap_point_size = QtWidgets.QDoubleSpinBox()
        self.hitmap_point_size.setRange(.1, 10.)
        self.hitmap_point_size.setSingleStep(.1)
        self.hitmap_point_size.setValue(1.)
        self.hitmap_point_size.valueChanged.connect(self.redraw_hitmap)
        controls.addWidget(self.hitmap_point_size)
        self.hitmap_limit = QtWidgets.QLineEdit('2000')
        self.hitmap_limit.setReadOnly(True)
        self.hitmap_limit.setFixedWidth(70)
        controls.addWidget(self.hitmap_limit)
        controls.addStretch()
        column.addLayout(controls)
        plots.addWidget(panel, 2)

    def reset_hitmap(self):
        self.hitmap_history.reset()
        self.redraw_hitmap()

    def redraw_hitmap(self):
        points = self.hitmap_history.coordinates()
        self.detector_scatter.setSize(self.hitmap_point_size.value())
        self.detector_scatter.setData(x=points[:, 0], y=points[:, 1])
        self.hitmap_count.setText(str(len(points)))
        fit = getattr(self.variables, 'alignment_status', {}).get('footprint', {})
        visible = bool(getattr(self.variables, 'automatic_alignment_enabled', False)
                       and fit.get('valid', False))
        self.footprint_circle.setVisible(visible)
        if visible:
            x, y = fit['centre_mm']
            radius = fit['radius_mm']
            self.footprint_circle.setRect(x-radius, y-radius, 2*radius, 2*radius)

    def reset_view(self):
        if self.view is not None:
            self.view.setCameraPosition(pos=QtGui.QVector3D(0, 0, 4), distance=42,
                                        elevation=28, azimuth=45)

    def set_monitoring(self, enabled):
        self.monitoring = enabled
        if enabled:
            self.timer.start()
            self.showNormal()
        else:
            self.timer.stop()
            self.hide()

    def closeEvent(self, event):
        if self.monitoring:
            event.ignore()
            self.showMinimized()
        else:
            event.accept()

    def refresh(self):
        snapshot = self.variables.alignment_plot_snapshot
        if (not self.variables.automatic_alignment_enabled or not self.variables.start_flag
                or not snapshot or not 0 <= time.monotonic()-snapshot.get('time', -math.inf) <= 2):
            self.status.setText('Waiting / experiment stopped — previous measured points retained.')
            return
        epoch = self.variables.alignment_window_epoch
        cleared = self.hitmap_history.prepare_epoch(epoch)
        window = self.variables.alignment_events
        if (window and window.get('epoch') == epoch and self.hitmap_history.append(window)) or cleared:
            self.redraw_hitmap()
        sequence_changed = snapshot.get('sequence_id') != self.history.sequence_id
        if not self.history.append(snapshot):
            return
        sample = snapshot['sample']
        changed = sequence_changed or sample != self.current_sample
        self.current_sample = sample
        self.sample_selector.blockSignals(True)
        if sequence_changed:
            self.sample_selector.clear()
        if self.sample_selector.findData(sample) < 0:
            self.sample_selector.addItem(f'Sample {sample}', sample)
        if changed:
            self.sample_selector.setCurrentIndex(self.sample_selector.findData(sample))
        self.sample_selector.blockSignals(False)
        self.redraw()

    def redraw(self):
        sample = self.sample_selector.currentData()
        coords, records = self.history.coordinates(sample)
        if not records:
            return
        last = records[-1]
        xyz = np.asarray(last['position_m'])*1000
        self.status.setText(
            f"Sample {sample} | {last['phase']} | {last['voltage']:.0f} V | "
            f"rate {last['rate_percent']:.3f}% / target {last['target_percent']:.3f}%\n"
            f"Actual stage: X {xyz[0]:.6f}, Y {xyz[1]:.6f}, Z {xyz[2]:.6f} mm")
        if self.view is None:
            return
        xy_span = max(1., float(np.max(np.abs(coords[:, :2]))))
        rate_max = max(last['target_percent'], float(coords[:, 2].max()), .001)
        rendered = coords / np.array([xy_span/10, xy_span/10, rate_max/12])
        colours = np.array([PHASE_COLOURS.get(r['phase'], (.6, .6, .6, 1.)) for r in records])
        self.points.setData(pos=rendered, color=colours)
        self.trace.setData(pos=rendered)
        self.latest.setData(pos=rendered[-1:])
        annotations = []
        for fraction in (-1., 0., 1.):
            annotations.extend([
                ((fraction*10, 11, 0), f'{fraction*xy_span:.3g}'),
                ((11, fraction*10, 0), f'{fraction*xy_span:.3g}'),
                ((-11, -10, (fraction+1)*6), f'{(fraction+1)*rate_max/2:.3g}%'),
            ])
        annotations.extend([((0, 13, 0), 'X offset (µm)'), ((13, 0, 0), 'Y offset (µm)'),
                            ((-10, -10, 14), 'Detection rate (%)')])
        for label, (pos, text) in zip(self.labels, annotations):
            label.setData(pos=pos, text=text)
