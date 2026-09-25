"""Bounded, read-only live alignment monitor. No motor or detector commands."""
from collections import deque
import math
import time

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets


PHASE_COLOURS = {
    'ramp': (.7, .5, 1., 1.), 'moving': (.6, .65, .7, 1.),
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


class AlignmentPlotWindow(QtWidgets.QWidget):
    def __init__(self, variables, parent=None):
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self.variables = variables
        self.history = AlignmentPlotHistory()
        self.monitoring = False
        self.current_sample = None
        self.setWindowTitle('Automatic Alignment — live stage / detection rate')
        self.resize(960, 700)
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
        self.view = None
        try:
            import pyqtgraph.opengl as gl
        except ImportError:
            message = QtWidgets.QLabel(
                '3D plotting needs PyOpenGL. Install in the Python environment used to run PyCCAPT:\n'
                'python -m pip install PyOpenGL\nThen restart PyCCAPT.')
            message.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            message.setWordWrap(True)
            layout.addWidget(message)
        else:
            self.view = gl.GLViewWidget()
            self.view.setBackgroundColor('#17212b')
            layout.addWidget(self.view, 1)
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
        layout.addWidget(QtWidgets.QLabel(
            'X / Y: offset from saved sample position (µm). Vertical: detection rate (%), not stage Z.\n'
            'Blue: coarse • amber: fine • green: aligned • purple: ramp • grey: moving • white: latest\n'
            'Drag to rotate; wheel to zoom. Last 10,000 display readings per sample; updates up to 5 Hz.'))
        self.sample_selector.currentIndexChanged.connect(self.redraw)
        reset.clicked.connect(self.reset_view)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.refresh)

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
