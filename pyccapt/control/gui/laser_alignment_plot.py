"""Embedded bounded OpenGL measurements plus a laser focus response curve."""
from collections import deque
import numpy as np
from PyQt6 import QtGui, QtWidgets


class LaserAlignmentPlot(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.records = deque(maxlen=5000)
        self.session = None
        self.last_time = None
        layout = QtWidgets.QVBoxLayout(self)
        self.metric = QtWidgets.QComboBox()
        self.metric.addItems(['Detection rate (%)', 'DC voltage (V)'])
        layout.addWidget(self.metric)
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)
        self.view = None
        try:
            import pyqtgraph as pg
            import pyqtgraph.opengl as gl
        except ImportError:
            layout.addWidget(QtWidgets.QLabel('Live plots require pyqtgraph and PyOpenGL.'))
        else:
            self.view = gl.GLViewWidget()
            self.view.setMinimumSize(350, 280)
            self.view.setBackgroundColor('#17212b')
            self.view.setCameraPosition(pos=QtGui.QVector3D(0, 0, 4), distance=40, elevation=28, azimuth=45)
            grid = gl.GLGridItem()
            grid.setSize(20, 20)
            grid.setSpacing(2, 2)
            self.view.addItem(grid)
            self.points = gl.GLScatterPlotItem(size=7, pxMode=True)
            self.trace = gl.GLLinePlotItem(mode='line_strip', color=(.6, .6, .7, .5))
            self.current = gl.GLScatterPlotItem(size=12, color=(1, 1, 1, 1), pxMode=True)
            for item in (self.points, self.trace, self.current): self.view.addItem(item)
            self.labels = [gl.GLTextItem(color='white') for _ in range(6)]
            for label in self.labels: self.view.addItem(label)
            self.tabs.addTab(self.view, 'XY response — 3D')
            self.focus = pg.PlotWidget()
            self.focus.setLabel('bottom', 'Laser stage Z offset', units='µm')
            self.focus.setLabel('left', 'Detection rate', units='%')
            self.focus_curve = self.focus.plot(pen=None, symbol='o', symbolSize=7)
            self.tabs.addTab(self.focus, 'Z focus')
        self.caption = QtWidgets.QLabel('Measured points only. Drag to rotate; wheel to zoom. White: latest.')
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)
        self.metric.currentIndexChanged.connect(self.redraw)

    def append(self, record):
        if not record or record.get('time') == self.last_time:
            return
        if record['session'] != self.session:
            self.session = record['session']
            self.records.clear()
        self.last_time = record['time']
        self.records.append(record)
        self.redraw()

    def redraw(self):
        if not self.records or self.view is None:
            return
        data = list(self.records)
        last = data[-1]
        xyz = (np.array([r['position_m'] for r in data])-np.array(last['origin_m']))*1e6
        key = 'rate_percent' if self.metric.currentIndex() == 0 else 'voltage'
        values = np.array([r[key] for r in data])
        xy_scale = max(1., float(abs(xyz[:, :2]).max()))
        z_scale = max(.001, float(values.max()))
        rendered = np.column_stack((xyz[:, :2]*10/xy_scale, values*12/z_scale))
        colours = {'coarse': (.2, .7, 1, 1), 'fine': (1, .7, .1, 1),
                   'focus': (.8, .4, 1, 1), 'tracking': (.2, 1, .5, 1)}
        self.points.setData(pos=rendered, color=np.array([colours.get(r['phase'], (1, 1, 1, 1)) for r in data]))
        self.trace.setData(pos=rendered)
        self.current.setData(pos=rendered[-1:])
        annotations = [((0, 12, 0), 'X offset (µm)'), ((12, 0, 0), 'Y offset (µm)'),
                       ((-10, -10, 14), self.metric.currentText()),
                       ((10, 10, 0), f'+{xy_scale:.3g}'), ((-10, 10, 0), f'-{xy_scale:.3g}'),
                       ((-10, -10, 12), f'{z_scale:.4g}')]
        for label, (pos, text) in zip(self.labels, annotations): label.setData(pos=pos, text=text)
        focus = [(xyz[i, 2], r['rate_percent']) for i, r in enumerate(data) if r['phase'] == 'focus']
        self.focus_curve.setData(x=[r[0] for r in focus], y=[r[1] for r in focus])
        quality = last.get('quality', {})
        quality_text = f"TOF width {quality['width_ns']:.3g} ns" if quality else 'Signal only; peak quality unavailable'
        self.caption.setText(f"{last['phase']} | {last['rate_percent']:.4g}% | {last['voltage']:.0f} V | {quality_text}\n"
                             f'X/Y: ±{xy_scale:.3g} µm. Vertical: 0–{z_scale:.4g} '+
                             ('%' if key == 'rate_percent' else 'V')+'. Offsets are laser-stage travel.')
