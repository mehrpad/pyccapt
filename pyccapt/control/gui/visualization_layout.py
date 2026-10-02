"""Equal-size live plot panels and compact visualization controls."""
from PyQt6 import QtCore, QtWidgets


class EqualPlotRow(QtWidgets.QWidget):
    """Give every plot identical pixel dimensions, even at odd window widths."""
    spacing = 6

    def __init__(self, plots, parent=None):
        super().__init__(parent)
        self.plots = tuple(plots)
        width = max(plot.minimumWidth() for plot in self.plots)
        height = max(plot.minimumHeight() for plot in self.plots)
        self.setMinimumSize(len(self.plots) * width + (len(self.plots)-1)*self.spacing, height)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        for plot in self.plots:
            plot.setParent(self)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        width = (self.width() - (len(self.plots)-1)*self.spacing) // len(self.plots)
        # Any indivisible remainder stays at the right edge; panels never
        # differ by a pixel as they can with equally stretched grid columns.
        for index, plot in enumerate(self.plots):
            plot.setGeometry(index*(width+self.spacing), 0, width, self.height())


class VisualizationLayoutMixin:
    def _setup_compact_visualization_layout(self, parent):
        main = self.gridLayout_5
        for layout in (main, self.gridLayout_4, self.gridLayout, self.gridLayout_3,
                       self.gridLayout_3b, self.dc_hold_row, self.horizontalLayout_2,
                       self.fdm_bottom_row, self.horizontalLayout):
            while layout.count():
                layout.takeAt(0)
        self.gridLayout_6.setContentsMargins(6, 6, 6, 6)
        main.setSpacing(6)
        for column, (label, field) in enumerate((
            (self.label_200, self.voltage), (self.label_201, self.detection_rate),
            (self.label_206, self.hitmap_count), (self.label_fdm_header, self.fdm_count),
        )):
            header = QtWidgets.QHBoxLayout()
            header.setSpacing(4)
            header.addWidget(label)
            field.setFixedSize(90, 25)
            header.addWidget(field)
            header.addStretch(1)
            if column == 3:
                # Keep the experiment indicator in the top-right corner on
                # the same row as the FDM count, freeing a full header row.
                self.experiment_status_row.setParent(None)
                header.addLayout(self.experiment_status_row)
            main.addLayout(header, 0, column)
            main.setColumnStretch(column, 1)
        self.top_plot_row = EqualPlotRow((self.vdc_time, self.detection_rate_viz,
                                         self.detector_heatmap, self.detector_fdm), parent)
        main.addWidget(self.top_plot_row, 1, 0, 1, 4)
        voltage = QtWidgets.QHBoxLayout()
        voltage.setSpacing(4)
        for button, width in ((self.dc_hold, 64), (self.set_dc_voltage, 56)):
            button.setFixedSize(max(width, button.fontMetrics().horizontalAdvance(button.text()) + 18), 25)
            voltage.addWidget(button)
        self.set_dc_voltage_value.setFixedSize(90, 25)
        voltage.addWidget(self.set_dc_voltage_value)
        voltage.addStretch(1)
        main.addLayout(voltage, 2, 0)
        main.addWidget(self.detection_rate_range_switch, 2, 1,
                       alignment=QtCore.Qt.AlignmentFlag.AlignTop)
        hitmap = QtWidgets.QHBoxLayout()
        hitmap.setSpacing(4)
        hitmap.addWidget(self.reset_heatmap_v)
        self.hitmap_plot_size.setFixedSize(65, 25)
        self.hit_displayed.setFixedSize(90, 25)
        hitmap.addWidget(self.hitmap_plot_size)
        hitmap.addWidget(self.hit_displayed)
        main.addLayout(hitmap, 2, 2, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
        fdm = QtWidgets.QHBoxLayout()
        fdm.setSpacing(4)
        fdm.addWidget(self.fdm_last_events_switch)
        self.fdm_max_ions.setFixedSize(90, 25)
        fdm.addWidget(self.fdm_max_ions)
        main.addLayout(fdm, 2, 3, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
        self.gridLayout_2.removeItem(self.horizontalLayout)
        spectrum = QtWidgets.QVBoxLayout()
        spectrum.setSpacing(4)
        views = QtWidgets.QHBoxLayout()
        views.setSpacing(4)
        for button in (self.btn_view_mc_cal, self.btn_view_mc, self.btn_view_tof_cal, self.btn_view_tof):
            views.addWidget(button)
        self.calib_status_label.setWordWrap(True)
        self.calib_status_label.setMinimumWidth(0)
        views.addWidget(self.calib_status_label, 1)
        spectrum.addLayout(views)
        limits = QtWidgets.QHBoxLayout()
        limits.setSpacing(4)
        for widget in (self.spectrum_last_events_switch, self.num_last_events,
                       self.label_208, self.max_mc, self.label_209, self.max_tof):
            if isinstance(widget, QtWidgets.QLineEdit):
                widget.setFixedSize(90, 25)
            limits.addWidget(widget)
        limits.addStretch(1)
        spectrum.addLayout(limits)
        self.gridLayout_2.addLayout(spectrum, 2, 0)
        self.gridLayout_2.setSpacing(4)
        self.gridLayout_2.setParent(None)
        self.Error.setMinimumWidth(0)
        main.addLayout(self.gridLayout_2, 3, 0, 1, 4)
        main.setRowStretch(1, 2)
        main.setRowStretch(3, 2)
