import multiprocessing
import os
import sys
import time

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
# from numba import njit
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QTimer

# Local module and scripts
from pyccapt.control.core import runtime, tof2mc_simple
from pyccapt.control.devices import initialize_devices


class Ui_Visualization(object):

	def __init__(self, variables, conf, x_plot, y_plot, t_plot, main_v_dc_plot):
		"""
		Constructor for the Visualization UI class.

		Args:
			variables (object): Global experiment variables.
			conf (dict): Configuration settings.
			x_plot (multiprocessing.Array): Array for storing the x-axis values of the mass spectrum.
			y_plot (multiprocessing.Array): Array for storing the y-axis values of the mass spectrum.
			t_plot (multiprocessing.Array): Array for storing the time values of the mass spectrum.
			main_v_dc_plot (multiprocessing.Array): Array for storing the main voltage values of the mass spectrum.

		"""
		self.path_meta = None
		self.num_hit_display = 0
		self.bins_detector = (256, 256)
		detector_diameter = conf["detector_diameter"]
		detector_diameter = detector_diameter / 2
		self.range = [[-detector_diameter, detector_diameter], [-detector_diameter, detector_diameter]]
		self.hist_fdm, xedges, yedges = np.histogram2d([], [], bins=self.bins_detector, range=self.range)
		self.index_hist_mc = None
		self.index_hist_tof = None
		self.max_tof_val = None
		self.max_mc_val = None
		self.last_100_thousand_det_x_heatmap = np.array([])
		self.last_100_thousand_det_y_heatmap = np.array([])
		self.last_100_thousand_t = np.array([])
		self.last_100_thousand_v = np.array([])
		self.last_100_thousand_det_x = np.array([])
		self.last_100_thousand_det_y = np.array([])
		self.length_events = 0
		self.styles = None
		self.num_event_mc_tof = None
		self.mc_tof_last_events_flag = False
		self.change_detection_rate_range = False
		self.start_time_metadata = 0
		self.start_main_exp = 0
		self.index_plot_start = 0
		self.variables = variables
		self.conf = conf
		self.x_plot = x_plot
		self.y_plot = y_plot
		self.t_plot = t_plot
		self.main_v_dc_plot = main_v_dc_plot
		self.counter_source = ''
		self.index_plot_save = 0
		self.index_plot = 0
		self.index_wait_on_plot_start = 0
		self.index_auto_scale_graph = 0
		self.heatmap_fdm_switch_flag = 'heatmap'

		self.bins_mc = np.arange(0, self.conf["max_mass"] + self.conf['bin_size'], self.conf['bin_size'])
		self.bins_tof = np.arange(0, self.conf["max_tof"] + self.conf['bin_size'], self.conf['bin_size'])
		self.hist_mc = np.zeros(len(self.bins_mc) - 1)
		self.hist_tof = np.zeros(len(self.bins_tof) - 1)

		self.update_timer = QTimer()  # Create a QTimer for updating graphs
		self.update_timer.timeout.connect(self.update_graphs)  # Connect it to the update_graphs slot
		self.visualization_window = None  # Inâ™ itialize the attribute

	def setupUi(self, Visualization):
		"""
        Setup the UI for the Visualization window.

        Args:
        Visualization (QMainWindow): Visualization window.

        Return:
        None
        """
		Visualization.setObjectName("Visualization")
		Visualization.resize(822, 647)
		self.gridLayout_6 = QtWidgets.QGridLayout(Visualization)
		self.gridLayout_6.setObjectName("gridLayout_6")
		self.gridLayout_5 = QtWidgets.QGridLayout()
		self.gridLayout_5.setObjectName("gridLayout_5")
		self.gridLayout_4 = QtWidgets.QGridLayout()
		self.gridLayout_4.setObjectName("gridLayout_4")
		self.label_200 = QtWidgets.QLabel(parent=Visualization)
		font = QtGui.QFont()
		font.setBold(True)
		self.label_200.setFont(font)
		self.label_200.setObjectName("label_200")
		self.gridLayout_4.addWidget(self.label_200, 0, 0, 1, 1)
		self.voltage = QtWidgets.QLineEdit(parent=Visualization)
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
		sizePolicy.setHorizontalStretch(0)
		sizePolicy.setVerticalStretch(0)
		sizePolicy.setHeightForWidth(self.voltage.sizePolicy().hasHeightForWidth())
		self.voltage.setSizePolicy(sizePolicy)
		self.voltage.setMinimumSize(QtCore.QSize(100, 20))
		self.voltage.setStyleSheet("QLineEdit{\n"
		                           "                                            background: rgb(223,223,233)\n"
		                           "                                            }\n"
		                           "                                        ")
		self.voltage.setObjectName("voltage")
		self.gridLayout_4.addWidget(self.voltage, 0, 1, 1, 1)
		spacerItem = QtWidgets.QSpacerItem(26, 17, QtWidgets.QSizePolicy.Policy.Expanding,
		                                   QtWidgets.QSizePolicy.Policy.Minimum)
		self.gridLayout_4.addItem(spacerItem, 0, 2, 1, 1)
		####
		# self.vdc_time = QtWidgets.QGraphicsView(parent=Visualization)
		self.vdc_time = pg.PlotWidget(parent=Visualization)
		self.vdc_time.setBackground('w')
		####
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
		                                   QtWidgets.QSizePolicy.Policy.Expanding)
		sizePolicy.setHorizontalStretch(1)
		sizePolicy.setVerticalStretch(1)
		sizePolicy.setHeightForWidth(self.vdc_time.sizePolicy().hasHeightForWidth())
		self.vdc_time.setSizePolicy(sizePolicy)
		self.vdc_time.setMinimumSize(QtCore.QSize(250, 250))
		self.vdc_time.setStyleSheet("QWidget{\n"
		                            "                                                    border: 0.5px solid gray;\n"
		                            "                                                    }\n"
		                            "                                                ")
		self.vdc_time.setObjectName("vdc_time")
		self.gridLayout_4.addWidget(self.vdc_time, 1, 0, 1, 3)
		self.dc_hold = QtWidgets.QPushButton(parent=Visualization)
		self.dc_hold.setMinimumSize(QtCore.QSize(100, 20))
		self.dc_hold.setMaximumSize(QtCore.QSize(100, 16777215))
		self.dc_hold.setObjectName("dc_hold")
		self.gridLayout_4.addWidget(self.dc_hold, 2, 0, 1, 2)
		self.gridLayout_5.addLayout(self.gridLayout_4, 0, 0, 1, 1)
		self.gridLayout = QtWidgets.QGridLayout()
		self.gridLayout.setObjectName("gridLayout")
		self.label_201 = QtWidgets.QLabel(parent=Visualization)
		font = QtGui.QFont()
		font.setBold(True)
		self.label_201.setFont(font)
		self.label_201.setObjectName("label_201")
		self.gridLayout.addWidget(self.label_201, 0, 0, 1, 1)
		self.detection_rate = QtWidgets.QLineEdit(parent=Visualization)
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
		sizePolicy.setHorizontalStretch(0)
		sizePolicy.setVerticalStretch(0)
		sizePolicy.setHeightForWidth(self.detection_rate.sizePolicy().hasHeightForWidth())
		self.detection_rate.setSizePolicy(sizePolicy)
		self.detection_rate.setMinimumSize(QtCore.QSize(100, 20))
		self.detection_rate.setStyleSheet("QLineEdit{\n"
		                                  "                                            background: rgb(223,223,233)\n"
		                                  "                                            }\n"
		                                  "                                        ")
		self.detection_rate.setObjectName("detection_rate")
		self.gridLayout.addWidget(self.detection_rate, 0, 1, 1, 1)
		spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
		                                    QtWidgets.QSizePolicy.Policy.Minimum)
		self.gridLayout.addItem(spacerItem1, 0, 2, 1, 1)
		####
		# self.detection_rate_viz = QtWidgets.QGraphicsView(parent=Visualization)
		self.detection_rate_viz = pg.PlotWidget(parent=Visualization)
		self.detection_rate_viz.setBackground('w')
		####
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
		                                   QtWidgets.QSizePolicy.Policy.Expanding)
		sizePolicy.setHorizontalStretch(1)
		sizePolicy.setVerticalStretch(1)
		sizePolicy.setHeightForWidth(self.detection_rate_viz.sizePolicy().hasHeightForWidth())
		self.detection_rate_viz.setSizePolicy(sizePolicy)
		self.detection_rate_viz.setMinimumSize(QtCore.QSize(250, 250))
		self.detection_rate_viz.setStyleSheet("QWidget{\n"
		                                      "                                            border: 0.5px solid gray;\n"
		                                      "                                            }\n"
		                                      "                                        ")
		self.detection_rate_viz.setObjectName("detection_rate_viz")
		self.gridLayout.addWidget(self.detection_rate_viz, 1, 0, 1, 3)
		self.detection_rate_range_switch = QtWidgets.QPushButton(parent=Visualization)
		self.detection_rate_range_switch.setMinimumSize(QtCore.QSize(0, 20))
		self.detection_rate_range_switch.setMaximumSize(QtCore.QSize(100, 16777215))
		self.detection_rate_range_switch.setObjectName("detection_rate_range_switch")
		self.gridLayout.addWidget(self.detection_rate_range_switch, 2, 0, 1, 1)
		self.gridLayout_5.addLayout(self.gridLayout, 0, 1, 1, 1)
		self.gridLayout_3 = QtWidgets.QGridLayout()
		self.gridLayout_3.setObjectName("gridLayout_3")
		self.label_206 = QtWidgets.QLabel(parent=Visualization)
		font = QtGui.QFont()
		font.setBold(True)
		self.label_206.setFont(font)
		self.label_206.setObjectName("label_206")
		self.gridLayout_3.addWidget(self.label_206, 0, 0, 1, 1)
		self.hitmap_count = QtWidgets.QLineEdit(parent=Visualization)
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
		sizePolicy.setHorizontalStretch(0)
		sizePolicy.setVerticalStretch(0)
		sizePolicy.setHeightForWidth(self.hitmap_count.sizePolicy().hasHeightForWidth())
		self.hitmap_count.setSizePolicy(sizePolicy)
		self.hitmap_count.setMinimumSize(QtCore.QSize(100, 20))
		self.hitmap_count.setStyleSheet("QLineEdit{\n"
		                                "                                            background: rgb(223,223,233)\n"
		                                "                                            }\n"
		                                "                                        ")
		self.hitmap_count.setObjectName("hitmap_count")
		self.gridLayout_3.addWidget(self.hitmap_count, 0, 1, 1, 1)
		spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
		                                    QtWidgets.QSizePolicy.Policy.Minimum)
		self.gridLayout_3.addItem(spacerItem2, 0, 2, 1, 1)
		###
		# self.detector_heatmap = QtWidgets.QGraphicsView(parent=Visualization)
		self.detector_heatmap = pg.PlotWidget(parent=Visualization)
		self.detector_heatmap.setBackground('w')
		###
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
		                                   QtWidgets.QSizePolicy.Policy.Expanding)
		sizePolicy.setHorizontalStretch(1)
		sizePolicy.setVerticalStretch(1)
		sizePolicy.setHeightForWidth(self.detector_heatmap.sizePolicy().hasHeightForWidth())
		self.detector_heatmap.setSizePolicy(sizePolicy)
		self.detector_heatmap.setMinimumSize(QtCore.QSize(250, 250))
		self.detector_heatmap.setStyleSheet("QWidget{\n"
		                                    "                                            border: 0.5px solid gray;\n"
		                                    "                                            }\n"
		                                    "                                        ")
		self.detector_heatmap.setObjectName("detector_heatmap")
		self.gridLayout_3.addWidget(self.detector_heatmap, 1, 0, 1, 3)
		self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
		self.horizontalLayout_2.setObjectName("horizontalLayout_2")
		self.reset_heatmap_v = QtWidgets.QPushButton(parent=Visualization)
		self.reset_heatmap_v.setMinimumSize(QtCore.QSize(0, 20))
		self.reset_heatmap_v.setMaximumSize(QtCore.QSize(60, 16777215))
		self.reset_heatmap_v.setObjectName("reset_heatmap_v")
		self.horizontalLayout_2.addWidget(self.reset_heatmap_v)
		self.hitmap_plot_size = QtWidgets.QDoubleSpinBox(parent=Visualization)
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
		sizePolicy.setHorizontalStretch(0)
		sizePolicy.setVerticalStretch(0)
		sizePolicy.setHeightForWidth(self.hitmap_plot_size.sizePolicy().hasHeightForWidth())
		self.hitmap_plot_size.setSizePolicy(sizePolicy)
		self.hitmap_plot_size.setMinimumSize(QtCore.QSize(0, 20))
		self.hitmap_plot_size.setStyleSheet("QDoubleSpinBox{\n"
		                                    "                                                background: rgb(223,223,233)\n"
		                                    "                                                }\n"
		                                    "                                            ")
		self.hitmap_plot_size.setObjectName("hitmap_plot_size")
		self.horizontalLayout_2.addWidget(self.hitmap_plot_size)
		self.hit_displayed = QtWidgets.QLineEdit(parent=Visualization)
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
		sizePolicy.setHorizontalStretch(0)
		sizePolicy.setVerticalStretch(0)
		sizePolicy.setHeightForWidth(self.hit_displayed.sizePolicy().hasHeightForWidth())
		self.hit_displayed.setSizePolicy(sizePolicy)
		self.hit_displayed.setMinimumSize(QtCore.QSize(50, 20))
		self.hit_displayed.setStyleSheet("QLineEdit{\n"
		                                 "                                            background: rgb(223,223,233)\n"
		                                 "                                            }\n"
		                                 "                                        ")
		self.hit_displayed.setObjectName("hit_displayed")
		self.horizontalLayout_2.addWidget(self.hit_displayed)
		self.heatmap_fdm_switch = QtWidgets.QPushButton(parent=Visualization)
		self.heatmap_fdm_switch.setMinimumSize(QtCore.QSize(100, 20))
		self.heatmap_fdm_switch.setMaximumSize(QtCore.QSize(60, 16777215))
		self.heatmap_fdm_switch.setObjectName("heatmap_fdm_switch")
		self.horizontalLayout_2.addWidget(self.heatmap_fdm_switch)
		self.gridLayout_3.addLayout(self.horizontalLayout_2, 2, 0, 1, 3)
		self.gridLayout_5.addLayout(self.gridLayout_3, 0, 2, 1, 1)
		self.gridLayout_2 = QtWidgets.QGridLayout()
		self.gridLayout_2.setObjectName("gridLayout_2")
		self.label_207 = QtWidgets.QLabel(parent=Visualization)
		self.label_207.setMinimumSize(QtCore.QSize(0, 25))
		font = QtGui.QFont()
		font.setBold(True)
		self.label_207.setFont(font)
		self.label_207.setObjectName("label_207")
		self.gridLayout_2.addWidget(self.label_207, 0, 0, 1, 1)
		####
		# self.histogram = QtWidgets.QGraphicsView(parent=Visualization)
		self.histogram = pg.PlotWidget(parent=Visualization)
		self.histogram.setBackground('w')
		####
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
		                                   QtWidgets.QSizePolicy.Policy.Expanding)
		sizePolicy.setHorizontalStretch(1)
		sizePolicy.setVerticalStretch(1)
		sizePolicy.setHeightForWidth(self.histogram.sizePolicy().hasHeightForWidth())
		self.histogram.setSizePolicy(sizePolicy)
		self.histogram.setMinimumSize(QtCore.QSize(750, 150))
		self.histogram.setStyleSheet("QWidget{\n"
		                             "                                            border: 0.5px solid gray;\n"
		                             "                                            }\n"
		                             "                                        ")
		self.histogram.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
		self.histogram.setObjectName("histogram")
		self.gridLayout_2.addWidget(self.histogram, 1, 0, 1, 1)
		self.horizontalLayout = QtWidgets.QHBoxLayout()
		self.horizontalLayout.setObjectName("horizontalLayout")
		self.spectrum_switch = QtWidgets.QPushButton(parent=Visualization)
		self.spectrum_switch.setMinimumSize(QtCore.QSize(0, 20))
		self.spectrum_switch.setMaximumSize(QtCore.QSize(60, 16777215))
		self.spectrum_switch.setObjectName("spectrum_switch")
		self.horizontalLayout.addWidget(self.spectrum_switch)
		self.spectrum_last_events_switch = QtWidgets.QPushButton(parent=Visualization)
		self.spectrum_last_events_switch.setMinimumSize(QtCore.QSize(0, 20))
		self.spectrum_last_events_switch.setMaximumSize(QtCore.QSize(100, 16777215))
		self.spectrum_last_events_switch.setObjectName("spectrum_last_events_switch")
		self.horizontalLayout.addWidget(self.spectrum_last_events_switch)
		self.num_last_events = QtWidgets.QLineEdit(parent=Visualization)
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
		sizePolicy.setHorizontalStretch(0)
		sizePolicy.setVerticalStretch(0)
		sizePolicy.setHeightForWidth(self.num_last_events.sizePolicy().hasHeightForWidth())
		self.num_last_events.setSizePolicy(sizePolicy)
		self.num_last_events.setMinimumSize(QtCore.QSize(100, 20))
		self.num_last_events.setStyleSheet("QLineEdit{\n"
		                                   "                                            background: rgb(223,223,233)\n"
		                                   "                                            }\n"
		                                   "                                        ")
		self.num_last_events.setObjectName("num_last_events")
		self.horizontalLayout.addWidget(self.num_last_events)
		self.label_208 = QtWidgets.QLabel(parent=Visualization)
		self.label_208.setMinimumSize(QtCore.QSize(0, 25))
		font = QtGui.QFont()
		font.setBold(True)
		self.label_208.setFont(font)
		self.label_208.setObjectName("label_208")
		self.horizontalLayout.addWidget(self.label_208)
		self.max_mc = QtWidgets.QLineEdit(parent=Visualization)
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
		sizePolicy.setHorizontalStretch(0)
		sizePolicy.setVerticalStretch(0)
		sizePolicy.setHeightForWidth(self.max_mc.sizePolicy().hasHeightForWidth())
		self.max_mc.setSizePolicy(sizePolicy)
		self.max_mc.setMinimumSize(QtCore.QSize(100, 20))
		self.max_mc.setStyleSheet("QLineEdit{\n"
		                          "                                            background: rgb(223,223,233)\n"
		                          "                                            }\n"
		                          "                                        ")
		self.max_mc.setObjectName("max_mc")
		self.horizontalLayout.addWidget(self.max_mc)
		self.label_209 = QtWidgets.QLabel(parent=Visualization)
		self.label_209.setMinimumSize(QtCore.QSize(0, 25))
		font = QtGui.QFont()
		font.setBold(True)
		self.label_209.setFont(font)
		self.label_209.setObjectName("label_209")
		self.horizontalLayout.addWidget(self.label_209)
		self.max_tof = QtWidgets.QLineEdit(parent=Visualization)
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
		sizePolicy.setHorizontalStretch(0)
		sizePolicy.setVerticalStretch(0)
		sizePolicy.setHeightForWidth(self.max_tof.sizePolicy().hasHeightForWidth())
		self.max_tof.setSizePolicy(sizePolicy)
		self.max_tof.setMinimumSize(QtCore.QSize(100, 20))
		self.max_tof.setStyleSheet("QLineEdit{\n"
		                           "                                            background: rgb(223,223,233)\n"
		                           "                                            }\n"
		                           "                                        ")
		self.max_tof.setObjectName("max_tof")
		self.horizontalLayout.addWidget(self.max_tof)
		spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
		                                    QtWidgets.QSizePolicy.Policy.Minimum)
		self.horizontalLayout.addItem(spacerItem3)
		self.gridLayout_2.addLayout(self.horizontalLayout, 2, 0, 1, 1)
		self.Error = QtWidgets.QLabel(parent=Visualization)
		self.Error.setMinimumSize(QtCore.QSize(800, 30))
		font = QtGui.QFont()
		font.setPointSize(10)
		font.setBold(True)
		font.setStrikeOut(False)
		self.Error.setFont(font)
		self.Error.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
		self.Error.setWordWrap(True)
		self.Error.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.LinksAccessibleByMouse)
		self.Error.setObjectName("Error")
		self.gridLayout_2.addWidget(self.Error, 3, 0, 1, 1)
		self.gridLayout_5.addLayout(self.gridLayout_2, 1, 0, 1, 3)
		self.gridLayout_6.addLayout(self.gridLayout_5, 0, 0, 1, 1)

		self.retranslateUi(Visualization)
		QtCore.QMetaObject.connectSlotsByName(Visualization)
		Visualization.setTabOrder(self.voltage, self.detection_rate)
		Visualization.setTabOrder(self.detection_rate, self.hitmap_count)
		Visualization.setTabOrder(self.hitmap_count, self.dc_hold)
		Visualization.setTabOrder(self.dc_hold, self.detection_rate_range_switch)
		Visualization.setTabOrder(self.detection_rate_range_switch, self.reset_heatmap_v)
		Visualization.setTabOrder(self.reset_heatmap_v, self.hitmap_plot_size)
		Visualization.setTabOrder(self.hitmap_plot_size, self.hit_displayed)
		Visualization.setTabOrder(self.hit_displayed, self.heatmap_fdm_switch)
		Visualization.setTabOrder(self.heatmap_fdm_switch, self.spectrum_switch)
		Visualization.setTabOrder(self.spectrum_switch, self.spectrum_last_events_switch)
		Visualization.setTabOrder(self.spectrum_last_events_switch, self.num_last_events)
		Visualization.setTabOrder(self.num_last_events, self.max_mc)
		Visualization.setTabOrder(self.max_mc, self.max_tof)
		Visualization.setTabOrder(self.max_tof, self.vdc_time)
		Visualization.setTabOrder(self.vdc_time, self.detection_rate_viz)
		Visualization.setTabOrder(self.detection_rate_viz, self.detector_heatmap)
		Visualization.setTabOrder(self.detector_heatmap, self.histogram)

		###
		# Start the update timer with a 500 ms interval (2 times per second)
		self.update_timer.start(500)

		# High Voltage visualization ################
		self.x_vdc = [i * 0.5 for i in range(200)]  # 100 time points
		self.y_vdc = [0.0] * 200  # 200 data points, all initialized to 0.0
		self.y_vdc[:] = [np.nan] * len(self.y_vdc)
		pen_vdc = pg.mkPen(color=(255, 0, 0), width=3)
		self.data_line_vdc = self.vdc_time.plot(self.x_vdc, self.y_vdc, pen=pen_vdc)
		self.vdc_time.plotItem.setMouseEnabled(x=False)  # Only allow zoom in Y-axis
		# Add Axis Labels
		self.styles = {"color": "#f00", "font-size": "12px"}
		self.vdc_time.setLabel("left", "High Voltage", units='V', **self.styles)
		self.vdc_time.setLabel("bottom", "Time (s)", **self.styles)
		# Add grid
		self.vdc_time.showGrid(x=True, y=True)
		# Add Range
		self.vdc_time.setXRange(0, 100)
		self.vdc_time.setYRange(0, 15000)

		# Detection Visualization #########################
		self.x_dtec = [i * 0.5 for i in range(200)]  # 100 time points
		self.y_dtec = [0.0] * 200  # 200 data points, all initialized to 0.0
		self.y_dtec[:] = [np.nan] * len(self.y_vdc)
		pen_dtec = pg.mkPen(color=(255, 0, 0), width=3)
		self.data_line_dtec = self.detection_rate_viz.plot(self.x_dtec, self.y_dtec, pen=pen_dtec)

		# Add Axis Labels
		self.detection_rate_viz.setLabel("left", "Detection rate (%)", **self.styles)
		self.detection_rate_viz.setLabel("bottom", "Time (s)", **self.styles)

		# Add grid
		self.detection_rate_viz.showGrid(x=True, y=True)
		self.detection_rate_viz.plotItem.setMouseEnabled(x=False)  # Only allow zoom in Y-axis
		# Add Range
		self.detection_rate_viz.setXRange(0, 100)
		self.detection_rate_viz.setYRange(0, 100)

		# detector heatmep #####################
		self.scatter = pg.ScatterPlotItem(
			size=self.hitmap_plot_size.value(), brush='black')
		self.detector_circle = QtWidgets.QGraphicsEllipseItem(-40, -40, 80, 80)  # x, y, width, height
		self.detector_circle.setPen(pg.mkPen(color=(255, 0, 0), width=2))
		self.detector_heatmap.addItem(self.detector_circle)
		self.detector_heatmap.setLabel("left", "X_det", units='mm', **self.styles)
		self.detector_heatmap.setLabel("bottom", "Y_det", units='mm', **self.styles)

		# Histogram #########################
		# Add Axis Labels
		self.histogram.plotItem.setMouseEnabled(y=False)  # Only allow zoom in X-axis
		self.histogram.setLabel("left", "Event Counts", **self.styles)
		self.histogram.setLogMode(y=True)
		if self.conf["visualization"] == "tof":
			self.histogram.setLabel("bottom", "Time", units='ns', **self.styles)
		elif self.conf["visualization"] == "mc":
			self.histogram.setLabel("bottom", "m/c", units='Da', **self.styles)

		self.visualization_window = Visualization  # Assign the attribute when setting up the UI

		self.reset_heatmap_v.clicked.connect(self.reset_heatmap)
		self.histogram.addLegend(offset=(-10, 10))

		self.original_button_style = self.detection_rate_range_switch.styleSheet()
		self.detection_rate_range_switch.clicked.connect(self.detection_rate_range)
		self.spectrum_switch.clicked.connect(self.spectrum_switch_mc_tof)
		self.spectrum_last_events_switch.clicked.connect(self.spectrum_last_events)
		self.num_last_events.editingFinished.connect(self.parameters_changes)
		self.max_mc.editingFinished.connect(self.parameters_changes)
		self.max_tof.editingFinished.connect(self.parameters_changes)

		self.num_event_mc_tof = int(self.num_last_events.text())

		self.heatmap_fdm_switch.clicked.connect(self.heatmap_fdm_switch_change)

		self.num_event_mc_tof = int(self.num_last_events.text())
		self.max_mc_val = int(self.max_mc.text())
		self.max_tof_val = int(self.max_tof.text())
		self.index_hist_tof = np.where(self.bins_tof == self.max_tof_val)[0][0]
		self.index_hist_mc = np.where(self.bins_mc == self.max_mc_val)[0][0]

		self.dc_hold.clicked.connect(self.dc_hold_clicked)

		self.hitmap_count.setReadOnly(True)
		self.voltage.setReadOnly(True)
		self.detection_rate.setReadOnly(True)
		self.hit_displayed.editingFinished.connect(self.parameters_changes)

		# Create a QTimer to hide the warning message after 8 seconds
		self.timer = QtCore.QTimer()
		self.timer.timeout.connect(self.hideMessage)

		self.hitmap_plot_size.setValue(1.0)
		self.hitmap_plot_size.setSingleStep(0.1)
		self.hitmap_plot_size.setDecimals(1)

	def retranslateUi(self, Visualization):
		"""
        Set the text of the widgets
        Args:
        Visualization: The main window

        Return:
        None
        """
		_translate = QtCore.QCoreApplication.translate
		###
		# Visualization.setWindowTitle(_translate("Visualization", "Form"))
		Visualization.setWindowTitle(_translate("Visualization", "PyCCAPT Visualization"))
		Visualization.setWindowIcon(QtGui.QIcon('./files/logo.png'))
		###
		self.label_200.setText(_translate("Visualization", "Voltage"))
		self.voltage.setText(_translate("Visualization", "0"))
		self.dc_hold.setText(_translate("Visualization", "Hold DC Voltage"))
		self.label_201.setText(_translate("Visualization", "Detection Rate"))
		self.detection_rate.setText(_translate("Visualization", "0"))
		self.detection_rate_range_switch.setText(_translate("Visualization", "Short Range"))
		self.label_206.setText(_translate("Visualization", "Detector"))
		self.hitmap_count.setText(_translate("Visualization", "0"))
		self.reset_heatmap_v.setText(_translate("Visualization", "Reset"))
		self.hit_displayed.setText(_translate("Visualization", "2000"))
		self.heatmap_fdm_switch.setText(_translate("Visualization", "Hitmap/FDM"))
		self.label_207.setText(_translate("Visualization", "Spectrum"))
		self.spectrum_switch.setText(_translate("Visualization", "mc/tof"))
		self.spectrum_last_events_switch.setText(_translate("Visualization", "Last Events"))
		self.num_last_events.setText(_translate("Visualization", "10000"))
		self.label_208.setText(_translate("Visualization", "Max mc (Da)"))
		self.max_mc.setText(_translate("Visualization", "400"))
		self.label_209.setText(_translate("Visualization", "Max tof (ns)"))
		self.max_tof.setText(_translate("Visualization", "5000"))
		self.Error.setText(_translate("Visualization", "<html><head/><body><p><br/></p></body></html>"))

	def dc_hold_clicked(self):
		"""
            Hold the DC voltage

            Args:
                None

            Return:
                None
        """
		if self.variables.start_flag or self.variables.last_screen_shot:
			if not self.variables.vdc_hold:
				self.variables.vdc_hold = True
				self.dc_hold.setStyleSheet("QPushButton{\n"
				                           "background: rgb(0, 255, 26)\n"
				                           "}")
			elif self.variables.vdc_hold:
				self.variables.vdc_hold = False
				self.dc_hold.setStyleSheet(self.original_button_style)

	def heatmap_fdm_switch_change(self):
		"""
        Change the heatmap type
        Args:
            None

        Return:
            None
        """
		if self.heatmap_fdm_switch_flag == 'heatmap':
			self.heatmap_fdm_switch_flag = 'fdm'

		elif self.heatmap_fdm_switch_flag == 'fdm':
			self.heatmap_fdm_switch_flag = 'heatmap'

	def reset_heatmap(self):
		"""
        Reset the heatmap
        Args:
            None

        Return:
            None
        """
		# with self.variables.lock_setup_parameters:
		if not self.variables.reset_heatmap:
			self.variables.reset_heatmap = True

	def detection_rate_range(self):
		"""
        Change the time range of the detection rate

        Args:
            None

        Return:
            None
        """
		self.change_detection_rate_range = not self.change_detection_rate_range

		if self.change_detection_rate_range:
			self.detection_rate_range_switch.setStyleSheet("QPushButton{\n"
			                                               "background: rgb(0, 255, 26)\n"
			                                               "}")
		else:
			self.detection_rate_range_switch.setStyleSheet(self.original_button_style)

	def update_graphs_helper(self, ):
		"""
		Update the graphs

		Args:
			None

		Return:
			None
		"""
		if self.index_plot_start == 0:
			self.num_hit_display = int(float(self.hit_displayed.text()))
			self.start_main_exp = time.time()
			self.start_time = time.time()
			self.start_time_metadata = time.time()
			self.index_plot_start += 1
			self.hitmap_count.setText(str(0))
		self.variables.elapsed_time = time.time() - self.start_time
		# with self.variables.lock_statistics:
		if self.index_wait_on_plot_start <= 16:
			if self.index_wait_on_plot_start == 0:
				self.counter_source = self.variables.counter_source
			self.index_wait_on_plot_start += 1

		# V_dc and V_p
		current_voltage = self.variables.specimen_voltage_plot
		if self.index_plot < len(self.y_vdc):
			self.y_vdc[self.index_plot] = int(current_voltage)  # Add a new value.

		else:
			x_vdc_last = self.x_vdc[-1]
			self.x_vdc.append(x_vdc_last + 0.5)  # Add a new value 1 higher than the last.
			self.y_vdc.append(int(current_voltage))
		# set the value of the voltage with two decimal places
		self.voltage.setText(str("{:.2f}".format(current_voltage)))

		# Set the maximum number of data points to display
		max_display_points = 200
		# Downsample the data if needed
		if len(self.x_vdc) > max_display_points:
			step = len(self.x_vdc) // max_display_points
			x_vdc_downsampled = self.x_vdc[::step]
			y_vdc_downsampled = self.y_vdc[::step]
			self.data_line_vdc.setData(x_vdc_downsampled, y_vdc_downsampled)
		else:
			self.data_line_vdc.setData(self.x_vdc, self.y_vdc)

		# Detection Rate Visualization
		# with self.variables.lock_statistics:
		current_detection_rate = self.variables.detection_rate_current_plot
		if self.index_plot < len(self.y_dtec):
			self.y_dtec[self.index_plot] = current_detection_rate  # Add a new value.
		else:
			# self.x_dtec = self.x_dtec[1:]  # Remove the first element.
			x_dtec_last = self.x_dtec[-1]
			self.x_dtec.append(x_dtec_last + 0.5)  # Add a new value 1 higher than the last.
			self.y_dtec.append(current_detection_rate)
		self.detection_rate.setText(str("{:.2f}".format(current_detection_rate)))
		# self.data_line_dtec.setData(self.x_dtec, self.y_dtec)
		# Set the maximum number of data points to display
		max_display_points = 200
		# Downsample the data if needed
		if len(self.x_dtec) > max_display_points and not self.change_detection_rate_range:
			step = len(self.x_dtec) // max_display_points
			x_dtec_downsampled = self.x_dtec[::step]
			y_dtec_downsampled = self.y_dtec[::step]
			self.data_line_dtec.setData(x_dtec_downsampled, y_dtec_downsampled)
		elif len(self.x_dtec) > max_display_points and self.change_detection_rate_range:
			x_dtec_downsampled = self.x_dtec[-max_display_points:]
			y_dtec_downsampled = self.y_dtec[-max_display_points:]
			self.data_line_dtec.setData(x_dtec_downsampled, y_dtec_downsampled)
		else:
			self.data_line_dtec.setData(self.x_dtec, self.y_dtec)
		# Increase the index
		# with self.variables.lock_statistics:
		self.index_plot += 1
		# mass spectrum

		if self.counter_source == 'TDC' and self.variables.total_ions > 0 and \
				self.index_wait_on_plot_start > 16:

			xx = np.array([])
			yy = np.array([])
			tt = np.array([])
			main_v_dc_dld = np.array([])

			while not self.x_plot.empty() and not self.y_plot.empty() and not self.t_plot.empty() and \
					not self.main_v_dc_plot.empty():
				data = self.x_plot.get()
				xx = np.append(xx, data)
				data = self.y_plot.get()
				yy = np.append(yy, data)
				data = self.t_plot.get()
				tt = np.append(tt, data)
				data = self.main_v_dc_plot.get()
				main_v_dc_dld = np.append(main_v_dc_dld, data)

			# self.length_events += len(self.tt)
			self.length_events += len(tt)

			if len(self.last_100_thousand_v) == 0:
				self.last_100_thousand_det_x_heatmap = xx
				self.last_100_thousand_det_y_heatmap = yy
				mask_t = tt < self.conf["max_tof"]
				self.last_100_thousand_v = main_v_dc_dld[mask_t]
				self.last_100_thousand_det_x = xx[mask_t]
				self.last_100_thousand_det_y = yy[mask_t]
				self.last_100_thousand_t = tt[mask_t]
			else:
				self.last_100_thousand_det_x_heatmap = np.concatenate((self.last_100_thousand_det_x_heatmap, xx))
				self.last_100_thousand_det_y_heatmap = np.concatenate((self.last_100_thousand_det_y_heatmap, yy))
				mask_t = tt < self.conf["max_tof"]
				self.last_100_thousand_v = np.concatenate((self.last_100_thousand_v, main_v_dc_dld[mask_t]))
				self.last_100_thousand_det_x = np.concatenate((self.last_100_thousand_det_x, xx[mask_t]))
				self.last_100_thousand_det_y = np.concatenate((self.last_100_thousand_det_y, yy[mask_t]))
				self.last_100_thousand_t = np.concatenate((self.last_100_thousand_t, tt[mask_t]))
			if len(self.last_100_thousand_v) > 100000:
				self.last_100_thousand_v = self.last_100_thousand_v[-100000:]
				self.last_100_thousand_det_x = self.last_100_thousand_det_x[-100000:]
				self.last_100_thousand_det_x_heatmap = self.last_100_thousand_det_x_heatmap[-100000:]
				self.last_100_thousand_det_y = self.last_100_thousand_det_y[-100000:]
				self.last_100_thousand_det_y_heatmap = self.last_100_thousand_det_y_heatmap[-100000:]
				self.last_100_thousand_t = self.last_100_thousand_t[-100000:]

			try:
				if self.variables.pulse_mode == 'Voltage':
					t_0 = self.conf["t_0_voltage"]
				elif self.variables.pulse_mode == 'Laser' or self.variables.pulse_mode == 'VoltageLaser':
					t_0 = self.conf["t_0_laser"]
				if self.mc_tof_last_events_flag and self.conf["visualization"] == "tof":
					tt_last_events = self.last_100_thousand_t[-self.num_event_mc_tof:]
					hist_tof_last_events, _ = np.histogram(tt_last_events, bins=self.bins_tof)

				elif self.mc_tof_last_events_flag and self.conf["visualization"] == "mc":
					t_last_events = self.last_100_thousand_t[-self.num_event_mc_tof:]
					main_v_dc_dld_last_events = self.last_100_thousand_v[-self.num_event_mc_tof:]
					x_last_events = self.last_100_thousand_det_x[-self.num_event_mc_tof:]
					y_last_events = self.last_100_thousand_det_y[-self.num_event_mc_tof:]

					mc_last_events = tof2mc_simple.tof_2_mc(t_last_events, t_0,
					                                        main_v_dc_dld_last_events,
					                                        x_last_events,
					                                        y_last_events,
					                                        flightPathLength=self.conf["flight_path_length"])
					hist_mc_last_events, _ = np.histogram(mc_last_events, bins=self.bins_mc)

				# hist_tof, _ = np.histogram(tt_max_lenght, bins=self.bins_tof)
				# self.hist_tof = hist_tof
				hist_tof, _ = np.histogram(tt[mask_t], bins=self.bins_tof)
				self.hist_tof += hist_tof

				# mc = tof2mc_simple.tof_2_mc(self.last_100_thousand_t, self.conf["t_0"],
				#                             self.last_100_thousand_v,
				#                             self.last_100_thousand_det_x,
				#                             self.last_100_thousand_det_y,
				#                             flightPathLength=self.conf["flight_path_length"])
				# hist_mc, _ = np.histogram(mc, bins=self.bins_mc)
				# self.hist_mc = hist_mc
				mc = tof2mc_simple.tof_2_mc(tt[mask_t], t_0,
				                            main_v_dc_dld[mask_t],
				                            xx[mask_t],
				                            yy[mask_t],
				                            flightPathLength=self.conf["flight_path_length"])
				hist_mc, _ = np.histogram(mc, bins=self.bins_mc)
				self.hist_mc += hist_mc

				self.histogram.clear()
				if self.conf["visualization"] == "tof" and not self.mc_tof_last_events_flag:
					hist = np.copy(self.hist_tof[:self.index_hist_tof])
					hist[hist == 0] = 1  # Avoid log(0) error
					bins = self.bins_tof[:self.index_hist_tof + 1]
					self.histogram.plot(bins, hist, stepMode="center", fillLevel=0,
					                    fillOutline=True, brush='black', name="num events: %s" % self.length_events)
				elif self.conf["visualization"] == "mc" and not self.mc_tof_last_events_flag:
					hist = np.copy(self.hist_mc[:self.index_hist_mc])
					hist[hist == 0] = 1  # Avoid log(0) error
					bins = self.bins_mc[:self.index_hist_mc + 1]
					self.histogram.plot(bins, hist, stepMode="center", fillLevel=0,
					                    fillOutline=True, brush='black', name="num events: %s" % self.length_events)
				elif self.conf["visualization"] == "tof" and self.mc_tof_last_events_flag:
					# remobe the bins bigger than the max_tof
					hist = np.copy(hist_tof_last_events[:self.index_hist_tof])
					hist[hist == 0] = 1  # Avoid log(0) error
					bins = self.bins_tof[:self.index_hist_tof + 1]
					self.histogram.plot(bins, hist, stepMode="center", fillLevel=0,
					                    fillOutline=True, brush='black', name="num events: %s" % self.length_events)
				elif self.conf["visualization"] == "mc" and self.mc_tof_last_events_flag:
					# remobe the bins bigger than the max_mc
					hist = np.copy(hist_mc_last_events[:self.index_hist_mc])
					hist[hist == 0] = 1  # Avoid log(0) error
					bins = self.bins_mc[:self.index_hist_mc + 1]
					self.histogram.plot(bins, hist, stepMode="center", fillLevel=0,
					                    fillOutline=True, brush='black', name="num events: %s" % self.length_events)

			except Exception as e:
				print(
					f"{initialize_devices.bcolors.FAIL}Error: Cannot plot Histogram correctly{initialize_devices.bcolors.ENDC}")
				print(e)
			# Visualization
			# try:
			# calculate the fdm for the current data
			hist, xedges, yedges = np.histogram2d(xx * 10, yy * 10, bins=self.bins_detector, range=self.range)
			self.hist_fdm += np.log10(hist + 1)  # Avoid log(0) error
			# self.hist_fdm += hist
			if self.heatmap_fdm_switch_flag == 'heatmap':
				if self.variables.reset_heatmap:
					self.variables.reset_heatmap = False
					self.last_100_thousand_det_x_heatmap = np.array([])
					self.last_100_thousand_det_y_heatmap = np.array([])
				x_last_events = self.last_100_thousand_det_x_heatmap[:]
				y_last_events = self.last_100_thousand_det_y_heatmap[:]
				# adding points to the scatter plot
				self.scatter.setSize(self.hitmap_plot_size.value())

				x = x_last_events * 10
				y = y_last_events * 10

				x = x[-self.num_hit_display:]
				y = y[-self.num_hit_display:]
				self.hitmap_count.setText(str(len(x)))  # number of points displayed
				self.scatter.clear()
				self.scatter.setData(x=x, y=y)
				# add item to plot window
				# adding scatter plot item to the plot window
				self.detector_heatmap.clear()
				self.detector_heatmap.addItem(self.scatter)
				self.detector_heatmap.addItem(self.detector_circle)

			elif self.heatmap_fdm_switch_flag == 'fdm':
				# plot fdm which is 2d hsogram of det_x and det_y
				# Create a 2D histogram
				if self.mc_tof_last_events_flag:
					x_last_events = self.last_100_thousand_det_x_heatmap[-self.num_event_mc_tof:]
					y_last_events = self.last_100_thousand_det_y_heatmap[-self.num_event_mc_tof:]
					hist_fdm_last_events, xedges, yedges = np.histogram2d(x_last_events * 10, y_last_events * 10,
					                                                      bins=self.bins_detector, range=self.range)
					hist_fdm_last_events = np.log10(hist_fdm_last_events + 1)
				if self.mc_tof_last_events_flag:
					hist_fdm_tmp = np.copy(hist_fdm_last_events)
				else:
					hist_fdm_tmp = np.copy(self.hist_fdm)

				img = pg.ImageItem()
				img.setImage(hist_fdm_tmp)  # Transpose if needed because pg.ImageItem assumes (row, col) format
				# set the length of histogram
				self.hitmap_count.setText(str(self.length_events))  # number of points displayed
				img.setRect(QtCore.QRectF(xedges[0], yedges[0], xedges[-1] - xedges[0], yedges[-1] - yedges[0]))

				# Apply a color map to the histogram
				# Load a preset color map (e.g., 'grey', 'thermal', 'flame', viridis, etc.)
				lut = pg.colormap.get('viridis').getLookupTable(start=0.0, stop=1.0, nPts=256)
				img.setLookupTable(lut)
				# add item to plot window
				# adding scatter plot item to the plot window
				self.detector_heatmap.clear()
				self.detector_heatmap.addItem(img)
				# Adjust the aspect ratio to match the data aspect ratio
				self.detector_heatmap.getViewBox().setAspectLocked(True)

	def update_graphs(self, ):
		"""
        Update the graphs
        Args:
            None

        Return:
            None
        """

		if self.variables.plot_clear_flag:
			self.x_vdc = [i * 0.5 for i in range(200)]  # 100 time points
			self.y_vdc = [0.0] * 200  # 200 data points, all initialized to 0.0
			self.y_vdc[:] = [np.nan] * len(self.y_vdc)

			self.vdc_time.clear()
			pen_vdc = pg.mkPen(color=(255, 0, 0), width=3)
			self.data_line_vdc = self.vdc_time.plot(self.x_vdc, self.y_vdc, pen=pen_vdc)

			self.x_dtec = [i * 0.5 for i in range(200)]  # 100 time points
			self.y_dtec = [0.0] * 200  # 200 data points, all initialized to 0.0
			self.y_dtec[:] = [np.nan] * len(self.y_vdc)

			self.detection_rate_viz.clear()
			pen_dtec = pg.mkPen(color=(255, 0, 0), width=3)
			self.data_line_dtec = self.detection_rate_viz.plot(self.x_dtec, self.y_dtec, pen=pen_dtec)

			self.histogram.clear()

			self.detector_heatmap.clear()
			self.detector_heatmap.addItem(self.detector_circle)
			self.variables.plot_clear_flag = False
			self.index_plot = 0
			self.index_plot_start = 0
			self.index_plot_save = 0
			self.start_time_metadata = 0
			self.variables.detection_rate_current_plot = 0

			self.last_100_thousand_det_x_heatmap = np.array([])
			self.last_100_thousand_det_x = np.array([])
			self.last_100_thousand_det_y_heatmap = np.array([])
			self.last_100_thousand_det_y = np.array([])
			self.last_100_thousand_t = np.array([])
			self.last_100_thousand_v = np.array([])
			self.length_events = 0
			self.hist_fdm, xedges, yedges = np.histogram2d([], [], bins=self.bins_detector, range=self.range)
			self.hist_mc = np.zeros(len(self.bins_mc) - 1)
			self.hist_tof = np.zeros(len(self.bins_tof) - 1)

		if self.index_auto_scale_graph == 30:
			self.vdc_time.enableAutoRange(axis='x')
			self.histogram.enableAutoRange(axis='y')
			self.detection_rate_viz.enableAutoRange(axis='x')
			self.detection_rate_viz.enableAutoRange(axis='y')
			self.detector_heatmap.enableAutoRange(axis='x')
			self.detector_heatmap.enableAutoRange(axis='y')
			self.index_auto_scale_graph = 0

		# with self.variables.lock_statistics and self.variables.lock_setup_parameters:
		if self.variables.start_flag and self.variables.flag_visualization_start:
			self.index_auto_scale_graph += 1
			self.update_graphs_helper()

			# save plots to the file
			if time.time() - self.start_time_metadata >= self.variables.save_meta_interval_visualization:
				self.path_meta = self.variables.path_meta
				exporter = pg.exporters.ImageExporter(self.vdc_time.plotItem)
				exporter.params['width'] = 1000  # Set the width of the image
				exporter.params['height'] = 800  # Set the height of the image
				exporter.export(self.variables.path_meta + '/visualization_v_dc_p_%s.png' % self.index_plot_save)
				exporter = pg.exporters.ImageExporter(self.detection_rate_viz.plotItem)
				exporter.params['width'] = 1000  # Set the width of the image
				exporter.params['height'] = 800  # Set the height of the image
				exporter.export(self.path_meta + '/visualization_detection_rate_%s.png' % self.index_plot_save)
				exporter = pg.exporters.ImageExporter(self.detector_heatmap.plotItem)
				exporter.params['width'] = 1000  # Set the width of the image
				exporter.params['height'] = 800  # Set the height of the image
				exporter.export(self.path_meta + '/visualization_detector_%s.png' % self.index_plot_save)
				exporter = pg.exporters.ImageExporter(self.histogram.plotItem)
				exporter.params['width'] = 1000  # Set the width of the image
				exporter.params['height'] = 800  # Set the height of the image
				exporter.export(self.path_meta + '/visualization_mc_tof_%s.png' % self.index_plot_save)

				screenshot = QtWidgets.QApplication.primaryScreen().grabWindow(self.visualization_window.winId())
				screenshot.save(self.path_meta + '/visualization_screenshot_%s.png' % self.index_plot_save, 'png')
				self.start_time_metadata = time.time()
				# Increase the index
				self.index_plot_save += 1

		elif self.variables.last_screen_shot:
			self.path_meta = self.variables.path_meta
			if self.variables.vdc_hold:
				self.dc_hold.click()
			if self.heatmap_fdm_switch_flag == 'heatmap':
				self.heatmap_fdm_switch.click()
			if self.mc_tof_last_events_flag:
				self.spectrum_last_events_switch.click()
			if self.change_detection_rate_range:
				self.detection_rate_range_switch.click()
			if self.conf["visualization"] == "tof":
				self.spectrum_switch.click()

			self.update_graphs_helper()

			exporter = pg.exporters.ImageExporter(self.vdc_time.plotItem)
			exporter.params['width'] = 1000  # Set the width of the image
			exporter.params['height'] = 800  # Set the height of the image
			exporter.export(self.path_meta + '/visualization_v_dc_p_final.png')
			exporter = pg.exporters.ImageExporter(self.detection_rate_viz.plotItem)
			exporter.params['width'] = 1000  # Set the width of the image
			exporter.params['height'] = 800  # Set the height of the image
			exporter.export(self.path_meta + '/visualization_detection_rate_final.png')
			exporter = pg.exporters.ImageExporter(self.detector_heatmap.plotItem)
			exporter.params['width'] = 1000  # Set the width of the image
			exporter.params['height'] = 800  # Set the height of the image
			exporter.export(self.path_meta + '/visualization_detector_final.png')
			exporter = pg.exporters.ImageExporter(self.histogram.plotItem)
			exporter.params['width'] = 1000  # Set the width of the image
			exporter.params['height'] = 800  # Set the height of the image
			exporter.export(self.path_meta + '/visualization_mc_tof_final.png')

			screenshot = QtWidgets.QApplication.primaryScreen().grabWindow(self.visualization_window.winId())
			screenshot.save(self.path_meta + '/visualization_screenshot_final.png', 'png')

			self.variables.last_screen_shot = False

	def spectrum_switch_mc_tof(self):
		"""
        Switch between mass spectrum and time of flight spectrum
        Args:
            None

        Return:
            None
        """
		if self.conf["visualization"] == "tof":
			self.conf["visualization"] = "mc"
			self.histogram.setLabel("bottom", "m/c", units='Da', **self.styles)
		elif self.conf["visualization"] == "mc":
			self.conf["visualization"] = "tof"
			self.histogram.setLabel("bottom", "Time", units='ns', **self.styles)

	def spectrum_last_events(self):
		"""
        Display the last events in the mass spectrum
        Args:
            None

        Return:
            None
        """
		self.mc_tof_last_events_flag = not self.mc_tof_last_events_flag
		if self.mc_tof_last_events_flag:
			self.spectrum_last_events_switch.setStyleSheet("QPushButton{\n"
			                                               "background: rgb(0, 255, 26)\n"
			                                               "}")
		else:
			self.spectrum_last_events_switch.setStyleSheet(self.original_button_style)

	def parameters_changes(self):
		"""
        Change the parameters for the mass spectrum
        Args:
            None

        Return:
            None
        """
		if self.num_last_events.text().isdigit():
			num_last_event_tmp = int(self.num_last_events.text())
			if num_last_event_tmp > 100000:
				self.num_last_events_val = 100000
				self.num_last_events.setText("100000")
			else:
				self.num_event_mc_tof = num_last_event_tmp

		if self.max_mc.text().isdigit():
			max_mc_tmp = int(self.max_mc.text())
			if max_mc_tmp > self.conf["max_mass"]:
				self.max_mc_val = self.conf["max_mass"]
				self.max_mc.setText(str(self.conf["max_mass"]))
				self.index_hist_mc = np.where(self.bins_mc == self.max_mc_val)[0][0]
			else:
				self.max_mc_val = max_mc_tmp
				self.index_hist_mc = np.where(self.bins_mc == self.max_mc_val)[0][0]
		if self.max_tof.text().isdigit():
			max_tof_tmp = int(self.max_tof.text())
			if max_tof_tmp > self.conf["max_tof"]:
				self.max_tof_val = self.conf["max_tof"]
				self.max_tof.setText(str(self.conf["max_tof"]))

				self.index_hist_tof = np.where(self.bins_tof == self.max_tof_val)[0][0]
			else:
				self.max_tof_val = max_tof_tmp
				self.index_hist_tof = np.where(self.bins_tof == self.max_tof_val)[0][0]
		if self.hit_displayed.text().isdigit():
			if int(float(self.hit_displayed.text())) > 100000:
				self.error_message("Maximum possible number is 100000")
				_translate = QtCore.QCoreApplication.translate
				self.hit_displayed.setText(_translate("PyCCAPT", "100000"))
			else:
				self.num_hit_display = int(float(self.hit_displayed.text()))

	def error_message(self, message):
		"""
		Display an error message and start a timer to hide it after 8 seconds

		Args:
			message (str): Error message to display

		Return:
			None
		"""
		_translate = QtCore.QCoreApplication.translate
		self.Error.setText(_translate("OXCART",
		                              "<html><head/><body><p><span style=\" color:#ff0000;\">"
		                              + message + "</span></p></body></html>"))

		self.timer.start(8000)

	def hideMessage(self, ):
		"""
        Hide the message and stop the timer
        Args:
            None

        Return:
            None
        """
		# Hide the message and stop the timer
		_translate = QtCore.QCoreApplication.translate
		self.Error.setText(_translate("OXCART",
		                              "<html><head/><body><p><span style=\" "
		                              "color:#ff0000;\"></span></p></body></html>"))

		self.timer.stop()

	def stop(self):
		"""
        Stop any background activity
        Args:
            None

        Return:
            None
            """
		# Add any additional cleanup code here
		pass


def efficient_histogram(viz, bin_size):
	bins = np.arange(np.min(viz), np.max(viz) + bin_size, bin_size)
	hist, edges = np.histogram(viz, bins=bins)
	hist[hist == 0] = 1  # Avoid log(0)
	return hist, edges


class VisualizationWindow(QtWidgets.QWidget):
	"""
	Widget for the Visualization window.
	"""
	closed = QtCore.pyqtSignal()  # Define a custom closed signal

	def __init__(self, variables, gui_visualization, visualization_close_event,
	             visualization_win_front, *args, **kwargs):
		"""
        Constructor for the VisualizationWindow class.

        Args:
            variables: Shared variables.
            gui_visualization: Instance of the Visualization.
            visualization_close_event: Event for the Visualization window closed.
            visualization_win_front: Event for the Visualization window front.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Return:
            None
        """
		super().__init__(*args, **kwargs)
		self.gui_visualization = gui_visualization
		self.variables = variables
		self.visualization_win_front = visualization_win_front
		self.visualization_close_event = visualization_close_event
		self.show()
		self.showMinimized()
		self.timer = QtCore.QTimer(self)
		self.timer.timeout.connect(self.check_if_should)
		self.timer.start(500)  # Check every 1000 milliseconds (1 second)

	def closeEvent(self, event):
		"""
        Close event for the window.

        Args:
            event: Close event.

        Return:
            None
        """
		event.ignore()
		self.showMinimized()
		self.visualization_close_event.set()

	def check_if_should(self):
		"""
        Check if the window should be shown.

        Args:
            None

        Return:
            None
        """
		if self.visualization_win_front.is_set():
			self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint)
			self.show()
			self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowType.WindowStaysOnTopHint)
			self.visualization_win_front.clear()  # Reset the flag
		if self.variables.flag_visualization_win_show:
			self.show()
			self.variables.flag_visualization_win_show = False

	def setWindowStyleFusion(self):
		# Set the Fusion style
		QtWidgets.QApplication.setStyle("Fusion")


def run_visualization_window(variables, conf, visualization_closed_event, visualization_win_front,
                             x_plot, y_plot, t_plot, main_v_dc_plot):
	"""
    Run the Cameras window in a separate process.

    Args:
        variables: Shared variables.
        conf: Configuration dictionary.
        visualization_closed_event: Event for the Visualization window closed.
        visualization_win_front: Event for the Visualization window front.
        x_plot: x plot
        y_plot: y plot
        t_plot: t plot
        main_v_dc_plot: main v dc plot

    Return:
        None
    """
	app = QtWidgets.QApplication(sys.argv)  # <-- Create a new QApplication instance
	app.setStyle('Fusion')

	gui_visualization = Ui_Visualization(variables, conf, x_plot, y_plot, t_plot, main_v_dc_plot)
	Cameras_alignment = VisualizationWindow(variables, gui_visualization, visualization_closed_event,
	                                        visualization_win_front, flags=QtCore.Qt.WindowType.Tool)
	gui_visualization.setupUi(Cameras_alignment)
	sys.exit(app.exec())  # <-- Start the event loop for this QApplication instance


if __name__ == "__main__":
	try:
		conf, _ = runtime.load_project_config()
	except Exception as exc:
		print('Can not load the configuration file')
		print(exc)
		sys.exit()
	shared = runtime.create_shared_context(conf)

	app = QtWidgets.QApplication(sys.argv)
	app.setStyle('Fusion')
	Visualization = QtWidgets.QWidget()
	ui = Ui_Visualization(
		shared.variables,
		conf,
		shared.x_plot,
		shared.y_plot,
		shared.t_plot,
		shared.main_v_dc_plot,
	)
	ui.setupUi(Visualization)
	Visualization.show()
	sys.exit(app.exec())

