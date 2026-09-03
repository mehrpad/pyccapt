import sys
import time

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
# from numba import njit
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QTimer

# Local module and scripts
from pyccapt.control.core import live_calibration, runtime, tof2mc_simple
from pyccapt.control.devices import initialize_devices
from pyccapt.control.gui import tooltips


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
        # Per-ion-aligned leftover from the ring-buffer drain, carried to
        # the next tick so no events are dropped when a read lands while the
        # producer is mid-write (see update_graphs_helper).
        self._carry_x = np.array([])
        self._carry_y = np.array([])
        self._carry_t = np.array([])
        self._carry_v = np.array([])
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
        # Two parallel cumulative histograms per axis: one binned with
        # the live calibration applied, one with raw values. We always
        # update both so that toggling the "Uncalibrate" button is
        # purely a display swap and never loses prior events.
        self.hist_mc = np.zeros(len(self.bins_mc) - 1)
        self.hist_tof = np.zeros(len(self.bins_tof) - 1)
        self.hist_mc_uncalib = np.zeros(len(self.bins_mc) - 1)
        self.hist_tof_uncalib = np.zeros(len(self.bins_tof) - 1)

        self.update_timer = QTimer()  # Create a QTimer for updating graphs
        self.update_timer.timeout.connect(self.update_graphs)  # Connect it to the update_graphs slot

        # ----- Live calibration state -------------------------------------
        # Four spectra are accumulated live and in parallel: raw mc, raw
        # tof, calibrated mc, calibrated tof. The user picks which one is
        # *displayed* with the four view buttons; switching is a pure
        # display swap and never resets data or refits.
        #
        # `uncalibrated_mode` + `conf["visualization"]` together pick the
        # displayed accumulator (see the four display branches further
        # down). uncalibrated_mode=True -> raw series, False -> calibrated.
        #
        # The mc and tof calibrations are independent fits, so we run TWO
        # background QThreads — one per mode — and keep their parameter
        # sets side by side. Each fitter emits new params every
        # refit_interval_s; the GUI thread (never the worker) applies them
        # so the histogram arrays are only ever mutated from one thread
        # (see _drain_calib_updates). Until a mode's first successful fit
        # its params stay None and that axis falls back to the raw formula.
        self.uncalibrated_mode = False
        self._calib_params_tof = None
        self._calib_params_mc = None
        self._calib_worker_tof = None
        self._calib_worker_mc = None
        # Set by the worker-thread slots, consumed on the GUI thread: a
        # request to zero the matching calibrated accumulator because its
        # bins were binned under now-superseded parameters.
        self._calib_reset_tof = False
        self._calib_reset_mc = False
        # Latest human-readable status per fitter, rendered into the GUI
        # banner by the GUI thread when _calib_status_dirty is set.
        self._calib_status = {"tof": "calibrating…", "mc": "calibrating…"}
        self._calib_status_dirty = True

        # Lock that protects every write to / read of the
        # ``last_100_thousand_*`` ring buffer. The GUI thread mutates
        # these arrays in update_graphs_helper while the calibration
        # worker thread reads them via _calibration_snapshot. NumPy
        # array assignments are NOT atomic at the array level (a
        # concatenate -> realloc may race with a copy), so an explicit
        # RLock is the correct fix even though Python's GIL alone
        # usually papers over the race in practice.
        import threading as _threading

        self._buffer_lock = _threading.RLock()
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
        self.voltage.setStyleSheet(
            "QLineEdit{\n"
            "                                            background: rgb(223,223,233)\n"
            "                                            }\n"
            "                                        "
        )
        self.voltage.setObjectName("voltage")
        self.gridLayout_4.addWidget(self.voltage, 0, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(
            26, 17, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum
        )
        self.gridLayout_4.addItem(spacerItem, 0, 2, 1, 1)
        ####
        # self.vdc_time = QtWidgets.QGraphicsView(parent=Visualization)
        self.vdc_time = pg.PlotWidget(parent=Visualization)
        self.vdc_time.setBackground('w')
        ####
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.vdc_time.sizePolicy().hasHeightForWidth())
        self.vdc_time.setSizePolicy(sizePolicy)
        self.vdc_time.setMinimumSize(QtCore.QSize(250, 250))
        self.vdc_time.setStyleSheet(
            "QWidget{\n"
            "                                                    border: 0.5px solid gray;\n"
            "                                                    }\n"
            "                                                "
        )
        self.vdc_time.setObjectName("vdc_time")
        self.gridLayout_4.addWidget(self.vdc_time, 1, 0, 1, 3)
        self.dc_hold = QtWidgets.QPushButton(parent=Visualization)
        self.dc_hold.setMinimumSize(QtCore.QSize(100, 20))
        self.dc_hold.setMaximumSize(QtCore.QSize(100, 16777215))
        self.dc_hold.setObjectName("dc_hold")
        # Row: Hold DC Voltage | Set DC Voltage | [target voltage field].
        # The Set button + field are disabled until the DC voltage is held
        # (enabled in dc_hold_clicked). Set applies the entered value to the
        # supply via the existing flag_new_min_voltage mechanism.
        self.dc_hold_row = QtWidgets.QHBoxLayout()
        self.dc_hold_row.setObjectName("dc_hold_row")
        self.dc_hold_row.addWidget(self.dc_hold)
        self.set_dc_voltage = QtWidgets.QPushButton(parent=Visualization)
        self.set_dc_voltage.setMinimumSize(QtCore.QSize(0, 20))
        self.set_dc_voltage.setMaximumSize(QtCore.QSize(120, 16777215))
        self.set_dc_voltage.setObjectName("set_dc_voltage")
        self.set_dc_voltage.setEnabled(False)
        self.dc_hold_row.addWidget(self.set_dc_voltage)
        self.set_dc_voltage_value = QtWidgets.QLineEdit(parent=Visualization)
        self.set_dc_voltage_value.setMinimumSize(QtCore.QSize(80, 20))
        self.set_dc_voltage_value.setMaximumSize(QtCore.QSize(100, 16777215))
        self.set_dc_voltage_value.setStyleSheet("QLineEdit{background: rgb(223,223,233)}")
        self.set_dc_voltage_value.setObjectName("set_dc_voltage_value")
        self.set_dc_voltage_value.setEnabled(False)
        self.dc_hold_row.addWidget(self.set_dc_voltage_value)
        self.dc_hold_row.addStretch(1)
        self.gridLayout_4.addLayout(self.dc_hold_row, 2, 0, 1, 3)
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
        self.detection_rate.setStyleSheet(
            "QLineEdit{\n"
            "                                            background: rgb(223,223,233)\n"
            "                                            }\n"
            "                                        "
        )
        self.detection_rate.setObjectName("detection_rate")
        self.gridLayout.addWidget(self.detection_rate, 0, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum
        )
        self.gridLayout.addItem(spacerItem1, 0, 2, 1, 1)
        ####
        # self.detection_rate_viz = QtWidgets.QGraphicsView(parent=Visualization)
        self.detection_rate_viz = pg.PlotWidget(parent=Visualization)
        self.detection_rate_viz.setBackground('w')
        ####
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.detection_rate_viz.sizePolicy().hasHeightForWidth())
        self.detection_rate_viz.setSizePolicy(sizePolicy)
        self.detection_rate_viz.setMinimumSize(QtCore.QSize(250, 250))
        self.detection_rate_viz.setStyleSheet(
            "QWidget{\n"
            "                                            border: 0.5px solid gray;\n"
            "                                            }\n"
            "                                        "
        )
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
        self.hitmap_count.setStyleSheet(
            "QLineEdit{\n"
            "                                            background: rgb(223,223,233)\n"
            "                                            }\n"
            "                                        "
        )
        self.hitmap_count.setObjectName("hitmap_count")
        self.gridLayout_3.addWidget(self.hitmap_count, 0, 1, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum
        )
        self.gridLayout_3.addItem(spacerItem2, 0, 2, 1, 1)
        ###
        # self.detector_heatmap = QtWidgets.QGraphicsView(parent=Visualization)
        self.detector_heatmap = pg.PlotWidget(parent=Visualization)
        self.detector_heatmap.setBackground('w')
        ###
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.detector_heatmap.sizePolicy().hasHeightForWidth())
        self.detector_heatmap.setSizePolicy(sizePolicy)
        self.detector_heatmap.setMinimumSize(QtCore.QSize(250, 250))
        self.detector_heatmap.setStyleSheet(
            "QWidget{\n"
            "                                            border: 0.5px solid gray;\n"
            "                                            }\n"
            "                                        "
        )
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
        self.hitmap_plot_size.setStyleSheet(
            "QDoubleSpinBox{\n"
            "                                                background: rgb(223,223,233)\n"
            "                                                }\n"
            "                                            "
        )
        self.hitmap_plot_size.setObjectName("hitmap_plot_size")
        self.horizontalLayout_2.addWidget(self.hitmap_plot_size)
        self.hit_displayed = QtWidgets.QLineEdit(parent=Visualization)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.hit_displayed.sizePolicy().hasHeightForWidth())
        self.hit_displayed.setSizePolicy(sizePolicy)
        self.hit_displayed.setMinimumSize(QtCore.QSize(50, 20))
        self.hit_displayed.setStyleSheet(
            "QLineEdit{\n"
            "                                            background: rgb(223,223,233)\n"
            "                                            }\n"
            "                                        "
        )
        self.hit_displayed.setObjectName("hit_displayed")
        self.horizontalLayout_2.addWidget(self.hit_displayed)
        # Hitmap and FDM are now SEPARATE panels (see gridLayout_3b
        # below).  The old heatmap_fdm_switch toggle is no longer
        # needed; we keep a hidden stub so any external code that
        # still references the attribute doesn't crash.
        self.heatmap_fdm_switch = QtWidgets.QPushButton(parent=Visualization)
        self.heatmap_fdm_switch.setVisible(False)
        self.gridLayout_3.addLayout(self.horizontalLayout_2, 2, 0, 1, 3)
        self.gridLayout_5.addLayout(self.gridLayout_3, 0, 2, 1, 1)

        # ------------------------------------------------------------------
        # FDM-only panel - mirrors the hitmap panel above but always shows
        # the field-desorption map.  Header has the live ion-count used in
        # the current FDM; bottom field is the max ion count that will be
        # accumulated before the histogram resets and starts over.
        # ------------------------------------------------------------------
        self.gridLayout_3b = QtWidgets.QGridLayout()
        self.gridLayout_3b.setObjectName("gridLayout_3b")
        self.label_fdm_header = QtWidgets.QLabel(parent=Visualization)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_fdm_header.setFont(font)
        self.label_fdm_header.setText("FDM")
        self.gridLayout_3b.addWidget(self.label_fdm_header, 0, 0, 1, 1)
        self.fdm_count = QtWidgets.QLineEdit(parent=Visualization)
        sp = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        self.fdm_count.setSizePolicy(sp)
        self.fdm_count.setMinimumSize(QtCore.QSize(100, 20))
        self.fdm_count.setStyleSheet("QLineEdit{background: rgb(223,223,233)}")
        self.fdm_count.setReadOnly(True)
        self.fdm_count.setText("0")
        self.fdm_count.setObjectName("fdm_count")
        self.gridLayout_3b.addWidget(self.fdm_count, 0, 1, 1, 1)
        self.gridLayout_3b.addItem(
            QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum),
            0,
            2,
            1,
            1,
        )
        # Experiment-state indicator in the top-right corner.  The existing
        # graph refresh timer keeps it synchronized with the shared
        # ``start_flag`` used by the acquisition process.
        self.experiment_status_row = QtWidgets.QHBoxLayout()
        self.experiment_status_row.setSpacing(5)
        self.experiment_status_led = QtWidgets.QLabel(parent=Visualization)
        self.experiment_status_led.setFixedSize(QtCore.QSize(16, 16))
        self.experiment_status_led.setObjectName("experiment_status_led")
        self.experiment_status_led.setAccessibleName("Experiment status")
        self.experiment_status_text = QtWidgets.QLabel(parent=Visualization)
        self.experiment_status_text.setObjectName("experiment_status_text")
        self.experiment_status_row.addWidget(self.experiment_status_led)
        self.experiment_status_row.addWidget(self.experiment_status_text)
        self.gridLayout_3b.addLayout(self.experiment_status_row, 0, 3, 1, 1)

        self.detector_fdm = pg.PlotWidget(parent=Visualization)
        self.detector_fdm.setBackground('w')
        sp = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sp.setHorizontalStretch(1)
        sp.setVerticalStretch(1)
        self.detector_fdm.setSizePolicy(sp)
        self.detector_fdm.setMinimumSize(QtCore.QSize(250, 250))
        self.detector_fdm.setStyleSheet("QWidget{border: 0.5px solid gray;}")
        self.detector_fdm.setObjectName("detector_fdm")
        self.gridLayout_3b.addWidget(self.detector_fdm, 1, 0, 1, 4)

        # Bottom row: [Last Events toggle] [N input]
        # When the toggle is OFF (default), the FDM accumulates ions
        # forever and the N field is ignored.  When ON, only the last N
        # ions are used to build the FDM (sliding window).
        self.fdm_bottom_row = QtWidgets.QHBoxLayout()
        self.fdm_last_events_switch = QtWidgets.QPushButton(parent=Visualization)
        self.fdm_last_events_switch.setMinimumSize(QtCore.QSize(0, 20))
        self.fdm_last_events_switch.setMaximumSize(QtCore.QSize(120, 16777215))
        self.fdm_last_events_switch.setText("Last Events")
        self.fdm_last_events_switch.setCheckable(True)
        self.fdm_last_events_switch.setObjectName("fdm_last_events_switch")
        self.fdm_bottom_row.addWidget(self.fdm_last_events_switch)
        self.fdm_max_ions = QtWidgets.QLineEdit(parent=Visualization)
        self.fdm_max_ions.setMinimumSize(QtCore.QSize(100, 20))
        self.fdm_max_ions.setStyleSheet("QLineEdit{background: rgb(223,223,233)}")
        self.fdm_max_ions.setText("1000000")
        self.fdm_max_ions.setObjectName("fdm_max_ions")
        self.fdm_bottom_row.addWidget(self.fdm_max_ions)
        self.gridLayout_3b.addLayout(self.fdm_bottom_row, 2, 0, 1, 4)

        self.gridLayout_5.addLayout(self.gridLayout_3b, 0, 3, 1, 1)
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
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.histogram.sizePolicy().hasHeightForWidth())
        self.histogram.setSizePolicy(sizePolicy)
        self.histogram.setMinimumSize(QtCore.QSize(750, 150))
        self.histogram.setStyleSheet(
            "QWidget{\n"
            "                                            border: 0.5px solid gray;\n"
            "                                            }\n"
            "                                        "
        )
        self.histogram.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.histogram.setObjectName("histogram")
        self.gridLayout_2.addWidget(self.histogram, 1, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        # Four explicit view buttons. Each selects one of the four live
        # spectra (raw/calibrated x mc/tof) as a pure display swap; the
        # active one is highlighted green. No reset, no refit on click.
        # Display order: MC cal, MC, TOF cal, TOF.
        self.btn_view_mc_cal = QtWidgets.QPushButton(parent=Visualization)
        self.btn_view_mc_cal.setMinimumSize(QtCore.QSize(0, 20))
        self.btn_view_mc_cal.setMaximumSize(QtCore.QSize(90, 16777215))
        self.btn_view_mc_cal.setObjectName("btn_view_mc_cal")
        self.horizontalLayout.addWidget(self.btn_view_mc_cal)
        self.btn_view_mc = QtWidgets.QPushButton(parent=Visualization)
        self.btn_view_mc.setMinimumSize(QtCore.QSize(0, 20))
        self.btn_view_mc.setMaximumSize(QtCore.QSize(70, 16777215))
        self.btn_view_mc.setObjectName("btn_view_mc")
        self.horizontalLayout.addWidget(self.btn_view_mc)
        self.btn_view_tof_cal = QtWidgets.QPushButton(parent=Visualization)
        self.btn_view_tof_cal.setMinimumSize(QtCore.QSize(0, 20))
        self.btn_view_tof_cal.setMaximumSize(QtCore.QSize(90, 16777215))
        self.btn_view_tof_cal.setObjectName("btn_view_tof_cal")
        self.horizontalLayout.addWidget(self.btn_view_tof_cal)
        self.btn_view_tof = QtWidgets.QPushButton(parent=Visualization)
        self.btn_view_tof.setMinimumSize(QtCore.QSize(0, 20))
        self.btn_view_tof.setMaximumSize(QtCore.QSize(70, 16777215))
        self.btn_view_tof.setObjectName("btn_view_tof")
        self.horizontalLayout.addWidget(self.btn_view_tof)
        # Small status label that surfaces what the live-calibration
        # worker is doing ("calibrating…", "no clear peak", "R²=0.81…").
        self.calib_status_label = QtWidgets.QLabel(parent=Visualization)
        self.calib_status_label.setMinimumSize(QtCore.QSize(120, 20))
        font_status = QtGui.QFont()
        font_status.setItalic(True)
        font_status.setPointSize(8)
        self.calib_status_label.setFont(font_status)
        self.calib_status_label.setObjectName("calib_status_label")
        self.horizontalLayout.addWidget(self.calib_status_label)
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
        self.num_last_events.setStyleSheet(
            "QLineEdit{\n"
            "                                            background: rgb(223,223,233)\n"
            "                                            }\n"
            "                                        "
        )
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
        self.max_mc.setStyleSheet(
            "QLineEdit{\n"
            "                                            background: rgb(223,223,233)\n"
            "                                            }\n"
            "                                        "
        )
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
        self.max_tof.setStyleSheet(
            "QLineEdit{\n"
            "                                            background: rgb(223,223,233)\n"
            "                                            }\n"
            "                                        "
        )
        self.max_tof.setObjectName("max_tof")
        self.horizontalLayout.addWidget(self.max_tof)
        spacerItem3 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum
        )
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
        self.gridLayout_5.addLayout(self.gridLayout_2, 1, 0, 1, 4)
        self.gridLayout_6.addLayout(self.gridLayout_5, 0, 0, 1, 1)

        self.retranslateUi(Visualization)
        QtCore.QMetaObject.connectSlotsByName(Visualization)
        tooltips.apply_tooltips(self, tooltips.VISUALIZATION_TOOLTIPS)
        Visualization.setTabOrder(self.voltage, self.detection_rate)
        Visualization.setTabOrder(self.detection_rate, self.hitmap_count)
        Visualization.setTabOrder(self.hitmap_count, self.dc_hold)
        Visualization.setTabOrder(self.dc_hold, self.set_dc_voltage)
        Visualization.setTabOrder(self.set_dc_voltage, self.set_dc_voltage_value)
        Visualization.setTabOrder(self.set_dc_voltage_value, self.detection_rate_range_switch)
        Visualization.setTabOrder(self.detection_rate_range_switch, self.reset_heatmap_v)
        Visualization.setTabOrder(self.reset_heatmap_v, self.hitmap_plot_size)
        Visualization.setTabOrder(self.hitmap_plot_size, self.hit_displayed)
        Visualization.setTabOrder(self.hit_displayed, self.fdm_last_events_switch)
        Visualization.setTabOrder(self.fdm_last_events_switch, self.fdm_max_ions)
        Visualization.setTabOrder(self.fdm_max_ions, self.btn_view_mc_cal)
        Visualization.setTabOrder(self.btn_view_mc_cal, self.btn_view_mc)
        Visualization.setTabOrder(self.btn_view_mc, self.btn_view_tof_cal)
        Visualization.setTabOrder(self.btn_view_tof_cal, self.btn_view_tof)
        Visualization.setTabOrder(self.btn_view_tof, self.spectrum_last_events_switch)
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
        self.scatter = pg.ScatterPlotItem(size=self.hitmap_plot_size.value(), brush='black')
        self.detector_circle = QtWidgets.QGraphicsEllipseItem(-40, -40, 80, 80)  # x, y, width, height
        self.detector_circle.setPen(pg.mkPen(color=(255, 0, 0), width=2))
        self.detector_heatmap.addItem(self.detector_circle)
        self.detector_heatmap.setLabel("left", "X_det", units='mm', **self.styles)
        self.detector_heatmap.setLabel("bottom", "Y_det", units='mm', **self.styles)

        # FDM panel - one detector circle per plot (Qt items can't be
        # shared between two PlotWidgets) plus matching axis labels.
        self.detector_circle_fdm = QtWidgets.QGraphicsEllipseItem(-40, -40, 80, 80)
        self.detector_circle_fdm.setPen(pg.mkPen(color=(255, 0, 0), width=2))
        self.detector_fdm.addItem(self.detector_circle_fdm)
        self.detector_fdm.setLabel("left", "X_det", units='mm', **self.styles)
        self.detector_fdm.setLabel("bottom", "Y_det", units='mm', **self.styles)
        self.detector_fdm.getViewBox().setAspectLocked(True)
        # The FDM keeps TWO accumulators live in parallel every tick so
        # the Last-Events toggle is a pure display swap that never loses
        # data — you can flip between the entire map and the last-N map
        # without either resetting:
        #   * entire : _fdm_hist_all accumulates every ion forever as raw
        #              counts (log10 is applied once at display time - see
        #              _draw_fdm_display - not per tick, otherwise summing
        #              log10(tick_count+1) every tick makes hot pixels grow
        #              unboundedly and washes out the rest of the map under
        #              autoscale), and _fdm_count_all is the running total
        #              ion count.
        #   * window : a circular buffer stores the bin of each recent ion;
        #              its histogram is updated by adding new bins and
        #              subtracting only the bins that leave the window.
        self._fdm_hist_all = np.zeros_like(self.hist_fdm)
        self._fdm_count_all = 0
        self._fdm_use_last_events = False
        self._fdm_window_capacity = 0
        self._fdm_window_bins = np.empty(0, dtype=np.int32)
        self._fdm_window_start = 0
        self._fdm_window_count = 0
        self._fdm_window_hist = np.zeros_like(self.hist_fdm)
        self._original_fdm_button_style = self.fdm_last_events_switch.styleSheet()
        self.fdm_last_events_switch.clicked.connect(self._fdm_last_events_toggle)

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
        # Each button is a preset of (visualization axis, calibrated?).
        self.btn_view_mc.clicked.connect(lambda: self._select_spectrum_view("mc", True))
        self.btn_view_tof.clicked.connect(lambda: self._select_spectrum_view("tof", True))
        self.btn_view_mc_cal.clicked.connect(lambda: self._select_spectrum_view("mc", False))
        self.btn_view_tof_cal.clicked.connect(lambda: self._select_spectrum_view("tof", False))
        self._highlight_active_view_button()
        self.spectrum_last_events_switch.clicked.connect(self.spectrum_last_events)
        # Start the background calibration worker. It snapshots the
        # 100 000-event ring buffer every refit_interval_s, fits new
        # parameters off the GUI thread, and emits parameters_updated
        # when ready. The render path picks up the new params on the
        # next tick. Disable via config: live_calibration_refit_interval_s = 0
        self._start_live_calibration_worker()
        self.num_last_events.editingFinished.connect(self.parameters_changes)
        self.max_mc.editingFinished.connect(self.parameters_changes)
        self.max_tof.editingFinished.connect(self.parameters_changes)

        self.num_event_mc_tof = int(self.num_last_events.text())

        # heatmap_fdm_switch is now hidden - hitmap and FDM are always
        # rendered side-by-side in their own panels, no toggle needed.

        self.num_event_mc_tof = int(self.num_last_events.text())
        self.max_mc_val = int(self.max_mc.text())
        self.max_tof_val = int(self.max_tof.text())
        self.index_hist_tof = np.where(self.bins_tof == self.max_tof_val)[0][0]
        self.index_hist_mc = np.where(self.bins_mc == self.max_mc_val)[0][0]

        self.dc_hold.clicked.connect(self.dc_hold_clicked)
        self.set_dc_voltage.clicked.connect(self.set_dc_voltage_clicked)
        self.set_dc_voltage_value.editingFinished.connect(self._clamp_set_dc_voltage_field)

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
        self._experiment_status_running = None
        self._update_experiment_status_indicator()

    def _update_experiment_status_indicator(self):
        """Show green while an experiment runs and red while it is stopped."""
        running = bool(self.variables.start_flag)
        if running == self._experiment_status_running:
            return

        self._experiment_status_running = running
        color = "#22a447" if running else "#d32f2f"
        state = "Running" if running else "Stopped"
        self.experiment_status_led.setStyleSheet(
            "QLabel {"
            f"background-color: {color};"
            "border: 1px solid #555;"
            "border-radius: 8px;"
            "}"
        )
        self.experiment_status_led.setToolTip(f"Experiment {state.lower()}")
        self.experiment_status_text.setText(state)

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
        self.set_dc_voltage.setText(_translate("Visualization", "Set DC Voltage"))
        self.set_dc_voltage_value.setText(_translate("Visualization", str(int(self.conf.get('default_vdc_min', 500)))))
        self.label_201.setText(_translate("Visualization", "Detection Rate"))
        self.detection_rate.setText(_translate("Visualization", "0"))
        self.detection_rate_range_switch.setText(_translate("Visualization", "Short Range"))
        self.label_206.setText(_translate("Visualization", "Detector"))
        self.hitmap_count.setText(_translate("Visualization", "0"))
        self.reset_heatmap_v.setText(_translate("Visualization", "Reset"))
        self.hit_displayed.setText(_translate("Visualization", "2000"))
        # heatmap_fdm_switch is hidden but we still set its text in case
        # any external code reads it.
        self.heatmap_fdm_switch.setText(_translate("Visualization", "Hitmap/FDM"))
        self.label_207.setText(_translate("Visualization", "Spectrum"))
        self.btn_view_mc.setText(_translate("Visualization", "MC"))
        self.btn_view_tof.setText(_translate("Visualization", "TOF"))
        self.btn_view_mc_cal.setText(_translate("Visualization", "MC cal."))
        self.btn_view_tof_cal.setText(_translate("Visualization", "TOF cal."))
        self.calib_status_label.setText(_translate("Visualization", "live cal: calibrating…"))
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
                self.dc_hold.setStyleSheet("QPushButton{\nbackground: rgb(0, 255, 26)\n}")
                self._set_dc_voltage_controls_enabled(True)
            elif self.variables.vdc_hold:
                self.variables.vdc_hold = False
                self.dc_hold.setStyleSheet(self.original_button_style)
                self._set_dc_voltage_controls_enabled(False)

    def _dc_voltage_limits(self):
        """(min, max) DC voltage the Set field allows, from config.toml."""
        lo = int(self.conf.get('default_vdc_min', 500))
        hi = int(self.conf.get('default_vdc_max', 9000))
        return (lo, hi) if lo <= hi else (hi, lo)

    def _set_dc_voltage_controls_enabled(self, enabled):
        """Enable the Set-DC-voltage field + button only while DC is held."""
        self.set_dc_voltage_value.setEnabled(enabled)
        self.set_dc_voltage.setEnabled(enabled)
        if enabled:
            # Seed the field with the current supply voltage (clamped) so the
            # operator nudges from where it is now.
            lo, hi = self._dc_voltage_limits()
            try:
                cur = int(float(getattr(self.variables, 'specimen_voltage', 0)))
            except (TypeError, ValueError):
                cur = lo
            self.set_dc_voltage_value.setText(str(min(max(cur, lo), hi)))

    def _clamp_set_dc_voltage_field(self):
        """Clamp the typed value into the config [min, max] DC range."""
        lo, hi = self._dc_voltage_limits()
        try:
            val = int(float(self.set_dc_voltage_value.text().strip()))
        except (TypeError, ValueError):
            self.set_dc_voltage_value.setText(str(lo))
            return
        self.set_dc_voltage_value.setText(str(min(max(val, lo), hi)))

    def set_dc_voltage_clicked(self):
        """Apply the entered DC voltage to the supply (only while DC is held).

        Reuses the existing mechanism unchanged: write the clamped target
        into ``variables.vdc_min`` and raise ``flag_new_min_voltage``, which
        the experiment control loop consumes to ramp the supply to it (the
        same path the old main-GUI 'Set' button used, just with a
        user-entered value instead of the Min. Voltage field).
        """
        if not self.variables.vdc_hold:
            self.error_message("Hold the DC voltage first")
            return
        lo, hi = self._dc_voltage_limits()
        try:
            val = int(float(self.set_dc_voltage_value.text().strip()))
        except (TypeError, ValueError):
            self.error_message("Enter a valid DC voltage (V)")
            return
        val = min(max(val, lo), hi)
        self.set_dc_voltage_value.setText(str(val))
        self.variables.vdc_min = val
        self.variables.flag_new_min_voltage = True

    def heatmap_fdm_switch_change(self):
        """No-op kept for backward compatibility.

        Hitmap and FDM are now rendered side-by-side in their own
        panels (detector_heatmap + detector_fdm) every refresh - there
        is no longer anything to toggle.  Any external code that still
        clicks the (now-hidden) heatmap_fdm_switch button just lands
        here harmlessly.
        """
        return

    def _fdm_last_events_toggle(self):
        """Swap the FDM display between the entire map and the last-N map.

        Both accumulators are maintained every tick (see the FDM block in
        update_graphs_helper), so this is a pure display swap: neither the
        entire histogram nor the sliding window is cleared, and toggling
        back and forth never loses data.

        Default (button up)  - show the entire FDM (every ion ever).
        Toggled on (green)   - show only the most recent fdm_max_ions hits.
        """
        self._fdm_use_last_events = self.fdm_last_events_switch.isChecked()
        if self._fdm_use_last_events:
            self._seed_fdm_window_from_retained_events()
            self.fdm_last_events_switch.setStyleSheet("QPushButton{background: rgb(0, 255, 26)}")
        else:
            self.fdm_last_events_switch.setStyleSheet(self._original_fdm_button_style)
        # Keep the toggle responsive after the experiment has stopped.
        self._redraw_if_stopped()

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
        # Apply the reset immediately if the experiment is already stopped.
        self._redraw_if_stopped()

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
            self.detection_rate_range_switch.setStyleSheet("QPushButton{\nbackground: rgb(0, 255, 26)\n}")
        else:
            self.detection_rate_range_switch.setStyleSheet(self.original_button_style)

    def update_graphs_helper(
        self,
        final_drain=False,
    ):
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
        if final_drain:
            # Final export must not inherit the live view's short startup
            # delay; the producer is stopped and all available hits should be
            # consumed now, even for a very short/aborted experiment.
            self.counter_source = self.variables.counter_source
            self.index_wait_on_plot_start = max(self.index_wait_on_plot_start, 17)
        elif self.index_wait_on_plot_start <= 16:
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

        if self.counter_source == 'TDC' and self.variables.total_ions > 0 and self.index_wait_on_plot_start > 16:
            # Drain all four ring buffers (zero-copy NumPy slices, no IPC).
            # The producer writes x -> y -> t -> v in sequence, so a read
            # that lands mid-write sees a longer x/y/t than v. We keep only
            # the per-ion-aligned prefix (min length) this tick AND carry
            # the unmatched tail over to the next tick instead of dropping
            # it. Dropping it (the old behaviour) permanently lost those
            # events, so the viz total drifted below total_ions — visibly
            # so for the final events after the experiment stopped.
            xx = np.concatenate((self._carry_x, self.x_plot.read_all()))
            yy = np.concatenate((self._carry_y, self.y_plot.read_all()))
            tt = np.concatenate((self._carry_t, self.t_plot.read_all()))
            main_v_dc_dld = np.concatenate((self._carry_v, self.main_v_dc_plot.read_all()))
            n = min(len(xx), len(yy), len(tt), len(main_v_dc_dld))
            # Stash the leftover tail of each buffer (everything past the
            # aligned prefix) for next tick. When the producer has finished
            # all four indices are equal, so the tails are empty and the
            # final drain matches every remaining event.
            self._carry_x = xx[n:]
            self._carry_y = yy[n:]
            self._carry_t = tt[n:]
            self._carry_v = main_v_dc_dld[n:]
            if n == 0:
                xx = np.array([])
                yy = np.array([])
                tt = np.array([])
                main_v_dc_dld = np.array([])
            else:
                xx = xx[:n]
                yy = yy[:n]
                tt = tt[:n]
                main_v_dc_dld = main_v_dc_dld[:n]

            # self.length_events += len(self.tt)
            self.length_events += len(tt)

            # All ring-buffer writes go through the lock so the
            # background calibration worker's snapshot can never see a
            # half-updated buffer (e.g. concatenated v_dc but pre-trim
            # t / x / y after the 100 k cap kicks in).
            with self._buffer_lock:
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
                t_0 = self._t_0()

                # Apply any pending live-calibration updates (parameter
                # swaps + accumulator resets) here on the GUI thread,
                # before we touch the histograms — the worker slots only
                # flag the work, they never mutate the arrays.
                self._drain_calib_updates()

                # "Last events" view: re-bin only the most recent N events
                # for whichever single view is currently displayed.
                le_hist = self._last_events_spectrum_hist(t_0) if self.mc_tof_last_events_flag else None

                # Four cumulative spectra, every one updated each tick so
                # switching the displayed view is a pure swap that never
                # loses events:
                #   raw tof / raw mc -> geometry-only, never depend on a fit
                #   calibrated tof   -> uses the tof-mode fit (independent)
                #   calibrated mc    -> uses the mc-mode fit  (independent)
                # We bin only this tick's new events into each accumulator.
                batch_t = tt[mask_t]
                batch_v = main_v_dc_dld[mask_t]
                batch_x = xx[mask_t]
                batch_y = yy[mask_t]
                tof_raw = batch_t
                mc_raw = tof2mc_simple.tof_2_mc(
                    batch_t, t_0, batch_v, batch_x, batch_y,
                    flightPathLength=self.conf["flight_path_length"],
                )
                self.hist_tof_uncalib += np.histogram(tof_raw, bins=self.bins_tof)[0]
                self.hist_mc_uncalib += np.histogram(mc_raw, bins=self.bins_mc)[0]
                tof_cal = self._apply_axis(
                    self._calib_params_tof, "tof", batch_t, batch_v, batch_x, batch_y, t_0
                )
                mc_cal = self._apply_axis(
                    self._calib_params_mc, "mc", batch_t, batch_v, batch_x, batch_y, t_0
                )
                self.hist_tof += np.histogram(tof_cal, bins=self.bins_tof)[0]
                self.hist_mc += np.histogram(mc_cal, bins=self.bins_mc)[0]

                # Draw the active spectrum from the freshly updated buffers.
                self._draw_spectrum(le_hist)
            except Exception as e:
                print(
                    f"{initialize_devices.bcolors.FAIL}Error: Cannot plot Histogram correctly{initialize_devices.bcolors.ENDC}"
                )
                print(e)
            # Hitmap and FDM are now rendered every tick into two
            # separate panels (detector_heatmap + detector_fdm).  The
            # heatmap_fdm_switch toggle is gone.
            hist, xedges, yedges = np.histogram2d(
                xx * 10,
                yy * 10,
                bins=self.bins_detector,
                range=self.range,
            )

            # --- Hitmap (left panel) -------------------------------------
            self._draw_hitmap()

            # --- FDM (right panel) ---------------------------------------
            # Both FDMs are updated every tick so the Last Events toggle is
            # a pure display swap (see _fdm_last_events_toggle):
            #   * entire : accumulate every ion into _fdm_hist_all forever.
            #   * window : keep a sliding window of the most recent fdm_max
            #              ions and rebuild the map from it each tick.
            try:
                fdm_max = max(1, int(float(self.fdm_max_ions.text())))
            except (ValueError, AttributeError):
                fdm_max = 1_000_000
            new_events = int(np.sum(hist))

            # Entire FDM: accumulate raw counts forever (log10 is applied
            # once at display time in _draw_fdm_display).
            self._fdm_hist_all += hist
            self._fdm_count_all += new_events

            # Maintain the last-events histogram incrementally. This adds
            # only this tick's ions and subtracts only expired ions.
            self._update_fdm_window(xx * 10, yy * 10, fdm_max)

            # Display whichever map the toggle selects (no new accumulation).
            self._draw_fdm_display()

    # ------------------------------------------------------------------ render
    # The rendering of the spectrum, hitmap and FDM is factored into the
    # helpers below so it can be reused both from the live update tick and
    # from _render_static_views (which redraws from the retained buffers
    # after the experiment has stopped, keeping the view buttons working).

    def _t_0(self):
        """Return the t_0 constant for the active pulse mode (s)."""
        if self.variables.pulse_mode in ('Laser', 'VoltageLaser'):
            return self.conf["t_0_laser"]
        return self.conf["t_0_voltage"]

    def _last_events_spectrum_hist(self, t_0):
        """Histogram of the most recent ``num_event_mc_tof`` events for the
        currently displayed axis (mc or tof), calibrated or raw per the
        active view. Reads only the retained ring buffer, so it works the
        same whether the experiment is running or stopped."""
        t_le = self.last_100_thousand_t[-self.num_event_mc_tof:]
        v_le = self.last_100_thousand_v[-self.num_event_mc_tof:]
        x_le = self.last_100_thousand_det_x[-self.num_event_mc_tof:]
        y_le = self.last_100_thousand_det_y[-self.num_event_mc_tof:]
        if self.conf["visualization"] == "tof":
            params = None if self.uncalibrated_mode else self._calib_params_tof
            vals = self._apply_axis(params, "tof", t_le, v_le, x_le, y_le, t_0)
            return np.histogram(vals, bins=self.bins_tof)[0]
        params = None if self.uncalibrated_mode else self._calib_params_mc
        vals = self._apply_axis(params, "mc", t_le, v_le, x_le, y_le, t_0)
        return np.histogram(vals, bins=self.bins_mc)[0]

    def _plot_spectrum_hist(self, hist, bins):
        """Plot a single spectrum histogram with the standard styling."""
        hist = np.copy(hist)
        hist[hist == 0] = 1  # Avoid log(0) error
        self.histogram.plot(
            bins,
            hist,
            stepMode="center",
            fillLevel=0,
            fillOutline=True,
            brush='black',
            name="num events: %s" % self.length_events,
        )

    def _draw_spectrum(self, le_hist=None):
        """Render the active spectrum.

        ``le_hist`` is the precomputed last-events histogram for the active
        view (from _last_events_spectrum_hist) when "Last Events" is on, or
        None to draw the cumulative accumulator instead.
        """
        self.histogram.clear()
        if self.conf["visualization"] == "tof":
            if self.mc_tof_last_events_flag and le_hist is not None:
                src = le_hist
            else:
                src = self.hist_tof_uncalib if self.uncalibrated_mode else self.hist_tof
            self._plot_spectrum_hist(src[: self.index_hist_tof], self.bins_tof[: self.index_hist_tof + 1])
        else:  # "mc"
            if self.mc_tof_last_events_flag and le_hist is not None:
                src = le_hist
            else:
                src = self.hist_mc_uncalib if self.uncalibrated_mode else self.hist_mc
            self._plot_spectrum_hist(src[: self.index_hist_mc], self.bins_mc[: self.index_hist_mc + 1])

    def _draw_hitmap(self):
        """Redraw the detector hitmap scatter from the retained heatmap
        buffer. Honours a pending Reset request."""
        if self.variables.reset_heatmap:
            self.variables.reset_heatmap = False
            self.last_100_thousand_det_x_heatmap = np.array([])
            self.last_100_thousand_det_y_heatmap = np.array([])
        x_last_events = self.last_100_thousand_det_x_heatmap[:]
        y_last_events = self.last_100_thousand_det_y_heatmap[:]
        self.scatter.setSize(self.hitmap_plot_size.value())
        x = (x_last_events * 10)[-self.num_hit_display:]
        y = (y_last_events * 10)[-self.num_hit_display:]
        self.hitmap_count.setText(str(len(x)))
        self.scatter.clear()
        self.scatter.setData(x=x, y=y)
        self.detector_heatmap.clear()
        self.detector_heatmap.addItem(self.scatter)
        self.detector_heatmap.addItem(self.detector_circle)

    def _fdm_flat_bins(self, x, y):
        """Map detector coordinates to flat histogram bins; discard outliers."""
        x = np.asarray(x)
        y = np.asarray(y)
        nx, ny = self.bins_detector
        xmin, xmax = self.range[0]
        ymin, ymax = self.range[1]
        valid = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        if not np.any(valid):
            return np.empty(0, dtype=np.int32)
        xi = np.minimum(((x[valid] - xmin) * nx / (xmax - xmin)).astype(np.int64), nx - 1)
        yi = np.minimum(((y[valid] - ymin) * ny / (ymax - ymin)).astype(np.int64), ny - 1)
        return (xi * ny + yi).astype(np.int32, copy=False)

    def _seed_fdm_window_from_retained_events(self):
        """Immediately build last-N state from ions already retained by the GUI."""
        try:
            capacity = max(1, int(float(self.fdm_max_ions.text())))
        except (ValueError, AttributeError):
            capacity = 1_000_000
        x = self.last_100_thousand_det_x_heatmap[-capacity:] * 10
        y = self.last_100_thousand_det_y_heatmap[-capacity:] * 10
        bins = self._fdm_flat_bins(x, y)
        self._fdm_window_capacity = capacity
        self._fdm_window_bins = np.empty(capacity, dtype=np.int32)
        self._fdm_window_start = 0
        self._fdm_window_count = min(len(bins), capacity)
        if self._fdm_window_count:
            bins = bins[-self._fdm_window_count:]
            self._fdm_window_bins[:self._fdm_window_count] = bins
        counts = np.bincount(
            bins if self._fdm_window_count else np.empty(0, dtype=np.int32),
            minlength=self.hist_fdm.size,
        )
        self._fdm_window_hist = counts.reshape(self.hist_fdm.shape).astype(float, copy=False)

    def _update_fdm_window(self, x, y, capacity):
        """Slide the last-N FDM using O(new + expired) work, without rebinning N."""
        if capacity != self._fdm_window_capacity:
            # The retained arrays already include this tick, so seeding is
            # sufficient and adding x/y again would duplicate those ions.
            self._seed_fdm_window_from_retained_events()
            return
        incoming = self._fdm_flat_bins(x, y)
        if not len(incoming):
            return
        if len(incoming) >= capacity:
            incoming = incoming[-capacity:]
            self._fdm_window_bins[:] = incoming
            self._fdm_window_start = 0
            self._fdm_window_count = capacity
            self._fdm_window_hist = np.bincount(
                incoming, minlength=self.hist_fdm.size
            ).reshape(self.hist_fdm.shape).astype(float, copy=False)
            return

        expired = max(0, self._fdm_window_count + len(incoming) - capacity)
        if expired:
            positions = (self._fdm_window_start + np.arange(expired)) % capacity
            outgoing = self._fdm_window_bins[positions]
            self._fdm_window_hist -= np.bincount(
                outgoing, minlength=self.hist_fdm.size
            ).reshape(self.hist_fdm.shape)
            self._fdm_window_start = (self._fdm_window_start + expired) % capacity
            self._fdm_window_count -= expired

        end = (self._fdm_window_start + self._fdm_window_count) % capacity
        first = min(len(incoming), capacity - end)
        self._fdm_window_bins[end:end + first] = incoming[:first]
        if first < len(incoming):
            self._fdm_window_bins[:len(incoming) - first] = incoming[first:]
        self._fdm_window_count += len(incoming)
        self._fdm_window_hist += np.bincount(
            incoming, minlength=self.hist_fdm.size
        ).reshape(self.hist_fdm.shape)

    def _draw_fdm_display(self):
        """Render the FDM panel from the existing accumulators (entire map
        or last-events window). Performs no accumulation, so it is safe to
        call both live and after the experiment has stopped."""
        try:
            fdm_max = max(1, int(float(self.fdm_max_ions.text())))
        except (ValueError, AttributeError):
            fdm_max = 1_000_000
        if self._fdm_use_last_events:
            if fdm_max != self._fdm_window_capacity:
                self._seed_fdm_window_from_retained_events()
            self.hist_fdm = np.log10(self._fdm_window_hist + 1)
        else:
            self.hist_fdm = np.log10(self._fdm_hist_all + 1)
        displayed_count = self._fdm_window_count if self._fdm_use_last_events else self._fdm_count_all
        self.fdm_count.setText(str(displayed_count))
        # Detector-plane edges are fixed by the configured range, so they
        # can be derived without re-histogramming this tick's events.
        xedges = np.linspace(self.range[0][0], self.range[0][1], self.bins_detector[0] + 1)
        yedges = np.linspace(self.range[1][0], self.range[1][1], self.bins_detector[1] + 1)
        img_fdm = pg.ImageItem()
        img_fdm.setImage(np.copy(self.hist_fdm))
        img_fdm.setRect(
            QtCore.QRectF(
                xedges[0],
                yedges[0],
                xedges[-1] - xedges[0],
                yedges[-1] - yedges[0],
            )
        )
        lut = pg.colormap.get('viridis').getLookupTable(start=0.0, stop=1.0, nPts=256)
        img_fdm.setLookupTable(lut)
        self.detector_fdm.clear()
        self.detector_fdm.addItem(img_fdm)
        self.detector_fdm.addItem(self.detector_circle_fdm)
        self.detector_fdm.getViewBox().setAspectLocked(True)

    def _render_static_views(self):
        """Re-render the spectrum, hitmap and FDM from the retained buffers.

        Called when a view button is used after the experiment has stopped
        so the controls stay interactive (e.g. show the last 10000 events on
        the FDM / mass spectrum). Reads no new ring-buffer data and never
        touches the V_dc / detection-rate time series — those stay frozen at
        their final values. The next experiment start clears everything via
        ``plot_clear_flag`` in update_graphs.
        """
        try:
            t_0 = self._t_0()
            self._drain_calib_updates()
            le_hist = self._last_events_spectrum_hist(t_0) if self.mc_tof_last_events_flag else None
            self._draw_spectrum(le_hist)
            self._draw_hitmap()
            self._draw_fdm_display()
        except Exception as e:
            print(
                f"{initialize_devices.bcolors.FAIL}Error: Cannot redraw stopped views{initialize_devices.bcolors.ENDC}"
            )
            print(e)

    def _redraw_if_stopped(self):
        """Redraw the static views when the experiment is not running.

        While an experiment runs the live timer redraws every tick, so this
        is a no-op then. After a stop the timer no longer renders, so the
        view buttons call this to reflect their change immediately.
        """
        if not self.variables.start_flag and self.length_events > 0:
            self._render_static_views()

    def _disable_last_event_views_for_final_export(self):
        """Select the cumulative 1D spectrum and 2D FDM for final metadata.

        Change both controls directly instead of clicking them: click handlers
        can trigger an intermediate stopped-view redraw before the remaining
        detector buffers have been drained. The caller performs exactly one
        final ``update_graphs_helper`` pass after this state change.
        """
        self.mc_tof_last_events_flag = False
        self.spectrum_last_events_switch.setStyleSheet(self.original_button_style)

        self._fdm_use_last_events = False
        self.fdm_last_events_switch.setChecked(False)
        self.fdm_last_events_switch.setStyleSheet(self._original_fdm_button_style)

    def _export_all_plots(self, path_meta, suffix, *, render_window_offscreen=False):
        """Export every Visualization plot (+ a full-window grab) to PNGs.

        ``suffix`` is appended to each filename, e.g. ``'email'`` for the
        interim-notification snapshot. Mirrors the periodic / final export
        blocks below so the e-mail attachment looks like the saved metadata.
        """
        targets = (
            (self.vdc_time.plotItem, 'visualization_v_dc_p'),
            (self.detection_rate_viz.plotItem, 'visualization_detection_rate'),
            (self.detector_heatmap.plotItem, 'visualization_detector_hitmap'),
            (self.detector_fdm.plotItem, 'visualization_detector_fdm'),
            (self.histogram.plotItem, 'visualization_mc_tof'),
        )
        for plot_item, name in targets:
            exporter = pg.exporters.ImageExporter(plot_item)
            exporter.params['width'] = 1000
            exporter.params['height'] = 800
            exporter.export('%s/%s_%s.png' % (path_meta, name, suffix))

        if render_window_offscreen:
            # Render directly into a pixmap. This captures the temporary
            # cumulative plot state without repainting it onto the operator's
            # visible window or requiring an event-loop turn.
            screenshot = QtGui.QPixmap(self.visualization_window.size())
            screenshot.fill(QtCore.Qt.GlobalColor.transparent)
            self.visualization_window.render(screenshot)
        else:
            screenshot = QtWidgets.QApplication.primaryScreen().grabWindow(
                self.visualization_window.winId()
            )
        screenshot.save('%s/visualization_screenshot_%s.png' % (path_meta, suffix), 'png')

    def _export_interval_snapshots(self, path_meta, suffix):
        """Export Last Events and cumulative plots without changing the live view.

        Plot items and controls are switched only for synchronous exporters and
        off-screen renders, then immediately restored to the user's selections.
        No event processing occurs while temporary states are active, so the
        visible GUI does not flicker.
        """
        spectrum_last_events = self.mc_tof_last_events_flag
        fdm_last_events = self._fdm_use_last_events
        fdm_checked = self.fdm_last_events_switch.isChecked()

        try:
            # Additional interval set: both histogram views limited to their
            # configured Last Events windows.
            self.mc_tof_last_events_flag = True
            self._fdm_use_last_events = True
            self.fdm_last_events_switch.setChecked(True)
            self.spectrum_last_events_switch.setStyleSheet(
                "QPushButton{background: rgb(0, 255, 26)}"
            )
            self.fdm_last_events_switch.setStyleSheet(
                "QPushButton{background: rgb(0, 255, 26)}"
            )
            self._draw_spectrum(self._last_events_spectrum_hist(self._t_0()))
            self._draw_fdm_display()
            self._export_all_plots(
                path_meta,
                f"last_events_{suffix}",
                render_window_offscreen=True,
            )

            # Existing interval set: complete cumulative experiment.
            self.mc_tof_last_events_flag = False
            self._fdm_use_last_events = False
            self.fdm_last_events_switch.setChecked(False)
            self.spectrum_last_events_switch.setStyleSheet(self.original_button_style)
            self.fdm_last_events_switch.setStyleSheet(self._original_fdm_button_style)
            self._draw_spectrum(None)
            self._draw_fdm_display()
            self._export_all_plots(
                path_meta,
                suffix,
                render_window_offscreen=True,
            )
        finally:
            self.mc_tof_last_events_flag = spectrum_last_events
            self._fdm_use_last_events = fdm_last_events
            self.fdm_last_events_switch.setChecked(fdm_checked)
            self.spectrum_last_events_switch.setStyleSheet(
                "QPushButton{background: rgb(0, 255, 26)}"
                if spectrum_last_events else self.original_button_style
            )
            self.fdm_last_events_switch.setStyleSheet(
                "QPushButton{background: rgb(0, 255, 26)}"
                if fdm_last_events else self._original_fdm_button_style
            )
            le_hist = (
                self._last_events_spectrum_hist(self._t_0())
                if spectrum_last_events else None
            )
            self._draw_spectrum(le_hist)
            self._draw_fdm_display()

    def update_graphs(
        self,
    ):
        """
        Update the graphs
        Args:
            None

        Return:
            None
        """

        self._update_experiment_status_indicator()

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
            # Reset the FDM panel too.
            self.detector_fdm.clear()
            self.detector_fdm.addItem(self.detector_circle_fdm)
            self._fdm_hist_all[:] = 0.0
            self._fdm_count_all = 0
            self._fdm_window_capacity = 0
            self._fdm_window_bins = np.empty(0, dtype=np.int32)
            self._fdm_window_start = 0
            self._fdm_window_count = 0
            self._fdm_window_hist = np.zeros_like(self.hist_fdm)
            self.fdm_count.setText("0")
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
            self._carry_x = np.array([])
            self._carry_y = np.array([])
            self._carry_t = np.array([])
            self._carry_v = np.array([])
            self.length_events = 0
            self.hist_fdm, xedges, yedges = np.histogram2d([], [], bins=self.bins_detector, range=self.range)
            self._fdm_hist_all = np.zeros_like(self.hist_fdm)
            self._fdm_count_all = 0
            self._fdm_window_capacity = 0
            self._fdm_window_bins = np.empty(0, dtype=np.int32)
            self._fdm_window_start = 0
            self._fdm_window_count = 0
            self._fdm_window_hist = np.zeros_like(self.hist_fdm)
            self.fdm_count.setText("0")
            self.hist_mc = np.zeros(len(self.bins_mc) - 1)
            self.hist_tof = np.zeros(len(self.bins_tof) - 1)
            self.hist_mc_uncalib = np.zeros(len(self.bins_mc) - 1)
            self.hist_tof_uncalib = np.zeros(len(self.bins_tof) - 1)

        if self.index_auto_scale_graph == 30:
            self.vdc_time.enableAutoRange(axis='x')
            self.histogram.enableAutoRange(axis='y')
            self.detection_rate_viz.enableAutoRange(axis='x')
            self.detection_rate_viz.enableAutoRange(axis='y')
            self.detector_heatmap.enableAutoRange(axis='x')
            self.detector_heatmap.enableAutoRange(axis='y')
            self.index_auto_scale_graph = 0

        # Fresh snapshot requested by the experiment process for an interim
        # notification e-mail. Export the live plots + a full-window grab to
        # fixed '*_email.png' names so email_send can attach the newest one,
        # then clear the flag to signal the experiment loop we are done.
        if self.variables.flag_save_email_screenshot:
            try:
                path_meta = self.variables.path_meta
                if path_meta:
                    self._export_all_plots(path_meta, 'email')
            except Exception as e:
                print(
                    f"{initialize_devices.bcolors.WARNING}Warning: Could not save e-mail "
                    f"screenshot{initialize_devices.bcolors.ENDC}"
                )
                print(e)
            finally:
                self.variables.flag_save_email_screenshot = False

        # with self.variables.lock_statistics and self.variables.lock_setup_parameters:
        if self.variables.start_flag and self.variables.flag_visualization_start:
            self.index_auto_scale_graph += 1
            self.update_graphs_helper()

            # save plots to the file
            if time.time() - self.start_time_metadata >= self.variables.save_meta_interval_visualization:
                self.path_meta = self.variables.path_meta
                self._export_interval_snapshots(
                    self.path_meta,
                    str(self.index_plot_save),
                )
                self.start_time_metadata = time.time()
                # Increase the index
                self.index_plot_save += 1

        elif self.variables.last_screen_shot:
            self.path_meta = self.variables.path_meta
            if self.variables.vdc_hold:
                self.dc_hold.click()
            # (No more heatmap_fdm_switch click - both views are always
            # rendered into their own panels.)
            # Final metadata must show the complete cumulative experiment,
            # regardless of the operator's live Last Events selections.
            self._disable_last_event_views_for_final_export()
            if self.change_detection_rate_range:
                self.detection_rate_range_switch.click()
            if self.conf["visualization"] == "tof":
                # Force the MC view for a consistent exported screenshot,
                # keeping the current calibrated/raw choice.
                self._select_spectrum_view("mc", self.uncalibrated_mode)

            # The producer has stopped by this point. Drain all remaining
            # detector events and redraw once with both Last Events filters
            # disabled before any plot or full-window screenshot is saved.
            self.update_graphs_helper(final_drain=True)
            QtWidgets.QApplication.processEvents()

            exporter = pg.exporters.ImageExporter(self.vdc_time.plotItem)
            exporter.params['width'] = 1000  # Set the width of the image
            exporter.params['height'] = 800  # Set the height of the image
            exporter.export(self.path_meta + '/visualization_v_dc_p_final.png')
            exporter = pg.exporters.ImageExporter(self.detection_rate_viz.plotItem)
            exporter.params['width'] = 1000  # Set the width of the image
            exporter.params['height'] = 800  # Set the height of the image
            exporter.export(self.path_meta + '/visualization_detection_rate_final.png')
            # Hitmap panel
            exporter = pg.exporters.ImageExporter(self.detector_heatmap.plotItem)
            exporter.params['width'] = 1000
            exporter.params['height'] = 800
            exporter.export(self.path_meta + '/visualization_detector_hitmap_final.png')
            # FDM panel
            exporter = pg.exporters.ImageExporter(self.detector_fdm.plotItem)
            exporter.params['width'] = 1000
            exporter.params['height'] = 800
            exporter.export(self.path_meta + '/visualization_detector_fdm_final.png')
            exporter = pg.exporters.ImageExporter(self.histogram.plotItem)
            exporter.params['width'] = 1000  # Set the width of the image
            exporter.params['height'] = 800  # Set the height of the image
            exporter.export(self.path_meta + '/visualization_mc_tof_final.png')

            screenshot = QtWidgets.QApplication.primaryScreen().grabWindow(self.visualization_window.winId())
            screenshot.save(self.path_meta + '/visualization_screenshot_final.png', 'png')

            self.variables.last_screen_shot = False

    def _select_spectrum_view(self, visualization, uncalibrated):
        """Select which of the four live spectra is displayed.

        ``visualization`` is "mc" or "tof"; ``uncalibrated`` True picks
        the raw series, False the calibrated one. This is a pure display
        swap — it never clears an accumulator or restarts a fitter, so
        the four spectra keep filling in the background and switching is
        instant. Only the "Last Events" button limits what is shown.
        """
        self.conf["visualization"] = visualization
        self.uncalibrated_mode = uncalibrated
        if visualization == "tof":
            self.histogram.setLabel("bottom", "Time", units='ns', **self.styles)
        else:
            self.histogram.setLabel("bottom", "m/c", units='Da', **self.styles)
        self._highlight_active_view_button()
        # Keep the view switch responsive after the experiment has stopped.
        self._redraw_if_stopped()

    def _highlight_active_view_button(self):
        """Paint the active view button green, the other three default."""
        buttons = {
            ("mc", True): self.btn_view_mc,
            ("tof", True): self.btn_view_tof,
            ("mc", False): self.btn_view_mc_cal,
            ("tof", False): self.btn_view_tof_cal,
        }
        active = buttons.get((self.conf.get("visualization", "mc"), self.uncalibrated_mode))
        for button in buttons.values():
            if button is active:
                button.setStyleSheet("QPushButton{background: rgb(0, 255, 26)}")
            else:
                button.setStyleSheet(self.original_button_style)

    # ---------------------------------------------------------------- live cal

    def _apply_axis(self, params, want, t, v, x, y, t_0):
        """Return the requested axis ("tof" or "mc") for these events.

        Applies ``params`` when available (the matching mode's live fit),
        otherwise falls back to the geometry-only raw value so the
        calibrated accumulator still fills before the first fit lands.
        """
        if params is not None:
            corrected = live_calibration.apply_corrections(t, v, x, y, params)
            if corrected is not None:
                t_corr, mc_corr = corrected
                return t_corr if want == "tof" else mc_corr
        if want == "tof":
            return t
        return tof2mc_simple.tof_2_mc(
            t, t_0, v, x, y, flightPathLength=self.conf["flight_path_length"],
        )

    def _calibration_snapshot(self):
        """Snapshot callback handed to the LiveCalibrationWorker.

        Returns the 100 000-event ring buffer's contents as plain numpy
        arrays, or ``None`` when there is not yet enough data. Runs on
        the worker thread; never touches Qt widgets. Takes the same
        ``_buffer_lock`` as the writer in update_graphs_helper so the
        snapshot is guaranteed consistent across the four arrays even
        when the GUI thread is mid-concatenate.
        """
        try:
            with self._buffer_lock:
                t = self.last_100_thousand_t
                v = self.last_100_thousand_v
                x = self.last_100_thousand_det_x
                y = self.last_100_thousand_det_y
                if t is None or t.size == 0:
                    return None
                # Lengths can desync briefly across the four arrays
                # while update_graphs_helper concatenates one at a time.
                # The lock above already prevents that, but trim to the
                # common length defensively in case any future code
                # path bypasses the lock.
                n = min(len(t), len(v), len(x), len(y))
                if n == 0:
                    return None
                return t[-n:].copy(), v[-n:].copy(), x[-n:].copy(), y[-n:].copy()
        except Exception:
            return None

    def _start_live_calibration_worker(self):
        """Spin up one background fitter per mode, unless disabled in config.

        The mc and tof calibrations are independent fits, so we run two
        workers in parallel and keep both parameter sets live. That is
        what lets the four view buttons switch instantly without ever
        refitting or clearing an accumulator.
        """
        try:
            interval = float(self.conf.get("live_calibration_refit_interval_s", 15.0))
        except (TypeError, ValueError):
            interval = 15.0
        if interval <= 0:
            # Operator disabled live calibration entirely.
            self._calib_status["tof"] = "disabled"
            self._calib_status["mc"] = "disabled"
            self._calib_status_dirty = True
            return
        # Tell the workers which t_0 to use by hinting at the active pulse mode.
        try:
            pulse_mode = str(getattr(self.variables, "pulse_mode", "")).strip()
            self.conf["_active_pulse_mode_is_laser"] = pulse_mode in {"Laser", "VoltageLaser"}
        except Exception:
            self.conf["_active_pulse_mode_is_laser"] = False
        # LiveCalibrationWorker captures its mode from conf at construction
        # (the two pipelines differ — TOF starts with a sqrt(V/V̄)
        # prescaling, MC with a bowl-only initial step), so set the mode
        # immediately before building each worker.
        # Re-fit cadence is driven by the GUI's running ion counter
        # (length_events): each worker re-fits every
        # live_calibration_refit_event_interval new events.
        event_count = lambda: self.length_events
        self.conf["live_calibration_mode"] = "tof"
        self._calib_worker_tof = live_calibration.LiveCalibrationWorker(
            self._calibration_snapshot, self.conf, event_count,
        )
        self._calib_worker_tof.parameters_updated.connect(self._on_calib_params_tof)
        self._calib_worker_tof.status_changed.connect(self._on_calib_status_tof)
        self.conf["live_calibration_mode"] = "mc"
        self._calib_worker_mc = live_calibration.LiveCalibrationWorker(
            self._calibration_snapshot, self.conf, event_count,
        )
        self._calib_worker_mc.parameters_updated.connect(self._on_calib_params_mc)
        self._calib_worker_mc.status_changed.connect(self._on_calib_status_mc)
        self._calib_worker_tof.start()
        self._calib_worker_mc.start()

    # --- live-calibration slots (may run on a worker thread) -------------
    # PyQt delivers these in the *emitting* worker's thread (the receiver
    # is a plain Ui_ helper, not a QObject), so they must stay tiny and
    # only touch plain attributes — never Qt widgets, never the histogram
    # arrays. The GUI thread picks the work up in _drain_calib_updates()
    # on the next render tick.
    def _on_calib_params_tof(self, params):
        # Reset the calibrated accumulator ONLY when calibration turns on or
        # off (None<->params), not on every refit. With the ~1000-event
        # cadence, resetting each refit would keep the calibrated spectrum
        # permanently near-empty. Between transitions the histogram keeps
        # accumulating and sharpens as the fit converges (small frequent
        # parameter nudges barely move already-binned events).
        if (self._calib_params_tof is None) != (params is None):
            self._calib_reset_tof = True
        self._calib_params_tof = params  # atomic reference swap
        self._calib_status["tof"] = self._params_status(params)
        self._calib_status_dirty = True

    def _on_calib_params_mc(self, params):
        if (self._calib_params_mc is None) != (params is None):
            self._calib_reset_mc = True
        self._calib_params_mc = params
        self._calib_status["mc"] = self._params_status(params)
        self._calib_status_dirty = True

    def _on_calib_status_tof(self, text):
        self._calib_status["tof"] = text
        self._calib_status_dirty = True

    def _on_calib_status_mc(self, text):
        self._calib_status["mc"] = text
        self._calib_status_dirty = True

    @staticmethod
    def _params_status(params):
        if params is None:
            return "raw (no fit)"
        return f"R²={params.fit_quality:.2f} n={params.num_events_used}"

    def _rebuild_calibrated_hist(self, axis):
        """Rebuild the "tof" or "mc" calibrated accumulator from retained events.

        Called right after a calibration on/off transition instead of
        zeroing the accumulator, so ions seen before the transition are
        re-binned under the new params (or the raw fallback) instead of
        vanishing. Limited to the last-100k retained raw events - anything
        older than that window is still lost.
        """
        try:
            with self._buffer_lock:
                t = self.last_100_thousand_t
                v = self.last_100_thousand_v
                x = self.last_100_thousand_det_x
                y = self.last_100_thousand_det_y
            t_0 = self._t_0()
            params = self._calib_params_tof if axis == "tof" else self._calib_params_mc
            values = self._apply_axis(params, axis, t, v, x, y, t_0)
            bins = self.bins_tof if axis == "tof" else self.bins_mc
            hist = np.histogram(values, bins=bins)[0].astype(float)
            if axis == "tof":
                self.hist_tof = hist
            else:
                self.hist_mc = hist
        except Exception:
            pass

    def _drain_calib_updates(self):
        """Apply pending live-calibration updates on the GUI thread.

        A calibrated accumulator's old bins were binned under
        now-superseded params, so they no longer mean the right thing on
        the new axis. Rather than zeroing them outright (which threw away
        every ion seen before calibration turned on/off), rebuild the
        accumulator from the retained last-100k raw event buffer under the
        new params - so events before the transition still show up,
        limited only by that 100k retention window. Updating the status
        banner happens here too so the histogram arrays and Qt widgets are
        only ever touched from the GUI thread. The raw accumulators never
        reset — their bin meanings don't depend on any fit.
        """
        if self._calib_reset_tof:
            self._calib_reset_tof = False
            self._rebuild_calibrated_hist("tof")
        if self._calib_reset_mc:
            self._calib_reset_mc = False
            self._rebuild_calibrated_hist("mc")
        if self._calib_status_dirty:
            self._calib_status_dirty = False
            try:
                ok = any(s.startswith("R²") for s in self._calib_status.values())
                color = "#0a7d20" if ok else "#666666"
                self.calib_status_label.setText(
                    f"live cal — tof: {self._calib_status['tof']} | mc: {self._calib_status['mc']}"
                )
                self.calib_status_label.setStyleSheet(f"QLabel{{color:{color};}}")
            except Exception:
                pass

    def _stop_live_calibration_worker(self):
        """Stop both background fitters cleanly; called from .stop()."""
        for attr in ("_calib_worker_tof", "_calib_worker_mc"):
            worker = getattr(self, attr, None)
            setattr(self, attr, None)
            if worker is None:
                continue
            try:
                worker.stop()
                worker.wait(2000)  # ms
            except Exception:
                pass

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
            self.spectrum_last_events_switch.setStyleSheet("QPushButton{\nbackground: rgb(0, 255, 26)\n}")
        else:
            self.spectrum_last_events_switch.setStyleSheet(self.original_button_style)
        # Keep the toggle responsive after the experiment has stopped.
        self._redraw_if_stopped()

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

        # Keep the fields responsive after the experiment has stopped so the
        # operator can still re-window the spectrum / hitmap (e.g. last 10000).
        self._redraw_if_stopped()

    def error_message(self, message):
        """
        Display an error message and start a timer to hide it after 8 seconds

        Args:
                message (str): Error message to display

        Return:
                None
        """
        _translate = QtCore.QCoreApplication.translate
        self.Error.setText(
            _translate(
                "OXCART", "<html><head/><body><p><span style=\" color:#ff0000;\">" + message + "</span></p></body></html>"
            )
        )

        self.timer.start(8000)

    def hideMessage(
        self,
    ):
        """
        Hide the message and stop the timer
        Args:
            None

        Return:
            None
        """
        # Hide the message and stop the timer
        _translate = QtCore.QCoreApplication.translate
        self.Error.setText(
            _translate("OXCART", "<html><head/><body><p><span style=\" color:#ff0000;\"></span></p></body></html>")
        )

        self.timer.stop()

    def stop(self):
        """
        Stop any background activity
        Args:
            None

        Return:
            None
        """
        # Stop the live-calibration QThread cleanly so the visualization
        # subprocess can exit. The worker's ``stop()`` sets a flag that
        # its ``run()`` loop checks every 200 ms; ``wait(2000)`` gives it
        # up to 2 s to actually exit.
        self._stop_live_calibration_worker()


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

    def __init__(self, variables, gui_visualization, visualization_close_event, command_queue, *args, **kwargs):
        """
        Constructor for the VisualizationWindow class.

        Args:
            variables: Shared variables.
            gui_visualization: Instance of the Visualization.
            visualization_close_event: multiprocessing.Event signalled by
                this window when closed by the user.
            command_queue: multiprocessing.Queue of typed string commands
                from the main GUI ("show", "show_front", "hide").
        """
        super().__init__(*args, **kwargs)
        self.gui_visualization = gui_visualization
        self.variables = variables
        self.command_queue = command_queue
        self.visualization_close_event = visualization_close_event
        # Diagnostic: log the first few QTimer ticks + every command we
        # receive to files/logs/visualization_subprocess.log so we can
        # tell whether the timer fires and the queue is being drained.
        self._diag_ticks_logged = 0
        self._diag_log_path = None
        try:
            from pyccapt.control.core import runtime as _runtime

            self._diag_log_path = _runtime.project_path("files", "logs", "visualization_subprocess.log")
        except Exception:
            pass
        # Start hidden - check_if_should() below brings the window up the
        # first time a "show" command arrives on the queue.
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.check_if_should)
        self.timer.start(500)

    def closeEvent(self, event):
        """
        Don't actually close - hide the window so the subprocess stays alive
        and the next "open" from the main GUI is instant.  Using hide()
        (not showMinimized) avoids leaving a leftover minimised stub in the
        taskbar / desktop.
        """
        event.ignore()
        self.hide()
        self.visualization_close_event.set()

    def check_if_should(self):
        """Drain the command queue and dispatch each message in order."""
        # Diagnostic: confirm the QTimer is actually firing (first 3 ticks
        # only, to avoid filling the log).
        if self._diag_ticks_logged < 3 and self._diag_log_path is not None:
            try:
                import datetime as _dt

                with open(self._diag_log_path, "a", encoding="utf-8") as fh:
                    fh.write(f"[{_dt.datetime.now().isoformat()}] timer tick #{self._diag_ticks_logged + 1}\n")
            except Exception:
                pass
            self._diag_ticks_logged += 1
        raise_to_front = False
        make_visible = False
        hide = False
        drained_msgs = []
        while True:
            try:
                msg = self.command_queue.get_nowait()
            except Exception:
                break
            drained_msgs.append(msg)
            if msg == "show":
                make_visible = True
            elif msg == "show_front":
                make_visible = True
                raise_to_front = True
            elif msg == "hide":
                hide = True
        if drained_msgs and self._diag_log_path is not None:
            try:
                import datetime as _dt

                with open(self._diag_log_path, "a", encoding="utf-8") as fh:
                    fh.write(f"[{_dt.datetime.now().isoformat()}] received: {drained_msgs}\n")
            except Exception:
                pass
        if hide and not make_visible:
            self.hide()
            return
        if not make_visible:
            return
        # Always call show() + showNormal() unconditionally.  After a
        # previous closeEvent->hide() Qt may not honour a single show()
        # call on every platform; the explicit showNormal() also brings
        # the window out of a minimised state if it's been there.  We
        # deliberately do NOT toggle setWindowFlags() - that hides the
        # widget on Windows (Qt docs).
        self.show()
        self.showNormal()
        self.raise_()
        if raise_to_front:
            self.activateWindow()

    def setWindowStyleFusion(self):
        # Set the Fusion style
        QtWidgets.QApplication.setStyle("Fusion")


def run_visualization_window(
    variables, conf, visualization_closed_event, visualization_command_queue, x_plot, y_plot, t_plot, main_v_dc_plot
):
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
    # Every subprocess startup writes a one-line breadcrumb to the log.
    # If the visualization subprocess never gets that far the file stays
    # empty and we know the unpickling of the Process args failed before
    # this body even ran.  Crashes inside this body land in the same
    # file with a full traceback.
    import os
    import traceback
    import datetime as _dt

    log_path = None
    try:
        log_path = runtime.project_path("files", "logs", "visualization_subprocess.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"[{_dt.datetime.now().isoformat()}] pid={os.getpid()} startup\n")
    except Exception:
        pass
    try:
        app = QtWidgets.QApplication(sys.argv)
        app.setStyle('Fusion')
        app.setQuitOnLastWindowClosed(False)

        gui_visualization = Ui_Visualization(variables, conf, x_plot, y_plot, t_plot, main_v_dc_plot)
        Cameras_alignment = VisualizationWindow(
            variables,
            gui_visualization,
            visualization_closed_event,
            visualization_command_queue,
            flags=QtCore.Qt.WindowType.Tool,
        )
        gui_visualization.setupUi(Cameras_alignment)
        try:
            if log_path is not None:
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(f"[{_dt.datetime.now().isoformat()}] setupUi finished, entering app.exec()\n")
        except Exception:
            pass
        sys.exit(app.exec())
    except Exception:
        try:
            if log_path is not None:
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(f"[{_dt.datetime.now().isoformat()}] CRASH:\n")
                    traceback.print_exc(file=fh)
        except Exception:
            pass
        traceback.print_exc()
        raise


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
