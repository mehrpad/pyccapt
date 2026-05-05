import re
import sys
import time

import serial.tools.list_ports
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QThread
from PyQt6.QtGui import QPixmap

# Local module and scripts
from pyccapt.control.core import runtime
from pyccapt.control.nkt_photonics import origamiClassCLI
from pyccapt.control.smaract_mcs2 import mcs2_stage


def _available_serial_ports_text():
    ports = sorted(port.device for port in serial.tools.list_ports.comports() if getattr(port, "device", ""))
    return ", ".join(ports) if ports else "none detected"


class Ui_Laser_Control(object):
    def __init__(self, variables, conf):
        """
        Initialize the Ui_Laser_Control class.

        Args:
            variables: Global experiment variables.
            conf: Configuration settings.
        """
        self.variables = variables
        self.conf = conf

        self.listen_mode = False
        self.standby_mode = False
        self.enable_mode = False
        self.laser_on_mode = False
        self.change_laser_wavelegnth = False
        self.change_laser_power = False
        self.change_laser_rate = False
        self.change_laser_divition_factor = False

        self.index = 0

    def setupUi(self, Laser_Control):
        """
        Setup the GUI for the laser control.
        Args:
            Laser_Control: The GUI window

        Return:
            None
        """
        Laser_Control.setObjectName("Laser_Control")
        Laser_Control.resize(1003, 345)
        self.gridLayout_6 = QtWidgets.QGridLayout(Laser_Control)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.laser_wavelegnth = QtWidgets.QComboBox(parent=Laser_Control)
        self.laser_wavelegnth.setStyleSheet("QComboBox{background: rgb(223,223,233)}")
        self.laser_wavelegnth.setObjectName("laser_wavelegnth")
        self.laser_wavelegnth.addItem("")
        self.laser_wavelegnth.addItem("")
        self.laser_wavelegnth.addItem("")
        self.gridLayout_3.addWidget(self.laser_wavelegnth, 0, 1, 1, 1)
        self.led_laser_on = QtWidgets.QLabel(parent=Laser_Control)
        font = QtGui.QFont()
        font.setBold(True)
        self.led_laser_on.setFont(font)
        self.led_laser_on.setObjectName("led_laser_on")
        self.gridLayout_3.addWidget(self.led_laser_on, 1, 3, 1, 1)
        self.laser_rate = QtWidgets.QComboBox(parent=Laser_Control)
        self.laser_rate.setStyleSheet("QComboBox{background: rgb(223,223,233)}")
        self.laser_rate.setObjectName("laser_rate")
        self.laser_rate.addItem("")
        self.laser_rate.addItem("")
        self.laser_rate.addItem("")
        self.laser_rate.addItem("")
        self.laser_rate.addItem("")
        self.laser_rate.addItem("")
        self.laser_rate.addItem("")
        self.gridLayout_3.addWidget(self.laser_rate, 2, 1, 1, 1)
        self.led_laser_enable = QtWidgets.QLabel(parent=Laser_Control)
        font = QtGui.QFont()
        font.setBold(True)
        self.led_laser_enable.setFont(font)
        self.led_laser_enable.setObjectName("led_laser_enable")
        self.gridLayout_3.addWidget(self.led_laser_enable, 0, 3, 1, 1)
        self.laser_standby = QtWidgets.QPushButton(parent=Laser_Control)
        self.laser_standby.setMinimumSize(QtCore.QSize(90, 25))
        self.laser_standby.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.laser_standby.setStyleSheet("")
        self.laser_standby.setObjectName("laser_standby")
        self.gridLayout_3.addWidget(self.laser_standby, 2, 2, 1, 1)
        self.label_2 = QtWidgets.QLabel(parent=Laser_Control)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout_3.addWidget(self.label_2, 2, 0, 1, 1)
        self.laser_on = QtWidgets.QPushButton(parent=Laser_Control)
        self.laser_on.setMinimumSize(QtCore.QSize(90, 25))
        self.laser_on.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.laser_on.setStyleSheet("")
        self.laser_on.setObjectName("laser_on")
        self.gridLayout_3.addWidget(self.laser_on, 1, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(parent=Laser_Control)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.gridLayout_3.addWidget(self.label_3, 3, 0, 1, 1)
        self.laser_enable = QtWidgets.QPushButton(parent=Laser_Control)
        self.laser_enable.setMinimumSize(QtCore.QSize(90, 25))
        self.laser_enable.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.laser_enable.setStyleSheet("")
        self.laser_enable.setObjectName("laser_enable")
        self.gridLayout_3.addWidget(self.laser_enable, 0, 2, 1, 1)
        self.led_laser_listen = QtWidgets.QLabel(parent=Laser_Control)
        font = QtGui.QFont()
        font.setBold(True)
        self.led_laser_listen.setFont(font)
        self.led_laser_listen.setObjectName("led_laser_listen")
        self.gridLayout_3.addWidget(self.led_laser_listen, 3, 3, 1, 1)
        self.led_laser_laser_standby = QtWidgets.QLabel(parent=Laser_Control)
        font = QtGui.QFont()
        font.setBold(True)
        self.led_laser_laser_standby.setFont(font)
        self.led_laser_laser_standby.setObjectName("led_laser_laser_standby")
        self.gridLayout_3.addWidget(self.led_laser_laser_standby, 2, 3, 1, 1)
        self.label = QtWidgets.QLabel(parent=Laser_Control)
        font = QtGui.QFont()
        font.setBold(True)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout_3.addWidget(self.label, 1, 0, 1, 1)
        self.laser_listen = QtWidgets.QPushButton(parent=Laser_Control)
        self.laser_listen.setMinimumSize(QtCore.QSize(90, 25))
        self.laser_listen.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.laser_listen.setStyleSheet("")
        self.laser_listen.setObjectName("laser_listen")
        self.gridLayout_3.addWidget(self.laser_listen, 3, 2, 1, 1)
        self.label_4 = QtWidgets.QLabel(parent=Laser_Control)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.gridLayout_3.addWidget(self.label_4, 0, 0, 1, 1)
        self.laser_divition_factor = QtWidgets.QSpinBox(parent=Laser_Control)
        self.laser_divition_factor.setObjectName("laser_divition_factor")
        self.gridLayout_3.addWidget(self.laser_divition_factor, 3, 1, 1, 1)
        self.laser_power = QtWidgets.QDoubleSpinBox(parent=Laser_Control)
        self.laser_power.setObjectName("doubleSpinBox")
        self.gridLayout_3.addWidget(self.laser_power, 1, 1, 1, 1)
        self.gridLayout_5.addLayout(self.gridLayout_3, 0, 0, 2, 3)
        self.label_12 = QtWidgets.QLabel(parent=Laser_Control)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.gridLayout_5.addWidget(self.label_12, 0, 4, 1, 1)
        self.laser_scan_mode5 = QtWidgets.QComboBox(parent=Laser_Control)
        self.laser_scan_mode5.setStyleSheet("QComboBox{background: rgb(223,223,233)}")
        self.laser_scan_mode5.setObjectName("laser_scan_mode5")
        self.laser_scan_mode5.addItem("")
        self.gridLayout_5.addWidget(self.laser_scan_mode5, 0, 5, 1, 1)
        self.scanning_disp = QtWidgets.QGraphicsView(parent=Laser_Control)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                           QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.scanning_disp.sizePolicy().hasHeightForWidth())
        self.scanning_disp.setSizePolicy(sizePolicy)
        self.scanning_disp.setMinimumSize(QtCore.QSize(250, 250))
        self.scanning_disp.setStyleSheet("QWidget{\n"
                                         "                                    border: 0.5px solid gray;\n"
                                         "                                    }\n"
                                         "                                ")
        self.scanning_disp.setObjectName("scanning_disp")
        self.gridLayout_5.addWidget(self.scanning_disp, 0, 6, 4, 1)
        self.label_13 = QtWidgets.QLabel(parent=Laser_Control)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.gridLayout_5.addWidget(self.label_13, 1, 4, 1, 1)
        self.laser_focus_mode = QtWidgets.QComboBox(parent=Laser_Control)
        self.laser_focus_mode.setStyleSheet("QComboBox{background: rgb(223,223,233)}")
        self.laser_focus_mode.setObjectName("laser_focus_mode")
        self.laser_focus_mode.addItem("")
        self.gridLayout_5.addWidget(self.laser_focus_mode, 1, 5, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
                                           QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.label_9 = QtWidgets.QLabel(parent=Laser_Control)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout.addWidget(self.label_9)
        self.laser_power_disp = QtWidgets.QLCDNumber(parent=Laser_Control)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                           QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.laser_power_disp.sizePolicy().hasHeightForWidth())
        self.laser_power_disp.setSizePolicy(sizePolicy)
        self.laser_power_disp.setMinimumSize(QtCore.QSize(100, 50))
        self.laser_power_disp.setMaximumSize(QtCore.QSize(100, 50))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.laser_power_disp.setFont(font)
        self.laser_power_disp.setStyleSheet("QLCDNumber{\n"
                                            "                                            border: 2px solid green;\n"
                                            "                                            border-radius: 10px;\n"
                                            "                                            padding: 0 8px;\n"
                                            "                                            }\n"
                                            "                                        ")
        self.laser_power_disp.setObjectName("laser_power_disp")
        self.horizontalLayout.addWidget(self.laser_power_disp)
        self.label_10 = QtWidgets.QLabel(parent=Laser_Control)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout.addWidget(self.label_10)
        self.laser_pulse_energy_disp = QtWidgets.QLCDNumber(parent=Laser_Control)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                           QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.laser_pulse_energy_disp.sizePolicy().hasHeightForWidth())
        self.laser_pulse_energy_disp.setSizePolicy(sizePolicy)
        self.laser_pulse_energy_disp.setMinimumSize(QtCore.QSize(100, 50))
        self.laser_pulse_energy_disp.setMaximumSize(QtCore.QSize(100, 50))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.laser_pulse_energy_disp.setFont(font)
        self.laser_pulse_energy_disp.setStyleSheet("QLCDNumber{\n"
                                                   "                                            border: 2px solid green;\n"
                                                   "                                            border-radius: 10px;\n"
                                                   "                                            padding: 0 8px;\n"
                                                   "                                            }\n"
                                                   "                                        ")
        self.laser_pulse_energy_disp.setObjectName("laser_pulse_energy_disp")
        self.horizontalLayout.addWidget(self.laser_pulse_energy_disp)
        self.label_11 = QtWidgets.QLabel(parent=Laser_Control)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.horizontalLayout.addWidget(self.label_11)
        self.laser_repetion_rate_disp = QtWidgets.QLCDNumber(parent=Laser_Control)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                           QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.laser_repetion_rate_disp.sizePolicy().hasHeightForWidth())
        self.laser_repetion_rate_disp.setSizePolicy(sizePolicy)
        self.laser_repetion_rate_disp.setMinimumSize(QtCore.QSize(100, 50))
        self.laser_repetion_rate_disp.setMaximumSize(QtCore.QSize(100, 50))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.laser_repetion_rate_disp.setFont(font)
        self.laser_repetion_rate_disp.setStyleSheet("QLCDNumber{\n"
                                                    "                                            border: 2px solid green;\n"
                                                    "                                            border-radius: 10px;\n"
                                                    "                                            padding: 0 8px;\n"
                                                    "                                            }\n"
                                                    "                                        ")
        self.laser_repetion_rate_disp.setObjectName("laser_repetion_rate_disp")
        self.horizontalLayout.addWidget(self.laser_repetion_rate_disp)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
                                            QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.gridLayout_5.addLayout(self.horizontalLayout, 2, 0, 1, 6)
        # ------------------------------------------------------------------
        # Laser focusing stage (SmarAct MCS2): 3 LCDs per axis (mm/um/nm)
        # ------------------------------------------------------------------
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        header_font = QtGui.QFont()
        header_font.setBold(True)
        header_font.setPointSize(8)
        for col, name in enumerate(("", "mm", "µm", "nm"), start=0):
	        lab = QtWidgets.QLabel(parent=Laser_Control)
	        lab.setText(name)
	        lab.setFont(header_font)
	        lab.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
	        self.gridLayout_4.addWidget(lab, 0, col, 1, 1)

        def _make_axis_lcd():
	        lcd = QtWidgets.QLCDNumber(parent=Laser_Control)
	        lcd.setDigitCount(5)
	        lcd.setSegmentStyle(QtWidgets.QLCDNumber.SegmentStyle.Flat)
	        lcd.setMinimumSize(QtCore.QSize(60, 28))
	        lcd.setStyleSheet(
		        "QLCDNumber{"
		        "background: rgb(220,235,245);"
		        "color: rgb(0,30,80);"
		        "border: 1px solid rgb(120,160,200);"
		        "border-radius: 4px;"
		        "}"
	        )
	        return lcd

        bold_font = QtGui.QFont()
        bold_font.setBold(True)
        self.label_19 = QtWidgets.QLabel("x", parent=Laser_Control)
        self.label_19.setFont(bold_font)
        self.label_17 = QtWidgets.QLabel("y", parent=Laser_Control)
        self.label_17.setFont(bold_font)
        self.label_18 = QtWidgets.QLabel("z", parent=Laser_Control)
        self.label_18.setFont(bold_font)

        self.laser_x_mm = _make_axis_lcd()
        self.laser_x_um = _make_axis_lcd()
        self.laser_x_nm = _make_axis_lcd()
        self.laser_y_mm = _make_axis_lcd()
        self.laser_y_um = _make_axis_lcd()
        self.laser_y_nm = _make_axis_lcd()
        self.laser_z_mm = _make_axis_lcd()
        self.laser_z_um = _make_axis_lcd()
        self.laser_z_nm = _make_axis_lcd()

        # Legacy single-LCD attributes kept for compatibility (hidden).
        self.laser_x_cord = QtWidgets.QLCDNumber(parent=Laser_Control)
        self.laser_y_cord = QtWidgets.QLCDNumber(parent=Laser_Control)
        self.laser_z_cord = QtWidgets.QLCDNumber(parent=Laser_Control)
        for w in (self.laser_x_cord, self.laser_y_cord, self.laser_z_cord):
	        w.setVisible(False)

        for row, (lbl, mm, um, nm) in enumerate(
		        (
				        (self.label_19, self.laser_x_mm, self.laser_x_um, self.laser_x_nm),
				        (self.label_17, self.laser_y_mm, self.laser_y_um, self.laser_y_nm),
				        (self.label_18, self.laser_z_mm, self.laser_z_um, self.laser_z_nm),
		        ),
		        start=1,
        ):
	        self.gridLayout_4.addWidget(lbl, row, 0, 1, 1)
	        self.gridLayout_4.addWidget(mm, row, 1, 1, 1)
	        self.gridLayout_4.addWidget(um, row, 2, 1, 1)
	        self.gridLayout_4.addWidget(nm, row, 3, 1, 1)
        self.gridLayout_5.addLayout(self.gridLayout_4, 3, 0, 1, 1)

        # ------------------------------------------------------------------
        # Speed slider (Simple-Mode-style 1..N) + jog step spin
        # ------------------------------------------------------------------
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")

        self._speed_max_mm_s = float(self.conf.get('stage_speed_max_mm_s', 1.0))
        self._speed_max_level = int(self.conf.get('stage_speed_level_max', 11))
        self._speed_min_level = int(self.conf.get('stage_speed_level_min', 1))
        self._speed_default = int(self.conf.get('stage_speed_level_default', 5))
        self._click_duration_s = float(self.conf.get('stage_click_duration_s', 0.2))
        self._speed_table = self.conf.get('stage_speed_table_mm_s') or None
        self._home_target_m = (
	        float(self.conf.get('laser_stage_home_x_mm', 0.0)) * 1e-3,
	        float(self.conf.get('laser_stage_home_y_mm', 0.0)) * 1e-3,
	        float(self.conf.get('laser_stage_home_z_mm', 0.0)) * 1e-3,
        )
        self._stage_locator = self.conf.get('stage_smartact_laser', '')
        self._stage_connect_error = ""
        self.flag_super_user_stage = False  # gates the Reference button

        # Header
        self.label_14 = QtWidgets.QLabel("Speed", parent=Laser_Control)
        self.label_14.setFont(bold_font)
        self.label_14.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.gridLayout_2.addWidget(self.label_14, 0, 1, 1, 1)

        # Per-axis labels
        self.label_15 = QtWidgets.QLabel("X", parent=Laser_Control);
        self.label_15.setFont(bold_font)
        self.label_16 = QtWidgets.QLabel("Y", parent=Laser_Control);
        self.label_16.setFont(bold_font)
        self.label_speed_z = QtWidgets.QLabel("Z", parent=Laser_Control);
        self.label_speed_z.setFont(bold_font)

        def _make_speed_slider():
	        s = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, parent=Laser_Control)
	        s.setMinimum(self._speed_min_level)
	        s.setMaximum(self._speed_max_level)
	        s.setValue(self._speed_default)
	        s.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
	        s.setTickInterval(1)
	        s.setMinimumWidth(160)
	        return s

        self.laser_speed_x = _make_speed_slider()
        self.laser_speed_y = _make_speed_slider()
        self.laser_speed_z = _make_speed_slider()

        self.laser_speed_x_label = QtWidgets.QLabel(parent=Laser_Control);
        self.laser_speed_x_label.setMinimumWidth(230)
        self.laser_speed_y_label = QtWidgets.QLabel(parent=Laser_Control);
        self.laser_speed_y_label.setMinimumWidth(230)
        self.laser_speed_z_label = QtWidgets.QLabel(parent=Laser_Control);
        self.laser_speed_z_label.setMinimumWidth(230)

        for row, (lbl, sl, val) in enumerate(
		        (
				        (self.label_15, self.laser_speed_x, self.laser_speed_x_label),
				        (self.label_16, self.laser_speed_y, self.laser_speed_y_label),
				        (self.label_speed_z, self.laser_speed_z, self.laser_speed_z_label),
		        ),
		        start=1,
        ):
	        self.gridLayout_2.addWidget(lbl, row, 0, 1, 1)
	        self.gridLayout_2.addWidget(sl, row, 1, 1, 1)
	        self.gridLayout_2.addWidget(val, row, 2, 1, 1)

        # Backwards-compat aliases (still referenced by some external code).
        self.laser_speed_lr = self.laser_speed_x
        self.laser_speed_ud = self.laser_speed_y
        self.laser_speed_fb = self.laser_speed_z

        self.gridLayout_5.addLayout(self.gridLayout_2, 3, 1, 1, 1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
                                            QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout.addItem(spacerItem2, 0, 0, 1, 1)
        self.laser_up = QtWidgets.QPushButton(parent=Laser_Control)
        self.laser_up.setMinimumSize(QtCore.QSize(50, 25))
        self.laser_up.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.laser_up.setStyleSheet("")
        self.laser_up.setObjectName("laser_up")
        self.gridLayout.addWidget(self.laser_up, 0, 1, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
                                            QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout.addItem(spacerItem3, 0, 2, 1, 1)
        self.laser_left = QtWidgets.QPushButton(parent=Laser_Control)
        self.laser_left.setMinimumSize(QtCore.QSize(50, 25))
        self.laser_left.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.laser_left.setStyleSheet("")
        self.laser_left.setObjectName("laser_left")
        self.gridLayout.addWidget(self.laser_left, 1, 0, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
                                            QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout.addItem(spacerItem4, 1, 1, 1, 1)
        self.leser_right = QtWidgets.QPushButton(parent=Laser_Control)
        self.leser_right.setMinimumSize(QtCore.QSize(50, 25))
        self.leser_right.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.leser_right.setStyleSheet("")
        self.leser_right.setObjectName("leser_right")
        self.gridLayout.addWidget(self.leser_right, 1, 2, 1, 1)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
                                            QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout.addItem(spacerItem5, 2, 0, 1, 1)
        self.laser_down = QtWidgets.QPushButton(parent=Laser_Control)
        self.laser_down.setMinimumSize(QtCore.QSize(50, 25))
        self.laser_down.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.laser_down.setStyleSheet("")
        self.laser_down.setObjectName("laser_down")
        self.gridLayout.addWidget(self.laser_down, 2, 1, 1, 1)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
                                            QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout.addItem(spacerItem6, 2, 2, 1, 1)
        self.gridLayout_5.addLayout(self.gridLayout, 3, 2, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.laser_forward = QtWidgets.QPushButton(parent=Laser_Control)
        self.laser_forward.setStyleSheet("")
        self.laser_forward.setObjectName("laser_forward")
        self.verticalLayout.addWidget(self.laser_forward)
        spacerItem7 = QtWidgets.QSpacerItem(17, 24, QtWidgets.QSizePolicy.Policy.Minimum,
                                            QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout.addItem(spacerItem7)
        self.laser_backward = QtWidgets.QPushButton(parent=Laser_Control)
        self.laser_backward.setStyleSheet("")
        self.laser_backward.setObjectName("laser_backward")
        self.verticalLayout.addWidget(self.laser_backward)
        self.gridLayout_5.addLayout(self.verticalLayout, 3, 3, 1, 2)
        # Home / Reference / Stop / Override column for the SmarAct stage.
        self._stage_button_layout = QtWidgets.QVBoxLayout()
        self.laser_home = QtWidgets.QPushButton(parent=Laser_Control)
        self.laser_home.setObjectName("laser_home")
        self._stage_button_layout.addWidget(self.laser_home)
        self.laser_stage_reference = QtWidgets.QPushButton("Reference", parent=Laser_Control)
        # Reference moves the stage on its own to find the physical reference
        # mark - dangerous if anything is in the way.  Gated behind Override
        # Access, same pattern as the gates / pumps GUIs.
        self.laser_stage_reference.setEnabled(False)
        self._stage_button_layout.addWidget(self.laser_stage_reference)
        self.laser_stage_stop = QtWidgets.QPushButton("STOP", parent=Laser_Control)
        self.laser_stage_stop.setStyleSheet(
	        "QPushButton{background: rgb(220,80,80); color: white; font-weight: bold;}"
        )
        self._stage_button_layout.addWidget(self.laser_stage_stop)
        self.laser_stage_superuser = QtWidgets.QPushButton("Override Access", parent=Laser_Control)
        self.laser_stage_superuser.setStyleSheet(
	        "QPushButton{background: rgb(193, 193, 193)}"
        )
        self._original_laser_stage_superuser_style = self.laser_stage_superuser.styleSheet()
        self._stage_button_layout.addWidget(self.laser_stage_superuser)
        self.gridLayout_5.addLayout(self._stage_button_layout, 3, 5, 1, 1)
        self.Error = QtWidgets.QLabel(parent=Laser_Control)
        self.Error.setMinimumSize(QtCore.QSize(500, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setStrikeOut(False)
        self.Error.setFont(font)
        self.Error.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.Error.setWordWrap(True)
        self.Error.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.LinksAccessibleByMouse)
        self.Error.setObjectName("Error")
        self.gridLayout_5.addWidget(self.Error, 4, 0, 1, 4)
        self.start_scanning = QtWidgets.QPushButton(parent=Laser_Control)
        self.start_scanning.setStyleSheet("QPushButton{background: rgb(193, 193, 193)}\n"
                                          "                                ")
        self.start_scanning.setObjectName("start_scanning")
        self.gridLayout_5.addWidget(self.start_scanning, 4, 6, 1, 1)
        self.nktpbus_mode_switch = QtWidgets.QPushButton(parent=Laser_Control)
        self.nktpbus_mode_switch.setStyleSheet("QPushButton{background: rgb(193, 193, 193)}\n"
                                               "                                ")
        self.nktpbus_mode_switch.setObjectName("nktpbus_mode_switch")
        # Switching to NKTPBus drops CLI control of the laser - gated behind
        # Override Access (same button as the stage Reference).
        self.nktpbus_mode_switch.setEnabled(False)
        self.gridLayout_5.addWidget(self.nktpbus_mode_switch, 4, 5, 1, 1)
        self.gridLayout_6.addLayout(self.gridLayout_5, 0, 0, 1, 1)

        self.retranslateUi(Laser_Control)
        QtCore.QMetaObject.connectSlotsByName(Laser_Control)
        Laser_Control.setTabOrder(self.laser_wavelegnth, self.laser_rate)
        Laser_Control.setTabOrder(self.laser_rate, self.laser_enable)
        Laser_Control.setTabOrder(self.laser_enable, self.laser_on)
        Laser_Control.setTabOrder(self.laser_on, self.laser_standby)
        Laser_Control.setTabOrder(self.laser_standby, self.laser_listen)
        Laser_Control.setTabOrder(self.laser_listen, self.laser_scan_mode5)
        Laser_Control.setTabOrder(self.laser_scan_mode5, self.laser_focus_mode)
        Laser_Control.setTabOrder(self.laser_focus_mode, self.laser_speed_x)
        Laser_Control.setTabOrder(self.laser_speed_x, self.laser_speed_y)
        Laser_Control.setTabOrder(self.laser_speed_y, self.laser_speed_z)
        Laser_Control.setTabOrder(self.laser_speed_z, self.laser_left)
        Laser_Control.setTabOrder(self.laser_left, self.laser_up)
        Laser_Control.setTabOrder(self.laser_up, self.leser_right)
        Laser_Control.setTabOrder(self.leser_right, self.laser_down)
        Laser_Control.setTabOrder(self.laser_down, self.laser_forward)
        Laser_Control.setTabOrder(self.laser_forward, self.laser_backward)
        Laser_Control.setTabOrder(self.laser_backward, self.laser_home)
        Laser_Control.setTabOrder(self.laser_home, self.start_scanning)
        Laser_Control.setTabOrder(self.start_scanning, self.scanning_disp)

        ######
        self.led_red = QPixmap('./files/led-red-on.png')
        self.led_green = QPixmap('./files/green-led-on.png')
        self.led_orange = QPixmap('./files/led-orange.png')
        self.led_laser_laser_standby.setPixmap(self.led_red)
        self.led_laser_on.setPixmap(self.led_red)
        self.led_laser_enable.setPixmap(self.led_red)
        self.led_laser_listen.setPixmap(self.led_red)

        self.laser_enable.setEnabled(False)
        self.laser_on.setEnabled(False)
        # self.laser_listen.clicked.connect(partial(self.start_task, self.laser_listen_clicked, self.laser_listen))
        # self.laser_standby.clicked.connect(partial(self.start_task, self.laser_standby_clicked, self.laser_standby))
        # self.laser_on.clicked.connect(partial(self.start_task, self.laser_on_clicked, self.laser_on))
        # self.laser_enable.clicked.connect(partial(self.start_task, self.laser_enable_clicked, self.laser_enable))

        self.listen_mode = False
        self.standby_mode = False
        self.on_mode = False
        self.enable_ouput_mode = False
        self.laser_listen.clicked.connect(self.laser_listen_clicked)
        self.laser_standby.clicked.connect(self.laser_standby_clicked)
        self.laser_on.clicked.connect(self.laser_on_clicked)
        self.laser_enable.clicked.connect(self.laser_enable_clicked)
        self.nktpbus_mode_switch.clicked.connect(self.switch_to_nktpbus_mode)

        self.laser_wavelegnth.currentIndexChanged.connect(self.laser_wavelegnth_changed)
        self.laser_power.valueChanged.connect(self.laser_power_changed)
        self.laser_rate.currentIndexChanged.connect(self.laser_rate_changed)
        self.laser_divition_factor.valueChanged.connect(self.laser_divition_factor_changed)
        self.laser_device = origamiClassCLI.origClass(self.conf['COM_PORT_laser'])

        self.variables.laser_pulse_energy = 0.0
        try:
            databack = self.laser_device.open_port()

            if databack == 0:
                self.laser_device.Listen()
                self.laser_device.wavelength_change(0)
                databack = self.laser_device.StatusRead()
                # reset the values to default
                self.laser_device.Power(float(self.laser_power.value()))
                self.laser_device.Freq(self.laser_rate.currentIndex() + 4)
                self.laser_repetion_rate_disp.display(400)
                self.variables.laser_freq = 400000
                self.laser_repetion_rate_disp.display(int(self.laser_rate.currentText()))
                self.laser_device.Div(float(self.laser_divition_factor.value()))
                if databack.strip() == 'ly_oxp2_dev_status 9':
                    self.led_laser_listen.setPixmap(self.led_green)
                else:
                    print("The laser status code is:", databack)
            else:
                error_text = self.laser_device.last_error or "no response from device"
                message = (
                    f"Laser is unavailable on {self.conf['COM_PORT_laser']}: {error_text}. "
                    f"Available serial ports: {_available_serial_ports_text()}."
                )
                print(message)
                self.error_message(message)
                self.laser_device = None
        except Exception as e:
            message = (
                f"Laser is unavailable on {self.conf['COM_PORT_laser']}: {e}. "
                f"Available serial ports: {_available_serial_ports_text()}."
            )
            print(message)
            self.error_message(message)
            self.laser_device = None

        self.worker = Worker(self.check_laser_status)
        self.worker.start()

        # ----- SmarAct laser focusing stage --------------------------------
        self.stage_device = None
        self._stage_poll_timer = None
        self.laser_speed_x.valueChanged.connect(lambda _v: self._update_stage_speed_label(self.laser_speed_x))
        self.laser_speed_y.valueChanged.connect(lambda _v: self._update_stage_speed_label(self.laser_speed_y))
        self.laser_speed_z.valueChanged.connect(lambda _v: self._update_stage_speed_label(self.laser_speed_z))
        self.laser_left.clicked.connect(lambda: self._stage_jog_axis(mcs2_stage.AXIS_X, -1))
        self.leser_right.clicked.connect(lambda: self._stage_jog_axis(mcs2_stage.AXIS_X, +1))
        self.laser_up.clicked.connect(lambda: self._stage_jog_axis(mcs2_stage.AXIS_Y, +1))
        self.laser_down.clicked.connect(lambda: self._stage_jog_axis(mcs2_stage.AXIS_Y, -1))
        self.laser_forward.clicked.connect(lambda: self._stage_jog_axis(mcs2_stage.AXIS_Z, +1))
        self.laser_backward.clicked.connect(lambda: self._stage_jog_axis(mcs2_stage.AXIS_Z, -1))
        self.laser_home.clicked.connect(self._stage_go_home)
        self.laser_stage_reference.clicked.connect(self._stage_reference)
        self.laser_stage_stop.clicked.connect(self._stage_stop)
        self.laser_stage_superuser.clicked.connect(self._stage_super_user_access)
        for sl in (self.laser_speed_x, self.laser_speed_y, self.laser_speed_z):
	        self._update_stage_speed_label(sl)
        self._connect_stage_device()

    # ------------------------------------------------------------------
    # SmarAct laser focusing stage
    # ------------------------------------------------------------------

    def _connect_stage_device(self):
	    if not self._stage_locator:
		    # Empty locator in config.toml means "no laser-side SmarAct
		    # controller in this rig" - skip silently, leave the panel
		    # disabled but don't bother the user with an error.
		    self._stage_connect_error = ""
		    self._set_stage_movement_enabled(False)
		    for sl_lbl in (self.laser_speed_x_label, self.laser_speed_y_label,
		                   self.laser_speed_z_label):
			    sl_lbl.setEnabled(False)
		    return
	    try:
		    self.stage_device = mcs2_stage.SmarActStage(self._stage_locator)
	    except mcs2_stage.SmarActStageError as exc:
		    self.stage_device = None
		    self._stage_connect_error = str(exc)
		    self.error_message(self._stage_connect_error)
		    self._set_stage_movement_enabled(False)
		    return
	    self._set_stage_movement_enabled(True)
	    self._stage_poll_timer = QtCore.QTimer()
	    self._stage_poll_timer.setInterval(500)
	    self._stage_poll_timer.timeout.connect(self._refresh_stage_position)
	    self._stage_poll_timer.start()
	    self._refresh_stage_position()

    def _set_stage_movement_enabled(self, enabled):
	    for btn in (self.laser_up, self.laser_down, self.laser_left,
	                self.leser_right, self.laser_forward, self.laser_backward,
	                self.laser_home):
		    btn.setEnabled(enabled)
	    # Reference stays gated behind Override Access (and also requires
	    # the device to be connected).
	    self.laser_stage_reference.setEnabled(enabled and self.flag_super_user_stage)
	    # STOP stays clickable so the user can always abort.

    def _stage_super_user_access(self):
	    """Toggle Override Access for the laser GUI's gated controls.

		Currently gates two operations:
		  * Stage Reference button (moves all axes on its own to find the
			physical reference mark - dangerous if anything is in the way)
		  * Nktpbus mode switch (drops CLI control of the laser, requires
			re-opening it from the NKT control software to come back)
		"""
	    if not self.flag_super_user_stage:
		    warning = QtWidgets.QMessageBox(parent=self.laser_stage_superuser)
		    warning.setIcon(QtWidgets.QMessageBox.Icon.Warning)
		    warning.setWindowTitle("Confirm Access Override")
		    warning.setText(
			    "Override Access enables two potentially disruptive controls:\n"
			    "  - Stage Reference (moves all axes on their own)\n"
			    "  - Nktpbus mode (hands the laser over to NKT control software)"
		    )
		    warning.setInformativeText(
			    "Make sure nothing is in the way of the laser stage and you really want to switch laser modes. Continue?")
		    warning.setStandardButtons(
			    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
		    )
		    warning.setDefaultButton(QtWidgets.QMessageBox.StandardButton.No)
		    if warning.exec() != QtWidgets.QMessageBox.StandardButton.Yes:
			    self.error_message("Override Access canceled.")
			    return
		    self.flag_super_user_stage = True
		    self.laser_stage_superuser.setStyleSheet("QPushButton{background: rgb(0, 255, 26)}")
		    self.error_message("!!! Override Access Granted !!!")
	    else:
		    self.flag_super_user_stage = False
		    self.laser_stage_superuser.setStyleSheet(self._original_laser_stage_superuser_style)
		    self.error_message("!!! Override Access deactivated !!!")
	    self.laser_stage_reference.setEnabled(
		    self.flag_super_user_stage and self.stage_device is not None
	    )
	    self.nktpbus_mode_switch.setEnabled(self.flag_super_user_stage)

    def _axis_velocity_m_s(self, axis):
	    slider = (self.laser_speed_x, self.laser_speed_y, self.laser_speed_z)[axis]
	    return mcs2_stage.speed_level_to_m_s(
		    slider.value(), self._speed_max_level, self._speed_max_mm_s,
		    table=self._speed_table,
	    )

    def _update_stage_speed_label(self, slider):
	    level = slider.value()
	    v_m_s = mcs2_stage.speed_level_to_m_s(
		    level, self._speed_max_level, self._speed_max_mm_s,
		    table=self._speed_table,
	    )
	    step_m = mcs2_stage.click_step_m(v_m_s, self._click_duration_s)
	    step_um = step_m * 1e6
	    step_text = f"{step_um:.0f}" if step_um >= 10 else f"{step_um:.2f}"
	    text = f"L{level}  {v_m_s * 1000:.3f} mm/s, step {step_text} µm"
	    mapping = {
		    self.laser_speed_x: self.laser_speed_x_label,
		    self.laser_speed_y: self.laser_speed_y_label,
		    self.laser_speed_z: self.laser_speed_z_label,
	    }
	    mapping[slider].setText(text)

    def _stage_jog_axis(self, axis, sign):
	    if self.stage_device is None:
		    self.error_message(self._stage_connect_error or "Laser stage not connected.")
		    return
	    vel = self._axis_velocity_m_s(axis)
	    step_m = mcs2_stage.click_step_m(vel, self._click_duration_s)
	    try:
		    self.stage_device.move_relative_axis(
			    axis=axis, delta_m=sign * step_m, velocity_m_s=vel, wait=False,
		    )
	    except mcs2_stage.SmarActStageError as exc:
		    self.error_message(f"Move failed: {exc}")

    def _stage_go_home(self):
	    if self.stage_device is None:
		    self.error_message(self._stage_connect_error or "Laser stage not connected.")
		    return
	    x_m, y_m, z_m = self._home_target_m
	    try:
		    self.stage_device.move_absolute(
			    x_m=x_m, y_m=y_m, z_m=z_m,
			    velocity_m_s=self._axis_velocity_m_s(mcs2_stage.AXIS_X),
			    wait=False,
		    )
	    except mcs2_stage.SmarActStageError as exc:
		    self.error_message(f"Home failed: {exc}")

    def _stage_reference(self):
	    if self.stage_device is None:
		    self.error_message(self._stage_connect_error or "Laser stage not connected.")
		    return
	    try:
		    self.stage_device.find_reference()
	    except mcs2_stage.SmarActStageError as exc:
		    self.error_message(f"Reference search failed: {exc}")

    def _stage_stop(self):
	    if self.stage_device is None:
		    return
	    self.stage_device.stop()

    def _refresh_stage_position(self):
	    if self.stage_device is None:
		    return
	    try:
		    pos = self.stage_device.get_position()
	    except mcs2_stage.SmarActStageError as exc:
		    self.error_message(f"Position read failed: {exc}")
		    return
	    self._set_stage_axis(pos['x'], self.laser_x_mm, self.laser_x_um,
	                         self.laser_x_nm, self.laser_x_cord)
	    self._set_stage_axis(pos['y'], self.laser_y_mm, self.laser_y_um,
	                         self.laser_y_nm, self.laser_y_cord)
	    self._set_stage_axis(pos['z'], self.laser_z_mm, self.laser_z_um,
	                         self.laser_z_nm, self.laser_z_cord)

    @staticmethod
    def _set_stage_axis(value_m, mm_lcd, um_lcd, nm_lcd, single_lcd):
	    mm, um, nm = mcs2_stage.split_meters_mm_um_nm(value_m)
	    mm_lcd.display(mm)
	    um_lcd.display(um)
	    nm_lcd.display(nm)
	    single_lcd.display(value_m * 1e6)  # legacy: micrometers

    def retranslateUi(self, Laser_Control):
        _translate = QtCore.QCoreApplication.translate
        ###
        # Laser_Control.setWindowTitle(_translate("Laser_Control", "Form"))
        Laser_Control.setWindowTitle(_translate("Laser_Control", "PyCCAPT Laser Control"))
        Laser_Control.setWindowIcon(QtGui.QIcon('./files/logo.png'))
        ###
        Laser_Control.setToolTip(_translate("Laser_Control", "<html><head/><body><p>1</p></body></html>"))
        self.laser_wavelegnth.setItemText(0, _translate("Laser_Control", "IR"))
        self.laser_wavelegnth.setItemText(1, _translate("Laser_Control", "Green"))
        self.laser_wavelegnth.setItemText(2, _translate("Laser_Control", "DUV"))
        self.led_laser_on.setText(_translate("Laser_Control", "Laser on"))
        self.laser_rate.setItemText(0, _translate("Laser_Control", "400000"))
        self.laser_rate.setItemText(1, _translate("Laser_Control", "500000"))
        self.laser_rate.setItemText(2, _translate("Laser_Control", "579710"))
        self.laser_rate.setItemText(3, _translate("Laser_Control", "720720"))
        self.laser_rate.setItemText(4, _translate("Laser_Control", "800000"))
        self.laser_rate.setItemText(5, _translate("Laser_Control", "898876"))
        self.laser_rate.setItemText(6, _translate("Laser_Control", "1000000"))
        self.led_laser_enable.setText(_translate("Laser_Control", "Output enable"))
        self.laser_standby.setText(_translate("Laser_Control", "Standby"))
        self.label_2.setText(_translate("Laser_Control", "Repetion rate (Hz)"))
        self.laser_on.setText(_translate("Laser_Control", "Laser on"))
        self.label_3.setText(_translate("Laser_Control", "Divition Factor"))
        self.laser_enable.setText(_translate("Laser_Control", "Output Enable"))
        self.led_laser_listen.setText(_translate("Laser_Control", "Listen"))
        self.led_laser_laser_standby.setText(_translate("Laser_Control", "Standby"))
        self.label.setText(_translate("Laser_Control", "Power control (mW)"))
        self.laser_listen.setText(_translate("Laser_Control", "Listen"))
        self.label_4.setText(_translate("Laser_Control", "Wavelength"))
        self.label_12.setText(_translate("Laser_Control", "Scan mode"))
        self.laser_scan_mode5.setItemText(0, _translate("Laser_Control", "Standard"))
        self.label_13.setText(_translate("Laser_Control", "Focus mode"))
        self.laser_focus_mode.setItemText(0, _translate("Laser_Control", "Standard"))
        self.label_9.setText(_translate("Laser_Control", "Laser power (mW)"))
        self.label_10.setText(_translate("Laser_Control", "Pulse energy (nJ)"))
        self.label_11.setText(_translate("Laser_Control", "Frequency (KHz)"))
        self.label_19.setText(_translate("Laser_Control", "x"))
        self.label_17.setText(_translate("Laser_Control", "y"))
        self.label_18.setText(_translate("Laser_Control", "z"))
        self.label_14.setText(_translate("Laser_Control", "Speed"))
        self.label_15.setText(_translate("Laser_Control", "X"))
        self.label_16.setText(_translate("Laser_Control", "Y"))
        self.label_speed_z.setText(_translate("Laser_Control", "Z"))
        self.laser_stage_reference.setText(_translate("Laser_Control", "Reference"))
        self.laser_stage_stop.setText(_translate("Laser_Control", "STOP"))
        self.laser_up.setText(_translate("Laser_Control", "up"))
        self.laser_left.setText(_translate("Laser_Control", "Left"))
        self.leser_right.setText(_translate("Laser_Control", "Right"))
        self.laser_down.setText(_translate("Laser_Control", "Down"))
        self.laser_forward.setText(_translate("Laser_Control", "Forward"))
        self.laser_backward.setText(_translate("Laser_Control", "Backward"))
        self.laser_home.setText(_translate("Laser_Control", "Home"))
        self.Error.setText(_translate("Laser_Control", "<html><head/><body><p><br/></p></body></html>"))
        self.start_scanning.setText(_translate("Laser_Control", "Start scaning"))
        self.nktpbus_mode_switch.setText(_translate("Laser_Control", "Nktpbus mode"))

        ####
        self.pattern_number = r'\b\d+\b'
        self.timer_hide_error = QtCore.QTimer()
        self.timer_hide_error.timeout.connect(self.hideMessage)
        self.laser_power.setMinimum(0.0)
        self.laser_power.setMaximum(self.conf['max_laser_power'])
        self.laser_power.setSingleStep(0.1)
        self.laser_divition_factor.setMinimum(1)
        self.laser_divition_factor.setMaximum(1000000)

    def laser_enable_clicked(self):
        """
            Handle the close event of the GatesWindow.

            Args:
                None
            Return:
                None
            """
        self.enable_ouput_mode = True

    def laser_on_clicked(self):
        """
            Handle the close event of the GatesWindow.

            Args:
                None
            Return:
                None
            """
        self.on_mode = True

    def laser_standby_clicked(self):
        """
            Handle the close event of the GatesWindow.

            Args:
                None
            Return:
                None
            """
        self.standby_mode = True

    def laser_listen_clicked(self):
        """
            Handle the close event of the GatesWindow.

            Args:
                None
            Return:
                None
            """
        self.listen_mode = True

    def laser_wavelegnth_changed(self):
        """
            Handle the close event of the GatesWindow.

            Args:
                None
            Return:
                None
            """
        self.change_laser_wavelegnth = True

    def laser_power_changed(self):
        """
            Handle the close event of the GatesWindow.

            Args:
                None
            Return:
                None
            """
        self.change_laser_power = True

    def laser_rate_changed(self):
        """
            Handle the close event of the GatesWindow.

            Args:
                None
            Return:
                None
            """
        self.change_laser_rate = True

    def laser_divition_factor_changed(self):
        """
            Handle the close event of the GatesWindow.

            Args:
                None
            Return:
                None
            """
        self.change_laser_divition_factor = True

    def get_frequency(self, index):
        """
            Handle the close event of the changing of laser rate.

        Args:
            None

        Return:
            None
        """
        repetition_rates = {
            4: 400000,
            5: 500000,
            6: 579710,
            7: 720720,
            8: 800000,
            9: 898876,
            10: 1000000
        }
        return repetition_rates.get(index, "Invalid index")

    def check_laser_status(self):
        if self.laser_device is not None:
            databack = self.laser_device.StatusRead()
            if self.listen_mode:
                if databack.strip() != 'ly_oxp2_dev_status 9':
                    self.laser_listen.setEnabled(False)
                    databack = self.laser_device.Listen()
                elif databack.strip() == 'ly_oxp2_dev_status 9':
                    self.laser_device.AOM(0)
                    self.led_laser_listen.setPixmap(self.led_green)
                    self.led_laser_enable.setPixmap(self.led_red)
                    self.led_laser_on.setPixmap(self.led_red)
                    self.led_laser_laser_standby.setPixmap(self.led_red)
                    self.laser_enable.setEnabled(False)
                    self.laser_on.setEnabled(False)
                    self.on_mode = False
                    self.enable_ouput_mode = False
                    self.standby_mode = False
                    self.listen_mode = False
                    self.laser_listen.setEnabled(True)
                    self.laser_standby.setEnabled(True)
                    self.laser_wavelegnth.setEnabled(True)

            elif self.standby_mode:
                if databack.strip() != 'ly_oxp2_dev_status 33':
                    if self.laser_standby.isEnabled():
                        self.laser_standby.setEnabled(False)
                        self.laser_wavelegnth.setEnabled(True)
                        self.laser_on.setEnabled(False)
                        self.led_laser_listen.setPixmap(self.led_orange)
                        self.led_laser_laser_standby.setPixmap(self.led_orange)
                        self.laser_device.Standby()
                    else:
                        if self.led_laser_laser_standby.pixmap().toImage() == self.led_orange.toImage():
                            self.led_laser_laser_standby.setPixmap(self.led_green)
                        elif self.led_laser_laser_standby.pixmap().toImage() == self.led_green.toImage():
                            self.led_laser_laser_standby.setPixmap(self.led_orange)
                elif databack.strip() == 'ly_oxp2_dev_status 33':
                    self.laser_device.AOM(0)
                    self.laser_on.setEnabled(True)
                    self.laser_standby.setEnabled(True)
                    self.led_laser_on.setPixmap(self.led_red)
                    self.led_laser_laser_standby.setPixmap(self.led_green)
                    self.led_laser_enable.setPixmap(self.led_red)
                    self.laser_enable.setEnabled(False)
                    self.standby_mode = False
            elif self.on_mode:
                if databack.strip() == 'ly_oxp2_dev_status 33':
                    if self.laser_on.isEnabled():
                        self.laser_on.setEnabled(False)
                        self.laser_wavelegnth.setEnabled(False)
                        self.led_laser_on.setPixmap(self.led_orange)
                        self.led_laser_laser_standby.setPixmap(self.led_orange)
                        self.laser_device.Enable()
                elif databack.strip() == 'ly_oxp2_dev_status 129':
                    self.laser_on.setEnabled(True)
                    self.led_laser_on.setPixmap(self.led_green)
                    self.led_laser_laser_standby.setPixmap(self.led_orange)
                    self.led_laser_enable.setPixmap(self.led_green)
                    self.laser_enable.setEnabled(True)
                    self.laser_device.AOM(4000)  # 4000 means AMO fully opeen
                    self.laser_device.AOMEnable()
                    self.on_mode = False
                elif databack.strip() == 'ly_oxp2_dev_status 1':
                    if self.led_laser_on.pixmap().toImage() == self.led_orange.toImage():
                        self.led_laser_on.setPixmap(self.led_green)
                    elif self.led_laser_on.pixmap().toImage() == self.led_green.toImage():
                        self.led_laser_on.setPixmap(self.led_orange)
                else:
                    self.on_mode = False
            elif self.enable_ouput_mode:
                self.laser_enable.setEnabled(False)
                if databack.strip() == 'ly_oxp2_dev_status 65':
                    self.laser_device.AOMEnable()
                    self.laser_device.AOM(4000)  # 4000 means AMO fully opeen
                    self.enable_ouput_mode = False
                    self.led_laser_enable.setPixmap(self.led_green)
                    self.laser_enable.setEnabled(True)
                elif databack.strip() == 'ly_oxp2_dev_status 129':
                    self.laser_device.AOMDisable()
                    self.laser_device.AOM(0)
                    self.enable_ouput_mode = False
                    self.led_laser_enable.setPixmap(self.led_red)
                    self.laser_enable.setEnabled(True)
            if self.change_laser_wavelegnth:
                # if emission is on we cannot change the wavelength
                if databack != 'ly_oxp2_dev_status 129':
                    self.laser_wavelegnth.setEnabled(False)
                    if self.laser_wavelegnth.currentText() == "IR":
                        dd = self.laser_device.wavelength_change(0)
                    elif self.laser_wavelegnth.currentText() == "Green":
                        dd = self.laser_device.wavelength_change(1)
                    elif self.laser_wavelegnth.currentText() == "DUV":
                        dd = self.laser_device.wavelength_change(3)
                    self.laser_wavelegnth.setEnabled(True)
                else:
                    print('The laser is on, you can not change the wavelength')
                self.change_laser_wavelegnth = False

            if self.change_laser_power:
                # only if the laser is on we can change the power
                # if databack.strip() == 'ly_oxp2_dev_status 129':
                self.laser_power.setEnabled(False)
                self.laser_device.Power(float(self.laser_power.value()))
                if databack.strip() == 'ly_oxp2_dev_status 129':
                    self.laser_device.AOM(4000)  # 4000 means AMO fully opeen
                else:
                    self.laser_device.AOM(0)

                # Pulse energy in nJ
                power_pe = self.laser_device.PowerRead()
                power_pe = re.search(r'[-+]?\d*\.\d+|\d+', power_pe)
                if power_pe:
                    power = float(power_pe.group())
                else:
                    power = 'Nan'
                self.laser_pulse_energy_disp.display(power)
                # update variables for laser power
                self.average_power = self.laser_device.read_average_power()
                self.variables.laser_power = float(self.laser_power.value())
                self.variables.laser_average_power = float(re.findall(self.pattern_number, self.average_power)[0])
                self.laser_power_disp.display(self.variables.laser_average_power)
                self.laser_power.setEnabled(True)
                # else:
                #     print('The laser is off, you can not change the power')
                self.change_laser_power = False

            if self.change_laser_rate:
                self.laser_rate.setEnabled(False)
                res = self.laser_device.Freq(self.laser_rate.currentIndex() + 4)
                # Repetition rate
                # At base frequencies above 100 kHz, the pulse energy linearly decreases.
                freq_o = self.laser_device.FreqRead()
                freq = re.search(r'[-+]?\d*\.\d+|\d+', freq_o)
                if freq:
                    freq = float(freq.group())
                else:
                    freq = 'Nan'
                if freq != 'Nan':
                    laser_rate = self.get_frequency(int(freq))
                    self.variables.laser_freq = laser_rate
                    self.laser_repetion_rate_disp.display(
                        (self.variables.laser_freq / 1000) / self.laser_divition_factor.value())
                else:
                    self.variables.laser_freq = 0
                    self.laser_repetion_rate_disp.display('Error')
                self.laser_rate.setEnabled(True)
                self.change_laser_rate = False

            if self.change_laser_divition_factor:
                self.laser_divition_factor.setEnabled(False)
                res = self.laser_device.Div(self.laser_divition_factor.value())
                self.variables.laser_division_factor = self.laser_divition_factor.value()
                print('dddddddddddddd', self.variables.laser_freq, self.laser_divition_factor.value())
                self.laser_repetion_rate_disp.display(
                    (self.variables.laser_freq / 1000) / self.laser_divition_factor.value())
                self.laser_divition_factor.setEnabled(True)
                self.change_laser_divition_factor = False

            if self.index == 5:
                res_error = self.laser_device.StatusMode()
                if "Error" in res_error:
                    self.listen_mode = True
                    self.error_message("Error:" + res_error)
        #
                print('==============================================')
                print('laser status is:', databack.strip())
                print("status mode is:", res_error)
                print("status is:", self.laser_device.StatusRead())
                print('Mode is', self.laser_device.ModeRead())  # 2: Internal power 3: External power 8: SPI power
                print('status LED is:', self.laser_device.status_led())
                print('wavelength is:', self.laser_device.wavelength_read())
                print("AMO status is:", self.laser_device.AOMState())
                print('pulse energy (mW):', self.laser_device.PowerRead())
                print('power W', self.laser_device.power_read_dv_green())
                print('avg power (mW):', self.laser_device.read_average_power())
                print('amo power:', self.laser_device.AOMRead())
                print('freq_o:', self.laser_device.FreqRead())
                print('Div:', self.laser_device.DivRead())
                # print("avaliable freq:", self.laser_device.freq_avaliable())
                print('----------------------------------------------')
                self.index = 0
            self.index += 1
            time.sleep(0.5)

    def switch_to_nktpbus_mode(self):
        """"
            Switch to NKTPBUS mode

            Args:
                None

            Return:
                None
            """
        if self.laser_device is not None:
            self.laser_device.InterbusEnable()
            self.laser_device.close_port()
            self.laser_device = None
            self.error_message("Switching to NKTPBUS mode. Back to CLImode with NKT control software")
        else:
            self.error_message("The laser is already in NKTPBUS mode or other connection error (check terminal)")

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

        self.timer_hide_error.start(8000)

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

        self.timer_hide_error.stop()


    def stop(self):
        """
            Handle the close event of the GatesWindow.

            Args:
                None

            Return:
                None
            """
        # Stop any background processes, timers, or threads here
        if getattr(self, '_stage_poll_timer', None) is not None:
	        self._stage_poll_timer.stop()
        if getattr(self, 'stage_device', None) is not None:
	        self.stage_device.close()
	        self.stage_device = None


class Worker(QThread):
    def __init__(self, task_function):
        super().__init__()
        self.task_function = task_function

    def run(self):
        while True:  # Run indefinitely
            self.task_function()
            self.msleep(1000)  # Sleep for 1000 milliseconds (1 second)


class LaserControlWindow(QtWidgets.QWidget):
    closed = QtCore.pyqtSignal()  # Define a custom closed signal

    def __init__(self, gui_laser_control, *args, **kwargs):
        """
        Initialize the LaserControlWindow class.

        Args:
            gui_laser_control: GUI for laser control.
            *args, **kwargs: Additional arguments for QWidget initialization.
        """
        super().__init__(*args, **kwargs)
        self.gui_laser_control = gui_laser_control

    def closeEvent(self, event):
        """
        Handle the close event of the LaserControlWindow.

        Args:
            event: Close event.
        """
        event.ignore()
        self.hide()
        self.closed.emit()

    def setWindowStyleFusion(self):
        # Set the Fusion style
        QtWidgets.QApplication.setStyle("Fusion")


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
    Laser_Control = QtWidgets.QWidget()
    ui = Ui_Laser_Control(shared.variables, conf)
    ui.setupUi(Laser_Control)
    Laser_Control.show()
    sys.exit(app.exec())

