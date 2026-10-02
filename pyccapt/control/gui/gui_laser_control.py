import logging
import re
import sys
import threading
import time

import serial.tools.list_ports
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QThread
from PyQt6.QtGui import QPixmap

# Local module and scripts
from pyccapt.control.gui.responsive import make_window_responsive
from pyccapt.control.core import runtime
from pyccapt.control.gui import tooltips
from pyccapt.control.gui.stage_control_widgets import (
    JOG_GROUP_STYLE,
    SpeedSelector,
    make_jog_button,
)
from pyccapt.control.nkt_photonics import nktpbus_switch, origamiClassCLI, readback
from pyccapt.control.gui.laser_readouts import LaserReadoutMixin
from pyccapt.control.gui.laser_alignment_gui import LaserAlignmentGuiMixin
from pyccapt.control.gui.laser_layout import LaserLayoutMixin
from pyccapt.control.smaract_mcs2 import mcs2_stage


class _LaserStageReferenceWorker(QtCore.QThread):
    """Background thread for the laser-stage Reference search.

    Mirrors the worker in gui_stage_control.py so STOP can interrupt the
    search instead of being blocked by the synchronous _wait loop.
    """

    finished_with_error = QtCore.pyqtSignal(str)
    finished_ok = QtCore.pyqtSignal()

    def __init__(self, stage_device, cancel_event, referencing_options, timeout_s, velocity_m_s):
        super().__init__()
        self.stage_device = stage_device
        self.cancel_event = cancel_event
        self.referencing_options = referencing_options
        self.timeout_s = timeout_s
        self.velocity_m_s = velocity_m_s

    def run(self):
        try:
            self.stage_device.find_reference(
                timeout_s=self.timeout_s,
                cancel_event=self.cancel_event,
                referencing_options=self.referencing_options,
                velocity_m_s=self.velocity_m_s,
            )
        except mcs2_stage.SmarActStageError as exc:
            self.finished_with_error.emit(str(exc))
            return
        self.finished_ok.emit()


def _available_serial_ports_text():
    ports = sorted(port.device for port in serial.tools.list_ports.comports() if getattr(port, "device", ""))
    return ", ".join(ports) if ports else "none detected"


# Compatibility alias; parsing excludes digits embedded in command names.
_parse_first_number = readback.scalar


class Ui_Laser_Control(LaserReadoutMixin, LaserAlignmentGuiMixin, LaserLayoutMixin):
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
        Laser_Control.resize(980, 650)
        self.gridLayout_6 = QtWidgets.QGridLayout(Laser_Control)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        # Nominal harmonic wavelength from the accepted device readback.
        # This is not the historical factory-measured spectrum of SN4906.
        wavelength_layout = QtWidgets.QHBoxLayout()
        self.laser_wavelegnth = QtWidgets.QComboBox(parent=Laser_Control)
        self.laser_wavelegnth.setStyleSheet("QComboBox{background: rgb(223,223,233)}")
        self.laser_wavelegnth.setObjectName("laser_wavelegnth")
        self.laser_wavelegnth.addItem("")
        self.laser_wavelegnth.addItem("")
        self.laser_wavelegnth.addItem("")
        wavelength_layout.addWidget(self.laser_wavelegnth)
        self.laser_wavelegnth_nm_label = QtWidgets.QLabel(parent=Laser_Control)
        self.laser_wavelegnth_nm_label.setObjectName("laser_wavelegnth_nm_label")
        wavelength_label_font = QtGui.QFont()
        wavelength_label_font.setItalic(True)
        self.laser_wavelegnth_nm_label.setFont(wavelength_label_font)
        self.laser_wavelegnth_nm_label.setMinimumWidth(70)
        wavelength_layout.addWidget(self.laser_wavelegnth_nm_label)
        self.gridLayout_3.addLayout(wavelength_layout, 0, 1, 1, 1)
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
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.scanning_disp.sizePolicy().hasHeightForWidth())
        self.scanning_disp.setSizePolicy(sizePolicy)
        self.scanning_disp.setMinimumSize(QtCore.QSize(250, 250))
        self.scanning_disp.setStyleSheet(
            "QWidget{\n"
            "                                    border: 0.5px solid gray;\n"
            "                                    }\n"
            "                                "
        )
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
        spacerItem = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum
        )
        self.horizontalLayout.addItem(spacerItem)
        self.label_9 = QtWidgets.QLabel(parent=Laser_Control)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout.addWidget(self.label_9)
        self.laser_power_disp = QtWidgets.QLCDNumber(parent=Laser_Control)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.laser_power_disp.sizePolicy().hasHeightForWidth())
        self.laser_power_disp.setSizePolicy(sizePolicy)
        self.laser_power_disp.setMinimumSize(QtCore.QSize(100, 50))
        self.laser_power_disp.setMaximumSize(QtCore.QSize(100, 50))
        # 6 digits so values like "5.028" fit comfortably (5 chars + a
        # spare position for trailing decimals or growth headroom).
        self.laser_power_disp.setDigitCount(6)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.laser_power_disp.setFont(font)
        self.laser_power_disp.setStyleSheet(
            "QLCDNumber{\n"
            "                                            border: 2px solid green;\n"
            "                                            border-radius: 10px;\n"
            "                                            padding: 0 8px;\n"
            "                                            }\n"
            "                                        "
        )
        self.laser_power_disp.setObjectName("laser_power_disp")
        self.horizontalLayout.addWidget(self.laser_power_disp)
        self.label_10 = QtWidgets.QLabel(parent=Laser_Control)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout.addWidget(self.label_10)
        self.laser_pulse_energy_disp = QtWidgets.QLCDNumber(parent=Laser_Control)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.laser_pulse_energy_disp.sizePolicy().hasHeightForWidth())
        self.laser_pulse_energy_disp.setSizePolicy(sizePolicy)
        self.laser_pulse_energy_disp.setMinimumSize(QtCore.QSize(100, 50))
        self.laser_pulse_energy_disp.setMaximumSize(QtCore.QSize(100, 50))
        # Same headroom as laser_power_disp -- "12.570" needs 6 digits.
        self.laser_pulse_energy_disp.setDigitCount(6)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.laser_pulse_energy_disp.setFont(font)
        self.laser_pulse_energy_disp.setStyleSheet(
            "QLCDNumber{\n"
            "                                            border: 2px solid green;\n"
            "                                            border-radius: 10px;\n"
            "                                            padding: 0 8px;\n"
            "                                            }\n"
            "                                        "
        )
        self.laser_pulse_energy_disp.setObjectName("laser_pulse_energy_disp")
        self.horizontalLayout.addWidget(self.laser_pulse_energy_disp)
        self.label_11 = QtWidgets.QLabel(parent=Laser_Control)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.horizontalLayout.addWidget(self.label_11)
        self.laser_repetion_rate_disp = QtWidgets.QLCDNumber(parent=Laser_Control)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.laser_repetion_rate_disp.sizePolicy().hasHeightForWidth())
        self.laser_repetion_rate_disp.setSizePolicy(sizePolicy)
        self.laser_repetion_rate_disp.setMinimumSize(QtCore.QSize(100, 50))
        self.laser_repetion_rate_disp.setMaximumSize(QtCore.QSize(100, 50))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.laser_repetion_rate_disp.setFont(font)
        self.laser_repetion_rate_disp.setStyleSheet(
            "QLCDNumber{\n"
            "                                            border: 2px solid green;\n"
            "                                            border-radius: 10px;\n"
            "                                            padding: 0 8px;\n"
            "                                            }\n"
            "                                        "
        )
        self.laser_repetion_rate_disp.setObjectName("laser_repetion_rate_disp")
        self.horizontalLayout.addWidget(self.laser_repetion_rate_disp)
        spacerItem1 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum
        )
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
        # Exact speed presets (Simple-Mode-style 1..N) and jog distance
        # ------------------------------------------------------------------
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")

        self._speed_max_mm_s = float(self.conf.get('stage_speed_max_mm_s', 1.0))
        self._speed_max_level = int(self.conf.get('stage_speed_level_max', 11))
        self._speed_min_level = int(self.conf.get('stage_speed_level_min', 1))
        self._speed_default = int(self.conf.get('stage_speed_level_default', 3))
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
        self._referencing_options = int(
            self.conf.get('stage_referencing_options', mcs2_stage.SmarActStage.REFERENCING_OPTIONS_DEFAULT)
        )
        self._reference_timeout_s = float(self.conf.get('stage_reference_timeout_s', 120))
        self._reference_velocity_m_s = float(self.conf.get('stage_reference_velocity_mm_s', 5.0)) * 1e-3
        self._home_velocity_m_s = float(self.conf.get('stage_home_velocity_mm_s', 1.0)) * 1e-3

        # Header
        self.label_14 = QtWidgets.QLabel("Speed preset", parent=Laser_Control)
        self.label_14.setFont(bold_font)
        self.label_14.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.gridLayout_2.addWidget(self.label_14, 0, 1, 1, 1)
        self.laser_jog_header = QtWidgets.QLabel(
            f"Jog / {self._click_duration_s:g} s", parent=Laser_Control
        )
        self.laser_jog_header.setFont(bold_font)
        self.laser_jog_header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.gridLayout_2.addWidget(self.laser_jog_header, 0, 2, 1, 1)

        # Per-axis labels
        self.label_15 = QtWidgets.QLabel("X", parent=Laser_Control)
        self.label_15.setFont(bold_font)
        self.label_16 = QtWidgets.QLabel("Y", parent=Laser_Control)
        self.label_16.setFont(bold_font)
        self.label_speed_z = QtWidgets.QLabel("Z", parent=Laser_Control)
        self.label_speed_z.setFont(bold_font)

        def _make_speed_selector():
            return SpeedSelector(
                Laser_Control,
                self._speed_min_level,
                self._speed_max_level,
                self._speed_default,
                self._speed_max_mm_s,
                self._speed_table,
            )

        self.laser_speed_x = _make_speed_selector()
        self.laser_speed_y = _make_speed_selector()
        self.laser_speed_z = _make_speed_selector()

        self.laser_speed_x_label = QtWidgets.QLabel(parent=Laser_Control)
        self.laser_speed_x_label.setMinimumWidth(85)
        self.laser_speed_y_label = QtWidgets.QLabel(parent=Laser_Control)
        self.laser_speed_y_label.setMinimumWidth(85)
        self.laser_speed_z_label = QtWidgets.QLabel(parent=Laser_Control)
        self.laser_speed_z_label.setMinimumWidth(85)

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
        # Standard three-axis layout matching the specimen-stage GUI.
        self.laser_xy_jog_group = QtWidgets.QGroupBox("X / Y Jog", parent=Laser_Control)
        self.laser_xy_jog_group.setStyleSheet(JOG_GROUP_STYLE)
        self.gridLayout = QtWidgets.QGridLayout(self.laser_xy_jog_group)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout.setContentsMargins(7, 11, 7, 7)
        self.gridLayout.setHorizontalSpacing(4)
        self.gridLayout.setVerticalSpacing(4)

        self.laser_up = make_jog_button(self.laser_xy_jog_group, "Y+\n▲")
        self.laser_up.setObjectName("laser_up")
        self.laser_left = make_jog_button(self.laser_xy_jog_group, "◀  X−")
        self.laser_left.setObjectName("laser_left")
        self.leser_right = make_jog_button(self.laser_xy_jog_group, "X+  ▶")
        self.leser_right.setObjectName("leser_right")
        # Correctly-spelled alias for new callers; legacy name remains valid.
        self.laser_right = self.leser_right
        self.laser_down = make_jog_button(self.laser_xy_jog_group, "▼\nY−")
        self.laser_down.setObjectName("laser_down")
        laser_xy_center = QtWidgets.QLabel("X / Y", parent=self.laser_xy_jog_group)
        laser_xy_center.setFixedSize(QtCore.QSize(58, 42))
        laser_xy_center.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        laser_xy_center.setStyleSheet(
            "QLabel{background:#f5f7f9;color:#526b7c;border:1px solid #c4ced6;"
            "border-radius:7px;font-weight:bold;}"
        )

        self.gridLayout.addWidget(self.laser_up, 0, 1)
        self.gridLayout.addWidget(self.laser_left, 1, 0)
        self.gridLayout.addWidget(laser_xy_center, 1, 1)
        self.gridLayout.addWidget(self.leser_right, 1, 2)
        self.gridLayout.addWidget(self.laser_down, 2, 1)
        self.gridLayout_5.addWidget(self.laser_xy_jog_group, 3, 2, 1, 1)

        self.laser_z_jog_group = QtWidgets.QGroupBox("Z Jog", parent=Laser_Control)
        self.laser_z_jog_group.setStyleSheet(JOG_GROUP_STYLE)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.laser_z_jog_group)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout.setContentsMargins(7, 11, 7, 7)
        self.verticalLayout.setSpacing(6)
        self.laser_forward = make_jog_button(self.laser_z_jog_group, "Z+\nForward", width=76)
        self.laser_forward.setObjectName("laser_forward")
        self.verticalLayout.addWidget(self.laser_forward)
        self.verticalLayout.addStretch(1)
        self.laser_backward = make_jog_button(self.laser_z_jog_group, "Z−\nBackward", width=76)
        self.laser_backward.setObjectName("laser_backward")
        self.verticalLayout.addWidget(self.laser_backward)
        self.gridLayout_5.addWidget(self.laser_z_jog_group, 3, 3, 1, 2)
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
        self.laser_stage_stop.setStyleSheet("QPushButton{background: rgb(220,80,80); color: white; font-weight: bold;}")
        self._stage_button_layout.addWidget(self.laser_stage_stop)
        self.laser_stage_superuser = QtWidgets.QPushButton("Override Access", parent=Laser_Control)
        self.laser_stage_superuser.setStyleSheet("QPushButton{background: rgb(193, 193, 193)}")
        self._original_laser_stage_superuser_style = self.laser_stage_superuser.styleSheet()
        self._stage_button_layout.addWidget(self.laser_stage_superuser)
        self.gridLayout_5.addLayout(self._stage_button_layout, 3, 5, 1, 1)
        # Persistent connection-state banner. Different from the Error
        # label below (which auto-hides after 8 s). This one stays visible
        # for as long as the laser is not reachable on CLI, and is
        # cleared automatically once a CLI session is open.
        self.laser_connection_banner = QtWidgets.QLabel(parent=Laser_Control)
        self.laser_connection_banner.setMinimumSize(QtCore.QSize(500, 24))
        banner_font = QtGui.QFont()
        banner_font.setPointSize(10)
        banner_font.setBold(True)
        self.laser_connection_banner.setFont(banner_font)
        self.laser_connection_banner.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.laser_connection_banner.setWordWrap(True)
        self.laser_connection_banner.setObjectName("laser_connection_banner")
        self.laser_connection_banner.setVisible(False)
        self.gridLayout_5.addWidget(self.laser_connection_banner, 5, 0, 1, 7)

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
        self.start_scanning.setStyleSheet("QPushButton{background: rgb(193, 193, 193)}\n                                ")
        self.start_scanning.setObjectName("start_scanning")
        self.gridLayout_5.addWidget(self.start_scanning, 4, 6, 1, 1)
        # Mode-switch buttons: NKTPBus <-> CLI. Both gated behind Override
        # Access. They sit side by side at the bottom-right because they
        # are closely related: one is the inverse of the other.
        mode_switch_layout = QtWidgets.QHBoxLayout()
        self.switch_to_cli_button = QtWidgets.QPushButton(parent=Laser_Control)
        self.switch_to_cli_button.setStyleSheet("QPushButton{background: rgb(193, 193, 193)}")
        self.switch_to_cli_button.setObjectName("switch_to_cli_button")
        # CLI is the "we want this for normal operation" direction. Same
        # gating as Nktpbus mode -- Override Access required.
        self.switch_to_cli_button.setEnabled(False)
        mode_switch_layout.addWidget(self.switch_to_cli_button)

        self.nktpbus_mode_switch = QtWidgets.QPushButton(parent=Laser_Control)
        self.nktpbus_mode_switch.setStyleSheet("QPushButton{background: rgb(193, 193, 193)}")
        self.nktpbus_mode_switch.setObjectName("nktpbus_mode_switch")
        # Switching to NKTPBus drops CLI control of the laser - gated behind
        # Override Access (same button as the stage Reference).
        self.nktpbus_mode_switch.setEnabled(False)
        mode_switch_layout.addWidget(self.nktpbus_mode_switch)
        self.gridLayout_5.addLayout(mode_switch_layout, 4, 5, 1, 1)
        self.gridLayout_6.addLayout(self.gridLayout_5, 0, 0, 1, 1)

        self.retranslateUi(Laser_Control)
        QtCore.QMetaObject.connectSlotsByName(Laser_Control)
        tooltips.apply_tooltips(self, tooltips.LASER_TOOLTIPS)
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
        self.switch_to_cli_button.clicked.connect(self.switch_to_cli_clicked)

        self.laser_wavelegnth.currentIndexChanged.connect(self.laser_wavelegnth_changed)
        self.laser_wavelegnth.currentIndexChanged.connect(lambda _i: self._update_wavelength_nm_label())
        self.laser_power.valueChanged.connect(self.laser_power_changed)
        self.laser_rate.currentIndexChanged.connect(self.laser_rate_changed)
        self.laser_rate.currentIndexChanged.connect(lambda _i: self._clamp_divider_to_min_output_rate())
        self.laser_divition_factor.valueChanged.connect(self.laser_divition_factor_changed)
        # Initialise the (nnnn nm) label now that the combo is populated.
        self._update_wavelength_nm_label()

        # Try to open the CLI session. If it fails (most commonly because
        # the laser is currently in NKTPBus mode) we do NOT pop a modal
        # dialog -- the user previously found that auto-recovery on the
        # same Python process did not actually restore communication
        # without a full software restart. Instead we leave the laser
        # disconnected, show a persistent red banner, and enable the
        # "Switch to CLI" button (gated behind Override Access).
        self.com_port_laser = self.conf['COM_PORT_laser']
        self.laser_device = None
        # Do not present the UI designer's example rates as instrument readback.
        self.laser_rate.blockSignals(True)
        self.laser_rate.clear()
        self.laser_rate.blockSignals(False)
        self._invalidate_laser_readouts('Waiting for laser readback')
        self._open_laser_cli(self.com_port_laser, initial_open=True)

        # Laser status loop.
        #
        # SAFETY: this loop both reads the laser state over serial AND
        # mutates GUI widgets (LED pixmaps, button enabled-states) that
        # operators rely on to know whether the laser is emitting. PyQt6
        # widget operations are only valid on the GUI thread; the previous
        # implementation ran ``check_laser_status`` inside a Worker
        # QThread, mutating widgets cross-thread, which can corrupt Qt's
        # internal state and produce undefined LED behaviour.
        #
        # Run it on the GUI thread via a QTimer instead (mirrors the
        # existing _stage_poll_timer). The serial transaction is short
        # (~once per second), and a re-entrancy guard prevents a slow
        # poll from stacking. This trades the cross-thread-widget hazard
        # for brief serial I/O on the GUI thread -- a deliberate, strictly
        # safer trade for laser status.
        self._laser_status_in_progress = False
        self._laser_status_timer = QtCore.QTimer()
        self._laser_status_timer.setInterval(1000)
        self._laser_status_timer.timeout.connect(self._poll_laser_status)
        self._laser_status_timer.start()

        # ----- SmarAct laser focusing stage --------------------------------
        self.stage_device = None
        self._stage_poll_timer = None
        self._stage_reference_worker = None
        self._stage_reference_cancel = None
        self._last_stage_position_error = ""
        self._consecutive_stage_position_errors = 0
        self.laser_speed_x.valueChanged.connect(lambda _v: self._update_stage_speed_label(self.laser_speed_x))
        self.laser_speed_y.valueChanged.connect(lambda _v: self._update_stage_speed_label(self.laser_speed_y))
        self.laser_speed_z.valueChanged.connect(lambda _v: self._update_stage_speed_label(self.laser_speed_z))
        # Direction buttons: continuous jog while held — see the stage
        # control GUI for the rationale. Tick interval matches
        # click_duration_s so consecutive relative steps chain into
        # smooth motion at the selected velocity.
        for button, axis, sign in (
            (self.laser_left, mcs2_stage.AXIS_X, -1),
            (self.leser_right, mcs2_stage.AXIS_X, +1),
            (self.laser_up, mcs2_stage.AXIS_Y, +1),
            (self.laser_down, mcs2_stage.AXIS_Y, -1),
            (self.laser_forward, mcs2_stage.AXIS_Z, +1),
            (self.laser_backward, mcs2_stage.AXIS_Z, -1),
        ):
            button.pressed.connect(
                lambda a=axis, s=sign: self._start_continuous_stage_jog(a, s)
            )
            button.released.connect(self._stop_continuous_stage_jog)
        self.laser_home.clicked.connect(self._stage_go_home)
        self.laser_stage_reference.clicked.connect(self._stage_reference)
        self.laser_stage_stop.clicked.connect(self._stage_stop)
        self.laser_stage_superuser.clicked.connect(self._stage_super_user_access)
        for selector in (self.laser_speed_x, self.laser_speed_y, self.laser_speed_z):
            self._update_stage_speed_label(selector)
        self._connect_stage_device()
        self._setup_laser_alignment(Laser_Control)
        self._setup_compact_laser_layout(Laser_Control)
        make_window_responsive(Laser_Control)

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
            for sl_lbl in (self.laser_speed_x_label, self.laser_speed_y_label, self.laser_speed_z_label):
                sl_lbl.setEnabled(False)
            logging.getLogger("pyccapt.gui").info(
	            "Laser Control: no laser-stage locator configured "
	            "(stage_smartact_laser); apt/laser_* will be logged as 0."
            )
            return
        try:
	        # Catch any exception (not just SmarActStageError): a connection
	        # problem must never crash GUI startup (runs from setupUi).
            self.stage_device = mcs2_stage.SmarActStage(self._stage_locator)
        except Exception as exc:
            self.stage_device = None
            self._stage_connect_error = str(exc)
            self.error_message(self._stage_connect_error)
            self._set_stage_movement_enabled(False)
            logging.getLogger("pyccapt.gui").warning(
	            "Laser Control: could not connect to SmarAct laser stage '%s': %s "
	            "Laser-stage position will be logged as 0 in apt/laser_* for "
	            "experiments started now.",
	            self._stage_locator, exc,
            )
            return
        self._set_stage_movement_enabled(True)
        logging.getLogger("pyccapt.gui").info(
	        "Laser Control: connected to SmarAct laser stage '%s'; publishing "
	        "position to apt/laser_*.",
	        self._stage_locator,
        )
        self._stage_poll_timer = QtCore.QTimer()
        self._stage_poll_timer.setInterval(500)
        self._stage_poll_timer.timeout.connect(self._refresh_stage_position)
        self._stage_poll_timer.start()
        self._refresh_stage_position()

    def reconnect_stage(self):
	    """Retry the laser-stage SmarAct connection if not connected.

		Lets the user recover a laser stage that was off/busy at GUI startup
		by re-opening the Laser Control window, instead of restarting the
		whole app. No-op when already connected (and when no laser-stage
		locator is configured), so safe to call on every open.
		"""
	    if self.stage_device is not None or not self._stage_locator:
		    return
	    logging.getLogger("pyccapt.gui").info(
		    "Laser Control: retrying connection to SmarAct laser stage '%s'...",
		    self._stage_locator,
	    )
	    self._connect_stage_device()

    def _set_stage_movement_enabled(self, enabled):
        # The laser focusing stage is a SEPARATE device from the laser
        # source, so its jog/Home buttons are deliberately NOT gated on the
        # stage connection (nor on the laser): they stay enabled and each
        # click handler shows a clear "Laser stage not connected" message if
        # the SmarAct controller is absent. This keeps the panel usable and
        # independent when the stage connects late or its SDK/locator isn't
        # set. ``enabled`` is ignored here, kept only for call-site
        # compatibility.
        for btn in (
            self.laser_up,
            self.laser_down,
            self.laser_left,
            self.leser_right,
            self.laser_forward,
            self.laser_backward,
            self.laser_home,
        ):
            btn.setEnabled(True)
        # Reference is gated solely behind Override Access; _stage_reference()
        # guards against a missing device.
        self.laser_stage_reference.setEnabled(self.flag_super_user_stage)
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
                "Make sure nothing is in the way of the laser stage and you really want to switch laser modes. Continue?"
            )
            warning.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
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
        self.laser_stage_reference.setEnabled(self.flag_super_user_stage)
        self.nktpbus_mode_switch.setEnabled(self.flag_super_user_stage)
        self.switch_to_cli_button.setEnabled(self.flag_super_user_stage)

    def _axis_velocity_m_s(self, axis):
        selector = (self.laser_speed_x, self.laser_speed_y, self.laser_speed_z)[axis]
        return mcs2_stage.speed_level_to_m_s(
            selector.value(),
            self._speed_max_level,
            self._speed_max_mm_s,
            table=self._speed_table,
        )

    def _update_stage_speed_label(self, selector):
        level = selector.value()
        v_m_s = mcs2_stage.speed_level_to_m_s(
            level,
            self._speed_max_level,
            self._speed_max_mm_s,
            table=self._speed_table,
        )
        step_m = mcs2_stage.click_step_m(v_m_s, self._click_duration_s)
        step_um = step_m * 1e6
        step_text = f"{step_um:.0f}" if step_um >= 10 else f"{step_um:.2f}"
        text = f"{step_text} µm"
        mapping = {
            self.laser_speed_x: self.laser_speed_x_label,
            self.laser_speed_y: self.laser_speed_y_label,
            self.laser_speed_z: self.laser_speed_z_label,
        }
        mapping[selector].setText(text)

    def _stage_jog_axis(self, axis, sign):
        if self._laser_alignment_busy():
            self.error_message('Stop laser alignment before manual stage movement.')
            return
        if self.stage_device is None:
            self.error_message(self._stage_connect_error or "Laser stage not connected.")
            return
        vel = self._axis_velocity_m_s(axis)
        step_m = mcs2_stage.click_step_m(vel, self._click_duration_s)
        try:
            self.stage_device.move_relative_axis(
                axis=axis,
                delta_m=sign * step_m,
                velocity_m_s=vel,
                wait=False,
            )
        except mcs2_stage.SmarActStageError as exc:
            self.error_message(f"Move failed: {exc}")

    def _start_continuous_stage_jog(self, axis, sign):
        """Hold-to-jog start: fire _stage_jog_axis on a timer.

        Mirrors the stage control window's behaviour — see
        gui_stage_control._start_continuous_jog for the full rationale.
        Tick period = click_duration_s so the per-step relative moves
        chain into smooth motion at the selected velocity.
        """
        if self.stage_device is None:
            self.error_message(self._stage_connect_error or "Laser stage not connected.")
            return
        timer = getattr(self, "_continuous_stage_jog_timer", None)
        if timer is None:
            # Ui_Laser_Control is a plain Python class, not a QObject,
            # so parenting the timer to `self` is a TypeError. Match
            # the existing parentless QTimer pattern used elsewhere in
            # this file.
            timer = QtCore.QTimer()
            timer.setSingleShot(False)
            self._continuous_stage_jog_timer = timer
        else:
            try:
                timer.timeout.disconnect()
            except (TypeError, RuntimeError):
                pass
        # Fire one step immediately so a quick tap still moves.
        self._stage_jog_axis(axis, sign)
        timer.timeout.connect(lambda: self._stage_jog_axis(axis, sign))
        interval_ms = max(50, int(self._click_duration_s * 1000))
        timer.start(interval_ms)

    def _stop_continuous_stage_jog(self):
        """Hold-to-jog stop: kill the timer and truncate the in-flight step."""
        timer = getattr(self, "_continuous_stage_jog_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()
        try:
            timer.timeout.disconnect()
        except (TypeError, RuntimeError, AttributeError):
            pass
        if self.stage_device is not None:
            try:
                self.stage_device.stop()
            except Exception:
                pass

    def _stage_go_home(self):
        if self._laser_alignment_busy():
            self.error_message('Stop laser alignment before manual stage movement.')
            return
        if self.stage_device is None:
            self.error_message(self._stage_connect_error or "Laser stage not connected.")
            return
        x_m, y_m, z_m = self._home_target_m
        # Home uses a dedicated velocity (stage_home_velocity_mm_s in
        # config.toml) instead of the per-axis speed presets - otherwise a
        # Home click with X at the slowest preset takes minutes.
        try:
            self.stage_device.move_absolute(
                x_m=x_m,
                y_m=y_m,
                z_m=z_m,
                velocity_m_s=self._home_velocity_m_s,
                wait=False,
            )
        except mcs2_stage.SmarActStageError as exc:
            self.error_message(f"Home failed: {exc}")

    def _stage_reference(self):
        if self._laser_alignment_busy():
            self.error_message('Stop laser alignment before manual stage movement.')
            return
        if self.stage_device is None:
            self.error_message(self._stage_connect_error or "Laser stage not connected.")
            return
        if self._stage_reference_worker is not None and self._stage_reference_worker.isRunning():
            self.error_message("Reference already in progress.")
            return
        self._stage_reference_cancel = threading.Event()
        self._stage_reference_worker = _LaserStageReferenceWorker(
            self.stage_device,
            self._stage_reference_cancel,
            self._referencing_options,
            self._reference_timeout_s,
            self._reference_velocity_m_s,
        )
        self._stage_reference_worker.finished_ok.connect(self._on_stage_reference_done)
        self._stage_reference_worker.finished_with_error.connect(self._on_stage_reference_failed)
        self._set_stage_jog_enabled(False)
        self.laser_stage_reference.setEnabled(False)
        self.error_message("Referencing - keep the path clear; press STOP to abort.")
        self._stage_reference_worker.start()

    def _on_stage_reference_done(self):
        self._stage_reference_worker = None
        self._stage_reference_cancel = None
        self._set_stage_jog_enabled(True)
        self.laser_stage_reference.setEnabled(self.flag_super_user_stage)
        self._last_stage_position_error = ""
        self._consecutive_stage_position_errors = 0
        self.error_message("Reference complete.")

    def _on_stage_reference_failed(self, message):
        self._stage_reference_worker = None
        self._stage_reference_cancel = None
        self._set_stage_jog_enabled(True)
        self.laser_stage_reference.setEnabled(self.flag_super_user_stage)
        self.error_message(f"Reference failed: {message}")

    def _set_stage_jog_enabled(self, enabled):
        """Enable/disable jog + Home for the duration of a reference run.

        Only the referencing lock toggles these (so the operator can't jog
        mid-reference); the connection state is handled by the click
        handlers, not by greying the buttons out.
        """
        for btn in (
            self.laser_up,
            self.laser_down,
            self.laser_left,
            self.leser_right,
            self.laser_forward,
            self.laser_backward,
            self.laser_home,
        ):
            btn.setEnabled(enabled)

    def _stage_stop(self):
        self._cancel_laser_alignment()
        self._stop_continuous_stage_jog()
        # Abort an in-flight reference search FIRST, then stop the axes.
        if self._stage_reference_cancel is not None:
            self._stage_reference_cancel.set()
        if self.stage_device is None:
            return
        self.stage_device.stop()

    def _refresh_stage_position(self):
        if self.stage_device is None:
            return
        try:
            pos = self.stage_device.get_position()
        except mcs2_stage.SmarActStageError as exc:
            text = str(exc)
            if text != self._last_stage_position_error:
                self._last_stage_position_error = text
                self._consecutive_stage_position_errors = 1
                self.error_message(f"Position read failed: {text}")
            else:
                self._consecutive_stage_position_errors += 1
                if self._consecutive_stage_position_errors == 4 and self._stage_poll_timer is not None:
                    self._stage_poll_timer.setInterval(5000)
            return
        if self._consecutive_stage_position_errors:
            self._last_stage_position_error = ""
            self._consecutive_stage_position_errors = 0
            if self._stage_poll_timer is not None:
                self._stage_poll_timer.setInterval(500)
        self._set_stage_axis(pos['x'], self.laser_x_mm, self.laser_x_um, self.laser_x_nm, self.laser_x_cord)
        self._set_stage_axis(pos['y'], self.laser_y_mm, self.laser_y_um, self.laser_y_nm, self.laser_y_cord)
        self._set_stage_axis(pos['z'], self.laser_z_mm, self.laser_z_um, self.laser_z_nm, self.laser_z_cord)
        # Publish to shared variables (meters) so the experiment loop can log
        # the laser-stage position per iteration into apt/*. Best-effort: a
        # Manager hiccup must never break the position display.
        try:
	        self.variables.laser_pos_x = float(pos['x'])
	        self.variables.laser_pos_y = float(pos['y'])
	        self.variables.laser_pos_z = float(pos['z'])
	        self.variables.laser_stage_snapshot = (float(pos['x']), float(pos['y']), float(pos['z']), time.monotonic())
        except Exception:
	        pass

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
        self.led_laser_on.setText(_translate("Laser_Control", "Laser On (emits)"))
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
        self.laser_on.setText(_translate("Laser_Control", "Laser On (emits)"))
        self.label_3.setText(_translate("Laser_Control", "Divition Factor"))
        self.laser_enable.setText(_translate("Laser_Control", "Output Enable"))
        self.led_laser_listen.setText(_translate("Laser_Control", "Listen"))
        self.led_laser_laser_standby.setText(_translate("Laser_Control", "Standby"))
        self.label.setText(_translate("Laser_Control", "IR AOM setting (%)"))
        self.laser_listen.setText(_translate("Laser_Control", "Listen"))
        self.label_4.setText(_translate("Laser_Control", "Wavelength"))
        self.label_12.setText(_translate("Laser_Control", "Scan mode"))
        self.laser_scan_mode5.setItemText(0, _translate("Laser_Control", "Standard"))
        self.label_13.setText(_translate("Laser_Control", "Focus mode"))
        self.laser_focus_mode.setItemText(0, _translate("Laser_Control", "Standard"))
        # Display in laser-physics-friendly units: W rather than mW (so a
        # 5 W laser shows "5.028" instead of "5028"), and µJ rather than nJ
        # (so a 12.5 µJ pulse shows "12.570" instead of "12570"). The
        # shared variables use mW / nJ; detector writers convert nJ to pJ.
        self.label_9.setText(_translate("Laser_Control", "Selected output (W)"))
        self.label_10.setText(_translate("Laser_Control", "Pulse energy (µJ)"))
        self.label_11.setText(_translate("Laser_Control", "Frequency (kHz)"))
        self.label_19.setText(_translate("Laser_Control", "x"))
        self.label_17.setText(_translate("Laser_Control", "y"))
        self.label_18.setText(_translate("Laser_Control", "z"))
        self.label_14.setText(_translate("Laser_Control", "Speed preset"))
        self.label_15.setText(_translate("Laser_Control", "X"))
        self.label_16.setText(_translate("Laser_Control", "Y"))
        self.label_speed_z.setText(_translate("Laser_Control", "Z"))
        self.laser_stage_reference.setText(_translate("Laser_Control", "Reference"))
        self.laser_stage_stop.setText(_translate("Laser_Control", "STOP"))
        self.laser_up.setText(_translate("Laser_Control", "Y+\n▲"))
        self.laser_left.setText(_translate("Laser_Control", "◀  X−"))
        self.leser_right.setText(_translate("Laser_Control", "X+  ▶"))
        self.laser_down.setText(_translate("Laser_Control", "▼\nY−"))
        self.laser_forward.setText(_translate("Laser_Control", "Z+\nForward"))
        self.laser_backward.setText(_translate("Laser_Control", "Z−\nBackward"))
        self.laser_home.setText(_translate("Laser_Control", "Home"))
        self.Error.setText(_translate("Laser_Control", "<html><head/><body><p><br/></p></body></html>"))
        self.start_scanning.setText(_translate("Laser_Control", "Start scaning"))
        self.nktpbus_mode_switch.setText(_translate("Laser_Control", "Nktpbus mode"))
        self.switch_to_cli_button.setText(_translate("Laser_Control", "Switch to CLI"))

        ####
        self.pattern_number = r'\b\d+\b'
        self.timer_hide_error = QtCore.QTimer()
        self.timer_hide_error.timeout.connect(self.hideMessage)
        self.laser_power.setMinimum(0.0)
        # Display actual readback even when it exceeds a local command limit.
        self.laser_power.setMaximum(100.)
        self.laser_power.setSingleStep(0.1)
        self.laser_divition_factor.setMinimum(1)
        # Manual p123: integer divider 1..10,000,000.
        self._clamp_divider_to_min_output_rate()

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

    def _poll_laser_status(self):
        """Main-thread QTimer slot that drives the laser status loop.

        Wraps ``check_laser_status`` with a re-entrancy guard so a slow
        serial transaction can't cause overlapping polls to stack, and
        swallows transient errors so a single failed read doesn't stop
        the timer (mirrors the old Worker.run try/except).
        """
        if self._laser_status_in_progress:
            return
        self._laser_status_in_progress = True
        try:
            self.check_laser_status()
        except Exception as exc:
            self._invalidate_laser_readouts(exc)
            self._apply_button_locks_for_status(None)
            print(f"Laser status poll failed: {exc}")
        finally:
            self._laser_status_in_progress = False

    def check_laser_status(self):
        if self.laser_device is None:
            return
        status = self.laser_device.StatusRead()
        code = readback.scalar(status)
        if code not in (9, 33, 65, 129):
            self._invalidate_laser_readouts(f'Laser status {code}: transitioning, warning or fault')
        requested = False
        # A later lower-state request must cancel any queued emission request.
        listen, standby, on, output = self.listen_mode, self.standby_mode, self.on_mode, self.enable_ouput_mode
        self.listen_mode = self.standby_mode = self.on_mode = self.enable_ouput_mode = False
        if listen:
            self.laser_device.Listen()
            requested = True
        elif standby:
            self.laser_device.Standby()
            requested = True
        elif on:
            if code == 33:
                # Manual pp116-117 / QSG p8: this command OPENS output.
                self.laser_device.Enable()
                requested = True
            elif code == 129:
                self.laser_device.AOMDisable()
                requested = True
        elif output:
            if code == 65:
                self.laser_device.AOMEnable()
                requested = True
            elif code == 129:
                self.laser_device.AOMDisable()
                requested = True
        changed = self._apply_laser_settings(code)
        if requested or changed:
            status = self.laser_device.StatusRead()
        self._apply_button_locks_for_status(status)
        if requested or changed or self.index >= 5:
            self._sync_controls_from_device()
            self.index = 0
        self.index += 1
        # Settings changes may reset e_power to zero. Only show readback;
        # never silently restore full AOM output.

    def switch_to_nktpbus_mode(self):
        """Switch the laser from CLI -> NKTPBus mode.

        After this returns, the CLI session is gone. The operator can
        either drive the laser from NKT Photonics CONTROL software (the
        usual reason for switching) or click the "Switch to CLI" button
        in this window to return to CLI mode.
        """
        if self.laser_device is not None:
            try:
                self.laser_device.InterbusEnable()
            except Exception as exc:
                self.error_message(f"Could not send NKTPBus switch command: {exc}")
                return
            try:
                self.laser_device.close_port()
            except Exception:
                pass
            self.laser_device = None
            # CLI is gone -- disable Listen / Standby / Laser On /
            # Output Enable / wavelength immediately so the operator
            # cannot send commands that would silently go nowhere. The
            # buttons will re-enable automatically the next time
            # _apply_button_locks_for_status runs after a successful
            # CLI re-open via the "Switch to CLI" button.
            self._apply_button_locks_for_status(None)
            # Make the new state explicit on both the persistent banner
            # and the auto-hide message line so the operator cannot miss
            # what happened.
            self._set_laser_disconnected_banner(
                "switched to NKTPBus mode -- use NKT Photonics software, or click 'Switch to CLI' to come back."
            )
            self.error_message(
                "Switched to NKTPBus mode. You can now use the NKT "
                "Photonics CONTROL software, or click 'Switch to CLI' "
                "to come back to CLI control."
            )
        else:
            # Either already in NKTPBus or never connected -- in both
            # cases the state buttons should be disabled.
            self._apply_button_locks_for_status(None)
            self.error_message(
                "The laser is already in NKTPBus mode (or the CLI "
                "session is closed). Click 'Switch to CLI' to return "
                "to CLI control."
            )

    # ------------------------------------------------------------------
    # Derived / live read-outs
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Strict adjacent-state button locks
    # ------------------------------------------------------------------

    def _apply_button_locks_for_status(self, status_text):
        code = readback.scalar(status_text) if self.laser_device is not None else None
        known = code in (9, 33, 65, 129)
        idle = code in (9, 33)
        running = bool(self.variables.start_flag)
        self.laser_listen.setEnabled(known and code != 9)
        self.laser_standby.setEnabled(known and code != 33)
        self.laser_on.setEnabled(code in (33, 129))
        self.laser_enable.setEnabled(code in (65, 129))
        self.laser_wavelegnth.setEnabled(idle and not running)
        self.laser_rate.setEnabled(idle and not running and bool(getattr(self, '_frequency_table', {})))
        self.laser_divition_factor.setEnabled(known and not running)
        self.laser_power.setEnabled(known and not self._laser_alignment_busy())
        for widget, active in ((self.led_laser_listen, code == 9),
                               (self.led_laser_laser_standby, code == 33),
                               (self.led_laser_on, code in (65, 129)),
                               (self.led_laser_enable, code == 129)):
            widget.setPixmap(self.led_green if active else self.led_red)

    # ------------------------------------------------------------------
    # CLI session lifecycle
    # ------------------------------------------------------------------

    def _set_laser_disconnected_banner(self, reason):
        """Show or hide the persistent 'laser not connected' red banner.

        Pass ``reason=""`` (or any falsy value) to hide the banner when
        the CLI session is healthy. Also keeps the shared
        ``flag_laser_connected`` flag in sync so the main GUI can show
        its own warning.
        """
        if not reason:
            self.laser_connection_banner.setText("")
            self.laser_connection_banner.setToolTip("")
            self.laser_connection_banner.setVisible(False)
            self.laser_connection_banner.setStyleSheet("")
            try:
                self.variables.flag_laser_connected = True
            except Exception:
                pass
            return
        self._invalidate_laser_readouts(reason)
        self.laser_connection_banner.setText(f"⚠  LASER NOT CONNECTED — {reason}")
        self.laser_connection_banner.setToolTip(str(reason))
        self.laser_connection_banner.setStyleSheet(
            "QLabel{background: #fff0f0;color: #c00000;border: 1px solid #c00000;border-radius: 4px;padding: 4px;}"
        )
        self.laser_connection_banner.setVisible(True)
        self._reset_laser_message_scroll()
        try:
            self.variables.flag_laser_connected = False
        except Exception:
            pass

    def _close_laser_device(self):
        """Close the existing CLI port if any. Safe on already-closed/None."""
        dev = getattr(self, 'laser_device', None)
        if dev is None:
            return
        try:
            dev.close_port()
        except Exception:
            pass
        self.laser_device = None

    def _open_laser_cli(self, com_port, *, initial_open=False):
        """Try to open a CLI session on ``com_port``.

        Returns True on success. On failure, sets the persistent
        disconnected banner and returns False without raising.

        ``initial_open`` controls only the wording on the banner
        (so we say "could not connect" the first time vs "still not
        connected" after a manual reconnect attempt).
        """
        # Always start from a known-good state -- close any previous
        # handle so reopening doesn't double-own the port.
        self._close_laser_device()
        self._frequency_table = {}

        device = origamiClassCLI.origClass(com_port)
        try:
            databack = device.open_port()
        except Exception as exc:
            reason = f"Laser: could not open {com_port}: {exc}"
            print(reason)
            self._set_laser_disconnected_banner(reason)
            self._apply_button_locks_for_status(None)
            return False

        if databack != 0:
            error_text = device.last_error or "no response from device"
            reason = f"Laser: could not open {com_port}: {error_text}. Available ports: {_available_serial_ports_text()}"
            print(reason)
            self._set_laser_disconnected_banner(reason)
            self._apply_button_locks_for_status(None)
            return False

        # Port opened. Now probe whether the laser actually answers
        # CLI -- if it's in NKTPBus mode, the port opens fine but
        # StatusRead returns garbage / nothing.
        try:
            status = device.StatusRead()
        except Exception as exc:
            reason = (
                f"Laser: port {com_port} opened but the laser did not reply to "
                f"any CLI command ({exc}). Most likely the laser is in "
                f"NKTPBus mode — use 'Switch to CLI', or check the cable."
            )
            print(reason)
            try:
                device.close_port()
            except Exception:
                pass
            self._set_laser_disconnected_banner(reason)
            self._apply_button_locks_for_status(None)
            return False
        if not status or 'ly_oxp2' not in status:
            reason = (
                f"Laser: port {com_port} opened but the laser did not reply "
                f"to CLI (probably in NKTPBus mode). Use 'Switch to CLI'."
            )
            print(reason)
            try:
                device.close_port()
            except Exception:
                pass
            self._set_laser_disconnected_banner(reason)
            self._apply_button_locks_for_status(None)
            return False

        # We have a working CLI session. Wire up the rest of the init
        # path that used to live inline in setupUi.
        self.laser_device = device
        try:
            # Send the laser to Listen mode -- the lowest-power, safest
            # state -- and read back the live settings into the GUI
            # widgets. Do NOT call wavelength_change here: writing the
            # wavelength register triggers an optical translation that
            # takes the laser out of Listen and into Standby, which the
            # operator did not ask for. The wavelength is whatever it
            # was last set to and will be reflected in the dropdown by
            # _sync_controls_from_device.
            self.laser_device.Listen()
            status_after = self.laser_device.StatusRead()
            self._sync_controls_from_device(initial=True)
            if status_after.strip() == 'ly_oxp2_dev_status 9':
                self.led_laser_listen.setPixmap(self.led_green)
            # Apply button locks immediately so the new state's buttons
            # become interactive without waiting for the next status poll.
            self._apply_button_locks_for_status(status_after)
        except Exception as exc:
            print(f"Laser CLI initial handshake failed: {exc}")

        self._set_laser_disconnected_banner("")  # clear
        return True

    # --- Tunables for the CLI switch retry loop --------------------------
    # 3 attempts is a sensible default: the first attempt usually fails
    # because the FTDI / Prolific virtual COM port is still cached at the
    # old baud rate; the second attempt typically succeeds. Bumping to 3
    # gives one more chance before we ask the user to restart pyccapt.
    CLI_SWITCH_NUM_ATTEMPTS = 3
    CLI_SWITCH_INITIAL_DELAY_S = 1.5  # after the register write
    CLI_SWITCH_RETRY_DELAY_S = 1.0  # between subsequent attempts

    def _wait_with_gui_pump(self, total_s, chunk_s=0.1):
        """Sleep for ``total_s`` seconds while keeping the UI responsive.

        ``time.sleep`` in the Qt GUI thread freezes the window. We
        instead sleep in 100 ms chunks and pump the event loop after
        each one so the window keeps repainting.
        """
        end = time.time() + max(0.0, float(total_s))
        while time.time() < end:
            remaining = end - time.time()
            time.sleep(min(chunk_s, max(remaining, 0.0)))
            QtWidgets.QApplication.processEvents()

    def switch_to_cli_clicked(self):
        """Operator-initiated NKTPBus -> CLI switch.

        Wired to the "Switch to CLI" button (gated behind Override Access).
        After writing register 0x39 := 1 we try to re-open the CLI session
        up to ``CLI_SWITCH_NUM_ATTEMPTS`` times with
        ``CLI_SWITCH_RETRY_DELAY_S`` between attempts. The first attempt
        fairly often fails because the host-side serial driver hasn't
        cleanly flipped from 115 200 baud (Interbus) to 38 400 baud
        (CLI); the second attempt usually succeeds.
        """
        port = getattr(self, 'com_port_laser', None) or self.conf.get('COM_PORT_laser', '')
        if not port:
            self.error_message("No COM port configured for the laser.")
            return

        # If CLI already works there is nothing to do; just try to
        # (re)open the session.
        if nktpbus_switch.is_cli_responding(port):
            self.error_message("CLI already responding -- reopening session...")
            if self._open_laser_cli(port):
                self.error_message("Laser CLI session opened.")
            return

        # Drop our own CLI handle (if any) before NKTPDLL takes the port.
        self._close_laser_device()

        self.error_message(f"Switching to CLI on {port}... please wait.")
        QtWidgets.QApplication.processEvents()

        try:
            nktpbus_switch.switch_to_cli(port)
        except nktpbus_switch.NKTPSwitchError as exc:
            self._set_laser_disconnected_banner(f"switch to CLI failed: {exc}")
            self.error_message(f"Could not switch laser to CLI: {exc}")
            return
        except Exception as exc:
            self._set_laser_disconnected_banner(f"switch to CLI failed: {exc}")
            self.error_message(f"Unexpected error switching to CLI: {exc}")
            return

        # Give the laser firmware a moment to flip its UART driver from
        # 115 200 (Interbus) down to 38 400 (CLI). Keep the GUI alive.
        self.error_message("Register written. Waiting for the laser UART to flip to 38 400 baud...")
        self._wait_with_gui_pump(self.CLI_SWITCH_INITIAL_DELAY_S)

        # Up to N tries with a delay in between. The status line gets
        # rewritten on each attempt so the operator can see progress
        # rather than staring at an unchanging banner.
        for attempt in range(1, self.CLI_SWITCH_NUM_ATTEMPTS + 1):
            self.error_message(f"Switching to CLI -- connection attempt {attempt} of {self.CLI_SWITCH_NUM_ATTEMPTS}...")
            QtWidgets.QApplication.processEvents()
            if self._open_laser_cli(port):
                self.error_message(
                    f"Laser switched to CLI and reconnected (attempt {attempt} of {self.CLI_SWITCH_NUM_ATTEMPTS})."
                )
                return
            if attempt < self.CLI_SWITCH_NUM_ATTEMPTS:
                self.error_message(
                    f"Attempt {attempt} did not respond yet. Retrying in {self.CLI_SWITCH_RETRY_DELAY_S:.1f} s..."
                )
                self._wait_with_gui_pump(self.CLI_SWITCH_RETRY_DELAY_S)

        # All attempts exhausted.
        self._set_laser_disconnected_banner("switch to CLI completed but the laser still does not reply")
        self.error_message(
            f"Switch register was written but CLI is still not "
            f"responding after {self.CLI_SWITCH_NUM_ATTEMPTS} attempts. "
            "If this persists, restart pyccapt; some USB serial "
            "drivers need a clean re-open after a baud-rate change."
        )

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
        self.Error.setToolTip(message)
        self._laser_error_active = True
        self.Error.show()
        self._reset_laser_message_scroll()
        self.timer_hide_error.start(8000)

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

        self.timer_hide_error.stop()
        self.Error.setToolTip('')
        self._laser_error_active = False
        self.Error.hide()
        self._reset_laser_message_scroll()

    def stop(self):
        """Stop background workers and release device handles.

        Called from gui_main.cleanup() when the user closes the main GUI;
        without this the laser status loop keeps running and prevents the
        Python process from exiting cleanly.
        """
        # Stop the laser-status QTimer (now a main-thread timer instead of
        # a Worker QThread; see _poll_laser_status). Stopping the timer is
        # synchronous so there's no thread to join.
        self._cancel_laser_alignment()
        alignment_timer = getattr(self, '_laser_alignment_timer', None)
        if alignment_timer is not None:
            alignment_timer.stop()
        status_timer = getattr(self, '_laser_status_timer', None)
        if status_timer is not None:
            try:
                status_timer.stop()
            except Exception:
                pass
        # Back-compat: if an old-style Worker QThread is still present
        # (e.g. a subclass or hot-reload), stop it too.
        worker = getattr(self, 'worker', None)
        if worker is not None:
            try:
                worker.stop()
                worker.wait(1000)  # ms
            except Exception:
                pass

        # Cancel any in-flight Reference search BEFORE we touch
        # ctl.Close - the SmarAct SDK is not thread-safe.
        cancel = getattr(self, '_stage_reference_cancel', None)
        if cancel is not None:
            try:
                cancel.set()
            except Exception:
                pass
        ref_worker = getattr(self, '_stage_reference_worker', None)
        if ref_worker is not None:
            try:
                ref_worker.wait(1000)
            except Exception:
                pass
        # Stop the laser-stage poll timer and release the SmarAct handle.
        if getattr(self, '_stage_poll_timer', None) is not None:
            self._stage_poll_timer.stop()
        if getattr(self, 'stage_device', None) is not None:
            try:
                self.stage_device.stop()
            except Exception:
                pass
            try:
                self.stage_device.close()
            except Exception:
                pass
            self.stage_device = None

        # Close the laser serial port if we still own it.
        # SAFETY: before closing, force the laser to a known-off state so
        # it can't continue emitting after the GUI has exited. The
        # previous code went straight to close_port() and the laser
        # stayed in whatever state the operator had set (potentially
        # state 129 = output enabled) until the rig was power-cycled.
        # We disable the AOM (kills the output gate), set its level to
        # zero, then issue Standby (firmware moves to state 33). Each
        # call is wrapped individually so a stuck command doesn't
        # block the rest of the safe-off sequence.
        laser_dev = getattr(self, 'laser_device', None)
        if laser_dev is not None:
            for safe_call, label in (
                (lambda: laser_dev.AOMDisable(), 'AOMDisable'),
                (lambda: laser_dev.AOM(0), 'AOM(0)'),
                (lambda: laser_dev.Standby(), 'Standby'),
            ):
                try:
                    safe_call()
                except Exception as exc:
                    print(f"laser safe-off: {label} failed (non-fatal): {exc}")
            try:
                laser_dev.close_port()
            except Exception:
                pass
            self.laser_device = None


class Worker(QThread):
    """Background poller for the laser status loop.

    The original implementation used `while True:` which made it
    unstoppable - the parent QApplication couldn't exit because the
    QThread never finished.  Now an internal _stop_flag is honoured at
    every iteration so the main GUI's cleanup() can shut us down.
    """

    def __init__(self, task_function):
        super().__init__()
        self.task_function = task_function
        self._stop_flag = False

    def stop(self):
        """Request that run() exits at the next loop iteration."""
        self._stop_flag = True

    def run(self):
        while not self._stop_flag:
            try:
                self.task_function()
            except Exception:
                # Don't let a transient error kill the polling loop.
                logging.getLogger(__name__).exception('Laser status poll failed')
            self.msleep(1000)


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
        if getattr(self, "force_close", False):
            event.accept()
            return
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
