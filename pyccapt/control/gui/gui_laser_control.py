import multiprocessing
import os
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
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_19 = QtWidgets.QLabel(parent=Laser_Control)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_19.setFont(font)
        self.label_19.setObjectName("label_19")
        self.gridLayout_4.addWidget(self.label_19, 0, 0, 1, 1)
        self.laser_x_cord = QtWidgets.QLCDNumber(parent=Laser_Control)
        self.laser_x_cord.setObjectName("laser_x_cord")
        self.gridLayout_4.addWidget(self.laser_x_cord, 0, 1, 1, 1)
        self.label_17 = QtWidgets.QLabel(parent=Laser_Control)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_17.setFont(font)
        self.label_17.setObjectName("label_17")
        self.gridLayout_4.addWidget(self.label_17, 1, 0, 1, 1)
        self.laser_y_cord = QtWidgets.QLCDNumber(parent=Laser_Control)
        self.laser_y_cord.setObjectName("laser_y_cord")
        self.gridLayout_4.addWidget(self.laser_y_cord, 1, 1, 1, 1)
        self.label_18 = QtWidgets.QLabel(parent=Laser_Control)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_18.setFont(font)
        self.label_18.setObjectName("label_18")
        self.gridLayout_4.addWidget(self.label_18, 2, 0, 1, 1)
        self.laser_z_cord = QtWidgets.QLCDNumber(parent=Laser_Control)
        self.laser_z_cord.setObjectName("laser_z_cord")
        self.gridLayout_4.addWidget(self.laser_z_cord, 2, 1, 1, 1)
        self.gridLayout_5.addLayout(self.gridLayout_4, 3, 0, 1, 1)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_14 = QtWidgets.QLabel(parent=Laser_Control)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.gridLayout_2.addWidget(self.label_14, 0, 0, 1, 1)
        self.laser_speed_lr = QtWidgets.QDoubleSpinBox(parent=Laser_Control)
        self.laser_speed_lr.setStyleSheet("QDoubleSpinBox{\n"
                                          "                                            background: rgb(223,223,233)\n"
                                          "                                            }\n"
                                          "                                        ")
        self.laser_speed_lr.setObjectName("laser_speed_lr")
        self.gridLayout_2.addWidget(self.laser_speed_lr, 0, 1, 1, 1)
        self.label_15 = QtWidgets.QLabel(parent=Laser_Control)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.gridLayout_2.addWidget(self.label_15, 1, 0, 1, 1)
        self.laser_speed_ud = QtWidgets.QDoubleSpinBox(parent=Laser_Control)
        self.laser_speed_ud.setStyleSheet("QDoubleSpinBox{\n"
                                          "                                            background: rgb(223,223,233)\n"
                                          "                                            }\n"
                                          "                                        ")
        self.laser_speed_ud.setObjectName("laser_speed_ud")
        self.gridLayout_2.addWidget(self.laser_speed_ud, 1, 1, 1, 1)
        self.label_16 = QtWidgets.QLabel(parent=Laser_Control)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.gridLayout_2.addWidget(self.label_16, 2, 0, 1, 1)
        self.laser_speed_fb = QtWidgets.QDoubleSpinBox(parent=Laser_Control)
        self.laser_speed_fb.setStyleSheet("QDoubleSpinBox{\n"
                                          "                                            background: rgb(223,223,233)\n"
                                          "                                            }\n"
                                          "                                        ")
        self.laser_speed_fb.setObjectName("laser_speed_fb")
        self.gridLayout_2.addWidget(self.laser_speed_fb, 2, 1, 1, 1)
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
        self.laser_home = QtWidgets.QPushButton(parent=Laser_Control)
        self.laser_home.setStyleSheet("")
        self.laser_home.setObjectName("laser_home")
        self.gridLayout_5.addWidget(self.laser_home, 3, 5, 1, 1)
        self.Error = QtWidgets.QLabel(parent=Laser_Control)
        self.Error.setMinimumSize(QtCore.QSize(500, 30))
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        font.setStrikeOut(False)
        self.Error.setFont(font)
        self.Error.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
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
        Laser_Control.setTabOrder(self.laser_focus_mode, self.laser_speed_lr)
        Laser_Control.setTabOrder(self.laser_speed_lr, self.laser_speed_ud)
        Laser_Control.setTabOrder(self.laser_speed_ud, self.laser_speed_fb)
        Laser_Control.setTabOrder(self.laser_speed_fb, self.laser_left)
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
        self.label_14.setText(_translate("Laser_Control", "Speed L/R"))
        self.label_15.setText(_translate("Laser_Control", "Speed U/D"))
        self.label_16.setText(_translate("Laser_Control", "Speed F/B"))
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
        pass


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
        self.gui_laser_control.stop()  # Call the stop method to stop any background activity
        self.closed.emit()  # Emit the custom closed signal
        # Additional cleanup code here if needed
        super().closeEvent(event)

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

