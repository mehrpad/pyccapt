import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import pyqtSignal, QObject, QThread
from PyQt6.QtGui import QPixmap

# Local module and scripts
from pyccapt.control.core import runtime
from pyccapt.control.devices import arduino_illumination, camera
from pyccapt.control.gui import tooltips


ILLUMINATION_RGB = {
    "green": (0, 255, 0),
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "white": (255, 255, 255),
}


class Ui_Cameras_Alignment(object):
    def __init__(self, variables, conf, SignalEmitter):
        """
        Initialize the UiCamerasAlignment class.

        Args:
                variables: Global experiment variables.
                conf: Configuration data.
                SignalEmitter: Signal emitter for communication.
        """
        # Cameras default to manual/preset exposure on startup: the Auto
        # Exposure Time button starts deselected (off). It is a plain
        # toggle that goes green while selected, like the other buttons.
        # This flag mirrors the worker's `exposure_auto` state.
        self.auto_exposure_time_flag = False
        # "Manual Exposure Time" toggle (only meaningful when auto is off):
        #   selected  -> the three µs fields are user-editable;
        #   unselected -> the worker drives them from light-dependent presets.
        self.manual_exposure_flag = False
        # This window keeps a local Override Access state for illumination and
        # camera-exposure changes.  Startup illumination is intentionally not
        # gated: it always turns on at the configured default brightness.
        self.flag_super_user = False
        self.illumination_controller = None
        self.conf = conf
        self.emitter = SignalEmitter
        self.variables = variables

    def setupUi(self, Cameras_Alignment):
        """
        Set up the GUI for the Cameras Alignment window.

        Args:
        Cameras_Alignment:

        Returns:
        None
        """
        Cameras_Alignment.setObjectName("Cameras_Alignment")
        # Keep the camera tool window practical on smaller displays. The
        # original 1210x938 footprint and image minimums are reduced by ~20%.
        Cameras_Alignment.resize(970, 750)
        self.gridLayout_5 = QtWidgets.QGridLayout(Cameras_Alignment)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_208 = QtWidgets.QLabel(parent=Cameras_Alignment)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_208.setFont(font)
        self.label_208.setObjectName("label_208")
        self.gridLayout.addWidget(self.label_208, 5, 0, 1, 1)
        ###
        # self.cam_s_d = QtWidgets.QLabel(parent=Cameras_alignment)
        self.cam_s_d = pg.ImageView(parent=Cameras_Alignment)
        self.cam_s_d.adjustSize()
        self.cam_s_d.ui.histogram.hide()
        self.cam_s_d.ui.roiBtn.hide()
        self.cam_s_d.ui.menuBtn.hide()
        self.cam_s_d.setObjectName("cam_s_o")
        ###
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.cam_s_d.sizePolicy().hasHeightForWidth())
        self.cam_s_d.setSizePolicy(sizePolicy)
        self.cam_s_d.setMinimumSize(QtCore.QSize(480, 200))
        self.cam_s_d.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.cam_s_d.setStyleSheet(
            "QWidget{\n"
            "                                            border: 2px solid gray;\n"
            "                                            }\n"
            "                                        "
        )
        # self.cam_s_d.setText("")
        self.cam_s_d.setObjectName("cam_s_d")
        self.gridLayout.addWidget(self.cam_s_d, 3, 4, 1, 1)
        self.label_209 = QtWidgets.QLabel(parent=Cameras_Alignment)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_209.setFont(font)
        self.label_209.setObjectName("label_209")
        self.gridLayout.addWidget(self.label_209, 5, 4, 1, 1)
        ###
        # self.cam_b_d = QtWidgets.QLabel(parent=Cameras_alignment)
        self.cam_b_d = pg.ImageView(parent=Cameras_Alignment)
        self.cam_b_d.adjustSize()
        self.cam_b_d.ui.histogram.hide()
        self.cam_b_d.ui.roiBtn.hide()
        self.cam_b_d.ui.menuBtn.hide()
        ###
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.cam_b_d.sizePolicy().hasHeightForWidth())
        self.cam_b_d.setSizePolicy(sizePolicy)
        self.cam_b_d.setMinimumSize(QtCore.QSize(480, 200))
        self.cam_b_d.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.cam_b_d.setStyleSheet(
            "QWidget{\n"
            "                                            border: 2px solid gray;\n"
            "                                            }\n"
            "                                        "
        )
        # self.cam_b_d.setText("")
        self.cam_b_d.setObjectName("cam_b_d")
        self.gridLayout.addWidget(self.cam_b_d, 6, 4, 1, 1)
        self.label_204 = QtWidgets.QLabel(parent=Cameras_Alignment)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_204.setFont(font)
        self.label_204.setObjectName("label_204")
        self.gridLayout.addWidget(self.label_204, 2, 4, 1, 1)
        ###
        # self.cam_s_o = QtWidgets.QLabel(parent=Cameras_alignment)
        self.cam_s_o = pg.ImageView(parent=Cameras_Alignment)
        self.cam_s_o.adjustSize()
        self.cam_s_o.ui.histogram.hide()
        self.cam_s_o.ui.roiBtn.hide()
        self.cam_s_o.ui.menuBtn.hide()
        self.cam_s_o.setObjectName("cam_s_o")
        ###
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.cam_s_o.sizePolicy().hasHeightForWidth())
        self.cam_s_o.setSizePolicy(sizePolicy)
        self.cam_s_o.setMinimumSize(QtCore.QSize(200, 200))
        self.cam_s_o.setStyleSheet(
            "QWidget{\n"
            "                                            border: 2px solid gray;\n"
            "                                            }\n"
            "                                        "
        )
        # self.cam_s_o.setText("")
        self.cam_s_o.setObjectName("cam_s_o")
        self.gridLayout.addWidget(self.cam_s_o, 3, 0, 1, 4)
        self.label_203 = QtWidgets.QLabel(parent=Cameras_Alignment)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_203.setFont(font)
        self.label_203.setObjectName("label_203")
        self.gridLayout.addWidget(self.label_203, 2, 0, 1, 1)
        ###
        # self.cam_b_o = QtWidgets.QLabel(parent=Cameras_alignment)
        self.cam_b_o = pg.ImageView(parent=Cameras_Alignment)
        self.cam_b_o.adjustSize()
        self.cam_b_o.ui.histogram.hide()
        self.cam_b_o.ui.roiBtn.hide()
        self.cam_b_o.ui.menuBtn.hide()
        self.cam_b_o.setObjectName("cam_s_o")
        ###
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.cam_b_o.sizePolicy().hasHeightForWidth())
        self.cam_b_o.setSizePolicy(sizePolicy)
        self.cam_b_o.setMinimumSize(QtCore.QSize(200, 200))
        self.cam_b_o.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.cam_b_o.setStyleSheet(
            "QWidget{\n"
            "                                            border: 2px solid gray;\n"
            "                                            }\n"
            "                                        "
        )
        # self.cam_b_o.setText("")
        self.cam_b_o.setObjectName("cam_b_o")
        self.gridLayout.addWidget(self.cam_b_o, 6, 0, 1, 4)
        self.label_205 = QtWidgets.QLabel(parent=Cameras_Alignment)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.label_205.setFont(font)
        self.label_205.setObjectName("label_205")
        self.gridLayout.addWidget(self.label_205, 4, 0, 1, 1)
        self.label_202 = QtWidgets.QLabel(parent=Cameras_Alignment)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.label_202.setFont(font)
        self.label_202.setObjectName("label_202")
        self.gridLayout.addWidget(self.label_202, 1, 0, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_211 = QtWidgets.QLabel(parent=Cameras_Alignment)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_211.setFont(font)
        self.label_211.setObjectName("label_211")
        self.gridLayout_3.addWidget(self.label_211, 2, 0, 1, 1)
        ###
        # self.cam_angle_o = QtWidgets.QLabel(parent=Cameras_alignment)
        self.cam_angle_o = pg.ImageView(parent=Cameras_Alignment)
        self.cam_angle_o.adjustSize()
        self.cam_angle_o.ui.histogram.hide()
        self.cam_angle_o.ui.roiBtn.hide()
        self.cam_angle_o.ui.menuBtn.hide()
        self.cam_angle_o.setObjectName("cam_s_o")
        ###
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.cam_angle_o.sizePolicy().hasHeightForWidth())
        self.cam_angle_o.setSizePolicy(sizePolicy)
        self.cam_angle_o.setMinimumSize(QtCore.QSize(200, 200))
        self.cam_angle_o.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.cam_angle_o.setStyleSheet(
            "QWidget{\n"
            "                                            border: 2px solid gray;\n"
            "                                            }\n"
            "                                        "
        )
        # self.cam_angle_o.setText("")
        self.cam_angle_o.setObjectName("cam_angle_o")
        self.gridLayout_3.addWidget(self.cam_angle_o, 3, 0, 1, 1)
        ###
        # self.cam_angle_d = QtWidgets.QLabel(parent=Cameras_alignment)
        self.cam_angle_d = pg.ImageView(parent=Cameras_Alignment)
        self.cam_angle_d.adjustSize()
        self.cam_angle_d.ui.histogram.hide()
        self.cam_angle_d.ui.roiBtn.hide()
        self.cam_angle_d.ui.menuBtn.hide()
        self.cam_angle_d.setObjectName("cam_s_o")
        ###
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.cam_angle_d.sizePolicy().hasHeightForWidth())
        self.cam_angle_d.setSizePolicy(sizePolicy)
        self.cam_angle_d.setMinimumSize(QtCore.QSize(480, 200))
        self.cam_angle_d.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.cam_angle_d.setStyleSheet(
            "QWidget{\n"
            "                                            border: 2px solid gray;\n"
            "                                            }\n"
            "                                        "
        )
        # self.cam_angle_d.setText("")
        self.cam_angle_d.setObjectName("cam_angle_d")
        self.gridLayout_3.addWidget(self.cam_angle_d, 3, 1, 1, 1)
        self.label_210 = QtWidgets.QLabel(parent=Cameras_Alignment)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_210.setFont(font)
        self.label_210.setObjectName("label_210")
        self.gridLayout_3.addWidget(self.label_210, 2, 1, 1, 1)
        self.label_206 = QtWidgets.QLabel(parent=Cameras_Alignment)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.label_206.setFont(font)
        self.label_206.setObjectName("label_206")
        self.gridLayout_3.addWidget(self.label_206, 1, 0, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_3)
        self.gridLayout_4.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.superuser = QtWidgets.QPushButton(parent=Cameras_Alignment)
        self.superuser.setMinimumSize(QtCore.QSize(0, 25))
        self.superuser.setStyleSheet(
            "QPushButton{\n"
            "    background: rgb(193, 193, 193)\n"
            "}\n"
        )
        self.superuser.setObjectName("superuser")
        self.horizontalLayout.addWidget(self.superuser)
        # The Auto Exposure button is placed below the exposure-time fields;
        # it is constructed here with the other controls and added later.
        self.auto_exposure_time = QtWidgets.QPushButton(parent=Cameras_Alignment)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.auto_exposure_time.sizePolicy().hasHeightForWidth())
        self.auto_exposure_time.setSizePolicy(sizePolicy)
        self.auto_exposure_time.setMinimumSize(QtCore.QSize(0, 25))
        self.auto_exposure_time.setStyleSheet(
            "QPushButton{\n"
            "                                            background: rgb(193, 193, 193)\n"
            "                                            }\n"
            "                                        "
        )
        self.auto_exposure_time.setObjectName("auto_exposure_time")
        spacerItem = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum
        )
        self.horizontalLayout.addItem(spacerItem)
        self.led_light = QtWidgets.QLabel(parent=Cameras_Alignment)
        self.led_light.setMaximumSize(QtCore.QSize(50, 50))
        self.led_light.setObjectName("led_light")
        self.horizontalLayout.addWidget(self.led_light)
        self.light = QtWidgets.QPushButton(parent=Cameras_Alignment)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.light.sizePolicy().hasHeightForWidth())
        self.light.setSizePolicy(sizePolicy)
        self.light.setMinimumSize(QtCore.QSize(0, 25))
        self.light.setStyleSheet(
            "QPushButton{\n"
            "                                            background: rgb(193, 193, 193)\n"
            "                                            }\n"
            "                                        "
        )
        self.light.setObjectName("light")
        self.horizontalLayout.addWidget(self.light)
        spacerItem1 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum
        )
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.illumination_percent_label = QtWidgets.QLabel(parent=Cameras_Alignment)
        self.illumination_percent_label.setMinimumSize(QtCore.QSize(130, 0))
        self.illumination_percent_label.setMaximumSize(QtCore.QSize(500, 50))
        self.illumination_percent_label.setObjectName("illumination_percent_label")
        self.gridLayout_2.addWidget(self.illumination_percent_label, 1, 0, 1, 1)
        self.illumination_percent = QtWidgets.QSpinBox(parent=Cameras_Alignment)
        self.illumination_percent.setRange(0, 100)
        self.illumination_percent.setSuffix(" %")
        self.illumination_percent.setValue(int(self.conf.get("camera_illumination_percent", 25)))
        self.illumination_percent.setObjectName("illumination_percent")
        self.gridLayout_2.addWidget(self.illumination_percent, 1, 1, 1, 1)
        self.dimming_separator = QtWidgets.QFrame(parent=Cameras_Alignment)
        self.dimming_separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.dimming_separator.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.dimming_separator.setObjectName("dimming_separator")
        self.gridLayout_2.addWidget(self.dimming_separator, 2, 0, 1, 2)
        self.led_light_2 = QtWidgets.QLabel(parent=Cameras_Alignment)
        self.led_light_2.setMinimumSize(QtCore.QSize(130, 0))
        self.led_light_2.setMaximumSize(QtCore.QSize(500, 50))
        self.led_light_2.setObjectName("led_light_2")
        self.gridLayout_2.addWidget(self.led_light_2, 3, 0, 1, 1)
        self.exposure_time_cam_1 = QtWidgets.QLineEdit(parent=Cameras_Alignment)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.exposure_time_cam_1.sizePolicy().hasHeightForWidth())
        self.exposure_time_cam_1.setSizePolicy(sizePolicy)
        self.exposure_time_cam_1.setMinimumSize(QtCore.QSize(0, 20))
        self.exposure_time_cam_1.setStyleSheet(
            "QLineEdit{\n"
            "                                                background: rgb(223,223,233)\n"
            "                                                }\n"
            "                                            "
        )
        self.exposure_time_cam_1.setObjectName("exposure_time_cam_1")
        self.gridLayout_2.addWidget(self.exposure_time_cam_1, 3, 1, 1, 1)
        self.exposure_time_cam_2 = QtWidgets.QLineEdit(parent=Cameras_Alignment)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.exposure_time_cam_2.sizePolicy().hasHeightForWidth())
        self.exposure_time_cam_2.setSizePolicy(sizePolicy)
        self.exposure_time_cam_2.setMinimumSize(QtCore.QSize(0, 20))
        self.exposure_time_cam_2.setStyleSheet(
            "QLineEdit{\n"
            "                                                background: rgb(223,223,233)\n"
            "                                                }\n"
            "                                            "
        )
        self.exposure_time_cam_2.setObjectName("exposure_time_cam_2")
        self.gridLayout_2.addWidget(self.exposure_time_cam_2, 4, 1, 1, 1)
        self.exposure_time_cam_3 = QtWidgets.QLineEdit(parent=Cameras_Alignment)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.exposure_time_cam_3.sizePolicy().hasHeightForWidth())
        self.exposure_time_cam_3.setSizePolicy(sizePolicy)
        self.exposure_time_cam_3.setMinimumSize(QtCore.QSize(0, 20))
        self.exposure_time_cam_3.setStyleSheet(
            "QLineEdit{\n"
            "                                                background: rgb(223,223,233)\n"
            "                                                }\n"
            "                                            "
        )
        self.exposure_time_cam_3.setObjectName("exposure_time_cam_3")
        self.gridLayout_2.addWidget(self.exposure_time_cam_3, 5, 1, 1, 1)
        self.led_light_3 = QtWidgets.QLabel(parent=Cameras_Alignment)
        self.led_light_3.setMinimumSize(QtCore.QSize(130, 0))
        self.led_light_3.setMaximumSize(QtCore.QSize(500, 50))
        self.led_light_3.setObjectName("led_light_3")
        self.gridLayout_2.addWidget(self.led_light_3, 4, 0, 1, 1)
        self.default_exposure_time = QtWidgets.QPushButton(parent=Cameras_Alignment)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.default_exposure_time.sizePolicy().hasHeightForWidth())
        self.default_exposure_time.setSizePolicy(sizePolicy)
        self.default_exposure_time.setMinimumSize(QtCore.QSize(0, 25))
        self.default_exposure_time.setStyleSheet(
            "QPushButton{\n"
            "                                            background: rgb(193, 193, 193)\n"
            "                                            }\n"
            "                                        "
        )
        self.default_exposure_time.setObjectName("default_exposure_time")
        self.led_light_4 = QtWidgets.QLabel(parent=Cameras_Alignment)
        self.led_light_4.setMinimumSize(QtCore.QSize(130, 0))
        self.led_light_4.setMaximumSize(QtCore.QSize(500, 50))
        self.led_light_4.setObjectName("led_light_4")
        self.gridLayout_2.addWidget(self.led_light_4, 5, 0, 1, 1)
        self.exposure_mode_layout = QtWidgets.QHBoxLayout()
        self.exposure_mode_layout.setObjectName("exposure_mode_layout")
        self.exposure_mode_layout.addWidget(self.auto_exposure_time)
        self.exposure_mode_layout.addWidget(self.default_exposure_time)
        self.gridLayout_2.addLayout(self.exposure_mode_layout, 6, 0, 1, 2)
        self.verticalLayout_2.addLayout(self.gridLayout_2)
        # ----- Camera list + connect/disconnect panel ---------------------
        # Compact box that lives right under the exposure-time controls
        # (inside verticalLayout_2) so it stacks under the "Exposure Time
        # Angle" row instead of widening the whole window. One row per
        # detected Basler camera; refreshed every 1.5 s.
        self.camera_list_box = QtWidgets.QGroupBox("Cameras detected", parent=Cameras_Alignment)
        self.camera_list_box.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum
        )
        self.camera_list_box.setMaximumWidth(310)
        self.camera_list_box.setMaximumHeight(135)
        self.camera_list_layout = QtWidgets.QVBoxLayout(self.camera_list_box)
        self.camera_list_layout.setContentsMargins(5, 3, 5, 3)
        self.camera_list_layout.setSpacing(1)
        self._camera_row_widgets = {}  # serial -> dict(widget, label, connect_btn, disconnect_btn)
        self.camera_list_empty_label = QtWidgets.QLabel("(scanning …)", parent=self.camera_list_box)
        self.camera_list_empty_label.setStyleSheet("color: gray;")
        self.camera_list_layout.addWidget(self.camera_list_empty_label)
        self.camera_list_layout.addStretch(1)
        self.verticalLayout_2.addWidget(self.camera_list_box)

        # Compact, display-only instrument monitor for alignment work. Values
        # come from the existing Manager namespace populated by the pumps and
        # vacuum process; the camera window never touches gauge hardware.
        self.instrument_monitor_box = QtWidgets.QGroupBox("Instrument monitor", parent=Cameras_Alignment)
        self.instrument_monitor_box.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Maximum
        )
        monitor_layout = QtWidgets.QGridLayout(self.instrument_monitor_box)
        monitor_layout.setContentsMargins(5, 3, 5, 4)
        monitor_layout.setHorizontalSpacing(6)
        monitor_layout.setVerticalSpacing(2)
        self.camera_monitor_labels = {}
        self.camera_monitor_lcds = {}
        stage_sensor_name = self.conf.get('cryo_sensor_3', 'stage').replace('_', ' ').title()
        monitor_specs = (
            ("vacuum_main", "Main Chamber (mBar)", "#2ca02c"),
            ("vacuum_buffer", "Buffer Chamber (mBar)", "#8c564b"),
            ("vacuum_load_lock", "Load Lock (mBar)", "#1f77b4"),
            ("vacuum_cryo_load_lock", "Cryo Load Lock (mBar)", "#d627a8"),
            ("temperature", f"Temp. {stage_sensor_name} (K)", "#ff8c00"),
        )
        for row, (attribute, label_text, color) in enumerate(monitor_specs):
            label = QtWidgets.QLabel(label_text, parent=self.instrument_monitor_box)
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            label.setMinimumWidth(145)
            label_font = label.font()
            label_font.setBold(True)
            label.setFont(label_font)
            lcd = QtWidgets.QLCDNumber(parent=self.instrument_monitor_box)
            lcd.setDigitCount(8)
            lcd.setSegmentStyle(QtWidgets.QLCDNumber.SegmentStyle.Flat)
            lcd.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed
            )
            lcd.setMinimumSize(QtCore.QSize(150, 34))
            lcd.setMaximumHeight(34)
            lcd.setStyleSheet(f"QLCDNumber{{border: 1px solid {color}; border-radius: 4px;}}")
            lcd.setToolTip(label_text)
            monitor_layout.addWidget(label, row, 0)
            monitor_layout.addWidget(lcd, row, 1)
            self.camera_monitor_labels[attribute] = label
            self.camera_monitor_lcds[attribute] = lcd
        monitor_layout.setColumnStretch(1, 1)
        self.instrument_monitor_box.setMaximumHeight(215)
        self.verticalLayout_2.addWidget(self.instrument_monitor_box)
        self.verticalLayout_2.addStretch(1)

        self.gridLayout_4.addLayout(self.verticalLayout_2, 0, 1, 1, 1)
        self.gridLayout_5.addLayout(self.gridLayout_4, 0, 0, 1, 1)

        # ----- Bottom status banner --------------------------------------
        # Mirrors the Error / status label used elsewhere in the GUI suite
        # so the user sees connection / grab failures without watching the
        # terminal.
        self.camera_status_label = QtWidgets.QLabel(parent=Cameras_Alignment)
        self.camera_status_label.setMinimumHeight(28)
        self.camera_status_label.setWordWrap(True)
        self.camera_status_label.setStyleSheet(
            "QLabel{ color: rgb(140,0,0); padding: 4px; border: 1px solid rgb(200,200,200); border-radius: 4px; }"
        )
        self.camera_status_label.setText("")
        self.gridLayout_5.addWidget(self.camera_status_label, 1, 0, 1, 1)

        self.retranslateUi(Cameras_Alignment)
        QtCore.QMetaObject.connectSlotsByName(Cameras_Alignment)
        tooltips.apply_tooltips(self, tooltips.CAMERAS_TOOLTIPS)
        Cameras_Alignment.setTabOrder(self.superuser, self.light)
        Cameras_Alignment.setTabOrder(self.light, self.illumination_percent)
        Cameras_Alignment.setTabOrder(self.illumination_percent, self.auto_exposure_time)
        Cameras_Alignment.setTabOrder(self.auto_exposure_time, self.default_exposure_time)
        Cameras_Alignment.setTabOrder(self.default_exposure_time, self.exposure_time_cam_1)
        Cameras_Alignment.setTabOrder(self.exposure_time_cam_1, self.exposure_time_cam_2)
        Cameras_Alignment.setTabOrder(self.exposure_time_cam_2, self.exposure_time_cam_3)

        ###
        # Create a ROI (Region of Interest) item
        self.roi_s = pg.ROI([1000, 1000], [500, 200], movable=True, resizable=True)
        self.roi_s.addScaleHandle([1, 1], [0, 0])  # Adding a scaling handle to the ROI
        self.roi_s.addScaleHandle([0, 0], [1, 1])
        self.roi_s.addScaleHandle([1, 0], [0, 1])
        self.roi_s.addScaleHandle([0, 1], [1, 0])
        self.roi_s.setZValue(10)  # Make sure ROI is drawn on top
        self.cam_s_o.addItem(self.roi_s)

        self.roi_b = pg.ROI([1000, 1000], [1000, 400], movable=True, resizable=True)
        self.roi_b.addScaleHandle([1, 1], [0, 0])  # Adding a scaling handle to the ROI
        self.roi_b.addScaleHandle([0, 0], [1, 1])
        self.roi_b.addScaleHandle([1, 0], [0, 1])
        self.roi_b.addScaleHandle([0, 1], [1, 0])
        self.roi_b.setZValue(10)  # Make sure ROI is drawn on top
        self.cam_b_o.addItem(self.roi_b)

        self.roi_angle = pg.ROI([1000, 1000], [1000, 400], movable=True, resizable=True)
        self.roi_angle.addScaleHandle([1, 1], [0, 0])  # Adding a scaling handle to the ROI
        self.roi_angle.addScaleHandle([0, 0], [1, 1])
        self.roi_angle.addScaleHandle([1, 0], [0, 1])
        self.roi_angle.addScaleHandle([0, 1], [1, 0])
        self.roi_angle.setZValue(10)  # Make sure ROI is drawn on top
        self.cam_angle_o.addItem(self.roi_angle)

        # Diagram and LEDs ##############
        self.led_red = QPixmap('./files/led-red-on.png')
        self.led_green = QPixmap('./files/green-led-on.png')
        self.led_light.setPixmap(self.led_red)

        # bottom camera (x, y)
        # arrow1 = pg.ArrowItem(pos=(925, 770), angle=0)
        # self.cam_b_o.addItem(arrow1)
        # Side camera (x, y) main arrow for puck exchange (Blue arrow)
        arrow1 = pg.ArrowItem(pos=(920, 830), angle=-90, brush='r')
        arrow3 = pg.ArrowItem(pos=(940, 600), angle=0, brush='g')
        arrow2 = pg.ArrowItem(pos=(795, 850), angle=-90, brush='b')

        self.cam_s_o.addItem(arrow1)
        self.cam_s_o.addItem(arrow2)
        self.cam_s_o.addItem(arrow3)
        # side camera zoom (x, y) zoom arrow
        # arrow1 = pg.ArrowItem(pos=(380, 115), angle=90, brush='r')
        # self.cam_s_d.addItem(arrow1)
        # bottom camera zoom (x, y) zoom arrow
        # arrow1 = pg.ArrowItem(pos=(620, 265), angle=90, brush='r')
        # self.cam_b_d.addItem(arrow1)
        ###
        self.superuser.clicked.connect(self.super_user_access)
        self.light.clicked.connect(self.light_switch)
        self.illumination_percent.valueChanged.connect(self.update_illumination_percent)
        self.auto_exposure_time.clicked.connect(self.auto_exposure_time_switch)
        self.default_exposure_time.clicked.connect(self.manual_exposure_time_switch)

        self.emitter.img0_orig.connect(self.update_cam_s_o)
        self.emitter.img1_orig.connect(self.update_cam_b_o)
        self.emitter.img2_orig.connect(self.update_cam_angle_o)
        # Connect the Arduino before starting camera capture, so the worker
        # begins with the correct illumination state.
        self._initialise_illumination()
        self.initialize_camera_thread()

        self.exposure_time_cam_1.editingFinished.connect(self.update_exposure_time)
        self.exposure_time_cam_2.editingFinished.connect(self.update_exposure_time)
        self.exposure_time_cam_3.editingFinished.connect(self.update_exposure_time)

        self.original_button_style = self.auto_exposure_time.styleSheet()
        self._original_superuser_style = self.superuser.styleSheet()
        # Captured so the "Manual Exposure Time" toggle can restore its own
        # look when deselected (it goes green while selected).
        self._original_manual_exposure_style = self.default_exposure_time.styleSheet()
        # Camera illumination and exposure controls start locked until the
        # operator deliberately grants Override Access.
        self._refresh_exposure_widgets()

        self.emitter.cams_exposure_time_default.connect(self.set_default_exposure_time)
        # Live read-back: each tick the worker tells us what
        # ExposureTime each camera is actually using. Most useful in
        # auto mode where the firmware picks the value; in manual mode
        # it just mirrors what the user typed.
        self.emitter.cams_exposure_time_current.connect(self._update_current_exposure_fields)

    def retranslateUi(self, Cameras_Alignment):
        """

        Args:
        Cameras_alignment:

        Returns:
        None
        """
        _translate = QtCore.QCoreApplication.translate
        ###
        # Cameras_alignment.setWindowTitle(_translate("Cameras_alignment", "Form"))
        Cameras_Alignment.setWindowTitle(_translate("Cameras_alignment", "PyCCAPT Cameras"))
        Cameras_Alignment.setWindowIcon(QtGui.QIcon('./files/logo.png'))
        self.Cameras_Alignment = Cameras_Alignment
        ###
        self.label_208.setText(_translate("Cameras_Alignment", "Overview"))
        self.label_209.setText(_translate("Cameras_Alignment", "Detail"))
        self.label_204.setText(_translate("Cameras_Alignment", "Detail"))
        self.label_203.setText(_translate("Cameras_Alignment", "Overview"))
        self.label_205.setText(_translate("Cameras_Alignment", "Camera Top"))
        self.label_202.setText(_translate("Cameras_Alignment", "Camera Side"))
        self.label_211.setText(_translate("Cameras_Alignment", "Overview"))
        self.label_210.setText(_translate("Cameras_Alignment", "Detail"))
        self.label_206.setText(_translate("Cameras_Alignment", "Camera Angle"))
        self.superuser.setText(_translate("Cameras_Alignment", "Override Access"))
        self.auto_exposure_time.setText(_translate("Cameras_Alignment", "Auto Exposure Time"))
        self.led_light.setText("")
        self.light.setText(_translate("Cameras_Alignment", "Light On / Off"))
        self.illumination_percent_label.setText(_translate("Cameras_Alignment", "Dimming"))
        self.led_light_2.setText(_translate("Cameras_Alignment", "Exposure Time Side (us)"))
        self.exposure_time_cam_1.setText(_translate("Cameras_Alignment", "2000000"))
        self.exposure_time_cam_2.setText(_translate("Cameras_Alignment", "1000000"))
        self.exposure_time_cam_3.setText(_translate("Cameras_Alignment", "2000000"))
        self.led_light_3.setText(_translate("Cameras_Alignment", "Exposure Time Top (us)"))
        self.default_exposure_time.setText(_translate("Cameras_Alignment", "Manual Exposure Time"))
        self.led_light_4.setText(_translate("Cameras_Alignment", "Exposure Time Angle (us)"))

        ###
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.cameras_screenshot)
        self.timer.start(2000)  # Check every 2000 milliseconds (1 second)

        # Periodic refresh of the camera list / status banner.
        self.camera_list_timer = QtCore.QTimer()
        self.camera_list_timer.timeout.connect(self._refresh_camera_panel)
        self.camera_list_timer.start(1500)
        # Run once immediately so the panel is populated before the
        # first 1.5s tick.
        QtCore.QTimer.singleShot(200, self._refresh_camera_panel)

        self.instrument_monitor_timer = QtCore.QTimer(self.Cameras_Alignment)
        self.instrument_monitor_timer.timeout.connect(self._refresh_instrument_monitor)
        self.instrument_monitor_timer.start(1000)
        self._refresh_instrument_monitor()

    def _refresh_instrument_monitor(self):
        """Refresh the compact vacuum and stage-temperature LCDs."""
        pressure_specs = (
            ("vacuum_main", "vacuum_threshold_main"),
            ("vacuum_buffer", "camera_warning_threshold_buffer"),
            ("vacuum_load_lock", "vacuum_threshold_load_lock"),
            ("vacuum_cryo_load_lock", "vacuum_threshold_cryo_load_lock"),
        )
        for attribute, threshold_key in pressure_specs:
            value = self._shared_numeric_value(attribute)
            self.camera_monitor_lcds[attribute].display("Error" if value is None or value < 0 else f"{value:.2e}")
            default_threshold = 1e-8 if attribute == "vacuum_buffer" else float("inf")
            threshold = float(self.conf.get(threshold_key, default_threshold))
            color = "red" if value is not None and value >= 0 and value > threshold else "black"
            self.camera_monitor_labels[attribute].setStyleSheet(f"color: {color};")

        temperature = self._shared_numeric_value("temperature")
        self.camera_monitor_lcds["temperature"].display(
            "Error" if temperature is None or temperature < 0 else f"{temperature:.2f}"
        )

    def _shared_numeric_value(self, attribute):
        try:
            return float(getattr(self.variables, attribute))
        except (AttributeError, TypeError, ValueError):
            return None

    def set_default_exposure_time(self, exposure_time_default):
        """
        Set the default exposure time

        Args:
        exposure_time_default: Default exposure time

        Return:
        None
        """
        self.exposure_time_cam_1.setText(str(exposure_time_default[0]))
        self.exposure_time_cam_2.setText(str(exposure_time_default[1]))
        self.exposure_time_cam_3.setText(str(exposure_time_default[2]))

    def _update_current_exposure_fields(self, values):
        """Reflect the camera's live ExposureTime in each line edit.

        Skips any field that currently has keyboard focus so the live
        update never stomps on a value the user is mid-typing. Also
        skips fields whose corresponding camera isn't attached (the
        worker publishes ``None`` for those).
        """
        if not values:
            return
        widgets = (self.exposure_time_cam_1, self.exposure_time_cam_2, self.exposure_time_cam_3)
        for widget, value in zip(widgets, values):
            if value is None:
                continue
            if widget.hasFocus():
                continue
            text = str(int(value))
            if widget.text() != text:
                widget.setText(text)

    def update_exposure_time(self):
        """
        Update the exposure time of the cameras

        Args:
        None

        Return:
        None
        """
        # Clamp manual exposure entry to a safe range before emitting.
        # The previous code passed whatever ``int(text)`` evaluated to
        # directly to the firmware -- including 0 (rejected silently,
        # leaving the UI showing a bogus value) and absurdly large
        # values (10**9 us = 1000 s, also rejected). Basler / generic
        # USB cameras typically accept ~100 us .. ~10 s; clamp here
        # and surface the bound to the user.
        EXPOSURE_MIN_US = 1
        EXPOSURE_MAX_US = 10_000_000  # 10 seconds

        def _clamp(field):
            txt = field.text().strip()
            if not txt:
                return None
            try:
                raw = int(txt)
            except ValueError:
                return None
            clamped = max(EXPOSURE_MIN_US, min(EXPOSURE_MAX_US, raw))
            if clamped != raw:
                print(
                    f'Camera exposure clamped: {raw} -> {clamped} us '
                    f'(allowed range {EXPOSURE_MIN_US} .. {EXPOSURE_MAX_US} us)'
                )
                # Reflect the clamped value back in the field so the
                # user sees what was actually sent.
                field.setText(str(clamped))
            return clamped

        try:
            v1 = _clamp(self.exposure_time_cam_1)
            if v1 is not None:
                self.emitter.cam_1_exposure_time.emit(v1)
            v2 = _clamp(self.exposure_time_cam_2)
            if v2 is not None:
                self.emitter.cam_2_exposure_time.emit(v2)
            v3 = _clamp(self.exposure_time_cam_3)
            if v3 is not None:
                self.emitter.cam_3_exposure_time.emit(v3)
        except Exception as e:
            print(e)
            print('type the exposure time in microseconds')

    def update_cam_s_o(self, img):
        # autoLevels=True stretches the per-frame intensity range to fill
        # the display, which is the most visible quality improvement for
        # alignment: dark "light off" frames stop looking nearly black
        # and bright "light on" frames stop saturating.
        self.cam_s_o.setImage(img, autoRange=False, autoLevels=True)
        roi_coords = self.roi_s.getArraySlice(img, self.cam_s_o.imageItem, axes=(0, 1))

        # Extract the region and update the second image view
        if roi_coords is not None:
            region = img[roi_coords[0][0], roi_coords[0][1]]
            self.cam_s_d.setImage(region, autoRange=False, autoLevels=True)

    def update_cam_b_o(self, img):
        self.cam_b_o.setImage(img, autoRange=False, autoLevels=True)
        roi_coords = self.roi_b.getArraySlice(img, self.cam_b_o.imageItem, axes=(0, 1))

        # Extract the region and update the second image view
        if roi_coords is not None:
            region = img[roi_coords[0][0], roi_coords[0][1]]
            self.cam_b_d.setImage(region, autoRange=False, autoLevels=True)

    def update_cam_angle_o(self, img):
        self.cam_angle_o.setImage(img, autoRange=False, autoLevels=True)
        roi_coords = self.roi_angle.getArraySlice(img, self.cam_angle_o.imageItem, axes=(0, 1))

        # Extract the region and update the second image view
        if roi_coords is not None:
            region = img[roi_coords[0][0], roi_coords[0][1]]
            self.cam_angle_d.setImage(region, autoRange=False, autoLevels=True)

    def _configured_illumination_color(self):
        """Return the configured RGB illumination colour, defaulting to green."""
        color_name = str(self.conf.get("camera_illumination_color", "green")).strip().lower()
        color = ILLUMINATION_RGB.get(color_name)
        if color is None:
            print(f"Unknown camera illumination color {color_name!r}; using green.")
            return "green", ILLUMINATION_RGB["green"]
        return color_name, color

    def _initialise_illumination(self):
        """Connect to the Arduino and turn the camera illumination on.

        This deliberately runs before local Override Access is granted.  The
        operator must still grant Override Access to subsequently change
        brightness, switch the light, or alter exposure modes.
        """
        if self.conf.get("camera_illumination", "off") != "on":
            self.variables.light = False
            self.led_light.setPixmap(self.led_red)
            return

        try:
            controller = arduino_illumination.ArduinoIllumination(
                self.conf.get("COM_PORT_camera_illumination", "auto")
            )
            port = controller.connect()
            color_name, color = self._configured_illumination_color()
            try:
                controller.set_color(*color)
            except RuntimeError as exc:
                # Keep an already-installed older sketch usable until the
                # Nano can be reflashed. Its previous colour is retained.
                print(
                    f"Camera illumination firmware does not accept colour "
                    f"control yet ({exc}); retaining its existing colour."
                )
            controller.set_on(self.illumination_percent.value())
            self.illumination_controller = controller
            self.variables.light = True
            self.variables.light_switch = True
            self.led_light.setPixmap(self.led_green)
            print(
                f"Camera illumination connected on {port}; "
                f"on at {self.illumination_percent.value()}% ({color_name})."
            )
        except Exception as exc:
            self.illumination_controller = None
            self.variables.light = False
            self.led_light.setPixmap(self.led_red)
            message = f"Camera illumination unavailable: {exc}"
            print(message)
            self.camera_status_label.setText(message)

    def _report_illumination_error(self, exc):
        message = f"Camera illumination command failed: {exc}"
        print(message)
        self.camera_status_label.setText(message)

    def super_user_access(self):
        """Toggle Override Access for camera illumination and exposure changes."""
        if not self.flag_super_user:
            warning = QtWidgets.QMessageBox(parent=self.superuser)
            warning.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            warning.setWindowTitle("Confirm Access Override")
            warning.setText("Camera Override Access unlocks illumination and exposure controls.")
            warning.setInformativeText(
                "It enables the illumination on/off button, brightness setting, "
                "Auto Exposure Time, and Manual Exposure Time. Continue only "
                "when it is safe to change the camera alignment conditions."
            )
            warning.setStandardButtons(
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
            )
            warning.setDefaultButton(QtWidgets.QMessageBox.StandardButton.No)
            if warning.exec() != QtWidgets.QMessageBox.StandardButton.Yes:
                return
            self.flag_super_user = True
            self.superuser.setStyleSheet("QPushButton{background: rgb(0, 255, 26)}")
        else:
            self.flag_super_user = False
            self.superuser.setStyleSheet(self._original_superuser_style)
        self._refresh_exposure_widgets()

    def update_illumination_percent(self, percent):
        """Reapply the configured colour, then the selected brightness."""
        if not self.flag_super_user or self.illumination_controller is None:
            return
        try:
            _, color = self._configured_illumination_color()
            self.illumination_controller.set_color(*color)
            self.illumination_controller.set_brightness(percent)
        except Exception as exc:
            self._report_illumination_error(exc)

    def light_switch(self):
        """Toggle Arduino-driven NeoPixel illumination on/off."""
        if not self.flag_super_user or self.illumination_controller is None:
            return
        try:
            if self.variables.light:
                self.illumination_controller.set_off()
                self.variables.light = False
                self.led_light.setPixmap(self.led_red)
            else:
                self.illumination_controller.set_on(self.illumination_percent.value())
                self.variables.light = True
                self.led_light.setPixmap(self.led_green)
            # Keep the camera worker's exposure presets in sync with light.
            self.variables.light_switch = True
        except Exception as exc:
            self._report_illumination_error(exc)

    def auto_exposure_time_switch(self):
        """
        Auto exposure time switch function

        Args:
        None

        Return:
        None
        """
        if not self.flag_super_user:
            return
        self.auto_exposure_time_flag = not self.auto_exposure_time_flag
        # Button goes green while selected (auto on), like the other toggles.
        if self.auto_exposure_time_flag:
            self.auto_exposure_time.setStyleSheet("QPushButton{background: rgb(0, 255, 26)}")
            # Auto owns exposure now; clear any manual selection so that
            # turning auto back off lands in preset mode by default.
            if self.manual_exposure_flag:
                self.manual_exposure_flag = False
                self.default_exposure_time.setStyleSheet(self._original_manual_exposure_style)
                self.emitter.default_exposure_time.emit(False)
        else:
            self.auto_exposure_time.setStyleSheet(self.original_button_style)
        self._refresh_exposure_widgets()
        self.emitter.auto_exposure_time.emit(self.auto_exposure_time_flag)

    def _refresh_exposure_widgets(self):
        """Enable/disable the exposure controls for the current mode.

        Three states:
          * auto on                 -> manual button + fields disabled
                                       (firmware owns ExposureTime).
          * auto off, manual on      -> the three µs fields are editable.
          * auto off, manual off     -> fields disabled; the worker drives
                                       them from the light-dependent presets.
        """
        access_granted = self.flag_super_user
        auto = self.auto_exposure_time_flag
        manual = self.manual_exposure_flag
        self.light.setEnabled(access_granted)
        self.illumination_percent.setEnabled(access_granted)
        self.auto_exposure_time.setEnabled(access_granted)
        # The Manual toggle is only selectable while auto exposure is off,
        # and all of these changes require Override Access.
        self.default_exposure_time.setEnabled(access_granted and not auto)
        fields_editable = access_granted and (not auto) and manual
        for widget in (self.exposure_time_cam_1, self.exposure_time_cam_2, self.exposure_time_cam_3):
            widget.setEnabled(fields_editable)

    def manual_exposure_time_switch(self):
        """Toggle user-manual exposure entry (only when auto is off).

        Selected (green)  -> the three µs fields become user-editable.
        Unselected        -> the worker reverts to the light-dependent
                             presets and the fields are locked.
        """
        if not self.flag_super_user or self.auto_exposure_time_flag:
            return
        self.manual_exposure_flag = not self.manual_exposure_flag
        if self.manual_exposure_flag:
            self.default_exposure_time.setStyleSheet("QPushButton{background: rgb(0, 255, 26)}")
        else:
            self.default_exposure_time.setStyleSheet(self._original_manual_exposure_style)
        self._refresh_exposure_widgets()
        # True = user-manual values; False = light presets.
        self.emitter.default_exposure_time.emit(self.manual_exposure_flag)

    def initialize_camera_thread(self):
        """
        Initialize camera thread

        Args:
        None

        Return:
        None
        """
        if self.conf['camera'] == "off":
            print('The cameras is off')
            return
        # Create a camera worker. The worker keeps running even when zero
        # cameras are currently connected so that hot-plugging a camera
        # later automatically populates the views — the GUI window itself
        # stays open in either case.
        self.camera_worker = camera.CameraWorker(variables=self.variables, emitter=self.emitter, conf=self.conf)
        if self.camera_worker.camera_status_message:
            print(self.camera_worker.camera_status_message)
        if not self.camera_worker.camera_available:
            # pypylon failed to load entirely — no point spinning up a thread.
            self.variables.flag_camera_grab = False
            return

        self.camera_thread = QThread()
        self.camera_worker.moveToThread(self.camera_thread)

        self.camera_thread.started.connect(self.camera_worker.start_capturing)
        self.camera_worker.finished.connect(self.camera_thread.quit)
        self.camera_worker.finished.connect(self.camera_worker.deleteLater)
        self.camera_thread.finished.connect(self.camera_thread.deleteLater)

        self.camera_thread.start()
        self.variables.flag_camera_grab = True

    def stop(self):
        """
        Stop the timer and any other background processes, timers, or threads here

        Args:
        None

        Return:
        None
        """
        # Add any additional cleanup code here
        # with self.variables.lock_setup_parameters:
        self.variables.flag_camera_grab = False
        for timer_name in ('timer', 'camera_list_timer', 'instrument_monitor_timer'):
            timer = getattr(self, timer_name, None)
            if timer is not None:
                timer.stop()
        if hasattr(self, 'camera_thread'):
            self.camera_thread.wait()
        if self.illumination_controller is not None:
            self.illumination_controller.close()

    def cameras_screenshot(self):
        if self.variables.flag_cameras_take_screenshot:
            screenshot = QtWidgets.QApplication.primaryScreen().grabWindow(self.Cameras_Alignment.winId())
            screenshot.save(str(Path(self.variables.path_meta) / "cameras_screenshot.png"), 'png')

    # -------------------------------------------------------------- list ui

    def _refresh_camera_panel(self):
        """Sync the camera-list rows and status banner with the worker."""
        worker = getattr(self, 'camera_worker', None)
        if worker is None:
            return

        # Status banner
        status = getattr(worker, 'latest_status', '') or ""
        if status != self.camera_status_label.text():
            self.camera_status_label.setText(status)

        # Camera list
        try:
            cams = worker.list_cameras()
        except Exception as e:
            cams = []
            print(f"camera list refresh failed: {e}")

        current_serials = {c['serial'] for c in cams}
        # Remove rows for cameras that are no longer detected.
        for sn in list(self._camera_row_widgets):
            if sn not in current_serials:
                row = self._camera_row_widgets.pop(sn)
                row['widget'].deleteLater()

        if not cams:
            self.camera_list_empty_label.setText("(no Basler cameras detected)")
            self.camera_list_empty_label.show()
            return
        self.camera_list_empty_label.hide()

        for cam in cams:
            sn = cam['serial']
            if sn in self._camera_row_widgets:
                self._update_camera_row(self._camera_row_widgets[sn], cam)
            else:
                self._camera_row_widgets[sn] = self._make_camera_row(cam)

    def _make_camera_row(self, cam):
        row = QtWidgets.QWidget(parent=self.camera_list_box)
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(6)
        label = QtWidgets.QLabel(parent=row)
        label.setMinimumWidth(130)
        label.setStyleSheet("font-size: 9px;")
        layout.addWidget(label, 1)
        connect_btn = QtWidgets.QPushButton("Connect", parent=row)
        disconnect_btn = QtWidgets.QPushButton("Disconnect", parent=row)
        connect_btn.setFixedWidth(62)
        disconnect_btn.setFixedWidth(76)
        layout.addWidget(connect_btn)
        layout.addWidget(disconnect_btn)
        # Insert above the trailing stretch.
        self.camera_list_layout.insertWidget(self.camera_list_layout.count() - 1, row)
        sn = cam['serial']
        connect_btn.clicked.connect(lambda _checked=False, s=sn: self._on_connect_clicked(s))
        disconnect_btn.clicked.connect(lambda _checked=False, s=sn: self._on_disconnect_clicked(s))
        entry = {
            'widget': row,
            'label': label,
            'connect_btn': connect_btn,
            'disconnect_btn': disconnect_btn,
        }
        self._update_camera_row(entry, cam)
        return entry

    def _update_camera_row(self, entry, cam):
        sn = cam['serial']
        model = cam['model'] or "Basler"
        if cam['user_disabled']:
            state = "disabled"
            color = "color: rgb(120,120,120);"
        elif cam['attached']:
            state = f"connected to slot {cam['slot']}"
            color = "color: rgb(0,120,0);"
        else:
            state = "detected (not connected)"
            color = "color: rgb(180,90,0);"
        entry['label'].setText(f"<b>{model}</b> &nbsp; {sn} &nbsp; — {state}")
        entry['label'].setStyleSheet(color)
        entry['connect_btn'].setEnabled(not cam['attached'])
        entry['disconnect_btn'].setEnabled(cam['attached'] or not cam['user_disabled'])

    def _on_connect_clicked(self, serial):
        worker = getattr(self, 'camera_worker', None)
        if worker is None:
            return
        try:
            worker.connect_serial(serial)
        except Exception as e:
            self.camera_status_label.setText(f"Connect failed: {e}")
        self._refresh_camera_panel()

    def _on_disconnect_clicked(self, serial):
        worker = getattr(self, 'camera_worker', None)
        if worker is None:
            return
        try:
            worker.disconnect_serial(serial)
        except Exception as e:
            self.camera_status_label.setText(f"Disconnect failed: {e}")
        self._refresh_camera_panel()


class SignalEmitter(QObject):
    img0_orig = pyqtSignal(np.ndarray)
    img1_orig = pyqtSignal(np.ndarray)
    img2_orig = pyqtSignal(np.ndarray)
    cam_1_exposure_time = pyqtSignal(int)
    cam_2_exposure_time = pyqtSignal(int)
    cam_3_exposure_time = pyqtSignal(int)
    cams_exposure_time_default = pyqtSignal(list)
    # Live read-back of ExposureTime from each slot — carries
    # [t_cam_1, t_cam_2, t_cam_3] in microseconds, with None for any
    # slot that isn't currently attached.
    cams_exposure_time_current = pyqtSignal(list)
    default_exposure_time = pyqtSignal(bool)
    auto_exposure_time = pyqtSignal(bool)


class CamerasAlignmentWindow(QtWidgets.QWidget):
    closed = QtCore.pyqtSignal()  # Define a custom closed signal

    def __init__(self, variables, gui_cameras_alignment, close_event, command_queue, *args, **kwargs):
        """
        Initialize the CamerasAlignmentWindow class.

        Args:
                        gui_cameras_alignment: GUI cameras alignment instance.
                        close_event: multiprocessing.Event signalled by this window
                                when it is closed by the user.
                        command_queue: multiprocessing.Queue of typed string commands
                                sent from the main GUI ("show", "show_front", "hide").
        """
        super().__init__(*args, **kwargs)
        self.variables = variables
        self.gui_cameras_alignment = gui_cameras_alignment
        self.command_queue = command_queue
        self.close_event = close_event
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
        self.close_event.set()

    def check_if_should(self):
        """Drain the command queue and dispatch each message in order."""
        raise_to_front = False
        make_visible = False
        hide = False
        while True:
            try:
                msg = self.command_queue.get_nowait()
            except Exception:
                break  # queue empty
            if msg == "show":
                make_visible = True
            elif msg == "show_front":
                make_visible = True
                raise_to_front = True
            elif msg == "hide":
                hide = True
        if hide and not make_visible:
            self.hide()
            return
        if not make_visible:
            return
        # Always call show() + showNormal() unconditionally - after a
        # previous closeEvent->hide() a single show() call doesn't
        # always re-display the window on every platform, and
        # showNormal() additionally brings it out of a minimised state.
        # We deliberately do NOT toggle setWindowFlags() - that call
        # implicitly hides the widget (Qt docs).
        self.show()
        self.showNormal()
        self.raise_()
        if raise_to_front:
            self.activateWindow()

    def setWindowStyleFusion(self):
        # Set the Fusion style
        QtWidgets.QApplication.setStyle("Fusion")


def run_camera_window(variables, conf, camera_closed_event, camera_command_queue):
    """
    Run the Cameras window in a separate process.

    Args:
            camera_command_queue: multiprocessing.Queue of typed string commands
                    from the main GUI ("show", "show_front", "hide").
    """
    # A startup crash in the subprocess otherwise dies silently - the
    # parent never sees it.  Funnel any exception to a log file under
    # files/logs/ so the user can pick it up after the fact.
    import traceback

    try:
        app = QtWidgets.QApplication(sys.argv)
        app.setStyle('Fusion')
        # The window starts hidden and only appears when the main GUI signals -
        # don't let Qt quit the subprocess just because no window is visible.
        app.setQuitOnLastWindowClosed(False)
        SignalEmitter_Cameras = SignalEmitter()

        gui_cameras_alignment = Ui_Cameras_Alignment(variables, conf, SignalEmitter_Cameras)
        Cameras_alignment = CamerasAlignmentWindow(
            variables, gui_cameras_alignment, camera_closed_event, camera_command_queue, flags=QtCore.Qt.WindowType.Tool
        )
        gui_cameras_alignment.setupUi(Cameras_alignment)
        sys.exit(app.exec())
    except Exception:
        try:
            log_path = runtime.project_path("files", "logs", "camera_subprocess_crash.log")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write("=" * 60 + "\n")
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
    Cameras_Alignment = QtWidgets.QWidget()
    signal_emitter = SignalEmitter()
    ui = Ui_Cameras_Alignment(shared.variables, conf, signal_emitter)
    ui.setupUi(Cameras_Alignment)
    Cameras_Alignment.show()
    sys.exit(app.exec())
