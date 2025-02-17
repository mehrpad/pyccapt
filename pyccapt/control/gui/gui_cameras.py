import multiprocessing
import os
import sys

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import pyqtSignal, QObject, QThread
from PyQt6.QtGui import QPixmap

# Local module and scripts
from pyccapt.control.control import share_variables, read_files
from pyccapt.control.devices import camera
from pyccapt.control.usb_switch import usb_switch


class Ui_Cameras_Alignment(object):

	def __init__(self, variables, conf, SignalEmitter):
		"""
		Initialize the UiCamerasAlignment class.

		Args:
			variables: Global experiment variables.
			conf: Configuration data.
			SignalEmitter: Signal emitter for communication.
		"""
		self.auto_exposure_time_flag = False
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
		Cameras_Alignment.resize(1210, 938)
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
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
		                                   QtWidgets.QSizePolicy.Policy.Expanding)
		sizePolicy.setHorizontalStretch(2)
		sizePolicy.setVerticalStretch(1)
		sizePolicy.setHeightForWidth(self.cam_s_d.sizePolicy().hasHeightForWidth())
		self.cam_s_d.setSizePolicy(sizePolicy)
		self.cam_s_d.setMinimumSize(QtCore.QSize(600, 250))
		self.cam_s_d.setMaximumSize(QtCore.QSize(16777215, 16777215))
		self.cam_s_d.setStyleSheet("QWidget{\n"
		                           "                                            border: 2px solid gray;\n"
		                           "                                            }\n"
		                           "                                        ")
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
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
		                                   QtWidgets.QSizePolicy.Policy.Expanding)
		sizePolicy.setHorizontalStretch(2)
		sizePolicy.setVerticalStretch(1)
		sizePolicy.setHeightForWidth(self.cam_b_d.sizePolicy().hasHeightForWidth())
		self.cam_b_d.setSizePolicy(sizePolicy)
		self.cam_b_d.setMinimumSize(QtCore.QSize(600, 250))
		self.cam_b_d.setMaximumSize(QtCore.QSize(16777215, 16777215))
		self.cam_b_d.setStyleSheet("QWidget{\n"
		                           "                                            border: 2px solid gray;\n"
		                           "                                            }\n"
		                           "                                        ")
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
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
		                                   QtWidgets.QSizePolicy.Policy.Expanding)
		sizePolicy.setHorizontalStretch(1)
		sizePolicy.setVerticalStretch(1)
		sizePolicy.setHeightForWidth(self.cam_s_o.sizePolicy().hasHeightForWidth())
		self.cam_s_o.setSizePolicy(sizePolicy)
		self.cam_s_o.setMinimumSize(QtCore.QSize(250, 250))
		self.cam_s_o.setStyleSheet("QWidget{\n"
		                           "                                            border: 2px solid gray;\n"
		                           "                                            }\n"
		                           "                                        ")
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
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
		                                   QtWidgets.QSizePolicy.Policy.Expanding)
		sizePolicy.setHorizontalStretch(1)
		sizePolicy.setVerticalStretch(1)
		sizePolicy.setHeightForWidth(self.cam_b_o.sizePolicy().hasHeightForWidth())
		self.cam_b_o.setSizePolicy(sizePolicy)
		self.cam_b_o.setMinimumSize(QtCore.QSize(250, 250))
		self.cam_b_o.setMaximumSize(QtCore.QSize(16777215, 16777215))
		self.cam_b_o.setStyleSheet("QWidget{\n"
		                           "                                            border: 2px solid gray;\n"
		                           "                                            }\n"
		                           "                                        ")
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
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
		                                   QtWidgets.QSizePolicy.Policy.Expanding)
		sizePolicy.setHorizontalStretch(1)
		sizePolicy.setVerticalStretch(1)
		sizePolicy.setHeightForWidth(self.cam_angle_o.sizePolicy().hasHeightForWidth())
		self.cam_angle_o.setSizePolicy(sizePolicy)
		self.cam_angle_o.setMinimumSize(QtCore.QSize(250, 250))
		self.cam_angle_o.setMaximumSize(QtCore.QSize(16777215, 16777215))
		self.cam_angle_o.setStyleSheet("QWidget{\n"
		                               "                                            border: 2px solid gray;\n"
		                               "                                            }\n"
		                               "                                        ")
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
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
		                                   QtWidgets.QSizePolicy.Policy.Expanding)
		sizePolicy.setHorizontalStretch(2)
		sizePolicy.setVerticalStretch(1)
		sizePolicy.setHeightForWidth(self.cam_angle_d.sizePolicy().hasHeightForWidth())
		self.cam_angle_d.setSizePolicy(sizePolicy)
		self.cam_angle_d.setMinimumSize(QtCore.QSize(600, 250))
		self.cam_angle_d.setMaximumSize(QtCore.QSize(16777215, 16777215))
		self.cam_angle_d.setStyleSheet("QWidget{\n"
		                               "                                            border: 2px solid gray;\n"
		                               "                                            }\n"
		                               "                                        ")
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
		self.auto_exposure_time = QtWidgets.QPushButton(parent=Cameras_Alignment)
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
		sizePolicy.setHorizontalStretch(0)
		sizePolicy.setVerticalStretch(0)
		sizePolicy.setHeightForWidth(self.auto_exposure_time.sizePolicy().hasHeightForWidth())
		self.auto_exposure_time.setSizePolicy(sizePolicy)
		self.auto_exposure_time.setMinimumSize(QtCore.QSize(0, 25))
		self.auto_exposure_time.setStyleSheet("QPushButton{\n"
		                                      "                                            background: rgb(193, 193, 193)\n"
		                                      "                                            }\n"
		                                      "                                        ")
		self.auto_exposure_time.setObjectName("auto_exposure_time")
		self.horizontalLayout.addWidget(self.auto_exposure_time)
		spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
		                                   QtWidgets.QSizePolicy.Policy.Minimum)
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
		self.light.setStyleSheet("QPushButton{\n"
		                         "                                            background: rgb(193, 193, 193)\n"
		                         "                                            }\n"
		                         "                                        ")
		self.light.setObjectName("light")
		self.horizontalLayout.addWidget(self.light)
		spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
		                                    QtWidgets.QSizePolicy.Policy.Minimum)
		self.horizontalLayout.addItem(spacerItem1)
		self.verticalLayout_2.addLayout(self.horizontalLayout)
		self.gridLayout_2 = QtWidgets.QGridLayout()
		self.gridLayout_2.setObjectName("gridLayout_2")
		self.led_light_2 = QtWidgets.QLabel(parent=Cameras_Alignment)
		self.led_light_2.setMinimumSize(QtCore.QSize(130, 0))
		self.led_light_2.setMaximumSize(QtCore.QSize(500, 50))
		self.led_light_2.setObjectName("led_light_2")
		self.gridLayout_2.addWidget(self.led_light_2, 1, 0, 1, 1)
		self.exposure_time_cam_1 = QtWidgets.QLineEdit(parent=Cameras_Alignment)
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
		sizePolicy.setHorizontalStretch(0)
		sizePolicy.setVerticalStretch(0)
		sizePolicy.setHeightForWidth(self.exposure_time_cam_1.sizePolicy().hasHeightForWidth())
		self.exposure_time_cam_1.setSizePolicy(sizePolicy)
		self.exposure_time_cam_1.setMinimumSize(QtCore.QSize(0, 20))
		self.exposure_time_cam_1.setStyleSheet("QLineEdit{\n"
		                                       "                                                background: rgb(223,223,233)\n"
		                                       "                                                }\n"
		                                       "                                            ")
		self.exposure_time_cam_1.setObjectName("exposure_time_cam_1")
		self.gridLayout_2.addWidget(self.exposure_time_cam_1, 1, 1, 1, 1)
		spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum,
		                                    QtWidgets.QSizePolicy.Policy.Expanding)
		self.gridLayout_2.addItem(spacerItem2, 4, 1, 1, 1)
		self.exposure_time_cam_2 = QtWidgets.QLineEdit(parent=Cameras_Alignment)
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
		sizePolicy.setHorizontalStretch(0)
		sizePolicy.setVerticalStretch(0)
		sizePolicy.setHeightForWidth(self.exposure_time_cam_2.sizePolicy().hasHeightForWidth())
		self.exposure_time_cam_2.setSizePolicy(sizePolicy)
		self.exposure_time_cam_2.setMinimumSize(QtCore.QSize(0, 20))
		self.exposure_time_cam_2.setStyleSheet("QLineEdit{\n"
		                                       "                                                background: rgb(223,223,233)\n"
		                                       "                                                }\n"
		                                       "                                            ")
		self.exposure_time_cam_2.setObjectName("exposure_time_cam_2")
		self.gridLayout_2.addWidget(self.exposure_time_cam_2, 2, 1, 1, 1)
		self.exposure_time_cam_3 = QtWidgets.QLineEdit(parent=Cameras_Alignment)
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
		sizePolicy.setHorizontalStretch(0)
		sizePolicy.setVerticalStretch(0)
		sizePolicy.setHeightForWidth(self.exposure_time_cam_3.sizePolicy().hasHeightForWidth())
		self.exposure_time_cam_3.setSizePolicy(sizePolicy)
		self.exposure_time_cam_3.setMinimumSize(QtCore.QSize(0, 20))
		self.exposure_time_cam_3.setStyleSheet("QLineEdit{\n"
		                                       "                                                background: rgb(223,223,233)\n"
		                                       "                                                }\n"
		                                       "                                            ")
		self.exposure_time_cam_3.setObjectName("exposure_time_cam_3")
		self.gridLayout_2.addWidget(self.exposure_time_cam_3, 3, 1, 1, 1)
		self.led_light_3 = QtWidgets.QLabel(parent=Cameras_Alignment)
		self.led_light_3.setMinimumSize(QtCore.QSize(130, 0))
		self.led_light_3.setMaximumSize(QtCore.QSize(500, 50))
		self.led_light_3.setObjectName("led_light_3")
		self.gridLayout_2.addWidget(self.led_light_3, 2, 0, 1, 1)
		self.default_exposure_time = QtWidgets.QPushButton(parent=Cameras_Alignment)
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
		sizePolicy.setHorizontalStretch(0)
		sizePolicy.setVerticalStretch(0)
		sizePolicy.setHeightForWidth(self.default_exposure_time.sizePolicy().hasHeightForWidth())
		self.default_exposure_time.setSizePolicy(sizePolicy)
		self.default_exposure_time.setMinimumSize(QtCore.QSize(0, 25))
		self.default_exposure_time.setStyleSheet("QPushButton{\n"
		                                         "                                            background: rgb(193, 193, 193)\n"
		                                         "                                            }\n"
		                                         "                                        ")
		self.default_exposure_time.setObjectName("default_exposure_time")
		self.gridLayout_2.addWidget(self.default_exposure_time, 0, 0, 1, 1, QtCore.Qt.AlignmentFlag.AlignHCenter)
		self.led_light_4 = QtWidgets.QLabel(parent=Cameras_Alignment)
		self.led_light_4.setMinimumSize(QtCore.QSize(130, 0))
		self.led_light_4.setMaximumSize(QtCore.QSize(500, 50))
		self.led_light_4.setObjectName("led_light_4")
		self.gridLayout_2.addWidget(self.led_light_4, 3, 0, 1, 1)
		self.verticalLayout_2.addLayout(self.gridLayout_2)
		self.gridLayout_4.addLayout(self.verticalLayout_2, 0, 1, 1, 1)
		self.gridLayout_5.addLayout(self.gridLayout_4, 0, 0, 1, 1)

		self.retranslateUi(Cameras_Alignment)
		QtCore.QMetaObject.connectSlotsByName(Cameras_Alignment)
		Cameras_Alignment.setTabOrder(self.auto_exposure_time, self.light)
		Cameras_Alignment.setTabOrder(self.light, self.default_exposure_time)
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
		arrow1 = pg.ArrowItem(pos=(805, 735), angle=-90, brush='r')
		arrow2 = pg.ArrowItem(pos=(795, 850), angle=-90)
		arrow3 = pg.ArrowItem(pos=(940, 570), angle=0, brush='g')

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
		self.light.clicked.connect(self.light_switch)
		self.auto_exposure_time.clicked.connect(self.auto_exposure_time_switch)
		self.default_exposure_time.clicked.connect(self.default_exposure_time_switch)

		self.emitter.img0_orig.connect(self.update_cam_s_o)
		self.emitter.img1_orig.connect(self.update_cam_b_o)
		self.emitter.img2_orig.connect(self.update_cam_angle_o)
		self.initialize_camera_thread()

		if self.conf['usb_lamp_switch'] == 'on':
			self.usb_lamp_switch = usb_switch.USBSwitch("./control/usb_switch/USBaccessX64.dll")  # 32 bit w/o X64

		self.exposure_time_cam_1.editingFinished.connect(self.update_exposure_time)
		self.exposure_time_cam_2.editingFinished.connect(self.update_exposure_time)
		self.exposure_time_cam_3.editingFinished.connect(self.update_exposure_time)

		self.original_button_style = self.auto_exposure_time.styleSheet()

		self.emitter.cams_exposure_time_default.connect(self.set_default_exposure_time)
		# switch off the light if it is one before opening the window
		self.usb_lamp_switch.switch_off(16)

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
		self.auto_exposure_time.setText(_translate("Cameras_Alignment", "Auto Exposure Time"))
		self.led_light.setText(_translate("Cameras_Alignment", "Light"))
		self.light.setText(_translate("Cameras_Alignment", "Light"))
		self.led_light_2.setText(_translate("Cameras_Alignment", "Exposure Time Side (us)"))
		self.exposure_time_cam_1.setText(_translate("Cameras_Alignment", "400000"))
		self.exposure_time_cam_2.setText(_translate("Cameras_Alignment", "1000000"))
		self.exposure_time_cam_3.setText(_translate("Cameras_Alignment", "400000"))
		self.led_light_3.setText(_translate("Cameras_Alignment", "Exposure Time Top (us)"))
		self.default_exposure_time.setText(_translate("Cameras_Alignment", "Default Exposure Time"))
		self.led_light_4.setText(_translate("Cameras_Alignment", "Exposure Time Angle (us)"))

		###
		self.timer = QtCore.QTimer()
		self.timer.timeout.connect(self.cameras_screenshot)
		self.timer.start(2000)  # Check every 2000 milliseconds (1 second)

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

	def update_exposure_time(self):
		"""
		Update the exposure time of the cameras

		Args:
		None

		Return:
		None
		"""
		try:
			if self.exposure_time_cam_1.text() != '':
				self.emitter.cam_1_exposure_time.emit(int(self.exposure_time_cam_1.text()))
			if self.exposure_time_cam_2.text() != '':
				self.emitter.cam_2_exposure_time.emit(int(self.exposure_time_cam_2.text()))
			if self.exposure_time_cam_3.text() != '':
				self.emitter.cam_3_exposure_time.emit(int(self.exposure_time_cam_3.text()))
		except Exception as e:
			print(e)
			print('type the exposure time in microseconds')

	def update_cam_s_o(self, img):
		self.cam_s_o.setImage(img, autoRange=False)
		roi_coords = self.roi_s.getArraySlice(img, self.cam_s_o.imageItem, axes=(0, 1))

		# Extract the region and update the second image view
		if roi_coords is not None:
			region = img[roi_coords[0][0], roi_coords[0][1]]
			self.cam_s_d.setImage(region, autoRange=False)

	def update_cam_b_o(self, img):
		self.cam_b_o.setImage(img, autoRange=False)
		roi_coords = self.roi_b.getArraySlice(img, self.cam_b_o.imageItem, axes=(0, 1))

		# Extract the region and update the second image view
		if roi_coords is not None:
			region = img[roi_coords[0][0], roi_coords[0][1]]
			self.cam_b_d.setImage(region, autoRange=False)

	def update_cam_angle_o(self, img):
		self.cam_angle_o.setImage(img, autoRange=False)
		roi_coords = self.roi_angle.getArraySlice(img, self.cam_angle_o.imageItem, axes=(0, 1))

		# Extract the region and update the second image view
		if roi_coords is not None:
			region = img[roi_coords[0][0], roi_coords[0][1]]
			self.cam_angle_d.setImage(region, autoRange=False)

	def light_switch(self):
		"""
		light switch function

		Args:
		None

		Return:
		None
		"""
		if not self.variables.light:
			self.led_light.setPixmap(self.led_green)
			if self.conf['usb_lamp_switch'] == 'on':
				self.usb_lamp_switch.switch_on(16)
			self.variables.light = True
			self.variables.light_switch = True

		elif self.variables.light:
			self.led_light.setPixmap(self.led_red)
			if self.conf['usb_lamp_switch'] == 'on':
				self.usb_lamp_switch.switch_off(16)
			self.variables.light = False
			self.variables.light_switch = True

	def auto_exposure_time_switch(self):
		"""
		Auto exposure time switch function

		Args:
		None

		Return:
		None
		"""
		self.auto_exposure_time_flag = not self.auto_exposure_time_flag
		if self.auto_exposure_time_flag:
			self.auto_exposure_time.setStyleSheet("QPushButton{\n"
			                                      "background: rgb(0, 255, 26)\n"
			                                      "}")
		else:
			self.auto_exposure_time.setStyleSheet(self.original_button_style)
		self.emitter.auto_exposure_time.emit(True)

	def default_exposure_time_switch(self):
		"""
		Default exposure time switch function

		Args:
		None

		Return:
		None
		"""
		self.emitter.default_exposure_time.emit(True)

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
		else:
			# Create cameras thread
			# Thread for reading cameras
			# Create a camera instance and move it to a new thread
			self.camera_worker = camera.CameraWorker(variables=self.variables, emitter=self.emitter)

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
		self.camera_thread.join()

	def cameras_screenshot(self):
		if self.variables.flag_cameras_take_screenshot:
			screenshot = QtWidgets.QApplication.primaryScreen().grabWindow(self.Cameras_Alignment.winId())
			screenshot.save(self.variables.path_meta + '\cameras_screenshot.png', 'png')


class SignalEmitter(QObject):
	img0_orig = pyqtSignal(np.ndarray)
	img1_orig = pyqtSignal(np.ndarray)
	img2_orig = pyqtSignal(np.ndarray)
	cam_1_exposure_time = pyqtSignal(int)
	cam_2_exposure_time = pyqtSignal(int)
	cam_3_exposure_time = pyqtSignal(int)
	cams_exposure_time_default = pyqtSignal(list)
	default_exposure_time = pyqtSignal(bool)
	auto_exposure_time = pyqtSignal(bool)


class CamerasAlignmentWindow(QtWidgets.QWidget):
	closed = QtCore.pyqtSignal()  # Define a custom closed signal

	def __init__(self, variables, gui_cameras_alignment, close_event,
	             camera_win_front, *args, **kwargs):
		"""
		Initialize the CamerasAlignmentWindow class.

		Args:
				gui_cameras_alignment: An instance of the GUI cameras alignment class.
				*args: Variable length argument list.
				**kwargs: Arbitrary keyword arguments.
		"""
		super().__init__(*args, **kwargs)
		self.variables = variables
		self.gui_cameras_alignment = gui_cameras_alignment
		self.camera_win_front = camera_win_front
		self.close_event = close_event
		self.show()
		self.showMinimized()
		self.timer = QtCore.QTimer(self)
		self.timer.timeout.connect(self.check_if_should)
		self.timer.start(500)  # Check every 1000 milliseconds (1 second)

	def closeEvent(self, event):
		"""
		Not close only hide the window

		Args:
				event: The close event.
		"""
		event.ignore()
		self.showMinimized()
		self.close_event.set()

	def check_if_should(self):
		if self.camera_win_front.is_set():
			self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint)
			self.show()
			self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowType.WindowStaysOnTopHint)
			self.camera_win_front.clear()  # Reset the flag
		if self.variables.flag_camera_win_show:
			self.show()
			self.variables.flag_camera_win_show = False

	def setWindowStyleFusion(self):
		# Set the Fusion style
		QtWidgets.QApplication.setStyle("Fusion")


def run_camera_window(variables, conf, camera_closed_event, camera_win_front):
	"""
	Run the Cameras window in a separate process.
	"""
	app = QtWidgets.QApplication(sys.argv)  # <-- Create a new QApplication instance
	app.setStyle('Fusion')
	SignalEmitter_Cameras = SignalEmitter()

	gui_cameras_alignment = Ui_Cameras_Alignment(variables, conf, SignalEmitter_Cameras)
	Cameras_alignment = CamerasAlignmentWindow(variables, gui_cameras_alignment, camera_closed_event, camera_win_front,
	                                           flags=QtCore.Qt.WindowType.Tool)
	gui_cameras_alignment.setupUi(Cameras_alignment)
	# Cameras_alignment.show()

	sys.exit(app.exec())  # <-- Start the event loop for this QApplication instance


if __name__ == "__main__":
	try:
		# Load the JSON file
		configFile = 'config.json'
		p = os.path.abspath(os.path.join(__file__, "../../.."))
		os.chdir(p)
		conf = read_files.read_json_file(configFile)
	except Exception as e:
		print('Can not load the configuration file')
		print(e)
		sys.exit()

	# Initialize global experiment variables
	manager = multiprocessing.Manager()
	ns = manager.Namespace()
	variables = share_variables.Variables(conf, ns)

	app = QtWidgets.QApplication(sys.argv)
	app.setStyle('Fusion')
	Cameras_Alignment = QtWidgets.QWidget()
	signal_emitter = SignalEmitter()
	ui = Ui_Cameras_Alignment(variables, conf, signal_emitter)
	ui.setupUi(Cameras_Alignment)
	Cameras_Alignment.show()
	sys.exit(app.exec())
