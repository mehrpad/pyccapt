import multiprocessing
import os
import sys
import threading
import time

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QPixmap

# Local module and scripts
from pyccapt.control.control import share_variables, read_files
from pyccapt.control.devices import initialize_devices


class Ui_Pumps_Vacuum(object):

    def __init__(self, variables, conf, SignalEmitter, parent=None):
        """
        Constructor for the Pumps and Vacuum UI class.

        Args:
                        variables (object): Global experiment variables.
                        conf (dict): Configuration settings.
                        SignalEmitter (object): Emitter for signals.
                        parent: Parent widget (optional).

        Return:
                        None
        """
        self.flag_super_user = None
        self.default_color = None
        self.variables = variables
        self.conf = conf
        self.parent = parent
        self.emitter = SignalEmitter

    def setupUi(self, Pumps_Vacuum):
        """
        Sets up the UI for the Pumps and Vacuum tab.
        Args:
                Pumps_Vacuum (object): Pumps and Vacuum tab widget.

        Return:
                None
        """
        Pumps_Vacuum.setObjectName("Pumps_Vacuum")
        Pumps_Vacuum.resize(757, 385)
        self.gridLayout_9 = QtWidgets.QGridLayout(Pumps_Vacuum)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_212 = QtWidgets.QLabel(parent=Pumps_Vacuum)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_212.setFont(font)
        self.label_212.setObjectName("label_212")
        self.gridLayout.addWidget(self.label_212, 0, 0, 1, 1)
        self.vacuum_main = QtWidgets.QLCDNumber(parent=Pumps_Vacuum)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                           QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.vacuum_main.sizePolicy().hasHeightForWidth())
        self.vacuum_main.setSizePolicy(sizePolicy)
        self.vacuum_main.setMinimumSize(QtCore.QSize(200, 50))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.vacuum_main.setFont(font)
        self.vacuum_main.setStyleSheet("QLCDNumber{\n"
                                       "                                    border: 2px solid green;\n"
                                       "                                    border-radius: 10px;\n"
                                       "                                    padding: 0 8px;\n"
                                       "                                    }\n"
                                       "                                ")
        self.vacuum_main.setObjectName("vacuum_main")
        self.gridLayout.addWidget(self.vacuum_main, 0, 1, 1, 2)
        self.label_211 = QtWidgets.QLabel(parent=Pumps_Vacuum)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_211.setFont(font)
        self.label_211.setObjectName("label_211")
        self.gridLayout.addWidget(self.label_211, 1, 0, 1, 2)
        self.vacuum_buffer = QtWidgets.QLCDNumber(parent=Pumps_Vacuum)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                           QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.vacuum_buffer.sizePolicy().hasHeightForWidth())
        self.vacuum_buffer.setSizePolicy(sizePolicy)
        self.vacuum_buffer.setMinimumSize(QtCore.QSize(150, 50))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.vacuum_buffer.setFont(font)
        self.vacuum_buffer.setStyleSheet("QLCDNumber{\n"
                                         "                                            border: 2px solid brown;\n"
                                         "                                            border-radius: 10px;\n"
                                         "                                            padding: 0 8px;\n"
                                         "                                            }\n"
                                         "                                        ")
        self.vacuum_buffer.setObjectName("vacuum_buffer")
        self.gridLayout.addWidget(self.vacuum_buffer, 1, 2, 1, 1)
        self.label_216 = QtWidgets.QLabel(parent=Pumps_Vacuum)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_216.setFont(font)
        self.label_216.setObjectName("label_216")
        self.gridLayout.addWidget(self.label_216, 2, 0, 1, 2)
        self.vacuum_cryo_load_lock = QtWidgets.QLCDNumber(parent=Pumps_Vacuum)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                           QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.vacuum_cryo_load_lock.sizePolicy().hasHeightForWidth())
        self.vacuum_cryo_load_lock.setSizePolicy(sizePolicy)
        self.vacuum_cryo_load_lock.setMinimumSize(QtCore.QSize(150, 50))
        self.vacuum_cryo_load_lock.setStyleSheet("QLCDNumber{\n"
                                                 "                                            border: 2px solid magenta;\n"
                                                 "                                            border-radius: 10px;\n"
                                                 "                                            padding: 0 8px;\n"
                                                 "                                            }\n"
                                                 "                                        ")
        self.vacuum_cryo_load_lock.setObjectName("vacuum_cryo_load_lock")
        self.gridLayout.addWidget(self.vacuum_cryo_load_lock, 2, 2, 1, 1)
        self.label_210 = QtWidgets.QLabel(parent=Pumps_Vacuum)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_210.setFont(font)
        self.label_210.setObjectName("label_210")
        self.gridLayout.addWidget(self.label_210, 3, 0, 1, 2)
        self.vacuum_load_lock = QtWidgets.QLCDNumber(parent=Pumps_Vacuum)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                           QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.vacuum_load_lock.sizePolicy().hasHeightForWidth())
        self.vacuum_load_lock.setSizePolicy(sizePolicy)
        self.vacuum_load_lock.setMinimumSize(QtCore.QSize(150, 50))
        self.vacuum_load_lock.setStyleSheet("QLCDNumber{\n"
                                            "                                            border: 2px solid blue;\n"
                                            "                                                            border-radius: 10px;\n"
                                            "                                                            padding: 0 8px;\n"
                                            "                                                            }\n"
                                            "                                        ")
        self.vacuum_load_lock.setObjectName("vacuum_load_lock")
        self.gridLayout.addWidget(self.vacuum_load_lock, 3, 2, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout, 0, 0, 2, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
                                           QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout_4.addItem(spacerItem, 0, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum,
                                            QtWidgets.QSizePolicy.Policy.Expanding)
        self.gridLayout_4.addItem(spacerItem1, 0, 2, 1, 1)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_214 = QtWidgets.QLabel(parent=Pumps_Vacuum)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_214.setFont(font)
        self.label_214.setObjectName("label_214")
        self.gridLayout_2.addWidget(self.label_214, 0, 0, 1, 1)
        self.vacuum_buffer_back = QtWidgets.QLCDNumber(parent=Pumps_Vacuum)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                           QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.vacuum_buffer_back.sizePolicy().hasHeightForWidth())
        self.vacuum_buffer_back.setSizePolicy(sizePolicy)
        self.vacuum_buffer_back.setMinimumSize(QtCore.QSize(150, 50))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.vacuum_buffer_back.setFont(font)
        self.vacuum_buffer_back.setStyleSheet("QLCDNumber{\n"
                                              "                                            border: 2px solid brown;\n"
                                              "                                                            border-radius: 10px;\n"
                                              "                                                            padding: 0 8px;\n"
                                              "                                                            }\n"
                                              "                                                        ")
        self.vacuum_buffer_back.setObjectName("vacuum_buffer_back")
        self.gridLayout_2.addWidget(self.vacuum_buffer_back, 0, 1, 1, 1)
        self.label_217 = QtWidgets.QLabel(parent=Pumps_Vacuum)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_217.setFont(font)
        self.label_217.setObjectName("label_217")
        self.gridLayout_2.addWidget(self.label_217, 1, 0, 1, 1)
        self.vacuum_cryo_load_lock_back = QtWidgets.QLCDNumber(parent=Pumps_Vacuum)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                           QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.vacuum_cryo_load_lock_back.sizePolicy().hasHeightForWidth())
        self.vacuum_cryo_load_lock_back.setSizePolicy(sizePolicy)
        self.vacuum_cryo_load_lock_back.setMinimumSize(QtCore.QSize(150, 50))
        self.vacuum_cryo_load_lock_back.setStyleSheet("QLCDNumber{\n"
                                                      "                                            border: 2px solid magenta;\n"
                                                      "                                            border-radius: 10px;\n"
                                                      "                                            padding: 0 8px;\n"
                                                      "                                            }\n"
                                                      "                                        ")
        self.vacuum_cryo_load_lock_back.setObjectName("vacuum_cryo_load_lock_back")
        self.gridLayout_2.addWidget(self.vacuum_cryo_load_lock_back, 1, 1, 1, 1)
        self.label_213 = QtWidgets.QLabel(parent=Pumps_Vacuum)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_213.setFont(font)
        self.label_213.setObjectName("label_213")
        self.gridLayout_2.addWidget(self.label_213, 2, 0, 1, 1)
        self.vacuum_load_lock_back = QtWidgets.QLCDNumber(parent=Pumps_Vacuum)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                           QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.vacuum_load_lock_back.sizePolicy().hasHeightForWidth())
        self.vacuum_load_lock_back.setSizePolicy(sizePolicy)
        self.vacuum_load_lock_back.setMinimumSize(QtCore.QSize(150, 50))
        self.vacuum_load_lock_back.setStyleSheet("QLCDNumber{\n"
                                                 "                                            border: 2px solid blue;\n"
                                                 "                                            border-radius: 10px;\n"
                                                 "                                            padding: 0 8px;\n"
                                                 "                                            }\n"
                                                 "                                        ")
        self.vacuum_load_lock_back.setObjectName("vacuum_load_lock_back")
        self.gridLayout_2.addWidget(self.vacuum_load_lock_back, 2, 1, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_2, 1, 1, 1, 2)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.superuser = QtWidgets.QPushButton(parent=Pumps_Vacuum)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.superuser.sizePolicy().hasHeightForWidth())
        self.superuser.setSizePolicy(sizePolicy)
        self.superuser.setMinimumSize(QtCore.QSize(0, 25))
        self.superuser.setStyleSheet("QPushButton{\n"
                                     "                                    background: rgb(193, 193, 193)\n"
                                     "                                    }\n"
                                     "                                ")
        self.superuser.setObjectName("superuser")
        self.gridLayout_3.addWidget(self.superuser, 0, 0, 1, 2)
        self.pump_cryo_load_lock_switch = QtWidgets.QPushButton(parent=Pumps_Vacuum)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pump_cryo_load_lock_switch.sizePolicy().hasHeightForWidth())
        self.pump_cryo_load_lock_switch.setSizePolicy(sizePolicy)
        self.pump_cryo_load_lock_switch.setMinimumSize(QtCore.QSize(0, 25))
        self.pump_cryo_load_lock_switch.setStyleSheet("QPushButton{\n"
                                                      "                                            background: rgb(193, 193, 193)\n"
                                                      "                                            }\n"
                                                      "                                        ")
        self.pump_cryo_load_lock_switch.setObjectName("pump_cryo_load_lock_switch")
        self.gridLayout_3.addWidget(self.pump_cryo_load_lock_switch, 1, 0, 1, 1)
        self.led_pump_cryo_load_lock = QtWidgets.QLabel(parent=Pumps_Vacuum)
        self.led_pump_cryo_load_lock.setMinimumSize(QtCore.QSize(50, 50))
        self.led_pump_cryo_load_lock.setMaximumSize(QtCore.QSize(50, 50))
        self.led_pump_cryo_load_lock.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.led_pump_cryo_load_lock.setObjectName("led_pump_cryo_load_lock")
        self.gridLayout_3.addWidget(self.led_pump_cryo_load_lock, 1, 1, 1, 1)
        self.pump_load_lock_switch = QtWidgets.QPushButton(parent=Pumps_Vacuum)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pump_load_lock_switch.sizePolicy().hasHeightForWidth())
        self.pump_load_lock_switch.setSizePolicy(sizePolicy)
        self.pump_load_lock_switch.setMinimumSize(QtCore.QSize(0, 25))
        self.pump_load_lock_switch.setStyleSheet("QPushButton{\n"
                                                 "                                            background: rgb(193, 193, 193)\n"
                                                 "                                            }\n"
                                                 "                                        ")
        self.pump_load_lock_switch.setObjectName("pump_load_lock_switch")
        self.gridLayout_3.addWidget(self.pump_load_lock_switch, 2, 0, 1, 1)
        self.led_pump_load_lock = QtWidgets.QLabel(parent=Pumps_Vacuum)
        self.led_pump_load_lock.setMinimumSize(QtCore.QSize(50, 50))
        self.led_pump_load_lock.setMaximumSize(QtCore.QSize(50, 50))
        self.led_pump_load_lock.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.led_pump_load_lock.setObjectName("led_pump_load_lock")
        self.gridLayout_3.addWidget(self.led_pump_load_lock, 2, 1, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_3, 1, 3, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_4)
        self.gridLayout_8 = QtWidgets.QGridLayout()
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.gridLayout_6 = QtWidgets.QGridLayout()
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.label_215 = QtWidgets.QLabel(parent=Pumps_Vacuum)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_215.setFont(font)
        self.label_215.setObjectName("label_215")
        self.gridLayout_6.addWidget(self.label_215, 0, 0, 1, 1)
        self.temp_stage = QtWidgets.QLCDNumber(parent=Pumps_Vacuum)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                           QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.temp_stage.sizePolicy().hasHeightForWidth())
        self.temp_stage.setSizePolicy(sizePolicy)
        self.temp_stage.setMinimumSize(QtCore.QSize(150, 50))
        self.temp_stage.setStyleSheet("QLCDNumber{\n"
                                      "                                                            border: 2px solid orange;\n"
                                      "                                                            border-radius: 10px;\n"
                                      "                                                            padding: 0 8px;\n"
                                      "                                                            }\n"
                                      "                                        ")
        self.temp_stage.setObjectName("temp_stage")
        self.gridLayout_6.addWidget(self.temp_stage, 0, 1, 1, 2)
        self.set_temperature_cryo = QtWidgets.QPushButton(parent=Pumps_Vacuum)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.set_temperature_cryo.sizePolicy().hasHeightForWidth())
        self.set_temperature_cryo.setSizePolicy(sizePolicy)
        self.set_temperature_cryo.setMinimumSize(QtCore.QSize(0, 25))
        self.set_temperature_cryo.setStyleSheet("QPushButton{\n"
                                                "                                    background: rgb(193, 193, 193)\n"
                                                "                                    }\n"
                                                "                                ")
        self.set_temperature_cryo.setObjectName("set_temperature_cryo")
        self.gridLayout_6.addWidget(self.set_temperature_cryo, 1, 0, 1, 2)
        self.target_tempreature_cryo = QtWidgets.QSpinBox(parent=Pumps_Vacuum)
        self.target_tempreature_cryo.setMinimumSize(QtCore.QSize(150, 0))
        self.target_tempreature_cryo.setMaximumSize(QtCore.QSize(70, 16777215))
        self.target_tempreature_cryo.setStyleSheet("QSpinBox{\n"
                                                   "                                    background: rgb(223,223,233)\n"
                                                   "                                    }\n"
                                                   "                                ")
        self.target_tempreature_cryo.setObjectName("target_tempreature_cryo")
        self.gridLayout_6.addWidget(self.target_tempreature_cryo, 1, 2, 1, 1)
        self.gridLayout_8.addLayout(self.gridLayout_6, 0, 0, 1, 1)
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.label_219 = QtWidgets.QLabel(parent=Pumps_Vacuum)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_219.setFont(font)
        self.label_219.setObjectName("label_219")
        self.gridLayout_5.addWidget(self.label_219, 0, 0, 1, 1)
        self.temp_ll = QtWidgets.QLCDNumber(parent=Pumps_Vacuum)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                           QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.temp_ll.sizePolicy().hasHeightForWidth())
        self.temp_ll.setSizePolicy(sizePolicy)
        self.temp_ll.setMinimumSize(QtCore.QSize(150, 50))
        self.temp_ll.setStyleSheet("QLCDNumber{\n"
                                   "                                    border: 2px solid orange;\n"
                                   "                                    border-radius: 10px;\n"
                                   "                                    padding: 0 8px;\n"
                                   "                                    }\n"
                                   "                                ")
        self.temp_ll.setObjectName("temp_ll")
        self.gridLayout_5.addWidget(self.temp_ll, 0, 1, 1, 2)
        self.set_temperature_ll = QtWidgets.QPushButton(parent=Pumps_Vacuum)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.set_temperature_ll.sizePolicy().hasHeightForWidth())
        self.set_temperature_ll.setSizePolicy(sizePolicy)
        self.set_temperature_ll.setMinimumSize(QtCore.QSize(0, 25))
        self.set_temperature_ll.setStyleSheet("QPushButton{\n"
                                              "                                    background: rgb(193, 193, 193)\n"
                                              "                                    }\n"
                                              "                                ")
        self.set_temperature_ll.setObjectName("set_temperature_ll")
        self.gridLayout_5.addWidget(self.set_temperature_ll, 1, 0, 1, 2)
        self.target_tempreature_ll = QtWidgets.QSpinBox(parent=Pumps_Vacuum)
        self.target_tempreature_ll.setMinimumSize(QtCore.QSize(150, 0))
        self.target_tempreature_ll.setMaximumSize(QtCore.QSize(70, 16777215))
        self.target_tempreature_ll.setStyleSheet("QSpinBox{\n"
                                                 "                                                    background: rgb(223,223,233)\n"
                                                 "                                                    }\n"
                                                 "                                                ")
        self.target_tempreature_ll.setObjectName("target_tempreature_ll")
        self.gridLayout_5.addWidget(self.target_tempreature_ll, 1, 2, 1, 1)
        self.gridLayout_8.addLayout(self.gridLayout_5, 0, 1, 1, 1)
        self.gridLayout_7 = QtWidgets.QGridLayout()
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.label_218 = QtWidgets.QLabel(parent=Pumps_Vacuum)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_218.setFont(font)
        self.label_218.setObjectName("label_218")
        self.gridLayout_7.addWidget(self.label_218, 0, 0, 1, 1)
        self.temp_cryo_head = QtWidgets.QLCDNumber(parent=Pumps_Vacuum)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                           QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.temp_cryo_head.sizePolicy().hasHeightForWidth())
        self.temp_cryo_head.setSizePolicy(sizePolicy)
        self.temp_cryo_head.setMinimumSize(QtCore.QSize(150, 50))
        self.temp_cryo_head.setStyleSheet("QLCDNumber{\n"
                                          "                                            border: 2px solid orange;\n"
                                          "                                                            border-radius: 10px;\n"
                                          "                                                            padding: 0 8px;\n"
                                          "                                                            }\n"
                                          "                                        ")
        self.temp_cryo_head.setObjectName("temp_cryo_head")
        self.gridLayout_7.addWidget(self.temp_cryo_head, 0, 1, 1, 1)
        self.label_220 = QtWidgets.QLabel(parent=Pumps_Vacuum)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_220.setFont(font)
        self.label_220.setObjectName("label_220")
        self.gridLayout_7.addWidget(self.label_220, 1, 0, 1, 1)
        self.ll_baking_time = QtWidgets.QLineEdit(parent=Pumps_Vacuum)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ll_baking_time.sizePolicy().hasHeightForWidth())
        self.ll_baking_time.setSizePolicy(sizePolicy)
        self.ll_baking_time.setMinimumSize(QtCore.QSize(0, 0))
        self.ll_baking_time.setStyleSheet("QLineEdit{\n"
                                          "                                    background: rgb(223,223,233)\n"
                                          "                                    }\n"
                                          "                                ")
        self.ll_baking_time.setObjectName("ll_baking_time")
        self.gridLayout_7.addWidget(self.ll_baking_time, 1, 1, 1, 1)
        self.gridLayout_8.addLayout(self.gridLayout_7, 0, 2, 1, 1)
        self.Error = QtWidgets.QLabel(parent=Pumps_Vacuum)
        self.Error.setMinimumSize(QtCore.QSize(600, 30))
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        font.setStrikeOut(False)
        self.Error.setFont(font)
        self.Error.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.Error.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.LinksAccessibleByMouse)
        self.Error.setObjectName("Error")
        self.gridLayout_8.addWidget(self.Error, 1, 0, 1, 3)
        self.verticalLayout.addLayout(self.gridLayout_8)
        self.gridLayout_9.addLayout(self.verticalLayout, 0, 0, 1, 1)

        self.retranslateUi(Pumps_Vacuum)
        QtCore.QMetaObject.connectSlotsByName(Pumps_Vacuum)
        Pumps_Vacuum.setTabOrder(self.set_temperature_cryo, self.target_tempreature_cryo)
        Pumps_Vacuum.setTabOrder(self.target_tempreature_cryo, self.set_temperature_ll)
        Pumps_Vacuum.setTabOrder(self.set_temperature_ll, self.target_tempreature_ll)
        Pumps_Vacuum.setTabOrder(self.target_tempreature_ll, self.ll_baking_time)
        Pumps_Vacuum.setTabOrder(self.ll_baking_time, self.pump_load_lock_switch)
        Pumps_Vacuum.setTabOrder(self.pump_load_lock_switch, self.pump_cryo_load_lock_switch)
        Pumps_Vacuum.setTabOrder(self.pump_cryo_load_lock_switch, self.superuser)

        ###
        self.led_red = QPixmap('./files/led-red-on.png')
        self.led_green = QPixmap('./files/green-led-on.png')
        self.led_pump_load_lock.setPixmap(self.led_green)
        self.led_pump_cryo_load_lock.setPixmap(self.led_green)
        self.pump_load_lock_switch.clicked.connect(self.pump_switch_ll)
        self.pump_cryo_load_lock_switch.clicked.connect(self.pump_switch_cryo_ll)
        # Set 8 digits for each LCD to show
        self.vacuum_main.setDigitCount(8)
        self.vacuum_buffer.setDigitCount(8)
        self.vacuum_buffer_back.setDigitCount(8)
        self.vacuum_load_lock.setDigitCount(8)
        self.vacuum_load_lock_back.setDigitCount(8)
        self.vacuum_cryo_load_lock.setDigitCount(8)
        self.vacuum_cryo_load_lock_back.setDigitCount(8)
        self.temp_stage.setDigitCount(8)
        self.temp_cryo_head.setDigitCount(8)
        self.target_tempreature_cryo.setValue(40)
        self.target_tempreature_ll.setValue(40)

        ###
        self.emitter.temp_stage.connect(self.update_temperature_stage)
        self.emitter.temp_cryo_head.connect(self.update_temperature_cryo)
        self.emitter.temp_ll.connect(self.update_temperature_ll)
        self.emitter.vacuum_main.connect(self.update_vacuum_main)
        self.set_temperature_cryo.clicked.connect(self.update_target_temperature_cryo)
        self.set_temperature_ll.clicked.connect(self.update_target_temperature_ll)
        self.emitter.vacuum_buffer.connect(self.update_vacuum_buffer)
        self.emitter.vacuum_buffer_back.connect(self.update_vacuum_buffer_back)
        self.emitter.vacuum_load_lock_back.connect(self.update_vacuum_load_back)
        self.emitter.vacuum_load_lock.connect(self.update_vacuum_load)
        self.emitter.vacuum_cryo_load_lock.connect(self.update_vacuum_cryo_load_lock)
        self.emitter.vacuum_cryo_load_lock_back.connect(self.update_vacuum_cryo_load_lock_back)
        # Connect the bool_flag_while_loop signal to a slot
        self.emitter.bool_flag_while_loop.emit(True)

        # Create a bold font
        font = QFont()
        font.setItalic(True)
        self.vacuum_main.setFont(font)

        # Thread for reading gauges
        if self.conf['gauges'] == "on":
            # Thread for reading gauges
            self.gauges_thread = threading.Thread(target=initialize_devices.state_update,
                                                  args=(self.conf, self.variables, self.emitter,))
            self.gauges_thread.setDaemon(True)
            self.gauges_thread.start()

        # Create a QTimer to hide the warning message after 8 seconds
        self.timer = QTimer(self.parent)
        self.timer.timeout.connect(self.hideMessage)

        # Create a Qtimer for baking time
        self.baking_timer = QTimer(self.parent)
        self.baking_timer.timeout.connect(self.update_target_temperature_ll)

        self.original_button_style = self.set_temperature_cryo.styleSheet()

        # default Qlcd color
        self.default_color = self.vacuum_buffer_back.style().standardPalette().color(
            QtGui.QPalette.ColorRole.WindowText)

        self.superuser.clicked.connect(self.super_user_access)
        self.original_button_style = self.superuser.styleSheet()

    def retranslateUi(self, Pumps_Vacuum):
        """
        Set the text and title of the widgets
        Args:
           Pumps_Vacuum: the main window

        Return:
            None
        """
        _translate = QtCore.QCoreApplication.translate
        ###
        # Pumps_Vacuum.setWindowTitle(_translate("Pumps_Vacuum", "Form"))
        Pumps_Vacuum.setWindowTitle(_translate("Pumps_Vacuum", "PyCCAPT Pumps and Vacuum Control"))
        Pumps_Vacuum.setWindowIcon(QtGui.QIcon('./files/logo.png'))
        ###
        self.label_212.setText(_translate("Pumps_Vacuum", "Main Chamber (mBar)"))
        self.label_211.setText(_translate("Pumps_Vacuum", "Buffer Chamber (mBar)"))
        self.label_216.setText(_translate("Pumps_Vacuum", "Cryo Load Lock (mBar)"))
        self.label_210.setText(_translate("Pumps_Vacuum", "Load Lock (mBar)"))
        self.label_214.setText(_translate("Pumps_Vacuum", "Buffer Chamber Pre (mBar)"))
        self.label_217.setText(_translate("Pumps_Vacuum", "CryoLoad Lock Pre(mBar)"))
        self.label_213.setText(_translate("Pumps_Vacuum", "Load Lock Pre(mBar)"))
        self.superuser.setText(_translate("Pumps_Vacuum", "Override Access"))
        self.pump_cryo_load_lock_switch.setText(_translate("Pumps_Vacuum", "Vent CLL"))
        self.led_pump_cryo_load_lock.setText(_translate("Pumps_Vacuum", "pump"))
        self.pump_load_lock_switch.setText(_translate("Pumps_Vacuum", "Vent LL"))
        self.led_pump_load_lock.setText(_translate("Pumps_Vacuum", "pump"))
        self.label_215.setText(_translate("Pumps_Vacuum", "Temp. Stage (K)"))
        self.set_temperature_cryo.setText(_translate("Pumps_Vacuum", "Set T Cryo (K)"))
        self.label_219.setText(_translate("Pumps_Vacuum", "LL Temp (°C)"))
        self.set_temperature_ll.setText(_translate("Pumps_Vacuum", "Set T LL (°C)"))
        self.label_218.setText(_translate("Pumps_Vacuum", "Temp. Cryo Head (K)"))
        self.label_220.setText(_translate("Pumps_Vacuum", "LL Backing Time (min)"))
        self.ll_baking_time.setText(_translate("Pumps_Vacuum", "60"))
        self.Error.setText(_translate("Pumps_Vacuum", "<html><head/><body><p><br/></p></body></html>"))

        self.target_tempreature_ll.setMaximum(1000)

    def update_temperature_stage(self, value):
        """
        Update the temperature value in the GUI
        Args:
                value: the temperature value of stage

        Return:
                None
        """
        if value == -1:
            self.temp_stage.display('Error')
        else:
            self.temp_stage.display(round(value, 2))

    def update_temperature_cryo(self, value):
        """
        Update the temperature value in the GUI
        Args:
                value: the temperature value of cryo head

        Return:
                None
        """
        if value == -1:
            self.temp_cryo_head.display('Error')
        else:
            # only up to 2 decimal points
            self.temp_cryo_head.display(round(value, 2))

    def update_temperature_ll(self, value):
        """
        Update the temperature value in the GUI
        Args:
                value: the temperature value of load lock

        Return:
                None
        """
        if value == -1:
            self.temp_ll.display('Error')
        else:
            self.temp_ll.display(round(value, 2))

    def update_target_temperature_cryo(self, ):
        """
        Update the temperature value of the cryo head
        Args:
            None

        Return:
            None
        """

        if self.target_tempreature_cryo.value() > self.conf['max_temperature_cryo']:
            self.error_message("!!! Highest possible temperature is Cryo %s !!!" % self.conf['max_temperature_cryo'])
            self.timer.start(8000)
        elif self.target_tempreature_cryo.value() < self.conf['min_temperature_cryo']:
            self.error_message("!!! Lowest possible temperature of Cryo is %s !!!" % self.conf['min_temperature_cryo'])
            self.timer.start(8000)
        else:
            if not self.variables.set_temperature_flag_cryo:
                self.variables.set_temperature_flag_cryo = True
                self.set_temperature_cryo.setStyleSheet("QPushButton{\n"
                                                        "background: rgb(0, 255, 26)\n"
                                                        "}")
                self.variables.set_temperature_cryo = self.target_tempreature_cryo.value()
            elif self.variables.set_temperature_flag_cryo:
                self.variables.set_temperature_flag_cryo = False
                self.set_temperature_cryo.setStyleSheet(self.original_button_style)

    def update_target_temperature_ll(self):
        """
        Update the temperature value of the load lock
        Args:
            None

        Return:
            None
        """
        if self.target_tempreature_ll.value() + 273.15 > self.conf['max_temperature_ll']:
            self.error_message("!!! Highest possible temperature of LL is %s !!!" % self.conf['max_temperature_ll'])
            self.timer.start(8000)
        elif self.target_tempreature_ll.value() + 273.15 < self.conf['min_temperature_ll']:
            self.error_message("!!! Lowest possible temperature of LL is %s !!!" % self.conf['min_temperature_ll'])
            self.timer.start(8000)
        else:
            if not self.variables.set_temperature_flag_ll:
                self.ll_baking_time.setEnabled(False)
                self.target_tempreature_ll.setEnabled(False)
                self.variables.set_temperature_flag_ll = True
                self.set_temperature_ll.setStyleSheet("QPushButton{\n"
                                                      "background: rgb(0, 255, 26)\n"
                                                      "}")
                self.variables.set_temperature_ll = self.target_tempreature_ll.value()
                # Start the timer for baking which is min * 60000
                self.baking_timer.start(int(self.ll_baking_time.text()) * 60000)
            elif self.variables.set_temperature_flag_ll:
                self.baking_timer.stop()
                self.ll_baking_time.setEnabled(True)
                self.target_tempreature_ll.setEnabled(True)
                self.variables.set_temperature_flag_ll = False
                self.set_temperature_ll.setStyleSheet(self.original_button_style)

    def update_vacuum_main(self, value):
        """
        Update the vacuum value in the GUI
        Args:
                value: the temperature value

        Return:
                None
        """
        if value == -1:
            self.vacuum_main.display('Error')
        else:
            self.vacuum_main.display('{:.2e}'.format(value))
        if value > 0.000000001:
            self.label_212.setStyleSheet("color: red")
        else:
            self.label_212.setStyleSheet("color: black")

    def update_vacuum_buffer(self, value):
        """
        Update the vacuum value in the GUI
        Args:
                value: the temperature value

        Return:
                None
        """
        if value == -1:
            self.vacuum_buffer.display('Error')
        else:
            self.vacuum_buffer.display('{:.2e}'.format(value))
        if value > 0.000000001:
            self.label_211.setStyleSheet("color: red")
        else:
            self.label_211.setStyleSheet("color: black")

    def update_vacuum_buffer_back(self, value):
        """
        Update the vacuum value in the GUI
        Args:
                value: the temperature value

        Return:
                None
        """
        if value == -1:
            self.vacuum_buffer_back.display('Error')
        else:
            self.vacuum_buffer_back.display('{:.2e}'.format(value))
        if value > 0.01:
            self.label_214.setStyleSheet("color: red")
        else:
            self.label_214.setStyleSheet("color: black")

    def update_vacuum_load_back(self, value):
        """
        Update the vacuum value in the GUI
        Args:
                value: the temperature value

        Return:
                None
        """
        if value == -1:
            self.vacuum_load_lock_back.display('Error')
        else:
            self.vacuum_load_lock_back.display('{:.2e}'.format(value))
        if value > 0.1:
            self.label_213.setStyleSheet("color: red")
        else:
            self.label_213.setStyleSheet("color: black")

    def update_vacuum_load(self, value):
        """
        Update the vacuum value in the GUI
        Args:
                value: the temperature value

        Return:
                None
        """
        if value == -1:
            self.vacuum_load_lock.display('Error')
        else:
            self.vacuum_load_lock.display('{:.2e}'.format(value))
        if value > 0.00001:
            self.label_210.setStyleSheet("color: red")
        else:
            self.label_210.setStyleSheet("color: black")

    def update_vacuum_cryo_load_lock(self, value):
        """
        Update the vacuum value in the GUI
        Args:
                value: the temperature value

        Return:
                None
        """
        if value == -1:
            self.vacuum_cryo_load_lock.display('Error')  # Or any other message you prefer
        else:
            self.vacuum_cryo_load_lock.display('{:.2e}'.format(value))

        if value > 0.00001:
            self.label_216.setStyleSheet("color: red")
        else:
            self.label_216.setStyleSheet("color: black")

    def update_vacuum_cryo_load_lock_back(self, value):
        """
        Update the vacuum value in the GUI
        Args:
                value: the temperature value

        Return:
                None
        """
        if value == -1:
            self.vacuum_cryo_load_lock_back.display('Error')
        else:
            self.vacuum_cryo_load_lock_back.display('{:.2e}'.format(value))
        if value > 0.1:
            self.label_217.setStyleSheet("color: red")
        else:
            self.label_217.setStyleSheet("color: black")

    def super_user_access(self):
        """
        The function for override access

        Args:
                None

        Returns:
                None
        """
        if not self.flag_super_user:
            self.flag_super_user = True
            self.superuser.setStyleSheet("QPushButton{\n"
                                         "background: rgb(0, 255, 26)\n"
                                         "}")
            self.error_message("!!! Override Access Granted !!!")
        elif self.flag_super_user:
            self.flag_super_user = False
            self.superuser.setStyleSheet(self.original_button_style)
            self.error_message("!!! Override Access deactivated !!!")
            self.timer.start(8000)

    def hideMessage(self):
        """
        Hide the warning message
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

    def pump_switch_ll(self):
        """
        Switch the pump on or off
        Args:
                None

        Return:
                None
        """
        try:
            if self.flag_super_user or (not self.variables.start_flag and not self.variables.flag_main_gate \
                                        and not self.variables.flag_cryo_gate and not self.variables.flag_load_gate):
                if self.variables.flag_pump_load_lock:
                    self.variables.flag_pump_load_lock_click = True
                    self.led_pump_load_lock.setPixmap(self.led_red)
                    self.pump_load_lock_switch.setEnabled(False)
                    time.sleep(1)
                    self.pump_load_lock_switch.setEnabled(True)
                elif not self.variables.flag_pump_load_lock:
                    self.variables.flag_pump_load_lock_click = True
                    self.led_pump_load_lock.setPixmap(self.led_green)
                    self.pump_load_lock_switch.setEnabled(False)
                    time.sleep(1)
                    self.pump_load_lock_switch.setEnabled(True)
            else:  # SHow error message in the GUI
                if self.variables.start_flag:
                    self.error_message("!!! An experiment is running !!!")
                else:
                    self.error_message("!!! First Close all the Gates !!!")

                self.timer.start(8000)
        except Exception as e:
            print('Error in pump_switch function')
            print(e)
            pass

    def pump_switch_cryo_ll(self):
        """
        Switch the pump on or off

        Args:
                None

        Return:
                None
        """
        try:
            if self.flag_super_user or (not self.variables.start_flag and not self.variables.flag_main_gate \
                                        and not self.variables.flag_cryo_gate and not self.variables.flag_load_gate):
                if self.variables.flag_pump_cryo_load_lock:
                    self.variables.flag_pump_cryo_load_lock_click = True
                    self.led_pump_cryo_load_lock.setPixmap(self.led_red)
                    self.pump_cryo_load_lock_switch.setEnabled(False)
                    time.sleep(1)
                    self.pump_cryo_load_lock_switch.setEnabled(True)
                elif not self.variables.flag_pump_cryo_load_lock:
                    self.variables.flag_pump_cryo_load_lock_click = True
                    self.led_pump_cryo_load_lock.setPixmap(self.led_green)
                    self.pump_cryo_load_lock_switch.setEnabled(False)
                    time.sleep(1)
                    self.pump_cryo_load_lock_switch.setEnabled(True)
            else:  # SHow error message in the GUI
                if self.variables.start_flag:
                    self.error_message("!!! An experiment is running !!!")
                else:
                    self.error_message("!!! First Close all the Gates !!!")

                self.timer.start(8000)
        except Exception as e:
            print('Error in pump_switch function')
            print(e)
            pass

    def error_message(self, message):
        """
        Show the warning message
        Args:
                message: the message to be shown

        Return:
                None
        """
        _translate = QtCore.QCoreApplication.translate
        self.Error.setText(_translate("OXCART",
                                      "<html><head/><body><p><span style=\" color:#ff0000;\">"
                                      + message + "</span></p></body></html>"))

    def stop(self):
        """
        Stop the timer
        Args:
                None

        Return:
                None
        """
        # Stop any background processes, timers, or threads here
        self.timer.stop()  # If you want to stop this timer when closing


class SignalEmitter(QObject):
    """
    Signal emitter class for emitting signals related to vacuum and pumps control.
    """

    temp_stage = pyqtSignal(float)
    temp_cryo_head = pyqtSignal(float)
    temp_ll = pyqtSignal(float)
    vacuum_main = pyqtSignal(float)
    vacuum_buffer = pyqtSignal(float)
    vacuum_buffer_back = pyqtSignal(float)
    vacuum_load_lock_back = pyqtSignal(float)
    vacuum_load_lock = pyqtSignal(float)
    vacuum_cryo_load_lock = pyqtSignal(float)
    vacuum_cryo_load_lock_back = pyqtSignal(float)
    bool_flag_while_loop = pyqtSignal(bool)


class PumpsVacuumWindow(QtWidgets.QWidget):
    """
    Widget for Pumps and Vacuum control window.
    """
    closed = QtCore.pyqtSignal()  # Define a custom closed signal

    def __init__(self, gui_pumps_vacuum, signal_emitter, *args, **kwargs):
        """
        Constructor for the PumpsVacuumWindow class.

        Args:
            gui_pumps_vacuum: Instance of the PumpsVacuum control.
            signal_emitter: SignalEmitter object for communication.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.gui_pumps_vacuum = gui_pumps_vacuum
        self.signal_emitter = signal_emitter

    def closeEvent(self, event):
        """
            Close event for the window.

            Args:
                event: Close event.
        """
        self.gui_pumps_vacuum.stop()  # Call the stop method to stop any background activity
        self.signal_emitter.bool_flag_while_loop.emit(False)
        self.gui_pumps_vacuum.gauges_thread.join(1)
        # Additional cleanup code here if needed
        self.closed.emit()  # Emit the custom closed signal
        super().closeEvent(event)

    def setWindowStyleFusion(self):
        # Set the Fusion style
        QtWidgets.QApplication.setStyle("Fusion")


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
    Pumps_vacuum = QtWidgets.QWidget()
    signal_emitter = SignalEmitter()
    ui = Ui_Pumps_Vacuum(variables, conf, signal_emitter)
    ui.setupUi(Pumps_vacuum)
    Pumps_vacuum.show()
    sys.exit(app.exec())
