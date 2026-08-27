import os
import sys
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QFont

# Local module and scripts
from pyccapt.control.core import runtime
from pyccapt.control.devices import initialize_devices
from pyccapt.control.gui import tooltips


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
        self.default_color = None
        self.variables = variables
        self.conf = conf
        self.parent = parent
        self.emitter = SignalEmitter

        # --- "Vent CLL" partial-vent state ---
        # Whether the fast CLL vent (sample/cryo exchange) is currently active,
        # and the persistent NI task that holds the vent-valve relay line high
        # while venting (see vent_cryo_load_lock_partial / _set_vent_valve).
        self.flag_vent_cll_partial = False
        self.variables.flag_vent_cryo_load_lock_partial = False
        self._vent_valve_task = None

        # --- LL baking log state ---
        # Latest LL temperature (deg C) seen on the temp_ll signal; cached so
        # the periodic log row always has a value even between signal updates.
        self._latest_temp_ll = None
        # DataFrame holding the current baking run (None when not baking).
        self.ll_baking_log_data = None
        self.ll_baking_log_file = None
        self.ll_baking_log_start = None

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
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.vacuum_main.sizePolicy().hasHeightForWidth())
        self.vacuum_main.setSizePolicy(sizePolicy)
        self.vacuum_main.setFixedSize(QtCore.QSize(220, 55))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.vacuum_main.setFont(font)
        self.vacuum_main.setStyleSheet(
            "QLCDNumber{\n"
            "                                    border: 2px solid green;\n"
            "                                    border-radius: 10px;\n"
            "                                    padding: 0 8px;\n"
            "                                    }\n"
            "                                "
        )
        self.vacuum_main.setObjectName("vacuum_main")
        self.gridLayout.addWidget(self.vacuum_main, 0, 1, 1, 2)
        self.label_211 = QtWidgets.QLabel(parent=Pumps_Vacuum)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_211.setFont(font)
        self.label_211.setObjectName("label_211")
        self.gridLayout.addWidget(self.label_211, 1, 0, 1, 2)
        self.vacuum_buffer = QtWidgets.QLCDNumber(parent=Pumps_Vacuum)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.vacuum_buffer.sizePolicy().hasHeightForWidth())
        self.vacuum_buffer.setSizePolicy(sizePolicy)
        self.vacuum_buffer.setMinimumSize(QtCore.QSize(150, 50))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.vacuum_buffer.setFont(font)
        self.vacuum_buffer.setStyleSheet(
            "QLCDNumber{\n"
            "                                            border: 2px solid brown;\n"
            "                                            border-radius: 10px;\n"
            "                                            padding: 0 8px;\n"
            "                                            }\n"
            "                                        "
        )
        self.vacuum_buffer.setObjectName("vacuum_buffer")
        self.gridLayout.addWidget(self.vacuum_buffer, 1, 2, 1, 1)
        self.label_216 = QtWidgets.QLabel(parent=Pumps_Vacuum)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_216.setFont(font)
        self.label_216.setObjectName("label_216")
        self.gridLayout.addWidget(self.label_216, 2, 0, 1, 2)
        self.vacuum_cryo_load_lock = QtWidgets.QLCDNumber(parent=Pumps_Vacuum)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.vacuum_cryo_load_lock.sizePolicy().hasHeightForWidth())
        self.vacuum_cryo_load_lock.setSizePolicy(sizePolicy)
        self.vacuum_cryo_load_lock.setMinimumSize(QtCore.QSize(150, 50))
        self.vacuum_cryo_load_lock.setStyleSheet(
            "QLCDNumber{\n"
            "                                            border: 2px solid magenta;\n"
            "                                            border-radius: 10px;\n"
            "                                            padding: 0 8px;\n"
            "                                            }\n"
            "                                        "
        )
        self.vacuum_cryo_load_lock.setObjectName("vacuum_cryo_load_lock")
        self.gridLayout.addWidget(self.vacuum_cryo_load_lock, 2, 2, 1, 1)
        self.label_210 = QtWidgets.QLabel(parent=Pumps_Vacuum)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_210.setFont(font)
        self.label_210.setObjectName("label_210")
        self.gridLayout.addWidget(self.label_210, 3, 0, 1, 2)
        self.vacuum_load_lock = QtWidgets.QLCDNumber(parent=Pumps_Vacuum)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.vacuum_load_lock.sizePolicy().hasHeightForWidth())
        self.vacuum_load_lock.setSizePolicy(sizePolicy)
        self.vacuum_load_lock.setMinimumSize(QtCore.QSize(150, 50))
        self.vacuum_load_lock.setStyleSheet(
            "QLCDNumber{\n"
            "                                            border: 2px solid blue;\n"
            "                                                            border-radius: 10px;\n"
            "                                                            padding: 0 8px;\n"
            "                                                            }\n"
            "                                        "
        )
        self.vacuum_load_lock.setObjectName("vacuum_load_lock")
        self.gridLayout.addWidget(self.vacuum_load_lock, 3, 2, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout, 0, 0, 2, 1)
        spacerItem = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum
        )
        self.gridLayout_4.addItem(spacerItem, 0, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding
        )
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
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.vacuum_buffer_back.sizePolicy().hasHeightForWidth())
        self.vacuum_buffer_back.setSizePolicy(sizePolicy)
        self.vacuum_buffer_back.setMinimumSize(QtCore.QSize(150, 50))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.vacuum_buffer_back.setFont(font)
        self.vacuum_buffer_back.setStyleSheet(
            "QLCDNumber{\n"
            "                                            border: 2px solid brown;\n"
            "                                                            border-radius: 10px;\n"
            "                                                            padding: 0 8px;\n"
            "                                                            }\n"
            "                                                        "
        )
        self.vacuum_buffer_back.setObjectName("vacuum_buffer_back")
        self.gridLayout_2.addWidget(self.vacuum_buffer_back, 0, 1, 1, 1)
        self.label_217 = QtWidgets.QLabel(parent=Pumps_Vacuum)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_217.setFont(font)
        self.label_217.setObjectName("label_217")
        self.gridLayout_2.addWidget(self.label_217, 1, 0, 1, 1)
        self.vacuum_cryo_load_lock_back = QtWidgets.QLCDNumber(parent=Pumps_Vacuum)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.vacuum_cryo_load_lock_back.sizePolicy().hasHeightForWidth())
        self.vacuum_cryo_load_lock_back.setSizePolicy(sizePolicy)
        self.vacuum_cryo_load_lock_back.setMinimumSize(QtCore.QSize(150, 50))
        self.vacuum_cryo_load_lock_back.setStyleSheet(
            "QLCDNumber{\n"
            "                                            border: 2px solid magenta;\n"
            "                                            border-radius: 10px;\n"
            "                                            padding: 0 8px;\n"
            "                                            }\n"
            "                                        "
        )
        self.vacuum_cryo_load_lock_back.setObjectName("vacuum_cryo_load_lock_back")
        self.gridLayout_2.addWidget(self.vacuum_cryo_load_lock_back, 1, 1, 1, 1)
        self.label_213 = QtWidgets.QLabel(parent=Pumps_Vacuum)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_213.setFont(font)
        self.label_213.setObjectName("label_213")
        self.gridLayout_2.addWidget(self.label_213, 2, 0, 1, 1)
        self.vacuum_load_lock_back = QtWidgets.QLCDNumber(parent=Pumps_Vacuum)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.vacuum_load_lock_back.sizePolicy().hasHeightForWidth())
        self.vacuum_load_lock_back.setSizePolicy(sizePolicy)
        self.vacuum_load_lock_back.setMinimumSize(QtCore.QSize(150, 50))
        self.vacuum_load_lock_back.setStyleSheet(
            "QLCDNumber{\n"
            "                                            border: 2px solid blue;\n"
            "                                            border-radius: 10px;\n"
            "                                            padding: 0 8px;\n"
            "                                            }\n"
            "                                        "
        )
        self.vacuum_load_lock_back.setObjectName("vacuum_load_lock_back")
        self.gridLayout_2.addWidget(self.vacuum_load_lock_back, 2, 1, 1, 1)
        # Keep the three chamber gauges and their three backing/pre-vacuum
        # gauges visually identical even when the neighboring controls change.
        for vacuum_lcd in (
            self.vacuum_buffer,
            self.vacuum_cryo_load_lock,
            self.vacuum_load_lock,
            self.vacuum_buffer_back,
            self.vacuum_cryo_load_lock_back,
            self.vacuum_load_lock_back,
        ):
            vacuum_lcd.setFixedSize(QtCore.QSize(150, 50))
        self.gridLayout_4.addLayout(self.gridLayout_2, 1, 1, 1, 2)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        # "Vent CLL" - partial vent of the cryo load lock for fast sample/cryo
        # exchange (drives a 3-valve sequence). Sits between "Fully Vent CLL"
        # and "Vent LL".
        self.vent_cryo_load_lock_partial_switch = QtWidgets.QPushButton(parent=Pumps_Vacuum)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.vent_cryo_load_lock_partial_switch.sizePolicy().hasHeightForWidth())
        self.vent_cryo_load_lock_partial_switch.setSizePolicy(sizePolicy)
        self.vent_cryo_load_lock_partial_switch.setMinimumSize(QtCore.QSize(0, 25))
        self.vent_cryo_load_lock_partial_switch.setStyleSheet(
            "QPushButton{\n"
            "                                            background: rgb(193, 193, 193)\n"
            "                                            }\n"
            "                                        "
        )
        self.vent_cryo_load_lock_partial_switch.setObjectName("vent_cryo_load_lock_partial_switch")
        self.gridLayout_3.addWidget(self.vent_cryo_load_lock_partial_switch, 2, 0, 1, 2)
        self.pump_cryo_load_lock_switch = QtWidgets.QPushButton(parent=Pumps_Vacuum)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pump_cryo_load_lock_switch.sizePolicy().hasHeightForWidth())
        self.pump_cryo_load_lock_switch.setSizePolicy(sizePolicy)
        self.pump_cryo_load_lock_switch.setMinimumSize(QtCore.QSize(0, 25))
        self.pump_cryo_load_lock_switch.setStyleSheet(
            "QPushButton{\n"
            "                                            background: rgb(193, 193, 193)\n"
            "                                            }\n"
            "                                        "
        )
        self.pump_cryo_load_lock_switch.setObjectName("pump_cryo_load_lock_switch")
        self.gridLayout_3.addWidget(self.pump_cryo_load_lock_switch, 1, 0, 1, 2)
        self.pump_load_lock_switch = QtWidgets.QPushButton(parent=Pumps_Vacuum)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pump_load_lock_switch.sizePolicy().hasHeightForWidth())
        self.pump_load_lock_switch.setSizePolicy(sizePolicy)
        self.pump_load_lock_switch.setMinimumSize(QtCore.QSize(0, 25))
        self.pump_load_lock_switch.setStyleSheet(
            "QPushButton{\n"
            "                                            background: rgb(193, 193, 193)\n"
            "                                            }\n"
            "                                        "
        )
        self.pump_load_lock_switch.setObjectName("pump_load_lock_switch")
        self.gridLayout_3.addWidget(self.pump_load_lock_switch, 3, 0, 1, 2)
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
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.temp_stage.sizePolicy().hasHeightForWidth())
        self.temp_stage.setSizePolicy(sizePolicy)
        self.temp_stage.setMinimumSize(QtCore.QSize(150, 50))
        self.temp_stage.setStyleSheet(
            "QLCDNumber{\n"
            "                                                            border: 2px solid orange;\n"
            "                                                            border-radius: 10px;\n"
            "                                                            padding: 0 8px;\n"
            "                                                            }\n"
            "                                        "
        )
        self.temp_stage.setObjectName("temp_stage")
        self.gridLayout_6.addWidget(self.temp_stage, 0, 1, 1, 2)
        self.set_temperature_cryo = QtWidgets.QPushButton(parent=Pumps_Vacuum)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.set_temperature_cryo.sizePolicy().hasHeightForWidth())
        self.set_temperature_cryo.setSizePolicy(sizePolicy)
        self.set_temperature_cryo.setMinimumSize(QtCore.QSize(0, 25))
        self.set_temperature_cryo.setStyleSheet(
            "QPushButton{\n"
            "                                    background: rgb(193, 193, 193)\n"
            "                                    }\n"
            "                                "
        )
        self.set_temperature_cryo.setObjectName("set_temperature_cryo")
        self.gridLayout_6.addWidget(self.set_temperature_cryo, 1, 0, 1, 2)
        self.target_tempreature_cryo = QtWidgets.QSpinBox(parent=Pumps_Vacuum)
        self.target_tempreature_cryo.setMinimumSize(QtCore.QSize(150, 0))
        self.target_tempreature_cryo.setMaximumSize(QtCore.QSize(70, 16777215))
        self.target_tempreature_cryo.setStyleSheet(
            "QSpinBox{\n"
            "                                    background: rgb(223,223,233)\n"
            "                                    }\n"
            "                                "
        )
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
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.temp_ll.sizePolicy().hasHeightForWidth())
        self.temp_ll.setSizePolicy(sizePolicy)
        self.temp_ll.setMinimumSize(QtCore.QSize(150, 50))
        self.temp_ll.setStyleSheet(
            "QLCDNumber{\n"
            "                                    border: 2px solid orange;\n"
            "                                    border-radius: 10px;\n"
            "                                    padding: 0 8px;\n"
            "                                    }\n"
            "                                "
        )
        self.temp_ll.setObjectName("temp_ll")
        self.gridLayout_5.addWidget(self.temp_ll, 0, 1, 1, 2)
        self.set_temperature_ll = QtWidgets.QPushButton(parent=Pumps_Vacuum)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.set_temperature_ll.sizePolicy().hasHeightForWidth())
        self.set_temperature_ll.setSizePolicy(sizePolicy)
        self.set_temperature_ll.setMinimumSize(QtCore.QSize(0, 25))
        self.set_temperature_ll.setStyleSheet(
            "QPushButton{\n"
            "                                    background: rgb(193, 193, 193)\n"
            "                                    }\n"
            "                                "
        )
        self.set_temperature_ll.setObjectName("set_temperature_ll")
        self.gridLayout_5.addWidget(self.set_temperature_ll, 1, 0, 1, 2)
        self.target_tempreature_ll = QtWidgets.QSpinBox(parent=Pumps_Vacuum)
        self.target_tempreature_ll.setMinimumSize(QtCore.QSize(150, 0))
        self.target_tempreature_ll.setMaximumSize(QtCore.QSize(70, 16777215))
        self.target_tempreature_ll.setStyleSheet(
            "QSpinBox{\n"
            "                                                    background: rgb(223,223,233)\n"
            "                                                    }\n"
            "                                                "
        )
        self.target_tempreature_ll.setObjectName("target_tempreature_ll")
        self.gridLayout_5.addWidget(self.target_tempreature_ll, 1, 2, 1, 1)
        # Load Lock temp block moved to column 2 (swapped with the Cryo Head block).
        self.gridLayout_8.addLayout(self.gridLayout_5, 0, 2, 1, 1)
        self.gridLayout_7 = QtWidgets.QGridLayout()
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.label_218 = QtWidgets.QLabel(parent=Pumps_Vacuum)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_218.setFont(font)
        self.label_218.setObjectName("label_218")
        self.gridLayout_7.addWidget(self.label_218, 0, 0, 1, 1)
        self.temp_cryo_head = QtWidgets.QLCDNumber(parent=Pumps_Vacuum)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.temp_cryo_head.sizePolicy().hasHeightForWidth())
        self.temp_cryo_head.setSizePolicy(sizePolicy)
        self.temp_cryo_head.setMinimumSize(QtCore.QSize(150, 50))
        self.temp_cryo_head.setStyleSheet(
            "QLCDNumber{\n"
            "                                            border: 2px solid orange;\n"
            "                                                            border-radius: 10px;\n"
            "                                                            padding: 0 8px;\n"
            "                                                            }\n"
            "                                        "
        )
        self.temp_cryo_head.setObjectName("temp_cryo_head")
        self.gridLayout_7.addWidget(self.temp_cryo_head, 0, 1, 1, 1)
        self.label_221 = QtWidgets.QLabel(parent=Pumps_Vacuum)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_221.setFont(font)
        self.label_221.setObjectName("label_221")
        self.gridLayout_7.addWidget(self.label_221, 1, 0, 1, 1)
        self.temp_cryo_head_inside = QtWidgets.QLCDNumber(parent=Pumps_Vacuum)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.temp_cryo_head_inside.sizePolicy().hasHeightForWidth())
        self.temp_cryo_head_inside.setSizePolicy(sizePolicy)
        self.temp_cryo_head_inside.setMinimumSize(QtCore.QSize(150, 50))
        self.temp_cryo_head_inside.setStyleSheet(
            "QLCDNumber{\n    border: 2px solid orange;\n    border-radius: 10px;\n    padding: 0 8px;\n    }\n"
        )
        self.temp_cryo_head_inside.setObjectName("temp_cryo_head_inside")
        self.gridLayout_7.addWidget(self.temp_cryo_head_inside, 1, 1, 1, 1)
        self.label_220 = QtWidgets.QLabel(parent=Pumps_Vacuum)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_220.setFont(font)
        self.label_220.setObjectName("label_220")
        self.gridLayout_7.addWidget(self.label_220, 2, 0, 1, 1)
        self.ll_baking_time = QtWidgets.QLineEdit(parent=Pumps_Vacuum)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ll_baking_time.sizePolicy().hasHeightForWidth())
        self.ll_baking_time.setSizePolicy(sizePolicy)
        self.ll_baking_time.setMinimumSize(QtCore.QSize(0, 0))
        self.ll_baking_time.setStyleSheet(
            "QLineEdit{\n"
            "                                    background: rgb(223,223,233)\n"
            "                                    }\n"
            "                                "
        )
        self.ll_baking_time.setObjectName("ll_baking_time")
        self.gridLayout_7.addWidget(self.ll_baking_time, 2, 1, 1, 1)
        # Cryo Head (outside/inside) block moved to column 1 (swapped with Load Lock).
        self.gridLayout_8.addLayout(self.gridLayout_7, 0, 1, 1, 1)
        self.Error = QtWidgets.QLabel(parent=Pumps_Vacuum)
        self.Error.setMinimumSize(QtCore.QSize(600, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setStrikeOut(False)
        self.Error.setFont(font)
        self.Error.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.Error.setWordWrap(True)
        self.Error.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.LinksAccessibleByMouse)
        self.Error.setObjectName("Error")
        self.gridLayout_8.addWidget(self.Error, 1, 0, 1, 3)
        self.verticalLayout.addLayout(self.gridLayout_8)
        self.gridLayout_9.addLayout(self.verticalLayout, 0, 0, 1, 1)

        self.retranslateUi(Pumps_Vacuum)
        QtCore.QMetaObject.connectSlotsByName(Pumps_Vacuum)
        tooltips.apply_tooltips(self, tooltips.PUMPS_TOOLTIPS)
        Pumps_Vacuum.setTabOrder(self.set_temperature_cryo, self.target_tempreature_cryo)
        Pumps_Vacuum.setTabOrder(self.target_tempreature_cryo, self.set_temperature_ll)
        Pumps_Vacuum.setTabOrder(self.set_temperature_ll, self.target_tempreature_ll)
        Pumps_Vacuum.setTabOrder(self.target_tempreature_ll, self.ll_baking_time)
        Pumps_Vacuum.setTabOrder(self.ll_baking_time, self.pump_load_lock_switch)
        Pumps_Vacuum.setTabOrder(self.pump_load_lock_switch, self.vent_cryo_load_lock_partial_switch)
        Pumps_Vacuum.setTabOrder(self.vent_cryo_load_lock_partial_switch, self.pump_cryo_load_lock_switch)

        self.pump_load_lock_switch.clicked.connect(self.pump_switch_ll)
        self.pump_cryo_load_lock_switch.clicked.connect(self.pump_switch_cryo_ll)
        self.vent_cryo_load_lock_partial_switch.clicked.connect(self.vent_cryo_load_lock_partial)
        # The buttons themselves replace the old LED icons as state indicators.
        # Green means the corresponding vent action is active.
        self.vent_partial_default_style = self.vent_cryo_load_lock_partial_switch.styleSheet()
        self.pump_load_lock_default_style = self.pump_load_lock_switch.styleSheet()
        self.pump_cryo_load_lock_default_style = self.pump_cryo_load_lock_switch.styleSheet()
        self._sync_pump_action_styles()
        # Full CLL venting is intentionally unavailable from this window.
        self.pump_cryo_load_lock_switch.setEnabled(False)
        # Initialise the CLL vent valve CLOSED and HOLD the line low. The
        # USB-6501 output floats HIGH via its pull-up when undriven, which would
        # leave the active-high vent relay energised (CLL venting) at startup.
        # Driving it low here opens a held task so the CLL starts un-vented.
        self._set_vent_valve(True)  # True = closed, False = open
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
        self.temp_cryo_head_inside.setDigitCount(8)
        self.target_tempreature_cryo.setValue(40)
        self.target_tempreature_ll.setValue(40)

        ###
        self.emitter.temp_stage.connect(self.update_temperature_stage)
        self.emitter.temp_cryo_head.connect(self.update_temperature_cryo)
        self.emitter.temp_cryo_head_inside.connect(self.update_temperature_cryo_inside)
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
            # Real threading.Event so cleanup() can actually stop the
            # gauge polling loop. The legacy ``emitter.bool_flag_while_loop``
            # is a pyqtSignal -- always truthy -- and never stopped the
            # thread; the OS held the COM ports until process exit.
            self.gauges_stop_event = threading.Event()
            self.gauges_thread = threading.Thread(
                target=initialize_devices.state_update,
                args=(
                    self.conf,
                    self.variables,
                    self.emitter,
                ),
                kwargs={'stop_event': self.gauges_stop_event},
                daemon=True,
            )
            self.gauges_thread.start()

        # Create a QTimer to hide the warning message after 8 seconds
        self.timer = QTimer(self.parent)
        self.timer.timeout.connect(self.hideMessage)

        # Create a Qtimer for baking time
        self.baking_timer = QTimer(self.parent)
        self.baking_timer.timeout.connect(self.update_target_temperature_ll)

        # Timer that samples LL temperature & vacuum into the baking CSV while
        # a baking run is active.
        self.ll_baking_log_timer = QTimer(self.parent)
        self.ll_baking_log_timer.timeout.connect(self._log_ll_baking_row)

        self.original_button_style = self.set_temperature_cryo.styleSheet()

        # default Qlcd color
        self.default_color = self.vacuum_buffer_back.style().standardPalette().color(QtGui.QPalette.ColorRole.WindowText)


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
        self.pump_cryo_load_lock_switch.setText(_translate("Pumps_Vacuum", "Fully Vent CLL"))
        self.vent_cryo_load_lock_partial_switch.setText(_translate("Pumps_Vacuum", "Vent CLL"))
        self.pump_load_lock_switch.setText(_translate("Pumps_Vacuum", "Vent LL"))
        # Cryo sensor labels driven by config.toml cryo_sensor_X keys
        _s1 = self.conf.get('cryo_sensor_1', 'cryo_head_outside').replace('_', ' ').title()
        _s2 = self.conf.get('cryo_sensor_2', 'cryo_head_inside').replace('_', ' ').title()
        _s3 = self.conf.get('cryo_sensor_3', 'stage').replace('_', ' ').title()
        _s4 = self.conf.get('cryo_sensor_4', 'load_lock').replace('_', ' ').title()
        self.label_215.setText(_translate("Pumps_Vacuum", f"Temp. {_s3} (K)"))
        self.set_temperature_cryo.setText(_translate("Pumps_Vacuum", "Set T Cryo (K)"))
        self.label_219.setText(_translate("Pumps_Vacuum", f"{_s4} Temp (°C)"))
        self.set_temperature_ll.setText(_translate("Pumps_Vacuum", "Set T LL (°C)"))
        self.label_218.setText(_translate("Pumps_Vacuum", f"Temp. {_s1} (K)"))
        self.label_221.setText(_translate("Pumps_Vacuum", f"Temp. {_s2} (K)"))
        self.label_220.setText(_translate("Pumps_Vacuum", "LL Baking Duration (min.)"))
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

    def update_temperature_cryo_inside(self, value):
        """
        Update the temperature value of cryo head inside sensor in the GUI.
        Args:
                value: the temperature value of cryo head inside

        Return:
                None
        """
        if value == -1:
            self.temp_cryo_head_inside.display('Error')
        else:
            self.temp_cryo_head_inside.display(round(value, 2))

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
            # Cache for the baking log (vacuum is read from variables directly).
            self._latest_temp_ll = round(value, 2)

    def update_target_temperature_cryo(
        self,
    ):
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
                self.set_temperature_cryo.setStyleSheet("QPushButton{\nbackground: rgb(0, 255, 26)\n}")
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
                self.set_temperature_ll.setStyleSheet("QPushButton{\nbackground: rgb(0, 255, 26)\n}")
                self.variables.set_temperature_ll = self.target_tempreature_ll.value()
                # Start the timer for baking which is min * 60000
                self.baking_timer.start(int(self.ll_baking_time.text()) * 60000)
                # Begin logging LL temperature & vacuum for this baking run.
                self._start_ll_baking_log()
            elif self.variables.set_temperature_flag_ll:
                self.baking_timer.stop()
                self.ll_baking_time.setEnabled(True)
                self.target_tempreature_ll.setEnabled(True)
                self.variables.set_temperature_flag_ll = False
                self.set_temperature_ll.setStyleSheet(self.original_button_style)
                # Baking finished -- either the duration elapsed (this slot is
                # also fired by baking_timer) or the user deselected the button.
                self._stop_ll_baking_log()

    def _start_ll_baking_log(self):
        """Create a CSV and start sampling LL temperature & vacuum.

        Called when the "Set T LL" button starts a baking run. Logging stops
        in :meth:`_stop_ll_baking_log` once the baking duration elapses or the
        user deselects the button.
        """
        try:
            now = datetime.now()
            now_time = now.strftime("%d-%m-%Y_%H-%M-%S")
            save_path = runtime.project_path("files", "logs", "ll_baking", now_time)
            os.makedirs(save_path, mode=0o777, exist_ok=True)
            self.ll_baking_log_file = str(save_path / f'll_baking_{now_time}.csv')
            self.ll_baking_log_data = pd.DataFrame(
                columns=['Date', 'Time', 'Elapsed_s', 'Target_T_LL_C', 'Temp_LL_C', 'Vacuum_LL_mBar']
            )
            self.ll_baking_log_start = time.perf_counter()
            # Sample once per second.
            self.ll_baking_log_timer.start(1000)
            # Write an initial row immediately so the file is never empty.
            self._log_ll_baking_row()
        except Exception as e:
            print(f'Cannot start LL baking log: {e}')

    def _log_ll_baking_row(self):
        """Append one sample (temperature + vacuum) and flush to disk."""
        if self.ll_baking_log_data is None:
            return
        now = datetime.now()
        elapsed = round(time.perf_counter() - self.ll_baking_log_start, 1)
        temp = self._latest_temp_ll if self._latest_temp_ll is not None else np.nan
        vacuum = self.variables.vacuum_load_lock
        self.ll_baking_log_data.loc[len(self.ll_baking_log_data)] = [
            now.strftime("%d-%m-%Y"),
            now.strftime('%H:%M:%S'),
            elapsed,
            self.variables.set_temperature_ll,
            temp,
            vacuum,
        ]
        try:
            self.ll_baking_log_data.to_csv(self.ll_baking_log_file, sep=';', index=False)
        except Exception as e:
            print(f'LL baking csv cannot be saved (close the file): {e}')

    def _stop_ll_baking_log(self):
        """Stop sampling, write a final row, and release the log buffer."""
        if self.ll_baking_log_timer.isActive():
            self.ll_baking_log_timer.stop()
        if self.ll_baking_log_data is not None:
            # Capture a final sample so the CSV records the end state.
            self._log_ll_baking_row()
        self.ll_baking_log_data = None
        self.ll_baking_log_file = None
        self.ll_baking_log_start = None

    def _update_gauge(self, display_widget, label_widget, value, threshold_key):
        """Show *value* on a gauge LCD and colour its label by threshold."""
        if value == -1:
            display_widget.display('Error')
        else:
            display_widget.display('{:.2e}'.format(value))
        threshold = float(self.conf.get(threshold_key, float('inf')))
        if value != -1 and value > threshold:
            label_widget.setStyleSheet("color: red")
        else:
            label_widget.setStyleSheet("color: black")

    def update_vacuum_main(self, value):
        self._update_gauge(self.vacuum_main, self.label_212, value, 'vacuum_threshold_main')

    def update_vacuum_buffer(self, value):
        self._update_gauge(self.vacuum_buffer, self.label_211, value, 'vacuum_threshold_buffer')

    def update_vacuum_buffer_back(self, value):
        self._update_gauge(self.vacuum_buffer_back, self.label_214, value, 'vacuum_threshold_buffer_back')

    def update_vacuum_load_back(self, value):
        self._update_gauge(self.vacuum_load_lock_back, self.label_213, value, 'vacuum_threshold_load_lock_back')

    def update_vacuum_load(self, value):
        self._update_gauge(self.vacuum_load_lock, self.label_210, value, 'vacuum_threshold_load_lock')
        self._sync_pump_action_styles()

    def update_vacuum_cryo_load_lock(self, value):
        self._update_gauge(self.vacuum_cryo_load_lock, self.label_216, value, 'vacuum_threshold_cryo_load_lock')
        self._sync_pump_action_styles()

    def update_vacuum_cryo_load_lock_back(self, value):
        self._update_gauge(self.vacuum_cryo_load_lock_back, self.label_217, value, 'vacuum_threshold_cryo_load_lock_back')

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
        self.Error.setText(
            _translate("OXCART", "<html><head/><body><p><span style=\" color:#ff0000;\"></span></p></body></html>")
        )

        self.timer.stop()

    @staticmethod
    def _set_action_button_active(button, active, default_style):
        """Show an active toggle action with the same green used by Set T."""
        if active:
            button.setStyleSheet("QPushButton{\nbackground: rgb(0, 255, 26)\n}")
        else:
            button.setStyleSheet(default_style)

    def _sync_pump_action_styles(self):
        """Synchronize vent button colors with the confirmed pump states."""
        self._set_action_button_active(
            self.pump_load_lock_switch,
            not bool(self.variables.flag_pump_load_lock),
            self.pump_load_lock_default_style,
        )
        self._set_action_button_active(
            self.pump_cryo_load_lock_switch,
            not bool(self.variables.flag_pump_cryo_load_lock),
            self.pump_cryo_load_lock_default_style,
        )

    def pump_switch_ll(self):
        """
        Switch the pump on or off
        Args:
                None

        Return:
                None
        """
        try:
            if (
                not self.variables.start_flag
                and not self.variables.flag_main_gate
                and not self.variables.flag_cryo_gate
                and not self.variables.flag_load_gate
            ):
                if self.variables.flag_pump_load_lock:
                    self._set_action_button_active(
                        self.pump_load_lock_switch, True, self.pump_load_lock_default_style
                    )
                    self.variables.flag_pump_load_lock_click = True
                    self.pump_load_lock_switch.setEnabled(False)
                    time.sleep(1)
                    self.pump_load_lock_switch.setEnabled(True)
                elif not self.variables.flag_pump_load_lock:
                    self._set_action_button_active(
                        self.pump_load_lock_switch, False, self.pump_load_lock_default_style
                    )
                    self.variables.flag_pump_load_lock_click = True
                    self.pump_load_lock_switch.setEnabled(False)
                    time.sleep(1)
                    self.pump_load_lock_switch.setEnabled(True)
                self._sync_pump_action_styles()
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
            if (
                not self.variables.start_flag
                and not self.variables.flag_main_gate
                and not self.variables.flag_cryo_gate
                and not self.variables.flag_load_gate
            ):
                if self.variables.flag_pump_cryo_load_lock:
                    # About to fully vent the CLL (stop the backing pump).
                    # Make the operator confirm first - see warning text.
                    if not self._confirm_full_vent_cll():
                        return
                    self._set_action_button_active(
                        self.pump_cryo_load_lock_switch, True, self.pump_cryo_load_lock_default_style
                    )
                    self.variables.flag_pump_cryo_load_lock_click = True
                    self.pump_cryo_load_lock_switch.setEnabled(False)
                    time.sleep(1)
                    self.pump_cryo_load_lock_switch.setEnabled(False)
                elif not self.variables.flag_pump_cryo_load_lock:
                    self._set_action_button_active(
                        self.pump_cryo_load_lock_switch, False, self.pump_cryo_load_lock_default_style
                    )
                    self.variables.flag_pump_cryo_load_lock_click = True
                    self.pump_cryo_load_lock_switch.setEnabled(False)
                    time.sleep(1)
                    self.pump_cryo_load_lock_switch.setEnabled(False)
                self._sync_pump_action_styles()
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

    def _confirm_full_vent_cll(self):
        """Confirm the operator really wants to fully vent the cryo load lock.

        The cryo head vacuum depends on the CLL backing pump; fully venting
        the CLL stops that backing and will spoil the cryo head vacuum. Warn
        the operator so they can check everything before venting.

        Args:
                None

        Return:
                True if the operator confirmed the vent, False if cancelled.
        """
        warning = QtWidgets.QMessageBox(parent=self.pump_cryo_load_lock_switch)
        warning.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        warning.setWindowTitle("Confirm full CLL vent")
        warning.setText("You are about to fully vent the cryo load lock (CLL).")
        warning.setInformativeText(
            "The cryo head vacuum depends on the CLL backing pump. Fully "
            "venting the CLL stops that backing and will spoil the cryo head "
            "vacuum.\n\nCheck everything before venting the CLL.\n\n"
            "Do you want to continue?"
        )
        warning.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )
        warning.setDefaultButton(QtWidgets.QMessageBox.StandardButton.No)
        return warning.exec() == QtWidgets.QMessageBox.StandardButton.Yes

    def vent_cryo_load_lock_partial(self):
        """Toggle a fast partial vent of the cryo load lock (CLL).

        Sequences the three CLL exchange valves for a fast sample/cryo swap.
        See config.toml (the ``cll_*`` keys) for the valve wiring and the full
        state table.

        Starting a vent is interlocked: it is refused while an experiment is
        running or any gate is open. Stopping a vent is always allowed.

        Press (start venting):
                - CLL backing valve OFF and CLL Turbo valve OFF (immediately)
                - CLL vent valve ON after ``cll_vent_on_delay`` s (default 2 s)
        Deselect (stop venting / restore pumping):
                - CLL vent valve OFF and CLL backing valve ON (immediately)
                - CLL backing valve OFF again after ``cll_backing_off_delay`` s (default 90 s)
                - CLL Turbo valve ON after ``cll_turbo_on_delay`` s (default 90 s)

        The delayed steps are guarded by ``flag_vent_cll_partial`` so a quick
        press/deselect within the delay window cancels the pending action.

        All three valves are level-controlled and energise-to-open. The
        backing/Turbo valves are single SSR channels on the gates NI
        (Dev2/USB-6525), which latches its output; the vent valve is held on a
        separate NI USB-6501. Long waits use QTimer.singleShot so the GUI stays
        responsive.

        Args:
                None

        Return:
                None
        """
        try:
            if not self.flag_vent_cll_partial:
                # ---- start venting (sample/cryo exchange) ----
                # Interlock: opening the CLL to atmosphere is only allowed when
                # no experiment is running and every gate is closed.
                if not (
                    not self.variables.start_flag
                    and not self.variables.flag_main_gate
                    and not self.variables.flag_cryo_gate
                    and not self.variables.flag_load_gate
                ):
                    if self.variables.start_flag:
                        self.error_message("!!! An experiment is running !!!")
                    else:
                        self.error_message("!!! First Close all the Gates !!!")
                    self.timer.start(8000)
                    return
                self.flag_vent_cll_partial = True
                self._set_action_button_active(
                    self.vent_cryo_load_lock_partial_switch, True, self.vent_partial_default_style
                )
                # Backing and Turbo valves off immediately.
                self._set_cll_valve_6525(self.conf['cll_backing_valve_line'], False)
                self._set_cll_valve_6525(self.conf['cll_turbo_valve_line'], False)
                # Vent valve on after a short delay (then held on).
                delay_ms = int(float(self.conf.get('cll_vent_on_delay', 2)) * 1000)
                QTimer.singleShot(delay_ms, self._vent_cll_press_delayed)
                self.error_message("!!! Venting CLL for sample/cryo exchange !!!")
            else:
                # ---- stop venting / restore pumping ----
                # Always allowed - restoring the pumps is the safe direction.
                self.flag_vent_cll_partial = False
                self._set_action_button_active(
                    self.vent_cryo_load_lock_partial_switch, False, self.vent_partial_default_style
                )
                # Close the vent valve and re-open the backing valve immediately.
                self._set_vent_valve(True)  # True = closed, False = open
                self.variables.flag_vent_cryo_load_lock_partial = False
                self._set_cll_valve_6525(self.conf['cll_backing_valve_line'], True)
                # Backing valve closes again after a delay; Turbo valve opens
                # after a longer delay (protects the turbo). Both are guarded
                # against a re-press within the delay window.
                backing_off_ms = int(float(self.conf.get('cll_backing_off_delay', 90)) * 1000)
                QTimer.singleShot(backing_off_ms, self._vent_cll_deselect_backing_off)
                turbo_on_ms = int(float(self.conf.get('cll_turbo_on_delay', 90)) * 1000)
                QTimer.singleShot(turbo_on_ms, self._vent_cll_deselect_turbo_on)
                self.error_message("!!! CLL vent closed - restoring pumps !!!")
        except Exception as e:
            print('Error in vent_cryo_load_lock_partial function')
            print(e)

    def _vent_cll_press_delayed(self):
        """Delayed part of the "Vent CLL" press sequence: open the vent valve.

        Guarded by ``flag_vent_cll_partial`` so a quick deselect within the
        delay window cancels it (leaves the vent valve closed).

        Args:
                None

        Return:
                None
        """
        if self.flag_vent_cll_partial:
            self._set_vent_valve(False) # True = closed, False = open
            self.variables.flag_vent_cryo_load_lock_partial = True

    def _vent_cll_deselect_backing_off(self):
        """Delayed part of the "Vent CLL" deselect sequence: close the backing valve.

        On deselect the backing valve opens immediately and then closes again
        after ``cll_backing_off_delay`` s. Guarded by ``flag_vent_cll_partial``
        so a quick re-press within the delay window cancels it.

        Args:
                None

        Return:
                None
        """
        if not self.flag_vent_cll_partial:
            self._set_cll_valve_6525(self.conf['cll_backing_valve_line'], False)

    def _vent_cll_deselect_turbo_on(self):
        """Delayed part of the "Vent CLL" deselect sequence: open the Turbo valve.

        Guarded by ``flag_vent_cll_partial`` so a quick re-press within the
        delay window cancels it (leaves the Turbo valve closed while venting).

        Args:
                None

        Return:
                None
        """
        if not self.flag_vent_cll_partial:
            self._set_cll_valve_6525(self.conf['cll_turbo_valve_line'], True)

    def _set_cll_valve_6525(self, line_num, state):
        """Set one CLL SSR valve (Dev2 / USB-6525) open or closed and hold it.

        The CLL backing/Turbo valves are single solid-state-relay channels on
        the gates NI (``COM_PORT_gates`` -> Dev2, a USB-6525). They are
        level-controlled and energise-to-open: line HIGH = SSR closed = valve
        powered/OPEN, line LOW = valve closed. The USB-6525 latches its output
        state after the task closes (until reprogrammed or powered off), so a
        single momentary write holds the valve - no persistent task is kept,
        which also avoids colliding with the gates' tasks on the same port.

        Only the requested line is placed in the task, so writing it never
        disturbs the gate lines (line0..5). Errors are logged, not raised.

        Args:
                line_num: SSR channel / DO line on ``COM_PORT_gates`` (Dev2).
                          Must be a CLL valve line (6 or 7), never a gate line.
                state: True to open (energise) the valve, False to close it.

        Return:
                None
        """
        import nidaqmx
        try:
            task = nidaqmx.Task()
        except Exception as e:
            print('Error creating NI task for CLL valve')
            print(e)
            return
        try:
            task.do_channels.add_do_chan(self.conf['COM_PORT_gates'] + 'line%s' % line_num)
            task.start()
            task.write([bool(state)])
        except Exception as e:
            print('Error setting CLL valve line %s' % line_num)
            print(e)
        finally:
            # Close the task immediately; the USB-6525 latches the value.
            try:
                task.close()
            except Exception:
                pass

    def _set_vent_valve(self, state):
        """Drive the CLL vent-valve relay on/off and HOLD the line.

        The vent valve is a level-controlled relay on the NI USB-6501
        (``COM_PORT_cll_vent_valve``). The 6501 DIO lines are open-collector
        with a weak on-board 4.7 kOhm pull-up, so a line that is NOT actively
        driven floats HIGH (~5 V) and leaves an active-high relay stuck ON.
        The DAQmx task is therefore kept OPEN for the life of the panel and
        just writes the level:
                True  -> relay ON  (vent open)   - line released/high
                False -> relay OFF (vent closed)  - line actively driven LOW
        The task is NOT closed on the off-path: closing it would release the
        line back to the pull-up and re-energise the relay (this was the bug
        where the vent could never be switched off).

        Args:
                state: True to open (vent), False to close the vent valve.

        Return:
                None
        """
        import nidaqmx
        try:
            if self._vent_valve_task is None:
                self._vent_valve_task = nidaqmx.Task()
                self._vent_valve_task.do_channels.add_do_chan(self.conf['COM_PORT_cll_vent_valve'])
                self._vent_valve_task.start()
            self._vent_valve_task.write([bool(state)])
        except Exception as e:
            print('Error setting CLL vent valve')
            print(e)
            # Best-effort cleanup so a half-open task does not wedge the line.
            try:
                if self._vent_valve_task is not None:
                    self._vent_valve_task.close()
            except Exception:
                pass
            self._vent_valve_task = None

    def error_message(self, message):
        """
        Show the warning message
        Args:
                message: the message to be shown

        Return:
                None
        """
        _translate = QtCore.QCoreApplication.translate
        self.Error.setText(
            _translate(
                "OXCART", "<html><head/><body><p><span style=\" color:#ff0000;\">" + message + "</span></p></body></html>"
            )
        )
        # Auto-hide the warning after 8 seconds so every message clears itself
        self.timer.start(8000)

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
        # Flush and stop the LL baking log if a run is in progress.
        self._stop_ll_baking_log()
        # The backing/Turbo valves are on the USB-6525, which latches its
        # output, so their state is preserved across a software restart with no
        # action here. The vent valve is on the USB-6501, whose line floats
        # HIGH via its pull-up once the process releases it on exit - i.e. the
        # vent relay energises (CLL vents) on shutdown. This is a hardware trait
        # of the 6501 that software cannot prevent; use the relay NC contact or
        # an active-drive + pull-down if a fail-safe-closed vent is required.


class SignalEmitter(QObject):
    """
    Signal emitter class for emitting signals related to vacuum and pumps control.
    """

    temp_stage = pyqtSignal(float)
    temp_cryo_head = pyqtSignal(float)
    temp_cryo_head_inside = pyqtSignal(float)
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
    Pumps_vacuum = QtWidgets.QWidget()
    signal_emitter = SignalEmitter()
    ui = Ui_Pumps_Vacuum(shared.variables, conf, signal_emitter)
    ui.setupUi(Pumps_vacuum)
    Pumps_vacuum.show()
    sys.exit(app.exec())
