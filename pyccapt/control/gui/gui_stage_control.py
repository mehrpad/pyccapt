import multiprocessing
import os
import sys

from PyQt6 import QtCore, QtGui, QtWidgets

# Local module and scripts
from pyccapt.control.core import runtime


class Ui_Stage_Control(object):

	def __init__(self, variables, conf):
		"""
		Constructor for the Stage Control UI class.

		Args:
			variables (object): Global experiment variables.
			conf (dict): Configuration settings.

		Attributes:
			variables: Global experiment variables.
			conf: Configuration settings.
		"""
		self.variables = variables
		self.conf = conf

	def setupUi(self, Stage_Control):
		"""
		Set up the Stage Control UI.
		Args:
			Stage_Control (object): The Stage Control UI object.

		Return:
			None
		"""
		Stage_Control.setObjectName("Stage_Control")
		Stage_Control.resize(636, 161)
		self.gridLayout_5 = QtWidgets.QGridLayout(Stage_Control)
		self.gridLayout_5.setObjectName("gridLayout_5")
		self.gridLayout_3 = QtWidgets.QGridLayout()
		self.gridLayout_3.setObjectName("gridLayout_3")
		self.gridLayout_4 = QtWidgets.QGridLayout()
		self.gridLayout_4.setObjectName("gridLayout_4")
		self.label_19 = QtWidgets.QLabel(parent=Stage_Control)
		font = QtGui.QFont()
		font.setBold(True)
		self.label_19.setFont(font)
		self.label_19.setObjectName("label_19")
		self.gridLayout_4.addWidget(self.label_19, 0, 0, 1, 1)
		self.stage_x_cord = QtWidgets.QLCDNumber(parent=Stage_Control)
		self.stage_x_cord.setObjectName("stage_x_cord")
		self.gridLayout_4.addWidget(self.stage_x_cord, 0, 1, 1, 1)
		self.label_17 = QtWidgets.QLabel(parent=Stage_Control)
		font = QtGui.QFont()
		font.setBold(True)
		self.label_17.setFont(font)
		self.label_17.setObjectName("label_17")
		self.gridLayout_4.addWidget(self.label_17, 1, 0, 1, 1)
		self.stage_y_cord = QtWidgets.QLCDNumber(parent=Stage_Control)
		self.stage_y_cord.setObjectName("stage_y_cord")
		self.gridLayout_4.addWidget(self.stage_y_cord, 1, 1, 1, 1)
		self.label_18 = QtWidgets.QLabel(parent=Stage_Control)
		font = QtGui.QFont()
		font.setBold(True)
		self.label_18.setFont(font)
		self.label_18.setObjectName("label_18")
		self.gridLayout_4.addWidget(self.label_18, 2, 0, 1, 1)
		self.stage_z_cord = QtWidgets.QLCDNumber(parent=Stage_Control)
		self.stage_z_cord.setObjectName("stage_z_cord")
		self.gridLayout_4.addWidget(self.stage_z_cord, 2, 1, 1, 1)
		self.gridLayout_3.addLayout(self.gridLayout_4, 0, 0, 1, 1)
		self.gridLayout_2 = QtWidgets.QGridLayout()
		self.gridLayout_2.setObjectName("gridLayout_2")
		self.label_14 = QtWidgets.QLabel(parent=Stage_Control)
		font = QtGui.QFont()
		font.setBold(True)
		self.label_14.setFont(font)
		self.label_14.setObjectName("label_14")
		self.gridLayout_2.addWidget(self.label_14, 0, 0, 1, 1)
		self.stage_speed_lr = QtWidgets.QDoubleSpinBox(parent=Stage_Control)
		self.stage_speed_lr.setStyleSheet("QDoubleSpinBox{\n"
		                                  "                                            background: rgb(223,223,233)\n"
		                                  "                                            }\n"
		                                  "                                        ")
		self.stage_speed_lr.setObjectName("stage_speed_lr")
		self.gridLayout_2.addWidget(self.stage_speed_lr, 0, 1, 1, 1)
		self.label_15 = QtWidgets.QLabel(parent=Stage_Control)
		font = QtGui.QFont()
		font.setBold(True)
		self.label_15.setFont(font)
		self.label_15.setObjectName("label_15")
		self.gridLayout_2.addWidget(self.label_15, 1, 0, 1, 1)
		self.stage_speed_ud = QtWidgets.QDoubleSpinBox(parent=Stage_Control)
		self.stage_speed_ud.setStyleSheet("QDoubleSpinBox{\n"
		                                  "                                            background: rgb(223,223,233)\n"
		                                  "                                            }\n"
		                                  "                                        ")
		self.stage_speed_ud.setObjectName("stage_speed_ud")
		self.gridLayout_2.addWidget(self.stage_speed_ud, 1, 1, 1, 1)
		self.label_16 = QtWidgets.QLabel(parent=Stage_Control)
		font = QtGui.QFont()
		font.setBold(True)
		self.label_16.setFont(font)
		self.label_16.setObjectName("label_16")
		self.gridLayout_2.addWidget(self.label_16, 2, 0, 1, 1)
		self.stage_speed_fb = QtWidgets.QDoubleSpinBox(parent=Stage_Control)
		self.stage_speed_fb.setStyleSheet("QDoubleSpinBox{\n"
		                                  "                                            background: rgb(223,223,233)\n"
		                                  "                                            }\n"
		                                  "                                        ")
		self.stage_speed_fb.setObjectName("stage_speed_fb")
		self.gridLayout_2.addWidget(self.stage_speed_fb, 2, 1, 1, 1)
		self.gridLayout_3.addLayout(self.gridLayout_2, 0, 1, 1, 1)
		self.gridLayout = QtWidgets.QGridLayout()
		self.gridLayout.setObjectName("gridLayout")
		spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
		                                   QtWidgets.QSizePolicy.Policy.Minimum)
		self.gridLayout.addItem(spacerItem, 0, 0, 1, 1)
		self.stage_up = QtWidgets.QPushButton(parent=Stage_Control)
		self.stage_up.setMinimumSize(QtCore.QSize(50, 25))
		self.stage_up.setMaximumSize(QtCore.QSize(16777215, 16777215))
		self.stage_up.setStyleSheet("")
		self.stage_up.setObjectName("stage_up")
		self.gridLayout.addWidget(self.stage_up, 0, 1, 1, 1)
		spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
		                                    QtWidgets.QSizePolicy.Policy.Minimum)
		self.gridLayout.addItem(spacerItem1, 0, 2, 1, 1)
		self.stage_left = QtWidgets.QPushButton(parent=Stage_Control)
		self.stage_left.setMinimumSize(QtCore.QSize(50, 25))
		self.stage_left.setMaximumSize(QtCore.QSize(16777215, 16777215))
		self.stage_left.setStyleSheet("")
		self.stage_left.setObjectName("stage_left")
		self.gridLayout.addWidget(self.stage_left, 1, 0, 1, 1)
		spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
		                                    QtWidgets.QSizePolicy.Policy.Minimum)
		self.gridLayout.addItem(spacerItem2, 1, 1, 1, 1)
		self.stage_right = QtWidgets.QPushButton(parent=Stage_Control)
		self.stage_right.setMinimumSize(QtCore.QSize(50, 25))
		self.stage_right.setMaximumSize(QtCore.QSize(16777215, 16777215))
		self.stage_right.setStyleSheet("")
		self.stage_right.setObjectName("stage_right")
		self.gridLayout.addWidget(self.stage_right, 1, 2, 1, 1)
		spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
		                                    QtWidgets.QSizePolicy.Policy.Minimum)
		self.gridLayout.addItem(spacerItem3, 2, 0, 1, 1)
		self.stage_down = QtWidgets.QPushButton(parent=Stage_Control)
		self.stage_down.setMinimumSize(QtCore.QSize(50, 25))
		self.stage_down.setMaximumSize(QtCore.QSize(16777215, 16777215))
		self.stage_down.setStyleSheet("")
		self.stage_down.setObjectName("stage_down")
		self.gridLayout.addWidget(self.stage_down, 2, 1, 1, 1)
		spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
		                                    QtWidgets.QSizePolicy.Policy.Minimum)
		self.gridLayout.addItem(spacerItem4, 2, 2, 1, 1)
		self.gridLayout_3.addLayout(self.gridLayout, 0, 2, 1, 1)
		self.verticalLayout = QtWidgets.QVBoxLayout()
		self.verticalLayout.setObjectName("verticalLayout")
		self.stage_forward = QtWidgets.QPushButton(parent=Stage_Control)
		self.stage_forward.setStyleSheet("")
		self.stage_forward.setObjectName("stage_forward")
		self.verticalLayout.addWidget(self.stage_forward)
		spacerItem5 = QtWidgets.QSpacerItem(17, 24, QtWidgets.QSizePolicy.Policy.Minimum,
		                                    QtWidgets.QSizePolicy.Policy.Expanding)
		self.verticalLayout.addItem(spacerItem5)
		self.stage_backward = QtWidgets.QPushButton(parent=Stage_Control)
		self.stage_backward.setStyleSheet("")
		self.stage_backward.setObjectName("stage_backward")
		self.verticalLayout.addWidget(self.stage_backward)
		self.gridLayout_3.addLayout(self.verticalLayout, 0, 3, 1, 1)
		self.stage_home = QtWidgets.QPushButton(parent=Stage_Control)
		self.stage_home.setStyleSheet("")
		self.stage_home.setObjectName("stage_home")
		self.gridLayout_3.addWidget(self.stage_home, 0, 4, 1, 1)
		self.Error = QtWidgets.QLabel(parent=Stage_Control)
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
		self.gridLayout_3.addWidget(self.Error, 1, 0, 1, 4)
		self.gridLayout_5.addLayout(self.gridLayout_3, 0, 0, 1, 1)

		self.retranslateUi(Stage_Control)
		QtCore.QMetaObject.connectSlotsByName(Stage_Control)
		Stage_Control.setTabOrder(self.stage_speed_lr, self.stage_speed_ud)
		Stage_Control.setTabOrder(self.stage_speed_ud, self.stage_speed_fb)
		Stage_Control.setTabOrder(self.stage_speed_fb, self.stage_left)
		Stage_Control.setTabOrder(self.stage_left, self.stage_right)
		Stage_Control.setTabOrder(self.stage_right, self.stage_up)
		Stage_Control.setTabOrder(self.stage_up, self.stage_down)
		Stage_Control.setTabOrder(self.stage_down, self.stage_forward)
		Stage_Control.setTabOrder(self.stage_forward, self.stage_backward)
		Stage_Control.setTabOrder(self.stage_backward, self.stage_home)

	def retranslateUi(self, Stage_Control):
		"""
				Set the text and titles of the UI elements
				Args:
					None

				Return:
					None
				"""
		_translate = QtCore.QCoreApplication.translate
		###
		# Stage_Control.setWindowTitle(_translate("Stage_Control", "Form"))
		Stage_Control.setWindowTitle(_translate("Stage_Control", "PyCCAPT Stage Control"))
		Stage_Control.setWindowIcon(QtGui.QIcon('./files/logo.png'))
		###
		self.label_19.setText(_translate("Stage_Control", "x"))
		self.label_17.setText(_translate("Stage_Control", "y"))
		self.label_18.setText(_translate("Stage_Control", "z"))
		self.label_14.setText(_translate("Stage_Control", "Speed L/R"))
		self.label_15.setText(_translate("Stage_Control", "Speed U/D"))
		self.label_16.setText(_translate("Stage_Control", "Speed F/B"))
		self.stage_up.setText(_translate("Stage_Control", "up"))
		self.stage_left.setText(_translate("Stage_Control", "Left"))
		self.stage_right.setText(_translate("Stage_Control", "Right"))
		self.stage_down.setText(_translate("Stage_Control", "Down"))
		self.stage_forward.setText(_translate("Stage_Control", "Forward"))
		self.stage_backward.setText(_translate("Stage_Control", "Backward"))
		self.stage_home.setText(_translate("Stage_Control", "Home"))
		self.Error.setText(_translate("Stage_Control", "<html><head/><body><p><br/></p></body></html>"))

	def stop(self):
		"""
		Stop any background processes, timers, or threads here
		Args:
			None

		Return:
			None
		"""
		# Add any additional cleanup code here
		pass


class StageControlWindow(QtWidgets.QWidget):
	"""
	Widget for the Stage Control window.
	"""
	closed = QtCore.pyqtSignal()  # Define a custom closed signal

	def __init__(self, gui_stage_control, *args, **kwargs):
		"""
		Constructor for the StageControlWindow class.

		Args:
			gui_stage_control: Instance of the StageControl.
			*args: Additional positional arguments.
			**kwargs: Additional keyword arguments.
		"""
		super().__init__(*args, **kwargs)
		self.gui_stage_control = gui_stage_control

	def closeEvent(self, event):
		"""
		Close event for the window.

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
	stage_control = QtWidgets.QWidget()
	ui = Ui_Stage_Control(shared.variables, conf)
	ui.setupUi(stage_control)
	stage_control.show()
	sys.exit(app.exec())

