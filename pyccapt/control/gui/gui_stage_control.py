import sys

from PyQt6 import QtCore, QtGui, QtWidgets

from pyccapt.control.core import runtime
from pyccapt.control.smaract_mcs2 import mcs2_stage


def _make_lcd(parent):
	lcd = QtWidgets.QLCDNumber(parent=parent)
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


def _make_axis_slider(parent, lo, hi, default):
	slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, parent=parent)
	slider.setMinimum(lo)
	slider.setMaximum(hi)
	slider.setValue(default)
	slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
	slider.setTickInterval(1)
	slider.setMinimumWidth(160)
	return slider


class Ui_Stage_Control(object):

	def __init__(self, variables, conf):
		self.variables = variables
		self.conf = conf

		self.stage_device = None
		self._poll_timer = None
		self._connect_error = ""  # original cause - kept across button clicks
		self.flag_super_user = False  # Reference is gated behind Override Access

		self._speed_max_mm_s = float(self.conf.get('stage_speed_max_mm_s', 1.0))
		self._speed_max_level = int(self.conf.get('stage_speed_level_max', 11))
		self._speed_min_level = int(self.conf.get('stage_speed_level_min', 1))
		self._speed_default = int(self.conf.get('stage_speed_level_default', 5))
		self._click_duration_s = float(self.conf.get('stage_click_duration_s', 0.2))
		self._speed_table = self.conf.get('stage_speed_table_mm_s') or None
		self._home_target_m = (
			float(self.conf.get('stage_home_x_mm', 0.0)) * 1e-3,
			float(self.conf.get('stage_home_y_mm', 0.0)) * 1e-3,
			float(self.conf.get('stage_home_z_mm', 0.0)) * 1e-3,
		)
		self._locator = self.conf.get('stage_smartact_main', '')

	# ------------------------------------------------------------------- ui

	def setupUi(self, Stage_Control):
		Stage_Control.setObjectName("Stage_Control")
		Stage_Control.resize(1020, 230)
		self.gridLayout_5 = QtWidgets.QGridLayout(Stage_Control)
		self.gridLayout_3 = QtWidgets.QGridLayout()

		# --- Position panel: header + 3 axes x (label, mm, um, nm) ---------
		self.gridLayout_4 = QtWidgets.QGridLayout()
		header_font = QtGui.QFont();
		header_font.setBold(True);
		header_font.setPointSize(8)
		for col, name in enumerate(("", "mm", "µm", "nm"), start=0):
			lab = QtWidgets.QLabel(name, parent=Stage_Control)
			lab.setFont(header_font)
			lab.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
			self.gridLayout_4.addWidget(lab, 0, col, 1, 1)

		bold = QtGui.QFont();
		bold.setBold(True)
		self.label_19 = QtWidgets.QLabel("x", parent=Stage_Control);
		self.label_19.setFont(bold)
		self.label_17 = QtWidgets.QLabel("y", parent=Stage_Control);
		self.label_17.setFont(bold)
		self.label_18 = QtWidgets.QLabel("z", parent=Stage_Control);
		self.label_18.setFont(bold)
		self.stage_x_mm = _make_lcd(Stage_Control)
		self.stage_x_um = _make_lcd(Stage_Control)
		self.stage_x_nm = _make_lcd(Stage_Control)
		self.stage_y_mm = _make_lcd(Stage_Control)
		self.stage_y_um = _make_lcd(Stage_Control)
		self.stage_y_nm = _make_lcd(Stage_Control)
		self.stage_z_mm = _make_lcd(Stage_Control)
		self.stage_z_um = _make_lcd(Stage_Control)
		self.stage_z_nm = _make_lcd(Stage_Control)

		# Backwards-compat single LCDs (hidden); some external code still
		# reads .stage_x_cord etc. as a position-in-µm value.
		self.stage_x_cord = QtWidgets.QLCDNumber(parent=Stage_Control)
		self.stage_y_cord = QtWidgets.QLCDNumber(parent=Stage_Control)
		self.stage_z_cord = QtWidgets.QLCDNumber(parent=Stage_Control)
		for w in (self.stage_x_cord, self.stage_y_cord, self.stage_z_cord):
			w.setVisible(False)

		for row, (lbl, mm, um, nm) in enumerate(
				(
						(self.label_19, self.stage_x_mm, self.stage_x_um, self.stage_x_nm),
						(self.label_17, self.stage_y_mm, self.stage_y_um, self.stage_y_nm),
						(self.label_18, self.stage_z_mm, self.stage_z_um, self.stage_z_nm),
				),
				start=1,
		):
			self.gridLayout_4.addWidget(lbl, row, 0, 1, 1)
			self.gridLayout_4.addWidget(mm, row, 1, 1, 1)
			self.gridLayout_4.addWidget(um, row, 2, 1, 1)
			self.gridLayout_4.addWidget(nm, row, 3, 1, 1)
		self.gridLayout_3.addLayout(self.gridLayout_4, 0, 0, 1, 1)

		# --- Per-axis speed sliders (X, Y, Z) ------------------------------
		self.gridLayout_2 = QtWidgets.QGridLayout()
		# Header
		header_label = QtWidgets.QLabel("Speed", parent=Stage_Control)
		header_label.setFont(bold)
		header_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
		self.gridLayout_2.addWidget(header_label, 0, 1, 1, 1)

		self.label_speed_x = QtWidgets.QLabel("X", parent=Stage_Control);
		self.label_speed_x.setFont(bold)
		self.label_speed_y = QtWidgets.QLabel("Y", parent=Stage_Control);
		self.label_speed_y.setFont(bold)
		self.label_speed_z = QtWidgets.QLabel("Z", parent=Stage_Control);
		self.label_speed_z.setFont(bold)

		self.stage_speed_x = _make_axis_slider(Stage_Control, self._speed_min_level,
		                                       self._speed_max_level, self._speed_default)
		self.stage_speed_y = _make_axis_slider(Stage_Control, self._speed_min_level,
		                                       self._speed_max_level, self._speed_default)
		self.stage_speed_z = _make_axis_slider(Stage_Control, self._speed_min_level,
		                                       self._speed_max_level, self._speed_default)

		self.stage_speed_x_label = QtWidgets.QLabel(parent=Stage_Control);
		self.stage_speed_x_label.setMinimumWidth(230)
		self.stage_speed_y_label = QtWidgets.QLabel(parent=Stage_Control);
		self.stage_speed_y_label.setMinimumWidth(230)
		self.stage_speed_z_label = QtWidgets.QLabel(parent=Stage_Control);
		self.stage_speed_z_label.setMinimumWidth(230)

		for row, (lbl, sl, val) in enumerate(
				(
						(self.label_speed_x, self.stage_speed_x, self.stage_speed_x_label),
						(self.label_speed_y, self.stage_speed_y, self.stage_speed_y_label),
						(self.label_speed_z, self.stage_speed_z, self.stage_speed_z_label),
				),
				start=1,
		):
			self.gridLayout_2.addWidget(lbl, row, 0, 1, 1)
			self.gridLayout_2.addWidget(sl, row, 1, 1, 1)
			self.gridLayout_2.addWidget(val, row, 2, 1, 1)

		# Backwards-compat aliases
		self.stage_speed_lr = self.stage_speed_x
		self.stage_speed_ud = self.stage_speed_y
		self.stage_speed_fb = self.stage_speed_z

		self.gridLayout_3.addLayout(self.gridLayout_2, 0, 1, 1, 1)

		# --- Direction buttons (X/Y plane) ---------------------------------
		self.gridLayout = QtWidgets.QGridLayout()
		self.gridLayout.addItem(QtWidgets.QSpacerItem(40, 20,
		                                              QtWidgets.QSizePolicy.Policy.Expanding,
		                                              QtWidgets.QSizePolicy.Policy.Minimum), 0, 0, 1, 1)
		self.stage_up = QtWidgets.QPushButton("up", parent=Stage_Control)
		self.stage_up.setMinimumSize(QtCore.QSize(50, 25))
		self.gridLayout.addWidget(self.stage_up, 0, 1, 1, 1)
		self.gridLayout.addItem(QtWidgets.QSpacerItem(40, 20,
		                                              QtWidgets.QSizePolicy.Policy.Expanding,
		                                              QtWidgets.QSizePolicy.Policy.Minimum), 0, 2, 1, 1)
		self.stage_left = QtWidgets.QPushButton("Left", parent=Stage_Control)
		self.stage_left.setMinimumSize(QtCore.QSize(50, 25))
		self.gridLayout.addWidget(self.stage_left, 1, 0, 1, 1)
		self.gridLayout.addItem(QtWidgets.QSpacerItem(40, 20,
		                                              QtWidgets.QSizePolicy.Policy.Expanding,
		                                              QtWidgets.QSizePolicy.Policy.Minimum), 1, 1, 1, 1)
		self.stage_right = QtWidgets.QPushButton("Right", parent=Stage_Control)
		self.stage_right.setMinimumSize(QtCore.QSize(50, 25))
		self.gridLayout.addWidget(self.stage_right, 1, 2, 1, 1)
		self.gridLayout.addItem(QtWidgets.QSpacerItem(40, 20,
		                                              QtWidgets.QSizePolicy.Policy.Expanding,
		                                              QtWidgets.QSizePolicy.Policy.Minimum), 2, 0, 1, 1)
		self.stage_down = QtWidgets.QPushButton("Down", parent=Stage_Control)
		self.stage_down.setMinimumSize(QtCore.QSize(50, 25))
		self.gridLayout.addWidget(self.stage_down, 2, 1, 1, 1)
		self.gridLayout.addItem(QtWidgets.QSpacerItem(40, 20,
		                                              QtWidgets.QSizePolicy.Policy.Expanding,
		                                              QtWidgets.QSizePolicy.Policy.Minimum), 2, 2, 1, 1)
		self.gridLayout_3.addLayout(self.gridLayout, 0, 2, 1, 1)

		# --- Forward / backward (Z) ----------------------------------------
		self.verticalLayout = QtWidgets.QVBoxLayout()
		self.stage_forward = QtWidgets.QPushButton("Forward", parent=Stage_Control)
		self.verticalLayout.addWidget(self.stage_forward)
		self.verticalLayout.addItem(QtWidgets.QSpacerItem(17, 24,
		                                                  QtWidgets.QSizePolicy.Policy.Minimum,
		                                                  QtWidgets.QSizePolicy.Policy.Expanding))
		self.stage_backward = QtWidgets.QPushButton("Backward", parent=Stage_Control)
		self.verticalLayout.addWidget(self.stage_backward)
		self.gridLayout_3.addLayout(self.verticalLayout, 0, 3, 1, 1)

		# --- Home / Reference / Stop / Override ----------------------------
		home_layout = QtWidgets.QVBoxLayout()
		self.stage_home = QtWidgets.QPushButton("Home", parent=Stage_Control)
		home_layout.addWidget(self.stage_home)
		self.stage_reference = QtWidgets.QPushButton("Reference", parent=Stage_Control)
		# Reference moves the stage on its own to find the physical reference
		# mark - dangerous if anything is in the way.  Gated behind Override
		# Access, same pattern as the gates / pumps GUIs.
		self.stage_reference.setEnabled(False)
		home_layout.addWidget(self.stage_reference)
		self.stage_stop = QtWidgets.QPushButton("STOP", parent=Stage_Control)
		self.stage_stop.setStyleSheet(
			"QPushButton{background: rgb(220,80,80); color: white; font-weight: bold;}"
		)
		home_layout.addWidget(self.stage_stop)
		self.superuser = QtWidgets.QPushButton("Override Access", parent=Stage_Control)
		self.superuser.setStyleSheet(
			"QPushButton{background: rgb(193, 193, 193)}"
		)
		self._original_superuser_style = self.superuser.styleSheet()
		home_layout.addWidget(self.superuser)
		self.gridLayout_3.addLayout(home_layout, 0, 4, 1, 1)

		# --- Status / error label ------------------------------------------
		self.Error = QtWidgets.QLabel(parent=Stage_Control)
		self.Error.setMinimumSize(QtCore.QSize(500, 30))
		efont = QtGui.QFont();
		efont.setPointSize(10);
		efont.setBold(True)
		self.Error.setFont(efont)
		self.Error.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
		self.Error.setWordWrap(True)
		self.gridLayout_3.addWidget(self.Error, 1, 0, 1, 5)

		self.gridLayout_5.addLayout(self.gridLayout_3, 0, 0, 1, 1)

		self.retranslateUi(Stage_Control)
		QtCore.QMetaObject.connectSlotsByName(Stage_Control)

		self._connect_signals()
		for sl in (self.stage_speed_x, self.stage_speed_y, self.stage_speed_z):
			self._update_speed_label(sl)
		self._connect_device()

	def retranslateUi(self, Stage_Control):
		_translate = QtCore.QCoreApplication.translate
		Stage_Control.setWindowTitle(_translate("Stage_Control", "PyCCAPT Stage Control"))
		Stage_Control.setWindowIcon(QtGui.QIcon('./files/logo.png'))

	# ------------------------------------------------------------------ wiring

	def _connect_signals(self):
		self.stage_speed_x.valueChanged.connect(lambda _v: self._update_speed_label(self.stage_speed_x))
		self.stage_speed_y.valueChanged.connect(lambda _v: self._update_speed_label(self.stage_speed_y))
		self.stage_speed_z.valueChanged.connect(lambda _v: self._update_speed_label(self.stage_speed_z))

		self.stage_left.clicked.connect(lambda: self._jog_axis(mcs2_stage.AXIS_X, -1))
		self.stage_right.clicked.connect(lambda: self._jog_axis(mcs2_stage.AXIS_X, +1))
		self.stage_up.clicked.connect(lambda: self._jog_axis(mcs2_stage.AXIS_Y, +1))
		self.stage_down.clicked.connect(lambda: self._jog_axis(mcs2_stage.AXIS_Y, -1))
		self.stage_forward.clicked.connect(lambda: self._jog_axis(mcs2_stage.AXIS_Z, +1))
		self.stage_backward.clicked.connect(lambda: self._jog_axis(mcs2_stage.AXIS_Z, -1))
		self.stage_home.clicked.connect(self._go_home)
		self.stage_reference.clicked.connect(self._reference)
		self.stage_stop.clicked.connect(self._stop_stage)
		self.superuser.clicked.connect(self._super_user_access)

	def _connect_device(self):
		if not self._locator:
			self._connect_error = ("No SmarAct stage locator configured "
			                       "(stage_smartact_main in config.toml).")
			self._set_error(self._connect_error)
			self._set_movement_enabled(False)
			return
		try:
			self.stage_device = mcs2_stage.SmarActStage(self._locator)
		except mcs2_stage.SmarActStageError as exc:
			self.stage_device = None
			self._connect_error = str(exc)
			self._set_error(self._connect_error)
			self._set_movement_enabled(False)
			return
		self._set_error("")
		self._set_movement_enabled(True)
		self._poll_timer = QtCore.QTimer()
		self._poll_timer.setInterval(500)
		self._poll_timer.timeout.connect(self._refresh_position)
		self._poll_timer.start()
		self._refresh_position()

	def _set_movement_enabled(self, enabled):
		for btn in (self.stage_up, self.stage_down, self.stage_left,
		            self.stage_right, self.stage_forward, self.stage_backward,
		            self.stage_home):
			btn.setEnabled(enabled)
		# Reference stays gated behind Override Access (and also requires
		# the device to be connected).
		self.stage_reference.setEnabled(enabled and self.flag_super_user)

	# STOP stays clickable so the user can always abort.

	def _super_user_access(self):
		"""Toggle Override Access; matches the pattern in gui_gates / gui_pumps."""
		if not self.flag_super_user:
			warning = QtWidgets.QMessageBox(parent=self.superuser)
			warning.setIcon(QtWidgets.QMessageBox.Icon.Warning)
			warning.setWindowTitle("Confirm Access Override")
			warning.setText("Stage Reference moves all axes on its own to find the physical reference mark.")
			warning.setInformativeText("Make sure nothing is in the way of the stage. Continue?")
			warning.setStandardButtons(
				QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
			)
			warning.setDefaultButton(QtWidgets.QMessageBox.StandardButton.No)
			if warning.exec() != QtWidgets.QMessageBox.StandardButton.Yes:
				self._set_error("Override Access canceled.")
				return
			self.flag_super_user = True
			self.superuser.setStyleSheet("QPushButton{background: rgb(0, 255, 26)}")
			self._set_error("!!! Override Access Granted !!!")
		else:
			self.flag_super_user = False
			self.superuser.setStyleSheet(self._original_superuser_style)
			self._set_error("!!! Override Access deactivated !!!")
		self.stage_reference.setEnabled(self.flag_super_user and self.stage_device is not None)

	# --------------------------------------------------------------- handlers

	def _axis_velocity_m_s(self, axis):
		slider = (self.stage_speed_x, self.stage_speed_y, self.stage_speed_z)[axis]
		return mcs2_stage.speed_level_to_m_s(
			slider.value(), self._speed_max_level, self._speed_max_mm_s,
			table=self._speed_table,
		)

	def _update_speed_label(self, slider):
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
			self.stage_speed_x: self.stage_speed_x_label,
			self.stage_speed_y: self.stage_speed_y_label,
			self.stage_speed_z: self.stage_speed_z_label,
		}
		mapping[slider].setText(text)

	def _jog_axis(self, axis, sign):
		if self.stage_device is None:
			self._set_error(self._connect_error or "Stage not connected.")
			return
		vel = self._axis_velocity_m_s(axis)
		step_m = mcs2_stage.click_step_m(vel, self._click_duration_s)
		try:
			self.stage_device.move_relative_axis(
				axis=axis, delta_m=sign * step_m, velocity_m_s=vel, wait=False,
			)
		except mcs2_stage.SmarActStageError as exc:
			self._set_error(f"Move failed: {exc}")

	def _go_home(self):
		if self.stage_device is None:
			self._set_error(self._connect_error or "Stage not connected.")
			return
		x_m, y_m, z_m = self._home_target_m
		# Use the X-axis slider as the global home velocity (tunable).
		try:
			self.stage_device.move_absolute(
				x_m=x_m, y_m=y_m, z_m=z_m,
				velocity_m_s=self._axis_velocity_m_s(mcs2_stage.AXIS_X),
				wait=False,
			)
		except mcs2_stage.SmarActStageError as exc:
			self._set_error(f"Home failed: {exc}")

	def _reference(self):
		if self.stage_device is None:
			self._set_error(self._connect_error or "Stage not connected.")
			return
		try:
			self.stage_device.find_reference()
		except mcs2_stage.SmarActStageError as exc:
			self._set_error(f"Reference search failed: {exc}")

	def _stop_stage(self):
		if self.stage_device is None:
			return
		self.stage_device.stop()

	def _refresh_position(self):
		if self.stage_device is None:
			return
		try:
			pos = self.stage_device.get_position()
		except mcs2_stage.SmarActStageError as exc:
			self._set_error(f"Position read failed: {exc}")
			return
		self._set_axis_display(pos['x'], self.stage_x_mm, self.stage_x_um, self.stage_x_nm,
		                       self.stage_x_cord)
		self._set_axis_display(pos['y'], self.stage_y_mm, self.stage_y_um, self.stage_y_nm,
		                       self.stage_y_cord)
		self._set_axis_display(pos['z'], self.stage_z_mm, self.stage_z_um, self.stage_z_nm,
		                       self.stage_z_cord)

	@staticmethod
	def _set_axis_display(value_m, mm_lcd, um_lcd, nm_lcd, single_lcd):
		mm, um, nm = mcs2_stage.split_meters_mm_um_nm(value_m)
		mm_lcd.display(mm)
		um_lcd.display(um)
		nm_lcd.display(nm)
		single_lcd.display(value_m * 1e6)  # legacy: micrometers

	def _set_error(self, message):
		if not message:
			self.Error.setText("")
			return
		self.Error.setText(
			f'<html><body><p style="color:#c00000;">{message}</p></body></html>'
		)

	def stop(self):
		if self._poll_timer is not None:
			self._poll_timer.stop()
		if self.stage_device is not None:
			self.stage_device.close()
			self.stage_device = None


class StageControlWindow(QtWidgets.QWidget):
	closed = QtCore.pyqtSignal()

	def __init__(self, gui_stage_control, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.gui_stage_control = gui_stage_control

	def closeEvent(self, event):
		event.ignore()
		self.hide()
		self.closed.emit()

	def setWindowStyleFusion(self):
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
