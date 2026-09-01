import sys

import nidaqmx
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QPixmap

# Local module and scripts
from pyccapt.control.core import runtime
from pyccapt.control.gui import tooltips


class Ui_Gates(object):
    """
    Load the GUI based on the configuration file.

    Args:
        None

    Returns:
        None
    """

    def __init__(self, variables, conf, parent=None, override_changed=None):
        """
        Load the GUI based on the configuration file.

        Args:
            variables (object): Global variables
            conf (dict): Configuration file
            parent (object): Parent object
            override_changed (callable): Optional callback receiving the
                shared gate/pump override state.

        Returns:
            None
        """
        self.flag_super_user = False
        self.variables = variables
        self.conf = conf
        self.parent = parent
        self.override_changed = override_changed

    def setupUi(self, Gates):
        """
        Load the GUI based on the configuration file.

        Args:
            Gates (object): Parent object

        Returns:
            None
        """
        Gates.setObjectName("Gates")
        self.Gates = Gates
        Gates.resize(434, 426)
        self.gridLayout_3 = QtWidgets.QGridLayout(Gates)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.diagram = QtWidgets.QLabel(parent=Gates)
        self.diagram.setMinimumSize(QtCore.QSize(378, 246))
        self.diagram.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.diagram.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.diagram.setStyleSheet(
            "QWidget{\n"
            "                                            border: 2px solid gray;\n"
            "                                            background: rgb(255, 255, 255)\n"
            "                                            }\n"
            "                                        "
        )
        self.diagram.setText("")
        self.diagram.setObjectName("diagram")
        self.verticalLayout_4.addWidget(self.diagram)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.cryo_switch = QtWidgets.QPushButton(parent=Gates)
        self.cryo_switch.setMinimumSize(QtCore.QSize(0, 25))
        self.cryo_switch.setStyleSheet(
            "QPushButton{\n"
            "                                                            background: rgb(193, 193, 193)\n"
            "                                                            }\n"
            "                                                        "
        )
        self.cryo_switch.setObjectName("cryo_switch")
        self.verticalLayout_3.addWidget(self.cryo_switch)
        # Gate controls follow the physical chamber order from left to right:
        # Cryo, Main Chamber, Load Lock. The diagram is the state indicator.
        self.gridLayout.addLayout(self.verticalLayout_3, 0, 0, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.main_chamber_switch = QtWidgets.QPushButton(parent=Gates)
        self.main_chamber_switch.setMinimumSize(QtCore.QSize(0, 25))
        self.main_chamber_switch.setStyleSheet(
            "QPushButton{\n"
            "                                                            background: rgb(193, 193, 193)\n"
            "                                                            }\n"
            "                                                        "
        )
        self.main_chamber_switch.setObjectName("main_chamber_switch")
        self.verticalLayout.addWidget(self.main_chamber_switch)
        self.gridLayout.addLayout(self.verticalLayout, 0, 1, 1, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.load_lock_switch = QtWidgets.QPushButton(parent=Gates)
        self.load_lock_switch.setMinimumSize(QtCore.QSize(0, 25))
        self.load_lock_switch.setStyleSheet(
            "QPushButton{\n"
            "                                                            background: rgb(193, 193, 193)\n"
            "                                                            }\n"
            "                                                        "
        )
        self.load_lock_switch.setObjectName("load_lock_switch")
        self.verticalLayout_2.addWidget(self.load_lock_switch)
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 2, 1, 1)
        self.verticalLayout_4.addLayout(self.gridLayout)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.superuser = QtWidgets.QPushButton(parent=Gates)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.superuser.sizePolicy().hasHeightForWidth())
        self.superuser.setSizePolicy(sizePolicy)
        self.superuser.setMinimumSize(QtCore.QSize(400, 25))
        self.superuser.setStyleSheet(
            "QPushButton{\n"
            "                                                    background: rgb(193, 193, 193)\n"
            "                                                    }\n"
            "                                                "
        )
        self.superuser.setObjectName("superuser")
        self.gridLayout_2.addWidget(self.superuser, 0, 1, 1, 1)
        self.Error = QtWidgets.QLabel(parent=Gates)
        self.Error.setMinimumSize(QtCore.QSize(400, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setStrikeOut(False)
        self.Error.setFont(font)
        self.Error.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.Error.setWordWrap(True)
        self.Error.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.LinksAccessibleByMouse)
        self.Error.setObjectName("Error")
        self.gridLayout_2.addWidget(self.Error, 1, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum
        )
        self.gridLayout_2.addItem(spacerItem, 0, 2, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum
        )
        self.gridLayout_2.addItem(spacerItem1, 0, 0, 1, 1)
        self.verticalLayout_4.addLayout(self.gridLayout_2)
        self.gridLayout_3.addLayout(self.verticalLayout_4, 0, 0, 1, 1)

        self.retranslateUi(Gates)
        QtCore.QMetaObject.connectSlotsByName(Gates)
        tooltips.apply_tooltips(self, tooltips.GATES_TOOLTIPS)

        # Four pressure backgrounds x eight gate combinations cover all 32
        # possible states without maintaining 32 near-identical bitmaps.
        self._diagram_backgrounds = {
            (False, False): self._load_gate_background('gates-v4-vacuum.png'),
            (True, False): self._load_gate_background('gates-v4-cryo-vented.png'),
            (False, True): self._load_gate_background('gates-v4-load-vented.png'),
            (True, True): self._load_gate_background('gates-v4-both-vented.png'),
        }
        self._diagram_state = None

        self._refresh_gate_diagram(force=True)

        ###
        self.main_chamber_switch.clicked.connect(lambda: self.gates(1))
        self.load_lock_switch.clicked.connect(lambda: self.gates(2))
        self.cryo_switch.clicked.connect(lambda: self.gates(3))
        self.superuser.clicked.connect(self.super_user_access)

        # Create a QTimer to hide the warning message after 8 seconds
        self.timer = QTimer(self.parent)
        self.timer.timeout.connect(self.hideMessage)
        self.diagram_timer = QTimer(self.parent)
        self.diagram_timer.timeout.connect(self._refresh_gate_diagram)
        self.diagram_timer.start(250)

        self.original_button_style = self.superuser.styleSheet()

    def attach_load_lock_temperature_controls(self, pumps_ui):
        """Place the Pumps UI's LL temperature controls below Override Access.

        The widgets are moved, not duplicated, so their existing signal
        connections and baking logic continue to be handled by ``pumps_ui``.
        """
        if hasattr(self, "load_lock_temperature_group"):
            return

        group = QtWidgets.QGroupBox("Load Lock temperature", parent=self.Gates)
        group.setObjectName("load_lock_temperature_group")
        group.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum
        )
        group.setMaximumHeight(155)
        group.setStyleSheet(
            "QGroupBox{font-weight: bold; border: 1px solid rgb(145,145,145); "
            "border-radius: 5px; margin-top: 8px; padding-top: 5px;}"
            "QGroupBox::title{subcontrol-origin: margin; left: 8px; padding: 0 4px;}"
        )
        layout = QtWidgets.QGridLayout(group)
        layout.setContentsMargins(8, 10, 8, 7)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(5)

        pumps_ui.temp_ll.setFixedSize(QtCore.QSize(150, 45))
        pumps_ui.target_tempreature_ll.setFixedSize(QtCore.QSize(150, 25))
        pumps_ui.set_temperature_ll.setFixedSize(QtCore.QSize(190, 25))
        pumps_ui.ll_baking_time.setFixedSize(QtCore.QSize(150, 25))

        layout.addWidget(pumps_ui.label_219, 0, 0)
        layout.addWidget(pumps_ui.temp_ll, 0, 1)
        layout.addWidget(pumps_ui.label_220, 1, 0)
        layout.addWidget(pumps_ui.ll_baking_time, 1, 1)
        layout.addWidget(pumps_ui.set_temperature_ll, 2, 0)
        layout.addWidget(pumps_ui.target_tempreature_ll, 2, 1)
        layout.setColumnStretch(0, 1)

        self.load_lock_temperature_group = group
        # Put the group immediately under Override Access. Move the status
        # label below it so an empty status row cannot create a large gap.
        self.gridLayout_2.addWidget(group, 1, 0, 1, 3)
        self.gridLayout_2.addWidget(self.Error, 2, 0, 1, 3)

    @staticmethod
    def _load_gate_background(filename):
        """Load one high-resolution pressure-state background."""
        return QPixmap(str(runtime.project_path('files', filename)))

    @staticmethod
    def _draw_gate_status(painter, center, is_open, vertical_pipe=False):
        """Draw an accessible valve symbol aligned with its connecting pipe."""
        fill = QtGui.QColor('#159A38' if is_open else '#E31B23')
        outline = QtGui.QColor('#0B6725' if is_open else '#991018')
        painter.setBrush(fill)
        painter.setPen(QtGui.QPen(outline, 6))
        painter.drawEllipse(QtCore.QPointF(*center), 42, 42)

        # The white stroke follows flow when open and blocks it when closed.
        line_is_vertical = is_open if vertical_pipe else not is_open
        painter.setPen(
            QtGui.QPen(
                QtGui.QColor('white'),
                14,
                QtCore.Qt.PenStyle.SolidLine,
                QtCore.Qt.PenCapStyle.RoundCap,
            )
        )
        x, y = center
        if line_is_vertical:
            painter.drawLine(QtCore.QPointF(x, y - 27), QtCore.QPointF(x, y + 27))
        else:
            painter.drawLine(QtCore.QPointF(x - 27, y), QtCore.QPointF(x + 27, y))

    def _refresh_gate_diagram(self, force=False):
        """Render the current three-gate/two-vent state (32 combinations)."""
        cryo_vented = (
            not bool(getattr(self.variables, 'flag_pump_cryo_load_lock', True))
            or bool(getattr(self.variables, 'flag_vent_cryo_load_lock_partial', False))
        )
        load_vented = not bool(getattr(self.variables, 'flag_pump_load_lock', True))
        state = (
            bool(self.variables.flag_cryo_gate),
            bool(self.variables.flag_main_gate),
            bool(self.variables.flag_load_gate),
            cryo_vented,
            load_vented,
        )
        if not force and state == self._diagram_state:
            return
        self._diagram_state = state

        pixmap = self._diagram_backgrounds[(cryo_vented, load_vented)].copy()
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self._draw_gate_status(painter, (462, 735), state[0])
        self._draw_gate_status(painter, (760, 439), state[1], vertical_pipe=True)
        self._draw_gate_status(painter, (1056, 735), state[2])
        painter.end()
        self.diagram.setPixmap(pixmap.scaled(
            378,
            246,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        ))

    def retranslateUi(self, Gates):
        """

        Args:
            Gates: The main window

        Returns:
            None
        """
        _translate = QtCore.QCoreApplication.translate
        ###
        Gates.setWindowTitle(_translate("Ui_Gates", "PyCCAPT Gates Control"))
        Gates.setWindowIcon(QtGui.QIcon('./files/logo.png'))
        ###
        self.cryo_switch.setText(_translate("Gates", "Cryo"))
        self.main_chamber_switch.setText(_translate("Gates", "Main Chamber"))
        self.load_lock_switch.setText(_translate("Gates", "Load Lock"))
        self.superuser.setText(_translate("Gates", "Override Access"))
        self.Error.setText(_translate("Gates", "<html><head/><body><p><br/></p></body></html>"))

    def super_user_access(self):
        """
        The function for super user access

        Args:
            None

        Returns:
            None
        """
        if not self.flag_super_user:
            warning = QtWidgets.QMessageBox(parent=self.superuser)
            warning.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            warning.setWindowTitle("Confirm Access Override")
            warning.setText("Gate and vacuum override can bypass safety interlocks.")
            warning.setInformativeText("Only continue if you really want to override gate and vacuum access.")
            warning.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
            warning.setDefaultButton(QtWidgets.QMessageBox.StandardButton.No)
            if warning.exec() != QtWidgets.QMessageBox.StandardButton.Yes:
                self.error_message("Override Access canceled.")
                self.timer.start(8000)
                return
            self.flag_super_user = True
            self.superuser.setStyleSheet("QPushButton{\nbackground: rgb(0, 255, 26)\n}")
            self.error_message("!!! Override Access Granted !!!")
        elif self.flag_super_user:
            self.flag_super_user = False
            self.superuser.setStyleSheet(self.original_button_style)
            self.error_message("!!! Override Access deactivated !!!")
            self.timer.start(8000)
        if self.override_changed is not None:
            self.override_changed(self.flag_super_user)

    def _vacuum_ok_to_open(self, gate_label, sides):
        """Confirm the vacuum is safe before opening a gate.

        A gate connects two chambers; opening it equalises their pressure.
        If either chamber is not pumped down to its safe threshold, opening
        can spoil a good vacuum (or worse). This pops a confirmation dialog
        naming the offending chamber(s) and lets the operator decide.

        Args:
            gate_label: Human-readable gate name for the dialog.
            sides: list of ``(chamber_label, pressure, threshold)`` for the
                two chambers the gate connects. ``pressure`` and
                ``threshold`` are in mBar (lower = better vacuum).

        Returns:
            True if it is safe to proceed (every side is within its
            threshold) or the operator confirmed the override; False if the
            operator cancelled.
        """
        # No gauges -> no pressure data to validate against; don't block.
        if self.conf.get('gauges', 'off') == 'off':
            return True

        unsafe = []
        for label, pressure, threshold in sides:
            try:
                pressure = float(pressure)
            except (TypeError, ValueError):
                pressure = -1.0
            if pressure <= 0:
                # -1 = gauge read error, 0 = no reading yet -> cannot
                # confirm the chamber is safe, so warn to be cautious.
                unsafe.append(
                    f"- {label}: pressure unknown (no valid gauge reading); "
                    f"should be ≤ {threshold:.1e} mBar"
                )
            elif pressure > threshold:
                unsafe.append(
                    f"- {label}: {pressure:.2e} mBar "
                    f"(should be ≤ {threshold:.1e} mBar)"
                )

        if not unsafe:
            return True

        parent = self.parent if isinstance(self.parent, QtWidgets.QWidget) else self.superuser
        warning = QtWidgets.QMessageBox(parent=parent)
        warning.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        warning.setWindowTitle("Unsafe vacuum - confirm gate opening")
        warning.setText(
            f"The vacuum is NOT at a safe level to open the {gate_label}.\n"
            "Opening it now may spoil the vacuum in the adjoining chamber."
        )
        warning.setInformativeText(
            "The following chamber(s) are outside the safe range:\n"
            + "\n".join(unsafe)
            + "\n\nDo you still want to open the gate?"
        )
        warning.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )
        warning.setDefaultButton(QtWidgets.QMessageBox.StandardButton.No)
        return warning.exec() == QtWidgets.QMessageBox.StandardButton.Yes

    def gates(self, gate_num):
        """
        The function for applying the command of closing or opening gate

        Args:
            gate_num: The number of the gate to be opened or closed

        Returns:
            None
        """

        def switch_gate(num):
            """
            The function for opening or closing the gate

            Args:
                num: The number of the gate to be opened or closed

            Returns:
                None
            """
            if self.conf["gates"] == "off":
                print('The gates control is off')
                return

            # Hardware sequence: drive the gate line high, wait 500 ms,
            # then drive it low. The previous synchronous ``time.sleep``
            # froze the entire GUI for the half-second; operators
            # toggling multiple gates in quick succession could stack
            # several seconds of UI lockup. Schedule the low write via
            # QTimer.singleShot so the Qt event loop stays responsive.
            task = nidaqmx.Task()
            try:
                task.do_channels.add_do_chan(self.conf["COM_PORT_gates"] + 'line%s' % num)
                task.start()
                task.write([True])
            except Exception:
                # Ensure the task is closed if the high-write blew up.
                try:
                    task.close()
                except Exception:
                    pass
                raise

            def _finish_gate_pulse():
                try:
                    task.write([False])
                finally:
                    try:
                        task.close()
                    except Exception:
                        pass

            QTimer.singleShot(500, _finish_gate_pulse)

        def error_gate():
            """
            The function for showing the error message in the GUI

            Args:
                None

            Returns:
                None
            """
            if self.variables.start_flag:
                self.error_message("!!! An experiment is running !!!")
            else:
                self.error_message("!!! Close the previous opened gate first !!!")
            self.timer.start(8000)

        # Main gate
        if gate_num == 1:
            if (
                not self.variables.start_flag
                and (
                    not self.variables.flag_load_gate
                    and not self.variables.flag_cryo_gate
                    and self.variables.flag_pump_load_lock
                )
            ) or self.flag_super_user:
                if not self.variables.flag_main_gate:  # Open the main gate
                    # Override Access explicitly bypasses the vacuum interlock.
                    if not self.flag_super_user and not self._vacuum_ok_to_open(
                        "main-chamber gate",
                        [
                            ("Main chamber", self.variables.vacuum_main, float(self.conf['vacuum_threshold_main'])),
                            ("Buffer chamber", self.variables.vacuum_buffer, float(self.conf['vacuum_threshold_buffer'])),
                        ],
                    ):
                        return  # operator cancelled - leave the gate closed
                    if self.conf["gates"] == "on":
                        switch_gate(0)
                    self.variables.flag_main_gate = True
                elif self.variables.flag_main_gate:  # Close the main gate
                    if self.conf["gates"] == "on":
                        switch_gate(1)
                    self.variables.flag_main_gate = False
            else:
                error_gate()
        # Buffer gate
        elif gate_num == 2:
            if (
                not self.variables.start_flag
                and (
                    not self.variables.flag_main_gate
                    and not self.variables.flag_cryo_gate
                    and self.variables.flag_pump_load_lock
                )
            ) or self.flag_super_user:
                if not self.variables.flag_load_gate:  # Open the load-lock gate
                    if not self.flag_super_user and not self._vacuum_ok_to_open(
                        "load-lock gate",
                        [
                            ("Load lock", self.variables.vacuum_load_lock, float(self.conf['vacuum_threshold_load_lock'])),
                            ("Buffer chamber", self.variables.vacuum_buffer, float(self.conf['vacuum_threshold_buffer'])),
                        ],
                    ):
                        return  # operator cancelled - leave the gate closed
                    if self.conf["gates"] == "on":
                        switch_gate(2)
                    self.variables.flag_load_gate = True
                elif self.variables.flag_load_gate:  # Close the main gate
                    if self.conf["gates"] == "on":
                        switch_gate(3)
                    self.variables.flag_load_gate = False
            else:
                error_gate()
        # Cryo gate
        elif gate_num == 3:
            if (
                not self.variables.start_flag
                and (
                    not self.variables.flag_main_gate
                    and not self.variables.flag_load_gate
                    and self.variables.flag_pump_load_lock
                )
            ) or self.flag_super_user:
                if not self.variables.flag_cryo_gate:  # Open the cryo gate
                    if not self.flag_super_user and not self._vacuum_ok_to_open(
                        "cryo gate",
                        [
                            (
                                "Cryo load lock",
                                self.variables.vacuum_cryo_load_lock,
                                float(self.conf['vacuum_threshold_cryo_load_lock']),
                            ),
                            ("Buffer chamber", self.variables.vacuum_buffer, float(self.conf['vacuum_threshold_buffer'])),
                        ],
                    ):
                        return  # operator cancelled - leave the gate closed
                    if self.conf["gates"] == "on":
                        switch_gate(4)
                    self.variables.flag_cryo_gate = True
                elif self.variables.flag_cryo_gate:  # Close the main gate
                    if self.conf["gates"] == "on":
                        switch_gate(5)
                    self.variables.flag_cryo_gate = False
            else:
                error_gate()

        else:
            print('The gate number is not correct')

        self._refresh_gate_diagram()

    def error_message(self, message):
        """
        The function for showing the error message in the GUI

        Args:
            message: The message to be shown in the GUI

        Returns:
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

    def hideMessage(self):
        """
        The function for hiding the error message in the GUI

        Args:
            None

        Returns:
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
        The function for stopping the background activity

        Args:
            None

        Returns:
            None
        """
        self.timer.stop()
        self.diagram_timer.stop()


class GatesWindow(QtWidgets.QWidget):
    closed = QtCore.pyqtSignal()  # Define a custom closed signal

    def __init__(self, gui_gates, *args, **kwargs):
        """
        Initialize the GatesWindow class.

        Args:
            gui_gates: GUI for gates.
            *args, **kwargs: Additional arguments for QWidget initialization.
        """
        super().__init__(*args, **kwargs)
        self.gui_gates = gui_gates

    def closeEvent(self, event):
        """
        Handle the close event of the GatesWindow.

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
    Gates = QtWidgets.QWidget()
    ui = Ui_Gates(shared.variables, conf)
    ui.setupUi(Gates)
    Gates.show()
    sys.exit(app.exec())
