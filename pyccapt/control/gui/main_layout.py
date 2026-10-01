"""Compact main-window layout and the infrequently edited experiment settings."""
from PyQt6 import QtCore, QtWidgets

from pyccapt.control.gui.responsive import make_window_responsive


class MainLayoutMixin:
    def _setup_compact_main_layout(self, window):
        # Keep the same widgets and signals in the dialog. This preserves config
        # defaults, live control-algorithm switching and experiment lock states.
        self.advanced_settings_dialog = QtWidgets.QDialog(window)
        self.advanced_settings_dialog.setWindowTitle('Advanced experiment settings')
        self.advanced_settings_dialog.setModal(False)
        self.advanced_settings_dialog.resize(460, 340)
        dialog_layout = QtWidgets.QVBoxLayout(self.advanced_settings_dialog)
        advanced_widgets = {self.gridLayout.itemAt(i).widget() for i in range(self.gridLayout.count())}
        self._plan_form_widgets = [widget for widget in self._plan_form_widgets if widget not in advanced_widgets]
        self.verticalLayout_2.removeItem(self.gridLayout)
        dialog_layout.addLayout(self.gridLayout)
        note = QtWidgets.QLabel('Changes take effect as you edit each field. Run controls retain their normal locks.')
        note.setWordWrap(True)
        dialog_layout.addWidget(note)
        close = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.advanced_settings_dialog.close)
        dialog_layout.addWidget(close)
        make_window_responsive(self.advanced_settings_dialog)
        self.advanced_settings_button = QtWidgets.QPushButton('Advanced settings…', self.centralwidget)
        self.advanced_settings_button.setObjectName('advanced_settings_button')
        self.advanced_settings_button.clicked.connect(self._show_advanced_settings)
        # Replace the two separators below the target rate with one full-width
        # separator below the parameters and statistics.
        for line in (self.line_3, self.line_4):
            self.verticalLayout.removeWidget(line)
            line.hide()
            self._plan_form_widgets.remove(line)
        self.run_controls_separator = QtWidgets.QFrame(self.centralwidget)
        self.run_controls_separator.setObjectName('run_controls_separator')
        self.run_controls_separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.run_controls_separator.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.gridLayout_6.addWidget(self.run_controls_separator, 2, 0, 1, 2)

        # Remove the third column and group operating controls below statistics.
        self.gridLayout_6.removeItem(self.electrode_controls)
        while self.electrode_controls.count():
            self.electrode_controls.takeAt(0)
        for widget in (self.start_button, self.stop_button, self.Error, self.plan_panel, self.text_line):
            self.gridLayout_6.removeWidget(widget)
        self.gridLayout_6.addWidget(self.plan_panel, 3, 0, 1, 2)
        self.gridLayout_6.addWidget(self.text_line, 3, 0, 1, 2)
        self.run_controls_panel = QtWidgets.QWidget(self.centralwidget)
        controls = QtWidgets.QGridLayout(self.run_controls_panel)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setVerticalSpacing(4)
        controls.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        for row, (label, widget) in enumerate(
            ((self.alignment_start_voltage_label, self.alignment_start_voltage),
             (self.alignment_voltage_increment_label, self.alignment_voltage_increment))
        ):
            label.setMinimumWidth(0)
            label.setMaximumWidth(16777215)
            label.setWordWrap(False)
            widget.setFixedWidth(100)
            controls.addWidget(label, row, 0)
            controls.addWidget(widget, row, 1, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        self.advanced_settings_button.setFixedWidth(self.flat_test_button.width())
        for widget, row, column in (
            (self.electrode_button, 2, 0), (self.advanced_settings_button, 2, 1),
            (self.automatic_alignment_button, 3, 0), (self.flat_test_button, 3, 1),
            (self.start_button, 4, 1), (self.stop_button, 5, 1),
        ):
            controls.addWidget(widget, row, column, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        self.verticalLayout_2.addWidget(self.run_controls_panel, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        self.gridLayout_5.setVerticalSpacing(4)
        self.Error.setMinimumWidth(0)
        self.gridLayout_6.addWidget(self.Error, 5, 0, 1, 2)

    def _show_advanced_settings(self):
        self.advanced_settings_dialog.show()
        self.advanced_settings_dialog.raise_()
        self.advanced_settings_dialog.activateWindow()
