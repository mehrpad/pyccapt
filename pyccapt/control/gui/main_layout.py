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
        self.advanced_settings_label = QtWidgets.QLabel('Advanced settings', self.centralwidget)
        self.advanced_settings_label.setObjectName('advanced_settings_label')
        self.gridLayout_2.addWidget(self.advanced_settings_label, 4, 0)
        self.gridLayout_2.addWidget(self.advanced_settings_button, 4, 1)
        self._plan_form_widgets.extend((self.advanced_settings_label, self.advanced_settings_button))
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
        self.gridLayout_6.addWidget(self.run_controls_separator, 3, 0, 1, 2)

        # Remove the third column and group operating controls below statistics.
        self.gridLayout_6.removeItem(self.electrode_controls)
        while self.electrode_controls.count():
            self.electrode_controls.takeAt(0)
        for widget in (self.start_button, self.stop_button, self.Error, self.plan_panel):
            self.gridLayout_6.removeWidget(widget)
        self.gridLayout_6.addWidget(self.plan_panel, 4, 0, 1, 2)
        self.run_controls_panel = QtWidgets.QWidget(self.centralwidget)
        controls = QtWidgets.QGridLayout(self.run_controls_panel)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setVerticalSpacing(2)
        controls.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        controls.addWidget(self.electrode_button, 0, 0, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        controls.addWidget(self.flat_test_button, 0, 1, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        self.electrode_separator = QtWidgets.QFrame(self.run_controls_panel)
        self.electrode_separator.setObjectName('electrode_separator')
        self.electrode_separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.electrode_separator.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        controls.addWidget(self.electrode_separator, 1, 0, 1, 2)
        self.alignment_group = QtWidgets.QGroupBox('Auto Alignment', self.run_controls_panel)
        self.alignment_group.setObjectName('alignment_group')
        self.alignment_group.setStyleSheet(
            'QGroupBox { font-weight: bold; border: 1px solid gray; border-radius: 5px; margin-top: 8px; } '
            'QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }')
        alignment_controls = QtWidgets.QGridLayout(self.alignment_group)
        alignment_controls.setContentsMargins(10, 12, 10, 8)
        alignment_controls.setVerticalSpacing(4)
        for row, (label, widget) in enumerate(
            ((self.alignment_start_voltage_label, self.alignment_start_voltage),
             (self.alignment_voltage_increment_label, self.alignment_voltage_increment))
        ):
            label.setMinimumWidth(0)
            label.setMaximumWidth(16777215)
            label.setWordWrap(False)
            widget.setFixedWidth(100)
            alignment_controls.addWidget(label, row, 0)
            alignment_controls.addWidget(widget, row, 1, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        alignment_controls.addWidget(self.automatic_alignment_button, 2, 0, 1, 2,
                                     alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        controls.addWidget(self.alignment_group, 2, 0, 1, 2)
        self.statistics_separator = QtWidgets.QFrame(self.centralwidget)
        self.statistics_separator.setObjectName('statistics_separator')
        self.statistics_separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.statistics_separator.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.verticalLayout_2.addWidget(self.statistics_separator)
        self.verticalLayout_2.addWidget(self.run_controls_panel, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        self.experiment_actions_panel = QtWidgets.QWidget(self.centralwidget)
        self.experiment_actions_panel.setObjectName('experiment_actions_panel')
        actions = QtWidgets.QVBoxLayout(self.experiment_actions_panel)
        actions.setContentsMargins(0, 8, 0, 0)
        actions.addWidget(self.start_button, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        actions.addWidget(self.stop_button, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.gridLayout_6.addWidget(self.experiment_actions_panel, 2, 0, 1, 2)
        self.gridLayout_5.setVerticalSpacing(3)
        self.Error.setMinimumWidth(0)
        self.gridLayout_6.addWidget(self.Error, 6, 0, 1, 2)

    def _show_advanced_settings(self):
        self.advanced_settings_dialog.show()
        self.advanced_settings_dialog.raise_()
        self.advanced_settings_dialog.activateWindow()
