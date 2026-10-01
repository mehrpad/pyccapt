"""Compact laser controls, retaining the original widgets and connections."""
from PyQt6 import QtCore, QtWidgets


class LaserLayoutMixin:
    def _setup_compact_laser_layout(self, parent):
        main = self.gridLayout_5
        while main.count():
            main.takeAt(0)
        # The former seven-column layout reserved space for obsolete scan
        # controls. Use two columns above one shared stage-control row.
        for column in range(7):
            main.setColumnStretch(column, 0)
            main.setColumnMinimumWidth(column, 0)
        self.gridLayout_6.setContentsMargins(6, 6, 6, 6)
        main.setSpacing(6)
        settings = QtWidgets.QWidget(parent)
        settings_layout = QtWidgets.QVBoxLayout(settings)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(4)
        self.gridLayout_3.setSpacing(4)
        self.gridLayout_3.setParent(None)
        settings_layout.addLayout(self.gridLayout_3)
        for field in (self.laser_power, self.laser_rate, self.laser_divition_factor):
            field.setFixedSize(159, 25)
        self.laser_wavelegnth.setFixedSize(85, 25)
        for button in (self.laser_enable, self.laser_on, self.laser_standby, self.laser_listen):
            button.setFixedHeight(25)
        while self.horizontalLayout.count():
            self.horizontalLayout.takeAt(0)
        readouts = QtWidgets.QGridLayout()
        readouts.setSpacing(3)
        for column, (label, lcd) in enumerate((
            (self.label_9, self.laser_power_disp),
            (self.label_10, self.laser_pulse_energy_disp),
            (self.label_11, self.laser_repetion_rate_disp),
        )):
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            lcd.setFixedSize(100, 40)
            readouts.addWidget(label, 0, column)
            readouts.addWidget(lcd, 1, column, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        settings_layout.addLayout(readouts)
        self.laser_alignment_panel.layout().setContentsMargins(6, 10, 6, 6)
        self.laser_alignment_panel.layout().setSpacing(4)
        for field in self.laser_alignment_fields.values():
            widest_value = f'{field.maximum():.{field.decimals()}f}{field.suffix()}'
            field.setFixedSize(max(130, field.fontMetrics().horizontalAdvance(widest_value) + 30), 25)
        settings_layout.addWidget(self.laser_alignment_panel)
        settings_layout.addStretch(1)
        main.addWidget(settings, 0, 0)
        main.addWidget(self.alignment_plot, 0, 1)
        self.alignment_plot.layout().setContentsMargins(0, 0, 0, 0)
        self.alignment_plot.layout().setSpacing(4)
        stage = QtWidgets.QWidget(parent)
        stage_layout = QtWidgets.QHBoxLayout(stage)
        stage_layout.setContentsMargins(0, 0, 0, 0)
        stage_layout.setSpacing(6)
        for layout in (self.gridLayout_4, self.gridLayout_2):
            layout.setParent(None)
            layout.setSpacing(3)
            stage_layout.addLayout(layout)
        for axis in ('x', 'y', 'z'):
            for unit in ('mm', 'um', 'nm'):
                getattr(self, f'laser_{axis}_{unit}').setFixedSize(64, 28)
            getattr(self, f'laser_speed_{axis}_label').setMinimumWidth(67)
            selector = getattr(self, f'laser_speed_{axis}')
            widest_text = max(selector.fontMetrics().horizontalAdvance(selector.itemText(index))
                              for index in range(selector.count()))
            selector.setFixedWidth(max(132, widest_text + 42))
        # Keep the jog boxes at their readable minimum; spare window width
        # belongs between the groups rather than inside the Z buttons.
        self.laser_xy_jog_group.setFixedWidth(self.laser_xy_jog_group.minimumSizeHint().width())
        self.laser_z_jog_group.setFixedWidth(self.laser_z_jog_group.minimumSizeHint().width())
        stage_layout.addWidget(self.laser_xy_jog_group)
        stage_layout.addWidget(self.laser_z_jog_group)
        self._stage_button_layout.setSpacing(4)
        self._stage_button_layout.setParent(None)
        for button in (self.laser_home, self.laser_stage_reference,
                       self.laser_stage_stop, self.laser_stage_superuser):
            button.setFixedSize(105, 28)
        stage_layout.addLayout(self._stage_button_layout)
        main.addWidget(stage, 1, 0, 1, 2)
        connection = QtWidgets.QHBoxLayout()
        connection.setSpacing(4)
        for button in (self.switch_to_cli_button, self.nktpbus_mode_switch):
            button.setFixedSize(115, 28)
            connection.addWidget(button)
        self.laser_connection_banner.setMinimumWidth(0)
        connection.addWidget(self.laser_connection_banner, 1)
        connection.addStretch(1)
        main.addLayout(connection, 2, 0, 1, 2)
        self.Error.setMinimumWidth(0)
        main.addWidget(self.Error, 3, 0, 1, 2)
        main.setColumnStretch(0, 3)
        main.setColumnStretch(1, 2)
        main.setRowStretch(0, 1)
