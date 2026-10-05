"""Compact visual grouping for pump controls, preserving the live widgets."""
from PyQt6 import QtCore, QtWidgets


class PumpLayoutMixin:
    def _setup_compact_pump_layout(self, window):
        def group(title, name):
            box = QtWidgets.QGroupBox(title, window)
            box.setObjectName(name)
            box.setStyleSheet(
                'QGroupBox { font-weight: bold; border: 1px solid gray; '
                'border-radius: 5px; margin-top: 8px; } '
                'QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }')
            layout = QtWidgets.QGridLayout(box)
            layout.setContentsMargins(8, 10, 8, 6)
            layout.setHorizontalSpacing(8)
            layout.setVerticalSpacing(3)
            return box, layout

        # Reparenting the original widgets keeps all signal connections, gauge
        # warning colours, overrides and temperature/vent interlocks intact.
        self.verticalLayout.removeItem(self.gridLayout_4)
        self.pump_readings_panel = QtWidgets.QWidget(window)
        readings = QtWidgets.QGridLayout(self.pump_readings_panel)
        readings.setContentsMargins(0, 0, 0, 0)
        readings.setSpacing(4)
        readings.setColumnStretch(1, 1)

        main = QtWidgets.QVBoxLayout()
        main.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        main.addWidget(self.label_212, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        main.addWidget(self.vacuum_main, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        readings.addLayout(main, 0, 0)

        self.cryo_temperature_group, cryo = group('Cryo temperature', 'cryo_temperature_group')
        cryo.addWidget(self.label_215, 0, 0)
        cryo.addWidget(self.temp_stage, 0, 1)
        cryo.addWidget(self.set_temperature_cryo, 0, 2)
        cryo.addWidget(self.target_tempreature_cryo, 0, 3)
        cryo.addWidget(self.label_218, 1, 0)
        cryo.addWidget(self.temp_cryo_head, 1, 1)
        cryo.addWidget(self.label_221, 1, 2)
        cryo.addWidget(self.temp_cryo_head_inside, 1, 3)
        for label in (self.label_215, self.label_218, self.label_221):
            label.setWordWrap(True)
            label.setMaximumWidth(150)
        readings.addWidget(self.cryo_temperature_group, 0, 1)

        self.vacuum_gauges_group, gauges = group('Buffer / LL / CLL vacuum (mBar)', 'vacuum_gauges_group')
        for row, pairs in enumerate((
            ((self.label_211, self.vacuum_buffer), (self.label_210, self.vacuum_load_lock),
             (self.label_216, self.vacuum_cryo_load_lock)),
            ((self.label_214, self.vacuum_buffer_back), (self.label_213, self.vacuum_load_lock_back),
             (self.label_217, self.vacuum_cryo_load_lock_back)),
        )):
            for column, (label, lcd) in enumerate(pairs):
                label.setWordWrap(True)
                label.setMaximumWidth(170)
                gauges.addWidget(label, row * 2, column)
                gauges.addWidget(lcd, row * 2 + 1, column)
        # Let this column grow to the full pre-vacuum label width rather
        # than wrapping its units onto a second line above the LCD.
        self.label_214.setMaximumWidth(16777215)
        self.label_214.setWordWrap(False)

        self.venting_group, venting = group('Venting', 'venting_group')
        for row, button in enumerate((self.pump_cryo_load_lock_switch,
                                      self.vent_cryo_load_lock_partial_switch,
                                      self.pump_load_lock_switch)):
            button.setFixedWidth(125)
            venting.addWidget(button, row, 0)
        venting.setRowStretch(3, 1)
        lower = QtWidgets.QHBoxLayout()
        lower.addWidget(self.vacuum_gauges_group)
        # Spare width separates the gauges from venting and keeps venting
        # beside the Gates diagram at the right edge of the pump panel.
        lower.addStretch(1)
        lower.addWidget(self.venting_group)
        readings.addLayout(lower, 1, 0, 1, 2)
        self.verticalLayout.insertWidget(0, self.pump_readings_panel)
        self.verticalLayout.setSpacing(4)
        self.ll_baking_time.setFixedHeight(25)
        self.Error.setMinimumHeight(20)
