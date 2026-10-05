"""Compact camera controls with the existing views and signal connections."""
from PyQt6 import QtCore


class CameraLayoutMixin:
    def _setup_compact_camera_layout(self):
        self.gridLayout_5.setContentsMargins(6, 6, 6, 6)
        self.gridLayout_5.setSpacing(4)
        self.gridLayout_4.setSpacing(6)
        self.gridLayout.setSpacing(3)
        self.gridLayout_3.setSpacing(3)
        self.verticalLayout.setSpacing(3)
        self.verticalLayout_2.setSpacing(4)
        self.horizontalLayout.setSpacing(4)
        for index in reversed(range(self.horizontalLayout.count())):
            if self.horizontalLayout.itemAt(index).spacerItem() is not None:
                self.horizontalLayout.takeAt(index)

        # Keep each exposure value next to its slider instead of allocating
        # two rows per camera. The original fields and sliders retain signals.
        while self.gridLayout_2.count():
            self.gridLayout_2.takeAt(0)
        self.gridLayout_2.setSpacing(4)
        self.gridLayout_2.addWidget(self.illumination_percent_label, 0, 0)
        self.illumination_percent.setFixedSize(90, 25)
        self.gridLayout_2.addWidget(self.illumination_percent, 0, 2)
        self.gridLayout_2.addWidget(self.dimming_separator, 1, 0, 1, 3)
        for row, (label, slider, field) in enumerate(zip(
            (self.led_light_2, self.led_light_3, self.led_light_4),
            self.exposure_sliders,
            (self.exposure_time_cam_1, self.exposure_time_cam_2, self.exposure_time_cam_3),
        ), start=2):
            field.setFixedSize(90, 25)
            slider.setMinimumWidth(80)
            self.gridLayout_2.addWidget(label, row, 0)
            self.gridLayout_2.addWidget(slider, row, 1)
            self.gridLayout_2.addWidget(field, row, 2)
        self.gridLayout_2.addLayout(self.exposure_mode_layout, 5, 0, 1, 3)
        self.exposure_mode_layout.setSpacing(4)
        self.gridLayout_2.setColumnStretch(1, 1)
        self.gridLayout_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.camera_list_box.setMaximumWidth(16777215)
        self.camera_list_box.setMaximumHeight(16777215)
        for lcd in self.camera_monitor_lcds.values():
            lcd.setFixedSize(150, 34)
        self.camera_status_label.setMinimumHeight(24)
