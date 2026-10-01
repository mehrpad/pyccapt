"""Laser alignment controls; no voltage supply is opened by the GUI."""
import logging
import time
import uuid
from PyQt6 import QtCore, QtWidgets

from pyccapt.control.apt.laser_alignment_config import LaserAlignmentConfig
from pyccapt.control.apt.laser_alignment_runtime import validate_laser_alignment
from pyccapt.control.devices.laser_alignment_stage import LaserAlignmentStageService
from pyccapt.control.gui.laser_alignment_plot import LaserAlignmentPlot


class LaserAlignmentGuiMixin:
    def _setup_laser_alignment(self, parent):
        for widget in (self.label_12, self.laser_scan_mode5, self.label_13, self.laser_focus_mode):
            self.gridLayout_5.removeWidget(widget)
            widget.hide()
        self.gridLayout_5.removeWidget(self.scanning_disp)
        self.scanning_disp.hide()
        self.alignment_plot = LaserAlignmentPlot(parent)
        self.gridLayout_5.addWidget(self.alignment_plot, 0, 6, 4, 1)
        self.gridLayout_5.removeWidget(self.start_scanning)
        self.start_scanning.hide()
        panel = QtWidgets.QGroupBox('Laser alignment', parent)
        self.laser_alignment_panel = panel
        layout = QtWidgets.QGridLayout(panel)
        self.laser_auto_alignment = QtWidgets.QCheckBox('Align when experiment starts')
        self.laser_tracking = QtWidgets.QCheckBox('Track during experiment')
        layout.addWidget(self.laser_auto_alignment, 0, 0, 1, 2)
        layout.addWidget(self.laser_tracking, 0, 2, 1, 2)
        cfg = LaserAlignmentConfig.from_config(self.conf)
        self.laser_alignment_fields = {}
        for row, prefix, title in ((1, 'coarse', 'Coarse XY'), (2, 'fine', 'Fine XY'), (3, 'focus', 'Focus Z')):
            layout.addWidget(QtWidgets.QLabel(title+' ± range'), row, 0)
            for column, suffix in ((1, '_range_um'), (3, '_step_um')):
                name = prefix+suffix
                field = QtWidgets.QDoubleSpinBox()
                field.setDecimals(3)
                field.setRange(.001, 10000)
                field.setSuffix(' µm')
                field.setValue(getattr(cfg, name))
                self.laser_alignment_fields[name] = field
                layout.addWidget(field, row, column)
                field.valueChanged.connect(self._publish_laser_alignment_settings)
            layout.addWidget(QtWidgets.QLabel('Step'), row, 2)
        increment = QtWidgets.QDoubleSpinBox()
        increment.setRange(1, 1000)
        increment.setDecimals(1)
        increment.setSuffix(' V')
        increment.setValue(cfg.voltage_increment)
        increment.setToolTip('After a complete unsuccessful coarse scan, return to its centre and raise DC by this amount. Voltage ceiling and the 10-second no-event timeout still apply.')
        self.laser_alignment_fields['voltage_increment'] = increment
        increment.valueChanged.connect(self._publish_laser_alignment_settings)
        layout.addWidget(QtWidgets.QLabel('DC increase after no signal'), 4, 0, 1, 2)
        layout.addWidget(increment, 4, 2, 1, 2)
        self.laser_alignment_buttons = []
        for column, (mode, label) in enumerate((('coarse', 'Coarse Alignment'), ('fine', 'Fine Alignment'), ('focus', 'Focus Z'))):
            button = QtWidgets.QPushButton(label)
            button.setMinimumHeight(28)
            button.clicked.connect(lambda _checked=False, m=mode: self._request_laser_alignment(m))
            layout.addWidget(button, 5, column)
            self.laser_alignment_buttons.append(button)
        self.laser_alignment_stop = QtWidgets.QPushButton('Stop Alignment')
        self.laser_alignment_stop.setMinimumHeight(28)
        self.laser_alignment_stop.clicked.connect(self._cancel_laser_alignment)
        layout.addWidget(self.laser_alignment_stop, 5, 3)
        self.laser_alignment_label = QtWidgets.QLabel('Start a Laser-mode experiment to scan. Ranges are laser-stage travel.')
        self.laser_alignment_label.setWordWrap(True)
        layout.addWidget(self.laser_alignment_label, 6, 0, 1, 4)
        self.gridLayout_5.addWidget(panel, 6, 0, 1, 7)
        self.laser_auto_alignment.toggled.connect(self._set_laser_auto_alignment)
        self.laser_tracking.toggled.connect(lambda enabled: setattr(self.variables, 'laser_alignment_tracking', enabled))
        self._publish_laser_alignment_settings()
        self._laser_alignment_service = LaserAlignmentStageService(self.variables, self.conf, lambda: self.stage_device)
        self._laser_alignment_timer = QtCore.QTimer(parent)
        self._laser_alignment_timer.setInterval(200)
        self._laser_alignment_timer.timeout.connect(self._tick_laser_alignment)
        self._laser_alignment_timer.start()
        self._tick_laser_alignment()

    def _publish_laser_alignment_settings(self, *_args):
        self.variables.laser_alignment_settings = {k: field.value() for k, field in self.laser_alignment_fields.items()}

    def _set_laser_auto_alignment(self, enabled):
        if enabled:
            try:
                cfg = LaserAlignmentConfig.from_config(self.conf, self.variables.laser_alignment_settings)
                cfg.validate(self.variables.laser_stage_snapshot[:3])
            except (ValueError, TypeError) as exc:
                self.laser_auto_alignment.setChecked(False)
                self.error_message(str(exc))
                return
        self.variables.laser_alignment_enabled = enabled

    def _laser_alignment_busy(self):
        return bool(getattr(self.variables, 'laser_alignment_status', {}).get('active'))

    def _request_laser_alignment(self, mode):
        try:
            if self.variables.experiment_state != 'running' or self.variables.stop_flag or self.variables.electrode_out:
                raise ValueError('Start a Laser-mode experiment with the electrode in before laser alignment.')
            if self._laser_alignment_busy():
                raise ValueError('Laser alignment is already running.')
            if self._stage_reference_worker is not None and self._stage_reference_worker.isRunning():
                raise ValueError('Wait for laser stage referencing to finish.')
            if self.stage_device is None or self.stage_device.is_moving():
                raise ValueError('Connect and stop the laser stage before scanning.')
            self.stage_device.validate_alignment_state()
            validate_laser_alignment(self.variables, self.conf)
            self.variables.laser_alignment_cancel = False
            self.variables.laser_alignment_status = {'phase': 'pending', 'active': True}
            self.variables.laser_alignment_command = {'id': uuid.uuid4().hex, 'mode': mode, 'issued': time.monotonic()}
            self._tick_laser_alignment()
        except Exception as exc:
            self.error_message(str(exc))

    def _cancel_laser_alignment(self):
        self.variables.laser_alignment_cancel = True
        self.variables.laser_alignment_command = {'id': uuid.uuid4().hex, 'mode': 'stop', 'issued': time.monotonic()}
        if hasattr(self, '_laser_alignment_service'):
            self._laser_alignment_service.cancel()
        self.variables.laser_alignment_status = {'phase': 'cancelled', 'active': False,
                                                  'reason': 'Laser alignment stopped by operator.'}

    def _tick_laser_alignment(self):
        try:
            self._laser_alignment_service.tick()
            status = self.variables.laser_alignment_status
            running = bool(self.variables.start_flag)
            busy = bool(status.get('active')) and running
            can_start = (self.variables.experiment_state == 'running' and not busy and
                         self.variables.pulse_mode == 'Laser' and not self.variables.stop_flag)
            for button in self.laser_alignment_buttons: button.setEnabled(can_start)
            self.laser_alignment_stop.setEnabled(busy or self.variables.laser_alignment_enabled and running)
            self.laser_auto_alignment.setEnabled(not running)
            for field in self.laser_alignment_fields.values(): field.setEnabled(not busy)
            reference = self._stage_reference_worker is not None and self._stage_reference_worker.isRunning()
            self._set_stage_jog_enabled(not busy and not reference)
            self.laser_stage_reference.setEnabled(not busy and not running and self.flag_super_user_stage)
            if busy: self.laser_power.setEnabled(False)
            text = status.get('phase', 'idle')+': '+status.get('reason', '')
            if status.get('target_voltage') is not None:
                text += f" | DC target {status['target_voltage']:.0f} V"
            self.laser_alignment_label.setText(text+'\nScan: coarse → fine XY → focus Z → fine XY. Stop Alignment leaves acquisition running.')
            self.alignment_plot.append(self.variables.laser_alignment_plot)
        except Exception:
            self.variables.laser_alignment_cancel = True
            if self.variables.start_flag and self._laser_alignment_busy():
                self.variables.experiment_error = 'Laser alignment stage service failed.'
                self.variables.stop_flag = True
            logging.getLogger(__name__).exception('Laser alignment service failed')
