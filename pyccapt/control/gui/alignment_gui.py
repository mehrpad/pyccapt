"""Main-GUI automatic sample sequence and pre-experiment positioning."""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import replace
from types import SimpleNamespace

from PyQt6 import QtCore, QtWidgets

from pyccapt.control.apt.alignment_config import AlignmentConfig
from pyccapt.control.apt.detector_models import normalize_tdc_model
from pyccapt.control.apt.laser_alignment_config import LaserAlignmentConfig
from pyccapt.control.apt.laser_alignment_runtime import validate_laser_alignment
from pyccapt.control.devices.alignment_stage import transfer_waypoints
from pyccapt.control.gui import main_parameters


_RUN_FIELDS = ('user_name', 'ex_name', 'electrode', 'ex_time', 'ex_freq', 'max_ions',
               'vdc_min', 'vdc_max', 'vdc_step_up', 'vdc_step_down', 'v_p_min', 'v_p_max',
               'pulse_fraction', 'pulse_frequency', 'pulse_mode', 'counter_source',
               'control_algorithm', 'detection_rate', 'email', 'criteria_time',
               'criteria_ions', 'criteria_vdc', 'criteria_email', 'email_interval_events',
               'hdf5_data_name', 'hit_display', 'pulse_amp_per_supply_voltage')


class AlignmentGuiMixin:
    def _setup_alignment_fields(self):
        self._alignment_batch = []
        self._alignment_batch_index = 0
        self._alignment_waiting_cleanup = False
        self._alignment_transfer = None
        self._operator_stopped = False
        self._alignment_plot_window = None
        self.alignment_start_voltage = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.alignment_voltage_increment = QtWidgets.QDoubleSpinBox(self.centralwidget)
        for widget, value in (
            (self.alignment_start_voltage, self.conf.get('alignment_start_voltage', 1500)),
            (self.alignment_voltage_increment, self.conf.get('alignment_voltage_increment', 200)),
        ):
            widget.setRange(1, float(self.conf['max_vdc']))
            widget.setDecimals(0)
            widget.setSuffix(' V')
            widget.setValue(float(value))
            widget.setFixedWidth(self.automatic_alignment_button.width())
        self.alignment_start_voltage.setObjectName('alignment_start_voltage')
        self.alignment_voltage_increment.setObjectName('alignment_voltage_increment')
        self.alignment_start_voltage.setToolTip('DC voltage for the first XY alignment search (default 1500 V).')
        self.alignment_voltage_increment.setToolTip('Voltage increase after a failed XY search and return to its origin (default 200 V).')
        for text, widget, name in (
            ('Alignment start voltage', self.alignment_start_voltage, 'alignment_start_voltage_label'),
            ('Alignment voltage increment', self.alignment_voltage_increment, 'alignment_voltage_increment_label'),
        ):
            label = QtWidgets.QLabel(text, self.centralwidget)
            setattr(self, name, label)
            label.setWordWrap(True)
            label.setFixedWidth(self.automatic_alignment_button.width())
            self.electrode_controls.addWidget(label)
            self.electrode_controls.addWidget(widget)
        self.automatic_alignment_button.toggled.connect(self._update_alignment_fields)
        self.automatic_alignment_button.toggled.connect(self._toggle_alignment_plot)
        self._update_alignment_fields()
        self._alignment_batch_timer = QtCore.QTimer(self.centralwidget)
        self._alignment_batch_timer.timeout.connect(self._alignment_batch_tick)
        self._alignment_batch_timer.start(100)

    def _toggle_alignment_plot(self, selected):
        if selected:
            if self._alignment_plot_window is None:
                from pyccapt.control.gui.alignment_plot import AlignmentPlotWindow
                self._alignment_plot_window = AlignmentPlotWindow(
                    self.variables, self.centralwidget,
                    detector_radius_mm=float(self.conf['detector_diameter'])/2)
            self._alignment_plot_window.set_monitoring(True)
        elif self._alignment_plot_window is not None:
            self._alignment_plot_window.set_monitoring(False)

    def _update_alignment_fields(self):
        enabled = self.automatic_alignment_button.isChecked() and not self.variables.sample_selection_locked
        self.alignment_start_voltage.setEnabled(enabled)
        self.alignment_voltage_increment.setEnabled(enabled)

    def _lock_alignment_controls(self):
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        for widget in (self.electrode_button, self.flat_test_button, self.automatic_alignment_button,
                       self.parameters_source):
            widget.setEnabled(False)
        self.variables.sample_selection_locked = True
        self._update_parameter_editor_mode()
        self._update_alignment_fields()

    def _start_alignment_batch(self, samples):
        """Validate every sample and parameter set before the first move."""
        try:
            if self.parameters_source.currentText() == 'TOML Plan':
                from pyccapt.control.core.experiment_plan import alignment_samples
                if alignment_samples(self.result_list, self.variables.sample_rough_positions) != tuple(samples):
                    raise ValueError('Sample sequence must match the experiment plan sample IDs and order.')
            cfg = AlignmentConfig.from_mapping(self.conf, self.alignment_start_voltage.value(),
                                               self.alignment_voltage_increment.value())
            if self.variables.laser_alignment_enabled:
                laser_cfg = LaserAlignmentConfig.from_config(self.conf, self.variables.laser_alignment_settings)
                cfg = replace(cfg, max_voltage=min(cfg.max_voltage, laser_cfg.max_voltage))
                cfg.validate()
            if normalize_tdc_model(self.conf.get('tdc_model')) not in ('Surface_Concept', 'RoentDek'):
                raise ValueError('Automatic alignment requires a Surface Concept or RoentDek position-resolving detector.')
            positions = {number: tuple(self.variables.sample_rough_positions[number]) for number in samples}
            cfg.validate_motion(positions.values())
            stage = self.gui_stage_control
            if stage.stage_device is None:
                raise ValueError('Connect the sample stage before automatic alignment.')
            stage.stage_device.validate_alignment_state()
            if stage._reference_worker is not None and stage._reference_worker.isRunning():
                raise ValueError('Finish stage referencing before automatic alignment.')
            if stage.stage_device.is_moving():
                raise ValueError('Stop manual stage movement before automatic alignment.')
            if not self.variables.hardware_safe:
                raise ValueError('Output shutdown must finish before positioning a sample.')
            batch = []
            for index, sample in enumerate(samples):
                if self.parameters_source.currentText() == 'TOML Plan':
                    temp = SimpleNamespace(pulse_amp_per_supply_voltage=self.variables.pulse_amp_per_supply_voltage)
                    errors = []
                    main_parameters.apply_experiment_item(temp, self.conf, self.result_list[index], errors.append)
                    if errors:
                        raise ValueError('; '.join(errors))
                else:
                    temp = self.variables
                values = {field: getattr(temp, field) for field in _RUN_FIELDS}
                values['experiment_plan_snapshot'] = (
                    {'source': self._plan_path, 'queue_index': index+1,
                     'experiment': dict(self.result_list[index])}
                    if self.parameters_source.currentText() == 'TOML Plan' else {})
                main_parameters.validate_run_parameters(SimpleNamespace(**values), self.conf)
                if (values['counter_source'] != 'TDC' or values['pulse_mode'] not in ('Voltage', 'Laser')
                        or str(self.conf.get('tdc', 'off')) != 'on'
                        or str(self.conf.get('v_dc', 'off')) != 'on'):
                    raise ValueError('Automatic alignment requires a position-resolving TDC and enabled DC voltage in Voltage or Laser mode.')
                if self.variables.laser_alignment_enabled:
                    validate_laser_alignment(SimpleNamespace(**values,
                        flat_test_active=False, laser_alignment_settings=self.variables.laser_alignment_settings,
                        laser_stage_snapshot=self.variables.laser_stage_snapshot), self.conf)
                if values['detection_rate'] <= 0:
                    raise ValueError('Automatic alignment requires a positive target detection rate.')
                if not values['vdc_min'] <= cfg.start_voltage <= min(cfg.max_voltage, values['vdc_max']):
                    raise ValueError(f'Sample {sample}: alignment start voltage is outside its experiment voltage range.')
                batch.append((sample, positions[sample], values))
            self._alignment_batch = batch
            self._alignment_batch_index = 0
            self._operator_stopped = False
            self.variables.alignment_settings = cfg.snapshot()
            self.variables.alignment_sequence_id = uuid.uuid4().hex
            self.variables.alignment_plot_snapshot = {}
            self.variables.automatic_alignment_enabled = True
            self.variables.automatic_alignment_samples = tuple(samples)
            self.variables.vdc_hold = False
            self._lock_alignment_controls()
            self._position_alignment_sample()
        except Exception as exc:
            self._end_alignment_batch(str(exc))

    def _position_alignment_sample(self):
        if self._operator_stopped:
            self._end_alignment_batch()
            return
        sample, target, values = self._alignment_batch[self._alignment_batch_index]
        cfg = AlignmentConfig(**dict(self.variables.alignment_settings))
        stage = self.gui_stage_control.stage_device
        position = stage.get_position()
        current = tuple(float(position[axis]) for axis in 'xyz')
        waypoints = transfer_waypoints(current, target, cfg)
        self.variables.stage_position_snapshot = (*current, time.monotonic())
        self.variables.alignment_sample = sample
        self.variables.alignment_sample_position = target
        self.variables.alignment_plot_snapshot = {}
        self.variables.alignment_move_request = {}
        self.variables.alignment_move_status = {}
        self.variables.alignment_cancel_motion = False
        self.variables.stop_flag = False
        self.variables.alignment_transfer_log = []
        self._alignment_transfer = {'waypoints': waypoints, 'pending': None, 'cfg': cfg,
                                    'values': values, 'sample': sample, 'step': 0, 'label': ''}
        self.statusbar.showMessage(f'Sample {sample}: preparing stage transfer')

    def _alignment_batch_tick(self):
        if not getattr(self, '_alignment_batch', []):
            return
        try:
            if self._alignment_waiting_cleanup:
                if self._operator_stopped:
                    self._end_alignment_batch()
                    return
                busy = bool(self.variables.last_screen_shot)
                if self.camera_available:
                    busy |= bool(self.variables.flag_cameras_take_screenshot)
                if busy:
                    if time.monotonic() > self._alignment_cleanup_deadline:
                        self._end_alignment_batch('Previous sample visualization/screenshot cleanup timed out.')
                    return
                self._alignment_waiting_cleanup = False
                self._position_alignment_sample()
            transfer = self._alignment_transfer
            if transfer is None:
                state = self.variables.alignment_status
                if state:
                    message = (f"Sample {state['sample']}: {state['phase']} | "
                               f"fine attempt {state['attempt']} | {state['target_voltage']:g} V")
                    if state.get('reason'):
                        message += f" | {state['reason']}"
                    self.statusbar.showMessage(message)
                return
            if self._operator_stopped or self.variables.alignment_cancel_motion or self.variables.stop_flag:
                raise ValueError('Sample positioning stopped.')
            if time.monotonic()-self.variables.alignment_stage_heartbeat > 2:
                raise ValueError('Stage controller is not responding during sample positioning.')
            pending = transfer['pending']
            if pending:
                status = self.variables.alignment_move_status
                if time.monotonic()-pending['issued'] > transfer['cfg'].move_timeout_s+2:
                    raise ValueError('Sample positioning timed out.')
                if status.get('id') != pending['id']:
                    return
                if status.get('state') == 'error':
                    raise ValueError(status.get('error', 'Sample positioning failed.'))
                if status.get('state') != 'done':
                    position = self.variables.stage_position_snapshot[:3]
                    remaining_mm = max(abs(a-b) for a, b in zip(position, pending['target_m']))*1000
                    self.statusbar.showMessage(
                        f"Sample {transfer['sample']}: {transfer['label']} "
                        f"({transfer['step']}/3), {remaining_mm:.3f} mm remaining")
                    return
                records = list(self.variables.alignment_transfer_log)
                records.append({**pending, 'completed_monotonic': time.monotonic(),
                                'position_m': self.variables.stage_position_snapshot[:3]})
                self.variables.alignment_transfer_log = records
                logging.getLogger('pyccapt.gui').info(
                    'Sample %s transfer step %s/3 complete at XYZ %s m',
                    transfer['sample'], transfer['step'], records[-1]['position_m'])
                transfer['pending'] = None
            if transfer['waypoints']:
                transfer['step'] += 1
                step = transfer['step']
                label = ('retracting Z to transfer height', 'moving XY to saved position',
                         'moving Z to saved position')[step-1]
                transfer['label'] = label
                self.statusbar.showMessage(f"Sample {transfer['sample']}: {label} "
                                           f'({step}/3)')
                request = {'id': uuid.uuid4().hex, 'kind': 'transfer',
                           'target_m': transfer['waypoints'].pop(0),
                           'speed_um_s': transfer['cfg'].transfer_speed_um_s, 'issued': time.monotonic()}
                transfer['pending'] = request
                logging.getLogger('pyccapt.gui').info(
                    'Sample %s transfer step %s/3: %s, target XYZ %s m, speed %.3f mm/s',
                    transfer['sample'], step, label, request['target_m'],
                    request['speed_um_s']*1e-3)
                self.variables.alignment_move_request = request
                return
            # No outputs are enabled until all three moves are acknowledged.
            for name, value in transfer['values'].items():
                setattr(self.variables, name, value)
            if self.parameters_source.currentText() == 'TOML Plan':
                self.plan_table.selectRow(self._alignment_batch_index)
            self.variables.alignment_move_request = {}
            self.variables.alignment_move_status = {}
            self.variables.alignment_events = {}
            self.variables.alignment_outcome = ''
            self.variables.alignment_status = {}
            self.variables.alignment_window_epoch = uuid.uuid4().hex
            self._alignment_transfer = None
            self.statusbar.showMessage(f"Sample {transfer['sample']}: stage positioned; starting experiment")
            if not self.start_experiment_worker():
                self._end_alignment_batch('Automatic sequence stopped because experiment startup failed.')
        except Exception as exc:
            if self._alignment_transfer is None and self.variables.start_flag:
                self.stop_experiment_clicked()
                self.error_message(str(exc))
            else:
                self._end_alignment_batch(str(exc))

    def _alignment_sample_finished(self, run_error):
        """Called only after worker exit and full normal per-run cleanup."""
        outcome = self.variables.alignment_outcome
        if outcome == 'voltage_limit':
            self.error_message(f'Sample {self.variables.alignment_sample}: alignment voltage limit reached; sample skipped.')
        can_continue = (not self._operator_stopped and not run_error and self.variables.hardware_safe
                        and outcome not in ('attempt_limit', 'timeout', 'fault', 'clearance_limit', 'cancelled')
                        and self._alignment_batch_index+1 < len(self._alignment_batch))
        if not can_continue:
            self._end_alignment_batch(run_error)
            return
        self._alignment_batch_index += 1
        self._alignment_waiting_cleanup = True
        self._alignment_cleanup_deadline = time.monotonic()+15
        self._lock_alignment_controls()

    def _end_alignment_batch(self, error=''):
        self.variables.alignment_cancel_motion = True
        stage = getattr(self, 'gui_stage_control', None)
        if stage is not None and self.variables.automatic_alignment_enabled:
            stage.alignment_service.cancel('Automatic sequence finished')
        self._alignment_batch = []
        self._alignment_transfer = None
        self._batch_items = None
        self._alignment_waiting_cleanup = False
        self.variables.automatic_alignment_enabled = False
        self.variables.automatic_alignment_samples = ()
        self.variables.sample_selection_locked = False
        self.variables.experiment_plan_index = 0
        self.parameters_source.setEnabled(True)
        self.electrode_button.setEnabled(True)
        self.automatic_alignment_button.setEnabled(True)
        self._sync_electrode_controls()
        self._update_parameter_editor_mode()
        self._update_alignment_fields()
        if error:
            self.error_message(error)
