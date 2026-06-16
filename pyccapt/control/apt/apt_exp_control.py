import copy
import datetime
import multiprocessing
import time
from pathlib import Path

import numpy as np
import serial.tools.list_ports
from simple_pid import PID

from pyccapt.control.apt import apt_exp_control_func
from pyccapt.control.apt.detector_runtime import (
    DetectorRuntime,
    join_detector_processes,
    start_detector_processes,
)
from pyccapt.control.apt.experiment_state import (
    append_main_loop_results,
    ensure_output_directories,
    prepare_experiment_output_paths,
    reset_runtime_variables,
    validate_detector_data_lengths,
)
from pyccapt.control.core import experiment_statistics, hdf5_creator, loggi, runtime
from pyccapt.control.devices import initialize_devices, signal_generator


class APT_Exp_Control:
    """
    This class is responsible for controlling the experiment.
    """

    def __init__(self, variables, conf, experiment_finished_event, x_plot, y_plot, t_plot, main_v_dc_plot):

        self.stop_event = None
        self.control_algorithm = None
        self.com_port_v_dc = None
        self.initialization_v_p = None
        self.initialization_v_dc = None
        self.initialization_signal_generator = None
        self.pulse_mode = None
        self.variables = variables
        self.conf = conf
        self.experiment_finished_event = experiment_finished_event
        self.x_plot = x_plot
        self.y_plot = y_plot
        self.t_plot = t_plot
        self.main_v_dc_plot = main_v_dc_plot

        self.com_port_v_p = None
        self.log_apt = None
        self.variables.start_time = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
        # Guard against ex_freq == 0 (or accidentally negative) so that the
        # experiment subprocess does not die with ZeroDivisionError before
        # any log is written.
        ex_freq_value = float(getattr(self.variables, 'ex_freq', 0) or 0)
        if ex_freq_value <= 0:
            ex_freq_value = 10.0  # safe default, matches the GUI default
            self.variables.ex_freq = ex_freq_value
        self.sleep_time = 1.0 / ex_freq_value

        self.detection_rate = 0
        self.specimen_voltage = 0
        self.pulse_voltage = 0
        self.count_last = 0
        self.vdc_max = 0
        self.pulse_frequency = 0
        self.pulse_fraction = 0
        self.pulse_amp_per_supply_voltage = 0
        self.pulse_voltage_max = 0
        self.pulse_voltage_min = 0
        self.total_ions = 0
        self.total_raw_signals = 0
        self.count_raw_signals_last = 0
        self.ex_freq = 0

        self.main_v_pulse = []
        self.main_l_pulse = []
        self.main_counter = []
        self.main_raw_counter = []
        self.main_temperature = []
        self.main_chamber_vacuum = []
        # Per-iteration stage positions (meters), logged into apt/*.
        self.main_laser_x = []
        self.main_laser_y = []
        self.main_laser_z = []
        self.main_stage_x = []
        self.main_stage_y = []
        self.main_stage_z = []

        self.initialization_error = False
        self.detector_runtime = DetectorRuntime()
        self.access_override_enabled = False
        self.override_disabled_devices = set()

        # apt/* metadata chunk state: how many items have already been flushed
        # to disk and which chunk file ID to use next.
        self._apt_meta_flush_offset = 0
        self._apt_meta_chunk_id = 0

    def _is_config_enabled(self, key):
        value = str(self.conf.get(key, "off")).strip().lower()
        return value in {"on", "enabled", "true", "1"}

    def _is_override_disabled(self, device_name):
        return self.access_override_enabled and device_name in self.override_disabled_devices

    def _vdc_active(self):
        return self._is_config_enabled("v_dc") and bool(self.initialization_v_dc) and self.com_port_v_dc is not None

    def _vp_active(self):
        return self._is_config_enabled("v_p") and bool(self.initialization_v_p) and self.com_port_v_p is not None

    def _signal_generator_active(self):
        return self._is_config_enabled("signal_generator") and bool(self.initialization_signal_generator)

    def initialize_detector_process(self):
        """
        Initialize the detector process based on the configured settings.

        This method initializes the necessary queues and processes for data acquisition based on the configured settings.

        Args:
           None

        Returns:
           None
        """
        self.detector_runtime = start_detector_processes(
            self.conf,
            self.variables,
            self.x_plot,
            self.y_plot,
            self.t_plot,
            self.main_v_dc_plot,
            process_factory=multiprocessing.Process,
            event_factory=multiprocessing.Event,
        )
        self.stop_event = self.detector_runtime.stop_event
        self.tdc_process = self.detector_runtime.tdc_process
        self.hsd_process = self.detector_runtime.hsd_process
        if self.tdc_process is None and self.hsd_process is None:
            print("No counter source selected")

    # Flush apt/* metadata to a chunk file every this many main-loop steps
    # (~20 s at the default 5 Hz rate).  Smaller = more frequent checkpoints
    # but slightly more disk I/O.
    _APT_META_CHUNK_SIZE = 100

    def _flush_apt_meta_chunks(self, time_counter: list, time_ex: list) -> None:
        """Write accumulated apt/* metadata items (since last flush) to a chunk file.

        Uses a half-open slice [_apt_meta_flush_offset : current_len] so
        successive calls each write only the NEW items.  The in-memory lists
        are NOT cleared, which means append_main_loop_results() still works
        normally if chunk recovery is not needed.
        """
        start = self._apt_meta_flush_offset
        end = len(time_counter)
        if end <= start:
            return
        try:
            chunk_dir = Path(self.variables.path) / "temp_data" / "chunks"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            self._apt_meta_chunk_id += 1
            cid = self._apt_meta_chunk_id

            def _save(stem: str, seq, dtype) -> None:
                arr = np.asarray(seq[start:end], dtype=dtype)
                np.save(chunk_dir / f"{stem}_chunk_{cid}.npy", arr)

            _save("apt_id",              time_counter,             np.uint64)
            _save("apt_timestamps",      time_ex,                  np.float64)
            _save("apt_num_events",      self.main_counter,        np.uint32)
            _save("apt_num_raw_signals", self.main_raw_counter,    np.uint32)
            _save("apt_temperature",     self.main_temperature,    np.float64)
            _save("apt_vacuum",          self.main_chamber_vacuum, np.float64)
            _save("apt_laser_x", self.main_laser_x, np.float64)
            _save("apt_laser_y", self.main_laser_y, np.float64)
            _save("apt_laser_z", self.main_laser_z, np.float64)
            _save("apt_stage_x", self.main_stage_x, np.float64)
            _save("apt_stage_y", self.main_stage_y, np.float64)
            _save("apt_stage_z", self.main_stage_z, np.float64)

            self._apt_meta_flush_offset = end
        except Exception as exc:
            if self.log_apt is not None:
                self.log_apt.warning("apt meta chunk flush failed (chunk %d): %s", self._apt_meta_chunk_id, exc)

    def main_ex_loop(
        self,
    ):
        """
        Execute main experiment loop.

        This method contains all methods that iteratively run to control the experiment. It reads the number of detected
        ions, calculates the error of the desired rate, and regulates the high voltage and pulser accordingly.

        Args:
            None

        Returns:
            None
        """
        # Update total_ions based on the counter_source...
        # Calculate count_temp and update variables...
        # Save high voltage, pulse, and current iteration ions...
        # Calculate counts_measured and counts_error...
        # Perform control algorithm with averaging...
        # Update v_dc and v_p...
        # Update other experiment variables...

        # with self.variables.lock_statistics:
        count_temp = self.total_ions - self.count_last
        self.count_last = self.total_ions

        count_raw_signals_temp = self.total_raw_signals - self.count_raw_signals_last
        self.count_raw_signals_last = self.total_raw_signals

        # saving the values of high dc voltage, pulse, and current iteration ions
        # with self.variables.lock_experiment_variables:
        self.main_counter.extend([count_temp])
        self.main_raw_counter.extend([count_raw_signals_temp])
        self.main_temperature.extend([self.variables.temperature])
        self.main_chamber_vacuum.extend([self.variables.vacuum_main])
        # Stage positions (meters) for this iteration, published by the
        # laser/stage GUIs. Read once per loop, like temperature/vacuum.
        self.main_laser_x.extend([self.variables.laser_pos_x])
        self.main_laser_y.extend([self.variables.laser_pos_y])
        self.main_laser_z.extend([self.variables.laser_pos_z])
        self.main_stage_x.extend([self.variables.stage_pos_x])
        self.main_stage_y.extend([self.variables.stage_pos_y])
        self.main_stage_z.extend([self.variables.stage_pos_z])

        # Re-read the algorithm choice every iteration so the user can
        # switch between Proportional / Aggressive / Adaptive / PID live
        # via the main GUI dropdown without restarting the experiment.
        live_algorithm = self.variables.control_algorithm
        if live_algorithm != self.control_algorithm:
            self._switch_control_algorithm(live_algorithm)

        error = self.detection_rate - self.variables.detection_rate_current

        if self.control_algorithm == 'Proportional':
            # P with deadband + asymmetric up/down gains + hard cap.
            if error > 0.05:
                voltage_step = error * self.variables.vdc_step_up * 10
            elif error < -0.05:
                voltage_step = error * self.variables.vdc_step_down * 10
            else:
                voltage_step = 0
            voltage_step = max(-40, min(40, voltage_step))

        elif self.control_algorithm == 'Proportional aggressive':
            # Same as Proportional but the upward gain is multiplied by
            # control_p_aggressive_up_factor (down-gain stays normal so
            # the loop still brakes gently when rate is too high).
            if error > 0.05:
                voltage_step = error * self.variables.vdc_step_up * 10 * self._p_aggressive_factor
            elif error < -0.05:
                voltage_step = error * self.variables.vdc_step_down * 10
            else:
                voltage_step = 0
            voltage_step = max(-40, min(40, voltage_step))

        elif self.control_algorithm == 'Adaptive P':
            # Proportional core, but the up-step gain is auto-scaled by
            # _adapt_factor based on observed loop behaviour:
            #   * many same-sign errors in a row -> loop is sluggish -> grow
            #   * sign-flipping errors -> loop is overshooting -> shrink
            sign = 1 if error > 0 else (-1 if error < 0 else 0)
            if sign != 0 and sign == self._adapt_last_sign:
                self._adapt_same_sign += 1
                self._adapt_flip_count = max(0, self._adapt_flip_count - 1)
            elif sign != 0 and self._adapt_last_sign != 0:
                self._adapt_flip_count += 1
                self._adapt_same_sign = 0
            self._adapt_last_sign = sign

            if self._adapt_same_sign >= self._adapt_grow_threshold:
                self._adapt_factor = min(self._adapt_factor * 1.1, self._adapt_max_factor)
                self._adapt_same_sign = 0
            elif self._adapt_flip_count >= self._adapt_shrink_threshold:
                self._adapt_factor = max(self._adapt_factor * 0.7, self._adapt_min_factor)
                self._adapt_flip_count = 0

            if error > 0.05:
                voltage_step = error * self.variables.vdc_step_up * 10 * self._adapt_factor
            elif error < -0.05:
                voltage_step = error * self.variables.vdc_step_down * 10 * self._adapt_factor
            else:
                voltage_step = 0
            voltage_step = max(-40, min(40, voltage_step))

        elif self.control_algorithm == 'PID':
            # simple_pid expects the *measurement* (it computes error itself).
            # output_limits are symmetric volts/iteration so the controller
            # can also drive voltage *down* when the rate is too high.
            voltage_step = self.pid(self.variables.detection_rate_current)
            cap = self._pid_max_step
            voltage_step = max(-cap, min(cap, voltage_step))

        else:
            voltage_step = 0

        # update v_dc
        if not self.variables.vdc_hold and voltage_step != 0:
            specimen_voltage_temp = min(self.specimen_voltage + voltage_step, self.vdc_max)
            if specimen_voltage_temp > self.vdc_min:
                if specimen_voltage_temp != self.specimen_voltage:
                    if self._vdc_active():
                        apt_exp_control_func.command_v_dc(self.com_port_v_dc, ">S0 %s" % specimen_voltage_temp)
                        self.specimen_voltage = specimen_voltage_temp
                        self.variables.specimen_voltage = self.specimen_voltage
                        self.variables.specimen_voltage_plot = self.specimen_voltage
                    if self.pulse_mode in ['Voltage', 'VoltageLaser']:
                        new_vp = self.specimen_voltage * (self.pulse_fraction / 100) / self.pulse_amp_per_supply_voltage
                        if self.pulse_voltage_max > new_vp > self.pulse_voltage_min and self._vp_active():
                            apt_exp_control_func.command_v_p(self.com_port_v_p, 'VOLT %s' % new_vp)
                            self.pulse_voltage = new_vp * self.pulse_amp_per_supply_voltage
                            self.variables.pulse_voltage = self.pulse_voltage

    def precise_sleep(self, seconds):
        """
        Precise sleep function.

        Args:
            seconds:    Seconds to sleep

        Returns:
            None
        """
        start_time = time.perf_counter()
        while time.perf_counter() - start_time < seconds:
            pass

    def _build_pid(self):
        """(Re)create the PID controller from the current config gains."""
        self.pid = PID(self._pid_kp, self._pid_ki, self._pid_kd, setpoint=self.detection_rate)
        self.pid.sample_time = 1.0 / self.variables.ex_freq
        self.pid.output_limits = (-self._pid_max_step, self._pid_max_step)
        # P-on-measurement avoids derivative kick on setpoint changes.
        self.pid.proportional_on_measurement = True

    def _switch_control_algorithm(self, new_algorithm):
        """Handle a live algorithm change from the GUI dropdown.

        Called from the per-iteration control branch.  Resets per-mode
        state so transitions don't leak old gains/integrators.
        """
        valid = ('Proportional', 'Proportional aggressive', 'Adaptive P', 'PID')
        if new_algorithm not in valid:
            return  # ignore unknown values
        self.control_algorithm = new_algorithm
        # Reset Adaptive-P state on every switch so the multiplier and
        # sign-tracker start fresh for the new mode.
        self._adapt_factor = 1.0
        self._adapt_same_sign = 0
        self._adapt_flip_count = 0
        self._adapt_last_sign = 0
        # Build / rebuild the PID object only when entering PID mode.
        if new_algorithm == 'PID':
            self._build_pid()
        else:
            # Drop the PID instance so its accumulated I-term doesn't
            # carry over if the user switches back later.
            self.pid = None

    def run_experiment(self):
        """
        Run the main experiment.

        This method initializes devices, starts the experiment loop, monitors various criteria, and manages experiment
        stop conditions and data storage.

        Returns:
            None
        """
        self.variables.flag_visualization_start = True
        self.pulse_mode = self.variables.pulse_mode
        self.control_algorithm = self.variables.control_algorithm
        self.access_override_enabled = bool(getattr(self.variables, "access_override_enabled", False))
        self.override_disabled_devices = set(getattr(self.variables, "override_disabled_devices", []))

        # if os.path.exists("./files/counter_experiments.txt"):
        #     # Read the experiment counter
        #     with open('./files/counter_experiments.txt') as f:
        #         self.variables.counter = int(f.readlines()[0])
        # else:
        #     # create a new txt file
        #     with open('./files/counter_experiments.txt', 'w') as f:
        #         f.write(str(1))  # Current time and date
        data_path, path_meta = prepare_experiment_output_paths(self.variables)
        # Create folder to save the data
        try:
            ensure_output_directories(data_path, path_meta)
        except Exception as exc:
            print('Can not create the directory for saving the data')
            print(exc)
            self.variables.stop_flag = True
            self.initialization_error = True

        if self._is_config_enabled('tdc') and not self.initialization_error and not self._is_override_disabled("tdc"):
            self.variables.flag_tdc_failure = False
            self.initialize_detector_process()
        else:
            self.variables.flag_finished_tdc = True

        self.log_apt = loggi.logger_creator('apt', self.variables, 'apt.log', path=self.variables.log_path)
        # Record the inputs that produced this experiment so the apt.log file
        # is self-contained for later debugging / forensics.
        try:
            self.log_apt.info('Experiment name: %s', getattr(self.variables, 'exp_name', '<unset>'))
            self.log_apt.info('Output folder  : %s', getattr(self.variables, 'path', '<unset>'))
            self.log_apt.info('Pulse mode     : %s', self.pulse_mode)
            self.log_apt.info('Counter source : %s', getattr(self.variables, 'counter_source', '<unset>'))
            self.log_apt.info('Control alg.   : %s', self.control_algorithm)
            self.log_apt.info('Detection rate : %s', getattr(self.variables, 'detection_rate', '<unset>'))
            self.log_apt.info('Ex frequency   : %s Hz', getattr(self.variables, 'ex_freq', '<unset>'))
            self.log_apt.info(
                'Vdc range      : %s -> %s V',
                getattr(self.variables, 'vdc_min', '<unset>'),
                getattr(self.variables, 'vdc_max', '<unset>'),
            )
            if self.access_override_enabled:
                self.log_apt.warning(
                    'Super-user override active. Disabled devices: %s', sorted(self.override_disabled_devices)
                )
            loggi.log_configuration_snapshot(self.log_apt, self.conf, self.variables)
        except Exception:
            self.log_apt.debug('Could not log experiment context', exc_info=True)
        if (
            self._is_config_enabled('signal_generator')
            and not self._is_override_disabled("signal_generator")
            and self.pulse_mode in ['Voltage', 'VoltageLaser']
            and not self.initialization_error
        ):
            self.initialization_error = apt_exp_control_func.initialization_signal_generator(self.variables, self.log_apt)
            if not self.initialization_error:
                self.initialization_signal_generator = True

        if self._is_config_enabled('v_dc') and not self._is_override_disabled("v_dc") and not self.initialization_error:
            try:
                self.com_port_v_dc = serial.Serial(
                    port=self.variables.COM_PORT_V_dc,
                    baudrate=115200,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                )
            except Exception as e:
                print('Can not open the COM port for V_dc')
                print(e)
                self.initialization_v_dc = True
            if not self.initialization_error:
                self.initialization_error = apt_exp_control_func.initialization_v_dc(
                    self.com_port_v_dc, self.log_apt, self.variables
                )
            if not self.initialization_error:
                self.initialization_v_dc = True

        if (
            self._is_config_enabled('v_p')
            and not self._is_override_disabled("v_p")
            and self.pulse_mode in ['Voltage', 'VoltageLaser']
        ):
            # Initialize pulser
            try:
                self.com_port_v_p = serial.Serial(self.variables.COM_PORT_V_p, baudrate=115200, timeout=0.01)
            except Exception as e:
                print('Can not open the COM port for V_p')
                print(e)
                self.initialization_v_p = True
            if not self.initialization_error:
                self.initialization_error = apt_exp_control_func.initialization_v_p(
                    self.com_port_v_p, self.log_apt, self.variables
                )

            if not self.initialization_error:
                self.initialization_v_p = True
        elif self._is_config_enabled('laser') and self.pulse_mode in ['Laser', 'VoltageLaser']:
            print(f"{initialize_devices.bcolors.WARNING}Warning: turn on the laser manually{initialize_devices.bcolors.ENDC}")

        self.variables.specimen_voltage = self.variables.vdc_min
        if self.pulse_mode in ['Voltage', 'VoltageLaser']:
            self.variables.pulse_voltage = self.variables.v_p_min

        time_ex = []
        time_counter = []

        steps = 0
        flag_achieved_high_voltage = 0
        index_time = 0

        desired_rate = self.variables.ex_freq  # Hz
        desired_period = 1.0 / desired_rate  # seconds
        self.pulse_frequency = self.variables.pulse_frequency * 1000
        self.counts_target = self.pulse_frequency * self.variables.detection_rate / 100

        # Turn on the v_dc and v_p
        if not self.initialization_error:
            if self.pulse_mode in ['Voltage', 'VoltageLaser']:
                if self._vp_active():
                    apt_exp_control_func.command_v_p(self.com_port_v_p, 'OUTPut ON')
                    vol = self.variables.v_p_min / self.variables.pulse_amp_per_supply_voltage
                    cmd = 'VOLT %s' % vol
                    apt_exp_control_func.command_v_p(self.com_port_v_p, cmd)
                    time.sleep(0.1)
            elif self.pulse_mode in ['Laser', 'VoltageLaser']:
                if self._is_config_enabled('laser'):
                    print(
                        f"{initialize_devices.bcolors.WARNING}Warning: enable output of laser manually"
                        f"{initialize_devices.bcolors.ENDC}"
                    )
            if self._vdc_active():
                apt_exp_control_func.command_v_dc(self.com_port_v_dc, "F1")
                time.sleep(0.1)

        self.pulse_fraction = self.variables.pulse_fraction
        self.pulse_amp_per_supply_voltage = self.variables.pulse_amp_per_supply_voltage
        self.specimen_voltage = self.variables.specimen_voltage
        if self.pulse_mode in ['Voltage', 'VoltageLaser']:
            self.pulse_voltage = self.variables.pulse_voltage

        # ----- Per-algorithm state ------------------------------------------
        # Loaded from config.toml and initialised once; the per-iteration
        # branch above reads them.  `_switch_control_algorithm` rebuilds the
        # PID object on demand when the user changes mode mid-run.
        self._p_aggressive_factor = float(self.conf.get('control_p_aggressive_up_factor', 3.0))
        self._adapt_min_factor = float(self.conf.get('control_adaptive_min_factor', 0.3))
        self._adapt_max_factor = float(self.conf.get('control_adaptive_max_factor', 3.0))
        self._adapt_grow_threshold = int(self.conf.get('control_adaptive_grow_threshold', 5))
        self._adapt_shrink_threshold = int(self.conf.get('control_adaptive_shrink_threshold', 3))
        self._adapt_factor = 1.0
        self._adapt_same_sign = 0
        self._adapt_flip_count = 0
        self._adapt_last_sign = 0

        self._pid_kp = float(self.conf.get('control_pid_kp', 1.0))
        self._pid_ki = float(self.conf.get('control_pid_ki', 0.1))
        self._pid_kd = float(self.conf.get('control_pid_kd', 0.05))
        self._pid_max_step = float(self.conf.get('control_pid_max_step_v', 40.0))
        self.pid = None
        if self.control_algorithm == 'PID':
            self._build_pid()

        self.ex_freq = self.variables.ex_freq

        # Wait for 8 second to all devices get ready specially tdc
        time.sleep(8)
        self.log_apt.info('Experiment is started')
        # Main loop of experiment
        remaining_time_list = []
        total_ions_tmp = 0
        index_tdc_failure = 0
        last_pulse_mode = self.pulse_mode
        flag_change_pulse_mode = False
        pulse_frequency_tmp = self.pulse_frequency
        if self.initialization_error:
            pass
        else:
            while True:
                start_time = time.perf_counter()
                self.vdc_max = self.variables.vdc_max
                self.vdc_min = self.variables.vdc_min
                self.pulse_frequency = self.variables.pulse_frequency * 1000
                if self.pulse_mode in ['Voltage', 'VoltageLaser']:
                    self.pulse_voltage_min = self.variables.v_p_min / self.pulse_amp_per_supply_voltage
                    self.pulse_voltage_max = self.variables.v_p_max / self.pulse_amp_per_supply_voltage
                if pulse_frequency_tmp != self.pulse_frequency:
                    self.pulse_frequency = self.variables.pulse_frequency * 1000
                    pulse_frequency_tmp = self.pulse_frequency
                    self.counts_target = self.pulse_frequency * self.detection_rate / 100
                    if self.pulse_mode in ['Voltage', 'VoltageLaser'] and self._signal_generator_active():
                        signal_generator.change_frequency_signal_generator(self.variables, self.pulse_frequency / 1000)
                    elif self.pulse_mode in ['Laser', 'VoltageLaser']:
                        pass

                if self.detection_rate != self.variables.detection_rate:
                    self.detection_rate = self.variables.detection_rate
                    self.counts_target = self.pulse_frequency * self.detection_rate / 100
                    if self.control_algorithm == 'PID' and self.pid is not None:
                        self.pid.setpoint = self.detection_rate

                self.total_ions = self.variables.total_ions
                self.total_raw_signals = self.variables.total_raw_signals
                # TDC failure detection should only run for an active TDC
                # process and only after the detector has produced at least
                # one event. Early zero-ion ramp-up is physically valid.
                detector_running = self.variables.counter_source == 'TDC' and self.detector_runtime.tdc_process is not None
                has_seen_tdc_events = self.total_ions > 0 or total_ions_tmp > 0
                if detector_running and has_seen_tdc_events and total_ions_tmp == self.total_ions and not self.variables.vdc_hold:
                    index_tdc_failure += 1
                    if index_tdc_failure > 200:
                        self.variables.flag_tdc_failure = True
                else:
                    index_tdc_failure = 0
                    total_ions_tmp = copy.deepcopy(self.total_ions)

                if self.variables.vdc_hold:
                    self.pulse_mode = self.variables.pulse_mode
                    # if the vdc is hold, we need to check if the pulse mode is changed or not to initialize the
                    # pulser and set the voltage
                    if last_pulse_mode != self.pulse_mode:
                        flag_change_pulse_mode = True
                        last_pulse_mode = self.pulse_mode
                    if flag_change_pulse_mode and self.pulse_mode in ['Voltage', 'VoltageLaser']:
                        # if the pulse mode is changed from laser to voltage, we need to initialize the pulser
                        if not self.initialization_v_p and not self._is_override_disabled("v_p"):
                            try:
                                # Initialize pulser
                                self.com_port_v_p = serial.Serial(self.variables.COM_PORT_V_p, baudrate=115200, timeout=0.01)
                                self.initialization_error = apt_exp_control_func.initialization_v_p(
                                    self.com_port_v_p, self.log_apt, self.variables
                                )
                                self.initialization_v_p = True
                                apt_exp_control_func.command_v_p(self.com_port_v_p, 'OUTPut ON')
                            except Exception as e:
                                print('Can not open the COM port for V_p')
                                print(e)
                        # if the pulse mode is changed from voltage to laser, we need to turn on the signal generator
                        if not self.initialization_signal_generator and not self._is_override_disabled("signal_generator"):
                            self.initialization_error = apt_exp_control_func.initialization_signal_generator(
                                self.variables, self.log_apt
                            )
                            if not self.initialization_error:
                                self.initialization_signal_generator = True
                        # set the v_dc and v_p
                        self.pulse_voltage_min = self.variables.v_p_min / self.pulse_amp_per_supply_voltage
                        self.pulse_voltage_max = self.variables.v_p_max / self.pulse_amp_per_supply_voltage
                        start_vp = self.specimen_voltage * (self.pulse_fraction / 100) / self.pulse_amp_per_supply_voltage
                        if start_vp < self.pulse_voltage_min:
                            start_vp = self.variables.v_p_min / self.variables.pulse_amp_per_supply_voltage

                        if self.pulse_voltage_max > start_vp > self.pulse_voltage_min - 1 and self._vp_active():
                            apt_exp_control_func.command_v_p(self.com_port_v_p, 'VOLT %s' % start_vp)
                            self.pulse_voltage = start_vp * self.pulse_amp_per_supply_voltage
                            self.variables.pulse_voltage = self.pulse_voltage
                        flag_change_pulse_mode = False
                    elif flag_change_pulse_mode and self.pulse_mode in ['Laser']:
                        if self.com_port_v_p is not None:
                            # if switch to laser mode chamge the voltage to zero
                            apt_exp_control_func.command_v_p(self.com_port_v_p, 'VOLT 0')
                            self.pulse_voltage = 0
                            self.variables.pulse_voltage = self.pulse_voltage
                            flag_change_pulse_mode = False

                    else:
                        if self.variables.flag_new_min_voltage:
                            if self.vdc_min > self.vdc_max:
                                self.vdc_min = self.vdc_max
                            decrement_vol = (self.specimen_voltage - self.vdc_min) / 10
                            for _ in range(10):
                                self.specimen_voltage -= decrement_vol
                                if self._vdc_active():
                                    apt_exp_control_func.command_v_dc(self.com_port_v_dc, ">S0 %s" % self.specimen_voltage)
                                time.sleep(0.3)
                            if self._vdc_active() and self.pulse_mode in ['Voltage', 'VoltageLaser']:
                                new_vp = (
                                    self.specimen_voltage * (self.pulse_fraction / 100) / self.pulse_amp_per_supply_voltage
                                )
                                if self.pulse_voltage_max > new_vp > self.pulse_voltage_min and self._vp_active():
                                    apt_exp_control_func.command_v_p(self.com_port_v_p, 'VOLT %s' % new_vp)
                                    self.pulse_voltage = new_vp * self.pulse_amp_per_supply_voltage
                                    self.variables.pulse_voltage = self.pulse_voltage

                            self.variables.specimen_voltage = self.specimen_voltage
                            self.variables.specimen_voltage_plot = self.specimen_voltage
                            self.variables.flag_new_min_voltage = False

                # main loop function
                self.main_ex_loop()

                # Counter of iteration
                time_counter.append(steps)

                # Measure time
                current_time = datetime.datetime.now()
                current_time_with_microseconds = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")  # Format with microseconds
                current_time_unix = datetime.datetime.strptime(
                    current_time_with_microseconds, "%Y-%m-%d %H:%M:%S.%f"
                ).timestamp()
                time_ex.append(current_time_unix)

                if self.variables.stop_flag:
                    self.log_apt.info('Experiment is stopped')
                    if self._is_config_enabled('tdc'):
                        if self.variables.counter_source == 'TDC':
                            self.variables.flag_stop_tdc = True
                            if self.stop_event is not None:
                                self.stop_event.set()  # Signal the tdc to stop
                    time.sleep(1)
                    break

                if self.variables.flag_tdc_failure:
                    self.log_apt.info('Experiment is stopped because of tdc failure')
                    print(f"{initialize_devices.bcolors.FAIL}Experiment is stopped because of TDC failure")
                    print(f"{initialize_devices.bcolors.FAIL}Restart the TDC and start the experiment again")
                    if self._is_config_enabled('tdc'):
                        if self.variables.counter_source == 'TDC':
                            self.variables.stop_flag = True  # Set the STOP flag
                            if self.stop_event is not None:
                                self.stop_event.set()  # Signal the tdc to stop
                    time.sleep(1)
                    break

                if self.variables.criteria_ions:
                    # Guard against the degenerate case max_ions == 0, which
                    # would otherwise satisfy the condition on iteration 1
                    # (both sides == 0) and stop the experiment immediately.
                    if self.variables.max_ions > 0 and self.variables.max_ions <= self.total_ions:
                        self.log_apt.info('Experiment is stopped because total number of ions is achieved')
                        if self._is_config_enabled('tdc'):
                            if self.variables.counter_source == 'TDC':
                                self.variables.flag_stop_tdc = True
                                self.variables.stop_flag = True  # Set the STOP flag
                                if self.stop_event is not None:
                                    self.stop_event.set()  # Signal the tdc to stop
                        time.sleep(1)
                        break
                if self.variables.criteria_vdc:
                    if self.vdc_max <= self.specimen_voltage:
                        if flag_achieved_high_voltage > self.ex_freq * 10:
                            self.log_apt.info('Experiment is stopped because dc voltage Max. is achieved')
                            if self._is_config_enabled('tdc'):
                                if self.variables.counter_source == 'TDC':
                                    self.variables.flag_stop_tdc = True
                                    self.variables.stop_flag = True  # Set the STOP flag
                                    if self.stop_event is not None:
                                        self.stop_event.set()  # Signal the tdc to stop
                            time.sleep(1)
                            break
                        flag_achieved_high_voltage += 1
                    else:
                        # Reset the dwell counter — we want ~10 s of
                        # *contiguous* time at Vdc max, not the cumulative
                        # time across separate excursions to max.
                        flag_achieved_high_voltage = 0

                if self.variables.criteria_time:
                    if self.variables.elapsed_time >= self.variables.ex_time:
                        self.log_apt.info('Experiment is stopped because experiment time Max. is achieved')
                        if self._is_config_enabled('tdc'):
                            if self.variables.counter_source == 'TDC':
                                self.variables.flag_stop_tdc = True
                                if self.stop_event is not None:
                                    self.stop_event.set()  # Signal the tdc to stop
                        # Set the stop flag and exit the loop unconditionally —
                        # without this break the loop kept running for one more
                        # iteration before stop_flag handling caught it on the
                        # next pass, which is inconsistent with the criteria_ions
                        # and criteria_vdc branches above.
                        self.variables.stop_flag = True
                        time.sleep(1)
                        break

                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                remaining_time = desired_period - elapsed_time

                if remaining_time > 0:
                    self.precise_sleep(remaining_time)
                elif remaining_time < 0:
                    index_time += 1
                    remaining_time_list.append(elapsed_time)

                steps += 1

                # Periodic apt/* metadata checkpoint so the data survives a crash.
                if steps % self._APT_META_CHUNK_SIZE == 0:
                    self._flush_apt_meta_chunks(time_counter, time_ex)

        self.variables.start_flag = False  # Set the START flag
        time.sleep(1)

        self.log_apt.info('Experiment is finished')
        print(
            "Experiment process: Experiment loop took longer than %s Millisecond for %s times out of %s "
            "iteration" % (int(1000 / self.variables.ex_freq), index_time, steps)
        )
        self.log_apt.warning(
            'Experiment loop took longer than %s (ms) for %s times out of %s iteration.'
            % (int(1000 / self.variables.ex_freq), index_time, steps)
        )

        if self.variables.counter_source == 'TDC' and self.detector_runtime.tdc_process is not None:
            # Teardown cost does NOT scale with total ion count: the bulk data is
            # streamed to temp_data/chunks/ continuously during the run (the save
            # worker sustains ~200k events/s, far above any APT detection rate, so
            # it never backs up). At shutdown only the final residual chunk
            # (< CHUNK_SIZE) is written and the device is deinitialized before
            # flag_finished_tdc is set -- a few seconds even for 100M+ ion runs.
            # 60 s is generous headroom. Timing out is non-fatal: the residual
            # chunk is already on disk before flag_finished_tdc would be set, and
            # join_detector_processes() below joins (never kills) the worker.
            tdc_finish_timeout_s = 60
            print('Waiting for TDC process to be finished for maximum %s seconds...' % tdc_finish_timeout_s)
            for i in range(tdc_finish_timeout_s):
                if self.variables.flag_finished_tdc:
                    print('TDC process is finished')
                    break
                print('%s seconds passed' % i)
                time.sleep(1)
            else:
                print('TDC process is not finished after %s seconds' % tdc_finish_timeout_s)
                self.log_apt.warning('TDC process is not finished after %s seconds', tdc_finish_timeout_s)
        else:
            self.variables.flag_finished_tdc = True

        try:
            join_detector_processes(self.conf, self.variables, self.detector_runtime)
        except Exception as exc:
            print(
                f"{initialize_devices.bcolors.WARNING}Warning: The TDC or HSD process cannot be terminated "
                f"properly{initialize_devices.bcolors.ENDC}"
            )
            print(exc)

        # Final apt/* metadata checkpoint (the tail that did not fill a full chunk).
        self._flush_apt_meta_chunks(time_counter, time_ex)

        append_main_loop_results(
            self.variables,
            self.main_counter,
            self.main_raw_counter,
            self.main_temperature,
            self.main_chamber_vacuum,
        )

        if not self._is_config_enabled('tdc'):
            if self.variables.counter_source == 'TDC':
                self.variables.total_ions = len(self.variables.x)
        elif self.variables.counter_source == 'HSD':
            pass

        # This flag set to True to save the last screenshot of the experiment in the GUI visualization
        self.variables.last_screen_shot = True
        validate_detector_data_lengths(self.variables, self.conf, self.log_apt)

        # Sanity check: stage/laser positions are published by the Stage Control
        # and Laser Control GUIs' poll timers. If a SmarAct stage was not
        # connected during the run those values stay at their 0.0 default, so
        # apt/stage_* (or apt/laser_*) is written all-zero. Surface that here so
        # it shows up in apt.log instead of only being discovered later in the
        # HDF5 file (the GUI session log records *why* the stage didn't connect).
        if self.log_apt is not None:
            if self.main_stage_x and not (
                    any(self.main_stage_x) or any(self.main_stage_y) or any(self.main_stage_z)
            ):
                self.log_apt.warning(
                    "Specimen-stage position was 0 for the whole run: the Stage "
                    "Control GUI never published a position (SmarAct main stage "
                    "not connected?). apt/stage_* will be all zero."
                )
            if self.main_laser_x and not (
                    any(self.main_laser_x) or any(self.main_laser_y) or any(self.main_laser_z)
            ):
                self.log_apt.warning(
                    "Laser-stage position was 0 for the whole run: the Laser "
                    "Control GUI never published a position (SmarAct laser stage "
                    "not connected?). apt/laser_* will be all zero."
                )

        try:
            self.variables.end_time = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
            # save data in hdf5 file
            hdf5_creator.hdf_creator(self.variables, self.conf, time_counter, time_ex)
            # Adding results of the experiment to the log file
            self.log_apt.info('Total number of Ions is: %s' % self.variables.total_ions)
            self.log_apt.info('HDF5 file is created')

            # save setup parameters and run statistics in a txt file
            experiment_statistics.save_statistics_apt(self.variables, self.conf)

            # send an email
            if len(self.variables.email) > 3:
                try:
                    apt_exp_control_func.send_info_email(self.log_apt, self.variables)
                except Exception:
                    # Email failure must not block counter increment, HDF5
                    # closure, or any other finalization step.
                    self.log_apt.exception('Email notification failed')

            # Save new value of experiment counter
            counter_path = runtime.ensure_counter_file()
            self.variables.counter += 1
            counter_path.write_text(str(self.variables.counter), encoding="utf-8")
            self.log_apt.info('Experiment counter is increased')
        except Exception as exc:
            message = f'Experiment finalization failed: {exc.__class__.__name__}: {exc}'
            print(message)
            if self.log_apt is not None:
                self.log_apt.exception(message)
        finally:
            try:
                # Clear up all the variables and deinitialize devices
                self.clear_up()
                if self.log_apt is not None:
                    self.log_apt.info('Variables and devices are cleared and deinitialized')
            except Exception as exc:
                print(f'Experiment cleanup failed: {exc.__class__.__name__}: {exc}')
                if self.log_apt is not None:
                    self.log_apt.exception('Experiment cleanup failed')
            self.experiment_finished_event.set()
            self.variables.flag_end_experiment = True

    def clear_up(self):
        """
        Clear class variables, deinitialize high voltage and pulser, and reset variables.

        This method performs the cleanup operations at the end of the experiment. It turns off the high voltage,
        pulser, and signal generator, resets global variables, and performs other cleanup tasks.

        Args:
            None

        Returns:
            None
        """

        self.log_apt.info('Starting cleanup')

        try:
            if self._vdc_active():
                # Turn off the v_dc
                apt_exp_control_func.command_v_dc(self.com_port_v_dc, 'F0')
                self.com_port_v_dc.close()
        except Exception as e:
            print(e)

        try:
            if self._vp_active():
                # Turn off the v_p
                apt_exp_control_func.command_v_p(self.com_port_v_p, 'VOLT 0')
                apt_exp_control_func.command_v_p(self.com_port_v_p, 'OUTPut OFF')
                self.com_port_v_p.close()
        except Exception as e:
            print(e)

        try:
            if self._signal_generator_active():
                # Turn off the signal generator. Pass variables so the
                # VISA resource address comes from config; the previous
                # hardcoded serial number silently no-op'd on every
                # rig except the original developer machine.
                signal_generator.turn_off_signal_generator(self.variables)

        except Exception as e:
            print(e)

        # Reset variables
        reset_runtime_variables(
            self.variables,
            self.x_plot,
            self.y_plot,
            self.t_plot,
            self.main_v_dc_plot,
        )
        self.log_apt.info('Cleanup is finished')


def run_experiment(variables, conf, experiment_finished_event, x_plot, y_plot, t_plot, main_v_dc_plot):
    """
    Run the main experiment.

    Args:
        variables:                  Global variables
        conf:                       Configuration dictionary
        experiment_finished_event:  Event to signal the end of the experiment
        x_plot:                     Array to store x data
        y_plot:                     Array to store y data
        t_plot:                     Array to store t data
        main_v_dc_plot:             Array to store main_v_dc data

    Returns:
        None

    """
    # On Windows ``multiprocessing`` uses spawn semantics, so the experiment
    # subprocess starts with no logging handlers attached. Re-initialise the
    # application logger here so that prints and uncaught exceptions in the
    # experiment process land in the same daily GUI log file as the main
    # GUI process. The function is idempotent and cheap.
    try:
        gui_log_root = getattr(variables, "log_path", "") or runtime.find_project_root()
        loggi.setup_application_logging(gui_log_root)
    except Exception as _exc:
        # Logging setup must never block the experiment from starting.
        print(f"[apt] Could not initialise application logging: {_exc}")

    apt_exp_control = APT_Exp_Control(variables, conf, experiment_finished_event, x_plot, y_plot, t_plot, main_v_dc_plot)

    apt_exp_control.run_experiment()
