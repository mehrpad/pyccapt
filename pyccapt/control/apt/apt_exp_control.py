import copy
import datetime
import multiprocessing
import os
import time

import serial.tools.list_ports
from simple_pid import PID

from pyccapt.control.apt import apt_exp_control_func
from pyccapt.control.control import experiment_statistics, hdf5_creator, loggi
from pyccapt.control.devices import initialize_devices, signal_generator
from pyccapt.control.drs import drs
from pyccapt.control.tdc_roentdek import tdc_roentdek
from pyccapt.control.tdc_surface_concept import tdc_surface_consept


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
        self.sleep_time = 1 / self.variables.ex_freq

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

        self.initialization_error = False

    def initialize_detector_process(self):
        """
        Initialize the detector process based on the configured settings.

        This method initializes the necessary queues and processes for data acquisition based on the configured settings.

        Args:
           None

        Returns:
           None
        """
        if self.conf['tdc'] == "on" and self.conf['tdc_model'] == 'Surface_Consept' \
                and self.variables.counter_source == 'TDC':

            # Initialize and initiate a process(Refer to imported file 'tdc_new' for process function declaration )
            # Module used: multiprocessing
            self.stop_event = multiprocessing.Event()
            self.tdc_process = multiprocessing.Process(target=tdc_surface_consept.experiment_measure,
                                                       args=(self.variables, self.x_plot, self.y_plot, self.t_plot,
                                                             self.main_v_dc_plot, self.stop_event))

            self.tdc_process.start()

        elif self.conf['tdc'] == "on" and self.conf[
            'tdc_model'] == 'RoentDek' and self.variables.counter_source == 'TDC':

            self.tdc_process = multiprocessing.Process(target=tdc_roentdek.experiment_measure,
                                                       args=(self.variables,))
            self.tdc_process.start()

        elif self.conf['tdc'] == "on" and self.conf['tdc_model'] == 'HSD' and self.variables.counter_source == 'HSD':

            # Initialize and initiate a process(Refer to imported file 'drs' for process function declaration)
            # Module used: multiprocessing
            self.hsd_process = multiprocessing.Process(target=drs.experiment_measure,
                                                       args=(self.variables,))
            self.hsd_process.start()

        else:
            print("No counter source selected")

    def main_ex_loop(self, ):
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

        if self.control_algorithm == 'Proportional':
            error = self.detection_rate - self.variables.detection_rate_current
            # simple proportional control with averaging
            if error > 0.05:
                voltage_step = error * self.variables.vdc_step_up * 10
            elif error < -0.05:
                voltage_step = error * self.variables.vdc_step_down * 10
            else:
                voltage_step = 0

            if voltage_step > 40:
                print('voltage step is too high: %s' % voltage_step)
                voltage_step = 40
        elif self.control_algorithm == 'PID' or self.control_algorithm == 'PID aggressive':
            error = self.detection_rate - self.variables.detection_rate_current
            print('error: %s' % error)
            voltage_step = self.pid(error) * 1000
            print('voltage step: %s' % voltage_step)

        # update v_dc
        if not self.variables.vdc_hold and voltage_step != 0:
            specimen_voltage_temp = min(self.specimen_voltage + voltage_step, self.vdc_max)
            if specimen_voltage_temp > self.vdc_min:
                if specimen_voltage_temp != self.specimen_voltage:
                    if self.conf['v_dc'] != "off":
                        apt_exp_control_func.command_v_dc(self.com_port_v_dc, ">S0 %s" % specimen_voltage_temp)
                        self.specimen_voltage = specimen_voltage_temp
                        self.variables.specimen_voltage = self.specimen_voltage
                        self.variables.specimen_voltage_plot = self.specimen_voltage
                    if self.pulse_mode in ['Voltage', 'VoltageLaser']:
                        new_vp = (self.specimen_voltage * (self.pulse_fraction / 100) /
                                  self.pulse_amp_per_supply_voltage)
                        if self.pulse_voltage_max > new_vp > self.pulse_voltage_min and self.conf['v_p'] != "off":
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

        # if os.path.exists("./files/counter_experiments.txt"):
        #     # Read the experiment counter
        #     with open('./files/counter_experiments.txt') as f:
        #         self.variables.counter = int(f.readlines()[0])
        # else:
        #     # create a new txt file
        #     with open('./files/counter_experiments.txt', 'w') as f:
        #         f.write(str(1))  # Current time and date
        now = datetime.datetime.now()
        self.variables.exp_name = "%s_" % self.variables.counter + \
                                  now.strftime("%b-%d-%Y_%H-%M") + "_%s" % self.variables.electrode + "_%s" % \
                                  self.variables.hdf5_data_name
        p = os.path.abspath(os.path.join(__file__, "../../.."))
        self.variables.path = os.path.join(p, 'data\\%s' % self.variables.exp_name)
        self.variables.path_meta = self.variables.path + '\\meta_data\\'

        self.variables.log_path = self.variables.path_meta
        # Create folder to save the data
        if not os.path.isdir(self.variables.path):
            try:
                os.makedirs(self.variables.path, mode=0o777, exist_ok=True)
            except Exception as e:
                print('Can not create the directory for saving the data')
                print(e)
                self.variables.stop_flag = True
                self.initialization_error = True
        if not os.path.isdir(self.variables.path_meta):
            try:
                os.makedirs(self.variables.path_meta, mode=0o777, exist_ok=True)
            except Exception as e:
                print('Can not create the directory for saving the data')
                print(e)
                self.variables.stop_flag = True
                self.initialization_error = True

        if self.conf['tdc'] == 'on' and not self.initialization_error:
            self.variables.flag_tdc_failure = False
            self.initialize_detector_process()

        self.log_apt = loggi.logger_creator('apt', self.variables, 'apt.log', path=self.variables.log_path)
        if self.conf['signal_generator'] == 'on' and self.pulse_mode in ['Voltage',
                                                                         'VoltageLaser'] and not self.initialization_error:
            self.initialization_error = apt_exp_control_func.initialization_signal_generator(self.variables,
                                                                                             self.log_apt)
            if not self.initialization_error:
                self.initialization_signal_generator = True

        if self.conf['v_dc'] == 'on' and not self.initialization_error:
            try:
                self.com_port_v_dc = serial.Serial(
                    port=self.variables.COM_PORT_V_dc,
                    baudrate=115200,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE
                )
            except Exception as e:
                print('Can not open the COM port for V_dc')
                print(e)
                self.initialization_v_dc = True
            if not self.initialization_error:
                self.initialization_error = apt_exp_control_func.initialization_v_dc(self.com_port_v_dc, self.log_apt,
                                                                                     self.variables)
            if not self.initialization_error:
                self.initialization_v_dc = True

        if self.conf['v_p'] == 'on' and self.pulse_mode in ['Voltage', 'VoltageLaser']:
            # Initialize pulser
            try:
                self.com_port_v_p = serial.Serial(self.variables.COM_PORT_V_p, baudrate=115200, timeout=0.01)
            except Exception as e:
                print('Can not open the COM port for V_p')
                print(e)
                self.initialization_v_p = True
            if not self.initialization_error:
                self.initialization_error = apt_exp_control_func.initialization_v_p(self.com_port_v_p, self.log_apt,
                                                                                    self.variables)

            if not self.initialization_error:
                self.initialization_v_p = True
        elif self.conf['laser'] == 'on' and self.pulse_mode in ['Laser', 'VoltageLaser']:
            print(f"{initialize_devices.bcolors.WARNING}Warning: turn on the laser manually"
                  f"{initialize_devices.bcolors.ENDC}")

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
                if self.conf['v_p'] == "on":
                    apt_exp_control_func.command_v_p(self.com_port_v_p, 'OUTPut ON')
                    vol = self.variables.v_p_min / self.variables.pulse_amp_per_supply_voltage
                    cmd = 'VOLT %s' % vol
                    apt_exp_control_func.command_v_p(self.com_port_v_p, cmd)
                    time.sleep(0.1)
            elif self.pulse_mode in ['Laser', 'VoltageLaser']:
                if self.conf['laser'] == "on":
                    print(f"{initialize_devices.bcolors.WARNING}Warning: enable output of laser manually"
                          f"{initialize_devices.bcolors.ENDC}")
            if self.conf['v_dc'] == "on":
                apt_exp_control_func.command_v_dc(self.com_port_v_dc, "F1")
                time.sleep(0.1)

        self.pulse_fraction = self.variables.pulse_fraction
        self.pulse_amp_per_supply_voltage = self.variables.pulse_amp_per_supply_voltage
        self.specimen_voltage = self.variables.specimen_voltage
        if self.pulse_mode in ['Voltage', 'VoltageLaser']:
            self.pulse_voltage = self.variables.pulse_voltage

        if self.control_algorithm == 'PID' or self.control_algorithm == 'PID aggressive':
            self.pid = PID(1, 0.1, 0.05, setpoint=self.detection_rate)
            self.pid.sample_time = 1 / self.variables.ex_freq
            self.pid.output_limits = (0, 100)
            self.pid.proportional_on_measurement = True
            if self.control_algorithm == 'PID aggressive':
                self.pid.tunings = (5, 0.1, 0.05)

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
                    if self.pulse_mode in ['Voltage', 'VoltageLaser']:
                        signal_generator.change_frequency_signal_generator(self.variables, self.pulse_frequency / 1000)
                    elif self.pulse_mode in ['Laser', 'VoltageLaser']:
                        pass

                if self.detection_rate != self.variables.detection_rate:
                    self.detection_rate = self.variables.detection_rate
                    self.counts_target = self.pulse_frequency * self.detection_rate / 100
                    self.detection_rate = self.variables.detection_rate
                    if self.control_algorithm == 'PID' or self.control_algorithm == 'PID aggressive':
                        self.pid.setpoint = self.detection_rate

                self.total_ions = self.variables.total_ions
                self.total_raw_signals = self.variables.total_raw_signals
                # here we check if tdc is failed or not by checking if the total number of ions is
                # constant for 100 iteration
                if total_ions_tmp == self.total_ions and not self.variables.vdc_hold:
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
                        if not self.initialization_v_p:
                            try:
                                # Initialize pulser
                                self.com_port_v_p = serial.Serial(self.variables.COM_PORT_V_p, baudrate=115200,
                                                                  timeout=0.01)
                                self.initialization_error = apt_exp_control_func.initialization_v_p(self.com_port_v_p,
                                                                                                    self.log_apt,
                                                                                                    self.variables)
                                self.initialization_v_p = True
                                apt_exp_control_func.command_v_p(self.com_port_v_p, 'OUTPut ON')
                            except Exception as e:
                                print('Can not open the COM port for V_p')
                                print(e)
                        # if the pulse mode is changed from voltage to laser, we need to turn on the signal generator
                        if not self.initialization_signal_generator:
                            self.initialization_error = apt_exp_control_func.initialization_signal_generator(
                                self.variables,
                                self.log_apt)
                            if not self.initialization_error:
                                self.initialization_signal_generator = True
                        # set the v_dc and v_p
                        self.pulse_voltage_min = self.variables.v_p_min / self.pulse_amp_per_supply_voltage
                        self.pulse_voltage_max = self.variables.v_p_max / self.pulse_amp_per_supply_voltage
                        start_vp = (self.specimen_voltage * (self.pulse_fraction / 100) /
                                    self.pulse_amp_per_supply_voltage)
                        if start_vp < self.pulse_voltage_min:
                            start_vp = self.variables.v_p_min / self.variables.pulse_amp_per_supply_voltage

                        if self.pulse_voltage_max > start_vp > self.pulse_voltage_min - 1 and self.conf[
                            'v_p'] != "off":
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
                                if self.conf['v_dc'] != "off":
                                    apt_exp_control_func.command_v_dc(self.com_port_v_dc,
                                                                      ">S0 %s" % self.specimen_voltage)
                                time.sleep(0.3)
                            if self.conf['v_dc'] != "off" and self.pulse_mode in ['Voltage', 'VoltageLaser']:
                                new_vp = (self.specimen_voltage * (self.pulse_fraction / 100) /
                                          self.pulse_amp_per_supply_voltage)
                                if self.pulse_voltage_max > new_vp > self.pulse_voltage_min and self.conf[
                                    'v_p'] != "off":
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
                current_time_with_microseconds = current_time.strftime(
                    "%Y-%m-%d %H:%M:%S.%f")  # Format with microseconds
                current_time_unix = datetime.datetime.strptime(current_time_with_microseconds,
                                                               "%Y-%m-%d %H:%M:%S.%f").timestamp()
                time_ex.append(current_time_unix)

                if self.variables.stop_flag:
                    self.log_apt.info('Experiment is stopped')
                    if self.conf['tdc'] != "off":
                        if self.variables.counter_source == 'TDC':
                            self.variables.flag_stop_tdc = True
                            self.stop_event.set()  # Signal the tdc to stop
                    time.sleep(1)
                    break

                if self.variables.flag_tdc_failure:
                    self.log_apt.info('Experiment is stopped because of tdc failure')
                    print(f"{initialize_devices.bcolors.FAIL}Experiment is stopped because of TDC failure")
                    print(f"{initialize_devices.bcolors.FAIL}Restart the TDC and start the experiment again")
                    if self.conf['tdc'] == "on":
                        if self.variables.counter_source == 'TDC':
                            self.variables.stop_flag = True  # Set the STOP flag
                            self.stop_event.set()  # Signal the tdc to stop
                    time.sleep(1)
                    break

                if self.variables.criteria_ions:
                    if self.variables.max_ions <= self.total_ions:
                        self.log_apt.info('Experiment is stopped because total number of ions is achieved')
                        if self.conf['tdc'] == "on":
                            if self.variables.counter_source == 'TDC':
                                self.variables.flag_stop_tdc = True
                                self.variables.stop_flag = True  # Set the STOP flag
                                self.stop_event.set()  # Signal the tdc to stop
                        time.sleep(1)
                        break
                if self.variables.criteria_vdc:
                    if self.vdc_max <= self.specimen_voltage:
                        if flag_achieved_high_voltage > self.ex_freq * 10:
                            self.log_apt.info('Experiment is stopped because dc voltage Max. is achieved')
                            if self.conf['tdc'] != "off":
                                if self.variables.counter_source == 'TDC':
                                    self.variables.flag_stop_tdc = True
                                    self.variables.stop_flag = True  # Set the STOP flag
                                    self.stop_event.set()  # Signal the tdc to stop
                            time.sleep(1)
                            break
                        flag_achieved_high_voltage += 1

                if self.variables.criteria_time:
                    if self.variables.elapsed_time >= self.variables.ex_time:
                        self.log_apt.info('Experiment is stopped because experiment time Max. is achieved')
                        if self.conf['tdc'] == "on":
                            if self.variables.counter_source == 'TDC':
                                self.variables.flag_stop_tdc = True
                                self.variables.stop_flag = True
                                self.stop_event.set()  # Signal the tdc to stop

                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                remaining_time = desired_period - elapsed_time

                if remaining_time > 0:
                    self.precise_sleep(remaining_time)
                elif remaining_time < 0:
                    index_time += 1
                    remaining_time_list.append(elapsed_time)

                steps += 1

        self.variables.start_flag = False  # Set the START flag
        time.sleep(1)

        self.log_apt.info('Experiment is finished')
        print("Experiment process: Experiment loop took longer than %s Millisecond for %s times out of %s "
              "iteration" % (int(1000 / self.variables.ex_freq), index_time, steps))
        self.log_apt.warning(
            'Experiment loop took longer than %s (ms) for %s times out of %s iteration.'
            % (int(1000 / self.variables.ex_freq), index_time, steps))

        print('Waiting for TDC process to be finished for maximum 60 seconds...')
        for i in range(900):
            if self.variables.flag_finished_tdc:
                print('TDC process is finished')
                break
            print('%s seconds passed' % i)
            time.sleep(1)
            if i == 599:
                print('TDC process is not finished')
                self.log_apt.warning('TDC process is not finished after 15 minutes')

        if self.conf['tdc'] == "on":
            # Stop the TDC process
            try:
                if self.variables.counter_source == 'TDC':
                    self.tdc_process.join(2)
                    if self.tdc_process.is_alive():
                        self.tdc_process.join(1)
                elif self.variables.counter_source == 'HSD':
                    self.hsd_process.join(2)
                    if self.hsd_process.is_alive():
                        self.hsd_process.join(1)

            except Exception as e:
                print(
                    f"{initialize_devices.bcolors.WARNING}Warning: The TDC or HSD process cannot be terminated "
                    f"properly{initialize_devices.bcolors.ENDC}")
                print(e)

        self.variables.extend_to('main_counter', self.main_counter)
        self.variables.extend_to('main_raw_counter', self.main_raw_counter)
        self.variables.extend_to('main_temperature', self.main_temperature)
        self.variables.extend_to('main_chamber_vacuum', self.main_chamber_vacuum)

        if self.conf['tdc'] == "off":
            if self.variables.counter_source == 'TDC':
                self.variables.total_ions = len(self.variables.x)
        elif self.variables.counter_source == 'HSD':
            pass

        # This flag set to True to save the last screenshot of the experiment in the GUI visualization
        self.variables.last_screen_shot = True
        # Check the length of arrays to be equal
        if self.variables.counter_source == 'TDC':
            if all(len(lst) == len(self.variables.x) for lst in [self.variables.x, self.variables.y,
                                                                 self.variables.t, self.variables.dld_start_counter,
                                                                 self.variables.main_v_dc_dld,
                                                                 self.variables.main_v_p_dld,
                                                                 self.variables.main_l_p_dld]):
                self.log_apt.warning('dld data have not same length')

            if all(len(lst) == len(self.variables.channel) for lst in [self.variables.channel, self.variables.time_data,
                                                                       self.variables.tdc_start_counter,
                                                                       self.variables.main_v_dc_tdc,
                                                                       self.variables.main_v_p_tdc,
                                                                       self.variables.main_l_p_tdc]):
                self.log_apt.warning('tdc data have not same length')
        elif self.variables.counter_source == 'DRS':
            if all(len(lst) == len(self.variables.ch0_time) for lst in
                   [self.variables.ch0_wave, self.variables.ch1_time,
                    self.variables.ch1_wave, self.variables.ch2_time,
                    self.variables.ch2_wave, self.variables.ch3_time,
                    self.variables.ch3_wave,
                    self.variables.main_v_dc_drs, self.variables.main_v_p_drs, self.variables.main_l_p_drs]):
                self.log_apt.warning('tdc data have not same length')

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
            apt_exp_control_func.send_info_email(self.log_apt, self.variables)

        # Save new value of experiment counter
        if os.path.exists("./files/counter_experiments.txt"):
            self.variables.counter += 1
            with open('./files/counter_experiments.txt', 'w') as f:
                f.write(str(self.variables.counter))
                self.log_apt.info('Experiment counter is increased')

        self.experiment_finished_event.set()
        # Clear up all the variables and deinitialize devices
        self.clear_up()
        self.log_apt.info('Variables and devices are cleared and deinitialized')
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

        def cleanup_variables():
            """
            Reset all the global variables.
            """
            self.variables.flag_finished_tdc = False
            self.variables.detection_rate_current = 0.0
            self.variables.count = 0
            self.variables.index_plot = 0
            self.variables.index_save_image = 0
            self.variables.index_wait_on_plot_start = 0
            self.variables.index_plot_save = 0
            self.variables.index_plot = 0
            self.variables.specimen_voltage = 0
            self.variables.specimen_voltage_plot = 0
            self.variables.pulse_voltage = 0

            while not self.x_plot.empty() or not self.y_plot.empty() or not self.t_plot.empty() or \
                    not self.main_v_dc_plot.empty():
                dumy = self.x_plot.get()
                dumy = self.y_plot.get()
                dumy = self.t_plot.get()
                dumy = self.main_v_dc_plot.get()

            self.variables.clear_to('x')
            self.variables.clear_to('y')
            self.variables.clear_to('t')

            self.variables.clear_to('channel')
            self.variables.clear_to('time_data')
            self.variables.clear_to('tdc_start_counter')
            self.variables.clear_to('dld_start_counter')

            self.variables.clear_to('time_stamp')
            self.variables.clear_to('ch0')
            self.variables.clear_to('ch1')
            self.variables.clear_to('ch2')
            self.variables.clear_to('ch3')
            self.variables.clear_to('ch4')
            self.variables.clear_to('ch5')
            self.variables.clear_to('ch6')
            self.variables.clear_to('ch7')
            self.variables.clear_to('laser_intensity')

            self.variables.clear_to('ch0_time')
            self.variables.clear_to('ch0_wave')
            self.variables.clear_to('ch1_time')
            self.variables.clear_to('ch1_wave')
            self.variables.clear_to('ch2_time')
            self.variables.clear_to('ch2_wave')
            self.variables.clear_to('ch3_time')
            self.variables.clear_to('ch3_wave')

            self.variables.clear_to('main_v_p')
            self.variables.clear_to('main_counter')
            self.variables.clear_to('main_raw_counter')
            self.variables.clear_to('main_temperature')
            self.variables.clear_to('main_chamber_vacuum')
            self.variables.clear_to('main_v_dc_dld')
            self.variables.clear_to('main_v_p_dld')
            self.variables.clear_to('main_l_p_dld')
            self.variables.clear_to('main_v_dc_tdc')
            self.variables.clear_to('main_v_p_tdc')
            self.variables.clear_to('main_l_p_tdc')
            self.variables.clear_to('main_v_dc_drs')
            self.variables.clear_to('main_v_p_drs')
            self.variables.clear_to('main_l_p_drs')

        self.log_apt.info('Starting cleanup')

        try:
            if self.conf['v_dc'] == "on" and self.initialization_v_dc:
                # Turn off the v_dc
                apt_exp_control_func.command_v_dc(self.com_port_v_dc, 'F0')
                self.com_port_v_dc.close()
        except Exception as e:
            print(e)

        try:
            if self.conf['v_p'] == "on" and self.initialization_v_p:
                # Turn off the v_p
                apt_exp_control_func.command_v_p(self.com_port_v_p, 'VOLT 0')
                apt_exp_control_func.command_v_p(self.com_port_v_p, 'OUTPut OFF')
                self.com_port_v_p.close()
        except Exception as e:
            print(e)

        try:
            if self.conf['signal_generator'] != "off" and self.initialization_signal_generator:
                # Turn off the signal generator
                signal_generator.turn_off_signal_generator()

        except Exception as e:
            print(e)

        # Reset variables
        cleanup_variables()
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

    apt_exp_control = APT_Exp_Control(variables, conf, experiment_finished_event, x_plot, y_plot, t_plot,
                                      main_v_dc_plot)

    apt_exp_control.run_experiment()
