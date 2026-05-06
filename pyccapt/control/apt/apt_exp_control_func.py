import serial.tools.list_ports

from pyccapt.control.devices import signal_generator, email_send


def initialization_signal_generator(variables, log_apt):
	"""
	Initialize the signal generator.

	Args:
		signal_generator: The class object of the SignalGenerator class.
		variables: The class object of the Variables class.
		log_apt: The logger object.

	Returns:
		initialization_error: The boolean flag to indicate if the initialization is successful.
	"""
	# Initialize the signal generator
	try:
		signal_generator.initialize_signal_generator(variables, variables.pulse_frequency)
		log_apt.info('Signal generator is initialized')
		initialization_error = False
	except Exception as e:
		log_apt.info('Signal generator is not initialized')
		print('Can not initialize the signal generator')
		print('Make the signal_generator off in the config file or fix the error below')
		print(e)
		variables.stop_flag = True
		initialization_error = True
		log_apt.info('Experiment is terminated')
	return initialization_error


def command_v_p(com_port_v_p, cmd):
	"""
	Send commands to the pulser.

	This method sends commands to the pulser over the COM port and reads the response.

	Args:
		com_port_v_p (serial.Serial): The COM port object for the pulser.
		cmd (str): The command to send.

	Returns:
		str: The response received from the device.
	"""
	if cmd == 'close':
		com_port_v_p.close()
	else:
		cmd = cmd + '\r\n'
		com_port_v_p.write(cmd.encode())
	# response = self.com_port_v_p.readline().decode().strip()
	# return response


def command_v_dc(com_port_v_dc, cmd):
	"""
	Send commands to the high voltage parameter: v_dc.

	This method sends commands to the V_dc source over the COM port and reads the response.

	Args:
		com_port_v_dc (serial.Serial): The COM port object for the V_dc source.
		cmd (str): The command to send.

	Returns:
		str: The response received from the device.
	"""

	if cmd == 'close':
		com_port_v_dc.close()
	else:
		com_port_v_dc.write((cmd + '\r\n').encode())
	# response = ''
	# try:
	#     while self.com_port_v_dc.in_waiting > 0:
	#         response = self.com_port_v_dc.readline()
	# except Exception as error:
	#     print(error)
	#
	# if isinstance(response, bytes):
	#     response = response.decode("utf-8")

	# return response


def initialization_v_dc(com_port_v_dc, log_apt, variables):
	"""
	Initialize the high voltage.

	Args:
		com_port_v_dc: The COM port object for the high voltage.
		log_apt: The logger object.
		variables: The class object of the Variables class.

	Returns:
		initialization_error: The boolean flag to indicate if the initialization is successful.
	"""

	try:
		# Initialize high voltage
		if com_port_v_dc.is_open:
			com_port_v_dc.flushInput()
			com_port_v_dc.flushOutput()

			cmd_list = [">S1 3.0e-4", ">S0B 0", ">S0 %s" % variables.vdc_min, "F0", ">S0?", ">DON?", ">S0A?"]
			for cmd in range(len(cmd_list)):
				command_v_dc(com_port_v_dc, cmd_list[cmd])
		else:
			print("Couldn't open Port!")
			exit()
		log_apt.info('High voltage is initialized')
		initialization_error = False
	except Exception as e:
		log_apt.info('High voltage is  not initialized')
		print('Can not initialize the high voltage')
		print('Make the v_dc off in the config file or fix the error below')
		print(e)
		variables.stop_flag = True
		initialization_error = True
		log_apt.info('Experiment is terminated')
	return initialization_error


def initialization_v_p(com_port_v_p, log_apt, variables):
	"""
	Initialize the pulser.

	Args:
		com_port_v_p: The COM port object for the pulser.
		log_apt: The logger object.
		variables: The class object of the Variables class.

	Return:
		initialization_error: The boolean flag to indicate if the initialization is successful.
	"""

	try:

		command_v_p(com_port_v_p, '*RST')
		log_apt.info('Pulser is initialized')
		initialization_error = False
	except Exception as e:
		log_apt.info('Pulser is not initialized')
		print('Can not initialize the pulser')
		print('Make the v_p off in the config file or fix the error below')
		print(e)
		variables.stop_flag = True
		initialization_error = True
		log_apt.info('Experiment is terminated')
	return initialization_error


def send_info_email(log_apt, variables):
	"""
	Send the information email.

	Args:
		log_apt: The logger object.
		variables: The class object of the Variables class.

	Returns:
		None
	"""
	subject = 'Experiment {} Report on {}'.format(variables.hdf5_data_name, variables.start_time)
	elapsed_time_temp = float("{:.3f}".format(variables.elapsed_time))
	message = 'The experiment was started at: {}\n' \
	          'The experiment was ended at: {}\n' \
	          'Experiment duration: {}\n' \
	          'Total number of ions: {}\n\n'.format(variables.start_time,
	                                                variables.end_time, elapsed_time_temp,
	                                                variables.total_ions)

	additional_info = 'Username: {}\n'.format(variables.user_name)
	additional_info += 'Experiment Name: {}\n'.format(variables.ex_name)
	additional_info += 'Electrode Name: {}\n'.format(variables.electrode)
	additional_info += 'Experiment number: {}\n'.format(variables.counter)
	additional_info += 'Detection Rate (%): {}\n'.format(variables.detection_rate)
	additional_info += 'Maximum Number of Ions: {}\n'.format(variables.max_ions)
	additional_info += 'Counter source: {}\n'.format(variables.counter_source)
	additional_info += 'Pulse Fraction (%): {}\n'.format(variables.pulse_fraction)
	additional_info += 'Pulse Frequency (kHz): {}\n'.format(variables.pulse_frequency)
	# variables.laser_freq is the BASE rep-rate in Hz; the OUTPUT rate at
	# the sample is base / division. Show both in kHz so the recipient does
	# not need to do the conversion.
	try:
		_base_khz = float(variables.laser_freq or 0) / 1000.0
		_div = max(int(getattr(variables, 'laser_division_factor', 1) or 1), 1)
		_output_khz = _base_khz / _div
		additional_info += 'Laser base pulse frequency (kHz): {:.3f}\n'.format(_base_khz)
		additional_info += 'Laser division factor: {}\n'.format(_div)
		additional_info += 'Laser output pulse frequency (kHz): {:.3f}\n'.format(_output_khz)
		_pulse_nJ = float(getattr(variables, 'laser_pulse_energy', 0) or 0)
		additional_info += 'Laser pulse energy (nJ): {:.3f}\n'.format(_pulse_nJ)
	except (TypeError, ValueError):
		pass
	additional_info += 'Control Algorithm: {}\n'.format(variables.control_algorithm)
	additional_info += 'pulse_mode: {}\n'.format(variables.pulse_mode)
	additional_info += 'Experiment Control Refresh freq. (Hz): {}\n'.format(variables.ex_freq)
	additional_info += 'K_p Upwards: {}\n'.format(variables.vdc_step_up)
	additional_info += 'K_p Downwards: {}\n'.format(variables.vdc_step_down)
	additional_info += 'Specimen start Voltage (V): {}\n'.format(variables.vdc_min)
	additional_info += 'Specimen Stop Voltage (V): {}\n'.format(variables.vdc_max)
	additional_info += 'Temperature (k): {}\n'.format(variables.temperature)
	additional_info += 'Vacuum (mbar): {}\n'.format(variables.vacuum_main)

	if variables.pulse_mode == 'Voltage':
		additional_info += 'Pulse start Voltage (V): {}\n'.format(variables.v_p_min)
		additional_info += 'Pulse Stop Voltage (V): {}\n'.format(variables.v_p_max)
		additional_info += 'Specimen Max Achieved Pulse Voltage (V): {:.3f}\n\n'.format(
			variables.pulse_voltage)
	elif variables.pulse_mode == 'Laser':
		additional_info += 'Specimen Laser Pulsed Energy (pJ): {:.3f}\n\n'.format(
			variables.laser_intensity)
		additional_info += 'Specimen Max Laser Power (W): {:.3f}\n\n'.format(
			variables.max_laser_power)
	additional_info += 'StopCriteria:\n'
	additional_info += 'Criteria Time:: {}\n'.format(variables.criteria_time)
	additional_info += 'Criteria DC Voltage:: {}\n'.format(variables.criteria_vdc)
	additional_info += 'Criteria Ions:: {}\n'.format(variables.criteria_ions)

	additional_info += 'Specimen Max Achieved dc Voltage (V): {:.3f}\n'.format(variables.specimen_voltage)
	additional_info += 'Experiment Elapsed Time (Sec): {:.3f}\n'.format(variables.elapsed_time)
	additional_info += 'Experiment Total Ions: {}\n\n'.format(variables.total_ions)

	additional_info += 'Email: {}\n'.format(variables.email)

	additional_info += 'The experiment was conducted using PyCCAPT Python package.'

	message += additional_info
	# Pass variables through so the email module can attach apt.log and
	# parameters.txt from this experiment's folder. Any failure (missing
	# credentials, SMTP error, attachment IO error) raises and is caught
	# by the caller's try/except in apt_exp_control.run_experiment.
	attached = email_send.send_email(variables.email, subject, message, variables=variables)
	if attached:
		log_apt.info('Email is sent (attachments: %s)', ', '.join(attached))
	else:
		log_apt.info('Email is sent (no attachments)')
