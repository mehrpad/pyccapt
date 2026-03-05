import h5py


def hdf_creator(variables, conf, time_counter, time_ex):
	"""
	Save experiment data to an HDF5 file.

	Args:
		variables (object): An object containing experiment variables.
		conf (dict): A dictionary containing configuration settings.
		time_counter (list): A list of time counter data.
		time_ex (list): A list of timestamp of iteration.

	Returns:
		None
	"""

	path = variables.path + '\\%s.h5' % variables.exp_name
	# Save HDF5 file
	with h5py.File(path, "w") as f:
		f.create_dataset("apt/id", data=time_counter, dtype='u8')
		f.create_dataset("apt/num_events", data=variables.main_counter, dtype='u4')
		f.create_dataset("apt/num_raw_signals", data=variables.main_raw_counter, dtype='u4')
		f.create_dataset('apt/temperature', data=variables.main_temperature, dtype='f8')
		f.create_dataset('apt/experiment_chamber_vacuum', data=variables.main_chamber_vacuum, dtype='f8')
		f.create_dataset("apt/timestamps", data=time_ex, dtype='f8')

		if conf['tdc'] == "on" and conf['tdc_model'] == 'Surface_Consept' \
				and variables.counter_source == 'TDC':
			f.create_dataset("dld/x", data=variables.x, dtype='f8')
			f.create_dataset("dld/y", data=variables.y, dtype='f8')
			f.create_dataset("dld/t", data=variables.t, dtype='f8')
			f.create_dataset("dld/high_voltage", data=variables.main_v_dc_dld, dtype='f8')
			f.create_dataset("dld/voltage_pulse", data=variables.main_v_p_dld, dtype='f8')
			f.create_dataset("dld/laser_pulse", data=variables.main_l_p_dld, dtype='f8')
			f.create_dataset("dld/start_counter", data=variables.dld_start_counter, dtype='u8')

			# raw data
			f.create_dataset("tdc/start_counter", data=variables.tdc_start_counter, dtype='u8')
			f.create_dataset("tdc/channel", data=variables.channel, dtype='u4')
			f.create_dataset("tdc/time_data", data=variables.time_data, dtype='u8')
			f.create_dataset("tdc/high_voltage", data=variables.main_v_dc_tdc, dtype='f8')
			f.create_dataset("tdc/voltage_pulse", data=variables.main_v_p_tdc, dtype='f8')
			f.create_dataset("tdc/laser_pulse", data=variables.main_l_p_tdc, dtype='f8')

		elif conf['tdc'] == "on" and conf[
			'tdc_model'] == 'RoentDek' and variables.counter_source == 'TDC':
			f.create_dataset("dld/x", data=variables.x, dtype='f8')
			f.create_dataset("dld/y", data=variables.y, dtype='f8')
			f.create_dataset("dld/t", data=variables.t, dtype='f8')
			f.create_dataset("dld/high_voltage", data=variables.main_v_dc_dld, dtype='f8')
			f.create_dataset("dld/voltage_pulse", data=variables.main_v_p_dld, dtype='f8')
			f.create_dataset("dld/laser_pulse", data=variables.main_l_p_dld, dtype='f8')
			f.create_dataset("dld/start_counter", data=variables.time_stamp, dtype='u8')
			# raw data
			f.create_dataset("tdc/ch0", data=variables.ch0, dtype='u8')
			f.create_dataset("tdc/ch1", data=variables.ch1, dtype='u8')
			f.create_dataset("tdc/ch2", data=variables.ch2, dtype='u8')
			f.create_dataset("tdc/ch3", data=variables.ch3, dtype='u8')
			f.create_dataset("tdc/ch4", data=variables.ch4, dtype='u8')
			f.create_dataset("tdc/ch5", data=variables.ch5, dtype='u8')
			f.create_dataset("tdc/ch6", data=variables.ch6, dtype='u8')
			f.create_dataset("tdc/ch7", data=variables.ch6, dtype='u8')
			f.create_dataset("tdc/high_voltage", data=variables.main_v_dc_tdc, dtype='f8')
			f.create_dataset("tdc/voltage_pulse", data=variables.main_v_p_tdc, dtype='f8')
			f.create_dataset("tdc/laser_pulse", data=variables.main_l_p_tdc, dtype='f8')
		elif conf['tdc'] == "on" and conf['tdc_model'] == 'HSD' and variables.counter_source == 'HSD':
			f.create_dataset("hsd/ch0_time", data=variables.ch0_time, dtype='u8')
			f.create_dataset("hsd/ch0_wave", data=variables.ch0_wave, dtype='u8')
			f.create_dataset("hsd/ch1_time", data=variables.ch1_time, dtype='u8')
			f.create_dataset("hsd/ch1_wave", data=variables.ch1_wave, dtype='u8')
			f.create_dataset("hsd/ch2_time", data=variables.ch2_time, dtype='u8')
			f.create_dataset("hsd/ch2_wave", data=variables.ch2_wave, dtype='u8')
			f.create_dataset("hsd/ch3_time", data=variables.ch3_time, dtype='u8')
			f.create_dataset("hsd/ch3_wave", data=variables.ch3_wave, dtype='u8')
			f.create_dataset("hsd/ch4_time", data=variables.ch4_time, dtype='u8')
			f.create_dataset("hsd/ch4_wave", data=variables.ch4_wave, dtype='u8')
			f.create_dataset("hsd/ch5_time", data=variables.ch5_time, dtype='u8')
			f.create_dataset("hsd/ch5_wave", data=variables.ch5_wave, dtype='u8')
			f.create_dataset("hsd/high_voltage", data=variables.main_v_dc_drs, dtype='f8')
			f.create_dataset("hsd/voltage_pulse", data=variables.main_v_p_drs, dtype='f8')
			f.create_dataset("hsd/laser_pulse", data=variables.main_l_p_drs, dtype='f8')
