import ctypes
from pathlib import Path

import numpy as np
from numpy.ctypeslib import ndpointer


class TDC:
	"""
	This class sets up the parameters for the TDC and allows users to read experiment TDC values.
	"""

	def __init__(self, tdc_lib, buf_size=30000, time_out=300):
		"""
		Constructor function which initializes function parameters.

		Args:
			tdc_lib (ctypes.CDLL): The TDC library.
			buf_size (int): Buffer size.
			time_out (int): Timeout value.
		"""
		self.tdc_lib = tdc_lib
		self.buf_size = buf_size
		self.time_out = time_out
		tdc_lib.Warraper_tdc_new.restype = ctypes.c_void_p
		tdc_lib.Warraper_tdc_new.argtypes = [ctypes.c_int, ctypes.c_int]
		tdc_lib.init_tdc.argtypes = [ctypes.c_void_p]
		tdc_lib.init_tdc.restype = ctypes.c_int
		tdc_lib.run_tdc.restype = ctypes.c_int
		tdc_lib.run_tdc.argtypes = [ctypes.c_void_p]
		tdc_lib.stop_tdc.restype = ctypes.c_int
		tdc_lib.stop_tdc.argtypes = [ctypes.c_void_p]
		tdc_lib.get_data_tdc_buf.restype = ndpointer(dtype=ctypes.c_double, shape=(12 * self.buf_size + 1,))
		tdc_lib.get_data_tdc_buf.argtypes = [ctypes.c_void_p]

		self.obj = tdc_lib.Warraper_tdc_new(self.buf_size, self.time_out)

	def stop_tdc(self):
		"""
		Stop the TDC.

		Returns:
			int: Return code.
		"""
		return self.tdc_lib.stop_tdc(self.obj)

	def init_tdc(self):
		"""
		Initialize the TDC.

		Returns:
			int: Return code.
		"""
		return self.tdc_lib.init_tdc(self.obj)

	def run_tdc(self):
		"""
		Run the TDC.
		"""
		self.tdc_lib.run_tdc(self.obj)

	def get_data_tdc_buf(self):
		"""
		Get data from the TDC buffer.

		Returns:
			np.ndarray: Data from the TDC buffer.
		"""
		data = self.tdc_lib.get_data_tdc_buf(self.obj)
		return data

def experiment_measure(variables):
	"""
	Measurement function: This function is called in a process to read data from the queue.

	Args:
		variables: Variables object

	Returns:
		int: Return code.
	"""
	try:
		# Load the library
		module_dir = Path(__file__).resolve().parent
		tdc_lib = ctypes.CDLL(str(module_dir / "wrapper_read_TDC8HP_x64.dll"))
	except Exception as exc:
		print("TDC DLL was not found")
		print(exc)
		return 1

	tdc = TDC(tdc_lib, buf_size=30000, time_out=100)

	ret_code = tdc.init_tdc()

	tdc.run_tdc()

	while True:
		returnVale = tdc.get_data_tdc_buf()
		buffer_length = int(returnVale[0])
		returnVale_tmp = np.copy(returnVale[1:buffer_length * 12 + 1].reshape(buffer_length, 12))

		xx = returnVale_tmp[:, 8]
		yy = returnVale_tmp[:, 9]
		tt = returnVale_tmp[:, 10]
		variables.extend_to('x', xx.tolist())
		variables.extend_to('y', yy.tolist())
		variables.extend_to('t', tt.tolist())
		variables.extend_to('time_stamp', returnVale_tmp[:, 11].tolist())
		variables.extend_to('ch0', returnVale_tmp[:, 0].tolist())
		variables.extend_to('ch1', returnVale_tmp[:, 1].tolist())
		variables.extend_to('ch2', returnVale_tmp[:, 2].tolist())
		variables.extend_to('ch3', returnVale_tmp[:, 3].tolist())
		variables.extend_to('ch4', returnVale_tmp[:, 4].tolist())
		variables.extend_to('ch5', returnVale_tmp[:, 5].tolist())
		variables.extend_to('ch6', returnVale_tmp[:, 6].tolist())
		variables.extend_to('ch7', returnVale_tmp[:, 7].tolist())
		main_v_dc_dld_list = np.tile(variables.specimen_voltage, len(xx))
		pulse_data = np.tile(variables.pulse_voltage, len(xx))
		variables.extend_to('main_v_dc_tdc', main_v_dc_dld_list.tolist())
		variables.extend_to('main_p_tdc_roentdek', pulse_data.tolist())

		# with self.variables.lock_data_plot:
		variables.extend_to('main_v_dc_plot', main_v_dc_dld_list.tolist())
		variables.extend_to('x_plot', xx.tolist())
		variables.extend_to('y_plot', yy.tolist())
		variables.extend_to('t_plot', tt.tolist())

		if variables.flag_stop_tdc:
			break

	tdc.stop_tdc()

	return 0
