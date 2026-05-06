"""Driver for Edwards TIC / Active Gauge Controller (AGC) vacuum gauges.

Vendor / origin
---------------
The serial command set used here (``!C910``, ``!C904``, ``?V911``,
``?V940``, etc.) is defined and owned by **Edwards Vacuum** for their
TIC / AGC vacuum-controller family. This file is a thin Python wrapper
around that ASCII serial protocol; the protocol itself is Edwards'
property. Refer to Edwards' Active Gauge Controller manual for the
authoritative command reference.
"""

import serial


class EdwardsAGC(object):
	"""
	Primitive driver for Edwards Active Gauge Controller.
	Complete manual found at
	http://www.idealvac.com/files/brochures/Edwards_AGC_D386-52-880_IssueM.pdf
	"""

	def __init__(self, port, variables):
		"""
		The constructor function to initialize serial lib parameters.

		Args:
			port (str): Port on which serial communication is established.
			variables: Variable container for shared variables.

		Returns:
			None
		"""
		self.port = port
		self.variables = variables
		self.serial = serial.Serial(self.port, baudrate=9600, timeout=0.5)

	def comm(self, command):
		"""
		This class method implements serial communication using the serial library.
		Reads the raw data through the serial line and returns it.

		Args:
			command (str): Command to be written on the serial line.

		Returns:
			str: String read through the serial.
		"""
		comm = command + "\r\n"

		self.serial.write(comm.encode())
		complete_string = self.serial.readline().decode()
		complete_string = complete_string.strip()

		return complete_string
