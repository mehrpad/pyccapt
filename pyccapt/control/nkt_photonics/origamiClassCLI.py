from time import sleep

import serial


# Control static object for Origami using CLI
# Version 1.1

# Author Ian Baker


class origClass:
	## Control methods

	def __init__(self, comPort):
		self.comPort = comPort
		self.ser = None
		self.last_error = None

	def open_port(self):
		try:
			self.ser = serial.Serial(
				port=self.comPort,
				baudrate=38400,
				stopbits=serial.STOPBITS_ONE,
				bytesize=serial.EIGHTBITS,
				rtscts=False)
			self.last_error = None
			return 0
		except Exception as e:
			self.last_error = e
			return -1

	def close_port(self):
		self.ser.close()

	def Listen(self):
		cmd = "ly_oxp2_listen\n"
		self.ser.write(cmd.encode())
		sleep(0.1)
		dataStore = []
		while self.ser.in_waiting:
			dataBack = self.ser.readline()
			dataStore.append(dataBack.decode())
		sleep(0.1)
		return dataBack.decode()

	def Standby(self):
		cmd = "ly_oxp2_standby\n"
		self.ser.write(cmd.encode())
		sleep(0.1)
		dataStore = []
		while self.ser.in_waiting:
			dataBack = self.ser.readline()
			dataStore.append(dataBack.decode())
		sleep(0.1)
		return dataBack.decode()

	def Enable(self):
		cmd = "ly_oxp2_enabled\n"
		self.ser.write(cmd.encode())
		sleep(0.1)
		dataStore = []
		while self.ser.in_waiting:
			dataBack = self.ser.readline()
			dataStore.append(dataBack.decode())
		sleep(0.1)
		return dataBack.decode()

	def Temp(self, comPort):  # Displays the laser system temperatures (needs a read back)
		# Open the port

		cmd = "ly_oxp2_temp_status\n"

		self.ser.write(cmd.encode())
		sleep(0.1)
		dataStore = []
		while self.ser.in_waiting:
			dataBack = self.ser.readline()
			dataStore.append(dataBack.decode())
		return dataBack.decode()

	def Power(self, power):
		cmd = "ly_oxp2_power=" + str(power) + "\n"
		self.ser.write(cmd.encode())
		sleep(0.1)
		dataStore = []
		while self.ser.in_waiting:
			dataBack = self.ser.readline()
			dataStore.append(dataBack.decode())
		sleep(0.1)
		return dataBack.decode()

	def PowerRead(self):
		cmd = "ly_oxp2_power?\n"
		self.ser.write(cmd.encode())
		sleep(0.1)
		dataStore = []
		while self.ser.in_waiting:
			dataBack = self.ser.readline()
			dataStore.append(dataBack.decode())
		sleep(0.1)
		return dataBack.decode()

	def AOM(self, power):
		cmd = "e_power=" + str(power) + "\n"
		self.ser.write(cmd.encode())
		sleep(0.1)
		dataStore = []
		while self.ser.in_waiting:
			dataBack = self.ser.readline()
			dataStore.append(dataBack.decode())
		sleep(0.1)
		return dataBack.decode()

	def AOMRead(self):  # Displays the e_power setting

		cmd = "e_power?\n"
		self.ser.write(cmd.encode())
		sleep(0.1)
		dataStore = []
		while self.ser.in_waiting:
			dataBack = self.ser.readline()
			dataStore.append(dataBack.decode())
		sleep(0.1)
		return dataBack.decode()

	def Freq(self, freq):
		cmd = "e_freq=" + str(freq) + "\n"
		self.ser.write(cmd.encode())
		sleep(0.1)
		dataStore = []
		while self.ser.in_waiting:
			dataBack = self.ser.readline()
			dataStore.append(dataBack.decode())
		sleep(0.1)
		return dataBack.decode()

	def FreqRead(self):
		cmd = "e_freq?\n"
		self.ser.write(cmd.encode())
		sleep(0.1)
		dataStore = []
		while self.ser.in_waiting:
			dataBack = self.ser.readline()
			dataStore.append(dataBack.decode())
		sleep(0.1)
		return dataBack.decode()

	def Div(self, division):
		cmd = "e_div=" + str(division) + "\n"
		self.ser.write(cmd.encode())
		sleep(0.1)
		dataStore = []
		while self.ser.in_waiting:
			dataBack = self.ser.readline()
			dataStore.append(dataBack.decode())
		sleep(0.1)
		return dataBack.decode()

	def DivRead(self):
		cmd = "e_div?\r\n"
		self.ser.write(cmd.encode())
		sleep(0.1)
		dataStore = []
		while self.ser.in_waiting:
			dataBack = self.ser.readline()
			dataStore.append(dataBack.decode())
		sleep(0.1)
		return dataBack.decode()

	def Mode(self, mode):
		cmd = "e_mode " + str(mode) + "\n"
		self.ser.write(cmd.encode())
		sleep(0.1)
		dataStore = []
		while self.ser.in_waiting:
			dataBack = self.ser.readline()
			dataStore.append(dataBack.decode())
		sleep(0.1)
		return dataBack.decode()

	def ModeRead(self):
		cmd = "e_mode?\n"
		self.ser.write(cmd.encode())
		sleep(0.1)
		dataStore = []
		while self.ser.in_waiting:
			dataBack = self.ser.readline()
			dataStore.append(dataBack.decode())
		sleep(0.1)
		return dataBack.decode()

	def StatusRead(self):
		cmd = "ly_oxp2_dev_status\n"
		self.ser.write(cmd.encode())
		sleep(0.1)
		dataStore = []
		while self.ser.in_waiting:
			dataBack = self.ser.readline()
			dataStore.append(dataBack.decode())
		sleep(0.1)
		return dataBack.decode()

	def StatusMode(self):
		cmd = "ly_oxp2_mode\n"
		self.ser.write(cmd.encode())
		sleep(0.1)
		dataStore = []
		while self.ser.in_waiting:
			dataBack = self.ser.readline()
			dataStore.append(dataBack.decode())
		sleep(0.1)
		return dataBack.decode()

	def ServiceMode(self):
		cmd = "ly_oxp2_service_mode\n"
		self.ser.write(cmd.encode())
		sleep(0.1)
		dataStore = []
		while self.ser.in_waiting:
			dataBack = self.ser.readline()
			dataStore.append(dataBack.decode())
		sleep(0.1)
		return dataBack.decode()

	def DigitalGateLogic(self, choice):
		cmd = "ly_oxp2_digiop=" + str(choice) + "\n"
		self.ser.write(cmd.encode())
		sleep(0.1)
		dataStore = []
		while self.ser.in_waiting:
			dataBack = self.ser.readline()
			dataStore.append(dataBack.decode())
		sleep(0.1)
		return dataBack.decode()

	def DigitalGateLogicRead(self):
		cmd = "ly_oxp2_digiop?\n"
		self.ser.write(cmd.encode())
		sleep(0.1)
		dataStore = []
		while self.ser.in_waiting:
			dataBack = self.ser.readline()
			dataStore.append(dataBack.decode())
		sleep(0.1)
		return dataBack.decode()

	def AOMEnable(self):
		cmd = "ly_oxp2_output_enable\n"
		self.ser.write(cmd.encode())
		sleep(0.1)
		dataStore = []
		while self.ser.in_waiting:
			dataBack = self.ser.readline()
			dataStore.append(dataBack.decode())
		sleep(0.1)
		return dataBack.decode()

	def AOMDisable(self):
		cmd = "ly_oxp2_output_disable\n"
		self.ser.write(cmd.encode())
		sleep(0.1)
		dataStore = []
		while self.ser.in_waiting:
			dataBack = self.ser.readline()
			dataStore.append(dataBack.decode())
		sleep(0.1)
		return dataBack.decode()

	def AOMState(self):
		cmd = "ly_oxp2_output?\n"
		self.ser.write(cmd.encode())
		sleep(0.1)
		dataStore = []
		while self.ser.in_waiting:
			dataBack = self.ser.readline()
			dataStore.append(dataBack.decode())
		sleep(0.1)
		return dataBack.decode()

	def InterbusEnable(self):

		# AOM is open if return value is 0 and closed if return value is 1]
		cmd = "ly_oxp2_nktpbus=1\n"
		self.ser.write(cmd.encode())
		sleep(0.1)
		dataStore = []
		while self.ser.in_waiting:
			dataBack = self.ser.readline()
			dataStore.append(dataBack.decode())
			sleep(0.1)
			return dataBack.decode()

	def wavelength_change(self, wavelength):
		cmd = "ls_wavelength=" + str(wavelength) + "\n"
		self.ser.write(cmd.encode())
		sleep(0.1)
		dataStore = []
		while self.ser.in_waiting:
			data_back = self.ser.readline()
			dataStore.append(data_back.decode())
		sleep(0.1)
		return data_back.decode()

	def wavelength_read(self):
		cmd = "ls_wavelength?\n"
		self.ser.write(cmd.encode())
		sleep(0.1)
		dataStore = []
		while self.ser.in_waiting:
			data_back = self.ser.readline()
			dataStore.append(data_back.decode())
		sleep(0.1)
		return data_back.decode()

	def read_average_power(self):
		cmd = "e_mlp?\n"
		self.ser.write(cmd.encode())
		sleep(0.1)
		dataStore = []
		while self.ser.in_waiting:
			data_back = self.ser.readline()
			dataStore.append(data_back.decode())
		sleep(0.1)
		return data_back.decode()

	def freq_avaliable(self):
		cmd = "e_freq_available?\n"
		self.ser.write(cmd.encode())
		sleep(0.1)
		dataStore = []
		while self.ser.in_waiting:
			data_back = self.ser.readline()
			print(data_back.decode())
			dataStore.append(data_back.decode())
		sleep(0.1)
		return data_back.decode()

	def power_read_dv_green(self):
		cmd = "ls_output_power?\n"
		self.ser.write(cmd.encode())
		sleep(0.1)
		dataStore = []
		while self.ser.in_waiting:
			data_back = self.ser.readline()
			dataStore.append(data_back.decode())
		sleep(0.1)
		return data_back.decode()

	def status_led(self):
		cmd = "ly_oxp2_mode?\n"
		self.ser.write(cmd.encode())
		sleep(0.1)
		dataStore = []
		while self.ser.in_waiting:
			data_back = self.ser.readline()
			dataStore.append(data_back.decode())
		sleep(0.1)
		return data_back.decode()
