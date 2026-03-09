import time
from datetime import datetime
from threading import Thread
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import style
try:
	from mcculw import ul
	from mcculw.enums import InfoType, BoardInfo, TcType, TempScale, TInOptions
except Exception as e:
	print('Cannot import mcculw library')
	print(e)

# local imports
from pyccapt.control.devices.pfeiffer_gauges import TPG362


def daq_tc():
	"""
	This function performs DAQ operations related to temperature measurement and control.
	"""
	device_to_show = "USB-TC"
	board_num = 0

	print("Looking for Board 0 in InstaCal to be {0} series...".format(device_to_show))

	try:
		board_name = ul.get_board_name(board_num)
	except Exception as e:
		if ul.ErrorCode(1):
			print("\nNo board found at Board 0.")
			print(e)
			return
	else:
		if device_to_show in board_name:
			print("{0} found as Board number {1}.\n".format(board_name, board_num))
			ul.flash_led(board_num)
		else:
			print("\nNo {0} series found as Board 0. Please run InstaCal.".format(device_to_show))
			return

	try:
		channel = 1
		ul.set_config(InfoType.BOARDINFO, board_num, channel, BoardInfo.CHANTCTYPE, TcType.K)
		ul.set_config(InfoType.BOARDINFO, board_num, channel, BoardInfo.TEMPSCALE, TempScale.CELSIUS)
		ul.set_config(InfoType.BOARDINFO, board_num, channel, BoardInfo.ADDATARATE, 60)
	except Exception as e:
		print('\n', e)


def read():
	"""
	This function reads data from sensors and logs it.
	"""
	tpg = TPG362(port='COM5')
	unit = tpg.pressure_unit()

	index = 0
	while True:
		print('-----------', index, 'seconds', '--------------')
		gauge_bc, _ = tpg.pressure_gauge(1)
		gauge_mc, _ = tpg.pressure_gauge(2)

		board_num = 0
		channel_list = ['MC_NEG', 'MC_Det', 'Mc_Top', 'MC_Gate', 'BC_Top', 'BC_Pump']
		value_temperature = []
		for i in range(6):
			options = TInOptions.NOFILTER
			val = float(ul.t_in(board_num, i, TempScale.CELSIUS, options))
			value_temperature.append(round(val, 3))

		value_temperature = np.array(value_temperature, dtype=np.dtype(float))

		new_row = [
			now.strftime("%d-%m-%Y"),
			datetime.now().strftime('%H:%M:%S'),
			gauge_mc, gauge_bc,
			value_temperature[0], value_temperature[1], value_temperature[2],
			value_temperature[3], value_temperature[4], value_temperature[5]
		]

		data.loc[len(data)] = new_row

		time.sleep(1)
		index = index + 1
		if index % 20 == 0:
			try:
				data.to_csv(file_name, sep=';', index=False)
			except:
				data.to_csv(file_name_backup, sep=';', index=False)
				print('csv File cannot be saved')
				print('close the csv file')


def animate(i):
	"""
	Animation function for plotting.
	"""
	time = data['Time'].to_numpy()
	MC_NEG = data['MC_NEG'].to_numpy()
	MC_Det = data['MC_Det'].to_numpy()
	Mc_Top = data['Mc_Top'].to_numpy()
	MC_Gate = data['MC_Gate'].to_numpy()
	BC_Top = data['BC_Top'].to_numpy()
	BC_Pump = data['BC_Pump'].to_numpy()
	MC_vacuum = data['MC_vacuum'].to_numpy()
	BC_vacuum = data['BC_vacuum'].to_numpy()

	ax1.clear()
	ax2.clear()

	ax1.plot(time[-20:], MC_NEG[-20:], label='MC_NEG', color='b')
	ax1.plot(time[-20:], MC_Det[-20:], label='MC_Det', color='g')
	ax1.plot(time[-20:], Mc_Top[-20:], label='Mc_Top', color='r')
	ax1.plot(time[-20:], MC_Gate[-20:], label='MC_Gate', color='c')
	ax1.plot(time[-20:], BC_Top[-20:], label='BC_Top', color='m')
	ax1.plot(time[-20:], BC_Pump[-20:], label='BC_Pump', color='y')

	ax2.plot(time[-20:], MC_vacuum[-20:], label='MC_vacuum', color='orange')
	ax2.plot(time[-20:], BC_vacuum[-20:], label='BC_vacuum', color='darkviolet')

	ax1.set_title('Baking Temperature')
	ax2.set_title('Baking Vacuum')

	ax1.set_ylabel('Temperature (C)')
	ax2.set_ylabel('Vacuum (mbar)')

	ax1.legend(loc='upper right')
	ax2.legend(['MC_vacuum', 'BC_vacuum'], loc='upper right')


style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)


def plot_baking(df, window=0):
	"""
	Plot baking data.
	"""
	MC_NEG = df['MC_NEG'].to_numpy()
	MC_Det = df['MC_Det'].to_numpy()
	Mc_Top = df['Mc_Top'].to_numpy()
	MC_Gate = df['MC_Gate'].to_numpy()
	BC_Top = df['BC_Top'].to_numpy()
	BC_Pump = df['BC_Pump'].to_numpy()
	MC_vacuum = df['MC_vacuum'].to_numpy()
	BC_vacuum = df['BC_vacuum'].to_numpy()
	time = np.arange(0, len(BC_vacuum))

	ax1.plot(time[window:], MC_NEG[window:], label='MC_NEG', color='b')
	ax1.plot(time[window:], MC_Det[window:], label='MC_Det', color='g')
	ax1.plot(time[window:], Mc_Top[window:], label='Mc_Top', color='r')
	ax1.plot(time[window:], MC_Gate[window:], label='MC_Gate', color='c')
	ax1.plot(time[window:], BC_Top[window:], label='BC_Top', color='m')
	ax1.plot(time[window:], BC_Pump[window:], label='BC_Pump', color='y')

	ax2.plot(time[window:], MC_vacuum[window:], label='MC_vacuum', color='orange')
	ax2.plot(time[window:], BC_vacuum[window:], label='BC_vacuum', color='darkviolet')

	ax1.set_title('Baking')

	ax1.set_ylabel('Temperature (C)')
	ax2.set_ylabel('Vacuum (mbar)')
	ax1.set_xlabel('Time (0.5 s)')
	ax2.set_xlabel('Time (0.5 s)')

	ax1.legend(loc='upper right')
	ax2.legend(['MC_vacuum', 'BC_vacuum'], loc='upper right')
	plt.show()


if __name__ == '__main__':
	# Set recording and plotting flags
	recording = True
	ploting = False

	if recording:
		# Get current date and time for file names
		now = datetime.now()
		now_time = now.strftime("%d-%m-%Y_%H-%M-%S")

		# Create file names
		file_name = 'baking_logging_%s.csv' % now_time
		file_name_backup = 'backup_baking_logging_%s.csv' % now_time

		# Create an empty DataFrame to store data
		data = pd.DataFrame(columns=[
			'data', 'Time', 'MC_vacuum', 'BC_vacuum', 'MC_NEG', 'MC_Det', 'Mc_Top', 'MC_Gate', 'BC_Top', 'BC_Pump'
		])

		# Create and start a thread for data reading
		thread_read = Thread(target=read)
		thread_read.daemon = True
		thread_read.start()

		# Create and start animation
		ani = animation.FuncAnimation(fig, animate, interval=1000)
		plt.show()

		try:
			data.to_csv(file_name, sep=';', index=False)
		except:
			data.to_csv(file_name_backup, sep=';', index=False)

	if ploting:
		# Read data from CSV file
		df = pd.read_csv('baking_logging_13-01-2023_11-48-02.csv', sep=';')

		# Plot data
		plot_baking(df, window=0)

		# Print DataFrame
		print(df)
