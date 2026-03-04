import sys

from PyQt6 import QtWidgets

from pyccapt.control.control import runtime
from pyccapt.control.gui import gui_main


def main():
	"""
	Load the GUI based on the configuration file.

	This function reads the configuration file, initializes global experiment variables, and
	shows the GUI window.

	Args:
		None

	Returns:
		None
	"""
	try:
		conf, _ = runtime.load_project_config()
	except Exception as exc:
		print("Cannot load the configuration file")
		print(exc)
		sys.exit()

	shared = runtime.create_shared_context(conf)


	app = QtWidgets.QApplication(sys.argv)
	app.setStyle('Fusion')
	window = gui_main.MyPyCCAPT(
		shared.variables,
		conf,
		shared.x_plot,
		shared.y_plot,
		shared.t_plot,
		shared.main_v_dc_plot,
	)
	window.show()
	sys.exit(app.exec())


if __name__ == '__main__':
	main()
