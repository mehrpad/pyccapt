import logging


def logger_creator(script_name, variables, log_name, path):
	"""
	Create and configure a logger object for logging.

	This function uses the native Python logging library to create a logger object
	that can be used to log statements of different levels.

	Args:
		script_name (str): The name of the script using the logger.
		variables (object): An object containing relevant variables.
		log_name (str): The name of the log file.
		path (str): The path to the log directory.

	Returns:
		logging.Logger: The configured logger object.
	"""
	try:
		log_creator = logging.getLogger(script_name)
		log_creator.setLevel(logging.INFO)
		formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
		                              '%m-%d-%Y %H:%M:%S')

		file_handler_creator = logging.FileHandler(path + '\\' + log_name)
		file_handler_creator.setLevel(logging.DEBUG)
		file_handler_creator.setFormatter(formatter)
		log_creator.addHandler(file_handler_creator)
		return log_creator
	except Exception as e:
		print(f"Error creating logger: {e}")
		return None
