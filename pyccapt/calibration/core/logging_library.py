import logging
import sys


def logger_creator(script_name, file_handler=None):
    """
    Instantiate and configure a logger object for logging.

    The function uses the native logging library in Python.

    Args:
        script_name (str): The name of the script or module using the logger.
        file_handler (dict, optional): Specification of the file handler. Defaults to None.

    Returns:
        logging.Logger: The logger object that can be used to log statements of different levels:
            - INFO: Useful information.
            - WARNING: Something is not right.
            - DEBUG: A debug message.
            - ERROR: A major error has occurred.
            - CRITICAL: A fatal error. Cannot continue.
    """
    log_creator = logging.getLogger(script_name)
    log_creator.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%m-%d-%Y %H:%M:%S')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)
    log_creator.addHandler(stdout_handler)

    if file_handler is None:
        return log_creator
    else:
        if 'path' not in file_handler or 'log_file_name' not in file_handler:
            print("Please provide the path and log_file_name in the dictionary")
        elif not file_handler['log_file_name'].endswith('.log'):
            print("Please provide the log file name with the .log extension")
        else:
            log_creator_file_handler = add_file_handler(log_creator, formatter, file_handler)
            return log_creator_file_handler


def add_file_handler(log_creator, formatter, file_handler):
    """
    Add a file handler to the logger.

    This function allows the respective module to add a file handler to the logger.

    Args:
        log_creator (logging.Logger): The logger object to which the file handler is added.
        formatter (logging.Formatter): The type of logging format.
        file_handler (dict): The module specification of the file handler.

    Returns:
        logging.Logger: The modified logger object with the additional file handler.
    """
    print("add_file_handler called", file_handler)
    path = file_handler['path']
    log_file_name = file_handler['log_file_name']
    file_handler_creator = logging.FileHandler(f"{path}/{log_file_name}")
    file_handler_creator.setLevel(logging.DEBUG)
    file_handler_creator.setFormatter(formatter)
    log_creator.addHandler(file_handler_creator)
    return log_creator
