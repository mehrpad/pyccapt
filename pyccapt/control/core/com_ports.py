import serial.tools.list_ports


def list_com_ports():
	"""
    List all available COM ports on the system.

    Args:
    	None
    Returns:
        list: A list of available COM ports.
    """
	com_ports = list(serial.tools.list_ports.comports())
	return com_ports


print(list_com_ports())
