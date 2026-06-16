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


if __name__ == "__main__":
    # Previously this print ran at IMPORT, so any module that did
    # ``from pyccapt.control.core import com_ports`` (or *) printed
    # the full COM port list to stdout on every Python start. Move
    # behind __main__ so the module can be imported safely.
    print(list_com_ports())
