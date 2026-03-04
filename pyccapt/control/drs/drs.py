import ctypes
from pathlib import Path

import numpy as np
from numpy.ctypeslib import ndpointer


class DRS:
    """
    This class sets up the parameters for the DRS group and allows users to read experiment DRS values.
    """

    def __init__(self, trigger, test, delay, sample_frequency):
        """
        Constructor function which initializes function parameters.

        Args:
            trigger (int): Trigger type. 0 for internal trigger, 1 for external trigger.
            test (int): Test mode. 0 for normal mode, 1 for test mode (connect 100 MHz clock to all channels).
            delay (int): Trigger delay in nanoseconds.
            sample_frequency (float): Sample frequency at which the data is being captured.
            log (bool): Enable logging.
            log_path (str): Path for logging.

        """
        try:
            module_dir = Path(__file__).resolve().parent
            self.drs_lib = ctypes.CDLL(str(module_dir / "drs_lib.dll"))
        except Exception as exc:
            print("DRS DLL was not found")
            print(exc)
            raise

        self.drs_lib.Drs_new.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
        self.drs_lib.Drs_new.restype = ctypes.c_void_p
        self.drs_lib.Drs_reader.argtypes = [ctypes.c_void_p]
        self.drs_lib.Drs_reader.restype = ndpointer(dtype=ctypes.c_float, shape=(8 * 1024,))
        self.drs_lib.Drs_delete_drs_ox.restype = ctypes.c_void_p
        self.drs_lib.Drs_delete_drs_ox.argtypes = [ctypes.c_void_p]
        self.obj = self.drs_lib.Drs_new(trigger, test, delay, sample_frequency)

    def reader(self):
        """
        Read and return the DRS values.

        Returns:
            data: Read DRS values.
        """
        data = self.drs_lib.Drs_reader(self.obj)
        return data

    def delete_drs_ox(self):
        """
        Destroy the object.
        """
        self.drs_lib.Drs_delete_drs_ox(self.obj)


def experiment_measure(variables):
    """
    Continuously reads the DRS data and puts it into the queues.

    Args:
        variables: Variables object
    """
    drs_ox = DRS(trigger=0, test=1, delay=0, sample_frequency=2)

    while True:
        returnVale = np.array(drs_ox.reader())
        data = returnVale.reshape(8, 1024)
        # with self.variables.lock_data:
        ch0_time = data[0, :]
        ch0_wave = data[1, :]
        ch1_time = data[2, :]
        ch1_wave = data[3, :]
        ch2_time = data[4, :]
        ch2_wave = data[5, :]
        ch3_time = data[6, :]
        ch3_wave = data[7, :]

        variables.extend_to('ch0_time', ch0_time.tolist())
        variables.extend_to('ch0_wave', ch0_wave.tolist())
        variables.extend_to('ch1_time', ch1_time.tolist())
        variables.extend_to('ch1_wave', ch1_wave.tolist())
        variables.extend_to('ch2_time', ch2_time.tolist())
        variables.extend_to('ch2_wave', ch2_wave.tolist())
        variables.extend_to('ch3_time', ch3_time.tolist())
        variables.extend_to('ch3_wave', ch3_wave.tolist())

        voltage_data = np.tile(variables.specimen_voltage, len(ch0_time))
        pulse_data = np.tile(variables.pulse_voltage, len(ch0_time))
        variables.extend_to('main_v_dc_drs', voltage_data.tolist())
        variables.extend_to('main_v_p_drs', pulse_data.tolist())

        # with self.variables.lock_data_plot:
        variables.extend_to('main_v_dc_plot', voltage_data.tolist())
        # we have to calculate x and y from the wave data here
        variables.extend_to('x_plot', ch0_time.tolist())
        variables.extend_to('y_plot', ch0_time.tolist())
        variables.extend_to('t_plot', ch0_time.tolist())

        if variables.flag_stop_tdc:
            print('DRS loop is break in child process')
            break

    drs_ox.delete_drs_ox()

