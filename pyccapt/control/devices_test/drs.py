import ctypes
from numpy.ctypeslib import ndpointer
import numpy as np
import matplotlib.pyplot as plt
import time


class DRS:
    """
    DRS class for interacting with the DRS board.
    """

    def __init__(self, trigger, test, delay, sample_frequency):
        """
        Initialize the DRS object.

        :param trigger: 0 for internal trigger, 1 for external trigger
        :param test: 0 for test mode off, 1 for test mode on
        :param delay: Trigger delay in ns
        :param sample_frequency: Sample frequency in GHz
        """
        self.drs_lib = ctypes.CDLL("../drs/drs_lib.dll")
        self.drs_lib.Drs_new.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
        self.drs_lib.Drs_new.restype = ctypes.c_void_p
        self.drs_lib.Drs_reader.argtypes = [ctypes.c_void_p]
        self.drs_lib.Drs_reader.restype = ndpointer(dtype=ctypes.c_float, shape=(8 * 1024,))
        self.drs_lib.Drs_delete_drs_ox.restype = ctypes.c_void_p
        self.drs_lib.Drs_delete_drs_ox.argtypes = [ctypes.c_void_p]
        self.obj = self.drs_lib.Drs_new(trigger, test, delay, sample_frequency)

    def reader(self):
        """
        Read data from the DRS board.

        :return: Data array
        """
        data = self.drs_lib.Drs_reader(self.obj)
        return data

    def delete_drs_ox(self):
        """
        Delete the DRS object.
        """
        self.drs_lib.Drs_delete_drs_ox(self.obj)


if __name__ == '__main__':
    # Create DRS object and initialize the DRS board
    drs_board = DRS(trigger=0, test=1, delay=0, sample_frequency=2)

    # Acquire data
    data_final = None
    for _ in range(20):
        # Read data from DRS
        start = time.time()
        return_value = np.array(drs_board.reader())
        print('Run time:', time.time() - start)

        # Reshape the data into 8 channels with 1024 samples
        data = return_value.reshape(8, 1024)

        if data_final is None:
            data_final = data
        else:
            data_final = np.concatenate((data_final, data), axis=1)

    # Delete the DRS object
    drs_board.delete_drs_ox()

    # Plot the data
    fig, axs = plt.subplots(nrows=2, ncols=2)
    fig.tight_layout()
    color_list = ['b', 'r', 'g', 'y']
    for i, ax in enumerate(axs.flatten()):
        channel_data = data_final[i * 2 : (i + 1) * 2, :]
        ax.plot(channel_data[0], channel_data[1], color_list[i])
        ax.set_title(f'Detector signal {i + 1}')
    fig.subplots_adjust(wspace=0.2)
    plt.show()
