import ctypes
import os
import time

import numpy as np
from numpy.ctypeslib import ndpointer


class TdcRoentec:
    def __init__(self, buf_size, time_out):
        p = os.path.abspath(os.path.join(__file__, "../../.."))
        p = os.path.join(p, 'control', 'pyccapt', 'tdc_roentdek')
        os.chdir(p)
        self.tdc_lib = ctypes.CDLL("./wrapper_read_TDC8HP_x64.dll")

        self.tdc_lib.Warraper_tdc_new.restype = ctypes.c_void_p
        self.tdc_lib.Warraper_tdc_new.argtypes = [ctypes.c_int, ctypes.c_int]
        self.tdc_lib.init_tdc.argtypes = [ctypes.c_void_p]
        self.tdc_lib.init_tdc.restype = ctypes.c_int
        self.tdc_lib.run_tdc.restype = ctypes.c_int
        self.tdc_lib.run_tdc.argtypes = [ctypes.c_void_p]
        self.tdc_lib.stop_tdc.restype = ctypes.c_int
        self.tdc_lib.stop_tdc.argtypes = [ctypes.c_void_p]
        self.tdc_lib.get_data_tdc_buf.restype = ndpointer(dtype=ctypes.c_double, shape=(12 * buf_size + 1,))
        self.tdc_lib.get_data_tdc_buf.argtypes = [ctypes.c_void_p]
        self.obj = self.tdc_lib.Warraper_tdc_new(buf_size, time_out)

    def stop_tdc(self):
        return self.tdc_lib.stop_tdc(self.obj)

    def init_tdc(self):
        self.tdc_lib.init_tdc(self.obj)

    def run_tdc(self):
        self.tdc_lib.run_tdc(self.obj)

    def get_data_tdc_buf(self):
        data = self.tdc_lib.get_data_tdc_buf(self.obj)
        return data


def experiment_measure_buf(buffer_size, time_out):
    tdc = TdcRoentec(buf_size=buffer_size, time_out=time_out)
    tdc.init_tdc()
    tdc.run_tdc()
    i = 0
    start = time.time()
    data = None

    while i < 30:
        return_value = tdc.get_data_tdc_buf()
        buffer_length = int(return_value[0])
        return_value_tmp = np.copy(return_value[1 : buffer_length * 12 + 1].reshape(buffer_length, 12))

        if data is not None:
            data = np.append(data, return_value_tmp, 0)
        else:
            data = np.copy(return_value_tmp)

        print('%s events recorded in (s):' % buffer_length, time.time() - start)
        i += 1

    print('Experiment time:', time.time() - start)
    print(data.shape)

    import pandas as pd

    pd.DataFrame(data).to_csv("data.csv")

    tdc.stop_tdc()

    os.system('"lmf2txt.exe output.lmf -f"')
    print('Finish the reading')


if __name__ == '__main__':
    experiment_measure_buf(1000, 300)
