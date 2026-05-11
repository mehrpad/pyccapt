"""Python wrapper around the DRS digitizer board library.

Vendor / origin
---------------
The DRS evaluation board, the ``DRS.dll`` / ``libDRS`` shared library,
and the underlying DRS4 ASIC are products of the
**Paul Scherrer Institute (PSI)**, Switzerland. The native source code
under ``pyccapt/control/drs/source/drs5_lib/`` is PSI's; it is
included here so PyCCAPT can be built on machines without a separate
DRS SDK install. See ``source/drs5_lib/`` for PSI's licence (the DRS
software is distributed under the GNU General Public License v3 with
the additional permission that linking against PSI's drivers does not
constitute a derivative work).

This Python wrapper is local pyccapt code that calls into PSI's
library via ctypes.
"""

import ctypes
import time
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
        # Stop check at the top so the user's Stop click does not
        # always cost one extra acquisition.
        if variables.flag_stop_tdc:
            print('DRS loop is break in child process')
            break

        try:
            returnVale = np.array(drs_ox.reader())
            data = returnVale.reshape(8, 1024)
        except Exception as exc:
            # Don't kill the worker silently on a single bad read
            # (USB hiccup, board reset, unexpected shape). Log and
            # let the next iteration retry.
            print(f"DRS read failed: {exc}")
            time.sleep(0.1)
            continue

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

        # One DC / pulse voltage reading per acquisition (per waveform),
        # not one per sample. The previous code tiled both to 1024 copies
        # which bloated the HDF5 by 1024x and only "worked" because every
        # downstream reader implicitly assumed N=1024 samples per event.
        variables.extend_to('main_v_dc_drs', [float(variables.specimen_voltage)])
        variables.extend_to('main_v_p_drs', [float(variables.pulse_voltage)])

        # Plot stream: one scalar per acquisition, matching the new
        # per-event voltage shape. (x_plot/y_plot/t_plot still feed the
        # raw waveform time array as a placeholder — proper hit
        # reconstruction from the waves is a separate task.)
        variables.extend_to('main_v_dc_plot', [float(variables.specimen_voltage)])
        # we have to calculate x and y from the wave data here
        variables.extend_to('x_plot', ch0_time.tolist())
        variables.extend_to('y_plot', ch0_time.tolist())
        variables.extend_to('t_plot', ch0_time.tolist())

    drs_ox.delete_drs_ox()

