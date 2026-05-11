import ctypes
import time
from pathlib import Path

import numpy as np
from numpy.ctypeslib import ndpointer


class TDC:
    """
    This class sets up the parameters for the TDC and allows users to read experiment TDC values.
    """

    def __init__(self, tdc_lib, buf_size=30000, time_out=300):
        """
        Constructor function which initializes function parameters.

        Args:
                tdc_lib (ctypes.CDLL): The TDC library.
                buf_size (int): Buffer size.
                time_out (int): Timeout value.
        """
        self.tdc_lib = tdc_lib
        self.buf_size = buf_size
        self.time_out = time_out
        tdc_lib.Warraper_tdc_new.restype = ctypes.c_void_p
        tdc_lib.Warraper_tdc_new.argtypes = [ctypes.c_int, ctypes.c_int]
        tdc_lib.init_tdc.argtypes = [ctypes.c_void_p]
        tdc_lib.init_tdc.restype = ctypes.c_int
        tdc_lib.run_tdc.restype = ctypes.c_int
        tdc_lib.run_tdc.argtypes = [ctypes.c_void_p]
        tdc_lib.stop_tdc.restype = ctypes.c_int
        tdc_lib.stop_tdc.argtypes = [ctypes.c_void_p]
        tdc_lib.get_data_tdc_buf.restype = ndpointer(dtype=ctypes.c_double, shape=(12 * self.buf_size + 1,))
        tdc_lib.get_data_tdc_buf.argtypes = [ctypes.c_void_p]

        self.obj = tdc_lib.Warraper_tdc_new(self.buf_size, self.time_out)

    def stop_tdc(self):
        """
        Stop the TDC.

        Returns:
                int: Return code.
        """
        return self.tdc_lib.stop_tdc(self.obj)

    def init_tdc(self):
        """
        Initialize the TDC.

        Returns:
                int: Return code.
        """
        return self.tdc_lib.init_tdc(self.obj)

    def run_tdc(self):
        """
        Run the TDC.
        """
        self.tdc_lib.run_tdc(self.obj)

    def get_data_tdc_buf(self):
        """
        Get data from the TDC buffer.

        Returns:
                np.ndarray: Data from the TDC buffer.
        """
        data = self.tdc_lib.get_data_tdc_buf(self.obj)
        return data


def experiment_measure(variables, x_plot, y_plot, t_plot, main_v_dc_plot, stop_event):
    """
    Measurement function: This function is called in a process to read data from the queue.

    Args:
            variables: Variables object

    Returns:
            int: Return code.
    """
    try:
        # Load the library
        module_dir = Path(__file__).resolve().parent
        tdc_lib = ctypes.CDLL(str(module_dir / "wrapper_read_TDC8HP_x64.dll"))
    except Exception as exc:
        print("TDC DLL was not found")
        print(exc)
        variables.flag_finished_tdc = True
        if not getattr(variables, "access_override_enabled", False):
            variables.flag_tdc_failure = True
            return 1
        print("Access Override is active. Continuing without a RoentDek detector.")
        variables.flag_tdc_failure = False
        return 0

    tdc = TDC(tdc_lib, buf_size=30000, time_out=100)

    ret_code = tdc.init_tdc()
    if ret_code < 0:
        print(f"Error initializing RoentDek TDC: {ret_code}")
        variables.flag_finished_tdc = True
        if not getattr(variables, "access_override_enabled", False):
            variables.flag_tdc_failure = True
            return ret_code
        print("Access Override is active. Continuing without a RoentDek detector.")
        variables.flag_tdc_failure = False
        return 0

    tdc.run_tdc()
    events_detected = 0
    raw_signal_detected = 0
    events_detected_tmp = 0
    start_time = time.time()
    pulse_frequency = max(float(variables.pulse_frequency) * 1000.0, 1.0)
    _last_buf_error = None  # dedup transient SDK read errors

    while not stop_event.is_set() and not variables.flag_stop_tdc:
        # A DLL fault here used to take down the subprocess silently.
        # Catch + dedup so transient hiccups don't spam stdout and
        # don't bring down the experiment.
        try:
            return_value = tdc.get_data_tdc_buf()
        except Exception as exc:
            msg = str(exc)
            if msg != _last_buf_error:
                _last_buf_error = msg
                print(f"RoentDek TDC: get_data_tdc_buf failed (non-fatal): {msg}")
            time.sleep(0.05)
            continue
        _last_buf_error = None
        buffer_length = int(return_value[0])
        if buffer_length <= 0:
            time.sleep(0.01)
            continue

        return_value_tmp = np.copy(return_value[1 : buffer_length * 12 + 1].reshape(buffer_length, 12))

        xx = return_value_tmp[:, 8]
        yy = return_value_tmp[:, 9]
        tt = return_value_tmp[:, 10]
        time_stamp = return_value_tmp[:, 11]
        events_detected += buffer_length
        raw_signal_detected += buffer_length
        events_detected_tmp += buffer_length

        main_v_dc_list = np.tile(variables.specimen_voltage, buffer_length)
        pulse_data = np.tile(variables.pulse_voltage, buffer_length)
        laser_data = np.tile(variables.laser_pulse_energy, buffer_length)

        # Push into the shared-memory ring buffers (one per signal).
        # Append is non-blocking and bounded; the visualization
        # subprocess drains the rings on its own cadence.  (Was
        # Queue.put before the migration to SharedRingBuffer; this file
        # was missed in that round.)
        x_plot.write(xx)
        y_plot.write(yy)
        t_plot.write(tt)
        main_v_dc_plot.write(main_v_dc_list)

        variables.extend_to('x', xx.tolist())
        variables.extend_to('y', yy.tolist())
        variables.extend_to('t', tt.tolist())
        variables.extend_to('time_stamp', time_stamp.tolist())
        variables.extend_to('ch0', return_value_tmp[:, 0].tolist())
        variables.extend_to('ch1', return_value_tmp[:, 1].tolist())
        variables.extend_to('ch2', return_value_tmp[:, 2].tolist())
        variables.extend_to('ch3', return_value_tmp[:, 3].tolist())
        variables.extend_to('ch4', return_value_tmp[:, 4].tolist())
        variables.extend_to('ch5', return_value_tmp[:, 5].tolist())
        variables.extend_to('ch6', return_value_tmp[:, 6].tolist())
        variables.extend_to('ch7', return_value_tmp[:, 7].tolist())
        variables.extend_to('main_v_dc_dld', main_v_dc_list.tolist())
        variables.extend_to('main_v_p_dld', pulse_data.tolist())
        variables.extend_to('main_l_p_dld', laser_data.tolist())
        variables.extend_to('main_v_dc_tdc', main_v_dc_list.tolist())
        variables.extend_to('main_v_p_tdc', pulse_data.tolist())
        variables.extend_to('main_l_p_tdc', laser_data.tolist())
        variables.extend_to('main_p_tdc_roentdek', pulse_data.tolist())

        current_time = time.time()
        if current_time - start_time >= 0.5:
            # Re-read pulse_frequency every interval - if the user
            # changes it mid-run the rate calc otherwise stays wrong.
            try:
                live_pulse_frequency = max(float(variables.pulse_frequency) * 1000.0, 1.0)
                pulse_frequency = live_pulse_frequency
            except Exception:
                pass
            detection_rate = events_detected_tmp * 100 / pulse_frequency
            variables.detection_rate_current = detection_rate * 2
            variables.detection_rate_current_plot = detection_rate * 2
            variables.total_ions = events_detected
            variables.total_raw_signals = raw_signal_detected
            events_detected_tmp = 0
            start_time = current_time

    tdc.stop_tdc()
    variables.total_ions = events_detected
    variables.total_raw_signals = raw_signal_detected
    variables.flag_finished_tdc = True

    return 0
