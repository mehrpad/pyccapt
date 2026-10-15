import ctypes
import time
from pathlib import Path

import numpy as np
from numpy.ctypeslib import ndpointer
from pyccapt.control.apt.detector_chunks import ChunkStream, ChunkWriter, preserve_unattributed_packet, read_event_metadata
from pyccapt.control.core.shared_ring_buffer import publish_plot_events
from pyccapt.control.nkt_photonics.readback import pulse_energy_pj, experiment_frequency_hz


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

    events_detected = raw_signal_detected = events_detected_tmp = 0
    start_time = time.monotonic()
    pulse_frequency = max(experiment_frequency_hz(variables), 1.)
    last_buf_error = None
    writer = stream = None
    failure = None
    from pyccapt.control.apt.laser_alignment_data import LaserAlignmentPublisher
    from pyccapt.control.apt.alignment_vision import AlignmentEventPublisher
    laser_alignment_publisher = LaserAlignmentPublisher(variables)
    alignment_publisher = AlignmentEventPublisher(variables)
    try:
        writer = ChunkWriter(Path(variables.path)/"temp_data"/"chunks", stop_event=stop_event)
        dtypes = {name: np.float64 for name in
                  ("x", "y", "t", "voltage", "voltage_pulse", "laser_pulse")}
        dtypes.update({name: np.uint64 for name in ("time_stamp", *(f"ch{i}" for i in range(8)))})
        stream = ChunkStream(writer, "roentdek", dtypes)
        tdc.run_tdc()
        while not stop_event.is_set() and not variables.flag_stop_tdc:
            writer.check()
            try:
                return_value = tdc.get_data_tdc_buf()
                last_buf_error = None
            except Exception as exc:
                if str(exc) != last_buf_error:
                    last_buf_error = str(exc)
                    print(f"RoentDek TDC read failed: {exc}")
                return_value = np.zeros(1)
                time.sleep(.05)
            buffer_length = int(return_value[0])
            if buffer_length < 0 or buffer_length > 30000:
                raise ValueError(f"Invalid RoentDek SDK event count: {buffer_length}")
            if buffer_length:
                data = np.copy(return_value[1:buffer_length*12+1].reshape(buffer_length, 12))
                try:
                    dc, pulse, laser = read_event_metadata(variables, pulse_energy_pj)
                except Exception:
                    preserve_unattributed_packet(Path(variables.path)/"temp_data", "roentdek", {"events": data})
                    raise
                voltage = np.full(buffer_length, dc)
                arrays = {"x": data[:, 8], "y": data[:, 9], "t": data[:, 10],
                          "time_stamp": data[:, 11], "voltage": voltage,
                          "voltage_pulse": np.full(buffer_length, pulse),
                          "laser_pulse": np.full(buffer_length, laser)}
                arrays.update({f"ch{i}": data[:, i] for i in range(8)})
                stream.append(arrays)
                events_detected += buffer_length
                raw_signal_detected += buffer_length
                events_detected_tmp += buffer_length
                publish_plot_events(x_plot, y_plot, t_plot, main_v_dc_plot,
                                    data[:, 8], data[:, 9], data[:, 10], voltage)
                alignment_publisher.append(data[:, 8], data[:, 9])
                laser_alignment_publisher.append(data[:, 10])
            else:
                time.sleep(.01)

            # Empty packets still advance the rate window. A silent detector
            # must publish zero rather than retain its last nonzero response.
            current_time = time.monotonic()
            elapsed_s = current_time-start_time
            if elapsed_s >= .5:
                try:
                    pulse_frequency = max(experiment_frequency_hz(variables), 1.)
                except Exception:
                    pass
                rate = events_detected_tmp*100/(pulse_frequency*elapsed_s)
                variables.detection_rate_current = rate
                variables.detection_rate_current_plot = rate
                variables.total_ions = events_detected
                variables.total_raw_signals = raw_signal_detected
                events_detected_tmp = 0
                start_time = current_time
    except Exception as exc:
        failure = exc
    finally:
        if stream is not None:
            try:
                stream.finish()
            except Exception as exc:
                failure = failure or exc
        if writer is not None:
            try:
                writer.close()
            except Exception as exc:
                failure = failure or exc
        try:
            tdc.stop_tdc()
        except Exception as exc:
            failure = failure or exc
        variables.total_ions = events_detected
        variables.total_raw_signals = raw_signal_detected
        if failure is not None:
            variables.flag_tdc_failure = True
            variables.detector_error = f"{type(failure).__name__}: {failure}"
            print(f"RoentDek acquisition failed: {variables.detector_error}")
        variables.flag_finished_tdc = True
    return -1 if failure is not None else 0
