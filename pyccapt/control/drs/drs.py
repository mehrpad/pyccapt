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
from pyccapt.control.apt.detector_chunks import ChunkStream, ChunkWriter, preserve_unattributed_packet
from pyccapt.control.core.shared_ring_buffer import publish_plot_events


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


def experiment_measure(variables, x_plot=None, y_plot=None, t_plot=None, main_v_dc_plot=None,
                       stop_event=None, conf=None):
    """Persist flattened waveforms and one voltage sample per acquisition."""
    drs_ox = writer = None
    streams = []
    failure = None
    conf = conf or {}
    try:
        mode = str(conf.get("drs_acquisition_mode", "normal")).strip().lower()
        if mode not in {"normal", "internal_calibration"}:
            raise ValueError("drs_acquisition_mode must be normal or internal_calibration")
        # Calibration routes the internal 100 MHz clock to every input. It is
        # never implicitly selected for ordinary specimen acquisition.
        drs_ox = DRS(trigger=0, test=int(mode == "internal_calibration"), delay=0, sample_frequency=2)
        writer = ChunkWriter(Path(variables.path)/"temp_data"/"chunks", stop_event=stop_event)
        waveforms = ChunkStream(writer, "hsd_waveforms",
                                {f"ch{i}_{kind}": np.float32 for i in range(4) for kind in ("time", "wave")},
                                chunk_size=64*1024)
        metadata = ChunkStream(writer, "hsd_events", {"voltage_drs": np.float64, "pulse_drs": np.float64},
                               chunk_size=64)
        streams = [waveforms, metadata]
        while not (variables.flag_stop_tdc or variables.stop_flag or
                   (stop_event is not None and stop_event.is_set())):
            writer.check()
            try:
                data = np.array(drs_ox.reader(), dtype=np.float32).reshape(8, 1024)
            except Exception as exc:
                print(f"DRS read failed: {exc}")
                time.sleep(.1)
                continue
            try:
                dc, pulse = float(variables.specimen_voltage), float(variables.pulse_voltage)
            except Exception:
                preserve_unattributed_packet(Path(variables.path)/"temp_data", "hsd", {"waveforms": data})
                raise
            waveforms.append({f"ch{i}_{kind}": data[2*i+j] for i in range(4)
                              for j, kind in enumerate(("time", "wave"))})
            metadata.append({"voltage_drs": [dc], "pulse_drs": [pulse]})
            variables.total_raw_signals = int(variables.total_raw_signals)+1
            variables.total_ions = int(variables.total_ions)+1
            # Live plots carry waveform samples, so voltage has one value per
            # plotted sample. Saved metadata remains one value per waveform.
            if all(ring is not None for ring in (x_plot, y_plot, t_plot, main_v_dc_plot)):
                publish_plot_events(x_plot, y_plot, t_plot, main_v_dc_plot,
                                    data[0], data[1], data[0], np.full(1024, dc))
    except Exception as exc:
        failure = exc
    finally:
        if writer is not None:
            try:
                for stream in streams:
                    stream.finish()
            except Exception as exc:
                failure = failure or exc
            try:
                writer.close()
            except Exception as exc:
                failure = failure or exc
        if drs_ox is not None:
            try:
                drs_ox.delete_drs_ox()
            except Exception as exc:
                failure = failure or exc
        if failure is not None:
            variables.flag_tdc_failure = True
            variables.detector_error = f"{type(failure).__name__}: {failure}"
            print(f"DRS acquisition failed: {variables.detector_error}")
        variables.flag_finished_tdc = True
    return -1 if failure is not None else 0
