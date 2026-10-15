import multiprocessing as mp
import os
import time
import threading
from pathlib import Path
from queue import Empty, Full, Queue

import numpy as np

# local imports
from pyccapt.control.core import runtime as _runtime
from pyccapt.control.core.chunk_store import atomic_write_chunk_group
from pyccapt.control.devices import initialize_devices
from pyccapt.control.nkt_photonics.readback import pulse_energy_pj, experiment_frequency_hz
from pyccapt.control.tdc_surface_concept import scTDC
from pyccapt.control.apt.detector_chunks import (
    ChunkStream, ChunkWriter, preserve_unattributed_packet, read_event_metadata,
)
from pyccapt.control.core.shared_ring_buffer import publish_plot_events

QUEUE_DATA = 0
QUEUE_ENDOFMEAS = 1
CHUNK_SIZE = 100_000  # Adjust the chunk size if needed

# Canonical on-disk dtype for each chunk stem, matching the HDF5 schema in
# core/hdf5_creator.py (chunk_mapping).  The integer stems are built from
# Python ints via list.extend(...), so a bare np.array(...) would infer the
# platform default int64.  hdf_creator._write_chunked_dataset declares these
# datasets as uint64/uint32 and (since the strict-dtype guard was added)
# refused to finalize when the chunk dtype did not match -- which broke every
# experiment longer than CHUNK_SIZE.  Pinning the dtype here keeps the writer
# and the HDF5 schema in agreement.  Stems absent from this map (e.g. the
# *_bin raw arrays, which hdf_creator does not consume) keep NumPy's inferred
# dtype.
CHUNK_DTYPES = {
    "x": np.float64,
    "y": np.float64,
    "t": np.float64,
    "voltage": np.float64,
    "voltage_pulse": np.float64,
    "laser_pulse": np.float64,
    "start_counter": np.uint64,
    "channel": np.uint32,
    "time": np.uint64,
    "tdc_start_counter": np.uint64,
    "voltage_tdc": np.float64,
    "voltage_pulse_tdc": np.float64,
    "laser_pulse_tdc": np.float64,
}
# Per-queue blocking-get timeout (s).  Two callbacks fire at different
# rates (DLD vs raw); using a small timeout instead of an indefinite
# wait avoids a deadlock when one queue is empty while the other is
# being drained.  Also lets us re-check stop_event between gets so
# STOP actually stops the TDC.
QUEUE_GET_TIMEOUT_S = 0.05


class BufDataCB4(scTDC.buffered_data_callbacks_pipe):
    """
    The class inherits from python wrapper module scTDC and class: buffered_data_callbacks_pipe
    """

    def __init__(self, lib, dev_desc, data_field_selection, dld_events, max_buffered_data_len=500_000):
        """
        Initialize the base class: scTDC.buffered_data_callbacks_pipe

        Args:
            lib (scTDClib): A scTDClib object.
            dev_desc (int): Device descriptor as returned by sc_tdc_init_inifile(...).
            data_field_selection (int): A 'bitwise or' combination of SC_DATA_FIELD_xyz constants.
            dld_events (bool): True to receive DLD events, False to receive TDC events.
            max_buffered_data_len (int): Number of events buffered before invoking callbacks.
        """
        super().__init__(lib, dev_desc, data_field_selection, max_buffered_data_len, dld_events)

        # Bound callback backlog so a stalled consumer cannot exhaust RAM.
        # put() intentionally applies backpressure; silently dropping detector
        # records would be worse than slowing acquisition.
        self.queue = Queue(maxsize=32)
        self.end_of_meas = False
        self._closing = threading.Event()
        self._publication_lock = threading.Lock()
        self.unattributed_directory = None
        self.stream_name = 'dld' if dld_events else 'tdc'
        self.shutdown_packets = 0
        self.publication_error = ''

    def _publish(self, item):
        # Closing sets the event before waiting for this bounded put. Once
        # stop_publication returns no producer can append behind the tail drain.
        with self._publication_lock:
            while not self._closing.is_set():
                try:
                    self.queue.put(item, timeout=QUEUE_GET_TIMEOUT_S)
                    return True
                except Full:
                    continue
        return False

    def stop_publication(self):
        self._closing.set()
        with self._publication_lock:
            pass

    def close(self):
        self.stop_publication()
        return super().close()

    def on_data(self, d):
        """
        This class method function:
            1. Makes a deep copy of numpy arrays in d
            2. Inserts basic values by simple assignment
            3. Inserts numpy arrays using the copy method of the source array

        Args:
            d (dict): Data dictionary.

        Returns:
            None
        """
        dcopy = {}
        for k in d.keys():
            if isinstance(d[k], np.ndarray):
                dcopy[k] = d[k].copy()
            else:
                dcopy[k] = d[k]
        if not self._publish((QUEUE_DATA, dcopy)):
            # Output safe-off runs immediately when Stop is requested. A late
            # SDK packet cannot safely inherit the now-changing voltage. Keep
            # its raw evidence explicitly unassigned instead of dropping it.
            try:
                if self.unattributed_directory is None:
                    raise RuntimeError('No directory configured for stopped callback evidence')
                preserve_unattributed_packet(self.unattributed_directory, self.stream_name+'_shutdown', dcopy)
                self.shutdown_packets += 1
            except Exception as exc:
                self.publication_error = f'{type(exc).__name__}: {exc}'
        if self.end_of_meas:
            self.end_of_meas = False
            self._publish((QUEUE_ENDOFMEAS, None))

    def on_end_of_meas(self):
        """
        This class method sets end_of_meas to True.

        Returns:
            True (bool)
        """
        self.end_of_meas = True
        return True


def errorcheck(device, bufdatacb, bufdatacb_raw, retcode):
    """
    This function checks return codes for errors and does cleanup.

    Args:
        retcode (int): Return code.
        bufdatacb (BufDataCB4): A BufDataCB4 object.
        bufdatacb_raw (BufDataCB4): A BufDataCB4 object.
        device (scTDC.Device): A scTDC.Device object.

    Returns:
        int: 0 if success return code or return code > 0, -1 if return code is error code or less than 0.
    """
    if retcode < 0:
        print(device.lib.sc_get_err_msg(retcode))
        bufdatacb.close()
        bufdatacb_raw.close()
        device.deinitialize()
        return -1
    else:
        return 0


def save_chunk_worker(save_queue):
    while True:
        task = save_queue.get()
        if task is None:  # Stop signal
            break

        chunk_id, path, chunk_data = task
        normalized = {
            key: np.asarray(data, dtype=CHUNK_DTYPES.get(key))
            if CHUNK_DTYPES.get(key) is not None
            else np.asarray(data)
            for key, data in chunk_data.items()
        }
        stream_name = "tdc" if "channel" in normalized else "dld"
        atomic_write_chunk_group(
            Path(path) / "chunks",
            stream_name=stream_name,
            chunk_id=chunk_id,
            arrays=normalized,
        )

        # No artificial throttle: save_queue.get() already blocks when idle,
        # so the worker never busy-waits. A fixed per-chunk sleep would cap
        # throughput to ~1 chunk per sleep -- and because the raw/TDC stream
        # chunks far more often than DLD, that let the queue (and RAM) grow
        # without bound on high-raw-rate runs.


def load_and_concatenate_chunks(path, chunk_id):
    attr_names = [
        "x_bin",
        "x",
        "y_bin",
        "y",
        "t_bin",
        "t",
        "voltage",
        "voltage_pulse",
        "laser_pulse",
        "start_counter",
        "channel",
        "time",
        "tdc_start_counter",
        "voltage_tdc",
        "voltage_pulse_tdc",
        "laser_pulse_tdc",
    ]

    all_data = {attr: [] for attr in attr_names}  # Initialize storage

    for i in range(1, chunk_id + 1):
        for attr in attr_names:
            chunk_file = os.path.join(path, f"chunks/{attr}_chunk_{i}.npy")
            if os.path.exists(chunk_file):
                try:
                    all_data[attr].append(np.load(chunk_file))  # Load NumPy array directly
                except Exception as e:
                    print(f"Error loading {chunk_file}: {e}")
            else:
                print(f"Warning: File '{chunk_file}' not found.")

    # Convert lists of arrays to single NumPy arrays (faster processing)
    return tuple(np.concatenate(all_data[attr]) if all_data[attr] else np.array([]) for attr in attr_names)


def run_experiment_measure(variables, x_plot, y_plot, t_plot, main_v_dc_plot, stop_event):
    """
    Measurement function: This function is called in a process to read data from the queue.

    Args:
        variables (share_variables.Variables): A share_variables.Variables object.
        x_plot (multiprocessing.Array): A multiprocessing.Array object.
        y_plot (multiprocessing.Array): A multiprocessing.Array object.
        t_plot (multiprocessing.Array): A multiprocessing.Array object.
        main_v_dc_plot (multiprocessing.Array): A multiprocessing.Array object.
        stop_event (multiprocessing.Event): A multiprocessing.Event object.

    Returns:
        int: Return code.
    """
    # Calibration constants - all configurable via config.toml so
    # different rigs can be tuned without editing source.  Defaults
    # match the values that were hard-coded here historically.
    try:
        _conf, _ = _runtime.load_project_config(change_cwd=False)
    except Exception:
        _conf = {}
    exposure_time = int(_conf.get("sc_exposure_time", 100))
    TOFFACTOR = float(_conf.get("sc_tof_ns_per_bin", 27.432 / (1000 * 4)))
    DETBINS = int(_conf.get("sc_detector_bins", 4900))
    BINNINGFAC = int(_conf.get("sc_detector_binning_factor", 2))
    DET_WIDTH_MM = float(_conf.get("sc_detector_width_mm", 80))
    XYFACTOR = DET_WIDTH_MM / DETBINS * BINNINGFAC  # mm per bin
    XYBINSHIFT = DETBINS / BINNINGFAC / 2  # to center the detector

    device = scTDC.Device(autoinit=False)
    retcode, errmsg = device.initialize()

    if retcode < 0:
        print("Error during init:", retcode, errmsg)
        print(
            f"{initialize_devices.bcolors.FAIL}Error: Restart the TDC manually "
            f"(Turn it On and Off){initialize_devices.bcolors.ENDC}"
        )
        variables.flag_finished_tdc = True
        if not getattr(variables, "access_override_enabled", False):
            variables.flag_tdc_failure = True
            return -1
        print("Access Override is active. Continuing without a Surface Concept detector.")
        variables.flag_tdc_failure = False
        return 0

    else:
        print("TDC is successfully initialized")
        variables.flag_tdc_failure = False

    DATA_FIELD_SEL = (
        scTDC.SC_DATA_FIELD_DIF1 | scTDC.SC_DATA_FIELD_DIF2 | scTDC.SC_DATA_FIELD_TIME | scTDC.SC_DATA_FIELD_START_COUNTER
    )
    DATA_FIELD_SEL_raw = scTDC.SC_DATA_FIELD_TIME | scTDC.SC_DATA_FIELD_CHANNEL | scTDC.SC_DATA_FIELD_START_COUNTER

    bufdatacb = BufDataCB4(device.lib, device.dev_desc, DATA_FIELD_SEL, dld_events=True)
    bufdatacb_raw = BufDataCB4(device.lib, device.dev_desc, DATA_FIELD_SEL_raw, dld_events=False)
    directory = Path(variables.path) / "temp_data"
    for callback in (bufdatacb, bufdatacb_raw):
        callback.unattributed_directory = directory

    retcode = bufdatacb.start_measurement(exposure_time)
    if errorcheck(device, bufdatacb, bufdatacb_raw, retcode) < 0:
        variables.flag_finished_tdc = True
        variables.flag_tdc_failure = not getattr(variables, "access_override_enabled", False)
        return -1 if variables.flag_tdc_failure else 0

    from pyccapt.control.apt.laser_alignment_data import LaserAlignmentPublisher
    from pyccapt.control.apt.alignment_vision import AlignmentEventPublisher
    laser_alignment_publisher = LaserAlignmentPublisher(variables)
    alignment_publisher = AlignmentEventPublisher(variables, _conf)
    writer = None
    events_detected = raw_signal_detected = events_detected_tmp = 0
    start_time = time.monotonic()
    pulse_frequency = experiment_frequency_hz(variables)
    streams = []
    failure = None
    try:
        writer = ChunkWriter(directory / "chunks", stop_event=stop_event)
        dld_types = {name: CHUNK_DTYPES[name] for name in
                     ("x", "y", "t", "voltage", "voltage_pulse", "laser_pulse", "start_counter")}
        dld_types.update(x_bin=np.int64, y_bin=np.int64, t_bin=np.int64)
        dld = ChunkStream(writer, "dld", dld_types, chunk_size=CHUNK_SIZE)
        raw = ChunkStream(writer, "tdc", {name: CHUNK_DTYPES[name] for name in
                          ("channel", "time", "tdc_start_counter", "voltage_tdc",
                           "voltage_pulse_tdc", "laser_pulse_tdc")}, chunk_size=CHUNK_SIZE)
        streams = [dld, raw]
        callbacks = (bufdatacb, bufdatacb_raw)
        idle_index = 0
        while not stop_event.is_set():
            writer.check()
            # Poll both independent streams before waiting. An empty DLD queue
            # must not impose a 50 ms delay on every available raw packet.
            packets = []
            for index, callback in enumerate(callbacks):
                try:
                    kind, data = callback.queue.get(block=False)
                except Empty:
                    continue
                if kind is not None:
                    packets.append((index, kind, data))
            if not packets:
                index = idle_index
                idle_index = 1-idle_index
                try:
                    kind, data = callbacks[index].queue.get(timeout=QUEUE_GET_TIMEOUT_S)
                    if kind is not None:
                        packets.append((index, kind, data))
                except Empty:
                    pass

            # Process packets already removed from the queues even if Stop
            # arrives now. Never drop a dequeued packet at a stop boundary.
            if any(kind == QUEUE_DATA for _, kind, _ in packets):
                try:
                    specimen_voltage, voltage_pulse, laser_pulse = read_event_metadata(variables, pulse_energy_pj)
                except Exception:
                    for index, kind, data in packets:
                        if kind == QUEUE_DATA:
                            preserve_unattributed_packet(directory, "dld" if index == 0 else "tdc", data)
                    raise
            for index, kind, data in packets:
                if kind == QUEUE_ENDOFMEAS and index == 0 and not stop_event.is_set():
                    retcode = bufdatacb.start_measurement(exposure_time, retries=10)
                    if retcode < 0:
                        raise RuntimeError(f"Surface Concept measurement failed: {device.lib.sc_get_err_msg(retcode)}")
                elif kind == QUEUE_DATA and index == 0:
                    x_bin, y_bin, t_bin = data["dif1"], data["dif2"], data["time"]
                    count = len(x_bin)
                    if not count:
                        continue
                    x = ((x_bin-XYBINSHIFT)*XYFACTOR)*.1
                    y = ((y_bin-XYBINSHIFT)*XYFACTOR)*.1
                    tof = t_bin*TOFFACTOR
                    voltage = np.full(count, specimen_voltage, dtype=np.float64)
                    dld.append({"x": x, "y": y, "t": tof, "x_bin": x_bin,
                                "y_bin": y_bin, "t_bin": t_bin,
                                "voltage": voltage,
                                "voltage_pulse": np.full(count, voltage_pulse),
                                "laser_pulse": np.full(count, laser_pulse),
                                "start_counter": data["start_counter"]})
                    events_detected += count
                    events_detected_tmp += count
                    publish_plot_events(x_plot, y_plot, t_plot, main_v_dc_plot, x, y, tof, voltage)
                    alignment_publisher.append(x, y)
                    laser_alignment_publisher.append(tof)
                elif kind == QUEUE_DATA and index == 1:
                    count = len(data["channel"])
                    raw.append({"channel": data["channel"], "time": data["time"],
                                "tdc_start_counter": data["start_counter"],
                                "voltage_tdc": np.full(count, specimen_voltage),
                                "voltage_pulse_tdc": np.full(count, voltage_pulse),
                                "laser_pulse_tdc": np.full(count, laser_pulse)})
                    raw_signal_detected += count

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
        # Stop callback publication immediately, but finalize the writer before
        # potentially blocking vendor close/deinitialize calls. No disk worker
        # remains to race HDF finalization if the SDK needs forced termination.
        for callback in (bufdatacb, bufdatacb_raw):
            if hasattr(callback, 'stop_publication'):
                callback.stop_publication()
        queued_tail = 0
        for index, callback in enumerate((bufdatacb, bufdatacb_raw)):
            while True:
                try:
                    kind, data = callback.queue.get(block=False)
                except Empty:
                    break
                if kind == QUEUE_DATA:
                    queued_tail += 1
                    try:
                        preserve_unattributed_packet(directory, ('dld' if index == 0 else 'tdc')+'_shutdown', data)
                    except Exception as exc:
                        failure = failure or exc
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
        for callback in (bufdatacb, bufdatacb_raw):
            try:
                callback.close()
            except Exception as exc:
                failure = failure or exc
        try:
            device.deinitialize()
        except Exception as exc:
            failure = failure or exc
        stopped_tail = queued_tail + sum(getattr(callback, 'shutdown_packets', 0)
                                        for callback in (bufdatacb, bufdatacb_raw))
        publication_errors = [getattr(callback, 'publication_error', '') for callback in (bufdatacb, bufdatacb_raw)]
        if stopped_tail or any(publication_errors):
            failure = failure or RuntimeError(
                f'{stopped_tail} detector packet(s) at shutdown retained without voltage attribution; '
                'inspect temp_data/unattributed' +
                (f'; callback preservation failed: {publication_errors}' if any(publication_errors) else ''))
        variables.total_ions = events_detected
        variables.total_raw_signals = raw_signal_detected
        if failure is not None:
            variables.flag_tdc_failure = True
            variables.detector_error = f"{type(failure).__name__}: {failure}"
            print(f"Surface Concept acquisition failed: {variables.detector_error}")
        variables.flag_finished_tdc = True
    return -1 if failure is not None else 0


def experiment_measure(variables, x_plot, y_plot, t_plot, main_v_dc_plot, stop_event):
    return run_experiment_measure(variables, x_plot, y_plot, t_plot, main_v_dc_plot, stop_event)
