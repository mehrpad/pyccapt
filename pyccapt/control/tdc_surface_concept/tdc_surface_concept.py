import multiprocessing as mp
import os
import time
from queue import Empty, Queue

import numpy as np

# local imports
from pyccapt.control.core import runtime as _runtime
from pyccapt.control.devices import initialize_devices
from pyccapt.control.tdc_surface_concept import scTDC

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

        self.queue = Queue()
        self.end_of_meas = False

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
        self.queue.put((QUEUE_DATA, dcopy))
        if self.end_of_meas:
            self.end_of_meas = False
            self.queue.put((QUEUE_ENDOFMEAS, None))

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

        chunk_id, path, chunk_data = task  # Extract data
        try:
            for key, data in chunk_data.items():
	            target_dtype = CHUNK_DTYPES.get(key)
	            if target_dtype is not None:
		            arr = np.asarray(data, dtype=target_dtype)
	            else:
		            arr = np.array(data)
	            np.save(os.path.join(path, f"chunks/{key}_chunk_{chunk_id}.npy"), arr)
        except Exception as e:
            print(f"Error saving chunk {chunk_id}: {e}")

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

    # DLD data
    xx = []
    yy = []
    tt = []
    voltage_data = []
    voltage_pulse_data = []
    laser_pulse_data = []
    start_counter = []

    # The binning of DLD events
    xx_list_bin = []
    yy_list_bin = []
    tt_list_bin = []

    # TDC data (Raw data)
    channel_data = []
    time_data = []
    tdc_start_counter = []
    voltage_data_tdc = []
    voltage_pulse_data_tdc = []
    laser_pulse_data_tdc = []

    retcode = bufdatacb.start_measurement(exposure_time)
    if errorcheck(device, bufdatacb, bufdatacb_raw, retcode) < 0:
        print("Error during read:", retcode, device.lib.sc_get_err_msg(retcode))
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

    loop_time = 1 / variables.ex_freq
    events_detected = 0
    events_detected_tmp = 0
    raw_signal_detected = 0
    start_time = time.time()
    pulse_frequency = variables.pulse_frequency * 1000
    loop_counter = 0
    loop_delay_counter = 0

    # DLD events and raw/TDC signals are independent streams of DIFFERENT
    # length, so they are chunked independently -- each on its own
    # CHUNK_SIZE threshold and its own chunk-id sequence. This keeps the
    # faster-growing raw stream flushed to disk on time instead of
    # accumulating in RAM (OOM) and being truncated to the DLD count (data
    # loss). The two streams write disjoint chunk stems (e.g. x_chunk_* vs
    # channel_chunk_*), so reusing the same id counter values never collides.
    dld_chunk_id = 0
    tdc_chunk_id = 0
    save_queue = mp.Queue()
    save_process = mp.Process(target=save_chunk_worker, args=(save_queue,))
    save_process.start()
    path = variables.path + "/temp_data/"
    # Create folder to save the data
    if not os.path.isdir(path):
        os.makedirs(path, mode=0o777, exist_ok=True)
    if not os.path.isdir(path + "chunks/"):
        os.makedirs(path + "chunks/", mode=0o777, exist_ok=True)

    _last_loop_warn = None  # dedup transient inner-loop errors
    while not stop_event.is_set():
        start_time_loop = time.time()

        # Two-queue gets used to be unconditional .get() calls back to
        # back.  Because the DLD callback fires only on completed DLD
        # events while the raw callback fires on every TDC channel hit,
        # the queues have very different rates - whichever lagged would
        # block the loop indefinitely and starve stop_event polling.
        # Now: timed gets, no-data sentinels, stop_event re-checked.
        try:
            eventtype, data = bufdatacb.queue.get(timeout=QUEUE_GET_TIMEOUT_S)
        except Empty:
            eventtype, data = None, None
        if stop_event.is_set():
            break
        try:
            eventtype_raw, data_raw = bufdatacb_raw.queue.get(timeout=QUEUE_GET_TIMEOUT_S)
        except Empty:
            eventtype_raw, data_raw = None, None
        if stop_event.is_set():
            break

        try:
            specimen_voltage = variables.specimen_voltage
            voltage_pulse = variables.pulse_voltage
            laser_pulse = variables.laser_pulse_energy
        except Exception as exc:
            # Manager IPC can transiently fail under heavy load; skip
            # this iteration but keep the loop alive.
            if str(exc) != _last_loop_warn:
                _last_loop_warn = str(exc)
                print(f"TDC: variable read failed (non-fatal): {exc}")
            continue

        if eventtype == QUEUE_DATA:
            # correct for binning of surface concept
            xx_dif = data["dif1"]
            length = len(xx_dif)
            if length > 0:
                events_detected_tmp += length
                events_detected += length
                yy_dif = data["dif2"]
                tt_dif = data["time"]
                start_counter.extend(data["start_counter"].tolist())
                xx_tmp = ((xx_dif - XYBINSHIFT) * XYFACTOR) * 0.1  # from mm to in cm by dividing by 10
                yy_tmp = ((yy_dif - XYBINSHIFT) * XYFACTOR) * 0.1  # from mm to in cm by dividing by 10
                tt_tmp = tt_dif * TOFFACTOR  # in ns
                dc_voltage_tmp = np.tile(specimen_voltage, len(xx_tmp))

                # Push into the shared-memory ring buffers (one per signal).
                # Append is non-blocking, bounded; the visualization
                # subprocess drains the rings on its own cadence.
                x_plot.write(xx_tmp)
                y_plot.write(yy_tmp)
                t_plot.write(tt_tmp)
                main_v_dc_plot.write(dc_voltage_tmp)

                # change to list
                xx_tmp = xx_tmp.tolist()
                yy_tmp = yy_tmp.tolist()
                tt_tmp = tt_tmp.tolist()

                # extend the main list with the new data
                xx.extend(xx_tmp)
                yy.extend(yy_tmp)
                tt.extend(tt_tmp)
                dc_voltage_tmp = dc_voltage_tmp.tolist()
                p_voltage_tmp = np.tile(voltage_pulse, len(xx_tmp)).tolist()
                p_laser_tmp = np.tile(laser_pulse, len(xx_tmp)).tolist()
                voltage_data.extend(dc_voltage_tmp)
                voltage_pulse_data.extend(p_voltage_tmp)
                laser_pulse_data.extend(p_laser_tmp)

                # The binning of DLD events
                xx_list_bin.extend(xx_dif.tolist())
                yy_list_bin.extend(yy_dif.tolist())
                tt_list_bin.extend(tt_dif.tolist())

        if eventtype_raw == QUEUE_DATA:
            channel_data_tmp = data_raw["channel"].tolist()
            if len(channel_data_tmp) > 0:
                raw_signal_detected += len(channel_data_tmp)
                tdc_start_counter.extend(data_raw["start_counter"].tolist())
                time_data.extend(data_raw["time"].tolist())
                # raw data
                channel_data.extend(channel_data_tmp)
                voltage_data_tdc.extend((np.tile(specimen_voltage, len(channel_data_tmp))).tolist())
                voltage_pulse_data_tdc.extend((np.tile(voltage_pulse, len(channel_data_tmp))).tolist())
                laser_pulse_data_tdc.extend((np.tile(laser_pulse, len(channel_data_tmp))).tolist())

        if eventtype == QUEUE_ENDOFMEAS:
            retcode = bufdatacb.start_measurement(exposure_time, retries=10)
            if retcode < 0:
                print("Error during read (error code: %s - error msg: %s):" % (retcode, device.lib.sc_get_err_msg(retcode)))
                variables.flag_tdc_failure = True
                # Clean teardown - the post-loop block at the end runs
                # bufdatacb.close() etc.  We just break out here.
                break

        # Calculate the detection rate
        current_time = time.time()
        if current_time - start_time >= 0.5:
            # Re-read pulse_frequency every interval so the rate calc
            # stays correct if the user changes it mid-run.  Guard
            # against zero (would divide by zero on first chunk after a
            # bad value).
            try:
                live_pulse_frequency = max(float(variables.pulse_frequency) * 1000.0, 1.0)
            except Exception:
                live_pulse_frequency = pulse_frequency
            pulse_frequency = live_pulse_frequency
            detection_rate = events_detected_tmp * 100 / pulse_frequency
            variables.detection_rate_current = detection_rate * 2  # rate per second
            variables.detection_rate_current_plot = detection_rate * 2
            variables.total_ions = events_detected
            variables.total_raw_signals = raw_signal_detected
            events_detected_tmp = 0
            start_time = current_time

        # Flush the DLD stream when it reaches a full chunk. We do NOT
        # mirror into the Manager here: the data is safe on disk under
        # temp_data/chunks/ and hdf_creator reads the chunk files directly.
        # (The old per-chunk extend_to shipped the whole growing list back
        # through the Manager every flush -- O(N) IPC -- which throttled
        # acquisition and OOM'd the Manager on large runs.)
        if len(xx) >= CHUNK_SIZE:
	        dld_chunk_id += 1
	        dld_chunk = {
                "x_bin": xx_list_bin[:CHUNK_SIZE],
                "x": xx[:CHUNK_SIZE],
                "y_bin": yy_list_bin[:CHUNK_SIZE],
                "y": yy[:CHUNK_SIZE],
                "t_bin": tt_list_bin[:CHUNK_SIZE],
                "t": tt[:CHUNK_SIZE],
                "voltage": voltage_data[:CHUNK_SIZE],
                "voltage_pulse": voltage_pulse_data[:CHUNK_SIZE],
                "laser_pulse": laser_pulse_data[:CHUNK_SIZE],
                "start_counter": start_counter[:CHUNK_SIZE],
	        }
	        save_queue.put((dld_chunk_id, path, dld_chunk))
	        del xx[:CHUNK_SIZE], yy[:CHUNK_SIZE], tt[:CHUNK_SIZE]
	        del xx_list_bin[:CHUNK_SIZE], yy_list_bin[:CHUNK_SIZE], tt_list_bin[:CHUNK_SIZE]
	        del voltage_data[:CHUNK_SIZE], voltage_pulse_data[:CHUNK_SIZE], laser_pulse_data[:CHUNK_SIZE]
	        del start_counter[:CHUNK_SIZE]

        # Flush the raw/TDC stream INDEPENDENTLY -- it grows much faster than
        # the DLD stream (many raw hits per DLD event), so it reaches
        # CHUNK_SIZE far more often. Gating this on len(xx) (the old code)
        # under-flushed the raw lists: they grew unbounded (OOM) and only
        # CHUNK_SIZE of the millions of raw rows were ever written per chunk.
        if len(channel_data) >= CHUNK_SIZE:
	        tdc_chunk_id += 1
	        tdc_chunk = {
                "channel": channel_data[:CHUNK_SIZE],
                "time": time_data[:CHUNK_SIZE],
                "tdc_start_counter": tdc_start_counter[:CHUNK_SIZE],
                "voltage_tdc": voltage_data_tdc[:CHUNK_SIZE],
                "voltage_pulse_tdc": voltage_pulse_data_tdc[:CHUNK_SIZE],
                "laser_pulse_tdc": laser_pulse_data_tdc[:CHUNK_SIZE],
            }
	        save_queue.put((tdc_chunk_id, path, tdc_chunk))
	        del channel_data[:CHUNK_SIZE], time_data[:CHUNK_SIZE], tdc_start_counter[:CHUNK_SIZE]
	        del voltage_data_tdc[:CHUNK_SIZE], voltage_pulse_data_tdc[:CHUNK_SIZE]
	        del laser_pulse_data_tdc[:CHUNK_SIZE]

        if time.time() - start_time_loop > loop_time:
            loop_delay_counter += 1
        loop_counter += 1

    print(
        "TDC process: for %s times loop time took longer than %s second" % (loop_delay_counter, loop_time),
        'out of %s iterations' % loop_counter,
    )
    variables.total_ions = events_detected
    variables.total_raw_signals = raw_signal_detected
    print("TDC Measurement stopped")

    # --- Teardown instrumentation -------------------------------------
    # Lightweight stdout logging (captured + timestamped by the parent
    # process) so a slow or hanging shutdown can be diagnosed -- this is
    # where the TDC subprocess "won't join". It runs AFTER the
    # acquisition loop, so it never touches the live hot path. flush=True
    # so each line survives even if the process is later force-joined.
    def _tlog(msg):
	    print("[TDC teardown] %s" % msg, flush=True)

    _teardown_t0 = time.time()
    _tlog(
	    "begin: dld_chunk_id=%d (DLD residual=%d), tdc_chunk_id=%d (raw residual=%d), "
	    "stop_event=%s, tdc_failure=%s"
	    % (dld_chunk_id, len(xx), tdc_chunk_id, len(channel_data),
	       stop_event.is_set(), getattr(variables, "flag_tdc_failure", "?"))
    )

    # Final residual chunks -- whatever is still in the in-memory lists after
    # the last in-loop flush. Each stream is flushed independently (mirroring
    # the in-loop logic) and in FULL (no [:CHUNK_SIZE] truncation), so the
    # faster raw stream is saved completely. A stream that already produced
    # chunks during the run just appends its final partial chunk here.
    if dld_chunk_id > 0 and len(xx) > 0:
	    dld_chunk_id += 1
	    save_queue.put((dld_chunk_id, path, {
		    "x_bin": xx_list_bin, "x": xx,
		    "y_bin": yy_list_bin, "y": yy,
		    "t_bin": tt_list_bin, "t": tt,
		    "voltage": voltage_data, "voltage_pulse": voltage_pulse_data,
		    "laser_pulse": laser_pulse_data, "start_counter": start_counter,
	    }))
	    _tlog("queued residual DLD chunk #%d (%d rows)" % (dld_chunk_id, len(xx)))
    if tdc_chunk_id > 0 and len(channel_data) > 0:
	    tdc_chunk_id += 1
	    save_queue.put((tdc_chunk_id, path, {
		    "channel": channel_data, "time": time_data,
		    "tdc_start_counter": tdc_start_counter,
		    "voltage_tdc": voltage_data_tdc,
		    "voltage_pulse_tdc": voltage_pulse_data_tdc,
		    "laser_pulse_tdc": laser_pulse_data_tdc,
	    }))
	    _tlog("queued residual raw/TDC chunk #%d (%d rows)" % (tdc_chunk_id, len(channel_data)))

    _tlog("waiting for save worker to drain queue (save_process.join)...")
    _save_join_t = time.time()
    save_queue.put(None)
    save_process.join()
    _tlog("save worker joined in %.1fs" % (time.time() - _save_join_t))

    # Per-stream fallback for a stream that NEVER reached a single full chunk
    # (its chunk-id is still 0): no chunk files exist for it, so hdf_creator
    # falls back to these in-memory lists -- mirror them into the Manager and
    # also drop a flat .npy as a manual-recovery copy. Done PER STREAM so a
    # run that chunked the fast raw stream but not the smaller DLD stream
    # (or vice versa) only mirrors the SMALL un-chunked stream, never the
    # large one already on disk -- avoiding the O(N) Manager blow-up.
    if dld_chunk_id == 0 and len(xx) > 0:
	    _tlog("DLD fallback: mirroring %d rows into Manager + flat .npy ..." % len(xx))
	    _t = time.time()
	    try:
            variables.extend_to('x', xx)
            variables.extend_to('y', yy)
            variables.extend_to('t', tt)
            variables.extend_to('dld_start_counter', start_counter)
            variables.extend_to('main_v_dc_dld', voltage_data)
            variables.extend_to('main_v_p_dld', voltage_pulse_data)
            variables.extend_to('main_l_p_dld', laser_pulse_data)
	    except Exception as exc:
		    print(f"TDC: DLD residual extend_to failed (non-fatal): {exc}")
        try:
            np.save(variables.path + "/temp_data/x.npy", np.array(xx))
            np.save(variables.path + "/temp_data/y.npy", np.array(yy))
            np.save(variables.path + "/temp_data/t.npy", np.array(tt))
            np.save(variables.path + "/temp_data/voltage.npy", np.array(voltage_data))
            np.save(variables.path + "/temp_data/voltage_pulse.npy", np.array(voltage_pulse_data))
            np.save(variables.path + "/temp_data/laser_pulse.npy", np.array(laser_pulse_data))
            np.save(variables.path + "/temp_data/start_counter.npy", np.array(start_counter))
            np.save(variables.path + "/temp_data/x_bin.npy", np.array(xx_list_bin))
            np.save(variables.path + "/temp_data/y_bin.npy", np.array(yy_list_bin))
            np.save(variables.path + "/temp_data/t_bin.npy", np.array(tt_list_bin))
        except Exception as exc:
	        print(f"TDC: DLD single-shot .npy save failed: {exc}")
	    _tlog("DLD fallback done in %.1fs" % (time.time() - _t))

    if tdc_chunk_id == 0 and len(channel_data) > 0:
	    _tlog("raw/TDC fallback: mirroring %d rows into Manager + flat .npy ..." % len(channel_data))
	    _t = time.time()
	    try:
		    variables.extend_to('channel', channel_data)
		    variables.extend_to('time_data', time_data)
		    variables.extend_to('tdc_start_counter', tdc_start_counter)
		    variables.extend_to('main_v_dc_tdc', voltage_data_tdc)
		    variables.extend_to('main_v_p_tdc', voltage_pulse_data_tdc)
		    variables.extend_to('main_l_p_tdc', laser_pulse_data_tdc)
	    except Exception as exc:
		    print(f"TDC: raw residual extend_to failed (non-fatal): {exc}")
	    try:
            np.save(variables.path + "/temp_data/channel.npy", np.array(channel_data))
            np.save(variables.path + "/temp_data/time.npy", np.array(time_data))
            # Canonical name matches the chunk stem (was the misnamed
            # "main_raw_counter.npy"; the recover script reads both).
            np.save(variables.path + "/temp_data/tdc_start_counter.npy", np.array(tdc_start_counter))
            np.save(variables.path + "/temp_data/voltage_tdc.npy", np.array(voltage_data_tdc))
            np.save(variables.path + "/temp_data/voltage_pulse_tdc.npy", np.array(voltage_pulse_data_tdc))
            np.save(variables.path + "/temp_data/laser_pulse_tdc.npy", np.array(laser_pulse_data_tdc))
        except Exception as exc:
	        print(f"TDC: raw single-shot .npy save failed: {exc}")
	    _tlog("raw/TDC fallback done in %.1fs" % (time.time() - _t))

    print("data saved in share variables")
    time.sleep(0.1)
    _tlog("closing data-callback buffers + deinitializing TDC device...")
    _dev_t = time.time()
    bufdatacb.close()
    bufdatacb_raw.close()
    device.deinitialize()
    _tlog("device deinitialized in %.1fs" % (time.time() - _dev_t))

    variables.flag_finished_tdc = True
    _tlog("flag_finished_tdc set; total teardown %.1fs" % (time.time() - _teardown_t0))

    return 0


def experiment_measure(variables, x_plot, y_plot, t_plot, main_v_dc_plot, stop_event):

    run_experiment_measure(variables, x_plot, y_plot, t_plot, main_v_dc_plot, stop_event)
