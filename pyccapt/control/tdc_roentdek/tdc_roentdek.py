import ctypes
import json
import time
from pathlib import Path

import numpy as np
from numpy.ctypeslib import ndpointer
from pyccapt.calibration.data_tools.roentdek_hexanode import (
    HexanodeCorrection, HexanodeParameters, RoentdekHits, first_particle, reconstruct_hexanode_multihit,
)
from pyccapt.control.apt.detector_chunks import (
    ChunkStream, ChunkWriter, preserve_unattributed_packet, finite_rate_frequency_hz, read_event_metadata,
)
from pyccapt.control.core import runtime
from pyccapt.control.core.shared_ring_buffer import publish_plot_events
from pyccapt.control.nkt_photonics.readback import pulse_energy_pj, experiment_frequency_hz


def hexanode_setup(conf):
    """Hexanode constants and optional correction map from ``config.toml``.

    ``ro_hexanode_ccf`` points to a CoboldPC DAn ``.ccf`` with the detector
    constants (defaults: the values in ``source/DAn-part2.ccf``);
    ``ro_hexanode_correction`` to a JSON map written from
    ``HexanodeCorrection.to_dict()``. Relative paths resolve from the
    project root.
    """
    conf = conf or {}

    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else runtime.project_path(str(path))

    params = HexanodeParameters()
    if conf.get("ro_hexanode_ccf"):
        params = HexanodeParameters.from_ccf(resolve(conf["ro_hexanode_ccf"]))
    correction = None
    if conf.get("ro_hexanode_correction"):
        with open(resolve(conf["ro_hexanode_correction"]), "r", encoding="utf-8") as handle:
            correction = HexanodeCorrection.from_dict(json.load(handle))
    return params, correction


FIRST_HIT_VALUES = 12
MCP_CHANNEL = 7
HIT_DTYPES = {"hit_time_stamp": np.uint64, "hit_channel": np.uint32, "hit_time": np.uint64,
              "hit_voltage": np.float64, "hit_voltage_pulse": np.float64, "hit_laser_pulse": np.float64}


def packet_layout(tdc_lib):
    """``(values_per_event, hits_per_channel)`` of the loaded wrapper DLL.

    Builds from 2026-11 export both and send every hit (up to
    ``hits_per_channel`` per channel); older builds send 12 first-hit values.
    """
    try:
        values, hits = int(tdc_lib.values_per_event()), int(tdc_lib.hits_per_channel())
    except (AttributeError, TypeError, ValueError, OSError):
        return FIRST_HIT_VALUES, 0
    if hits < 1 or values != FIRST_HIT_VALUES + 8 + 8 * hits:
        raise ValueError(f"Unexpected RoentDek DLL packet layout: {values} values, {hits} hits per channel")
    return values, hits


def packet_hits(data, hits_per_channel):
    """``(event, channel, bins)`` of every hit in an SDK packet.

    Rows are ordered by event, channel and arrival. The MCP trigger of every
    event is always included (bin 0), so each event keeps at least one row.
    """
    n = len(data)
    if hits_per_channel:
        counts = np.rint(data[:, FIRST_HIT_VALUES:FIRST_HIT_VALUES + 8]).astype(np.int64)
        start = FIRST_HIT_VALUES + 8
        block = np.rint(data[:, start:start + 8 * hits_per_channel]).astype(np.int64).reshape(n, 8, hits_per_channel)
        present = np.arange(hits_per_channel)[None, None, :] < np.minimum(counts, hits_per_channel)[:, :, None]
    else:
        block = np.rint(data[:, :8]).astype(np.int64)[:, :, None]
        present = block != 0
    present[:, MCP_CHANNEL, 0] = True
    event, channel, slot = np.nonzero(present)
    return event, channel, block[event, channel, slot]


def events_from_packet(data, params, correction=None, *, hits_per_channel=0):
    """Detector columns for one SDK packet of ``(n, values_per_event)`` event rows.

    Columns 0..7 are the first-hit TDC bins (0 = no hit); with a DLL that
    sends every hit (``hits_per_channel > 0``) the hit counts and hits follow
    from column 12 (see ``source/wrapper_read_TDC8HP.cpp``). x/y/t are the
    first (MCP-referenced) particle of the multi-hit hexanode reconstruction,
    which recovers ions whose first hit on a channel is a reflection or noise
    pulse, instead of trusting the DLL's columns 8..10. Events without a
    position get x = y = t = 0 (CoboldPC convention). Channel bins are signed
    (the start channel precedes the MCP trigger) and are stored
    two's-complement in the uint64 schema; read them back with
    ``.view(np.int64)``.

    Returns ``(arrays, valid, hit_arrays)``: the per-event columns, the mask of
    events with a position and the per-hit columns of ``HIT_DTYPES`` (without
    the voltage columns, which the caller fills per packet).
    """
    n = len(data)
    channels = np.rint(data[:, :8]).astype(np.int64)
    event, channel, value = packet_hits(data, hits_per_channel)
    hits = RoentdekHits.from_rows(event, channel, value, n)
    sum_walk = correction.sum_walk if correction is not None and correction.sum_walk is not None else False
    result = first_particle(reconstruct_hexanode_multihit(hits, params, correction, sum_walk=sum_walk), n)
    valid = result["valid"]
    arrays = {"x": np.where(valid, result["x_mm"] / 10., 0.),
              "y": np.where(valid, result["y_mm"] / 10., 0.),
              "t": np.where(valid, np.nan_to_num(result["tof_ns"]), 0.),
              "time_stamp": data[:, 11]}
    arrays.update({f"ch{i}": channels[:, i].view(np.uint64) for i in range(8)})
    hit_arrays = {"hit_time_stamp": data[event, 11], "hit_channel": channel.astype(np.uint32),
                  "hit_time": value.view(np.uint64)}
    return arrays, valid, hit_arrays


class TDC:
    """
    This class sets up the parameters for the TDC and allows users to read experiment TDC values.
    """

    def __init__(self, tdc_lib, buf_size=30000, time_out=300, values_per_event=FIRST_HIT_VALUES):
        """
        Constructor function which initializes function parameters.

        Args:
                tdc_lib (ctypes.CDLL): The TDC library.
                buf_size (int): Buffer size.
                time_out (int): Timeout value.
                values_per_event (int): Packet row length (see ``packet_layout``).
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
        tdc_lib.get_data_tdc_buf.restype = ndpointer(dtype=ctypes.c_double,
                                                     shape=(values_per_event * self.buf_size + 1,))
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


def experiment_measure(variables, x_plot, y_plot, t_plot, main_v_dc_plot, stop_event, conf=None):
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

    try:
        hex_params, hex_correction = hexanode_setup(conf)
    except Exception as exc:
        print(f"RoentDek hexanode configuration is invalid: {exc}")
        variables.detector_error = f"{type(exc).__name__}: {exc}"
        variables.flag_tdc_failure = True
        variables.flag_finished_tdc = True
        return 1
    try:
        values_per_event, hits_per_channel = packet_layout(tdc_lib)
    except ValueError as exc:
        print(exc)
        variables.detector_error = f"{type(exc).__name__}: {exc}"
        variables.flag_tdc_failure = True
        variables.flag_finished_tdc = True
        return 1
    tdc = TDC(tdc_lib, buf_size=30000, time_out=100, values_per_event=values_per_event)

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
    pulse_frequency = finite_rate_frequency_hz(experiment_frequency_hz(variables))
    last_buf_error = None
    writer = stream = hit_stream = None
    failure = None
    from pyccapt.control.apt.laser_alignment_data import LaserAlignmentPublisher
    from pyccapt.control.apt.alignment_vision import AlignmentEventPublisher
    laser_alignment_publisher = LaserAlignmentPublisher(variables)
    alignment_publisher = AlignmentEventPublisher(variables, conf)
    try:
        writer = ChunkWriter(Path(variables.path)/"temp_data"/"chunks", stop_event=stop_event)
        dtypes = {name: np.float64 for name in
                  ("x", "y", "t", "voltage", "voltage_pulse", "laser_pulse")}
        dtypes.update({name: np.uint64 for name in ("time_stamp", *(f"ch{i}" for i in range(8)))})
        stream = ChunkStream(writer, "roentdek", dtypes)
        # Every TDC hit (one row per hit, MCP trigger included) for offline
        # multi-hit reconstruction; finalized into the flat tdc/* table.
        hit_stream = ChunkStream(writer, "roentdek_hits", HIT_DTYPES)
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
                data = np.copy(return_value[1:buffer_length*values_per_event+1].reshape(buffer_length, values_per_event))
                try:
                    dc, pulse, laser = read_event_metadata(variables, pulse_energy_pj)
                except Exception:
                    preserve_unattributed_packet(Path(variables.path)/"temp_data", "roentdek", {"events": data})
                    raise
                voltage = np.full(buffer_length, dc)
                arrays, valid, hit_arrays = events_from_packet(data, hex_params, hex_correction,
                                                               hits_per_channel=hits_per_channel)
                arrays.update({"voltage": voltage,
                               "voltage_pulse": np.full(buffer_length, pulse),
                               "laser_pulse": np.full(buffer_length, laser)})
                n_hits = len(hit_arrays["hit_channel"])
                hit_arrays.update({"hit_voltage": np.full(n_hits, dc), "hit_voltage_pulse": np.full(n_hits, pulse),
                                   "hit_laser_pulse": np.full(n_hits, laser)})
                stream.append(arrays)
                hit_stream.append(hit_arrays)
                events_detected += buffer_length
                raw_signal_detected += buffer_length
                events_detected_tmp += buffer_length
                if valid.any():
                    publish_plot_events(x_plot, y_plot, t_plot, main_v_dc_plot, arrays["x"][valid],
                                        arrays["y"][valid], arrays["t"][valid], voltage[valid])
                    alignment_publisher.append(arrays["x"][valid], arrays["y"][valid])
                    laser_alignment_publisher.append(arrays["t"][valid])
            else:
                time.sleep(.01)

            # Empty packets still advance the rate window. A silent detector
            # must publish zero rather than retain its last nonzero response.
            current_time = time.monotonic()
            elapsed_s = current_time-start_time
            if elapsed_s >= .5:
                try:
                    pulse_frequency = finite_rate_frequency_hz(experiment_frequency_hz(variables), pulse_frequency)
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
        for open_stream in (stream, hit_stream):
            if open_stream is not None:
                try:
                    open_stream.finish()
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
