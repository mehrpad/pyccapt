"""RoentDek TDC8HP (hexanode) acquisition process.

The wrapper DLL (``wrapper_read_TDC8HP_x64.dll``, version 3, source in ``source/``)
reads the TDC exactly like the RoentDek example, writes every TDC group unchanged
to ``output.lmf`` (CoboldPC format, byte-identical to the previous DLLs) and returns
ALL hits of every group (``cnt[n, 8]``, ``tdc[n, 8, 10]`` in TDC bins, 64-bit time
stamp).

The groups are sorted with RoentDek's Resort64c (the library CoboldPC uses),
configured from the Cobold DAn ccf (``roentdek_resort.cfg``); see
``roentdek_resort.py``. If that DLL cannot be loaded, the PyCCAPT reconstruction
(``roentdek_reco.py``) is used and a warning is printed.

Streams written to the chunk directory:

``roentdek``      one row per reconstructed ion: x, y (cm), t (ns; NaN if the group has no
                  laser start), voltage, voltage_pulse, laser_pulse, time_stamp (TDC bins of
                  the ion's group), ro_reco_method (Resort/Cobold method number)
``roentdek_raw``  one row per TDC hit of every group: ro_raw_channel (0..7), ro_raw_time
                  (TDC bins + RAW_TIME_OFFSET, see below), ro_raw_time_stamp, ro_raw_voltage,
                  ro_raw_voltage_pulse, ro_raw_laser_pulse

``tdc/time_data`` is uint64, but RoentDek hit times are signed (relative to the group
trigger, the laser start comes before the MCP). They are stored as
``bins + RAW_TIME_OFFSET`` (2**23, the TDC data are 24-bit signed), so a stored value is
never 0 for a real hit and channel differences are unchanged.
"""
import ctypes
import os
import shutil
import time
from pathlib import Path

import numpy as np
from numpy.ctypeslib import ndpointer
from pyccapt.control.apt.detector_chunks import (
    ChunkStream, ChunkWriter, preserve_unattributed_packet, finite_rate_frequency_hz, read_event_metadata,
)
from pyccapt.control.core.shared_ring_buffer import publish_plot_events
from pyccapt.control.nkt_photonics.readback import pulse_energy_pj, experiment_frequency_hz
from pyccapt.control.tdc_roentdek.roentdek_resort import make_reconstructor

MODULE_DIR = Path(__file__).resolve().parent
DLL_NAME = "wrapper_read_TDC8HP_x64.dll"
LMF_TEMPLATE = "CoboldPC2011R5-2_Header-Template_1TDC8HP.lmf"
MAX_EVENTS_PER_READ = 30000   # TDC groups per DLL call
READ_TIMEOUT_MS = 100         # a DLL call returns after this time also without data
RAW_TIME_OFFSET = 1 << 23     # tdc/time_data = signed TDC bins + RAW_TIME_OFFSET
MM_PER_CM = 10.0
STATUS_EVERY_S = 10.0
RECONSTRUCTION_FILE = "roentdek_reconstruction.txt"   # in temp_data; detector_hdf stores it as dld attribute

ION_DTYPES = {"x": np.float64, "y": np.float64, "t": np.float64, "voltage": np.float64,
              "voltage_pulse": np.float64, "laser_pulse": np.float64, "time_stamp": np.uint64,
              "ro_reco_method": np.int16}
RAW_DTYPES = {"ro_raw_channel": np.uint32, "ro_raw_time": np.uint64, "ro_raw_time_stamp": np.uint64,
              "ro_raw_voltage": np.float64, "ro_raw_voltage_pulse": np.float64, "ro_raw_laser_pulse": np.float64}


def reconstruction_info(reco):
    """one line saying which reconstruction produced dld/x, y, t"""
    from pyccapt.control.tdc_roentdek.roentdek_resort import ResortReconstructor
    if isinstance(reco, ResortReconstructor):
        return "Resort64c: " + reco.describe()
    return ("PyCCAPT fallback reconstruction (roentdek_reco.py, roentdek_calibration.json); "
            "Resort64c was not available")


def record_reconstruction(temp_dir, reco):
    """write reconstruction_info() to temp_data; never stops the acquisition"""
    try:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        (Path(temp_dir)/RECONSTRUCTION_FILE).write_text(reconstruction_info(reco) + "\n")
    except Exception as exc:
        print(f"RoentDek: could not record the reconstruction type: {exc}")


class TDC:
    """ctypes interface of the wrapper DLL (version >= 2: all hits per group)."""

    DIAG_NAMES = ('read_calls', 'read_nonzero', 'read_negative', 'words', 'hit_words', 'hit_words_bad_channel',
                  'hit_words_dropped_over_10', 'group_words', 'rollover_words', 'other_words', 'groups_ok',
                  'groups_without_hit', 'driver_exceptions', 'lmf_events', 'state')

    def __init__(self, tdc_lib, buf_size=MAX_EVENTS_PER_READ, time_out=READ_TIMEOUT_MS):
        self.tdc_lib = tdc_lib
        self.buf_size = int(buf_size)
        self.time_out = int(time_out)
        tdc_lib.Warraper_tdc_new.restype = ctypes.c_void_p
        tdc_lib.Warraper_tdc_new.argtypes = [ctypes.c_int, ctypes.c_int]
        for name in ('init_tdc', 'run_tdc', 'stop_tdc'):
            getattr(tdc_lib, name).argtypes = [ctypes.c_void_p]
            getattr(tdc_lib, name).restype = ctypes.c_int
        try:
            self.version = int(tdc_lib.get_wrapper_version())
        except AttributeError:
            self.version = 1
        if self.version < 2:
            raise RuntimeError(f"{DLL_NAME} is an old wrapper DLL without get_data_tdc_raw; "
                               "build it from source/wrapper_read_TDC8HP.cpp (source/build_wrapper.bat)")
        tdc_lib.get_data_tdc_raw.argtypes = [ctypes.c_void_p, ndpointer(np.uint64, flags='C_CONTIGUOUS'),
                                             ndpointer(np.uint32, flags='C_CONTIGUOUS'),
                                             ndpointer(np.int32, flags='C_CONTIGUOUS'), ctypes.c_int, ctypes.c_int]
        tdc_lib.get_data_tdc_raw.restype = ctypes.c_int
        tdc_lib.get_bin_size_ns.argtypes = [ctypes.c_void_p]
        tdc_lib.get_bin_size_ns.restype = ctypes.c_double
        self.has_diag = self.version >= 3
        if self.has_diag:
            tdc_lib.get_diagnostics.argtypes = [ctypes.c_void_p, ndpointer(np.int64, flags='C_CONTIGUOUS'), ctypes.c_int]
            tdc_lib.get_diagnostics.restype = ctypes.c_int
            tdc_lib.get_last_words.argtypes = [ctypes.c_void_p, ndpointer(np.uint32, flags='C_CONTIGUOUS'), ctypes.c_int]
            tdc_lib.get_last_words.restype = ctypes.c_int
            tdc_lib.get_last_error.argtypes = [ctypes.c_void_p]
            tdc_lib.get_last_error.restype = ctypes.c_char_p
            tdc_lib.get_tdc_info.argtypes = [ctypes.c_void_p, ndpointer(np.float64, flags='C_CONTIGUOUS'), ctypes.c_int]
            tdc_lib.get_tdc_info.restype = ctypes.c_int
        self.obj = tdc_lib.Warraper_tdc_new(self.buf_size, self.time_out)
        self._ts = np.zeros(self.buf_size, np.uint64)
        self._cnt = np.zeros((self.buf_size, 8), np.uint32)
        self._tdc = np.zeros((self.buf_size, 8, 10), np.int32)

    def init_tdc(self):
        return self.tdc_lib.init_tdc(self.obj)

    def run_tdc(self):
        return self.tdc_lib.run_tdc(self.obj)

    def stop_tdc(self):
        return self.tdc_lib.stop_tdc(self.obj)

    def bin_size_ns(self):
        return float(self.tdc_lib.get_bin_size_ns(self.obj))

    def read(self):
        """(time_stamp[n], cnt[n, 8], tdc[n, 8, 10]) as copies, or None if no group arrived"""
        n = self.tdc_lib.get_data_tdc_raw(self.obj, self._ts, self._cnt, self._tdc, self.buf_size, self.time_out)
        if n <= 0:
            return None
        return self._ts[:n].copy(), self._cnt[:n].copy(), self._tdc[:n].copy()

    def diagnostics(self):
        if not self.has_diag:
            return {}
        out = np.zeros(len(self.DIAG_NAMES), np.int64)
        n = self.tdc_lib.get_diagnostics(self.obj, out, len(out))
        return dict(zip(self.DIAG_NAMES[:n], out[:n].tolist()))

    def last_error(self):
        if not self.has_diag:
            return ''
        error = self.tdc_lib.get_last_error(self.obj)
        return error.decode(errors='replace') if error else ''

    def tdc_info(self):
        if not self.has_diag:
            return {}
        out = np.zeros(4, np.float64)
        self.tdc_lib.get_tdc_info(self.obj, out, 4)
        return {'resolution_ns': out[0] * 1e9, 'version': int(out[1]), 'serial': int(out[2]), 'channels': int(out[3])}


def check_tdc_files(directory=MODULE_DIR):
    """global.cfg and the LMF template must be in the DLL's working directory.

    Missing files are restored from ``source/default_files`` with a warning.
    """
    directory = Path(directory)
    ok = True
    for name in (LMF_TEMPLATE, "global.cfg"):
        path = directory / name
        if path.is_file():
            continue
        backup = directory / "source" / "default_files" / name
        if backup.is_file():
            shutil.copy2(backup, path)
            print(f"WARNING: {name} was missing in {directory}; restored from {backup}. Check that it is correct!")
        else:
            print(f"Error: {name} is missing in {directory}; the RoentDek TDC cannot start without it.")
            ok = False
    return ok


def raw_hit_rows(time_stamp, cnt, hits):
    """All hits of all groups as flat rows (group order, channel, hit)."""
    mask = np.arange(10)[None, None, :] < np.minimum(cnt, 10)[:, :, None]
    group, channel, _ = np.nonzero(mask)
    times = hits[mask].astype(np.int64) + RAW_TIME_OFFSET
    return group, channel.astype(np.uint32), times.astype(np.uint64)


def _copy_lmf(experiment_path):
    source = MODULE_DIR / "output.lmf"
    if not source.is_file() or not experiment_path:
        return None
    target = Path(experiment_path) / "output.lmf"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return target
    except OSError as exc:
        print(f"Warning: cannot copy {source} to the experiment folder: {exc}")
        return None


def experiment_measure(variables, x_plot, y_plot, t_plot, main_v_dc_plot, stop_event, conf=None):
    """TDC process: read all hits, reconstruct ions, stream ions + raw hits to chunks."""
    conf = conf or {}
    old_cwd = os.getcwd()
    tdc = None
    tdc_ready = False
    try:
        # The DLL reads global.cfg and the LMF template from the working directory and
        # writes output.lmf there (same as all previous RoentDek wrappers).
        os.chdir(MODULE_DIR)
        if not check_tdc_files(MODULE_DIR):
            raise FileNotFoundError("RoentDek TDC files missing (global.cfg / LMF template)")
        tdc_lib = ctypes.CDLL(str(MODULE_DIR / DLL_NAME))
        tdc = TDC(tdc_lib, buf_size=MAX_EVENTS_PER_READ, time_out=READ_TIMEOUT_MS)
        ret_code = tdc.init_tdc()
        if ret_code != 0:
            raise RuntimeError(f"RoentDek TDC init failed (code {ret_code}): {tdc.last_error()}")
        tdc_ready = True
    except Exception as exc:
        os.chdir(old_cwd)
        print(f"RoentDek TDC could not be started: {exc}")
        variables.flag_finished_tdc = True
        if not getattr(variables, "access_override_enabled", False):
            variables.flag_tdc_failure = True
            variables.detector_error = f"{type(exc).__name__}: {exc}"
            return 1
        print("Access Override is active. Continuing without a RoentDek detector.")
        variables.flag_tdc_failure = False
        return 0

    events_detected = raw_signal_detected = events_detected_tmp = 0
    start_time = time.monotonic()
    status_time = start_time
    pulse_frequency = finite_rate_frequency_hz(experiment_frequency_hz(variables))
    writer = stream = raw_stream = reco = None
    failure = None
    from pyccapt.control.apt.laser_alignment_data import LaserAlignmentPublisher
    from pyccapt.control.apt.alignment_vision import AlignmentEventPublisher
    laser_alignment_publisher = LaserAlignmentPublisher(variables)
    alignment_publisher = AlignmentEventPublisher(variables, conf)
    try:
        bin_ns = tdc.bin_size_ns()
        reco = make_reconstructor(bin_ns=bin_ns if bin_ns > 0 else None,
                                  config_file=conf.get("roentdek_resort_config"))
        bin_ns = float(reco.bin_ns)
        record_reconstruction(Path(variables.path)/"temp_data", reco)
        writer = ChunkWriter(Path(variables.path)/"temp_data"/"chunks", stop_event=stop_event)
        stream = ChunkStream(writer, "roentdek", ION_DTYPES)
        raw_stream = ChunkStream(writer, "roentdek_raw", RAW_DTYPES)
        ret_code = tdc.run_tdc()
        if ret_code != 0:
            raise RuntimeError(f"RoentDek TDC start failed (code {ret_code}): {tdc.last_error()}")
        print("RoentDek TDC is running")
        while not stop_event.is_set() and not variables.flag_stop_tdc:
            writer.check()
            packet = tdc.read()
            if packet is not None:
                time_stamp, cnt, hits = packet
                n_groups = len(time_stamp)
                ions = reco.reconstruct(cnt, hits, time_stamp)
                group, channel, raw_time = raw_hit_rows(time_stamp, cnt, hits)
                try:
                    dc, pulse, laser = read_event_metadata(variables, pulse_energy_pj)
                except Exception:
                    preserve_unattributed_packet(Path(variables.path)/"temp_data", "roentdek",
                                                 {"time_stamp": time_stamp, "cnt": cnt, "tdc": hits})
                    raise
                event = np.asarray(ions["event"], dtype=np.int64)
                n_ions = len(event)
                x_cm = np.asarray(ions["x"], dtype=np.float64) / MM_PER_CM
                y_cm = np.asarray(ions["y"], dtype=np.float64) / MM_PER_CM
                tof = np.asarray(ions["t"], dtype=np.float64)
                voltage = np.full(n_ions, dc)
                stream.append({"x": x_cm, "y": y_cm, "t": tof, "voltage": voltage,
                               "voltage_pulse": np.full(n_ions, pulse), "laser_pulse": np.full(n_ions, laser),
                               "time_stamp": time_stamp[event], "ro_reco_method": ions["method"]})
                n_hits = len(group)
                raw_stream.append({"ro_raw_channel": channel, "ro_raw_time": raw_time,
                                   "ro_raw_time_stamp": time_stamp[group],
                                   "ro_raw_voltage": np.full(n_hits, dc),
                                   "ro_raw_voltage_pulse": np.full(n_hits, pulse),
                                   "ro_raw_laser_pulse": np.full(n_hits, laser)})
                events_detected += n_ions
                raw_signal_detected += n_groups
                events_detected_tmp += n_ions
                if n_ions:
                    publish_plot_events(x_plot, y_plot, t_plot, main_v_dc_plot, x_cm, y_cm, tof, voltage)
                    alignment_publisher.append(x_cm, y_cm)
                    laser_alignment_publisher.append(tof)
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
            if current_time - status_time >= STATUS_EVERY_S:
                status_time = current_time
                print(f"[RoentDek] {reco.stats_text()}")
                for warning in reco.signal_warnings():
                    print(f"[RoentDek] WARNING - check cable/CFD/laser trigger: {warning}")
    except Exception as exc:
        failure = exc
    finally:
        for closable in (stream, raw_stream):
            if closable is not None:
                try:
                    closable.finish()
                except Exception as exc:
                    failure = failure or exc
        if writer is not None:
            try:
                writer.close()
            except Exception as exc:
                failure = failure or exc
        if tdc is not None and tdc_ready:
            try:
                tdc.stop_tdc()   # closes output.lmf
            except Exception as exc:
                failure = failure or exc
            diag = tdc.diagnostics()
            if diag:
                print(f"[RoentDek] DLL counters: {diag}")
            if tdc.last_error():
                print(f"[RoentDek] last driver error: {tdc.last_error()}")
        if reco is not None:
            print(f"[RoentDek] final: {reco.stats_text()}")
        os.chdir(old_cwd)
        # only the LMF of THIS run (an old output.lmf must not be copied after a failed start)
        copied = _copy_lmf(getattr(variables, "path", None)) if tdc_ready else None
        if copied is not None:
            print(f"RoentDek raw data (CoboldPC LMF) saved: {copied}")
        variables.total_ions = events_detected
        variables.total_raw_signals = raw_signal_detected
        if failure is not None:
            variables.flag_tdc_failure = True
            variables.detector_error = f"{type(failure).__name__}: {failure}"
            print(f"RoentDek acquisition failed: {variables.detector_error}")
        variables.flag_finished_tdc = True
    return -1 if failure is not None else 0
