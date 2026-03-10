# -*- coding: utf-8 -*-
"""
Copyright 2018, 2019, 2021 Surface Concept GmbH

This file is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This file is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this file.  If not, see <https://www.gnu.org/licenses/>.

----------------------------------------------------------------------------

Python wrapper for a subset of the scTDC library built on pythons ctypes
module. The class scTDClib holds the dynamically loaded library object and
provides all low-level functions supported by this wrapper. Additionally,
this module defines structures from the scTDC_types.h header file. The
functions defined in the scTDC class aim to match the original scTDC C API
as close as possible, except for a few cases where they have been changed
for more convenient use (sc_get_err_msg, for example).

A higher-level API is defined in the Device and Pipe classes, which
facilitates usage of 1D,2D,3D histogram pipes, statistics data, and time
histograms of stand-alone TDCs.

A higher-level API for processing of event data is provided via the classes
buffered_data_callbacks_pipe and usercallbacks_pipe. usercallbacks_pipe has
been in existence for a longer time, but it suffers from a high number of
callbacks into python code which makes it rather slow. It is still included
for backwards compatibility.
The buffered_data_callbacks_pipe improves the performance by buffering a
user-defined number of events and invoking callbacks into python code after
filling this buffer and optionally at the end of each measurement.
The buffer size can be tuned large enough to reduce the number of callbacks
into python code. The data offered in the callbacks are in the form of 1D
numpy arrays --- one array for one selected event data field --- ready to be
processed in vectorized mathematical operations.
The buffered_data_callbacks_pipe requires scTDC1 library versions not older
than 1.3010.0.

Following constants are defined for selecting which event data fields shall
be buffered by the buffered_data_callbacks_pipe interface:
SC_DATA_FIELD_SUBDEVICE, SC_DATA_FIELD_CHANNEL, SC_DATA_FIELD_START_COUNTER,
SC_DATA_FIELD_TIME_TAG, SC_DATA_FIELD_DIF1, SC_DATA_FIELD_DIF2,
SC_DATA_FIELD_TIME, SC_DATA_FIELD_MASTER_RST_COUNTER, SC_DATA_FIELD_ADC,
SC_DATA_FIELD_SIGNAL1BIT.

Some remarks concerning low-level interfaces:
* Following constants are defined for pipe types: TDC_HISTO, DLD_IMAGE_XY,
  DLD_IMAGE_XT, DLD_IMAGE_YT, DLD_IMAGE_3D, DLD_SUM_HISTO, STATISTICS,
  TMSTAMP_TDC_HISTO, TDC_STATISTICS, DLD_STATISTICS, USER_CALLBACKS,
  DLD_IMAGE_XY_EXT, BUFFERED_DATA_CALLBACKS.
  These are to be used for the pipe_type parameter in sc_pipe_open2.
* Following constants are defined for bit sizes: BS8, BS16, BS32, BS64.
  These are to be used for the depth field in the structures
  sc_pipe_dld_image_xyt_params_t,
  sc_pipe_tdc_histo_params_t.

----------------------------------------------------------------------------

Additions 2019-09-19 : structures for USER_CALLBACKS pipe
Additions 2019-10-15 : higher-level Device and Pipe classes
Additions 2020-05-14 : added wrapper for a separate library to save DLD
                       events to HDF5
Additions 2021-06-15 : added support for BUFFERED_DATA_CALLBACKS pipe
                       (requires scTDC1 library version >= 1.3010.0)
"""

__version__ = "1.2.0"

import ctypes
import os
from pathlib import Path
import time
import traceback
from contextlib import contextmanager

try:  # most stuff works without numpy
    import numpy as np
except:
    pass


# pipe types
TDC_HISTO = 0
DLD_IMAGE_XY = 1
DLD_IMAGE_XT = 2
DLD_IMAGE_YT = 3
DLD_IMAGE_3D = 4
DLD_SUM_HISTO = 5  # Used to get dld time histogram data
STATISTICS = 6  # Used to get statistics for last exposure
TMSTAMP_TDC_HISTO = 7
TDC_STATISTICS = 8
DLD_STATISTICS = 9
USER_CALLBACKS = 10  # slow in python
DLD_IMAGE_XY_EXT = 11
BUFFERED_DATA_CALLBACKS = 12  # more efficient variant of USER_CALLBACKS

# bitsizes for depth parameter in
#  sc_pipe_dld_image_xyt_params_t  and
#  sc_pipe_tdc_histo_params_t
BS8 = 0
BS16 = 1
BS32 = 2
BS64 = 3

# callback reasons for end-of-measurement callback in conjunction with
# sc_tdc_set_complete_callback2
CBR_COMPLETE = 1
CBR_USER_ABORT = 2
CBR_BUFFER_FULL = 3
CBR_EARLY_NOTIF = 4
CBR_DICT = {CBR_COMPLETE: "Measurement and data processing completed.",
            CBR_USER_ABORT: "Measurement was interrupted by user.",
            CBR_BUFFER_FULL: "Measurement was aborted because buffers were full.",
            CBR_EARLY_NOTIF: "Acquisition finished, not all data processed yet."}

# enum sc_data_field_t
SC_DATA_FIELD_SUBDEVICE = 0x0001
SC_DATA_FIELD_CHANNEL = 0x0002
SC_DATA_FIELD_START_COUNTER = 0x0004
SC_DATA_FIELD_TIME_TAG = 0x0008
SC_DATA_FIELD_DIF1 = 0x0010
SC_DATA_FIELD_DIF2 = 0x0020
SC_DATA_FIELD_TIME = 0x0040
SC_DATA_FIELD_MASTER_RST_COUNTER = 0x0080
SC_DATA_FIELD_ADC = 0x0100
SC_DATA_FIELD_SIGNAL1BIT = 0x0200

_FUNCTYPE = None
if os.name == 'nt':
    _FUNCTYPE = ctypes.WINFUNCTYPE
else:
    _FUNCTYPE = ctypes.CFUNCTYPE


class sc3du_t(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint),
                ("y", ctypes.c_uint),
                ("time", ctypes.c_uint64)]


class sc3d_t(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int),
                ("y", ctypes.c_int),
                ("time", ctypes.c_int64)]


class roi_t(ctypes.Structure):
    _fields_ = [("offset", sc3d_t),
                ("size", sc3du_t)]


ALLOCATORFUNC = _FUNCTYPE(ctypes.c_int, ctypes.POINTER(None),
                          ctypes.POINTER(ctypes.POINTER(None)))


class sc_pipe_dld_image_xyt_params_t(ctypes.Structure):
    _fields_ = [("depth", ctypes.c_int),
                ("channel", ctypes.c_int),
                ("modulo", ctypes.c_uint64),
                ("binning", sc3du_t),
                ("roi", roi_t),
                ("accumulation_ms", ctypes.c_uint),
                ("allocator_owner", ctypes.c_char_p),
                ("allocator_cb", ALLOCATORFUNC)]


class sc_pipe_tdc_histo_params_t(ctypes.Structure):
    _fields_ = [("depth", ctypes.c_int),
                ("channel", ctypes.c_uint),
                ("modulo", ctypes.c_uint64),
                ("binning", ctypes.c_uint),
                ("offset", ctypes.c_uint64),
                ("size", ctypes.c_uint),
                ("accumulation_ms", ctypes.c_uint),
                ("allocator_owner", ctypes.c_char_p),
                ("allocator_cb", ALLOCATORFUNC)]


class sc_pipe_statistics_params_t(ctypes.Structure):
    _fields_ = [("allocator_owner", ctypes.c_char_p),
                ("allocator_cb", ALLOCATORFUNC)]


class statistics_t(ctypes.Structure):
    _fields_ = [("counts_read", ctypes.c_uint * 64),
                ("counts_received", ctypes.c_uint * 64),
                ("events_found", ctypes.c_uint * 4),
                ("events_in_roi", ctypes.c_uint * 4),
                ("events_received", ctypes.c_uint * 4),
                ("counters", ctypes.c_uint * 64),
                ("reserved", ctypes.c_uint * 52)]


class tdc_event_t(ctypes.Structure):
    _fields_ = [("subdevice", ctypes.c_uint),
                ("channel", ctypes.c_uint),
                ("start_counter", ctypes.c_ulonglong),
                ("time_tag", ctypes.c_ulonglong),
                ("time_data", ctypes.c_ulonglong),
                ("sign_counter", ctypes.c_ulonglong)]


class dld_event_t(ctypes.Structure):
    _fields_ = [("start_counter", ctypes.c_ulonglong),
                ("time_tag", ctypes.c_ulonglong),
                ("subdevice", ctypes.c_uint),
                ("channel", ctypes.c_uint),
                ("sum", ctypes.c_ulonglong),
                ("dif1", ctypes.c_ushort),
                ("dif2", ctypes.c_ushort),
                ("master_rst_counter", ctypes.c_uint),
                ("adc", ctypes.c_ushort),
                ("signal1bit", ctypes.c_ushort)]


class sc_pipe_buf_callback_args(ctypes.Structure):
    _fields_ = [("event_index", ctypes.c_ulonglong),
                ("som_indices", ctypes.POINTER(ctypes.c_ulonglong)),
                ("ms_indices", ctypes.POINTER(ctypes.c_ulonglong)),
                ("subdevice", ctypes.POINTER(ctypes.c_uint)),
                ("channel", ctypes.POINTER(ctypes.c_uint)),
                ("start_counter", ctypes.POINTER(ctypes.c_ulonglong)),
                ("time_tag", ctypes.POINTER(ctypes.c_uint)),
                ("dif1", ctypes.POINTER(ctypes.c_uint)),
                ("dif2", ctypes.POINTER(ctypes.c_uint)),
                ("time", ctypes.POINTER(ctypes.c_ulonglong)),
                ("master_rst_counter", ctypes.POINTER(ctypes.c_uint)),
                ("adc", ctypes.POINTER(ctypes.c_int)),
                ("signal1bit", ctypes.POINTER(ctypes.c_ushort)),
                ("som_indices_len", ctypes.c_uint),
                ("ms_indices_len", ctypes.c_uint),
                ("data_len", ctypes.c_uint),
                ("reserved", ctypes.c_char * 12)]


### ----    callbacks   -------------------------------------------------------
# void (*start_of_measure) (void *priv);
CB_STARTMEAS = _FUNCTYPE(None, ctypes.POINTER(None))
# void (*end_of_measure) (void *priv);
CB_ENDMEAS = CB_STARTMEAS
# void (*millisecond_countup) (void *priv);
CB_MILLISEC = CB_STARTMEAS
# void (*statistics) (void *priv, const struct statistics_t *stat);
CB_STATISTICS = _FUNCTYPE(None, ctypes.POINTER(None),
                          ctypes.POINTER(statistics_t))
# void (*tdc_event)
# (void *priv, const struct sc_TdcEvent *const event_array,
#  size_t event_array_len);
CB_TDCEVENT = _FUNCTYPE(None, ctypes.POINTER(None),
                        ctypes.POINTER(tdc_event_t), ctypes.c_size_t)
#   void (*dld_event)
#    (void *priv, const struct sc_DldEvent *const event_array,
#     size_t event_array_len);
CB_DLDEVENT = _FUNCTYPE(None, ctypes.POINTER(None),
                        ctypes.POINTER(dld_event_t), ctypes.c_size_t)
# the following callback type does not belong to the user callbacks, but is
# used in the sc_tdc_set_complete_callback2 function
# void (*cb)(void *, int));
CB_COMPLETE = _FUNCTYPE(None, ctypes.c_void_p, ctypes.c_int)
# the following callback belongs to the BUFFERED_DATA_CALLBACKS pipe
CB_BUFDATA_DATA = _FUNCTYPE(None, ctypes.c_void_p,
                            ctypes.POINTER(sc_pipe_buf_callback_args))
CB_BUFDATA_END_OF_MEAS = _FUNCTYPE(ctypes.c_bool, ctypes.c_void_p)


### ---------------------------------------------------------------------------

class sc_pipe_callbacks(ctypes.Structure):
    _fields_ = [("priv", ctypes.POINTER(None)),
                ("start_of_measure", CB_STARTMEAS),
                ("end_of_measure", CB_ENDMEAS),
                ("millisecond_countup", CB_MILLISEC),
                ("statistics", CB_STATISTICS),
                ("tdc_event", CB_TDCEVENT),
                ("dld_event", CB_DLDEVENT)]


class sc_pipe_callback_params_t(ctypes.Structure):
    _fields_ = [("callbacks", ctypes.POINTER(sc_pipe_callbacks))]


class sc_pipe_buf_callbacks_params_t(ctypes.Structure):
    _fields_ = [("priv", ctypes.POINTER(None)),
                ("data", CB_BUFDATA_DATA),
                ("end_of_measurement", CB_BUFDATA_END_OF_MEAS),
                ("data_field_selection", ctypes.c_uint),
                ("max_buffered_data_len", ctypes.c_uint),
                ("dld_events", ctypes.c_int),
                ("version", ctypes.c_int),
                ("reserved", ctypes.c_ubyte * 24)]


def copy_statistics(s):
    assert (type(s) == statistics_t)
    r = statistics_t()
    ctypes.memmove(ctypes.byref(r), ctypes.byref(s), ctypes.sizeof(s))
    return r


@contextmanager
def _vendor_runtime_environment(module_dir: Path):
    old_cwd = Path.cwd()
    dll_directory = None
    set_dll_directory = None
    if os.name == 'nt':
        if hasattr(os, "add_dll_directory"):
            dll_directory = os.add_dll_directory(str(module_dir))
        try:
            kernel32 = ctypes.windll.kernel32
            kernel32.SetDllDirectoryW.argtypes = [ctypes.c_wchar_p]
            kernel32.SetDllDirectoryW.restype = ctypes.c_bool
            set_dll_directory = kernel32.SetDllDirectoryW
            set_dll_directory(str(module_dir))
        except Exception:
            set_dll_directory = None

    try:
        os.chdir(module_dir)
        yield
    finally:
        os.chdir(old_cwd)
        if set_dll_directory is not None:
            try:
                set_dll_directory(None)
            except Exception:
                pass
        if dll_directory is not None:
            dll_directory.close()


class scTDClib:

    def __init__(self):
        """
        loads the shared library
        """
        self._module_dir = Path(__file__).resolve().parent

        if os.name == 'nt':
            with _vendor_runtime_environment(self._module_dir):
                self.lib = ctypes.WinDLL(str(self._module_dir / "scTDC1.dll"))
                self.lib.sc_tdc_init_inifile.argtypes = [ctypes.c_char_p]
                self.lib.sc_get_err_msg.argtypes = [ctypes.c_int, ctypes.c_char_p]
        else:
            self.lib = ctypes.CDLL("libscTDC.so.1")
        self.lib.sc_tdc_init_inifile.argtypes = [ctypes.c_char_p]
        self.lib.sc_tdc_init_inifile.restype = ctypes.c_int
        self.lib.sc_get_err_msg.argtypes = [ctypes.c_int, ctypes.c_char_p]
        self.lib.sc_get_err_msg.restype = None
        self.lib.sc_tdc_deinit2.argtypes = [ctypes.c_int]
        self.lib.sc_tdc_deinit2.restype = ctypes.c_int
        self.lib.sc_tdc_start_measure2.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.sc_tdc_start_measure2.restype = ctypes.c_int
        self.lib.sc_tdc_interrupt2.argtypes = [ctypes.c_int]
        self.lib.sc_tdc_interrupt2.restype = ctypes.c_int
        self.lib.sc_pipe_open2.argtypes = [ctypes.c_int, ctypes.c_int,
                                           ctypes.POINTER(None)]
        self.lib.sc_pipe_open2.restype = ctypes.c_int
        self.lib.sc_pipe_close2.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.sc_pipe_close2.restype = ctypes.c_int
        self.lib.sc_tdc_get_status2.argtypes = [ctypes.c_int,
                                                ctypes.POINTER(ctypes.c_int)]
        self.lib.sc_tdc_get_status2.restype = ctypes.c_int
        self.lib.sc_pipe_read2.argtypes = \
            [ctypes.c_int, ctypes.c_int,
             ctypes.POINTER(ctypes.POINTER(None)),
             ctypes.c_uint]
        self.lib.sc_pipe_read2.restype = ctypes.c_int
        self.lib.sc_tdc_get_statistics2.argtypes = \
            [ctypes.c_int, ctypes.POINTER(statistics_t)]
        self.lib.sc_tdc_set_complete_callback2.argtypes = \
            [ctypes.c_int, ctypes.c_void_p, CB_COMPLETE]
        self.lib.sc_tdc_set_complete_callback2.restype = ctypes.c_int

    def sc_tdc_init_inifile(self, inifile_path="tdc_gpx3.ini"):
        """
        Initializes the hardware and loads the initial settings reading
        it from the specified ini file. Returns an integer, containing
        a non-negative device iterator on success, or, a negative error code in
        case of failure. The device descriptor is needed for almost all other
        functions.
        """

        ini_path = Path(inifile_path)
        if not ini_path.is_absolute():
            candidate = self._module_dir / ini_path
            if candidate.exists():
                ini_path = candidate
        if os.name == 'nt':
            with _vendor_runtime_environment(self._module_dir):
                return self.lib.sc_tdc_init_inifile(str(ini_path).encode('utf-8'))
        return self.lib.sc_tdc_init_inifile(str(ini_path).encode('utf-8'))

    def sc_get_err_msg(self, errcode):
        """
        Returns an error message to the given error code (signed integer)
        """
        if errcode >= 0:
            return ""
        sbuf = ctypes.create_string_buffer(1024)
        self.lib.sc_get_err_msg(errcode, sbuf)
        if type(sbuf.value) == type(b''):
            return sbuf.value.decode('utf-8')
        else:
            return sbuf.value

    def sc_tdc_deinit2(self, dev_desc):
        """
        Deinitialize the hardware for the given device descriptor
        which was retrieved from sc_tdc_init_inifile. Returns 0 on success
        or negative error code.
        """
        return self.lib.sc_tdc_deinit2(dev_desc)

    def sc_tdc_start_measure2(self, dev_desc, exposure_ms):
        """
        Start a measurement (asynchronously/non-blocking) for the hardware
        indicated by the device descriptor with the given exposure time in
        milliseconds. Returns 0 on success, or negative error code.
        """
        return self.lib.sc_tdc_start_measure2(dev_desc, exposure_ms)

    def sc_tdc_interrupt2(self, dev_desc):
        """Interrupts a measurement asynchronously (non-blocking)"""
        return self.lib.sc_tdc_interrupt2(dev_desc)

    def sc_pipe_open2(self, dev_desc, pipe_type, pipe_params):
        """
        Open a pipe. dev_desc is the device descriptor.
        Here is a table of which pipe_type requires which type of pipe_params:
        DLD_IMAGE_XY     : sc_pipe_dld_image_xyt_params_t
        DLD_IMAGE_XT     : sc_pipe_dld_image_xyt_params_t
        DLD_IMAGE_YT     : sc_pipe_dld_image_xyt_params_t
        DLD_IMAGE_3D     : sc_pipe_dld_image_xyt_params_t
        DLD_SUM_HISTO    : sc_pipe_dld_image_xyt_params_t
        TDC_HISTO        : sc_pipe_tdc_histo_params_t
        STATISTICS       : sc_pipe_statistics_params_t
        This python function expects a structure object for pipe_params. Do not
        pass a pointer-to-structure here. ctypes.addressof(pipe_params) is
        already internally performed for the call of the C function.
        /// Support for other pipe types is a TODO item ///
        Returns an integer containing either the pipe handle (non-negative),
        or, in case of failure, an error code (negative number)
        """
        assert (pipe_type == DLD_IMAGE_XY and
                isinstance(pipe_params, sc_pipe_dld_image_xyt_params_t)) \
               or (pipe_type == DLD_IMAGE_XT and
                   isinstance(pipe_params, sc_pipe_dld_image_xyt_params_t)) \
               or (pipe_type == DLD_IMAGE_YT and
                   isinstance(pipe_params, sc_pipe_dld_image_xyt_params_t)) \
               or (pipe_type == DLD_IMAGE_3D and
                   isinstance(pipe_params, sc_pipe_dld_image_xyt_params_t)) \
               or (pipe_type == DLD_SUM_HISTO and
                   isinstance(pipe_params, sc_pipe_dld_image_xyt_params_t)) \
               or (pipe_type == TDC_HISTO and
                   isinstance(pipe_params, sc_pipe_tdc_histo_params_t)) \
               or (pipe_type == STATISTICS and
                   isinstance(pipe_params, sc_pipe_statistics_params_t)) \
               or (pipe_type == USER_CALLBACKS and
                   isinstance(pipe_params, sc_pipe_callback_params_t)) \
               or (pipe_type == BUFFERED_DATA_CALLBACKS and
                   isinstance(pipe_params, sc_pipe_buf_callbacks_params_t))
        return self.lib.sc_pipe_open2(dev_desc, pipe_type,
                                      ctypes.addressof(pipe_params))

    def sc_pipe_close2(self, dev_desc, pipe_handle):
        """
        Close a pipe. dev_desc is the device descriptor.
        pipe_handle is the pipe handle as returned by sc_pipe_open2.
        Returns 0 on success, or negative error code.
        """
        return self.lib.sc_pipe_close2(dev_desc, pipe_handle)

    def sc_pipe_read2(self, dev_desc, pipe_handle, timeout):
        """
        Read a pipe.
        dev_desc is the device descriptor.
        pipe_handle is the pipe handle as returned by sc_pipe_open2.
        timeout is the timeout after which the function returns when no
        data is available.
        Returns a tuple containing the return code and a ctypes.POINTER to
        the databuffer.
        """
        bufptr = ctypes.POINTER(None)
        retcode = self.lib.sc_pipe_read2(dev_desc, pipe_handle,
                                         ctypes.byref(bufptr), timeout)
        return (retcode, bufptr)

    def sc_tdc_get_status2(self, dev_desc):
        """
        Get status. Returns 0 (idle), 1 (exposure), or negative
        error codes
        """
        statuscode = ctypes.c_int()
        # statuscodeptr = ctypes.POINTER(ctypes.c_int)(statuscode)
        retcode = self.lib.sc_tdc_get_status2(dev_desc,
                                              ctypes.byref(statuscode))
        if retcode < 0:
            return retcode
        else:
            return 1 if statuscode.value == 0 else 0

    def sc_tdc_get_statistics2(self, dev_desc):
        """
        This function is deprecated! Use the statistics pipe, instead.
        This function is kept for older scTDC library versions.
        """
        stat1 = statistics_t()
        retcode = self.lib.sc_tdc_get_statistics2(dev_desc,
                                                  ctypes.byref(stat1))
        if retcode < 0:
            return retcode
        else:
            return stat1

    def sc_tdc_set_complete_callback2(self, dev_desc, privptr, callback):
        """
        Set a measurement complete callback
        """
        return self.lib.sc_tdc_set_complete_callback2(dev_desc, privptr,
                                                      callback)


class scTDC_hdf5lib:
    def __init__(self):
        """
        loads the shared library scTDC_hdf5
        """
        self._module_dir = Path(__file__).resolve().parent
        if os.name == 'nt':
            with _vendor_runtime_environment(self._module_dir):
                self.lib = ctypes.WinDLL(str(self._module_dir / "scTDC_hdf50.dll"))
        else:
            self.lib = ctypes.CDLL("libscTDC_hdf5.so.0")
        l = self.lib
        l.sc_tdc_hdf5_create.argtypes = []
        l.sc_tdc_hdf5_create.restype = ctypes.c_int
        l.sc_tdc_hdf5_destroy.argtypes = [ctypes.c_int]
        l.sc_tdc_hdf5_destroy.restype = ctypes.c_int
        l.sc_tdc_hdf5_connect.argtypes = [ctypes.c_int, ctypes.c_int]
        l.sc_tdc_hdf5_connect.restype = ctypes.c_int
        l.sc_tdc_hdf5_disconnect.argtypes = [ctypes.c_int]
        l.sc_tdc_hdf5_disconnect.restype = ctypes.c_int
        l.sc_tdc_hdf5_setactive.argtypes = [ctypes.c_int, ctypes.c_int]
        l.sc_tdc_hdf5_setactive.restype = ctypes.c_int
        l.sc_tdc_hdf5_isactive.argtypes = [ctypes.c_int]
        l.sc_tdc_hdf5_isactive.restype = ctypes.c_int
        l.sc_tdc_hdf5_cfg_outfile.argtypes = [ctypes.c_int, ctypes.c_char_p]
        l.sc_tdc_hdf5_cfg_outfile.restype = ctypes.c_int
        l.sc_tdc_hdf5_cfg_comment.argtypes = [ctypes.c_int, ctypes.c_char_p]
        l.sc_tdc_hdf5_cfg_comment.restype = ctypes.c_int
        l.sc_tdc_hdf5_cfg_datasel.argtypes = [ctypes.c_int, ctypes.c_uint]
        l.sc_tdc_hdf5_cfg_datasel.restype = ctypes.c_int
        l.sc_tdc_hdf5_version.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        l.sc_tdc_hdf5_version.restype = None
        # abbreviations
        self.sc_tdc_hdf5_create = l.sc_tdc_hdf5_create
        self.sc_tdc_hdf5_destroy = l.sc_tdc_hdf5_destroy
        self.sc_tdc_hdf5_connect = l.sc_tdc_hdf5_connect
        self.sc_tdc_hdf5_disconnect = l.sc_tdc_hdf5_disconnect
        self.sc_tdc_hdf5_setactive = l.sc_tdc_hdf5_setactive
        self.sc_tdc_hdf5_isactive = l.sc_tdc_hdf5_isactive
        self.sc_tdc_hdf5_cfg_outfile = l.sc_tdc_hdf5_cfg_outfile
        self.sc_tdc_hdf5_cfg_comment = l.sc_tdc_hdf5_cfg_comment
        self.sc_tdc_hdf5_cfg_datasel = l.sc_tdc_hdf5_cfg_datasel

    def version(self):
        BUFLEN = 32
        sbuf = ctypes.create_string_buffer(BUFLEN)
        self.lib.sc_tdc_hdf5_version(sbuf, BUFLEN)
        if type(sbuf.value) == bytes:
            return sbuf.value.decode('utf-8')
        else:
            return sbuf.value

    def cfg_outfile(self, objID, filepath):
        self.lib.sc_tdc_hdf5_cfg_outfile(objID, filepath.encode('utf-8'))

    def cfg_comment(self, objID, comment):
        self.lib.sc_tdc_hdf5_cfg_comment(objID, comment.encode('utf-8'))


class buffered_data_callbacks_pipe(object):
    """
    Base class for using the "BUFFERED_DATA_CALLBACKS" interface.
    Requires scTDC1 library version >= 1.3010.0.
    In comparison to the USER_CALLBACKS pipe, this pipe reduces the number of
    callbacks into python, buffering a higher number of events within the
    library before invoking the callbacks. The on_data callback receives a
    dictionary containing 1D numpy arrays where the size of these arrays can be
    as large as specified by the max_buffered_data_len parameter.
    To use this interface, write a class that derives from this class and
    override the methods
      on_data,
      on_end_of_meas
    """

    def __init__(self,
                 lib,
                 dev_desc,
                 data_field_selection=SC_DATA_FIELD_TIME,
                 max_buffered_data_len=(1 << 16),
                 dld_events=True):
        """
        Parameters
        ----------
        lib : scTDClib
          a scTDClib object.
        dev_desc : int
          device descriptor as returned by sc_tdc_init_inifile(...).
        data_field_selection : int, optional
          a 'bitwise or' combination of SC_DATA_FIELD_xyz constants. The
          default is SC_DATA_FIELD_TIME.
        max_buffered_data_len : int, optional
          The number of events that are buffered before invoking the on_data
          callback. Less events can also be received in the on_data callback,
          when the user chooses to return True from the on_end_of_meas
          callback.
          The default is (1<<16).
        dld_events : bool, optional
          if True, receive DLD events. If False, receive TDC events.
          Depending on the configuration in the tdc_gpx3.ini file, only one
          type of events may be available. The default is True.

        Returns
        -------
        None.
        """
        self.dev_desc = dev_desc
        self.lib = lib
        self._pipe_desc = None
        self._open_pipe(data_field_selection, max_buffered_data_len,
                        dld_events)

    def _open_pipe(self, data_field_selection, max_buffered_data_len,
                   dld_events):
        p = sc_pipe_buf_callbacks_params_t()
        p.priv = None
        self._cb_data = CB_BUFDATA_DATA(lambda x, y: self._data_cb(y))
        self._cb_eom = CB_BUFDATA_END_OF_MEAS(lambda x: self.on_end_of_meas())
        p.data = self._cb_data
        p.end_of_measurement = self._cb_eom
        p.data_field_selection = data_field_selection
        p.max_buffered_data_len = max_buffered_data_len
        p.dld_events = 1 if dld_events else 0
        p.version = 0
        reservedlist = [0] * 24
        p.reserved = (ctypes.c_ubyte * 24)(*reservedlist)
        self._pipe_args = p  # prevent garbage collection!
        self._pipe_desc = self.lib.sc_pipe_open2(
            self.dev_desc, BUFFERED_DATA_CALLBACKS, p)

    def _data_cb(self, dptr):
        d = dptr.contents
        x = {"event_index": d.event_index, "data_len": d.data_len}
        f = np.ctypeslib.as_array
        if d.subdevice:
            x["subdevice"] = f(d.subdevice, shape=(d.data_len,))
        if d.channel:
            x["channel"] = f(d.channel, shape=(d.data_len,))
        if d.start_counter:
            x["start_counter"] = f(d.start_counter, shape=(d.data_len,))
        if d.time_tag:
            x["time_tag"] = f(d.time_tag, shape=(d.data_len,))
        if d.dif1:
            x["dif1"] = f(d.dif1, shape=(d.data_len,))
        if d.dif2:
            x["dif2"] = f(d.dif2, shape=(d.data_len,))
        if d.time:
            x["time"] = f(d.time, shape=(d.data_len,))
        if d.master_rst_counter:
            x["master_rst_counter"] = f(d.master_rst_counter, shape=(d.data_len,))
        if d.adc:
            x["adc"] = f(d.adc, shape=(d.data_len,))
        if d.signal1bit:
            x["signal1bit"] = f(d.signal1bit, shape=(d.data_len,))
        if d.som_indices:
            x["som_indices"] = f(d.som_indices, shape=(d.som_indices_len,))
        if d.ms_indices:
            x["ms_indices"] = f(d.ms_indices, shape=(d.ms_indices_len,))
        self.on_data(x)

    def on_data(self, data):
        """
        Override this method to process the data.

        Parameters
        ----------
        data : dict
            A dictionary containing several numpy arrays. The selection of
            arrays depends on the data_field_selection value used during
            initialization of the class.
            The key names in this dictionary, that are always present, are:
              event_index, data_len
            Keywords related to regular event data are:
              subdevice, channel, start_counter, time_tag, dif1, dif2, time,
              master_rst_counter, adc, signal1bit
            Keywords related to indexing arrays are:
              som_indices, ms_indices. These contain event indices where the
              start of a measurement happened, or a millisecond count up
              happened.

        Returns
        -------
        None.

        """
        pass

    def on_end_of_meas(self):
        """
        Override this method to trigger actions at the end of the measurement.
        Do not call methods that start the next measurement from this callback.
        This cannot succeed. Use a signalling mechanism into your main thread,
        instead.

        Returns
        -------
        bool
            True indicates that the pipe should transfer the remaining buffered
            events immediately after returning from this callback.
            False indicates that the pipe may continue buffering the next
            measurements until the max_buffered_data_len threshold is reached.
        """
        return True  # True signalizes that all buffered data shall be emitted

    def close(self):
        """
        Close the pipe.

        Returns
        -------
        None.

        """
        self.lib.sc_pipe_close2(self.dev_desc, self._pipe_desc)

    def start_measurement_sync(self, time_ms):
        """
        Start a measurement and wait until it is finished.

        Parameters
        ----------
        time_ms : int
            the duration of the measurement in milliseconds.

        Returns
        -------
        int
            0 on success or a negative error code.

        """
        retcode = self.lib.sc_tdc_start_measure2(self.dev_desc, time_ms)
        if retcode < 0:
            return retcode
        time.sleep(time_ms / 1000.0)  # sleep expects floating point seconds
        while self.lib.sc_tdc_get_status2(self.dev_desc) == 1:
            time.sleep(0.01)
        return 0

    def start_measurement(self, time_ms, retries=3):
        """
        Start a measurement 'in the background', i.e. don't wait for it
        to finish.

        Parameters
        ----------
        time_ms : int
            the duration of the measurement in milliseconds.
        retries : int
            in an asynchronous scheme of measurement sequences, trying to
            start the next measurement can occasionally result in a
            "NOT READY" error. Often some thread of the scTDC1 library just
            needs a few more cycles to reach the "idle" state again, where
            the start of the next measurement will be accepted.
            The retries parameter specifies how many retries with 0.001 s
            sleeps in between will be made before giving up.

        Returns
        -------
        int
            0 on success or a negative error code.

        """
        while True:
            retcode = self.lib.sc_tdc_start_measure2(self.dev_desc, time_ms)
            if retcode != -11:  # "not ready" error
                return retcode
            retries -= 1
            if retries <= 0:
                return -11
            time.sleep(0.001)


class usercallbacks_pipe(object):
    """
    Base class for user implementations of the "USER_CALLBACKS" interface.
    Derive from this class and override some or all of the methods
      on_start_of_meas,
      on_end_of_meas,
      on_millisecond,
      on_statistics,
      on_tdc_event,
      on_dld_event
    The lib argument in the constructor expects a scTDClib object.
    The dev_desc argument in the constructor expects the device descriptor
    as returned by sc_tdc_init_inifile(...).
    """

    def __init__(self, lib, dev_desc):
        self.dev_desc = dev_desc
        self.lib = lib
        self._pipe_desc = None
        self._open_pipe()

    def _open_pipe(self):
        p = sc_pipe_callbacks()
        p.priv = None
        p.start_of_measure = CB_STARTMEAS(lambda x: self.on_start_of_meas())
        p.end_of_measure = CB_ENDMEAS(lambda x: self.on_end_of_meas())
        p.millisecond_countup = CB_MILLISEC(lambda x: self.on_millisecond())
        p.tdc_event = CB_TDCEVENT(lambda x, y, z: self.on_tdc_event(y, z))
        p.dld_event = CB_DLDEVENT(lambda x, y, z: self.on_dld_event(y, z))
        p.statistics = CB_STATISTICS(lambda x, y: self.on_statistics(y))
        self.struct_callbacks = p
        p2 = sc_pipe_callback_params_t()
        p2.callbacks = ctypes.pointer(self.struct_callbacks)
        self._pipe_args = p
        self._pipe_args2 = p2
        self._pipe_desc = self.lib.sc_pipe_open2(self.dev_desc, USER_CALLBACKS,
                                                 p2)

    def do_measurement(self, time_ms):
        self.lib.sc_tdc_start_measure2(self.dev_desc, time_ms)
        time.sleep(time_ms / 1000.0)  # sleep expects floating point seconds
        while self.lib.sc_tdc_get_status2(self.dev_desc) == 1:
            time.sleep(0.01)

    def on_start_of_meas(self):
        pass

    def on_end_of_meas(self):
        pass

    def on_millisecond(self):
        pass

    def on_statistics(self, stats):
        pass

    def on_tdc_event(self, tdc_events, nr_tdc_events):
        pass

    def on_dld_event(self, dld_events, nr_dld_events):
        pass

    def close(self):
        self.lib.sc_pipe_close2(self.dev_desc, self._pipe_desc)


def _get_voxel_type(depth):
    if depth == BS8:
        return ctypes.c_uint8
    elif depth == BS16:
        return ctypes.c_uint16
    elif depth == BS32:
        return ctypes.c_uint32
    elif depth == BS64:
        return ctypes.c_uint64
    else:
        return -1


# * 0x1 start counter, 0x2 time tag, 0x4 subdevice, 0x8 channel,
# * 0x10 time since start pulse ("sum"), 0x20 "x" detector coordinate ("dif1"),
# * 0x40 "y" detector coordinate ("dif2"), 0x80 master reset counter,
# * 0x100 ADC value, 0x200 signal bit. If this function is not called, the

class HDF5DataSelection:
    STARTCTR = 0x001
    TIMETAG = 0x002
    SUBDEVICE = 0x004
    CHANNEL = 0x008
    TIME = 0x010
    X = 0x020
    Y = 0x040
    MASTER_RESET_CTR = 0x080
    ADC = 0x100
    SIGNALBIT = 0x200

    def __init__(self, value=0):
        self.value = value

    def add(self, value):
        self.value = self.value | value

    def remove(self, value):
        self.value = self.value & (~value)


class Device(object):
    def __init__(self, inifilepath="tdc_gpx3.ini", autoinit=True, lib=None):
        """
        Creates a Device object that will use the specified inifilepath
        during initialization. If autoinit==True, initialize the hardware
        immediately. lib can be specified to reuse an existing scTDClib object.
        """
        self.inifilepath = inifilepath
        self.dev_desc = None
        self.pipes = {}
        self.eomcb = {}  # end of measurement callbacks
        if lib is None:
            self.lib = scTDClib()
        else:
            self.lib = lib
        if autoinit:
            self.initialize()

    def initialize(self):
        """
        Initialize the hardware. Returns a tuple, containing an error code
        and a human-readable error message (zero and empty string in case of
        success).
        """
        retcode = self.lib.sc_tdc_init_inifile(self.inifilepath)
        if retcode < 0:
            return (retcode, self.lib.sc_get_err_msg(retcode))
        else:
            self.dev_desc = retcode
            # register end of measurement callback
            if not hasattr(self, "_eomcbfobj"):
                def _eomcb(privptr, reason):
                    for i in self.eomcb.keys():
                        self.eomcb[i](reason)

                self._eomcbfobj = CB_COMPLETE(_eomcb)  # extend lifetime!
            ret2 = self.lib.sc_tdc_set_complete_callback2(self.dev_desc, None,
                                                          self._eomcbfobj)
            if ret2 < 0:
                print("Registering measurement-complete callback failed")
                print(" message:", self.lib.sc_get_err_msg(ret2))
            return (0, "")

    def deinitialize(self):
        """
        Deinitialize the hardware. Returns a tuple, containing an error
        code and a human-readable error message (zero and empty string in case
        of success).
        """
        if self.dev_desc is None or self.dev_desc < 0:
            return (0, "")  # don't argue if there is nothing to do
        retcode = self.lib.sc_tdc_deinit2(self.dev_desc)
        if retcode < 0:
            return (retcode, self.lib.sc_get_err_msg(retcode))
        else:
            try:
                pipekeys = [x for x in self.pipes.keys()]
                for p in pipekeys:
                    del self.pipes[p]
            except:
                traceback.print_exc()
                traceback.print_stack()
            self.dev_desc = None
            return (0, "")

    def is_initialized(self):
        """ Returns True, if the device is initialized """
        return self.dev_desc is not None

    def do_measurement(self, time_ms=100, synchronous=False):
        """
        Perform a measurement. If synchronous is True, block until the
        measurement has finished. Returns a tuple (0, "") in case of success,
        or a negative error code and a string with the error message.
        """
        retcode = self.lib.sc_tdc_start_measure2(self.dev_desc, time_ms)
        if retcode < 0:
            return (retcode, self.lib.sc_get_err_msg(retcode))
        else:
            if synchronous:
                time.sleep(time_ms / 1000.0)
                while self.lib.sc_tdc_get_status2(self.dev_desc) == 1:
                    time.sleep(0.01)
            return (0, "")

    def interrupt_measurement(self):
        """
        Interrupt a measurement that was started with synchronous=False.
        Returns a tuple (0, "") in case of success, or a negative error code
        and a string with the error message.
        """
        retcode = self.lib.sc_tdc_interrupt2(self.dev_desc)
        return (retcode, self.lib.sc_get_err_msg(retcode))

    def add_end_of_measurement_callback(self, cb):
        """
        Adds a callback function for the end of measurement, the callback
        function needs to accept one argument which indicates the reason for
        callback. Notification via callback is useful if you want to use
        do_measurement(...) with synchronous=False, for example in GUIs that
        need to be responsive during measurement. This functions returns a
        non-negative ID for the callback that can be used for later removal,
        or -1 on error (that would be a bug, though).
        """
        for i in range(len(self.eomcb), -1, -1):
            if not i in self.eomcb:
                self.eomcb[i] = cb
                return i
        return -1

    def remove_end_of_measurement_callback(self, id_of_cb):
        """
        Removes a previously added callback function for the end of
        measurement by its ID. This function returns 0 on success, or -1 if the
        id_of_cb is unknown.
        """
        if id_of_cb in self.eomcb:
            del self.eomcb[id_of_cb]
            return 0
        else:
            return -1

    def _make_new_pipe(self, typestr, par, parent):
        for i in range(1000):
            if i not in self.pipes:
                self.pipes[i] = Pipe(typestr, par, parent)
                return (i, self.pipes[i])
        return None

    def _add_img_pipe_impl(self, depth, modulo, binning, roi, typestr):
        par = sc_pipe_dld_image_xyt_params_t()
        par.depth = depth
        par.channel = -1
        par.modulo = modulo
        par.binning.x = binning[0]
        par.binning.y = binning[1]
        par.binning.time = binning[2]
        par.roi.offset.x = roi[0][0]
        par.roi.offset.y = roi[1][0]
        par.roi.offset.time = roi[2][0]
        par.roi.size.x = roi[0][1]
        par.roi.size.y = roi[1][1]
        par.roi.size.time = roi[2][1]
        par.accumulation_ms = 1 << 31
        pipe = self._make_new_pipe(typestr, par, self)
        if pipe is None:
            return (-1, "Too many pipes open")
        else:
            return pipe  # return id and object

    def add_3d_pipe(self, depth, modulo, binning, roi):
        """
        Adds a 3D pipe (x,y,time) with static buffer. Returns a tuple
        containing a non-negative pipe ID on success and the Pipe object --- or
        negative error code and a string containing the error message. depth is
        either BS8, BS16, BS32, BS64 and determines the voxel bit width. modulo
        unequal zero applies a modulo operation to the time before sorting it
        into the 3D buffer. Note that the modulo value is in digital time units
        _times_ 32. binning is a triple specifying the binning in x, y, and
        time and each entry has to be a power of 2. roi is a triple of
        (offset,size) pairs specifying ranges in the x, y, and time axis. The
        3D buffer is organized such that a point (x,y,time_slice) is addressed
        by x + y * size_x + time_slice * size_x * size_y. When getting a numpy
        array view/copy of the buffer, the 'F' (Fortran) indexing order can be
        chosen, such that the indices are intuitively ordered x, y, time.
        """
        return self._add_img_pipe_impl(depth, modulo, binning, roi,
                                       typestr="3d")

    def add_xy_pipe(self, depth, modulo, binning, roi):
        """
        Adds a 2D pipe (x,y) with static buffer. Returns a tuple
        containing a non-negative pipe ID on success and the Pipe object --- or
        negative error code and a string containing the error message. depth is
        either BS8, BS16, BS32, BS64 and determines the voxel bit width. modulo
        unequal zero has the effect that the time value is transformed by the
        modulo operation and the transformed value is checked whether it is
        within the specified integration range, given in the third element of
        the roi parameter. Note that the modulo value is in digital time units
        _times_ 32. binning is a triple specifying the binning in x, y, and
        time and each entry has to be a power of 2. roi is a triple of
        (offset,size) pairs specifying ranges in the x, y, and time axis. The
        2D buffer is organized such that a point (x,y) is addressed
        by x + y * size_x. When getting a numpy array view/copy of the buffer,
        the 'F' (Fortran) indexing order can be chosen, such that the indices
        are intuitively ordered x, y. The binning in time has an influence only
        on the time units in the roi. The time part in the roi specifies
        the integration range, such that only events inside this time range are
        inserted into the data buffer.
        """
        return self._add_img_pipe_impl(depth, modulo, binning, roi,
                                       typestr="xy")

    def add_xt_pipe(self, depth, modulo, binning, roi):
        """
        Adds a 2D pipe (x,t) with static buffer. Returns a tuple
        containing a non-negative pipe ID on success and the Pipe object --- or
        negative error code and a string containing the error message. depth is
        either BS8, BS16, BS32, BS64 and determines the voxel bit width. modulo
        unequal zero applies a modulo operation to the time before sorting it
        into the 2D buffer. Note that the modulo value is in digital time units
        _times_ 32. binning is a triple specifying the binning in x, y, and
        time and each entry has to be a power of 2. roi is a triple of
        (offset,size) pairs specifying ranges in the x, y, and time axis. The
        2D buffer is organized such that a point (x,time) is addressed by
        x + time * size_x. When getting a numpy array view/copy of the buffer,
        the 'F' (Fortran) indexing order can be chosen, such that the indices
        are intuitively ordered x, time. The binning in y has an influence only
        on the y units in the roi. The y part in the roi specifies the
        integration range, such that only events inside this y range are
        inserted into the data buffer.
        """
        return self._add_img_pipe_impl(depth, modulo, binning, roi,
                                       typestr="xt")

    def add_yt_pipe(self, depth, modulo, binning, roi):
        """
        Adds a 2D pipe (y,t) with static buffer. Returns a tuple
        containing a non-negative pipe ID on success and the Pipe object --- or
        negative error code and a string containing the error message. depth is
        either BS8, BS16, BS32, BS64 and determines the voxel bit width. modulo
        unequal zero applies a modulo operation to the time before sorting it
        into the 2D buffer. Note that the modulo value is in digital time units
        _times_ 32. binning is a triple specifying the binning in x, y, and
        time and each entry has to be a power of 2. roi is a triple of
        (offset,size) pairs specifying ranges in the x, y, and time axis. The
        2D buffer is organized such that a point (y,time) is addressed by
        y + time * size_y. When getting a numpy array view/copy of the buffer,
        the 'F' (Fortran) indexing order can be chosen, such that the indices
        are intuitively ordered y, time. The binning in x has an influence only
        on the x units in the roi. The x part in the roi specifies the
        integration range, such that only events inside this x range are
        inserted into the data buffer.
        """
        return self._add_img_pipe_impl(depth, modulo, binning, roi,
                                       typestr="yt")

    def add_t_pipe(self, depth, modulo, binning, roi):
        """
        Adds a 1D time histogram pipe, integrated over a rectangular region
        in the (x,y) plane (for delay-line detectors) with static buffer.
        Returns a tuple containing a non-negative pipe ID on success and the
        Pipe object --- or negative error code and a string containing the
        error message. depth is either BS8, BS16, BS32, BS64 and determines the
        bit width of the intensity entries. modulo unequal zero applies a
        modulo operation to the time before sorting it into the 1D array. Note
        that the modulo value is in digital time units _times_ 32. binning is a
        triple specifying the binning in x, y, and time and each entry has to
        be a power of 2. roi is a triple of (offset,size) pairs specifying
        ranges in the x, y, and time axis. The buffer is a 1D array of the
        intensity values for all resolved time bins. The binning in x and y has
        an influence only on the x and y units in the roi. The x and y parts in
        the roi specify the integration ranges, such that only events inside
        the x and y ranges are inserted into the data buffer.
        """
        return self._add_img_pipe_impl(depth, modulo, binning, roi,
                                       typestr="t")

    def add_statistics_pipe(self):
        """
        Adds a pipe for statistics data (what is shown in rate meters).
        The statistics data is only updated at the end of each measurement.
        Returns a tuple with a non-negative pipe ID and the Pipe object in case
        of success --- or a negative error code and a string containing the
        error message in case of error."""
        par = sc_pipe_statistics_params_t()
        pipe = self._make_new_pipe("stat", par, self)
        if pipe is None:
            return (-1, "Too many pipes open")
        else:
            return pipe  # return id and object

    def add_tdc_histo_pipe(self, depth, channel, modulo, binning, offset,
                           size):
        """
        Adds a pipe for time histograms from a stand-alone TDC.
        Returns a tuple containing a non-negative pipe ID on success and the
        Pipe object --- or negative error code and a string containing the
        error message. depth is either BS8, BS16, BS32, BS64 and determines the
        bit width of the histogram values. The channel parameter selects a
        channel of the TDC, or accepts all channels if channel==-1. modulo
        unequal zero applies a modulo operation to the time before sorting it
        into the 1D array. Note that the modulo value is in digital time units
        _times_ 32. Binning is an integer number and has to be a power of 2.
        offset and size specify the range of the time axis. The size is equal
        to the number of entries in the histogram. The original time value is
        first transformed by modulo, then by binning, then the offset is
        subtracted and in the last step it is clipped to the size value
        followed by insertion into the histogram."""
        par = sc_pipe_tdc_histo_params_t()
        par.depth = depth
        par.channel = channel
        par.modulo = modulo
        par.binning = binning
        par.offset = offset
        par.size = size
        par.accumulation_ms = 1 << 31
        pipe = self._make_new_pipe("tdch", par, self)
        if pipe is None:
            return (-1, "Too many pipes open")
        else:
            return pipe  # return id and object

    def remove_pipe(self, pipeid):
        """
        Remove a pipe specified by the pipe ID as returned in the first
        entry of a tuple by all add_XXX_pipe functions. Note that deinitialize
        will remove all pipes, as well."""
        try:
            self.pipes[pipeid].close()
        except KeyError:
            return -1
        try:
            del self.pipes[pipeid]
        except KeyError:
            return -1
        return 0

    def hdf5_enable(self):
        """
        Attempts to load the scTDC_hdf5 library which implements event
        streaming to HDF5 files. This should be called only once with an
        initialized Device.
        Returns (True, "") if HDF5 is enabled or (False, error_message)
        """
        if not hasattr(self, 'libh5'):
            try:
                self.libh5 = scTDC_hdf5lib()
            except OSError:
                traceback.print_exc()
                return (False, "Loading of scTDC_hdf5 library failed")
        if self.dev_desc >= 0:
            result = self.libh5.sc_tdc_hdf5_create()
            if result >= 0:
                self.h5obj = result  # store the HDF5 streamer instance handle
                r2 = self.libh5.sc_tdc_hdf5_connect(self.h5obj, self.dev_desc)
                if r2 == 0:
                    return (True, "")
            else:
                return (False, "Instantiation of HDF5 streamer failed")
        else:
            return (False, "Need initialized device to enable hdf5 streaming")

    def hdf5_disable(self):
        """ Disconnects the HDF5 streamer instance from this Device. Does
        not return anything """
        if not hasattr(self, 'libh5') or not hasattr(self, 'h5obj'):
            return
        self.libh5.sc_tdc_hdf5_disconnect(self.h5obj)

    def hdf5_open(self, filepath, comment, data_selection):
        """
        Opens an HDF5 file and activates streaming. Comment is a string
        that will be included as the '/user_comment' attribute in the HDF5
        file. data_selection is an instance of the HDF5DataSelection class and
        specifies the selection of data fields from DLD events to be included
        in the HDF5 file (as separate 1D datasets such that in each dataset,
        the first event is represented by the first element of the dataset
        and so on). (Technical remark: Internally, this involves creation of a
        thread and registration of a USER_CALLBACKS pipe, done by the
        scTDC_hdf5 library.)
        Returns (True, "") if successful or (False, error_message)
        """
        if not hasattr(self, 'libh5') or not hasattr(self, 'h5obj'):
            return (False, "HDF5 streaming is not enabled.")
        if self.libh5.sc_tdc_hdf5_isactive(self.h5obj) > 0:
            return (False, "an HDF5 file is already open")
        self.libh5.cfg_outfile(self.h5obj, filepath)
        self.libh5.cfg_comment(self.h5obj, comment)
        self.libh5.sc_tdc_hdf5_cfg_datasel(self.h5obj, data_selection.value)
        r1 = self.libh5.sc_tdc_hdf5_setactive(self.h5obj, 1)
        if r1 == 1:
            return (True, "")
        elif r1 == 0:
            return (False, "Opening HDF5 file failed")
        else:
            return (False, "Streaming instance became invalid")

    def hdf5_close(self):
        """
        Writes the remaining, internally buffered events and closes the
        HDF5 file. (Technical remark: Internally, this also involves
        termination of a thread and deregistration of a USER_CALLBACKS pipe).
        Returns (True, "") if successful or (False, error_message)
        """
        if not hasattr(self, 'libh5') or not hasattr(self, 'h5obj'):
            return (False, "HDF5 streaming is not enabled.")
        r1 = self.libh5.sc_tdc_hdf5_setactive(self.h5obj, 0)
        if r1 == 0:
            return (True, "")
        else:
            return (False, "Error")

    def hdf5_lib_version(self):
        if not hasattr(self, 'libh5') or not hasattr(self, 'h5obj'):
            return "HDF5 streaming is not enabled."
        else:
            return self.libh5.version()


class Pipe(object):
    """
    Pipe objects are used to let the scTDC library construct 1D, 2D, 3D
    histograms from DLD events occuring during measurements, or to collect
    statistics data after measurements. They are preferably created via the
    Device object and its add_XXX_pipe functions.
    """

    def __init__(self, typestr, par, parent):
        """ Constructs a pipe object. typestr must be one of '3d', 'xy', 'xt',
        'yt', 't', 'stat'. par is of the sc_pipe_dld_image_xyt_params_t type,
        or in case of typestr=='stat', of the sc_pipe_statistics_params_t type.
        parent must be a Device object. Creates the data buffer and opens the
        pipe in the scTDC library for the parent device.
        """
        self.par = par
        self.parent = parent
        self.typestr = typestr
        self.handle = None
        self.buf = None
        self.bufptr = None
        self.bufsize = None
        self.par.allocator_owner = None
        self.pipetypeconst = None
        # ---------------------------------------------------------------------
        # ---     statistics case     -----------------------------------------
        # ---------------------------------------------------------------------
        if self.typestr == 'stat':
            self.pipetypeconst = STATISTICS
            self.par.allocator_cb = self._get_stat_allocator()
            retcode, errmsg = self.reopen()
            if retcode < 0:
                print("scTDC.Pipe.__init__ : error during creation:\n"
                      + "  ({}) {}".format(errmsg, retcode))
            return
        # ---------------------------------------------------------------------
        self.nrvoxels = None
        self.voxeltype = _get_voxel_type(self.par.depth)
        if self.typestr == '3d':
            self.nrvoxels = par.roi.size.x * par.roi.size.y * par.roi.size.time
            self.pipetypeconst = DLD_IMAGE_3D
        elif self.typestr == 'xy':
            self.nrvoxels = par.roi.size.x * par.roi.size.y
            self.pipetypeconst = DLD_IMAGE_XY
        elif self.typestr == 'xt':
            self.nrvoxels = par.roi.size.x * par.roi.size.time
            self.pipetypeconst = DLD_IMAGE_XT
        elif self.typestr == 'yt':
            self.nrvoxels = par.roi.size.y * par.roi.size.time
            self.pipetypeconst = DLD_IMAGE_YT
        elif self.typestr == 't':
            self.nrvoxels = par.roi.size.time
            self.pipetypeconst = DLD_SUM_HISTO
        elif self.typestr == 'tdch':
            self.nrvoxels = par.size
            self.pipetypeconst = TDC_HISTO
        if self.nrvoxels is not None:
            self.par.allocator_cb = self._get_allocator(self.nrvoxels,
                                                        self.voxeltype)
            retcode, errmsg = self.reopen()
            if retcode < 0:
                print("scTDC.Pipe.__init__ : error during creation:\n"
                      + "  ({}) {}".format(errmsg, retcode))

    def _get_allocator(self, nrvoxels, voxeltype):
        if self.buf is not None and self.nrvoxels != nrvoxels:
            return None  # already have a buffer, cannot change
        elif self.buf is None:
            self.buf = (voxeltype * nrvoxels)()  # fixed-size array of voxeltype
            self.bufptr = ctypes.POINTER(type(self.buf))(self.buf)
            self.nrvoxels = nrvoxels
            self.bufsize = nrvoxels * ctypes.sizeof(voxeltype)
        if not hasattr(self, '_allocatorfunc'):
            def _allocator(privptr, bufptrptr):
                bufptrptr[0] = ctypes.cast(self.bufptr, ctypes.c_void_p)
                return 0

            self._allocatorfunc = ALLOCATORFUNC(_allocator)
        return self._allocatorfunc

    def _get_stat_allocator(self):
        if not hasattr(self, '_stat_allocfunc'):
            self.buf = statistics_t()
            self.bufptr = ctypes.POINTER(type(self.buf))(self.buf)
            self.bufsize = ctypes.sizeof(statistics_t)

            def _stat_alloc(privptr, bufptrptr):
                bufptrptr[0] = ctypes.cast(self.bufptr, ctypes.c_void_p)
                return 0

            self._stat_allocfunc = ALLOCATORFUNC(_stat_alloc)
        return self._stat_allocfunc

    def is_open(self):
        """
        Returns True if the pipe is active in the scTDC library (i.e. the
        library will access the data buffer and increment voxels on incoming
        events.
        """
        return self.handle is not None

    def reopen(self, force=False):
        """
        Open a pipe with previous parameters, if currently not open.
        Use force=True, if the pipe had not been explicitly closed, but the
        device had been deinitialized, causing an implicit destruction of the
        pipe (implicit desctruction only happens through low-level API calls,
        whereas Device.deinitialize will close all pipe objects and delete its
        references to them)."""
        if self.handle is not None and not force:
            return
        retcode = self.parent.lib.sc_pipe_open2(
            self.parent.dev_desc, self.pipetypeconst, self.par)
        if retcode < 0:
            return (retcode, self.parent.lib.sc_get_err_msg(retcode))
        else:
            self.handle = retcode
            return (0, "")

    def close(self):
        """
        Closes the pipe such that no events are sorted into the data buffer
        anymore. The data buffer remains unchanged. In that sense, closing acts
        more like setting the pipe inactive and you can reopen it, later. The
        data buffer can only be garbage-collected after deleting the pipe
        object via the parent device and discarding all other references to the
        Pipe object, as well.
        """
        retcode = self.parent.lib.sc_pipe_close2(self.parent.dev_desc,
                                                 self.handle)
        if retcode < 0:
            return (retcode, self.parent.lib.sc_get_err_msg(retcode))
        else:
            self.handle = None
            return (0, "")

    def _reshape(self, a):
        if self.typestr == '3d':
            return np.reshape(a, (self.par.roi.size.x, self.par.roi.size.y,
                                  self.par.roi.size.time), order='F')
        elif self.typestr == 'xy':
            return np.reshape(a, (self.par.roi.size.x, self.par.roi.size.y),
                              order='F')
        elif self.typestr == 'xt':
            return np.reshape(a, (self.par.roi.size.x, self.par.roi.size.time),
                              order='F')
        elif self.typestr == 'yt':
            return np.reshape(a, (self.par.roi.size.y, self.par.roi.size.time),
                              order='F')
        elif self.typestr == 't' or self.typestr == 'tdch':
            # return np.reshape(a, (self.par.roi.size.time,))
            return a  # buffer is already 1D, needs no reshaping

    def get_buffer_view(self):
        """
        For 1D, 2D, 3D pipes, returns a numpy array of the data buffer,
        constructed without copying. As a consequence, changes to the data
        buffer, made by the scTDC library after getting the buffer view, will
        be visible to the numpy array returned from this function.
        The indexing is in Fortran order, i.e. x, y, time.
        If the pipe is a statistics pipe, return the statistics_t object which
        may be modified subsequently by the scTDC.
        """
        if self.typestr == 'stat':
            return self.buf
        else:
            return self._reshape(np.ctypeslib.as_array(self.buf))

    def get_buffer_copy(self):
        """
        For 1D, 2D, 3D pipes, returns a numpy array of a copy of the data
        buffer. The indexing is in Fortran order, i.e. x, y, time.
        If the pipe is a statistics pipe, return a copy of the statistics_t
        object.
        """
        if self.typestr == 'stat':
            return copy_statistics(self.buf)
        else:
            return self._reshape(np.array(self.buf, copy=True))

    def clear(self):
        """
        Set all voxels of the data buffer to zero
        """
        ctypes.memset(self.buf, 0, self.bufsize)
