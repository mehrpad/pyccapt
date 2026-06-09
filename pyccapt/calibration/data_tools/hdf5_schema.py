"""Single source of truth for the pyccapt HDF5 dataset schema.

The control side (``pyccapt.control.core.hdf5_creator``) WRITES the
``/dld``, ``/tdc`` and ``/hsd`` groups; the calibration side
(``pyccapt.calibration.data_tools.data_loadcrop``) READS them. Those
two layers historically drifted -- different group names, 1-D vs 2-D
shapes, dtype changes -- with no shared definition, so a writer change
could silently break the reader (and vice versa) without any single
place to check the contract.

This module is that single place. It documents:

* the canonical HDF5 group paths each detector mode writes,
* the per-event column ORDER the reader assembles into the in-memory
  DataFrame (must match ``create_pandas_dataframe`` in data_loadcrop),
* the reader's accepted ALIASES for groups that were renamed across
  pyccapt versions, and
* a small ``stack_columns`` helper that assembles per-column arrays of
  EITHER shape ``(N,)`` or ``(N, 1)`` into a single ``(N, k)`` matrix,
  so the reader no longer depends on the writer emitting one specific
  shape.

Keep this in sync with both hdf5_creator (writer) and
create_pandas_dataframe (reader) whenever the on-disk layout changes.
"""
from __future__ import annotations

import numpy as np

# --- DLD (single-hit detector) -------------------------------------------
# Reader output column order (see create_pandas_dataframe mode='dld').
DLD_COLUMNS = (
    "high_voltage (V)",
    "pulse_v (V)",
    "pulse_l (pJ)",
    "start_counter",
    "t (ns)",
    "x_det (cm)",
    "y_det (cm)",
)

# Canonical HDF5 group for each reader column, plus accepted aliases the
# reader falls back to (older pyccapt files / dld-vs-tdc naming). The
# first present group wins; ``None`` means "synthesize zeros".
DLD_GROUP_ALIASES = {
    "high_voltage (V)": ("dld/high_voltage",),
    "pulse_v (V)": ("dld/pulse", "dld/voltage_pulse", "dld/pulse_voltage"),
    "pulse_l (pJ)": ("dld/laser_pulse", "dld/laser_intensity"),
    "start_counter": ("dld/start_counter",),
    "t (ns)": ("dld/t",),
    "x_det (cm)": ("dld/x",),
    "y_det (cm)": ("dld/y",),
}

# --- TDC (multi-hit, Surface Concept / RoentDek) -------------------------
TDC_COLUMNS = (
    "channel",
    "start_counter",
    "high_voltage (V)",
    "pulse_v (V)",
    "pulse_l (pJ)",
    "time_data",
)

TDC_GROUP_ALIASES = {
    "channel": ("tdc/channel",),
    "start_counter": ("tdc/start_counter",),
    "high_voltage (V)": ("tdc/high_voltage",),
    "pulse_v (V)": ("tdc/pulse", "tdc/voltage_pulse"),
    "pulse_l (pJ)": ("tdc/laser_pulse",),
    "time_data": ("tdc/time_data",),
}

# --- HSD (DRS digitizer waveform) ----------------------------------------
# Written by hdf5_creator as float32 (GetTime ns / GetWave mV are signed
# reals, NOT unsigned ints -- casting to uint64 wraps negative samples).
HSD_GROUPS = (
    "hsd/ch0_time", "hsd/ch0_wave",
    "hsd/ch1_time", "hsd/ch1_wave",
    "hsd/ch2_time", "hsd/ch2_wave",
    "hsd/ch3_time", "hsd/ch3_wave",
    "hsd/high_voltage", "hsd/voltage_pulse",
)

# Integer-typed reader columns (cast after assembly).
INTEGER_COLUMNS = ("channel", "start_counter", "time_data")


def stack_columns(columns):
    """Assemble per-column arrays into a single ``(N, k)`` matrix.

    Each input may be shape ``(N,)`` or ``(N, 1)`` (the reader gets
    ``(N, 1)`` today because ``read_hdf5`` wraps every dataset in a
    one-column DataFrame, but a future lazy/raw path could yield ``(N,)``).
    Normalising to 1-D and using ``np.column_stack`` makes the reader
    independent of that shape, replacing a bare ``np.concatenate(axis=1)``
    that raised ``AxisError`` on 1-D input.
    """
    normalized = []
    n_rows = None
    for i, col in enumerate(columns):
        arr = np.asarray(col).reshape(-1)
        if n_rows is None:
            n_rows = arr.shape[0]
        elif arr.shape[0] != n_rows:
            raise ValueError(
                f"HDF5 column {i} has length {arr.shape[0]}, expected "
                f"{n_rows}; the /dld or /tdc groups have inconsistent "
                f"lengths (writer/reader schema mismatch)."
            )
        normalized.append(arr.astype(object, copy=False))
    if not normalized:
        raise ValueError("stack_columns received no columns")
    return np.column_stack(normalized)
