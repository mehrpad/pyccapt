"""Shared calibration journal and verified same-source result restoration.

The journal records actions/models; the saved table is the exact result checkpoint.
Restoration never refits or silently transfers a calibration to another experiment.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from pyccapt.calibration.core.exceptions import CalibrationInputError


ATTR = "pyccapt_calibration_metadata"
SCHEMA = "pyccapt-calibration-history-1"
RAW_COLUMNS = ("event_id", "t (ns)", "high_voltage (V)", "pulse_v (V)", "pulse_l (pJ)",
               "x_det (cm)", "y_det (cm)", "start_counter")
RESULT_COLUMNS = ("mc (Da)", "mc_uc (Da)", "t_c (ns)", "x (nm)", "y (nm)", "z (nm)")


def portable(value):
    """JSON-only model/settings state; arbitrary Python objects are never pickled."""
    if isinstance(value, dict):
        return {str(key): portable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return portable(value.tolist())
    if isinstance(value, (list, tuple)):
        return [portable(item) for item in value]
    if isinstance(value, np.generic):
        return portable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        from pyccapt.calibration.data_tools.model_state import export_model_state
        return export_model_state(value)
    except (ValueError, TypeError, AttributeError):
        return {"python_type": type(value).__name__, "portable_model_available": False}


def vector_digest(values):
    array = np.ascontiguousarray(values)
    if array.dtype.kind not in "biuf":
        raise CalibrationInputError("Calibration fingerprints require numeric columns")
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def frame_identity(frame, columns=None):
    columns = list(columns) if columns is not None else [name for name in RAW_COLUMNS if name in frame]
    if not columns or columns == ['event_id']:
        return {"rows": len(frame), "columns": columns, "sha256": None, "replayable": False}
    if any(name not in frame for name in columns):
        raise CalibrationInputError("Raw measurements required for provenance are missing")
    digest = hashlib.sha256()
    for name in columns:
        digest.update(name.encode())
        digest.update(vector_digest(frame[name].to_numpy()).encode())
    return {"rows": len(frame), "columns": columns, "sha256": digest.hexdigest()}


class CalibrationMetadata:
    """One journal contract for legacy Variables and the guided workflow."""

    def __init__(self, state=None):
        from pyccapt import __version__
        self.state = deepcopy(state) if state else {
            "schema_version": SCHEMA, "software": {"pyccapt": __version__, "numpy": np.__version__},
            "source": None, "steps": [], "checkpoint": None,
            "history_scope": "Recorded from this import onward; missing historical steps are not inferred.",
            "replay_scope": "Exact saved outputs on matching source measurements/event IDs; no refit or new-run transfer."}
        if self.state.get("schema_version") != SCHEMA:
            raise CalibrationInputError("Unsupported calibration metadata schema")

    @classmethod
    def from_frame(cls, frame):
        metadata = cls(frame.attrs.get(ATTR))
        if metadata.state["source"] is None:
            metadata.state["source"] = frame_identity(frame)
        return metadata

    def record(self, operation, *, parameters=None, model=None, status="applied"):
        step = {"sequence": len(self.state["steps"])+1, "operation": operation, "status": status,
                "time_utc": datetime.now(timezone.utc).isoformat(), "parameters": portable(parameters or {})}
        if model is not None:
            step["model"] = portable(model)
        self.state["steps"].append(step)
        return deepcopy(step)

    def attach(self, frame):
        frame.attrs[ATTR] = deepcopy(self.state)
        return frame

    def checkpoint(self, frame):
        self.state["checkpoint"] = {
            "input": frame_identity(frame),
            "outputs": {name: vector_digest(frame[name].to_numpy()) for name in RESULT_COLUMNS if name in frame},
            "selection": portable(frame.attrs.get("pyccapt_selection", {})),
            "instrument": portable(frame.attrs.get("pyccapt_instrument_metadata", {})),
            "time_utc": datetime.now(timezone.utc).isoformat()}
        self.attach(frame)
        return deepcopy(self.state)

    def to_dict(self):
        return deepcopy(self.state)

    def write_json(self, path):
        """Atomic sidecar for formats that cannot contain processing metadata."""
        path = Path(path)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(self.state, stream, allow_nan=False, sort_keys=True, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def write_hdf5(self, handle):
        import h5py
        if "calibration_provenance" in handle:
            del handle["calibration_provenance"]
        group = handle.create_group("calibration_provenance")
        group.attrs.update(NX_class="NXcollection", schema_version=SCHEMA,
                           conformance="PyCCAPT processing history extension; not a full NXapm conformance assertion")
        group.create_dataset("json", data=json.dumps(self.state, allow_nan=False, sort_keys=True),
                             dtype=h5py.string_dtype("utf-8"))


def record_step(variables, operation, *, parameters=None, model=None, status="applied"):
    """Record only on real loaded data; widget rendering cannot fabricate a run."""
    frame = getattr(variables, "data", None)
    if frame is None or not hasattr(frame, "columns") or not len(frame):
        return None
    metadata = getattr(variables, "calibration_metadata", None)
    if not isinstance(metadata, CalibrationMetadata):
        metadata = CalibrationMetadata.from_frame(frame)
        variables.calibration_metadata = metadata
    metadata.record(operation, parameters=parameters, model=model, status=status)
    metadata.attach(frame)
    return metadata


def restore_saved_calibration(path, original_source):
    """Return saved selected results after verifying every selected raw measurement.

    Requires the original recorded analysis input in its original event order.
    Changes in voltage, ToF, detector hits or counters reject restoration.
    This restores the accepted/exported result, not the outcome of a new fit.
    """
    from pyccapt.calibration.data_tools.data_tools import load_data
    from pyccapt.calibration.core.dataset import CalibrationDataset
    import h5py
    with h5py.File(path, "r") as handle:
        kind = "pyccapt" if "df" in handle else "nexus"
    saved = load_data(str(Path(path)), kind, mode="processed")
    if ATTR not in saved.attrs:
        raise CalibrationInputError("No calibration provenance checkpoint was saved")
    metadata = CalibrationMetadata(saved.attrs[ATTR])
    checkpoint = metadata.state.get("checkpoint")
    if not checkpoint or "event_id" not in saved:
        raise CalibrationInputError("Replay needs a checkpoint with stable event IDs")
    original = CalibrationDataset.from_frame(original_source).frame
    source = metadata.state.get("source")
    if not source or source.get('sha256') is None or frame_identity(original, source["columns"]) != source:
        raise CalibrationInputError("The complete original analysis input/order does not match this calibration history")
    import pandas as pd
    ids = pd.Index(original["event_id"])
    indices = ids.get_indexer(saved["event_id"])
    if np.any(indices < 0):
        raise CalibrationInputError("Original source does not contain every saved event ID")
    selected = original.iloc[indices].copy().reset_index(drop=True)
    if frame_identity(selected, checkpoint["input"]["columns"]) != checkpoint["input"]:
        raise CalibrationInputError("Raw source measurements differ from the saved calibration checkpoint")
    for name, expected in checkpoint["outputs"].items():
        if name not in saved or vector_digest(saved[name].to_numpy()) != expected:
            raise CalibrationInputError(f"Saved calibrated output failed its fingerprint: {name}")
        selected[name] = saved[name].to_numpy(copy=True)
    selected.attrs = deepcopy(saved.attrs)
    selected.attrs["pyccapt_calibration_restoration"] = {"source_verified": True, "refit": False}
    return selected
