"""Accepted-state notebook workflow with explicit dependency invalidation and undo."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from pyccapt.calibration.core.exceptions import CalibrationInputError, CalibrationStateError
from pyccapt.calibration.core.ranging_review import PeakReviewQueue
from pyccapt.calibration.core.share_variables import Variables


STAGES = ("Import", "Inspect", "Calibrate", "Range", "Reconstruct", "Analyze", "Export")
DEPENDENTS = {"calibration": ("ranges", "reconstruction", "analysis"),
              "ranges": ("reconstruction", "analysis"), "reconstruction": ("analysis",), "analysis": ()}


def _owned(array, dtype=None):
    value = np.array(array, dtype=dtype, copy=True)
    value.flags.writeable = False
    return value


def _pack(value, arrays):
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "SU":
            return value.tolist()
        if value.dtype.kind not in "biufc":
            raise CalibrationInputError("Session arrays must have numeric, boolean or string dtypes")
        raw = np.ascontiguousarray(value)
        from pyccapt.calibration.data_tools.analysis_session import array_digest
        key = "a_" + array_digest(raw)
        arrays[key] = raw
        return {"array_ref": key}
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _pack(item, arrays) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_pack(item, arrays) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise CalibrationInputError(f"Unsupported session metadata type {type(value).__name__}")


def _unpack(value, arrays, cache=None):
    cache = {} if cache is None else cache
    if isinstance(value, dict):
        if set(value) == {"array_ref"}:
            if value["array_ref"] not in arrays:
                raise CalibrationStateError("Session array reference is missing")
            key = value["array_ref"]
            if key not in cache:
                cache[key] = _owned(arrays[key])
            return cache[key]
        return {key: _unpack(item, arrays, cache) for key, item in value.items()}
    if isinstance(value, list):
        return [_unpack(item, arrays, cache) for item in value]
    return value


class GuidedWorkflow:
    """Own accepted derived state while retaining an immutable imported source.

    Preview payloads cannot affect shared Variables until accepted. Snapshots use
    immutable array references in memory, so undo does not copy every event for
    each button press. Session persistence deduplicates identical numeric arrays.
    """
    def __init__(self, variables, *, source_path=None):
        if variables.dataset is None or not len(variables.data):
            raise CalibrationStateError("Import a nonempty dataset before creating the guided workflow")
        self.variables = variables
        self._source_dataset = variables.dataset
        self.source = variables.dataset.frame.copy(deep=True)
        self.source.attrs = deepcopy(variables.dataset.frame.attrs)
        self.source_hash = variables.dataset.source_hash
        self.source_path = str(source_path or "")
        self.event_ids = _owned(variables.dataset.event_ids, np.uint64)
        self.n = len(self.event_ids)
        self.accepted = {key: None for key in DEPENDENTS}
        self.preview = {}
        self.review = None
        self.history, self.log = [], []
        self.revision, self.mass_revision = 0, 0
        self.inspected = False
        self._sync_outputs()

    @classmethod
    def import_file(cls, path, *, data_type="pyccapt", mode="processed", variables=None):
        from pyccapt.calibration.data_tools.data_tools import load_data
        frame = load_data(str(Path(path).expanduser()), data_type, mode=mode)
        if isinstance(frame, tuple):
            frame = frame[0]
        variables = variables or Variables()
        variables.sync_from_data(frame, update_backups=True)
        return cls(variables, source_path=path)

    def _check(self):
        ids = np.asarray(getattr(self.variables, "event_ids", []))
        if (self.variables.dataset is not self._source_dataset or ids.shape != self.event_ids.shape or
                not np.array_equal(ids, self.event_ids)):
            raise CalibrationStateError("Source event order changed outside the workflow; import/create a new workflow")

    def _snapshot(self, operation):
        self.history.append({"accepted": self.accepted.copy(), "mass_revision": self.mass_revision,
                             "review": None if self.review is None else self.review.to_state(),
                             "inspected": self.inspected, "operation": operation})
        self.history = self.history[-12:]

    def _invalidate(self, stage):
        for name in DEPENDENTS[stage]:
            self.accepted[name] = None
        self.preview.clear()
        if stage == "calibration":
            self.mass_revision += 1
            self.review = None

    def _sync_outputs(self):
        v = self.variables
        frame = self.source.copy(deep=True)
        calibration = self.accepted["calibration"]
        mass = calibration["mass"] if calibration else np.full(self.n, np.nan)
        v.mc = np.array(mass, copy=True)
        v.mc_calib = np.array(mass, copy=True)
        v.has_mass_spectrum = bool(np.isfinite(mass).any())
        frame["mc (Da)"] = mass
        tof = calibration.get("tof") if calibration else None
        v.dld_t_c = np.array(tof, copy=True) if tof is not None else np.full(self.n, np.nan)
        v.dld_t_calib = v.dld_t_c.copy() if tof is not None else np.array(v.dld_t, copy=True)
        if tof is not None:
            frame["t_c (ns)"] = tof
        elif "t_c (ns)" in frame:
            frame.drop(columns="t_c (ns)", inplace=True)
        ranges = self.accepted["ranges"]
        v.range_data = pd.DataFrame(deepcopy(ranges["rows"])) if ranges else v.range_data.iloc[:0].copy()
        spatial = self.accepted["reconstruction"]
        xyz = spatial["xyz"] if spatial else np.full((self.n, 3), np.nan)
        for index, name in enumerate(("x", "y", "z")):
            setattr(v, name, np.array(xyz[:, index], copy=True))
            if spatial:
                frame[f"{name} (nm)"] = xyz[:, index]
            elif f"{name} (nm)" in frame:
                frame.drop(columns=f"{name} (nm)", inplace=True)
        v.has_reconstruction = bool(np.isfinite(xyz).all(axis=1).any())
        v.reconstruction_parameters = deepcopy(spatial["parameters"]) if spatial else {}
        v.reconstruction_ratio_result = deepcopy(spatial.get("ratio_result", {})) if spatial else {}
        v.guided_calibration_receipt = deepcopy(calibration["receipt"]) if calibration else {}
        v.pulse_mode = (spatial["parameters"].get("pulse_mode") if spatial else None) or (
            calibration["receipt"].get("pulse_mode") if calibration else None) or getattr(v, "pulse_mode", "laser")
        replaced = ({"mc (Da)"} if calibration else set()) | ({"t_c (ns)"} if tof is not None else set())
        if spatial:
            replaced.update(("x (nm)", "y (nm)", "z (nm)"))
        frame.attrs["pyccapt_unavailable_columns"] = [name for name in
            frame.attrs.get("pyccapt_unavailable_columns", []) if name not in replaced]
        v.data = frame

    def working_variables(self):
        """Independent sandbox for legacy/calibration/ratio panels; acceptance is separate."""
        self._check()
        working = Variables()
        working.sync_from_data(self.source.copy(deep=True), update_backups=True)
        working.pulse_mode = self.variables.pulse_mode
        # Legacy tables can contain numerical zero placeholders. In this
        # independent preview workspace, availability takes precedence over the
        # numerical value; do not invent detector-center or pulse measurements.
        for column, attribute in (("t (ns)", "dld_t"), ("high_voltage (V)", "dld_high_voltage"),
                                  ("x_det (cm)", "dld_x_det"), ("y_det (cm)", "dld_y_det"),
                                  ("pulse_v (V)", "dld_pulse_v")):
            values = np.asarray(getattr(working, attribute), dtype=float).copy()
            values[working.dataset.validity_status(column) != 0] = np.nan
            setattr(working, attribute, values)
        calibration = self.accepted["calibration"]
        if calibration:
            working.mc = np.array(calibration["mass"], copy=True)
            working.mc_calib = working.mc.copy()
            working.has_mass_spectrum = bool(np.isfinite(working.mc).any())
            working.data["mc (Da)"] = working.mc
            working.data.attrs["pyccapt_unavailable_columns"] = [name for name in
                working.data.attrs.get("pyccapt_unavailable_columns", []) if name != "mc (Da)"]
            working.guided_calibration_receipt = deepcopy(calibration["receipt"])
            if calibration.get("tof") is not None:
                working.dld_t_calib = np.array(calibration["tof"], copy=True)
                working.dld_t_c = working.dld_t_calib.copy()
                working.data["t_c (ns)"] = working.dld_t_c
                working.data.attrs["pyccapt_unavailable_columns"] = [name for name in
                    working.data.attrs.get("pyccapt_unavailable_columns", []) if name != "t_c (ns)"]
        return working

    def inspect(self):
        self._check()
        self.inspected = True
        return {"events": self.n, "source_hash": self.source_hash,
                "columns": list(self.source.columns), "capabilities": dict(self.variables.dataset.capabilities),
                "unknown_positions": int((~self.variables.dataset.finite_mask("detector_positions")).sum()),
                "accepted_stages": self.status()}

    def preview_calibration(self, masses, *, receipt=None, model_state=None, tof=None, expected_revision=None):
        self._check()
        self._revision_check(expected_revision)
        mass = _owned(masses, float)
        if mass.shape != (self.n,) or np.isinf(mass).any() or not np.any(np.isfinite(mass) & (mass > 0)):
            raise CalibrationInputError("Mass preview must be event-aligned, contain positive finite masses, and use NaN for unavailable events")
        if np.any(mass[np.isfinite(mass)] < 0):
            raise CalibrationInputError("Mass values cannot be negative")
        if tof is not None:
            tof = _owned(tof, float)
            if (tof.shape != (self.n,) or np.isinf(tof).any() or not np.any(tof > 0) or
                    np.any(tof[np.isfinite(tof)] < 0)):
                raise CalibrationInputError("Calibrated ToF must be aligned and nonnegative with positive finite values, NaN where unavailable")
        self.preview["calibration"] = {"revision": self.revision, "mass": mass,
            "receipt": deepcopy(receipt or {"method": "explicit imported/externally calibrated mass"}),
            "model_state": deepcopy(model_state), "tof": tof, "valid_mask": _owned(np.isfinite(mass), bool)}
        return self.compare("calibration")

    def preview_imported_mass(self, *, pulse_mode=None, flight_length_mm=None):
        unavailable = self.source.attrs.get("pyccapt_unavailable_columns", [])
        columns = [name for name in ("mc (Da)", "mc_uc (Da)") if name in self.source and name not in unavailable
                   and np.any(np.isfinite(self.source[name].to_numpy(dtype=float)) & (self.source[name].to_numpy(dtype=float) > 0))]
        if not columns:
            raise CalibrationStateError("Imported data has no mass column; calibrate ToF first")
        receipt = {"method": "imported mass", "column": columns[0]}
        if pulse_mode is not None:
            if pulse_mode not in ("laser", "voltage"):
                raise CalibrationInputError("Pulse mode must be laser or voltage")
            receipt["pulse_mode"] = pulse_mode
        if flight_length_mm is not None:
            if not np.isfinite(flight_length_mm) or flight_length_mm <= 0:
                raise CalibrationInputError("Flight length must be positive in mm")
            receipt["flight_length_mm"] = float(flight_length_mm)
        tof = None
        if columns[0] == "mc (Da)" and "t_c (ns)" in self.source and "t_c (ns)" not in unavailable:
            values = self.source["t_c (ns)"].to_numpy(dtype=float)
            valid = np.isfinite(values) & (values >= 0)
            if np.any(valid & (values > 0)):
                tof = np.where(valid, values, np.nan)
                receipt.update(tof_column="t_c (ns)", unavailable_tof_events=int((~valid).sum()))
        result = self.preview_calibration(self.source[columns[0]].to_numpy(), receipt=receipt, tof=tof)
        return dict(result, imported_mass_column=columns[0], imported_tof_column=receipt.get("tof_column"))

    def _revision_check(self, expected):
        if expected is not None and expected != self.revision:
            raise CalibrationStateError("Workflow changed while computing; discard this preview and run again")

    def begin_review(self):
        self._check()
        if self.accepted["calibration"] is None:
            raise CalibrationStateError("Accept mass calibration before proposing ranges")
        if self.review is None:
            self.review = PeakReviewQueue(self.accepted["calibration"]["mass"], mass_revision=self.mass_revision)
        return self.review

    def preview_ranges(self):
        review = self.begin_review()
        rows = review.ranges()
        if rows.empty:
            raise CalibrationStateError("Accept at least one reviewed peak identity first")
        self.preview["ranges"] = {"revision": self.revision, "rows": rows.to_dict("records"),
                                  "assignment": _owned(review.assignments(), np.int32),
                                  "review_state": review.to_state(), "overlap_policy": "first accepted table row"}
        return self.compare("ranges")

    def preview_reconstruction(self, xyz, parameters, *, ratio_result=None, expected_revision=None):
        self._check()
        self._revision_check(expected_revision)
        if self.accepted["ranges"] is None:
            raise CalibrationStateError("Accept reviewed ranges before the reconstruction stage")
        xyz = _owned(xyz, float)
        if xyz.shape != (self.n, 3) or np.isinf(xyz).any() or not np.isfinite(xyz).all(axis=1).any():
            raise CalibrationInputError("Reconstruction must be an aligned N-by-3 nm array with finite valid rows")
        self.preview["reconstruction"] = {"revision": self.revision, "xyz": xyz,
            "parameters": deepcopy(parameters), "ratio_result": deepcopy(ratio_result or {})}
        return self.compare("reconstruction")

    def accept(self, stage):
        self._check()
        if stage not in self.preview or stage not in DEPENDENTS:
            raise CalibrationStateError("Create a preview before accepting this stage")
        payload = self.preview[stage]
        self._revision_check(payload["revision"])
        if stage == "ranges":
            from pyccapt.calibration.data_tools.analysis_session import canonical_json
            if canonical_json(_pack(self.review.to_state(), {})) != canonical_json(_pack(payload["review_state"], {})):
                raise CalibrationStateError("Peak review changed after preview; preview ranges again")
        self._snapshot(f"accept {stage}")
        accepted = {key: value for key, value in payload.items() if key != "revision"}
        self._invalidate(stage)
        self.accepted[stage] = accepted
        self.revision += 1
        self.log.append({"action": "accept", "stage": stage, "revision": self.revision})
        self._sync_outputs()
        return self.status()

    def undo(self):
        self._check()
        if not self.history:
            raise CalibrationStateError("No accepted workflow change to undo")
        previous = self.history.pop()
        self.accepted, self.mass_revision = previous["accepted"], previous["mass_revision"]
        self.inspected = previous["inspected"]
        state = previous["review"]
        self.review = None if state is None else PeakReviewQueue.from_state(self.accepted["calibration"]["mass"], state)
        self.preview.clear()
        self.revision += 1
        self.log.append({"action": "undo", "operation": previous["operation"], "revision": self.revision})
        self._sync_outputs()
        return self.status()

    def compare(self, stage):
        candidate, accepted = self.preview.get(stage), self.accepted.get(stage)
        if candidate is None:
            raise CalibrationStateError("There is no preview to compare")
        if stage == "ranges":
            return {"accepted_ranges": len(accepted["rows"]) if accepted else 0,
                    "preview_ranges": len(candidate["rows"]),
                    "preview_ranged_events": int(np.count_nonzero(candidate["assignment"] >= 0)),
                    "unranged_events": int(np.count_nonzero(candidate["assignment"] < 0))}
        key = "mass" if stage == "calibration" else "xyz"
        values = candidate[key]
        valid = np.isfinite(values)
        result = {"finite_preview_values": int(valid.sum()), "finite_accepted_values": int(np.isfinite(accepted[key]).sum()) if accepted else 0,
                  "preview_quantiles": np.percentile(values[valid], [1, 50, 99]).tolist()}
        if accepted:
            overlap = valid & np.isfinite(accepted[key])
            result["rms_change"] = float(np.sqrt(np.mean((values[overlap]-accepted[key][overlap])**2))) if overlap.any() else None
        return result

    def analyze(self):
        self._check()
        if self.accepted["ranges"] is None:
            raise CalibrationStateError("Accept reviewed ranges before composition analysis")
        ranged = self.accepted["ranges"]
        counts = np.bincount(ranged["assignment"][ranged["assignment"] >= 0], minlength=len(ranged["rows"]))
        atoms, ions = {}, {}
        for row, count in zip(ranged["rows"], counts):
            ions[row["ion"]] = ions.get(row["ion"], 0) + int(count)
            for element, multiplicity in zip(row["element"], row["complex"]):
                atoms[element] = atoms.get(element, 0.) + float(count*multiplicity)
        total = sum(atoms.values())
        result = {"events_per_ion": ions,
                  "atoms_per_element": atoms, "atomic_fractions_of_assigned_atoms": {k: v/total for k, v in atoms.items()} if total else {},
                  "unranged_events": int(np.count_nonzero(ranged["assignment"] < 0)),
                  "interpretation": "Hard interval assignments, not a background-corrected composition or uncertainty estimate"}
        self._snapshot("analyze")
        self.accepted["analysis"] = result
        self.revision += 1
        self.preview.clear()
        self.log.append({"action": "analyze", "revision": self.revision})
        return result

    def status(self):
        result = {"Import": "accepted", "Inspect": "reviewed" if self.inspected else "ready",
                "Calibrate": "accepted" if self.accepted["calibration"] else "ready",
                "Range": "accepted" if self.accepted["ranges"] else "ready" if self.accepted["calibration"] else "blocked",
                "Reconstruct": "accepted" if self.accepted["reconstruction"] else "ready" if self.accepted["ranges"] else "blocked",
                "Analyze": "accepted" if self.accepted["analysis"] else "ready" if self.accepted["ranges"] else "blocked",
                "Export": "session ready; derived export ready" if self.accepted["calibration"] else "session ready"}
        for key, label in (("calibration", "Calibrate"), ("ranges", "Range"), ("reconstruction", "Reconstruct")):
            if key in self.preview:
                result[label] += "; preview awaiting acceptance"
        if self.review is not None and self.accepted["ranges"] is not None:
            if self.review.ranges().to_dict("records") != self.accepted["ranges"]["rows"]:
                result["Range"] += "; review edits not applied"
        return result

    def to_state(self):
        self._check()
        # Hash at the persistence boundary, not on every peak-review click.
        from pyccapt.calibration.core.dataset import CalibrationDataset
        if CalibrationDataset.from_frame(self.source).source_hash != self.source_hash:
            raise CalibrationStateError("Imported source was modified; refusing an inconsistent session snapshot")
        columns = []
        for name in self.source:
            series = self.source[name]
            data = series.to_numpy()
            if data.dtype.kind == "O":
                data = data.tolist()
            columns.append({"name": str(name), "dtype": str(series.dtype), "data": data})
        arrays = {}
        metadata = _pack({"workflow_schema": "1.0", "source_hash": self.source_hash,
            "source_path": self.source_path, "source_attrs": self.source.attrs, "source_columns": columns,
            "event_ids": self.event_ids, "accepted": self.accepted, "preview": self.preview,
            "review": None if self.review is None else self.review.to_state(), "history": self.history,
            "log": self.log, "revision": self.revision, "mass_revision": self.mass_revision, "inspected": self.inspected}, arrays)
        return metadata, arrays

    @classmethod
    def from_state(cls, metadata, arrays, *, variables=None):
        state = _unpack(metadata, arrays)
        if state.get("workflow_schema") != "1.0":
            raise CalibrationInputError("Unsupported guided workflow session schema")
        frame = pd.DataFrame({column["name"]: pd.Series(column["data"], dtype=column["dtype"])
                              for column in state["source_columns"]})
        frame.attrs = state["source_attrs"]
        temporary = Variables()
        temporary.sync_from_data(frame, update_backups=True)
        workflow = cls(temporary, source_path=state["source_path"])
        if workflow.source_hash != state["source_hash"] or not np.array_equal(workflow.event_ids, state["event_ids"]):
            raise CalibrationStateError("Session source identity/order/hash does not match the saved workflow")
        workflow.accepted, workflow.preview = state["accepted"], state["preview"]
        workflow.history, workflow.log = state["history"], state["log"]
        workflow.revision, workflow.mass_revision, workflow.inspected = state["revision"], state["mass_revision"], state["inspected"]
        if state["review"] is not None:
            workflow.review = PeakReviewQueue.from_state(workflow.accepted["calibration"]["mass"], state["review"])
        from pyccapt.calibration.core.guided_workflow_validation import validate_restored_workflow
        validate_restored_workflow(workflow)
        workflow._sync_outputs()
        if variables is not None:
            # Commit only after every source/output/review validation succeeds.
            variables.__dict__.clear()
            variables.__dict__.update(temporary.__dict__)
            workflow.variables = variables
        return workflow

    def save(self, path):
        from pyccapt.calibration.data_tools.analysis_session import save_analysis_session
        return save_analysis_session(path, *self.to_state())

    @classmethod
    def load(cls, path, *, variables=None):
        from pyccapt.calibration.data_tools.analysis_session import load_analysis_session
        return cls.from_state(*load_analysis_session(path), variables=variables)
