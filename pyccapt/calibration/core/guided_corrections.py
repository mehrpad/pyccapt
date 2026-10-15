"""Event-aligned bridge to the original voltage/bowl calibration tools.

The tools only mutate a private selected-event workspace. Capture scatters its
results back into source order; the workflow's separate acceptance is unchanged.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from pyccapt.calibration.core.exceptions import CalibrationInputError, CalibrationStateError
from pyccapt.calibration.core.share_variables import Variables


METHOD = "original voltage/bowl correction tools"


class CorrectionWorkspace:
    """Bind a correction trial to one source, accepted crop and instrument setup."""

    def __init__(self, workflow, *, flight_length_mm, t0_ns, pulse_mode, detector_diameter_mm=None):
        workflow._check()
        if not np.isfinite([flight_length_mm, t0_ns]).all() or flight_length_mm <= 0 or pulse_mode not in ("laser", "voltage"):
            raise CalibrationInputError("Check flight length (positive mm), time zero (ns) and pulse mode")
        baseline = workflow.preview.get("calibration", workflow.accepted["calibration"])
        if baseline is None:
            raise CalibrationStateError("Start calibration or preview an initial mass before opening correction tools")
        self.workflow, self.revision = workflow, workflow.revision
        self.settings = (float(flight_length_mm), float(t0_ns), pulse_mode)
        if detector_diameter_mm is not None and (not np.isfinite(detector_diameter_mm) or detector_diameter_mm <= 0):
            raise CalibrationInputError("Detector diameter must be finite and positive in mm")
        self.detector_diameter_mm = detector_diameter_mm
        self.initial_receipt = deepcopy(baseline["receipt"])
        self.initial_mass = np.array(baseline["mass"], copy=True)
        self.initial_tof = None if baseline.get("tof") is None else np.array(baseline["tof"], copy=True)
        frame = workflow.source
        valid = workflow.active_mask & np.isfinite(self.initial_mass) & (self.initial_mass > 0)
        if self.initial_tof is not None:
            valid &= np.isfinite(self.initial_tof) & (self.initial_tof > 0)
        unavailable = frame.attrs.get("pyccapt_unavailable_columns", [])
        for name in ("t (ns)", "high_voltage (V)", "x_det (cm)", "y_det (cm)"):
            if name not in frame or name in unavailable:
                raise CalibrationStateError(f"Correction tools require measured {name}; keep the existing mass or use a supported source")
            valid &= workflow.variables.dataset.validity_status(name) == 0
            valid &= np.isfinite(frame[name].to_numpy(dtype=float))
        valid &= (frame["t (ns)"].to_numpy() > t0_ns) & (frame["high_voltage (V)"].to_numpy() > 0)
        if pulse_mode == "voltage":
            if "pulse_v (V)" not in frame or "pulse_v (V)" in unavailable:
                raise CalibrationStateError("Voltage-mode correction requires measured pulse voltage")
            valid &= (workflow.variables.dataset.validity_status("pulse_v (V)") == 0) & np.isfinite(frame["pulse_v (V)"])
        self.indices = np.flatnonzero(valid)
        if not len(self.indices):
            raise CalibrationStateError("No selected events have the measurements required for voltage/bowl correction")
        self.event_ids = workflow.event_ids[self.indices].copy()
        selected = frame.iloc[self.indices].copy(deep=True).reset_index(drop=True)
        selected["mc (Da)"] = self.initial_mass[self.indices]
        selected["mc_uc (Da)"] = self.initial_mass[self.indices]
        selected.attrs["pyccapt_unavailable_columns"] = [name for name in unavailable if name not in ("mc (Da)", "mc_uc (Da)")]
        working = self.variables = Variables()
        working.sync_from_data(selected)
        working.pulse_mode, working.flight_path_length = pulse_mode, float(flight_length_mm)
        working.max_tof = int(np.ceil(np.max(working.dld_t)))
        if self.initial_tof is not None:
            working.dld_t_calib = self.initial_tof[self.indices].copy()
        working.mc_calib = self.initial_mass[self.indices].copy()
        working.mc_calib_backup = working.mc_calib.copy()
        working.dld_t_calib_backup = working.dld_t_calib.copy()
        self.start_mass, self.start_tof = working.mc_calib.copy(), working.dld_t_calib.copy()
        directory = Path(workflow.source_path).parent if workflow.source_path else Path.cwd()
        working.set_result_directory(directory)
        self.summary = {"workspace_events": len(self.indices),
            "selected_events_not_corrected": int(workflow.active_mask.sum())-len(self.indices),
            "excluded_event_policy": "Retain initial mass/ToF for events lacking required correction inputs."}

    def preview(self, workflow, settings):
        """Capture working outputs without accepting or refitting either domain."""
        if workflow is not self.workflow or workflow.revision != self.revision or tuple(settings) != self.settings:
            raise CalibrationStateError("Crop, workflow or instrument settings changed; start a new correction workspace")
        workflow._check()
        working = self.variables
        for ids in (np.asarray(working.event_ids), working.data["event_id"].to_numpy()):
            if ids.dtype.kind not in "iu" or not np.array_equal(ids, self.event_ids):
                raise CalibrationStateError("Correction workspace event order changed; start calibration again")
        mass, time = np.asarray(working.mc_calib, dtype=float), np.asarray(working.dld_t_calib, dtype=float)
        for values in (mass, time):
            if values.shape != self.start_mass.shape or not np.isfinite(values).all() or np.any(values <= 0):
                raise CalibrationInputError("Correction outputs must be positive, finite and event-aligned; reset or repair the working trial")
        masses = self.initial_mass.copy()
        masses[self.indices] = mass
        tof_changed = not np.array_equal(time, self.start_tof)
        tof = None if self.initial_tof is None else self.initial_tof.copy()
        if tof_changed:
            if tof is None:
                tof = np.full(workflow.n, np.nan)
            tof[self.indices] = time
        receipt = dict(method=METHOD, flight_length_mm=self.settings[0], t0_ns=self.settings[1], pulse_mode=self.settings[2],
            initial_mass_receipt=self.initial_receipt, **self.summary,
            mass_changed=not np.array_equal(mass, self.start_mass), tof_changed=tof_changed,
            tof_convention="Independent original-tool ToF normalization/corrections; never reconverted as raw ToF.",
            working_history=working.calibration_metadata.to_dict())
        if self.detector_diameter_mm is not None:
            receipt["detector_diameter_mm"] = float(self.detector_diameter_mm)
        return workflow.preview_calibration(masses, tof=tof, receipt=receipt,
            model_state={"workspace_event_ids": self.event_ids}, expected_revision=self.revision)
