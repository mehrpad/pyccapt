"""Reviewable fit/selection/confirmation/accept/apply workflow for ToF calibration."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from pyccapt.calibration.core.exceptions import CalibrationInputError, CalibrationStateError
from pyccapt.calibration.core.multireference_tof import MultiReferenceToFModel, physical_tof_baseline


def _digest(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _immutable(values, dtype=float):
    result = np.array(values, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class ValidationPolicy:
    max_abs_relative_bias: float = .005
    max_relative_rmse: float = .02
    min_events_per_reference_block: int = 10

    def __post_init__(self):
        if (not np.isfinite(self.max_abs_relative_bias) or self.max_abs_relative_bias <= 0 or
                not np.isfinite(self.max_relative_rmse) or self.max_relative_rmse <= 0 or
                not isinstance(self.min_events_per_reference_block, int) or self.min_events_per_reference_block < 2):
            raise CalibrationInputError("Validation tolerances must be positive; each reference/block needs >=2 events")


class GuidedCalibrationSession:
    """Snapshot a dataset and user-confirmed reference identities without applying a fit.

    ``references`` maps known mass-to-charge (Da) to event-aligned boolean masks.
    ``roles`` contains disjoint train/selection/confirmation boolean masks. Use
    independent acquisition regions where possible; the session never relabels
    references or reuses confirmation events for optimization.
    """
    SOURCE_FIELDS = ("dld_t", "dld_high_voltage", "dld_x_det", "dld_y_det", "dld_pulse_v")

    def __init__(self, variables, references, roles, *, mode, flight_length_mm,
                 pulse_mode="laser", model_config=None, policy=None, apply_mask=None,
                 reference_voltage_V=5000., reference_flight_length_mm=None,
                 max_fit_events_per_group=5000):
        if mode not in ("tof", "mc"):
            raise CalibrationInputError("mode must be tof or mc")
        if getattr(variables, "dataset", None) is None:
            raise CalibrationStateError("Load data with Variables.sync_from_data before creating a session")
        if not isinstance(max_fit_events_per_group, int) or max_fit_events_per_group < 10:
            raise CalibrationInputError("max_fit_events_per_group must be an integer >=10")
        self.variables, self.mode = variables, mode
        self._dataset = variables.dataset
        required = ["t (ns)", "high_voltage (V)", "x_det (cm)", "y_det (cm)"]
        if pulse_mode == "voltage":
            required.append("pulse_v (V)" if "pulse_v (V)" in self._dataset.frame else "pulse")
        unavailable = self._dataset.frame.attrs.get("pyccapt_unavailable_columns", [])
        missing = [name for name in required if name not in self._dataset.frame or name in unavailable]
        if missing:
            raise CalibrationInputError("Physical ToF calibration requires recorded source fields; unavailable: " + ", ".join(missing))
        self.dataset_hash = self._dataset.source_hash
        self.raw = tuple(_immutable(getattr(variables, name)) for name in self.SOURCE_FIELDS)
        self.n = len(self.raw[0])
        if not self.n or any(a.shape != (self.n,) for a in self.raw):
            raise CalibrationInputError("Raw source arrays must be nonempty and event-aligned")
        self.source_hash = _digest(*self.raw)
        self._active = _immutable(variables.get_calibration_array(mode))
        if self._active.shape != (self.n,):
            raise CalibrationInputError("Active calibration array is not event-aligned")
        self._active_hash = _digest(self._active)
        self.flight_length_mm, self.pulse_mode = float(flight_length_mm), pulse_mode
        self.reference_voltage_V = float(reference_voltage_V)
        self.reference_flight_length_mm = float(flight_length_mm if reference_flight_length_mm is None else reference_flight_length_mm)
        if not np.isfinite(self.reference_voltage_V) or self.reference_voltage_V <= 0:
            raise CalibrationInputError("reference_voltage_V must be positive and finite")
        self.reference_scale = float(physical_tof_baseline(self.reference_voltage_V, 0., 0.,
                                                          self.reference_flight_length_mm))
        if not np.isfinite(self.reference_scale) or self.reference_scale <= 0:
            raise CalibrationInputError("Reference voltage/flight length produce an invalid normalization")
        t, voltage, x, y, pulse = self.raw
        baseline = physical_tof_baseline(voltage, x, y, self.flight_length_mm, pulse, pulse_mode)
        self.features = _immutable(np.column_stack((voltage, x, y, np.linspace(0., 1., self.n), baseline)))
        finite = np.all(np.isfinite(self.features), axis=1) & np.isfinite(t) & (t > 0) & (voltage > 0) & (baseline > 0)
        self.apply_mask = _immutable(finite if apply_mask is None else self._mask(apply_mask, "apply_mask"), bool)
        if not np.any(self.apply_mask) or np.any(self.apply_mask & ~finite):
            raise CalibrationInputError("apply_mask must contain at least one event and only finite valid source rows")
        labels = np.full(self.n, np.nan)
        for mass, mask in references.items():
            mass = float(mass)
            selected = self._mask(mask, f"reference {mass}")
            if not np.isfinite(mass) or mass <= 0 or not selected.any():
                raise CalibrationInputError("Reference masses must be positive and masks nonempty")
            if np.any(np.isfinite(labels[selected])) or np.any(selected & ~finite):
                raise CalibrationInputError("Reference masks must not overlap or include invalid source rows")
            labels[selected] = mass
        if len(np.unique(labels[np.isfinite(labels)])) < 2:
            raise CalibrationInputError("Provide at least two user-confirmed reference masses")
        self.labels = _immutable(labels)
        if set(roles) != {"train", "selection", "confirmation"}:
            raise CalibrationInputError("roles must contain train, selection and confirmation")
        masks = {name: _immutable(self._mask(mask, name), bool) for name, mask in roles.items()}
        if np.any(sum(mask.astype(int) for mask in masks.values()) > 1):
            raise CalibrationInputError("Train, selection and confirmation must be disjoint")
        self.roles = MappingProxyType(masks)
        self.policy = policy or ValidationPolicy()
        self.model = MultiReferenceToFModel(model_config)
        self.blocks = _immutable(np.searchsorted(self.model.config.block_edges, self.features[:, 3], side="right"), int)
        self.max_fit_events_per_group = max_fit_events_per_group
        self.status, self.diagnostics = "preview", {}
        self._copied_to_mass = False
        self._model_hash = None
        for role in masks:
            self._check_coverage(role)
        self._protocol_hash = self._fingerprint_protocol()

    def _mask(self, mask, name):
        value = np.asarray(mask)
        if value.dtype != bool or value.shape != (self.n,):
            raise CalibrationInputError(f"{name} must be an event-aligned boolean mask")
        return value

    def _groups(self, role):
        for mass in np.unique(self.labels[np.isfinite(self.labels)]):
            for block in range(len(self.model.config.block_edges) + 1):
                yield mass, block, self.roles[role] & (self.labels == mass) & (self.blocks == block)

    def _check_coverage(self, role):
        for mass, block, mask in self._groups(role):
            if np.count_nonzero(mask) < self.policy.min_events_per_reference_block:
                raise CalibrationInputError(f"{role}: insufficient reference {mass:g} Da in declared block {block}")

    def _check_current(self):
        if self.status == "invalidated":
            raise CalibrationStateError("Session was invalidated; create a new session")
        if self._protocol_hash != self._fingerprint_protocol():
            self.status = "invalidated"
            raise CalibrationStateError("Session protocol or masks changed; create a new session")
        current = tuple(np.asarray(getattr(self.variables, field), dtype=float) for field in self.SOURCE_FIELDS)
        if (self.variables.dataset is not self._dataset or _digest(*current) != self.source_hash or
                _digest(np.asarray(self.variables.get_calibration_array(self.mode), dtype=float)) != self._active_hash):
            self.status = "invalidated"
            raise CalibrationStateError("Source, event order or active calibration changed; create a new session")
        if self._model_hash is not None and self._model_hash != self._fingerprint_model():
            self.status = "invalidated"
            raise CalibrationStateError("Fitted model changed outside the session; create a new session")

    def _fingerprint_model(self):
        return hashlib.sha256(json.dumps(self.model.provenance(), sort_keys=True).encode()).hexdigest()

    def _fingerprint_protocol(self):
        protocol = (self.mode, self.flight_length_mm, self.pulse_mode, self.reference_scale,
                    self.reference_voltage_V, self.reference_flight_length_mm,
                    asdict(self.policy), asdict(self.model.config), self.max_fit_events_per_group,
                    _digest(self.labels, self.apply_mask, *self.roles.values()))
        return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()

    def preview(self):
        self._check_current()
        return dict(status=self.status, events=self.n, apply_events=int(self.apply_mask.sum()),
                    excluded_events=int((~self.apply_mask).sum()), mode=self.mode,
                    role_counts={role: int(mask.sum()) for role, mask in self.roles.items()},
                    role_reference_counts={role: int((mask & np.isfinite(self.labels)).sum()) for role, mask in self.roles.items()},
                    reference_masses_da=np.unique(self.labels[np.isfinite(self.labels)]).tolist(),
                    block_edges=list(self.model.config.block_edges), policy=asdict(self.policy))

    def fit(self):
        self._check_current()
        if self.status not in ("preview", "fitted", "selection_failed", "selection_passed"):
            raise CalibrationStateError("Confirmation freezes the model; create a new session to refit")
        indices = []
        for _, _, mask in self._groups("train"):
            group = np.flatnonzero(mask)
            indices.append(group[np.linspace(0, len(group)-1, min(len(group), self.max_fit_events_per_group), dtype=int)])
        indices = np.sort(np.concatenate(indices))
        self.status, self._model_hash = "preview", None
        self.model.fit(self.features[indices], np.column_stack((self.raw[0][indices], self.labels[indices])))
        self._model_hash = self._fingerprint_model()
        self.diagnostics = {"training": dict(self.model.fit_diagnostics, fit_event_hash=_digest(indices),
                                               available_events=int((self.roles["train"] & np.isfinite(self.labels)).sum()))}
        self.status = "fitted"
        return deepcopy(self.diagnostics["training"])

    def _evaluate(self, role):
        rows, passed = [], True
        for mass, block, mask in self._groups(role):
            features = self.features[mask]
            in_domain = self.model.domain_mask(features)
            row = dict(mass_da=float(mass), block=int(block), events=len(features),
                       outside_domain=int((~in_domain).sum()))
            if not np.all(in_domain):
                row.update(passed=False, reason="outside training domain")
            else:
                relative = (self.model.predict_mass_da(features, self.raw[0][mask]) - mass) / mass
                bias, rmse = float(np.median(relative)), float(np.sqrt(np.mean(relative**2)))
                row.update(relative_median_bias=bias, relative_rmse=rmse,
                           relative_error_quantiles=np.quantile(relative, [.025, .5, .975]).tolist(),
                           passed=abs(bias) <= self.policy.max_abs_relative_bias and rmse <= self.policy.max_relative_rmse)
            passed &= row["passed"]
            rows.append(row)
        return dict(passed=bool(passed), groups=rows)

    def validate_selection(self):
        self._check_current()
        if self.status not in ("fitted", "selection_failed", "selection_passed"):
            raise CalibrationStateError("Fit before selection validation; confirmation remains untouched")
        self.status = "selection_failed"
        result = self._evaluate("selection")
        self.diagnostics["selection"] = result
        self.status = "selection_passed" if result["passed"] else "selection_failed"
        return deepcopy(result)

    def confirm(self):
        self._check_current()
        if self.status != "selection_passed":
            raise CalibrationStateError("Selection validation must pass before one-time confirmation")
        # Mark consumed BEFORE evaluating: even an evaluation error cannot reopen this holdout.
        self.status = "confirmation_failed"
        result = self._evaluate("confirmation")
        self.diagnostics["confirmation"] = result
        if result["passed"]:
            self.status = "confirmed"
        return deepcopy(result)

    def accept(self):
        self._check_current()
        if self.status != "confirmed":
            raise CalibrationStateError("Independent confirmation must pass before acceptance")
        self.status = "accepted"
        self.variables.guided_calibration_receipt = self.receipt()
        return deepcopy(self.variables.guided_calibration_receipt)

    def apply(self, *, active_mode, chunk_size=250_000):
        self._check_current()
        if self.status != "accepted":
            raise CalibrationStateError("Explicit acceptance is required before applying calibration")
        if active_mode != self.mode:
            raise CalibrationStateError("Active ToF/mass mode does not match this session")
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise CalibrationInputError("chunk_size must be a positive integer")
        output = np.full(self.n, np.nan)
        for start in range(0, self.n, chunk_size):
            indices = np.flatnonzero(self.apply_mask[start:start+chunk_size]) + start
            if not len(indices):
                continue
            features = self.features[indices]
            if not np.all(self.model.domain_mask(features)):
                raise CalibrationInputError("Apply includes events outside the training domain; narrow the mask or collect references")
            mass = self.model.predict_mass_da(features, self.raw[0][indices])
            output[indices] = mass if self.mode == "mc" else self.reference_scale * np.sqrt(mass)
        # All chunks validated before a single shared-array write.
        self._check_current()
        attr = self.variables._CALIBRATION_ATTR[self.mode]
        setattr(self.variables, attr + "_backup", self._active.copy())
        setattr(self.variables, attr, output)
        setattr(self.variables, "initial_calibration_done_" + self.mode, True)
        self.variables.clear_calibration_selection_mask()
        self.variables.clear_calibration_peak_range()
        self.variables.guided_calibration_valid_mask = self.apply_mask.copy()
        self._active_hash = _digest(output)
        self.status = "applied"
        self.variables.guided_calibration_receipt = self.receipt()
        return output

    def mass_from_normalized_tof(self, normalized_tof_ns):
        """Use the public mass converter with THIS output's reference geometry and zero t0."""
        self._check_current()
        from pyccapt.calibration.mc.mc_tools import tof2mc
        tof = np.asarray(normalized_tof_ns, dtype=float)
        zeros = np.zeros_like(tof)
        return tof2mc(tof, 0., np.full_like(tof, self.reference_voltage_V), zeros, zeros,
                      self.reference_flight_length_mm, zeros, mode="laser")

    def convert_applied_tof_to_mass(self):
        """Explicitly copy accepted normalized ToF to ``mc_calib`` with matching units."""
        self._check_current()
        if self.mode != "tof" or self.status != "applied":
            raise CalibrationStateError("Apply an accepted ToF session before converting it to mass")
        result = self.mass_from_normalized_tof(self.variables.dld_t_calib)
        self.variables.mc_calib_backup = self.variables.mc_calib.copy()
        self.variables.mc_calib = result
        self.variables.initial_calibration_done_mc = True
        self._copied_to_mass = True
        self.variables.guided_calibration_receipt = self.receipt()
        return result

    def receipt(self):
        return dict(schema_version="guided-calibration-1", status=self.status, mode=self.mode,
                    copied_normalized_tof_to_mass=self._copied_to_mass,
                    dataset_hash=self.dataset_hash, source_hash=self.source_hash,
                    references_hash=_digest(self.labels), roles_hashes={k: _digest(v) for k, v in self.roles.items()},
                    apply_mask_hash=_digest(self.apply_mask), applied_events=int(self.apply_mask.sum()),
                    excluded_events=int((~self.apply_mask).sum()), policy=asdict(self.policy),
                    model=self.model.provenance(), diagnostics=deepcopy(self.diagnostics),
                    input_units="raw ToF ns, detector cm, voltage V, flight length mm, reference mass-to-charge Da",
                    pulse_mode=self.pulse_mode, flight_length_mm=self.flight_length_mm,
                    output=dict(units="Da" if self.mode == "mc" else "ns", t0_ns=0.,
                                convention="inferred mass" if self.mode == "mc" else "normalized reference ToF",
                                reference_voltage_V=self.reference_voltage_V,
                                reference_flight_length_mm=self.reference_flight_length_mm,
                                ns_per_sqrt_Da=self.reference_scale,
                                excluded_policy="NaN"))

    def save_receipt(self, path):
        """Persist diagnostics and hashes; this is not a substitute for saving the raw dataset/masks."""
        self._check_current()
        destination = Path(path)
        destination.write_text(json.dumps(self.receipt(), indent=2, allow_nan=False), encoding="utf-8")
        return destination
