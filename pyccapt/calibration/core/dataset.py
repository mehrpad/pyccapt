"""Schema-aware calibration dataset boundary."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from pyccapt.calibration.core.exceptions import CalibrationInputError


SCHEMA_VERSION = "1.1"
EVENT_ID_COLUMN = "event_id"
UNITS = MappingProxyType(
    {
        "high_voltage (V)": "V",
        "pulse_v (V)": "V",
        "pulse_l (pJ)": "pJ",
        "t (ns)": "ns",
        "t_c (ns)": "ns",
        "x_det (cm)": "cm",
        "y_det (cm)": "cm",
        "mc (Da)": "Da",
        "mc_uc (Da)": "Da",
        "x (nm)": "nm",
        "y (nm)": "nm",
        "z (nm)": "nm",
    }
)


@dataclass(frozen=True)
class CalibrationDataset:
    """One normalized, aligned calibration table plus derived capabilities."""

    frame: pd.DataFrame
    units: Mapping[str, str]
    capabilities: Mapping[str, bool]
    finite_masks: Mapping[str, np.ndarray]
    source_hash: str
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "CalibrationDataset":
        if not isinstance(frame, pd.DataFrame):
            raise CalibrationInputError("Calibration data must be a pandas DataFrame")
        normalized = frame.reset_index(drop=True).copy()
        row_count = len(normalized)
        if EVENT_ID_COLUMN not in normalized:
            normalized[EVENT_ID_COLUMN] = np.arange(row_count, dtype=np.uint64)
        else:
            ids = normalized[EVENT_ID_COLUMN].to_numpy()
            if ids.dtype.kind not in 'iu' or np.any(ids < 0) or len(np.unique(ids)) != row_count:
                raise CalibrationInputError('event_id must contain unique nonnegative integers; floats are not event identities')
            normalized[EVENT_ID_COLUMN] = ids.astype(np.uint64, copy=False)
        for column in normalized.columns:
            if len(normalized[column]) != row_count:
                raise CalibrationInputError(f"Column {column!r} is not aligned to {row_count} rows")

        finite_masks: dict[str, np.ndarray] = {}
        for column in UNITS:
            if column in normalized.columns and column not in normalized.attrs.get('pyccapt_unavailable_columns', []):
                try:
                    finite_masks[column] = np.isfinite(normalized[column].to_numpy(dtype=float, copy=False))
                except (TypeError, ValueError):
                    finite_masks[column] = np.zeros(row_count, dtype=bool)
            else:
                finite_masks[column] = np.zeros(row_count, dtype=bool)

        detector_mask = finite_masks["x_det (cm)"] & finite_masks["y_det (cm)"]
        reconstruction_mask = finite_masks["x (nm)"] & finite_masks["y (nm)"] & finite_masks["z (nm)"]
        mass_mask = finite_masks["mc (Da)"] | finite_masks["mc_uc (Da)"]
        tof_mask = finite_masks["t (ns)"]
        finite_masks.update(
            detector_positions=detector_mask,
            reconstruction=reconstruction_mask,
            mass_spectrum=mass_mask,
            time_of_flight=tof_mask,
        )
        capabilities = MappingProxyType(
            {
                "detector_positions": bool(detector_mask.any()),
                "reconstruction": bool(reconstruction_mask.any()),
                "mass_spectrum": bool(mass_mask.any()),
                "time_of_flight": bool(tof_mask.any()),
            }
        )
        row_hashes = pd.util.hash_pandas_object(normalized, index=False).to_numpy(dtype=np.uint64)
        digest = hashlib.sha256(repr([(str(c), str(normalized[c].dtype)) for c in normalized]).encode())
        digest.update(row_hashes.tobytes())
        source_hash = digest.hexdigest()
        # Pandas preserves attrs through ordinary slicing/reset_index. Identity
        # is (source_id, event_id), never a wrapping hardware pulse counter.
        normalized.attrs.setdefault('pyccapt_source_id', source_hash)
        return cls(
            frame=normalized,
            units=UNITS,
            capabilities=capabilities,
            finite_masks=MappingProxyType(finite_masks),
            source_hash=source_hash,
        )

    @property
    def row_count(self) -> int:
        return len(self.frame)

    @property
    def event_ids(self) -> np.ndarray:
        ids = self.frame[EVENT_ID_COLUMN].to_numpy(copy=True)
        ids.flags.writeable = False
        return ids

    @property
    def source_id(self) -> str:
        return str(self.frame.attrs['pyccapt_source_id'])

    def validity_status(self, column: str) -> np.ndarray:
        """0=measured (zero is valid), 1=unknown/nonfinite, 2=unavailable field."""
        if column not in self.frame or column in self.frame.attrs.get('pyccapt_unavailable_columns', []):
            return np.full(self.row_count, 2, dtype=np.uint8)
        if column not in self.finite_masks:
            values = self.frame[column].to_numpy()
            if values.dtype.kind in 'biufc':
                return (~np.isfinite(values)).astype(np.uint8)
            return np.asarray(self.frame[column].isna(), dtype=np.uint8)
        return (~self.finite_masks[column]).astype(np.uint8)

    def values(self, column: str, *, missing: float = 0.0) -> np.ndarray:
        if column in self.frame.columns:
            return self.frame[column].to_numpy()
        return np.full(self.row_count, missing, dtype=float)

    def finite_mask(self, *columns_or_capabilities: str) -> np.ndarray:
        mask: np.ndarray = np.ones(self.row_count, dtype=bool)
        for name in columns_or_capabilities:
            try:
                mask &= self.finite_masks[name]
            except KeyError as exc:
                raise CalibrationInputError(f"Unknown finite-mask field: {name}") from exc
        return mask
