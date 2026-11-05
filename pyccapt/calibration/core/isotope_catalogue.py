"""Bounded, canonical isotope candidates for interactive review (not identification)."""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
import hashlib
from itertools import combinations_with_replacement
from math import comb
from pathlib import Path
from threading import RLock

import numpy as np
import pandas as pd

from pyccapt.calibration.core.exceptions import CalibrationInputError, CalibrationStateError


@dataclass(frozen=True)
class IonCandidate:
    key: str
    ion: str
    mass: float
    element: tuple
    isotope: tuple
    complex: tuple
    charge: int
    abundance_product: float

    @property
    def family(self):
        counts = Counter()
        for element, count in zip(self.element, self.complex):
            counts[element] += count
        return tuple(sorted(counts.items())), self.charge

    def record(self, target=None):
        value = asdict(self)
        for field in ("element", "isotope", "complex"):
            value[field] = list(value[field])
        value["family"] = [[list(pair) for pair in self.family[0]], self.charge]
        if target is not None:
            value["mass_residual_Da"] = float(target - self.mass)
            value["ppm_residual"] = float((target-self.mass)/self.mass*1e6)
        return value


class IsotopeCatalogue:
    """Sorted immutable candidates; searches inspect only the requested mass window."""
    def __init__(self, candidates, key):
        self.candidates = tuple(sorted(candidates, key=lambda item: (item.mass, item.key)))
        self.masses = np.asarray([item.mass for item in self.candidates])
        self.masses.flags.writeable = False
        self.key = key

    def search(self, mass, tolerance=.15, limit=30):
        if not np.isfinite([mass, tolerance]).all() or tolerance <= 0 or int(limit) < 1:
            raise CalibrationInputError("Mass and tolerance must be finite; tolerance and limit positive")
        left, right = np.searchsorted(self.masses, [mass-tolerance, mass+tolerance], side="left")
        right = np.searchsorted(self.masses, mass+tolerance, side="right")
        options = self.candidates[left:right]
        # A deterministic display order, deliberately not a probability model.
        return sorted(options, key=lambda item: (abs(item.mass-mass), sum(item.complex), item.key))[:int(limit)]


_CACHE = OrderedDict()
_LOCK = RLock()


def get_isotope_catalogue(elements, charges=(1, 2, 3), max_atoms=2, *, table=None,
                          max_candidates=100000, cancel=None, progress=None):
    """Cache by selected elements/charges/complexity AND isotope-table content hash.

    Canonical multisets avoid generating molecular permutations. A strict size
    bound rejects combinatorial requests before allocation; no silent truncation.
    Natural isotope abundances are metadata only, never identity probabilities.
    """
    if cancel is not None and cancel.is_set():
        raise CalibrationStateError("Catalogue generation cancelled")
    if table is None:
        table = pd.read_hdf(Path(__file__).resolve().parents[2] / "files" / "isotopeTable.h5")
    required = ["element", "isotope", "weight", "abundance"]
    if not set(required).issubset(table.columns):
        raise CalibrationInputError("Isotope table requires element, isotope, weight and abundance")
    selected_elements = tuple(sorted(set(part.strip() for part in
                                        (elements.split(",") if isinstance(elements, str) else elements) if part.strip())))
    selected_charges = tuple(sorted(set(charges)))
    if (not selected_elements or not selected_charges or
            any(isinstance(c, bool) or not isinstance(c, (int, np.integer)) or not 1 <= c <= 32 for c in selected_charges) or
            isinstance(max_atoms, bool) or not isinstance(max_atoms, (int, np.integer)) or not 1 <= max_atoms <= 6 or
            not isinstance(max_candidates, (int, np.integer)) or not 1 <= max_candidates <= 1000000):
        raise CalibrationInputError("Select elements, integer charges 1–32, atom limit 1–6 and candidate budget 1–1,000,000")
    missing = set(selected_elements) - set(table.element.astype(str))
    if missing:
        raise CalibrationInputError(f"Elements missing from isotope table: {sorted(missing)}")
    content_hash = hashlib.sha256(pd.util.hash_pandas_object(table[required], index=False).values.tobytes()).hexdigest()
    key = (selected_elements, selected_charges, int(max_atoms), content_hash)
    with _LOCK:
        if key in _CACHE:
            result = _CACHE.pop(key)
            if len(result.candidates) > max_candidates:
                raise CalibrationInputError("Cached catalogue exceeds requested candidate budget")
            _CACHE[key] = result
            return result
    rows = table.loc[table.element.isin(selected_elements), required].sort_values(["element", "isotope"])
    if rows.duplicated(["element", "isotope"]).any():
        raise CalibrationInputError("Isotope table contains duplicate element/isotope rows")
    number = len(rows)
    estimate = sum(comb(number+k-1, k) for k in range(1, max_atoms+1)) * len(selected_charges)
    if estimate > max_candidates:
        raise CalibrationInputError(f"Requested catalogue has {estimate:,} candidates; reduce elements/charges/atoms (limit {max_candidates:,})")
    atoms = list(rows.itertuples(index=False, name=None))
    if any(not np.isfinite([mass, abundance]).all() or mass <= 0 or not 0 <= abundance <= 100
           for _, _, mass, abundance in atoms):
        raise CalibrationInputError("Invalid isotope masses or percentage abundances")
    candidates = []
    for size in range(1, max_atoms+1):
        for combo in combinations_with_replacement(range(number), size):
            if cancel is not None and cancel.is_set():
                raise CalibrationStateError("Catalogue generation cancelled")
            counts = Counter(combo)
            entries = [atoms[i] for i in counts]
            composition = tuple(counts.values())
            mass = sum(atoms[i][2] for i in combo)
            abundance = float(np.prod([atoms[i][3]/100 for i in combo]))
            formula = "".join(f"{{{int(row[1])}}}{row[0]}{count if count > 1 else ''}"
                              for row, count in zip(entries, composition))
            for charge in selected_charges:
                ion_key = f"{formula}^{charge}+"
                candidates.append(IonCandidate(ion_key, ion_key, float(mass/charge),
                    tuple(str(row[0]) for row in entries), tuple(int(row[1]) for row in entries),
                    composition, int(charge), abundance))
            if progress and len(candidates) % 1000 < len(selected_charges):
                progress(len(candidates)/max(estimate, 1), "Building isotope catalogue")
    result = IsotopeCatalogue(candidates, key)
    with _LOCK:
        _CACHE[key] = result
        while len(_CACHE) > 4:
            _CACHE.popitem(last=False)
    if progress:
        progress(1., "Isotope catalogue ready")
    return result
