"""Human-reviewed mass windows with reversible decisions and compact assignment."""
from __future__ import annotations

from collections import Counter, OrderedDict
from copy import deepcopy
import hashlib

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths

from pyccapt.calibration.core.exceptions import CalibrationInputError, CalibrationStateError
from pyccapt.calibration.core.range_assignment import assign_intervals


def _cancelled(cancel):
    if cancel is not None and cancel.is_set():
        raise CalibrationStateError("Operation cancelled; previous decisions retained")


def _family(candidate):
    counts = Counter()
    for element, count in zip(candidate["element"], candidate["complex"]):
        counts[str(element)] += int(count)
    return tuple(sorted(counts.items())), int(candidate["charge"])


class PeakReviewQueue:
    """Proposals never write ranges. Only explicit accepted decisions become ranges.

    Rerunning detection on the same immutable mass vector retains locked accepted
    and not-a-peak decisions. Changed calibration requires a new queue.
    """
    def __init__(self, masses, *, mass_revision=0):
        self.masses = np.array(masses, dtype=float, copy=True)
        if self.masses.ndim != 1 or not np.isfinite(self.masses).any():
            raise CalibrationInputError("Review requires a nonempty finite mass spectrum")
        self.masses.flags.writeable = False
        self.mass_hash = hashlib.sha256(self.masses.tobytes()).hexdigest()
        self.mass_revision = int(mass_revision)
        self.sorted_mass = np.sort(self.masses[np.isfinite(self.masses)])
        self.records, self.history = [], []
        self.current, self.counter = 0, 0
        self._histograms = OrderedDict()
        self.catalogue = None
        self.catalogue_spec = None
        self._catalogues = OrderedDict()
        self.tolerance = .15

    def _snapshot(self):
        self.history.append({"records": deepcopy(self.records), "current": self.current, "counter": self.counter,
                             "catalogue_spec": deepcopy(self.catalogue_spec), "tolerance": self.tolerance})
        self.history = self.history[-50:]

    def _get(self, peak_id):
        for record in self.records:
            if record["id"] == peak_id:
                return record
        raise CalibrationInputError(f"Unknown review peak {peak_id}")

    def _new(self, center, lower, upper, catalogue=None):
        self.counter += 1
        key = hashlib.sha256(f"{self.mass_hash}:{self.counter}".encode()).hexdigest()[:16]
        return {"id": key, "center": float(center), "lower": float(lower), "upper": float(upper),
                "status": "proposed", "locked": False, "chosen": None, "rejected": [],
                "candidates": [item.record(center) for item in catalogue.search(center, self.tolerance)] if catalogue else [],
                "reviewed_ambiguity": False, "family_reviewed": False, "evidence": {}}

    def _ensure_catalogue(self):
        if self.catalogue is None and self.catalogue_spec is not None:
            from pyccapt.calibration.core.isotope_catalogue import get_isotope_catalogue
            elements, charges, atoms, content_hash = self.catalogue_spec
            candidate = get_isotope_catalogue(elements, charges, atoms)
            if candidate.key[-1] != content_hash:
                raise CalibrationStateError("Saved isotope table differs from the installed table; rerun proposals and review the new candidates")
            self.catalogue = candidate

    def histogram(self, bin_width=.02, lower=None, upper=None, *, cancel=None, progress=None):
        _cancelled(cancel)
        lower = float(self.sorted_mass[0] if lower is None else lower)
        upper = float(self.sorted_mass[-1] if upper is None else upper)
        if not np.isfinite([lower, upper, bin_width]).all() or upper <= lower or bin_width <= 0:
            raise CalibrationInputError("Histogram bounds and bin width must be finite and increasing")
        bins = int(np.ceil((upper-lower)/bin_width))
        if bins > 200000:
            raise CalibrationInputError("Histogram exceeds 200,000 bins; widen the bin size or narrow the mass window")
        key = (lower, upper, float(bin_width))
        if key not in self._histograms:
            edges = lower + np.arange(bins+1)*bin_width
            counts = np.zeros(bins, dtype=np.int64)
            first = np.searchsorted(self.sorted_mass, edges[0], side="left")
            last = np.searchsorted(self.sorted_mass, edges[-1], side="right")
            local = self.sorted_mass[first:last]
            for start in range(0, len(local), 250000):
                _cancelled(cancel)
                chunk = local[start:start+250000]
                counts += np.histogram(chunk, bins=edges)[0]
                if progress:
                    progress(min(1., (start+len(chunk))/len(local)), "Building spectrum")
            _cancelled(cancel)
            edges.flags.writeable = counts.flags.writeable = False
            self._histograms[key] = counts, edges
            while len(self._histograms) > 3:
                self._histograms.popitem(last=False)
        return self._histograms[key]

    def propose(self, catalogue, *, bin_width=.02, prominence=20, min_distance_Da=.1,
                tolerance=.15, lower=None, upper=None, max_peaks=500, cancel=None, progress=None):
        if not np.isfinite([prominence, min_distance_Da, tolerance]).all() or min(prominence, min_distance_Da, tolerance) <= 0:
            raise CalibrationInputError("Prominence, peak distance and candidate tolerance must be positive")
        counts, edges = self.histogram(bin_width, lower, upper, cancel=cancel, progress=progress)
        found, info = find_peaks(counts, prominence=prominence, distance=max(1, int(np.ceil(min_distance_Da/bin_width))))
        if len(found) > max_peaks:
            raise CalibrationInputError(f"Found {len(found)} peaks; increase prominence or narrow the spectrum (limit {max_peaks})")
        widths = peak_widths(counts, found, rel_height=.95) if len(found) else ([], [], [], [])
        centers = (edges[:-1]+edges[1:])/2
        old = self.records
        proposed, used = [], set()
        counter_before = self.counter
        tolerance_before = self.tolerance
        self.tolerance = float(tolerance)
        try:
            for i, index in enumerate(found):
                _cancelled(cancel)
                center = float(centers[index])
                left = max(edges[0], float(np.interp(widths[2][i], np.arange(len(centers)), centers))-bin_width)
                right = min(edges[-1], float(np.interp(widths[3][i], np.arange(len(centers)), centers))+bin_width)
                matches = [r for r in old if r["lower"] <= center < r["upper"] and r["id"] not in used]
                existing = min(matches, key=lambda r: abs(r["center"]-center)) if matches else None
                if existing:
                    row = deepcopy(existing)
                    used.add(row["id"])
                    if not row["locked"]:
                        row.update(center=center, chosen=None, reviewed_ambiguity=False, family_reviewed=False, evidence={})
                        row["candidates"] = [item.record(center) for item in catalogue.search(center, tolerance)]
                else:
                    row = self._new(center, left, right, catalogue)
                row["evidence"]["detection"] = {"prominence_counts": float(info["prominences"][i]),
                    "bin_width_Da": bin_width, "proposal_only": True}
                proposed.append(row)
                if progress:
                    progress((i+1)/max(1, len(found)), f"Proposing peak {i+1}/{len(found)}")
            # A locked decision is never discarded simply because detection settings changed.
            proposed.extend(deepcopy(r) for r in old if r["locked"] and r["id"] not in used)
            _cancelled(cancel)
        except Exception:
            self.counter = counter_before
            self.tolerance = tolerance_before
            raise
        self._snapshot()
        self.history[-1]["counter"] = counter_before
        self.history[-1]["tolerance"] = tolerance_before
        self.records = sorted(proposed, key=lambda r: (r["center"], r["id"]))
        self.current, self.catalogue = 0, catalogue
        self.catalogue_spec = [list(catalogue.key[0]), list(catalogue.key[1]), catalogue.key[2], catalogue.key[3]]
        self._catalogues[repr(self.catalogue_spec)] = catalogue
        while len(self._catalogues) > 4:
            self._catalogues.popitem(last=False)
        return len(self.records)

    def warnings(self, peak_id):
        row = self._get(peak_id)
        warnings = []
        overlaps = [r["id"] for r in self.records if r["id"] != peak_id and r["status"] != "not_peak"
                    and max(row["lower"], r["lower"]) < min(row["upper"], r["upper"])]
        if overlaps:
            warnings.append("Window overlaps another peak; hard assignment uses first accepted row. Split/edit before acceptance when possible.")
        options = [r for r in row["candidates"] if r["key"] not in row["rejected"]]
        if len(options) > 1:
            warnings.append("Multiple plausible identities; mass proximity is not an identity probability.")
        if row["chosen"] and not row["lower"] <= row["chosen"]["mass"] < row["upper"]:
            warnings.append("Ideal mass lies outside this window; inspect calibration and identity.")
        family = row["evidence"].get("mixture", {})
        if family.get("diagnostics", {}).get("ambiguous", False):
            warnings.append("Mixture fit is ambiguous; fitted component counts do not identify individual events.")
        return warnings

    def count(self, peak_id):
        row = self._get(peak_id)
        bounds = np.searchsorted(self.sorted_mass, [row["lower"], row["upper"]], side="left")
        return int(bounds[1]-bounds[0])

    def choose(self, peak_id, candidate_key):
        row = self._get(peak_id)
        matches = [item for item in row["candidates"] if item["key"] == candidate_key and item["key"] not in row["rejected"]]
        if len(matches) != 1:
            raise CalibrationInputError("Choose an available candidate from this peak's list")
        self._snapshot()
        row.update(chosen=deepcopy(matches[0]), status="proposed", locked=False, reviewed_ambiguity=False,
                   family_reviewed=False, evidence={})

    def decide(self, peak_id, action, *, ambiguity_reviewed=False):
        row = self._get(peak_id)
        if action not in {"accept", "reject_identity", "not_peak", "defer"}:
            raise CalibrationInputError("Unknown peak-review decision")
        if action == "accept":
            if row["chosen"] is None:
                raise CalibrationStateError("Choose an identity before accepting")
            if self.count(peak_id) == 0:
                raise CalibrationStateError("This interval contains no events; edit its bounds before acceptance")
            if self.warnings(peak_id) and not ambiguity_reviewed:
                raise CalibrationStateError("Review the displayed ambiguity/overlap warnings before acceptance")
        if action == "reject_identity" and row["chosen"] is None:
            raise CalibrationStateError("Choose the identity to reject")
        self._snapshot()
        if action == "reject_identity":
            row["rejected"].append(row["chosen"]["key"])
            row.update(chosen=None, status="proposed", locked=False, family_reviewed=False, evidence={})
        else:
            row.update(status={"accept": "accepted", "not_peak": "not_peak", "defer": "deferred"}[action],
                       locked=action in {"accept", "not_peak"}, reviewed_ambiguity=bool(ambiguity_reviewed))

    def edit_bounds(self, peak_id, lower, upper):
        if not np.isfinite([lower, upper]).all() or upper <= lower:
            raise CalibrationInputError("Range bounds must be finite and increasing")
        row = self._get(peak_id)
        self._snapshot()
        row.update(lower=float(lower), upper=float(upper), status="proposed", locked=False,
                   reviewed_ambiguity=False, family_reviewed=False, evidence={})

    def split(self, peak_id, at):
        row = self._get(peak_id)
        if not row["lower"] < at < row["upper"]:
            raise CalibrationInputError("Split mass must be inside the peak window")
        self._ensure_catalogue()
        self._snapshot()
        children = [self._new((a+b)/2, a, b, self.catalogue) for a, b in
                    ((row["lower"], at), (at, row["upper"]))]
        self.records = [r for r in self.records if r["id"] != peak_id] + children
        self.records.sort(key=lambda r: r["center"])
        return [r["id"] for r in children]

    def merge(self, peak_ids):
        rows = [self._get(key) for key in dict.fromkeys(peak_ids)]
        if len(rows) < 2:
            raise CalibrationInputError("Select at least two peaks to merge")
        self._ensure_catalogue()
        lower, upper = min(r["lower"] for r in rows), max(r["upper"] for r in rows)
        self._snapshot()
        merged = self._new((lower+upper)/2, lower, upper, self.catalogue)
        self.records = [r for r in self.records if r["id"] not in peak_ids] + [merged]
        self.records.sort(key=lambda r: r["center"])
        return merged["id"]

    def accept_family(self, peak_ids, *, consistency_reviewed=False, residual_tolerance=.05, ambiguity_reviewed=False):
        rows = [self._get(key) for key in dict.fromkeys(peak_ids)]
        if not consistency_reviewed or len(rows) < 2 or any(r["chosen"] is None for r in rows):
            raise CalibrationStateError("Review at least two chosen identities and explicitly confirm family consistency")
        if len({_family(r["chosen"]) for r in rows}) != 1:
            raise CalibrationInputError("Selected identities do not have the same elemental composition and charge")
        if len({r["chosen"]["key"] for r in rows}) != len(rows) or any(self.count(r["id"]) == 0 for r in rows):
            raise CalibrationInputError("A family needs distinct isotope identities in nonempty peak intervals")
        residuals = [r["center"]-r["chosen"]["mass"] for r in rows]
        if not np.isfinite(residual_tolerance) or residual_tolerance <= 0 or np.ptp(residuals) > residual_tolerance:
            raise CalibrationInputError("Family mass residuals are inconsistent; review calibration/identities")
        if any(self.warnings(r["id"]) for r in rows) and not ambiguity_reviewed:
            raise CalibrationStateError("Review family ambiguity and overlap before accepting")
        self._snapshot()
        for row in rows:
            row.update(status="accepted", locked=True, family_reviewed=True, reviewed_ambiguity=bool(ambiguity_reviewed))

    def next(self):
        for offset in range(1, len(self.records)+1):
            index = (self.current+offset) % len(self.records)
            if not self.records[index]["locked"]:
                self.current = index
                break
        return self.records[self.current] if self.records else None

    def undo(self):
        if not self.history:
            raise CalibrationStateError("No review decision to undo")
        previous = self.history.pop()
        self.records, self.current, self.counter = previous["records"], previous["current"], previous["counter"]
        self.catalogue_spec, self.tolerance = previous.get("catalogue_spec"), previous.get("tolerance", .15)
        self.catalogue = self._catalogues.get(repr(self.catalogue_spec))

    def ranges(self):
        rows = []
        for item in self.records:
            if item["status"] != "accepted":
                continue
            candidate = item["chosen"]
            rows.append({"name": candidate["ion"], "ion": candidate["ion"], "mass": candidate["mass"],
                "mc": item["center"], "mc_low": item["lower"], "mc_up": item["upper"],
                "color": "#" + hashlib.sha256(candidate["key"].encode()).hexdigest()[:6],
                "element": list(candidate["element"]), "complex": list(candidate["complex"]),
                "isotope": list(candidate["isotope"]), "charge": candidate["charge"], "review_id": item["id"]})
        return pd.DataFrame(rows)

    def assignments(self):
        ranges = self.ranges()
        return assign_intervals(self.masses, ranges.mc_low, ranges.mc_up) if len(ranges) else np.full(len(self.masses), -1, np.int32)

    def profile_evidence(self, peak_id):
        """Optional existing asymmetric/Voigt evidence; never an acceptance action."""
        from pyccapt.calibration.core.mc_plot_peak_helpers import gaussian_mrp_report
        row = self._get(peak_id)
        report = gaussian_mrp_report(self.masses, row["lower"], row["upper"],
                                     bin_size=max(.0001, (row["upper"]-row["lower"])/100))
        if report is None:
            raise CalibrationStateError("Insufficient peak evidence for profile fitting")
        evidence = {key: report.get(key) for key in ("peak_position", "num_ions", "recommended_label",
            "recommended_mrp", "window_warning", "asymmetric_ok", "asymmetric_fwhm")}
        row["evidence"]["profile"] = evidence
        return evidence

    def mixture_evidence(self, peak_ids, reference_id, *, identities_reviewed=False, bin_width=.005):
        from pyccapt.calibration.core.isotope_family import build_empirical_template, fit_isotope_family
        rows = [self._get(key) for key in peak_ids]
        reference = self._get(reference_id)
        if not identities_reviewed or not rows or any(r["chosen"] is None for r in rows) or reference["status"] != "accepted":
            raise CalibrationStateError("Confirm component identities and select an accepted isolated reference peak")
        if any(max(reference["lower"], r["lower"]) < min(reference["upper"], r["upper"])
               for r in self.records if r["id"] != reference_id and r["status"] != "not_peak"):
            raise CalibrationInputError("Reference window overlaps another peak; use an isolated template")
        reference_counts, reference_edges = self.histogram(bin_width, reference["lower"], reference["upper"])
        template = build_empirical_template((reference_edges[:-1]+reference_edges[1:])/2, reference_counts,
                                             reference_mass=reference["chosen"]["mass"])
        counts, edges = self.histogram(bin_width, min(r["lower"] for r in rows), max(r["upper"] for r in rows))
        fit = fit_isotope_family((edges[:-1]+edges[1:])/2, counts,
                                 isotope_masses=[r["chosen"]["mass"] for r in rows], template=template)
        for row in rows:
            row["evidence"]["mixture"] = fit.summary()
        rows[0]["evidence"]["mixture_curve"] = {"mass": (edges[:-1]+edges[1:])/2,
            "observed": counts, "prediction": fit.prediction, "residual": counts-fit.prediction,
            "validation_mask": fit.validation_mask}
        return fit

    def to_state(self):
        return deepcopy({"mass_hash": self.mass_hash, "mass_revision": self.mass_revision,
            "records": self.records, "current": self.current, "counter": self.counter,
            "history": self.history, "tolerance": self.tolerance, "catalogue_spec": self.catalogue_spec})

    @classmethod
    def from_state(cls, masses, state):
        queue = cls(masses, mass_revision=state["mass_revision"])
        if queue.mass_hash != state["mass_hash"]:
            raise CalibrationStateError("Saved review decisions refer to a different mass spectrum")
        queue.records, queue.history = deepcopy(state["records"]), deepcopy(state["history"])
        queue.current, queue.counter, queue.tolerance = state["current"], state["counter"], state["tolerance"]
        queue.catalogue_spec = deepcopy(state.get("catalogue_spec"))
        for records in [queue.records, *(snapshot["records"] for snapshot in queue.history)]:
            ids = [row["id"] for row in records]
            if len(set(ids)) != len(ids):
                raise CalibrationStateError("Saved review contains duplicate peak IDs")
            for row in records:
                if (not np.isfinite([row["center"], row["lower"], row["upper"]]).all() or row["lower"] >= row["upper"] or
                        row["status"] not in {"proposed", "deferred", "accepted", "not_peak"}):
                    raise CalibrationStateError("Saved review contains invalid peak bounds/status")
                if row["status"] == "accepted" and (row["chosen"] is None or not row["locked"]):
                    raise CalibrationStateError("Saved accepted review has no locked identity")
                for candidate in row["candidates"] + ([row["chosen"]] if row["chosen"] else []):
                    if (not np.isfinite(candidate["mass"]) or candidate["mass"] <= 0 or
                            not len(candidate["element"]) == len(candidate["complex"]) == len(candidate["isotope"]) or
                            not candidate["element"] or any(not isinstance(c, int) or c <= 0 for c in candidate["complex"]) or
                            not isinstance(candidate["charge"], int) or candidate["charge"] <= 0):
                        raise CalibrationStateError("Saved review candidate has invalid mass/stoichiometry")
        if (not isinstance(queue.current, int) or not 0 <= queue.current < max(1, len(queue.records)) or
                not isinstance(queue.counter, int) or queue.counter < 0 or not np.isfinite(queue.tolerance) or queue.tolerance <= 0):
            raise CalibrationStateError("Saved review cursor or tolerance is invalid")
        return queue
