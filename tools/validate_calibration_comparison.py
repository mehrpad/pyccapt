"""Compare saved calibration with an explicitly conditional physical-model fit.

Historical range identities are assumptions, not independent ground truth.
This diagnostic never accepts/applies a calibration or edits either input.
Selection windows come from the saved calibration, so reported errors are
conditional on that selection. No confirmation data is evaluated if selection
fails. See docs/workflow_validation.md for the interpretation of real runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_masks(frame, ranges, rows, isotope_table):
    """Resolve explicit historical compositions; reject overlaps/duplicate masses."""
    if not rows or len(set(rows)) != len(rows):
        raise ValueError("Reference row indices must be nonempty and unique")
    mass = frame["mc (Da)"].to_numpy()
    masks, descriptions = {}, []
    selected = np.zeros(len(frame), dtype=bool)
    for index in rows:
        if index < 0 or index >= len(ranges):
            raise ValueError("Reference row index is outside the range table")
        row = ranges.iloc[index]
        charge = float(row["charge"])
        components = [row[name] for name in ("element", "isotope", "complex")]
        if (not np.isfinite(charge) or charge <= 0 or charge != int(charge) or
                not len(components[0]) or len({len(values) for values in components}) != 1):
            raise ValueError("Reference composition and positive integer charge are required")
        target = 0.
        for element, isotope, count in zip(*components):
            if not np.isfinite(count) or count <= 0 or count != int(count):
                raise ValueError("Reference atom counts must be positive integers")
            matches = isotope_table[(isotope_table.element == element) & (isotope_table.isotope == isotope)]
            if len(matches) != 1:
                raise ValueError(f"Reference isotope {element}-{isotope} must match exactly one catalogue entry")
            target += float(matches.iloc[0].weight) * count / charge
        lo, hi = float(row.mc_low), float(row.mc_up)
        if not np.isfinite([lo, hi, target]).all() or lo >= hi or target <= 0:
            raise ValueError("Reference bounds and target must be finite and valid")
        mask = (mass >= lo) & (mass < hi)
        if not mask.any() or (selected & mask).any() or target in masks:
            raise ValueError("Reference windows must be populated, disjoint and have distinct target masses")
        selected |= mask
        masks[target] = mask
        neighbours = []
        if len(components[0]) == 1 and components[2][0] == 1:
            for isotope in isotope_table[isotope_table.element == components[0][0]].itertuples():
                mass_to_charge = float(isotope.weight) / charge
                if isotope.isotope != components[1][0] and lo <= mass_to_charge < hi:
                    neighbours.append(dict(element=str(isotope.element), isotope=int(isotope.isotope),
                                           mass_Da=mass_to_charge))
        descriptions.append(dict(row=index, historical_ion=str(row.ion), target_Da=target,
            bounds_Da=[lo, hi], events=int(mask.sum()), nominal_target_inside_window=bool(lo <= target < hi),
            other_same_element_isotopes_inside_window=neighbours,
            interpretation="Window membership is conditional evidence, not a pure or independently identified reference"))
    return masks, descriptions


def metrics(values, target):
    error = (values - target) / target
    return dict(events=len(values), relative_median_bias=float(np.median(error)),
                relative_rmse=float(np.sqrt(np.mean(error**2))),
                central_90_percent_width_Da=float(np.diff(np.quantile(values, [.05, .95]))[0]))


def compare(data, ranges_path, destination, *, rows, flight_mm, t0_ns, pulse_mode, seed=2643):
    from pyccapt.calibration.core.calibration_sampling import _voltage_segments
    from pyccapt.calibration.core.flight_path_t0 import load_isotope_table
    from pyccapt.calibration.core.guided_calibration import GuidedCalibrationSession
    from pyccapt.calibration.core.share_variables import Variables
    from pyccapt.calibration.mc.mc_tools import tof2mc

    data, ranges_path, destination = (Path(path).resolve() for path in (data, ranges_path, destination))
    if destination in (data, ranges_path) or (destination.exists() and any(
            source.exists() and destination.samefile(source) for source in (data, ranges_path))):
        raise ValueError("Diagnostic output must not overwrite an input")
    if not np.isfinite([flight_mm, t0_ns]).all() or flight_mm <= 0:
        raise ValueError("Flight length must be positive; instrument settings must be finite")
    before = {"data": fingerprint(data), "ranges": fingerprint(ranges_path)}
    started = time.perf_counter()
    frame = pd.read_hdf(data, key="df", mode="r")
    ranges = pd.read_hdf(ranges_path, key="df", mode="r")
    references, descriptions = reference_masks(frame, ranges, rows, load_isotope_table())
    raw_t = frame["t (ns)"].to_numpy()
    voltage = frame["high_voltage (V)"].to_numpy()
    x, y = frame["x_det (cm)"].to_numpy(), frame["y_det (cm)"].to_numpy()
    pulse = frame["pulse_v (V)"].to_numpy()
    physical = tof2mc(raw_t, t0_ns, voltage, x, y, flight_mm, pulse, mode=pulse_mode)
    report = dict(source=str(data), ranges=str(ranges_path), source_sha256=before, events=len(frame),
                  assumptions="Historical identities and saved-calibration windows are conditional labels, not ground truth. "
                              "Seeded interleaved holdouts are not independent experiments. No model is accepted or applied.",
                  instrument=dict(flight_mm=flight_mm, t0_ns=t0_ns, pulse_mode=pulse_mode),
                  references=descriptions, split_seed=seed, model_config="default ToFModelConfig; no drift terms",
                  invalid_physical_elapsed_events=int(np.count_nonzero(raw_t <= t0_ns)),
                  physical_vs_saved_uncorrected=dict(
                      max_abs_difference_Da=float(np.max(np.abs(physical-frame["mc_uc (Da)"].to_numpy()))),
                      allclose_rtol_1e_12=bool(np.allclose(physical, frame["mc_uc (Da)"], rtol=1e-12, atol=1e-12))))
    # Exact grouping benchmark on the identical reference population and bin boundaries.
    mask = np.logical_or.reduce(list(references.values()))
    volts, values = voltage[mask], frame["mc (Da)"].to_numpy()[mask]
    step = 100.
    lower, upper = float(volts.min()), float(volts.max())
    count = max(1, int(np.ceil((upper-lower)/step)))
    def repeated_masks():
        result = []
        for i in range(count):
            selected = volts >= lower+i*step
            if i < count-1:
                selected &= volts < lower+(i+1)*step
            result.append((volts[selected], values[selected]))
        return result
    timings = {"repeated_masks": [], "grouped": []}
    for repeat in range(3):
        for name, operation in (("repeated_masks", repeated_masks), ("grouped", lambda: _voltage_segments(volts, values, step))):
            tick = time.perf_counter()
            result = operation()
            timings[name].append(time.perf_counter()-tick)
            if repeat == 0 and name == "repeated_masks":
                expected = result
            else:
                assert len(result) == len(expected)
                for actual, wanted in zip(result, expected):
                    np.testing.assert_array_equal(actual[0], wanted[0])
                    np.testing.assert_array_equal(actual[1], wanted[1])
    report["grouping_benchmark"] = dict(events=len(volts), bins=count, step_V=step,
                                        exact_values_and_order=True, timings_seconds=timings,
                                        ratio_repeated_over_grouped=float(np.median(timings["repeated_masks"])/np.median(timings["grouped"])))
    variables = Variables()
    variables.sync_from_data(frame)
    original_active = variables.get_calibration_array("mc").copy()
    split = np.random.default_rng(seed).integers(0, 5, len(frame), dtype=np.int8)
    roles = dict(train=split < 3, selection=split == 3, confirmation=split == 4)
    session = GuidedCalibrationSession(variables, references, roles, mode="mc", flight_length_mm=flight_mm,
                                       pulse_mode=pulse_mode, baseline_mass_da=frame["mc (Da)"].to_numpy())
    tick = time.perf_counter()
    report["training"] = session.fit()
    report["fit_seconds"] = time.perf_counter()-tick
    report["selection"] = session.validate_selection()
    report["replacement_assessment"] = dict(
        passed=report["selection"]["passed"],
        failed_reference_time_groups=sum(not group["passed"] for group in report["selection"]["groups"]),
        baseline_regressions=sum(not group.get("baseline_comparison", {}).get("passed", True)
                                 for group in report["selection"]["groups"]),
        interpretation="Absolute tolerances alone do not establish improvement; every declared validation group must pass.")
    if report["selection"]["passed"]:
        report["confirmation"] = session.confirm()
    else:
        report["confirmation"] = {"evaluated": False, "reason": "Selection failed; confirmation remains untouched"}
    report["model"] = session.model.provenance()
    report["state"] = session.status
    report["full_source_domain_check"] = dict(
        outside_training_domain=int(np.count_nonzero(~session.model.domain_mask(session.features))),
        nonpositive_fitted_elapsed_events=int(np.count_nonzero(raw_t <= session.model.t0_ns)))
    report["selection_comparison"] = []
    for target, reference in references.items():
        selected = reference & roles["selection"]
        features = session.features[selected]
        valid = session.model.domain_mask(features)
        # Diagnostic comparisons use exactly the same in-domain events in both methods.
        selected_indices = np.flatnonzero(selected)[valid]
        if len(selected_indices):
            report["selection_comparison"].append(dict(target_Da=target, outside_domain=int((~valid).sum()),
                saved_calibration=metrics(frame["mc (Da)"].to_numpy()[selected_indices], target),
                uncorrected_physical=metrics(physical[selected_indices], target),
                candidate=metrics(session.model.predict_mass_da(features[valid], raw_t[selected_indices]), target)))
    np.testing.assert_array_equal(variables.get_calibration_array("mc"), original_active)
    report["active_calibration_unchanged"] = True
    report["source_files_unchanged"] = before == {"data": fingerprint(data), "ranges": fingerprint(ranges_path)}
    assert report["source_files_unchanged"]
    report["total_seconds"] = time.perf_counter()-started
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--ranges", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-rows", type=int, nargs="+", required=True, help="Zero-based explicitly assumed historical rows")
    parser.add_argument("--flight-mm", type=float, required=True)
    parser.add_argument("--t0-ns", type=float, required=True)
    parser.add_argument("--pulse-mode", choices=("laser", "voltage"), required=True)
    args = parser.parse_args()
    result = compare(args.data, args.ranges, args.output, rows=args.reference_rows, flight_mm=args.flight_mm,
                     t0_ns=args.t0_ns, pulse_mode=args.pulse_mode)
    print(json.dumps({key: result[key] for key in ("events", "physical_vs_saved_uncorrected", "grouping_benchmark", "fit_seconds", "state", "selection_comparison", "total_seconds")}, indent=2))
