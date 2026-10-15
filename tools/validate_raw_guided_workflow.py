"""Validate raw /dld import, crop and resume through shipped notebook callbacks.

No instrument settings are inferred. Without all three explicit physical inputs,
the runner stops at a resumable raw-data session. The optional physical branch
checks public-converter parity, not calibration accuracy or reference identities.
This is an in-process notebook/callback check, not browser/kernel rendering.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import h5py
import nbformat
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.validate_guided_notebook_ui import InProcessJob, file_sha256

OUTPUTS = ("session.h5", "accepted.h5", "ui_transcript.txt", "summary.json")
ALIASES = {
    "high_voltage (V)": ("high_voltage",), "pulse_v (V)": ("pulse", "voltage_pulse", "pulse_voltage"),
    "pulse_l (pJ)": ("laser_pulse", "laser_intensity"), "start_counter": ("start_counter",),
    "t (ns)": ("t",), "x_det (cm)": ("x",), "y_det (cm)": ("y",),
}


def _outputs(data, output):
    for name in OUTPUTS:
        destination = (output / name).resolve()
        if destination == data or (destination.exists() and destination.samefile(data)):
            raise ValueError(f"Validation output {name} would overwrite an input; choose another directory")
    # A failed widget callback must never appear successful because a previous
    # run already left a valid export/session at the expected destination.
    for name in OUTPUTS:
        if (output / name).exists():
            raise ValueError(f"Validation output {name} already exists; choose a fresh output directory")


def verify_source(frame, data, *, chunk_size=250000):
    """Compare every imported measurement/counter to HDF source chunks independently."""
    checked, missing = {}, []
    with h5py.File(data, "r") as handle:
        group = handle["dld"]
        legacy = not any(name in group for name in ("pulse", "voltage_pulse")) and "pulse_voltage" in group
        for column, aliases in ALIASES.items():
            key = next((name for name in aliases if name in group), None)
            if key is None:
                missing.append(column)
                assert column in frame.attrs.get("pyccapt_unavailable_columns", []), column
                continue
            dataset = group[key]
            assert len(dataset) == len(frame), f"Event length differs: {key}"
            actual = frame[column].to_numpy()
            if column == "start_counter":
                assert actual.dtype == np.dtype("uint64"), "Counter lost its exact uint64 representation"
            for lo in range(0, len(frame), chunk_size):
                hi = min(len(frame), lo+chunk_size)
                expected = dataset[lo:hi].reshape(-1)
                if legacy and column == "t (ns)":
                    expected = expected * (27.432 / 4000)
                elif legacy and column in ("x_det (cm)", "y_det (cm)"):
                    expected = ((expected - 1225.) * (80 / 4900 * 2)) * .1
                assert np.array_equal(actual[lo:hi], expected, equal_nan=True), f"Imported values differ: {key}, rows {lo}:{hi}"
            checked[column] = {"source": f"/dld/{key}", "source_dtype": str(dataset.dtype),
                               "imported_dtype": str(actual.dtype), "all_rows_equal": True}
    return {"columns": checked, "unavailable_columns": missing, "legacy_bin_conversion": legacy,
            "comparison": "Every row, source read in bounded chunks; uint64 counters compared without float conversion"}


def _check_selection(workflow, n, start, stop, attrs, source_hash):
    assert workflow.n == n and len(workflow.source) == n, "Crop discarded full source context"
    assert workflow.source_hash == source_hash and workflow.source.attrs == attrs
    assert workflow.active_mask.dtype == bool and workflow.active_mask.shape == (n,)
    assert not workflow.active_mask[:start].any() and workflow.active_mask[start:stop].all()
    assert not workflow.active_mask[stop:].any()
    for lo in range(0, n, 250000):
        hi = min(n, lo+250000)
        assert np.array_equal(workflow.event_ids[lo:hi], np.arange(lo, hi, dtype=np.uint64))
    receipt = workflow.selection_metadata(preserves_full_source=True)
    assert receipt["source_events"] == n and receipt["selected_events"] == stop-start
    assert receipt["preserves_full_source"]
    return receipt


def _check_physical(workflow, flight_mm, t0_ns, pulse_mode):
    from pyccapt.calibration.mc.mc_tools import tof2mc
    mass = workflow.preview.get("calibration", workflow.accepted["calibration"])["mass"]
    frame, selected = workflow.source, workflow.active_mask
    unavailable = frame.attrs.get("pyccapt_unavailable_columns", [])
    finite_count = 0
    for lo in range(0, workflow.n, 250000):
        hi = min(workflow.n, lo+250000)
        arrays = {name: frame[name].to_numpy()[lo:hi] for name in
                  ("t (ns)", "high_voltage (V)", "pulse_v (V)", "x_det (cm)", "y_det (cm)")}
        valid = selected[lo:hi].copy()
        for name in ("t (ns)", "high_voltage (V)", "x_det (cm)", "y_det (cm)"):
            valid &= np.isfinite(arrays[name]) & (name not in unavailable)
        valid &= (arrays["t (ns)"] > t0_ns) & (arrays["high_voltage (V)"] > 0)
        if pulse_mode == "voltage":
            valid &= np.isfinite(arrays["pulse_v (V)"]) & ("pulse_v (V)" not in unavailable)
        expected = np.full(hi-lo, np.nan)
        expected[valid] = tof2mc(arrays["t (ns)"][valid], t0_ns, arrays["high_voltage (V)"][valid],
            arrays["x_det (cm)"][valid], arrays["y_det (cm)"][valid], flight_mm,
            arrays["pulse_v (V)"][valid], pulse_mode)
        assert np.array_equal(mass[lo:hi], expected, equal_nan=True), "Physical output differs from public converter"
        finite_count += int(np.isfinite(expected).sum())
    raw_tof = frame["t (ns)"].to_numpy()
    return {"public_converter_exact_parity": True, "finite_mass_events": finite_count,
            "selected_events_without_valid_mass": int(selected.sum())-finite_count,
            "excluded_events": int((~selected).sum()),
            "selected_recorded_tof_at_or_below_t0": int(np.count_nonzero(selected & np.isfinite(raw_tof) & (raw_tof <= t0_ns))),
            "selected_nonfinite_tof": int(np.count_nonzero(selected & ~np.isfinite(raw_tof))),
            "mass_quantiles_Da": dict(zip(("minimum", "p01", "median", "p99", "maximum"),
                np.nanpercentile(mass, [0, 1, 50, 99, 100]).tolist())),
            "excluded_and_invalid_events_are_nan": True, "accuracy_claim": False}


def _verify_export(path, workflow):
    selected_ids = workflow.event_ids[workflow.active_mask]
    selected_mass = workflow.accepted["calibration"]["mass"][workflow.active_mask]
    verified = set()
    with h5py.File(path, "r") as handle:
        group = handle["df"]
        attrs = json.loads(group.attrs["pyccapt_source_json"])
        assert len(group["axis1"]) == len(selected_ids)
        for item in group:
            if not item.endswith("_items"):
                continue
            names = [name.decode() if isinstance(name, bytes) else str(name) for name in group[item][:]]
            assert not {"x (nm)", "y (nm)", "z (nm)"}.intersection(names)
            for column, expected in (("event_id", selected_ids), ("mc (Da)", selected_mass)):
                if column not in names:
                    continue
                block = group[item.replace("_items", "_values")]
                for lo in range(0, len(expected), 250000):
                    hi = min(len(expected), lo+250000)
                    assert np.array_equal(block[lo:hi, names.index(column)], expected[lo:hi], equal_nan=True)
                verified.add(column)
        assert verified == {"event_id", "mc (Da)"}, "Export omitted accepted mass or stable event IDs"
        receipt = attrs["pyccapt_selection"]
        assert receipt["source_events"] == workflow.n
        assert receipt["preserves_full_source"] == bool(workflow.active_mask.all())
    return receipt


def validate(data, output_directory, *, event_start=0, event_stop=None,
             flight_mm=None, t0_ns=None, pulse_mode=None):
    """Replay real notebook cells and controls; never infer unknown physical inputs."""
    supplied = (flight_mm is not None, t0_ns is not None, pulse_mode is not None)
    physical = all(supplied)
    if any(supplied) and not physical:
        raise ValueError("Physical conversion requires all of flight_mm, t0_ns and pulse_mode explicitly")
    if physical and (not np.isfinite([flight_mm, t0_ns]).all() or flight_mm <= 0 or pulse_mode not in ("laser", "voltage")):
        raise ValueError("Physical settings require positive finite flight_mm, finite t0_ns and laser/voltage pulse_mode")
    data, output = Path(data).expanduser().resolve(), Path(output_directory).expanduser().resolve()
    _outputs(data, output)
    with h5py.File(data, "r") as handle:
        n = len(handle["dld/t"])
    stop = n if event_stop is None else event_stop
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in (event_start, stop)) or not 0 <= event_start < stop <= n:
        raise ValueError("Crop requires integer positions 0 <= event_start < event_stop <= source events")
    output.mkdir(parents=True, exist_ok=True)
    started, source_digest = perf_counter(), file_sha256(data)
    report = {"status": "started", "source": str(data), "source_sha256": source_digest, "source_events": n,
        "validation_mode": "in-process shipped notebook cells and real widget callbacks; no browser or live kernel",
        "crop": {"event_start": int(event_start), "event_stop": int(stop), "purpose": "Explicit validation selection; not inferred scientific cleanup"},
        "instrument_settings": {"flight_mm": flight_mm, "t0_ns": t0_ns, "pulse_mode": pulse_mode},
        "physical_conversion_requested": physical, "phases_seconds": {},
        "limitations": ["Source coverage before this file is unknown; full source means all loaded DLD events.",
            "No peak identities, ranging or ICF/kf reconstruction are inferred or accepted.",
            "Physical conversion, if explicitly requested, checks numerical compatibility, not calibration accuracy."]}
    os.environ.setdefault("MPLBACKEND", "Agg")
    from pyccapt.calibration.tutorials.tutorials_helpers import helper_guided_workflow as helper
    from pyccapt.calibration.core.guided_workflow import GuidedWorkflow
    try:
        with ExitStack() as stack:
            stream = stack.enter_context((output / "ui_transcript.txt").open("w", encoding="utf8"))
            stack.enter_context(redirect_stdout(stream))
            stack.enter_context(redirect_stderr(stream))
            stack.enter_context(patch.object(helper, "NotebookJob", InProcessJob))
            stack.enter_context(patch.object(helper, "display", lambda *_: None))
            notebook = nbformat.read(ROOT / "pyccapt/calibration/tutorials/jupyter_files/data_processing_guided.ipynb", as_version=4)
            nbformat.validate(notebook)
            namespace = {}
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    exec(compile(cell.source, "data_processing_guided.ipynb", "exec"), namespace)
            panel = namespace["guided_panel"]
            panel.controls["path"].value = str(data)
            panel.controls["mode"].value = "raw"
            phase = perf_counter()
            panel.controls["Import dataset"].click()
            workflow = panel.controller["workflow"]
            assert workflow is not None, "Import callback failed; inspect ui_transcript.txt"
            panel.controls["Inspect source"].click()
            assert workflow.inspected
            assert not {"mc (Da)", "mc_uc (Da)", "x (nm)", "y (nm)", "z (nm)"}.intersection(workflow.source.columns)
            assert panel.controls["Preview imported mass"].disabled
            report["imported_mass_disabled"] = True
            report["source_columns"] = list(workflow.source.columns)
            report["phases_seconds"]["import_inspect"] = perf_counter()-phase
            phase = perf_counter()
            report["source_measurements"] = verify_source(workflow.source, data)
            source_hash, attrs = workflow.source_hash, deepcopy(workflow.source.attrs)
            report["source_metadata"] = attrs
            report["phases_seconds"]["verify_all_source_rows"] = perf_counter()-phase
            panel.controls["Preview physical mass"].click()
            assert "calibration" not in workflow.preview and workflow.accepted["calibration"] is None
            report["unconfirmed_physical_conversion_refused"] = True
            phase = perf_counter()
            panel.controls["crop_start"].value, panel.controls["crop_stop"].value = int(event_start), int(stop)
            panel.controls["Preview crop"].click()
            assert workflow.active_mask.all() and "crop" in workflow.preview
            panel.controls["Accept crop"].click()
            report["selection"] = _check_selection(workflow, n, event_start, stop, attrs, source_hash)
            report["phases_seconds"]["crop_preview_accept"] = perf_counter()-phase
            if physical:
                phase = perf_counter()
                panel.controls["flight"].value, panel.controls["t0"].value = flight_mm, t0_ns
                panel.controls["pulse_mode"].value = pulse_mode
                panel.controls["instrument_confirmed"].value = True
                panel.controls["Preview physical mass"].click()
                assert "calibration" in workflow.preview, "Physical conversion failed; inspect ui_transcript.txt"
                assert workflow.accepted["calibration"] is None
                report["physical_conversion"] = _check_physical(workflow, flight_mm, t0_ns, pulse_mode)
                panel.controls["Accept calibration"].click()
                assert workflow.accepted["calibration"] is not None
                panel.controls["export_path"].value = str(output / "accepted.h5")
                panel.controls["Export accepted HDF5"].click()
                assert (output / "accepted.h5").is_file(), "Export failed; inspect ui_transcript.txt"
                report["export_selection"] = _verify_export(output / "accepted.h5", workflow)
                embedded = GuidedWorkflow.load(output / "accepted.h5")
                _check_selection(embedded, n, event_start, stop, attrs, source_hash)
                _check_physical(embedded, flight_mm, t0_ns, pulse_mode)
                del embedded
                report["embedded_session_verified"] = True
                report["phases_seconds"]["physical_preview_accept_export_resume"] = perf_counter()-phase
            phase = perf_counter()
            panel.controls["session_path"].value = str(output / "session.h5")
            panel.controls["Save resumable session"].click()
            assert (output / "session.h5").is_file(), "Save failed; inspect ui_transcript.txt"
            report["phases_seconds"]["save_session"] = perf_counter()-phase
            phase = perf_counter()
            panel.controls["Resume saved session"].click()
            restored = panel.controller["workflow"]
            assert restored is not workflow
            _check_selection(restored, n, event_start, stop, attrs, source_hash)
            verify_source(restored.source, data)
            if physical:
                _check_physical(restored, flight_mm, t0_ns, pulse_mode)
            else:
                assert restored.accepted["calibration"] is None and not restored.variables.has_mass_spectrum
            assert restored.accepted["ranges"] is None and restored.accepted["reconstruction"] is None
            report["phases_seconds"]["resume_verify"] = perf_counter()-phase
            report.update(session_resume_verified=True, full_loaded_source_retained=True, stable_event_ids_verified=True,
                          selected_events=int(restored.active_mask.sum()), final_status=restored.status())
        report["source_file_unchanged"] = file_sha256(data) == source_digest
        assert report["source_file_unchanged"]
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["total_seconds"] = perf_counter()-started
        (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--event-start", type=int, default=0)
    parser.add_argument("--event-stop", type=int)
    parser.add_argument("--flight-mm", type=float)
    parser.add_argument("--t0-ns", type=float)
    parser.add_argument("--pulse-mode", choices=("laser", "voltage"))
    report = validate(**vars(parser.parse_args()))
    print(json.dumps({key: report[key] for key in ("status", "source_events", "selected_events",
        "physical_conversion_requested", "source_file_unchanged", "phases_seconds")}, indent=2))


if __name__ == "__main__":
    main()
