"""Replay reviewed source ranges through the shipped guided notebook UI.

This is an in-process callback integration check, not a browser or Jupyter-kernel
test. It does not infer reference identities, fit a new calibration, or invent
instrument/reconstruction settings. Pass the previously reviewed range file
explicitly; unmatched proposals remain unaccepted. Source files are read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
from threading import Event
import time

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class InProcessJob:
    """Run the real operation and its delivery callbacks synchronously for validation."""
    running = False

    def start(self, operation, *, on_result, on_error, on_progress=lambda *_: None, on_finished=lambda: None):
        self.running = True
        try:
            on_result(operation(Event(), on_progress))
        except Exception as error:
            on_error(error)
            raise
        finally:
            self.running = False
            on_finished()

    def cancel(self):
        raise RuntimeError("Cancellation is covered by separate tests; this replay runs synchronously")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(row):
    composition = Counter()
    for element, isotope, count in zip(row["element"], row["isotope"], row["complex"]):
        composition[str(element), int(isotope)] += int(count)
    return tuple(sorted(composition.items())), int(row["charge"])


def validate(data, ranges, output_directory, *, bin_width=.02, prominence=20., tolerance=.15, max_atoms=1):
    from pyccapt.calibration.core.guided_workflow import GuidedWorkflow
    from pyccapt.calibration.tutorials.tutorials_helpers import helper_guided_workflow, helper_peak_review

    data, ranges = Path(data).resolve(), Path(ranges).resolve()
    output = Path(output_directory).resolve()
    inputs = {data, ranges}
    for filename in ("session.h5", "accepted.h5", "ui_transcript.txt", "summary.json"):
        if (output / filename).resolve() in inputs:
            raise ValueError(f"Validation output {filename} would overwrite an input; select a separate output directory")
    output.mkdir(parents=True, exist_ok=True)
    initial_hashes = {"data": file_sha256(data), "ranges": file_sha256(ranges)}
    historical = pd.read_hdf(ranges, key="df").reset_index(drop=True)
    report = {"validation_mode": "in-process shipped notebook cells and real widget callbacks; no live kernel/browser",
        "source": str(data), "range_source": str(ranges), "source_sha256": initial_hashes,
        "instrument_settings_known": False, "new_physical_calibration_or_q_claim": False,
        "identity_policy": "Replay only explicitly supplied historical identities; unmatched proposals remain unaccepted",
        "settings": dict(bin_width_Da=bin_width, prominence_counts=prominence, tolerance_Da=tolerance, max_atoms=max_atoms),
        "phases_seconds": {}}
    original_jobs = helper_guided_workflow.NotebookJob, helper_peak_review.NotebookJob
    helper_guided_workflow.NotebookJob = helper_peak_review.NotebookJob = InProcessJob
    transcript = io.StringIO()
    started = time.perf_counter()
    try:
        with redirect_stdout(transcript):
            notebook = nbformat.read(ROOT / "pyccapt/calibration/tutorials/jupyter_files/data_processing_guided.ipynb", as_version=4)
            nbformat.validate(notebook)
            namespace = {}
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    exec(compile(cell.source, "data_processing_guided.ipynb", "exec"), namespace)
            panel = namespace["guided_panel"]
            panel.controls["path"].value = str(data)
            phase = time.perf_counter()
            panel.controls["Import dataset"].click()
            workflow = panel.controller["workflow"]
            if workflow is None:
                raise AssertionError("Import callback did not create a workflow")
            panel.controls["Inspect source"].click()
            panel.controls["Preview imported mass"].click()
            assert workflow.accepted["calibration"] is None
            panel.controls["Accept calibration"].click()
            accepted_mass = workflow.accepted["calibration"]["mass"]
            column = workflow.accepted["calibration"]["receipt"]["column"]
            np.testing.assert_array_equal(accepted_mass, workflow.source[column].to_numpy())
            report.update(events=workflow.n, source_columns=list(workflow.source.columns),
                          accepted_mass_column=column, imported_mass_bitwise_unchanged=True,
                          imported_tof_preserved=workflow.accepted["calibration"]["tof"] is not None,
                          mass_quantiles_Da=np.nanpercentile(accepted_mass, [0, 1, 50, 99, 100]).tolist(),
                          source_hash=workflow.source_hash)
            report["phases_seconds"]["import_inspect_accept"] = time.perf_counter()-phase
            phase = time.perf_counter()
            panel.controls["Open peak-review queue"].click()
            review = panel.controller["range_panel"]
            review.controls["elements"].value = ",".join(sorted({str(element) for values in historical.element for element in values}))
            review.controls["charge"].value = int(historical.charge.max())
            review.controls["atoms"].value = max_atoms
            review.controls["bin_width"].value = bin_width
            review.controls["prominence"].value = prominence
            review.controls["tolerance"].value = tolerance
            review.controls["Propose peaks"].click()
            queue = workflow.review
            assert queue.ranges().empty
            report["proposal_count"] = len(queue.records)
            report["automatic_acceptances"] = 0
            report["phases_seconds"]["proposal_generation"] = time.perf_counter()-phase
            phase = time.perf_counter()
            matches, unmatched, used = [], [], set()
            for index, historical_row in historical.iterrows():
                expected_identity = identity(historical_row)
                available = [row for row in queue.records if row["id"] not in used and
                    historical_row.mc_low <= row["center"] < historical_row.mc_up]
                options = [(row, candidate) for row in available for candidate in row["candidates"]
                           if identity(candidate) == expected_identity]
                if not options:
                    unmatched.append({"historical_row": int(index), "ion": str(historical_row.ion),
                        "reason": "No proposed center inside the historical window with its identity in the bounded candidate list"})
                    continue
                row, candidate = min(options, key=lambda item: abs(item[0]["center"]-float(historical_row.mc)))
                proposed_low, proposed_high = row["lower"], row["upper"]
                historical_mask = (accepted_mass >= historical_row.mc_low) & (accepted_mass < historical_row.mc_up)
                proposed_mask = (accepted_mass >= proposed_low) & (accepted_mass < proposed_high)
                union = np.count_nonzero(historical_mask | proposed_mask)
                matches.append({"historical_row": int(index), "review_id": row["id"], "ion": str(historical_row.ion),
                    "proposal_center_Da": row["center"], "identity_mass_Da": candidate["mass"],
                    "mass_residual_Da": row["center"]-candidate["mass"],
                    "historical_events": int(historical_mask.sum()), "proposed_events": int(proposed_mask.sum()),
                    "window_jaccard": float(np.count_nonzero(historical_mask & proposed_mask)/union) if union else None,
                    "historical_bounds_Da": [float(historical_row.mc_low), float(historical_row.mc_up)],
                    "proposed_bounds_Da": [proposed_low, proposed_high]})
                review.controls["peaks"].value = row["id"]
                review.controls["candidate"].value = candidate["key"]
                review.controls["Choose identity"].click()
                review.controls["lower"].value = float(historical_row.mc_low)
                review.controls["upper"].value = float(historical_row.mc_up)
                review.controls["Edit bounds"].click()
                review.controls["reviewed"].value = True
                review.controls["Accept identity"].click()
                assert queue._get(row["id"])["status"] == "accepted"
                used.add(row["id"])
            if not matches:
                raise AssertionError("No historical identity matched a proposed peak; review settings, do not fabricate acceptance")
            review.controls["Preview ranges"].click()
            assert workflow.accepted["ranges"] is None
            review.controls["Accept ranges"].click()
            ranged = workflow.accepted["ranges"]
            # Independent reference implementation: one mask at a time, with
            # historical bounds/identity and the actual accepted table order.
            # This preserves overlap ownership without testing the production
            # assignment function against itself.
            accepted_to_reference = {item["review_id"]: item["historical_row"] for item in matches}
            expected = np.full(workflow.n, -1, dtype=np.int32)
            for index, row in enumerate(ranged["rows"]):
                reference = historical.iloc[accepted_to_reference[row["review_id"]]]
                assert identity(row) == identity(reference)
                assert row["mc_low"] == float(reference.mc_low) and row["mc_up"] == float(reference.mc_up)
                mask = (accepted_mass >= reference.mc_low) & (accepted_mass < reference.mc_up) & (expected < 0)
                expected[mask] = index
            actual = ranged["assignment"]
            np.testing.assert_array_equal(actual, expected)
            historical_all = np.full(workflow.n, -1, dtype=np.int32)
            for index, reference in historical.iterrows():
                mask = (accepted_mass >= reference.mc_low) & (accepted_mass < reference.mc_up) & (historical_all < 0)
                historical_all[mask] = index
            report.update(historical_range_count=len(historical), replayed_range_count=len(matches),
                unmatched_historical_ranges=unmatched, matched_ranges=matches,
                replayed_ranged_events=int(np.count_nonzero(actual >= 0)),
                historical_ranged_events=int(np.count_nonzero(historical_all >= 0)),
                replayed_interval_assignments_match_reference=True)
            panel.controls["Analyze accepted ranges"].click()
            report["composition_of_replayed_assigned_atoms"] = workflow.accepted["analysis"]["atomic_fractions_of_assigned_atoms"]
            report["phases_seconds"]["historical_review_replay"] = time.perf_counter()-phase
            phase = time.perf_counter()
            session_path, exported_path = output / "session.h5", output / "accepted.h5"
            panel.controls["session_path"].value = str(session_path)
            panel.controls["Save resumable session"].click()
            panel.controls["export_path"].value = str(exported_path)
            panel.controls["Export accepted HDF5"].click()
            exported = pd.read_hdf(exported_path, "df")
            np.testing.assert_array_equal(exported["mc (Da)"], accepted_mass)
            np.testing.assert_array_equal(exported["event_id"], workflow.event_ids)
            # This run has no verified reconstruction settings. Imported XYZ must
            # not masquerade as newly accepted geometry in derived export.
            assert not {"x (nm)", "y (nm)", "z (nm)"}.intersection(exported.columns)
            panel.controls["Resume saved session"].click()
            restored = panel.controller["workflow"]
            assert restored is not workflow and restored.source_hash == workflow.source_hash
            np.testing.assert_array_equal(restored.accepted["ranges"]["assignment"], actual)
            np.testing.assert_array_equal(restored.variables.mc, accepted_mass)
            embedded = GuidedWorkflow.load(exported_path)
            np.testing.assert_array_equal(embedded.event_ids, workflow.event_ids)
            report.update(session_resume_verified=True, embedded_session_verified=True, accepted_export_verified=True,
                          reconstruction_abstained_without_instrument_settings=True, final_status=restored.status())
            report["phases_seconds"]["save_export_resume"] = time.perf_counter()-phase
    finally:
        helper_guided_workflow.NotebookJob, helper_peak_review.NotebookJob = original_jobs
        (output / "ui_transcript.txt").write_text(transcript.getvalue(), encoding="utf-8")
    report["source_files_unchanged"] = initial_hashes == {"data": file_sha256(data), "ranges": file_sha256(ranges)}
    assert report["source_files_unchanged"]
    report["total_seconds"] = time.perf_counter()-started
    report["limitations"] = ["Supplementary dataset only; not a substitute for the specifically requested original acquisition.",
        "Imported mass retained exactly; no new calibration accuracy or q/ICF/kf accuracy established.",
        "Historical identities/bounds are validation inputs, not independent ground truth or new user approvals.",
        "Unmatched weak, molecular, ambiguous, or out-of-window ranges remain unaccepted.",
        "Notebook cells/callbacks executed in-process; live Jupyter kernel and browser rendering are separate checks."]
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--ranges", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--bin-width", type=float, default=.02)
    parser.add_argument("--prominence", type=float, default=20.)
    parser.add_argument("--tolerance", type=float, default=.15)
    parser.add_argument("--max-atoms", type=int, default=1)
    arguments = parser.parse_args()
    result = validate(arguments.data, arguments.ranges, arguments.output_directory,
                      bin_width=arguments.bin_width, prominence=arguments.prominence,
                      tolerance=arguments.tolerance, max_atoms=arguments.max_atoms)
    print(json.dumps({key: result[key] for key in ("events", "proposal_count", "replayed_range_count", "replayed_ranged_events",
        "historical_ranged_events", "source_files_unchanged", "total_seconds")}, indent=2))


if __name__ == "__main__":
    main()
