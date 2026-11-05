"""Human decision/state and exact assignment contracts for the separate guided notebook."""
from threading import Event

import numpy as np
import pandas as pd
import pytest

from pyccapt.calibration.core.exceptions import CalibrationInputError, CalibrationStateError
from pyccapt.calibration.core.guided_workflow import GuidedWorkflow
from pyccapt.calibration.core.isotope_catalogue import get_isotope_catalogue
from pyccapt.calibration.core.ranging_review import PeakReviewQueue
from pyccapt.calibration.core.share_variables import Variables


def isotope_table():
    return pd.DataFrame({"element": ["Fe", "Fe", "Ni"], "isotope": [54, 56, 58],
                         "weight": [54., 56., 58.], "abundance": [10., 90., 100.]})


def catalogue():
    return get_isotope_catalogue("Fe,Ni", (1, 2), 1, table=isotope_table())


def workflow():
    rng = np.random.default_rng(3)
    mass = np.r_[rng.normal(27., .025, 1000), rng.normal(28., .025, 2000), rng.normal(29., .025, 1000)]
    frame = pd.DataFrame({"mc (Da)": mass, "mc_uc (Da)": mass, "t (ns)": np.sqrt(mass)*20,
        "high_voltage (V)": np.full(len(mass), 5000.), "x_det (cm)": rng.normal(0, .1, len(mass)),
        "y_det (cm)": rng.normal(0, .1, len(mass)), "event_id": np.arange(len(mass), dtype=np.uint64)+2**60})
    variables = Variables()
    variables.sync_from_data(frame)
    return GuidedWorkflow(variables)


def accepted_mass():
    session = workflow()
    session.preview_imported_mass()
    session.accept("calibration")
    return session


def accept_one_range(session):
    queue = session.begin_review()
    queue.propose(catalogue(), prominence=20, bin_width=.01, tolerance=.08)
    row = queue.records[0]
    queue.choose(row["id"], row["candidates"][0]["key"])
    queue.decide(row["id"], "accept", ambiguity_reviewed=True)
    session.preview_ranges()
    session.accept("ranges")
    return row["id"]


def test_catalogue_uses_canonical_multisets_and_exact_window_search():
    result = get_isotope_catalogue("Fe", (1,), 2, table=isotope_table())
    assert len(result.candidates) == 5  # two atoms + three unordered two-atom combinations
    matches = result.search(110., .01)
    assert len(matches) == 1
    assert matches[0].isotope == (54, 56)
    assert matches[0].mass == 110.
    assert not result.search(111., .01)


def test_catalogue_cache_includes_content_and_rejects_excessive_work():
    first = catalogue()
    assert first is catalogue()
    changed = isotope_table()
    changed.loc[0, "weight"] += .001
    assert first is not get_isotope_catalogue("Ni,Fe", (2, 1), 1, table=changed)
    with pytest.raises(CalibrationInputError, match="limit"):
        get_isotope_catalogue("Fe,Ni", (1, 2), 5, table=isotope_table(), max_candidates=10)


def test_builtin_catalogue_preserves_explicit_deuterium_and_hydrogen_isotope_identities():
    result = get_isotope_catalogue("H,D", (1,), 1)
    deuterium = next(item for item in result.candidates if item.element == ("D",))
    alternatives = result.search(deuterium.mass, tolerance=.001)
    identities = {(item.element, item.isotope, item.complex, item.charge) for item in alternatives}
    assert identities == {(("D",), (2,), (1,), 1), (("H",), (2,), (1,), 1)}
    assert len({item.key for item in alternatives}) == 2


def test_proposals_do_not_accept_and_locked_decisions_survive_redetection():
    session = accepted_mass()
    queue = session.begin_review()
    queue.propose(catalogue(), bin_width=.01, prominence=20)
    assert len(queue.records) == 3
    assert queue.ranges().empty
    row = queue.records[1]
    queue.choose(row["id"], row["candidates"][0]["key"])
    queue.decide(row["id"], "accept", ambiguity_reviewed=True)
    queue.propose(catalogue(), bin_width=.02, prominence=1e9)
    assert len(queue.records) == 1
    assert queue.records[0]["id"] == row["id"]
    assert queue.records[0]["status"] == "accepted"


def test_review_reject_not_peak_defer_and_undo_are_distinct():
    queue = PeakReviewQueue(np.linspace(26., 30., 100))
    queue.catalogue = catalogue()
    queue.records = [queue._new(28., 27.8, 28.2, queue.catalogue)]
    row = queue.records[0]
    queue.choose(row["id"], row["candidates"][0]["key"])
    key = row["chosen"]["key"]
    queue.decide(row["id"], "reject_identity")
    assert row["status"] == "proposed" and key in row["rejected"]
    queue.decide(row["id"], "defer")
    assert row["status"] == "deferred" and not row["locked"]
    queue.decide(row["id"], "not_peak")
    assert row["locked"] and row["status"] == "not_peak"
    queue.undo()
    assert queue.records[0]["status"] == "deferred"


def test_split_merge_bounds_and_overlap_require_renewed_review():
    queue = PeakReviewQueue(np.array([27.9, 28., 28.1, 28.2, np.nan]))
    queue.catalogue = catalogue()
    queue.records = [queue._new(28., 27.8, 28.3, queue.catalogue)]
    children = queue.split(queue.records[0]["id"], 28.05)
    assert len(children) == 2 and all(row["chosen"] is None for row in queue.records)
    merged = queue.merge(children)
    assert len(queue.records) == 1 and queue._get(merged)["upper"] == 28.3
    queue.choose(merged, queue._get(merged)["candidates"][0]["key"])
    queue.decide(merged, "accept", ambiguity_reviewed=True)
    assigned = queue.assignments()
    np.testing.assert_array_equal(assigned, [0, 0, 0, 0, -1])
    assert assigned.dtype == np.int32
    queue.edit_bounds(merged, 28., 28.2)
    assert queue.ranges().empty


def test_cancelled_proposals_leave_previous_decisions_unchanged():
    session = accepted_mass()
    accept_one_range(session)
    queue = session.review
    before = queue.to_state()
    cancel = Event()
    cancel.set()
    with pytest.raises(CalibrationStateError, match="cancelled"):
        queue.propose(catalogue(), bin_width=.03, cancel=cancel)
    assert queue.to_state() == before


def test_dependency_invalidation_and_undo_restore_exact_accepted_state():
    session = accepted_mass()
    accept_one_range(session)
    xyz = np.column_stack([np.arange(session.n), np.zeros(session.n), np.arange(session.n)*.1])
    session.preview_reconstruction(xyz, {"icf": 1.4, "kf": 3.3}, ratio_result={"q": .4242})
    session.accept("reconstruction")
    session.analyze()
    original_mass = session.accepted["calibration"]["mass"]
    session.preview_calibration(original_mass*1.001, receipt={"method": "synthetic correction"})
    np.testing.assert_array_equal(session.variables.mc, original_mass)
    session.accept("calibration")
    assert all(session.accepted[key] is None for key in ("ranges", "reconstruction", "analysis"))
    assert session.review is None and not session.variables.has_reconstruction
    assert "x (nm)" not in session.variables.data
    session.undo()
    np.testing.assert_array_equal(session.variables.mc, original_mass)
    np.testing.assert_array_equal(session.variables.x, xyz[:, 0])
    assert session.variables.reconstruction_ratio_result["q"] == .4242
    assert len(session.review.ranges()) == 1


def test_changes_after_range_preview_and_stale_async_results_cannot_commit():
    session = accepted_mass()
    key = accept_one_range(session)
    session.preview_ranges()
    session.review.edit_bounds(key, 26.8, 27.1)
    with pytest.raises(CalibrationStateError, match="review changed"):
        session.accept("ranges")
    with pytest.raises(CalibrationStateError, match="changed while computing"):
        session.preview_calibration(session.variables.mc, expected_revision=-1)


def test_session_roundtrip_preserves_ids_dtype_models_queue_and_undo(tmp_path):
    session = accepted_mass()
    accept_one_range(session)
    path = tmp_path / "guided.h5"
    session.save(path)
    restored = GuidedWorkflow.load(path)
    assert restored.source_hash == session.source_hash
    assert restored.event_ids.dtype == np.uint64 and restored.event_ids[0] == 2**60
    np.testing.assert_array_equal(restored.variables.mc, session.variables.mc)
    np.testing.assert_array_equal(restored.accepted["ranges"]["assignment"], session.accepted["ranges"]["assignment"])
    assert restored.review.to_state() == session.review.to_state()
    restored.undo()
    assert restored.accepted["ranges"] is None
    assert restored.accepted["calibration"] is not None


def test_resume_rejects_source_reordering():
    session = accepted_mass()
    metadata, arrays = session.to_state()
    mass_column = next(column for column in metadata["source_columns"] if column["name"] == "event_id")
    key = mass_column["data"]["array_ref"]
    arrays[key] = arrays[key][::-1].copy()
    with pytest.raises(CalibrationStateError, match="identity/order/hash"):
        GuidedWorkflow.from_state(metadata, arrays)


def test_resync_with_same_event_ids_cannot_bypass_source_guard():
    session = accepted_mass()
    changed = session.source.copy()
    changed["t (ns)"] += 10
    session.variables.sync_from_data(changed)
    with pytest.raises(CalibrationStateError, match="Source event order"):
        session.inspect()
    with pytest.raises(CalibrationStateError):
        session.analyze()


def test_preview_workspace_inherits_only_accepted_tof_and_receipt():
    session = workflow()
    mass = session.source["mc (Da)"].to_numpy()
    tof = session.source["t (ns)"].to_numpy()-2
    session.preview_calibration(mass, tof=tof, receipt={"method": "known synthetic offset"})
    session.accept("calibration")
    working = session.working_variables()
    np.testing.assert_array_equal(working.dld_t_calib, tof)
    assert working.guided_calibration_receipt["method"] == "known synthetic offset"
    working.mc_calib[:] = 999
    np.testing.assert_array_equal(session.variables.mc, mass)


def test_background_cancel_discards_a_completed_result():
    from pyccapt.calibration.tutorials.tutorials_helpers.guided_jobs import NotebookJob
    deliveries, accepted, errors = [], [], []
    job = NotebookJob(dispatch=deliveries.append)
    job.start(lambda cancel, progress: 42, on_result=accepted.append, on_error=errors.append)
    job.thread.join(timeout=5)
    job.cancel()
    for callback in deliveries:
        callback()
    assert not accepted and isinstance(errors[0], CalibrationStateError) and not job.running


def test_local_histogram_matches_full_histogram_and_reads_only_window(monkeypatch):
    mass = np.r_[np.arange(10000., 1010000.), 1., 1.5, 2., np.nan, np.inf]
    queue = PeakReviewQueue(mass)
    original = np.histogram
    visited = []
    def counted(values, **kwargs):
        visited.append(len(values))
        return original(values, **kwargs)
    monkeypatch.setattr(np, "histogram", counted)
    counts, edges = queue.histogram(.5, 1., 2.)
    np.testing.assert_array_equal(counts, original(mass[np.isfinite(mass)], bins=edges)[0])
    assert counts.tolist() == [1, 2]  # final-edge equality belongs to the last bin
    assert sum(visited) == 3
    queue.histogram(.5, 1., 2.)
    assert sum(visited) == 3  # repeat view uses the bounded cache


def test_family_needs_explicit_consistency_and_identity_change_clears_approval():
    queue = accepted_mass().begin_review()
    queue.propose(catalogue(), bin_width=.01, prominence=20, tolerance=.08)
    ids = [row["id"] for row in queue.records[:2]]
    for row in queue.records[:2]:
        queue.choose(row["id"], row["candidates"][0]["key"])
    with pytest.raises(CalibrationStateError, match="explicitly confirm"):
        queue.accept_family(ids)
    queue.accept_family(ids, consistency_reviewed=True, ambiguity_reviewed=True)
    assert all(queue._get(key)["family_reviewed"] for key in ids)
    row = queue._get(ids[0])
    queue.choose(row["id"], row["chosen"]["key"])
    assert not row["family_reviewed"] and not row["locked"]
    assert len(queue.ranges()) == 1


def test_proposal_undo_restores_catalogue_and_tolerance_for_split():
    queue = accepted_mass().begin_review()
    fe = get_isotope_catalogue("Fe", (2,), 1, table=isotope_table())
    ni = get_isotope_catalogue("Ni", (2,), 1, table=isotope_table())
    queue.propose(fe, bin_width=.01, prominence=20, tolerance=.08)
    first = queue.to_state()
    queue.propose(ni, bin_width=.01, prominence=20, tolerance=.2)
    queue.undo()
    assert queue.catalogue is fe and queue.tolerance == .08
    assert queue.to_state()["catalogue_spec"] == first["catalogue_spec"]
    row = queue.records[0]
    children = queue.split(row["id"], row["center"])
    assert all(candidate["element"] == ["Fe"] for key in children for candidate in queue._get(key)["candidates"])


def test_failed_resume_does_not_mutate_existing_workspace():
    source = accepted_mass()
    accept_one_range(source)
    metadata, arrays = source.to_state()
    assignment = metadata["accepted"]["ranges"]["assignment"]["array_ref"]
    arrays[assignment] = np.full(source.n, 99, dtype=np.int32)
    destination = accepted_mass().variables
    original_dataset, original_mass = destination.dataset, destination.mc.copy()
    with pytest.raises(CalibrationStateError, match="assignments"):
        GuidedWorkflow.from_state(metadata, arrays, variables=destination)
    assert destination.dataset is original_dataset
    np.testing.assert_array_equal(destination.mc, original_mass)


def test_source_mutation_refuses_inconsistent_snapshot():
    session = accepted_mass()
    session.source.loc[0, "t (ns)"] += 1
    with pytest.raises(CalibrationStateError, match="source was modified"):
        session.to_state()


def test_resume_shares_identical_immutable_arrays_across_undo_history():
    session = accepted_mass()
    accept_one_range(session)
    session.analyze()
    restored = GuidedWorkflow.from_state(*session.to_state())
    mass = restored.accepted["calibration"]["mass"]
    snapshot = next(item for item in restored.history if item["accepted"]["calibration"] is not None)
    assert snapshot["accepted"]["calibration"]["mass"] is mass
    assert not mass.flags.writeable


def test_learned_model_and_masks_roundtrip(tmp_path):
    from sklearn.linear_model import LinearRegression
    from pyccapt.calibration.data_tools.model_state import export_model_state, restore_model_state
    session = workflow()
    model = LinearRegression().fit(np.arange(20).reshape(-1, 1), np.arange(20)*3.+7)
    roles = np.arange(session.n) % 2 == 0
    state = {"learned_model": export_model_state(model), "roles": {"train": roles, "selection": ~roles},
             "reference_labels": session.source["mc (Da)"].to_numpy(), "apply_mask": np.ones(session.n, bool)}
    session.preview_calibration(session.source["mc (Da)"], model_state=state)
    session.accept("calibration")
    path = tmp_path / "model.h5"
    session.save(path)
    restored = GuidedWorkflow.load(path)
    saved = restored.accepted["calibration"]["model_state"]
    np.testing.assert_array_equal(saved["roles"]["train"], roles)
    np.testing.assert_allclose(restore_model_state(saved["learned_model"]).predict([[2.], [5.]]), [13., 22.])


def test_unavailable_mass_is_not_an_importable_measurement():
    session = workflow()
    session.source.attrs["pyccapt_unavailable_columns"] = ["mc (Da)", "mc_uc (Da)"]
    with pytest.raises(CalibrationStateError, match="no mass column"):
        session.preview_imported_mass()
    session.preview_calibration(np.ones(session.n)*27.)
    session.accept("calibration")
    assert "mc (Da)" not in session.variables.data.attrs["pyccapt_unavailable_columns"]


def test_catalogue_limits_and_cached_cancellation_are_explicit():
    with pytest.raises(CalibrationInputError, match="atom limit"):
        get_isotope_catalogue("Fe", max_atoms=10**9, table=isotope_table())
    catalogue()
    cancelled = Event()
    cancelled.set()
    with pytest.raises(CalibrationStateError, match="cancelled"):
        get_isotope_catalogue("Fe,Ni", (1, 2), 1, table=isotope_table(), cancel=cancelled)


@pytest.mark.parametrize("unusable", [0., np.nan, np.inf])
def test_imported_mass_falls_back_to_usable_uncalibrated_column_without_pairing_tof(unusable):
    session = workflow()
    frame = session.source.copy()
    frame["mc (Da)"] = unusable
    frame["t_c (ns)"] = frame["t (ns)"]-2
    session.variables.sync_from_data(frame)
    session = GuidedWorkflow(session.variables)
    preview = session.preview_imported_mass()
    assert preview["imported_mass_column"] == "mc_uc (Da)"
    assert session.preview["calibration"]["tof"] is None
    session.accept("calibration")
    np.testing.assert_array_equal(session.variables.mc, frame["mc_uc (Da)"])


def test_imported_calibrated_mass_preserves_corresponding_available_tof(tmp_path):
    session = workflow()
    frame = session.source.copy()
    frame["t_c (ns)"] = frame["t (ns)"]-2.
    frame.loc[0, "t_c (ns)"] = 0.  # a measured zero is retained in an otherwise usable ToF vector
    session.variables.sync_from_data(frame)
    session = GuidedWorkflow(session.variables)
    preview = session.preview_imported_mass()
    assert preview["imported_tof_column"] == "t_c (ns)"
    session.accept("calibration")
    np.testing.assert_array_equal(session.variables.dld_t_c, frame["t_c (ns)"])
    path = tmp_path / "retained_tof.h5"
    session.save(path)
    restored = GuidedWorkflow.load(path)
    np.testing.assert_array_equal(restored.variables.dld_t_calib, frame["t_c (ns)"])


def test_calibrated_tof_preview_cannot_accept_state_that_fails_resume():
    session = workflow()
    with pytest.raises(CalibrationInputError, match="ToF"):
        session.preview_calibration(session.source["mc (Da)"], tof=-np.ones(session.n))


def test_redetection_clears_unaccepted_identity_from_changed_catalogue():
    queue = accepted_mass().begin_review()
    queue.propose(catalogue(), bin_width=.01, prominence=20)
    row = queue.records[0]
    queue.choose(row["id"], row["candidates"][0]["key"])
    row["evidence"]["mixture"] = {"old_identity_fit": True}
    nickel = get_isotope_catalogue("Ni", (2,), 1, table=isotope_table())
    queue.propose(nickel, bin_width=.01, prominence=20)
    refreshed = queue._get(row["id"])
    assert refreshed["chosen"] is None and "mixture" not in refreshed["evidence"]
    with pytest.raises(CalibrationStateError, match="Choose an identity"):
        queue.decide(row["id"], "accept", ambiguity_reviewed=True)
