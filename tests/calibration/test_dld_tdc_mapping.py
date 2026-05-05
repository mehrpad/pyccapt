"""Tests for the dld<->tdc event-group mapping and the save_tdc round-trip."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from pyccapt.calibration.data_tools import data_loadcrop, data_tools


# ---------------------------------------------------------------------------
# Mapping algorithm
# ---------------------------------------------------------------------------


def test_build_event_group_mapping_user_example():
    """Reproduce the exact example the user described.

    dld has counters [7537, 15374, 11858]; tdc has runs of consecutive equal
    counters with several "orphan" runs (10994, 5082, 12979, 16462, 16852)
    that did not produce reconstructible dld events.
    """
    dld_sc = np.array([7537, 15374, 11858])
    tdc_sc = np.array([
        16852, 16852,
        7537, 7537, 7537, 7537,
        15374, 15374, 15374, 15374,
        10994,
        11858, 11858, 11858, 11858,
        5082, 5082, 5082, 5082,
        12979, 12979, 12979,
        16462, 16462,
    ])

    dld_gid, tdc_gid, has_match = data_loadcrop.build_event_group_mapping(dld_sc, tdc_sc)

    assert dld_gid.tolist() == [0, 1, 2]
    expected_tdc_gid = [
        -1, -1,
        0, 0, 0, 0,
        1, 1, 1, 1,
        -1,
        2, 2, 2, 2,
        -1, -1, -1, -1,
        -1, -1, -1,
        -1, -1,
    ]
    assert tdc_gid.tolist() == expected_tdc_gid
    expected_has_match = [bool(g >= 0) for g in expected_tdc_gid]
    assert has_match.tolist() == expected_has_match


def test_build_event_group_mapping_handles_counter_wraparound():
    """The same counter value can repeat after wraparound.

    The mapping must rely on consecutive runs in the time-ordered arrays,
    never on counter values being unique.
    """
    # Two cycles, both ending with counter 5 (the wrap value).
    dld_sc = np.array([1, 5, 2, 5])
    tdc_sc = np.array([
        1, 1, 1, 1,
        5, 5, 5, 5,
        2, 2, 2, 2,
        5, 5, 5, 5,
    ])

    dld_gid, tdc_gid, has_match = data_loadcrop.build_event_group_mapping(dld_sc, tdc_sc)

    assert dld_gid.tolist() == [0, 1, 2, 3]
    assert tdc_gid.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    assert has_match.all()


def test_build_event_group_mapping_handles_multi_hit_dld():
    """One pulse trigger can produce multiple dld rows (multi-hit)."""
    dld_sc = np.array([7, 7, 9])  # two dld rows in pulse 7
    tdc_sc = np.array([7, 7, 7, 7, 9, 9, 9, 9])

    dld_gid, tdc_gid, has_match = data_loadcrop.build_event_group_mapping(dld_sc, tdc_sc)

    # Both dld rows in pulse 7 share the same group id.
    assert dld_gid.tolist() == [0, 0, 1]
    assert tdc_gid.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert has_match.all()


def test_build_event_group_mapping_empty_inputs():
    dld_gid, tdc_gid, has_match = data_loadcrop.build_event_group_mapping(
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
    )
    assert dld_gid.size == 0
    assert tdc_gid.size == 0
    assert has_match.size == 0


def test_build_event_group_mapping_raises_when_dld_has_no_tdc_match():
    """A dld pulse without a corresponding tdc run signals inconsistent inputs."""
    dld_sc = np.array([1, 2, 3])
    tdc_sc = np.array([1, 1, 3, 3])  # dld pulse 2 has no tdc rows

    with pytest.raises(ValueError, match="without a matching tdc"):
        data_loadcrop.build_event_group_mapping(dld_sc, tdc_sc)


# ---------------------------------------------------------------------------
# filter_tdc_by_dld
# ---------------------------------------------------------------------------


def _make_linked_pair():
    """Build a small dld/tdc pair using the user's example layout."""
    dld_sc = np.array([7537, 15374, 11858])
    tdc_sc = np.array([
        16852, 16852,
        7537, 7537, 7537, 7537,
        15374, 15374, 15374, 15374,
        10994,
        11858, 11858, 11858, 11858,
        5082, 5082, 5082, 5082,
        12979, 12979, 12979,
        16462, 16462,
    ])
    dld_gid, tdc_gid, has_match = data_loadcrop.build_event_group_mapping(dld_sc, tdc_sc)
    dld_df = pd.DataFrame({
        "start_counter": dld_sc,
        "t (ns)": [100.0, 200.0, 300.0],
        "event_group_id": dld_gid,
    })
    tdc_df = pd.DataFrame({
        "start_counter": tdc_sc,
        "channel": np.arange(len(tdc_sc)),
        "event_group_id": tdc_gid,
        "has_dld_match": has_match,
    })
    return dld_df, tdc_df


def test_filter_tdc_by_dld_keeps_orphans_and_drops_only_deleted_groups():
    dld_df, tdc_df = _make_linked_pair()

    # Drop the middle dld row (event_group_id == 1, counter 15374).
    dld_after = dld_df.drop(index=[1]).reset_index(drop=True)

    filtered = data_loadcrop.filter_tdc_by_dld(dld_after, tdc_df)

    # The 4 tdc rows linked to group 1 should be gone (counter 15374).
    # All orphan tdc rows should remain.
    assert (filtered["start_counter"] != 15374).all() or (
        not (filtered["event_group_id"] == 1).any()
    )
    # All orphan rows survived (they have has_dld_match == False).
    orphan_count_before = int((~tdc_df["has_dld_match"]).sum())
    orphan_count_after = int((~filtered["has_dld_match"]).sum())
    assert orphan_count_before == orphan_count_after
    # Group 1 tdc rows are dropped (4 rows).
    group1_before = int((tdc_df["event_group_id"] == 1).sum())
    group1_after = int((filtered["event_group_id"] == 1).sum())
    assert group1_before == 4
    assert group1_after == 0
    # Group 0 (counter 7537) and group 2 (counter 11858) tdc rows still there.
    assert (filtered["event_group_id"] == 0).sum() == 4
    assert (filtered["event_group_id"] == 2).sum() == 4


def test_filter_tdc_by_dld_full_dld_keeps_all_tdc():
    dld_df, tdc_df = _make_linked_pair()

    filtered = data_loadcrop.filter_tdc_by_dld(dld_df, tdc_df)
    assert len(filtered) == len(tdc_df)


def test_filter_tdc_by_dld_empty_dld_keeps_only_orphans():
    dld_df, tdc_df = _make_linked_pair()

    filtered = data_loadcrop.filter_tdc_by_dld(dld_df.iloc[0:0], tdc_df)
    # Only orphan rows survive.
    assert (~filtered["has_dld_match"]).all()
    assert len(filtered) == int((~tdc_df["has_dld_match"]).sum())


def test_filter_tdc_by_dld_requires_event_group_columns():
    dld_df = pd.DataFrame({"start_counter": [1, 2]})
    tdc_df = pd.DataFrame({"start_counter": [1, 2]})
    with pytest.raises(ValueError):
        data_loadcrop.filter_tdc_by_dld(dld_df, tdc_df)


# ---------------------------------------------------------------------------
# fetch_dataset_with_tdc on a synthetic h5 file
# ---------------------------------------------------------------------------


def _write_minimal_pyccapt_h5(path: Path, dld_sc: np.ndarray, tdc_sc: np.ndarray):
    """Write a tiny synthetic pyccapt-format h5 file with both dld and tdc groups."""
    n_dld = len(dld_sc)
    n_tdc = len(tdc_sc)
    with h5py.File(path, "w") as hdf:
        dld = hdf.create_group("dld")
        # all of these are written as (n, 1) arrays by pyccapt
        dld.create_dataset("high_voltage", data=np.full((n_dld, 1), 1000.0))
        dld.create_dataset("voltage_pulse", data=np.full((n_dld, 1), 200.0))
        dld.create_dataset("laser_intensity", data=np.zeros((n_dld, 1)))
        dld.create_dataset("start_counter", data=dld_sc.reshape(-1, 1).astype(np.int64))
        dld.create_dataset("t", data=np.linspace(100.0, 200.0, n_dld).reshape(-1, 1))
        dld.create_dataset("x", data=np.full((n_dld, 1), 0.5))
        dld.create_dataset("y", data=np.full((n_dld, 1), -0.5))

        tdc = hdf.create_group("tdc")
        tdc.create_dataset("channel", data=np.arange(n_tdc).reshape(-1, 1).astype(np.int64))
        tdc.create_dataset("start_counter", data=tdc_sc.reshape(-1, 1).astype(np.int64))
        tdc.create_dataset("high_voltage", data=np.full((n_tdc, 1), 1000.0))
        tdc.create_dataset("voltage_pulse", data=np.full((n_tdc, 1), 200.0))
        tdc.create_dataset("laser_pulse", data=np.zeros((n_tdc, 1)))
        tdc.create_dataset("time_data", data=np.linspace(50.0, 80.0, n_tdc).reshape(-1, 1))


def test_fetch_dataset_with_tdc_assigns_shared_event_group_id(tmp_path: Path):
    dld_sc = np.array([7537, 15374, 11858])
    tdc_sc = np.array([
        16852, 16852,
        7537, 7537, 7537, 7537,
        15374, 15374, 15374, 15374,
        10994,
        11858, 11858, 11858, 11858,
        5082, 5082, 5082, 5082,
    ])
    h5_path = tmp_path / "synthetic.h5"
    _write_minimal_pyccapt_h5(h5_path, dld_sc, tdc_sc)

    dld_df, tdc_df = data_loadcrop.fetch_dataset_with_tdc(str(h5_path))

    assert "event_group_id" in dld_df.columns
    assert "event_group_id" in tdc_df.columns
    assert "has_dld_match" in tdc_df.columns
    # 3 distinct dld groups -> ids 0, 1, 2.
    assert sorted(dld_df["event_group_id"].unique().tolist()) == [0, 1, 2]
    # Orphan tdc rows (counters 16852, 10994, 5082) still appear.
    assert (~tdc_df["has_dld_match"]).sum() == 2 + 1 + 4


def test_fetch_dataset_with_tdc_supports_roentdek_extract_mode(tmp_path: Path):
    dld_sc = np.array([10, 11])
    tdc_sc = np.array([10, 10, 10, 10, 10, 10, 11, 11, 11, 11])
    h5_path = tmp_path / "synthetic_roentdek.h5"
    _write_minimal_pyccapt_h5(h5_path, dld_sc, tdc_sc)

    dld_df, tdc_df = data_loadcrop.fetch_dataset_with_tdc(str(h5_path), tdc_extract_mode="tdc_ro")

    assert "event_group_id" in dld_df.columns
    assert "event_group_id" in tdc_df.columns
    assert "has_dld_match" in tdc_df.columns
    assert list(tdc_df.columns[:6]) == [
        "channel",
        "start_counter",
        "high_voltage (V)",
        "pulse_v (V)",
        "pulse_l (pJ)",
        "time_data",
    ]


# ---------------------------------------------------------------------------
# save_data with save_tdc round-trip
# ---------------------------------------------------------------------------


class _StubVariables:
    """Minimal stand-in for `Variables` that resolves the result file path."""

    def __init__(self, output_dir: Path, name: str = "calibrated"):
        self._output_dir = output_dir
        self.result_data_name = name
        self.result_data_path = str(output_dir) + "/"
        self.result_path = str(output_dir) + "/"
        self.data_tdc = None

    def resolve_result_data_file(self, filename: str) -> str:
        return str(self._output_dir / filename)

    def resolve_result_file(self, filename: str) -> str:
        return str(self._output_dir / filename)


def test_save_data_with_save_tdc_writes_filtered_tdc_group(tmp_path: Path):
    dld_df, tdc_df = _make_linked_pair()
    # Drop the middle dld row to test that its tdc group gets removed at save.
    dld_after = dld_df.drop(index=[1]).reset_index(drop=True)

    variables = _StubVariables(tmp_path, name="testset")
    variables.data_tdc = tdc_df

    data_tools.save_data(
        dld_after,
        variables,
        hdf=True,
        save_tdc=True,
    )

    out_path = tmp_path / "testset.h5"
    assert out_path.exists()

    loaded_dld = pd.read_hdf(out_path, key="df")
    loaded_tdc = pd.read_hdf(out_path, key="tdc")

    # dld round-trip
    assert len(loaded_dld) == 2
    assert set(loaded_dld["start_counter"]) == {7537, 11858}

    # tdc round-trip: orphans preserved, group-1 (15374) dropped.
    assert (~loaded_tdc["has_dld_match"]).sum() == int((~tdc_df["has_dld_match"]).sum())
    assert (loaded_tdc["event_group_id"] == 1).sum() == 0
    assert (loaded_tdc["event_group_id"] == 0).sum() == 4
    assert (loaded_tdc["event_group_id"] == 2).sum() == 4


def test_save_data_warns_when_save_tdc_but_no_tdc_loaded(tmp_path: Path):
    dld_df, _ = _make_linked_pair()
    variables = _StubVariables(tmp_path, name="testset")
    # variables.data_tdc is None

    with pytest.warns(RuntimeWarning, match="data_tdc is None"):
        data_tools.save_data(dld_df, variables, hdf=True, save_tdc=True)

    out_path = tmp_path / "testset.h5"
    assert out_path.exists()
    # No /tdc key was written.
    with pytest.raises(KeyError):
        pd.read_hdf(out_path, key="tdc")


def test_save_data_warns_when_dld_lacks_event_group_id(tmp_path: Path):
    # dld has no event_group_id column (raw tdc was not loaded with the link).
    dld_df = pd.DataFrame({"start_counter": [1, 2, 3], "t (ns)": [10.0, 20.0, 30.0]})
    tdc_df = pd.DataFrame({
        "start_counter": [1, 2, 3],
        "event_group_id": [0, 1, 2],
        "has_dld_match": [True, True, True],
    })
    variables = _StubVariables(tmp_path, name="testset")
    variables.data_tdc = tdc_df

    with pytest.warns(RuntimeWarning, match="event_group_id"):
        data_tools.save_data(dld_df, variables, hdf=True, save_tdc=True)


def test_event_group_id_survives_typical_filtering_chain():
    """The group id rides through drop/iloc/reset_index without being mangled."""
    dld_df = pd.DataFrame({
        "start_counter": [1, 2, 3, 4, 5],
        # remove_invalid_data drops rows with t<50 or t>max_tof; pick valid TOFs.
        "t (ns)": [100.0, 6000.0, 200.0, 300.0, 400.0],
        "x_det (cm)": [0.5] * 5,
        "y_det (cm)": [0.5] * 5,
        "high_voltage (V)": [1000.0] * 5,
        "event_group_id": [10, 11, 12, 13, 14],
    })

    # 1) remove_invalid_data drops the second row (TOF > 5000).
    cleaned = data_tools.remove_invalid_data(dld_df.copy(), max_tof=5000)
    assert "event_group_id" in cleaned.columns
    assert cleaned["event_group_id"].tolist() == [10, 12, 13, 14]

    # 2) iloc-style temporal crop (positional slice).
    cropped = cleaned.iloc[1:3].reset_index(drop=True).copy()
    assert cropped["event_group_id"].tolist() == [12, 13]

    # 3) boolean mask drop.
    keep_mask = cropped["event_group_id"] != 13
    final = cropped.loc[keep_mask].reset_index(drop=True).copy()
    assert final["event_group_id"].tolist() == [12]
