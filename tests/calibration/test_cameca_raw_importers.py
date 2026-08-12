import numpy as np
import pandas as pd
import pytest

from pyccapt.calibration.leap_tools.cameca_raw import (
    rhit_apply_calibration,
    rhit_to_ccapt,
    str_calculate_positions,
    str_load,
    str_to_ccapt,
)


def test_rhit_apply_calibration_uses_saved_polynomial():
    hits = pd.DataFrame(
        {
            "detx": [0.0, 1.0],
            "dety": [0.0, 2.0],
            "tof": [10.0, 12.0],
            "VDC": [1000.0, 1000.0],
        }
    )
    calibration = {"t_offset": 1.0, "C_poly": [1.0, 0.5, 0.1]}

    calibrated = rhit_apply_calibration(hits, calibration)

    assert calibrated["mc"].tolist() == pytest.approx([81000.0, 726000.0])


def test_rhit_to_ccapt_builds_processed_dataset_columns():
    hits = pd.DataFrame(
        {
            "mc": [1.0, 2.0],
            "VDC": [1000.0, 1100.0],
            "tof": [5.0, 6.0],
            "detx": [1.0, 2.0],
            "dety": [3.0, 4.0],
            "pulse": [10.0, 20.0],
        }
    )

    dataset = rhit_to_ccapt(hits)

    assert list(dataset.columns[:5]) == ["x (nm)", "y (nm)", "z (nm)", "mc (Da)", "mc_uc (Da)"]
    assert dataset["x_det (cm)"].tolist() == pytest.approx([0.1, 0.2])
    assert dataset["pulse_v (V)"].tolist() == pytest.approx([10.0, 20.0])


def test_str_calculate_positions_recovers_partial_hits():
    hits = pd.DataFrame(
        {
            "ionIdx": [1, 2, 3, 4],
            "detxt1": [100.0, 110.0, np.nan, 120.0],
            "detxt2": [80.0, 90.0, np.nan, 100.0],
            "detyt1": [200.0, 210.0, 220.0, np.nan],
            "detyt2": [180.0, 190.0, 200.0, np.nan],
            "detwt1": [314.1421, np.nan, 334.1421, 344.1421],
            "detwt2": [285.8579, np.nan, 305.8579, 315.8579],
            "quality": [0.0, 0.0, 0.0, 0.0],
        }
    )

    result = str_calculate_positions(hits)

    assert result["detxRaw"].tolist() == pytest.approx([20.0, 20.0, 20.0, 20.0], abs=1e-3)
    assert result["detyRaw"].tolist() == pytest.approx([20.0, 20.0, 20.0, 20.0], abs=1e-3)
    assert result["hitType"].tolist() == [3, 2, 2, 2]
    assert result["tof"].notna().all()


def test_str_v2_loader_skips_header_record_and_retains_raw_counters(tmp_path):
    """The first 0x18 closes metadata, not a detector event (R56 STR v2)."""
    records = [
        # Header closed by 0x18.  This must not appear as ion/event row 1.
        (0xA0, 2), (0x1B, 500), (0x18, 2005),
        # First complete detector event.
        (0x01, 100), (0x02, 90), (0x03, 150), (0x04, 120),
        (0x21, 140), (0x22, 110), (0x05, 1000), (0x0B, 2000), (0x18, 7),
        # Empty raw record: it remains in the source event index but is not a hit.
        (0x18, 8),
        # Second detector event.
        (0x01, 210), (0x02, 200), (0x03, 260), (0x04, 230),
        (0x21, 250), (0x22, 220), (0x05, 1010), (0x0B, 2010), (0x18, 9),
        # Incomplete trailing bytes must not become an out-of-range event.
        (0x01, 999),
    ]
    raw = bytearray()
    for tag, value in records:
        assert 0 <= value < (1 << 23)
        raw.extend((value & 0xFF, (value >> 8) & 0xFF, (value >> 16) & 0xFF, tag))
    path = tmp_path / "minimal.STR"
    path.write_bytes(bytes(raw))

    hits, metadata = str_load(path, verbose=False)

    assert metadata["nHeaderRecords"] == 3
    assert metadata["nEventRecords"] == 3
    assert metadata["nEvents"] == 2
    assert hits["strEventIdx"].tolist() == [1, 3]
    assert hits["pulseCounterA"].tolist() == pytest.approx([1000.0, 1010.0])
    assert hits["pulseCounterB"].tolist() == pytest.approx([2000.0, 2010.0])
    assert hits["eventEndValue"].tolist() == pytest.approx([7.0, 9.0])
    assert hits["quality"].tolist() == pytest.approx([7.0, 9.0])


def test_str_v2_loader_preserves_unsigned_24_bit_delay_line_values(tmp_path):
    """TDC timestamps above 2**23 are valid unsigned 24-bit payloads."""
    records = [
        (0xA0, 2), (0x18, 0),
        (0x01, 9_000_000), (0x02, 9_000_100),
        (0x03, 9_000_200), (0x04, 9_000_300),
        (0x21, 9_000_400), (0x22, 9_000_500), (0x18, 1),
    ]
    raw = bytearray()
    for tag, value in records:
        raw.extend((value & 0xFF, (value >> 8) & 0xFF, (value >> 16) & 0xFF, tag))
    path = tmp_path / "unsigned_tdc.STR"
    path.write_bytes(bytes(raw))

    hits, metadata = str_load(path, verbose=False)

    assert metadata["payloadEncoding"] == "unsigned 24-bit little-endian TLV payload"
    assert hits.loc[0, "detxt1"] == pytest.approx(9_000_000)
    assert hits.loc[0, "detwt2"] == pytest.approx(9_000_500)


def test_str_to_ccapt_requires_calibration_columns():
    hits = pd.DataFrame({"tof": [1.0], "detxRaw": [2.0], "detyRaw": [3.0], "hitType": [2]})

    with pytest.raises(ValueError):
        str_to_ccapt(hits)


def test_str_to_ccapt_uses_unit_multi_not_hit_type():
    hits = pd.DataFrame(
        {
            "mc": [10.0, 11.0],
            "tof_ns": [1.0, 2.0],
            "VDC": [1000.0, 1001.0],
            "detx": [1.0, 2.0],
            "dety": [3.0, 4.0],
            "hitType": [2, 3],
            "ionIdx": [1, 2],
        }
    )

    dataset = str_to_ccapt(hits)

    assert dataset["multi"].tolist() == [1, 1]
