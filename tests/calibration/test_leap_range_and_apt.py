import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyccapt.calibration.data_tools import data_tools
from pyccapt.calibration.leap_tools import ccapt_tools, leap_tools


RRNG_TEXT = """[Ions]
Ion1=Al
Ion2=AlO
[Ranges]
Range1=26.90 27.10 Vol:1.00 Al:1 Color:FF0000
Range2=42,40 42,80 Vol:2,50 Al:1 O:1 Color:00FF00
Range3=0.00 0.10 Vol:0.00 Name:0 Color:FFFFFF
"""

RRNG_MISMATCH_TEXT = """[Ions]
Ion1=WrongAl
Ion2=WrongCr
[Ranges]
Range1=26.90 27.10 Vol:1.00 Al:1 Color:FF0000
Range2=51.90 52.20 Vol:1.00 Cr:1 Color:00FF00
"""

RRNG_NAME_TEXT = """[Ions]
Ion1=27CrMo
[Ranges]
Range1=51.90 52.20 Vol:1.00 Name:27CrMo Color:AA5500
"""

RNG_TEXT = """2 3
Cr
Cr 1.00 0.20 0.80
O
O 0.00 0.80 1.00
----------------- Cr O
. 51.6990 54.2430 1 0
. 67.6220 69.5740 1 1
. 83.5950 86.5550 1 2

--- polyatomic extension
2 2
CrO
CrO 1.00 0.00 0.00
CrO2
CrO2 0.00 1.00 0.00
----------------- CrO CrO2
. 67.6220 69.5740 1 0
. 83.5950 86.5550 0 1
"""


def _write_rrng(path: Path) -> None:
    path.write_text(RRNG_TEXT, encoding="utf-8")


def _write_rng(path: Path) -> None:
    path.write_text(RNG_TEXT, encoding="utf-8")


def _pack_utf16_field(text: str, char_count: int) -> bytes:
    encoded = text.encode("utf-16")
    return encoded.ljust(char_count * 2, b"\x00")[: char_count * 2]


def _build_apt_section(name: str, data_bytes: bytes, record_size: int, record_count: int, unit: str = "") -> bytes:
    header = (
        struct.pack("<4sii", b"SECT", 148, 1)
        + _pack_utf16_field(name, 32)
        + struct.pack("<iiiiii", 1, 1, 1, 3, 32, record_size)
        + _pack_utf16_field(unit, 16)
        + struct.pack("<qq", record_count, len(data_bytes))
    )
    return header + data_bytes


def _write_minimal_apt(path: Path) -> np.ndarray:
    ion_count = 2
    file_header = (
        struct.pack("<4sii", b"APT\0", 540, 1)
        + _pack_utf16_field("synthetic.apt", 256)
        + struct.pack("<Qq", 0, ion_count)
    )

    mass_values = np.asarray([27.0, 54.0], dtype=np.float32)
    tipbox = np.asarray([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    position_values = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    mass_section = _build_apt_section("Mass", mass_values.tobytes(), record_size=4, record_count=ion_count, unit="Da")
    position_section = _build_apt_section(
        "Position",
        tipbox.tobytes() + position_values.tobytes(),
        record_size=12,
        record_count=ion_count,
        unit="nm",
    )

    path.write_bytes(file_header + mass_section + position_section)
    return tipbox


def _write_minimal_epos(path: Path, trailing_bytes: bytes = b"") -> None:
    record_dtype = np.dtype([
        ("x", ">f4"),
        ("y", ">f4"),
        ("z", ">f4"),
        ("mc", ">f4"),
        ("tof", ">f4"),
        ("hv", ">f4"),
        ("pulse", ">f4"),
        ("det_x", ">f4"),
        ("det_y", ">f4"),
        ("pslep", ">u4"),
        ("ipp", ">u4"),
    ])
    records = np.array(
        [
            (1.0, 2.0, 3.0, 27.0, 100.0, 5000.0, 12.0, 1.5, -1.5, 7, 1),
            (4.0, 5.0, 6.0, 54.0, 200.0, 5100.0, 15.0, 2.5, -2.5, 9, 2),
        ],
        dtype=record_dtype,
    )
    path.write_bytes(records.tobytes() + trailing_bytes)


def test_read_rrng_returns_raw_and_normalized_tables(tmp_path: Path):
    range_path = tmp_path / "example.rrng"
    _write_rrng(range_path)

    ions, rrngs = leap_tools.read_rrng(range_path, return_tables=True)
    assert ions["name"].tolist() == ["Al", "AlO"]
    assert rrngs.loc["2", "upper"] == 42.8
    assert rrngs.loc["2", "comp"] == "Al:1 O:1"

    normalized = leap_tools.read_rrng(range_path)
    via_shared_reader = data_tools.read_range(range_path)

    assert normalized["element"].tolist()[0] == ["Al"]
    assert normalized["element"].tolist()[1] == ["Al", "O"]
    assert normalized["complex"].tolist()[1] == [1, 1]
    assert normalized["color"].tolist()[1] == "#00FF00"
    assert normalized["raw_comp"].tolist()[2] == "Name:0"
    assert via_shared_reader["mc_low"].tolist() == normalized["mc_low"].tolist()


def test_read_rrng_accepts_name_species_tokens(tmp_path: Path):
    range_path = tmp_path / "name_tokens.rrng"
    range_path.write_text(RRNG_NAME_TEXT, encoding="utf-8")

    ions, rrngs = leap_tools.read_rrng(range_path, return_tables=True)
    normalized = leap_tools.read_rrng(range_path)

    assert ions.loc["1", "name"] == "27CrMo"
    assert "ion_name" not in rrngs.columns or rrngs.loc["1"].get("ion_name", "") in ("", "27CrMo")
    assert normalized["name"].tolist() == ["27CrMo"]
    assert normalized["element"].tolist()[0] == ["Cr", "Mo"]
    assert normalized["complex"].tolist()[0] == [1, 1]
    assert normalized["isotope"].tolist()[0] == [27, 0]


def test_read_rrng_does_not_map_ion_names_by_range_index(tmp_path: Path):
    range_path = tmp_path / "mismatch.rrng"
    range_path.write_text(RRNG_MISMATCH_TEXT, encoding="utf-8")

    normalized = leap_tools.read_rrng(range_path)

    assert normalized["name"].tolist() == ["Al", "Cr"]
    assert normalized["element"].tolist() == [["Al"], ["Cr"]]
    assert normalized["color"].tolist() == ["#FF0000", "#00FF00"]


def test_read_rng_returns_raw_and_normalized_tables(tmp_path: Path):
    range_path = tmp_path / "example.rng"
    _write_rng(range_path)

    ions, ranges = leap_tools.read_rng(range_path, return_tables=True)
    assert ions["name"].tolist() == ["Cr", "O"]
    assert ranges.loc["2", "comp"] == "Cr:1 O:1"
    assert ranges.loc["2", "ion_name"] == "CrO"
    assert ranges.loc["2", "colour"] == "FF0000"
    assert ranges.loc["3", "ion_name"] == "CrO2"
    assert ranges.loc["3", "colour"] == "00FF00"

    normalized = leap_tools.read_rng(range_path)
    via_shared_reader = data_tools.read_range(range_path)

    assert normalized["name"].tolist() == ["Cr", "CrO", "CrO2"]
    assert normalized["element"].tolist()[2] == ["Cr", "O"]
    assert normalized["complex"].tolist()[2] == [1, 2]
    assert normalized["color"].tolist()[1] == "#FF0000"
    assert via_shared_reader["name"].tolist() == normalized["name"].tolist()


def test_label_ions_accepts_mass_column_aliases(tmp_path: Path):
    range_path = tmp_path / "example.rrng"
    _write_rrng(range_path)
    _, rrngs = leap_tools.read_rrng(range_path, return_tables=True)

    pos = pd.DataFrame({"m/n (Da)": [27.0, 60.0]})
    labeled = leap_tools.label_ions(pos, rrngs)

    assert labeled["comp"].tolist() == ["Al:1", ""]
    assert labeled["colour"].tolist() == ["#FF0000", "#FFFFFF"]


def test_read_apt_skips_position_tipbox(tmp_path: Path):
    apt_path = tmp_path / "synthetic.apt"
    expected_tipbox = _write_minimal_apt(apt_path)

    dataframe = leap_tools.read_apt(apt_path)

    np.testing.assert_allclose(dataframe["Mass"].to_numpy(), [27.0, 54.0])
    np.testing.assert_allclose(dataframe["x"].to_numpy(), [1.0, 4.0])
    np.testing.assert_allclose(dataframe["y"].to_numpy(), [2.0, 5.0])
    np.testing.assert_allclose(dataframe["z"].to_numpy(), [-3.0, -6.0])
    np.testing.assert_allclose(dataframe.attrs["tipbox"], expected_tipbox)


def test_apt_to_ccapt_uses_position_fallbacks_and_elapsed_counter(monkeypatch):
    apt_dataframe = pd.DataFrame(
        {
            "Mass": [27.0, 54.0],
            "x": [1.0, 4.0],
            "y": [2.0, 5.0],
            "z": [-3.0, -6.0],
            "tElapsed": [10, 20],
        }
    )

    monkeypatch.setattr(ccapt_tools.leap_tools, "read_apt", lambda path: apt_dataframe.copy())
    ccapt = ccapt_tools.apt_to_ccapt("synthetic.apt")

    np.testing.assert_allclose(ccapt["x (nm)"].to_numpy(), [1.0, 4.0])
    np.testing.assert_allclose(ccapt["z (nm)"].to_numpy(), [-3.0, -6.0])
    assert ccapt["start_counter"].tolist() == [10, 20]


def test_read_epos_uses_record_reader_for_large_files(tmp_path: Path):
    epos_path = tmp_path / "synthetic.epos"
    _write_minimal_epos(epos_path)

    dataframe = leap_tools.read_epos(epos_path)

    np.testing.assert_allclose(dataframe["m/n (Da)"].to_numpy(), [27.0, 54.0])
    np.testing.assert_allclose(dataframe["TOF (ns)"].to_numpy(), [100.0, 200.0])
    assert dataframe["pslep"].tolist() == [7, 9]
    assert dataframe["ipp"].tolist() == [1, 2]


def test_read_epos_ignores_partial_trailing_bytes(tmp_path: Path):
    epos_path = tmp_path / "truncated.epos"
    _write_minimal_epos(epos_path, trailing_bytes=b"\x00\x01\x02\x03")

    with pytest.warns(RuntimeWarning, match="Ignoring the final 4 trailing bytes"):
        dataframe = leap_tools.read_epos(epos_path)

    assert len(dataframe) == 2
    np.testing.assert_allclose(dataframe["x (nm)"].to_numpy(), [1.0, 4.0])
