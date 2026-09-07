from __future__ import annotations

import struct

import pandas as pd
import pytest

from pyccapt.calibration.leap_tools.ccapt_tools import ccapt_to_epos, ccapt_to_pos


@pytest.fixture
def ccapt_frame():
    return pd.DataFrame(
        {
            'x (nm)': [1.0, 2.0], 'y (nm)': [3.0, 4.0],
            'z (nm)': [5.0, 6.0], 'mc (Da)': [7.0, 8.0],
            't (ns)': [9.0, 10.0], 'high_voltage (V)': [11.0, 12.0],
            'pulse_v (V)': [13.0, 14.0], 'x_det (cm)': [1.5, 2.5],
            'y_det (cm)': [3.5, 4.5], 'delta_p': [15, 16], 'multi': [17, 18],
        }
    )


def test_pos_writer_matches_binary_format(ccapt_frame):
    expected = b''.join(
        struct.pack('>ffff', *row)
        for row in ccapt_frame[['x (nm)', 'y (nm)', 'z (nm)', 'mc (Da)']].itertuples(index=False, name=None)
    )
    assert ccapt_to_pos(ccapt_frame) == expected


def test_epos_writer_matches_binary_format_across_chunks(ccapt_frame):
    expected = b''.join(
        struct.pack('>fffffffffII', x, y, z, mc, tof, hv, pulse, dx * 10, dy * 10, dp, multi)
        for x, y, z, mc, tof, hv, pulse, dx, dy, dp, multi in ccapt_frame.itertuples(index=False, name=None)
    )
    assert ccapt_to_epos(ccapt_frame, chunk_size=1) == expected


def test_epos_writer_rejects_invalid_chunk_size(ccapt_frame):
    with pytest.raises(ValueError, match='positive integer'):
        ccapt_to_epos(ccapt_frame, chunk_size=0)
