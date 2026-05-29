"""Tests for helper_data_loader.summarize_loaded_events.

Covers the two TDC-link summaries:

* ``dld_without_match`` — DLD events with no raw-TDC pulse behind them
  (negative ``event_group_id`` assigned by ``build_event_group_mapping``
  when a dld start_counter run has no tdc counterpart).
* ``match_check`` — the TDC-vs-DLD position/ToF cross-check.
"""
import pandas as pd

from pyccapt.calibration.data_tools._raw_workflow_surface_concept import (
    _surface_concept_hit_from_time_data as _hit4,
)
from pyccapt.calibration.tutorials.tutorials_helpers.helper_data_loader import (
    summarize_loaded_events,
)


class _V:
    def __init__(self, data, data_tdc):
        self.data = data
        self.data_tdc = data_tdc


def _tdc(rows):
    out = []
    for sc, gid, chs, ts, hm in rows:
        for c, tt in zip(chs, ts):
            out.append(
                dict(channel=c, start_counter=sc, time_data=tt, event_group_id=gid, has_dld_match=hm)
            )
    return pd.DataFrame(out)


def _dld(events):
    rows = []
    for gid, x, y, t in events:
        rows.append(
            {
                "x_det (cm)": x, "y_det (cm)": y, "t (ns)": t,
                "event_group_id": gid, "dlts": 4, "dlts_quality": "native",
            }
        )
    return pd.DataFrame(rows)


def test_dld_without_match_counts_negative_gids():
    x0, y0, t0 = _hit4([100, 140, 130, 110])
    x1, y1, t1 = _hit4([200, 260, 250, 210])
    tdc = _tdc([
        (5, 0, [0, 1, 2, 3], [100, 140, 130, 110], True),
        (6, 1, [0, 1, 2, 3], [200, 260, 250, 210], True),
    ])
    # Two matched dld events (gid 0, 1) and two unmatched ones (gid -1, -2).
    dld = _dld([(0, x0, y0, t0), (1, x1, y1, t1), (-1, 0.1, 0.2, 50.0), (-2, 0.3, 0.4, 51.0)])
    s = summarize_loaded_events(_V(dld, tdc), print_summary=False)
    assert s["dld_without_match_known"] is True
    assert s["dld_without_match"] == 2
    assert s["dld_total"] == 4


def test_dld_without_match_unknown_when_no_link_column():
    # No event_group_id column (raw tdc not loaded) -> count is "unknown".
    dld = pd.DataFrame({"x_det (cm)": [0.1], "y_det (cm)": [0.2], "t (ns)": [50.0]})
    s = summarize_loaded_events(_V(dld, None), print_summary=False)
    assert s["dld_without_match_known"] is False
    assert s["dld_without_match"] == 0


def test_match_check_zero_residual_for_correct_link():
    x0, y0, t0 = _hit4([100, 140, 130, 110])
    tdc = _tdc([(5, 0, [0, 1, 2, 3], [100, 140, 130, 110], True)])
    dld = _dld([(0, x0, y0, t0)])
    s = summarize_loaded_events(_V(dld, tdc), print_summary=False)
    assert s["match_check"]["n_compared"] == 1
    assert s["match_check"]["dx_max_cm"] < 1e-9
    assert s["match_check"]["dtof_max_ns"] < 1e-9
