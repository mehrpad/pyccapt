"""Surface Concept raw-stop positions must match the firmware's dld values.

The FPGA quadrupel finder reports ``x_bin = floor((t0 - t1 - TimeDif1Min) / 2)``
(``TimeDif1Min = -2402``, ``TimeDif2Min = -2380`` in ``tdc_gpx3.ini``), so the
detector centre is not at bin 1225. Positions rebuilt from raw stops (raw
analysis, partial-hit recovery) must use the same formula to line up with
native dld hits.
"""
import numpy as np
import pytest

from pyccapt.calibration.data_tools import _raw_workflow_common as common
from pyccapt.calibration.data_tools import _raw_workflow_surface_concept as sc
from pyccapt.calibration.data_tools import partial_recovery
from pyccapt.calibration.mc import mc_tools

# (t0, t1, t2, t3, dld x [cm], dld y [cm], dld t [ns]) of single-ion pulses
# from run 2559 (Jul-09-2026, NiC6 316L HPT): the four raw stops of the pulse
# and the event the firmware wrote for it.
RUN_2559_EVENTS = [
    (154070, 153952, 153412, 154598, 0.11428571428571431, -2.0506122448979593, 4224.747456),
    (30029, 27953, 29160, 28819, 3.311020408163266, 0.4408163265306123, 795.260538),
    (32224, 33220, 33071, 32359, -1.7044897959183676, 1.0481632653061226, 897.533892),
    (23692, 23345, 22424, 24600, 0.486530612244898, -3.6669387755102045, 645.070338),
    (23712, 22496, 22512, 23684, 1.9069387755102043, -2.0277551020408167, 633.706632),
    (44214, 46075, 45585, 44715, -3.118367346938776, 1.306122448979592, 1238.479362),
]


@pytest.mark.parametrize("event", RUN_2559_EVENTS)
def test_raw_stops_reproduce_firmware_dld_event(event):
    *stops, x, y, tof = event
    det_x, det_y, det_t = sc._surface_concept_hit_from_time_data(stops)
    assert det_x == pytest.approx(x, abs=1e-9)
    assert det_y == pytest.approx(y, abs=1e-9)
    assert det_t == pytest.approx(tof, abs=1e-6)


def test_vectorised_position_matches_scalar():
    stops = np.array([event[:4] for event in RUN_2559_EVENTS], dtype=np.int64)
    np.testing.assert_allclose(common.sc_position_cm(stops[:, 0], stops[:, 1], "x"),
                               [event[4] for event in RUN_2559_EVENTS], atol=1e-9)
    np.testing.assert_allclose(common.sc_position_cm(stops[:, 2], stops[:, 3], "y"),
                               [event[5] for event in RUN_2559_EVENTS], atol=1e-9)


def test_quadrupel_window_is_configurable():
    constants = common.load_detector_constants("surface_concept", {"sc_time_dif1_min": -2450, "sc_time_dif2_min": -2450})
    assert (constants["time_dif_min_x"], constants["time_dif_min_y"]) == (-2450, -2450)
    # With a symmetric window the centre is back at XY_BIN_SHIFT.
    assert common.sc_position_cm(5000, 5000, "x", constants) == pytest.approx(0.0)
    defaults = common.load_detector_constants("surface_concept")
    assert (defaults["time_dif_min_x"], defaults["time_dif_min_y"]) == (-2402, -2380)


def test_estimate_pair_sum_offset_uses_clean_single_ion_pulses():
    rng = np.random.default_rng(0)
    channels, times, counters = [], [], []
    for pulse in range(500):
        t0, t1, t2 = rng.integers(1000, 5000, size=3)
        t3 = t0 + t1 - t2 - 12 + int(rng.integers(-3, 4))  # (t0+t1)-(t2+t3) = 12 +- 3
        channels += [2, 0, 3, 1]
        times += [t2, t0, t3, t1]
        counters += [pulse] * 4
    # A multi-hit pulse and a pulse outside the firmware window must be ignored.
    channels += [0, 1, 2, 3, 0, 0, 1, 2, 3]
    times += [100, 100, 100, 5000, 1, 2, 3, 4, 5]
    counters += [900] * 4 + [901] * 5
    offset = sc.estimate_pair_sum_offset(np.array(channels), np.array(times), np.array(counters))
    assert offset == pytest.approx(12, abs=1)
    assert sc.estimate_pair_sum_offset(np.array(channels[:8]), np.array(times[:8]), np.array(counters[:8])) == 0.0


def test_three_of_four_and_single_axis_tof_use_pair_sum_offset():
    t0, t1, t2, t3 = 30000, 31000, 30200, 30790  # (t0+t1)-(t2+t3) = 10
    full_x, full_y, full_tof = sc._surface_concept_hit_from_time_data([t0, t1, t2, t3])

    hits = sc._recover_surface_concept_partial_hits(
        np.array([0, 1, 2]), np.array([t0, t1, t2]), pair_sum_offset_bins=10.0
    )
    assert [hit.get("recovery_method") for hit in hits] == ["3of4"]
    assert hits[0]["x_det (cm)"] == pytest.approx(full_x)
    assert hits[0]["y_det (cm)"] == pytest.approx(full_y)
    assert hits[0]["tof (ns)"] == pytest.approx(full_tof)

    for channels, stops in (([0, 1], [t0, t1]), ([2, 3], [t2, t3])):
        hit, = sc._recover_surface_concept_partial_hits(np.array(channels), np.array(stops), pair_sum_offset_bins=10.0)
        assert hit["tof (ns)"] == pytest.approx(full_tof)


def test_single_axis_partial_mc_uses_chord_rms_for_missing_axis():
    row = partial_recovery._build_recovered_row(
        start_counter=1, high_voltage=5000.0, pulse_v=0.0, pulse_l=0.0,
        recovered_hit={"x_det (cm)": 2.0, "y_det (cm)": np.nan, "tof (ns)": 800.0, "detector_axis": "x"},
        new_gid=0, dld_columns=["mc (Da)", "mc_uc (Da)"], flight_path_length_mm=110.0, detector_radius_cm=4.0,
    )
    expected = mc_tools.tof2mc(t=np.array([800.0]), t0=0, V=np.array([5000.0]), xDet=np.array([2.0]),
                               yDet=np.array([np.sqrt((16.0 - 4.0) / 3.0)]), flightPathLength=110.0,
                               V_pulse=np.array([0.0]), mode="voltage")[0]
    assert row["mc_uc (Da)"] == pytest.approx(expected)
    assert np.isnan(row["y_det (cm)"])
    assert row["dlts"] == 2


def test_matched_multihit_subtraction_finds_firmware_stops():
    # Firmware ion from run 2559 plus a second ion's stops in the same pulse.
    *stops, x, y, tof = RUN_2559_EVENTS[1]
    channels = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    times = np.array([*stops, 31000, 31400, 31190, 31200])
    residual = partial_recovery._subtract_firmware_stops(channels, times, [(x, y, tof)], match_tol_cm=0.1)
    assert residual is not None
    np.testing.assert_array_equal(residual[1], [31000, 31400, 31190, 31200])
