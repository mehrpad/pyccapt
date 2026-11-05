"""Surface Concept raw-stop positions must match the firmware's dld values.

The FPGA quadrupel finder reports ``x_bin = floor((t0 - t1 - TimeDif1Min) / 2)``
(``TimeDif1Min = -2402``, ``TimeDif2Min = -2380`` in ``tdc_gpx3.ini``), so the
detector centre is pixel 1201 / 1190, not 1225. Positions rebuilt from raw
stops (raw analysis, partial-hit recovery) and dld x/y (live, or converted from
legacy files on load) use the same geometry (sc_geometry).
"""
import numpy as np
import pytest

from pyccapt.calibration.data_tools import _raw_workflow_common as common
from pyccapt.calibration.data_tools import _raw_workflow_surface_concept as sc
from pyccapt.calibration.data_tools import partial_recovery, sc_geometry
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
    # Run 2559 predates the geometry fix; its dld x/y are converted on load.
    *stops, x, y, tof = event
    det_x, det_y, det_t = sc._surface_concept_hit_from_time_data(stops)
    assert det_x == pytest.approx(sc_geometry.legacy_cm_to_cm(x, "x"), abs=1e-9)
    assert det_y == pytest.approx(sc_geometry.legacy_cm_to_cm(y, "y"), abs=1e-9)
    assert det_t == pytest.approx(tof, abs=1e-6)


def test_vectorised_position_matches_scalar():
    stops = np.array([event[:4] for event in RUN_2559_EVENTS], dtype=np.int64)
    np.testing.assert_allclose(common.sc_position_cm(stops[:, 0], stops[:, 1], "x"),
                               sc_geometry.legacy_cm_to_cm([event[4] for event in RUN_2559_EVENTS], "x"), atol=1e-9)
    np.testing.assert_allclose(common.sc_position_cm(stops[:, 2], stops[:, 3], "y"),
                               sc_geometry.legacy_cm_to_cm([event[5] for event in RUN_2559_EVENTS], "y"), atol=1e-9)


def test_geometry_centres_the_firmware_window():
    # Pixels 0 and 2401 (x) / 2379 (y) are the window edges: +-4.0 cm.
    assert sc_geometry.bins_to_cm(0, "x") == pytest.approx(-sc_geometry.bins_to_cm(2401, "x"))
    assert sc_geometry.bins_to_cm(0, "y") == pytest.approx(-sc_geometry.bins_to_cm(2379, "y"))
    assert sc_geometry.bins_to_cm(2401, "x") == pytest.approx(3.998, abs=1e-3)
    # Equal stop times (ion at the delay-line centre) land on the centre pixel.
    assert abs(common.sc_position_cm(5000, 5000, "x")) <= 0.5 * sc_geometry.PIXEL_SIZE_X_MM * 0.1


def test_legacy_coordinates_convert_exactly():
    pixels = np.array([0, 17, 1201, 1225, 2401])
    legacy = (pixels - 1225) * 80.0 / 2450 * 0.1
    assert sc_geometry.looks_like_legacy_cm(legacy, legacy)
    np.testing.assert_allclose(sc_geometry.legacy_cm_to_cm(legacy, "x"), sc_geometry.bins_to_cm(pixels, "x"))
    assert not sc_geometry.looks_like_legacy_cm([-1.23483, 0.0], [-1.55235, 0.0])  # CoboldPC values
    assert not sc_geometry.looks_like_legacy_cm([0.0, 0.0], [0.0, 0.0])


def test_quadrupel_window_is_configurable():
    conf = {"sc_time_dif1_min": -2450, "sc_time_dif2_min": -2450, "sc_pixel_size_x_mm": 0.05}
    constants = common.load_detector_constants("surface_concept", conf)
    assert (constants["time_dif_min_x"], constants["time_dif_min_y"]) == (-2450, -2450)
    assert constants["pixel_size_x_mm"] == 0.05
    assert common.sc_position_cm(5100, 5000, "x", constants) == pytest.approx((1275 + 0.5 - 1225) * 0.05 * 0.1)
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
    x, y = sc_geometry.legacy_cm_to_cm(x, "x"), sc_geometry.legacy_cm_to_cm(y, "y")
    channels = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    times = np.array([*stops, 31000, 31400, 31190, 31200])
    residual = partial_recovery._subtract_firmware_stops(channels, times, [(x, y, tof)], match_tol_cm=0.1)
    assert residual is not None
    np.testing.assert_array_equal(residual[1], [31000, 31400, 31190, 31200])


@pytest.mark.parametrize("legacy_coordinates", [False, True])
def test_combinatorial_keeps_firmware_ion_and_searches_only_leftover_stops(legacy_coordinates):
    from pyccapt.calibration.data_tools import _raw_workflow_sc_combinatorial as combinatorial

    first = RUN_2559_EVENTS[2]          # firmware ion
    second = (32210, 33190, 33080, 32340)  # another ion in the same pulse, not in dld
    channels = [0, 1, 2, 3, 0, 1, 2, 3]
    times = [*first[:4], *second]
    x, y, tof = first[4], first[5], first[6]
    if not legacy_coordinates:
        x, y = sc_geometry.legacy_cm_to_cm(x, "x"), sc_geometry.legacy_cm_to_cm(y, "y")
    record = {"channels": channels, "time_data": times, "start_counter": [1], "high_voltage": 5000.0, "pulse": 0.0}

    emitted, _ = combinatorial.extract_valid_hits_combinatorial(
        record, mode="exhaustive", max_tof_ns=6000.0, firmware_events=[(x, y, tof)]
    )
    firmware = [hit for hit in emitted if hit["firmware"]]
    assert len(firmware) == 1
    assert firmware[0]["used_indices"] == frozenset({0, 1, 2, 3})
    assert firmware[0]["tof (ns)"] == pytest.approx(tof)
    others = [hit for hit in emitted if not hit["firmware"]]
    assert [hit["dlts"] for hit in others] == [4]
    assert others[0]["used_indices"] == frozenset({4, 5, 6, 7})
