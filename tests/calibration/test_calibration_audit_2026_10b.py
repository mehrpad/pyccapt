"""Regression tests for the second 2026-10 calibration audit pass."""
import json

import h5py
import numpy as np
import pandas as pd
import pytest

from pyccapt.calibration.core import calibration, peak_models
from pyccapt.calibration.core.concentration_profile import calculate_concentration_profile
from pyccapt.calibration.core.guided_workflow import GuidedWorkflow
from pyccapt.calibration.core.isotope_catalogue import get_isotope_catalogue
from pyccapt.calibration.core.ranging_review import PeakReviewQueue
from pyccapt.calibration.core.share_variables import Variables
from pyccapt.calibration.data_tools.nexus_export import export_nexus
from pyccapt.instrument_metadata import ATTR, calibrated_tof_reference, extract_instrument_settings


def _isotope_table():
    return pd.DataFrame({"element": ["Fe", "Fe", "Ni"], "isotope": [54, 56, 58],
                         "weight": [54., 56., 58.], "abundance": [10., 90., 100.]})


def _workflow():
    rng = np.random.default_rng(3)
    mass = np.r_[rng.normal(27., .025, 1000), rng.normal(28., .025, 2000), rng.normal(29., .025, 1000)]
    frame = pd.DataFrame({"mc (Da)": mass, "mc_uc (Da)": mass, "t (ns)": np.sqrt(mass)*20,
        "high_voltage (V)": np.full(len(mass), 5000.), "x_det (cm)": rng.normal(0, .1, len(mass)),
        "y_det (cm)": rng.normal(0, .1, len(mass)), "event_id": np.arange(len(mass), dtype=np.uint64)+2**60})
    variables = Variables()
    variables.sync_from_data(frame)
    return GuidedWorkflow(variables)


# --- 1. isotope-family mixture: the reference background is not a pedestal ---

def test_mixture_evidence_subtracts_reference_background_from_template():
    catalogue = get_isotope_catalogue("Fe,Ni", (2,), 1, table=_isotope_table())
    rng = np.random.default_rng(0)
    true_areas = np.array([4000., 16000.])
    background = rng.uniform(26.5, 29.5, int(50/.005*3))  # 50 counts per 0.005 Da bin
    mass = np.r_[rng.normal(27., .02, int(true_areas[0])), rng.normal(28., .02, int(true_areas[1])),
                 rng.normal(29., .02, 10000), background]
    queue = PeakReviewQueue(mass)
    queue.catalogue = catalogue
    queue.catalogue_spec = [list(catalogue.key[0]), list(catalogue.key[1]), catalogue.key[2], catalogue.key[3]]
    ids = []
    for centre in (27., 28., 29.):
        row = queue._new(centre, centre-.15, centre+.15, catalogue)
        queue.records.append(row)
        key = next(item["key"] for item in row["candidates"] if abs(item["mass"]-centre) < .01)
        queue.choose(row["id"], key)
        queue.decide(row["id"], "accept", ambiguity_reviewed=True)
        ids.append(row["id"])

    fit = queue.mixture_evidence(ids[:2], ids[2], identities_reviewed=True)

    # Without background subtraction the template pedestal inflated the small
    # isotope by ~16% and the large one by ~9%.
    np.testing.assert_allclose(fit.amplitudes/true_areas, 1., atol=.05)
    np.testing.assert_allclose(fit.background_density, 10000., rtol=.05)
    evidence = queue._get(ids[0])["evidence"]["mixture"]["reference_background"]
    assert evidence["method"] == "adjacent_sidebands"
    assert evidence["counts_per_bin"] == pytest.approx(50., rel=.1)


# --- 2. non-finite voltage rows keep their value -----------------------------

@pytest.mark.parametrize("mode", ["tof", "mc"])
def test_voltage_corr_main_passes_nan_voltage_rows_through(mode):
    voltage = np.repeat(np.linspace(4000.0, 6000.0, 30), 100)
    measured = 100.0 * (1.0 + 0.0001 * (voltage - 5000.0))
    missing = np.zeros(voltage.size, dtype=bool)
    missing[::97] = True
    voltage[missing] = np.nan
    variables = Variables()
    variables.sync_from_data(pd.DataFrame({'t (ns)': measured, 'high_voltage (V)': np.nan_to_num(voltage, nan=5000.)}))
    variables.dld_t_calib = measured.copy()
    variables.mc_calib = measured.copy()
    variables.set_calibration_peak_range(mode, 89.0, 111.0)
    calibration.voltage_corr_main(
        voltage, variables, sample_size=100, mode="ion_seq",
        calibration_mode=mode, index_fig=0, plot=False, save=False,
        model="curve_fit", maximum_sample_method="mean",
        maximum_cal_method="mean", peak_maximum=100.0,
    )
    corrected = variables.get_calibration_array(mode)
    np.testing.assert_array_equal(corrected[missing], measured[missing])
    np.testing.assert_allclose(corrected[~missing], 100.0, atol=1e-6)


@pytest.mark.parametrize("model", ["curve_fit", "robust_fit"])
def test_multi_peak_voltage_corr_passes_nan_voltage_rows_through(model):
    rng = np.random.default_rng(0)
    voltage = np.tile(np.linspace(4000.0, 6000.0, 20000), 3)
    labels = np.repeat([28.0, 58.0, 100.0], 20000)
    measured = labels * (1.0 + 0.002 * (voltage - 5000.0) / 1000.0) * (1.0 + rng.normal(0, .0005, labels.size))
    missing = np.zeros(voltage.size, dtype=bool)
    missing[::50] = True
    voltage[missing] = np.nan
    variables = Variables()
    variables.dld_t_calib = measured.copy()
    variables.mc_calib = measured.copy()
    calibration.multi_peak_voltage_corr_main(
        voltage, variables, calibration_mode="mc", model=model,
        bin_size=0.01, n_peaks=3, prominence=10, distance=10,
    )
    corrected = variables.mc_calib
    np.testing.assert_array_equal(corrected[missing], measured[missing])
    assert np.isfinite(corrected).all()


# --- 3./4./6. guided workflow journal, availability and t_c reference -------

def test_guided_accept_ranges_records_the_accepted_range_table():
    session = _workflow()
    session.preview_imported_mass()
    session.accept("calibration")
    queue = session.begin_review()
    queue.propose(get_isotope_catalogue("Fe,Ni", (1, 2), 1, table=_isotope_table()),
                  prominence=20, bin_width=.01, tolerance=.08)
    row = queue.records[0]
    queue.choose(row["id"], row["candidates"][0]["key"])
    queue.decide(row["id"], "accept", ambiguity_reviewed=True)
    session.preview_ranges()
    session.accept("ranges")
    step = session.variables.calibration_metadata.state["steps"][-1]
    assert step["operation"] == "guided_accept_ranges"
    ranges = step["parameters"]["ranges"]
    accepted = session.accepted["ranges"]["rows"][0]
    assert len(ranges) == 1
    assert ranges[0] == {key: accepted[key] for key in ("ion", "mc_low", "mc_up", "element", "complex", "charge")}
    assert step["parameters"]["ranged_events"] == int(np.count_nonzero(session.accepted["ranges"]["assignment"] >= 0))


def test_mass_column_is_unavailable_until_a_calibration_is_accepted():
    session = _workflow()
    data = session.variables.data
    assert np.isnan(data["mc (Da)"]).all()
    assert "mc (Da)" in data.attrs["pyccapt_unavailable_columns"]
    session.preview_imported_mass()
    session.accept("calibration")
    assert "mc (Da)" not in session.variables.data.attrs["pyccapt_unavailable_columns"]
    session.undo()
    assert "mc (Da)" in session.variables.data.attrs["pyccapt_unavailable_columns"]


def test_fitted_t0_is_not_offered_and_normalized_tof_reference_is_exported(tmp_path):
    session = _workflow()
    mass = session.source["mc (Da)"].to_numpy()
    receipt = {"method": "validated reference model", "flight_length_mm": 110., "pulse_mode": "voltage",
               "model": {"t0_ns": 17.4},
               "output": {"units": "ns", "t0_ns": 0., "convention": "normalized reference ToF",
                          "reference_voltage_V": 5000., "reference_flight_length_mm": 125.}}
    session.preview_calibration(mass, tof=np.sqrt(mass)*20, receipt=receipt)
    session.accept("calibration")
    frame = session.variables.data
    # The fitted offset stays as provenance but is never an instrument setting.
    assert frame.attrs[ATTR]["calibration"]["t0_ns"]["status"] == "fitted_model"
    assert frame.attrs[ATTR]["calibration"]["t0_ns"]["value"] == 17.4
    assert extract_instrument_settings(frame) == {"flight_path_length_mm": 110., "pulse_mode": "voltage"}
    reference = calibrated_tof_reference(frame)
    assert reference["convention"] == "normalized reference ToF"
    assert (reference["reference_voltage_V"], reference["reference_flight_length_mm"], reference["t0_ns"]) == (5000., 125., 0.)

    path = tmp_path / "normalized.nxs"
    export_nexus(path, session.export_frame())
    with h5py.File(path, "r") as handle:
        report = json.loads(handle["entry/mapping_report_json"].asstr()[()])
        assert "t0_ns" in report["missing_instrument_settings"]
        attrs = handle["entry/atom_probe1/voltage_and_bowl/calibrated_tof"].attrs
        assert attrs["convention"] == "normalized reference ToF"
        assert attrs["reference_flight_length_mm"] == 125. and attrs["reference_voltage_V"] == 5000.
        assert attrs["t0_ns"] == 0.
        group = handle["entry/atom_probe1/instrument_metadata/calibrated_tof"]
        assert group["reference_flight_length_mm"][()] == 125.
        assert group["reference_flight_length_mm"].attrs["units"] == "mm"

    session.undo()
    assert calibrated_tof_reference(session.variables.data) == {}


# --- 5./8. peak-model convergence and asymmetric MRP position ---------------

def test_gaussian_and_voigt_fits_pinned_on_bounds_are_not_converged():
    x = np.arange(27.0, 29.0, .002)
    ramp = np.linspace(0., 100., x.size)  # no peak: the centre runs to the window edge
    index = x.size // 2
    assert peak_models._fit_gaussian_mrp(x, ramp, index)[1] is False
    assert peak_models._fit_voigt_mrp(x, ramp, index)[1] is False
    # A genuine zero-background peak (background on its lower bound) still fits.
    gaussian = 1000 * np.exp(-(x - 28) ** 2 / (2 * .01 ** 2))
    result, ok = peak_models._fit_gaussian_mrp(x, gaussian, int(np.argmax(gaussian)))
    assert ok and result[0] == pytest.approx(28 / (peak_models._FWHM_FACTOR * .01), rel=1e-3)
    lorentzian = 1000 * .01 ** 2 / ((x - 28) ** 2 + .01 ** 2)
    assert peak_models._fit_voigt_mrp(x, lorentzian, int(np.argmax(lorentzian)))[1] is True


def test_gaussian_fit_without_finite_covariance_is_not_converged():
    x = np.arange(27.0, 29.0, .002)
    y = 1000 * np.exp(-(x - x[1]) ** 2 / (2 * .002 ** 2)) + 3
    # A peak one bin from the edge leaves three samples for four parameters:
    # curve_fit "succeeds" with an infinite covariance (old MRP ~25000).
    assert peak_models._fit_gaussian_mrp(x, y, 1)[1] is False


def test_asymmetric_mrp_uses_profile_maximum_not_erf_centre():
    x = np.arange(1.7, 2.6, .001)
    centre, sigma, tail = 2.0, .01, .1
    y = peak_models._err_exp_decay(x, 1000., centre, sigma, tail, 5.)
    result, ok, fwhm = peak_models._fit_asymmetric_mrp(x, y, int(np.argmax(y)))
    dense = np.linspace(1.9, 2.2, 300001)
    apex = dense[np.argmax(peak_models._err_exp_decay(dense, 1000., centre, sigma, tail, 0.))]
    assert ok and apex > centre + sigma
    assert result[0] == pytest.approx(apex / fwhm, abs=.02)
    assert abs(result[0] - centre / fwhm) > .1


# --- 7. ion selections are labelled as atoms contributed --------------------

def test_ion_selection_is_labelled_as_atoms_contributed():
    ranges = pd.DataFrame({"name": ["H2"], "mc_low": [1.5], "mc_up": [2.5],
                           "element": [["H"]], "complex": [[2]]})
    profile = calculate_concentration_profile(np.array([2.0, 99.0]), ranges, ["ion:0"], window_size=2)
    # Two atoms from one H2 event plus one unranged atom-equivalent: 2/3 at.%,
    # whereas the ion (event) fraction would be 1/2.
    assert "H2 (ion)" not in profile
    assert profile["H2 (ion atoms)"].iloc[0] == pytest.approx(200 / 3)
