"""Quick benchmark to validate the 3 calibration fixes."""

from __future__ import annotations
import sys, time, io, json
from pathlib import Path
from contextlib import redirect_stdout

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from pyccapt.calibration.core import calibration
from pyccapt.calibration.core.adaptive_residual_calibration import adaptive_residual_calibration
from pyccapt.calibration.core.mc_plot_peak_helpers import gaussian_mrp_report
from pyccapt.calibration.core.share_variables import Variables

DATASET = REPO_ROOT / "2470_Apr-17-2025_13-10_NiC2_Nimonic90_HT_HPCF_1.h5"
E_CHARGE = 1.6e-19
AMU = 1.66e-27
ALPHA = 1.015
BETA = 0.7
MAIN_PROMINENCE = 100
MAIN_DISTANCE = 500


def current_voltage(variables):
    pulse = getattr(variables, "dld_pulse_v", np.zeros_like(variables.dld_high_voltage))
    return variables.dld_high_voltage + BETA * pulse


def run_mc_initial(variables, det_diam):
    calibration.bowl_correction_main(
        variables.dld_x_det,
        variables.dld_y_det,
        current_voltage(variables),
        variables,
        det_diam,
        sample_size=5,
        fit_mode="robust_fit",
        maximum_cal_method="mean",
        maximum_sample_method="histogram",
        fig_size=(5, 5),
        calibration_mode="mc",
        index_fig=1,
        plot=False,
        save=False,
        fast_calibration=False,
        bin_size=0.01,
        peak_maximum=0,
        sampling_mode="polar",
    )


def run_voltage(variables, mode):
    calibration.voltage_corr_main(
        current_voltage(variables),
        variables,
        sample_size=10000,
        mode="ion_seq",
        calibration_mode=mode,
        index_fig=1,
        plot=False,
        save=False,
        maximum_cal_method="mean",
        maximum_sample_method="histogram",
        fig_size=(5, 5),
        fast_calibration=False,
        model="robust_fit",
        bin_size=0.01,
        peak_maximum=0,
    )


def run_bowl(variables, mode, det_diam):
    calibration.bowl_correction_main(
        variables.dld_x_det,
        variables.dld_y_det,
        current_voltage(variables),
        variables,
        det_diam,
        sample_size=5,
        fit_mode="robust_fit",
        maximum_cal_method="mean",
        maximum_sample_method="histogram",
        fig_size=(5, 5),
        calibration_mode=mode,
        index_fig=1,
        plot=False,
        save=False,
        fast_calibration=False,
        bin_size=0.01,
        peak_maximum=0,
        sampling_mode="polar",
    )


def select_peak(variables, mode):
    arr = variables.dld_t_calib if mode == "tof" else variables.mc_calib
    peaks = calibration.auto_detect_reference_peaks(
        arr,
        n_peaks=1,
        prominence=MAIN_PROMINENCE,
        distance=MAIN_DISTANCE,
        hist_bin_size=1.0 if mode == "tof" else 0.1,
    )
    if peaks:
        variables.selected_x1 = float(peaks[0]["x1"])
        variables.selected_x2 = float(peaks[0]["x2"])
        mask = (arr > variables.selected_x1) & (arr < variables.selected_x2)
        variables.set_calibration_selection_mask(mode, mask)
    return peaks


def measure_mrp(arr, label="", n_peaks=3):
    peaks = calibration.auto_detect_reference_peaks(
        arr, n_peaks=n_peaks, prominence=MAIN_PROMINENCE, distance=MAIN_DISTANCE, hist_bin_size=0.1
    )
    results = []
    for p in peaks:
        report = gaussian_mrp_report(arr, p["x1"], p["x2"], bin_size=0.001)
        if report:
            mrp = report["recommended_mrp"]
            rec_label = report["recommended_label"]
            print(
                f"  {label} Peak {p['position']:.3f}: "
                f"MRP(0.5)={mrp[0]:.1f}, MRP(0.1)={mrp[1]:.1f}, MRP(0.01)={mrp[2]:.1f} [{rec_label}]"
            )
            results.append({"position": p["position"], "mrp_50": mrp[0], "mrp_10": mrp[1], "mrp_01": mrp[2]})
        else:
            print(f"  {label} Peak {p['position']:.3f}: report=None")
    return results


def avg_mrp(results):
    vals = [r["mrp_50"] for r in results if np.isfinite(r["mrp_50"])]
    return float(np.mean(vals)) if vals else float("nan")


def fresh_variables(data):
    v = Variables()
    v.data = data
    v.sync_from_data(data)
    # Select the strongest peak on the raw mc_uc so that
    # bowl_correction_main has a valid peak range from the start.
    arr = v.mc_calib  # mc_calib starts as mc_uc
    peaks = calibration.auto_detect_reference_peaks(
        arr, n_peaks=1, prominence=MAIN_PROMINENCE, distance=MAIN_DISTANCE, hist_bin_size=0.1
    )
    if peaks:
        v.selected_x1 = float(peaks[0]["x1"])
        v.selected_x2 = float(peaks[0]["x2"])
        mask = (arr > v.selected_x1) & (arr < v.selected_x2)
        v.set_calibration_selection_mask("mc", mask)
    return v


def main():
    print(f"Loading dataset: {DATASET}")
    data = pd.read_hdf(str(DATASET))
    print(f"  {len(data)} ions")

    mc = data["mc_uc (Da)"].to_numpy(float)
    tof = data["t (ns)"].to_numpy(float)
    hv = data["high_voltage (V)"].to_numpy(float)
    pv = data["pulse_v (V)"].to_numpy(float)
    valid = np.isfinite(mc) & np.isfinite(tof) & (mc > 0) & (tof > 0) & (hv > 0)
    eff_v = ALPHA * (hv[valid] + BETA * pv[valid])
    mass_kg = mc[valid] * AMU
    length_m = np.sqrt(2.0 * E_CHARGE * eff_v * (tof[valid] * 1e-9) ** 2 / mass_kg)
    flight_path_mm = float(np.median(length_m) * 1e3)

    x_det = data["x_det (cm)"].to_numpy(float) * 10.0
    y_det = data["y_det (cm)"].to_numpy(float) * 10.0
    radius = np.sqrt(x_det**2 + y_det**2)
    det_diam = float(np.percentile(radius[np.isfinite(radius)], 99.9) * 2.0)
    print(f"  Flight path: {flight_path_mm:.1f} mm, Det diam: {det_diam:.1f} mm\n")

    results_all = {}

    def run_test(name, fn):
        print("=" * 60)
        print(f">>> {name}")
        sys.stdout.flush()
        t0 = time.perf_counter()
        try:
            fn()
        except Exception as e:
            import traceback

            traceback.print_exc()
            elapsed = time.perf_counter() - t0
            print(f"  FAILED: {e}")
            results_all[name] = {"score": float("nan"), "time_s": elapsed, "error": str(e)}
            return None
        elapsed = time.perf_counter() - t0
        return elapsed

    # 1) Baseline
    v = fresh_variables(data)
    elapsed = run_test("baseline_mc", lambda: None)
    r = measure_mrp(v.mc_uc, "baseline")
    results_all["baseline_mc"] = {"score": avg_mrp(r), "time_s": elapsed, "mrps": r}
    print(f"  Score: {results_all['baseline_mc']['score']:.1f}\n")

    # 2) MC Initial
    v = fresh_variables(data)

    def do_initial():
        with redirect_stdout(io.StringIO()):
            run_mc_initial(v, det_diam)

    elapsed = run_test("mc_initial", do_initial)
    if elapsed is not None:
        r = measure_mrp(v.mc_calib, "mc_init")
        results_all["mc_initial"] = {"score": avg_mrp(r), "time_s": elapsed, "mrps": r}
        print(f"  Score: {results_all['mc_initial']['score']:.1f}\n")

    # 3) MC Initial + Voltage
    v = fresh_variables(data)

    def do_init_voltage():
        with redirect_stdout(io.StringIO()):
            run_mc_initial(v, det_diam)
            select_peak(v, "mc")
            run_voltage(v, "mc")

    elapsed = run_test("mc_initial+voltage", do_init_voltage)
    if elapsed is not None:
        r = measure_mrp(v.mc_calib, "init+V")
        results_all["mc_initial_voltage"] = {"score": avg_mrp(r), "time_s": elapsed, "mrps": r}
        print(f"  Score: {results_all['mc_initial_voltage']['score']:.1f}\n")

    # 4) MC Initial + Bowl
    v = fresh_variables(data)

    def do_init_bowl():
        with redirect_stdout(io.StringIO()):
            run_mc_initial(v, det_diam)
            select_peak(v, "mc")
            run_bowl(v, "mc", det_diam)

    elapsed = run_test("mc_initial+bowl", do_init_bowl)
    if elapsed is not None:
        r = measure_mrp(v.mc_calib, "init+bowl")
        results_all["mc_initial_bowl"] = {"score": avg_mrp(r), "time_s": elapsed, "mrps": r}
        print(f"  Score: {results_all['mc_initial_bowl']['score']:.1f}\n")

    # 5) MC Initial + V + Bowl
    v = fresh_variables(data)

    def do_init_v_bowl():
        with redirect_stdout(io.StringIO()):
            run_mc_initial(v, det_diam)
            select_peak(v, "mc")
            run_voltage(v, "mc")
            select_peak(v, "mc")
            run_bowl(v, "mc", det_diam)

    elapsed = run_test("mc_initial+V+bowl", do_init_v_bowl)
    if elapsed is not None:
        r = measure_mrp(v.mc_calib, "init+V+bowl")
        results_all["mc_initial_v_bowl"] = {"score": avg_mrp(r), "time_s": elapsed, "mrps": r}
        print(f"  Score: {results_all['mc_initial_v_bowl']['score']:.1f}\n")

    # 6) MC Initial + Bowl + V
    v = fresh_variables(data)

    def do_init_bowl_v():
        with redirect_stdout(io.StringIO()):
            run_mc_initial(v, det_diam)
            select_peak(v, "mc")
            run_bowl(v, "mc", det_diam)
            select_peak(v, "mc")
            run_voltage(v, "mc")

    elapsed = run_test("mc_initial+bowl+V", do_init_bowl_v)
    if elapsed is not None:
        r = measure_mrp(v.mc_calib, "init+bowl+V")
        results_all["mc_initial_bowl_v"] = {"score": avg_mrp(r), "time_s": elapsed, "mrps": r}
        print(f"  Score: {results_all['mc_initial_bowl_v']['score']:.1f}\n")

    # 7) MC Initial + Adaptive Residual
    v = fresh_variables(data)

    def do_init_ar():
        with redirect_stdout(io.StringIO()):
            run_mc_initial(v, det_diam)
        adaptive_residual_calibration(
            v,
            calibration_mode="mc",
            n_peaks=6,
            prominence=100,
            distance=10,
            n_windows=24,
            overlap=0.5,
            template_bin_size=0.01,
            temporal_smoothing=0.5,
            apply_spatial=True,
            spatial_grid=12,
            min_window_ions=40,
            min_cell_ions=35,
            verbose=True,
        )

    elapsed = run_test("mc_initial+AR", do_init_ar)
    if elapsed is not None:
        r = measure_mrp(v.mc_calib, "init+AR")
        results_all["mc_initial_ar"] = {"score": avg_mrp(r), "time_s": elapsed, "mrps": r}
        print(f"  Score: {results_all['mc_initial_ar']['score']:.1f}\n")

    # 8) Full: Initial + V + Bowl + AR
    v = fresh_variables(data)

    def do_full():
        with redirect_stdout(io.StringIO()):
            run_mc_initial(v, det_diam)
            select_peak(v, "mc")
            run_voltage(v, "mc")
            select_peak(v, "mc")
            run_bowl(v, "mc", det_diam)
        adaptive_residual_calibration(
            v,
            calibration_mode="mc",
            n_peaks=6,
            prominence=100,
            distance=10,
            n_windows=24,
            overlap=0.5,
            template_bin_size=0.01,
            temporal_smoothing=0.5,
            apply_spatial=True,
            spatial_grid=12,
            min_window_ions=40,
            min_cell_ions=35,
            verbose=True,
        )

    elapsed = run_test("mc_full_V_bowl_AR", do_full)
    if elapsed is not None:
        r = measure_mrp(v.mc_calib, "full")
        results_all["mc_full_v_bowl_ar"] = {"score": avg_mrp(r), "time_s": elapsed, "mrps": r}
        print(f"  Score: {results_all['mc_full_v_bowl_ar']['score']:.1f}\n")

    # 9) Full: Initial + Bowl + V + AR
    v = fresh_variables(data)

    def do_full2():
        with redirect_stdout(io.StringIO()):
            run_mc_initial(v, det_diam)
            select_peak(v, "mc")
            run_bowl(v, "mc", det_diam)
            select_peak(v, "mc")
            run_voltage(v, "mc")
        adaptive_residual_calibration(
            v,
            calibration_mode="mc",
            n_peaks=6,
            prominence=100,
            distance=10,
            n_windows=24,
            overlap=0.5,
            template_bin_size=0.01,
            temporal_smoothing=0.5,
            apply_spatial=True,
            spatial_grid=12,
            min_window_ions=40,
            min_cell_ions=35,
            verbose=True,
        )

    elapsed = run_test("mc_full_bowl_V_AR", do_full2)
    if elapsed is not None:
        r = measure_mrp(v.mc_calib, "full2")
        results_all["mc_full_bowl_v_ar"] = {"score": avg_mrp(r), "time_s": elapsed, "mrps": r}
        print(f"  Score: {results_all['mc_full_bowl_v_ar']['score']:.1f}\n")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<30} {'Avg MRP(0.5)':>12} {'Time':>8}")
    print("-" * 55)
    for name, info in results_all.items():
        score = info.get("score", float("nan"))
        t = info.get("time_s", 0)
        err = info.get("error", "")
        suffix = f"  ERROR: {err}" if err else ""
        print(f"{name:<30} {score:>12.1f} {t:>7.1f}s{suffix}")

    out_path = REPO_ROOT / "quick_benchmark_v3.json"
    with open(out_path, "w") as f:
        json.dump(results_all, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
