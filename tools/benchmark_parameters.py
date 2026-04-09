"""Comprehensive benchmark for PyCCAPT calibration methods and parameter combinations.

Produces a detailed results table showing which parameter / method combinations
produce the best, most stable calibration for both mc and tof modes.

Usage:
    python tools/benchmark_parameters.py <dataset.h5> [--output-json results.json]
"""

from __future__ import annotations

import argparse
import io
import json
import time
from contextlib import redirect_stdout
from itertools import product
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from pyccapt.calibration.core import calibration
from pyccapt.calibration.core.adaptive_residual_calibration import adaptive_residual_calibration
from pyccapt.calibration.core.mc_plot_peak_helpers import gaussian_mrp_report
from pyccapt.calibration.core.share_variables import Variables
try:
    from pyccapt.calibration.core.ml_calibration import ml_calibration
except ImportError:
    ml_calibration = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
E_CHARGE = 1.6e-19
AMU = 1.66e-27
ALPHA = 1.015
DEFAULT_PULSE_TIMING_FACTOR = 0.7


# ---------------------------------------------------------------------------
# Helpers (same as benchmark_calibrations.py)
# ---------------------------------------------------------------------------
def estimate_flight_path_mm(data: pd.DataFrame) -> float:
    mc = data["mc_uc (Da)"].to_numpy(dtype=float)
    tof = data["t (ns)"].to_numpy(dtype=float)
    hv = data["high_voltage (V)"].to_numpy(dtype=float)
    pv = data["pulse_v (V)"].to_numpy(dtype=float) if "pulse_v (V)" in data.columns else np.zeros(len(data))
    valid = np.isfinite(mc) & np.isfinite(tof) & np.isfinite(hv) & np.isfinite(pv) & (mc > 0) & (tof > 0) & (hv > 0)
    if np.count_nonzero(valid) < 100:
        return 120.0
    eff_voltage = ALPHA * (hv[valid] + DEFAULT_PULSE_TIMING_FACTOR * pv[valid])
    mass_kg = mc[valid] * AMU
    length_m = np.sqrt(2.0 * E_CHARGE * eff_voltage * (tof[valid] * 1e-9) ** 2 / mass_kg)
    return float(np.median(length_m) * 1e3)


def estimate_det_diam_mm(data: pd.DataFrame) -> float:
    if "x_det (cm)" not in data.columns:
        return 60.0
    x_mm = data["x_det (cm)"].to_numpy(dtype=float) * 10.0
    y_mm = data["y_det (cm)"].to_numpy(dtype=float) * 10.0
    return float(np.percentile(np.sqrt(x_mm ** 2 + y_mm ** 2)[np.isfinite(x_mm)], 99.9) * 2.0)


def fresh_variables(data: pd.DataFrame) -> Variables:
    v = Variables()
    v.data = data
    v.sync_from_data(data)
    return v


def current_voltage(v: Variables) -> np.ndarray:
    pulse = getattr(v, "dld_pulse_v", np.zeros_like(v.dld_high_voltage))
    return v.dld_high_voltage + DEFAULT_PULSE_TIMING_FACTOR * pulse


def select_peak(v: Variables, mode: str, prominence: int = 100, distance: int = 500) -> dict:
    arr = v.dld_t_calib if mode == "tof" else v.mc_calib
    hist_bin = 1.0 if mode == "tof" else 0.1
    peaks = calibration.auto_detect_reference_peaks(arr, n_peaks=1, prominence=prominence, distance=distance, hist_bin_size=hist_bin)
    p = peaks[0]
    v.selected_x1 = float(p["x1"])
    v.selected_x2 = float(p["x2"])
    mask = (arr > v.selected_x1) & (arr < v.selected_x2)
    if np.any(mask):
        v.set_calibration_selection_mask(mode, mask)
    v.set_calibration_peak_range(mode, v.selected_x1, v.selected_x2)
    return p


def evaluate_spectrum(arr: np.ndarray, mode: str) -> dict:
    """Evaluate MRP quality of a calibrated spectrum."""
    specs = [(50, 50, 0.05), (30, 20, 0.05), (10, 10, 0.05)] if mode == "mc" else [(50, 50, 1.0), (30, 20, 1.0), (10, 10, 1.0)]
    peaks = []
    for prom, dist, hbs in specs:
        try:
            peaks = calibration.auto_detect_reference_peaks(arr, n_peaks=4, prominence=prom, distance=dist, hist_bin_size=hbs)
        except Exception:
            peaks = []
        if len(peaks) >= 2:
            break
    details = []
    weighted = []
    dominant_mrp50 = float("nan")
    for i, peak in enumerate(peaks):
        local_bin = max(1e-4, min(0.02 if np.nanmax(arr) < 500 else 2.0, (peak["x2"] - peak["x1"]) / 80.0))
        report = gaussian_mrp_report(arr, peak["x1"], peak["x2"], bin_size=local_bin)
        if report is None:
            continue
        mrp = report["recommended_mrp"]
        n_ions = int(report["num_ions"])
        w = (0.6, 0.3, 0.1)
        s = sum(wt * max(0, float(val)) for wt, val in zip(w, mrp) if np.isfinite(val))
        ws = sum(wt for wt, val in zip(w, mrp) if np.isfinite(val))
        if ws > 0:
            score = s / ws
            weighted.append((score, max(1.0, np.sqrt(n_ions))))
        if i == 0 and np.isfinite(mrp[0]):
            dominant_mrp50 = float(mrp[0])
        details.append({
            "position": round(float(peak["position"]), 4),
            "n_ions": n_ions,
            "mrp50": round(float(mrp[0]), 2) if np.isfinite(mrp[0]) else None,
            "mrp10": round(float(mrp[1]), 2) if np.isfinite(mrp[1]) else None,
            "mrp01": round(float(mrp[2]), 2) if np.isfinite(mrp[2]) else None,
        })
    total = sum(w for _, w in weighted)
    score = float(sum(s * w for s, w in weighted) / total) if total > 0 else float("nan")
    return {"score": round(score, 2) if np.isfinite(score) else None, "dominant_mrp50": round(dominant_mrp50, 2) if np.isfinite(dominant_mrp50) else None, "peaks": details}


# ---------------------------------------------------------------------------
# Calibration building blocks
# ---------------------------------------------------------------------------
def _run_initial_mc(v, det_diam, fit_mode="robust_fit", sampling_mode="polar", sample_size=5, bin_size=0.01):
    calibration.bowl_correction_main(
        v.dld_x_det, v.dld_y_det, current_voltage(v), v, det_diam,
        sample_size=sample_size, fit_mode=fit_mode, maximum_cal_method="mean",
        maximum_sample_method="histogram", fig_size=(5, 5), calibration_mode="mc",
        index_fig=1, plot=False, save=False, bin_size=bin_size, sampling_mode=sampling_mode,
    )


def _run_initial_tof(v, flight_path_mm, det_diam, fit_mode="robust_fit", sampling_mode="polar", sample_size=5, bin_size=0.01):
    v.dld_t_calib = calibration.initial_calibration(v.data, flight_path_mm)
    select_peak(v, "tof")
    _run_voltage(v, "tof", model="robust_fit", mode="ion_seq", sample_size=10000, bin_size=bin_size)
    select_peak(v, "tof")
    calibration.bowl_correction_main(
        v.dld_x_det, v.dld_y_det, current_voltage(v), v, det_diam,
        sample_size=sample_size, fit_mode=fit_mode, maximum_cal_method="mean",
        maximum_sample_method="histogram", fig_size=(5, 5), calibration_mode="tof",
        index_fig=1, plot=False, save=False, bin_size=bin_size, sampling_mode=sampling_mode,
    )


def _run_voltage(v, cal_mode, model="robust_fit", mode="ion_seq", sample_size=10000, bin_size=0.01):
    calibration.voltage_corr_main(
        current_voltage(v), v, sample_size=sample_size, mode=mode,
        calibration_mode=cal_mode, index_fig=1, plot=False, save=False,
        maximum_cal_method="mean", maximum_sample_method="histogram",
        model=model, bin_size=bin_size,
    )


def _run_bowl(v, cal_mode, det_diam, fit_mode="robust_fit", sampling_mode="polar", sample_size=5, bin_size=0.01):
    calibration.bowl_correction_main(
        v.dld_x_det, v.dld_y_det, current_voltage(v), v, det_diam,
        sample_size=sample_size, fit_mode=fit_mode, maximum_cal_method="mean",
        maximum_sample_method="histogram", fig_size=(5, 5), calibration_mode=cal_mode,
        index_fig=1, plot=False, save=False, bin_size=bin_size, sampling_mode=sampling_mode,
    )


def _run_multi_peak(v, cal_mode, det_diam, model="robust_fit", fit_mode="robust_fit",
                    n_peaks=3, prominence=100, distance=500, sampling_mode="polar"):
    calibration.multi_peak_voltage_corr_main(
        current_voltage(v), v, calibration_mode=cal_mode, model=model,
        bin_size=0.01, n_peaks=n_peaks, prominence=prominence, distance=distance,
    )
    calibration.multi_peak_bowl_corr_main(
        v.dld_x_det, v.dld_y_det, current_voltage(v), v, det_diam,
        calibration_mode=cal_mode, fit_mode=fit_mode, sample_size=5,
        bin_size=0.01, n_peaks=n_peaks, prominence=prominence, distance=distance,
        sampling_mode=sampling_mode,
    )


def _run_adaptive_residual(v, cal_mode, n_peaks=6, n_windows=24, apply_spatial=True,
                           spatial_grid=12, max_rounds=8, template_bin_size=0.01,
                           temporal_smoothing=0.5, prominence=100, distance=10):
    return adaptive_residual_calibration(
        v, calibration_mode=cal_mode, n_peaks=n_peaks, prominence=prominence,
        distance=distance, n_windows=n_windows, overlap=0.5,
        template_bin_size=template_bin_size, temporal_smoothing=temporal_smoothing,
        apply_spatial=apply_spatial, spatial_grid=spatial_grid,
        min_window_ions=40, min_cell_ions=35, max_rounds=max_rounds, verbose=False,
    )


def _run_ml(v, cal_mode, n_peaks=6, prominence=100, distance=10,
            n_estimators=200, max_depth=5):
    if ml_calibration is None:
        raise RuntimeError("ml_calibration not available")
    return ml_calibration(
        v, calibration_mode=cal_mode, n_peaks=n_peaks,
        prominence=prominence, distance=distance,
        n_estimators=n_estimators, max_depth=max_depth, verbose=False,
    )


# ---------------------------------------------------------------------------
# Pipeline definitions
# ---------------------------------------------------------------------------
def define_pipelines(flight_path_mm, det_diam):
    """Return a list of (name, params_dict, run_func) for each benchmark case.

    Each run_func takes (variables, mode) and runs the full pipeline.
    """
    pipelines = []

    # ── Baseline ──
    pipelines.append(("baseline", {}, lambda v, m: None))

    # ── Initial only ──
    pipelines.append(("initial", {"init": "standard"},
        lambda v, m: (
            _run_initial_mc(v, det_diam) if m == "mc"
            else _run_initial_tof(v, flight_path_mm, det_diam)
        )))

    # ── Initial + Voltage + Bowl (1 pass) ──
    for v_model in ["curve_fit", "robust_fit"]:
        for b_fit in ["curve_fit", "robust_fit"]:
            for v_mode in ["ion_seq"]:
                for samp_mode in ["polar", "cartesian"]:
                    for samp_size in [5, 9]:
                        name = f"V+B_{v_model}_{b_fit}_{v_mode}_{samp_mode}_s{samp_size}"
                        params = {
                            "voltage_model": v_model, "bowl_fit": b_fit,
                            "voltage_mode": v_mode, "sampling_mode": samp_mode,
                            "sample_size": samp_size,
                        }
                        def _make_fn(vm=v_model, bf=b_fit, vmode=v_mode, sm=samp_mode, ss=samp_size):
                            def _run(v, m):
                                if m == "mc":
                                    _run_initial_mc(v, det_diam)
                                else:
                                    _run_initial_tof(v, flight_path_mm, det_diam)
                                select_peak(v, m)
                                _run_voltage(v, m, model=vm, mode=vmode)
                                select_peak(v, m)
                                _run_bowl(v, m, det_diam, fit_mode=bf, sampling_mode=sm, sample_size=ss)
                            return _run
                        pipelines.append((name, params, _make_fn()))

    # ── Initial + V+B x2 (two passes, matching user manual workflow) ──
    for v_model in ["curve_fit", "robust_fit"]:
        for b_fit in ["curve_fit", "robust_fit"]:
            name = f"V+Bx2_{v_model}_{b_fit}"
            params = {"voltage_model": v_model, "bowl_fit": b_fit, "passes": 2}
            def _make_fn2(vm=v_model, bf=b_fit):
                def _run(v, m):
                    if m == "mc":
                        _run_initial_mc(v, det_diam)
                    else:
                        _run_initial_tof(v, flight_path_mm, det_diam)
                    for _ in range(2):
                        select_peak(v, m)
                        _run_voltage(v, m, model=vm)
                        select_peak(v, m)
                        _run_bowl(v, m, det_diam, fit_mode=bf)
                return _run
            pipelines.append((name, params, _make_fn2()))

    # ── Initial + V+B + Residual ──
    for apply_spatial in [False, True]:
        for n_windows in [16, 24]:
            for max_rounds in [2, 8]:
                name = f"V+B+Res_sp{int(apply_spatial)}_w{n_windows}_r{max_rounds}"
                params = {"apply_spatial": apply_spatial, "n_windows": n_windows, "max_rounds": max_rounds}
                def _make_fn3(sp=apply_spatial, nw=n_windows, mr=max_rounds):
                    def _run(v, m):
                        if m == "mc":
                            _run_initial_mc(v, det_diam)
                        else:
                            _run_initial_tof(v, flight_path_mm, det_diam)
                        select_peak(v, m)
                        _run_voltage(v, m, model="robust_fit")
                        select_peak(v, m)
                        _run_bowl(v, m, det_diam, fit_mode="robust_fit")
                        _run_adaptive_residual(v, m, apply_spatial=sp, n_windows=nw, max_rounds=mr)
                    return _run
                pipelines.append((name, params, _make_fn3()))

    # ── Initial + V+Bx2 + Residual (user manual workflow) ──
    for apply_spatial in [False, True]:
        name = f"V+Bx2+Res_sp{int(apply_spatial)}"
        params = {"passes": 2, "apply_spatial": apply_spatial, "n_windows": 24, "max_rounds": 8}
        def _make_fn4(sp=apply_spatial):
            def _run(v, m):
                if m == "mc":
                    _run_initial_mc(v, det_diam)
                else:
                    _run_initial_tof(v, flight_path_mm, det_diam)
                for _ in range(2):
                    select_peak(v, m)
                    _run_voltage(v, m, model="robust_fit")
                    select_peak(v, m)
                    _run_bowl(v, m, det_diam, fit_mode="robust_fit")
                _run_adaptive_residual(v, m, apply_spatial=sp, n_windows=24, max_rounds=8)
            return _run
        pipelines.append((name, params, _make_fn4()))

    # ── Multi-peak ──
    for n_peaks in [3, 6]:
        name = f"multi_peak_n{n_peaks}"
        params = {"n_peaks": n_peaks}
        def _make_fn5(np_=n_peaks):
            def _run(v, m):
                if m == "mc":
                    _run_initial_mc(v, det_diam)
                else:
                    _run_initial_tof(v, flight_path_mm, det_diam)
                select_peak(v, m)
                _run_multi_peak(v, m, det_diam, n_peaks=np_)
            return _run
        pipelines.append((name, params, _make_fn5()))

    # ── ML Calibration ──
    if ml_calibration is not None:
        for n_est in [100, 200]:
            for md in [4, 5]:
                name = f"ML_n{n_est}_d{md}"
                params = {"n_estimators": n_est, "max_depth": md}
                def _make_fn6(ne=n_est, mdd=md):
                    def _run(v, m):
                        if m == "mc":
                            _run_initial_mc(v, det_diam)
                        else:
                            _run_initial_tof(v, flight_path_mm, det_diam)
                        select_peak(v, m)
                        _run_voltage(v, m, model="robust_fit")
                        select_peak(v, m)
                        _run_bowl(v, m, det_diam, fit_mode="robust_fit")
                        _run_ml(v, m, n_estimators=ne, max_depth=mdd)
                    return _run
                pipelines.append((name, params, _make_fn6()))

    return pipelines


# ---------------------------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------------------------
def run_benchmark(data, flight_path_mm, det_diam, modes=("mc", "tof"), max_pipelines=None):
    """Run all pipeline combinations and return a list of result dicts."""
    pipelines = define_pipelines(flight_path_mm, det_diam)
    if max_pipelines:
        pipelines = pipelines[:max_pipelines]

    results = []
    total = len(pipelines) * len(modes)
    for idx, ((name, params, run_fn), mode) in enumerate(product(pipelines, modes)):
        print(f"[{idx+1}/{total}] {mode}: {name} ...", end=" ", flush=True)
        v = fresh_variables(data)
        select_peak(v, mode)

        start = time.perf_counter()
        status = "ok"
        error = None
        try:
            with redirect_stdout(io.StringIO()):
                run_fn(v, mode)
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - start

        arr = v.dld_t_calib if mode == "tof" else v.mc_calib
        ev = evaluate_spectrum(arr, mode) if status == "ok" else {"score": None, "dominant_mrp50": None, "peaks": []}

        row = {
            "pipeline": name,
            "mode": mode,
            "params": params,
            "status": status,
            "error": error,
            "runtime_s": round(elapsed, 2),
            "score": ev["score"],
            "dominant_mrp50": ev["dominant_mrp50"],
            "n_peaks_found": len(ev["peaks"]),
            "peak_details": ev["peaks"],
        }
        results.append(row)
        score_str = f"score={ev['score']}" if ev['score'] else "N/A"
        print(f"{status} {score_str} ({elapsed:.1f}s)")

    return results


# ---------------------------------------------------------------------------
# Format results table
# ---------------------------------------------------------------------------
def format_results_table(results, top_n=30):
    """Format results as a readable text table, sorted by score descending."""
    by_mode = {}
    for r in results:
        by_mode.setdefault(r["mode"], []).append(r)

    lines = []
    for mode in sorted(by_mode.keys()):
        rows = by_mode[mode]
        ok_rows = [r for r in rows if r["status"] == "ok" and r["score"] is not None]
        ok_rows.sort(key=lambda r: -(r["score"] or 0))

        lines.append(f"\n{'='*100}")
        lines.append(f" MODE: {mode.upper()}  ({len(ok_rows)} successful / {len(rows)} total)")
        lines.append(f"{'='*100}")
        lines.append(f"{'Rank':<5} {'Pipeline':<50} {'Score':>8} {'MRP50':>8} {'Time(s)':>8} {'Peaks':>6}")
        lines.append("-" * 100)

        for i, r in enumerate(ok_rows[:top_n]):
            lines.append(
                f"{i+1:<5} {r['pipeline']:<50} {r['score']:>8.2f} "
                f"{r['dominant_mrp50'] or 0:>8.2f} {r['runtime_s']:>8.1f} {r['n_peaks_found']:>6}"
            )

        # Failed entries
        failed = [r for r in rows if r["status"] == "failed"]
        if failed:
            lines.append(f"\nFailed ({len(failed)}):")
            for r in failed[:5]:
                lines.append(f"  {r['pipeline']}: {r['error']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Comprehensive calibration parameter benchmark")
    parser.add_argument("dataset", help="Path to .h5 dataset file")
    parser.add_argument("--output-json", default="benchmark_params_results.json", help="Output JSON file")
    parser.add_argument("--output-table", default="benchmark_params_table.txt", help="Output text table")
    parser.add_argument("--max-pipelines", type=int, default=None, help="Limit number of pipelines (for quick tests)")
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}")
    data = pd.read_hdf(args.dataset)
    fpl = estimate_flight_path_mm(data)
    dd = estimate_det_diam_mm(data)
    print(f"Ions: {len(data):,}, flight path: {fpl:.1f} mm, detector diam: {dd:.1f} mm")

    results = run_benchmark(data, fpl, dd, max_pipelines=args.max_pipelines)

    # Save JSON
    with open(args.output_json, "w") as f:
        json.dump({"dataset": args.dataset, "n_ions": len(data), "flight_path_mm": fpl,
                    "det_diam_mm": dd, "results": results}, f, indent=2, default=str)
    print(f"\nJSON results saved to: {args.output_json}")

    # Format and save table
    table = format_results_table(results)
    with open(args.output_table, "w") as f:
        f.write(table)
    print(f"Table saved to: {args.output_table}")
    print(table)


if __name__ == "__main__":
    main()
