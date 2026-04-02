from __future__ import annotations

import argparse
import io
import json
import math
import time
from contextlib import redirect_stdout
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from pyccapt.calibration.core import calibration
from pyccapt.calibration.core.adaptive_residual_calibration import adaptive_residual_calibration
from pyccapt.calibration.core.joint_tof_mc_calibration import joint_tof_mc_calibration
from pyccapt.calibration.core.mc_plot_peak_helpers import gaussian_mrp_report
from pyccapt.calibration.core.share_variables import Variables


E_CHARGE = 1.6e-19
AMU = 1.66e-27
ALPHA = 1.015
DEFAULT_PULSE_TIMING_FACTOR = 0.7

MAIN_PROMINENCE = 100
MAIN_DISTANCE = 500
MAIN_BIN_SIZE = 0.1

JOINT_PROMINENCE = 40
JOINT_DISTANCE_TOF = 100
JOINT_DISTANCE_MC = 5

ADAPTIVE_PROMINENCE = 100
ADAPTIVE_DISTANCE = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PyCCAPT calibration methods.")
    parser.add_argument("dataset", help="Absolute path to the dataset .h5 file")
    parser.add_argument("--flight-path-mm", type=float, default=None, help="Override nominal flight path in mm")
    parser.add_argument("--det-diam-mm", type=float, default=None, help="Override detector diameter in mm")
    parser.add_argument("--joint-t0", type=float, default=0.0, help="Joint-calibration t0 in ns")
    parser.add_argument("--output-json", type=str, default="", help="Optional path to save raw benchmark results as JSON")
    return parser.parse_args()


def estimate_flight_path_length_mm(data: pd.DataFrame) -> float:
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


def estimate_detector_diameter_mm(data: pd.DataFrame) -> float:
    if "x_det (cm)" not in data.columns or "y_det (cm)" not in data.columns:
        return 60.0
    x_mm = data["x_det (cm)"].to_numpy(dtype=float) * 10.0
    y_mm = data["y_det (cm)"].to_numpy(dtype=float) * 10.0
    radius = np.sqrt(x_mm ** 2 + y_mm ** 2)
    return float(np.percentile(radius[np.isfinite(radius)], 99.9) * 2.0)


def fresh_variables(data: pd.DataFrame) -> Variables:
    variables = Variables()
    variables.data = data
    variables.sync_from_data(data)
    return variables


def current_voltage(variables: Variables) -> np.ndarray:
    pulse = getattr(variables, "dld_pulse_v", np.zeros_like(variables.dld_high_voltage))
    return variables.dld_high_voltage + (DEFAULT_PULSE_TIMING_FACTOR * pulse)


def effective_flight_path_m(variables: Variables, flight_path_mm: float) -> np.ndarray:
    x_m = np.asarray(variables.dld_x_det, dtype=float) * 1e-2
    y_m = np.asarray(variables.dld_y_det, dtype=float) * 1e-2
    return np.sqrt(x_m ** 2 + y_m ** 2 + (float(flight_path_mm) * 1e-3) ** 2)


def tof_to_mc_from_calibrated_tof(variables: Variables, flight_path_mm: float) -> np.ndarray:
    eff_voltage = ALPHA * np.asarray(current_voltage(variables), dtype=float)
    path_m = effective_flight_path_m(variables, flight_path_mm)
    tof_s = np.asarray(variables.dld_t_calib, dtype=float) * 1e-9
    mass_kg_per_c = 2.0 * E_CHARGE * eff_voltage * (tof_s / path_m) ** 2
    return mass_kg_per_c / AMU


def capture_state(variables: Variables, mode: str) -> np.ndarray:
    return np.copy(variables.dld_t_calib if mode == "tof" else variables.mc_calib)


def restore_state(variables: Variables, mode: str, state: np.ndarray) -> None:
    if mode == "tof":
        variables.dld_t_calib = np.copy(state)
    else:
        variables.mc_calib = np.copy(state)


def current_array(variables: Variables, mode: str) -> np.ndarray:
    return variables.dld_t_calib if mode == "tof" else variables.mc_calib


def state_is_valid(state: np.ndarray) -> bool:
    arr = np.asarray(state, dtype=float)
    return arr.size > 0 and np.all(np.isfinite(arr)) and np.nanstd(arr) > 0


def selected_peak_entry(variables: Variables, mode: str) -> dict | None:
    if not (variables.selected_x2 > variables.selected_x1):
        return None
    data = current_array(variables, mode)
    mask = np.logical_and(data > variables.selected_x1, data < variables.selected_x2)
    n_ions = int(np.count_nonzero(mask))
    if n_ions == 0:
        return None
    return {
        "position": float((variables.selected_x1 + variables.selected_x2) * 0.5),
        "x1": float(variables.selected_x1),
        "x2": float(variables.selected_x2),
        "label": "selected",
        "n_ions": n_ions,
        "weight": max(1.0, float(np.sqrt(n_ions))),
    }


def peaks_overlap(first_peak: dict, second_peak: dict) -> bool:
    return not (first_peak["x2"] <= second_peak["x1"] or second_peak["x2"] <= first_peak["x1"])


def select_and_lock_peak(variables: Variables, mode: str, prominence: int = MAIN_PROMINENCE,
                         distance: int = MAIN_DISTANCE) -> dict:
    arr = current_array(variables, mode)
    hist_bin_size = 1.0 if mode == "tof" else 0.1
    peaks = calibration.auto_detect_reference_peaks(
        arr,
        n_peaks=1,
        prominence=prominence,
        distance=distance,
        hist_bin_size=hist_bin_size,
    )
    peak = peaks[0]
    variables.selected_x1 = float(peak["x1"])
    variables.selected_x2 = float(peak["x2"])
    mask = np.logical_and(arr > variables.selected_x1, arr < variables.selected_x2)
    if np.any(mask):
        variables.set_calibration_selection_mask(mode, mask)
    return peak


def peak_quality_score(arr: np.ndarray, peak: dict) -> tuple[float, dict | None]:
    local_bin_size = max(1e-4, min(0.02 if np.nanmax(arr) < 500 else 2.0, (peak["x2"] - peak["x1"]) / 80.0))
    report = gaussian_mrp_report(arr, peak["x1"], peak["x2"], bin_size=local_bin_size)
    if report is None:
        return float("nan"), None
    mrp_values = report["gaussian_mrp"] if report["gaussian_ok"] else report["histogram_mrp"]
    weights = [0.6, 0.3, 0.1]
    score = 0.0
    weight_sum = 0.0
    for weight, value in zip(weights, mrp_values):
        if np.isfinite(value):
            score += weight * float(value)
            weight_sum += weight
    if weight_sum == 0:
        return float("nan"), report
    return score / weight_sum, report


def collect_reference_peaks(variables: Variables, mode: str) -> dict:
    data = current_array(variables, mode)
    selected_peak = selected_peak_entry(variables, mode)
    auto_peaks = []
    try:
        auto_peaks = calibration.auto_detect_reference_peaks(
            data,
            n_peaks=6,
            prominence=MAIN_PROMINENCE,
            distance=MAIN_DISTANCE,
            hist_bin_size=max(0.01, MAIN_BIN_SIZE),
        )
    except Exception:
        auto_peaks = []

    reference_peaks = []
    if selected_peak is not None:
        reference_peaks.append(selected_peak)

    for peak in auto_peaks:
        candidate = dict(peak)
        if selected_peak is not None and peaks_overlap(selected_peak, candidate):
            continue
        peak_mask = np.logical_and(data > candidate["x1"], data < candidate["x2"])
        n_ions = int(np.count_nonzero(peak_mask))
        if n_ions < 25:
            continue
        candidate["label"] = f'auto@{candidate["position"]:.2f}'
        candidate["n_ions"] = n_ions
        candidate["weight"] = max(1.0, float(np.sqrt(n_ions)))
        reference_peaks.append(candidate)

    auto_only = [peak for peak in reference_peaks if peak.get("label") != "selected"]
    holdout_n = min(2, max(0, len(auto_only) // 3))
    if holdout_n > 0:
        train_auto = auto_only[:-holdout_n]
        holdout = auto_only[-holdout_n:]
    else:
        train_auto = auto_only
        holdout = []

    train = []
    if selected_peak is not None:
        train.append(selected_peak)
    train.extend(train_auto)
    if not train and holdout:
        train.append(holdout.pop(0))
    return {"train": train, "holdout": holdout}


def evaluate_peak_group(arr: np.ndarray, peaks: list[dict]) -> tuple[float, list[dict]]:
    weighted_scores = []
    details = []
    for peak in peaks:
        score, report = peak_quality_score(arr, peak)
        if not np.isfinite(score):
            continue
        weight = float(peak.get("weight", 1.0))
        weighted_scores.append((score, weight))
        details.append({
            "label": peak.get("label", f'{peak["position"]:.2f}'),
            "score": float(score),
            "weight": weight,
            "position": float(peak["position"]),
            "num_ions": int(peak.get("n_ions", report["num_ions"] if report is not None else 0)),
        })
    if not weighted_scores:
        return float("nan"), details
    total_weight = sum(weight for _, weight in weighted_scores)
    group_score = sum(score * weight for score, weight in weighted_scores) / total_weight
    return float(group_score), details


def evaluate_helper_quality(variables: Variables, mode: str, reference_peaks: dict) -> dict:
    arr = current_array(variables, mode)
    train_score, train_details = evaluate_peak_group(arr, reference_peaks["train"])
    holdout_score, holdout_details = evaluate_peak_group(arr, reference_peaks["holdout"])
    selected_score = float("nan")
    selected_peak = selected_peak_entry(variables, mode)
    if selected_peak is not None:
        selected_score, _ = peak_quality_score(arr, selected_peak)
    return {
        "train_score": train_score,
        "train_details": train_details,
        "holdout_score": holdout_score,
        "holdout_details": holdout_details,
        "selected_score": selected_score,
    }


def score_improved(candidate: float, best: float) -> bool:
    if not np.isfinite(candidate):
        return False
    if not np.isfinite(best):
        return True
    return candidate > best + max(0.1, abs(best) * 0.002)


def score_not_worse(candidate: float, best: float, tolerance_ratio: float = 0.01) -> bool:
    if not np.isfinite(best):
        return True
    if not np.isfinite(candidate):
        return False
    return candidate >= best - max(0.1, abs(best) * tolerance_ratio)


def optimize_sequence(variables: Variables, mode: str, action_specs: list[tuple[str, callable]],
                      max_iterations: int, max_no_improve: int) -> dict:
    reference_peaks = collect_reference_peaks(variables, mode)
    best_state = capture_state(variables, mode)
    best_selection = (float(variables.selected_x1), float(variables.selected_x2))
    initial_quality = evaluate_helper_quality(variables, mode, reference_peaks)
    best_train = initial_quality["train_score"]
    best_holdout = initial_quality["holdout_score"]
    best_selected = initial_quality["selected_score"]
    no_improve = 0
    history = []

    for iteration in range(1, max_iterations + 1):
        improved_this_round = False
        round_log = {"iteration": iteration, "actions": []}
        for action_name, action in action_specs:
            before_state = capture_state(variables, mode)
            before_selection = (float(variables.selected_x1), float(variables.selected_x2))
            try:
                with redirect_stdout(io.StringIO()):
                    action()
                candidate_state = capture_state(variables, mode)
                if not state_is_valid(candidate_state):
                    restore_state(variables, mode, before_state)
                    variables.selected_x1, variables.selected_x2 = before_selection
                    round_log["actions"].append({"name": action_name, "accepted": False, "reason": "invalid_state"})
                    continue
                quality = evaluate_helper_quality(variables, mode, reference_peaks)
                has_valid_signal = (
                    np.isfinite(quality["train_score"]) and quality["train_score"] > 0
                ) or (
                    np.isfinite(quality["selected_score"]) and quality["selected_score"] > 0
                )
                if not has_valid_signal:
                    restore_state(variables, mode, before_state)
                    variables.selected_x1, variables.selected_x2 = before_selection
                    round_log["actions"].append({"name": action_name, "accepted": False, "reason": "invalid_quality"})
                    continue

                train_improved = score_improved(quality["train_score"], best_train)
                selected_improved = score_improved(quality["selected_score"], best_selected)
                holdout_improved = score_improved(quality["holdout_score"], best_holdout)
                holdout_stable = score_not_worse(quality["holdout_score"], best_holdout, tolerance_ratio=0.01)
                train_stable = score_not_worse(quality["train_score"], best_train, tolerance_ratio=0.01)

                accepted = False
                if train_improved and holdout_stable:
                    accepted = True
                elif holdout_improved and train_stable:
                    accepted = True
                elif not np.isfinite(best_train) and selected_improved and holdout_stable:
                    accepted = True

                if accepted:
                    best_train = quality["train_score"]
                    best_holdout = quality["holdout_score"]
                    best_selected = quality["selected_score"]
                    best_state = capture_state(variables, mode)
                    best_selection = (float(variables.selected_x1), float(variables.selected_x2))
                    improved_this_round = True
                    round_log["actions"].append({"name": action_name, "accepted": True, "quality": quality})
                else:
                    restore_state(variables, mode, before_state)
                    variables.selected_x1, variables.selected_x2 = before_selection
                    round_log["actions"].append({"name": action_name, "accepted": False, "reason": "no_stable_improvement", "quality": quality})
            except Exception as exc:
                restore_state(variables, mode, before_state)
                variables.selected_x1, variables.selected_x2 = before_selection
                round_log["actions"].append({"name": action_name, "accepted": False, "reason": f"{type(exc).__name__}: {exc}"})
        history.append(round_log)
        if improved_this_round:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= max_no_improve:
                break

    restore_state(variables, mode, best_state)
    variables.selected_x1, variables.selected_x2 = best_selection
    return {
        "initial_quality": initial_quality,
        "final_quality": evaluate_helper_quality(variables, mode, reference_peaks),
        "history": history,
    }


def evaluate_spectrum(arr: np.ndarray, mode: str) -> dict:
    fallback_specs = (
        [(50, 50, 0.05), (30, 20, 0.05), (10, 10, 0.05)]
        if mode == "mc"
        else [(50, 50, 1.0), (30, 20, 1.0), (10, 10, 1.0)]
    )
    peaks = []
    for prominence, distance, hist_bin_size in fallback_specs:
        try:
            peaks = calibration.auto_detect_reference_peaks(
                arr,
                n_peaks=4,
                prominence=prominence,
                distance=distance,
                hist_bin_size=hist_bin_size,
            )
        except Exception:
            peaks = []
        if len(peaks) >= 2:
            break

    details = []
    weighted_scores = []
    dominant_mrp50 = float("nan")
    for idx, peak in enumerate(peaks):
        score, report = peak_quality_score(arr, peak)
        if not np.isfinite(score) or report is None:
            continue
        mrp_values = report["gaussian_mrp"] if report["gaussian_ok"] else report["histogram_mrp"]
        n_ions = int(report["num_ions"])
        weight = max(1.0, math.sqrt(n_ions))
        weighted_scores.append((score, weight))
        details.append({
            "position": float(peak["position"]),
            "window": [float(peak["x1"]), float(peak["x2"])],
            "n_ions": n_ions,
            "mrp": [float(x) if np.isfinite(x) else float("nan") for x in mrp_values],
            "gaussian_ok": bool(report["gaussian_ok"]),
        })
        if idx == 0 and np.isfinite(mrp_values[0]):
            dominant_mrp50 = float(mrp_values[0])

    score = float("nan")
    if weighted_scores:
        total_weight = sum(weight for _, weight in weighted_scores)
        score = float(sum(value * weight for value, weight in weighted_scores) / total_weight)
    return {"score": score, "dominant_mrp50": dominant_mrp50, "peaks": details}


def run_mc_initial(variables: Variables, det_diam: float) -> None:
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


def run_tof_initial(variables: Variables, flight_path_mm: float, det_diam: float) -> None:
    variables.dld_t_calib = calibration.initial_calibration(variables.data, flight_path_mm)
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
        calibration_mode="tof",
        index_fig=1,
        plot=False,
        save=False,
        fast_calibration=False,
        bin_size=0.01,
        peak_maximum=0,
        sampling_mode="polar",
    )


def run_voltage(variables: Variables, mode: str) -> None:
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


def run_bowl(variables: Variables, mode: str, det_diam: float) -> None:
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


def run_multi_peak(variables: Variables, mode: str, det_diam: float) -> None:
    calibration.multi_peak_voltage_corr_main(
        current_voltage(variables),
        variables,
        calibration_mode=mode,
        model="robust_fit",
        bin_size=0.01,
        n_peaks=3,
        prominence=MAIN_PROMINENCE,
        distance=MAIN_DISTANCE,
    )
    calibration.multi_peak_bowl_corr_main(
        variables.dld_x_det,
        variables.dld_y_det,
        current_voltage(variables),
        variables,
        det_diam,
        calibration_mode=mode,
        fit_mode="robust_fit",
        sample_size=5,
        bin_size=0.01,
        n_peaks=3,
        prominence=MAIN_PROMINENCE,
        distance=MAIN_DISTANCE,
        sampling_mode="polar",
    )


def run_joint_refinement(variables: Variables, mode: str, det_diam: float) -> None:
    calibration.joint_voltage_bowl_corr_main(
        variables.dld_x_det,
        variables.dld_y_det,
        current_voltage(variables),
        variables,
        det_diam,
        calibration_mode=mode,
        sample_size=5,
        bin_size=0.01,
        n_peaks=4,
        prominence=MAIN_PROMINENCE,
        distance=MAIN_DISTANCE,
        sampling_mode="polar",
    )


def benchmark_method(data: pd.DataFrame, name: str, mode: str, flight_path_mm: float,
                     det_diam: float, joint_t0: float) -> dict:
    variables = fresh_variables(data)
    if mode in {"mc", "tof"}:
        select_and_lock_peak(variables, mode)
    elif mode == "joint":
        select_and_lock_peak(variables, "mc")
        select_and_lock_peak(variables, "tof")

    start = time.perf_counter()
    extra = {}
    try:
        if name == "baseline_mc":
            pass
        elif name == "baseline_tof":
            pass
        elif name == "mc_initial":
            with redirect_stdout(io.StringIO()):
                run_mc_initial(variables, det_diam)
        elif name == "mc_voltage":
            with redirect_stdout(io.StringIO()):
                run_voltage(variables, "mc")
        elif name == "mc_voltage_bowl":
            with redirect_stdout(io.StringIO()):
                run_voltage(variables, "mc")
                run_bowl(variables, "mc", det_diam)
        elif name == "mc_auto_bowl":
            extra = optimize_sequence(
                variables, "mc",
                [("Bowl correction", lambda: run_bowl(variables, "mc", det_diam))],
                max_iterations=10,
                max_no_improve=3,
            )
        elif name == "mc_auto_calibration":
            extra = optimize_sequence(
                variables, "mc",
                [
                    ("Voltage correction", lambda: run_voltage(variables, "mc")),
                    ("Bowl correction", lambda: run_bowl(variables, "mc", det_diam)),
                    ("Joint voltage+bowl refinement", lambda: run_joint_refinement(variables, "mc", det_diam)),
                ],
                max_iterations=10,
                max_no_improve=3,
            )
        elif name == "mc_auto_multi_peak":
            extra = optimize_sequence(
                variables, "mc",
                [("Auto multi-peak calibration", lambda: run_multi_peak(variables, "mc", det_diam))],
                max_iterations=10,
                max_no_improve=3,
            )
        elif name == "mc_auto_optimize":
            extra = optimize_sequence(
                variables, "mc",
                [
                    ("Voltage correction", lambda: run_voltage(variables, "mc")),
                    ("Bowl correction", lambda: run_bowl(variables, "mc", det_diam)),
                    ("Auto multi-peak calibration", lambda: run_multi_peak(variables, "mc", det_diam)),
                    ("Joint voltage+bowl refinement", lambda: run_joint_refinement(variables, "mc", det_diam)),
                ],
                max_iterations=20,
                max_no_improve=3,
            )
        elif name == "mc_adaptive_residual":
            with redirect_stdout(io.StringIO()):
                run_mc_initial(variables, det_diam)
                extra = adaptive_residual_calibration(
                    variables,
                    calibration_mode="mc",
                    n_peaks=6,
                    prominence=ADAPTIVE_PROMINENCE,
                    distance=ADAPTIVE_DISTANCE,
                    n_windows=24,
                    overlap=0.5,
                    template_bin_size=0.01,
                    temporal_smoothing=0.5,
                    apply_spatial=True,
                    spatial_grid=12,
                    min_window_ions=40,
                    min_cell_ions=35,
                    verbose=False,
                )
        elif name == "tof_initial":
            with redirect_stdout(io.StringIO()):
                run_tof_initial(variables, flight_path_mm, det_diam)
        elif name == "tof_voltage":
            with redirect_stdout(io.StringIO()):
                run_voltage(variables, "tof")
        elif name == "tof_voltage_bowl":
            with redirect_stdout(io.StringIO()):
                run_voltage(variables, "tof")
                run_bowl(variables, "tof", det_diam)
        elif name == "tof_auto_bowl":
            extra = optimize_sequence(
                variables, "tof",
                [("Bowl correction", lambda: run_bowl(variables, "tof", det_diam))],
                max_iterations=10,
                max_no_improve=3,
            )
        elif name == "tof_auto_calibration":
            extra = optimize_sequence(
                variables, "tof",
                [
                    ("Voltage correction", lambda: run_voltage(variables, "tof")),
                    ("Bowl correction", lambda: run_bowl(variables, "tof", det_diam)),
                    ("Joint voltage+bowl refinement", lambda: run_joint_refinement(variables, "tof", det_diam)),
                ],
                max_iterations=10,
                max_no_improve=3,
            )
        elif name == "tof_auto_multi_peak":
            extra = optimize_sequence(
                variables, "tof",
                [("Auto multi-peak calibration", lambda: run_multi_peak(variables, "tof", det_diam))],
                max_iterations=10,
                max_no_improve=3,
            )
        elif name == "tof_auto_optimize":
            extra = optimize_sequence(
                variables, "tof",
                [
                    ("Voltage correction", lambda: run_voltage(variables, "tof")),
                    ("Bowl correction", lambda: run_bowl(variables, "tof", det_diam)),
                    ("Auto multi-peak calibration", lambda: run_multi_peak(variables, "tof", det_diam)),
                    ("Joint voltage+bowl refinement", lambda: run_joint_refinement(variables, "tof", det_diam)),
                ],
                max_iterations=20,
                max_no_improve=3,
            )
        elif name == "tof_adaptive_residual":
            with redirect_stdout(io.StringIO()):
                run_tof_initial(variables, flight_path_mm, det_diam)
                extra = adaptive_residual_calibration(
                    variables,
                    calibration_mode="tof",
                    n_peaks=6,
                    prominence=ADAPTIVE_PROMINENCE,
                    distance=ADAPTIVE_DISTANCE,
                    n_windows=24,
                    overlap=0.5,
                    template_bin_size=0.01,
                    temporal_smoothing=0.5,
                    apply_spatial=True,
                    spatial_grid=12,
                    min_window_ions=40,
                    min_cell_ions=35,
                    verbose=False,
                )
        elif name == "joint_after_initial":
            with redirect_stdout(io.StringIO()):
                run_mc_initial(variables, det_diam)
                run_tof_initial(variables, flight_path_mm, det_diam)
                extra = joint_tof_mc_calibration(
                    variables,
                    flight_path_length=flight_path_mm,
                    t0=joint_t0,
                    det_diam=det_diam,
                    pulse_mode="voltage",
                    n_peaks=6,
                    prominence=JOINT_PROMINENCE,
                    distance_tof=JOINT_DISTANCE_TOF,
                    distance_mc=JOINT_DISTANCE_MC,
                    bin_size_mc=0.1,
                    bin_size_tof=1.0,
                    max_iterations=10,
                    convergence_tol=1e-4,
                    tof_weight=0.7,
                    mc_weight=0.3,
                    verbose=False,
                )
        else:
            raise ValueError(f"Unknown benchmark method: {name}")

        elapsed = time.perf_counter() - start
        result = {
            "name": name,
            "mode": mode,
            "runtime_s": elapsed,
            "status": "ok",
            "extra": extra,
        }
        if mode == "mc":
            result["mc_eval"] = evaluate_spectrum(variables.mc_calib, "mc")
        elif mode == "tof":
            result["tof_eval"] = evaluate_spectrum(variables.dld_t_calib, "tof")
            result["derived_mc_eval"] = evaluate_spectrum(tof_to_mc_from_calibrated_tof(variables, flight_path_mm), "mc")
        else:
            result["tof_eval"] = evaluate_spectrum(variables.dld_t_calib, "tof")
            result["mc_eval"] = evaluate_spectrum(variables.mc_calib, "mc")
        return result
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return {
            "name": name,
            "mode": mode,
            "runtime_s": elapsed,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def main() -> None:
    args = parse_args()
    data = pd.read_hdf(args.dataset)
    flight_path_mm = float(args.flight_path_mm or estimate_flight_path_length_mm(data))
    det_diam_mm = float(args.det_diam_mm or estimate_detector_diameter_mm(data))

    methods = [
        ("baseline_mc", "mc"),
        ("mc_initial", "mc"),
        ("mc_voltage", "mc"),
        ("mc_voltage_bowl", "mc"),
        ("mc_auto_bowl", "mc"),
        ("mc_auto_calibration", "mc"),
        ("mc_auto_multi_peak", "mc"),
        ("mc_auto_optimize", "mc"),
        ("mc_adaptive_residual", "mc"),
        ("baseline_tof", "tof"),
        ("tof_initial", "tof"),
        ("tof_voltage", "tof"),
        ("tof_voltage_bowl", "tof"),
        ("tof_auto_bowl", "tof"),
        ("tof_auto_calibration", "tof"),
        ("tof_auto_multi_peak", "tof"),
        ("tof_auto_optimize", "tof"),
        ("tof_adaptive_residual", "tof"),
        ("joint_after_initial", "joint"),
    ]

    results = {
        "dataset": args.dataset,
        "n_ions": int(len(data)),
        "flight_path_mm": flight_path_mm,
        "det_diam_mm": det_diam_mm,
        "joint_t0_ns": float(args.joint_t0),
        "results": [],
    }

    for method_name, mode in methods:
        print(f"Running {method_name}...")
        result = benchmark_method(
            data,
            name=method_name,
            mode=mode,
            flight_path_mm=flight_path_mm,
            det_diam=det_diam_mm,
            joint_t0=args.joint_t0,
        )
        results["results"].append(result)
        print(json.dumps({
            "name": result["name"],
            "status": result["status"],
            "runtime_s": round(result["runtime_s"], 2),
            "mc_score": result.get("mc_eval", {}).get("score"),
            "tof_score": result.get("tof_eval", {}).get("score"),
            "derived_mc_score": result.get("derived_mc_eval", {}).get("score"),
            "error": result.get("error"),
        }, indent=2))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
