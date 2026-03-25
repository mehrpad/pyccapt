"""Reusable helpers for instrument t0 and flight-path determination."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyccapt.calibration.mc import mc_tools

AMU = 1.66e-27
ELEMENTARY_CHARGE = 1.6e-19
VOLTAGE_ALPHA = 1.015
VOLTAGE_BETA = 0.7


def isotope_table_path() -> Path:
    """Return the default isotope table path bundled with PyCCAPT."""
    return Path(__file__).resolve().parents[2] / "files" / "isotopeTable.h5"


def load_isotope_table(table_path: str | Path | None = None) -> pd.DataFrame:
    """Load the isotope reference table used in calibration workflows."""
    path = isotope_table_path() if table_path is None else Path(table_path)
    dataframe = pd.read_hdf(path, mode="r")
    required_columns = {"element", "isotope", "weight", "abundance"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Isotope table is missing required columns: {sorted(missing_columns)}")
    return dataframe.copy()


def format_isotope_options(isotope_table: pd.DataFrame) -> tuple[list[str], dict[str, float]]:
    """Build combobox labels and a lookup table for isotope masses."""
    labels: list[str] = []
    lookup: dict[str, float] = {}
    for _, row in isotope_table.iterrows():
        label = (
            f"{row['element']}-{int(row['isotope'])} | "
            f"mass={float(row['weight']):.5f} | abundance={float(row['abundance']):.4f}"
        )
        labels.append(label)
        lookup[label] = float(row["weight"])
    return labels, lookup


def suggest_isotope_option(
    measured_mc: float,
    charge: int,
    isotope_table: pd.DataFrame,
) -> str:
    """Return the nearest isotope label for a measured peak and charge state."""
    charge_value = max(int(charge), 1)
    target_mass = float(measured_mc) * charge_value
    index = int((isotope_table["weight"].astype(float) - target_mass).abs().idxmin())
    row = isotope_table.loc[index]
    return (
        f"{row['element']}-{int(row['isotope'])} | "
        f"mass={float(row['weight']):.5f} | abundance={float(row['abundance']):.4f}"
    )


def reference_mass_from_selection(selection: str, charge: int, lookup: dict[str, float]) -> float | None:
    """Convert an isotope selection label into an ideal mass-to-charge value."""
    if not selection:
        return None
    mass = lookup.get(selection)
    if mass is None:
        return None
    charge_value = max(int(charge), 1)
    return float(mass) / charge_value


def configure_result_directories(variables, dataset_path: str | Path, subdirectory: str = "t_0_flight_path_distance") -> str:
    """Point workflow outputs to a dedicated t0/flight-path result directory."""
    dataset = Path(dataset_path)
    output_directory = dataset.parent / dataset.stem / subdirectory
    output_directory.mkdir(parents=True, exist_ok=True)
    variables.set_result_directory(output_directory)
    variables.set_result_data_directory(output_directory)
    variables.result_data_name = dataset.stem
    return str(output_directory)


def build_selected_peak_ranges(variables) -> list[dict[str, float | int]]:
    """Build measured peak ranges from the current interactive histogram selection."""
    if getattr(variables, "peak_widths", None) is None or len(getattr(variables, "peak_widths", [])) < 4:
        raise ValueError("No peak widths are available yet. Plot the histogram and detect peaks first.")
    if getattr(variables, "peak_x", None) is None or len(getattr(variables, "peak_x", [])) == 0:
        raise ValueError("No detected peak centers are available yet.")
    if not getattr(variables, "peaks_index_list", None):
        raise ValueError("No peaks are selected yet. Click the desired peaks in the histogram first.")

    peak_centers = np.asarray(variables.peak_x, dtype=float)
    x_hist = np.asarray(variables.x_hist, dtype=float)
    left_ips = np.asarray(variables.peak_widths[2], dtype=float)
    right_ips = np.asarray(variables.peak_widths[3], dtype=float)
    mc_values = np.asarray(variables.mc_uc, dtype=float)

    selected: list[dict[str, float | int]] = []
    for peak_index in sorted({int(index) for index in variables.peaks_index_list}):
        if peak_index < 0 or peak_index >= len(peak_centers):
            continue
        left_index = int(np.clip(round(left_ips[peak_index]), 0, len(x_hist) - 1))
        right_index = int(np.clip(round(right_ips[peak_index]), 0, len(x_hist) - 1))
        left = float(min(x_hist[left_index], x_hist[right_index]))
        right = float(max(x_hist[left_index], x_hist[right_index]))
        if right <= left:
            continue
        mask = np.logical_and(mc_values > left, mc_values < right)
        selected.append(
            {
                "peak_index": peak_index,
                "measured_mc": float(peak_centers[peak_index]),
                "left": left,
                "right": right,
                "num_ions": int(np.count_nonzero(mask)),
            }
        )

    if not selected:
        raise ValueError("The selected peaks do not define any valid ion windows.")
    return selected


def build_peak_regression_table(
    variables,
    selected_peaks: list[dict[str, float | int | str]],
    *,
    max_ions_per_peak: int = 5000,
    random_seed: int = 13,
) -> pd.DataFrame:
    """Expand selected peak windows into an ion-level regression table."""
    if max_ions_per_peak <= 0:
        raise ValueError("max_ions_per_peak must be positive")

    mc_values = np.asarray(variables.mc_uc, dtype=float)
    tof_values = np.asarray(variables.dld_t, dtype=float)
    x_det = np.asarray(variables.dld_x_det, dtype=float)
    y_det = np.asarray(variables.dld_y_det, dtype=float)
    high_voltage = np.asarray(variables.dld_high_voltage, dtype=float)
    pulse_values = np.asarray(getattr(variables, "dld_pulse_v", np.zeros_like(high_voltage)), dtype=float)
    rng = np.random.default_rng(random_seed)

    frames: list[pd.DataFrame] = []
    for peak_number, peak in enumerate(selected_peaks, start=1):
        left = float(peak["left"])
        right = float(peak["right"])
        ideal_mc = float(peak["ideal_mc"])
        label = str(peak.get("label") or f"Peak {peak_number}")
        measured_mc = float(peak["measured_mc"])

        indices = np.flatnonzero(np.logical_and(mc_values > left, mc_values < right))
        if indices.size == 0:
            continue
        if indices.size > max_ions_per_peak:
            indices = np.sort(rng.choice(indices, size=max_ions_per_peak, replace=False))

        frames.append(
            pd.DataFrame(
                {
                    "peak_label": label,
                    "peak_index": int(peak["peak_index"]),
                    "measured_mc": measured_mc,
                    "ideal_mc": ideal_mc,
                    "window_left": left,
                    "window_right": right,
                    "tof_ns": tof_values[indices],
                    "x_det_cm": x_det[indices],
                    "y_det_cm": y_det[indices],
                    "high_voltage_v": high_voltage[indices],
                    "pulse_v": pulse_values[indices],
                }
            )
        )

    if not frames:
        raise ValueError("No ions were found inside the selected peak windows.")
    return pd.concat(frames, ignore_index=True)


def filter_peak_regression_table(
    peak_table: pd.DataFrame,
    *,
    center_radius_cm: float = 0.5,
    voltage_tolerance_v: float = 0.0,
) -> pd.DataFrame:
    """Keep ions near the detector center and optionally near the median voltage."""
    if peak_table.empty:
        raise ValueError("peak_table is empty")

    filtered = peak_table.copy()
    filtered["detector_radius_cm"] = np.hypot(filtered["x_det_cm"], filtered["y_det_cm"])

    if center_radius_cm > 0:
        filtered = filtered[filtered["detector_radius_cm"] <= float(center_radius_cm)]

    if voltage_tolerance_v > 0:
        median_voltage = float(np.nanmedian(filtered["high_voltage_v"]))
        filtered = filtered[np.abs(filtered["high_voltage_v"] - median_voltage) <= float(voltage_tolerance_v)]

    filtered = filtered.reset_index(drop=True)
    if filtered.empty:
        raise ValueError("The current center-radius and voltage filters remove all selected ions.")
    return filtered


def compute_effective_voltage(high_voltage_v, pulse_v, pulse_mode: str) -> np.ndarray:
    """Return the effective accelerating voltage used in the mass conversion model."""
    high_voltage = np.asarray(high_voltage_v, dtype=float)
    pulse = np.asarray(pulse_v, dtype=float)
    if pulse_mode == "laser":
        effective_voltage = high_voltage
    elif pulse_mode == "voltage":
        effective_voltage = VOLTAGE_ALPHA * (high_voltage + VOLTAGE_BETA * pulse)
    else:
        raise ValueError(f"Unsupported pulse mode: {pulse_mode!r}")

    if np.any(~np.isfinite(effective_voltage)) or np.any(effective_voltage <= 0):
        raise ValueError("Effective accelerating voltage must stay finite and positive.")
    return effective_voltage


def compute_time_scaling_factor(ideal_mc, high_voltage_v, pulse_v, pulse_mode: str) -> np.ndarray:
    """Return the linearized factor so that `tof = t0 + L * factor`."""
    ideal_values = np.asarray(ideal_mc, dtype=float)
    if np.any(~np.isfinite(ideal_values)) or np.any(ideal_values <= 0):
        raise ValueError("Ideal mass-to-charge values must stay finite and positive.")
    effective_voltage = compute_effective_voltage(high_voltage_v, pulse_v, pulse_mode)
    return np.sqrt(ideal_values * AMU / (2.0 * ELEMENTARY_CHARGE * effective_voltage)) * 1.0e6


def estimate_fixed_path_t0(
    peak_table: pd.DataFrame,
    *,
    flight_path_length_mm: float,
    pulse_mode: str,
) -> dict[str, object]:
    """Estimate instrument t0 using a fixed flight-path length."""
    factor = compute_time_scaling_factor(
        peak_table["ideal_mc"],
        peak_table["high_voltage_v"],
        peak_table["pulse_v"],
        pulse_mode,
    )
    tof_ns = peak_table["tof_ns"].to_numpy(dtype=float)
    t0_values = tof_ns - float(flight_path_length_mm) * factor
    return {
        "flight_path_length_mm": float(flight_path_length_mm),
        "t0_ns": float(np.mean(t0_values)),
        "t0_std_ns": float(np.std(t0_values)),
        "num_ions": int(len(peak_table)),
        "factor": factor,
        "tof_ns": tof_ns,
        "t0_values_ns": t0_values,
    }


def fit_flight_path_and_t0(peak_table: pd.DataFrame, *, pulse_mode: str) -> dict[str, object]:
    """Fit both flight-path length and t0 by linear regression."""
    factor = compute_time_scaling_factor(
        peak_table["ideal_mc"],
        peak_table["high_voltage_v"],
        peak_table["pulse_v"],
        pulse_mode,
    )
    tof_ns = peak_table["tof_ns"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(factor, tof_ns, deg=1)
    predicted = intercept + slope * factor
    residuals = tof_ns - predicted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((tof_ns - np.mean(tof_ns)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {
        "flight_path_length_mm": float(slope),
        "t0_ns": float(intercept),
        "rmse_ns": float(np.sqrt(np.mean(residuals ** 2))),
        "r_squared": float(r_squared),
        "num_ions": int(len(peak_table)),
        "factor": factor,
        "tof_ns": tof_ns,
        "predicted_tof_ns": predicted,
        "residual_ns": residuals,
    }


def summarize_peak_table(peak_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize ion usage per selected peak."""
    summary = (
        peak_table.groupby(["peak_label", "measured_mc", "ideal_mc", "window_left", "window_right"], dropna=False)
        .size()
        .reset_index(name="num_ions")
        .sort_values("measured_mc")
        .reset_index(drop=True)
    )
    return summary


def plot_detector_selection(
    peak_table: pd.DataFrame,
    filtered_peak_table: pd.DataFrame,
    *,
    center_radius_cm: float,
) -> plt.Figure:
    """Plot detector hits before and after the center filter."""
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(
        peak_table["x_det_cm"],
        peak_table["y_det_cm"],
        s=8,
        alpha=0.1,
        color="#94a3b8",
        label="Selected peak ions",
    )
    ax.scatter(
        filtered_peak_table["x_det_cm"],
        filtered_peak_table["y_det_cm"],
        s=10,
        alpha=0.35,
        color="#2563eb",
        label="Used in t0 fit",
    )
    if center_radius_cm > 0:
        circle = plt.Circle((0.0, 0.0), center_radius_cm, fill=False, linestyle="--", color="#dc2626", linewidth=1.5)
        ax.add_patch(circle)
    ax.set_xlabel("x_det (cm)")
    ax.set_ylabel("y_det (cm)")
    ax.set_title("Detector-center selection")
    ax.legend(loc="upper right")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig


def plot_regression_results(
    fixed_result: dict[str, object],
    regression_result: dict[str, object],
) -> plt.Figure:
    """Plot the fixed-path and fitted linear models against the selected ions."""
    factor = np.asarray(regression_result["factor"], dtype=float)
    tof_ns = np.asarray(regression_result["tof_ns"], dtype=float)
    x_min = float(np.min(factor))
    x_max = float(np.max(factor))
    x_line = np.linspace(x_min, x_max, 200)

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.scatter(factor, tof_ns, s=10, alpha=0.18, color="#475569", label="Selected ions")
    ax.plot(
        x_line,
        float(fixed_result["t0_ns"]) + float(fixed_result["flight_path_length_mm"]) * x_line,
        linestyle="--",
        color="#dc2626",
        label="Fixed flight path",
    )
    ax.plot(
        x_line,
        float(regression_result["t0_ns"]) + float(regression_result["flight_path_length_mm"]) * x_line,
        linestyle="-",
        color="#2563eb",
        label="Fitted line",
    )
    ax.set_xlabel(r"$\sqrt{\frac{m/n}{2eV_\mathrm{eff}}}$ (ns/mm)")
    ax.set_ylabel("Time of flight (ns)")
    ax.set_title("Instrument t0 regression")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def preview_mass_spectrum_after_t0(
    variables,
    *,
    t0_ns: float,
    flight_path_length_mm: float,
    pulse_mode: str,
    max_mc: float = 120.0,
    bin_width: float = 0.1,
) -> plt.Figure:
    """Preview the recalculated mass spectrum after applying a candidate t0."""
    recalculated = mc_tools.tof2mc(
        np.asarray(variables.dld_t, dtype=float),
        float(t0_ns),
        np.asarray(variables.dld_high_voltage, dtype=float),
        np.asarray(variables.dld_x_det, dtype=float),
        np.asarray(variables.dld_y_det, dtype=float),
        float(flight_path_length_mm),
        np.asarray(getattr(variables, "dld_pulse_v", np.zeros_like(variables.dld_high_voltage)), dtype=float),
        mode=pulse_mode,
    )
    original = np.asarray(getattr(variables, "mc_uc", np.zeros_like(recalculated)), dtype=float)
    mask_original = np.logical_and(np.isfinite(original), original <= float(max_mc))
    mask_recalculated = np.logical_and(np.isfinite(recalculated), recalculated <= float(max_mc))
    if not np.any(mask_original) and not np.any(mask_recalculated):
        raise ValueError("No finite mass/charge values are available for the preview.")

    upper = float(max(np.max(original[mask_original]) if np.any(mask_original) else 0.0,
                      np.max(recalculated[mask_recalculated]) if np.any(mask_recalculated) else 0.0))
    upper = max(upper, bin_width)
    bins = np.arange(0.0, upper + bin_width, bin_width)

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    if np.any(mask_original):
        ax.hist(original[mask_original], bins=bins, histtype="step", linewidth=1.1, color="#94a3b8", label="Current mc")
    if np.any(mask_recalculated):
        ax.hist(
            recalculated[mask_recalculated],
            bins=bins,
            histtype="step",
            linewidth=1.1,
            color="#2563eb",
            label="Using fitted t0",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Mass-to-charge (Da)")
    ax.set_ylabel("Counts")
    ax.set_title("Mass-spectrum preview after t0 update")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


__all__ = [
    "build_peak_regression_table",
    "build_selected_peak_ranges",
    "compute_effective_voltage",
    "compute_time_scaling_factor",
    "configure_result_directories",
    "estimate_fixed_path_t0",
    "fit_flight_path_and_t0",
    "format_isotope_options",
    "isotope_table_path",
    "load_isotope_table",
    "plot_detector_selection",
    "plot_regression_results",
    "preview_mass_spectrum_after_t0",
    "reference_mass_from_selection",
    "suggest_isotope_option",
    "summarize_peak_table",
]
