"""Sequential concentration profiles for ranged atom-probe events.

The profile is evaluated in fixed, non-overlapping acquisition-order windows.
Every ranged species contributes its stoichiometric atom count to the
denominator; only user-selected elements or ions are returned as curves.
Unranged events are excluded because their atomic identity is unknown.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class _RangeSpecies:
    index: int
    name: str
    lower: float
    upper: float
    elements: tuple[str, ...]
    counts: tuple[float, ...]

    @property
    def atom_count(self) -> float:
        return float(sum(self.counts))

    def element_count(self, symbol: str) -> float:
        return float(sum(count for element, count in zip(self.elements, self.counts) if element == symbol))


def _list_value(value) -> list:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") or text.startswith("("):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return [text]
            return _list_value(parsed)
        return [text]
    return [value]


def _range_species(range_data: pd.DataFrame) -> list[_RangeSpecies]:
    required = {"mc_low", "mc_up", "element", "complex"}
    missing = required.difference(range_data.columns)
    if missing:
        raise ValueError(f"Range table is missing required columns: {sorted(missing)}")

    species = []
    for position, (_, row) in enumerate(range_data.iterrows()):
        elements = tuple(str(value).strip() for value in _list_value(row["element"]))
        raw_counts = _list_value(row["complex"])
        try:
            counts = tuple(float(value) for value in raw_counts)
            lower = float(row["mc_low"])
            upper = float(row["mc_up"])
        except (TypeError, ValueError):
            continue
        if len(elements) != len(counts) or upper <= lower or sum(counts) <= 0:
            continue
        if any(element.lower() in {"un", "unranged", "unknown"} for element in elements):
            continue
        name = str(row.get("name", row.get("ion", f"range_{position}"))).strip()
        species.append(_RangeSpecies(position, name, lower, upper, elements, counts))
    return species


def profile_species_options(range_data: pd.DataFrame) -> list[tuple[str, str]]:
    """Return ``(display label, selector token)`` options for notebook widgets."""
    species = _range_species(range_data)
    elements = sorted({element for item in species for element in item.elements})
    options = [(f"Element: {element}", f"element:{element}") for element in elements]
    name_counts: dict[str, int] = {}
    for item in species:
        name_counts[item.name] = name_counts.get(item.name, 0) + 1
    for item in species:
        suffix = f" (range {item.index})" if name_counts[item.name] > 1 else ""
        options.append((f"Ion: {item.name}{suffix}", f"ion:{item.index}"))
    return options


def _resolve_selectors(selectors: Iterable[str], species: list[_RangeSpecies]):
    element_names = {element for item in species for element in item.elements}
    resolved = []
    used_labels = set()
    for raw in selectors:
        token = str(raw).strip()
        if token.startswith("element:"):
            symbol = token.split(":", 1)[1].strip()
            if symbol not in element_names:
                raise ValueError(f"Element {symbol!r} is not present in the range table")
            label, kind, value = f"{symbol} (element)", "element", symbol
        elif token.startswith("ion:"):
            try:
                index = int(token.split(":", 1)[1])
                item = next(item for item in species if item.index == index)
            except (ValueError, StopIteration) as exc:
                raise ValueError(f"Unknown ion selector {token!r}") from exc
            label, kind, value = f"{item.name} (ion)", "ion", index
        elif token in element_names:
            label, kind, value = f"{token} (element)", "element", token
        else:
            matches = [item for item in species if item.name == token]
            if not matches:
                raise ValueError(f"Selected material {token!r} is not present in the range table")
            if len(matches) > 1:
                raise ValueError(f"Ion name {token!r} occurs more than once; select a specific range")
            label, kind, value = f"{token} (ion)", "ion", matches[0].index
        if label not in used_labels:
            resolved.append((label, kind, value))
            used_labels.add(label)
    if not resolved:
        raise ValueError("Select at least one element or ion to plot")
    return resolved


def _assign_species(mc: np.ndarray, species: list[_RangeSpecies]) -> np.ndarray:
    """Return one species-list index per event, or -1 when it is unranged."""
    ordered = sorted(enumerate(species), key=lambda pair: pair[1].lower)
    non_overlapping = all(
        left.upper <= right.lower
        for (_, left), (_, right) in zip(ordered, ordered[1:])
    )
    assigned = np.full(mc.size, -1, dtype=np.int32)
    if non_overlapping:
        lower = np.asarray([item.lower for _, item in ordered], dtype=float)
        upper = np.asarray([item.upper for _, item in ordered], dtype=float)
        original_indices = np.asarray([index for index, _ in ordered], dtype=np.int32)
        candidates = np.searchsorted(lower, mc, side="right") - 1
        safe = np.clip(candidates, 0, len(ordered) - 1)
        valid = np.isfinite(mc) & (candidates >= 0) & (mc < upper[safe])
        assigned[valid] = original_indices[safe[valid]]
        return assigned

    # Preserve range-table priority for unusual overlapping ranges.
    unassigned = np.isfinite(mc)
    for species_index, item in enumerate(species):
        matched = unassigned & (mc >= item.lower) & (mc < item.upper)
        assigned[matched] = species_index
        unassigned[matched] = False
    return assigned


def calculate_concentration_profile(
    mc_values,
    range_data: pd.DataFrame,
    selected_species: Iterable[str],
    *,
    window_size: int = 50_000,
    include_partial_window: bool = True,
) -> pd.DataFrame:
    """Calculate atomic percentages in sequential acquisition-order windows.

    Parameters
    ----------
    mc_values:
        Calibrated mass-to-charge values in acquisition order.
    range_data:
        Ranging table containing ``mc_low``, ``mc_up``, ``element`` and
        stoichiometric ``complex`` columns.
    selected_species:
        Selector tokens returned by :func:`profile_species_options`, or plain
        element/range names for programmatic use.
    window_size:
        Number of detected events per non-overlapping window.
    include_partial_window:
        Include the final shorter window when the event count is not an exact
        multiple of ``window_size``.
    """
    try:
        window_size = int(window_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("Window length must be an integer") from exc
    if window_size <= 0:
        raise ValueError("Window length must be greater than zero")

    mc = np.asarray(mc_values, dtype=float).reshape(-1)
    species = _range_species(range_data)
    if not species:
        raise ValueError("The range table contains no usable ranged ions")
    selectors = _resolve_selectors(selected_species, species)
    assigned = _assign_species(mc, species)
    denominator_weights = np.asarray([item.atom_count for item in species], dtype=float)
    numerator_weights = {}
    for label, kind, value in selectors:
        if kind == "element":
            numerator_weights[label] = np.asarray(
                [item.element_count(value) for item in species], dtype=float
            )
        else:
            numerator_weights[label] = np.asarray(
                [item.atom_count if item.index == value else 0.0 for item in species],
                dtype=float,
            )

    # Whole-dataset values are attached as dataframe metadata for the plot
    # legend. "Other" uses the union of selected atoms, so selecting both an
    # element and one of its molecular ions never produces a negative value.
    selected_elements = {value for _, kind, value in selectors if kind == "element"}
    selected_ions = {value for _, kind, value in selectors if kind == "ion"}
    selected_union_weights = np.asarray(
        [
            item.atom_count
            if item.index in selected_ions
            else sum(item.element_count(element) for element in selected_elements)
            for item in species
        ],
        dtype=float,
    )
    all_ranged = assigned >= 0
    all_counts = np.bincount(assigned[all_ranged], minlength=len(species))
    overall_denominator = float(all_counts @ denominator_weights)
    overall_percentages = {
        label: (float(all_counts @ weights) / overall_denominator * 100.0)
        if overall_denominator else np.nan
        for label, weights in numerator_weights.items()
    }
    selected_union = float(all_counts @ selected_union_weights)
    other_percentage = (
        max(0.0, overall_denominator - selected_union) / overall_denominator * 100.0
        if overall_denominator else np.nan
    )

    rows = []
    for start in range(0, mc.size, window_size):
        stop = min(start + window_size, mc.size)
        if stop - start < window_size and not include_partial_window:
            break
        assigned_window = assigned[start:stop]
        ranged = assigned_window >= 0
        counts = np.bincount(assigned_window[ranged], minlength=len(species))
        denominator = float(counts @ denominator_weights)
        numerators = {
            label: float(counts @ weights)
            for label, weights in numerator_weights.items()
        }

        record = {
            "sequence": (start + 1 + stop) / 2.0,
            "sequence_start": start + 1,
            "sequence_end": stop,
            "detected_events": stop - start,
            "ranged_atoms": denominator,
        }
        for label, _, _ in selectors:
            record[label] = numerators[label] / denominator * 100.0 if denominator else np.nan
        rows.append(record)

    profile = pd.DataFrame(rows)
    profile.attrs["overall_percentages"] = overall_percentages
    profile.attrs["other_percentage"] = other_percentage
    profile.attrs["overall_ranged_atoms"] = overall_denominator
    return profile


def plot_concentration_profile(profile: pd.DataFrame, *, figure_size=(9.0, 5.0)):
    """Plot concentration-profile columns and return ``(figure, axis)``."""
    import matplotlib.pyplot as plt

    metadata = {"sequence", "sequence_start", "sequence_end", "detected_events", "ranged_atoms"}
    curves = [column for column in profile.columns if column not in metadata]
    if profile.empty or not curves:
        raise ValueError("The concentration profile has no windows or selected curves")
    fig, ax = plt.subplots(figsize=figure_size)
    for column in curves:
        overall = profile.attrs.get("overall_percentages", {}).get(column, np.nan)
        display_name = column.removesuffix(" (element)")
        label = f"{display_name}: {overall:.2f} at.%" if np.isfinite(overall) else display_name
        ax.plot(profile["sequence"], profile[column], marker="o", markersize=3, linewidth=1.5, label=label)
    other = profile.attrs.get("other_percentage", np.nan)
    if np.isfinite(other):
        # Legend-only entry: Other atoms are part of normalization but are not
        # plotted as a curve, matching the notebook control's description.
        ax.plot([], [], linestyle="none", marker="", label=f"Others: {other:.2f} at.%")
    ax.set_xlabel("Ion sequence")
    ax.set_ylabel("Concentration [at.%]")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


__all__ = ["calculate_concentration_profile", "plot_concentration_profile", "profile_species_options"]
