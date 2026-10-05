"""Detector model name helpers for control runtime."""

from __future__ import annotations

from typing import Any

SURFACE_CONCEPT_MODELS = {"Surface_Concept", "Surface_Consept"}
ROENTDEK_MODEL = "RoentDek"
HSD_MODEL = "HSD"
SIMULATOR_MODEL = "Simulator"
SUPPORTED_TDC_MODELS = (*sorted(SURFACE_CONCEPT_MODELS), ROENTDEK_MODEL, HSD_MODEL, SIMULATOR_MODEL)
COUNTER_SOURCE_ALIASES = {
    "tdc": "TDC",
    "hsd": "HSD",
    "drs": "HSD",
    "digitizer": "HSD",
    "hsd / digitizer": "HSD",
}


def normalize_tdc_model(model: Any) -> str:
    """Return the canonical detector model name.

    ``Surface_Consept`` is kept as a legacy alias because existing
    config.toml files use that misspelling.
    """
    value = str(model or "").strip()
    if value in SURFACE_CONCEPT_MODELS:
        return "Surface_Concept"
    if value in {ROENTDEK_MODEL, HSD_MODEL, SIMULATOR_MODEL}:
        return value
    allowed = ", ".join(SUPPORTED_TDC_MODELS)
    raise ValueError(f"Unsupported tdc_model {value!r}. Supported values: {allowed}")


def normalize_counter_source(source: Any) -> str:
    """Normalize GUI/config aliases to the two runtime counter sources."""
    value = str(source or "").strip()
    normalized = COUNTER_SOURCE_ALIASES.get(value.lower())
    if normalized is None:
        raise ValueError(f"Unsupported counter source {value!r}. Supported values: TDC, HSD")
    return normalized
