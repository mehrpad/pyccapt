"""Typed, versioned TOML experiment plans without GUI or hardware dependencies."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pyccapt.control.core.read_files import read_toml_file


class PlanError(ValueError):
    """An experiment plan has invalid structure or settings."""


STRING_FIELDS = ('ex_user', 'ex_name', 'electrode', 'email', 'pulse_mode',
                 'counter_source', 'control_algorithm')
INTEGER_FIELDS = ('ex_time', 'max_ions', 'ex_freq', 'vdc_min', 'vdc_max', 'vp_min',
                  'vp_max', 'pulse_fraction', 'pulse_frequency', 'hit_displayed')
NUMBER_FIELDS = ('vdc_steps_up', 'vdc_steps_down', 'detection_rate_init')
BOOL_FIELDS = ('criteria_time', 'criteria_ions', 'criteria_vdc')
REQUIRED_FIELDS = STRING_FIELDS + INTEGER_FIELDS + NUMBER_FIELDS + BOOL_FIELDS
OPTIONAL_DEFAULTS = {'criteria_email': False, 'email_interval_events': 1000000}
ALLOWED_FIELDS = set(REQUIRED_FIELDS) | set(OPTIONAL_DEFAULTS) | {'sample_id'}
CHOICES = {
    'pulse_mode': ('Voltage', 'Laser', 'VoltageLaser'),
    'counter_source': ('TDC', 'HSD'),
    'control_algorithm': ('Proportional', 'Proportional aggressive', 'Adaptive P', 'PID'),
}
UNITS = {
    'ex_time': 's', 'ex_freq': 'Hz', 'vdc_min': 'V', 'vdc_max': 'V',
    'vp_min': 'V', 'vp_max': 'V', 'vdc_steps_up': 'V/update',
    'vdc_steps_down': 'V/update', 'pulse_frequency': 'kHz',
    'pulse_fraction': '%', 'detection_rate_init': '%', 'max_ions': 'ions',
    'hit_displayed': 'hits', 'email_interval_events': 'ions',
}


def validate_item(item: Mapping[str, Any], label: str = 'Experiment') -> dict[str, Any]:
    """Reject unknown keys, implicit conversions, nonfinite and invalid values."""
    unknown = set(item) - ALLOWED_FIELDS
    missing = set(REQUIRED_FIELDS) - set(item)
    if unknown or missing:
        raise PlanError(f'{label}: unknown keys {sorted(unknown)}; missing keys {sorted(missing)}')
    resolved = {**OPTIONAL_DEFAULTS, **item}
    for key in STRING_FIELDS:
        if not isinstance(resolved[key], str):
            raise PlanError(f'{label}: {key} must be a string')
    for key in ('ex_user', 'ex_name', 'electrode'):
        if not resolved[key].strip():
            raise PlanError(f'{label}: {key} must not be empty')
    for key in INTEGER_FIELDS + ('email_interval_events', 'sample_id'):
        if key in resolved and type(resolved[key]) is not int:
            raise PlanError(f'{label}: {key} must be a whole number')
    for key in NUMBER_FIELDS:
        value = resolved[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise PlanError(f'{label}: {key} must be a finite number')
    for key in BOOL_FIELDS + ('criteria_email',):
        if type(resolved[key]) is not bool:
            raise PlanError(f'{label}: {key} must be true or false')
    for key, choices in CHOICES.items():
        if resolved[key] not in choices:
            raise PlanError(f'{label}: {key} must be one of {", ".join(choices)}')
    if 'sample_id' in resolved and resolved['sample_id'] not in (1, 2, 3):
        raise PlanError(f'{label}: sample_id must be 1, 2 or 3')
    for key in ('ex_time', 'max_ions', 'vdc_steps_up', 'vdc_steps_down'):
        if resolved[key] < 0:
            raise PlanError(f'{label}: {key} must not be negative')
    for key in ('ex_freq', 'hit_displayed', 'email_interval_events'):
        if resolved[key] <= 0:
            raise PlanError(f'{label}: {key} must be positive')
    if not any(resolved[key] for key in BOOL_FIELDS):
        raise PlanError(f'{label}: enable at least one stopping criterion')
    return resolved


def resolve_plan(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    if set(data) - {'schema_version', 'defaults', 'experiments'}:
        raise PlanError('Plan supports only schema_version, defaults and experiments')
    if type(data.get('schema_version')) is not int or data['schema_version'] != 1:
        raise PlanError('Plan requires schema_version = 1')
    defaults = data.get('defaults', {})
    if not isinstance(defaults, Mapping) or set(defaults) - (ALLOWED_FIELDS - {'sample_id'}):
        raise PlanError('Invalid defaults table (sample_id belongs to each experiment)')
    rows = data.get('experiments')
    if not isinstance(rows, list) or not rows:
        raise PlanError('Plan requires at least one [[experiments]] table')
    result = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, Mapping):
            raise PlanError(f'Experiment {index} must be a table')
        result.append(validate_item({**defaults, **row}, f'Experiment {index}'))
    return result


def load_plan(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() != '.toml':
        raise PlanError('Experiment plans must use a .toml file')
    try:
        return resolve_plan(read_toml_file(path))
    except (OSError, ValueError) as exc:
        raise PlanError(f'Cannot load plan {path.name}: {exc}') from exc


def dumps_plan(items: Sequence[Mapping[str, Any]]) -> str:
    """Export common settings as defaults; strings use TOML basic-string escapes."""
    rows = [validate_item(item, f'Experiment {i}') for i, item in enumerate(items, 1)]
    if not rows:
        raise PlanError('Plan requires at least one experiment')
    common = {key: value for key, value in rows[0].items()
              if key != 'sample_id' and all(row.get(key) == value for row in rows)}
    def assignments(values):
        return [f'{key} = {json.dumps(value, ensure_ascii=False)}' for key, value in values.items()]
    lines = ['# PyCCAPT experiment plan; pulse_frequency is in kHz.', 'schema_version = 1',
             '', '[defaults]', *assignments(common)]
    for row in rows:
        lines.extend(['', '[[experiments]]', *assignments({key: value for key, value in row.items()
                                                         if key not in common})])
    return '\n'.join(lines) + '\n'


def save_plan(path: str | Path, items: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    if path.suffix.lower() != '.toml':
        raise PlanError('Experiment plans must use a .toml file')
    path.write_text(dumps_plan(items), encoding='utf-8')


def alignment_samples(items: Sequence[Mapping[str, Any]], positions: Mapping) -> tuple[int, ...]:
    """Keep plan order and require a saved position for every explicit sample ID."""
    samples = []
    for index, item in enumerate(items, 1):
        sample = item.get('sample_id')
        if sample not in (1, 2, 3) or sample not in positions:
            raise PlanError(f'Experiment {index}: sample_id needs a saved Cameras coarse position')
        samples.append(sample)
    return tuple(samples)


def write_plan_snapshot(directory: str | Path, snapshot: Mapping[str, Any]) -> None:
    """Record the resolved row and its source before enabling experiment outputs."""
    if not snapshot:
        return
    directory = Path(directory)
    save_plan(directory / 'experiment_plan.toml', [snapshot['experiment']])
    (directory / 'experiment_plan_source.json').write_text(
        json.dumps({key: value for key, value in snapshot.items() if key != 'experiment'},
                   ensure_ascii=False, indent=2), encoding='utf-8')
