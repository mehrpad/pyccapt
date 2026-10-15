"""Shared instrument-setting provenance; no device I/O or scientific defaults.

Acquisition settings, user-supplied conversion inputs and fitted models are
different evidence. Only explicit, valid settings are offered to downstream UIs.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math


SCHEMA = 'pyccapt-instrument-1'
ATTR = 'pyccapt_instrument_metadata'
UNITS = {'flight_path_length_mm': 'mm', 't0_ns': 'ns', 't0_voltage_ns': 'ns',
         't0_laser_ns': 'ns', 'detector_diameter_mm': 'mm', 'pulse_mode': '1',
         'instrument_name': '1', 'detector_model': '1'}
_TEXT = {'pulse_mode', 'instrument_name', 'detector_model'}


def normalize_pulse_mode(value):
    text = str(value or '').strip().lower().replace('_', '').replace('-', '')
    return {'laser': 'laser', 'voltage': 'voltage', 'voltagelaser': 'voltage_laser',
            'mixed': 'mixed'}.get(text)


def _record(name, value, source, status='configured'):
    valid = value is not None
    if name == 'pulse_mode':
        value = normalize_pulse_mode(value)
        valid = value is not None
    elif name in _TEXT:
        value = str(value).strip() if valid else None
        valid = bool(value)
    else:
        try:
            # TOML booleans are not physical quantities, even though Python's
            # float(True/False) would silently manufacture a 1/0 value.
            is_boolean = isinstance(value, bool)
            value = float(value)
            valid = not is_boolean and math.isfinite(value) and (value > 0 if name.endswith('_mm') else True)
        except (TypeError, ValueError, OverflowError):
            valid = False
    return {'value': value if valid else None, 'units': UNITS[name],
            'status': status if valid else 'unknown', 'source': str(source)}


def acquisition_metadata(config, *, pulse_mode=None, observed_at=None):
    """Snapshot configured constants and an observed run mode (never a GUI default)."""
    names = {'flight_path_length_mm': 'flight_path_length', 't0_voltage_ns': 't_0_voltage',
             't0_laser_ns': 't_0_laser', 'detector_diameter_mm': 'detector_diameter',
             'instrument_name': 'device_name', 'detector_model': 'tdc_model'}
    context = {name: _record(name, config.get(key), 'control.config.' + key) for name, key in names.items()}
    context['pulse_mode'] = _record('pulse_mode', pulse_mode, 'control run mode; not detector-event attribution', 'selected')
    context['pulse_mode_history'] = []
    metadata = {'schema_version': SCHEMA, 'acquisition': context}
    if pulse_mode is not None:
        metadata = record_mode_observation(metadata, pulse_mode, observed_at=observed_at)
    else:
        context['t0_ns'] = _record('t0_ns', None, 'No unambiguous recorded run mode')
    return metadata


def record_mode_observation(metadata, pulse_mode, *, observed_at=None, control_iteration=None):
    """Record selected control modes, without implying an exact ion-level boundary."""
    result = deepcopy(metadata)
    context = result['acquisition']
    mode = normalize_pulse_mode(pulse_mode)
    history = context.setdefault('pulse_mode_history', [])
    if not history or history[-1]['pulse_mode'] != mode:
        history.append({'pulse_mode': mode, 'observed_utc': observed_at,
                        'control_iteration': control_iteration,
                        'scope': 'control observation; transition is not assigned to individual detector events'})
    modes = {item['pulse_mode'] for item in history}
    single = next(iter(modes)) if len(modes) == 1 else 'mixed'
    context['pulse_mode'] = _record('pulse_mode', single, 'control run mode history', 'selected')
    key = {'voltage': 't0_voltage_ns', 'laser': 't0_laser_ns'}.get(single)
    context['t0_ns'] = (deepcopy(context[key]) if key else
                        _record('t0_ns', None, 'Mixed, combined or unknown pulse mode has no single offset'))
    return result


def record_calibration_settings(frame, *, flight_path_length_mm=None, t0_ns=None,
                                pulse_mode=None, source, status='user_supplied', context='calibration'):
    """Record a complete explicit input context on a dataframe; None stays unknown.

    ``context='conversion_preview'`` records a preview without claiming that it
    produced the saved calibrated mass. Model-fit offsets belong in receipts.
    """
    if context not in {'calibration', 'conversion_preview', 'uncorrected_conversion', 'legacy_workflow_inputs', 'import_inputs'}:
        raise ValueError('Unsupported calibration metadata context')
    metadata = deepcopy(frame.attrs.get(ATTR, {'schema_version': SCHEMA}))
    metadata[context] = {name: _record(name, value, source, status.get(name, 'user_supplied') if isinstance(status, dict) else status)
                         for name, value in (('flight_path_length_mm', flight_path_length_mm),
                                             ('t0_ns', t0_ns), ('pulse_mode', pulse_mode))}
    frame.attrs[ATTR] = metadata
    return deepcopy(metadata)


def extract_instrument_settings(frame, *, context='calibration'):
    """Return only known flight_path_length_mm/t0_ns/pulse_mode suggestions.

    A calibration context takes precedence as a whole; partial contexts never
    borrow a different mode's acquisition offset. Mixed modes require review.
    """
    metadata = frame.attrs.get(ATTR, {})
    values = metadata.get(context, metadata.get('acquisition', {}))
    result = {}
    for name in ('flight_path_length_mm', 't0_ns', 'pulse_mode'):
        record = values.get(name, {})
        if not isinstance(record, dict) or record.get('status', 'unknown') == 'unknown':
            continue
        if record.get('units') != UNITS[name]:
            continue
        checked = _record(name, record.get('value'), '')
        if checked['value'] is not None:
            result[name] = checked['value']
    if result.get('pulse_mode') not in {'laser', 'voltage'}:
        result.pop('pulse_mode', None)
        result.pop('t0_ns', None)
    return result


def write_instrument_metadata(parent, metadata, *, name='instrument_metadata'):
    """Write a versioned JSON contract and discoverable typed scalar settings."""
    import h5py
    group = parent.create_group(name)
    group.attrs.update(
        NX_class='NXcollection', schema_version=SCHEMA,
        nx_class_reference='https://manual.nexusformat.org/classes/base_classes/NXcollection.html',
        conformance='PyCCAPT metadata extension; NXcollection content is not NXDL validated; not full NXapm conformance')
    group.create_dataset('json', data=json.dumps(metadata, sort_keys=True, allow_nan=False),
                         dtype=h5py.string_dtype('utf-8'))
    for context_name in ('acquisition', 'calibration', 'conversion_preview', 'uncorrected_conversion', 'legacy_workflow_inputs', 'import_inputs'):
        if context_name not in metadata:
            continue
        context = group.create_group(context_name)
        context.attrs['NX_class'] = 'NXcollection'
        for field, record in metadata[context_name].items():
            if field not in UNITS or not isinstance(record, dict):
                continue
            value = record.get('value')
            if field in _TEXT:
                dataset = context.create_dataset(field, data='' if value is None else str(value),
                                                 dtype=h5py.string_dtype('utf-8'))
            else:
                dataset = context.create_dataset(field, data=float('nan') if value is None else value)
            dataset.attrs.update(units=record.get('units', UNITS[field]),
                                 status=record.get('status', 'unknown'), source=record.get('source', 'unknown'))
    return group


def read_instrument_metadata(path):
    """Read settings/provenance only, including older config-only acquisition files."""
    import h5py
    result = {}
    with h5py.File(path, 'r') as handle:
        for name in ('instrument_metadata/json', 'entry/atom_probe1/instrument_metadata/json'):
            if name in handle:
                result[ATTR] = json.loads(handle[name].asstr()[()])
                break
        if 'provenance' in handle:
            provenance = handle['provenance'].attrs
            if ATTR not in result and 'control_config_json' in provenance:
                # The GUI default is not evidence of the mode actually used.
                result[ATTR] = acquisition_metadata(json.loads(provenance['control_config_json']))
            result['pyccapt_acquisition_metadata'] = {
                name: str(provenance[name]) for name in
                ('experiment_name', 'created_utc', 'acquisition_completeness', 'control_config_sha256',
                 'control_first_observation_utc', 'control_last_observation_utc')
                if name in provenance}
    return result


def attach_instrument_metadata(frame, path):
    """Attach file metadata while retaining explicit processed-frame attributes."""
    for key, value in read_instrument_metadata(path).items():
        frame.attrs.setdefault(key, value)
    return frame
