"""Origami CLI parsing, with explicit units from 800-621-01 pp.119-125/139.

No bare e_mlp number is assumed to be mW. A returned unit is required.
"""
import math
import re
import time

NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
WAVELENGTHS = {0: ('IR', 1030.), 1: ('Green', 515.), 2: ('UV', 343.), 3: ('DUV', 257.5)}


def scalar(text):
    """Last standalone numeric value, never digits embedded in command names."""
    if not text or re.search(r'\b(?:error|invalid|failed|unknown)\b', text, re.I):
        return None
    lines = [s.strip().lstrip('>') for s in str(text).splitlines()]
    lines = [s for s in lines if s and not s.endswith('?') and s not in ('>', '?>')]
    matches = re.findall(r'(?<![\w.])('+NUMBER+r')(?![\w.])', '\n'.join(lines))
    if len(matches) != 1:
        return None
    value = float(matches[0])
    return value if math.isfinite(value) else None


def quantity(text):
    """Return (value, unit) for an unambiguous unit-bearing monitor reply."""
    if not text or re.search(r'\b(?:error|invalid|failed|unknown)\b', text, re.I):
        return None
    matches = re.findall(r'(?<![\w.])('+NUMBER+r')\s*(mW|uW|µW|μW|W|pJ|nJ|uJ|µJ|μJ|mJ|J)\b', text)
    if len(matches) != 1:
        return None
    value, unit = matches[0]
    value = float(value)
    return (value, unit.replace('µ', 'u').replace('μ', 'u')) if math.isfinite(value) and value >= 0 else None


def frequency_table(text):
    pairs = re.findall(r'e_freq\s*=\s*(\d+)\s*-+>\s*('+NUMBER+r')\s*Hz', text or '', re.I)
    return {int(index): float(hz) for index, hz in pairs if math.isfinite(float(hz)) and float(hz) > 0}


def wavelength(text):
    if not text or re.search(r'\b(?:error|invalid|failed|unknown|moving)\b', text, re.I):
        return None
    # Parse the position/value, not SHG-2P / THG / FHG in the device prefix.
    match = re.search(r'(?:position|wavelength)\s*[=: ]\s*(IR|GREEN|DUV|FHG|UV|THG|SHG|[0-3])\b', text, re.I)
    if not match:
        return None
    value = match[1].upper()
    return {'IR': 0, 'GREEN': 1, 'SHG': 1, 'UV': 2, 'THG': 2, 'DUV': 3, 'FHG': 3}.get(
        value, int(value) if value.isdigit() else None)


def optical_values(text, output_hz):
    """Return power mW and energy nJ from the monitor's explicit physical unit."""
    measured = quantity(text)
    if measured is None:
        return math.nan, math.nan
    value, unit = measured
    if unit.endswith('W'):
        mw = value * {'W': 1000., 'mW': 1., 'uW': .001}[unit]
        return mw, mw*1e6/output_hz if output_hz > 0 else math.nan
    nj = value * {'J': 1e9, 'mJ': 1e6, 'uJ': 1000., 'nJ': 1., 'pJ': .001}[unit]
    return nj*output_hz/1e6 if output_hz > 0 else math.nan, nj


def fresh_snapshot(variables, now=None):
    data = dict(getattr(variables, 'laser_telemetry', {}) or {})
    now = time.monotonic() if now is None else now
    if not data or not 0 <= now-data.get('monotonic', -math.inf) <= 10:
        return {}
    return data


def pulse_energy_pj(variables):
    """HDF5 per-event energy is pJ; unknown/stale laser telemetry is NaN."""
    if getattr(variables, 'pulse_mode', 'Voltage') == 'Voltage':
        return 0.
    data = fresh_snapshot(variables)
    return float(data.get('pulse_energy_nj', math.nan))*1000.


def experiment_frequency_hz(variables):
    if getattr(variables, 'pulse_mode', 'Voltage') == 'Laser':
        return float(fresh_snapshot(variables).get('output_frequency_hz', math.nan))
    return float(variables.pulse_frequency)*1000.


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


# Numeric per-control-iteration HDF5 telemetry fields and units.
TELEMETRY_UNITS = {
    'output_power_mw': 'mW', 'ir_power_mw': 'mW', 'pulse_energy_nj': 'nJ',
    'base_frequency_hz': 'Hz', 'output_frequency_hz': 'Hz', 'divider': '1',
    'wavelength_nm': 'nm', 'wavelength_index': '1', 'aom_percent': '%',
    'status_code': '1', 'valid': '1',
}
