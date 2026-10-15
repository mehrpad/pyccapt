"""Non-destructive NeXus mapping targeting the NXapm v2026.01 conventions.

This is a labelled partial mapping, not a claim of complete NXapm conformance.
Missing instrument/coordinate metadata is reported, never fabricated. Existing
PyCCAPT groups remain readable alongside the new /entry representation.
"""
from __future__ import annotations

import hashlib

import h5py
import numpy as np

from pyccapt.calibration.core.dataset import CalibrationDataset, UNITS
from pyccapt.calibration.core.exceptions import CalibrationInputError
from pyccapt.calibration.data_tools.analysis_session import (
    _atomic_hdf, canonical_json, dataset_options, software_identity,
)


MAPPING_VERSION = 'pyccapt-nxapm-map-1'
NXAPM_VERSION = 'v2026.01'
NXAPM_REFERENCE = 'https://manual.nexusformat.org/classes/applications/NXapm.html'


def _group(parent, name, nx_class):
    group = parent.require_group(name)
    group.attrs['NX_class'] = nx_class
    return group


def _text(parent, name, value):
    return parent.create_dataset(name, data=str(value), dtype=h5py.string_dtype('utf-8'))


def _numeric(parent, name, values, units=None):
    array = np.asarray(values)
    dataset = parent.create_dataset(name, data=array, **dataset_options(array))
    if units is not None:
        dataset.attrs['units'] = units
    return dataset


def _program(group, identity):
    program = _group(group, 'program1', 'NXprogram')
    dataset = _text(program, 'program', 'PyCCAPT')
    dataset.attrs['version'] = identity['pyccapt_source_version']


def _measurement(parent, name, source, columns, units, scale=1.):
    """Map measured values without promoting importer placeholders to evidence."""
    values = np.column_stack([source.frame[column].to_numpy(dtype=float) for column in columns])
    status = np.column_stack([source.validity_status(column) for column in columns])
    values[status != 0] = np.nan
    values *= scale
    if len(columns) == 1:
        values, status = values[:, 0], status[:, 0]
    _numeric(parent, name, values, units)
    validity = _numeric(parent, name + '_status', status, '1')
    validity.attrs['codes'] = '0=measured;1=unknown/nonfinite;2=unavailable'


def write_nexus_mapping(handle, frame, *, metadata=None, ranges=None, reconstruction=None):
    metadata, reconstruction = dict(metadata or {}), dict(reconstruction or {})
    if 'entry' in handle:
        raise CalibrationInputError('Refusing to replace an existing NeXus entry')
    had_ids = 'event_id' in frame
    dataset = CalibrationDataset.from_frame(frame)
    frame = dataset.frame
    identity = software_identity()
    entry = _group(handle, 'entry', 'NXentry')
    entry.attrs.update(mapping_version=MAPPING_VERSION, application_definition_target='NXapm',
                       application_definition_version=NXAPM_VERSION,
                       application_definition_reference=NXAPM_REFERENCE,
                       conformance='partial mapping; full NXDL validation has not been performed')
    # Do not write definition=NXapm: it would assert an application definition
    # whose mandatory instrument/specimen/coordinate fields may be unavailable.
    for key in ('start_time', 'end_time', 'experiment_alias', 'experiment_description', 'operation_mode'):
        if key in metadata:
            _text(entry, key, metadata[key])
    collection = _group(entry, 'pyccapt_data', 'NXcollection')
    collection.attrs.update(source_id=dataset.source_id, source_hash=dataset.source_hash,
                            event_id_origin='source' if had_ids else 'assigned at export; full source order unknown')
    _numeric(collection, 'event_id', dataset.event_ids, '1')
    columns = _group(collection, 'columns', 'NXcollection')
    for index, column in enumerate(frame):
        values = frame[column].to_numpy()
        if values.dtype.kind not in 'biufc':
            continue
        child = _group(columns, f'column{index}', 'NXcollection')
        child.attrs['name'] = str(column)
        _numeric(child, 'value', values, UNITS.get(column))
        _numeric(child, 'status', dataset.validity_status(column), '1').attrs['codes'] = '0=measured;1=unknown/nonfinite;2=unavailable'
    _text(collection, 'field_availability_json', canonical_json(
        {column: 'available' if column in frame and column not in frame.attrs.get('pyccapt_unavailable_columns', [])
         else 'unavailable' for column in UNITS}))
    _text(collection, 'software_json', canonical_json(identity))
    _text(collection, 'experiment_metadata_json', canonical_json(metadata))
    instrument = _group(entry, 'atom_probe1', 'NXinstrument')
    if {'x_det (cm)', 'y_det (cm)'}.issubset(frame):
        hits = _group(instrument, 'hit_finding', 'NXprocess')
        _measurement(hits, 'hit_positions', dataset, ['x_det (cm)', 'y_det (cm)'], 'mm', scale=10.)
        _program(hits, identity)
    if 't (ns)' in frame:
        tof = _group(instrument, 'voltage_and_bowl', 'NXprocess')
        _measurement(tof, 'raw_tof', dataset, ['t (ns)'], 'ns')
        if 't_c (ns)' in frame:
            _measurement(tof, 'calibrated_tof', dataset, ['t_c (ns)'], 'ns')
        _program(tof, identity)
    if 'mc (Da)' in frame:
        mass = _group(instrument, 'mass_to_charge_conversion', 'NXprocess')
        _measurement(mass, 'mass_to_charge', dataset, ['mc (Da)'], 'Da')
        _program(mass, identity)
    if {'x (nm)', 'y (nm)', 'z (nm)'}.issubset(frame):
        recon = _group(instrument, 'reconstruction', 'NXapm_reconstruction')
        _measurement(recon, 'reconstructed_positions', dataset, ['x (nm)', 'y (nm)', 'z (nm)'], 'nm')
        _program(recon, identity)
        config = _group(recon, 'config', 'NXparameters')
        mapping = {'icf': ('image_compression', '1'), 'kf': ('kfactor', '1'),
                   'det_eff': ('efficiency', '1'), 'flight_path_length': ('flight_path', 'mm')}
        for name, (target, units) in mapping.items():
            if name in reconstruction:
                _numeric(config, target, float(reconstruction[name]), units)
        _text(config, 'pyccapt_parameters_json', canonical_json(reconstruction))
    if ranges is not None and len(ranges):
        ranging = _group(instrument, 'ranging', 'NXapm_ranging')
        _program(ranging, identity)
        # Preserve identities/compositions losslessly; nuclide hashes require
        # the pinned NeXus encoding and are not invented from internal row IDs.
        rows = ranges.to_dict(orient='records') if hasattr(ranges, 'to_dict') else ranges
        _text(ranging, 'pyccapt_ranges_json', canonical_json(rows))
        ranging.attrs['interval_policy'] = '[low,high); first valid accepted range owns an overlap'
    report = {'mapping_version': MAPPING_VERSION, 'target_version': NXAPM_VERSION,
              'conformance': 'partial', 'structural_mapping_valid': True,
              'unmapped_required_context': ['full NXapm instrument metadata', 'consistent coordinate/rotation systems',
                                           'specimen characterization', 'reconstruction voxel discretization, if present',
                                           'NeXus nuclide encoding for ranged ions, if present'],
              'missing_user_metadata': [key for key in ('start_time', 'operation_mode') if key not in metadata],
              'event_identity_preserved': had_ids,
              'acquisition_completeness': metadata.get('acquisition_completeness', 'unverified')}
    _text(entry, 'mapping_report_json', canonical_json(report))
    entry.attrs['mapping_report_sha256'] = hashlib.sha256(canonical_json(report).encode()).hexdigest()
    return report


def export_nexus(path, frame, *, metadata=None, ranges=None, reconstruction=None):
    return _atomic_hdf(path, lambda handle: write_nexus_mapping(
        handle, frame, metadata=metadata, ranges=ranges, reconstruction=reconstruction))
