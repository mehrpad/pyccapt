"""Versioned, checksummed HDF5 analysis sessions without executable serializers.

The workflow supplies JSON metadata and named numeric arrays. Every source,
selection mask, learned-model array and accepted output needed for resumption
belongs in that snapshot. Saving settings alone does not save a fitted model.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import subprocess
import tempfile

import h5py
import numpy as np

from pyccapt.calibration.core.exceptions import CalibrationInputError


SESSION_SCHEMA = 'pyccapt-session-2'
LEGACY_SCHEMA = 'pyccapt-session-1'


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Unsupported session metadata: {type(value).__name__}')


def canonical_json(value):
    try:
        return json.dumps(value, default=_json_default, allow_nan=False, sort_keys=True,
                          separators=(',', ':'), ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise CalibrationInputError(f'Session metadata must be finite JSON: {exc}') from exc


def software_identity():
    from pyccapt import __version__

    packages = {}
    for name in ('pyccapt', 'numpy', 'pandas', 'h5py', 'scikit-learn'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = 'unavailable'
    root = Path(__file__).resolve().parents[3]
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                           stderr=subprocess.DEVNULL, timeout=2, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root,
                                             stderr=subprocess.DEVNULL, timeout=2, text=True).strip())
    except (OSError, subprocess.SubprocessError):
        revision, dirty = 'unavailable', None
    return {'packages': packages, 'pyccapt_source_version': __version__,
            'git_revision': revision, 'working_tree_dirty': dirty}


def array_digest(values, block_rows=65536):
    """Hash dtype/shape and exact ordered bytes; usable on arrays or HDF datasets."""
    digest = hashlib.sha256()
    digest.update(canonical_json({'dtype': values.dtype.str, 'shape': values.shape}).encode('utf-8'))
    if not values.shape:
        digest.update(np.asarray(values[()] if isinstance(values, h5py.Dataset) else values).tobytes())
    else:
        for start in range(0, values.shape[0], block_rows):
            digest.update(np.ascontiguousarray(values[start:start+block_rows]).tobytes())
    return digest.hexdigest()


def dataset_options(array, *, compression='gzip', chunk_rows=65536):
    if compression not in (None, 'gzip', 'lzf') or not isinstance(chunk_rows, int) or chunk_rows <= 0:
        raise CalibrationInputError('compression must be None/gzip/lzf; chunk_rows must be positive')
    if array.ndim == 0 or array.size == 0:
        return {}
    row_bytes = max(1, array.dtype.itemsize * int(np.prod(array.shape[1:])))
    rows = min(len(array), chunk_rows, max(1, (1024 * 1024) // row_bytes))
    options = {'chunks': (rows, *array.shape[1:]), 'fletcher32': True}
    if compression:
        options.update(compression=compression, shuffle=True)
        if compression == 'gzip':
            options['compression_opts'] = 4
    return options


def write_session_group(handle, metadata, arrays, *, compression='gzip', chunk_rows=65536):
    """Write a snapshot into an already staged file; the caller owns atomic rename."""
    if not isinstance(metadata, dict) or not isinstance(arrays, dict):
        raise CalibrationInputError('Session metadata and arrays must be dictionaries')
    payload = canonical_json(metadata)
    normalized = {}
    for name, values in arrays.items():
        if not isinstance(name, str) or not name or len(name) > 512:
            raise CalibrationInputError('Array names must be nonempty strings of at most 512 characters')
        array = np.asarray(values)
        if array.dtype.kind not in 'biufc':
            raise CalibrationInputError(f'{name}: only numeric/bool arrays are stored; no objects or pickles')
        normalized[name] = array
    group = handle.create_group('analysis_session')
    group.attrs['schema_version'] = SESSION_SCHEMA
    group.attrs['created_utc'] = datetime.now(timezone.utc).isoformat()
    group.attrs['software_json'] = canonical_json(software_identity())
    group.attrs['metadata_sha256'] = hashlib.sha256(payload.encode('utf-8')).hexdigest()
    group.create_dataset('metadata_json', data=payload, dtype=h5py.string_dtype('utf-8'))
    data = group.create_group('arrays')
    manifest = []
    for index, (name, values) in enumerate(normalized.items()):
        key = f'a{index:06d}'
        dataset = data.create_dataset(key, data=values, **dataset_options(values, compression=compression,
                                                                        chunk_rows=chunk_rows))
        manifest.append({'name': name, 'key': key, 'dtype': values.dtype.str, 'shape': list(values.shape),
                         'sha256': array_digest(dataset)})
    manifest_json = canonical_json(manifest)
    group.create_dataset('manifest_json', data=manifest_json, dtype=h5py.string_dtype('utf-8'))
    group.attrs['manifest_sha256'] = hashlib.sha256(manifest_json.encode('utf-8')).hexdigest()
    group.attrs['status'] = 'complete'
    return group


def _embedded(parent, name, expected_type):
    if not isinstance(parent.get(name, getlink=True), h5py.HardLink):
        raise CalibrationInputError(f'Session {name} must be embedded, not an external or symbolic link')
    node = parent[name]
    if not isinstance(node, expected_type):
        raise CalibrationInputError(f'Session {name} has an invalid object type')
    if isinstance(node, h5py.Dataset) and (node.is_virtual or node.external):
        raise CalibrationInputError(f'Session {name} must embed its data, not virtual or external storage')
    return node


def _read_header(group):
    schema = group.attrs.get('schema_version', '')
    if schema not in (SESSION_SCHEMA, LEGACY_SCHEMA):
        raise CalibrationInputError(f'Unsupported analysis session schema: {schema}')
    if group.attrs.get('status') != 'complete':
        raise CalibrationInputError('Analysis session is not marked complete')
    for key in ('metadata_json', 'manifest_json'):
        _embedded(group, key, h5py.Dataset)
    _embedded(group, 'arrays', h5py.Group)
    payload = group['metadata_json'].asstr()[()]
    manifest_json = group['manifest_json'].asstr()[()]
    for payload_value, attr in ((payload, 'metadata_sha256'), (manifest_json, 'manifest_sha256')):
        # v1 had per-array and metadata hashes but no manifest checksum.
        if schema == LEGACY_SCHEMA and attr == 'manifest_sha256':
            continue
        if hashlib.sha256(payload_value.encode('utf-8')).hexdigest() != group.attrs.get(attr):
            raise CalibrationInputError(f'Analysis session {attr} mismatch')
    metadata, manifest = json.loads(payload), json.loads(manifest_json)
    if not isinstance(metadata, dict) or not isinstance(manifest, list):
        raise CalibrationInputError('Invalid session metadata/array manifest')
    names, keys = set(), set()
    for entry in manifest:
        if (not isinstance(entry, dict) or not isinstance(entry.get('name'), str) or
                entry['name'] in names or entry.get('key') in keys or
                not isinstance(entry.get('key'), str) or '/' in entry['key']):
            raise CalibrationInputError('Invalid or duplicate array manifest entry')
        names.add(entry['name'])
        keys.add(entry['key'])
        dataset = _embedded(group['arrays'], entry['key'], h5py.Dataset)
        if (not isinstance(dataset, h5py.Dataset) or dataset.dtype.kind not in 'biufc' or
                dataset.dtype.str != entry.get('dtype') or list(dataset.shape) != entry.get('shape')):
            raise CalibrationInputError(f"Array schema mismatch: {entry['name']}")
    if set(group['arrays']) != keys:
        raise CalibrationInputError('Session contains unmanifested arrays')
    return metadata, manifest, schema


def read_session_group(group, *, verify=True, max_bytes=None):
    metadata, manifest, schema = _read_header(group)
    total_bytes = sum(group['arrays'][entry['key']].size * group['arrays'][entry['key']].dtype.itemsize
                      for entry in manifest)
    if max_bytes is not None and total_bytes > max_bytes:
        raise CalibrationInputError(f'Session arrays require {total_bytes} bytes, exceeding the requested limit')
    arrays = {}
    for entry in manifest:
        dataset = group['arrays'][entry['key']]
        if verify and array_digest(dataset) != entry.get('sha256'):
            raise CalibrationInputError(f"Array checksum mismatch: {entry['name']}")
        arrays[entry['name']] = dataset[()]
    if schema == LEGACY_SCHEMA:
        metadata = dict(metadata)
        metadata.setdefault('storage_migrations', []).append({'from': LEGACY_SCHEMA, 'to': SESSION_SCHEMA,
                                                            'action': 'verified original arrays; added manifest checksum on next save'})
    return metadata, arrays


def validate_analysis_session(path):
    issues, count, byte_count = [], 0, 0
    try:
        with h5py.File(path, 'r') as handle:
            group = _embedded(handle, 'analysis_session', h5py.Group)
            _, manifest, schema = _read_header(group)
            for entry in manifest:
                dataset = group['arrays'][entry['key']]
                count += 1
                byte_count += dataset.size * dataset.dtype.itemsize
                if array_digest(dataset) != entry.get('sha256'):
                    issues.append(f"Array checksum mismatch: {entry['name']}")
    except (OSError, ValueError, TypeError, KeyError, CalibrationInputError) as exc:
        issues.append(str(exc))
        schema = ''
    return {'valid': not issues, 'schema_version': schema, 'arrays': count, 'array_bytes': byte_count,
            'issues': issues, 'scope': 'stored snapshot integrity; not scientific validity or acquisition completeness'}


def _atomic_hdf(path, writer):
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f'.{destination.name}.', suffix='.tmp', dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, 'w') as handle:
            writer(handle)
            handle.flush()
        # Windows FlushFileBuffers requires a writable descriptor.
        with temporary.open('rb+') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def save_analysis_session(path, metadata, arrays, *, compression='gzip', chunk_rows=65536):
    """Atomically replace a complete snapshot, leaving an existing file intact on failure."""
    def writer(handle):
        write_session_group(handle, metadata, arrays, compression=compression, chunk_rows=chunk_rows)
    return _atomic_hdf(path, writer)


def load_analysis_session(path, *, max_bytes=None):
    with h5py.File(path, 'r') as handle:
        if 'analysis_session' not in handle:
            raise CalibrationInputError('File has no resumable analysis session; a legacy receipt alone is insufficient')
        return read_session_group(_embedded(handle, 'analysis_session', h5py.Group), max_bytes=max_bytes)


def list_analysis_sessions(directory, query=''):
    """Read compact metadata only; bulk arrays are verified when a session is loaded."""
    result = []
    for path in sorted(Path(directory).expanduser().glob('*.h5')):
        try:
            with h5py.File(path, 'r') as handle:
                if 'analysis_session' not in handle:
                    continue
                group = _embedded(handle, 'analysis_session', h5py.Group)
                metadata, manifest, schema = _read_header(group)
                if query.casefold() not in (path.name + canonical_json(metadata)).casefold():
                    continue
                result.append({'path': str(path.resolve()), 'schema_version': schema,
                               'created_utc': group.attrs.get('created_utc', ''), 'metadata': metadata,
                               'arrays': len(manifest), 'integrity': 'header checked; arrays not yet verified'})
        except (OSError, ValueError, TypeError, KeyError, CalibrationInputError) as exc:
            result.append({'path': str(path.resolve()), 'integrity': 'invalid', 'error': str(exc)})
    return result
