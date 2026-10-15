"""Source-field availability is independent of a stored numerical zero."""
from copy import deepcopy
import io
import json

import h5py
import numpy as np
import pandas as pd
import pytest

from pyccapt.calibration.core.dataset import CalibrationDataset
from pyccapt.calibration.data_tools import data_tools
from pyccapt.calibration.data_tools.nexus_export import write_nexus_mapping
from pyccapt.calibration.leap_tools import ccapt_tools


EPOS_MISSING = {'mc_uc (Da)', 'pulse_l (pJ)', 't_c (ns)', 'start_counter'}


def _epos_source(n=3):
    frame = pd.DataFrame({name: np.zeros(n, dtype=np.float32) for name in
        ('x (nm)', 'y (nm)', 'z (nm)', 'm/n (Da)', 'HV_DC (V)', 'pulse (V)',
         'TOF (ns)', 'det_x (mm)', 'det_y (mm)')})
    frame['pslep'] = np.zeros(n, dtype=np.uint32)
    frame['ipp'] = np.ones(n, dtype=np.uint32)
    return frame


class _LazyEPOS:
    def __init__(self, frame):
        self.frame = frame
        self.attrs = deepcopy(frame.attrs)
        self.n_rows = len(frame)

    def __getitem__(self, name):
        return self.frame[name].to_numpy()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def test_pos_tags_absent_fields_while_preserving_measured_zero_and_source_attrs(monkeypatch):
    raw = pd.DataFrame({'x (nm)': [0., 1.], 'y (nm)': [0., 2.],
                        'z (nm)': [0., 3.], 'm/n (Da)': [0., 28.]})
    raw.attrs = {'pyccapt_source_id': 'original', 'instrument_notes': ['provided']}
    monkeypatch.setattr(ccapt_tools.leap_tools, 'read_pos', lambda _: raw)
    result = ccapt_tools.pos_to_ccapt('dummy.pos')
    present = {'x (nm)', 'y (nm)', 'z (nm)', 'mc (Da)'}
    assert set(result.attrs['pyccapt_unavailable_columns']) == set(result).difference(present)
    assert result.attrs['pyccapt_source_id'] == 'original'
    result.attrs['instrument_notes'].append('local')
    assert raw.attrs['instrument_notes'] == ['provided']
    dataset = CalibrationDataset.from_frame(result)
    for name in present:
        assert dataset.validity_status(name).tolist() == [0, 0]
    assert dataset.validity_status('t (ns)').tolist() == [2, 2]
    assert not dataset.capabilities['time_of_flight']
    assert not dataset.capabilities['detector_positions']
    assert dataset.capabilities['reconstruction']
    assert result['t (ns)'].dtype == np.dtype('float32')
    np.testing.assert_array_equal(result['t (ns)'], [0., 0.])


def test_epos_eager_and_lazy_tag_same_placeholders_without_changing_values(monkeypatch):
    raw = _epos_source()
    raw.loc[1, 'TOF (ns)'] = np.nan
    raw.loc[2, 'det_x (mm)'] = 20.
    raw.attrs['pyccapt_source_id'] = 'epos-source'
    monkeypatch.setattr(ccapt_tools.leap_tools, 'read_epos', lambda _: raw)
    eager = ccapt_tools.epos_to_ccapt('dummy.epos')
    chunks = list(ccapt_tools.epos_lazy_to_ccapt_chunks(_LazyEPOS(raw), chunk_size=2))
    assert all(set(chunk.attrs['pyccapt_unavailable_columns']) == EPOS_MISSING for chunk in chunks)
    pd.testing.assert_frame_equal(pd.concat(chunks, ignore_index=True), eager)
    dataset = CalibrationDataset.from_frame(eager)
    assert dataset.validity_status('t (ns)').tolist() == [0, 1, 0]
    assert dataset.validity_status('pulse_v (V)').tolist() == [0, 0, 0]
    assert dataset.validity_status('pulse_l (pJ)').tolist() == [2, 2, 2]
    assert eager['x_det (cm)'].tolist() == [0., 0., 2.]


def test_inherited_source_unavailability_is_mapped_across_epos_column_names(monkeypatch):
    raw = _epos_source(1)
    raw.attrs['pyccapt_unavailable_columns'] = ['TOF (ns)', 'HV_DC (V)']
    monkeypatch.setattr(ccapt_tools.leap_tools, 'read_epos', lambda _: raw)
    result = CalibrationDataset.from_frame(ccapt_tools.epos_to_ccapt('dummy.epos'))
    assert result.validity_status('t (ns)').tolist() == [2]
    assert result.validity_status('high_voltage (V)').tolist() == [2]
    assert result.validity_status('pulse_v (V)').tolist() == [0]


@pytest.mark.parametrize('count', [0, 3])
def test_streamed_epos_hdf_persists_availability_for_processed_loader(monkeypatch, tmp_path, count):
    raw = _epos_source(count)
    raw.attrs['pyccapt_source_id'] = 'stream-source'
    monkeypatch.setattr(ccapt_tools.leap_tools, 'read_epos_lazy', lambda _: _LazyEPOS(raw))
    path = tmp_path/'stream.h5'
    result = ccapt_tools.epos_to_ccapt_h5_streaming('dummy.epos', path, chunk_size=2)
    assert result['rows'] == count
    loaded = data_tools.load_data(path, 'pyccapt', mode='processed')
    assert len(loaded) == count
    assert loaded.attrs['pyccapt_source_id'] == 'stream-source'
    assert set(loaded.attrs['pyccapt_unavailable_columns']) == EPOS_MISSING
    assert 't (ns)' not in loaded.attrs['pyccapt_unavailable_columns']


def test_apt_fallbacks_tag_only_absent_source_fields_and_preserve_tipbox(monkeypatch):
    raw = pd.DataFrame({'Mass': [0., 28.], 'xs': [0., 1.], 'zs': [0., 2.],
                        'Vref': [0., 5000.], 'tof': [0., np.nan], 'XDet_mm': [0., 20.],
                        'laserpower': [0., 2.], 'tElapsed': [0, 5]})
    raw.attrs = {'tipbox': np.array([1., 2., 3.]), 'pyccapt_source_id': 'apt-source'}
    monkeypatch.setattr(ccapt_tools.leap_tools, 'read_apt', lambda _: raw)
    result = ccapt_tools.apt_to_ccapt('dummy.apt')
    assert set(result.attrs['pyccapt_unavailable_columns']) == {
        'y (nm)', 'mc_uc (Da)', 'pulse_v (V)', 't_c (ns)', 'y_det (cm)', 'delta_p', 'multi'}
    np.testing.assert_array_equal(result.attrs['tipbox'], raw.attrs['tipbox'])
    result.attrs['tipbox'][0] = 100.
    assert raw.attrs['tipbox'][0] == 1.
    assert result['z (nm)'].tolist() == [0., -2.]
    assert result['x_det (cm)'].tolist() == [0., 2.]
    dataset = CalibrationDataset.from_frame(result)
    assert dataset.validity_status('high_voltage (V)').tolist() == [0, 0]
    assert dataset.validity_status('t (ns)').tolist() == [0, 1]
    assert dataset.validity_status('pulse_v (V)').tolist() == [2, 2]
    assert dataset.validity_status('start_counter').tolist() == [0, 0]


def test_nexus_mapping_preserves_import_availability_not_zero_heuristics(monkeypatch):
    raw = _epos_source(1)
    monkeypatch.setattr(ccapt_tools.leap_tools, 'read_epos', lambda _: raw)
    result = ccapt_tools.epos_to_ccapt('dummy.epos')
    with h5py.File(io.BytesIO(), 'w') as handle:
        write_nexus_mapping(handle, result)
        collection = handle['entry/pyccapt_data']
        availability = json.loads(collection['field_availability_json'].asstr()[()])
        assert availability['pulse_v (V)'] == 'available'
        assert availability['pulse_l (pJ)'] == 'unavailable'
        statuses = {child.attrs['name']: child['status'][:].tolist() for child in collection['columns'].values()}
        assert statuses['pulse_v (V)'] == [0]
        assert statuses['pulse_l (pJ)'] == [2]


def test_raw_processed_conversion_preserves_ids_attrs_and_tags_its_placeholders():
    raw = pd.DataFrame({'high_voltage (V)': [5000., 5000.], 'pulse_v (V)': [0., 0.],
        'pulse_l (pJ)': [0., 0.], 't (ns)': [100., 110.], 'x_det (cm)': [0., .1],
        'y_det (cm)': [0., .1], 'start_counter': np.array([2**60+3, 2**60+5], dtype=np.uint64),
        'event_id': np.array([7, 19], dtype=np.uint64), 'mc (Da)': [27., 28.]})
    raw.attrs = {'pyccapt_source_id': 'raw-source', 'pyccapt_unavailable_columns': ['pulse_l (pJ)']}
    result = data_tools.pyccapt_raw_to_processed(raw)
    np.testing.assert_array_equal(result.event_id, [7, 19])
    np.testing.assert_array_equal(result.start_counter, raw.start_counter)
    assert result.attrs['pyccapt_source_id'] == 'raw-source'
    dataset = CalibrationDataset.from_frame(result)
    for name in ('x (nm)', 'y (nm)', 'z (nm)', 't_c (ns)', 'delta_p', 'multi', 'pulse_l (pJ)'):
        assert dataset.validity_status(name).tolist() == [2, 2]
    assert dataset.validity_status('pulse_v (V)').tolist() == [0, 0]
    assert dataset.validity_status('mc (Da)').tolist() == [0, 0]
