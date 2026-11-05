import json
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd
import pytest

from pyccapt.calibration.core.dataset import CalibrationDataset
from pyccapt.calibration.core.exceptions import CalibrationInputError
from pyccapt.calibration.data_tools import analysis_session as sessions
from pyccapt.calibration.data_tools.live_hdf5 import LiveEventWriter, read_live_events
from pyccapt.calibration.data_tools.model_state import export_model_state, restore_model_state
from pyccapt.calibration.data_tools.nexus_export import export_nexus, NXAPM_VERSION


def test_event_identity_survives_crop_reorder_and_large_counters():
    original = CalibrationDataset.from_frame(pd.DataFrame({'t (ns)': [3., 1., 2.],
                                                          'start_counter': np.array([2**60+7]*3, dtype=np.uint64)}))
    cropped = CalibrationDataset.from_frame(original.frame.iloc[[2, 0]].reset_index(drop=True))
    np.testing.assert_array_equal(cropped.event_ids, [2, 0])
    assert cropped.source_id == original.source_id
    assert cropped.source_hash != original.source_hash
    assert not cropped.event_ids.flags.writeable
    assert cropped.frame['start_counter'].iloc[0] == 2**60+7


@pytest.mark.parametrize('ids', [[1., 2.], [-1, 2], [1, 1]])
def test_invalid_event_identity_is_rejected(ids):
    with pytest.raises(CalibrationInputError, match='event_id'):
        CalibrationDataset.from_frame(pd.DataFrame({'event_id': ids}))


def test_validity_distinguishes_zero_unknown_and_absent():
    dataset = CalibrationDataset.from_frame(pd.DataFrame({'t (ns)': [0., np.nan, 1.]}))
    np.testing.assert_array_equal(dataset.validity_status('t (ns)'), [0, 1, 0])
    np.testing.assert_array_equal(dataset.validity_status('x_det (cm)'), [2, 2, 2])


def test_session_roundtrip_integrity_and_search(tmp_path):
    path = tmp_path/'session.h5'
    arrays = {'event_ids': np.array([2**60+7, 2**60+9], dtype=np.uint64),
              'reference mask': np.array([True, False]), 'source/t (ns)': np.array([0., np.nan])}
    metadata = {'sample': 'Fe study', 'history': [{'action': 'accept', 'model': {'coefficients': [1., 2.]}}]}
    sessions.save_analysis_session(path, metadata, arrays)
    restored, values = sessions.load_analysis_session(path)
    assert restored == metadata
    for name, value in arrays.items():
        assert values[name].dtype == value.dtype
        np.testing.assert_array_equal(values[name], value)
    assert sessions.validate_analysis_session(path)['valid']
    assert len(sessions.list_analysis_sessions(tmp_path, 'FE study')) == 1
    assert sessions.list_analysis_sessions(tmp_path, 'nonexistent') == []
    with pytest.raises(CalibrationInputError, match='limit'):
        sessions.load_analysis_session(path, max_bytes=1)


@pytest.mark.parametrize('target', ['metadata', 'array', 'manifest'])
def test_session_corruption_fails_before_restoring_state(tmp_path, target):
    path = tmp_path/'session.h5'
    sessions.save_analysis_session(path, {'sample': 'A'}, {'x': np.arange(8.)})
    with h5py.File(path, 'a') as handle:
        group = handle['analysis_session']
        if target == 'array':
            group['arrays/a000000'][2] = 99.
        else:
            group[target+'_json'][()] = '{}' if target == 'metadata' else '[]'
    assert not sessions.validate_analysis_session(path)['valid']
    with pytest.raises(CalibrationInputError, match='mismatch|checksum'):
        sessions.load_analysis_session(path)


def test_atomic_session_failure_preserves_previous_file(tmp_path):
    path = tmp_path/'session.h5'
    sessions.save_analysis_session(path, {'sample': 'A'}, {'x': np.arange(8.)})
    before = path.read_bytes()
    with pytest.raises(CalibrationInputError):
        sessions.save_analysis_session(path, {}, {'unsafe': np.array([object()], dtype=object)})
    assert path.read_bytes() == before
    assert not list(tmp_path.glob('*.tmp'))


def test_session_v1_migration_preserves_arrays(tmp_path):
    path = tmp_path/'legacy.h5'
    sessions.save_analysis_session(path, {'sample': 'prototype'}, {'x': np.arange(3.)})
    with h5py.File(path, 'a') as handle:
        group = handle['analysis_session']
        group.attrs['schema_version'] = sessions.LEGACY_SCHEMA
        del group.attrs['manifest_sha256']
    metadata, arrays = sessions.load_analysis_session(path)
    assert metadata['storage_migrations'][0]['to'] == sessions.SESSION_SCHEMA
    np.testing.assert_array_equal(arrays['x'], np.arange(3.))


def test_receipt_only_file_is_not_misrepresented_as_resumable(tmp_path):
    path = tmp_path/'old.h5'
    with h5py.File(path, 'w') as handle:
        handle.attrs['pyccapt_analysis_json'] = '{}'
    with pytest.raises(CalibrationInputError, match='receipt alone'):
        sessions.load_analysis_session(path)


@pytest.mark.parametrize('kind', ['root_external', 'virtual', 'external_storage'])
def test_sessions_reject_nonembedded_data_even_when_bytes_match(tmp_path, kind):
    path = tmp_path/'session.h5'
    values = np.arange(3.)
    sessions.save_analysis_session(path, {}, {'x': values})
    if kind == 'root_external':
        reference = tmp_path/'reference.h5'
        with h5py.File(reference, 'w') as handle:
            handle['analysis_session'] = h5py.ExternalLink(str(path), '/analysis_session')
        path = reference
    else:
        with h5py.File(path, 'a') as handle:
            group = handle['analysis_session/arrays']
            del group['a000000']
            if kind == 'virtual':
                source = tmp_path/'source.h5'
                with h5py.File(source, 'w') as other:
                    other['x'] = values
                layout = h5py.VirtualLayout(shape=values.shape, dtype=values.dtype)
                layout[:] = h5py.VirtualSource(str(source), 'x', shape=values.shape)
                group.create_virtual_dataset('a000000', layout)
            else:
                external = tmp_path/'values.bin'
                group.create_dataset('a000000', data=values, external=[(str(external), 0, values.nbytes)])
    assert not sessions.validate_analysis_session(path)['valid']
    with pytest.raises(CalibrationInputError, match='embed'):
        sessions.load_analysis_session(path)


@pytest.mark.parametrize('kind', ['forest', 'boosting', 'linear', 'pipeline'])
def test_learned_model_roundtrip_reproduces_predictions(kind):
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    rng = np.random.default_rng(519)
    x = rng.normal(size=(70, 2))
    target = 1 + x[:, 0]**2 + .1*x[:, 1]
    models = {'forest': RandomForestRegressor(n_estimators=5, max_depth=4, random_state=1),
              'boosting': GradientBoostingRegressor(n_estimators=8, random_state=1),
              'linear': LinearRegression(),
              'pipeline': make_pipeline(StandardScaler(), PolynomialFeatures(2), LinearRegression())}
    model = models[kind].fit(x, target)
    state = json.loads(json.dumps(export_model_state(model), allow_nan=False))
    restored = restore_model_state(state)
    query = rng.normal(size=(100, 2))
    np.testing.assert_allclose(restored.predict(query), model.predict(query), rtol=1e-12, atol=1e-12)


def test_cyclic_tree_snapshot_rejected():
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor().fit([[0], [1]], [0, 1])
    state = export_model_state(model)
    state['left'][0] = 0
    with pytest.raises(CalibrationInputError, match='child'):
        restore_model_state(state)


def test_physical_tof_model_roundtrip_preserves_terms_mass_and_domain():
    from pyccapt.calibration.core.multireference_tof import MultiReferenceToFModel, ToFModelConfig
    rng = np.random.default_rng(732)
    n = 600
    voltage = rng.uniform(4000., 6000., n)
    event = np.linspace(0., 1., n)
    # Constant spatial coordinates drop five raw design terms during fitting.
    baseline = .110 * np.sqrt(1.66e-27 / (2 * 1.6e-19 * voltage)) * 1e9
    features = np.column_stack((voltage, np.zeros(n), np.zeros(n), event, baseline))
    mass = np.resize([14., 28., 56.], n)
    tof = 12. + np.sqrt(mass)*baseline*np.exp(.01 + .00001*(voltage-5000) + .005*event + .006*(event >= .5))
    model = MultiReferenceToFModel(ToFModelConfig(linear_drift=True, block_edges=(.5,)))
    model.fit(features, np.column_stack((tof, mass)))
    state = json.loads(sessions.canonical_json(export_model_state(model)))
    restored = restore_model_state(state)
    assert restored.term_names == model.term_names
    np.testing.assert_allclose(restored.predict_factor(features), model.predict_factor(features), rtol=1e-14)
    np.testing.assert_allclose(restored.predict_mass_da(features, tof), mass, rtol=1e-8)
    query = features.copy()
    query[0, 0], query[1, 1], query[2, 3] = 1e6, 10., 2.
    np.testing.assert_array_equal(restored.domain_mask(query), model.domain_mask(query))
    state['keep'][0] = False
    with pytest.raises(CalibrationInputError, match='terms'):
        restore_model_state(state)


def test_swmr_publishes_only_committed_complete_rows(tmp_path):
    path = tmp_path/'live.h5'
    with LiveEventWriter(path, {'event_id': 'uint64', 't': 'float64'}, units={'t': 'ns'}) as writer:
        writer.append({'event_id': np.array([2**60+7], dtype=np.uint64), 't': np.array([0.])})
        arrays, status = read_live_events(path)
        assert status['committed_rows'] == 1
        assert arrays['event_id'][0] == 2**60+7
        # Simulate an interrupted next append before its publication marker.
        writer.file['events/t'].resize((2,))
        writer.file['events/t'][1] = 99.
        writer.file['events/t'].flush()
        arrays, status = read_live_events(path)
        np.testing.assert_array_equal(arrays['t'], [0.])
        with pytest.raises(CalibrationInputError, match='dtype'):
            writer.append({'event_id': np.array([1.]), 't': np.array([0.])})
    assert read_live_events(path)[1]['writer_state'] == 1


def test_swmr_reader_in_independent_process_sees_flushed_commit(tmp_path):
    path = tmp_path/'live-process.h5'
    with LiveEventWriter(path, {'event_id': 'uint64', 't': 'float64'}) as writer:
        writer.append({'event_id': np.array([2**60+7], dtype=np.uint64), 't': np.array([0.])})
        # A separate interpreter has no shared h5py file handles or caches.
        code = ('import sys; from pyccapt.calibration.data_tools.live_hdf5 import read_live_events; '
                'a,s=read_live_events(sys.argv[1]); '
                'assert s["committed_rows"]==1 and s["writer_state"]==0; '
                'assert int(a["event_id"][0])==2**60+7 and a["t"][0]==0.')
        result = subprocess.run([sys.executable, '-c', code, str(path)], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr


def test_swmr_close_during_read_cannot_hide_final_committed_rows(tmp_path, monkeypatch):
    path = tmp_path/'live-close-race.h5'
    writer = LiveEventWriter(path, {'event_id': 'uint64'})
    writer.append({'event_id': np.array([0], dtype=np.uint64)})
    original_refresh = h5py.Dataset.refresh
    closed = False
    def refresh(dataset):
        nonlocal closed
        if dataset.name == '/writer_state' and not closed:
            # Finish acquisition exactly as the reader checks terminal state.
            writer.append({'event_id': np.array([1], dtype=np.uint64)})
            writer.close()
            closed = True
        return original_refresh(dataset)
    monkeypatch.setattr(h5py.Dataset, 'refresh', refresh)
    try:
        arrays, status = read_live_events(path)
        assert status['writer_state'] == 1 and status['committed_rows'] == 2
        np.testing.assert_array_equal(arrays['event_id'], [0, 1])
    finally:
        writer.close()


def test_nexus_mapping_preserves_identity_units_and_reports_missing_context(tmp_path):
    frame = pd.DataFrame({'event_id': np.array([2**60+7, 2**60+9], dtype=np.uint64),
                          'x_det (cm)': [1., np.nan], 'y_det (cm)': [2., 3.],
                          'mc (Da)': [27., 28.], 't (ns)': [0., 100.]})
    path = export_nexus(tmp_path/'nexus.h5', frame, metadata={'experiment_alias': 'test'})
    with h5py.File(path, 'r') as handle:
        entry = handle['entry']
        assert entry.attrs['application_definition_version'] == NXAPM_VERSION
        assert 'definition' not in entry  # Partial mapping must not assert full application conformance.
        np.testing.assert_array_equal(entry['pyccapt_data/event_id'][:], frame.event_id.to_numpy())
        positions = entry['atom_probe1/hit_finding/hit_positions']
        np.testing.assert_array_equal(positions[0], [10., 20.])
        assert positions.attrs['units'] == 'mm'
        report = json.loads(entry['mapping_report_json'].asstr()[()])
        assert report['conformance'] == 'partial' and report['event_identity_preserved']
        assert 'start_time' in report['missing_user_metadata']


def test_storage_benchmark_checks_exact_full_roi_and_preview_reads(tmp_path):
    from pyccapt.calibration.data_tools.storage_benchmark import benchmark_storage
    report = benchmark_storage(tmp_path, np.arange(1000, dtype=np.uint64)+2**60,
                               chunk_rows=(128,), compressions=(None, 'gzip'), repeats=1)
    assert len(report['results']) == 2
    assert all(result['exact_values'] and result['roi_median_s'] >= 0 for result in report['results'])
    assert not list(tmp_path.iterdir())


def test_nexus_measurements_do_not_publish_unavailable_placeholders_as_real_zeros(tmp_path):
    frame = pd.DataFrame({'x_det (cm)': [0., 1.], 'y_det (cm)': [0., 0.],
                          't (ns)': [0., np.inf], 't_c (ns)': [0., 0.],
                          'mc (Da)': [0., 0.], 'x (nm)': [0., 0.],
                          'y (nm)': [0., 0.], 'z (nm)': [0., 0.]})
    frame.attrs['pyccapt_unavailable_columns'] = [
        'y_det (cm)', 't_c (ns)', 'mc (Da)', 'x (nm)', 'y (nm)', 'z (nm)']
    path = export_nexus(tmp_path/'missing.h5', frame)
    with h5py.File(path, 'r') as handle:
        instrument = handle['entry/atom_probe1']
        np.testing.assert_array_equal(instrument['hit_finding/hit_positions'][:], [[0., np.nan], [10., np.nan]])
        np.testing.assert_array_equal(instrument['hit_finding/hit_positions_status'][:], [[0, 2], [0, 2]])
        np.testing.assert_array_equal(instrument['voltage_and_bowl/raw_tof'][:], [0., np.nan])
        np.testing.assert_array_equal(instrument['voltage_and_bowl/raw_tof_status'][:], [0, 1])
        for name in ('voltage_and_bowl/calibrated_tof', 'mass_to_charge_conversion/mass_to_charge',
                     'reconstruction/reconstructed_positions'):
            assert np.isnan(instrument[name][:]).all()
            assert (instrument[name + '_status'][:] == 2).all()
        # The original compatibility columns retain their exact placeholders.
        columns = handle['entry/pyccapt_data/columns']
        original = next(node for node in columns.values() if node.attrs['name'] == 'mc (Da)')
        np.testing.assert_array_equal(original['value'][:], [0., 0.])


def test_export_records_running_source_version_when_distribution_metadata_is_stale(tmp_path, monkeypatch):
    import pyccapt
    monkeypatch.setattr(sessions, 'version', lambda name: 'stale-installed-version')
    identity = sessions.software_identity()
    assert identity['packages']['pyccapt'] == 'stale-installed-version'
    assert identity['pyccapt_source_version'] == pyccapt.__version__
    path = export_nexus(tmp_path/'source-version.h5', pd.DataFrame({'mc (Da)': [27.]}))
    with h5py.File(path, 'r') as handle:
        program = handle['entry/atom_probe1/mass_to_charge_conversion/program1/program']
        assert program.attrs['version'] == pyccapt.__version__
