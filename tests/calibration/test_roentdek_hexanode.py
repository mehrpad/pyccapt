from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from pyccapt.calibration.data_tools import roentdek_hexanode as hx

CCF = Path(__file__).resolve().parents[2] / 'pyccapt' / 'control' / 'tdc_roentdek' / 'source' / 'DAn-part2.ccf'


def _event(x_mm, y_mm, tof_ns=5000., params=hx.HexanodeParameters()):
    """First-hit TDC bins of one ideal hexanode hit at detector position (x, y)."""
    u = x_mm - params.x_offset_mm
    y = y_mm - params.y_offset_mm
    v, w = (u - np.sqrt(3) * y) / 2, (u + np.sqrt(3) * y) / 2
    times = []
    for coordinate, factor, offset, sum_offset in (
        (u, params.fu, 0., params.sum_offset_u_ns),
        (v, params.fv, 0., params.sum_offset_v_ns),
        (w, params.fw, params.w_offset_ns, params.sum_offset_w_ns),
    ):
        diff = 2 * coordinate / factor - offset
        times += [(-sum_offset + diff) / 2, (-sum_offset - diff) / 2]
    times += [-tof_ns, 0.]
    return np.rint(np.asarray(times) / params.tdc_bin_ns).astype(np.int64)


def test_full_event_reproduces_position_and_tof():
    channels = np.stack([_event(12.3, -7.8, 4321.), _event(-30., 20., 900.)])
    result = hx.reconstruct_hexanode(channels)
    np.testing.assert_allclose(result['x_mm'], [12.3, -30.], atol=0.02)
    np.testing.assert_allclose(result['y_mm'], [-7.8, 20.], atol=0.02)
    np.testing.assert_allclose(result['tof_ns'], [4321., 900.], atol=0.02)
    assert result['valid'].all()
    np.testing.assert_array_equal(result['layer_mask'], [7, 7])
    np.testing.assert_array_equal(result['n_reconstructed'], [0, 0])


@pytest.mark.parametrize(('dropped', 'mask', 'rebuilt'), [
    ((0,), 6, 0),       # u1 missing, v and w complete: use the vw pair
    ((0, 2), 7, 2),     # u1 and v1 missing: rebuild both from their time sums next to w
    ((5,), 3, 0),       # w2 missing: uv pair
])
def test_missing_signals_fall_back_to_remaining_layers(dropped, mask, rebuilt):
    channels = _event(8., 15.)[None, :]
    channels[0, list(dropped)] = 0
    result = hx.reconstruct_hexanode(channels)
    assert result['valid'][0]
    assert result['layer_mask'][0] == mask
    assert result['n_reconstructed'][0] == rebuilt
    np.testing.assert_allclose([result['x_mm'][0], result['y_mm'][0]], [8., 15.], atol=0.05)


def test_wrong_first_hit_is_rejected_by_time_sum_gate():
    channels = _event(-5., 25.)[None, :]
    channels[0, 0] += 400  # u1 10 ns late: a reflection, not the hit
    result = hx.reconstruct_hexanode(channels)
    assert result['layer_mask'][0] == 6
    np.testing.assert_allclose([result['x_mm'][0], result['y_mm'][0]], [-5., 25.], atol=0.05)


def test_events_without_start_or_enough_layers_are_invalid():
    no_start = _event(1., 1.)
    no_start[6] = 0
    one_layer = _event(1., 1.)
    one_layer[2:6] = 0
    empty = np.zeros(8, dtype=np.int64)
    result = hx.reconstruct_hexanode(np.stack([no_start, one_layer, empty]))
    assert not result['valid'].any()
    assert np.isnan(result['x_mm']).all()
    np.testing.assert_array_equal(result['layer_mask'], [0, 0, 0])
    assert np.isnan(result['tof_ns'][0]) and np.isfinite(result['tof_ns'][1])


def test_ccf_parameters_match_defaults():
    params = hx.HexanodeParameters.from_ccf(CCF)
    assert params == hx.HexanodeParameters()


def test_hdf5_loader_reads_legacy_and_flat_layouts(tmp_path):
    channels = np.stack([_event(3., 4.), _event(-10., 2., 800.)])
    legacy = tmp_path / 'legacy.h5'
    with h5py.File(legacy, 'w') as handle:
        for i in range(8):
            handle.create_dataset(f'tdc/ch{i}', data=channels[:, i].astype(np.int32))
    flat = tmp_path / 'flat.h5'
    with h5py.File(flat, 'w') as handle:
        handle['tdc/time_data'] = channels.reshape(-1).view(np.uint64)
        handle['tdc/channel'] = np.tile(np.arange(8, dtype=np.uint32), 2)
        handle['tdc/start_counter'] = np.repeat(np.array([7, 9], dtype=np.uint64), 8)

    np.testing.assert_array_equal(hx.load_roentdek_tdc_channels(legacy)['channels'], channels)
    loaded = hx.load_roentdek_tdc_channels(flat)
    np.testing.assert_array_equal(loaded['channels'], channels)
    np.testing.assert_array_equal(loaded['start_counter'], [7, 9])

    frame = hx.reconstruct_roentdek_hdf5(flat)
    np.testing.assert_allclose(frame['x_det (cm)'], [0.3, -1.0], atol=0.002)
    np.testing.assert_allclose(frame['t (ns)'], [5000., 800.], atol=0.02)
    np.testing.assert_array_equal(frame['start_counter'], [7, 9])


def test_correction_fit_round_trips_and_matches_cobold(tmp_path):
    rng = np.random.default_rng(0)
    radius = 38 * np.sqrt(rng.random(3000))
    angle = 2 * np.pi * rng.random(3000)
    x, y = radius * np.cos(angle), radius * np.sin(angle)
    result = hx.reconstruct_hexanode(np.stack([_event(a, b) for a, b in zip(x, y)]))
    cobold = pd.DataFrame({'n_particles': 1., 'x1_mm': 1.01 * x + 0.002 * y ** 2, 'y1_mm': 0.99 * y - 0.3})

    correction = hx.fit_cobold_correction(result, cobold, degree=3)
    restored = hx.HexanodeCorrection.from_dict(correction.to_dict())
    corrected = hx.reconstruct_hexanode(np.stack([_event(a, b) for a, b in zip(x, y)]), correction=restored)
    stats = hx.compare_with_cobold(corrected, cobold)
    assert stats['both'] == 3000
    assert stats['rms_x_mm'] < 0.02 and stats['rms_y_mm'] < 0.02


def test_read_cobold_txt(tmp_path):
    path = tmp_path / 'run.txt'
    path.write_text(
        '1 0.000000 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n'
        '133 0.000000 2 0.828495 2 10 37.0948 7.51468 535.0565 14 24.7025 -25.2118 559.8800 19 '
        '0 0 0 0 0 0 0 0\n'
    )
    frame = hx.read_cobold_txt(path)
    assert list(frame['n_particles']) == [0, 2]
    assert frame.loc[1, 'x2_mm'] == pytest.approx(24.7025)
    assert frame.loc[1, 'method1'] == 14


# --- all hits (LMF-like) -----------------------------------------------------

def _hits(*events):
    """RoentdekHits from per-event lists of (channel, bin)."""
    event = [i for i, hits in enumerate(events) for _ in hits]
    channel = [c for hits in events for c, _ in hits]
    value = [v for hits in events for _, v in hits]
    return hx.RoentdekHits.from_rows(event, channel, value, len(events))


def _as_hits(bins, arrival_bins=0):
    """(channel, bin) pairs of one ideal ion; anode times shifted by ``arrival_bins``."""
    return [(c, int(b) + (arrival_bins if c < 6 else 0)) for c, b in enumerate(bins) if c != 7]


def test_hits_container_counts_and_first_hits():
    hits = _hits([(0, 5), (3, 7), (0, 9), (7, 0)], [])
    assert len(hits) == 2
    np.testing.assert_array_equal(hits.counts()[0], [2, 0, 0, 1, 0, 0, 0, 0])  # zeros (MCP rows) are dropped
    np.testing.assert_array_equal(hits.first_hits()[0], [5, 0, 0, 7, 0, 0, 0, 0])
    np.testing.assert_array_equal(hits.offsets, [0, 3, 3])
    single = hx.RoentdekHits.from_first_hits(np.array([[1, 2, 0, 4, 0, 0, -3, 0]]))
    np.testing.assert_array_equal(single.first_hits(), [[1, 2, 0, 4, 0, 0, -3, 0]])


def test_read_lmf_txt(tmp_path):
    path = tmp_path / 'run_raw.txt'
    path.write_text(
        'Number of events = 2\n\n------- #1 -------\nT  = 0.000 ns = \t0.000000 s\n'
        'chan     1     2  1163  1500\nchan     2     0\nchan     7     1 -201845\nchan     8     1     0\n'
        'Levelinfo: 0\n\n------- #2 -------\nT  = 131.350 ns = \t0.000000 s\nchan     1     0\n'
        'chan     8     1     0\n'
    )
    hits = hx.read_lmf_txt(path)
    assert len(hits) == 2
    np.testing.assert_array_equal(hits.first_hits()[0], [1163, 0, 0, 0, 0, 0, -201845, 0])
    np.testing.assert_array_equal(hits.counts()[:, 0], [2, 0])
    np.testing.assert_allclose(hits.timestamp_ns, [0.0, 131.35])


def test_multihit_recovers_ion_behind_reflections_on_two_layers():
    ion = _event(10.0, -5.0, 3000.)
    reflections = [(0, int(ion[0]) - 900), (2, int(ion[2]) - 700)]  # earlier pulses on u1 and v1
    hits = _hits(reflections + _as_hits(ion))
    assert not hx.reconstruct_hexanode(hits.first_hits())['valid'][0]
    result = hx.reconstruct_hexanode_multihit(hits, sum_walk=False)
    np.testing.assert_array_equal(result['n_particles'], [1])
    assert result['x_mm'][0] == pytest.approx(10.0, abs=0.05)
    assert result['y_mm'][0] == pytest.approx(-5.0, abs=0.05)
    assert result['tof_ns'][0] == pytest.approx(3000., abs=0.05)
    assert result['layer_mask'][0] == 7 and result['n_reconstructed'][0] == 0


def test_multihit_single_signals_need_the_hexanode_check():
    ion = _event(-12.0, 8.0, 2000.)
    one_per_layer = [(0, ion[0]), (3, ion[3]), (4, ion[4]), (6, ion[6])]  # u1, v2, w1 + start
    result = hx.reconstruct_hexanode_multihit(_hits(one_per_layer), sum_walk=False)
    assert result['layer_mask'][0] == 7 and result['n_reconstructed'][0] == 3
    assert result['x_mm'][0] == pytest.approx(-12.0, abs=0.1)
    assert result['y_mm'][0] == pytest.approx(8.0, abs=0.1)
    inconsistent = [(0, ion[0]), (3, ion[3] + 400), (4, ion[4]), (6, ion[6])]
    assert hx.reconstruct_hexanode_multihit(_hits(inconsistent), sum_walk=False)['n_particles'][0] == 0


def test_multihit_finds_a_later_ion_without_its_own_mcp_signal():
    first, second = _event(5.0, 5.0, 2500.), _event(-20.0, 15.0, 2500.)
    later = int(round(15.0 / hx.HexanodeParameters().tdc_bin_ns))  # second ion arrives 15 ns after the trigger
    hits = _hits(_as_hits(first) + [hit for hit in _as_hits(second, later) if hit[0] < 6])
    result = hx.reconstruct_hexanode_multihit(hits, sum_walk=False)
    np.testing.assert_array_equal(result['n_particles'], [2])
    np.testing.assert_allclose(result['x_mm'], [5.0, -20.0], atol=0.05)
    np.testing.assert_allclose(result['y_mm'], [5.0, 15.0], atol=0.05)
    np.testing.assert_allclose(result['tof_ns'], [2500., 2515.], atol=0.05)
    assert result['arrival_ns'][1] == pytest.approx(15.0, abs=0.05)
    assert hx.first_particle(result, 1)['x_mm'][0] == pytest.approx(5.0, abs=0.05)


def test_multihit_rejects_a_later_ion_before_the_trigger():
    first, second = _event(5.0, 5.0, 2500.), _event(-20.0, 15.0, 2500.)
    early = -int(round(15.0 / hx.HexanodeParameters().tdc_bin_ns))
    hits = _hits(_as_hits(first) + [hit for hit in _as_hits(second, early) if hit[0] < 6])
    assert hx.reconstruct_hexanode_multihit(hits, sum_walk=False)['n_particles'][0] == 1


def test_sum_walk_round_trips_through_the_correction():
    walk = np.array([[0.01, 0.2], [0.0, -0.1], [0.02, 0.3]])
    correction = hx.HexanodeCorrection(1, np.array([0., 0., 1.]), np.array([0., 1., 0.]), sum_walk=walk)
    restored = hx.HexanodeCorrection.from_dict(correction.to_dict())
    np.testing.assert_array_equal(restored.sum_walk, walk)
    hits = _hits(_as_hits(_event(1.0, 2.0)))
    np.testing.assert_array_equal(hx.reconstruct_hexanode_multihit(hits, correction=restored)['sum_walk'], walk)
    assert np.all(hx.fit_sum_walk(np.stack([_event(1.0, 2.0)] * 10)) == 0)  # too few events to fit


def test_all_particles_from_a_per_hit_hdf5_table(tmp_path):
    ion = _event(3.0, 4.0, 4000.)
    rows = [(0, int(ion[0]) - 900)] + _as_hits(ion) + [(7, 0)]  # reflection on u1, then the ion
    path = tmp_path / 'hits.h5'
    with h5py.File(path, 'w') as handle:
        handle['tdc/channel'] = np.array([c for c, _ in rows] + [7], dtype=np.uint32)
        handle['tdc/time_data'] = np.array([v for _, v in rows] + [0], dtype=np.int64).view(np.uint64)
        handle['tdc/start_counter'] = np.array([50] * len(rows) + [51], dtype=np.uint64)
    frame = hx.reconstruct_roentdek_hdf5(path, all_particles=True, sum_walk=False)
    assert frame['event'].tolist() == [0] and frame['particle'].tolist() == [0]
    assert frame['x_det (cm)'][0] == pytest.approx(0.3, abs=0.005)
    assert frame['start_counter'][0] == 50
    assert len(hx.load_roentdek_hits(path)[0]) == 2


def test_loader_reads_reader_v3_offset_layout(tmp_path):
    # RoentDek reader v3 (Resort64c): one tdc row per hit, signed bins + 2**23.
    import h5py

    path = tmp_path / 'v3.h5'
    event = _event(8., 15.)
    bins = np.r_[event[:6], 900, event[6], 0]                      # extra (later) start hit on channel 6
    channel = np.array([0, 1, 2, 3, 4, 5, 6, 6, 7], dtype=np.uint32)
    with h5py.File(path, 'w') as handle:
        data = handle.create_dataset('tdc/time_data', data=(bins + (1 << 23)).astype(np.uint64))
        data.attrs['roentdek_time_offset'] = 1 << 23
        handle['tdc/channel'] = channel
        handle['tdc/start_counter'] = np.full(len(bins), 42, dtype=np.uint64)
    hits, extras = hx.load_roentdek_hits(path)
    assert len(hits) == 1
    np.testing.assert_array_equal(extras['start_counter'], [42])
    np.testing.assert_array_equal(hx.load_roentdek_tdc_channels(path)['channels'][0, :6], event[:6])
    first = hx.first_particle(hx.reconstruct_hexanode_multihit(hits), len(hits))
    assert first['valid'][0]
    np.testing.assert_allclose([first['x_mm'][0], first['y_mm'][0]], [8., 15.], atol=0.05)
