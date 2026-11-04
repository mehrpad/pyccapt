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
