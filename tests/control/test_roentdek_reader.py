"""RoentDek TDC8HP reader: raw-hit encoding, HDF5 layout, and (on Windows) the real DLLs.

The DLL tests replay 3000 recorded groups of run 198 through
* the acquisition DLL built with MOCK_TDC (``source/test/wrapper_read_TDC8HP_x64_mock.dll``),
  which feeds the recorded groups as HPTDC data words into the unchanged decoding code, and
* RoentDek's Resort64c via ``roentdek_resort_x64.dll``,
and compare with the CoboldPC export of the same groups.
"""
import ctypes
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from pyccapt.control.core import chunk_store
from pyccapt.control.core.data_integrity import validate_hdf5
from pyccapt.control.core.hdf5_creator import hdf_creator
from pyccapt.control.tdc_roentdek import tdc_roentdek

MODULE_DIR = Path(tdc_roentdek.__file__).resolve().parent
SAMPLE = Path(__file__).resolve().parents[1] / "data" / "roentdek" / "run198_first3000_groups.npz"


def _variables(path, **kwargs):
    defaults = dict(path=str(path), exp_name='run', counter_source='TDC', main_counter=[1],
                    main_raw_counter=[1], main_temperature=[40.], main_chamber_vacuum=[1e-8])
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _sample():
    data = np.load(SAMPLE)
    return {key: data[key] for key in data.files}


def test_raw_hit_rows_keep_every_hit_and_encode_signed_times():
    cnt = np.array([[1, 0, 2, 0, 0, 0, 1, 1]], dtype=np.uint32)
    hits = np.zeros((1, 8, 10), np.int32)
    hits[0, 0, 0] = 5
    hits[0, 2, :2] = [-7, 9]
    hits[0, 6, 0] = -136605
    group, channel, times = tdc_roentdek.raw_hit_rows(np.array([42], np.uint64), cnt, hits)
    np.testing.assert_array_equal(group, [0, 0, 0, 0, 0])
    np.testing.assert_array_equal(channel, [0, 2, 2, 6, 7])
    decoded = times.astype(np.int64) - tdc_roentdek.RAW_TIME_OFFSET
    np.testing.assert_array_equal(decoded, [5, -7, 9, -136605, 0])
    assert times.dtype == np.uint64 and np.all(times != 0)


def test_new_roentdek_layout_finalizes_ions_and_all_hits(tmp_path):
    directory = tmp_path/'temp_data'/'chunks'
    ions = {'x': np.array([.1, -.2, 3.]), 'y': np.array([.5, .6, -.7]), 't': np.array([1000., np.nan, 2500.]),
            'voltage': np.full(3, 4000.), 'voltage_pulse': np.zeros(3), 'laser_pulse': np.full(3, 12.),
            'time_stamp': np.array([10, 10, 30], np.uint64), 'ro_reco_method': np.array([0, 6, 0], np.int16)}
    raw = {'ro_raw_channel': np.array([0, 1, 7, 0, 7], np.uint32),
           'ro_raw_time': (np.array([3, 4, 0, -5, 0]) + tdc_roentdek.RAW_TIME_OFFSET).astype(np.uint64),
           'ro_raw_time_stamp': np.array([10, 10, 10, 30, 30], np.uint64),
           'ro_raw_voltage': np.full(5, 4000.), 'ro_raw_voltage_pulse': np.zeros(5),
           'ro_raw_laser_pulse': np.full(5, 12.)}
    chunk_store.atomic_write_chunk_group(directory, stream_name='roentdek', chunk_id=1, arrays=ions)
    chunk_store.atomic_write_chunk_group(directory, stream_name='roentdek_raw', chunk_id=1, arrays=raw)
    hdf_creator(_variables(tmp_path), {'tdc': 'on', 'tdc_model': 'RoentDek'}, [1], [0.])
    with h5py.File(tmp_path/'run.h5') as handle:
        np.testing.assert_array_equal(handle['dld/x'], ions['x'])
        np.testing.assert_array_equal(handle['dld/t'], ions['t'])
        np.testing.assert_array_equal(handle['dld/start_counter'], ions['time_stamp'])
        np.testing.assert_array_equal(handle['dld/reco_method'], ions['ro_reco_method'])
        np.testing.assert_array_equal(handle['tdc/channel'], raw['ro_raw_channel'])
        np.testing.assert_array_equal(handle['tdc/time_data'], raw['ro_raw_time'])
        np.testing.assert_array_equal(handle['tdc/start_counter'], raw['ro_raw_time_stamp'])
        assert handle['tdc/time_data'].attrs['roentdek_time_offset'] == tdc_roentdek.RAW_TIME_OFFSET
        assert handle['dld/x'].attrs['units'] == 'cm'
    assert validate_hdf5(tmp_path/'run.h5')['valid']


def test_without_resort64c_the_fallback_is_used_and_recorded(tmp_path, monkeypatch):
    from pyccapt.control.tdc_roentdek import roentdek_resort
    from pyccapt.control.tdc_roentdek.roentdek_reco import RoentDekReconstructor
    monkeypatch.setattr(roentdek_resort, 'DLL_PATH', str(tmp_path/'missing'/'roentdek_resort_x64.dll'))
    reco = roentdek_resort.make_reconstructor(bin_ns=0.025, verbose=False)
    assert isinstance(reco, RoentDekReconstructor)
    tdc_roentdek.record_reconstruction(tmp_path/'temp_data', reco)
    directory = tmp_path/'temp_data'/'chunks'
    ions = {'x': np.array([.1]), 'y': np.array([.5]), 't': np.array([1000.]), 'voltage': np.full(1, 4000.),
            'voltage_pulse': np.zeros(1), 'laser_pulse': np.zeros(1), 'time_stamp': np.array([10], np.uint64),
            'ro_reco_method': np.array([0], np.int16)}
    raw = {'ro_raw_channel': np.array([7], np.uint32),
           'ro_raw_time': np.array([tdc_roentdek.RAW_TIME_OFFSET], np.uint64),
           'ro_raw_time_stamp': np.array([10], np.uint64), 'ro_raw_voltage': np.full(1, 4000.),
           'ro_raw_voltage_pulse': np.zeros(1), 'ro_raw_laser_pulse': np.zeros(1)}
    chunk_store.atomic_write_chunk_group(directory, stream_name='roentdek', chunk_id=1, arrays=ions)
    chunk_store.atomic_write_chunk_group(directory, stream_name='roentdek_raw', chunk_id=1, arrays=raw)
    hdf_creator(_variables(tmp_path), {'tdc': 'on', 'tdc_model': 'RoentDek'}, [1], [0.])
    with h5py.File(tmp_path/'run.h5') as handle:
        assert handle['dld'].attrs['roentdek_reconstruction'].startswith('PyCCAPT fallback')
        assert 'roentdek_resort_cfg' not in handle['dld'].attrs
    assert validate_hdf5(tmp_path/'run.h5')['valid']


def test_missing_tdc_files_are_restored(tmp_path):
    (tmp_path/'source'/'default_files').mkdir(parents=True)
    for name in (tdc_roentdek.LMF_TEMPLATE, 'global.cfg'):
        shutil.copy2(MODULE_DIR/'source'/'default_files'/name, tmp_path/'source'/'default_files'/name)
    assert tdc_roentdek.check_tdc_files(tmp_path)
    assert (tmp_path/'global.cfg').is_file() and (tmp_path/tdc_roentdek.LMF_TEMPLATE).is_file()
    os.remove(tmp_path/'source'/'default_files'/'global.cfg')
    os.remove(tmp_path/'global.cfg')
    assert not tdc_roentdek.check_tdc_files(tmp_path)


def test_global_cfg_matches_the_shipped_default():
    # the instrument was run with MCP trigger and a -6000..1000 ns group window
    text = (MODULE_DIR/'global.cfg').read_text()
    assert 'GroupRangeStart -6000 ns' in text and 'GroupRangeEnd 1000 ns' in text
    assert text == (MODULE_DIR/'source'/'default_files'/'global.cfg').read_text()


windows_only = pytest.mark.skipif(sys.platform != 'win32', reason='RoentDek DLLs are Windows x64 binaries')


@windows_only
def test_mock_dll_returns_every_hit_and_writes_the_lmf(tmp_path, monkeypatch):
    sample = _sample()
    n = len(sample['time_stamp'])
    record = np.dtype([('ts', '<u8'), ('cnt', '<u4', (8,)), ('tdc', '<i4', (8, 10))])
    events = np.zeros(n, record)
    events['ts'], events['cnt'], events['tdc'] = sample['time_stamp'], sample['cnt'], sample['tdc']
    events.tofile(tmp_path/'events.bin')
    shutil.copy2(MODULE_DIR/tdc_roentdek.LMF_TEMPLATE, tmp_path/tdc_roentdek.LMF_TEMPLATE)
    monkeypatch.setenv('MOCK_TDC_EVENTS', str(tmp_path/'events.bin'))
    monkeypatch.setenv('MOCK_TDC_EMPTY_EVERY', '500')
    monkeypatch.chdir(tmp_path)
    try:
        lib = ctypes.CDLL(str(MODULE_DIR/'source'/'test'/'wrapper_read_TDC8HP_x64_mock.dll'))
    except OSError as exc:
        pytest.skip(f'mock DLL not loadable: {exc}')
    tdc = tdc_roentdek.TDC(lib, buf_size=1000, time_out=50)
    assert tdc.version >= 3
    assert tdc.init_tdc() == 0 and tdc.run_tdc() == 0
    got = []
    empty = 0
    while empty < 20:
        packet = tdc.read()
        if packet is None:
            empty += 1
            continue
        got.append(packet)
    tdc.stop_tdc()
    ts = np.concatenate([p[0] for p in got])
    cnt = np.concatenate([p[1] for p in got])
    hits = np.concatenate([p[2] for p in got])
    np.testing.assert_array_equal(ts, sample['time_stamp'])
    np.testing.assert_array_equal(cnt, sample['cnt'])
    mask = np.arange(10)[None, None, :] < sample['cnt'][:, :, None]
    np.testing.assert_array_equal(hits[mask], sample['tdc'][mask])
    assert tdc.diagnostics()['lmf_events'] == n
    # the LMF written by the DLL contains every hit of every group (read back with RoentDek's LMF_IO)
    import subprocess
    dump = tmp_path/'dump.bin'
    subprocess.run([str(MODULE_DIR/'tools'/'lmf_dump.exe'), str(tmp_path/'output.lmf'), str(dump)],
                   check=True, capture_output=True)
    back = np.fromfile(dump, dtype=record)
    np.testing.assert_array_equal(back['ts'], sample['time_stamp'])
    np.testing.assert_array_equal(back['cnt'], sample['cnt'])
    np.testing.assert_array_equal(back['tdc'][mask], sample['tdc'][mask])


@windows_only
def test_resort64c_reproduces_cobold_for_run198():
    from pyccapt.control.tdc_roentdek.roentdek_resort import ResortReconstructor
    sample = _sample()
    try:
        reco = ResortReconstructor(MODULE_DIR/'source'/'cobold_ccf'/'roentdek_resort_run198.cfg',
                                   bin_ns=float(sample['bin_ns']))
    except OSError as exc:
        pytest.skip('Resort64c not loadable (resort64c_VS2010_x64.dll from CoboldPC and the VC++ 2010 runtime '
                    f'are needed, see tdc_roentdek/README.md): {exc}')
    ions = reco.reconstruct(sample['cnt'], sample['tdc'], sample['time_stamp'])
    cob = sample['cobold'].astype(np.float64)
    n_hits = cob[:, 4].astype(int)
    rows = [(e, h, cob[e, 6+4*h], cob[e, 7+4*h], cob[e, 8+4*h], cob[e, 9+4*h] if 9+4*h < cob.shape[1] else -1)
            for e in range(len(cob)) for h in range(n_hits[e])]
    ref = {(e, h): (x, y, t, m) for e, h, x, y, t, m in rows}
    mine = {(int(e), int(h)): (x, y, t, m) for e, h, x, y, t, m in
            zip(ions['event'], ions['hit'], ions['x'], ions['y'], ions['t'], ions['method'])}
    common = sorted(set(ref) & set(mine))
    assert len(common) >= 0.99*len(ref)
    d_xy = np.array([np.hypot(mine[k][0]-ref[k][0], mine[k][1]-ref[k][1]) for k in common])
    d_t = np.array([abs(mine[k][2]-ref[k][2]) for k in common])
    same_method = np.array([mine[k][3] == ref[k][3] for k in common])
    assert np.mean(d_xy < 0.1) > 0.97          # mm
    assert np.mean(d_t < 0.03) > 0.95          # ns
    assert np.mean(same_method) > 0.97
    # groups without laser start still give x/y with TOF = NaN
    no_start = sample['cnt'][ions['event'], 6] == 0
    assert np.all(np.isnan(ions['t'][no_start])) and np.all(np.isfinite(ions['x'][no_start]))
