"""Check a finished RoentDek experiment against its raw data.

    python -m pyccapt.control.tdc_roentdek.check_run <experiment folder>

Checks (PASS/FAIL):
  1. dld/ and tdc/ exist, have consistent lengths and types (dld x/y in cm, t in ns)
  2. tdc/ (all hits of all groups, start_counter = TDC time stamp) is identical to the
     output.lmf saved in the experiment folder (read with tools/lmf_dump.exe)
  3. dld/x, y, t equal a new Resort64c evaluation of the LMF with the configuration stored
     in the HDF5 file (within the rounding of the TDC bin size)
  4. plausibility: positions inside the detector, fraction of ions with TOF, channel hit rates

``pyccapt-data validate-hdf5`` checks the general schema; this tool checks the RoentDek data.
"""
import glob
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
RECORD = np.dtype([('ts', '<u8'), ('cnt', '<u4', (8,)), ('tdc', '<i4', (8, 10))])


def read_lmf(lmf_path):
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / 'dump.bin'
        subprocess.run([str(MODULE_DIR / 'tools' / 'lmf_dump.exe'), str(lmf_path), str(out)],
                       check=True, capture_output=True)
        return np.fromfile(out, dtype=RECORD)


def check(folder):
    import h5py
    from pyccapt.control.tdc_roentdek.roentdek_resort import ResortReconstructor
    from pyccapt.control.tdc_roentdek.tdc_roentdek import RAW_TIME_OFFSET

    results = []

    def report(name, ok, info=''):
        results.append(bool(ok))
        print(f"{'PASS' if ok else 'FAIL'} {name} {info}", flush=True)

    h5s = glob.glob(os.path.join(folder, '*.h5'))
    lmfs = glob.glob(os.path.join(folder, '*.lmf'))
    report('HDF5 file found', len(h5s) == 1, h5s[0] if h5s else '')
    report('LMF file found', len(lmfs) >= 1, lmfs[0] if lmfs else '')
    if not h5s:
        return 1
    f = h5py.File(h5s[0], 'r')
    dld = ['dld/x', 'dld/y', 'dld/t', 'dld/high_voltage', 'dld/start_counter', 'dld/reco_method']
    tdc = ['tdc/channel', 'tdc/time_data', 'tdc/start_counter']
    missing = [k for k in dld + tdc if k not in f]
    report('RoentDek datasets present', not missing, str(missing) if missing else '')
    if missing:
        return 1
    n_ion = len(f['dld/x'])
    report(f'dld/ lengths equal ({n_ion} ions)', len({len(f[k]) for k in dld}) == 1)
    report(f'tdc/ lengths equal ({len(f["tdc/channel"])} hits)', len({len(f[k]) for k in tdc}) == 1)
    report('dld/x, y, t are float64', all(f[k].dtype == np.float64 for k in dld[:3]))
    channel = f['tdc/channel'][:]
    times = f['tdc/time_data'][:].astype(np.int64) - RAW_TIME_OFFSET
    start_counter = f['tdc/start_counter'][:]

    if lmfs:
        raw = read_lmf(lmfs[0])
        mask = np.arange(10)[None, None, :] < np.minimum(raw['cnt'], 10)[:, :, None]
        group, ch, _ = np.nonzero(mask)
        same = (len(ch) == len(channel) and np.array_equal(ch, channel) and np.array_equal(raw['tdc'][mask], times)
                and np.array_equal(raw['ts'][group], start_counter))
        report(f'tdc/ identical to the LMF ({len(raw)} groups, {len(ch)} hits)', same)
        cfg_text = f['dld'].attrs.get('roentdek_resort_cfg', None)
        reconstruction = str(f['dld'].attrs.get('roentdek_reconstruction', ''))
        if reconstruction and not reconstruction.startswith('Resort64c'):
            print(f'SKIP dld/ vs. Resort64c: the run used {reconstruction}')
            cfg_text = None
        if cfg_text is not None:
            with tempfile.TemporaryDirectory() as directory:
                cfg = Path(directory) / 'roentdek_resort.cfg'
                lines = []
                for line in str(cfg_text).splitlines():
                    if line.startswith('correction_table'):
                        rel = line.split(None, 1)[1].strip()
                        line = 'correction_table ' + (rel if os.path.isabs(rel) else str(MODULE_DIR / rel))
                    lines.append(line)
                cfg.write_text('\n'.join(lines) + '\n')
                try:
                    reco = ResortReconstructor(str(cfg))
                except OSError as exc:
                    reco = None
                    print(f'SKIP dld/ vs. Resort64c: Resort64c not loadable on this PC ({exc}); '
                          'copy resort64c_VS2010_x64.dll from CoboldPC, see tdc_roentdek/README.md')
                parts = [] if reco is None else [
                    reco.reconstruct(raw['cnt'][i:i + 30000], raw['tdc'][i:i + 30000], raw['ts'][i:i + 30000])
                    for i in range(0, len(raw), 30000)]
            if reco is not None:
                x = np.concatenate([p['x'] for p in parts]) / 10. if parts else np.zeros(0)
                y = np.concatenate([p['y'] for p in parts]) / 10. if parts else np.zeros(0)
                t = np.concatenate([p['t'] for p in parts]) if parts else np.zeros(0)
                ok = (len(x) == n_ion and np.allclose(x, f['dld/x'][:], rtol=0, atol=5e-4)
                      and np.allclose(y, f['dld/y'][:], rtol=0, atol=5e-4)
                      and np.allclose(t, f['dld/t'][:], rtol=2e-5, atol=1e-3, equal_nan=True))
                report('dld/ equal to a new Resort64c evaluation of the LMF', ok,
                       '' if ok else f'(new {len(x)} ions, file {n_ion})')

    if n_ion:
        x, y, t = f['dld/x'][:], f['dld/y'][:], f['dld/t'][:]
        r_cm = np.hypot(x, y)
        report('positions inside the detector (r <= 4.1 cm)', r_cm.max() <= 4.1, f'r max {r_cm.max():.2f} cm')
        finite = np.isfinite(t)
        print(f'     ions {n_ion}, with TOF {finite.sum()} ({100 * finite.mean():.1f} %), '
              f'without laser start (TOF = NaN) {(~finite).sum()}')
        if not finite.any():
            print('     WARNING: no ion has a TOF -> no laser start signal (Cobold channel 7)')
        hit_groups = np.zeros(8)
        if len(start_counter):
            groups = np.r_[0, np.nonzero(np.diff(start_counter.astype(np.int64)))[0] + 1]
            n_groups = len(groups)
            for c in range(8):
                hit_groups[c] = len(np.unique(start_counter[channel == c])) / n_groups
            print('     groups with a hit per Cobold channel 1..8: ' + ' '.join(f'{100 * v:.0f}%' for v in hit_groups))
    print(f"\n{'ALL OK' if all(results) else 'PROBLEMS FOUND'}: {sum(results)} of {len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    sys.exit(check(sys.argv[1]))
