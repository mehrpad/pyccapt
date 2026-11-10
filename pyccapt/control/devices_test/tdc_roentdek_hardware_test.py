"""Stand-alone RoentDek TDC8HP test (no GUI, no high voltage control).

    python -m pyccapt.control.devices_test.tdc_roentdek_hardware_test [seconds]

Prints every second what the wrapper DLL receives from the driver, the first groups,
per-channel hit rates, measured time sums and the Resort64c reconstruction result.
output.lmf is written in pyccapt/control/tdc_roentdek, like during an experiment.
"""
import ctypes
import os
import sys
import time

import numpy as np

from pyccapt.control.tdc_roentdek import tdc_roentdek
from pyccapt.control.tdc_roentdek.roentdek_resort import make_reconstructor


def main(seconds=20.0):
    old_cwd = os.getcwd()
    os.chdir(tdc_roentdek.MODULE_DIR)
    try:
        if not tdc_roentdek.check_tdc_files():
            return 1
        tdc = tdc_roentdek.TDC(ctypes.CDLL(str(tdc_roentdek.MODULE_DIR / tdc_roentdek.DLL_NAME)))
        print('wrapper DLL version', tdc.version)
        rc = tdc.init_tdc()
        print('init_tdc ->', rc, tdc.last_error())
        if rc != 0:
            return 1
        print('TDC info:', tdc.tdc_info(), ' bin size %.7f ns' % tdc.bin_size_ns())
        print('run_tdc ->', tdc.run_tdc(), tdc.last_error())
        reco = make_reconstructor(bin_ns=tdc.bin_size_ns() or None)
        counts, hits_all = [], []
        ions = shown = 0
        t0 = last = time.time()
        try:
            while time.time() - t0 < seconds:
                packet = tdc.read()
                if packet is not None:
                    ts, cnt, hits = packet
                    counts.append(cnt)
                    hits_all.append(hits)
                    ions += len(reco.reconstruct(cnt, hits, ts)['event'])
                    while shown < 5 and shown < len(cnt):
                        print(f'--- group {shown}: ' + ', '.join(
                            f'ch{c}: {np.round(hits[shown, c, :cnt[shown, c]] * reco.bin_ns, 1).tolist()}'
                            for c in range(8)))
                        shown += 1
                if time.time() - last >= 1.:
                    last = time.time()
                    d = tdc.diagnostics()
                    print(f"{last - t0:.0f} s | DLL reads {d.get('read_calls')} non-empty {d.get('read_nonzero')} "
                          f"words {d.get('words')} groups {d.get('groups_ok')} | ions {ions}")
        finally:
            tdc.stop_tdc()
        print('last driver error:', tdc.last_error() or '-')
        print('reconstruction:', reco.stats_text())
        for warning in reco.signal_warnings():
            print('WARNING:', warning)
        if not counts:
            print('NO TDC GROUPS RECEIVED')
            return 1
        cnt = np.concatenate(counts)
        hits = np.concatenate(hits_all).astype(float) * reco.bin_ns
        print('groups with a hit per Cobold channel 1..8:', np.round((cnt > 0).mean(0), 3))
        complete = np.all(cnt[:, :6] == 1, axis=1) & (cnt[:, 7] > 0)
        for name, (a, b) in (('u', (0, 1)), ('v', (2, 3)), ('w', (4, 5))):
            if complete.sum() > 10:
                s = hits[complete, a, 0] + hits[complete, b, 0] - 2 * hits[complete, 7, 0]
                print(f'time sum {name}: median {np.median(s):.2f} ns')
        return 0
    finally:
        os.chdir(old_cwd)


if __name__ == '__main__':
    sys.exit(main(float(sys.argv[1]) if len(sys.argv) > 1 else 20.0))
