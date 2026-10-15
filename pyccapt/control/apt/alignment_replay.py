"""Read-only alignment journal metrics, with explicit optional replay truth."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def summarize_replay(records, *, expected_position_m=None, tolerance_um=1.):
    """Summarize one session; false-alignment metrics need independent truth.

    Laser ``observation`` counts are disjoint epochs. Sample snapshots overlap,
    so only sequence increments are counted, reported as a lower bound. These
    metrics never feed the physical controller or certify instrument safety.
    """
    records = list(records)
    aligned = [r for r in records if r.get('event') == 'aligned']
    observations = [r for r in records if r.get('event') == 'observation']
    elapsed = [float(r.get('elapsed_s', 0.)) for r in records]
    moves = [r for r in records if r.get('event') in ('move', 'move_requested')]
    positions = [np.asarray(r['target_m'], dtype=float) for r in moves if 'target_m' in r]
    travel = sum(float(np.linalg.norm(b-a))*1e6 for a, b in zip(positions, positions[1:]))
    counted = [r for r in observations if 'count' in r]
    if counted:
        ions = sum(int(r['count']) for r in counted)
        ion_scope = 'ions in disjoint measured laser epochs; excludes moves and discarded transition packets'
    else:
        sequences = [int(r['sequence']) for r in observations if 'sequence' in r]
        ions = max(sequences)-min(sequences) if sequences else 0
        ion_scope = 'lower bound from sample event sequence span; initial window not counted'
    false_aligned = None
    if expected_position_m is not None:
        truth = np.asarray(expected_position_m, dtype=float)
        if truth.shape != (3,) or not np.isfinite(truth).all() or not np.isfinite(tolerance_um) or tolerance_um <= 0:
            raise ValueError('Replay truth needs finite XYZ and positive tolerance')
        tested = [r for r in aligned if 'position_m' in r]
        if len(tested) != len(aligned):
            raise ValueError('Aligned records lack positions required to score replay truth')
        false_aligned = sum(np.linalg.norm(np.asarray(r['position_m'])-truth)*1e6 > tolerance_um for r in tested)
    return dict(aligned=bool(aligned), aligned_count=len(aligned),
                false_aligned_count=None if false_aligned is None else int(false_aligned),
                truth_available=expected_position_m is not None,
                elapsed_s=max(elapsed, default=0.)-min(elapsed, default=0.),
                measurement_windows=len(observations), observed_ions=ions, ion_scope=ion_scope,
                moves=len(moves), commanded_travel_um=travel,
                surface_fallbacks=sum(r.get('event') == 'response_surface' and not r.get('usable') for r in records),
                coverage_probes=sum(r.get('event') == 'coverage_probe' for r in records))


def read_sessions(path):
    """Group a JSONL journal by session; preserve malformed input as an error."""
    sessions = {}
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            sessions.setdefault(record.get('session', 'sample_experiment'), []).append(record)
    return sessions
