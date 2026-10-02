"""Sparse expanding search and repeatable neighbour-rate contrast at fixed DC."""
import math

import numpy as np


def expanding_positions(origin, ranges_um, first_um):
    origin = np.asarray(origin, dtype=float)
    ranges = np.asarray(ranges_um, dtype=float)
    yield tuple(origin)
    radius = min(float(first_um), float(max(ranges)))
    seen = {(0., 0.)}
    while True:
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1),
                      (1, 1), (-1, 1), (-1, -1), (1, -1)]
        for x, y in directions:
            offset = tuple(np.clip(np.array((x, y))*radius, -ranges, ranges))
            if offset in seen:
                continue
            seen.add(offset)
            yield (origin[0]+offset[0]*1e-6, origin[1]+offset[1]*1e-6, origin[2])
        if radius >= max(ranges):
            return
        radius = min(radius*2, float(max(ranges)))


def significant_jump(high, low, ratio, sigma):
    """Use ion-count uncertainty, including a nonzero zero-count noise floor."""
    if not high['coherent'] or high['rate'] <= 0:
        return False
    if abs(high['voltage']-low['voltage']) > .5:
        return False
    return (high['rate'] >= ratio*low['rate'] and
            high['rate']-low['rate'] >= sigma*math.hypot(high['error'], low['error']))


class RelativeSearch:
    def __init__(self, cfg):
        self.cfg = cfg
        self.records = []

    def observe(self, record):
        neighbours = [item for item in self.records
                      if (abs(item['voltage']-record['voltage']) <= .5 and
                          np.linalg.norm(np.subtract(item['position_m'][:2], record['position_m'][:2])) > 1e-10)]
        self.records = [item for item in self.records if item['position_m'] != record['position_m']]
        self.records.append(record)
        if not neighbours:
            return None
        # Always include the closest neighbour; farther low points cannot turn
        # a uniformly high local plateau into a false spatial jump.
        reference = min(neighbours, key=lambda item: np.linalg.norm(
            np.subtract(item['position_m'][:2], record['position_m'][:2])))
        high, low = (record, reference) if record['rate'] >= reference['rate'] else (reference, record)
        if significant_jump(high, low, self.cfg.jump_ratio, self.cfg.jump_sigma):
            return high, low
        return None
