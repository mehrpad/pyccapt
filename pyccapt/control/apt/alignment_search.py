"""Bounded two-scale probes and repeatable neighbour-rate contrast at fixed DC."""
import math

import numpy as np


def range_positions(origin, ranges_um, lower_xy=None, upper_xy=None, first_um=None):
    """Origin plus eight directions at two scales; at most 16 lateral probes.

    Each axis scales with its own range. Optional clipping keeps a local second
    pass inside the original sample envelope, with duplicate targets removed.
    """
    origin = np.asarray(origin, dtype=float)
    ranges = np.asarray(ranges_um, dtype=float)
    first = .5 if first_um is None else min(.5, float(first_um)/float(max(ranges)))
    seen = set()
    for scale in (0., first, 1.):
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1),
                      (1, 1), (-1, 1), (-1, -1), (1, -1)]
        for x, y in directions:
            target = origin.copy()
            target[:2] += np.array((x, y))*ranges*scale*1e-6
            if lower_xy is not None:
                target[:2] = np.clip(target[:2], lower_xy, upper_xy)
            key = tuple(target)
            if key in seen:
                continue
            seen.add(key)
            yield key


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
        position = tuple(record.get('target_m', record['position_m']))
        def target(item):
            return tuple(item.get('target_m', item['position_m']))
        neighbours = [item for item in self.records
                      if (abs(item['voltage']-record['voltage']) <= .5 and
                          np.linalg.norm(np.subtract(target(item)[:2], position[:2])) > 1e-10)]
        self.records = [item for item in self.records if target(item) != position]
        self.records.append(record)
        if not neighbours:
            return None
        # Always include the closest neighbour; farther low points cannot turn
        # a uniformly high local plateau into a false spatial jump.
        reference = min(neighbours, key=lambda item: np.linalg.norm(
            np.subtract(target(item)[:2], position[:2])))
        high, low = (record, reference) if record['rate'] >= reference['rate'] else (reference, record)
        if significant_jump(high, low, self.cfg.jump_ratio, self.cfg.jump_sigma):
            return high, low
        return None
