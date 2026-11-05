"""Detector-only search proposals; these helpers never authorize movement.

Rates are percentages measured at fixed DC/laser settings. A local log-quadratic
surface is useful only on a smooth, single-lobe response. Every proposal must be
bounded, settled and remeasured by the existing alignment state machine.
"""
from __future__ import annotations

import math
import numpy as np


def _design(xy):
    x, y = np.asarray(xy).T
    return np.column_stack((np.ones(len(x)), x, y, .5*x*x, x*y, .5*y*y))


def informative_cross(centre, lower, upper, radius_um, speed_um_s):
    """Centre/axes then informative diagonals, favouring shorter equal probes."""
    centre = np.asarray(centre, dtype=float)
    candidates = []
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        point = centre.copy()
        point[:2] = np.clip(centre[:2]+np.array((dx, dy))*radius_um*1e-6, lower[:2], upper[:2])
        if tuple(point) != tuple(centre) and tuple(point) not in candidates:
            candidates.append(tuple(point))
    result = [tuple(centre)]
    # Axial probes come first. Among equivalent geometric information gains,
    # travel time breaks ties. Diagonals identify the mixed curvature term.
    while candidates:
        def priority(point):
            xy = (np.asarray(result)[:, :2]-centre[:2])*1e6/radius_um
            candidate = (np.asarray(point[:2])-centre[:2])*1e6/radius_um
            matrix = _design(np.vstack((xy, candidate)))
            information = matrix.T@matrix+np.eye(6)*1e-6
            gain = np.linalg.slogdet(information)[1]
            axes = np.count_nonzero(np.abs(candidate) > 1e-8) == 1
            travel = np.max(np.abs(np.subtract(point, result[-1])))*1e6/speed_um_s
            return (axes and len(result) < 5, gain-.05*travel)
        point = max(candidates, key=priority)
        candidates.remove(point)
        result.append(point)
    return result


def fit_local_response(records, centre, radius_um, lower, upper, *, max_reduced_chi2=4., max_condition=1e5):
    """Return a bounded proposal and diagnostics, or an explicit fallback reason.

    Weighted log-rate regression approximates count uncertainty; leave-one-out
    residuals guard against a poor local shape. No prediction is an acceptance
    probability and an unobserved narrow lobe cannot be ruled out by this model.
    """
    def reject(reason, **details):
        return {'usable': False, 'reason': reason, **details}
    if len(records) < 8:
        return reject('insufficient independent positions')
    positions = np.array([r['position_m'][:2] for r in records])
    xy = (positions-np.asarray(centre[:2]))*1e6/radius_um
    if len(np.unique(np.round(xy, 8), axis=0)) != len(xy):
        return reject('repeated positions cannot identify curvature')
    rate = np.array([r['rate_percent'] for r in records])
    error = np.array([r['error_percent'] for r in records])
    if np.any(~np.isfinite(rate)) or np.any(~np.isfinite(error)) or np.any(rate <= 3*error) or np.any(rate <= 0):
        return reject('insufficient rate evidence for a local surface')
    if np.ptp([r['voltage'] for r in records]) > .1:
        return reject('observations have different DC voltages')
    sigma = np.maximum(error/rate, 1e-6)
    design = _design(xy)
    weighted = design/sigma[:, None]
    condition = float(np.linalg.cond(weighted))
    if np.linalg.matrix_rank(weighted) < 6 or not np.isfinite(condition) or condition > max_condition:
        return reject('poor geometric coverage', condition=condition if np.isfinite(condition) else None)
    beta = np.linalg.lstsq(weighted, np.log(rate)/sigma, rcond=None)[0]
    residual = (design@beta-np.log(rate))/sigma
    covariance = np.linalg.inv(weighted.T@weighted)
    leverage = np.sum((weighted@covariance)*weighted, axis=1)
    if np.any(leverage >= .995):
        return reject('surface depends on a single unverified probe')
    loo = residual/(1-leverage)
    reduced = float(residual@residual/(len(rate)-6))
    evidence = dict(condition=condition, reduced_chi2=reduced,
                    leave_one_out_rms_sigma=float(np.sqrt(np.mean(loo**2))))
    if reduced > max_reduced_chi2 or evidence['leave_one_out_rms_sigma'] > 4.:
        return reject('response is not explained by one local lobe', **evidence)
    hessian = np.array([[beta[3], beta[4]], [beta[4], beta[5]]])
    curvature_noise = 3*math.sqrt(float(covariance[3, 3]+2*covariance[4, 4]+covariance[5, 5])*max(1., reduced))
    if np.linalg.eigvalsh(hessian).max() >= -curvature_noise:
        return reject('flat, uncertain or non-concave response', **evidence)
    optimum = -np.linalg.solve(hessian, beta[1:3])
    if np.any(np.abs(optimum) > 1):
        return reject('predicted optimum lies outside probed range', **evidence)
    point = np.asarray(centre, dtype=float).copy()
    point[:2] += optimum*radius_um*1e-6
    if np.any(point < np.asarray(lower)-1e-14) or np.any(point > np.asarray(upper)+1e-14):
        return reject('predicted optimum lies outside permitted travel', **evidence)
    return dict(usable=True, reason='bounded local proposal requires remeasurement',
                position_m=point.tolist(), curvature_per_um2=(hessian/radius_um**2).tolist(),
                radius_um=float(radius_um), **evidence)


def coverage_probe(origin, lower, upper, visited, *, subdivisions=8, travel_weight=.05):
    """Choose the largest unsampled grid gap, with movement-cost tie breaking.

    The finite grid establishes an explicit coverage resolution, not a promise
    to find islands narrower than that spacing. All outputs remain at saved Z.
    """
    origin, lower, upper = map(lambda x: np.asarray(x, dtype=float), (origin, lower, upper))
    width = upper-lower
    if width.shape != (2,) or np.any(width <= 0):
        raise ValueError('Coverage search needs positive XY bounds')
    grid = np.array([(x, y) for x in np.linspace(lower[0], upper[0], subdivisions+1)
                     for y in np.linspace(lower[1], upper[1], subdivisions+1)])
    previous = np.asarray([p[:2] for p in visited], dtype=float).reshape(-1, 2)
    if not len(previous):
        previous = origin[None, :2]
    distances = np.linalg.norm((grid[:, None, :]-previous[None, :, :])/width, axis=2).min(axis=1)
    eligible = distances >= distances.max()*.95
    eligible &= distances > 1e-8
    if not eligible.any():
        return None
    travel = np.linalg.norm((grid-previous[-1])/width, axis=1)
    score = np.where(eligible, distances-travel_weight*travel, -np.inf)
    selected = int(np.argmax(score))
    return (*grid[selected], float(origin[2]))
