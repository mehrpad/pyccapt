"""Conservative near-axis ICF/kf ratio profiling with cached full-event depth.

Adapted in concept from local APT3D near_axis_depth_ratio.py and the paraxial
comparator in exact_single_pole_phase.py. Our q = ICF/kf is the reciprocal of
APT3D's kf/ICF. No APT3D runtime dependency or unique-pair inference is used.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from pyccapt.calibration.core.exceptions import CalibrationInputError, CalibrationStateError
from pyccapt.calibration.reconstructions.input_validation import validate_reconstruction_inputs
from pyccapt.calibration.reconstructions.sdm_calibration import (
    active_selection_mask, build_hv, validate_full_source_metadata,
)


def _hash_source(variables):
    digest = hashlib.sha256(str(getattr(variables, 'pulse_mode', None)).encode())
    for values in (variables.dld_x_det, variables.dld_y_det, build_hv(variables)):
        array = np.ascontiguousarray(values, dtype=np.float64)
        digest.update(str(array.shape).encode())
        digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def _readonly(values, dtype=float):
    result = np.array(values, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def selection_hash(variables):
    """Fingerprint the analysis selection separately from acquisition coverage."""
    mask = active_selection_mask(variables)
    return None if mask is None else hashlib.sha256(mask.tobytes()).hexdigest()


@dataclass(frozen=True)
class RatioGeometry:
    """Full prefix is computed before selecting a species, ROI or sample."""
    cumulative_inverse_voltage2: np.ndarray
    selected_indices: np.ndarray
    base_depth_nm: np.ndarray
    detector_x_cm: np.ndarray
    detector_y_cm: np.ndarray
    voltage_V: np.ndarray
    bands: np.ndarray
    metadata: dict


def prepare_ratio_geometry(variables, roi_center_cm, roi_radius_cm, *, det_eff,
                           field_evap_v_nm, atomic_density_nm3, flight_path_mm,
                           full_acquisition_confirmed=False, event_mask=None,
                           depth_bands=6, max_events_per_band=2500, sample_seed=0):
    """Cache full depth, then sample uniformly without replacement within each band.

    The caller must confirm event-zero/full acquisition coverage. A detected
    crop against Variables.data_backup is rejected; an already cropped input
    cannot be certified from its numbers alone, so provenance records the claim.
    Detector area is the full observed circular extent, matching current PyCCAPT.
    Seeded random sampling avoids the artificial periodic comb created by a
    regular event-index stride. Selected events retain their acquisition order.
    """
    if not full_acquisition_confirmed:
        raise CalibrationInputError('Confirm the loaded source contains the full acquisition from event zero before profiling')
    validate_full_source_metadata(variables)
    x, y, voltage = validate_reconstruction_inputs(
        variables.dld_x_det, variables.dld_y_det, build_hv(variables), flight_path_mm,
        1., det_eff, 1., field_evap_v_nm, atomic_density_nm3)
    if getattr(variables, 'data_backup', None) is not None and len(variables.data_backup) > len(voltage):
        raise CalibrationInputError('A cropped source was detected; reload the full acquisition before computing its depth prefix')
    center = np.asarray(roi_center_cm, dtype=float)
    if center.shape != (2,) or not np.all(np.isfinite(center)) or not np.isfinite(roi_radius_cm) or roi_radius_cm <= 0:
        raise CalibrationInputError('ROI center must be two finite cm coordinates and radius positive')
    if not isinstance(depth_bands, int) or depth_bands < 4 or depth_bands % 2:
        raise CalibrationInputError('depth_bands must be an even integer >=4')
    if not isinstance(max_events_per_band, int) or max_events_per_band < 50:
        raise CalibrationInputError('max_events_per_band must be an integer >=50')
    if not isinstance(sample_seed, (int, np.integer)) or sample_seed < 0:
        raise CalibrationInputError('sample_seed must be a nonnegative integer')
    area = float(np.pi * np.nanmax(np.hypot(x*10., y*10.))**2)
    # This is intentionally BEFORE every mask and subsample.
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        prefix = np.cumsum(1. / voltage**2)
        coefficient = flight_path_mm**2 * field_evap_v_nm**2 / (atomic_density_nm3 * det_eff * area)
    if not np.all(np.isfinite(prefix)) or not np.isfinite(coefficient) or coefficient <= 0:
        raise CalibrationInputError('Source voltage or geometry cannot form a finite positive depth prefix')
    active = active_selection_mask(variables)
    first, stop = 0, len(voltage)
    selected = np.isfinite(x) & np.isfinite(y) & (np.hypot(x-center[0], y-center[1]) <= roi_radius_cm)
    if active is not None:
        active = np.array(active, copy=True)
        first, stop = int(np.argmax(active)), len(active)-int(np.argmax(active[::-1]))
        selected &= active
    if event_mask is not None:
        event_mask = np.asarray(event_mask)
        if event_mask.dtype != bool or event_mask.shape != selected.shape:
            raise CalibrationInputError('event_mask must be an event-aligned boolean mask; it never filters the prefix')
        selected &= event_mask
    # Bands span the active acquisition interval, using ORIGINAL source indices.
    # Internal gaps remain gaps. Neither active nor species selection changes H.
    all_bands = np.clip((np.arange(len(voltage))-first) * depth_bands // (stop-first), 0, depth_bands-1)
    indices, available = [], []
    rng = np.random.default_rng(sample_seed)
    for band in range(depth_bands):
        group = np.flatnonzero(selected & (all_bands == band))
        available.append(int(len(group)))
        if len(group) > max_events_per_band:
            group = np.sort(rng.choice(group, size=max_events_per_band, replace=False))
        indices.extend(group)
    indices = np.asarray(indices, dtype=int)
    metadata = dict(source_hash=_hash_source(variables), source_events=len(voltage),
                    source_coverage='user_declared_full_acquisition_from_event_zero',
                    pulse_mode=getattr(variables, 'pulse_mode', None),
                    hv_convention='DC plus pulse voltage in voltage mode, DC otherwise; current reconstruction convention',
                    sampled_event_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                    event_mask_sha256=None if event_mask is None else hashlib.sha256(event_mask.tobytes()).hexdigest(),
                    active_selection_sha256=None if active is None else hashlib.sha256(active.tobytes()).hexdigest(),
                    active_selection_events=len(voltage) if active is None else int(active.sum()),
                    analysis_source_index_span=[first, stop],
                    band_convention='equal original-event intervals over active first-to-last source index; internal gaps retained',
                    detector_partial_events=int(np.count_nonzero(~np.isfinite(x) | ~np.isfinite(y))),
                    depth_bands=depth_bands, band_available_events=available,
                    detector_area_mm2=area, area_convention='full observed circular detector extent',
                    full_detector_radius_cm=float(np.sqrt(area/np.pi)/10.),
                    det_eff=float(det_eff), field_evap=float(field_evap_v_nm), avg_dens=float(atomic_density_nm3),
                    flight_path_length=float(flight_path_mm), roi_center_cm=center.tolist(),
                    roi_radius_cm=float(roi_radius_cm), depth_coefficient_nm_V2=float(coefficient),
                    max_events_per_band=max_events_per_band, sample_seed=int(sample_seed),
                    sampling_method='uniform_without_replacement_per_band_then_acquisition_sort',
                    full_prefix_cached_before_selection=True)
    return RatioGeometry(_readonly(prefix), _readonly(indices, int), _readonly(coefficient*prefix[indices]),
                         _readonly(x[indices]), _readonly(y[indices]), _readonly(voltage[indices]),
                         _readonly(all_bands[indices], int), metadata)


def ratio_next_steps(result):
    """Actionable abstention guidance; no threshold relaxation or guessed identity."""
    status = result['status']
    if status == 'supported':
        return ['Use an independently justified ICF anchor to select one compatible family member; this profile does not infer it.']
    steps = {
        'source_or_geometry_unavailable': ['Load the full original event sequence and retain crop/species selections as masks; confirm coverage separately.'],
        'insufficient_events': ['Inspect counts in each acquisition band. Use a justified larger indexed ROI or a longer contiguous interval; keep the full prefix.'],
        'grid_underresolved': ['Narrow q bounds using independent information or explicitly increase the grid cap; do not treat a coarse maximum as evidence.'],
        'geometry_unsupported': ['Use a tighter independently indexed near-axis ROI or a justified narrower ICF interval. Off-axis poles require a geometry model validated for that pole.'],
        'insufficient_depth_span': ['Include a longer acquisition interval containing at least five indexed spacings in every band.'],
        'no_periodic_evidence': [
            'Inspect the detector map and choose an independently indexed pole. Detector center and density extrema are only ROI candidates.',
            'Confirm the phase, lattice parameter and plane order for that ROI; defaults are not measured spacing.',
            'Inspect individual-band evidence. Weak maxima in every band cannot establish a ratio; mixed strong/weak bands can indicate changing structure, geometry or sampling.'],
        'heldout_disagreement': ['A single ratio did not transfer between acquisition intervals. Inspect independent segments for changing structure or geometry before considering a time-dependent model.'],
        'flat': ['The profile does not localize q. Add independent spacing/geometry evidence; retain manual parameters.'],
        'boundary': ['A boundary maximum does not identify q. Review independent parameter bounds and spacing before expanding a search.'],
        'ambiguous': ['Competing branches require another independent spacing or angular reference; do not choose a branch only because it is highest.'],
    }.get(status, [])
    return steps + ['Further ROI, spacing or q searches are exploratory; freeze those choices before evaluating fresh confirmation data.']


def _band_diagnostics(q, scores, coherence, counts, chosen, floors, minimum_score):
    maxima = np.argmax(coherence, axis=1)
    rows = []
    for band, peak in enumerate(maxima):
        rows.append(dict(band=int(band), role='training' if band % 2 == 0 else 'held_out',
            events=int(counts[band]), coherence_floor=float(floors[band]),
            exploratory_peak_q=float(q[peak]), exploratory_peak_coherence=float(coherence[band, peak]),
            exploratory_peak_contrast=float(scores[band, peak]),
            exploratory_peak_passes_gates=bool(coherence[band, peak] >= floors[band] and scores[band, peak] >= minimum_score),
            coherence_at_training_choice=float(coherence[band, chosen]), contrast_at_training_choice=float(scores[band, chosen])))
    return dict(bands=rows,
        interpretation='Per-band maxima are exploratory diagnostics, not independent estimates or corrected significance tests. Selection and acceptance still use the frozen training choice.',
        bands_with_exploratory_signal=sum(row['exploratory_peak_passes_gates'] for row in rows))


def periodic_band_evidence(coordinate_nm, spacing_nm):
    """Absolute exact-period coherence against neighboring frequencies; no q normalization."""
    values = np.asarray(coordinate_nm, dtype=float)
    phase = (2*np.pi/spacing_nm) * (values - np.median(values))
    exact = float(abs(np.mean(np.exp(1j*phase))))
    side = [abs(np.mean(np.exp(1j*phase*ratio))) for ratio in (.82, .90, 1.10, 1.18)]
    noise = 1./np.sqrt(len(values))
    return float(np.log((exact+noise)/(np.median(side)+noise))), exact


def _launch(x_cm, y_cm, icf, flight_path_mm, mode):
    radius = np.hypot(x_cm, y_cm)*10.
    observed = np.arctan2(radius, flight_path_mm)
    argument = icf*radius/flight_path_mm if mode == 'Bas' else (icf-1)*np.sin(observed)
    if np.any(np.abs(argument) > 1):
        raise CalibrationInputError('The ICF family lies outside the projection domain')
    theta = np.arcsin(argument) + (0. if mode == 'Bas' else observed)
    if np.any(theta >= np.pi/2):
        raise CalibrationInputError('The ICF family crosses the apex horizon')
    phi = np.arctan2(y_cm, x_cm)
    return np.column_stack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)))


def ratio_geometry_error(geometry, q_values, icf_bounds, spacing_nm, mode):
    """Bound omitted legacy-projection terms on the declared family, by depth band.

    Uses the corrected fixed-origin term R*(p_z-u.p), NOT R*(1-u.p).
    A constant per-band phase is irrelevant; varying omitted terms are not.
    """
    m = geometry.metadata
    worst, cosine_bias = 0., 0.
    for icf in np.linspace(*icf_bounds, 5):
        # Acceptance reconstructs every event, not just the central pole ROI.
        # A central profile cannot certify an ICF that puts outer hits beyond
        # the projection domain or onto the rear side of the spherical cap.
        try:
            _launch(np.array([m['full_detector_radius_cm']]), np.array([0.]),
                    icf, m['flight_path_length'], mode)
        except CalibrationInputError as exc:
            raise CalibrationInputError(
                'The ICF family lies outside the full-detector projection domain; '
                'review the flight length, detector units or narrow the justified ICF family') from exc
        direction = _launch(geometry.detector_x_cm, geometry.detector_y_cm, icf, m['flight_path_length'], mode)
        pole = _launch(np.array([m['roi_center_cm'][0]]), np.array([m['roi_center_cm'][1]]),
                       icf, m['flight_path_length'], mode)[0]
        cosine_bias = max(cosine_bias, abs(np.sqrt(pole[2])-1.))
        for q in (q_values[0], q_values[-1]):
            depth = geometry.base_depth_nm/q**2
            radius = geometry.voltage_V*q/(icf*m['field_evap'])
            correction = (pole[2]-1.)*depth + radius*(pole[2]-direction@pole)
            for band in range(m['depth_bands']):
                values = correction[geometry.bands == band]
                if len(values):
                    worst = max(worst, float(np.ptp(values))/spacing_nm)
    return dict(max_omitted_phase_span_in_periods=worst, max_pole_cosine_relative_q_bias=cosine_bias,
                method='sampled family-endpoint and five-ICF geometry check; not an external metrology bound')


def _profile_peak(q, profile, drop):
    peak = int(np.argmax(profile))
    support = profile >= profile[peak]-drop
    components = np.flatnonzero(np.diff(np.r_[False, support, False]))
    interval = [float(q[peak]), float(q[peak])]
    for first, end in zip(components[::2], components[1::2]):
        if first <= peak < end:
            interval = [float(q[first]), float(q[end-1])]
            threshold = profile[peak]-drop
            if first > 0:
                interval[0] = float(np.interp(threshold, profile[first-1:first+1], q[first-1:first+1]))
            if end < len(q):
                interval[1] = float(np.interp(threshold, profile[end-1:end+1][::-1], q[end-1:end+1][::-1]))
    if np.ptp(profile) < drop*2:
        status = 'flat'
    elif peak in (0, len(q)-1) or support[0] or support[-1]:
        status = 'boundary'
    elif len(components) > 2:
        status = 'ambiguous'
    else:
        status = 'interior'
    return peak, interval, status


def profile_ratio_geometry(geometry, spacing_nm, q_values, *, icf_bounds=(1.1, 1.9), mode='Geiser',
                           minimum_events_per_band=100, profile_drop=.10, minimum_score=.35,
                           maximum_phase_span=.15, maximum_cosine_q_bias=.005,
                           maximum_heldout_disagreement=.02, bootstrap_repeats=200, seed=0,
                           max_grid_points=4001, progress=None):
    """Profile q=ICF/kf on even bands; validate the fixed choice on odd bands.

    Supported intervals are descriptive score-drop regions, NOT confidence
    intervals or probabilities. Bootstrap ranges resample training bands only.
    Noise/evidence gates are explicit conservative heuristics, not null p-values.
    """
    q = np.asarray(q_values, dtype=float)
    bounds = np.asarray(icf_bounds, dtype=float)
    if q.ndim != 1 or len(q) < 9 or np.any(~np.isfinite(q) | (q <= 0)) or np.any(np.diff(q) <= 0):
        raise CalibrationInputError('q=ICF/kf grid must contain >=9 strictly increasing positive finite values')
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] <= 0 or bounds[1] < bounds[0]:
        raise CalibrationInputError('ICF family bounds must be positive finite and ordered')
    if not np.isfinite(spacing_nm) or spacing_nm <= 0 or mode not in ('Geiser', 'Bas'):
        raise CalibrationInputError('Supply positive indexed spacing in nm and mode Geiser or Bas')
    if (minimum_events_per_band < 50 or not isinstance(bootstrap_repeats, int) or bootstrap_repeats < 0
            or not isinstance(max_grid_points, int) or max_grid_points < 9
            or not np.isfinite([profile_drop, minimum_score, maximum_phase_span, maximum_cosine_q_bias,
                               maximum_heldout_disagreement]).all()
            or min(profile_drop, minimum_score, maximum_phase_span, maximum_cosine_q_bias, maximum_heldout_disagreement) <= 0):
        raise CalibrationInputError('Evidence thresholds must be finite and positive, with >=50 events per band')
    metadata = dict(geometry.metadata, q_convention='ICF/kf', spacing_nm=float(spacing_nm),
                    icf_bounds=bounds.tolist(), mode=mode, profile_drop=float(profile_drop),
                    minimum_score=float(minimum_score), minimum_events_per_band=int(minimum_events_per_band),
                    maximum_phase_span=float(maximum_phase_span), maximum_cosine_q_bias=float(maximum_cosine_q_bias),
                    maximum_heldout_disagreement=float(maximum_heldout_disagreement))
    result = dict(status='insufficient_events', q=None, supported_interval=None, q_grid=q.tolist(),
                  metadata=metadata, family=None, diagnostics={}, training_profile=[], heldout_profile=[],
                  interpretation='conditional near-axis ratio support, not a unique pair or probability')
    band_indices = [np.flatnonzero(geometry.bands == band) for band in range(metadata['depth_bands'])]
    result['diagnostics']['sampled_events_per_band'] = [len(indices) for indices in band_indices]
    if min(len(indices) for indices in band_indices) < minimum_events_per_band:
        result['message'] = 'Too few ROI events in one or more acquisition bands; enlarge a central ROI or use more data.'
        return result
    # H varies as q^-2: keep the phase-span error at a half-grid offset below
    # pi/2. A coarse search can otherwise miss a narrow true coherence maximum.
    max_base_span = max(float(np.ptp(geometry.base_depth_nm[indices])) for indices in band_indices)
    allowed_step = q[0]**3*spacing_nm/(4.*max_base_span) if max_base_span > 0 else np.inf
    required_points = max(9, int(np.ceil((q[-1]-q[0])/allowed_step))+1)
    result['diagnostics']['grid_resolution'] = dict(requested_points=len(q), required_points=required_points,
                                                   maximum_q_step=float(allowed_step))
    if np.max(np.diff(q)) > allowed_step:
        if required_points > max(max_grid_points, len(q)):
            result.update(status='grid_underresolved', message=f'At least {required_points} grid points are needed for this depth span; narrow justified q bounds or explicitly increase the grid cap.')
            return result
        q = np.linspace(q[0], q[-1], required_points)
        result['q_grid'] = q.tolist()
    try:
        error = ratio_geometry_error(geometry, q, bounds, spacing_nm, mode)
    except CalibrationInputError as exc:
        result.update(status='geometry_unsupported', message=str(exc))
        return result
    result['diagnostics']['geometry'] = error
    if error['max_omitted_phase_span_in_periods'] > maximum_phase_span or error['max_pole_cosine_relative_q_bias'] > maximum_cosine_q_bias:
        result.update(status='geometry_unsupported', message='Select a tighter near-axis indexed pole or narrower justified ICF family; use manual reconstruction for off-axis geometry.')
        return result
    scores = np.empty((len(band_indices), len(q)))
    coherence = np.empty_like(scores)
    if progress is not None:
        progress(0, len(q))  # Permit cooperative cancellation before the first profile point.
    for j, ratio in enumerate(q):
        depth = geometry.base_depth_nm/ratio**2
        for band, indices in enumerate(band_indices):
            scores[band, j], coherence[band, j] = periodic_band_evidence(depth[indices], spacing_nm)
        if progress is not None:
            progress(j+1, len(q))
    # Select the physical exact-period maximum. A sideband denominator minimum
    # can shift a log-ratio maximum, especially in short depth bands; use that
    # contrast only as an independent evidence gate, never as the ruler.
    train = np.median(coherence[::2], axis=0)
    held = np.median(coherence[1::2], axis=0)
    result.update(training_profile=train.tolist(), heldout_profile=held.tolist())
    peak, interval, status = _profile_peak(q, train, profile_drop)
    result['diagnostics']['grid_peak_q'] = float(q[peak])
    evidence_floor = np.array([max(.08, 4./np.sqrt(len(indices))) for indices in band_indices])
    result['diagnostics']['coherence_floor'] = evidence_floor.tolist()
    result['diagnostics']['exact_coherence_at_training_choice'] = coherence[:, peak].tolist()
    result['diagnostics']['scores_at_training_choice'] = scores[:, peak].tolist()
    result['diagnostics']['band_evidence'] = _band_diagnostics(
        q, scores, coherence, [len(indices) for indices in band_indices], peak, evidence_floor, minimum_score)
    periods = [float(np.ptp(geometry.base_depth_nm[indices]/q[peak]**2)/spacing_nm) for indices in band_indices]
    result['diagnostics']['period_span_per_band_at_training_choice'] = periods
    if min(periods) < 5:
        result.update(status='insufficient_depth_span', message='Every acquisition band needs at least five indexed periods for a ratio profile.')
        return result
    if np.any(coherence[::2, peak] < evidence_floor[::2]) or np.any(scores[::2, peak] < minimum_score):
        result.update(status='no_periodic_evidence', message='Indexed-period coherence does not exceed the declared evidence gates in every training band.')
        return result
    if status != 'interior':
        result.update(status=status, message='The ratio profile is flat, has competing branches, or reaches its search boundary. Review spacing and justified search bounds.')
        return result
    held_q = q[np.argmax(coherence[1::2], axis=1)]
    disagreement = np.abs(held_q/q[peak]-1.)
    result['diagnostics']['heldout_peak_q'] = held_q.tolist()
    result['diagnostics']['heldout_relative_disagreement'] = disagreement.tolist()
    if (np.any(coherence[1::2, peak] < evidence_floor[1::2]) or np.any(scores[1::2, peak] < minimum_score)
            or np.any(disagreement > maximum_heldout_disagreement)):
        result.update(status='heldout_disagreement', message='The training-selected ratio does not transfer across acquisition bands; no static ratio is accepted.')
        return result
    rng = np.random.default_rng(seed)
    train_scores = coherence[::2]
    peaks = [q[np.argmax(np.median(train_scores[rng.integers(0, len(train_scores), len(train_scores))], axis=0))]
             for _ in range(bootstrap_repeats)]
    result['diagnostics']['training_band_bootstrap'] = dict(
        repeats=bootstrap_repeats, seed=int(seed),
        percentile_range=None if not peaks else np.quantile(peaks, [.025, .975]).tolist(),
        interpretation='training-band resampling spread; not calibrated coverage or probability')
    result.update(status='supported', q=float(q[peak]), supported_interval=interval,
                  message='A conditional ICF/kf ratio is supported. Choose an independent/manual ICF anchor before applying one compatible family member.',
                  family=dict(relation='kf = ICF/q', icf_bounds=bounds.tolist(),
                              kf_at_icf_bounds=(bounds/q[peak]).tolist()))
    return result


def profile_icf_kf_ratio(variables, roi_center_cm, roi_radius_cm, spacing_nm, q_values, *,
                         det_eff, field_evap_v_nm, atomic_density_nm3, flight_path_mm,
                         full_acquisition_confirmed=False, event_mask=None, depth_bands=6,
                         max_events_per_band=2500, sample_seed=0, **profile_options):
    """User-facing profile with useful abstention for missing full-source coverage."""
    try:
        geometry = prepare_ratio_geometry(variables, roi_center_cm, roi_radius_cm, det_eff=det_eff,
            field_evap_v_nm=field_evap_v_nm, atomic_density_nm3=atomic_density_nm3, flight_path_mm=flight_path_mm,
            full_acquisition_confirmed=full_acquisition_confirmed, event_mask=event_mask,
            depth_bands=depth_bands, max_events_per_band=max_events_per_band, sample_seed=sample_seed)
    except CalibrationInputError as exc:
        return dict(status='source_or_geometry_unavailable', q=None, supported_interval=None,
                    message=str(exc), metadata={}, diagnostics={}, training_profile=[], heldout_profile=[])
    return profile_ratio_geometry(geometry, spacing_nm, q_values, **profile_options)


def compatible_parameter_family(result, icf_values):
    """Return compatible pairs/ranges; ICF values are anchors, not predictions."""
    if result['status'] != 'supported':
        raise CalibrationStateError('No supported ratio is available')
    icf = np.atleast_1d(np.asarray(icf_values, dtype=float))
    low, high = result['metadata']['icf_bounds']
    if np.any(~np.isfinite(icf) | (icf < low) | (icf > high)):
        raise CalibrationInputError('ICF anchors must lie within the validated family interval')
    qlow, qhigh = result['supported_interval']
    return [dict(icf=float(value), kf=float(value/result['q']),
                 kf_profile_interval=[float(value/qhigh), float(value/qlow)]) for value in icf]


def accept_ratio_family(variables, result, *, icf_anchor):
    """Explicitly accept one anchored family member into shared reconstruction settings."""
    pair = compatible_parameter_family(result, [icf_anchor])[0]
    if _hash_source(variables) != result['metadata']['source_hash']:
        raise CalibrationStateError('Acquisition arrays changed after profiling; rerun the ratio profile')
    if selection_hash(variables) != result['metadata'].get('active_selection_sha256'):
        raise CalibrationStateError('Active reconstruction selection changed after profiling; rerun the ratio profile')
    m = result['metadata']
    parameters = {key: m[key] for key in ('det_eff', 'field_evap', 'avg_dens', 'flight_path_length', 'mode')}
    parameters.update(kf=pair['kf'], icf=pair['icf'])
    variables.reconstruction_parameters = parameters
    variables.reconstruction_ratio_result = dict(result, accepted_anchor=pair,
                                                  anchor_source='explicit user/manual ICF; not inferred from this ratio')
    return parameters
