"""New experimental calibration methods.

All opt-in. The existing pipeline is untouched. These functions return
a corrected calibration array (or modify ``variables.*_calib`` in place)
and can be A/B-tested against the legacy path via the benchmark harness.

Methods implemented here:

  3) ``voltage_corr_time_dependent``       — voltage correction with an
        explicit ion-index drift correction on top of the voltage poly.
  4) ``adaptive_residual_peak_fit``        — per-window shift estimated
        by fitting each reference peak as a Voigt and using the fit
        centre, instead of cross-correlation.
  5) ``bowl_correction_spline``            — bowl correction with a
        thin-plate / smoothing spline instead of a 2-D polynomial.
  6) ``adaptive_residual_gbm``             — replaces the temporal +
        spatial fits with one gradient-boosted regressor
        (sklearn.ensemble.GradientBoostingRegressor by default;
        falls back gracefully if lightgbm is available).

Methods #1, #2 already exist in calibration.py as
``joint_voltage_bowl_corr_main`` and ``multi_peak_voltage_corr_main`` /
``multi_peak_bowl_corr_main``; they're wired into the harness via
calibration_benchmark.py.

Method #7 (diagnostic plot) is in scripts/diagnostic_plot.py.
Method #8 (differentiable pipeline + SGD) is deferred.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.interpolate import SmoothBivariateSpline
from scipy.optimize import curve_fit

from pyccapt.calibration.core.validation import (
    CALIBRATION_MODES,
    ensure_choice,
    ensure_positive,
)

# Re-use the pseudo-Voigt + Gaussian helpers from mc_plot_peak_helpers so we
# stay consistent with the rest of the pipeline.
from pyccapt.calibration.core.mc_plot_peak_helpers import (
    _gaussian,
    _voigt,
    _FWHM_FACTOR,
)


# =====================================================================
# Method 3: time-dependent voltage correction
# =====================================================================


def voltage_corr_time_dependent(
    dld_highVoltage,
    variables,
    calibration_mode='mc',
    n_time_bins=12,
    bin_size=0.01,
    sample_size=100,
    apply=True,
    gate_min_ions=200,
    max_shift_fraction=0.005,
    use_legacy_v=True,
):
    """V2: Apply legacy voltage correction first, then a SMALL ion-index drift.

    The previous version was too aggressive. Key fixes:

    1. Use the legacy ``voltage_corr_main`` for the V-vs-voltage part
       (it's tuned and stable). Only the time-drift residual is the new
       part.
    2. Per chunk, gate against MRP improvement: only apply the shift if
       it doesn't regress the chunk's peak FWHM.
    3. Hard-clip shifts at ``max_shift_fraction * target_peak`` (default
       0.5%) so noisy chunks can't move ions far.
    4. Weight shifts by ion count and use trimmed-mean instead of median.
    5. Drop the convolution smoothing (over-smoothed across chunks).
    """
    from pyccapt.calibration.core import calibration as cal

    ensure_choice(calibration_mode, field_name='calibration_mode', allowed=CALIBRATION_MODES)
    bin_size = ensure_positive(bin_size, field_name='bin_size')

    # --- Stage 1: legacy voltage correction ---
    if use_legacy_v:
        cal.voltage_corr_main(
            np.asarray(dld_highVoltage, dtype=float),
            variables,
            sample_size=max(int(sample_size), 30),
            mode='ion_seq',
            calibration_mode=calibration_mode,
            index_fig=1, plot=False, save=False,
            bin_size=bin_size, model='robust_fit',
            peak_maximum=0, refine_nelder_mead=False,
        )

    calib = np.asarray(variables.get_calibration_array(calibration_mode), dtype=float).copy()

    # --- Stage 2: ion-index drift residual (gated, clipped) ---
    left, right = variables.get_calibration_peak_range(calibration_mode)
    mask = (calib > left) & (calib < right)
    if mask.sum() < gate_min_ions:
        if apply:
            _write_calib(variables, calibration_mode, calib)
        return calib
    target = _trimmed_mean(calib[mask], trim=0.10)
    abs_clip = abs(target) * float(max_shift_fraction)
    n_ions = calib.size

    edges = np.linspace(0, n_ions, int(n_time_bins) + 1, dtype=int)
    calib_out = calib.copy()
    for i in range(int(n_time_bins)):
        lo, hi = int(edges[i]), int(edges[i + 1])
        if hi - lo < gate_min_ions:
            continue
        m = mask[lo:hi]
        if m.sum() < gate_min_ions * 0.25:
            continue
        chunk_values = calib[lo:hi][m]
        chunk_center = _trimmed_mean(chunk_values, trim=0.10)
        raw_shift = chunk_center - target
        # Clip to physical bound
        shift = float(np.clip(raw_shift, -abs_clip, abs_clip))
        if abs(shift) < bin_size * 0.25:
            continue  # below noise floor, skip
        # Gate: only apply if it would improve the chunk's FWHM
        before_fwhm = _local_fwhm(chunk_values, target)
        candidate = chunk_values - shift
        after_fwhm = _local_fwhm(candidate, target)
        if not (np.isfinite(after_fwhm) and np.isfinite(before_fwhm)):
            continue
        if after_fwhm > before_fwhm * 0.99:  # require >1% improvement
            continue
        # Apply the shift to ALL ions in the chunk (not just masked ones)
        calib_out[lo:hi] = calib[lo:hi] - shift

    if apply:
        _write_calib(variables, calibration_mode, calib_out)
    return calib_out


def voltage_corr_v_t_joint(
    dld_highVoltage,
    variables,
    calibration_mode='mc',
    poly_v=2,
    poly_t=2,
    bin_size=0.05,
    sample_size=200,
    apply=True,
    smoothness=1e-3,
):
    """M3v3: joint 2-D polynomial fit factor(V, t_index).

    Replaces M3v2's per-chunk shift with a globally smooth correction
    surface. The correction is multiplicative (like legacy V), the
    coefficient on the t_index axis is regularised toward zero so that
    if there's no real drift the fit collapses back to the legacy V
    behavior.

    Fit:  t_observed = c0 + c1*V_n + c2*V_n^2 + c3*t_n + c4*t_n^2 + c5*V_n*t_n
        (degrees configurable; t_idx is normalised to [-1, 1]).
    """
    from pyccapt.calibration.core.validation import (
        CALIBRATION_MODES, ensure_choice, ensure_positive,
    )

    ensure_choice(calibration_mode, field_name='calibration_mode', allowed=CALIBRATION_MODES)
    bin_size = ensure_positive(bin_size, field_name='bin_size')
    calib = np.asarray(variables.get_calibration_array(calibration_mode), dtype=float).copy()
    voltage = np.asarray(dld_highVoltage, dtype=float)
    if voltage.size != calib.size:
        raise ValueError('voltage and calib arrays must match length')

    left, right = variables.get_calibration_peak_range(calibration_mode)
    mask = (calib > left) & (calib < right)
    if mask.sum() < 200:
        return calib

    # Normalise the regressors so coefficients are commensurate
    v_in = voltage[mask]
    n_ions = calib.size
    t_in_idx = np.where(mask)[0]
    v_mean = float(np.mean(v_in))
    v_std = max(float(np.std(v_in)), 1e-6)
    v_n_in = (v_in - v_mean) / v_std
    t_n_in = (t_in_idx.astype(float) / max(1.0, n_ions - 1)) * 2.0 - 1.0
    t_in = calib[mask]

    # Build design matrix up to (poly_v, poly_t) cross terms
    cols = [np.ones_like(v_n_in)]
    col_names = ['1']
    for i in range(1, int(poly_v) + 1):
        cols.append(v_n_in ** i); col_names.append(f'v^{i}')
    for j in range(1, int(poly_t) + 1):
        cols.append(t_n_in ** j); col_names.append(f't^{j}')
    for i in range(1, int(poly_v) + 1):
        for j in range(1, int(poly_t) + 1):
            cols.append((v_n_in ** i) * (t_n_in ** j))
            col_names.append(f'v^{i} t^{j}')
    X = np.column_stack(cols)
    y = t_in

    # Ridge-regularised on the t-only and cross-terms (NOT the V-only terms
    # so the legacy V fit isn't penalised). Effective if smoothness > 0.
    n_params = X.shape[1]
    reg = np.zeros(n_params, dtype=float)
    for k, nm in enumerate(col_names):
        if 't' in nm:
            reg[k] = smoothness * y.size
    # Solve (X^T X + diag(reg)) c = X^T y
    XtX = X.T @ X + np.diag(reg)
    Xty = X.T @ y
    try:
        coeffs = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        return calib

    # Predict on the FULL array
    v_n = (voltage - v_mean) / v_std
    t_n = (np.arange(n_ions, dtype=float) / max(1.0, n_ions - 1)) * 2.0 - 1.0
    cols_full = [np.ones_like(v_n)]
    for i in range(1, int(poly_v) + 1):
        cols_full.append(v_n ** i)
    for j in range(1, int(poly_t) + 1):
        cols_full.append(t_n ** j)
    for i in range(1, int(poly_v) + 1):
        for j in range(1, int(poly_t) + 1):
            cols_full.append((v_n ** i) * (t_n ** j))
    X_full = np.column_stack(cols_full)
    predicted = X_full @ coeffs
    target = float(np.median(t_in))

    factor = predicted / target
    factor = np.where(np.isfinite(factor) & (factor > 0.5) & (factor < 1.5), factor, 1.0)
    if calibration_mode == 'tof':
        factor = np.sqrt(np.maximum(factor, 1e-6))
    candidate = calib / factor

    # Global FWHM gate
    before_fwhm = _local_fwhm(calib[mask], target)
    cand_mask = (candidate > left) & (candidate < right)
    if cand_mask.sum() < 50:
        return calib
    after_fwhm = _local_fwhm(candidate[cand_mask], target)
    if not (np.isfinite(after_fwhm) and np.isfinite(before_fwhm)):
        return calib
    if after_fwhm > before_fwhm:  # require strict improvement
        return calib

    if apply:
        _write_calib(variables, calibration_mode, candidate)
    return candidate


def _trimmed_mean(values, trim=0.10):
    """Trimmed mean: drop top and bottom `trim` fraction, mean the rest."""
    if values.size == 0:
        return 0.0
    s = np.sort(values)
    k = int(values.size * trim)
    if 2 * k >= values.size:
        return float(np.mean(s))
    return float(np.mean(s[k:values.size - k]))


def _local_fwhm(values, center_hint=None, bin_size=0.005):
    """Rough FWHM of the dominant peak in `values`."""
    if values.size < 30:
        return float('nan')
    if center_hint is None:
        center_hint = float(np.median(values))
    half_w = (values.max() - values.min()) * 0.5
    edges = np.arange(center_hint - half_w, center_hint + half_w + bin_size, bin_size)
    if edges.size < 8:
        return float('nan')
    counts, _ = np.histogram(values, bins=edges)
    if counts.max() == 0:
        return float('nan')
    half_max = counts.max() / 2.0
    above = counts >= half_max
    if not above.any():
        return float('nan')
    idxs = np.where(above)[0]
    centres = 0.5 * (edges[:-1] + edges[1:])
    return float(centres[idxs[-1]] - centres[idxs[0]])


def _write_calib(variables, calibration_mode, calib_array):
    if calibration_mode == 'tof':
        variables.dld_t_calib = calib_array
    else:
        variables.mc_calib = calib_array


# =====================================================================
# Method 4: peak-deconvolution-based residual
# =====================================================================


def adaptive_residual_peak_fit(
    variables,
    calibration_mode='mc',
    n_windows=24,
    overlap=0.5,
    min_window_ions=40,
    bin_size=0.005,
    n_peaks=3,
    apply=True,
):
    """Residual correction using per-window Voigt fits instead of cross-correlation.

    1. Detect top reference peaks on the current calibration.
    2. Split ions into overlapping windows of equal index width.
    3. In each window, fit each detected reference peak as a pseudo-Voigt
       and take its fit centre as the observed centre.
    4. Compute a shift per peak (observed - target). Average the
       per-peak shifts to get a robust window shift.
    5. Interpolate window-shift to per-ion shift; apply.
    """
    from pyccapt.calibration.core import calibration as cal
    from pyccapt.calibration.core.adaptive_residual_calibration import _window_slices

    ensure_choice(calibration_mode, field_name='calibration_mode', allowed=CALIBRATION_MODES)
    calib = np.asarray(variables.get_calibration_array(calibration_mode), dtype=float).copy()
    n_ions = calib.size
    if n_ions < 1000:
        return calib

    arr_in_range = calib[(calib > 0) & np.isfinite(calib)]
    peaks = cal.auto_detect_reference_peaks(
        arr_in_range, n_peaks=max(2, int(n_peaks)),
        prominence=100, distance=10, hist_bin_size=max(0.02, bin_size * 2),
    )
    peaks = [p for p in peaks if p.get('n_ions', 0) > 50]
    if len(peaks) < 1:
        return calib

    targets = np.array([float(p['position']) for p in peaks])

    obs_centres = []  # per (window, peak) observed centre
    obs_indices = []  # per window, the centre ion index
    obs_weights = []  # per (window, peak) ion count

    for start, stop in _window_slices(n_ions, n_windows=int(n_windows), overlap=float(overlap)):
        window = calib[start:stop]
        if window.size < min_window_ions:
            continue
        per_peak_centres = []
        per_peak_weights = []
        for peak in peaks:
            x1, x2 = float(peak['x1']), float(peak['x2'])
            mask = (window > x1) & (window < x2)
            if mask.sum() < 30:
                per_peak_centres.append(np.nan)
                per_peak_weights.append(0.0)
                continue
            values = window[mask]
            centre = _fit_peak_centre(values, x1, x2, bin_size=bin_size)
            per_peak_centres.append(centre)
            per_peak_weights.append(float(mask.sum()))
        obs_centres.append(per_peak_centres)
        obs_weights.append(per_peak_weights)
        obs_indices.append(0.5 * (start + stop - 1))

    obs_centres = np.array(obs_centres, dtype=float)
    obs_weights = np.array(obs_weights, dtype=float)
    obs_indices = np.array(obs_indices, dtype=float)
    if obs_indices.size < 4:
        return calib

    # Per peak: centre - target. Then average across peaks weighted by ion count.
    shifts_per_peak = obs_centres - targets  # broadcast
    shifts_per_peak = np.where(np.isfinite(shifts_per_peak), shifts_per_peak, 0.0)
    eff_weight = np.sum(obs_weights, axis=1)
    avg_shift = np.where(
        eff_weight > 0,
        np.sum(shifts_per_peak * obs_weights, axis=1) / np.maximum(eff_weight, 1.0),
        0.0,
    )
    # Smooth across windows
    if avg_shift.size >= 5:
        kernel = np.ones(3) / 3
        avg_shift = np.convolve(avg_shift, kernel, mode='same')

    # Interpolate to per-ion shifts
    indices = np.arange(n_ions, dtype=float)
    per_ion_shift = np.interp(indices, obs_indices, avg_shift)
    # Clip to a sensible bound
    bound = max(float(np.nanstd(avg_shift)) * 4.0, 1e-6)
    per_ion_shift = np.clip(per_ion_shift, -bound, bound)
    calib_out = calib - per_ion_shift

    if apply:
        _write_calib(variables, calibration_mode, calib_out)
    return calib_out


def _fit_peak_centre(values, x1, x2, bin_size):
    """Fit a pseudo-Voigt to the values in [x1, x2] and return the centre.

    Falls back to the histogram-bin argmax if the fit fails.
    """
    if values.size < 30:
        return float('nan')
    edges = np.arange(x1, x2 + bin_size, bin_size)
    if edges.size < 5:
        return float(np.median(values))
    counts, _ = np.histogram(values, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    if counts.sum() == 0:
        return float('nan')
    mu0 = centres[int(np.argmax(counts))]
    amp0 = float(counts.max())
    sigma0 = max((x2 - x1) * 0.05, bin_size * 2.0)
    gamma0 = sigma0
    bg0 = float(np.median(counts))
    try:
        popt, _ = curve_fit(
            _voigt, centres, counts,
            p0=[amp0, mu0, sigma0, gamma0, bg0],
            bounds=(
                [0.0, x1, 1e-6, 1e-6, 0.0],
                [amp0 * 5.0, x2, x2 - x1, x2 - x1, amp0],
            ),
            maxfev=500,
        )
        return float(popt[1])
    except Exception:
        return float(mu0)


# =====================================================================
# Method 5: spline-based bowl correction
# =====================================================================


def bowl_correction_spline(
    dld_x_cm,
    dld_y_cm,
    dld_highVoltage,
    variables,
    det_diam,
    calibration_mode='mc',
    sample_size=20,
    bin_size=0.05,
    smoothing=None,
    apply=True,
    use_rbf=True,
    gate_min_improvement=0.0,
):
    """V2 spline bowl: RBF interpolator with stronger smoothing.

    Fixes from V1:
    - ``SmoothBivariateSpline`` overfits at the small N (grid² cell centres)
      so use ``scipy.interpolate.RBFInterpolator`` with a thin-plate spline
      kernel + explicit smoothing regularisation.
    - Cell-mean weighted by ion count (was: equal weight per cell).
    - Hard-clip correction factors to [0.95, 1.05] so over-fit cells can't
      blow ions out of the calibration window.
    - Gate against MRP improvement: only adopt the spline correction if
      the bracketed-peak FWHM doesn't regress.
    """
    from pyccapt.calibration.core import calibration as cal

    ensure_choice(calibration_mode, field_name='calibration_mode', allowed=CALIBRATION_MODES)
    calib = np.asarray(variables.get_calibration_array(calibration_mode), dtype=float).copy()
    x_mm = np.asarray(dld_x_cm, dtype=float) * 10.0
    y_mm = np.asarray(dld_y_cm, dtype=float) * 10.0

    left, right = variables.get_calibration_peak_range(calibration_mode)
    mask = (calib > left) & (calib < right)
    if mask.sum() < 500:
        return calib

    # Coarser grid (was: det_diam / sample_size = up to 16x16; now half that)
    grid = int(max(3, min(8, det_diam / max(2.0, sample_size * 1.5))))
    xb = np.linspace(np.nanmin(x_mm[mask]), np.nanmax(x_mm[mask]), grid + 1)
    yb = np.linspace(np.nanmin(y_mm[mask]), np.nanmax(y_mm[mask]), grid + 1)
    x_idx = np.clip(np.searchsorted(xb, x_mm[mask]) - 1, 0, grid - 1)
    y_idx = np.clip(np.searchsorted(yb, y_mm[mask]) - 1, 0, grid - 1)
    cell_id = x_idx * grid + y_idx
    n_cells = grid * grid
    sums = np.bincount(cell_id, weights=calib[mask], minlength=n_cells)
    counts = np.bincount(cell_id, minlength=n_cells)
    min_per_cell = max(20, int(mask.sum() / (n_cells * 4)))
    cell_means = np.divide(sums, counts, out=np.full(n_cells, np.nan),
                            where=counts >= min_per_cell)

    ix = np.arange(n_cells) // grid
    iy = np.arange(n_cells) % grid
    cx = 0.5 * (xb[ix] + xb[ix + 1])
    cy = 0.5 * (yb[iy] + yb[iy + 1])
    ok = np.isfinite(cell_means)
    if ok.sum() < 12:
        return calib

    target = float(np.average(cell_means[ok], weights=counts[ok]))
    z = cell_means[ok] / target

    factors = None
    if use_rbf:
        try:
            from scipy.interpolate import RBFInterpolator
            pts = np.column_stack([cx[ok], cy[ok]])
            # smoothing > 0 regularises; large enough to suppress overfit
            sm = smoothing if smoothing is not None else max(1.0, ok.sum() * 0.05)
            rbf = RBFInterpolator(pts, z, smoothing=sm, kernel='thin_plate_spline')
            xy = np.column_stack([x_mm, y_mm])
            factors = rbf(xy)
        except Exception:
            factors = None

    if factors is None:
        # Fallback to SmoothBivariateSpline with stronger smoothing
        try:
            sm = smoothing if smoothing is not None else ok.sum() * 0.02
            spline = SmoothBivariateSpline(cx[ok], cy[ok], z, s=sm, kx=3, ky=3)
            factors = spline.ev(x_mm, y_mm)
        except Exception:
            return calib

    # Aggressive clip: bowl corrections in real APT are < 5%
    factors = np.where(np.isfinite(factors), factors, 1.0)
    factors = np.clip(factors, 0.95, 1.05)
    if calibration_mode == 'tof':
        factors = np.sqrt(np.maximum(factors, 1e-6))
    candidate = calib / factors

    # Gate: require FWHM improvement on the bracketed peak
    before_fwhm = _local_fwhm(calib[mask], target)
    after_mask = (candidate > left) & (candidate < right)
    after_fwhm = (
        _local_fwhm(candidate[after_mask], target)
        if after_mask.sum() > 50 else float('nan')
    )
    if (
        not np.isfinite(after_fwhm)
        or not np.isfinite(before_fwhm)
        or after_fwhm > before_fwhm * (1.0 - float(gate_min_improvement))
    ):
        return calib  # no improvement -- keep original

    if apply:
        _write_calib(variables, calibration_mode, candidate)
    return candidate


# =====================================================================
# Method 6: gradient-boosted residual fitter
# =====================================================================


def adaptive_residual_gbm(
    variables,
    calibration_mode='mc',
    n_windows=24,
    overlap=0.5,
    min_window_ions=40,
    bin_size=0.005,
    n_peaks=3,
    gbm_n_estimators=80,
    gbm_max_depth=4,
    apply=True,
):
    """Residual correction via a gradient-boosted regressor.

    Per window, computes (window_centre_index, x_det, y_det, voltage) as
    features and the observed peak shift as the label. Trains a sklearn
    GradientBoostingRegressor (no lightgbm dependency) to predict the
    per-ion shift, then applies it.
    """
    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError:
        warnings.warn('scikit-learn not available; gbm residual not applied', RuntimeWarning)
        return np.asarray(variables.get_calibration_array(calibration_mode), dtype=float)

    from pyccapt.calibration.core import calibration as cal
    from pyccapt.calibration.core.adaptive_residual_calibration import _window_slices

    ensure_choice(calibration_mode, field_name='calibration_mode', allowed=CALIBRATION_MODES)
    calib = np.asarray(variables.get_calibration_array(calibration_mode), dtype=float).copy()
    n_ions = calib.size
    if n_ions < 1000:
        return calib

    voltage = np.asarray(variables.dld_high_voltage, dtype=float)
    x_det = np.asarray(variables.dld_x_det, dtype=float)
    y_det = np.asarray(variables.dld_y_det, dtype=float)

    peaks = cal.auto_detect_reference_peaks(
        calib[np.isfinite(calib) & (calib > 0)],
        n_peaks=max(2, int(n_peaks)),
        prominence=100, distance=10, hist_bin_size=max(0.02, bin_size * 2),
    )
    peaks = [p for p in peaks if p.get('n_ions', 0) > 50]
    if not peaks:
        return calib
    target = float(peaks[0]['position'])

    feats = []
    labels = []
    weights = []
    for start, stop in _window_slices(n_ions, n_windows=int(n_windows), overlap=float(overlap)):
        if stop - start < min_window_ions:
            continue
        for peak in peaks:
            x1, x2 = float(peak['x1']), float(peak['x2'])
            tgt = float(peak['position'])
            window = calib[start:stop]
            mask = (window > x1) & (window < x2)
            if mask.sum() < 20:
                continue
            observed_centre = _fit_peak_centre(window[mask], x1, x2, bin_size=bin_size)
            if not np.isfinite(observed_centre):
                continue
            shift = observed_centre - tgt
            # One feature row per peak per window.
            # Use nanmean so partial-recovered rows (NaN x_det / y_det)
            # don't poison the window-average features. If the entire
            # window is partial (all NaN), nanmean returns NaN with a
            # warning; guard so we just skip the row instead.
            xw = x_det[start:stop]
            yw = y_det[start:stop]
            if not np.any(np.isfinite(xw)) or not np.any(np.isfinite(yw)):
                continue
            feats.append([
                0.5 * (start + stop - 1) / max(1, n_ions),
                float(np.nanmean(xw)),
                float(np.nanmean(yw)),
                float(np.mean(voltage[start:stop])),
                tgt,  # which peak
            ])
            labels.append(shift)
            weights.append(int(mask.sum()))

    if len(feats) < 16:
        return calib
    X = np.array(feats, dtype=float)
    y = np.array(labels, dtype=float)
    w = np.array(weights, dtype=float)

    gbm = GradientBoostingRegressor(
        n_estimators=int(gbm_n_estimators), max_depth=int(gbm_max_depth), learning_rate=0.05
    )
    gbm.fit(X, y, sample_weight=w)

    # Predict per-ion shift (use the dominant peak's target as the reference per ion).
    # Partial rows have NaN x_det / y_det -> GBM.predict raises on NaN
    # features. Predict only on position-capable rows; partials get zero
    # shift so their uncalibrated value passes through unchanged.
    indices = np.arange(n_ions, dtype=float) / max(1.0, n_ions)
    finite_xy_pred = np.isfinite(x_det) & np.isfinite(y_det)
    X_pred = np.column_stack([
        indices[finite_xy_pred],
        x_det[finite_xy_pred],
        y_det[finite_xy_pred],
        voltage[finite_xy_pred],
        np.full(int(finite_xy_pred.sum()), target),
    ])
    per_ion_shift = np.zeros(n_ions, dtype=float)
    if X_pred.size > 0:
        per_ion_shift[finite_xy_pred] = gbm.predict(X_pred)
    bound = max(float(np.std(y)) * 4.0, 1e-6)
    per_ion_shift = np.clip(per_ion_shift, -bound, bound)
    calib_out = calib - per_ion_shift
    if apply:
        _write_calib(variables, calibration_mode, calib_out)
    return calib_out


# =====================================================================
# Method 3-NIST: full-spectrum self-similarity drift correction
# =====================================================================
#
# Per Coakley & Sanford (NIST, 2022): rather than tracking a single peak's
# centroid through time (which is what voltage_corr_time_dependent does),
# align EACH ion-index chunk's WHOLE spectrum to a reference spectrum
# by finding the multiplicative shift that maximises cross-correlation.
#
# This is robust to peak shape changes, dissociation tracks, and
# multi-peak content. It catches drift that single-peak methods miss.


def _chunk_spectrum(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    s = float(counts.sum())
    if s <= 0:
        return counts.astype(float)
    return counts.astype(float) / s


def _best_log_shift(
    chunk_vals: np.ndarray, ref_spec: np.ndarray, edges: np.ndarray,
    log_shift_grid: np.ndarray,
) -> tuple:
    """Find the log-shift in ``log_shift_grid`` that maximises overlap
    between ``hist(chunk_vals * exp(log_shift))`` and ``ref_spec``.

    Returns (best_log_shift, best_score).
    """
    best_s = 0.0
    best_score = -np.inf
    for s in log_shift_grid:
        if s == 0.0:
            scaled = chunk_vals
        else:
            scaled = chunk_vals * np.exp(s)
        spec = _chunk_spectrum(scaled, edges)
        # Cross-correlation at zero lag (both spectra share the same bin
        # grid) = sum of pointwise products. Normalise: cosine similarity.
        n1 = float(np.sqrt((spec * spec).sum()))
        if n1 == 0:
            continue
        score = float((spec * ref_spec).sum()) / n1
        if score > best_score:
            best_score = score
            best_s = float(s)
    return best_s, best_score


def drift_correction_self_similarity(
    variables,
    calibration_mode: str = 'mc',
    *,
    n_chunks: int = 12,
    bin_size: float = 0.02,
    max_shift_frac: float = 0.01,
    n_grid: int = 51,
    poly_degree: int = 3,
    smooth_lambda: float = 1e-2,
    contamination_policy: str = 'single_hit_only',
    mrp_drop_tolerance: float = 0.05,
    apply: bool = True,
    verbose: bool = True,
):
    """Self-similarity drift correction.

    Splits ions into ``n_chunks`` consecutive blocks along the
    acquisition axis. For each block, finds the multiplicative shift
    that maximises spectral overlap with a reference (the union of all
    blocks). Fits a smooth polynomial through the per-chunk shifts and
    applies it to all ions.

    Parameters
    ----------
    n_chunks : int
        Number of ion-index blocks. More chunks = finer drift detection
        but each chunk has fewer ions / noisier spectrum. 12 is a good
        default for ~1M ions.
    bin_size : float
        Histogram bin size in Da (mc) or ns (tof). Set ~ peak-FWHM/3.
    max_shift_frac : float
        Maximum |log-shift| considered; values are clipped to
        exp(+- max_shift_frac). 0.01 = +-1% per-ion correction.
    n_grid : int
        Grid resolution for the log-shift search per chunk.
    poly_degree : int
        Polynomial degree for the smooth(per-chunk shifts -> per-ion shift)
        fit. 2-3 is normal; higher overfits at low n_chunks.
    smooth_lambda : float
        L2 regulariser on the polynomial coefficients.
    contamination_policy : str
        Forwarded to event_filters.build_calibration_mask.
    mrp_drop_tolerance : float
        Acceptance gate.
    apply : bool
        Write back to variables when accepted.
    """
    from pyccapt.calibration.core import event_filters as _ef
    from pyccapt.calibration.core import calibration as _cal
    from pyccapt.calibration.core.mc_plot_peak_helpers import fast_mrp as _fast_mrp

    mode = ensure_choice(calibration_mode, field_name='calibration_mode', allowed=CALIBRATION_MODES)
    raw = np.asarray(variables.get_calibration_array(mode), dtype=float).copy()
    n_ions = raw.size
    if n_ions < max(2000, n_chunks * 200):
        if verbose:
            print(f"[drift-ss] {mode}: only {n_ions} ions; aborting")
        return {"ok": False, "reason": "too_few_ions"}

    mask = _ef.build_calibration_mask(getattr(variables, 'data', None),
                                       policy=contamination_policy)
    if mask.size != n_ions:
        mask = np.ones(n_ions, dtype=bool)

    lim = variables.max_tof if mode == 'tof' else 400.0
    finite_mask = (raw > 0) & (raw < lim) & np.isfinite(raw) & mask
    if int(finite_mask.sum()) < n_chunks * 200:
        if verbose:
            print(f"[drift-ss] {mode}: insufficient clean ions; aborting")
        return {"ok": False, "reason": "too_few_clean_ions"}

    # Histogram edges shared across all chunks (so spectra align bin-for-bin).
    lo = max(float(np.percentile(raw[finite_mask], 0.5)), 1e-3)
    hi = float(np.percentile(raw[finite_mask], 99.5))
    if hi <= lo:
        return {"ok": False, "reason": "degenerate_range"}
    edges = np.arange(lo, hi + bin_size, bin_size)
    if edges.size < 50:
        return {"ok": False, "reason": "too_few_bins"}

    # Build the reference spectrum (all clean ions).
    ref_spec = _chunk_spectrum(raw[finite_mask], edges)

    # Chunk boundaries along ion index (only clean ions count).
    clean_idx = np.where(finite_mask)[0]
    chunk_starts = np.linspace(0, clean_idx.size, n_chunks + 1).astype(int)
    log_shift_grid = np.linspace(-max_shift_frac, max_shift_frac, n_grid)

    per_chunk_log_shift = np.zeros(n_chunks)
    per_chunk_centre = np.zeros(n_chunks)
    per_chunk_score = np.zeros(n_chunks)
    for ci in range(n_chunks):
        sub = clean_idx[chunk_starts[ci]:chunk_starts[ci + 1]]
        if sub.size < 100:
            per_chunk_log_shift[ci] = 0.0
            per_chunk_centre[ci] = (chunk_starts[ci] + chunk_starts[ci + 1]) / 2.0
            continue
        s, score = _best_log_shift(raw[sub], ref_spec, edges, log_shift_grid)
        per_chunk_log_shift[ci] = s
        # Use the MEAN ion-index (over ALL ions in this chunk's range) as
        # the predictor for the smoother below.
        idx_range = clean_idx[chunk_starts[ci]:chunk_starts[ci + 1]]
        per_chunk_centre[ci] = float(np.mean(idx_range))
        per_chunk_score[ci] = score

    if verbose:
        s_min = float(per_chunk_log_shift.min())
        s_max = float(per_chunk_log_shift.max())
        print(f"[drift-ss] {mode}: per-chunk log-shift range [{s_min:+.4f}, {s_max:+.4f}] "
              f"({np.exp(s_min):.4f}x .. {np.exp(s_max):.4f}x), "
              f"mean score={float(per_chunk_score.mean()):.3f}")

    # Smooth polynomial fit through (chunk_centre, log_shift), weighted
    # by the chunk score. Center & scale predictor for stability.
    x_idx = per_chunk_centre
    x_idx_n = (x_idx - x_idx.mean()) / max(x_idx.std(), 1e-9)
    poly = np.polynomial.polynomial.Polynomial.fit(
        x_idx_n, per_chunk_log_shift, deg=int(poly_degree),
        w=np.sqrt(np.maximum(per_chunk_score, 0.0)),
    )
    # Predict per-ion log-shift.
    all_idx = np.arange(n_ions, dtype=float)
    all_idx_n = (all_idx - x_idx.mean()) / max(x_idx.std(), 1e-9)
    per_ion_log_shift = poly(all_idx_n)
    # Hard clip to bounds.
    per_ion_log_shift = np.clip(per_ion_log_shift, -max_shift_frac, max_shift_frac)
    factor = np.exp(per_ion_log_shift)
    # Apply: corrected = raw * factor (factor brings raw onto reference grid).
    if mode == 'tof':
        candidate = raw * np.sqrt(factor)
    else:
        candidate = raw * factor

    # Audit gate
    def _audit(arr):
        a = arr[(arr > 0) & (arr < lim) & np.isfinite(arr)]
        try:
            peaks = _cal.auto_detect_reference_peaks(
                a, n_peaks=20, prominence=25, distance=10, hist_bin_size=0.05,
            )
        except Exception:
            return {"n_peaks": 0, "geomean_mrp": float('nan')}
        if not peaks:
            return {"n_peaks": 0, "geomean_mrp": float('nan')}
        mrps = []
        for p in peaks:
            m = _fast_mrp(a, p['x1'], p['x2'], bin_size=0.01)
            m0 = float(m[0]) if m and len(m) > 0 and np.isfinite(m[0]) else float('nan')
            if np.isfinite(m0) and m0 > 0:
                mrps.append(min(m0, 5000.0))
        gm = float(np.exp(np.mean(np.log(np.asarray(mrps))))) if mrps else float('nan')
        return {"n_peaks": len(peaks), "geomean_mrp": gm}

    pre = _audit(raw)
    post = _audit(candidate)
    accepted = (
        post['n_peaks'] >= pre['n_peaks']
        and (
            not np.isfinite(pre['geomean_mrp'])
            or not np.isfinite(post['geomean_mrp'])
            or post['geomean_mrp'] >= pre['geomean_mrp'] * (1.0 - float(mrp_drop_tolerance))
        )
    )
    info = {
        "ok": bool(accepted),
        "calibration_mode": mode,
        "n_chunks": int(n_chunks),
        "per_chunk_log_shift": per_chunk_log_shift.tolist(),
        "per_chunk_score": per_chunk_score.tolist(),
        "factor_min": float(factor.min()),
        "factor_max": float(factor.max()),
        "pre": pre,
        "post": post,
    }
    if verbose:
        print(f"[drift-ss] {mode}: pre n={pre['n_peaks']} mrp={pre['geomean_mrp']:.0f}  "
              f"-> post n={post['n_peaks']} mrp={post['geomean_mrp']:.0f}  "
              f"factor_range=[{info['factor_min']:.4f}, {info['factor_max']:.4f}]  "
              f"{'ACCEPTED' if accepted else 'REVERTED'}")

    if apply and accepted:
        _write_calib(variables, mode, candidate)
    return info
