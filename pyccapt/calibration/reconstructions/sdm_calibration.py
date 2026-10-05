"""Self-contained SDM-based ICF/kf reconstruction calibration.

Pure (widget-free) algorithms for calibrating the two reconstruction
parameters that control depth/lateral scaling:

* ``kf``  -- field-reduction factor (sets the depth scale)
* ``icf`` -- image-compression factor (sets the lateral projection)

The calibration target is the **z-axis Spatial Distribution Map (SDM)** of the
atoms under a crystallographic pole: when ``kf``/``icf`` are correct, the
measured interplanar spacing (distance between SDM peaks along z) matches the
theoretical d-spacing of the indexed plane, and the peaks are sharp.

Workflow mirrored here (manual = the Day & Breen spreadsheet, automatic = the
``reconstruction_correction`` grid-sweep):

1. Pick a pole on the detector hit-map (FDM). ``sharp_fdm_map`` produces a
   high-contrast "best" FDM where poles stand out as density minima.
2. The pole ROI is a **detector** disk -> its atom membership is invariant to
   ``(kf, icf)`` (only the reconstructed coordinates change), so the same atoms
   are followed across every iteration.
3. Reconstruct, rotate so the pole faces +z (``align_pole_to_z``), then build
   the z-SDM (``z_sdm``) and measure peak spacing (``find_sdm_peaks``).
4. Correct kf from the theoretical/observed spacing ratio
   (``corrected_kf``); the theoretical d-spacing comes from the lattice +
   Miller index (``d_spacing``). Iterate.
5. ``grid_sweep_kf_icf`` automates step 4 by maximising SDM peakiness
   (``sdm_peakiness``) over a (kf, icf) grid in the fixed ROI.

All units: detector coordinates in cm (as stored in ``variables.dld_x_det``),
lattice constants and SDM spacings in nm, angles in degrees on input.
"""

from __future__ import annotations

import numpy as np

from pyccapt.calibration.reconstructions import reconstruction

__all__ = [
    "lattice_vectors",
    "d_spacing",
    "interplanar_angle",
    "build_hv",
    "reconstruct_xyz",
    "detector_roi_mask",
    "sharp_fdm_map",
    "detect_pole_candidates",
    "pole_axis_from_detector",
    "align_pole_to_z",
    "z_sdm",
    "find_sdm_peaks",
    "sdm_peakiness",
    "corrected_kf",
	"observed_pole_angle",
	"icf_from_pole_angles",
    "grid_sweep_kf_icf",
]


# ---------------------------------------------------------------------------
# Crystallography (mirrors the spreadsheet's "crystal calculator")
# ---------------------------------------------------------------------------

def lattice_vectors(a, b, c, alpha=90.0, beta=90.0, gamma=90.0):
    """Direct lattice vectors (rows) in a Cartesian frame, units = input.

    Standard convention: a along x, b in the x-y plane.
    Angles in degrees.
    """
    al, be, ga = np.radians([alpha, beta, gamma])
    ca, cb, cg = np.cos([al, be, ga])
    sg = np.sin(ga)
    a_vec = np.array([a, 0.0, 0.0])
    b_vec = np.array([b * cg, b * sg, 0.0])
    cx = c * cb
    cy = c * (ca - cb * cg) / sg
    cz2 = c * c - cx * cx - cy * cy
    cz = np.sqrt(max(cz2, 0.0))
    c_vec = np.array([cx, cy, cz])
    return np.vstack([a_vec, b_vec, c_vec])


def _reciprocal_vectors(direct):
    """Reciprocal lattice vectors (rows) from direct vectors (rows)."""
    a_vec, b_vec, c_vec = direct
    vol = np.dot(a_vec, np.cross(b_vec, c_vec))
    a_star = np.cross(b_vec, c_vec) / vol
    b_star = np.cross(c_vec, a_vec) / vol
    c_star = np.cross(a_vec, b_vec) / vol
    return np.vstack([a_star, b_star, c_star])


def _reciprocal_g(hkl, direct):
    """Cartesian reciprocal-lattice vector g* = h a* + k b* + l c*."""
    recip = _reciprocal_vectors(direct)
    h, k, l = hkl
    return h * recip[0] + k * recip[1] + l * recip[2]


def d_spacing(hkl, a, b, c, alpha=90.0, beta=90.0, gamma=90.0):
    """Theoretical interplanar spacing d_hkl (same length unit as a, b, c).

    Works for any crystal system (triclinic-general via the reciprocal
    metric). For cubic this reduces to a / sqrt(h^2+k^2+l^2).
    """
    direct = lattice_vectors(a, b, c, alpha, beta, gamma)
    g = _reciprocal_g(hkl, direct)
    norm = np.linalg.norm(g)
    if norm == 0:
        return float("nan")
    return float(1.0 / norm)


def interplanar_angle(hkl1, hkl2, a, b, c, alpha=90.0, beta=90.0, gamma=90.0):
    """Angle (degrees) between the normals of planes (hkl1) and (hkl2)."""
    direct = lattice_vectors(a, b, c, alpha, beta, gamma)
    g1 = _reciprocal_g(hkl1, direct)
    g2 = _reciprocal_g(hkl2, direct)
    n1, n2 = np.linalg.norm(g1), np.linalg.norm(g2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    cosang = np.clip(np.dot(g1, g2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


# ---------------------------------------------------------------------------
# Reconstruction wrapper (reuses pyccapt's Geiser/Bas implementation)
# ---------------------------------------------------------------------------

def build_hv(variables):
    """Per-event high voltage used for reconstruction (matches reconstruction.py)."""
    if getattr(variables, "pulse_mode", "laser") == "voltage":
        return np.asarray(variables.dld_high_voltage, dtype=float) + np.asarray(
            variables.dld_pulse_v, dtype=float
        )
    return np.asarray(variables.dld_high_voltage, dtype=float)


def reconstruct_xyz(variables, kf, icf, det_eff, field_evap, avg_dens,
                    flight_path_length, mode="Geiser"):
    """Reconstruct (x, y, z) in nm for ALL events without mutating ``variables``.

    z is a global cumulative sum, so the full set must be reconstructed even
    when only a pole ROI is analysed afterwards.
    """
    detx = np.asarray(variables.dld_x_det, dtype=float)
    dety = np.asarray(variables.dld_y_det, dtype=float)
    hv = build_hv(variables)
    if mode == "Bas":
        fn = reconstruction.atom_probe_recons_Bas_et_al
    else:
        fn = reconstruction.atom_probe_recons_from_detector_Geiser_et_al
    x, y, z = fn(detx, dety, hv, flight_path_length, kf, det_eff, icf, field_evap, avg_dens)
    return np.asarray(x), np.asarray(y), np.asarray(z)


def detector_roi_mask(variables, cx, cy, radius):
    """Boolean mask of events whose detector hit falls in the disk (cx, cy, radius) [cm].

    This membership is invariant to (kf, icf) -- the ROI is "constant over
    iterations" as required.
    """
    detx = np.asarray(variables.dld_x_det, dtype=float)
    dety = np.asarray(variables.dld_y_det, dtype=float)
    return (detx - cx) ** 2 + (dety - cy) ** 2 <= radius ** 2


# ---------------------------------------------------------------------------
# FDM (field-desorption / detector hit map): normal + sharp/"best"
# ---------------------------------------------------------------------------

def sharp_fdm_map(detx, dety, bins=200, smooth_sigma=1.4, extent=None):
    """Return (density, contrast, x_edges, y_edges) detector hit maps.

    ``density``  -- raw 2-D histogram of detector hits (the "normal" FDM).
    ``contrast`` -- detrended, smoothed, robust-z map where crystallographic
                    poles (local hit-density minima, De Geuser & Gault 2017)
                    stand out as bright spots -- the "best" / sharp FDM that
                    reveals clear pole positions.

    detx, dety in cm. ``extent`` = (xmin, xmax, ymin, ymax) in cm or None.
    """
    from scipy.ndimage import gaussian_filter, binary_erosion

    detx = np.asarray(detx, dtype=float)
    dety = np.asarray(dety, dtype=float)
    ok = np.isfinite(detx) & np.isfinite(dety)
    detx, dety = detx[ok], dety[ok]
    if extent is None:
        r = np.nanmax(np.hypot(detx, dety)) if detx.size else 1.0
        extent = (-r, r, -r, r)
    rng = [[extent[0], extent[1]], [extent[2], extent[3]]]
    density, xe, ye = np.histogram2d(detx, dety, bins=bins, range=rng)

    # Restrict to the *interior* of the populated detector area. The square
    # histogram has a sharp density step at the round detector edge; without
    # masking, detrending turns that edge into the dominant "contrast" and the
    # interior poles are lost. Erode the occupancy mask by a few bins.
    occ = gaussian_filter((density > 0).astype(float), sigma=2.0) > 0.5
    erode = max(2, int(bins / 40))
    interior = binary_erosion(occ, iterations=erode)
    if interior.sum() < 16:
        interior = occ

    # Detrend the smooth (bell-shaped) background, then smooth, then robust-z
    # using interior statistics only.
    background = gaussian_filter(density, sigma=max(bins / 12.0, 4.0))
    detr = density - background
    smoothed = gaussian_filter(detr, sigma=smooth_sigma)
    vals = smoothed[interior]
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) * 1.4826
    if mad == 0:
        mad = vals.std() or 1.0
    # Poles are density MINIMA (De Geuser & Gault) -> negate so poles become
    # positive contrast. Mask the boundary out (NaN) so it is never selected.
    contrast = -(smoothed - med) / mad
    contrast[~interior] = np.nan
    return density, contrast, xe, ye


def detect_pole_candidates(contrast, x_edges, y_edges, n=6, min_separation=5, min_sigma=3.0):
    """Top ``n`` pole candidates ``[(cx, cy, strength), ...]`` from a sharp FDM.

    ``contrast`` is the robust-sigma map from :func:`sharp_fdm_map`; coordinates
    are returned in detector cm, sorted by descending strength (sigma).
    """
    from scipy.ndimage import maximum_filter

    c = np.asarray(contrast, dtype=float)
    finite = np.isfinite(c)
    cc = np.where(finite, c, -np.inf)
    local_max = maximum_filter(cc, size=max(3, int(min_separation)))
    peaks = (cc == local_max) & (cc >= min_sigma) & finite
    ii, jj = np.where(peaks)
    if ii.size == 0:
        return []
    xc = 0.5 * (np.asarray(x_edges)[:-1] + np.asarray(x_edges)[1:])
    yc = 0.5 * (np.asarray(y_edges)[:-1] + np.asarray(y_edges)[1:])
    cand = [(float(xc[i]), float(yc[j]), float(c[i, j])) for i, j in zip(ii, jj)]
    cand.sort(key=lambda t: -t[2])
    return cand[:n]


# ---------------------------------------------------------------------------
# Pole -> +z rotation (planes become perpendicular to z -> clean z-SDM)
# ---------------------------------------------------------------------------

def pole_axis_from_detector(cx, cy, flight_path_length, icf):
    """Unit pole-axis direction in reconstruction space from a detector ROI center.

    cx, cy in cm; flight_path_length in mm. Uses the same launch-angle geometry
    as the Geiser reconstruction. A pole at the detector centre returns +z.
    """
    rad = np.hypot(cx * 10.0, cy * 10.0)  # cm -> mm to match flight path units
    az = np.arctan2(cy, cx)
    theta_p = np.arctan2(rad, flight_path_length)
    theta_a = theta_p + np.arcsin(np.clip((icf - 1.0) * np.sin(theta_p), -1.0, 1.0))
    d = np.array([
        np.sin(theta_a) * np.cos(az),
        np.sin(theta_a) * np.sin(az),
        np.cos(theta_a),
    ])
    n = np.linalg.norm(d)
    return d / n if n else np.array([0.0, 0.0, 1.0])


def _rotation_aligning(v_from, v_to):
    """Rotation matrix R (3x3) with R @ v_from parallel to v_to (unit vectors)."""
    a = v_from / (np.linalg.norm(v_from) or 1.0)
    b = v_to / (np.linalg.norm(v_to) or 1.0)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    if s < 1e-12:
        # Parallel or anti-parallel.
        if c > 0:
            return np.eye(3)
        # 180 deg: rotate about any axis perpendicular to a.
        perp = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + 2 * K @ K
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def align_pole_to_z(x, y, z, pole_dir, origin=None):
    """Rotate the point cloud so ``pole_dir`` points along +z.

    Returns (xr, yr, zr, R). ``origin`` (3,) is the rotation centre (defaults
    to the centroid); distances are preserved.
    """
    pts = np.column_stack([x, y, z]).astype(float)
    if origin is None:
        origin = np.nanmean(pts, axis=0)
    origin = np.asarray(origin, dtype=float)
    R = _rotation_aligning(np.asarray(pole_dir, dtype=float), np.array([0.0, 0.0, 1.0]))
    out = (pts - origin) @ R.T + origin
    return out[:, 0], out[:, 1], out[:, 2], R


# ---------------------------------------------------------------------------
# z-SDM and peak measurement
# ---------------------------------------------------------------------------

def z_sdm(x, y, z, bin_size=0.01, z_max=2.0, lateral_radius=1.5, max_atoms=20000,
          seed=0):
    """1-D z-axis SDM of a (rotated) pole ROI: histogram of pair Delta-z.

    For atom pairs within a cylinder (|Delta_xy| < lateral_radius,
    |Delta_z| < z_max), the signed Delta-z values are histogrammed
    symmetrically about 0. Peaks at +/- multiples of the interplanar spacing.

    Returns (centers, counts) -- both length = number of bins.
    """
    from scipy.spatial import cKDTree

    pts = np.column_stack([x, y, z]).astype(float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] > max_atoms:
        rng = np.random.default_rng(seed)
        pts = pts[rng.choice(pts.shape[0], max_atoms, replace=False)]
    if pts.shape[0] < 2:
        centers = np.arange(-z_max, z_max, bin_size)
        return centers, np.zeros_like(centers)

    tree = cKDTree(pts)
    r3d = float(np.hypot(lateral_radius, z_max))
    pairs = tree.query_pairs(r=r3d, output_type="ndarray")
    if pairs.size == 0:
        centers = np.arange(-z_max, z_max, bin_size)
        return centers, np.zeros_like(centers)

    dxy = np.hypot(pts[pairs[:, 0], 0] - pts[pairs[:, 1], 0],
                   pts[pairs[:, 0], 1] - pts[pairs[:, 1], 1])
    dz = pts[pairs[:, 0], 2] - pts[pairs[:, 1], 2]
    keep = (dxy < lateral_radius) & (np.abs(dz) < z_max)
    dz = dz[keep]
    dz = np.concatenate([dz, -dz])  # symmetric

    edges = np.arange(-z_max, z_max + bin_size, bin_size)
    counts, edges = np.histogram(dz, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts.astype(float)


def find_sdm_peaks(centers, counts, min_spacing=0.05, prominence_frac=0.05):
    """Find z-SDM peaks (excluding the central self-pair peak) and the mean spacing.

    Returns dict(peak_positions, spacing, n_peaks). ``spacing`` is the mean of
    successive positive-side peak separations (the measured d), or the position
    of the first positive peak if only one is found.
    """
    from scipy.signal import find_peaks

    centers = np.asarray(centers, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if counts.max() <= 0:
        return {"peak_positions": np.array([]), "spacing": float("nan"), "n_peaks": 0}

    bin_size = float(np.median(np.diff(centers))) if centers.size > 1 else min_spacing
    distance = max(1, int(round(min_spacing / bin_size)))
    prominence = prominence_frac * counts.max()
    idx, _ = find_peaks(counts, distance=distance, prominence=prominence)
    pos = centers[idx]

    # Drop the central peak around 0.
    pos = pos[np.abs(pos) > min_spacing]
    pos_side = np.sort(pos[pos > 0])
    if pos_side.size >= 2:
        spacing = float(np.mean(np.diff(pos_side)))
    elif pos_side.size == 1:
        spacing = float(pos_side[0])
    else:
        spacing = float("nan")
    return {"peak_positions": np.sort(pos), "spacing": spacing, "n_peaks": int(pos.size)}


def sdm_peakiness(centers, counts, dmin=0.10, dmax=0.40):
    """Periodicity strength of a z-SDM = peak FFT power / mean power in the band.

    Higher = sharper, more periodic peaks. This is the objective maximised by
    the automatic (kf, icf) grid sweep. ``dmin``/``dmax`` (nm) bound the
    physically plausible interplanar-spacing band.
    """
    counts = np.asarray(counts, dtype=float)
    centers = np.asarray(centers, dtype=float)
    if counts.size < 8 or counts.max() <= 0 or centers.size < 2:
        return 0.0
    bin_size = float(np.median(np.diff(centers)))
    sig = counts - counts.mean()
    spec = np.abs(np.fft.rfft(sig)) ** 2
    freqs = np.fft.rfftfreq(sig.size, d=bin_size)  # cycles per nm
    band = (freqs >= 1.0 / dmax) & (freqs <= 1.0 / dmin)
    if not np.any(band) or spec.mean() <= 0:
        return 0.0
    return float(spec[band].max() / spec.mean())


# ---------------------------------------------------------------------------
# kf correction (spreadsheet STEP logic) and automatic grid sweep
# ---------------------------------------------------------------------------

def corrected_kf(kf, d_theo, d_obs):
    """Spreadsheet kf correction: kf_new = kf * sqrt(d_theo / d_obs).

    From the Day & Breen template: k_correct = sqrt(kf^2 / (d_obs/d_theo)).
    """
    if d_obs is None or d_obs <= 0 or d_theo is None or d_theo <= 0:
        return float("nan")
    return float(kf * np.sqrt(d_theo / d_obs))


def observed_pole_angle(xy1, xy2, flight_path_length):
	"""Observed angle (degrees) between two detector poles, seen from the tip.

	Mirrors the Day & Breen spreadsheet: ``S = |p1 - p2|`` is the in-plane
	detector separation and the angle is ``atan(S / flight_path_length)``.
	``xy1``/``xy2`` and ``flight_path_length`` must share the same length unit
	(the spreadsheet uses mm for both).
	"""
	(x1, y1), (x2, y2) = xy1, xy2
	s = float(np.hypot(x1 - x2, y1 - y2))
	return float(np.degrees(np.arctan2(s, float(flight_path_length))))


def icf_from_pole_angles(poles, a, b, c, alpha, beta, gamma, flight_path_length):
	"""Image-compression factor from inter-pole angles (spreadsheet ICF method).

	For every pair of indexed poles the theoretical interplanar angle (from the
	lattice + Miller indices) is divided by the angle the same two poles subtend
	on the detector; ``icf`` is the mean of those per-pair ratios. This is the
	angle-based ICF calibration of Gault et al. 2009 implemented in the Day &
	Breen reconstruction-calibration spreadsheet.

	``poles`` is a sequence of ``{'xy': (X, Y), 'hkl': (h, k, l)}`` mappings with
	the detector position in the same length unit as ``flight_path_length``
	(mm). Poles with a missing position, a missing index or an all-zero index
	are ignored.

	Returns ``dict(icf, pairs)`` where ``pairs`` is a list of
	``(i, j, theo_deg, obs_deg, ratio)`` for each contributing pole pair.
	"""
	items = []
	for p in poles:
		xy = p.get('xy')
		hkl = p.get('hkl')
		if xy is None or hkl is None:
			continue
		hkl = tuple(int(round(v)) for v in hkl)
		if all(v == 0 for v in hkl):
			continue
		items.append((tuple(float(v) for v in xy), hkl))

	pairs = []
	ratios = []
	for i in range(len(items)):
		for j in range(i + 1, len(items)):
			(xy1, hkl1), (xy2, hkl2) = items[i], items[j]
			theo = interplanar_angle(hkl1, hkl2, a, b, c, alpha, beta, gamma)
			obs = observed_pole_angle(xy1, xy2, flight_path_length)
			if not np.isfinite(theo) or obs <= 0:
				continue
			ratio = theo / obs
			pairs.append((i, j, float(theo), float(obs), float(ratio)))
			ratios.append(ratio)

	icf = float(np.mean(ratios)) if ratios else float('nan')
	return {'icf': icf, 'pairs': pairs}


def grid_sweep_kf_icf(variables, roi_center, roi_radius, det_eff, field_evap,
                      avg_dens, flight_path_length, kf_values, icf_values,
                      mode="Geiser", bin_size=0.01, z_max=2.0, lateral_radius=1.5,
                      max_atoms=20000, progress=None):
    """Sweep a (kf, icf) grid maximising z-SDM peakiness in the fixed pole ROI.

    ``roi_center`` = (cx, cy) detector coords (cm); ``roi_radius`` in cm.
    ``kf_values``/``icf_values`` are 1-D arrays. ``progress`` (optional) is a
    callable ``progress(done, total)`` for a status bar.

    Returns dict(strength [len(kf) x len(icf)], kf_values, icf_values,
    best=dict(kf, icf, strength, spacing)).
    """
    cx, cy = roi_center
    roi_mask = detector_roi_mask(variables, cx, cy, roi_radius)
    n_roi = int(roi_mask.sum())
    kf_values = np.asarray(kf_values, dtype=float)
    icf_values = np.asarray(icf_values, dtype=float)
    S = np.zeros((kf_values.size, icf_values.size))
    spacing = np.full_like(S, np.nan)

    total = kf_values.size * icf_values.size
    done = 0
    for i, kf in enumerate(kf_values):
        for j, icf in enumerate(icf_values):
            x, y, z = reconstruct_xyz(variables, kf, icf, det_eff, field_evap,
                                      avg_dens, flight_path_length, mode=mode)
            xr = x[roi_mask]; yr = y[roi_mask]; zr = z[roi_mask]
            pole_dir = pole_axis_from_detector(cx, cy, flight_path_length, icf)
            xr, yr, zr, _ = align_pole_to_z(xr, yr, zr, pole_dir)
            centers, counts = z_sdm(xr, yr, zr, bin_size=bin_size, z_max=z_max,
                                    lateral_radius=lateral_radius, max_atoms=max_atoms)
            S[i, j] = sdm_peakiness(centers, counts)
            spacing[i, j] = find_sdm_peaks(centers, counts)["spacing"]
            done += 1
            if progress is not None:
                progress(done, total)

    bi, bj = np.unravel_index(np.argmax(S), S.shape)
    best = {
        "kf": float(kf_values[bi]),
        "icf": float(icf_values[bj]),
        "strength": float(S[bi, bj]),
        "spacing": float(spacing[bi, bj]),
    }
    return {
        "strength": S,
        "spacing": spacing,
        "kf_values": kf_values,
        "icf_values": icf_values,
        "roi_atoms": n_roi,
        "best": best,
    }
