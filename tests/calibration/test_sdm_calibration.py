"""Unit tests for the widget-free SDM ICF/kf calibration algorithms."""

import numpy as np
import pytest

from pyccapt.calibration.reconstructions import sdm_calibration as sc

pytestmark = pytest.mark.calibration


def test_d_spacing_cubic():
    # cubic a = 0.405 nm
    assert sc.d_spacing((0, 0, 2), 0.405, 0.405, 0.405) == pytest.approx(0.2025, abs=1e-6)
    assert sc.d_spacing((1, 1, 1), 0.405, 0.405, 0.405) == pytest.approx(0.405 / np.sqrt(3), abs=1e-6)
    assert sc.d_spacing((1, 0, 0), 0.405, 0.405, 0.405) == pytest.approx(0.405, abs=1e-6)


def test_d_spacing_tetragonal():
    # tetragonal a=b=0.3, c=0.5: d(001) = c, d(100) = a
    assert sc.d_spacing((0, 0, 1), 0.3, 0.3, 0.5) == pytest.approx(0.5, abs=1e-6)
    assert sc.d_spacing((1, 0, 0), 0.3, 0.3, 0.5) == pytest.approx(0.3, abs=1e-6)


def test_interplanar_angle_cubic():
    assert sc.interplanar_angle((1, 0, 0), (0, 1, 0), 0.4, 0.4, 0.4) == pytest.approx(90.0, abs=1e-4)
    assert sc.interplanar_angle((1, 0, 0), (1, 1, 0), 0.4, 0.4, 0.4) == pytest.approx(45.0, abs=1e-4)
    assert sc.interplanar_angle((1, 1, 1), (1, 1, 1), 0.4, 0.4, 0.4) == pytest.approx(0.0, abs=1e-4)


def test_align_pole_to_z():
    pole = np.array([0.3, 0.4, 0.866])
    _, _, _, R = sc.align_pole_to_z([0.0], [0.0], [0.0], pole)
    aligned = R @ (pole / np.linalg.norm(pole))
    assert aligned == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)


def test_corrected_kf():
    assert sc.corrected_kf(3.3, 0.2025, 0.18) == pytest.approx(3.3 * np.sqrt(0.2025 / 0.18), abs=1e-6)
    assert np.isnan(sc.corrected_kf(3.3, 0.2, 0.0))


def _synthetic_tilted_lattice(d=0.20, pole=np.array([0.3, 0.4, 0.866]), seed=0):
    rng = np.random.default_rng(seed)
    nz, npl = 40, 400
    zc = np.repeat(np.arange(nz) * d, npl)
    xy = rng.uniform(-3, 3, (zc.size, 2))
    pts = np.column_stack([xy, zc]).astype(float)
    pts[:, 2] += rng.normal(0, 0.02, zc.size)
    R_z_to_pole = sc._rotation_aligning(np.array([0.0, 0.0, 1.0]), pole)
    return pts @ R_z_to_pole.T


def test_z_sdm_recovers_spacing_after_alignment():
    pole = np.array([0.3, 0.4, 0.866])
    tilted = _synthetic_tilted_lattice(d=0.20, pole=pole)
    xa, ya, za, _ = sc.align_pole_to_z(tilted[:, 0], tilted[:, 1], tilted[:, 2], pole)
    centers, counts = sc.z_sdm(xa, ya, za, bin_size=0.01, z_max=1.0, lateral_radius=1.0, max_atoms=8000)
    pk = sc.find_sdm_peaks(centers, counts, min_spacing=0.05)
    assert pk["spacing"] == pytest.approx(0.20, abs=0.02)
    assert pk["n_peaks"] >= 4


def test_peakiness_higher_when_pole_aligned_to_z():
    pole = np.array([0.3, 0.4, 0.866])
    tilted = _synthetic_tilted_lattice(d=0.20, pole=pole)
    xa, ya, za, _ = sc.align_pole_to_z(tilted[:, 0], tilted[:, 1], tilted[:, 2], pole)
    c_a, k_a = sc.z_sdm(xa, ya, za, bin_size=0.01, z_max=1.0, lateral_radius=1.0, max_atoms=8000)
    c_m, k_m = sc.z_sdm(tilted[:, 0], tilted[:, 1], tilted[:, 2], bin_size=0.01, z_max=1.0,
                        lateral_radius=1.0, max_atoms=8000)
    assert sc.sdm_peakiness(c_a, k_a) > 2.0 * sc.sdm_peakiness(c_m, k_m)


def test_sharp_fdm_detects_carved_pole():
    rng = np.random.default_rng(1)
    n = 300000
    r = np.sqrt(rng.uniform(0, 1, n)) * 1.5
    th = rng.uniform(0, 2 * np.pi, n)
    dx = r * np.cos(th)
    dy = r * np.sin(th)
    # carve a hit-density depletion (pole) at (0.3, 0.2)
    keep = ~((np.hypot(dx - 0.3, dy - 0.2) < 0.10) & (rng.uniform(0, 1, n) < 0.8))
    dx, dy = dx[keep], dy[keep]
    _, contrast, xe, ye = sc.sharp_fdm_map(dx, dy, bins=160, smooth_sigma=1.4)
    cands = sc.detect_pole_candidates(contrast, xe, ye, n=5)
    assert cands, "expected at least one pole candidate"
    cx, cy, _ = cands[0]
    assert cx == pytest.approx(0.3, abs=0.1)
    assert cy == pytest.approx(0.2, abs=0.1)


def test_pole_axis_center_is_z():
    axis = sc.pole_axis_from_detector(0.0, 0.0, 110.0, 1.65)
    assert axis == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)
