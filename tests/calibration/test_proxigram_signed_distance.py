"""Regression test for the proxigram face-based signed-distance fix (R1).

The proxigram previously computed the signed distance from each ion to
the interface using the nearest VERTEX's normal and the vector to that
vertex. For points whose closest mesh feature is a face interior (the
common case on coarse / clipped iso-surfaces) that introduces an
in-plane error and can flip the sign. The fix projects onto the closest
FACE (find_closest_cell + closest point on the face) and takes the sign
from the cell normal.

For a planar interface the perpendicular signed distance must equal the
point's offset along the plane normal exactly -- this test pins that.
"""
import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from pyccapt.calibration.reconstructions import proxigram as px


def test_signed_distance_to_plane_equals_perpendicular_offset():
    # A z=0 plane: signed distance must equal |z| in magnitude for every
    # point, regardless of the in-plane (x, y) position.
    plane = pv.Plane(
        center=(0, 0, 0), direction=(0, 0, 1),
        i_size=20, j_size=20, i_resolution=20, j_resolution=20,
    ).triangulate()
    surf, verts, normals = px._interface_vertices_and_normals(plane)

    rng = np.random.default_rng(0)
    pts = rng.uniform(-5, 5, size=(500, 3))
    dist = px._signed_distance_to_interface(pts, surf, verts, normals)

    # Exact for a planar face projection (the nearest-vertex method would
    # introduce error proportional to the in-plane vertex spacing).
    assert np.allclose(np.abs(dist), np.abs(pts[:, 2]), atol=1e-6), (
        "Face-based signed distance to a z=0 plane must equal |z| exactly."
    )


def test_signed_distance_falls_back_to_nearest_vertex_on_error():
    plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1)).triangulate()
    surf, verts, normals = px._interface_vertices_and_normals(plane)
    pts = np.random.default_rng(1).uniform(-3, 3, size=(200, 3))

    class _BrokenSurf:
        n_cells = surf.n_cells
        cell_normals = surf.cell_normals

        def find_closest_cell(self, *args, **kwargs):
            raise RuntimeError("forced fallback")

    dist = px._signed_distance_to_interface(pts, _BrokenSurf(), verts, normals)
    # The nearest-vertex fallback still recovers the plane geometry well.
    corr = float(np.corrcoef(np.abs(dist), np.abs(pts[:, 2]))[0, 1])
    assert corr > 0.99, f"Fallback signed distance correlation too low: {corr}"
