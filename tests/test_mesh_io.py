"""Tests for whippersnappy/geometry/mesh_io.py and the updated resolve_mesh.

All tests use in-memory strings written to temporary files so no external
data is required (except the bundled tetra.off sample).
"""

import os
import tempfile

import numpy as np
import pytest

from whippersnappy.geometry.inputs import resolve_mesh
from whippersnappy.geometry.mesh_io import (
    read_gifti_surface,
    read_mesh,
    read_off,
    read_ply_ascii,
    read_vtk_ascii_polydata,
)

# ---------------------------------------------------------------------------
# Shared sample content strings
# ---------------------------------------------------------------------------

_TETRA_OFF = """\
OFF
# tetrahedron
4 4 6
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
3 0 2 1
3 0 1 3
3 0 3 2
3 1 2 3
"""

_TETRA_VTK = """\
# vtk DataFile Version 3.0
tetrahedron
ASCII
DATASET POLYDATA
POINTS 4 float
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
POLYGONS 4 16
3 0 2 1
3 0 1 3
3 0 3 2
3 1 2 3
"""

_TETRA_PLY = """\
ply
format ascii 1.0
comment tetrahedron
element vertex 4
property float x
property float y
property float z
element face 4
property list uchar int vertex_indices
end_header
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
3 0 2 1
3 0 1 3
3 0 3 2
3 1 2 3
"""


def _write_tmp(content, suffix):
    """Write *content* to a named temp file, return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as fh:
        fh.write(content)
    return path


def _expected_verts():
    return np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
    )


def _expected_faces():
    return np.array(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.uint32
    )


# ---------------------------------------------------------------------------
# read_off
# ---------------------------------------------------------------------------

class TestReadOff:
    def test_basic(self):
        path = _write_tmp(_TETRA_OFF, ".off")
        try:
            v, f = read_off(path)
        finally:
            os.unlink(path)
        assert v.shape == (4, 3)
        assert v.dtype == np.float32
        assert f.shape == (4, 3)
        assert f.dtype == np.uint32
        np.testing.assert_array_equal(v, _expected_verts())
        np.testing.assert_array_equal(f, _expected_faces())

    def test_bundled_sample(self):
        """Verify the bundled tests/data/tetra.off file loads correctly."""
        here = os.path.dirname(__file__)
        sample = os.path.join(here, "data", "tetra.off")
        v, f = read_off(sample)
        assert v.shape == (4, 3)
        assert f.shape == (4, 3)

    def test_bad_header_raises(self):
        content = "NOFF\n4 4 6\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n3 0 1 2\n3 0 1 3\n3 0 2 3\n3 1 2 3\n"
        path = _write_tmp(content, ".off")
        try:
            with pytest.raises(ValueError, match="OFF"):
                read_off(path)
        finally:
            os.unlink(path)

    def test_quad_face_raises(self):
        content = "OFF\n4 1 4\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n4 0 1 2 3\n"
        path = _write_tmp(content, ".off")
        try:
            with pytest.raises(ValueError, match="triangles"):
                read_off(path)
        finally:
            os.unlink(path)

    def test_out_of_range_indices_raises(self):
        content = "OFF\n3 1 3\n0 0 0\n1 0 0\n0 1 0\n3 0 1 99\n"
        path = _write_tmp(content, ".off")
        try:
            with pytest.raises(ValueError, match="out of range"):
                read_off(path)
        finally:
            os.unlink(path)

    def test_empty_file_raises(self):
        path = _write_tmp("", ".off")
        try:
            with pytest.raises(ValueError, match="empty"):
                read_off(path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# read_vtk_ascii_polydata
# ---------------------------------------------------------------------------

class TestReadVtkAsciiPolydata:
    def test_basic(self):
        path = _write_tmp(_TETRA_VTK, ".vtk")
        try:
            v, f = read_vtk_ascii_polydata(path)
        finally:
            os.unlink(path)
        assert v.shape == (4, 3)
        assert v.dtype == np.float32
        assert f.shape == (4, 3)
        assert f.dtype == np.uint32
        np.testing.assert_array_equal(v, _expected_verts())
        np.testing.assert_array_equal(f, _expected_faces())

    def test_binary_vtk_raises(self):
        content = "# vtk DataFile Version 3.0\ntest\nBINARY\nDATASET POLYDATA\n"
        path = _write_tmp(content, ".vtk")
        try:
            with pytest.raises(ValueError, match="BINARY"):
                read_vtk_ascii_polydata(path)
        finally:
            os.unlink(path)

    def test_non_polydata_raises(self):
        content = "# vtk DataFile Version 3.0\ntest\nASCII\nDATASET UNSTRUCTURED_GRID\n"
        path = _write_tmp(content, ".vtk")
        try:
            with pytest.raises(ValueError, match="POLYDATA"):
                read_vtk_ascii_polydata(path)
        finally:
            os.unlink(path)

    def test_quad_polygon_raises(self):
        content = (
            "# vtk DataFile Version 3.0\ntest\nASCII\nDATASET POLYDATA\n"
            "POINTS 4 float\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n"
            "POLYGONS 1 5\n4 0 1 2 3\n"
        )
        path = _write_tmp(content, ".vtk")
        try:
            with pytest.raises(ValueError, match="triangles"):
                read_vtk_ascii_polydata(path)
        finally:
            os.unlink(path)

    def test_missing_points_raises(self):
        content = (
            "# vtk DataFile Version 3.0\ntest\nASCII\nDATASET POLYDATA\n"
            "POLYGONS 1 4\n3 0 1 2\n"
        )
        path = _write_tmp(content, ".vtk")
        try:
            with pytest.raises(ValueError, match="POINTS"):
                read_vtk_ascii_polydata(path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# read_ply_ascii
# ---------------------------------------------------------------------------

class TestReadPlyAscii:
    def test_basic(self):
        path = _write_tmp(_TETRA_PLY, ".ply")
        try:
            v, f = read_ply_ascii(path)
        finally:
            os.unlink(path)
        assert v.shape == (4, 3)
        assert v.dtype == np.float32
        assert f.shape == (4, 3)
        assert f.dtype == np.uint32
        np.testing.assert_array_equal(v, _expected_verts())
        np.testing.assert_array_equal(f, _expected_faces())

    def test_extra_vertex_props(self):
        """PLY with extra per-vertex properties (e.g. nx ny nz) should still load."""
        content = """\
ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
property float nx
property float ny
property float nz
element face 1
property list uchar int vertex_indices
end_header
0.0 0.0 0.0 0.0 0.0 1.0
1.0 0.0 0.0 0.0 0.0 1.0
0.0 1.0 0.0 0.0 0.0 1.0
3 0 1 2
"""
        path = _write_tmp(content, ".ply")
        try:
            v, f = read_ply_ascii(path)
        finally:
            os.unlink(path)
        assert v.shape == (3, 3)
        assert f.shape == (1, 3)

    def test_binary_ply_raises(self):
        content = "ply\nformat binary_little_endian 1.0\nelement vertex 4\nend_header\n"
        path = _write_tmp(content, ".ply")
        try:
            with pytest.raises(ValueError, match="binary"):
                read_ply_ascii(path)
        finally:
            os.unlink(path)

    def test_not_ply_raises(self):
        content = "OFF\n4 4 6\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n3 0 1 2\n"
        path = _write_tmp(content, ".ply")
        try:
            with pytest.raises(ValueError, match="ply"):
                read_ply_ascii(path)
        finally:
            os.unlink(path)

    def test_quad_face_raises(self):
        content = """\
ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
element face 1
property list uchar int vertex_indices
end_header
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
4 0 1 2 3
"""
        path = _write_tmp(content, ".ply")
        try:
            with pytest.raises(ValueError, match="triangles"):
                read_ply_ascii(path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# read_mesh dispatcher
# ---------------------------------------------------------------------------

class TestReadMeshDispatcher:
    def test_off_dispatch(self):
        path = _write_tmp(_TETRA_OFF, ".off")
        try:
            v, f = read_mesh(path)
        finally:
            os.unlink(path)
        assert v.shape == (4, 3)

    def test_vtk_dispatch(self):
        path = _write_tmp(_TETRA_VTK, ".vtk")
        try:
            v, f = read_mesh(path)
        finally:
            os.unlink(path)
        assert v.shape == (4, 3)

    def test_ply_dispatch(self):
        path = _write_tmp(_TETRA_PLY, ".ply")
        try:
            v, f = read_mesh(path)
        finally:
            os.unlink(path)
        assert v.shape == (4, 3)

    def test_unknown_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            read_mesh("/some/file.stl")

    def test_case_insensitive_extension(self):
        """Uppercase .OFF extension should be recognised."""
        path = _write_tmp(_TETRA_OFF, ".OFF")
        try:
            v, f = read_mesh(path)
        finally:
            os.unlink(path)
        assert v.shape == (4, 3)


# ---------------------------------------------------------------------------
# resolve_mesh routing
# ---------------------------------------------------------------------------

class TestResolveMeshRouting:
    def test_off_path_routed(self):
        path = _write_tmp(_TETRA_OFF, ".off")
        try:
            v, f = resolve_mesh(path)
        finally:
            os.unlink(path)
        assert v.shape == (4, 3)
        assert v.dtype == np.float32
        assert f.shape == (4, 3)
        assert f.dtype == np.uint32

    def test_vtk_path_routed(self):
        path = _write_tmp(_TETRA_VTK, ".vtk")
        try:
            v, f = resolve_mesh(path)
        finally:
            os.unlink(path)
        assert v.shape == (4, 3)

    def test_ply_path_routed(self):
        path = _write_tmp(_TETRA_PLY, ".ply")
        try:
            v, f = resolve_mesh(path)
        finally:
            os.unlink(path)
        assert v.shape == (4, 3)

    def test_array_tuple_still_works(self):
        v_in = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        f_in = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.uint32)
        v, f = resolve_mesh((v_in, f_in))
        assert v.shape == (4, 3)
        assert f.shape == (4, 3)

    def test_array_out_of_range_raises(self):
        v_in = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        f_in = np.array([[0, 1, 99]], dtype=np.uint32)  # index 99 out of range
        with pytest.raises(ValueError, match="out of range"):
            resolve_mesh((v_in, f_in))


# ---------------------------------------------------------------------------
# GIfTI surface reader
# ---------------------------------------------------------------------------

def _make_surf_gii(verts, faces, suffix=".surf.gii"):
    """Write a minimal GIfTI surface file and return its path."""
    import nibabel as nib
    from nibabel import nifti1
    _INTENT_POINTSET = 1008
    _INTENT_TRIANGLE = 1009
    coords_da = nib.gifti.GiftiDataArray(
        data=verts.astype(np.float32),
        intent=_INTENT_POINTSET,
        datatype="NIFTI_TYPE_FLOAT32",
    )
    faces_da = nib.gifti.GiftiDataArray(
        data=faces.astype(np.int32),
        intent=_INTENT_TRIANGLE,
        datatype="NIFTI_TYPE_INT32",
    )
    img = nib.gifti.GiftiImage(darrays=[coords_da, faces_da])
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    nib.save(img, path)
    return path


_V4 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
_F4 = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.uint32)


class TestReadGiftiSurface:
    def test_surf_gii_basic(self):
        path = _make_surf_gii(_V4, _F4, ".surf.gii")
        try:
            v, f = read_gifti_surface(path)
        finally:
            os.unlink(path)
        assert v.shape == (4, 3)
        assert v.dtype == np.float32
        assert f.shape == (4, 3)
        assert f.dtype == np.uint32
        np.testing.assert_allclose(v, _V4, atol=1e-6)
        np.testing.assert_array_equal(f, _F4)

    def test_plain_gii_extension(self):
        """A .gii file with POINTSET+TRIANGLE should also be loaded as surface."""
        path = _make_surf_gii(_V4, _F4, ".gii")
        try:
            v, f = read_gifti_surface(path)
        finally:
            os.unlink(path)
        assert v.shape == (4, 3)
        assert f.shape == (4, 3)

    def test_no_pointset_raises(self):
        """A plain scalar .gii without POINTSET arrays should raise."""
        import nibabel as nib
        scalar_da = nib.gifti.GiftiDataArray(
            data=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            intent=0,  # NIFTI_INTENT_NONE
        )
        img = nib.gifti.GiftiImage(darrays=[scalar_da])
        fd, path = tempfile.mkstemp(suffix=".gii")
        os.close(fd)
        nib.save(img, path)
        try:
            with pytest.raises(ValueError, match="POINTSET"):
                read_gifti_surface(path)
        finally:
            os.unlink(path)

    def test_no_triangle_raises(self):
        """A .gii with only a POINTSET but no TRIANGLE array should raise."""
        import nibabel as nib
        coords_da = nib.gifti.GiftiDataArray(
            data=_V4.astype(np.float32),
            intent=1008,
        )
        img = nib.gifti.GiftiImage(darrays=[coords_da])
        fd, path = tempfile.mkstemp(suffix=".gii")
        os.close(fd)
        nib.save(img, path)
        try:
            with pytest.raises(ValueError, match="TRIANGLE"):
                read_gifti_surface(path)
        finally:
            os.unlink(path)


class TestReadMeshGiftiDispatch:
    def test_surf_gii_dispatched(self):
        path = _make_surf_gii(_V4, _F4, ".surf.gii")
        try:
            v, f = read_mesh(path)
        finally:
            os.unlink(path)
        assert v.shape == (4, 3)

    def test_gii_dispatched(self):
        path = _make_surf_gii(_V4, _F4, ".gii")
        try:
            v, f = read_mesh(path)
        finally:
            os.unlink(path)
        assert v.shape == (4, 3)

    def test_resolve_mesh_surf_gii(self):
        path = _make_surf_gii(_V4, _F4, ".surf.gii")
        try:
            v, f = resolve_mesh(path)
        finally:
            os.unlink(path)
        assert v.shape == (4, 3)
        assert v.dtype == np.float32
        assert f.dtype == np.uint32

