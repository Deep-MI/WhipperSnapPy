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

_SAMPLES = {
    ".off": _TETRA_OFF,
    ".vtk": _TETRA_VTK,
    ".ply": _TETRA_PLY,
}


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
        assert v.shape == (4, 3) and v.dtype == np.float32
        assert f.shape == (4, 3) and f.dtype == np.uint32
        np.testing.assert_array_equal(v, _expected_verts())
        np.testing.assert_array_equal(f, _expected_faces())

    def test_bundled_sample(self):
        """Verify the bundled tests/data/tetra.off file loads correctly."""
        here = os.path.dirname(__file__)
        v, f = read_off(os.path.join(here, "data", "tetra.off"))
        assert v.shape == (4, 3) and f.shape == (4, 3)

    def test_error_cases(self):
        cases = [
            ("", ".off", "empty"),
            ("NOFF\n4 4 6\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n3 0 1 2\n3 0 1 3\n3 0 2 3\n3 1 2 3\n", ".off", "OFF"),
            ("OFF\n4 1 4\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n4 0 1 2 3\n", ".off", "triangles"),
            ("OFF\n3 1 3\n0 0 0\n1 0 0\n0 1 0\n3 0 1 99\n", ".off", "out of range"),
        ]
        for content, suffix, match in cases:
            path = _write_tmp(content, suffix)
            try:
                with pytest.raises(ValueError, match=match):
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
        assert v.shape == (4, 3) and v.dtype == np.float32
        assert f.shape == (4, 3) and f.dtype == np.uint32
        np.testing.assert_array_equal(v, _expected_verts())
        np.testing.assert_array_equal(f, _expected_faces())

    def test_error_cases(self):
        cases = [
            ("# vtk DataFile Version 3.0\ntest\nBINARY\nDATASET POLYDATA\n", "BINARY"),
            ("# vtk DataFile Version 3.0\ntest\nASCII\nDATASET UNSTRUCTURED_GRID\n", "POLYDATA"),
            (
                "# vtk DataFile Version 3.0\ntest\nASCII\nDATASET POLYDATA\n"
                "POINTS 4 float\n0 0 0\n1 0 0\n0 1 0\n0 0 1\nPOLYGONS 1 5\n4 0 1 2 3\n",
                "triangles",
            ),
            (
                "# vtk DataFile Version 3.0\ntest\nASCII\nDATASET POLYDATA\n"
                "POLYGONS 1 4\n3 0 1 2\n",
                "POINTS",
            ),
        ]
        for content, match in cases:
            path = _write_tmp(content, ".vtk")
            try:
                with pytest.raises(ValueError, match=match):
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
        assert v.shape == (4, 3) and v.dtype == np.float32
        assert f.shape == (4, 3) and f.dtype == np.uint32
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
        assert v.shape == (3, 3) and f.shape == (1, 3)

    def test_error_cases(self):
        quad_ply = """\
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
        cases = [
            ("ply\nformat binary_little_endian 1.0\nelement vertex 4\nend_header\n", "binary"),
            ("OFF\n4 4 6\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n3 0 1 2\n", "ply"),
            (quad_ply, "triangles"),
        ]
        for content, match in cases:
            path = _write_tmp(content, ".ply")
            try:
                with pytest.raises(ValueError, match=match):
                    read_ply_ascii(path)
            finally:
                os.unlink(path)


# ---------------------------------------------------------------------------
# read_mesh dispatcher + resolve_mesh routing (combined)
# Each format is tested end-to-end through resolve_mesh (highest-level call).
# read_mesh itself is only tested for error cases not covered above.
# ---------------------------------------------------------------------------

class TestMeshDispatchAndRouting:
    @pytest.mark.parametrize("suffix,content", list(_SAMPLES.items()))
    def test_resolve_mesh_path(self, suffix, content):
        """resolve_mesh routes each format to the right reader and returns correct dtypes."""
        path = _write_tmp(content, suffix)
        try:
            v, f = resolve_mesh(path)
        finally:
            os.unlink(path)
        assert v.shape == (4, 3) and v.dtype == np.float32
        assert f.shape == (4, 3) and f.dtype == np.uint32

    def test_case_insensitive_extension(self):
        path = _write_tmp(_TETRA_OFF, ".OFF")
        try:
            v, f = read_mesh(path)
        finally:
            os.unlink(path)
        assert v.shape == (4, 3)

    def test_unknown_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            read_mesh("/some/file.stl")

    def test_array_tuple_still_works(self):
        v_in = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        f_in = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.uint32)
        v, f = resolve_mesh((v_in, f_in))
        assert v.shape == (4, 3) and f.shape == (4, 3)

    def test_array_out_of_range_raises(self):
        v_in = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        with pytest.raises(ValueError, match="out of range"):
            resolve_mesh((v_in, np.array([[0, 1, 99]], dtype=np.uint32)))


# ---------------------------------------------------------------------------
# GIfTI surface reader
# ---------------------------------------------------------------------------

def _make_surf_gii(verts, faces, suffix=".surf.gii"):
    """Write a minimal GIfTI surface file and return its path."""
    import nibabel as nib
    coords_da = nib.gifti.GiftiDataArray(
        data=verts.astype(np.float32), intent=1008, datatype="NIFTI_TYPE_FLOAT32",
    )
    faces_da = nib.gifti.GiftiDataArray(
        data=faces.astype(np.int32), intent=1009, datatype="NIFTI_TYPE_INT32",
    )
    img = nib.gifti.GiftiImage(darrays=[coords_da, faces_da])
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    nib.save(img, path)
    return path


_V4 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
_F4 = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.uint32)


class TestReadGiftiSurface:
    def test_basic_and_dispatch(self):
        """read_gifti_surface, read_mesh, and resolve_mesh all load .surf.gii correctly."""
        path = _make_surf_gii(_V4, _F4, ".surf.gii")
        try:
            v, f = read_gifti_surface(path)
            assert v.shape == (4, 3) and v.dtype == np.float32
            assert f.shape == (4, 3) and f.dtype == np.uint32
            np.testing.assert_allclose(v, _V4, atol=1e-6)
            # dispatch through read_mesh and resolve_mesh also work
            v2, _ = read_mesh(path)
            assert v2.shape == (4, 3)
            v3, f3 = resolve_mesh(path)
            assert v3.dtype == np.float32 and f3.dtype == np.uint32
        finally:
            os.unlink(path)

    def test_plain_gii_extension(self):
        path = _make_surf_gii(_V4, _F4, ".gii")
        try:
            v, f = read_gifti_surface(path)
            assert v.shape == (4, 3) and f.shape == (4, 3)
            # dispatch also works for plain .gii
            v2, _ = read_mesh(path)
            assert v2.shape == (4, 3)
        finally:
            os.unlink(path)

    def test_missing_arrays_raise(self):
        """Missing POINTSET or TRIANGLE array raises a clear ValueError."""
        import nibabel as nib
        # No POINTSET — only a scalar array
        scalar_da = nib.gifti.GiftiDataArray(
            data=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), intent=0,
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

        # POINTSET but no TRIANGLE
        coords_da = nib.gifti.GiftiDataArray(data=_V4.astype(np.float32), intent=1008)
        img2 = nib.gifti.GiftiImage(darrays=[coords_da])
        fd, path2 = tempfile.mkstemp(suffix=".gii")
        os.close(fd)
        nib.save(img2, path2)
        try:
            with pytest.raises(ValueError, match="TRIANGLE"):
                read_gifti_surface(path2)
        finally:
            os.unlink(path2)
