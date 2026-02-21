"""Tests for whippersnappy/geometry/overlay_io.py and the updated
_load_overlay_from_file routing in inputs.py.

All tests write content to temporary files so no external data is required.
"""

import os
import tempfile

import numpy as np
import pytest

from whippersnappy.geometry.inputs import resolve_overlay, resolve_bg_map, resolve_roi
from whippersnappy.geometry.overlay_io import (
    read_npy,
    read_npz,
    read_overlay,
    read_txt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tmp(content, suffix, binary=False):
    fd, path = tempfile.mkstemp(suffix=suffix)
    mode = "wb" if binary else "w"
    with os.fdopen(fd, mode) as fh:
        fh.write(content)
    return path


_FLOAT_VALUES = [0.1, -1.5, 2.0, 0.0]
_INT_VALUES   = [0, 1, 3, 2]


# ---------------------------------------------------------------------------
# read_txt — float
# ---------------------------------------------------------------------------

class TestReadTxtFloat:
    def test_basic_floats(self):
        content = "\n".join(str(v) for v in _FLOAT_VALUES) + "\n"
        path = _write_tmp(content, ".txt")
        try:
            arr = read_txt(path)
        finally:
            os.unlink(path)
        assert arr.shape == (4,)
        assert arr.dtype == np.float32
        np.testing.assert_allclose(arr, _FLOAT_VALUES, atol=1e-6)

    def test_hash_comment_skipped(self):
        content = "# this is a comment\n0.5\n1.5\n"
        path = _write_tmp(content, ".txt")
        try:
            arr = read_txt(path)
        finally:
            os.unlink(path)
        assert arr.shape == (2,)

    def test_text_header_skipped(self):
        content = "value\n0.1\n0.2\n0.3\n"
        path = _write_tmp(content, ".txt")
        try:
            arr = read_txt(path)
        finally:
            os.unlink(path)
        assert arr.shape == (3,)
        np.testing.assert_allclose(arr, [0.1, 0.2, 0.3], atol=1e-6)

    def test_csv_first_column_used(self):
        content = "label,ignore\n1.0,extra\n2.0,extra\n"
        path = _write_tmp(content, ".csv")
        try:
            arr = read_txt(path)
        finally:
            os.unlink(path)
        assert arr.shape == (2,)

    def test_empty_file_raises(self):
        path = _write_tmp("", ".txt")
        try:
            with pytest.raises(ValueError, match="No numeric"):
                read_txt(path)
        finally:
            os.unlink(path)

    def test_bad_value_raises(self):
        content = "1.0\nbadvalue\n3.0\n"
        path = _write_tmp(content, ".txt")
        try:
            with pytest.raises(ValueError, match="Could not parse"):
                read_txt(path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# read_txt — integer promotion
# ---------------------------------------------------------------------------

class TestReadTxtInt:
    def test_integer_values_promoted_to_int32(self):
        content = "\n".join(str(v) for v in _INT_VALUES) + "\n"
        path = _write_tmp(content, ".txt")
        try:
            arr = read_txt(path)
        finally:
            os.unlink(path)
        assert arr.dtype == np.int32
        np.testing.assert_array_equal(arr, _INT_VALUES)

    def test_mixed_float_stays_float32(self):
        content = "0\n1\n2.5\n3\n"
        path = _write_tmp(content, ".txt")
        try:
            arr = read_txt(path)
        finally:
            os.unlink(path)
        assert arr.dtype == np.float32


# ---------------------------------------------------------------------------
# read_npy / read_npz
# ---------------------------------------------------------------------------

class TestReadNpy:
    def test_basic(self):
        arr_in = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        fd, path = tempfile.mkstemp(suffix=".npy")
        os.close(fd)
        np.save(path, arr_in)
        try:
            arr = read_npy(path)
        finally:
            os.unlink(path)
        np.testing.assert_array_equal(arr, arr_in)

    def test_2d_raises(self):
        arr_in = np.ones((3, 4), dtype=np.float32)
        fd, path = tempfile.mkstemp(suffix=".npy")
        os.close(fd)
        np.save(path, arr_in)
        try:
            with pytest.raises(ValueError, match="1-D"):
                read_npy(path)
        finally:
            os.unlink(path)

    def test_column_vector_squeezed(self):
        """Shape (N,1) should be squeezed to (N,) successfully."""
        arr_in = np.ones((5, 1), dtype=np.float32)
        fd, path = tempfile.mkstemp(suffix=".npy")
        os.close(fd)
        np.save(path, arr_in)
        try:
            arr = read_npy(path)
        finally:
            os.unlink(path)
        assert arr.shape == (5,)


class TestReadNpz:
    def test_data_key(self):
        arr_in = np.array([0, 1, 2], dtype=np.int32)
        fd, path = tempfile.mkstemp(suffix=".npz")
        os.close(fd)
        np.savez(path, data=arr_in, other=np.zeros(3))
        try:
            arr = read_npz(path)
        finally:
            os.unlink(path)
        np.testing.assert_array_equal(arr, arr_in)

    def test_first_array_fallback(self):
        arr_in = np.array([9.0, 8.0], dtype=np.float32)
        fd, path = tempfile.mkstemp(suffix=".npz")
        os.close(fd)
        np.savez(path, arr_0=arr_in)
        try:
            arr = read_npz(path)
        finally:
            os.unlink(path)
        np.testing.assert_array_equal(arr, arr_in)

    def test_empty_raises(self):
        fd, path = tempfile.mkstemp(suffix=".npz")
        os.close(fd)
        np.savez(path)
        try:
            with pytest.raises(ValueError, match="no arrays"):
                read_npz(path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# read_overlay dispatcher
# ---------------------------------------------------------------------------

class TestReadOverlayDispatcher:
    def test_txt_dispatched(self):
        path = _write_tmp("1.0\n2.0\n3.0\n", ".txt")
        try:
            arr = read_overlay(path)
        finally:
            os.unlink(path)
        assert arr.shape == (3,)

    def test_csv_dispatched(self):
        path = _write_tmp("0.5\n1.5\n", ".csv")
        try:
            arr = read_overlay(path)
        finally:
            os.unlink(path)
        assert arr.shape == (2,)

    def test_npy_dispatched(self):
        arr_in = np.array([1.0, 2.0], dtype=np.float32)
        fd, path = tempfile.mkstemp(suffix=".npy")
        os.close(fd)
        np.save(path, arr_in)
        try:
            arr = read_overlay(path)
        finally:
            os.unlink(path)
        np.testing.assert_array_equal(arr, arr_in)

    def test_npz_dispatched(self):
        arr_in = np.array([0, 1, 2], dtype=np.int32)
        fd, path = tempfile.mkstemp(suffix=".npz")
        os.close(fd)
        np.savez(path, data=arr_in)
        try:
            arr = read_overlay(path)
        finally:
            os.unlink(path)
        np.testing.assert_array_equal(arr, arr_in)

    def test_surface_gii_rejected_with_helpful_error(self):
        """A .surf.gii passed to read_gifti (overlay reader) should raise clearly."""
        import nibabel as nib
        coords_da = nib.gifti.GiftiDataArray(
            data=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32),
            intent=1008,  # POINTSET
        )
        faces_da = nib.gifti.GiftiDataArray(
            data=np.array([[0,1,2],[0,1,3]], dtype=np.int32),
            intent=1009,  # TRIANGLE
        )
        img = nib.gifti.GiftiImage(darrays=[coords_da, faces_da])
        fd, path = tempfile.mkstemp(suffix=".surf.gii")
        os.close(fd)
        nib.save(img, path)
        try:
            with pytest.raises(ValueError, match="surface geometry"):
                from whippersnappy.geometry.overlay_io import read_gifti
                read_gifti(path)
        finally:
            os.unlink(path)

    def test_unknown_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            read_overlay("/some/file.xyz")

    def test_case_insensitive_extension(self):
        path = _write_tmp("1.0\n2.0\n", ".TXT")
        try:
            arr = read_overlay(path)
        finally:
            os.unlink(path)
        assert arr.shape == (2,)


# ---------------------------------------------------------------------------
# resolve_overlay / resolve_bg_map / resolve_roi routing via inputs.py
# ---------------------------------------------------------------------------

class TestResolveOverlayRouting:
    """Verify that resolve_overlay and resolve_bg_map pick up .txt/.npy files."""

    def test_txt_path_routed_as_float(self):
        content = "0.1\n0.5\n0.9\n0.3\n"
        path = _write_tmp(content, ".txt")
        try:
            arr = resolve_overlay(path, n_vertices=4)
        finally:
            os.unlink(path)
        assert arr.shape == (4,)
        assert arr.dtype == np.float32

    def test_npy_path_routed(self):
        arr_in = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        fd, path = tempfile.mkstemp(suffix=".npy")
        os.close(fd)
        np.save(path, arr_in)
        try:
            arr = resolve_overlay(path, n_vertices=3)
        finally:
            os.unlink(path)
        np.testing.assert_array_equal(arr, arr_in)

    def test_shape_mismatch_raises(self):
        content = "0.1\n0.5\n"
        path = _write_tmp(content, ".txt")
        try:
            with pytest.raises(ValueError, match="vertices"):
                resolve_overlay(path, n_vertices=5)
        finally:
            os.unlink(path)

    def test_bg_map_txt_routed(self):
        content = "1\n-1\n1\n-1\n"
        path = _write_tmp(content, ".txt")
        try:
            arr = resolve_bg_map(path, n_vertices=4)
        finally:
            os.unlink(path)
        assert arr.shape == (4,)
        assert arr.dtype == np.float32  # always cast to float32 by resolve_bg_map

    def test_roi_from_bool_npy(self):
        """Boolean .npy array is a valid ROI input after resolve_roi casts it."""
        arr_in = np.array([True, False, True, True])
        fd, path = tempfile.mkstemp(suffix=".npy")
        os.close(fd)
        np.save(path, arr_in)
        try:
            # resolve_roi receives a str path; _load_overlay_from_file loads npy
            # and returns the array; then resolve_roi casts it to bool.
            arr = resolve_roi(path, n_vertices=4)
        finally:
            os.unlink(path)
        assert arr.dtype == bool
        np.testing.assert_array_equal(arr, arr_in)

    def test_label_txt_integer_values(self):
        """Integer .txt file (parcellation) should be loadable as overlay."""
        content = "3\n0\n1\n3\n"
        path = _write_tmp(content, ".txt")
        try:
            # resolve_overlay casts to float32; original int32 from read_txt
            arr = resolve_overlay(path, n_vertices=4)
        finally:
            os.unlink(path)
        assert arr.shape == (4,)
        # Values should be preserved numerically
        np.testing.assert_array_equal(arr, [3.0, 0.0, 1.0, 3.0])

