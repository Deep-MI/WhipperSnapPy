"""Tests for whippersnappy/geometry/overlay_io.py and the updated
_load_overlay_from_file routing in inputs.py.

All tests write content to temporary files so no external data is required.
"""

import os
import tempfile

import numpy as np
import pytest

from whippersnappy.geometry.inputs import resolve_bg_map, resolve_overlay, resolve_roi
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
# read_txt
# ---------------------------------------------------------------------------

class TestReadTxt:
    def test_float_basic(self):
        path = _write_tmp("\n".join(str(v) for v in _FLOAT_VALUES) + "\n", ".txt")
        try:
            arr = read_txt(path)
        finally:
            os.unlink(path)
        assert arr.shape == (4,) and arr.dtype == np.float32
        np.testing.assert_allclose(arr, _FLOAT_VALUES, atol=1e-6)

    def test_integer_promoted_to_int32(self):
        path = _write_tmp("\n".join(str(v) for v in _INT_VALUES) + "\n", ".txt")
        try:
            arr = read_txt(path)
        finally:
            os.unlink(path)
        assert arr.dtype == np.int32
        np.testing.assert_array_equal(arr, _INT_VALUES)

    def test_mixed_float_stays_float32(self):
        path = _write_tmp("0\n1\n2.5\n3\n", ".txt")
        try:
            arr = read_txt(path)
        finally:
            os.unlink(path)
        assert arr.dtype == np.float32

    def test_headers_and_comments_skipped(self):
        # hash comment, text header, CSV first column
        for content, suffix, expected_len in [
            ("# comment\n0.5\n1.5\n", ".txt", 2),
            ("value\n0.1\n0.2\n0.3\n", ".txt", 3),
            ("label,ignore\n1.0,extra\n2.0,extra\n", ".csv", 2),
        ]:
            path = _write_tmp(content, suffix)
            try:
                arr = read_txt(path)
            finally:
                os.unlink(path)
            assert arr.shape == (expected_len,)

    def test_error_cases(self):
        for content, match in [
            ("", "No numeric"),
            ("1.0\nbadvalue\n3.0\n", "Could not parse"),
        ]:
            path = _write_tmp(content, ".txt")
            try:
                with pytest.raises(ValueError, match=match):
                    read_txt(path)
            finally:
                os.unlink(path)


# ---------------------------------------------------------------------------
# read_npy / read_npz
# ---------------------------------------------------------------------------

class TestReadNpyNpz:
    def test_npy_basic(self):
        arr_in = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        fd, path = tempfile.mkstemp(suffix=".npy")
        os.close(fd)
        np.save(path, arr_in)
        try:
            np.testing.assert_array_equal(read_npy(path), arr_in)
        finally:
            os.unlink(path)

    def test_npy_column_vector_squeezed(self):
        fd, path = tempfile.mkstemp(suffix=".npy")
        os.close(fd)
        np.save(path, np.ones((5, 1), dtype=np.float32))
        try:
            assert read_npy(path).shape == (5,)
        finally:
            os.unlink(path)

    def test_npy_2d_raises(self):
        fd, path = tempfile.mkstemp(suffix=".npy")
        os.close(fd)
        np.save(path, np.ones((3, 4), dtype=np.float32))
        try:
            with pytest.raises(ValueError, match="1-D"):
                read_npy(path)
        finally:
            os.unlink(path)

    def test_npz_data_key_and_fallback(self):
        arr_in = np.array([0, 1, 2], dtype=np.int32)
        # named 'data' key
        fd, path = tempfile.mkstemp(suffix=".npz")
        os.close(fd)
        np.savez(path, data=arr_in, other=np.zeros(3))
        try:
            np.testing.assert_array_equal(read_npz(path), arr_in)
        finally:
            os.unlink(path)
        # first-array fallback
        arr2 = np.array([9.0, 8.0], dtype=np.float32)
        fd, path2 = tempfile.mkstemp(suffix=".npz")
        os.close(fd)
        np.savez(path2, arr_0=arr2)
        try:
            np.testing.assert_array_equal(read_npz(path2), arr2)
        finally:
            os.unlink(path2)

    def test_npz_empty_raises(self):
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
# Dispatch is implicitly covered by TestResolveOverlayRouting below; here we
# only test the error cases and the gifti rejection that are not exercised
# through resolve_overlay.
# ---------------------------------------------------------------------------

class TestReadOverlayDispatcher:
    def test_unknown_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            read_overlay("/some/file.xyz")

    def test_case_insensitive_extension(self):
        path = _write_tmp("1.0\n2.0\n", ".TXT")
        try:
            assert read_overlay(path).shape == (2,)
        finally:
            os.unlink(path)

    def test_surface_gii_rejected_with_helpful_error(self):
        import nibabel as nib
        coords_da = nib.gifti.GiftiDataArray(
            data=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32),
            intent=1008,
        )
        faces_da = nib.gifti.GiftiDataArray(
            data=np.array([[0,1,2],[0,1,3]], dtype=np.int32),
            intent=1009,
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


# ---------------------------------------------------------------------------
# resolve_overlay / resolve_bg_map / resolve_roi routing via inputs.py
# ---------------------------------------------------------------------------

class TestResolveOverlayRouting:
    """End-to-end: file path → resolve_overlay / resolve_bg_map / resolve_roi."""

    @pytest.mark.parametrize("suffix,content,n", [
        (".txt",  "0.1\n0.5\n0.9\n0.3\n", 4),
        (".csv",  "0.5\n1.5\n",            2),
    ])
    def test_txt_csv_routed(self, suffix, content, n):
        path = _write_tmp(content, suffix)
        try:
            arr = resolve_overlay(path, n_vertices=n)
        finally:
            os.unlink(path)
        assert arr.shape == (n,) and arr.dtype == np.float32

    def test_npy_routed(self):
        arr_in = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        fd, path = tempfile.mkstemp(suffix=".npy")
        os.close(fd)
        np.save(path, arr_in)
        try:
            np.testing.assert_array_equal(resolve_overlay(path, n_vertices=3), arr_in)
        finally:
            os.unlink(path)

    def test_shape_mismatch_raises(self):
        path = _write_tmp("0.1\n0.5\n", ".txt")
        try:
            with pytest.raises(ValueError, match="vertices"):
                resolve_overlay(path, n_vertices=5)
        finally:
            os.unlink(path)

    def test_bg_map_and_roi_routed(self):
        """resolve_bg_map and resolve_roi also correctly route .txt and .npy."""
        # bg_map from txt — always float32
        path = _write_tmp("1\n-1\n1\n-1\n", ".txt")
        try:
            arr = resolve_bg_map(path, n_vertices=4)
        finally:
            os.unlink(path)
        assert arr.shape == (4,) and arr.dtype == np.float32

        # roi from bool npy
        arr_in = np.array([True, False, True, True])
        fd, path2 = tempfile.mkstemp(suffix=".npy")
        os.close(fd)
        np.save(path2, arr_in)
        try:
            roi = resolve_roi(path2, n_vertices=4)
        finally:
            os.unlink(path2)
        assert roi.dtype == bool
        np.testing.assert_array_equal(roi, arr_in)

    def test_integer_txt_as_overlay(self):
        """Integer txt (parcellation) values are numerically preserved."""
        path = _write_tmp("3\n0\n1\n3\n", ".txt")
        try:
            arr = resolve_overlay(path, n_vertices=4)
        finally:
            os.unlink(path)
        np.testing.assert_array_equal(arr, [3.0, 0.0, 1.0, 3.0])
