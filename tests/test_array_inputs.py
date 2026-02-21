"""Tests for the array-input pathway introduced in v2.0-rc.

These tests exercise the resolver functions and the geometry-preparation
pipeline entirely without touching any file on disk, using small synthetic
triangle meshes.
"""

import numpy as np
import pytest

from whippersnappy.geometry.inputs import (
    resolve_annot,
    resolve_bg_map,
    resolve_mesh,
    resolve_overlay,
    resolve_roi,
)
from whippersnappy.geometry.prepare import (
    estimate_overlay_thresholds,
    prepare_geometry,
    prepare_geometry_from_arrays,
)

# ---------------------------------------------------------------------------
# Minimal synthetic mesh (tetrahedron)
# ---------------------------------------------------------------------------

_V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
_F = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.uint32)
_N = _V.shape[0]  # 4 vertices


# ---------------------------------------------------------------------------
# resolve_mesh
# ---------------------------------------------------------------------------

class TestResolveMesh:
    def test_tuple_input(self):
        v, f = resolve_mesh((_V, _F))
        assert v.shape == (4, 3)
        assert v.dtype == np.float32
        assert f.shape == (4, 3)
        assert f.dtype == np.uint32

    def test_list_input(self):
        v, f = resolve_mesh([_V, _F])
        assert v.shape == (4, 3)
        assert f.shape == (4, 3)

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            resolve_mesh(42)

    def test_wrong_shape_vertices_raises(self):
        bad_v = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            resolve_mesh((bad_v, _F))

    def test_wrong_shape_faces_raises(self):
        bad_f = np.ones((4, 4), dtype=np.uint32)
        with pytest.raises(ValueError):
            resolve_mesh((_V, bad_f))


# ---------------------------------------------------------------------------
# resolve_overlay / resolve_bg_map
# ---------------------------------------------------------------------------

class TestResolveOverlay:
    def test_none_returns_none(self):
        assert resolve_overlay(None, n_vertices=_N) is None

    def test_array_input(self):
        arr = np.array([0.1, 0.5, 0.9, 0.3], dtype=np.float32)
        result = resolve_overlay(arr, n_vertices=_N)
        assert result.shape == (_N,)
        assert result.dtype == np.float32

    def test_shape_mismatch_raises(self):
        arr = np.array([0.1, 0.5], dtype=np.float32)
        with pytest.raises(ValueError):
            resolve_overlay(arr, n_vertices=_N)

    def test_n_vertices_none_skips_check(self):
        arr = np.array([0.1, 0.5], dtype=np.float32)
        result = resolve_overlay(arr, n_vertices=None)
        assert result.shape == (2,)


class TestResolveBgMap:
    def test_none_returns_none(self):
        assert resolve_bg_map(None, n_vertices=_N) is None

    def test_array_input(self):
        arr = np.array([-1.0, 1.0, -0.5, 0.5], dtype=np.float32)
        result = resolve_bg_map(arr, n_vertices=_N)
        assert result.shape == (_N,)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            resolve_bg_map(np.ones(2), n_vertices=_N)


# ---------------------------------------------------------------------------
# resolve_roi
# ---------------------------------------------------------------------------

class TestResolveRoi:
    def test_none_returns_none(self):
        assert resolve_roi(None, n_vertices=_N) is None

    def test_bool_array(self):
        roi = np.array([True, True, True, False], dtype=bool)
        result = resolve_roi(roi, n_vertices=_N)
        assert result.dtype == bool
        assert result.shape == (_N,)
        assert result[3] is np.bool_(False)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            resolve_roi(np.ones(2, dtype=bool), n_vertices=_N)


# ---------------------------------------------------------------------------
# resolve_annot
# ---------------------------------------------------------------------------

class TestResolveAnnot:
    def test_none_returns_none(self):
        assert resolve_annot(None, n_vertices=_N) is None

    def test_two_tuple(self):
        labels = np.array([0, 1, 0, 1])
        ctab = np.array([[255, 0, 0, 0, 0], [0, 255, 0, 0, 1]])
        result = resolve_annot((labels, ctab), n_vertices=_N)
        assert result is not None
        assert len(result) == 3
        assert result[2] is None  # names

    def test_three_tuple(self):
        labels = np.zeros(_N, dtype=int)
        ctab = np.array([[200, 100, 50, 0, 1]])
        names = ["region0"]
        result = resolve_annot((labels, ctab, names), n_vertices=_N)
        assert result[2] == names

    def test_shape_mismatch_raises(self):
        labels = np.zeros(2, dtype=int)
        ctab = np.array([[255, 0, 0, 0, 0]])
        with pytest.raises(ValueError):
            resolve_annot((labels, ctab), n_vertices=_N)

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            resolve_annot(42, n_vertices=_N)


# ---------------------------------------------------------------------------
# estimate_overlay_thresholds
# ---------------------------------------------------------------------------

class TestEstimateOverlayThresholds:
    def test_array_input(self):
        arr = np.array([1.0, 2.0, 3.0, -1.5], dtype=np.float32)
        fmin, fmax = estimate_overlay_thresholds(arr)
        assert fmin >= 0
        assert fmax == pytest.approx(3.0)

    def test_passthrough_when_provided(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        fmin, fmax = estimate_overlay_thresholds(arr, minval=0.5, maxval=5.0)
        assert fmin == pytest.approx(0.5)
        assert fmax == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# prepare_geometry_from_arrays — pure array pipeline
# ---------------------------------------------------------------------------

class TestPrepareGeometryFromArrays:
    def test_no_overlay(self):
        vdata, tris, fmin, fmax, pos, neg = prepare_geometry_from_arrays(_V, _F)
        assert vdata.shape == (_N, 9)
        assert tris.shape == (4, 3)
        assert fmin is None and fmax is None

    def test_with_overlay(self):
        overlay = np.array([0.1, 0.5, 0.9, 0.3], dtype=np.float32)
        vdata, tris, fmin, fmax, pos, neg = prepare_geometry_from_arrays(
            _V, _F, overlay=overlay
        )
        assert vdata.shape == (_N, 9)
        assert fmin is not None and fmax is not None

    def test_with_bg_map(self):
        bg = np.array([-1.0, 1.0, -0.5, 0.5], dtype=np.float32)
        vdata, tris, *_ = prepare_geometry_from_arrays(_V, _F, bg_map=bg)
        assert vdata.shape == (_N, 9)

    def test_with_roi(self):
        overlay = np.array([0.1, 0.5, 0.9, 0.3], dtype=np.float32)
        roi = np.array([True, True, True, False], dtype=bool)
        vdata, tris, *_ = prepare_geometry_from_arrays(_V, _F, overlay=overlay, roi=roi)
        assert vdata.shape == (_N, 9)

    def test_overlay_shape_mismatch_raises(self):
        overlay = np.array([0.1, 0.5], dtype=np.float32)  # wrong length
        with pytest.raises(ValueError):
            prepare_geometry_from_arrays(_V, _F, overlay=overlay)


# ---------------------------------------------------------------------------
# prepare_geometry — thin wrapper (array path)
# ---------------------------------------------------------------------------

class TestPrepareGeometry:
    def test_tuple_mesh_no_overlay(self):
        vdata, tris, *_ = prepare_geometry((_V, _F))
        assert vdata.shape == (_N, 9)

    def test_tuple_mesh_with_overlay_and_roi(self):
        overlay = np.array([0.1, 0.5, 0.9, 0.3], dtype=np.float32)
        roi = np.array([True, True, True, False], dtype=bool)
        vdata, tris, fmin, fmax, pos, neg = prepare_geometry(
            (_V, _F), overlay=overlay, roi=roi
        )
        assert vdata.shape == (_N, 9)
        assert fmin is not None

    def test_tuple_mesh_with_bg_map(self):
        bg = np.array([-1.0, 1.0, -0.5, 0.5], dtype=np.float32)
        vdata, tris, *_ = prepare_geometry((_V, _F), bg_map=bg)
        assert vdata.shape == (_N, 9)

    def test_invalid_mesh_type_raises(self):
        with pytest.raises(TypeError):
            prepare_geometry(12345)


# ---------------------------------------------------------------------------
# snap1 array-input integration (no OpenGL — just geometry prep)
# ---------------------------------------------------------------------------

class TestSnap1ArrayInputs:
    """Tests for the geometry-preparation layer used by snap1.

    These run without any OpenGL context and are safe for headless CI.
    """

    def test_prepare_geometry_called_by_snap1_path(self):
        """prepare_geometry accepts the same args snap1 would pass."""
        overlay = np.array([0.1, 0.5, 0.9, 0.3], dtype=np.float32)
        roi = np.array([True, True, True, False], dtype=bool)
        bg = np.array([-1.0, 1.0, -0.5, 0.5], dtype=np.float32)
        vdata, tris, fmin, fmax, pos, neg = prepare_geometry(
            (_V, _F), overlay=overlay, roi=roi, bg_map=bg
        )
        assert vdata is not None
        assert tris is not None

    def test_prepare_geometry_bg_map_array_no_error(self):
        """bg_map as array raises no error."""
        bg = np.array([-0.5, 0.5, -0.3, 0.3], dtype=np.float32)
        vdata, tris, *_ = prepare_geometry((_V, _F), bg_map=bg)
        assert vdata.shape == (_N, 9)


# ---------------------------------------------------------------------------
# snap1 rendering — actual OpenGL image output
# ---------------------------------------------------------------------------

def _snap1_offscreen(**kwargs):
    """Call snap1 with an invisible (offscreen) GLFW context.

    On macOS a visible GLFW window goes through the Cocoa compositor; the
    first glReadPixels call may return all-black before the compositor has
    finished its first composite pass.  An invisible context renders
    directly to the driver framebuffer and reads back correctly.

    Skips the test automatically if no OpenGL context can be created
    (headless CI without GPU or EGL support).
    """
    import whippersnappy.gl.utils as gl_utils  # noqa: PLC0415

    original = gl_utils.create_window_with_fallback

    def _invisible(*args, **kw):
        kw["visible"] = False
        return original(*args, **kw)

    gl_utils.create_window_with_fallback = _invisible
    try:
        from whippersnappy import snap1  # noqa: PLC0415
        return snap1(width=200, height=200, colorbar=False, **kwargs)
    except RuntimeError as exc:
        if "context" in str(exc).lower() or "opengl" in str(exc).lower():
            pytest.skip(f"No OpenGL context available: {exc}")
        raise
    finally:
        gl_utils.create_window_with_fallback = original


class TestSnap1Rendering:
    """End-to-end rendering tests: snap1 must return a non-empty PIL Image.

    All tests use the tetrahedron mesh (_V, _F) — a true 3-D shape that is
    visible from any camera direction, unlike a flat surface which can
    appear edge-on and produce an all-black image.

    Tests use an offscreen GLFW context (see ``_snap1_offscreen``) and are
    skipped automatically when no OpenGL context is available.
    """

    def test_snap1_basic(self):
        """snap1 returns the right size and renders a non-uniform image."""
        img = _snap1_offscreen(mesh=(_V, _F))
        assert img.width == 200
        assert img.height == 200
        arr = np.array(img)
        assert arr.min() != arr.max(), (
            "Rendered image is completely uniform — shading is not working."
        )

    def test_snap1_with_overlay_and_roi(self):
        """Overlay + ROI mask: image is non-uniform and correct size."""
        overlay = np.array([0.1, 0.5, 0.9, 0.3], dtype=np.float32)
        roi = np.array([True, True, True, False], dtype=bool)
        img = _snap1_offscreen(
            mesh=(_V, _F), overlay=overlay, roi=roi, fthresh=0.0, fmax=1.0
        )
        assert img.width == 200
        arr = np.array(img)
        assert arr.min() != arr.max(), "Image with overlay+ROI is completely uniform."

    def test_snap1_with_bg_map(self):
        """bg_map array: image is non-uniform."""
        bg = np.array([-1.0, 1.0, -0.5, 0.5], dtype=np.float32)
        img = _snap1_offscreen(mesh=(_V, _F), bg_map=bg)
        arr = np.array(img)
        assert arr.min() != arr.max(), "Image with bg_map is completely uniform."

