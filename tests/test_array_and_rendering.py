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
    def test_valid_inputs(self):
        v, f = resolve_mesh((_V, _F))
        assert v.shape == (4, 3) and v.dtype == np.float32
        assert f.shape == (4, 3) and f.dtype == np.uint32
        # list input should also work
        v2, f2 = resolve_mesh([_V, _F])
        assert v2.shape == (4, 3) and f2.shape == (4, 3)

    def test_invalid_inputs_raise(self):
        with pytest.raises(TypeError):
            resolve_mesh(42)
        with pytest.raises(ValueError):
            resolve_mesh((np.ones((4, 4), dtype=np.float32), _F))
        with pytest.raises(ValueError):
            resolve_mesh((_V, np.ones((4, 4), dtype=np.uint32)))


# ---------------------------------------------------------------------------
# resolve_overlay / resolve_bg_map (identical logic, tested together)
# ---------------------------------------------------------------------------

class TestResolveScalarOverlay:
    """Tests for resolve_overlay and resolve_bg_map (same logic)."""

    @pytest.mark.parametrize("fn", [resolve_overlay, resolve_bg_map])
    def test_none_returns_none(self, fn):
        assert fn(None, n_vertices=_N) is None

    @pytest.mark.parametrize("fn", [resolve_overlay, resolve_bg_map])
    def test_array_input_shape_and_dtype(self, fn):
        arr = np.array([0.1, 0.5, 0.9, 0.3], dtype=np.float32)
        result = fn(arr, n_vertices=_N)
        assert result.shape == (_N,) and result.dtype == np.float32

    @pytest.mark.parametrize("fn", [resolve_overlay, resolve_bg_map])
    def test_shape_mismatch_raises(self, fn):
        with pytest.raises(ValueError):
            fn(np.ones(2), n_vertices=_N)

    def test_n_vertices_none_skips_shape_check(self):
        arr = np.array([0.1, 0.5], dtype=np.float32)
        assert resolve_overlay(arr, n_vertices=None).shape == (2,)


# ---------------------------------------------------------------------------
# resolve_roi
# ---------------------------------------------------------------------------

class TestResolveRoi:
    def test_none_returns_none(self):
        assert resolve_roi(None, n_vertices=_N) is None

    def test_bool_array(self):
        roi = np.array([True, True, True, False], dtype=bool)
        result = resolve_roi(roi, n_vertices=_N)
        assert result.dtype == bool and result.shape == (_N,)
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

    def test_two_and_three_tuple(self):
        labels = np.array([0, 1, 0, 1])
        ctab = np.array([[255, 0, 0, 0, 0], [0, 255, 0, 0, 1]])
        # two-tuple: names should be None
        r2 = resolve_annot((labels, ctab), n_vertices=_N)
        assert len(r2) == 3 and r2[2] is None
        # three-tuple: names passed through
        names = ["a", "b"]
        r3 = resolve_annot((labels, ctab, names), n_vertices=_N)
        assert r3[2] == names

    def test_invalid_inputs_raise(self):
        with pytest.raises(TypeError):
            resolve_annot(42, n_vertices=_N)
        with pytest.raises(ValueError):
            resolve_annot((np.zeros(2, dtype=int), np.array([[255, 0, 0, 0, 0]])), n_vertices=_N)


# ---------------------------------------------------------------------------
# estimate_overlay_thresholds
# ---------------------------------------------------------------------------

class TestEstimateOverlayThresholds:
    def test_auto_and_passthrough(self):
        arr = np.array([1.0, 2.0, 3.0, -1.5], dtype=np.float32)
        fmin, fmax = estimate_overlay_thresholds(arr)
        assert fmin >= 0 and fmax == pytest.approx(3.0)
        # explicit values passed through unchanged
        fmin2, fmax2 = estimate_overlay_thresholds(arr, minval=0.5, maxval=5.0)
        assert fmin2 == pytest.approx(0.5) and fmax2 == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# prepare_geometry_from_arrays — pure array pipeline
# ---------------------------------------------------------------------------

class TestPrepareGeometryFromArrays:
    def test_no_overlay(self):
        vdata, tris, fmin, fmax, pos, neg = prepare_geometry_from_arrays(_V, _F)
        assert vdata.shape == (_N, 9) and tris.shape == (4, 3)
        assert fmin is None and fmax is None

    def test_with_overlay_bg_map_roi(self):
        overlay = np.array([0.1, 0.5, 0.9, 0.3], dtype=np.float32)
        bg = np.array([-1.0, 1.0, -0.5, 0.5], dtype=np.float32)
        roi = np.array([True, True, True, False], dtype=bool)
        vdata, tris, fmin, fmax, pos, neg = prepare_geometry_from_arrays(
            _V, _F, overlay=overlay, bg_map=bg, roi=roi
        )
        assert vdata.shape == (_N, 9) and fmin is not None

    def test_overlay_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            prepare_geometry_from_arrays(_V, _F, overlay=np.array([0.1, 0.5]))


# ---------------------------------------------------------------------------
# prepare_geometry — thin wrapper (array path)
# ---------------------------------------------------------------------------

class TestPrepareGeometry:
    def test_tuple_mesh_various_inputs(self):
        """One call covers: tuple mesh, overlay, roi, bg_map — all together."""
        overlay = np.array([0.1, 0.5, 0.9, 0.3], dtype=np.float32)
        roi = np.array([True, True, True, False], dtype=bool)
        bg = np.array([-1.0, 1.0, -0.5, 0.5], dtype=np.float32)
        vdata, tris, fmin, fmax, pos, neg = prepare_geometry(
            (_V, _F), overlay=overlay, roi=roi, bg_map=bg
        )
        assert vdata.shape == (_N, 9) and fmin is not None

    def test_invalid_mesh_type_raises(self):
        with pytest.raises(TypeError):
            prepare_geometry(12345)


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
        assert img.width == 200 and img.height == 200
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
        assert np.array(img).min() != np.array(img).max(), (
            "Image with overlay+ROI is completely uniform."
        )

    def test_snap1_with_bg_map(self):
        """bg_map array: image is non-uniform."""
        bg = np.array([-1.0, 1.0, -0.5, 0.5], dtype=np.float32)
        arr = np.array(_snap1_offscreen(mesh=(_V, _F), bg_map=bg))
        assert arr.min() != arr.max(), "Image with bg_map is completely uniform."

    def test_label_map_and_lut_rendering(self):
        import numpy as np

        from whippersnappy.snap import snap1
        # Minimal tetra mesh
        v = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
        f = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.uint32)
        # Label map: 4 vertices, 2 labels
        labels = np.array([1,2,1,2], dtype=int)
        # LUT: label id, R, G, B (values in 0-255)
        lut = np.array([[1,255,0,0],[2,0,255,0]], dtype=float)
        # Normalize LUT colors
        lut[:,1:] = lut[:,1:] / 255.0
        annot = (labels, lut)
        img = snap1((v, f), annot=annot)
        assert img is not None
        arr = np.array(img)
        assert arr.shape[0] > 0 and arr.shape[1] > 0
        assert np.any(arr != arr[0,0])  # Not uniform
