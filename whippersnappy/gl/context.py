"""OpenGL context management and rendering pipeline.

Owns the full lifecycle of an OpenGL context — creation, scene rendering,
framebuffer capture, and teardown.

Context creation tries up to three paths (Linux; macOS/Windows use GLFW only):

1. **GLFW invisible window** — standard path when a display is available.
2. **EGL pbuffer** — headless GPU rendering (Linux, no display needed).
   Only used when :mod:`~whippersnappy.gl._headless` set
   ``PYOPENGL_PLATFORM=egl`` at import time (no display + accessible
   ``/dev/dri/renderD*``).  PyOpenGL selects its platform backend on the
   first ``import OpenGL.GL`` and cannot be changed afterwards — so EGL is
   only safe when it was selected before any ``OpenGL.GL`` import.
3. **OSMesa** — CPU software renderer (Linux only).
   Used when neither GLFW nor EGL succeeds.

The :mod:`whippersnappy.gl._headless` guard runs before ``OpenGL.GL`` is
imported and sets ``PYOPENGL_PLATFORM`` to ``"egl"`` or ``"osmesa"``
as appropriate.
"""
# ruff: noqa: I001  — import order is intentional: _headless must precede OpenGL.GL

import logging
import os
import sys
import warnings
from typing import Any

# Ensure PYOPENGL_PLATFORM=osmesa is set on headless Linux before OpenGL.GL
# is imported.  The actual guard lives in _headless so it runs whenever any
# submodule of this package is imported first.
from . import _headless  # noqa: F401, I001

import glfw  # noqa: E402
import OpenGL.GL as gl  # noqa: E402
from PIL import Image  # noqa: E402

# Module logger
logger = logging.getLogger(__name__)

# Module-level offscreen context handle (None when GLFW is used instead).
# May hold an EGLContext or OSMesaContext instance on headless environments.
_offscreen_context: Any = None


def _try_glfw_window(width, height, title, visible, core_profile):
    """Attempt to create a single GLFW window with the given profile settings.

    Calls ``glfw.init()`` before and ``glfw.terminate()`` on failure so that
    each attempt starts from a clean GLFW state.

    Parameters
    ----------
    core_profile : bool
        If True request OpenGL 3.3 Core Profile + ``FORWARD_COMPAT``
        (preferred on all platforms).
        If False request OpenGL 3.3 Compatibility Profile (fallback for
        Windows software renderers that don't support Core Profile).

    Returns
    -------
    window or None
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if not glfw.init():
            return None

    glfw.default_window_hints()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    if core_profile:
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    else:
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, False)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)
    if not visible:
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

    window = glfw.create_window(width, height, title, None, None)
    if not window:
        glfw.terminate()
        return None
    return window


def init_window(width, height, title="WhipperSnapPy", visible=True):
    """Create a GLFW window, make an OpenGL context current and return the window handle.

    Tries OpenGL 3.3 Core Profile + ``FORWARD_COMPAT`` first, then falls
    back to Compatibility Profile on non-macOS platforms.  NSGL (macOS) does
    not support Compatibility Profile, so only one attempt is made there.

    Each attempt calls ``glfw.init()`` / ``glfw.terminate()`` independently
    so that a failed attempt leaves no stale GLFW state for the next.

    Parameters
    ----------
    width, height : int
        Window dimensions in pixels.
    title : str, optional, default 'WhipperSnapPy'
        Window title.
    visible : bool, optional, default True
        If False create an invisible/offscreen window.

    Returns
    -------
    window or False
        GLFW window handle on success, or False on failure.
    """
    # Core Profile (required on macOS, preferred everywhere).
    window = _try_glfw_window(width, height, title, visible, core_profile=True)
    if window:
        glfw.set_input_mode(window, glfw.STICKY_KEYS, gl.GL_TRUE)
        glfw.make_context_current(window)
        glfw.swap_interval(0)
        return window

    # macOS NSGL does not support Compatibility Profile — don't retry.
    if sys.platform == "darwin":
        return False

    # Non-macOS: retry with Compatibility Profile (helps on some Windows CI
    # runners with software renderers that support compat but not core).
    logger.debug(
        "OpenGL 3.3 Core Profile unavailable; retrying with Compatibility Profile."
    )
    window = _try_glfw_window(width, height, title, visible, core_profile=False)
    if window:
        glfw.set_input_mode(window, glfw.STICKY_KEYS, gl.GL_TRUE)
        glfw.make_context_current(window)
        glfw.swap_interval(0)
        return window

    return False


def init_offscreen_context(width, height):
    """Create an invisible OpenGL context for off-screen rendering.

    Tries up to three paths on Linux; macOS and Windows use GLFW only.

    1. **GLFW invisible window** — used when ``PYOPENGL_PLATFORM`` is not
       ``"egl"`` (i.e. a display is available and EGL was not pre-selected).
       Skipped on Linux when EGL was selected at import time to avoid spurious
       GLX warnings.
    2. **EGL** — used when ``PYOPENGL_PLATFORM=egl`` was set by
       :mod:`~whippersnappy.gl._headless` at import time (no display detected
       and ``libEGL`` is installed).  EGL handles both GPU and CPU (llvmpipe)
       rendering without needing ``/dev/dri`` access — works in Docker without
       ``--device``.
    3. **OSMesa** — CPU software renderer (Linux only).  Used when EGL is not
       installed (``PYOPENGL_PLATFORM=osmesa``) or when EGL context creation
       fails.

    Parameters
    ----------
    width : int
        Render target width in pixels.
    height : int
        Render target height in pixels.

    Returns
    -------
    GLFWwindow or None
        GLFW window handle when GLFW succeeded, ``None`` when EGL or OSMesa
        is used (the context is already current).

    Raises
    ------
    RuntimeError
        If no usable OpenGL context can be created.
    """
    global _offscreen_context

    # --- Step 1: GLFW invisible window ---
    # Skip when PYOPENGL_PLATFORM=egl — OpenGL.GL is already bound to EGL,
    # so a GLFW/GLX attempt would print GLX warnings and fail anyway.
    if os.environ.get("PYOPENGL_PLATFORM") != "egl":
        window = init_window(width, height, visible=False)
        if window:
            return window

    # Steps 2 & 3 are Linux-only.
    if sys.platform != "linux":
        raise RuntimeError(
            "Could not create a GLFW OpenGL context. "
            "On macOS a display connection is required (NSGL does not support "
            "headless rendering). "
            "On Windows ensure a GPU driver or Mesa opengl32.dll is available."
        )

    # --- Step 2: EGL headless rendering ---
    # PYOPENGL_PLATFORM=egl was set by _headless.py before OpenGL.GL was
    # imported (no display detected + libEGL available).  PyOpenGL is already
    # bound to EGL; GLFW was intentionally skipped above.
    if os.environ.get("PYOPENGL_PLATFORM") == "egl":
        logger.debug("Using EGL headless context.")
        try:
            from .egl_context import EGLContext  # noqa: PLC0415
            ctx = EGLContext(width, height)
            ctx.make_current()
            _offscreen_context = ctx
            logger.info("Using EGL headless context (no display required).")
            return None
        except (ImportError, RuntimeError) as exc:
            logger.warning("EGL failed (%s) — falling back to OSMesa.", exc)

    # --- Step 3: OSMesa software rendering ---
    logger.debug("Trying OSMesa software rendering (CPU).")
    try:
        from .osmesa_context import OSMesaContext  # noqa: PLC0415
        ctx = OSMesaContext(width, height)
        ctx.make_current()
        _offscreen_context = ctx
        logger.info("Using OSMesa headless context (CPU, no display or GPU required).")
        return None
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "Could not create any OpenGL context (tried GLFW invisible window and OSMesa). "
            f"Last error: {exc}"
        ) from exc


def terminate_context(window):
    """Release the active OpenGL context regardless of how it was created.

    This is a drop-in replacement for ``glfw.terminate()`` that also
    handles EGL and OSMesa headless paths.  Call it at the end of every
    rendering function instead of calling ``glfw.terminate()`` directly.

    Parameters
    ----------
    window : GLFWwindow or None
        The GLFW window handle returned by :func:`init_offscreen_context`,
        or ``None`` when an EGL or OSMesa offscreen context is active.
    """
    global _offscreen_context
    if _offscreen_context is not None:
        _offscreen_context.destroy()  # type: ignore[union-attr]
        _offscreen_context = None
    else:
        if window:
            glfw.destroy_window(window)
        glfw.terminate()


def capture_window(window):
    """Read the current GL framebuffer and return it as a PIL Image (RGB).

    Works for GLFW windows and all headless contexts (EGL, OSMesa).  When an
    offscreen context is active (``window`` is ``None``) pixels are read via
    ``_offscreen_context.read_pixels()``: EGL reads from its FBO, OSMesa reads
    from its pixel buffer (the default framebuffer).  In both cases there is no
    HiDPI scaling to account for.

    Parameters
    ----------
    window : GLFWwindow or None
        GLFW window handle, or ``None`` when an EGL or OSMesa offscreen
        context is active.

    Returns
    -------
    PIL.Image.Image
        RGB image of the rendered frame, with the vertical flip applied so
        that the origin is at the top-left (image convention).
    """
    global _offscreen_context

    # --- Offscreen path (EGL or OSMesa): delegate to the context's read_pixels() ---
    if _offscreen_context is not None:
        return _offscreen_context.read_pixels()  # type: ignore[union-attr]

    # --- GLFW path: read from the default framebuffer ---
    monitor = glfw.get_primary_monitor()
    if monitor is None:
        # Invisible / offscreen GLFW window — no monitor, no HiDPI scaling.
        x_scale, y_scale = 1.0, 1.0
    else:
        x_scale, y_scale = glfw.get_monitor_content_scale(monitor)
    width, height = glfw.get_framebuffer_size(window)

    logger.debug("Framebuffer size = (%s,%s)", width, height)
    logger.debug("Monitor scale    = (%s,%s)", x_scale, y_scale)

    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
    img_buf = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGB", (width, height), img_buf)
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    if x_scale != 1 or y_scale != 1:
        rwidth = int(round(width / x_scale))
        rheight = int(round(height / y_scale))
        logger.debug("Rescale to       = (%s,%s)", rwidth, rheight)
        image.thumbnail((rwidth, rheight), Image.Resampling.LANCZOS)

    return image


def render_scene(shader, triangles, transform):
    """Render a single draw call using the supplied shader/indices.

    Parameters
    ----------
    shader : int
        OpenGL shader program handle.
    triangles : numpy.ndarray
        Element/index array used for the draw call.
    transform : array-like
        4x4 transform matrix (model/view/projection combined) to upload to
        the shader uniform named ``transform``.

    Raises
    ------
    RuntimeError
        If a GL error occurs during rendering.
    """
    try:
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    except Exception as exc:
        logger.error("glClear failed: %s", exc)
        raise RuntimeError(f"glClear failed: {exc}") from exc

    transform_loc = gl.glGetUniformLocation(shader, "transform")
    gl.glUniformMatrix4fv(transform_loc, 1, gl.GL_FALSE, transform)
    gl.glDrawElements(gl.GL_TRIANGLES, triangles.size, gl.GL_UNSIGNED_INT, None)

    err = gl.glGetError()
    if err != gl.GL_NO_ERROR:
        logger.error("OpenGL error after draw: %s", err)
        raise RuntimeError(f"OpenGL error after draw: {err}")
