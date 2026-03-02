"""OpenGL context management and rendering pipeline.

Owns the full lifecycle of an OpenGL context — creation (GLFW invisible
window or OSMesa fallback on Linux), scene rendering, framebuffer capture,
and teardown.

Headless rendering on Linux uses OSMesa (CPU software renderer) via
:class:`~whippersnappy.gl.osmesa_context.OSMesaContext` when no display
server or GPU is available.  No EGL or GPU driver is required.

On Linux with no ``DISPLAY`` or ``WAYLAND_DISPLAY`` set,
``PYOPENGL_PLATFORM=osmesa`` is set automatically before ``OpenGL.GL`` is
imported.  The guard lives in :mod:`whippersnappy.gl._headless`, which is
imported first both here and in every other GL submodule, so the variable
is always set in time regardless of which submodule is imported first.
"""
# ruff: noqa: I001  — import order is intentional: _headless must precede OpenGL.GL

import logging
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
# May hold an OSMesaContext instance on headless environments.
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

    Always creates an invisible GLFW window — no title is shown.
    Interactive GUI code calls :func:`init_window` directly with
    ``visible=True``.

    On Linux, if GLFW fails (no display server), falls back to OSMesa
    software rendering.  On macOS and Windows, GLFW is the only supported
    path — a ``RuntimeError`` is raised if it fails.

    Parameters
    ----------
    width : int
        Render target width in pixels.
    height : int
        Render target height in pixels.

    Returns
    -------
    GLFWwindow or None
        GLFW window handle when GLFW succeeded, ``None`` when OSMesa is used
        (the context is already current via ``_offscreen_context.make_current()``).

    Raises
    ------
    RuntimeError
        If no usable OpenGL context can be created.  On Linux this means
        both GLFW and OSMesa failed.  On macOS/Windows it means GLFW failed
        (those platforms have no OSMesa fallback).
    """
    global _offscreen_context

    # --- Step 1: GLFW invisible window ---
    window = init_window(width, height, visible=False)
    if window:
        return window

    # --- Step 2: OSMesa software rendering (Linux headless only) ---
    # Only reached on Linux when GLFW failed entirely (no display server).
    # On macOS and Windows GLFW is the only supported path; raise a clear
    # platform-specific error instead of a confusing libOSMesa hint.
    if sys.platform != "linux":
        raise RuntimeError(
            "Could not create a GLFW OpenGL context. "
            "On macOS a display connection is required (NSGL does not support "
            "headless rendering). "
            "On Windows ensure a GPU driver or Mesa opengl32.dll is available."
        )
    # PYOPENGL_PLATFORM=osmesa was set at module import time when no display
    # was detected, so PyOpenGL uses OSMesaGetProcAddress for all GL calls.
    logger.info("No display detected — trying OSMesa software rendering (CPU).")
    try:
        from .osmesa_context import OSMesaContext  # noqa: PLC0415
        ctx = OSMesaContext(width, height)
        ctx.make_current()
        _offscreen_context = ctx
        logger.info("Using OSMesa headless context — no display server or GPU required.")
        return None
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "Could not create any OpenGL context (tried GLFW invisible window "
            f"and OSMesa). Last error: {exc}"
        ) from exc


def terminate_context(window):
    """Release the active OpenGL context regardless of how it was created.

    This is a drop-in replacement for ``glfw.terminate()`` that also
    handles the OSMesa headless path.  Call it at the end of every rendering
    function instead of calling ``glfw.terminate()`` directly.

    Parameters
    ----------
    window : GLFWwindow or None
        The GLFW window handle returned by :func:`init_offscreen_context`,
        or ``None`` when an OSMesa context is active.
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

    Works for both GLFW windows and OSMesa headless contexts.  When OSMesa is
    active (``window`` is ``None``) the pixels are read directly from the
    OSMesa pixel buffer, which acts as the default framebuffer (FBO 0) — no
    explicit FBO is created by :class:`~whippersnappy.gl.osmesa_context.OSMesaContext`;
    in that case there is no HiDPI scaling to account for.

    Parameters
    ----------
    window : GLFWwindow or None
        GLFW window handle, or ``None`` when an OSMesa context is active.

    Returns
    -------
    PIL.Image.Image
        RGB image of the rendered frame, with the vertical flip applied so
        that the origin is at the top-left (image convention).
    """
    global _offscreen_context

    # --- OSMesa path: read from the OSMesa pixel buffer (default framebuffer) ---
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
