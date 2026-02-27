"""GL helper utilities.

Contains the implementation of OpenGL helpers used by the package.
Headless rendering on Linux uses OSMesa (CPU software renderer) via
:class:`~whippersnappy.gl.osmesa_context.OSMesaContext` as a fallback
when no display server or GPU is available.  No EGL or GPU driver is
required for headless operation.

On Linux with no ``DISPLAY`` or ``WAYLAND_DISPLAY`` set,
``PYOPENGL_PLATFORM=osmesa`` is set automatically at import time (before
``OpenGL.GL`` is imported) so that PyOpenGL resolves all function pointers
via ``OSMesaGetProcAddress`` rather than GLX.
"""

import logging
import os
import sys
import warnings
from typing import Any

# On Linux with no display, pre-set PYOPENGL_PLATFORM=osmesa before
# importing OpenGL.GL so PyOpenGL uses OSMesaGetProcAddress for function
# pointer resolution instead of GLX (which returns null pointers without
# an X11/Wayland display).  Has no effect on macOS or Windows.
if (
    sys.platform == "linux"
    and "PYOPENGL_PLATFORM" not in os.environ
    and not os.environ.get("DISPLAY")
    and not os.environ.get("WAYLAND_DISPLAY")
):
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import glfw
import OpenGL.GL as gl
import OpenGL.GL.shaders as shaders
from PIL import Image

from .camera import make_model, make_projection, make_view
from .shaders import get_default_shaders

# Module logger
logger = logging.getLogger(__name__)

# Module-level offscreen context handle (None when GLFW is used instead).
# May hold an OSMesaContext instance on headless environments.
_offscreen_context: Any = None


def create_vao():
    """Create and bind a Vertex Array Object (VAO).

    Returns
    -------
    int
        OpenGL handle for the created VAO.
    """
    vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vao)
    return vao


def compile_shader_program(vertex_src, fragment_src):
    """Compile GLSL vertex and fragment sources and link them into a program.

    Parameters
    ----------
    vertex_src : str
        Vertex shader source code.
    fragment_src : str
        Fragment shader source code.

    Returns
    -------
    int
        OpenGL program handle.
    """
    return shaders.compileProgram(
        shaders.compileShader(vertex_src, gl.GL_VERTEX_SHADER),
        shaders.compileShader(fragment_src, gl.GL_FRAGMENT_SHADER),
    )


def setup_buffers(meshdata, triangles):
    """Create and upload vertex and element buffers for the mesh.

    Parameters
    ----------
    meshdata : numpy.ndarray
        Vertex array with interleaved attributes (position, normal, color).
    triangles : numpy.ndarray
        Face index array.

    Returns
    -------
    (vbo, ebo) : tuple
        OpenGL buffer handles for the VBO and EBO.
    """
    vbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, meshdata.nbytes, meshdata, gl.GL_STATIC_DRAW)

    ebo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
    gl.glBufferData(
        gl.GL_ELEMENT_ARRAY_BUFFER, triangles.nbytes, triangles, gl.GL_STATIC_DRAW
    )

    return vbo, ebo


def setup_vertex_attributes(shader):
    """Configure vertex attribute pointers for position, normal and color.

    Parameters
    ----------
    shader : int
        OpenGL shader program handle used to query attribute locations.
    """
    position = gl.glGetAttribLocation(shader, "aPos")
    gl.glVertexAttribPointer(
        position, 3, gl.GL_FLOAT, gl.GL_FALSE, 9 * 4, gl.ctypes.c_void_p(0)
    )
    gl.glEnableVertexAttribArray(position)

    vnormalpos = gl.glGetAttribLocation(shader, "aNormal")
    gl.glVertexAttribPointer(
        vnormalpos, 3, gl.GL_FLOAT, gl.GL_FALSE, 9 * 4, gl.ctypes.c_void_p(3 * 4)
    )
    gl.glEnableVertexAttribArray(vnormalpos)

    colorpos = gl.glGetAttribLocation(shader, "aColor")
    gl.glVertexAttribPointer(
        colorpos, 3, gl.GL_FLOAT, gl.GL_FALSE, 9 * 4, gl.ctypes.c_void_p(6 * 4)
    )
    gl.glEnableVertexAttribArray(colorpos)


def set_default_gl_state():
    """Set frequently used default OpenGL state for rendering.

    This function enables depth testing and sets a default clear color.
    """
    gl.glClearColor(0.0, 0.0, 0.0, 1.0)
    gl.glEnable(gl.GL_DEPTH_TEST)


def set_camera_uniforms(shader, view, projection, model):
    """Upload camera MVP (view, projection, model) matrices to the shader.

    Parameters
    ----------
    shader : int
        OpenGL shader program handle.
    view, projection, model : array-like
        4x4 matrices to be uploaded to the corresponding shader uniforms.
    """
    view_loc = gl.glGetUniformLocation(shader, "view")
    proj_loc = gl.glGetUniformLocation(shader, "projection")
    model_loc = gl.glGetUniformLocation(shader, "model")
    gl.glUniformMatrix4fv(view_loc, 1, gl.GL_FALSE, view)
    gl.glUniformMatrix4fv(proj_loc, 1, gl.GL_FALSE, projection)
    gl.glUniformMatrix4fv(model_loc, 1, gl.GL_FALSE, model)


def set_lighting_uniforms(shader, specular=True, ambient=0.0, light_color=(1.0, 1.0, 1.0)):
    """Set lighting-related uniforms (specular toggle, ambient, light color).

    Parameters
    ----------
    shader : int
        OpenGL shader program handle.
    specular : bool, optional, default True
        Enable specular highlights.
    ambient : float, optional, default 0.0
        Ambient light strength.
    light_color : tuple, optional, default (1.0, 1.0, 1.0)
        RGB light color.
    """
    specular_loc = gl.glGetUniformLocation(shader, "doSpecular")
    gl.glUniform1i(specular_loc, specular)

    light_color_loc = gl.glGetUniformLocation(shader, "lightColor")
    gl.glUniform3f(light_color_loc, *light_color)

    ambient_loc = gl.glGetUniformLocation(shader, "ambientStrength")
    gl.glUniform1f(ambient_loc, ambient)



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


def init_window(width, height, title="PyOpenGL", visible=True):
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
    title : str, optional, default 'PyOpenGL'
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


def create_window_with_fallback(width, height, title="WhipperSnapPy", visible=True):
    """Create an OpenGL context, trying GLFW first and OSMesa as a fallback.

    The function attempts context creation in this priority order:

    1. **GLFW visible window** — normal path on workstations with a display.
    2. **GLFW invisible window** — when a display exists but no on-screen
       window is needed (e.g. batch rendering).  Core Profile +
       ``FORWARD_COMPAT`` is tried first; Compatibility Profile is retried
       on non-macOS platforms (NSGL does not support Compatibility Profile).
    3. **OSMesa software rendering** — fully headless; no display server,
       no GPU, and no ``/dev/dri/`` devices required.  Only attempted on
       Linux.  Requires ``libosmesa6`` (Debian/Ubuntu) or
       ``mesa-libOSMesa`` (RHEL/Fedora).

    When OSMesa is used the module-level ``_offscreen_context`` is set and
    ``make_current()`` is called so that subsequent OpenGL calls work
    identically to the GLFW path.

    Parameters
    ----------
    width : int
        Render target width in pixels.
    height : int
        Render target height in pixels.
    title : str, optional
        Window title (used for GLFW paths only). Default is ``'WhipperSnapPy'``.
    visible : bool, optional
        Prefer a visible window. Default is ``True``.

    Returns
    -------
    GLFWwindow or None
        GLFW window handle when GLFW succeeded, ``None`` when OSMesa is used
        (the context is already current via ``_offscreen_context.make_current()``).

    Raises
    ------
    RuntimeError
        If all three methods fail to produce a usable OpenGL context.
    """
    global _offscreen_context

    # --- Step 1: GLFW visible window ---
    window = init_window(width, height, title, visible=visible)
    if window:
        return window

    # --- Step 2: GLFW invisible window ---
    if visible:
        logger.warning(
            "Could not create visible GLFW window; retrying with invisible window."
        )
        window = init_window(width, height, title, visible=False)
        if window:
            return window

    # --- Step 3: OSMesa software rendering (Linux headless, no display needed) ---
    # On macOS and Windows GLFW should have succeeded above via the GPU driver.
    # OSMesa is the fallback for Linux environments without a display server.
    # PYOPENGL_PLATFORM=osmesa was already set at module import time (top of
    # this file) when no display was detected, so PyOpenGL already uses
    # OSMesaGetProcAddress for function pointer resolution.
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
            "Could not create any OpenGL context (tried GLFW visible, "
            f"GLFW invisible, OSMesa). Last error: {exc}"
        ) from exc


def terminate_context(window):
    """Release the active OpenGL context regardless of how it was created.

    This is a drop-in replacement for ``glfw.terminate()`` that also
    handles the OSMesa headless path.  Call it at the end of every rendering
    function instead of calling ``glfw.terminate()`` directly.

    Parameters
    ----------
    window : GLFWwindow or None
        The GLFW window handle returned by ``create_window_with_fallback``,
        or ``None`` when an OSMesa context is active.
    """
    global _offscreen_context
    if _offscreen_context is not None:
        _offscreen_context.destroy()  # type: ignore[union-attr]
        _offscreen_context = None
    else:
        glfw.terminate()


def setup_shader(meshdata, triangles, width, height, specular=True, ambient=0.0):
    """Create shader program, upload mesh and initialize camera & lighting.

    This is a convenience wrapper that compiles default shaders, creates
    VAO/VBO/EBO, and configures common uniforms (camera matrices, lighting).

    Parameters
    ----------
    meshdata : numpy.ndarray
        Interleaved vertex data.
    triangles : numpy.ndarray
        Triangle indices.
    width, height : int
        Framebuffer size used to compute projection matrix.
    specular : bool, optional, default True
        Enable specular highlights.
    ambient : float, optional, default 0.0
        Ambient lighting strength.

    Returns
    -------
    shader : int
        Compiled OpenGL shader program handle.
    """
    vertex_shader, fragment_shader = get_default_shaders()

    create_vao()
    shader = compile_shader_program(vertex_shader, fragment_shader)
    setup_buffers(meshdata, triangles)
    setup_vertex_attributes(shader)

    gl.glUseProgram(shader)
    set_default_gl_state()

    view = make_view()
    projection = make_projection(width, height)
    model = make_model()
    set_camera_uniforms(shader, view, projection, model)
    set_lighting_uniforms(shader, specular=specular, ambient=ambient)

    return shader

def capture_window(window):
    """Read the current GL framebuffer and return it as a PIL Image (RGB).

    Works for both GLFW windows and OSMesa headless contexts.  When OSMesa is
    active (``window`` is ``None``) the pixels are read from the FBO that
    was set up by :class:`~whippersnappy.gl.osmesa_context.OSMesaContext`; in
    that case there is no HiDPI scaling to account for.

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

    # --- OSMesa path: read directly from the FBO ---
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
