"""EGL off-screen (headless) OpenGL context via pbuffer + FBO.

This module provides a drop-in alternative to GLFW window creation for
headless environments (CI, Docker, HPC clusters) where no X11/Wayland
display is available.  It requires:

  - A system EGL library (``libegl1`` on Debian/Ubuntu, already present
    in the WhipperSnapPy Dockerfile).
  - PyOpenGL >= 3.1 (already a project dependency), which ships
    ``OpenGL.EGL`` bindings.
  - Either an NVIDIA GPU with the EGL driver, or Mesa ``libEGL-mesa0``
    (llvmpipe software renderer) for CPU-only systems.

Typical usage (internal, called from ``create_window_with_fallback``)::

    from whippersnappy.gl.egl_context import EGLContext

    ctx = EGLContext(width, height)
    ctx.make_current()
    # ... OpenGL calls ...
    img = ctx.read_pixels()
    ctx.destroy()
"""

import ctypes
import logging

import OpenGL.GL as gl
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EGL constants not exposed by all PyOpenGL versions
# ---------------------------------------------------------------------------
_EGL_SURFACE_TYPE         = 0x3033
_EGL_PBUFFER_BIT          = 0x0001
_EGL_RENDERABLE_TYPE      = 0x3040
_EGL_OPENGL_BIT           = 0x0008
_EGL_NONE                 = 0x3038
_EGL_WIDTH                = 0x3057
_EGL_HEIGHT               = 0x3056
_EGL_OPENGL_API           = 0x30A2
_EGL_CONTEXT_MAJOR_VERSION = 0x3098
_EGL_CONTEXT_MINOR_VERSION = 0x30FB


def _check_egl_available():
    """Raise ImportError with a helpful message if EGL bindings are absent."""
    try:
        from OpenGL import EGL as _EGL  # noqa: F401
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "OpenGL.EGL is not available. Make sure pyopengl >= 3.1 is "
            "installed and libegl1 (or equivalent) is present on the system."
        ) from exc


class EGLContext:
    """A headless OpenGL 3.3 Core context backed by an EGL pbuffer + FBO.

    The pbuffer surface is created solely to satisfy EGL's requirement for
    a surface when calling ``eglMakeCurrent``.  All rendering is directed
    into an off-screen Framebuffer Object (FBO) so that ``glReadPixels``
    captures exactly what was rendered regardless of platform quirks with
    pbuffer readback.

    Parameters
    ----------
    width, height : int
        Dimensions of the off-screen render target in pixels.

    Attributes
    ----------
    width, height : int
        Render target dimensions.
    fbo : int
        OpenGL FBO handle (valid after ``make_current`` is called).

    Raises
    ------
    ImportError
        If ``OpenGL.EGL`` bindings are not available.
    RuntimeError
        If any EGL initialisation step fails.
    """

    def __init__(self, width: int, height: int):
        _check_egl_available()
        from OpenGL import EGL

        self._EGL = EGL
        self.width = width
        self.height = height
        self._display = None
        self._surface = None
        self._context = None
        self.fbo = None
        self._rbo_color = None
        self._rbo_depth = None

        self._init_egl()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_egl(self):
        EGL = self._EGL

        # 1. Get the default EGL display (works for both GPU and Mesa)
        self._display = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
        if self._display == EGL.EGL_NO_DISPLAY:
            raise RuntimeError("eglGetDisplay returned EGL_NO_DISPLAY.")

        major = ctypes.c_int(0)
        minor = ctypes.c_int(0)
        if not EGL.eglInitialize(self._display, major, minor):
            raise RuntimeError("eglInitialize failed.")
        logger.debug("EGL version %d.%d", major.value, minor.value)

        # 2. Bind the OpenGL API (not OpenGL ES)
        if not EGL.eglBindAPI(_EGL_OPENGL_API):
            raise RuntimeError("eglBindAPI(OpenGL) failed.")

        # 3. Choose a framebuffer config
        cfg_attribs = (ctypes.c_int * 7)(
            _EGL_SURFACE_TYPE,    _EGL_PBUFFER_BIT,
            _EGL_RENDERABLE_TYPE, _EGL_OPENGL_BIT,
            _EGL_NONE,
        )
        configs = (EGL.EGLConfig * 1)()
        num_cfg = ctypes.c_int(0)
        if not EGL.eglChooseConfig(
            self._display, cfg_attribs, configs, 1, ctypes.byref(num_cfg)
        ) or num_cfg.value == 0:
            raise RuntimeError(
                "eglChooseConfig found no suitable configs. "
                "Ensure a Mesa or GPU EGL driver is installed (libegl1-mesa or libegl1)."
            )

        # 4. Create a minimal pbuffer surface (1×1 is sufficient — rendering
        #    goes into the FBO, not this surface)
        pbuf_attribs = (ctypes.c_int * 5)(
            _EGL_WIDTH,  1,
            _EGL_HEIGHT, 1,
            _EGL_NONE,
        )
        self._surface = EGL.eglCreatePbufferSurface(
            self._display, configs[0], pbuf_attribs
        )
        if self._surface == EGL.EGL_NO_SURFACE:
            raise RuntimeError("eglCreatePbufferSurface failed.")

        # 5. Create an OpenGL 3.3 Core context
        ctx_attribs = (ctypes.c_int * 5)(
            _EGL_CONTEXT_MAJOR_VERSION, 3,
            _EGL_CONTEXT_MINOR_VERSION, 3,
            _EGL_NONE,
        )
        self._context = EGL.eglCreateContext(
            self._display, configs[0], EGL.EGL_NO_CONTEXT, ctx_attribs
        )
        if self._context == EGL.EGL_NO_CONTEXT:
            raise RuntimeError(
                "eglCreateContext failed. "
                "The EGL driver may not support OpenGL 3.3 Core. "
                "Check with: glxinfo | grep 'OpenGL version'"
            )

        logger.info("EGL headless context created (%dx%d)", self.width, self.height)

    def make_current(self):
        """Make this EGL context current and set up the FBO render target.

        Must be called before any OpenGL commands.  Creates and binds an
        FBO backed by two renderbuffers (RGBA color + depth/stencil).
        """
        EGL = self._EGL
        if not EGL.eglMakeCurrent(
            self._display, self._surface, self._surface, self._context
        ):
            raise RuntimeError("eglMakeCurrent failed.")

        # Build FBO so rendering is directed off-screen
        self.fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)

        # Color renderbuffer
        self._rbo_color = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._rbo_color)
        gl.glRenderbufferStorage(
            gl.GL_RENDERBUFFER, gl.GL_RGBA8, self.width, self.height
        )
        gl.glFramebufferRenderbuffer(
            gl.GL_FRAMEBUFFER,
            gl.GL_COLOR_ATTACHMENT0,
            gl.GL_RENDERBUFFER,
            self._rbo_color,
        )

        # Depth + stencil renderbuffer
        self._rbo_depth = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._rbo_depth)
        gl.glRenderbufferStorage(
            gl.GL_RENDERBUFFER,
            gl.GL_DEPTH24_STENCIL8,
            self.width,
            self.height,
        )
        gl.glFramebufferRenderbuffer(
            gl.GL_FRAMEBUFFER,
            gl.GL_DEPTH_STENCIL_ATTACHMENT,
            gl.GL_RENDERBUFFER,
            self._rbo_depth,
        )

        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        if status != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(
                f"FBO is not complete after EGL setup (status=0x{status:X})."
            )

        # Set the viewport to match the render target
        gl.glViewport(0, 0, self.width, self.height)
        logger.debug("EGL FBO complete and bound (%dx%d)", self.width, self.height)

    def read_pixels(self) -> Image.Image:
        """Read the FBO contents and return a PIL RGB Image.

        Returns
        -------
        PIL.Image.Image
            Captured frame, vertically flipped to convert from OpenGL's
            bottom-left origin to image top-left convention.
        """
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        buf = gl.glReadPixels(
            0, 0, self.width, self.height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE
        )
        img = Image.frombytes("RGB", (self.width, self.height), buf)
        return img.transpose(Image.FLIP_TOP_BOTTOM)

    def destroy(self):
        """Release the FBO, renderbuffers, EGL context and surface.

        Safe to call multiple times; subsequent calls are no-ops.
        """
        EGL = self._EGL

        # Clean up GL objects first (context must still be current)
        if self.fbo is not None:
            gl.glDeleteFramebuffers(1, [self.fbo])
            self.fbo = None
        if self._rbo_color is not None:
            gl.glDeleteRenderbuffers(1, [self._rbo_color])
            self._rbo_color = None
        if self._rbo_depth is not None:
            gl.glDeleteRenderbuffers(1, [self._rbo_depth])
            self._rbo_depth = None

        if self._display is not None:
            EGL.eglMakeCurrent(
                self._display,
                EGL.EGL_NO_SURFACE,
                EGL.EGL_NO_SURFACE,
                EGL.EGL_NO_CONTEXT,
            )
            if self._context is not None:
                EGL.eglDestroyContext(self._display, self._context)
                self._context = None
            if self._surface is not None:
                EGL.eglDestroySurface(self._display, self._surface)
                self._surface = None
            EGL.eglTerminate(self._display)
            self._display = None

        logger.debug("EGL context destroyed.")

    # Allow use as a context manager
    def __enter__(self):
        self.make_current()
        return self

    def __exit__(self, *_):
        self.destroy()
