"""OSMesa off-screen (headless) OpenGL context via software rendering + FBO.

This module provides a drop-in alternative to GLFW window creation for
headless environments (CI, Docker, HPC clusters) where no X11/Wayland
display is available and no GPU is required.  It requires:

  - The system OSMesa library (``libosmesa6`` on Debian/Ubuntu,
    ``mesa-libOSMesa`` on RHEL/Fedora — loaded via ctypes at runtime).
  - PyOpenGL >= 3.1 (already a project dependency).
  - No GPU, no ``/dev/dri/`` devices, no display server.

Typical usage (internal, called from ``create_window_with_fallback``)::

    from whippersnappy.gl.osmesa_context import OSMesaContext

    ctx = OSMesaContext(width, height)
    ctx.make_current()
    # ... OpenGL calls ...
    img = ctx.read_pixels()
    ctx.destroy()

Notes
-----
OSMesa renders entirely in software (CPU).  The OpenGL API surface is
identical to EGL or GLFW — shaders, FBOs, VAOs etc. all work unchanged.
"""

import ctypes
import ctypes.util
import logging
import sys

if sys.platform != "linux":
    raise ImportError(
        "OSMesaContext is only supported on Linux. "
        "On macOS use GLFW/CGL and on Windows use GLFW/WGL instead."
    )

import OpenGL.GL as gl
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OSMesa constants
# ---------------------------------------------------------------------------
_OSMESA_RGBA         = 0x1908
_OSMESA_ROW_LENGTH   = 0x10
_OSMESA_Y_UP         = 0x11
_GL_UNSIGNED_BYTE    = 0x1401


def _load_libosmesa():
    """Try several candidate names and return the loaded ctypes CDLL."""
    candidates = ["OSMesa", "libOSMesa.so.8", "libOSMesa.so.6", "libOSMesa.so"]
    # ctypes.util.find_library may give us a name that LoadLibrary knows
    found = ctypes.util.find_library("OSMesa")
    if found:
        candidates.insert(0, found)
    for name in candidates:
        try:
            lib = ctypes.CDLL(name)
            # Verify it has the symbol we need
            _ = lib.OSMesaCreateContextExt
            logger.debug("Loaded OSMesa from: %s", name)
            return lib
        except (OSError, AttributeError):
            continue
    raise RuntimeError(
        "Could not load libOSMesa. "
        "Install it with:  sudo apt-get install libosmesa6  (Debian/Ubuntu) "
        "or:  sudo dnf install mesa-libOSMesa  (RHEL/Fedora)."
    )


class OSMesaContext:
    """A headless OpenGL 3.3-compatible context backed by OSMesa + FBO.

    OSMesa renders entirely in software (CPU).  The rendered pixels are
    read via the same FBO + ``glReadPixels`` path used by the EGL context,
    so the output is identical.

    Parameters
    ----------
    width, height : int
        Dimensions of the off-screen render target in pixels.

    Raises
    ------
    ImportError
        If run on a non-Linux platform.
    RuntimeError
        If libOSMesa cannot be loaded or context creation fails.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self._libosmesa = None
        self._ctx = None
        self._buf = None
        self.fbo = None
        self._rbo_color = None
        self._rbo_depth = None
        self._init_osmesa()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_osmesa(self):
        lib = _load_libosmesa()
        self._libosmesa = lib

        # Set ctypes signatures
        lib.OSMesaCreateContextExt.restype  = ctypes.c_void_p
        lib.OSMesaCreateContextExt.argtypes = [
            ctypes.c_uint,   # format  (OSMESA_RGBA)
            ctypes.c_int,    # depthBits (24)
            ctypes.c_int,    # stencilBits (8)
            ctypes.c_int,    # accumBits (0)
            ctypes.c_void_p, # sharelist (None)
        ]
        lib.OSMesaMakeCurrent.restype  = ctypes.c_bool
        lib.OSMesaMakeCurrent.argtypes = [
            ctypes.c_void_p,  # ctx
            ctypes.c_void_p,  # buffer
            ctypes.c_uint,    # type (GL_UNSIGNED_BYTE)
            ctypes.c_int,     # width
            ctypes.c_int,     # height
        ]
        lib.OSMesaDestroyContext.restype  = None
        lib.OSMesaDestroyContext.argtypes = [ctypes.c_void_p]

        ctx = lib.OSMesaCreateContextExt(
            _OSMESA_RGBA,  # pixel format
            24,            # depth bits
            8,             # stencil bits
            0,             # accum bits
            None,          # no shared context
        )
        if not ctx:
            raise RuntimeError(
                "OSMesaCreateContextExt failed. "
                "Try: MESA_GL_VERSION_OVERRIDE=3.3 MESA_GLSL_VERSION_OVERRIDE=330"
            )
        self._ctx = ctx

        # Allocate the pixel buffer that OSMesa renders into
        buf_size = self.width * self.height * 4  # RGBA bytes
        self._buf = (ctypes.c_ubyte * buf_size)()
        logger.info("OSMesa context created (%dx%d)", self.width, self.height)

    # ------------------------------------------------------------------
    # Public API (same as EGLContext)
    # ------------------------------------------------------------------

    def make_current(self):
        """Make this OSMesa context current and set up the FBO render target.

        Must be called before any OpenGL commands.  Creates and binds an
        FBO backed by two renderbuffers (RGBA color + depth/stencil).
        """
        ok = self._libosmesa.OSMesaMakeCurrent(
            self._ctx,
            self._buf,
            _GL_UNSIGNED_BYTE,
            self.width,
            self.height,
        )
        if not ok:
            raise RuntimeError("OSMesaMakeCurrent failed.")

        # Force PyOpenGL to discover and cache the context we just made current.
        gl.glGetError()

        # Build FBO so rendering is directed off-screen into a proper
        # renderbuffer (same pattern as EGLContext.make_current)
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
                f"FBO is not complete after OSMesa setup (status=0x{status:X})."
            )

        gl.glViewport(0, 0, self.width, self.height)
        logger.debug("OSMesa FBO complete and bound (%dx%d)", self.width, self.height)

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
        return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    def destroy(self):
        """Release the FBO resources and destroy the OSMesa context."""
        # GL cleanup first (context must still be current)
        if self.fbo is not None:
            gl.glDeleteFramebuffers(1, [self.fbo])
            self.fbo = None
        if self._rbo_color is not None:
            gl.glDeleteRenderbuffers(1, [self._rbo_color])
            self._rbo_color = None
        if self._rbo_depth is not None:
            gl.glDeleteRenderbuffers(1, [self._rbo_depth])
            self._rbo_depth = None
        if self._ctx is not None:
            self._libosmesa.OSMesaDestroyContext(self._ctx)
            self._ctx = None
        self._buf = None
        logger.debug("OSMesa context destroyed.")

    # Allow use as a context manager
    def __enter__(self):
        self.make_current()
        return self

    def __exit__(self, *_):
        self.destroy()


