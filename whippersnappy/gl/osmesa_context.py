"""OSMesa off-screen (headless) OpenGL context via software rendering.

This module provides a drop-in alternative to GLFW window creation for
headless environments (CI, Docker, HPC clusters) where no X11/Wayland
display is available and no GPU is required.  It requires:

  - The system OSMesa library (``libosmesa6`` on Debian/Ubuntu,
    ``mesa-libOSMesa`` on RHEL/Fedora — loaded via ctypes at runtime).
  - PyOpenGL >= 3.1 (already a project dependency).
  - No GPU, no ``/dev/dri/`` devices, no display server.

This module is not intended to be used directly.  It is instantiated by
:func:`~whippersnappy.gl.utils.create_window_with_fallback` when GLFW
cannot create a window (Linux headless).  That function also sets
``PYOPENGL_PLATFORM=osmesa`` at import time so that PyOpenGL resolves
function pointers via ``OSMesaGetProcAddress``.

Notes
-----
OSMesa renders into its own pixel buffer which acts as the default
framebuffer (FBO 0).  No explicit FBO creation is needed — ``glReadPixels``
reads directly from the OSMesa buffer.
"""

import ctypes
import ctypes.util
import logging

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
    """Try several candidate library names and return the loaded ctypes CDLL.

    Only called on Linux — OSMesa is not attempted on macOS or Windows
    (GLFW handles those platforms).
    """
    candidates = ["libOSMesa.so.8", "libOSMesa.so.6", "libOSMesa.so"]

    # ctypes.util.find_library may resolve a shorter name to the real path
    found = ctypes.util.find_library("OSMesa")
    if found and found not in candidates:
        candidates.insert(0, found)

    last_err = None
    for name in candidates:
        try:
            lib = ctypes.CDLL(name)
            _ = lib.OSMesaCreateContextExt
            logger.debug("Loaded OSMesa from: %s", name)
            return lib
        except (OSError, AttributeError) as exc:
            last_err = exc

    raise RuntimeError(
        f"Could not load libOSMesa ({last_err}). "
        "Install with:  sudo apt-get install libosmesa6  (Debian/Ubuntu) "
        "or:  sudo dnf install mesa-libOSMesa  (RHEL/Fedora)"
    )


class OSMesaContext:
    """A headless OpenGL context backed by OSMesa software rendering.

    OSMesa renders into its own pixel buffer which serves as the default
    framebuffer.  No explicit FBO is required — ``glReadPixels`` reads
    directly from the OSMesa buffer.

    Parameters
    ----------
    width, height : int
        Dimensions of the off-screen render target in pixels.

    Raises
    ------
    RuntimeError
        If libOSMesa cannot be loaded or context creation fails.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self._libosmesa = None
        self._ctx = None
        self._buf = None
        self._init_osmesa()

    def _init_osmesa(self):
        lib = _load_libosmesa()
        self._libosmesa = lib

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

    def make_current(self):
        """Make this OSMesa context current.

        ``OSMesaMakeCurrent`` binds ``self._buf`` as the default framebuffer
        (FBO 0) of this context.  The buffer already has colour (RGBA8),
        depth (24-bit) and stencil (8-bit) attached — requested via
        ``OSMesaCreateContextExt`` — so no explicit FBO creation is needed.
        All rendering goes into ``self._buf`` automatically.
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

        gl.glViewport(0, 0, self.width, self.height)
        logger.debug("OSMesa context is current (%dx%d)", self.width, self.height)

    def read_pixels(self) -> Image.Image:
        """Read the OSMesa framebuffer and return a PIL RGB Image.

        FBO 0 is always current (set by ``OSMesaMakeCurrent``), so
        ``glReadPixels`` reads from ``self._buf`` directly.

        Returns
        -------
        PIL.Image.Image
            Captured frame, vertically flipped to top-left origin convention.
        """
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        buf = gl.glReadPixels(
            0, 0, self.width, self.height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE
        )
        img = Image.frombytes("RGB", (self.width, self.height), buf)
        return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    def destroy(self):
        """Destroy the OSMesa context and release the pixel buffer."""
        if self._ctx is not None:
            self._libosmesa.OSMesaDestroyContext(self._ctx)
            self._ctx = None
        self._buf = None
        logger.debug("OSMesa context destroyed.")

    def __enter__(self):
        self.make_current()
        return self

    def __exit__(self, *_):
        self.destroy()

