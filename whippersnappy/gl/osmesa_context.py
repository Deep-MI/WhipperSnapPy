"""OSMesa off-screen (headless) OpenGL context via software rendering.

This module provides a drop-in alternative to GLFW window creation for
headless environments (CI, Docker, HPC clusters) where no X11/Wayland
display is available and no GPU is required.  It requires:

  - The system OSMesa library (``libosmesa6`` on Debian/Ubuntu,
    ``mesa-libOSMesa`` on RHEL/Fedora — loaded via ctypes at runtime).
  - PyOpenGL >= 3.1 (already a project dependency).
  - No GPU, no ``/dev/dri/`` devices, no display server.

**Important:** ``PYOPENGL_PLATFORM=osmesa`` must be set in the environment
*before* the first ``import OpenGL.GL`` anywhere in the process.
:func:`~whippersnappy.gl.utils.create_window_with_fallback` takes care of
this automatically.

Typical usage (internal, called from ``create_window_with_fallback``)::

    import os
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"   # must be first
    from whippersnappy.gl.osmesa_context import OSMesaContext

    ctx = OSMesaContext(width, height)
    ctx.make_current()
    # ... OpenGL calls render into ctx's internal pixel buffer ...
    img = ctx.read_pixels()
    ctx.destroy()

Notes
-----
OSMesa renders into its own pixel buffer which acts as the default
framebuffer (FBO 0).  No explicit FBO creation is needed — ``glReadPixels``
reads directly from the OSMesa buffer.
"""

import ctypes
import ctypes.util
import logging
import sys

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
    """Try several candidate names and return the loaded ctypes CDLL.

    Candidate names are platform-specific but we always try all of them so
    that e.g. a macOS user with a non-default Homebrew prefix still has a
    chance of success.
    """
    if sys.platform == "win32":
        candidates = ["osmesa", "osmesa.dll", "libOSMesa.dll"]
    elif sys.platform == "darwin":
        # Homebrew mesa does NOT build libOSMesa — a custom build with
        # -Dosmesa=true is needed.  We still try in case the user has one.
        candidates = ["libOSMesa.dylib", "libOSMesa.8.dylib"]
    else:  # Linux / WSL2
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

    if sys.platform == "win32":
        hint = "Install via MSYS2: pacman -S mingw-w64-x86_64-mesa"
    elif sys.platform == "darwin":
        hint = (
            "libOSMesa is not available via 'brew install mesa' (Homebrew does not "
            "build OSMesa). A custom Mesa build with -Dosmesa=true is required, or "
            "use a display/GLFW for rendering on macOS."
        )
    else:
        hint = (
            "Install with: sudo apt-get install libosmesa6  (Debian/Ubuntu) "
            "or sudo dnf install mesa-libOSMesa  (RHEL/Fedora)"
        )

    raise RuntimeError(f"Could not load libOSMesa ({last_err}). {hint}")


class OSMesaContext:
    """A headless OpenGL context backed by OSMesa software rendering.

    OSMesa renders into its own pixel buffer which serves as the default
    framebuffer.  No explicit FBO is required — ``glReadPixels`` reads
    directly from the OSMesa buffer.

    ``PYOPENGL_PLATFORM=osmesa`` **must** already be set in the environment
    before this class is imported (handled by
    :func:`~whippersnappy.gl.utils.create_window_with_fallback`).

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
            _OSMESA_RGBA, 24, 8, 0, None,
        )
        if not ctx:
            raise RuntimeError(
                "OSMesaCreateContextExt failed. "
                "Try: MESA_GL_VERSION_OVERRIDE=3.3 MESA_GLSL_VERSION_OVERRIDE=330"
            )
        self._ctx = ctx

        buf_size = self.width * self.height * 4  # RGBA bytes
        self._buf = (ctypes.c_ubyte * buf_size)()
        logger.info("OSMesa context created (%dx%d)", self.width, self.height)

    def make_current(self):
        """Make this OSMesa context current.

        OSMesa renders into its own internal pixel buffer which acts as the
        default framebuffer (FBO 0).  No FBO creation is needed.
        ``PYOPENGL_PLATFORM=osmesa`` must already be set so that PyOpenGL
        resolves function pointers via OSMesaGetProcAddress.
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

        Reads from the default framebuffer (FBO 0), which is OSMesa's own
        pixel buffer.  No FBO bind needed.

        Returns
        -------
        PIL.Image.Image
            Captured frame, vertically flipped to top-left origin convention.
        """
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
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

