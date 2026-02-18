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
_EGL_PLATFORM_DEVICE_EXT  = 0x313F


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
        self.width = width
        self.height = height
        self._libegl = None
        self._display = None
        self._surface = None
        self._context = None
        self._config = None
        self.fbo = None
        self._rbo_color = None
        self._rbo_depth = None
        self._init_egl()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_ext_fn(self, name, restype, argtypes):
        """Load an EGL extension function via eglGetProcAddress."""
        addr = self._libegl.eglGetProcAddress(name.encode())
        if not addr:
            raise RuntimeError(
                f"eglGetProcAddress('{name}') returned NULL — "
                f"extension not available on this driver."
            )
        FuncType = ctypes.CFUNCTYPE(restype, *argtypes)
        return FuncType(addr)

    def _init_egl(self):
        import ctypes.util

        egl_name = ctypes.util.find_library("EGL") or "libEGL.so.1"
        try:
            libegl = ctypes.CDLL(egl_name)
        except OSError as e:
            raise RuntimeError(
                f"Could not load {egl_name}. "
                "Install libegl1-mesa and retry."
            ) from e
        self._libegl = libegl  # keep reference alive

        # Set signatures for direct (non-extension) EGL symbols
        libegl.eglGetProcAddress.restype = ctypes.c_void_p
        libegl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
        libegl.eglQueryString.restype = ctypes.c_char_p
        libegl.eglQueryString.argtypes = [ctypes.c_void_p, ctypes.c_int]
        libegl.eglInitialize.restype = ctypes.c_bool
        libegl.eglInitialize.argtypes = [ctypes.c_void_p,
                                         ctypes.POINTER(ctypes.c_int),
                                         ctypes.POINTER(ctypes.c_int)]
        libegl.eglBindAPI.restype = ctypes.c_bool
        libegl.eglBindAPI.argtypes = [ctypes.c_uint]
        libegl.eglChooseConfig.restype = ctypes.c_bool
        libegl.eglChooseConfig.argtypes = [ctypes.c_void_p,
                                           ctypes.POINTER(ctypes.c_int),
                                           ctypes.c_void_p, ctypes.c_int,
                                           ctypes.POINTER(ctypes.c_int)]
        libegl.eglCreatePbufferSurface.restype = ctypes.c_void_p
        libegl.eglCreatePbufferSurface.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                   ctypes.POINTER(ctypes.c_int)]
        libegl.eglCreateContext.restype = ctypes.c_void_p
        libegl.eglCreateContext.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.POINTER(ctypes.c_int)]
        libegl.eglMakeCurrent.restype = ctypes.c_bool
        libegl.eglMakeCurrent.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.c_void_p, ctypes.c_void_p]
        libegl.eglDestroyContext.restype = ctypes.c_bool
        libegl.eglDestroyContext.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        libegl.eglDestroySurface.restype = ctypes.c_bool
        libegl.eglDestroySurface.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        libegl.eglTerminate.restype = ctypes.c_bool
        libegl.eglTerminate.argtypes = [ctypes.c_void_p]

        # Check extensions and load ext functions via eglGetProcAddress
        _EGL_EXTENSIONS = 0x3055
        client_exts = libegl.eglQueryString(None, _EGL_EXTENSIONS) or b""
        logger.debug("EGL client extensions: %s", client_exts.decode())

        has_device_enum = b"EGL_EXT_device_enumeration" in client_exts
        has_platform_base = b"EGL_EXT_platform_base" in client_exts

        display = None
        if has_device_enum and has_platform_base:
            eglQueryDevicesEXT = self._get_ext_fn(
                "eglQueryDevicesEXT",
                ctypes.c_bool,
                [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)],
            )
            eglGetPlatformDisplayEXT = self._get_ext_fn(
                "eglGetPlatformDisplayEXT",
                ctypes.c_void_p,
                [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)],
            )
            display = self._open_device_display(
                eglQueryDevicesEXT, eglGetPlatformDisplayEXT
            )

        if display is None:
            logger.debug("Falling back to eglGetDisplay(EGL_DEFAULT_DISPLAY)")
            libegl.eglGetDisplay.restype = ctypes.c_void_p
            libegl.eglGetDisplay.argtypes = [ctypes.c_void_p]
            display = libegl.eglGetDisplay(ctypes.c_void_p(0))

        if not display:
            raise RuntimeError(
                "Could not obtain any EGL display. "
                "Install libegl1-mesa for CPU rendering."
            )
        self._display = display

        major, minor = ctypes.c_int(0), ctypes.c_int(0)
        if not libegl.eglInitialize(
                self._display, ctypes.byref(major), ctypes.byref(minor)
        ):
            raise RuntimeError("eglInitialize failed.")
        logger.debug("EGL %d.%d", major.value, minor.value)

        if not libegl.eglBindAPI(_EGL_OPENGL_API):
            raise RuntimeError("eglBindAPI(OpenGL) failed.")

        cfg_attribs = (ctypes.c_int * 7)(
            _EGL_SURFACE_TYPE, _EGL_PBUFFER_BIT,
            _EGL_RENDERABLE_TYPE, _EGL_OPENGL_BIT,
            _EGL_NONE,
        )
        configs = (ctypes.c_void_p * 1)()
        num_cfgs = ctypes.c_int(0)
        if not libegl.eglChooseConfig(
                self._display, cfg_attribs, configs, 1, ctypes.byref(num_cfgs)
        ) or num_cfgs.value == 0:
            raise RuntimeError("eglChooseConfig: no suitable config.")
        self._config = configs[0]

        pbuf_attribs = (ctypes.c_int * 5)(
            _EGL_WIDTH, 1, _EGL_HEIGHT, 1, _EGL_NONE
        )
        self._surface = libegl.eglCreatePbufferSurface(
            self._display, self._config, pbuf_attribs
        )
        if not self._surface:
            raise RuntimeError("eglCreatePbufferSurface failed.")

        ctx_attribs = (ctypes.c_int * 5)(
            _EGL_CONTEXT_MAJOR_VERSION, 3,
            _EGL_CONTEXT_MINOR_VERSION, 3,
            _EGL_NONE,
        )
        self._context = libegl.eglCreateContext(
            self._display, self._config, None, ctx_attribs
        )
        if not self._context:
            raise RuntimeError(
                "eglCreateContext for OpenGL 3.3 Core failed. "
                "Try: MESA_GL_VERSION_OVERRIDE=3.3 MESA_GLSL_VERSION_OVERRIDE=330"
            )
        logger.info("EGL context created (%dx%d)", self.width, self.height)


    def _open_device_display(self, eglQueryDevicesEXT, eglGetPlatformDisplayEXT):
        """Enumerate EGL devices and return first usable display pointer."""
        n = ctypes.c_int(0)
        if not eglQueryDevicesEXT(0, None, ctypes.byref(n)) or n.value == 0:
            logger.warning("eglQueryDevicesEXT: no devices.")
            return None
        logger.debug("EGL: %d device(s) found", n.value)
        devices = (ctypes.c_void_p * n.value)()
        eglQueryDevicesEXT(n.value, devices, ctypes.byref(n))
        no_attribs = (ctypes.c_int * 1)(_EGL_NONE)
        for i, dev in enumerate(devices):
            dpy = eglGetPlatformDisplayEXT(
                _EGL_PLATFORM_DEVICE_EXT, ctypes.c_void_p(dev), no_attribs
            )
            if dpy:
                logger.debug("EGL: using device %d", i)
                return dpy
        return None


    def make_current(self):
        """Make this EGL context current and set up the FBO render target.

        Must be called before any OpenGL commands.  Creates and binds an
        FBO backed by two renderbuffers (RGBA color + depth/stencil).
        """
        if not self._libegl.eglMakeCurrent(
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
        libegl = self._libegl
        # GL cleanup first (context must be current)
        if self.fbo is not None:
            gl.glDeleteFramebuffers(1, [self.fbo]);
            self.fbo = None
        if self._rbo_color is not None:
            gl.glDeleteRenderbuffers(1, [self._rbo_color]);
            self._rbo_color = None
        if self._rbo_depth is not None:
            gl.glDeleteRenderbuffers(1, [self._rbo_depth]);
            self._rbo_depth = None
        if self._display:
            libegl.eglMakeCurrent(self._display, None, None, None)
            if self._context: libegl.eglDestroyContext(self._display, self._context)
            if self._surface: libegl.eglDestroySurface(self._display, self._surface)
            libegl.eglTerminate(self._display)
            self._display = self._context = self._surface = None
        logger.debug("EGL context destroyed.")

    # Allow use as a context manager
    def __enter__(self):
        self.make_current()
        return self

    def __exit__(self, *_):
        self.destroy()
