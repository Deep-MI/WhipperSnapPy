"""Headless OpenGL platform detection.

This module MUST be imported before any ``import OpenGL.GL`` statement in the
package.  On Linux with no display it sets ``PYOPENGL_PLATFORM`` so that
PyOpenGL resolves function pointers via the correct backend before
``OpenGL.GL`` is first imported.

Priority chain on Linux when no display is detected
(``DISPLAY`` / ``WAYLAND_DISPLAY`` unset):

1. **EGL** — tried first when ``libEGL`` is installed.  A lightweight ctypes
   probe (``eglGetDisplay`` + ``eglInitialize``) confirms EGL can actually
   initialise before ``PYOPENGL_PLATFORM=egl`` is set.  This covers both GPU
   rendering (real device) and CPU software rendering (Mesa llvmpipe) —
   works in Docker without ``--device``.
2. **OSMesa** — fallback when EGL is not installed or the probe fails (e.g.
   no GPU and no llvmpipe).  Sets ``PYOPENGL_PLATFORM=osmesa``.
3. **Neither** — raises ``RuntimeError`` with install instructions.

When ``DISPLAY`` is set the module does not intervene; GLFW is tried first
in :func:`~whippersnappy.gl.context.init_offscreen_context`.  If GLFW then
fails (e.g. broken ``ssh -X`` forward), the same EGL/OSMesa chain is
attempted there.

``PYOPENGL_PLATFORM`` is not consulted by GLFW, so setting it here does not
affect the interactive GUI (``whippersnap``).

No OpenGL, GLFW, or other heavy imports are done here — only stdlib.
"""

import ctypes
import logging
import os
import sys

logger = logging.getLogger(__name__)


def _egl_is_available():
    """Return True if libEGL can be loaded via ctypes."""
    for name in ("libEGL.so.1", "libEGL.so"):
        try:
            ctypes.CDLL(name)
            return True
        except OSError:
            continue
    return False


def _egl_context_works():
    """Probe EGL via ctypes to confirm a context can actually be created headlessly.

    Tries display-independent EGL paths in order:

    1. ``EGL_EXT_device_enumeration`` — enumerate GPU devices directly; works
       headlessly without a display server.  With ``--gpus all`` (NVIDIA) or
       ``--device`` (AMD/Intel) the GPU device appears here and is preferred.
    2. ``EGL_MESA_platform_surfaceless`` — Mesa CPU software rendering
       (llvmpipe); no GPU or display server needed.  Used when no GPU device
       is found (e.g. Docker without ``--gpus``/``--device``).
    3. ``eglGetDisplay(EGL_DEFAULT_DISPLAY)`` — last resort; only succeeds
       when a display server (X11/Wayland) is reachable.

    No ``OpenGL.GL`` import and no ``PYOPENGL_PLATFORM`` change are made.
    Returns ``True`` only when EGL can actually initialise a display.
    """
    for lib_name in ("libEGL.so.1", "libEGL.so"):
        try:
            libegl = ctypes.CDLL(lib_name)
            break
        except OSError:
            continue
    else:
        logger.debug("EGL probe: libEGL not loadable.")
        return False

    try:
        libegl.eglGetProcAddress.restype  = ctypes.c_void_p
        libegl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
        libegl.eglQueryString.restype     = ctypes.c_char_p
        libegl.eglQueryString.argtypes    = [ctypes.c_void_p, ctypes.c_int]
        libegl.eglGetDisplay.restype      = ctypes.c_void_p
        libegl.eglGetDisplay.argtypes     = [ctypes.c_void_p]
        libegl.eglInitialize.restype      = ctypes.c_bool
        libegl.eglInitialize.argtypes     = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        libegl.eglTerminate.restype  = ctypes.c_bool
        libegl.eglTerminate.argtypes = [ctypes.c_void_p]

        _EGL_EXTENSIONS      = 0x3055
        _EGL_NONE            = 0x3038
        _EGL_PLATFORM_DEVICE = 0x313F

        def _try_init(dpy):
            if not dpy:
                return False
            major, minor = ctypes.c_int(0), ctypes.c_int(0)
            ok = libegl.eglInitialize(dpy, ctypes.byref(major), ctypes.byref(minor))
            libegl.eglTerminate(dpy)
            if ok:
                logger.debug("EGL probe: eglInitialize OK (EGL %d.%d).",
                             major.value, minor.value)
            return bool(ok)

        client_exts = libegl.eglQueryString(None, _EGL_EXTENSIONS) or b""
        logger.debug("EGL client extensions: %s", client_exts.decode())

        _GetPlatformDisplayEXT = None
        if b"EGL_EXT_platform_base" in client_exts:
            addr = libegl.eglGetProcAddress(b"eglGetPlatformDisplayEXT")
            if addr:
                _GetPlatformDisplayEXT = ctypes.CFUNCTYPE(
                    ctypes.c_void_p,
                    ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                )(addr)

        no_attribs = (ctypes.c_int * 1)(_EGL_NONE)

        # --- Path 1: EGL_EXT_device_enumeration ---
        # GPU devices — preferred. With --gpus all (NVIDIA) or --device
        # (AMD/Intel) the GPU appears here before surfaceless/llvmpipe.
        if (_GetPlatformDisplayEXT
                and b"EGL_EXT_device_enumeration" in client_exts):
            addr = libegl.eglGetProcAddress(b"eglQueryDevicesEXT")
            if addr:
                _QueryDevices = ctypes.CFUNCTYPE(
                    ctypes.c_bool,
                    ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                )(addr)
                n = ctypes.c_int(0)
                if _QueryDevices(0, None, ctypes.byref(n)) and n.value > 0:
                    logger.info("EGL probe: %d EGL device(s) found via enumeration.", n.value)
                    devices = (ctypes.c_void_p * n.value)()
                    _QueryDevices(n.value, devices, ctypes.byref(n))
                    for dev in devices:
                        dpy = _GetPlatformDisplayEXT(
                            _EGL_PLATFORM_DEVICE,
                            ctypes.c_void_p(dev),
                            no_attribs,
                        )
                        if _try_init(dpy):
                            logger.debug("EGL probe: device enumeration succeeded (GPU).")
                            return True
                else:
                    logger.info("EGL probe: device enumeration found 0 devices.")

        # --- Path 2: EGL_MESA_platform_surfaceless ---
        # CPU software rendering (llvmpipe) — no GPU needed.
        _EGL_PLATFORM_SURFACELESS = 0x31DD
        if _GetPlatformDisplayEXT and b"EGL_MESA_platform_surfaceless" in client_exts:
            dpy = _GetPlatformDisplayEXT(
                _EGL_PLATFORM_SURFACELESS, ctypes.c_void_p(0), no_attribs
            )
            if _try_init(dpy):
                logger.debug("EGL probe: surfaceless platform succeeded (CPU/llvmpipe).")
                return True

        # --- Path 3: EGL_DEFAULT_DISPLAY ---
        # Works only when a display server is reachable (DISPLAY set).
        dpy = libegl.eglGetDisplay(ctypes.c_void_p(0))
        if _try_init(dpy):
            logger.debug("EGL probe: EGL_DEFAULT_DISPLAY succeeded.")
            return True

        logger.info("EGL probe: no EGL display could be initialised — will use OSMesa.")
        return False

    except Exception as exc:  # noqa: BLE001
        logger.debug("EGL probe: unexpected error (%s) — will use OSMesa.", exc)
        return False


def _osmesa_is_available():
    """Return True if libOSMesa can be loaded via ctypes."""
    for name in ("libOSMesa.so.8", "libOSMesa.so", "OSMesa"):
        try:
            ctypes.CDLL(name)
            return True
        except OSError:
            continue
    return False


if sys.platform == "linux" and "PYOPENGL_PLATFORM" not in os.environ:
    _has_display = (
        bool(os.environ.get("DISPLAY")) or bool(os.environ.get("WAYLAND_DISPLAY"))
    )
    if not _has_display:
        # No display — choose headless backend before OpenGL.GL is imported.
        # Use _egl_context_works() not just _egl_is_available(): libEGL may be
        # installed but still fail (no GPU, no llvmpipe, bad driver).  We must
        # know the outcome before setting PYOPENGL_PLATFORM because OpenGL.GL
        # binds its function pointers on first import and cannot be re-bound.
        if _egl_context_works():
            os.environ["PYOPENGL_PLATFORM"] = "egl"
            logger.info("No display detected; EGL available — using EGL headless rendering.")
        elif _osmesa_is_available():
            os.environ["PYOPENGL_PLATFORM"] = "osmesa"
            logger.info("No display detected; EGL unavailable — using OSMesa CPU rendering.")
        else:
            raise RuntimeError(
                "whippersnappy requires an OpenGL context but none could be found.\n"
                "\n"
                "No display server detected (DISPLAY/WAYLAND_DISPLAY unset),\n"
                "EGL initialisation failed, and OSMesa is not installed.\n"
                "\n"
                "To fix this, choose one of:\n"
                "  1. Install EGL (recommended, if GPU is installed):\n"
                "       Debian/Ubuntu:  sudo apt-get install libegl1\n"
                "       RHEL/Fedora:    sudo dnf install mesa-libEGL\n"
                "  2. Install OSMesa (CPU-only alternative):\n"
                "       Debian/Ubuntu:  sudo apt-get install libosmesa6\n"
                "       RHEL/Fedora:    sudo dnf install mesa-libOSMesa\n"
                "  3. Set DISPLAY if a local X server is running:\n"
                "       export DISPLAY=:0\n"
            )
