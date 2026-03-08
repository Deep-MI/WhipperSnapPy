"""Headless OpenGL platform detection.

This module MUST be imported before any ``import OpenGL.GL`` statement in the
package.  On Linux with no display it sets ``PYOPENGL_PLATFORM`` so that
PyOpenGL resolves function pointers via the correct backend before
``OpenGL.GL`` is first imported.

Priority chain on Linux when no display is detected
(``DISPLAY`` / ``WAYLAND_DISPLAY`` unset):

1. **EGL** — tried first.  A lightweight ctypes probe confirms EGL can
   actually initialise a display before ``PYOPENGL_PLATFORM=egl`` is set.
   When a GPU is accessible (native install) EGL uses it; otherwise EGL falls
   back to Mesa's llvmpipe CPU software renderer.  Works in Docker and
   Singularity without any special flags.
2. **OSMesa** — fallback when EGL cannot initialise at all (e.g. ``libegl1``
   not installed).  Sets ``PYOPENGL_PLATFORM=osmesa``.
3. **Neither** — raises ``RuntimeError`` with install instructions.

When ``DISPLAY`` is set the module does not intervene; GLFW is tried first
in :func:`~whippersnappy.gl.context.init_offscreen_context`.

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

    1. ``EGL_EXT_device_enumeration`` — enumerate GPU/software devices.
    2. ``EGL_MESA_platform_surfaceless`` — Mesa CPU software rendering
       (llvmpipe); no GPU or display server needed.
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

    from contextlib import contextmanager

    @contextmanager
    def _suppress_stderr():
        """Suppress C-level stderr (e.g. Mesa/EGL driver warnings)."""
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved = os.dup(2)
        try:
            os.dup2(devnull, 2)
            yield
        finally:
            os.dup2(saved, 2)
            os.close(saved)
            os.close(devnull)

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

        _EGL_EXTENSIONS           = 0x3055
        _EGL_NONE                 = 0x3038
        _EGL_PLATFORM_DEVICE      = 0x313F
        _EGL_PLATFORM_SURFACELESS = 0x31DD

        client_exts = libegl.eglQueryString(None, _EGL_EXTENSIONS) or b""
        logger.debug("EGL client extensions: %s", client_exts.decode())

        no_attribs = (ctypes.c_int * 1)(_EGL_NONE)

        def _get_proc(signature, name):
            """Resolve an EGL extension function; return None if unavailable."""
            addr = libegl.eglGetProcAddress(name)
            return signature(addr) if addr else None

        def _try_init(dpy):
            """Try eglInitialize on dpy with stderr suppressed; terminate on success."""
            if not dpy:
                return False
            major, minor = ctypes.c_int(0), ctypes.c_int(0)
            with _suppress_stderr():
                ok = libegl.eglInitialize(dpy, ctypes.byref(major), ctypes.byref(minor))
                libegl.eglTerminate(dpy)
            if ok:
                logger.debug("EGL probe: eglInitialize OK (EGL %d.%d).",
                             major.value, minor.value)
            return bool(ok)

        # Resolve eglGetPlatformDisplayEXT once; used by paths 1 and 2.
        _GetPlatformDisplayEXT = None
        if b"EGL_EXT_platform_base" in client_exts:
            _GetPlatformDisplayEXT = _get_proc(
                ctypes.CFUNCTYPE(ctypes.c_void_p,
                                 ctypes.c_int, ctypes.c_void_p,
                                 ctypes.POINTER(ctypes.c_int)),
                b"eglGetPlatformDisplayEXT",
            )

        # --- Path 1: EGL_EXT_device_enumeration ---
        # Try GPU and software devices; prefer GPU when available natively.
        if _GetPlatformDisplayEXT and b"EGL_EXT_device_enumeration" in client_exts:
            _QueryDevices = _get_proc(
                ctypes.CFUNCTYPE(ctypes.c_bool,
                                 ctypes.c_int, ctypes.c_void_p,
                                 ctypes.POINTER(ctypes.c_int)),
                b"eglQueryDevicesEXT",
            )
            if _QueryDevices:
                n = ctypes.c_int(0)
                with _suppress_stderr():
                    found = _QueryDevices(0, None, ctypes.byref(n))
                logger.debug("EGL probe: %d EGL device(s) found.", n.value if found else 0)
                if found and n.value > 0:
                    devices = (ctypes.c_void_p * n.value)()
                    with _suppress_stderr():
                        _QueryDevices(n.value, devices, ctypes.byref(n))
                    for dev in devices:
                        dpy = _GetPlatformDisplayEXT(
                            _EGL_PLATFORM_DEVICE,
                            ctypes.c_void_p(dev),
                            no_attribs,
                        )
                        if _try_init(dpy):
                            logger.debug("EGL probe: device enumeration succeeded.")
                            return True

        # --- Path 2: EGL_MESA_platform_surfaceless ---
        # CPU software rendering (llvmpipe) — no GPU needed.
        if _GetPlatformDisplayEXT and b"EGL_MESA_platform_surfaceless" in client_exts:
            dpy = _GetPlatformDisplayEXT(
                _EGL_PLATFORM_SURFACELESS, ctypes.c_void_p(0), no_attribs
            )
            if _try_init(dpy):
                logger.debug("EGL probe: surfaceless platform succeeded (CPU/llvmpipe).")
                return True

        # --- Path 3: EGL_DEFAULT_DISPLAY ---
        # Works only when a display server is reachable (DISPLAY set).
        if _try_init(libegl.eglGetDisplay(ctypes.c_void_p(0))):
            logger.debug("EGL probe: EGL_DEFAULT_DISPLAY succeeded.")
            return True

        logger.info("EGL probe: no EGL display could be initialised.")
        return False

    except Exception as exc:  # noqa: BLE001
        logger.debug("EGL probe: unexpected error (%s).", exc)
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
        # OpenGL.GL binds its function pointers on first import and cannot be
        # re-bound, so PYOPENGL_PLATFORM must be set correctly here.
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
                "  1. Install EGL (recommended, for GPU or CPU rendering):\n"
                "       Debian/Ubuntu:  sudo apt-get install libegl1\n"
                "       RHEL/Fedora:    sudo dnf install mesa-libEGL\n"
                "  2. Install OSMesa (CPU-only alternative):\n"
                "       Debian/Ubuntu:  sudo apt-get install libosmesa6\n"
                "       RHEL/Fedora:    sudo dnf install mesa-libOSMesa\n"
                "  3. Set DISPLAY if a local X server is running:\n"
                "       export DISPLAY=:0\n"
            )
