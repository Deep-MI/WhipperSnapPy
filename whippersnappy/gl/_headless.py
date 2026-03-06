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
    """Probe EGL via ctypes to confirm a context can actually be created.

    Calls ``eglGetDisplay(EGL_DEFAULT_DISPLAY)`` + ``eglInitialize`` only —
    no ``OpenGL.GL`` import, no ``PYOPENGL_PLATFORM`` change.  Returns
    ``True`` only when EGL is loadable **and** a display can be initialised.
    This means both the GPU path (real device) and the CPU path (llvmpipe)
    are confirmed before we commit to ``PYOPENGL_PLATFORM=egl``.

    If this returns ``False``, callers should fall back to OSMesa so that
    ``OpenGL.GL`` is imported with the correct backend on its first import —
    mixing EGL-bound function pointers with an OSMesa context causes silent
    failures.
    """
    for lib_name in ("libEGL.so.1", "libEGL.so"):
        try:
            libegl = ctypes.CDLL(lib_name)
            break
        except OSError:
            continue
    else:
        return False

    try:
        libegl.eglGetDisplay.restype  = ctypes.c_void_p
        libegl.eglGetDisplay.argtypes = [ctypes.c_void_p]
        libegl.eglInitialize.restype  = ctypes.c_bool
        libegl.eglInitialize.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        libegl.eglTerminate.restype  = ctypes.c_bool
        libegl.eglTerminate.argtypes = [ctypes.c_void_p]

        dpy = libegl.eglGetDisplay(ctypes.c_void_p(0))  # EGL_DEFAULT_DISPLAY
        if not dpy:
            logger.debug("EGL probe: eglGetDisplay returned NULL.")
            return False
        major, minor = ctypes.c_int(0), ctypes.c_int(0)
        ok = libegl.eglInitialize(dpy, ctypes.byref(major), ctypes.byref(minor))
        libegl.eglTerminate(dpy)
        if ok:
            logger.debug("EGL probe: eglInitialize succeeded (EGL %d.%d).",
                         major.value, minor.value)
            return True
        logger.debug("EGL probe: eglInitialize failed.")
        return False
    except Exception:  # noqa: BLE001
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
            logger.debug("No display; EGL probe succeeded — PYOPENGL_PLATFORM=egl set.")
        elif _osmesa_is_available():
            os.environ["PYOPENGL_PLATFORM"] = "osmesa"
            logger.debug("No display; EGL unavailable — PYOPENGL_PLATFORM=osmesa set.")
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
