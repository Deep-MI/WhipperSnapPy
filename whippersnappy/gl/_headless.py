"""Headless OpenGL platform detection.

This module MUST be imported before any ``import OpenGL.GL`` statement in the
package.  On Linux with no display server it sets ``PYOPENGL_PLATFORM`` so
that PyOpenGL resolves function pointers via the right backend.

Priority chain when no display is detected (Linux only):

1. **EGL + GPU device** — ``/dev/dri/renderD*`` readable and ``libEGL``
   loadable.  ``PYOPENGL_PLATFORM`` is left **unset** here so that
   :mod:`whippersnappy.gl.egl_context` can set it to ``"egl"`` before
   ``OpenGL.GL`` is imported.
2. **OSMesa** — CPU software renderer.  Sets ``PYOPENGL_PLATFORM=osmesa``.
3. **Neither** — raises ``RuntimeError`` with install instructions.

When GLFW fails even though ``DISPLAY`` is set (e.g. ``ssh -Y`` with a
forwarded X that lacks GLX 3.3), ``PYOPENGL_PLATFORM`` is already bound to
whatever was set at import time.  :func:`init_offscreen_context` handles
this by trying EGL as a second step after GLFW failure, before OSMesa.

No OpenGL, GLFW, or other heavy imports are done here — only stdlib.
"""

import ctypes
import glob
import logging
import os
import sys

logger = logging.getLogger(__name__)


def _osmesa_is_available():
    """Return True if libOSMesa can be loaded via ctypes."""
    for name in ("libOSMesa.so.8", "libOSMesa.so", "OSMesa"):
        try:
            ctypes.CDLL(name)
            return True
        except OSError:
            continue
    return False


def egl_device_is_available():
    """Return True if libEGL is loadable AND a DRI render node is accessible.

    Checking for ``/dev/dri/renderD*`` existence and readability guards
    against Singularity/Docker containers that have EGL libraries installed
    but no device nodes bound in — in those cases EGL context creation would
    fail and we should fall back to OSMesa instead.

    This function is called both here (at import time) and from
    :func:`~whippersnappy.gl.context.init_offscreen_context` (at context
    creation time, to decide whether to attempt EGL after GLFW fails).
    """
    render_nodes = glob.glob("/dev/dri/renderD*")
    if not render_nodes:
        logger.debug("EGL: no /dev/dri/renderD* device nodes found — skipping EGL.")
        return False
    if not any(os.access(n, os.R_OK) for n in render_nodes):
        logger.debug("EGL: /dev/dri/renderD* exists but not readable — skipping EGL.")
        return False
    for name in ("libEGL.so.1", "libEGL.so"):
        try:
            ctypes.CDLL(name)
            logger.debug("EGL: libEGL found and render node accessible.")
            return True
        except OSError:
            continue
    logger.debug("EGL: libEGL not found.")
    return False


if (
    sys.platform == "linux"
    and "PYOPENGL_PLATFORM" not in os.environ
    and not os.environ.get("DISPLAY")
    and not os.environ.get("WAYLAND_DISPLAY")
):
    if egl_device_is_available():
        # Defer: egl_context.py will set PYOPENGL_PLATFORM=egl before importing
        # OpenGL.GL.  Do NOT set osmesa here or GL will bind to the wrong backend.
        logger.debug(
            "No display, EGL + GPU device available — "
            "deferring platform selection to EGL context creation."
        )
    elif _osmesa_is_available():
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        logger.debug(
            "No display, no EGL device — PYOPENGL_PLATFORM=osmesa set (CPU rendering)."
        )
    else:
        raise RuntimeError(
            "whippersnappy requires an OpenGL context but none could be found.\n"
            "\n"
            "No display server detected (DISPLAY / WAYLAND_DISPLAY are unset),\n"
            "no accessible GPU render device (/dev/dri/renderD*), and OSMesa\n"
            "is not installed.\n"
            "\n"
            "To fix this, choose one of:\n"
            "  1. Install OSMesa (recommended for headless/SSH use):\n"
            "       Debian/Ubuntu:  sudo apt-get install libosmesa6\n"
            "       RHEL/Fedora:    sudo dnf install mesa-libOSMesa\n"
            "  2. Use EGL GPU rendering by ensuring /dev/dri/renderD* is accessible\n"
            "     and libEGL is installed (libegl1 on Debian/Ubuntu).\n"
            "  3. Set DISPLAY if a local X server is running:\n"
            "       export DISPLAY=:1\n"
        )
elif sys.platform == "linux":
    _display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    logger.debug(
        "Display set (%s) — will try GLFW; EGL on failure if GPU device available.",
        _display,
    )
