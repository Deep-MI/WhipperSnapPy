"""Headless OpenGL platform detection.

This module MUST be imported before any ``import OpenGL.GL`` statement in the
package.  It sets ``PYOPENGL_PLATFORM=osmesa`` when running on a Linux machine
with no display server so that PyOpenGL resolves all function pointers via
``OSMesaGetProcAddress`` rather than GLX.

Importing ``OpenGL.GL`` before this runs (e.g. by importing
:mod:`whippersnappy.gl.shaders` or :mod:`whippersnappy.gl.pipeline` directly)
would allow PyOpenGL to bind via GLX on a headless machine, causing the later
OSMesa context to fail.  Placing the guard here and importing this module first
in :mod:`whippersnappy.gl.__init__` ensures the variable is always set in time,
regardless of which submodule a caller imports first.

No OpenGL, GLFW, or other heavy imports are done here — only stdlib.

SSH sessions
------------
If you connect via SSH to a machine with a running X server and want GPU
rendering rather than OSMesa, set ``DISPLAY`` before running::

    export DISPLAY=:1   # or whichever display the X server is on
    python my_script.py

Without an explicit ``DISPLAY``, whippersnappy falls back to OSMesa (CPU
software rendering) automatically so that scripts work without any setup on
headless servers.  OSMesa requires ``libosmesa6`` (Debian/Ubuntu) or
``mesa-libOSMesa`` (RHEL/Fedora) to be installed as a system package.
"""

import ctypes
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


if (
    sys.platform == "linux"
    and "PYOPENGL_PLATFORM" not in os.environ
    and not os.environ.get("DISPLAY")
    and not os.environ.get("WAYLAND_DISPLAY")
):
    if _osmesa_is_available():
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        logger.debug("No display detected — using OSMesa software rendering (CPU).")
    else:
        raise RuntimeError(
            "whippersnappy requires an OpenGL context but none could be found.\n"
            "\n"
            "No display server detected (DISPLAY / WAYLAND_DISPLAY are unset) "
            "and the OSMesa software renderer is not installed.\n"
            "\n"
            "To fix this, choose one of:\n"
            "  1. Install OSMesa (recommended for headless/SSH use):\n"
            "       Debian/Ubuntu:  sudo apt-get install libosmesa6\n"
            "       RHEL/Fedora:    sudo dnf install mesa-libOSMesa\n"
            "  2. Set DISPLAY if a local X server is running:\n"
            "       export DISPLAY=:1\n"
        )
else:
    _display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or "n/a"
    logger.debug("Display detected (%s) — using GLFW/GPU rendering.", _display)

