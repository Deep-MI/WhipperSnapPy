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
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)


def _find_x_display():
    """Return the first running X display found via /tmp/.X<n>-lock, or None."""
    import glob
    locks = sorted(glob.glob("/tmp/.X*-lock"))
    for lock in locks:
        # Lock filename is /tmp/.X<n>-lock  →  display is :<n>
        name = os.path.basename(lock)  # .X1-lock
        try:
            n = name[2:name.index("-lock")]
            if n.isdigit():
                return f":{n}"
        except ValueError:
            continue
    return None


if (
    sys.platform == "linux"
    and "PYOPENGL_PLATFORM" not in os.environ
    and not os.environ.get("DISPLAY")
    and not os.environ.get("WAYLAND_DISPLAY")
):
    # Common in SSH sessions: the workstation has a running X server but the
    # session didn't inherit DISPLAY.  Try to auto-detect it so that GLFW
    # (GPU rendering) works without the user needing to export DISPLAY manually.
    detected = _find_x_display()
    if detected:
        os.environ["DISPLAY"] = detected
        logger.debug(
            "No DISPLAY set; auto-detected X display %s from lock file. "
            "Set DISPLAY explicitly to override.",
            detected,
        )
    else:
        # Truly headless — fall back to OSMesa CPU rendering.
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"

