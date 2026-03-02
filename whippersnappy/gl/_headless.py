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

import os
import sys

if (
    sys.platform == "linux"
    and "PYOPENGL_PLATFORM" not in os.environ
    and not os.environ.get("DISPLAY")
    and not os.environ.get("WAYLAND_DISPLAY")
):
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

