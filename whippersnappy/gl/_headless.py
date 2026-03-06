"""Headless OpenGL platform detection.

This module MUST be imported before any ``import OpenGL.GL`` statement in the
package.  On Linux it sets ``PYOPENGL_PLATFORM`` so that PyOpenGL resolves
function pointers via the correct backend before ``OpenGL.GL`` is first
imported.

Priority chain on Linux (applied unconditionally — ``DISPLAY`` is irrelevant
for offscreen rendering):

1. **EGL + GPU device** — ``/dev/dri/renderD*`` present and ``libEGL``
   loadable.  Sets ``PYOPENGL_PLATFORM=egl``.  Works with or without a
   display server, including headless servers, Docker/Singularity, and
   ``ssh`` sessions (with or without ``-X``/``-Y``).
2. **OSMesa** — CPU software renderer.  Sets ``PYOPENGL_PLATFORM=osmesa``.
3. **Neither** — raises ``RuntimeError`` with install instructions.

``PYOPENGL_PLATFORM`` is not consulted by GLFW, so setting it here does not
affect the interactive GUI (``whippersnap``), which creates its own visible
GLFW window independently.

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
    """Return True if libEGL is loadable AND a DRI render node can be opened.

    Unlike a simple existence check, this function actually tries to
    ``open()`` each ``/dev/dri/renderD*`` node so that a permission error
    (e.g. inside Docker when ``--group-add render`` is missing) is caught
    here — before ``PYOPENGL_PLATFORM=egl`` is set and ``OpenGL.GL`` is
    bound to EGL function pointers.  If EGL were allowed to fail *after*
    PyOpenGL has already bound to EGL, any OSMesa fallback in the same
    process would use EGL function pointers for OSMesa calls, causing
    cryptic GL errors (e.g. ``Validation failure``).

    We still skip EGL if *no* device node exists at all — that is the
    reliable Singularity/Docker signal where no device is bound in.

    This function is called both here (at import time) and from
    :func:`~whippersnappy.gl.context.init_offscreen_context` (at context
    creation time, to decide whether to attempt EGL after GLFW fails).
    """
    render_nodes = glob.glob("/dev/dri/renderD*")
    if not render_nodes:
        logger.debug("EGL: no /dev/dri/renderD* device nodes found — skipping EGL.")
        return False

    # Try to open at least one node to verify actual access permission.
    # os.access() is unreliable for supplementary groups and POSIX ACLs,
    # but open() uses the real kernel permission check.
    accessible = []
    for node in render_nodes:
        try:
            fd = os.open(node, os.O_RDWR | os.O_NONBLOCK)
            os.close(fd)
            accessible.append(node)
        except OSError:
            continue
    if not accessible:
        logger.debug(
            "EGL: /dev/dri/renderD* node(s) exist but none could be opened "
            "(permission denied?) — skipping EGL to avoid broken fallback."
        )
        return False

    for name in ("libEGL.so.1", "libEGL.so"):
        try:
            ctypes.CDLL(name)
            logger.debug(
                "EGL: libEGL found and %d accessible render node(s) — EGL available.",
                len(accessible),
            )
            return True
        except OSError:
            continue
    logger.debug("EGL: render node accessible but libEGL not loadable.")
    return False


if sys.platform == "linux" and "PYOPENGL_PLATFORM" not in os.environ:
    if egl_device_is_available():
        # Prefer EGL for all offscreen rendering on Linux — regardless of
        # whether DISPLAY is set.  GLFW does not use PYOPENGL_PLATFORM, so
        # the interactive GUI is unaffected.  Setting this here, before any
        # import of OpenGL.GL, ensures PyOpenGL binds EGL function pointers.
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        logger.debug("EGL device available — PYOPENGL_PLATFORM=egl set.")
    elif _osmesa_is_available():
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        logger.debug("No EGL device — PYOPENGL_PLATFORM=osmesa set (CPU rendering).")
    else:
        _has_display = (
            bool(os.environ.get("DISPLAY")) or bool(os.environ.get("WAYLAND_DISPLAY"))
        )
        if not _has_display:
            # No display and no headless backend — raise immediately with
            # instructions.  When DISPLAY is set we stay silent and let GLFW
            # try; if it fails too, context.py will raise a clearer error.
            raise RuntimeError(
                "whippersnappy requires an OpenGL context but none could be found.\n"
                "\n"
                "No display server detected (DISPLAY/WAYLAND_DISPLAY unset),\n"
                "no accessible GPU render device (no /dev/dri/renderD* node could be\n"
                "opened — device absent or permission denied), and OSMesa is not\n"
                "installed.\n"
                "\n"
                "To fix this, choose one of:\n"
                "  1. Install OSMesa (recommended for headless/SSH use):\n"
                "       Debian/Ubuntu:  sudo apt-get install libosmesa6\n"
                "       RHEL/Fedora:    sudo dnf install mesa-libOSMesa\n"
                "  2. Use EGL GPU rendering — ensure /dev/dri/renderD* is accessible:\n"
                "       • Add your user to the render group:\n"
                "           sudo usermod -aG render $USER  (then log out and back in)\n"
                "       • Inside Docker: add --group-add render (or --group-add <GID>)\n"
                "         and --device /dev/dri/renderD128 to your docker run command.\n"
                "  3. Set DISPLAY if a local X server is running:\n"
                "       export DISPLAY=:0\n"
            )
