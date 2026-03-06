"""Headless OpenGL platform detection.

This module MUST be imported before any ``import OpenGL.GL`` statement in the
package.  On Linux with no display server it sets ``PYOPENGL_PLATFORM`` so
that PyOpenGL resolves function pointers via the correct backend before
``OpenGL.GL`` is first imported.

Priority chain when no display is detected (Linux only):

1. **EGL + GPU device** — ``/dev/dri/renderD*`` readable and ``libEGL``
   loadable.  Sets ``PYOPENGL_PLATFORM=egl`` immediately so that PyOpenGL
   binds function pointers via EGL when ``OpenGL.GL`` is first imported.
2. **OSMesa** — CPU software renderer.  Sets ``PYOPENGL_PLATFORM=osmesa``.
3. **Neither** — raises ``RuntimeError`` with install instructions.

When ``DISPLAY`` is set (e.g. normal desktop or ``ssh -Y``), ``_headless``
does not intervene: GLFW is tried first in :func:`init_offscreen_context`.
If GLFW fails (e.g. GLX 3.3 unavailable on the forwarded display), EGL is
attempted only when ``PYOPENGL_PLATFORM`` was already set to ``"egl"`` by
this module at import time — i.e. only for the no-display + EGL-device case.
In all other GLFW-failure scenarios, OSMesa is used as the final fallback.

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
    """Return True if libEGL is loadable AND a DRI render node is present.

    We check that at least one ``/dev/dri/renderD*`` node exists and that
    ``libEGL`` can be loaded.  We intentionally do **not** gate on
    ``os.access(node, os.R_OK)`` here because that POSIX check does not
    honour supplementary group memberships on all kernels (e.g. when the
    process inherits group ``render`` via newgrp or a login session) and
    does not account for POSIX ACL entries (the ``+`` suffix in ``ls -l``
    output).  If the node exists and EGL is installed we optimistically try
    EGL and let the context-creation call fail gracefully if the device truly
    turns out to be inaccessible.

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
    for name in ("libEGL.so.1", "libEGL.so"):
        try:
            ctypes.CDLL(name)
            logger.debug(
                "EGL: libEGL found and %d render node(s) present.", len(render_nodes)
            )
            return True
        except OSError:
            continue
    logger.debug("EGL: /dev/dri/renderD* found but libEGL not loadable.")
    return False


if (
    sys.platform == "linux"
    and "PYOPENGL_PLATFORM" not in os.environ
    and not os.environ.get("DISPLAY")
    and not os.environ.get("WAYLAND_DISPLAY")
):
    if egl_device_is_available():
        # Set PYOPENGL_PLATFORM=egl NOW, before OpenGL.GL is imported anywhere.
        # PyOpenGL selects its platform backend on first import and cannot be
        # changed afterwards; deferring to egl_context.py would mean OpenGL.GL
        # is already bound to the wrong backend by the time EGL is tried.
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        logger.debug(
            "No display, EGL + GPU device available — PYOPENGL_PLATFORM=egl set."
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
            "no GPU render device found (/dev/dri/renderD* absent or libEGL missing),\n"
            "and OSMesa is not installed.\n"
            "\n"
            "To fix this, choose one of:\n"
            "  1. Install OSMesa (recommended for headless/SSH use):\n"
            "       Debian/Ubuntu:  sudo apt-get install libosmesa6\n"
            "       RHEL/Fedora:    sudo dnf install mesa-libOSMesa\n"
            "  2. Use EGL GPU rendering — ensure /dev/dri/renderD* exists and\n"
            "     libEGL is installed (libegl1 on Debian/Ubuntu).  If the device\n"
            "     exists but you still see this error, add your user to the\n"
            "     'render' group:  sudo usermod -aG render $USER\n"
            "     (then log out and back in).\n"
            "  3. Set DISPLAY if a local X server is running:\n"
            "       export DISPLAY=:1\n"
        )
elif sys.platform == "linux":
    _display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    logger.debug(
        "Display set (%s) — will try GLFW first.",
        _display,
    )
