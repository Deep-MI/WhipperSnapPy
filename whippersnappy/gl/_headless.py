"""Headless OpenGL platform detection.

This module MUST be imported before any ``import OpenGL.GL`` statement in the
package.  On Linux with no display server it sets ``PYOPENGL_PLATFORM`` so
that PyOpenGL resolves function pointers via the correct backend before
``OpenGL.GL`` is first imported.

Priority chain on Linux when no usable display is detected:

1. **EGL + GPU device** — ``/dev/dri/renderD*`` present and ``libEGL``
   loadable.  Sets ``PYOPENGL_PLATFORM=egl`` immediately so that PyOpenGL
   binds function pointers via EGL when ``OpenGL.GL`` is first imported.
2. **OSMesa** — CPU software renderer.  Sets ``PYOPENGL_PLATFORM=osmesa``.
3. **Neither** — raises ``RuntimeError`` with install instructions.

"No usable display" covers both:

* ``DISPLAY`` / ``WAYLAND_DISPLAY`` are unset entirely.
* ``DISPLAY`` is set but the X server is unreachable, refuses the
  connection, or does not advertise the ``GLX_ARB_create_context_profile``
  extension that GLFW requires to create an OpenGL 3.3 Core Profile context
  (e.g. ``ssh -X``/``ssh -Y`` to a server with only old software-rendered
  GLX).  In these cases GLFW fails with
  ``GLX_ARB_create_context_profile is unavailable``, so we pre-empt it by
  trying EGL/OSMesa instead.

When ``DISPLAY`` is set **and** the X server is reachable, this module
does not intervene: GLFW is tried first in
:func:`~whippersnappy.gl.context.init_offscreen_context`.

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


def _display_is_usable():
    """Return True if ``DISPLAY`` points to an X server with usable GLX support.

    A display is considered usable only when **all** of the following hold:

    1. ``DISPLAY`` is set and non-empty.
    2. ``libX11`` is loadable and ``XOpenDisplay`` succeeds (server reachable).
    3. ``libGL`` (or ``libGLX``) exposes ``glXQueryExtensionsString`` and the
       returned string contains ``GLX_ARB_create_context_profile``.  This
       extension is what GLFW requires to create an OpenGL 3.3 Core Profile
       context.  A forwarded ``ssh -X``/``ssh -Y`` display typically provides
       old software-rendered GLX that lacks this extension, causing GLFW to
       fail with ``GLX_ARB_create_context_profile is unavailable``.

    Returns ``False`` in all other cases so that ``_headless.py`` falls through
    to EGL or OSMesa instead of letting GLFW attempt and print GLX warnings.

    Note: Wayland displays (``WAYLAND_DISPLAY``) are not probed here because
    GLFW handles the Wayland path natively and does not go through GLX.
    """
    display_str = os.environ.get("DISPLAY")
    if not display_str:
        return False

    # --- Step 1: open X connection ---
    for lib_name in ("libX11.so.6", "libX11.so"):
        try:
            libx11 = ctypes.CDLL(lib_name)
            break
        except OSError:
            continue
    else:
        return False  # libX11 not installed

    try:
        libx11.XOpenDisplay.restype = ctypes.c_void_p
        libx11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        dpy = libx11.XOpenDisplay(display_str.encode())
    except Exception:  # noqa: BLE001
        return False
    if not dpy:
        return False  # X server unreachable / access denied

    # --- Step 2: check for GLX_ARB_create_context_profile extension ---
    # This is the extension GLFW needs to request an OpenGL 3.3 Core Profile.
    # It is absent on old/software-rendered forwarded displays.
    glx_ok = False
    for lib_name in ("libGL.so.1", "libGL.so", "libGLX.so.0", "libGLX.so"):
        try:
            libgl = ctypes.CDLL(lib_name)
            break
        except OSError:
            continue
    else:
        libgl = None

    if libgl is not None:
        try:
            libgl.glXQueryExtensionsString.restype = ctypes.c_char_p
            libgl.glXQueryExtensionsString.argtypes = [
                ctypes.c_void_p,  # display
                ctypes.c_int,     # screen
            ]
            ext_bytes = libgl.glXQueryExtensionsString(dpy, 0)
            if ext_bytes:
                exts = ext_bytes.decode("ascii", errors="replace")
                if "GLX_ARB_create_context_profile" in exts:
                    glx_ok = True
                else:
                    logger.debug(
                        "GLX_ARB_create_context_profile absent on %s "
                        "(GLFW would fail); treating display as unusable.",
                        display_str,
                    )
        except Exception:  # noqa: BLE001
            pass

    try:
        libx11.XCloseDisplay(dpy)
    except Exception:  # noqa: BLE001
        pass

    return glx_ok


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


if sys.platform == "linux" and "PYOPENGL_PLATFORM" not in os.environ:
    _has_display = (
        bool(os.environ.get("DISPLAY")) or bool(os.environ.get("WAYLAND_DISPLAY"))
    )

    if _has_display and _display_is_usable():
        # Reachable X/Wayland server — let GLFW try first; don't intervene.
        _display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        logger.debug("Display set and reachable (%s) — will try GLFW first.", _display)
    else:
        # No display, or display is set but unreachable (e.g. bad ssh -X forward).
        # Must choose a headless backend NOW before OpenGL.GL is imported.
        if _has_display:
            logger.debug(
                "DISPLAY is set (%s) but X server is unreachable — "
                "skipping GLFW/GLX and trying headless backends.",
                os.environ.get("DISPLAY"),
            )
        if egl_device_is_available():
            os.environ["PYOPENGL_PLATFORM"] = "egl"
            logger.debug(
                "No usable display; EGL + GPU device available — "
                "PYOPENGL_PLATFORM=egl set."
            )
        elif _osmesa_is_available():
            os.environ["PYOPENGL_PLATFORM"] = "osmesa"
            logger.debug(
                "No usable display; no EGL device — "
                "PYOPENGL_PLATFORM=osmesa set (CPU rendering)."
            )
        else:
            raise RuntimeError(
                "whippersnappy requires an OpenGL context but none could be found.\n"
                "\n"
                "No usable display detected (DISPLAY/WAYLAND_DISPLAY unset or X server\n"
                "unreachable), no GPU render device found (/dev/dri/renderD* absent or\n"
                "libEGL missing), and OSMesa is not installed.\n"
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
                "  3. If you used ssh -X/-Y, try without X forwarding:\n"
                "       unset DISPLAY\n"
            )
