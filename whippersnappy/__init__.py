"""WhipperSnapPy: Plot and capture FastSurfer and FreeSurfer-style surface overlays.

WhipperSnapPy provides tools for rendering brain surface meshes with statistical
overlays and annotations. It includes:

- **Static rendering**: `snap1()` and `snap4()` functions for publication-quality images
- **3D plotting**: For Jupyter notebooks with mouse-controlled 3D (via Three.js)
- **GUI**: Interactive desktop viewer via the ``whippersnap`` command
- **CLI tools**: ``whippersnap1`` and ``whippersnap4`` for batch processing
- **Local mesh IO**: OFF, VTK ASCII PolyData, and PLY in addition to FreeSurfer surfaces

For static image generation::

    from whippersnappy import snap1, snap4
    from whippersnappy.utils.types import ViewType
    from IPython.display import display

    img = snap1('path/to/lh.white', view=ViewType.LEFT)
    display(img)

For interactive 3D in Jupyter notebooks::

    # Requires: pip install 'whippersnappy[notebook]'
    from whippersnappy import plot3d

    viewer = plot3d(
        mesh='path/to/lh.white',
        bg_map='path/to/lh.curv',
        overlay='path/to/lh.thickness',  # optional: for colors
    )
    display(viewer)

For the interactive desktop GUI::

    # Requires: pip install 'whippersnappy[gui]'
    # General mode — any mesh file:
    whippersnap --mesh lh.white --overlay lh.thickness --bg-map lh.curv
    # FreeSurfer shortcut — derive paths from subject directory:
    whippersnap -sd path/to/subject_dir --hemi lh -lh lh.thickness

"""

import os
import sys


def _check_display():
    """Return True if a display is available or the platform handles GL natively.

    On macOS (CGL) and Windows (WGL) PyOpenGL does not use EGL, so we always
    return True to avoid setting ``PYOPENGL_PLATFORM=egl`` on those systems.
    On Linux we probe for an X11/Wayland display server.
    """
    if sys.platform != "linux":
        # macOS uses CGL, Windows uses WGL — no EGL needed on either.
        return True
    display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    if not display:
        return False
    try:
        import ctypes
        import ctypes.util
        libx11 = ctypes.CDLL(ctypes.util.find_library("X11") or "libX11.so.6")
        libx11.XOpenDisplay.restype = ctypes.c_void_p
        libx11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        libx11.XCloseDisplay.restype = None
        libx11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        dpy = libx11.XOpenDisplay(display.encode())
        if dpy:
            libx11.XCloseDisplay(dpy)
            return True
        return False
    except Exception:
        return False

if "PYOPENGL_PLATFORM" not in os.environ:
    if not _check_display():
        os.environ["PYOPENGL_PLATFORM"] = "egl"

from ._config import sys_info  # noqa: F401, E402
from ._version import __version__  # noqa: F401, E402
from .snap import snap1, snap4, snap_rotate  # noqa: E402
from .utils.datasets import fetch_sample_subject  # noqa: E402
from .utils.types import ViewType  # noqa: E402

# 3D plotting for notebooks (Three.js-based, works in all Jupyter environments)
try:
    from .plot3d import plot3d  # noqa: E402
    _has_plot3d = True
except ImportError:
    _has_plot3d = False

# Export list
__all__ = [
    "__version__",
    "sys_info",
    "snap1",
    "snap4",
    "snap_rotate",
    "fetch_sample_subject",
    "ViewType",
]

if _has_plot3d:
    __all__.append("plot3d")
