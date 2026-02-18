"""WhipperSnapPy: Plot and capture FastSurfer and FreeSurfer-style surface overlays.

WhipperSnapPy provides tools for rendering brain surface meshes with statistical
overlays and annotations. It includes:

- **Static rendering**: `snap1()` and `snap4()` functions for publication-quality images
- **3D plotting**: For Jupyter notebooks with mouse-controlled 3D (via Three.js)
- **GUI**: Desktop application with `--interactive` flag
- **CLI tools**: Command-line interface for batch processing
- **Custom shaders**: Full control over OpenGL lighting and rendering

For static image generation:

    from whippersnappy import snap1, snap4
    from whippersnappy.utils.types import ViewType
    from IPython.display import display

    img = snap1(meshpath='path/to/surface.white', view=ViewType.LEFT)
    display(img)

For interactive 3D in Jupyter notebooks:

    # Requires: pip install 'whippersnappy[notebook]'
    from whippersnappy import plot3d

    viewer = plot3d(
        meshpath='path/to/surface.white',
        curvpath='path/to/curv',
        overlaypath='path/to/thickness.mgh'  # optional: for colors
    )
    display(viewer)

For desktop GUI:

    # Command line
    whippersnap --interactive -lh path/to/lh.white -rh path/to/rh.white

Features:
- Works in ALL Jupyter environments (browser, JupyterLab, Colab, VS Code)
- Mouse-controlled rotation, zoom, and pan
- Professional lighting via Three.js/WebGL
- Same technology Plotly uses for 3D plots

"""

import os

def _check_display():
    """Return True if a working X11 display connection can be opened."""
    # macOS uses CGL/Cocoa — GLFW handles context creation natively, no EGL needed
    if sys.platform == "darwin":
        return True
    display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    if not display:
        return False
    try:
        import ctypes, ctypes.util
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

from ._config import sys_info  # noqa: F401
from ._version import __version__  # noqa: F401
from .snap import snap1, snap4
from .utils.types import ViewType

# 3D plotting for notebooks (Three.js-based, works in all Jupyter environments)
try:
    from .plot3d import plot3d
    _has_plot3d = True
except ImportError:
    _has_plot3d = False

# Export list
__all__ = [
    "__version__",
    "sys_info",
    "snap1",
    "snap4",
]

if _has_plot3d:
    __all__.append("plot3d")
# Top-level convenience export for frequently used enum
__all__.append("ViewType")
