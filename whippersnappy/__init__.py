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

from ._config import sys_info  # noqa: F401
from ._version import __version__  # noqa: F401
from .snap import snap1, snap4
from .utils.types import ColorSelection, OrientationType, ViewType

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
