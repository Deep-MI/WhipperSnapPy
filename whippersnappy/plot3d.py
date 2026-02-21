"""3D plotting for WhipperSnapPy using pythreejs (Three.js) for Jupyter.

This module provides interactive 3D brain visualization for Jupyter notebooks
using Three.js/WebGL.  It works in all Jupyter environments (browser,
JupyterLab, Colab, VS Code).

Unlike the desktop GUI (``whippersnap`` command), this renders entirely in the
browser via WebGL and is designed for notebook environments.

Usage::

    from whippersnappy import plot3d
    viewer = plot3d(mesh='path/to/lh.white', bg_map='path/to/lh.curv')
    display(viewer)

Dependencies:
    pythreejs, ipywidgets, numpy
"""

import logging

import numpy as np
import pythreejs as p3js
from ipywidgets import HTML, VBox

from .geometry import prepare_geometry
from .gl import get_webgl_shaders
from .utils.types import ColorSelection

# Module logger
logger = logging.getLogger(__name__)


def plot3d(
    mesh,
    overlay=None,
    annot=None,
    bg_map=None,
    roi=None,
    minval=None,
    maxval=None,
    invert=False,
    scale=1.85,
    color_mode=None,
    width=800,
    height=800,
    ambient=0.1,
):
    """Create an interactive 3D notebook viewer using pythreejs (Three.js).

    This function prepares geometry and color information (via
    :func:`whippersnappy.geometry.prepare_geometry`) and constructs a
    pythreejs renderer and controls wrapped in an ``ipywidgets.VBox`` for
    display inside a Jupyter notebook.

    The mesh can be any triangular surface — not just brain surfaces.
    Supported file formats: FreeSurfer binary surface (e.g. ``lh.white``),
    ASCII OFF (``.off``), legacy ASCII VTK PolyData (``.vtk``), ASCII PLY
    (``.ply``), or a ``(vertices, faces)`` numpy array tuple.

    Parameters
    ----------
    mesh : str or tuple of (array-like, array-like)
        Path to a mesh file (FreeSurfer binary, ``.off``, ``.vtk``, or
        ``.ply``) **or** a ``(vertices, faces)`` tuple.
    overlay : str, array-like, or None, optional
        Path to a per-vertex scalar file, or a (N,) array of per-vertex
        scalar values.
    annot : str, tuple, or None, optional
        Path to a FreeSurfer .annot file, or a ``(labels, ctab)`` /
        ``(labels, ctab, names)`` tuple for categorical labeling.
    bg_map : str, array-like, or None, optional
        Path to a per-vertex scalar file **or** a (N,) array used as
        grayscale background shading for non-overlay regions.
    roi : str, array-like, or None, optional
        Path to a FreeSurfer label file **or** a (N,) boolean array to
        restrict overlay coloring to a subset of vertices.
    minval, maxval : float or None, optional
        Threshold and saturation values used for color mapping.
        If ``None``, sensible defaults are chosen automatically.
    invert : bool, optional
        If True, invert the overlay color map. Default is ``False``.
    scale : float, optional
        Global geometry scale applied during preparation. Default is ``1.85``.
    color_mode : ColorSelection or None, optional
        Which sign of overlay values to color (BOTH/POSITIVE/NEGATIVE).
        If None, defaults to ``ColorSelection.BOTH``.
    width, height : int, optional
        Canvas dimensions for the generated renderer. Default is ``800``.
    ambient : float, optional
        Ambient lighting strength passed to the Three.js shader. Default is ``0.1``.

    Returns
    -------
    ipywidgets.VBox
        A widget containing the pythreejs Renderer and a small info panel.

    Raises
    ------
    ValueError, FileNotFoundError
        Errors from :func:`prepare_geometry` are propagated (for example
        shape mismatches between overlay and mesh vertex count).

    Examples
    --------
    In a Jupyter notebook::

        from whippersnappy import plot3d
        from IPython.display import display

        # FreeSurfer surface
        viewer = plot3d('lh.white', overlay='lh.thickness', bg_map='lh.curv')
        display(viewer)

        # Any triangular mesh via OFF / VTK / PLY
        viewer = plot3d('mesh.off', overlay='values.mgh')
        display(viewer)

        # Array inputs
        import numpy as np
        v = np.random.randn(500, 3).astype(np.float32)
        f = np.zeros((1, 3), dtype=np.uint32)
        viewer = plot3d((v, f))
        display(viewer)
    """
    # Load and prepare mesh data
    color_mode = color_mode or ColorSelection.BOTH
    meshdata, triangles, fmin, fmax, pos, neg = prepare_geometry(
        mesh, overlay, annot, bg_map, roi,
        minval, maxval, invert, scale, color_mode
    )

    logger.info("Loaded mesh: %d vertices, %d faces", meshdata.shape[0], triangles.shape[0])

    # Extract vertices, normals, and colors
    vertices = meshdata[:, :3]  # x, y, z
    normals = meshdata[:, 3:6]  # nx, ny, nz
    colors = meshdata[:, 6:9]   # r, g, b

    # Center and scale the mesh
    center = vertices.mean(axis=0)
    vertices = vertices - center
    max_extent = np.abs(vertices).max()
    vertices = vertices / max_extent * 2.0

    # Create Three.js mesh
    mesh = create_threejs_mesh_with_custom_shaders(vertices, triangles, colors, normals, ambient=ambient)

    camera = p3js.PerspectiveCamera(
        position=[-5, 0, 0],
        up=[0, 0, 1],
        aspect=width/height,
        fov=45,
        near=0.1,
        far=1000
    )

    # Create scene without lights (use our own custom shader):
    scene = p3js.Scene(
        children=[mesh, camera],  # No lights needed
        background='#000000'
    )

    # Create renderer
    renderer = p3js.Renderer(
        camera=camera,
        scene=scene,
        controls=[p3js.OrbitControls(controlling=camera)],
        width=width,
        height=height,
        antialias=True
    )

    # Create info display
    info_text = f"""
        <div style='font-family: monospace; font-size: 12px; color: #666;'>
        <b>Interactive 3D Viewer (Three.js) ✓</b><br>
        Vertices: {len(vertices):,}<br>
        Triangles: {len(triangles):,}<br>
        <br>
        <i>Drag to rotate, scroll to zoom, right-drag to pan</i><br>
        """

    if overlay or annot:
        info_text += "<br>📊 Overlay/annotation loaded"
    elif bg_map:
        info_text += "<br>🧠 Curvature (grayscale is correct)"

    info_text += "</div>"

    info = HTML(value=info_text)

    # Combine renderer and info
    viewer = VBox([renderer, info])

    return viewer

def create_threejs_mesh_with_custom_shaders(vertices, faces, colors, normals, ambient=0.1):
    """Create a pythreejs.Mesh using custom shader material and buffers.

    The function builds a BufferGeometry with position, color and normal
    attributes, attaches an index buffer, and creates a ShaderMaterial
    using the WebGL shader snippets returned by :func:`get_webgl_shaders`.

    Parameters
    ----------
    vertices : numpy.ndarray
        Array of shape (N, 3) with vertex positions (float32).
    faces : numpy.ndarray
        Integer face index array shape (M, 3) or flattened (3*M,) dtype uint32.
    colors : numpy.ndarray
        Array of shape (N, 3) with per-vertex RGB colors (float32).
    normals : numpy.ndarray
        Array of shape (N, 3) with per-vertex normals (float32).
    ambient : float, optional, default 0.1
        Ambient lighting strength for the shader (passed to Three.js uniform).

    Returns
    -------
    pythreejs.Mesh
        Mesh object ready to be inserted into a pythreejs.Scene.
    """
    vertices = vertices.astype(np.float32)
    colors = colors.astype(np.float32)
    normals = normals.astype(np.float32)
    faces = faces.astype(np.uint32).flatten()

    vertex_shader, fragment_shader = get_webgl_shaders()

    geometry = p3js.BufferGeometry(
        attributes={
            'position': p3js.BufferAttribute(array=vertices),
            'color': p3js.BufferAttribute(array=colors),
            'normal': p3js.BufferAttribute(array=normals),
        }
    )
    geometry.index = p3js.BufferAttribute(array=faces)

    material = p3js.ShaderMaterial(
        vertexShader=vertex_shader,
        fragmentShader=fragment_shader,
        uniforms={
            'lightColor': {'value': [1.0, 1.0, 1.0]},
            'ambientStrength': {'value': ambient}
        }
    )

    mesh = p3js.Mesh(geometry=geometry, material=material)
    return mesh
