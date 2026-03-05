"""OpenGL context management and rendering helpers (gl package).

This package contains the low-level OpenGL helpers used by the renderers:

- :mod:`~whippersnappy.gl._headless` — headless platform detection; sets
  ``PYOPENGL_PLATFORM`` to ``"egl"`` or ``"osmesa"`` on headless Linux
  *before* any ``OpenGL.GL`` import.  Imported first here and in every GL
  submodule so the guard takes effect even when a submodule is imported
  directly.  PyOpenGL binds its platform backend on the first
  ``import OpenGL.GL`` and cannot be changed afterwards, so this ordering
  is critical.
- :mod:`~whippersnappy.gl.context` — context lifecycle (create, capture, destroy).
  On Linux tries GLFW → EGL (only when ``PYOPENGL_PLATFORM=egl``) → OSMesa.
- :mod:`~whippersnappy.gl.egl_context` — EGL headless GPU context (Linux).
- :mod:`~whippersnappy.gl.osmesa_context` — OSMesa headless CPU context (Linux).
- :mod:`~whippersnappy.gl.pipeline` — one-time GPU upload: VAO, buffers, uniforms, shader setup.
- :mod:`~whippersnappy.gl.shaders` — GLSL shader source and compilation.
- :mod:`~whippersnappy.gl.camera` — projection, view, and model matrices.

Public API re-exported at package level::

    from whippersnappy.gl import init_window, init_offscreen_context, setup_shader

"""

from . import _headless  # noqa: F401 — must be first; sets PYOPENGL_PLATFORM before any OpenGL.GL import
from .camera import make_model, make_projection, make_view
from .context import (
    capture_window,
    init_offscreen_context,
    init_window,
    render_scene,
    terminate_context,
)
from .pipeline import setup_shader
from .shaders import compile_shader_program, get_default_shaders, get_webgl_shaders

__all__ = [
    # context lifecycle
    'init_window', 'init_offscreen_context', 'terminate_context',
    # rendering
    'render_scene', 'capture_window',
    # pipeline setup
    'setup_shader',
    # shaders
    'compile_shader_program', 'get_default_shaders', 'get_webgl_shaders',
    # camera
    'make_model', 'make_projection', 'make_view',
]
