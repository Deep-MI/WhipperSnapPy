"""OpenGL helper utilities (gl package).

This package replaces the previous `gl_utils.py` module.
Functions are re-exported at package level for convenience, e.g.:

    from whippersnappy.gl import init_window, setup_shader

"""

from . import _platform   # noqa: F401 — MUST be first; sets PYOPENGL_PLATFORM

from .camera import make_model, make_projection, make_view
from .shaders import get_default_shaders, get_webgl_shaders
from .utils import (
    capture_window,
    compile_shader_program,
    create_vao,
    create_window_with_fallback,
    init_window,
    render_scene,
    set_camera_uniforms,
    set_default_gl_state,
    set_lighting_uniforms,
    setup_buffers,
    setup_shader,
    setup_vertex_attributes,
    terminate_context,
)
from .views import get_view_matrices, get_view_matrix

__all__ = [
    'create_vao', 'compile_shader_program', 'setup_buffers', 'setup_vertex_attributes',
    'set_default_gl_state', 'set_camera_uniforms', 'set_lighting_uniforms',
    'init_window', 'render_scene', 'setup_shader', 'capture_window',
    'make_model', 'make_projection', 'make_view',
    'get_default_shaders', 'get_view_matrices', 'get_view_matrix',
    'get_webgl_shaders', 'terminate_context',
]
