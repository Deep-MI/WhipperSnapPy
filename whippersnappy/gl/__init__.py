"""OpenGL helper utilities (gl package).

This package contains the low-level OpenGL helpers used by the renderers.
View preset matrices (:func:`~whippersnappy.utils.types.get_view_matrices`)
live in :mod:`whippersnappy.utils.types` alongside :class:`~whippersnappy.utils.types.ViewType`.

Functions are re-exported at package level for convenience::

    from whippersnappy.gl import init_window, setup_shader

"""

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

__all__ = [
    'create_vao', 'compile_shader_program', 'setup_buffers', 'setup_vertex_attributes',
    'set_default_gl_state', 'set_camera_uniforms', 'set_lighting_uniforms',
    'init_window', 'render_scene', 'setup_shader', 'capture_window',
    'create_window_with_fallback', 'terminate_context',
    'make_model', 'make_projection', 'make_view',
    'get_default_shaders', 'get_webgl_shaders',
]
