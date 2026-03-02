"""Rendering pipeline setup: VAO/VBO/EBO creation, vertex attributes, uniforms, and shader setup.

This module owns the one-time GPU upload step that happens after an OpenGL
context is created and before the first draw call:

1. Compile shaders and link program (:func:`~whippersnappy.gl.shaders.compile_shader_program`)
2. Create VAO (:func:`create_vao`)
3. Upload mesh to VBO/EBO (:func:`setup_buffers`)
4. Configure vertex attribute pointers (:func:`setup_vertex_attributes`)
5. Upload camera and lighting uniforms (:func:`set_camera_uniforms`, :func:`set_lighting_uniforms`)

:func:`setup_shader` is the single convenience entry-point that runs all
five steps in order and returns the compiled shader program handle.
"""

import OpenGL.GL as gl

from .camera import make_model, make_projection, make_view
from .shaders import compile_shader_program, get_default_shaders


def create_vao():
    """Create and bind a Vertex Array Object (VAO).

    Returns
    -------
    int
        OpenGL handle for the created VAO.
    """
    vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vao)
    return vao


def setup_buffers(meshdata, triangles):
    """Create and upload vertex and element buffers for the mesh.

    Parameters
    ----------
    meshdata : numpy.ndarray
        Vertex array with interleaved attributes (position, normal, color).
    triangles : numpy.ndarray
        Face index array.

    Returns
    -------
    (vbo, ebo) : tuple
        OpenGL buffer handles for the VBO and EBO.
    """
    vbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, meshdata.nbytes, meshdata, gl.GL_STATIC_DRAW)

    ebo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
    gl.glBufferData(
        gl.GL_ELEMENT_ARRAY_BUFFER, triangles.nbytes, triangles, gl.GL_STATIC_DRAW
    )

    return vbo, ebo


def setup_vertex_attributes(shader):
    """Configure vertex attribute pointers for position, normal and color.

    Parameters
    ----------
    shader : int
        OpenGL shader program handle used to query attribute locations.
    """
    position = gl.glGetAttribLocation(shader, "aPos")
    gl.glVertexAttribPointer(
        position, 3, gl.GL_FLOAT, gl.GL_FALSE, 9 * 4, gl.ctypes.c_void_p(0)
    )
    gl.glEnableVertexAttribArray(position)

    vnormalpos = gl.glGetAttribLocation(shader, "aNormal")
    gl.glVertexAttribPointer(
        vnormalpos, 3, gl.GL_FLOAT, gl.GL_FALSE, 9 * 4, gl.ctypes.c_void_p(3 * 4)
    )
    gl.glEnableVertexAttribArray(vnormalpos)

    colorpos = gl.glGetAttribLocation(shader, "aColor")
    gl.glVertexAttribPointer(
        colorpos, 3, gl.GL_FLOAT, gl.GL_FALSE, 9 * 4, gl.ctypes.c_void_p(6 * 4)
    )
    gl.glEnableVertexAttribArray(colorpos)


def set_default_gl_state():
    """Set frequently used default OpenGL state for rendering.

    Enables depth testing and sets a black clear color.
    """
    gl.glClearColor(0.0, 0.0, 0.0, 1.0)
    gl.glEnable(gl.GL_DEPTH_TEST)


def set_camera_uniforms(shader, view, projection, model):
    """Upload camera MVP (view, projection, model) matrices to the shader.

    Parameters
    ----------
    shader : int
        OpenGL shader program handle.
    view, projection, model : array-like
        4×4 matrices uploaded to the corresponding shader uniforms.
    """
    gl.glUniformMatrix4fv(gl.glGetUniformLocation(shader, "view"),       1, gl.GL_FALSE, view)
    gl.glUniformMatrix4fv(gl.glGetUniformLocation(shader, "projection"), 1, gl.GL_FALSE, projection)
    gl.glUniformMatrix4fv(gl.glGetUniformLocation(shader, "model"),      1, gl.GL_FALSE, model)


def set_lighting_uniforms(shader, specular=True, ambient=0.0, light_color=(1.0, 1.0, 1.0)):
    """Set lighting-related uniforms (specular toggle, ambient, light color).

    Parameters
    ----------
    shader : int
        OpenGL shader program handle.
    specular : bool, optional, default True
        Enable specular highlights.
    ambient : float, optional, default 0.0
        Ambient light strength.
    light_color : tuple, optional, default (1.0, 1.0, 1.0)
        RGB light color.
    """
    gl.glUniform1i(gl.glGetUniformLocation(shader, "doSpecular"),    specular)
    gl.glUniform3f(gl.glGetUniformLocation(shader, "lightColor"),    *light_color)
    gl.glUniform1f(gl.glGetUniformLocation(shader, "ambientStrength"), ambient)


def setup_shader(meshdata, triangles, width, height, specular=True, ambient=0.0):
    """Compile shaders, upload mesh, and initialise camera & lighting uniforms.

    Convenience entry-point that runs the full one-time pipeline setup in
    order: compile → VAO → VBO/EBO → attributes → GL state → uniforms.

    Parameters
    ----------
    meshdata : numpy.ndarray
        Interleaved vertex data (position, normal, color).
    triangles : numpy.ndarray
        Triangle index array.
    width, height : int
        Framebuffer size used to compute the projection matrix.
    specular : bool, optional, default True
        Enable specular highlights.
    ambient : float, optional, default 0.0
        Ambient lighting strength.

    Returns
    -------
    shader : int
        Compiled and linked OpenGL shader program handle.
    """
    vertex_shader, fragment_shader = get_default_shaders()

    create_vao()
    shader = compile_shader_program(vertex_shader, fragment_shader)
    setup_buffers(meshdata, triangles)
    setup_vertex_attributes(shader)

    gl.glUseProgram(shader)
    set_default_gl_state()

    set_camera_uniforms(shader, make_view(), make_projection(width, height), make_model())
    set_lighting_uniforms(shader, specular=specular, ambient=ambient)

    return shader

