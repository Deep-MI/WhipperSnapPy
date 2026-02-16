"""GL helper utilities.

Contains the implementation of OpenGL helpers used by the package.
"""

import sys

import glfw
import OpenGL.GL as gl
import OpenGL.GL.shaders as shaders
from PIL import Image

from .camera import make_model, make_projection, make_view
from .shaders import get_default_shaders


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


def compile_shader_program(vertex_src, fragment_src):
    """Compile GLSL vertex and fragment sources and link them into a program.

    Parameters
    ----------
    vertex_src : str
        Vertex shader source code.
    fragment_src : str
        Fragment shader source code.

    Returns
    -------
    int
        OpenGL program handle.
    """
    return gl.shaders.compileProgram(
        shaders.compileShader(vertex_src, gl.GL_VERTEX_SHADER),
        shaders.compileShader(fragment_src, gl.GL_FRAGMENT_SHADER),
    )


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

    This function enables depth testing and sets a default clear color.
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
        4x4 matrices to be uploaded to the corresponding shader uniforms.
    """
    view_loc = gl.glGetUniformLocation(shader, "view")
    proj_loc = gl.glGetUniformLocation(shader, "projection")
    model_loc = gl.glGetUniformLocation(shader, "model")
    gl.glUniformMatrix4fv(view_loc, 1, gl.GL_FALSE, view)
    gl.glUniformMatrix4fv(proj_loc, 1, gl.GL_FALSE, projection)
    gl.glUniformMatrix4fv(model_loc, 1, gl.GL_FALSE, model)


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
    specular_loc = gl.glGetUniformLocation(shader, "doSpecular")
    gl.glUniform1i(specular_loc, specular)

    light_color_loc = gl.glGetUniformLocation(shader, "lightColor")
    gl.glUniform3f(light_color_loc, *light_color)

    ambient_loc = gl.glGetUniformLocation(shader, "ambientStrength")
    gl.glUniform1f(ambient_loc, ambient)


def init_window(width, height, title="PyOpenGL", visible=True):
    """Create a GLFW window, make an OpenGL context current and return the window handle.

    Parameters
    ----------
    width, height : int
        Window dimensions in pixels.
    title : str, optional, default 'PyOpenGL'
        Window title.
    visible : bool, optional, default True
        If False create an invisible/offscreen window (useful for headless rendering).

    Returns
    -------
    window or False
        GLFW window handle on success, or False on failure.
    """
    if not glfw.init():
        return False

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    if not visible:
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(width, height, title, None, None)
    if not window:
        glfw.terminate()
        return False
    glfw.set_input_mode(window, glfw.STICKY_KEYS, gl.GL_TRUE)
    glfw.make_context_current(window)
    glfw.swap_interval(0)
    return window


def setup_shader(meshdata, triangles, width, height, specular=True, ambient=0.0):
    """Create shader program, upload mesh and initialize camera & lighting.

    This is a convenience wrapper that compiles default shaders, creates
    VAO/VBO/EBO, and configures common uniforms (camera matrices, lighting).

    Parameters
    ----------
    meshdata : numpy.ndarray
        Interleaved vertex data.
    triangles : numpy.ndarray
        Triangle indices.
    width, height : int
        Framebuffer size used to compute projection matrix.
    specular : bool, optional, default True
        Enable specular highlights.
    ambient : float, optional, default 0.0
        Ambient lighting strength.

    Returns
    -------
    shader : int
        Compiled OpenGL shader program handle.
    """
    vertex_shader, fragment_shader = get_default_shaders()

    create_vao()
    shader = compile_shader_program(vertex_shader, fragment_shader)
    setup_buffers(meshdata, triangles)
    setup_vertex_attributes(shader)

    gl.glUseProgram(shader)
    set_default_gl_state()

    view = make_view()
    projection = make_projection(width, height)
    model = make_model()
    set_camera_uniforms(shader, view, projection, model)
    set_lighting_uniforms(shader, specular=specular, ambient=ambient)

    return shader


def capture_window(width, height):
    """Read the current GL framebuffer and return it as a PIL.Image (RGB).

    On macOS (retina) this function reads at double resolution and downscales
    the result to compensate for pixel-density differences.

    Parameters
    ----------
    width, height : int
        Desired output image dimensions.

    Returns
    -------
    PIL.Image.Image
        RGB image containing the captured framebuffer content.
    """
    if sys.platform == "darwin":
        rwidth = 2 * width
        rheight = 2 * height
    else:
        rwidth = width
        rheight = height

    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
    img_buf = gl.glReadPixels(0, 0, rwidth, rheight, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGB", (rwidth, rheight), img_buf)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    if sys.platform == "darwin":
        image.thumbnail((0.5 * rwidth, 0.5 * rheight), Image.Resampling.LANCZOS)
    return image
