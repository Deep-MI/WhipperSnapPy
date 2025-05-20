"""Contains the core functionalities of WhipperSnapPy.

Dependencies:
    numpy, glfw, pyrr, PyOpenGL, pillow

@Author    : Martin Reuter
@Created   : 27.02.2022
@Revised   : 16.03.2022

"""

import math
import os
import sys

import glfw
import numpy as np
import OpenGL.GL as gl
import OpenGL.GL.shaders as shaders
import pyrr
from PIL import Image, ImageDraw, ImageFont

from .read_geometry import read_geometry, read_mgh_data, read_morph_data


def normalize_mesh(v, scale=1.0):
    """
    Normalize mesh vertex coordinates.

    - Center their bounding box at the origin.
    - Ensure that the longest side-length is equal to the scale variable (default 1).

    Parameters
    ----------
    v : numpy.ndarray
        Vertex array (Nvert X 3).
    scale : float
        Scaling constant.

    Returns
    -------
    v: numpy.ndarray
        Normalized vertex array (Nvert X 3).
    """
    # center bounding box at origin
    # scale longest side to scale (default 1)
    bbmax = np.max(v, axis=0)
    bbmin = np.min(v, axis=0)
    v = v - 0.5 * (bbmax + bbmin)
    v = scale * v / np.max(bbmax - bbmin)
    return v


# adopted from lapy
def vertex_normals(v, t):
    """
    Compute vertex normals.

    Triangle normals around each vertex are averaged, weighted by the angle
    that they contribute.
    Vertex ordering is important in t: counterclockwise when looking at the
    triangle from above, so that normals point outwards.

    Parameters
    ----------
    v : numpy.ndarray
        Vertex array (Nvert X 3).
    t : numpy.ndarray
        Triangle array (Ntria X 3).

    Returns
    -------
    normals: numpy.ndarray
        Normals array: n - normals (Nvert X 3).
    """
    # Compute vertex coordinates and a difference vector for each triangle:
    v0 = v[t[:, 0], :]
    v1 = v[t[:, 1], :]
    v2 = v[t[:, 2], :]
    v1mv0 = v1 - v0
    v2mv1 = v2 - v1
    v0mv2 = v0 - v2
    # Compute cross product at every vertex
    # will point into the same direction with lengths depending on spanned area
    cr0 = np.cross(v1mv0, -v0mv2)
    cr1 = np.cross(v2mv1, -v1mv0)
    cr2 = np.cross(v0mv2, -v2mv1)
    # Add normals at each vertex (there can be duplicate indices in t at vertex i)
    n = np.zeros(v.shape)
    np.add.at(n, t[:, 0], cr0)
    np.add.at(n, t[:, 1], cr1)
    np.add.at(n, t[:, 2], cr2)
    # Normalize normals
    ln = np.sqrt(np.sum(n * n, axis=1))
    ln[ln < sys.float_info.epsilon] = 1  # avoid division by zero
    n = n / ln.reshape(-1, 1)
    return n


def heat_color(values, invert=False):
    """
    Convert an array of float values into RBG heat color values.

    Only values between -1 and 1 will receive gradient and colors will
    max-out at -1 and 1. Negative values will be blue and positive
    red (unless invert is passed to flip the heatmap). Masked values
    (nan) will map to masked colors (nan,nan,nan).

    Parameters
    ----------
    values : numpy.ndarray
        Float values of function on the surface mesh (length Nvert).
    invert : bool
        Whether to invert the heat map (blue is positive and red negative).

    Returns
    -------
    colors: numpy.ndarray
        (Nvert x 3) array of RGB of heat map as 0.0 .. 1.0 floats.
    """
    # values (1 dim array length n) will receive gradient between -1 and 1
    # nan will return (nan,nan,nan)
    # returns colors (r,g,b)  as n x 3 array
    if invert:
        values = -1.0 * values
    vabs = np.abs(values)
    colors = np.zeros((vabs.size, 3), dtype=np.float32)
    crb = 0.5625 + 3 * 0.4375 * vabs
    cg = 1.5 * (vabs - (1.0 / 3.0))
    n1 = values < -1.0
    nm = (values >= -1.0) & (values < -(1.0 / 3.0))
    n0 = (values >= -(1.0 / 3.0)) & (values < 0)
    p0 = (values >= 0) & (values < (1.0 / 3.0))
    pm = (values >= (1.0 / 3.0)) & (values < 1.0)
    p1 = values >= 1.0
    # fill in colors for the 5 blocks
    colors[n1, 1:3] = 1.0  # bright blue
    colors[nm, 1] = cg[nm]  # cg increasing green channel
    colors[nm, 2] = 1.0  # and keeping blue on full
    colors[n0, 2] = crb[n0]  # crb increasing blue channel
    colors[p0, 0] = crb[p0]  # crb increasing red channel
    colors[pm, 1] = cg[pm]  # cg increasing green channel
    colors[pm, 0] = 1.0  # and keeping red on full
    colors[p1, 0:2] = 1.0  # yellow
    colors[np.isnan(values), :] = np.nan
    return colors


def rescale_overlay(values, minval=None, maxval=None):
    """
    Rescale values for color map computation.

    minval and maxval are two positive floats (maxval>minval).
    Values between -minval and minval will be masked (np.nan);
    others will be shifted towards zero (from both sides)
    and scaled so that -maxval and maxval are at -1 and +1.

    Parameters
    ----------
    values : numpy.ndarray
        Float values of function on the surface (length Nvert).
    minval : float
        Minimum value.
    maxval : float
        Maximum value.

    Returns
    -------
    values: numpy.ndarray
        Float array of input function on mesh (length Nvert).
    minval: float
        Positive minimum value (crop values whose absolute value is below).
    maxval: float
        Positive maximum value (saturate color at maxval and -maxval).
    neg: bool
        Whether negative values are present at all after cropping.
    """
    valsign = np.sign(values)
    valabs = np.abs(values)
    realmin = np.min(values)
    if maxval is None:
        maxval = np.max(valabs)
    if minval is None:
        minval = max(0.0, np.min(valabs))
    if maxval < 0 or minval < 0:
        print("resacle_overlay ERROR: min and maxval should both be positive!")
        exit(1)
    # print("Using min {:.2f} and max {:.2f}".format(minval,maxval))
    # rescale map symmetrically to -1 .. 1 (keeping minval at 0)
    # mask values below minval
    values[valabs < minval] = np.nan
    # shift towards 0 from both sides
    values = values - valsign * minval
    # rescale so that former maxval is at 1 (and -1 for negative values)
    values = values / (maxval - minval)
    return values, minval, maxval, (realmin < 0 and realmin < -minval)


def binary_color(values, thres, color_low, color_high):
    """
    Create a binary colormap based on a threshold value.

    This function assigns colors to input values based on whether they are
    below or equal to the threshold (thres) or greater than the threshold.

    Values below thres are color_low, others are color_high.
    color_low and color_high can be float (gray scale), or 1x3 array of RGB.

    Parameters
    ----------
    values : numpy.ndarray
        Input vertex function as float array (length Nvert).
    thres : float
        Threshold value.
    color_low : float or numpy.ndarray
        Lower color value(s).
    color_high : float or numpy.ndarray
        Higher color value(s).

    Returns
    -------
    colors : numpy.ndarray
        Binary colormap.
    """
    if np.isscalar(color_low):
        color_low = np.array((color_low, color_low, color_low), dtype=np.float32)
    if np.isscalar(color_high):
        color_high = np.array((color_high, color_high, color_high), dtype=np.float32)
    colors = np.empty((values.size, 3), dtype=np.float32)
    colors[values < thres, :] = color_low
    colors[values >= thres, :] = color_high
    return colors


def mask_label(values, labelpath=None):
    """
    Apply a labelfile as a mask.

    Labelfile freesurfer format has indices of values that should be kept;
    all other values will be set to np.nan.

    Parameters
    ----------
    values : numpy.ndarray
        Float values of function defined at vertices (a 1-dim array).
    labelpath : str
        Absolute path to label file.

    Returns
    -------
    values: numpy.ndarray
        Masked surface function values.
    """
    if not labelpath:
        return values
    # this is the mask of vertices to keep, e.g. cortex labels
    maskvids = np.loadtxt(labelpath, dtype=int, skiprows=2, usecols=[0])
    imask = np.ones(values.shape, dtype=bool)
    imask[maskvids] = False
    values[imask] = np.nan
    return values


def prepare_geometry(
    surfpath,
    overlaypath=None,
    curvpath=None,
    labelpath=None,
    minval=None,
    maxval=None,
    invert=False,
):
    """
    Prepare meshdata for upload to GPU.

    Vertex coordinates, vertex normals and color values are concatenated into
    large vertexdata array. Also returns triangles, minimum and maximum overlay
    values as well as whether negative values are present or not in triangles.

    Parameters
    ----------
    surfpath : str
        Path to surface file (usually lh or rh.pial_semi_inflated).
    overlaypath : str
        Path to overlay file.
    curvpath : str
        Path to curvature file (usually lh or rh.curv).
    labelpath : str
        Path to label file (mask; usually cortex.label).
    minval : float
        Minimum threshold to stop coloring (-minval used for neg values).
    maxval : float
        Maximum value to saturate (-maxval used for negative values).
    invert : bool
        Invert color map.

    Returns
    -------
    vertexdata: numpy.ndarray
        Concatenated array with vertex coords, vertex normals and colors
        as a (Nvert X 9) float32 array.
    triangles: numpy.ndarray
        Triangle array as a (Ntria X 3) uint32 array.
    fmin: float
        Minimum value of overlay function after rescale.
    fmax: float
        Maximum value of overlay function after rescale.
    neg: bool
        Whether negative values are there after rescale/cropping.
    """

    # read vertices and triangles
    surf = read_geometry(surfpath, read_metadata=False)
    vertices = normalize_mesh(np.array(surf[0], dtype=np.float32), 1.85)
    triangles = np.array(surf[1], dtype=np.uint32)
    # compute vertex normals
    vnormals = np.array(vertex_normals(vertices, triangles), dtype=np.float32)
    # read curvature
    if curvpath:
        curv = read_morph_data(curvpath)
        sulcmap = binary_color(curv, 0.0, color_low=0.5, color_high=0.33)
    else:
        # if no curv pattern, color mesh in mid-gray
        sulcmap = 0.5 * np.ones(vertices.shape, dtype=np.float32)
    # read map (stats etc)
    if overlaypath:
        _, file_extension = os.path.splitext(overlaypath)

        if file_extension == ".mgh":
            mapdata = read_mgh_data(overlaypath)
        else:
            mapdata = read_morph_data(overlaypath)
        mapdata, fmin, fmax, neg = rescale_overlay(mapdata, minval, maxval)
        # mask map with label
        mapdata = mask_label(mapdata, labelpath)
        # compute color
        colors = heat_color(mapdata, invert)
        missing = np.isnan(mapdata)
        colors[missing, :] = sulcmap[missing, :]
    else:
        colors = sulcmap
    # concatenate matrices
    vertexdata = np.concatenate((vertices, vnormals, colors), axis=1)
    return vertexdata, triangles, fmin, fmax, neg


def init_window(width, height, title="PyOpenGL", visible=True):
    """
    Create window with width, height, title.

    If visible False, hide window.

    Parameters
    ----------
    width : int
        Window width.
    height : int
        Window height.
    title : str
        Window title.
    visible : bool
       Window visibility.

    Returns
    -------
    window: glfw.LP__GLFWwindow
        GUI window.
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
    # Enable key events
    glfw.set_input_mode(window, glfw.STICKY_KEYS, gl.GL_TRUE)
    # Enable key event callback
    # glfw.set_key_callback(window,key_event)
    glfw.make_context_current(window)
    # vsync and glfw do not play nice.  when vsync is enabled mouse movement is jittery.
    glfw.swap_interval(0)
    return window


def setup_shader(meshdata, triangles, width, height, specular=True):
    """
    Create vertex and fragment shaders.

    Set up data and parameters (such as the initial view matrix) on the GPU.

    In meshdata:
      - the first 3 columns are the vertex coordinates
      - the next  3 columns are the vertex normals
      - the final 3 columns are the color RGB values

    Parameters
    ----------
    meshdata : numpy.ndarray
        Mesh array (shape: n x 9, dtype: np.float32).
    triangles : bool
       Triangle indices array (shape: m x 3).
    width : int
        Window width (to set perspective projection).
    height : int
        Window height (to set perspective projection).
    specular : Boolean
        By default specular is set as True.

    Returns
    -------
    shader: ShaderProgram
        Compiled OpenGL shader program.
    """

    VERTEX_SHADER = """

        #version 330

        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aNormal;
        layout (location = 2) in vec3 aColor;

        out vec3 FragPos;
        out vec3 Normal;
        out vec3 Color;

        uniform mat4 transform;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        void main()
        {
          gl_Position = projection * view * model * transform * vec4(aPos, 1.0f);
          FragPos = vec3(model * transform * vec4(aPos, 1.0));
          // normal matrix should be computed outside and passed!
          Normal = mat3(transpose(inverse(view * model * transform))) * aNormal;
          Color = aColor;
        }

    """

    FRAGMENT_SHADER = """
        #version 330

        in vec3 Normal;
        in vec3 FragPos;
        in vec3 Color;

        out vec4 FragColor;

        uniform vec3 lightColor;
        uniform bool doSpecular;

        void main()
        {
          // ambient
          float ambientStrength = 0.0;
          vec3 ambient = ambientStrength * lightColor;

          // diffuse
          vec3 norm = normalize(Normal);
          // key light (overhead)
          vec3 lightPos1 = vec3(0.0,5.0,5.0);
          vec3 lightDir = normalize(lightPos1 - FragPos);
          float diff = max(dot(norm, lightDir), 0.0);
          float key = 0.6;
          vec3 diffuse = key * diff * lightColor;

          // headlight (at camera)
          vec3 lightPos2 = vec3(0.0,0.0,5.0);
          lightDir = normalize(lightPos2 - FragPos);
          vec3 ohlightDir = lightDir;
          diff = max(dot(norm, lightDir), 0.0);
          diffuse = diffuse + 0.68  * key * diff * lightColor;

          // fill light (from below)
          vec3 lightPos3 = vec3(0.0,-5.0,5.0);
          lightDir = normalize(lightPos3 - FragPos);
          diff = max(dot(norm, lightDir), 0.0);
          diffuse = diffuse + 0.6  * key * diff * lightColor;

          // left right back lights
          vec3 lightPos4 = vec3(5.0,0.0,-5.0);
          lightDir = normalize(lightPos4 - FragPos);
          diff = max(dot(norm, lightDir), 0.0);
          diffuse = diffuse + 0.52 * key * diff * lightColor;
          vec3 lightPos5 = vec3(-5.0,0.0,-5.0);
          lightDir = normalize(lightPos5 - FragPos);
          diff = max(dot(norm, lightDir), 0.0);
          diffuse = diffuse + 0.52 * key * diff * lightColor;

          // specular
          vec3 result;
          if (doSpecular)
          {
            float specularStrength = 0.5;
            // the viewer is always at (0,0,0) in view-space,
            // so viewDir is (0,0,0) - Position => -Position
            vec3 viewDir = normalize(-FragPos);
            vec3 reflectDir = reflect(ohlightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
            vec3 specular = specularStrength * spec * lightColor;
            // final color
            result = (ambient + diffuse + specular) * Color;
          }
          else
          {
            // final color no specular
            result = (ambient + diffuse) * Color;
          }
          FragColor = vec4(result, 1.0);
        }

    """

    # Create Vertex Buffer object in gpu
    VBO = gl.glGenBuffers(1)
    # Bind the buffer
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, VBO)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, meshdata.nbytes, meshdata, gl.GL_STATIC_DRAW)

    # Create Vertex Array object
    VAO = gl.glGenVertexArrays(1)
    # Bind array
    gl.glBindVertexArray(VAO)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, meshdata.nbytes, meshdata, gl.GL_STATIC_DRAW)

    # Create Element Buffer Object
    EBO = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, EBO)
    gl.glBufferData(
        gl.GL_ELEMENT_ARRAY_BUFFER, triangles.nbytes, triangles, gl.GL_STATIC_DRAW
    )

    # Compile The Program and shaders
    shader = gl.shaders.compileProgram(
        shaders.compileShader(VERTEX_SHADER, gl.GL_VERTEX_SHADER),
        shaders.compileShader(FRAGMENT_SHADER, gl.GL_FRAGMENT_SHADER),
    )

    # get the position from shader
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

    gl.glUseProgram(shader)

    gl.glClearColor(0.0, 0.0, 0.0, 1.0)
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Creating Projection Matrix
    view = pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0, 0.0, -5.0]))
    projection = pyrr.matrix44.create_perspective_projection(
        20.0, width / height, 0.1, 100.0
    )
    model = pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0, 0.0, 0.0]))

    # Set matrices in vertex shader
    view_loc = gl.glGetUniformLocation(shader, "view")
    proj_loc = gl.glGetUniformLocation(shader, "projection")
    model_loc = gl.glGetUniformLocation(shader, "model")
    gl.glUniformMatrix4fv(view_loc, 1, gl.GL_FALSE, view)
    gl.glUniformMatrix4fv(proj_loc, 1, gl.GL_FALSE, projection)
    gl.glUniformMatrix4fv(model_loc, 1, gl.GL_FALSE, model)

    # setup doSpecular in fragment shader
    specular_loc = gl.glGetUniformLocation(shader, "doSpecular")
    gl.glUniform1i(specular_loc, specular)

    # setup light color in fragment shader
    lightColor_loc = gl.glGetUniformLocation(shader, "lightColor")
    gl.glUniform3f(lightColor_loc, 1.0, 1.0, 1.0)

    return shader


def capture_window(width, height):
    """
    Capture the GL region (0,0) .. (width,height) into PIL Image.

    Parameters
    ----------
    width : int
        Window width.
    height : int
        Window height.

    Returns
    -------
    image: PIL.Image.Image
        Captured image.
    """
    if sys.platform == "darwin":
        # not sure why on mac the drawing area is 4 times as large (2x2):
        width = 2 * width
        height = 2 * height
    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)  # may not be needed
    img_buf = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGB", (width, height), img_buf)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    if sys.platform == "darwin":
        image.thumbnail((0.5 * width, 0.5 * height), Image.Resampling.LANCZOS)
    return image


def create_colorbar(fmin, fmax, invert, neg=True, font_file=None):
    """
    Create colorbar image with text indicating min and max values.

    Parameters
    ----------
    fmin : int
        Absolute min value that receives color (threshold).
    fmax : int
        Absolute max value where color saturates.
    invert : bool
        Color invert.
    neg : bool
        Show negative axis.
    font_file : str
        Path to the file describing the font to be used.

    Returns
    -------
    image: PIL.Image.Image
        Colorbar image.
    """
    cwidth = 200
    cheight = 30
    # img = Image.new("RGB", (cwidth, cheight), color=(90, 90, 90))
    values = np.nan * np.ones(cwidth)
    gapspace = 0
    if fmin > 0.01:
        # leave gray gap
        num = int(0.42 * cwidth)
        gapspace = 0.08 * cwidth
    else:
        num = int(0.5 * cwidth)
    if not neg:
        num = num * 2
        gapspace = gapspace * 2
    vals = np.linspace(0.01, 1, num)
    if not neg:
        values[-vals.size :] = vals
    else:
        values[: vals.size] = -1.0 * np.flip(vals)
        values[-vals.size :] = vals

    colors = heat_color(values, invert)
    colors[np.isnan(values), :] = 0.33 * np.ones((1, 3))
    img_bar = np.uint8(np.tile(colors, (cheight, 1, 1)) * 255)
    # pad with black
    img_buf = np.zeros((cheight + 20, cwidth + 20, 3), dtype=np.uint8)
    img_buf[3 : cheight + 3, 10 : cwidth + 10, :] = img_bar
    image = Image.fromarray(img_buf)

    if font_file is None:
        script_dir = "/".join(str(__file__).split("/")[:-1])
        font_file = os.path.join(script_dir, "Roboto-Regular.ttf")
    font = ImageFont.truetype(font_file, 12)
    if neg:
        # Left
        caption = f" <{-fmax:.2f}"
        xpos = 0  # 10- 0.5*(font.getlength(caption))
        ImageDraw.Draw(image).text(
            (xpos, image.height - 17), caption, (220, 220, 220), font=font
        )
        # Right
        caption = f">{fmax:.2f} "
        xpos = image.width - (font.getlength(caption))
        ImageDraw.Draw(image).text(
            (xpos, image.height - 17), caption, (220, 220, 220), font=font
        )
        if gapspace == 0:
            caption = "0"
            xpos = 0.5 * image.width - 0.5 * font.getlength(caption)
            ImageDraw.Draw(image).text(
                (xpos, image.height - 17), caption, (220, 220, 220), font=font
            )
        else:
            caption = f"{-fmin:.2f}"
            xpos = 0.5 * image.width - 0.5 * font.getlength(caption) - gapspace - 5
            ImageDraw.Draw(image).text(
                (xpos, image.height - 17), caption, (220, 220, 220), font=font
            )
            caption = f"{fmin:.2f}"
            xpos = 0.5 * image.width - 0.5 * font.getlength(caption) + gapspace + 5
            ImageDraw.Draw(image).text(
                (xpos, image.height - 17), caption, (220, 220, 220), font=font
            )
    else:
        # Right
        caption = f">{fmax:.2f} "
        xpos = image.width - (font.getlength(caption))
        ImageDraw.Draw(image).text(
            (xpos, image.height - 17), caption, (220, 220, 220), font=font
        )
        # Left
        caption = f" {fmin:.2f}"
        xpos = gapspace
        if gapspace == 0:
            caption = " 0"
            xpos = 5
        ImageDraw.Draw(image).text(
            (xpos, image.height - 17), caption, (220, 220, 220), font=font
        )

    return image


def snap4(
    lhoverlaypath,
    rhoverlaypath,
    fthresh=None,
    fmax=None,
    sdir=None,
    caption=None,
    invert=False,
    labelname="cortex.label",
    surfname=None,
    curvname="curv",
    colorbar=True,
    outpath=None,
    font_file=None,
    specular=True,
):
    """
    Snap four views (front and back for left and right hemispheres).

    Save an image that includes the views and a color bar.

    Parameters
    ----------
    lhoverlaypath : str
        Path to the overlay files for left hemi (FreeSurfer format).
    rhoverlaypath : str
        Path to the overlay files for right hemi (FreeSurfer format).
    fthresh : float
        Pos absolute value under which no color is shown.
    fmax : float
        Pos absolute value above which color is saturated.
    sdir : str
       Subject dir containing surf files.
    caption : str
       Caption text to be placed on the image.
    invert : bool
       Invert color (blue positive, red negative).
    labelname : str
       Label for masking, usually cortex.label.
    surfname : str
       Surface to display values on, usually pial_semi_inflated from fsaverage.
    curvname : str
       Curvature file for texture in non-colored regions (default curv).
    colorbar : bool
       Show colorbar on image.
    outpath : str
        Path to the output image file.
    font_file : str
        Path to the file describing the font to be used in captions.
    specular : bool
        Specular is by default set as True.

    Returns
    -------
    None
        This function returns None.
    """
    # setup window
    # (keep aspect ratio, as the mesh scale and distances are set accordingly)
    wwidth = 540
    weight = 450
    visible = True
    window = init_window(wwidth, weight, "WhipperSnapPy 2.0", visible)
    if not window:
        return False  # need raise error here in future

    # set up matrices to show object left and right side:
    rot_z = pyrr.Matrix44.from_z_rotation(-0.5 * math.pi)
    rot_x = pyrr.Matrix44.from_x_rotation(0.5 * math.pi)
    # rot_y = pyrr.Matrix44.from_y_rotation(math.pi/6)
    viewLeft = rot_x * rot_z
    rot_y = pyrr.Matrix44.from_y_rotation(math.pi)
    viewRight = rot_y * viewLeft
    transl = pyrr.Matrix44.from_translation((0, 0, 0.4))

    for hemi in ("lh", "rh"):
        if surfname is None:
            print(
                "[INFO] No surf_name provided. Looking for options in surf directory..."
            )

            if sdir is None:
                sdir = os.environ.get("SUBJECTS_DIR")
                if not sdir:
                    print(
                        "[INFO] No surf_name or subjects directory (sdir) \
provided, can not find surf file"
                    )
                    sys.exit(1)

            found_surfname = get_surf_name(sdir, hemi)

            if found_surfname is None:
                print(
                    f"[ERROR] Could not find valid surface in {sdir} for hemi: {hemi}!"
                )
                sys.exit(1)
            meshpath = os.path.join(sdir, "surf", hemi + "." + found_surfname)
        else:
            meshpath = os.path.join(sdir, "surf", hemi + "." + surfname)

        curvpath = None
        if curvname:
            curvpath = os.path.join(sdir, "surf", hemi + "." + curvname)
        labelpath = None
        if labelname:
            labelpath = os.path.join(sdir, "label", hemi + "." + labelname)
        if hemi == "lh":
            overlaypath = lhoverlaypath
        else:
            overlaypath = rhoverlaypath

        # load and colorzie data
        meshdata, triangles, fthresh, fmax, neg = prepare_geometry(
            meshpath, overlaypath, curvpath, labelpath, fthresh, fmax, invert
        )
        # upload to GPU and compile shaders
        shader = setup_shader(meshdata, triangles, wwidth, weight, specular=specular)

        # draw
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        transformLoc = gl.glGetUniformLocation(shader, "transform")
        viewmat = viewLeft
        if hemi == "lh":
            viewmat = transl * viewmat
        gl.glUniformMatrix4fv(transformLoc, 1, gl.GL_FALSE, viewmat)
        gl.glDrawElements(gl.GL_TRIANGLES, triangles.size, gl.GL_UNSIGNED_INT, None)

        im1 = capture_window(wwidth, weight)

        glfw.swap_buffers(window)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        viewmat = viewRight
        if hemi == "rh":
            viewmat = transl * viewmat
        gl.glUniformMatrix4fv(transformLoc, 1, gl.GL_FALSE, viewmat)
        gl.glDrawElements(gl.GL_TRIANGLES, triangles.size, gl.GL_UNSIGNED_INT, None)

        im2 = capture_window(wwidth, weight)

        if hemi == "lh":
            lhimg = Image.new("RGB", (im1.width, im1.height + im2.height))
            lhimg.paste(im1, (0, 0))
            lhimg.paste(im2, (0, im1.height))
        else:
            rhimg = Image.new("RGB", (im1.width, im1.height + im2.height))
            rhimg.paste(im2, (0, 0))
            rhimg.paste(im1, (0, im2.height))

    image = Image.new("RGB", (lhimg.width + rhimg.width, lhimg.height))
    image.paste(lhimg, (0, 0))
    image.paste(rhimg, (im1.width, 0))

    if caption:
        if font_file is None:
            script_dir = "/".join(str(__file__).split("/")[:-1])
            font_file = os.path.join(script_dir, "Roboto-Regular.ttf")
        font = ImageFont.truetype(font_file, 20)
        xpos = 0.5 * (image.width - font.getlength(caption))
        ImageDraw.Draw(image).text(
            (xpos, image.height - 40), caption, (220, 220, 220), font=font
        )

    if colorbar:
        bar = create_colorbar(fthresh, fmax, invert, neg)
        xpos = int(0.5 * (image.width - bar.width))
        ypos = int(0.5 * (image.height - bar.height))
        image.paste(bar, (xpos, ypos))

    if outpath:
        print(f"[INFO] Saving snapshot to {outpath}")
        image.save(outpath)


def get_surf_name(sdir, hemi):
    """
    Find a valid surface file in the specified subject directory.

    A valid file can be one of: ['pial_semi_inflated', 'white', 'inflated'].

    Parameters
    ----------
    sdir : str
        Subject directory.
    hemi : str
        Hemisphere; one of: ['lh', 'rh'].

    Returns
    -------
    surfname: str
        Valid and existing surf file's name; otherwise, None.
    """
    for surf_name_option in ["pial_semi_inflated", "white", "inflated"]:
        if os.path.exists(os.path.join(sdir, "surf", hemi + "." + surf_name_option)):
            print("[INFO] Found {}".format(hemi + "." + surf_name_option))
            return surf_name_option
        else:
            print("[INFO] No {} file found".format(hemi + "." + surf_name_option))
    else:
        return None
