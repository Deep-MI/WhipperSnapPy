"""Camera and transform helpers (moved under gl package)."""

import pyrr


def make_projection(width, height, fov=20.0, near=0.1, far=100.0):
    """Create a 4x4 perspective projection matrix.

    Parameters
    ----------
    width, height : int
        Viewport dimensions in pixels (used to compute aspect ratio).
    fov : float, optional, default 20.0
        Vertical field of view in degrees.
    near, far : float, optional, default 0.1, 100.0
        Near and far clipping planes. Default are 0.1 and 100.0, respectively.

    Returns
    -------
    numpy.ndarray
        4x4 projection matrix.
    """
    return pyrr.matrix44.create_perspective_projection(fov, width / height, near, far)


def make_view(camera_pos=(0.0, 0.0, -5.0)):
    """Create a view matrix for a camera located at ``camera_pos``.

    Parameters
    ----------
    camera_pos : sequence of float, optional, default (0.0, 0.0, -5.0)
        3-element position of the camera in world space.
        Default is (0.0, 0.0, -5.0).

    Returns
    -------
    numpy.ndarray
        4x4 view matrix.
    """
    return pyrr.matrix44.create_from_translation(pyrr.Vector3(camera_pos))


def make_model():
    """Create a default model matrix (identity translation).

    Returns
    -------
    numpy.ndarray
        4x4 model matrix.
    """
    return pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0, 0.0, 0.0]))


def make_transform(translation, rotation, scale):
    """Build a model transform matrix from translation, rotation and uniform scale.

    Parameters
    ----------
    translation : sequence of float
        3-element translation vector.
    rotation : numpy.ndarray
        4x4 rotation matrix.
    scale : float
        Uniform scaling factor.

    Returns
    -------
    numpy.ndarray
        4x4 transformation matrix (translation * rotation * scale).
    """
    scale_matrix = pyrr.matrix44.create_from_scale([scale, scale, scale])
    return (
        pyrr.matrix44.create_from_translation(translation)
        * rotation
        * scale_matrix
    )
