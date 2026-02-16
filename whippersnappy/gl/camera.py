"""Camera and transform helpers (moved under gl package)."""

import pyrr


def make_projection(width, height, fov=20.0, near=0.1, far=100.0):
    """Create a perspective projection matrix."""
    return pyrr.matrix44.create_perspective_projection(fov, width / height, near, far)


def make_view(camera_pos=(0.0, 0.0, -5.0)):
    """Create a view matrix from a camera position."""
    return pyrr.matrix44.create_from_translation(pyrr.Vector3(camera_pos))


def make_model():
    """Create a default model matrix."""
    return pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0, 0.0, 0.0]))


def make_transform(translation, rotation, scale):
    """Create a model transform from translation, rotation and scale."""
    scale_matrix = pyrr.matrix44.create_from_scale([scale, scale, scale])
    return (
        pyrr.matrix44.create_from_translation(translation)
        * rotation
        * scale_matrix
    )

