"""View matrices and presets under gl package."""

import numpy as np

from whippersnappy.utils.types import ViewType


def get_view_matrices():
    """Return canonical 4x4 view matrices for common brain orientations.

    The returned dictionary maps :class:`whippersnappy.utils.types.ViewType`
    enum members to corresponding 4x4 view matrices (dtype float32) that
    can be used as camera/view transforms in the OpenGL renderer.

    Returns
    -------
    dict
        Mapping of :class:`ViewType` -> 4x4 numpy.ndarray view matrix.
    """
    view_left = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
    view_right = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
    view_back = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
    view_front = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
    view_bottom = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)
    view_top = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

    return {
        ViewType.LEFT: view_left,
        ViewType.RIGHT: view_right,
        ViewType.BACK: view_back,
        ViewType.FRONT: view_front,
        ViewType.TOP: view_top,
        ViewType.BOTTOM: view_bottom,
    }


def get_view_matrix(view_type):
    """Return the 4x4 view matrix for a single :class:`ViewType`.

    Parameters
    ----------
    view_type : ViewType
        Enum member indicating the requested view.

    Returns
    -------
    numpy.ndarray
        4x4 float32 view matrix.
    """
    return get_view_matrices()[view_type]
