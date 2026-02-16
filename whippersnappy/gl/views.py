"""View matrices and presets under gl package."""

import numpy as np

from whippersnappy.utils.types import ViewType


def get_view_matrices():
    """Return canonical view matrices for left/right/front/back/top/bottom."""
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
    """Return a view matrix for a single view type."""
    return get_view_matrices()[view_type]
