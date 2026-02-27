"""Contains the types used in WhipperSnapPy.

This module defines small enumeration types used across the package for
controlling color selection, colorbar orientation, and predefined views.
It also provides the canonical per-view 4×4 matrices used by the renderers.

Classes
-------
ColorSelection
    Which sign(s) of overlay values should be used to produce colors.
OrientationType
    Orientation of UI elements such as the colorbar (horizontal or vertical).
ViewType
    Predefined canonical view orientations for rendering the brain surface.

Functions
---------
get_view_matrices
    Return a dict mapping every :class:`ViewType` to its 4×4 numpy matrix.
get_view_matrix
    Return the 4×4 numpy matrix for a single :class:`ViewType`.
"""

import enum

import numpy as np


class ColorSelection(enum.Enum):
    """Enum to select which sign(s) of overlay values to color.

    Parameters
    ----------
    *values : tuple
        Positional arguments passed to the Enum constructor (not used by
        consumers of this enum). Documented here to satisfy documentation
        linters that inspect the class signature.

    Attributes
    ----------
    BOTH : int
        Use both positive and negative values for coloring.
    POSITIVE : int
        Use only positive values for coloring.
    NEGATIVE : int
        Use only negative values for coloring.
    """
    BOTH = 1
    POSITIVE = 2
    NEGATIVE = 3


class OrientationType(enum.Enum):
    """Enum describing orientation choices for elements like the colorbar.

    Parameters
    ----------
    *values : tuple
        Positional arguments passed to the Enum constructor (not used by
        consumers of this enum).

    Attributes
    ----------
    HORIZONTAL : int
        Layout along the horizontal axis.
    VERTICAL : int
        Layout along the vertical axis.
    """
    HORIZONTAL = 1
    VERTICAL = 2


class ViewType(enum.Enum):
    """Predefined canonical view directions used by snapshot renderers.

    Parameters
    ----------
    *values : tuple
        Positional arguments passed to the Enum constructor (not used by
        consumers of this enum).

    Attributes
    ----------
    LEFT : int
        Left hemisphere lateral view.
    RIGHT : int
        Right hemisphere lateral view.
    BACK : int
        Posterior view.
    FRONT : int
        Anterior/frontal view.
    TOP : int
        Superior/top view.
    BOTTOM : int
        Inferior/bottom view.
    """
    LEFT = 1
    RIGHT = 2
    BACK = 3
    FRONT = 4
    TOP = 5
    BOTTOM = 6


def get_view_matrices() -> dict:
    """Return canonical 4×4 view matrices for every :class:`ViewType`.

    The matrices are pure numpy arrays (no OpenGL calls) and can be used
    as the ``base_view`` argument to any renderer.

    Returns
    -------
    dict
        Mapping of :class:`ViewType` → 4×4 float32 numpy.ndarray.
    """
    return {
        ViewType.LEFT:   np.array([[ 0, 0,-1, 0],[-1, 0, 0, 0],[ 0, 1, 0, 0],[0, 0, 0, 1]], dtype=np.float32),
        ViewType.RIGHT:  np.array([[ 0, 0, 1, 0],[ 1, 0, 0, 0],[ 0, 1, 0, 0],[0, 0, 0, 1]], dtype=np.float32),
        ViewType.BACK:   np.array([[ 1, 0, 0, 0],[ 0, 0,-1, 0],[ 0, 1, 0, 0],[0, 0, 0, 1]], dtype=np.float32),
        ViewType.FRONT:  np.array([[-1, 0, 0, 0],[ 0, 0, 1, 0],[ 0, 1, 0, 0],[0, 0, 0, 1]], dtype=np.float32),
        ViewType.TOP:    np.array([[ 1, 0, 0, 0],[ 0, 1, 0, 0],[ 0, 0, 1, 0],[0, 0, 0, 1]], dtype=np.float32),
        ViewType.BOTTOM: np.array([[-1, 0, 0, 0],[ 0, 1, 0, 0],[ 0, 0,-1, 0],[0, 0, 0, 1]], dtype=np.float32),
    }


def get_view_matrix(view_type: "ViewType") -> np.ndarray:
    """Return the 4×4 view matrix for a single :class:`ViewType`.

    Parameters
    ----------
    view_type : ViewType
        Enum member indicating the requested view.

    Returns
    -------
    numpy.ndarray
        4×4 float32 view matrix.
    """
    return get_view_matrices()[view_type]

