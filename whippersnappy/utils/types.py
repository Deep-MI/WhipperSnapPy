"""Contains the types used in WhipperSnapPy.

This module defines small enumeration types used across the package for
controlling color selection, colorbar orientation, and predefined views.

Classes
-------
ColorSelection
    Which sign(s) of overlay values should be used to produce colors.
OrientationType
    Orientation of UI elements such as the colorbar (horizontal or vertical).
ViewType
    Predefined canonical view orientations for rendering the brain surface.

Examples
--------
>>> from whippersnappy.utils.types import ColorSelection, ViewType
>>> ColorSelection.BOTH
<ColorSelection.BOTH: 1>
>>> ViewType.LEFT
<ViewType.LEFT: 1>
"""

import enum


class ColorSelection(enum.Enum):
    """Enum to select which sign(s) of overlay values to color.

    Members
    -------
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

    Members
    -------
    HORIZONTAL : int
        Layout along the horizontal axis.
    VERTICAL : int
        Layout along the vertical axis.
    """
    HORIZONTAL = 1
    VERTICAL = 2


class ViewType(enum.Enum):
    """Predefined canonical view directions used by snapshot renderers.

    Members
    -------
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

