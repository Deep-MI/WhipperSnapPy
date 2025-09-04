"""Contains the types used in WhipperSnapPy.

Dependencies:
    enum

@Author    : Abdulla Ahmadkhan
@Created   : 02.10.2025
@Revised   : 02.10.2025

"""
import enum


class ColorSelection(enum.Enum):
    BOTH = 1
    POSITIVE = 2
    NEGATIVE = 3

class OrientationType(enum.Enum):
    HORIZONTAL = 1
    VERTICAL = 2

class ViewType(enum.Enum):
    LEFT = 1
    RIGHT = 2
    BACK = 3
    FRONT = 4
    TOP = 5
    BOTTOM = 6