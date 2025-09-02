"""Contains the types used in WhipperSnapPy.

Dependencies:
    enum

@Author    : Abdulla Ahmadkhan
@Created   : 02.10.2025
@Revised   : 02.10.2025

"""

# Standard library imports
from enum import Enum

class ViewType(Enum):
    LEFT = 1
    RIGHT = 2
    BACK = 3
    FRONT = 4
    TOP = 5
    BOTTOM = 6

class OrientationType(Enum):
    HORIZONTAL = 1
    VERTICAL = 2