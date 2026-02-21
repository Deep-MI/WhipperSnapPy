"""View matrices, presets, and interactive view state under gl package."""

from dataclasses import dataclass, field

import numpy as np
import pyrr  # still needed for compute_view_matrix

from ..utils.types import ViewType

# ---------------------------------------------------------------------------
# ViewState — single source of truth for all mutable view parameters
# ---------------------------------------------------------------------------

@dataclass
class ViewState:
    """Mutable view parameters for the interactive GUI render loop.

    All mouse/keyboard interaction updates this object; the view matrix is
    recomputed from it each frame via :func:`compute_view_matrix`.

    Attributes
    ----------
    rotation : np.ndarray
        4×4 float32 rotation matrix (identity = no rotation applied).
    pan : np.ndarray
        (x, y) pan offset in normalised screen-space units.
    zoom : float
        Z-translation applied before the base view.  Larger = further away.
    last_mouse_pos : np.ndarray or None
        Last recorded mouse position (pixels); ``None`` when no button held.
    left_button_down, right_button_down, middle_button_down : bool
        Current pressed state of each mouse button.
    """
    rotation: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float32)
    )
    pan: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32)
    )
    zoom: float = 0.4
    last_mouse_pos: np.ndarray | None = None
    left_button_down: bool = False
    right_button_down: bool = False
    middle_button_down: bool = False


def compute_view_matrix(view_state: ViewState, base_view: np.ndarray) -> np.ndarray:
    """Return the ``transform`` uniform — exactly as snap_rotate does it.

    Packs ``transl * rotation * base_view`` into a single matrix, matching
    the snap_rotate convention (line: ``viewmat = transl * rot * base_view``).
    The ``model`` and ``view`` uniforms are left as set by ``setup_shader``
    (identity and camera respectively) and must not be overwritten.

    Parameters
    ----------
    view_state : ViewState
        Current interactive view state.
    base_view : np.ndarray
        Fixed 4×4 orientation preset from :func:`get_view_matrices`.

    Returns
    -------
    np.ndarray
        4×4 float32 matrix for the ``transform`` shader uniform.
    """
    transl = pyrr.Matrix44.from_translation((
        view_state.pan[0],
        view_state.pan[1],
        0.4 + view_state.zoom,
    ))
    rot = pyrr.Matrix44(view_state.rotation)
    return np.array(transl * rot * pyrr.Matrix44(base_view), dtype=np.float32)



# ---------------------------------------------------------------------------
# Arcball helpers
# ---------------------------------------------------------------------------

def arcball_vector(x: float, y: float, width: int, height: int) -> np.ndarray:
    """Map a 2-D screen pixel to a point on the unit arcball sphere.

    Normalises (x, y) to [-1, 1] NDC, then projects onto the unit sphere.
    Points outside the sphere radius are clamped to the rim (z = 0).

    Parameters
    ----------
    x, y : float
        Mouse position in pixels.
    width, height : int
        Window dimensions in pixels.

    Returns
    -------
    np.ndarray
        Unit 3-vector on (or clamped to) the arcball sphere.
    """
    s = min(width, height)
    p = np.array([
        (2.0 * x - width)  / s,
        -(2.0 * y - height) / s,
        0.0,
    ], dtype=np.float64)
    sq = p[0] ** 2 + p[1] ** 2
    if sq <= 1.0:
        p[2] = np.sqrt(1.0 - sq)
    else:
        p /= np.sqrt(sq)           # clamp to rim
    n = np.linalg.norm(p)
    return p / n if n > 0 else p


def arcball_rotation_matrix(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Return a 4×4 rotation matrix that rotates unit vector *v1* to *v2*.

    Uses Rodrigues' rotation formula in pure numpy — no pyrr dependency.
    Returns identity when *v1* and *v2* are coincident.

    Parameters
    ----------
    v1, v2 : np.ndarray
        Unit 3-vectors on the arcball sphere.

    Returns
    -------
    np.ndarray
        4×4 float32 rotation matrix compatible with pyrr.
    """
    axis = np.cross(v1, v2)
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-10:
        return np.eye(4, dtype=np.float32)

    axis = axis / axis_len
    angle = np.arctan2(axis_len, np.dot(v1, v2))

    # Rodrigues' formula: R = I cos(a) + sin(a) [axis]× + (1-cos(a)) axis⊗axis
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    x, y, z = axis
    r3 = np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ], dtype=np.float32)

    r4 = np.eye(4, dtype=np.float32)
    r4[:3, :3] = r3
    return r4


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
