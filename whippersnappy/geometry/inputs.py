"""Input resolver functions for WhipperSnapPy geometry loading.

This module is the single source of truth for loading and validating all
user-facing inputs (mesh, overlay, background map, ROI, annotation).  No
other module should call ``read_geometry``, ``read_morph_data``,
``read_mgh_data``, ``read_annot_data``, or ``mask_label`` directly — all
calls should go through the resolver functions defined here.
"""

import os

import numpy as np

from ..utils.colormap import mask_label
from .read_geometry import read_annot_data, read_geometry, read_mgh_data, read_morph_data


def resolve_mesh(mesh):
    """Resolve a mesh input to ``(vertices, faces)`` numpy arrays.

    Parameters
    ----------
    mesh : str or tuple/list of two array-likes
        Either a file path to a FreeSurfer-format surface file, or a
        two-element sequence whose first element is the vertex coordinate
        array and whose second element is the face index array.

    Returns
    -------
    vertices : numpy.ndarray
        Vertex coordinate array of shape (N, 3), dtype float32.
    faces : numpy.ndarray
        Triangle face index array of shape (M, 3), dtype uint32.

    Raises
    ------
    TypeError
        If *mesh* is neither a ``str`` nor a two-element tuple/list.
    ValueError
        If the resulting arrays do not have the expected shapes.
    """
    if isinstance(mesh, str):
        v_raw, f_raw = read_geometry(mesh, read_metadata=False)
        vertices = np.asarray(v_raw, dtype=np.float32)
        faces = np.asarray(f_raw, dtype=np.uint32)
    elif isinstance(mesh, (tuple, list)) and len(mesh) == 2:
        vertices = np.asarray(mesh[0], dtype=np.float32)
        faces = np.asarray(mesh[1], dtype=np.uint32)
    else:
        raise TypeError(
            f"mesh must be a file path (str) or a (vertices, faces) tuple/list, "
            f"got {type(mesh).__name__!r}."
        )

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            f"vertices must be an array of shape (N, 3), got shape {vertices.shape}."
        )
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"faces must be an array of shape (M, 3), got shape {faces.shape}."
        )
    return vertices, faces


def _load_overlay_from_file(path):
    """Load a 1-D per-vertex overlay array from a file path."""
    _, ext = os.path.splitext(path)
    if ext == ".mgh":
        return read_mgh_data(path)
    return read_morph_data(path)


def resolve_overlay(overlay, *, n_vertices):
    """Resolve an overlay input to a 1-D float32 numpy array, or ``None``.

    Parameters
    ----------
    overlay : None, str, or array-like
        * ``None`` — no overlay; returns ``None``.
        * ``str`` — path to an overlay file (.mgh or FreeSurfer morph format).
        * array-like — converted to ``np.float32``; must have shape
          ``(n_vertices,)`` when *n_vertices* is not ``None``.
    n_vertices : int or None
        Expected number of vertices.  Shape validation is skipped when
        ``None`` (useful for ``estimate_overlay_thresholds``).

    Returns
    -------
    numpy.ndarray of shape (n_vertices,) or None

    Raises
    ------
    ValueError
        If the loaded/converted array does not match *n_vertices*.
    """
    if overlay is None:
        return None
    if isinstance(overlay, str):
        arr = _load_overlay_from_file(overlay).astype(np.float32)
    else:
        arr = np.asarray(overlay, dtype=np.float32)
    if n_vertices is not None and arr.shape != (n_vertices,):
        raise ValueError(
            f"overlay has shape {arr.shape} but mesh has {n_vertices} vertices."
        )
    return arr


def resolve_bg_map(bg_map, *, n_vertices):
    """Resolve a background-map input to a 1-D float32 numpy array, or ``None``.

    Identical logic to :func:`resolve_overlay`.

    Parameters
    ----------
    bg_map : None, str, or array-like
        Background shading data (typically curvature).
    n_vertices : int or None
        Expected number of vertices for shape validation.

    Returns
    -------
    numpy.ndarray of shape (n_vertices,) or None
    """
    if bg_map is None:
        return None
    if isinstance(bg_map, str):
        arr = _load_overlay_from_file(bg_map).astype(np.float32)
    else:
        arr = np.asarray(bg_map, dtype=np.float32)
    if n_vertices is not None and arr.shape != (n_vertices,):
        raise ValueError(
            f"bg_map has shape {arr.shape} but mesh has {n_vertices} vertices."
        )
    return arr


def resolve_roi(roi, *, n_vertices):
    """Resolve a region-of-interest input to a boolean numpy array, or ``None``.

    The returned boolean array has ``True`` for vertices that are *included*
    in the overlay coloring and ``False`` for vertices that fall back to
    background (``bg_map``) shading.

    Parameters
    ----------
    roi : None, str, or array-like
        * ``None`` — no masking; returns ``None``.
        * ``str`` — path to a FreeSurfer label file.  Vertices listed in the
          file are marked ``True``; all others are ``False``.
        * array-like — converted to ``np.bool_``; must have shape
          ``(n_vertices,)``.
    n_vertices : int
        Expected number of vertices.

    Returns
    -------
    numpy.ndarray of shape (n_vertices,) bool, or None

    Raises
    ------
    ValueError
        If the resolved array does not match *n_vertices*.
    """
    if roi is None:
        return None
    if isinstance(roi, str):
        # Use mask_label to get vertices included in the label (NaN = excluded).
        sentinel = np.ones(n_vertices, dtype=np.float32)
        masked = mask_label(sentinel, roi)
        # Vertices NOT in the label were set to NaN → roi = ~isnan
        arr = ~np.isnan(masked)
    else:
        arr = np.asarray(roi, dtype=bool)
    if arr.shape != (n_vertices,):
        raise ValueError(
            f"roi has shape {arr.shape} but mesh has {n_vertices} vertices."
        )
    return arr


def resolve_annot(annot, *, n_vertices):
    """Resolve an annotation input to ``(labels, ctab, names)`` or ``None``.

    Parameters
    ----------
    annot : None, str, or tuple of length 2 or 3
        * ``None`` — no annotation; returns ``None``.
        * ``str`` — path to a FreeSurfer .annot file; loaded via
          :func:`read_annot_data`.
        * 2-tuple ``(labels, ctab)`` — validated and returned as
          ``(labels, ctab, None)``.
        * 3-tuple ``(labels, ctab, names)`` — validated and returned as-is.
    n_vertices : int
        Expected number of vertices for shape validation of *labels*.

    Returns
    -------
    tuple of (labels, ctab, names) or None
        * *labels* — integer array of shape (n_vertices,).
        * *ctab* — color table array of shape (n_labels, ≥3).
        * *names* — list of label names, or ``None``.

    Raises
    ------
    TypeError
        If *annot* is not one of the accepted types.
    ValueError
        If *labels* does not match *n_vertices*.
    """
    if annot is None:
        return None
    if isinstance(annot, str):
        labels, ctab, names = read_annot_data(annot)
    elif isinstance(annot, (tuple, list)) and len(annot) == 2:
        labels = np.asarray(annot[0])
        ctab = np.asarray(annot[1])
        names = None
    elif isinstance(annot, (tuple, list)) and len(annot) == 3:
        labels = np.asarray(annot[0])
        ctab = np.asarray(annot[1])
        names = annot[2]
    else:
        raise TypeError(
            f"annot must be a file path (str), a (labels, ctab) tuple, "
            f"or a (labels, ctab, names) tuple; got {type(annot).__name__!r}."
        )
    if labels.shape != (n_vertices,):
        raise ValueError(
            f"annot labels have shape {labels.shape} but mesh has {n_vertices} vertices."
        )
    return labels, ctab, names

