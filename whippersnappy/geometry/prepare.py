"""Geometry helpers for mesh processing and GPU preparation (prepare.py).

This module contains the primary geometry-preparation pipeline.  The
low-level workhorse is :func:`prepare_geometry_from_arrays` which operates
entirely on numpy arrays.  :func:`prepare_geometry` is a thin file-loading
wrapper that delegates to the resolver functions in
:mod:`whippersnappy.geometry.inputs` before calling
:func:`prepare_geometry_from_arrays`.
"""

import warnings

import numpy as np

from ..utils.colormap import binary_color, heat_color, mask_sign, rescale_overlay
from ..utils.types import ColorSelection
from .inputs import resolve_annot, resolve_bg_map, resolve_mesh, resolve_overlay, resolve_roi


def normalize_mesh(v, scale=1.0):
    """Center and scale mesh vertex coordinates to a unit cube.

    The function recenters the vertices around the origin and scales them so
    that the maximum extent fits into a unit cube, optionally applying an
    additional scale factor.

    Parameters
    ----------
    v : numpy.ndarray
        Vertex coordinate array of shape (n_vertices, 3).
    scale : float, optional
        Additional multiplicative scale applied after normalization.

    Returns
    -------
    numpy.ndarray
        Normalized vertex coordinates with same shape as ``v``.
    """
    bbmax = np.max(v, axis=0)
    bbmin = np.min(v, axis=0)
    v = v - 0.5 * (bbmax + bbmin)
    v = scale * v / np.max(bbmax - bbmin)
    return v


def vertex_normals(v, t):
    """Compute per-vertex normals from triangle connectivity.

    Parameters
    ----------
    v : numpy.ndarray
        Vertex coordinates (n_vertices, 3).
    t : numpy.ndarray
        Triangle indices (n_faces, 3).

    Returns
    -------
    numpy.ndarray
        Per-vertex unit normals (n_vertices, 3).
    """
    v0 = v[t[:, 0], :]
    v1 = v[t[:, 1], :]
    v2 = v[t[:, 2], :]
    v1mv0 = v1 - v0
    v2mv1 = v2 - v1
    v0mv2 = v0 - v2
    cr0 = np.cross(v1mv0, -v0mv2)
    cr1 = np.cross(v2mv1, -v1mv0)
    cr2 = np.cross(v0mv2, -v2mv1)
    # Vectorized accumulation using bincount
    idx = np.concatenate([t[:, 0], t[:, 1], t[:, 2]])
    contribs = np.vstack([cr0, cr1, cr2])
    n = np.empty((v.shape[0], 3), dtype=np.float64)
    for j in range(3):
        n[:, j] = np.bincount(idx, weights=contribs[:, j], minlength=v.shape[0])
    ln = np.sqrt(np.sum(n * n, axis=1))
    ln[ln < np.finfo(float).eps] = 1
    n = n / ln.reshape(-1, 1)
    return n


def _estimate_thresholds_from_array(mapdata, minval=None, maxval=None):
    """Estimate threshold and saturation values from an already-loaded array.

    Parameters
    ----------
    mapdata : numpy.ndarray
        Per-vertex overlay values.
    minval : float or None, optional
        If provided, used as-is; otherwise estimated as the minimum absolute
        value in the data.
    maxval : float or None, optional
        If provided, used as-is; otherwise estimated as the maximum absolute
        value in the data.

    Returns
    -------
    minval : float
        Threshold value (lower bound of the color scale).
    maxval : float
        Saturation value (upper bound of the color scale).
    """
    valabs = np.abs(mapdata)
    if maxval is None:
        maxval = float(np.max(valabs)) if np.any(valabs) else 0.0
    if minval is None:
        minval = float(max(0.0, np.min(valabs) if np.any(valabs) else 0.0))
    return minval, maxval


def estimate_overlay_thresholds(overlay, minval=None, maxval=None):
    """Estimate threshold and saturation values from an overlay file or array.

    Reads the overlay data and derives ``fmin`` / ``fmax`` from the absolute
    values without performing any geometry or color work.  Both values are
    returned unchanged when they are already provided by the caller, making
    the function safe to call unconditionally.

    Parameters
    ----------
    overlay : str or array-like
        Path to the overlay file (.mgh or FreeSurfer morph format), or a
        numpy array / array-like of per-vertex scalar values.
    minval : float or None, optional
        If provided, used as-is for the threshold; otherwise estimated as
        the minimum absolute value in the overlay.
    maxval : float or None, optional
        If provided, used as-is for the saturation; otherwise estimated as
        the maximum absolute value in the overlay.

    Returns
    -------
    minval : float
        Threshold value (lower bound of the color scale).
    maxval : float
        Saturation value (upper bound of the color scale).
    """
    if isinstance(overlay, str):
        # Use resolve_overlay with n_vertices=None to skip shape validation
        overlay_arr = resolve_overlay(overlay, n_vertices=None)
    else:
        overlay_arr = np.asarray(overlay)
    return _estimate_thresholds_from_array(overlay_arr, minval, maxval)


def prepare_geometry_from_arrays(
    vertices,
    faces,
    overlay=None,
    annot=None,
    ctab=None,
    bg_map=None,
    roi=None,
    minval=None,
    maxval=None,
    invert=False,
    scale=1.85,
    color_mode=ColorSelection.BOTH,
):
    """Prepare vertex and color arrays for GPU upload from numpy arrays.

    This is the core geometry preparation function.  All inputs must already
    be resolved numpy arrays; for file-path support use the thin wrapper
    :func:`prepare_geometry`.

    Parameters
    ----------
    vertices : numpy.ndarray
        Vertex coordinate array of shape (N, 3), dtype float32.
    faces : numpy.ndarray
        Triangle index array of shape (M, 3), dtype uint32.
    overlay : numpy.ndarray or None, optional
        Per-vertex scalar values of shape (N,) float32 used for coloring.
    annot : numpy.ndarray or None, optional
        Per-vertex integer label indices of shape (N,) int32.
    ctab : numpy.ndarray or None, optional
        Color table array (n_labels, ≥3) associated with *annot*.
    bg_map : numpy.ndarray or None, optional
        Per-vertex scalar values of shape (N,) float32 whose sign determines
        background shading (binary light/dark).  When ``None`` a flat gray
        background is used.
    roi : numpy.ndarray of bool or None, optional
        Boolean mask of shape (N,).  ``True`` = vertex is inside the region
        of interest and receives overlay coloring; ``False`` = vertex falls
        back to background shading.  When ``None`` all vertices are in-ROI.
    minval, maxval : float or None, optional
        Threshold and saturation values for overlay scaling.
    invert : bool, optional, default False
        Invert color mapping.
    scale : float, optional, default 1.85
        Geometry scaling factor applied by :func:`normalize_mesh`.
    color_mode : ColorSelection, optional, default ColorSelection.BOTH
        Which sign(s) of overlay values to use for coloring.

    Returns
    -------
    vertexdata : numpy.ndarray
        Nx9 array (position x3, normal x3, color x3) ready for GPU upload.
    triangles : numpy.ndarray
        Mx3 uint32 triangle index array.
    fmin, fmax : float or None
        Final threshold and saturation values used for color mapping.
    pos, neg : bool or None
        Flags indicating whether positive/negative overlay values are present.

    Raises
    ------
    ValueError
        If overlay or annotation arrays do not match the surface vertex count.
    """
    vertices = normalize_mesh(np.array(vertices, dtype=np.float32), scale)
    triangles = np.array(faces, dtype=np.uint32)
    vnormals = np.array(vertex_normals(vertices, triangles), dtype=np.float32)
    num_vertices = vertices.shape[0]

    # Build background (sulcal) colormap
    if bg_map is not None:
        if bg_map.shape[0] != num_vertices:
            warnings.warn(
                f"bg_map has {bg_map.shape[0]} values but mesh has {num_vertices}.",
                stacklevel=2,
            )
            sulcmap = 0.5 * np.ones(vertices.shape, dtype=np.float32)
        else:
            sulcmap = binary_color(bg_map, 0.0, color_low=0.5, color_high=0.33)
    else:
        sulcmap = 0.5 * np.ones(vertices.shape, dtype=np.float32)

    # Initialize defaults for overlay outputs
    fmin = None
    fmax = None
    pos = None
    neg = None
    colors = sulcmap  # use as default

    # Apply overlay coloring
    if overlay is not None:
        if overlay.shape[0] != num_vertices:
            raise ValueError(
                f"overlay has {overlay.shape[0]} values but mesh has {num_vertices}.\n"
                "This usually means the overlay does not match the provided surface "
                "(e.g. RH overlay used with LH surface). Provide the correct overlay."
            )
        mapdata = overlay.copy().astype(np.float64)
        minval, maxval = _estimate_thresholds_from_array(mapdata, minval, maxval)
        mapdata = mask_sign(mapdata, color_mode)
        mapdata, fmin, fmax, pos, neg = rescale_overlay(mapdata, minval, maxval)
        colors = heat_color(mapdata, invert)
        # Some mapdata values could be nan (below min threshold) — fall back to bg
        missing = np.isnan(mapdata)
        if np.any(missing):
            colors[missing, :] = sulcmap[missing, :]

    elif annot is not None and ctab is not None:
        # Per-vertex annotation coloring
        if annot.shape[0] != num_vertices:
            raise ValueError(
                f"annot has {annot.shape[0]} values but mesh has {num_vertices}.\n"
                "This usually means the .annot does not match the provided surface "
                "(e.g. RH annot used with LH surface). Provide the correct annot file."
            )
        annot = annot.astype(np.int32)
        colors = np.array(sulcmap, dtype=np.float32)
        ctab_rgb = np.asarray(ctab[:, 0:3], dtype=np.float32)
        denom = 255.0 if np.max(ctab_rgb) > 1 else 1.0
        valid = (annot >= 0) & (annot < ctab.shape[0])
        if np.any(valid):
            colors[valid, :] = ctab_rgb[annot[valid], :] / denom

    # Ensure colors dtype matches vertices/normals
    colors = np.asarray(colors, dtype=np.float32)

    # Apply ROI mask: vertices where roi == False fall back to sulcmap
    if roi is not None:
        outside = ~roi
        if np.any(outside):
            colors[outside, :] = sulcmap[outside, :]

    vertexdata = np.concatenate((vertices, vnormals, colors), axis=1)
    return vertexdata, triangles, fmin, fmax, pos, neg


def prepare_geometry(
    mesh,
    overlay=None,
    annot=None,
    bg_map=None,
    roi=None,
    minval=None,
    maxval=None,
    invert=False,
    scale=1.85,
    color_mode=ColorSelection.BOTH,
):
    """Prepare vertex and color arrays for GPU upload.

    This is a thin file-loading wrapper around
    :func:`prepare_geometry_from_arrays`.  Inputs are resolved via the
    functions in :mod:`whippersnappy.geometry.inputs` so that every
    parameter can be either a file path or a numpy array.

    Parameters
    ----------
    mesh : str or tuple of (array-like, array-like)
        Surface file path (FreeSurfer format) **or** a ``(vertices, faces)``
        tuple/list where *vertices* is (N, 3) float and *faces* is (M, 3) int.
    overlay : str, array-like, or None, optional
        Path to an overlay (.mgh / FreeSurfer morph) file, or a (N,) array
        of per-vertex scalar values.
    annot : str, tuple, or None, optional
        Path to a FreeSurfer .annot file, or a ``(labels, ctab)`` /
        ``(labels, ctab, names)`` tuple.
    bg_map : str, array-like, or None, optional
        Path to a curvature/morph file used for background shading, or a
        (N,) array whose sign determines light/dark shading.
    roi : str, array-like, or None, optional
        Path to a FreeSurfer label file or a (N,) boolean array.  Vertices
        with ``True`` receive overlay coloring; others fall back to *bg_map*.
    minval, maxval : float or None, optional
        Threshold and saturation values for overlay scaling.
    invert : bool, optional, default False
        Invert color mapping.
    scale : float, optional, default 1.85
        Geometry scaling factor applied by :func:`normalize_mesh`.
    color_mode : ColorSelection, optional, default ColorSelection.BOTH
        Which sign(s) of overlay values to use for coloring.

    Returns
    -------
    vertexdata : numpy.ndarray
        Nx9 array (position x3, normal x3, color x3) ready for GPU upload.
    triangles : numpy.ndarray
        Mx3 uint32 triangle index array.
    fmin, fmax : float or None
        Final threshold and saturation values used for color mapping.
    pos, neg : bool or None
        Flags indicating whether positive/negative overlay values are present.

    Raises
    ------
    TypeError
        If *mesh* is not a valid type.
    ValueError
        If overlay or annotation arrays do not match the surface vertex count.

    Examples
    --------
    File-path usage::

        vdata, tris, fmin, fmax, pos, neg = prepare_geometry(
            'fsaverage/surf/lh.white',
            overlay='fsaverage/surf/lh.thickness',
            bg_map='fsaverage/surf/lh.curv',
            roi='fsaverage/label/lh.cortex.label',
        )

    Array inputs::

        import numpy as np
        v = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
        f = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.uint32)
        vdata, tris, *_ = prepare_geometry((v, f))
    """
    vertices, faces = resolve_mesh(mesh)
    n = vertices.shape[0]
    overlay_arr = resolve_overlay(overlay, n_vertices=n)
    bg_map_arr = resolve_bg_map(bg_map, n_vertices=n)
    roi_arr = resolve_roi(roi, n_vertices=n)
    annot_result = resolve_annot(annot, n_vertices=n)
    annot_arr = annot_result[0] if annot_result is not None else None
    ctab_arr = annot_result[1] if annot_result is not None else None
    return prepare_geometry_from_arrays(
        vertices, faces, overlay_arr, annot_arr, ctab_arr,
        bg_map_arr, roi_arr, minval, maxval, invert, scale, color_mode,
    )


def prepare_and_validate_geometry(
    mesh,
    overlay=None,
    annot=None,
    bg_map=None,
    roi=None,
    fthresh=None,
    fmax=None,
    invert=False,
    scale=1.85,
    color_mode=ColorSelection.BOTH,
):
    """Load and validate mesh geometry and overlay/annotation inputs.

    This is a small wrapper around :func:`prepare_geometry` that performs
    the same overlay-presence validation used throughout the static snapshot
    helpers.

    Parameters
    ----------
    mesh : str or tuple
        Passed through to :func:`prepare_geometry`.
    overlay, annot, bg_map, roi : str, array-like, or None
        Passed through to :func:`prepare_geometry`.
    fthresh, fmax : float or None
        Threshold and saturation values passed to the geometry preparer.
    invert : bool
        Passed to the geometry preparer.
    scale : float
        Scaling factor passed to the geometry preparer.
    color_mode : ColorSelection
        Which sign of overlay to display (POSITIVE/NEGATIVE/BOTH).

    Returns
    -------
    tuple
        ``(meshdata, triangles, fthresh, fmax, pos, neg)`` as returned by
        :func:`prepare_geometry`.

    Raises
    ------
    ValueError
        If the overlay contains no values appropriate for ``color_mode``.
    """
    import logging
    logger = logging.getLogger(__name__)
    meshdata, triangles, out_fthresh, out_fmax, pos, neg = prepare_geometry(
        mesh,
        overlay,
        annot,
        bg_map,
        roi,
        fthresh,
        fmax,
        invert,
        scale=scale,
        color_mode=color_mode,
    )

    # Validate overlay presence similar to previous inline checks
    if overlay is not None:
        if color_mode == ColorSelection.POSITIVE:
            if not pos and neg:
                logger.error("Overlay has no values to display with positive color_mode")
                raise ValueError("Overlay has no values to display with positive color_mode")
            neg = False
        elif color_mode == ColorSelection.NEGATIVE:
            if pos and not neg:
                logger.error("Overlay has no values to display with negative color_mode")
                raise ValueError("Overlay has no values to display with negative color_mode")
            pos = False
        if not pos and not neg:
            logger.error("Overlay has no values to display")
            raise ValueError("Overlay has no values to display")

    return meshdata, triangles, out_fthresh, out_fmax, pos, neg

