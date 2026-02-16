"""Geometry helpers for mesh processing and GPU preparation (prepare.py).

This module contains the primary `prepare_geometry` function used to
normalize meshes, compute normals and assemble vertex arrays for OpenGL.
"""

import os
import warnings

import numpy as np

from whippersnappy.geometry.read_geometry import read_annot_data, read_geometry, read_mgh_data, read_morph_data
from whippersnappy.utils.colormap import binary_color, heat_color, mask_label, mask_sign, rescale_overlay
from whippersnappy.utils.types import ColorSelection


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
    n = np.zeros(v.shape)
    np.add.at(n, t[:, 0], cr0)
    np.add.at(n, t[:, 1], cr1)
    np.add.at(n, t[:, 2], cr2)
    ln = np.sqrt(np.sum(n * n, axis=1))
    ln[ln < np.finfo(float).eps] = 1
    n = n / ln.reshape(-1, 1)
    return n


def prepare_geometry(
    surfpath,
    overlaypath=None,
    annotpath=None,
    curvpath=None,
    labelpath=None,
    minval=None,
    maxval=None,
    invert=False,
    scale=1.85,
    color_mode=ColorSelection.BOTH,
):
    """Prepare vertex and color arrays for GPU upload.

    This function loads a surface geometry from ``surfpath``, optionally
    loads an overlay (mgh/curv) or annotation (.annot) and produces an
    interleaved vertex array containing positions, normals and colors
    suitable for uploading to OpenGL (vertex buffer objects).

    Parameters
    ----------
    surfpath : str
        Path to the surface file.
    overlaypath : str or None, optional
        Path to an overlay (mgh/curv) file providing per-vertex scalar
        values used for coloring.
    annotpath : str or None, optional
        Path to a FreeSurfer .annot file for categorical labeling.
    curvpath : str or None, optional
        Path to curvature data used as fallback texture.
    labelpath : str or None, optional
        Path to a label file used to mask vertices.
    minval, maxval : float or None, optional
        Threshold and saturation values for overlay scaling.
    invert : bool, optional, default False
        Invert color mapping.
    scale : float, optional, default 1.85
        Geometry scaling factor applied by ``normalize_mesh``.
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
    surf = read_geometry(surfpath, read_metadata=False)
    vertices = normalize_mesh(np.array(surf[0], dtype=np.float32), scale)
    triangles = np.array(surf[1], dtype=np.uint32)
    vnormals = np.array(vertex_normals(vertices, triangles), dtype=np.float32)
    num_vertices = vertices.shape[0]

    # try to load sulcal colormap
    sulcmap = 0.5 * np.ones(vertices.shape, dtype=np.float32)
    if curvpath:
        curv = read_morph_data(curvpath)
        if curv.shape[0] != num_vertices:
            warnings.warn(f"Curvature file {curvpath} has {curv.shape[0]} values, but mesh has {num_vertices}.",
                          stacklevel=2)
        else:
            sulcmap = binary_color(curv, 0.0, color_low=0.5, color_high=0.33)

    # Initialize defaults for overlay outputs
    fmin = None
    fmax = None
    pos = None
    neg = None
    colors = sulcmap # use as default

    # try to load overlay data
    if overlaypath:
        _, file_extension = os.path.splitext(overlaypath)
        if file_extension == ".mgh":
            mapdata = read_mgh_data(overlaypath)
        else:
            mapdata = read_morph_data(overlaypath)

        # Check if overlay length matches number of vertices. If not, raise an error.
        if mapdata.shape[0] != num_vertices:
            raise ValueError(
                f"Overlay file {overlaypath} has {mapdata.shape[0]} values but mesh has {num_vertices}.\n"
                "This usually means the overlay does not match the provided surface "
                "(e.g. RH overlay used with LH surface). Provide the correct overlay "
                "file."
            )
        else:
            valabs = np.abs(mapdata)
            if maxval is None:
                maxval = np.max(valabs) if np.any(valabs) else 0
            if minval is None:
                minval = max(0.0, np.min(valabs) if np.any(valabs) else 0)

            mapdata = mask_sign(mapdata, color_mode)
            mapdata, fmin, fmax, pos, neg = rescale_overlay(mapdata, minval, maxval)
            colors = heat_color(mapdata, invert)
            # some mapdata values could be nan (below min threshold)
            missing = np.isnan(mapdata)
            if np.any(missing):
                colors[missing, :] = sulcmap[missing, :]
    # alternatively try to load annotation data
    elif annotpath:
        # Read annotation (per-vertex labels) and colormap table.
        annot, ctab, _ = read_annot_data(annotpath)

        # Check if annotation length matches number of vertices. If not, raise an error.
        if annot.shape[0] != num_vertices:
            raise ValueError(
                f"Annotation file {annotpath} has {annot.shape[0]} values but mesh has {num_vertices}.\n"
                "This usually means the .annot does not match the provided surface "
                "(e.g. RH annot used with LH surface). Provide the correct annot "
                "file."
            )
        else:
            # If annot is shorter, pad with -1 (meaning 'no label') to match
            # mesh vertices.
            if annot.shape[0] < num_vertices:
                pad_len = num_vertices - annot.shape[0]
                annot = np.pad(annot, (0, pad_len), mode="constant", constant_values=-1)

            # Ensure integer type for safe indexing
            annot = annot.astype(np.int32)

            # Start with sulcmap as the default and only overwrite valid label indices
            colors = np.array(sulcmap, dtype=np.float32)

            # Normalize colortable: detect whether ctab is 0-255 or 0-1
            ctab_rgb = ctab[:, 0:3].astype(np.float32)
            denom = 255.0 if np.max(ctab_rgb) > 1 else 1.0

            # Only assign colors for valid annotation indices (>=0 and within the color table)
            valid = (annot >= 0) & (annot < ctab.shape[0])
            if np.any(valid):
                colors[valid, :] = ctab_rgb[annot[valid], :] / denom

    # Ensure colors dtype matches vertices/normals
    colors = np.asarray(colors, dtype=np.float32)

    # Apply label mask to colors if labelpath is provided,
    # regardless of whether overlay or annot data was loaded
    if labelpath:
        mask = np.isnan(mask_label(np.ones(num_vertices), labelpath))
        if np.any(mask):
            colors[mask, :] = sulcmap[mask, :]

    vertexdata = np.concatenate((vertices, vnormals, colors), axis=1)
    return vertexdata, triangles, fmin, fmax, pos, neg
