"""Colormap and value preprocessing utilities."""

import logging

import numpy as np

from .types import ColorSelection

# Module logger
logger = logging.getLogger(__name__)


def heat_color(values, invert=False):
    """Convert an array of float values into RGB heat color values.

    Maps scalar values to RGB triplets suitable for visualization. Input
    values are expected to be in a symmetric range around zero; mapping
    produces blue-to-red heat colors. NaN inputs propagate to NaN outputs.

    Parameters
    ----------
    values : array_like
        1-D array of float values to map. May include NaNs.
    invert : bool, optional
        If True, invert the sign of the input values before mapping.
        Default is False.

    Returns
    -------
    numpy.ndarray
        Array of shape (N, 3) and dtype float32 with RGB channels in [0, 1].
    """
    if invert:
        values = -1.0 * values
    vabs = np.abs(values)
    colors = np.zeros((vabs.size, 3), dtype=np.float32)
    crb = 0.5625 + 3 * 0.4375 * vabs
    cg = 1.5 * (vabs - (1.0 / 3.0))
    n1 = values < -1.0
    nm = (values >= -1.0) & (values < -(1.0 / 3.0))
    n0 = (values >= -(1.0 / 3.0)) & (values < 0)
    p0 = (values >= 0) & (values < (1.0 / 3.0))
    pm = (values >= (1.0 / 3.0)) & (values < 1.0)
    p1 = values >= 1.0
    colors[n1, 1:3] = 1.0
    colors[nm, 1] = cg[nm]
    colors[nm, 2] = 1.0
    colors[n0, 2] = crb[n0]
    colors[p0, 0] = crb[p0]
    colors[pm, 1] = cg[pm]
    colors[pm, 0] = 1.0
    colors[p1, 0:2] = 1.0
    colors[np.isnan(values), :] = np.nan
    return colors


def mask_sign(values, color_mode):
    """Mask values that don't match the requested sign selection.

    Parameters
    ----------
    values : array_like
        Input numeric array.
    color_mode : ColorSelection
        Enum indicating which sign to preserve (POSITIVE, NEGATIVE, BOTH).

    Returns
    -------
    numpy.ndarray
        Copy of ``values`` where elements not matching the requested sign
        are set to ``np.nan``.
    """
    masked_values = np.copy(values)
    if color_mode == ColorSelection.POSITIVE:
        masked_values[masked_values < 0] = np.nan
    elif color_mode == ColorSelection.NEGATIVE:
        masked_values[masked_values > 0] = np.nan
    return masked_values


def rescale_overlay(values, minval, maxval):
    """Rescale overlay values into a normalized range for colormap computation.

    Values whose absolute magnitude is below ``minval`` are set to ``NaN``.
    Remaining values are shifted by ``minval`` and divided by ``(maxval - minval)``.

    Parameters
    ----------
    values : numpy.ndarray
        Numeric array of overlay values (1-D).
    minval : float
        Minimum absolute threshold — values with abs < minval are treated as absent.
    maxval : float
        Maximum absolute value used for normalization.

    Returns
    -------
    tuple
        ``(values, minval, maxval, pos, neg)`` where ``values`` is the rescaled
        array, and ``pos``/``neg`` are booleans indicating presence of positive
        / negative values after rescaling.

    Raises
    ------
    ValueError
        If ``minval`` or ``maxval`` is negative.
    """
    valsign = np.sign(values)
    valabs = np.abs(values)

    if maxval < 0 or minval < 0:
        logger.error("rescale_overlay ERROR: min and maxval should both be positive!")
        raise ValueError("minval and maxval must be non-negative")

    values[valabs < minval] = np.nan
    range_val = maxval - minval
    if range_val == 0:
        values = np.zeros_like(values)
    else:
        values = values - valsign * minval
        values = values / range_val

    pos = np.any(values[~np.isnan(values)] > 0)
    neg = np.any(values[~np.isnan(values)] < 0)

    return values, minval, maxval, pos, neg


def binary_color(values, thres, color_low, color_high):
    """Create a binary colormap for values based on a threshold.

    Parameters
    ----------
    values : array_like
        1-D array of values to map.
    thres : float
        Threshold value used to split the colors.
    color_low, color_high : scalar or sequence
        Colors assigned to values below/above the threshold. Scalars are
        expanded to RGB triplets.

    Returns
    -------
    numpy.ndarray
        Array of shape (N, 3) and dtype float32 containing RGB colors.
    """
    if np.isscalar(color_low):
        color_low = np.array((color_low, color_low, color_low), dtype=np.float32)
    if np.isscalar(color_high):
        color_high = np.array((color_high, color_high, color_high), dtype=np.float32)
    colors = np.empty((values.size, 3), dtype=np.float32)
    colors[values < thres, :] = color_low
    colors[values >= thres, :] = color_high
    return colors


def mask_label(values, labelpath=None):
    """Apply a label file as a mask to an array of per-vertex values.

    If ``labelpath`` is provided the function loads vertex indices from the
    label file and sets all entries not listed in the label to ``NaN``.

    Parameters
    ----------
    values : numpy.ndarray
        1-D array indexed by vertex id.
    labelpath : str or None, optional
        Path to a label file readable by ``numpy.loadtxt`` (expected format
        with vertex ids in the first column after two header lines).

    Returns
    -------
    numpy.ndarray
        Array with vertices not included in the label set to ``np.nan``.
    """
    if not labelpath:
        return values
    maskvids = np.loadtxt(labelpath, dtype=int, skiprows=2, usecols=[0])
    imask = np.ones(values.shape, dtype=bool)
    imask[maskvids] = False
    values[imask] = np.nan
    return values

