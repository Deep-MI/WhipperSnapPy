"""Colormap and value preprocessing utilities."""

import numpy as np

from whippersnappy.utils.types import ColorSelection


def heat_color(values, invert=False):
    """Convert an array of float values into RBG heat color values."""
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
    """Mask values that don't have the same sign as color_mode."""
    masked_values = np.copy(values)
    if color_mode == ColorSelection.POSITIVE:
        masked_values[masked_values < 0] = np.nan
    elif color_mode == ColorSelection.NEGATIVE:
        masked_values[masked_values > 0] = np.nan
    return masked_values


def rescale_overlay(values, minval=None, maxval=None):
    """Rescale values for color map computation."""
    valsign = np.sign(values)
    valabs = np.abs(values)

    if maxval < 0 or minval < 0:
        print("rescale_overlay ERROR: min and maxval should both be positive!")
        exit(1)

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
    """Create a binary colormap based on a threshold value."""
    if np.isscalar(color_low):
        color_low = np.array((color_low, color_low, color_low), dtype=np.float32)
    if np.isscalar(color_high):
        color_high = np.array((color_high, color_high, color_high), dtype=np.float32)
    colors = np.empty((values.size, 3), dtype=np.float32)
    colors[values < thres, :] = color_low
    colors[values >= thres, :] = color_high
    return colors


def mask_label(values, labelpath=None):
    """Apply a label file as a mask."""
    if not labelpath:
        return values
    maskvids = np.loadtxt(labelpath, dtype=int, skiprows=2, usecols=[0])
    imask = np.ones(values.shape, dtype=bool)
    imask[maskvids] = False
    values[imask] = np.nan
    return values

