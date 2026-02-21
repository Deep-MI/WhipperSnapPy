"""Lightweight per-vertex scalar and label readers for common open formats.

This module implements pure-Python (stdlib + numpy only) readers for simple
per-vertex data files, plus a GIfTI reader that reuses the nibabel dependency
already present in the project.

Supported formats
-----------------
* **ASCII text** (``.txt``, ``.csv``) — one numeric value per line; optional
  single non-numeric header line (skipped automatically); whitespace or
  comma-separated.  Integer values are loaded as ``int32``; all others as
  ``float32``.

* **NumPy array** (``.npy``) — single 1-D array saved with
  ``numpy.save``.  Any numeric dtype is accepted and kept as-is; callers
  cast to the required dtype.

* **NumPy archive** (``.npz``) — multi-array archive saved with
  ``numpy.savez``.  The array named ``"data"`` is used if present,
  otherwise the first array in the archive is used.

* **GIfTI functional / label** (``.func.gii``, ``.label.gii``, ``.gii``) —
  loaded via ``nibabel``; the first data array is returned.  Covers HCP,
  fMRIPrep, and Connectome Workbench outputs.

The public dispatcher :func:`read_overlay` routes by file extension.
FreeSurfer binary morph files and MGH/MGZ files are *not* handled here —
they are loaded by :mod:`whippersnappy.geometry.freesurfer_io` and
dispatched from :func:`whippersnappy.geometry.inputs._load_overlay_from_file`.

All readers return a flat ``numpy.ndarray`` of shape ``(N,)``.  The caller
is responsible for casting to the desired dtype (``float32`` for overlays and
background maps, ``bool`` for ROI masks, ``int32`` for label/parcellation
maps).
"""

import os

import numpy as np

# ---------------------------------------------------------------------------
# ASCII text / CSV reader
# ---------------------------------------------------------------------------

def read_txt(path):
    """Read a per-vertex scalar file in plain ASCII format.

    The file must contain exactly one numeric value per line.  An optional
    single-line text header (non-numeric first line) is silently skipped.
    Whitespace and comma separators are both accepted; only the *first*
    value on each line is used (allowing simple CSV files with a single
    data column).

    Integer-valued files (every value equal to its ``int`` cast) are
    returned as ``int32``; all others as ``float32``.

    Parameters
    ----------
    path : str
        Path to the ``.txt`` or ``.csv`` file.

    Returns
    -------
    numpy.ndarray, shape (N,), dtype float32 or int32

    Raises
    ------
    ValueError
        If no numeric values can be parsed from the file.
    IOError
        If the file cannot be opened.

    Examples
    --------
    A valid ``overlay.txt``::

        # optional comment line (skipped)
        0.123
        -1.456
        2.0

    A valid ``labels.csv`` (first column used, header skipped)::

        label
        3
        0
        1
        3
    """
    values = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # Take only the first token (handles CSV with a single data column)
            token = line.split(",")[0].split()[0]
            try:
                values.append(float(token))
            except ValueError as exc:
                if lineno == 1:
                    # Treat the very first non-numeric line as a header and skip it
                    continue
                raise ValueError(
                    f"Could not parse numeric value on line {lineno} of {path!r}: "
                    f"{raw.strip()!r}"
                ) from exc

    if not values:
        raise ValueError(f"No numeric values found in {path!r}.")

    arr = np.array(values, dtype=np.float32)

    # Promote to int32 if all values are integers (label / parcellation file)
    if np.all(arr == arr.astype(np.int32)):
        return arr.astype(np.int32)
    return arr


# ---------------------------------------------------------------------------
# NumPy readers
# ---------------------------------------------------------------------------

def read_npy(path):
    """Read a per-vertex scalar array from a NumPy ``.npy`` file.

    Parameters
    ----------
    path : str
        Path to the ``.npy`` file.

    Returns
    -------
    numpy.ndarray, shape (N,)
        The stored array, squeezed to 1-D.

    Raises
    ------
    ValueError
        If the stored array is not 1-D after squeezing, or is empty.
    IOError
        If the file cannot be opened.
    """
    arr = np.load(path)
    arr = np.squeeze(arr)
    if arr.ndim != 1:
        raise ValueError(
            f"NumPy file {path!r} contains an array of shape {arr.shape}; "
            f"expected a 1-D per-vertex array."
        )
    if arr.size == 0:
        raise ValueError(f"NumPy file {path!r} contains an empty array.")
    return arr


def read_npz(path):
    """Read a per-vertex scalar array from a NumPy ``.npz`` archive.

    The array named ``"data"`` is returned if it exists; otherwise the
    first array in the archive is used.

    Parameters
    ----------
    path : str
        Path to the ``.npz`` file.

    Returns
    -------
    numpy.ndarray, shape (N,)
        The selected array, squeezed to 1-D.

    Raises
    ------
    ValueError
        If no arrays are found, or the selected array is not 1-D after
        squeezing.
    IOError
        If the file cannot be opened.
    """
    archive = np.load(path)
    keys = list(archive.keys())
    if not keys:
        raise ValueError(f"NumPy archive {path!r} contains no arrays.")

    key = "data" if "data" in keys else keys[0]
    arr = np.squeeze(archive[key])
    if arr.ndim != 1:
        raise ValueError(
            f"NumPy archive {path!r}, array {key!r} has shape {arr.shape}; "
            f"expected a 1-D per-vertex array."
        )
    if arr.size == 0:
        raise ValueError(f"NumPy archive {path!r}, array {key!r} is empty.")
    return arr


# ---------------------------------------------------------------------------
# GIfTI reader
# ---------------------------------------------------------------------------

def read_gifti(path):
    """Read a per-vertex scalar array from a GIfTI functional or label file.

    Supports ``.func.gii`` (continuous scalars, e.g. HCP thickness) and
    ``.label.gii`` (integer parcellation labels, e.g. HCP parcellation).
    Plain ``.gii`` files are also accepted provided they contain a scalar
    data array — **not** a surface geometry file.  For surface GIfTI files
    (``.surf.gii`` or ``.gii`` files with POINTSET+TRIANGLE arrays) use
    :func:`whippersnappy.geometry.mesh_io.read_gifti_surface` or pass the
    path to :func:`whippersnappy.geometry.inputs.resolve_mesh`.

    The first non-POINTSET, non-TRIANGLE data array in the file is returned.

    Parameters
    ----------
    path : str
        Path to a GIfTI file.

    Returns
    -------
    numpy.ndarray, shape (N,)
        The first scalar data array, squeezed to 1-D.

    Raises
    ------
    ImportError
        If ``nibabel`` is not installed.
    ValueError
        If the file is a surface GIfTI (POINTSET+TRIANGLE only), contains
        no usable scalar arrays, or the first scalar array is not 1-D.
    IOError
        If the file cannot be opened or is not a valid GIfTI file.
    """
    try:
        import nibabel as nib  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Reading GIfTI files requires nibabel. "
            "Install with: pip install nibabel"
        ) from exc

    img = nib.load(path)
    if not hasattr(img, "darrays") or not img.darrays:
        raise ValueError(
            f"GIfTI file {path!r} contains no data arrays."
        )

    # Intent codes for surface geometry — skip these
    _SURFACE_INTENTS = {1008, 1009}   # POINTSET, TRIANGLE

    scalar_da = None
    has_surface_arrays = False
    for da in img.darrays:
        if da.intent in _SURFACE_INTENTS:
            has_surface_arrays = True
        elif scalar_da is None:
            scalar_da = da

    if scalar_da is None:
        if has_surface_arrays:
            raise ValueError(
                f"GIfTI file {path!r} appears to be a surface geometry file "
                f"(contains only POINTSET/TRIANGLE arrays).  "
                f"Use resolve_mesh() or read_gifti_surface() to load it as a mesh."
            )
        raise ValueError(
            f"GIfTI file {path!r} contains no scalar data arrays."
        )

    arr = np.squeeze(scalar_da.data)
    if arr.ndim != 1:
        raise ValueError(
            f"GIfTI file {path!r}: first scalar data array has shape "
            f"{scalar_da.data.shape}; expected a 1-D per-vertex array."
        )
    if arr.size == 0:
        raise ValueError(f"GIfTI file {path!r}: first scalar data array is empty.")
    return arr


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

# Map from lower-case file extension to reader function.
# Note: ".func.gii" and ".label.gii" have a compound extension; we handle
# them by matching the last *two* dot-separated components as well.
_READERS = {
    ".txt":       read_txt,
    ".csv":       read_txt,
    ".npy":       read_npy,
    ".npz":       read_npz,
    ".gii":       read_gifti,
    ".func.gii":  read_gifti,
    ".label.gii": read_gifti,
}

_SUPPORTED = ", ".join(sorted(_READERS))


def read_overlay(path):
    """Read a per-vertex scalar or label array from a file.

    Dispatches to the appropriate reader based on the file extension.
    FreeSurfer binary morph files (e.g. ``lh.curv``, ``lh.thickness``) and
    MGH/MGZ files are **not** handled here — pass them through
    :func:`whippersnappy.geometry.inputs._load_overlay_from_file` which
    already routes those formats via :mod:`~whippersnappy.geometry.freesurfer_io`.

    Parameters
    ----------
    path : str
        Path to an overlay/label file.  Recognised extensions:

        * ``.txt``, ``.csv`` — plain ASCII, one value per line
        * ``.npy`` — NumPy binary array
        * ``.npz`` — NumPy archive (key ``"data"`` or first array)
        * ``.gii``, ``.func.gii``, ``.label.gii`` — GIfTI

    Returns
    -------
    numpy.ndarray, shape (N,)

    Raises
    ------
    ValueError
        If the file extension is not recognised.
    """
    # Check compound extensions first (.func.gii, .label.gii)
    lower = path.lower()
    for compound in (".func.gii", ".label.gii"):
        if lower.endswith(compound):
            return _READERS[compound](path)

    ext = os.path.splitext(path)[1].lower()
    reader = _READERS.get(ext)
    if reader is None:
        raise ValueError(
            f"Unsupported overlay file extension {ext!r} for {path!r}.  "
            f"Supported formats: {_SUPPORTED}.  "
            f"For FreeSurfer morph files (no extension) or .mgh/.mgz files "
            f"the routing is handled automatically by resolve_overlay()."
        )
    return reader(path)

