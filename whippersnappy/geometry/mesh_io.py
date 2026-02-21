"""Lightweight ASCII mesh readers for common open formats.

This module implements pure-Python (stdlib + numpy only) readers for:

* **OFF** — Object File Format, ASCII triangles
* **VTK legacy ASCII PolyData** — ``DATASET POLYDATA`` with POINTS/POLYGONS
* **PLY ASCII** — Stanford PLY, ASCII encoding, triangles only

All readers return ``(vertices, faces)`` where

* ``vertices`` — ``float32`` array of shape ``(N, 3)``
* ``faces``    — ``uint32``  array of shape ``(M, 3)``

The public dispatcher :func:`read_mesh` routes by file extension
(``.off``, ``.vtk``, ``.ply``).  For FreeSurfer surfaces (no standard
extension) use the existing :func:`whippersnappy.geometry.read_geometry`
directly, or go through :func:`whippersnappy.geometry.inputs.resolve_mesh`
which handles the routing automatically.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _non_empty_lines(path):
    """Yield stripped, non-empty, non-comment lines from a text file."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if line and not line.startswith("#"):
                yield line


# ---------------------------------------------------------------------------
# OFF reader
# ---------------------------------------------------------------------------

def read_off(path):
    """Read an ASCII OFF (Object File Format) triangle mesh.

    Parameters
    ----------
    path : str
        Path to the ``.off`` file.

    Returns
    -------
    vertices : numpy.ndarray, shape (N, 3), dtype float32
    faces : numpy.ndarray, shape (M, 3), dtype uint32

    Raises
    ------
    ValueError
        If the file does not start with ``OFF``, if any face is not a
        triangle, or if the declared counts don't match the data.
    IOError
        If the file cannot be opened.

    Notes
    -----
    Comments (lines starting with ``#``) and blank lines are ignored
    anywhere in the file.  The optional ``COFF`` / ``NOFF`` / ``CNOFF``
    variants are *not* supported; only plain ``OFF`` is accepted.
    """
    lines = list(_non_empty_lines(path))
    if not lines:
        raise ValueError(f"OFF file is empty: {path!r}")

    # First line must be exactly "OFF"
    header = lines[0].upper()
    if header != "OFF":
        raise ValueError(
            f"Expected 'OFF' header on first non-comment line, got {lines[0]!r} "
            f"in {path!r}.  Only plain ASCII OFF is supported."
        )

    if len(lines) < 2:
        raise ValueError(f"OFF file has no count line after header: {path!r}")

    # Second line: n_vertices n_faces n_edges
    parts = lines[1].split()
    if len(parts) < 2:
        raise ValueError(
            f"OFF count line must have at least 2 integers (n_vertices n_faces), "
            f"got {lines[1]!r} in {path!r}."
        )
    try:
        n_verts = int(parts[0])
        n_faces = int(parts[1])
    except ValueError as exc:
        raise ValueError(
            f"Could not parse OFF count line {lines[1]!r} in {path!r}."
        ) from exc

    # Validate we have enough data lines
    data_lines = lines[2:]
    if len(data_lines) < n_verts + n_faces:
        raise ValueError(
            f"OFF file declares {n_verts} vertices and {n_faces} faces "
            f"but only {len(data_lines)} data lines follow in {path!r}."
        )

    # Parse vertices
    vertices = np.empty((n_verts, 3), dtype=np.float32)
    for i in range(n_verts):
        try:
            coords = data_lines[i].split()[:3]
            vertices[i] = [float(c) for c in coords]
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"Could not parse vertex {i} in OFF file {path!r}: "
                f"{data_lines[i]!r}"
            ) from exc

    # Parse faces
    faces = np.empty((n_faces, 3), dtype=np.uint32)
    for j in range(n_faces):
        try:
            tokens = data_lines[n_verts + j].split()
            n_poly = int(tokens[0])
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"Could not parse face {j} in OFF file {path!r}: "
                f"{data_lines[n_verts + j]!r}"
            ) from exc
        if n_poly != 3:
            raise ValueError(
                f"OFF face {j} has {n_poly} vertices; only triangles (3) are "
                f"supported in {path!r}.  Convert to a triangle mesh first."
            )
        try:
            faces[j] = [int(tokens[k]) for k in range(1, 4)]
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"Could not parse face indices at face {j} in {path!r}: "
                f"{data_lines[n_verts + j]!r}"
            ) from exc

    # Bounds check
    if n_faces > 0:
        if int(faces.max()) >= n_verts or int(faces.min()) < 0:
            raise ValueError(
                f"OFF face indices out of range [0, {n_verts}) in {path!r}."
            )

    return vertices, faces


# ---------------------------------------------------------------------------
# Legacy VTK ASCII PolyData reader
# ---------------------------------------------------------------------------

def read_vtk_ascii_polydata(path):
    """Read a legacy ASCII VTK PolyData triangle mesh.

    Only the *ASCII legacy format* with ``DATASET POLYDATA`` is supported.
    Binary VTK files are explicitly rejected.

    Parameters
    ----------
    path : str
        Path to the ``.vtk`` file.

    Returns
    -------
    vertices : numpy.ndarray, shape (N, 3), dtype float32
    faces : numpy.ndarray, shape (M, 3), dtype uint32

    Raises
    ------
    ValueError
        If the file is binary, is not POLYDATA, contains non-triangle
        polygons, or if required sections are missing.
    IOError
        If the file cannot be opened.
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        raw_lines = fh.readlines()

    # Legacy VTK format:
    #   line 0: "# vtk DataFile Version x.x"
    #   line 1: title (arbitrary free text)
    #   line 2: "ASCII" or "BINARY"
    #   line 3: "DATASET <type>"
    if len(raw_lines) < 3:
        raise ValueError(f"VTK file too short: {path!r}")

    fmt_line = raw_lines[2].strip().upper()
    if "BINARY" in fmt_line:
        raise ValueError(
            f"Only ASCII legacy VTK POLYDATA is supported; "
            f"file appears to be BINARY: {path!r}.  "
            f"Convert with: vtk-convert or meshio-convert."
        )
    if "ASCII" not in fmt_line:
        raise ValueError(
            f"Could not determine VTK format from line 3 (expected 'ASCII' or "
            f"'BINARY'): {raw_lines[2]!r} in {path!r}."
        )

    # Scan for DATASET POLYDATA
    dataset_found = False
    for line in raw_lines:
        if line.strip().upper().startswith("DATASET"):
            if "POLYDATA" not in line.upper():
                raise ValueError(
                    f"Only POLYDATA VTK datasets are supported, "
                    f"got: {line.strip()!r} in {path!r}."
                )
            dataset_found = True
            break
    if not dataset_found:
        raise ValueError(f"No DATASET line found in VTK file {path!r}.")

    # Tokenise everything into a flat list for easy sectioned parsing
    lines = [raw_ln.strip() for raw_ln in raw_lines if raw_ln.strip() and not raw_ln.strip().startswith("#")]

    vertices = None
    faces = None
    i = 0
    while i < len(lines):
        upper = lines[i].upper()

        if upper.startswith("POINTS"):
            parts = lines[i].split()
            n_pts = int(parts[1])
            # Collect 3*n_pts floats; they may span multiple lines
            floats = []
            i += 1
            while len(floats) < 3 * n_pts and i < len(lines):
                # Stop at next keyword section
                if lines[i].upper().split()[0] in (
                    "POLYGONS", "LINES", "STRIPS", "VERTICES",
                    "POINT_DATA", "CELL_DATA", "FIELD", "NORMALS",
                    "TEXTURE_COORDINATES", "SCALARS", "LOOKUP_TABLE",
                ):
                    break
                floats.extend(float(x) for x in lines[i].split())
                i += 1
            if len(floats) < 3 * n_pts:
                raise ValueError(
                    f"Expected {3 * n_pts} floats for POINTS but got "
                    f"{len(floats)} in {path!r}."
                )
            vertices = np.array(floats[: 3 * n_pts], dtype=np.float32).reshape(n_pts, 3)
            continue  # i already advanced

        elif upper.startswith("POLYGONS"):
            parts = lines[i].split()
            n_polys = int(parts[1])
            face_list = []
            i += 1
            while len(face_list) < n_polys and i < len(lines):
                if lines[i].upper().split()[0] in (
                    "POINTS", "LINES", "STRIPS", "VERTICES",
                    "POINT_DATA", "CELL_DATA", "FIELD", "NORMALS",
                    "TEXTURE_COORDINATES", "SCALARS", "LOOKUP_TABLE",
                ):
                    break
                tokens = lines[i].split()
                n_poly = int(tokens[0])
                if n_poly != 3:
                    raise ValueError(
                        f"VTK polygon {len(face_list)} has {n_poly} vertices; "
                        f"only triangles (3) are supported in {path!r}.  "
                        f"Triangulate the mesh before loading."
                    )
                face_list.append([int(tokens[1]), int(tokens[2]), int(tokens[3])])
                i += 1
            if len(face_list) < n_polys:
                raise ValueError(
                    f"Expected {n_polys} polygons but only parsed "
                    f"{len(face_list)} in {path!r}."
                )
            faces = np.array(face_list, dtype=np.uint32)
            continue  # i already advanced

        else:
            i += 1

    if vertices is None:
        raise ValueError(f"No POINTS section found in VTK file {path!r}.")
    if faces is None:
        raise ValueError(f"No POLYGONS section found in VTK file {path!r}.")

    # Bounds check
    n_verts = vertices.shape[0]
    if faces.size > 0 and (int(faces.max()) >= n_verts or int(faces.min()) < 0):
        raise ValueError(
            f"VTK face indices out of range [0, {n_verts}) in {path!r}."
        )

    return vertices, faces


# ---------------------------------------------------------------------------
# PLY ASCII reader
# ---------------------------------------------------------------------------

def read_ply_ascii(path):
    """Read an ASCII PLY triangle mesh.

    Only ASCII PLY files are supported.  Binary PLY files are explicitly
    rejected with a helpful error message.

    Parameters
    ----------
    path : str
        Path to the ``.ply`` file.

    Returns
    -------
    vertices : numpy.ndarray, shape (N, 3), dtype float32
    faces : numpy.ndarray, shape (M, 3), dtype uint32

    Raises
    ------
    ValueError
        If the file is binary PLY, if faces are not triangles, or if the
        header is malformed.
    IOError
        If the file cannot be opened.
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        raw_lines = fh.readlines()

    if not raw_lines or raw_lines[0].strip() != "ply":
        raise ValueError(
            f"File does not start with 'ply' magic; not a PLY file: {path!r}."
        )

    # Check encoding
    for line in raw_lines[1:4]:
        stripped = line.strip().lower()
        if stripped.startswith("format"):
            if "ascii" not in stripped:
                raise ValueError(
                    f"PLY binary format not supported; only ASCII PLY is "
                    f"accepted: {path!r}.  "
                    f"Convert with: plyconvert or meshio-convert."
                )
            break

    # Parse header
    n_verts = None
    n_faces = None
    vertex_props = []   # ordered list of property names for the vertex element
    in_vertex = False
    in_face = False
    header_end = 0

    for idx, line in enumerate(raw_lines):
        stripped = line.strip()
        lower = stripped.lower()

        if lower == "end_header":
            header_end = idx + 1
            break
        if lower.startswith("element vertex"):
            n_verts = int(stripped.split()[2])
            in_vertex = True
            in_face = False
        elif lower.startswith("element face"):
            n_faces = int(stripped.split()[2])
            in_face = True
            in_vertex = False
        elif lower.startswith("property") and in_vertex:
            # e.g. "property float x"
            parts = stripped.split()
            if len(parts) >= 3:
                vertex_props.append(parts[-1])
        elif lower.startswith("element") and not lower.startswith("element vertex") and not lower.startswith("element face"):
            in_vertex = False

    if n_verts is None:
        raise ValueError(f"No 'element vertex' found in PLY header: {path!r}.")
    if n_faces is None:
        raise ValueError(f"No 'element face' found in PLY header: {path!r}.")

    # Determine column indices for x, y, z
    try:
        xi = vertex_props.index("x")
        yi = vertex_props.index("y")
        zi = vertex_props.index("z")
    except ValueError as exc:
        raise ValueError(
            f"PLY vertex element missing x/y/z properties in {path!r}; "
            f"found: {vertex_props!r}."
        ) from exc

    data_lines = [raw_line.strip() for raw_line in raw_lines[header_end:] if raw_line.strip()]

    if len(data_lines) < n_verts + n_faces:
        raise ValueError(
            f"PLY file has {len(data_lines)} data lines but expects "
            f"{n_verts} vertices + {n_faces} faces in {path!r}."
        )

    # Parse vertices
    vertices = np.empty((n_verts, 3), dtype=np.float32)
    for i in range(n_verts):
        try:
            tokens = data_lines[i].split()
            vertices[i] = [float(tokens[xi]), float(tokens[yi]), float(tokens[zi])]
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"Could not parse PLY vertex {i} in {path!r}: "
                f"{data_lines[i]!r}"
            ) from exc

    # Parse faces
    faces = np.empty((n_faces, 3), dtype=np.uint32)
    for j in range(n_faces):
        try:
            tokens = data_lines[n_verts + j].split()
            count = int(tokens[0])
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"Could not parse PLY face {j} in {path!r}: "
                f"{data_lines[n_verts + j]!r}"
            ) from exc
        if count != 3:
            raise ValueError(
                f"PLY face {j} has {count} vertices; only triangles (3) are "
                f"supported in {path!r}.  Triangulate the mesh first."
            )
        try:
            faces[j] = [int(tokens[1]), int(tokens[2]), int(tokens[3])]
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"Could not parse PLY face indices at face {j} in {path!r}: "
                f"{data_lines[n_verts + j]!r}"
            ) from exc

    # Bounds check
    if n_faces > 0 and (int(faces.max()) >= n_verts or int(faces.min()) < 0):
        raise ValueError(
            f"PLY face indices out of range [0, {n_verts}) in {path!r}."
        )

    return vertices, faces


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_READERS = {
    ".off": read_off,
    ".vtk": read_vtk_ascii_polydata,
    ".ply": read_ply_ascii,
}

_SUPPORTED = ", ".join(sorted(_READERS))


def read_mesh(path):
    """Read a triangle mesh from an OFF, VTK, or PLY file.

    Dispatches to the appropriate reader based on the file extension.
    For FreeSurfer binary surfaces (which typically have no standard
    extension, e.g. ``lh.white``) use
    :func:`whippersnappy.geometry.read_geometry` directly, or pass the
    path through :func:`whippersnappy.geometry.inputs.resolve_mesh` which
    handles the routing automatically.

    Parameters
    ----------
    path : str
        Path to a mesh file.  Extension must be one of:
        ``.off``, ``.vtk``, ``.ply`` (case-insensitive).

    Returns
    -------
    vertices : numpy.ndarray, shape (N, 3), dtype float32
    faces : numpy.ndarray, shape (M, 3), dtype uint32

    Raises
    ------
    ValueError
        If the extension is not recognised.
    """
    import os
    ext = os.path.splitext(path)[1].lower()
    reader = _READERS.get(ext)
    if reader is None:
        raise ValueError(
            f"Unsupported mesh file extension {ext!r} for {path!r}.  "
            f"Supported formats: {_SUPPORTED}.  "
            f"For FreeSurfer surfaces (no extension) use resolve_mesh() "
            f"or read_geometry() directly."
        )
    return reader(path)





