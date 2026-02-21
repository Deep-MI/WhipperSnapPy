"""Geometry subpackage — mesh IO, overlay IO, and rendering preparation.

Architecture
------------
The subpackage has three layers:

**Layer 1 — low-level format readers** (one file per format family):

* :mod:`~whippersnappy.geometry.freesurfer_io` — FreeSurfer binary formats:
  surface geometry (``read_geometry``), morphometry scalars
  (``read_morph_data``), MGH overlays (``read_mgh_data``), and annotation
  files (``read_annot_data``).  Contains derived nibabel code (MIT licence).
* :mod:`~whippersnappy.geometry.mesh_io` — open ASCII mesh formats:
  OFF (``read_off``), legacy VTK PolyData (``read_vtk_ascii_polydata``),
  ASCII PLY (``read_ply_ascii``), and GIfTI surface (``read_gifti_surface``).
  Pure stdlib + numpy, except GIfTI which uses nibabel.
* :mod:`~whippersnappy.geometry.overlay_io` — open scalar/label formats:
  plain ASCII (``read_txt``), NumPy binary (``read_npy``, ``read_npz``),
  and GIfTI functional/label (``read_gifti``).  Pure stdlib + numpy, except
  GIfTI.

Each family also exposes a dispatcher (``read_mesh`` / ``read_overlay``)
that routes by file extension.

**Layer 2 — resolvers** (:mod:`~whippersnappy.geometry.inputs`):

``resolve_mesh``, ``resolve_overlay``, ``resolve_bg_map``, ``resolve_roi``,
``resolve_annot`` — the **single public interface** for the rest of the
package.  Each resolver accepts a file path *or* a numpy array *or* ``None``,
dispatches to the correct layer-1 reader, validates shapes and dtypes, and
returns a clean numpy array.  All higher-level code (``prepare.py``,
``snap.py``, ``plot3d.py``, CLIs) should go through resolvers only.

**Layer 3 — geometry preparation** (:mod:`~whippersnappy.geometry.prepare`):

``prepare_geometry`` / ``prepare_geometry_from_arrays`` — load, normalise,
colour, and pack vertex data into the GPU-ready format consumed by the
OpenGL shaders.
"""
from .freesurfer_io import read_annot_data, read_geometry, read_mgh_data, read_morph_data
from .inputs import resolve_annot, resolve_bg_map, resolve_mesh, resolve_overlay, resolve_roi
from .mesh_io import read_gifti_surface, read_mesh, read_off, read_ply_ascii, read_vtk_ascii_polydata
from .overlay_io import read_gifti, read_npy, read_npz, read_overlay, read_txt
from .prepare import (
    estimate_overlay_thresholds,
    prepare_and_validate_geometry,
    prepare_geometry,
    prepare_geometry_from_arrays,
)
from .surf_name import get_surf_name

__all__ = [
    # Layer 3 — geometry preparation
    'prepare_geometry',
    'prepare_geometry_from_arrays',
    'prepare_and_validate_geometry',
    'estimate_overlay_thresholds',
    # Layer 2 — resolvers (preferred public interface)
    'resolve_mesh',
    'resolve_overlay',
    'resolve_bg_map',
    'resolve_roi',
    'resolve_annot',
    # Layer 1 — mesh readers
    'read_mesh',
    'read_off',
    'read_vtk_ascii_polydata',
    'read_ply_ascii',
    'read_gifti_surface',
    # Layer 1 — overlay/scalar readers
    'read_overlay',
    'read_txt',
    'read_npy',
    'read_npz',
    'read_gifti',
    # Layer 1 — FreeSurfer binary readers
    'read_geometry',
    'read_morph_data',
    'read_mgh_data',
    'read_annot_data',
    # Utilities
    'get_surf_name',
]
