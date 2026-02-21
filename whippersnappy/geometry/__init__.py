"""Geometry subpackage exports.

Expose prepare_geometry and small IO helpers under `whippersnappy.geometry`.
"""
from .inputs import resolve_annot, resolve_bg_map, resolve_mesh, resolve_overlay, resolve_roi
from .mesh_io import read_gifti_surface, read_mesh, read_off, read_ply_ascii, read_vtk_ascii_polydata
from .overlay_io import read_gifti, read_npy, read_npz, read_overlay, read_txt
from .prepare import (
    estimate_overlay_thresholds,
    prepare_and_validate_geometry,
    prepare_geometry,
    prepare_geometry_from_arrays,
)
from .read_geometry import read_annot_data, read_geometry, read_mgh_data, read_morph_data
from .surf_name import get_surf_name

__all__ = [
    'prepare_geometry',
    'prepare_geometry_from_arrays',
    'prepare_and_validate_geometry',
    'estimate_overlay_thresholds',
    'resolve_mesh',
    'resolve_overlay',
    'resolve_bg_map',
    'resolve_roi',
    'resolve_annot',
    'read_mesh',
    'read_off',
    'read_vtk_ascii_polydata',
    'read_ply_ascii',
    'read_gifti_surface',
    'read_overlay',
    'read_txt',
    'read_npy',
    'read_npz',
    'read_gifti',
    'read_geometry',
    'read_annot_data',
    'read_mgh_data',
    'read_morph_data',
    'get_surf_name',
]
