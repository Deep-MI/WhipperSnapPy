"""Geometry subpackage exports.

Expose prepare_geometry and small IO helpers under `whippersnappy.geometry`.
"""
from .prepare import prepare_geometry
from .read_geometry import read_annot_data, read_geometry, read_mgh_data, read_morph_data
from .surf_name import get_surf_name

__all__ = [
    'prepare_geometry', 'read_geometry', 'read_annot_data', 'read_mgh_data', 'read_morph_data', 'get_surf_name'
]
