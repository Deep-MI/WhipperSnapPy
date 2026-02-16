"""Helper for finding a surface file name inside a subject directory.

This replaces the previous `io.py` name which was generic; `surf_name.py` is
more descriptive (it provides `get_surf_name`).
"""
import os


def get_surf_name(sdir, hemi):
    """Find a valid surface file in the specified subject directory.

    Returns the surface basename (e.g. 'white', 'inflated', etc.) or None.
    """
    for surf_name_option in ["pial_semi_inflated", "white", "inflated"]:
        path = os.path.join(sdir, "surf", f"{hemi}.{surf_name_option}")
        if os.path.exists(path):
            return surf_name_option
    return None

