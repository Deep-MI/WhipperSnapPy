"""Helper for finding a surface file name inside a subject directory.

This replaces the previous `io.py` name which was generic; `surf_name.py` is
more descriptive (it provides `get_surf_name`).
"""
import os


def get_surf_name(sdir, hemi):
    """Find a suitable surface basename in a subject directory.

    The function searches the standard FreeSurfer `surf/` directory for a
    common surface name in order of preference and returns the basename
    (search for 'pial_semi_inflated', 'white', and then 'inflated').

    Parameters
    ----------
    sdir : str
        Path to the subject directory containing a `surf/` subdirectory.
    hemi : {'lh','rh'}
        Hemisphere prefix to use when searching for surface files.

    Returns
    -------
    surf_name : str or None
        The surface basename if found, otherwise ``None``.
    """
    for surf_name_option in ["pial_semi_inflated", "white", "inflated"]:
        path = os.path.join(sdir, "surf", f"{hemi}.{surf_name_option}")
        if os.path.exists(path):
            return surf_name_option
    return None
