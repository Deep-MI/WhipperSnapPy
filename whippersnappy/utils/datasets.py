"""Sample dataset download utility for WhipperSnapPy.

Downloads and caches a small anonymized FreeSurfer subject from the
WhipperSnapPy GitHub release assets for use in tutorials and tests.
"""

from pathlib import Path

RELEASE_URL = (
    "https://github.com/Deep-MI/WhipperSnapPy"
    "/releases/download/v2.0.0/{file_name}"
)

# Mapping of relative path inside the subject directory → SHA-256 hash.
# GitHub release assets are flat (no subdirectories), so the URL uses only
# the basename while pooch.retrieve() reconstructs the subdirectory locally.
_FILES = {
    "README.md":                            "sha256:ecb6ddf31cec17f3a8636fc3ecac90099c441228811efed56104e29fcd301bc5",
    "surf/lh.white":                        "sha256:4ab049fb42ca882ba9b56f8fe0d0e8814973e7fa2e0575a794d8e468abf7d62f",
    "surf/lh.curv":                         "sha256:9edbde57be8593cd9d89d9d1124e2175edd8ecfee55d53e066d89700c480b12a",
    "surf/lh.thickness":                    "sha256:40ab3483284608c6c5cca2d3d794a60cd1bcbeb0140bb1ca6ad0fce7962c57c6",
    "surf/rh.white":                        "sha256:43035c53a8b04bebe4e843c34f80588f253f79052a8dbf7194b706495b11f8d2",
    "surf/rh.curv":                         "sha256:af2bc71133d7ef17ce1a3a6f4208d2495a5a4c96da00c80b59be03bb7c8ea83f",
    "surf/rh.thickness":                    "sha256:50ec291c73928cd697156edd9e0e77f5c54d15c56cf84810d2564b496876e132",
    "label/lh.aparc.DKTatlas.mapped.annot": "sha256:4d48d33f4fd8278ab973a1552f6ea9c396dfc1791b707ed17ad8e761299c4960",
    "label/lh.cortex.label":                "sha256:79ae17fcfde6b2e0a75a0652fcc0f3c072e4ea62a541843b7338e01c598b0b6e",
    "label/rh.aparc.DKTatlas.mapped.annot": "sha256:12217166d8ef43ee1fa280511ec2ba0796c6885f527a4455b93760acc73ce273",
    "label/rh.cortex.label":                "sha256:162c97c887eb1ec857fe575b8cc4e4b950c7dd5ec181a581d709bbe7fca58f9e",
}


def _build_dict(base: Path) -> dict:
    """Build the return dictionary of paths from a subject base directory."""
    return {
        "sdir":         str(base),
        "lh_white":     str(base / "surf/lh.white"),
        "lh_curv":      str(base / "surf/lh.curv"),
        "lh_thickness": str(base / "surf/lh.thickness"),
        "rh_white":     str(base / "surf/rh.white"),
        "rh_curv":      str(base / "surf/rh.curv"),
        "rh_thickness": str(base / "surf/rh.thickness"),
        "lh_annot":     str(base / "label/lh.aparc.DKTatlas.mapped.annot"),
        "lh_label":     str(base / "label/lh.cortex.label"),
        "rh_annot":     str(base / "label/rh.aparc.DKTatlas.mapped.annot"),
        "rh_label":     str(base / "label/rh.cortex.label"),
    }


def fetch_sample_subject() -> dict:
    """Download and cache the WhipperSnapPy sample subject (Rhineland Study).

    Downloads FreeSurfer surface files for one anonymized subject into the
    OS-specific user cache directory and returns a dictionary of paths to
    all files. Files are only downloaded once; subsequent calls use the
    local cache.

    If a ``sub-rs/`` directory with all required files is found next to the
    package root (i.e. inside the source repository), it is used directly
    without any network access. This allows the Sphinx doc build to work
    before the GitHub release assets are published.

    Returns
    -------
    dict
        Dictionary with the following keys:

        * ``sdir`` -- path to the subject root directory (``sub-rs/``),
          usable directly as the ``sdir`` argument to :func:`~whippersnappy.snap4`.
        * ``lh_white`` -- path to ``surf/lh.white``.
        * ``lh_curv`` -- path to ``surf/lh.curv``.
        * ``lh_thickness`` -- path to ``surf/lh.thickness``.
        * ``rh_white`` -- path to ``surf/rh.white``.
        * ``rh_curv`` -- path to ``surf/rh.curv``.
        * ``rh_thickness`` -- path to ``surf/rh.thickness``.
        * ``lh_annot`` -- path to ``label/lh.aparc.DKTatlas.mapped.annot``.
        * ``lh_label`` -- path to ``label/lh.cortex.label``.
        * ``rh_annot`` -- path to ``label/rh.aparc.DKTatlas.mapped.annot``.
        * ``rh_label`` -- path to ``label/rh.cortex.label``.

    Raises
    ------
    ImportError
        If ``pooch`` is not installed. Install with
        ``pip install 'whippersnappy[notebook]'``.

    Notes
    -----
    Data from the Rhineland Study (Koch et al.),
    https://doi.org/10.5281/zenodo.11186582, CC BY 4.0.

    Examples
    --------
    >>> from whippersnappy import fetch_sample_subject
    >>> data = fetch_sample_subject()
    >>> print(data["sdir"])
    """
    try:
        import pooch
    except ImportError as e:
        raise ImportError(
            "fetch_sample_subject() requires pooch. "
            "Install with: pip install 'whippersnappy[notebook]'"
        ) from e

    # Use a local sub-rs/ directory (present in the source repo) when all
    # required files are already there — no network access needed.
    _pkg_root = Path(__file__).parent.parent.parent  # .../whippersnappy/
    _local = _pkg_root / "sub-rs"
    if _local.is_dir() and all((_local / p).exists() for p in _FILES):
        return _build_dict(_local)

    # Otherwise download from the GitHub release and cache in the OS cache dir.
    base = Path(pooch.os_cache("whippersnappy")) / "sub-rs"

    for rel_path, known_hash in _FILES.items():
        rel = Path(rel_path)
        pooch.retrieve(
            url=RELEASE_URL.format(file_name=rel.name),
            known_hash=known_hash,
            fname=rel.name,
            path=base / rel.parent,
        )

    return _build_dict(base)
