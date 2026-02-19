#!/usr/bin/env python3
"""CLI entry point for four-view batch rendering via snap4.

Renders left and right hemisphere surfaces in lateral and medial views,
stitches them into a single image and writes it to disk.

Usage::

    whippersnap4 -lh <lh_overlay> -rh <rh_overlay> -sd <subject_dir> -o out.png
    whippersnap4 --lh_annot <lh.annot> --rh_annot <rh.annot> -sd <subject_dir>

See ``whippersnap4 --help`` for the full list of options.
For the interactive GUI use ``whippersnap``.
"""

import argparse
import logging
import os
import tempfile

from .. import snap4
from .._version import __version__

# Module logger
logger = logging.getLogger(__name__)


def run():
    """Command-line entry point for WhipperSnapPy four-view batch rendering.

    Parses command-line arguments, validates them, and calls
    :func:`whippersnappy.snap4` to produce a four-view composed image.

    Raises
    ------
    ValueError
        For invalid or mutually exclusive argument combinations.
    RuntimeError
        If the OpenGL context cannot be initialised.
    FileNotFoundError
        If required surface files cannot be found.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        prog="whippersnap4",
        description=(
            "Render a four-view (left/right hemisphere, lateral/medial) "
            "batch snapshot without a GUI. "
            "For the interactive GUI use whippersnap."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    # --- Overlay / annotation inputs ---
    parser.add_argument("-lh", "--lh_overlay", type=str, default=None,
                        help="Path to the lh overlay file.")
    parser.add_argument("-rh", "--rh_overlay", type=str, default=None,
                        help="Path to the rh overlay file.")
    parser.add_argument("--lh_annot", type=str, default=None,
                        help="Path to the lh annotation (.annot) file.")
    parser.add_argument("--rh_annot", type=str, default=None,
                        help="Path to the rh annotation (.annot) file.")

    # --- Subject directory / surface ---
    parser.add_argument("-sd", "--sdir", type=str, required=True,
                        help="Subject directory containing surf/ and label/ subdirectories.")
    parser.add_argument("-s", "--surf_name", type=str, default=None,
                        help="Surface basename to load (e.g. 'white'); "
                             "auto-detected if not provided.")

    # --- Output ---
    parser.add_argument(
        "-o", "--output_path",
        type=str,
        default=os.path.join(tempfile.gettempdir(), "whippersnappy_snap4.png"),
        help="Output image path (default: temp file).",
    )

    # --- Overlay appearance ---
    parser.add_argument("--fmax", type=float, default=None,
                        help="Overlay saturation value (auto-estimated if not set).")
    parser.add_argument("--fthresh", type=float, default=None,
                        help="Overlay threshold value (auto-estimated if not set).")
    parser.add_argument("--invert", action="store_true",
                        help="Invert the color scale.")
    parser.add_argument("--no-colorbar", dest="no_colorbar", action="store_true",
                        default=False, help="Suppress the colorbar.")
    parser.add_argument("-c", "--caption", type=str, default="",
                        help="Caption text to place on the figure.")

    # --- Rendering ---
    parser.add_argument("--diffuse", dest="specular", action="store_false", default=True,
                        help="Diffuse-only shading (no specular).")
    parser.add_argument("--ambient", type=float, default=0.0,
                        help="Ambient light strength (default: 0.0).")
    parser.add_argument("--brain-scale", type=float, default=1.85,
                        help="Geometry scale factor (default: 1.85).")
    parser.add_argument("--font", type=str, default=None,
                        help="Path to a TTF font for captions.")

    args = parser.parse_args()

    try:
        if (args.lh_overlay or args.rh_overlay) and (args.lh_annot or args.rh_annot):
            raise ValueError(
                "Cannot use lh_overlay/rh_overlay and lh_annot/rh_annot at the same time."
            )
        if not any([args.lh_overlay, args.rh_overlay, args.lh_annot, args.rh_annot]):
            raise ValueError(
                "Either lh_overlay/rh_overlay or lh_annot/rh_annot must be present."
            )
        if (args.lh_overlay is None) != (args.rh_overlay is None):
            raise ValueError("Both -lh and -rh overlays must be provided together.")
        if (args.lh_annot is None) != (args.rh_annot is None):
            raise ValueError("Both --lh_annot and --rh_annot must be provided together.")
    except ValueError as e:
        parser.error(str(e))

    logger.debug("Parsed args: %s", vars(args))

    try:
        img = snap4(
            lhoverlaypath=args.lh_overlay,
            rhoverlaypath=args.rh_overlay,
            lhannotpath=args.lh_annot,
            rhannotpath=args.rh_annot,
            sdir=args.sdir,
            caption=args.caption,
            surfname=args.surf_name,
            fthresh=args.fthresh,
            fmax=args.fmax,
            invert=args.invert,
            colorbar=not args.no_colorbar,
            outpath=args.output_path,
            font_file=args.font,
            specular=args.specular,
            ambient=args.ambient,
            brain_scale=args.brain_scale,
        )
        logger.info(
            "Snapshot saved to %s (%dx%d)", args.output_path, img.width, img.height
        )
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        parser.error(str(e))

