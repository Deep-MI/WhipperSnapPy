#!/usr/bin/env python3
"""CLI entry point for single-mesh snapshot via snap1."""

import argparse
import logging
import os
import tempfile

from .. import snap1
from .._version import __version__
from ..utils.types import ColorSelection, OrientationType, ViewType

_VIEW_CHOICES = {v.name.lower(): v for v in ViewType}
_ORIENT_CHOICES = {o.name.lower(): o for o in OrientationType}
_COLOR_CHOICES = {c.name.lower(): c for c in ColorSelection}


def run():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        prog="whippersnap1",
        description=(
            "Render a single-view screenshot of any triangular surface mesh "
            "(FreeSurfer or otherwise) without a GUI."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    # --- Required ---
    parser.add_argument(
        "meshpath",
        type=str,
        help="Path to the surface file. FreeSurfer binary format (e.g. lh.white) "
             "or any mesh readable by the geometry module.",
    )

    # --- Output ---
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=os.path.join(tempfile.gettempdir(), "whippersnappy_snap1.png"),
        help="Output PNG path. Defaults to a temp file.",
    )

    # --- Optional overlay / annotation / label / curv ---
    parser.add_argument("--overlay",  type=str, default=None, help="Per-vertex overlay file.")
    parser.add_argument("--annot",    type=str, default=None, help="FreeSurfer .annot file.")
    parser.add_argument("--label",    type=str, default=None, help="Label file for masking.")
    parser.add_argument("--curv",     type=str, default=None, help="Curvature file for texturing.")

    # --- View ---
    parser.add_argument(
        "--view",
        type=str,
        default="left",
        choices=list(_VIEW_CHOICES),
        help="Pre-defined view direction (default: left).",
    )

    # --- Appearance ---
    parser.add_argument("--width",   type=int,   default=700)
    parser.add_argument("--height",  type=int,   default=500)
    parser.add_argument("--fthresh", type=float, default=None, help="Overlay threshold.")
    parser.add_argument("--fmax",    type=float, default=None, help="Overlay saturation value.")
    parser.add_argument("--caption", type=str,   default=None)
    parser.add_argument("--invert",  action="store_true", help="Invert color scale.")
    parser.add_argument("--no-colorbar", dest="no_colorbar", action="store_true")
    parser.add_argument(
        "--color-mode",
        type=str,
        default="both",
        choices=list(_COLOR_CHOICES),
        help="Which overlay sign to display (default: both).",
    )
    parser.add_argument(
        "--orientation",
        type=str,
        default="horizontal",
        choices=list(_ORIENT_CHOICES),
        help="Colorbar orientation (default: horizontal).",
    )
    parser.add_argument("--diffuse", dest="specular", action="store_false", default=True,
                        help="Use diffuse-only shading (no specular).")
    parser.add_argument("--brain-scale", type=float, default=1.5,
                        help="Geometry scale factor (default: 1.5).")
    parser.add_argument("--ambient", type=float, default=0.0,
                        help="Ambient light strength (default: 0.0).")
    parser.add_argument("--font",    type=str,   default=None,
                        help="Path to a TTF font for captions.")

    args = parser.parse_args()

    try:
        img = snap1(
            meshpath=args.meshpath,
            outpath=args.output,
            overlaypath=args.overlay,
            annotpath=args.annot,
            labelpath=args.label,
            curvpath=args.curv,
            view=_VIEW_CHOICES[args.view],
            width=args.width,
            height=args.height,
            fthresh=args.fthresh,
            fmax=args.fmax,
            caption=args.caption,
            invert=args.invert,
            colorbar=not args.no_colorbar,
            color_mode=_COLOR_CHOICES[args.color_mode],
            orientation=_ORIENT_CHOICES[args.orientation],
            font_file=args.font,
            specular=args.specular,
            brain_scale=args.brain_scale,
            ambient=args.ambient,
        )
        logging.getLogger(__name__).info("Snapshot saved to %s (%dx%d)", args.output, img.width, img.height)
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        parser.error(str(e))
