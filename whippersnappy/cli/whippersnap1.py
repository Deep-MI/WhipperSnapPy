#!/usr/bin/env python3
"""CLI entry point for single-mesh snapshot and rotation video via snap1/snap_rotate.

Renders any triangular surface mesh from a chosen viewpoint and saves it as a
PNG image. Alternatively, pass ``--rotate`` to produce a full 360° rotation
video (MP4, WebM, or GIF).

The mesh can be a FreeSurfer binary surface (e.g. ``lh.white``), an ASCII OFF
file (``mesh.off``), a legacy ASCII VTK PolyData file (``mesh.vtk``), or an
ASCII PLY file (``mesh.ply``).

Usage::

    # FreeSurfer surface — lateral view with thickness overlay
    whippersnap1 --mesh <subject_dir>/surf/lh.white \\
        --overlay <subject_dir>/surf/lh.thickness \\
        --bg-map  <subject_dir>/surf/lh.curv \\
        --roi     <subject_dir>/label/lh.cortex.label \\
        --view left --fthresh 1.5 --fmax 4.0 \\
        -o snap1.png

    # OFF / VTK / PLY mesh with a numpy-saved overlay
    whippersnap1 --mesh mesh.off --overlay values.mgh -o snap1.png
    whippersnap1 --mesh mesh.vtk -o snap1.png
    whippersnap1 --mesh mesh.ply --overlay values.mgh -o snap1.png

    # 360° rotation video
    whippersnap1 --mesh <subject_dir>/surf/lh.white \\
        --overlay <subject_dir>/surf/lh.thickness \\
        --rotate --rotate-frames 72 --rotate-fps 24 \\
        -o rotation.mp4

    # Parcellation annotation
    whippersnap1 --mesh <subject_dir>/surf/lh.white \\
        --annot <subject_dir>/label/lh.aparc.annot \\
        --view left -o snap_annot.png

See ``whippersnap1 --help`` for the full list of options.
For four-view batch rendering use ``whippersnap4``.
For the interactive GUI use ``whippersnap``.
"""

import argparse
import logging
import os
import tempfile

if __name__ == "__main__" and __package__ is None:
    import sys
    os.execv(sys.executable, [sys.executable, "-m", "whippersnappy.cli.whippersnap1"] + sys.argv[1:])

from .. import snap1, snap_rotate
from .._version import __version__
from ..utils.types import ColorSelection, OrientationType, ViewType

_VIEW_CHOICES = {v.name.lower(): v for v in ViewType}
_ORIENT_CHOICES = {o.name.lower(): o for o in OrientationType}
_COLOR_CHOICES = {c.name.lower(): c for c in ColorSelection}


def run():
    """Command-line entry point for single-view snapshot or rotation video.

    Parses command-line arguments, validates them, and calls either
    :func:`whippersnappy.snap1` (static snapshot) or
    :func:`whippersnappy.snap_rotate` (360° rotation video) depending on
    whether ``--rotate`` is passed.
    All input is read from ``sys.argv`` via :mod:`argparse`.

    Raises
    ------
    FileNotFoundError
        If the mesh file or any overlay/annotation/label file cannot be found.
    RuntimeError
        If the OpenGL context cannot be initialised.
    ValueError
        For invalid argument combinations.

    Notes
    -----
    **Snapshot options** (default mode):

    * ``--mesh`` — path to any triangular surface mesh: FreeSurfer binary
      (e.g. ``lh.white``), ASCII OFF (``.off``), legacy ASCII VTK PolyData
      (``.vtk``), or ASCII PLY (``.ply``).
    * ``--overlay`` — per-vertex scalar overlay (e.g. ``lh.thickness`` or a ``.mgh`` file).
    * ``--annot`` — FreeSurfer ``.annot`` parcellation file.
    * ``--roi`` — FreeSurfer label file defining vertices to include in overlay coloring.
    * ``--bg-map`` — per-vertex scalar file whose sign controls light/dark background shading.
    * ``--view`` — camera direction: ``left``, ``right``, ``front``, ``back``,
      ``top``, ``bottom`` (default: ``left``).
    * ``--fthresh`` / ``--fmax`` — overlay threshold and saturation values.
    * ``--invert`` — invert the color scale.
    * ``--no-colorbar`` — suppress the color bar.
    * ``--caption`` — text label placed on the image.
    * ``--width`` / ``--height`` — output resolution in pixels (default: 700×500).
    * ``-o`` / ``--output`` — output file path (default: temp ``.png``).

    **Rotation video options** (pass ``--rotate``):

    * ``--rotate-frames`` — number of frames for one full rotation (default: 72).
    * ``--rotate-fps`` — output frame rate (default: 24).
    * ``--rotate-start-view`` — starting camera direction (default: ``left``).
    * ``-o`` — output path; extension controls format: ``.mp4``, ``.webm``,
      or ``.gif`` (GIF requires no ffmpeg).
    """

    parser = argparse.ArgumentParser(
        prog="whippersnap1",
        description=(
            "Render a single-view screenshot of any triangular surface mesh "
            "(FreeSurfer or otherwise) without a GUI. "
            "Pass --rotate to produce a 360° rotation video instead."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    # --- Mesh input: --mesh flag (preferred) or bare positional (legacy) ---
    parser.add_argument(
        "--mesh",
        type=str,
        default=None,
        help=(
            "Path to the surface mesh file. Supported formats: "
            "FreeSurfer binary surface (e.g. lh.white, rh.pial), "
            "ASCII OFF (.off), legacy ASCII VTK PolyData (.vtk), ASCII PLY (.ply), "
            "GIfTI surface (.gii, .surf.gii)."
        ),
    )
    # Keep positional for backward compatibility (silently accepted)
    parser.add_argument(
        "_mesh_positional",
        nargs="?",
        default=None,
        metavar="MESH",
        help=argparse.SUPPRESS,
    )

    # --- Output ---
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help=(
            "Output file path. For snapshots defaults to a temp .png file; "
            "for rotation videos defaults to a temp .mp4 file. "
            "Use .gif for an animated GIF (no ffmpeg required)."
        ),
    )

    # --- Optional overlay / annotation / roi / bg-map ---
    parser.add_argument("--overlay",  type=str, default=None, help="Per-vertex overlay file.")
    parser.add_argument("--annot",    type=str, default=None, help="FreeSurfer .annot file.")
    parser.add_argument("--roi",      type=str, default=None,
                        help="Path to a FreeSurfer label file defining the region of interest "
                             "(vertices to include in overlay coloring).")
    parser.add_argument("--bg-map",   type=str, default=None, dest="bg_map",
                        help="Path to a per-vertex scalar file used as background shading "
                             "(sign determines light/dark).")

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

    # --- Rotation video ---
    rotate_group = parser.add_argument_group("rotation video (--rotate)")
    rotate_group.add_argument(
        "--rotate",
        action="store_true",
        help="Produce a 360° rotation video instead of a static snapshot.",
    )
    rotate_group.add_argument(
        "--rotate-frames",
        type=int,
        default=72,
        metavar="N",
        help="Number of frames for a full rotation (default: 72, i.e. 5° per frame).",
    )
    rotate_group.add_argument(
        "--rotate-fps",
        type=int,
        default=24,
        metavar="FPS",
        help="Frame rate of the output video (default: 24).",
    )
    rotate_group.add_argument(
        "--rotate-start-view",
        type=str,
        default="left",
        choices=list(_VIEW_CHOICES),
        metavar="VIEW",
        help="Starting view for the rotation (default: left).",
    )

    args = parser.parse_args()

    # Resolve mesh: --mesh takes precedence over bare positional argument
    mesh_path = args.mesh or args._mesh_positional
    if mesh_path is None:
        parser.error("A mesh file is required: use --mesh <path>.")

    log = logging.getLogger(__name__)

    try:
        if args.rotate:
            outpath = args.output or os.path.join(
                tempfile.gettempdir(), "whippersnappy_rotation.mp4"
            )
            snap_rotate(
                mesh=mesh_path,
                outpath=outpath,
                n_frames=args.rotate_frames,
                fps=args.rotate_fps,
                width=args.width,
                height=args.height,
                overlay=args.overlay,
                bg_map=args.bg_map,
                annot=args.annot,
                roi=args.roi,
                fthresh=args.fthresh,
                fmax=args.fmax,
                invert=args.invert,
                specular=args.specular,
                ambient=args.ambient,
                brain_scale=args.brain_scale,
                start_view=_VIEW_CHOICES[args.rotate_start_view],
                color_mode=_COLOR_CHOICES[args.color_mode],
            )
            log.info("Rotation video saved to %s", outpath)
        else:
            outpath = args.output or os.path.join(
                tempfile.gettempdir(), "whippersnappy_snap1.png"
            )
            img = snap1(
                mesh=mesh_path,
                outpath=outpath,
                overlay=args.overlay,
                annot=args.annot,
                roi=args.roi,
                bg_map=args.bg_map,
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
            log.info("Snapshot saved to %s (%dx%d)", outpath, img.width, img.height)
    except (RuntimeError, FileNotFoundError, ValueError, ImportError) as e:
        parser.error(str(e))


if __name__ == "__main__":
    run()


