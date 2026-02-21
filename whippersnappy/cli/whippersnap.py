#!/usr/bin/python3

"""Interactive GUI viewer for WhipperSnapPy.

Opens a live OpenGL window for any triangular surface mesh together with a
Qt-based configuration panel that allows adjusting overlay thresholds at
runtime.

Two input modes are supported:

**General mode** — pass any mesh file directly::

    whippersnap --mesh mesh.off --overlay values.mgh
    whippersnap --mesh lh.white --overlay lh.thickness --bg-map lh.curv

**FreeSurfer shortcut** — pass a subject directory and hemisphere; all
FreeSurfer paths are derived automatically::

    whippersnap -sd <subject_dir> --hemi lh --overlay lh.thickness
    whippersnap -sd <subject_dir> --hemi rh --annot rh.aparc.annot

See ``whippersnap --help`` for the full list of options.
For non-interactive four-view batch rendering use ``whippersnap4``.
For single-view non-interactive snapshots use ``whippersnap1``.
"""

import argparse
import logging
import os
import signal
import sys
import threading

import glfw
import OpenGL.GL as gl
import pyrr

try:
    from PyQt6.QtWidgets import QApplication
except Exception:
    # GUI dependency missing; raise a clear error at runtime
    QApplication = None

from .._version import __version__
from ..geometry import get_surf_name, prepare_geometry
from ..gl import get_view_matrices, init_window, setup_shader
from ..utils.types import ViewType

# Module logger
logger = logging.getLogger(__name__)

# Global state shared between the GL thread and the Qt main thread
current_fthresh_ = None
current_fmax_ = None
app_ = None
app_window_ = None
app_window_closed_ = False


def show_window(
    mesh,
    overlay=None,
    annot=None,
    bg_map=None,
    roi=None,
    invert=False,
    specular=True,
    view=ViewType.LEFT,
):
    """Start a live interactive OpenGL window for viewing a triangular mesh.

    The function initializes a GLFW window and renders the provided mesh
    with any supplied overlay or annotation.  It polls for threshold updates
    from the Qt configuration panel and re-renders whenever the thresholds
    change.

    ``mesh`` is a fully resolved path (or ``(vertices, faces)`` tuple);
    all FreeSurfer path-building is performed in :func:`run` before this
    function is called.

    Parameters
    ----------
    mesh : str or tuple of (array-like, array-like)
        Path to any mesh file supported by :func:`whippersnappy.geometry.inputs.resolve_mesh`
        (FreeSurfer binary, ``.off``, ``.vtk``, ``.ply``) **or** a
        ``(vertices, faces)`` array tuple.
    overlay : str, array-like, or None, optional
        Per-vertex scalar overlay — file path or (N,) array.
    annot : str, tuple, or None, optional
        FreeSurfer ``.annot`` file path or ``(labels, ctab[, names])`` tuple.
    bg_map : str, array-like, or None, optional
        Per-vertex scalar file or array for background shading (sign → light/dark).
    roi : str, array-like, or None, optional
        FreeSurfer label file path or boolean (N,) array masking overlay coloring.
    invert : bool, optional
        Invert the overlay color mapping. Default is ``False``.
    specular : bool, optional
        Enable specular highlights in the shader. Default is ``True``.
    view : ViewType, optional
        Initial camera view direction. Default is ``ViewType.LEFT``.

    Raises
    ------
    RuntimeError
        If the GLFW window or OpenGL context could not be created.
    """
    global current_fthresh_, current_fmax_, app_, app_window_, app_window_closed_

    wwidth = 720
    wheight = 600
    window = init_window(wwidth, wheight, "WhipperSnapPy", visible=True)
    if not window:
        logger.error("Could not create any GLFW window/context. OpenGL context unavailable.")
        raise RuntimeError("Could not create any GLFW window/context. OpenGL context unavailable.")

    view_mats = get_view_matrices()
    viewmat = view_mats[view]
    rot_y = pyrr.Matrix44.from_y_rotation(0)

    meshdata, triangles, fthresh, fmax, neg = prepare_geometry(
        mesh, overlay, annot, bg_map, roi, current_fthresh_, current_fmax_,
        invert=invert,
    )
    shader = setup_shader(meshdata, triangles, wwidth, wheight, specular=specular)

    logger.info("\nKeys:\nLeft - Right : Rotate Geometry\nESC          : Quit\n")

    ypos = 0
    while glfw.get_key(window, glfw.KEY_ESCAPE) != glfw.PRESS and not glfw.window_should_close(window):
        if app_window_closed_:
            break

        glfw.poll_events()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        if app_window_ is not None:
            if (
                app_window_.get_fthresh_value() != current_fthresh_
                or app_window_.get_fmax_value() != current_fmax_
            ):
                current_fthresh_ = app_window_.get_fthresh_value()
                current_fmax_ = app_window_.get_fmax_value()
                meshdata, triangles, fthresh, fmax, neg = prepare_geometry(
                    mesh, overlay, annot, bg_map, roi,
                    current_fthresh_, current_fmax_,
                    invert=invert,
                )
                shader = setup_shader(meshdata, triangles, wwidth, wheight, specular=specular)

        transformLoc = gl.glGetUniformLocation(shader, "transform")
        gl.glUniformMatrix4fv(transformLoc, 1, gl.GL_FALSE, rot_y * viewmat)

        if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
            ypos += 0.0004
        if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
            ypos -= 0.0004
        rot_y = pyrr.Matrix44.from_y_rotation(ypos)

        gl.glDrawElements(gl.GL_TRIANGLES, triangles.size, gl.GL_UNSIGNED_INT, None)
        glfw.swap_buffers(window)

    glfw.terminate()
    # Signal the main thread to tear down the Qt app
    app_window_closed_ = True


def config_app_exit_handler():
    """Mark the configuration application as closed.

    Connected to the Qt app's ``aboutToQuit`` signal so the OpenGL loop
    in the worker thread terminates cleanly.
    """
    global app_window_closed_
    app_window_closed_ = True


def run():
    """Command-line entry point for the WhipperSnapPy interactive GUI.

    Parses command-line arguments, validates them, then spawns the OpenGL
    viewer thread and launches the PyQt6 configuration window in the main
    thread.

    Two mutually exclusive input modes are supported:

    **General mode** — supply a mesh file directly with ``--mesh``.
    Works with FreeSurfer binary surfaces, OFF, VTK, and PLY files.

    **FreeSurfer shortcut** — supply ``-sd``/``--sdir`` and
    ``--hemi``; the surface, curvature, and cortex-label paths are all
    derived automatically from the subject directory.

    Raises
    ------
    RuntimeError
        If PyQt6 is not installed (``pip install 'whippersnappy[gui]'``).
    ValueError
        For invalid or mutually exclusive argument combinations.

    Notes
    -----
    **General mode** (``--mesh`` required):

    * ``--mesh`` — path to any triangular mesh file (FreeSurfer binary,
      ``.off``, ``.vtk``, ``.ply``).
    * ``--overlay`` — per-vertex scalar overlay file path or ``.mgh``.
    * ``--annot`` — FreeSurfer ``.annot`` file.
    * ``--bg-map`` — per-vertex scalar file for background shading.
    * ``--roi`` — FreeSurfer label file or boolean mask for overlay region.

    **FreeSurfer shortcut** (``-sd``/``--sdir`` + ``--hemi`` required):

    * ``-sd`` / ``--sdir`` — subject directory containing ``surf/`` and
      ``label/`` subdirectories.
    * ``--hemi`` — hemisphere to display: ``lh`` or ``rh``.
    * ``-lh`` / ``--lh_overlay`` or ``-rh`` / ``--rh_overlay`` — overlay
      for the respective hemisphere (shorthand; equivalent to ``--overlay``).
    * ``--annot`` — annotation file (full path).
    * ``-s`` / ``--surf_name`` — surface basename (e.g. ``white``);
      auto-detected if not provided.
    * ``--curv-name`` — curvature basename (default: ``curv``).
    * ``--label-name`` — cortex label basename (default: ``cortex.label``).

    **Common options** (both modes):

    * ``--fthresh`` — overlay threshold (default: 2.0); adjustable live in the GUI.
    * ``--fmax`` — overlay saturation (default: 4.0); adjustable live in the GUI.
    * ``--invert`` — invert the color scale.
    * ``--diffuse`` — use diffuse-only shading (no specular highlights).
    * ``--view`` — initial camera view (default: ``left``).

    Requires ``pip install 'whippersnappy[gui]'``.
    For non-interactive batch rendering use ``whippersnap4`` or ``whippersnap1``.
    """
    global current_fthresh_, current_fmax_, app_, app_window_
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    _VIEW_CHOICES = {v.name.lower(): v for v in ViewType}

    parser = argparse.ArgumentParser(
        prog="whippersnap",
        description=(
            "Interactive GUI viewer for any triangular surface mesh. "
            "Pass --mesh for a direct mesh file, or -sd/--sdir + --hemi "
            "for the FreeSurfer subject-directory shortcut. "
            "For non-interactive batch rendering use whippersnap4 or whippersnap1."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    # --- General mesh mode ---
    general = parser.add_argument_group(
        "general mode",
        "Load any triangular mesh directly (OFF, VTK, PLY, FreeSurfer binary).",
    )
    general.add_argument(
        "--mesh", type=str, default=None,
        help="Path to any triangular mesh file (.off, .vtk, .ply, or FreeSurfer binary).",
    )
    general.add_argument(
        "--bg-map", dest="bg_map", type=str, default=None,
        help="Per-vertex scalar file for background shading (sign → light/dark).",
    )
    general.add_argument(
        "--roi", type=str, default=None,
        help="FreeSurfer label file or boolean mask restricting overlay coloring.",
    )

    # --- FreeSurfer shortcut mode ---
    fs = parser.add_argument_group(
        "FreeSurfer shortcut",
        "Derive mesh, curvature, and label paths from a subject directory.",
    )
    fs.add_argument(
        "-sd", "--sdir", type=str, default=None,
        help="Subject directory containing surf/ and label/ subdirectories.",
    )
    fs.add_argument(
        "--hemi", type=str, default=None, choices=["lh", "rh"],
        help="Hemisphere to display: lh or rh.",
    )
    fs.add_argument(
        "-s", "--surf_name", type=str, default=None,
        help="Surface basename (e.g. 'white'); auto-detected if not provided.",
    )
    fs.add_argument(
        "--curv-name", dest="curv_name", type=str, default="curv",
        help="Curvature file basename for background shading (default: curv).",
    )
    fs.add_argument(
        "--label-name", dest="label_name", type=str, default="cortex.label",
        help="Cortex label basename for overlay masking (default: cortex.label).",
    )

    # Hemisphere-prefixed overlay shortcuts (FreeSurfer convention)
    fs.add_argument(
        "-lh", "--lh_overlay", type=str, default=None,
        help="Shorthand for --overlay when using lh hemisphere (e.g. lh.thickness).",
    )
    fs.add_argument(
        "-rh", "--rh_overlay", type=str, default=None,
        help="Shorthand for --overlay when using rh hemisphere (e.g. rh.thickness).",
    )

    # --- Inputs common to both modes ---
    common = parser.add_argument_group("overlay / annotation (both modes)")
    common.add_argument(
        "--overlay", type=str, default=None,
        help="Per-vertex scalar overlay file path.",
    )
    common.add_argument(
        "--annot", type=str, default=None,
        help="FreeSurfer .annot file for parcellation coloring.",
    )

    # --- Appearance / rendering ---
    rend = parser.add_argument_group("appearance")
    rend.add_argument("--fmax",    type=float, default=4.0,
                      help="Overlay saturation value (default: 4.0).")
    rend.add_argument("--fthresh", type=float, default=2.0,
                      help="Overlay threshold value (default: 2.0).")
    rend.add_argument("--invert",  action="store_true",
                      help="Invert the color scale.")
    rend.add_argument("--diffuse", dest="specular", action="store_false", default=True,
                      help="Diffuse-only shading (no specular highlights).")
    rend.add_argument(
        "--view", type=str, default="left", choices=list(_VIEW_CHOICES),
        help="Initial camera view direction (default: left).",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve the two modes and build the final mesh / bg_map / roi paths
    # ------------------------------------------------------------------
    fs_mode     = args.sdir is not None or args.hemi is not None
    general_mode = args.mesh is not None

    try:
        if fs_mode and general_mode:
            raise ValueError(
                "Cannot combine --mesh with -sd/--sdir or --hemi. "
                "Use either general mode (--mesh) or FreeSurfer mode (-sd + --hemi)."
            )
        if not fs_mode and not general_mode:
            raise ValueError(
                "Either --mesh (general mode) or both -sd/--sdir and --hemi "
                "(FreeSurfer shortcut) must be provided."
            )

        if fs_mode:
            if args.sdir is None or args.hemi is None:
                raise ValueError(
                    "FreeSurfer mode requires both -sd/--sdir and --hemi."
                )

        # Resolve overlay: --overlay takes precedence; -lh/-rh are shorthands
        overlay = args.overlay
        if overlay is None:
            if args.hemi == "lh" and args.lh_overlay:
                overlay = args.lh_overlay
            elif args.hemi == "rh" and args.rh_overlay:
                overlay = args.rh_overlay
            elif args.lh_overlay and not fs_mode:
                raise ValueError(
                    "-lh/--lh_overlay is only valid in FreeSurfer mode (with --hemi lh)."
                )
            elif args.rh_overlay and not fs_mode:
                raise ValueError(
                    "-rh/--rh_overlay is only valid in FreeSurfer mode (with --hemi rh)."
                )

        if overlay and args.annot:
            raise ValueError("Cannot combine --overlay/hemisphere overlay and --annot.")
        if not overlay and not args.annot and not general_mode:
            raise ValueError(
                "Either an overlay (-lh/-rh/--overlay) or --annot must be provided."
            )

    except ValueError as e:
        parser.error(str(e))

    # Build resolved mesh / bg_map / roi
    if fs_mode:
        hemi  = args.hemi
        sdir  = args.sdir
        if args.surf_name is None:
            found = get_surf_name(sdir, hemi)
            if found is None:
                parser.error(f"Could not find a valid surface in {sdir} for hemi {hemi!r}.")
            mesh_path = os.path.join(sdir, "surf", f"{hemi}.{found}")
        else:
            mesh_path = os.path.join(sdir, "surf", f"{hemi}.{args.surf_name}")
        bg_map = os.path.join(sdir, "surf", f"{hemi}.{args.curv_name}") if args.curv_name else None
        roi    = os.path.join(sdir, "label", f"{hemi}.{args.label_name}") if args.label_name else None
        view   = ViewType.RIGHT if hemi == "rh" else ViewType.LEFT
    else:
        mesh_path = args.mesh
        bg_map    = args.bg_map
        roi       = args.roi
        view      = _VIEW_CHOICES[args.view]

    # ------------------------------------------------------------------
    # Start the Qt app + OpenGL thread
    # ------------------------------------------------------------------
    if QApplication is None:
        print(
            "ERROR: Interactive mode requires PyQt6. "
            "Install with: pip install 'whippersnappy[gui]'",
            file=sys.stderr,
        )
        raise RuntimeError(
            "Interactive mode requires PyQt6. "
            "Install with: pip install 'whippersnappy[gui]'"
        )

    try:
        from ..gui import ConfigWindow  # noqa: PLC0415
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Interactive mode requires PyQt6. "
            "Install with: pip install 'whippersnappy[gui]'"
        ) from e

    current_fthresh_ = args.fthresh
    current_fmax_    = args.fmax

    thread = threading.Thread(
        target=show_window,
        kwargs=dict(
            mesh=mesh_path,
            overlay=overlay,
            annot=args.annot,
            bg_map=bg_map,
            roi=roi,
            invert=args.invert,
            specular=args.specular,
            view=view,
        ),
    )
    thread.start()

    app_ = QApplication([])
    app_.setStyle("Fusion")
    app_.aboutToQuit.connect(config_app_exit_handler)

    screen_geometry = app_.primaryScreen().availableGeometry()
    app_window_ = ConfigWindow(
        screen_dims=(screen_geometry.width(), screen_geometry.height()),
        initial_fthresh_value=current_fthresh_,
        initial_fmax_value=current_fmax_,
    )

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app_window_.show()
    app_.exec()

