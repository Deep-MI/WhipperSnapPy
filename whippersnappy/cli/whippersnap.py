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
import sys

if __name__ == "__main__" and __package__ is None:
    # Replace the current process with `python -m whippersnappy.cli.whippersnap`
    # so that relative imports work. os.execv replaces this process in-place
    # (no child process, no blocking wait, signals work correctly).
    os.execv(sys.executable, [sys.executable, "-m", "whippersnappy.cli.whippersnap"] + sys.argv[1:])

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

# Global state shared between the GL render loop and the Qt config panel.
# All access is from the main thread — no locking needed.
current_fthresh_ = None
current_fmax_ = None
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
    app=None,
    config_window=None,
):
    """Start a live interactive OpenGL+Qt window for viewing a triangular mesh.

    On macOS both GLFW/Cocoa and Qt require the main thread.  This function
    creates a GLFW window, then hands control to a ``QTimer``-driven render
    loop so that GLFW polling and Qt event processing share the main thread.

    Parameters
    ----------
    mesh : str or tuple of (array-like, array-like)
        Path to any mesh file or a ``(vertices, faces)`` array tuple.
    overlay : str, array-like, or None, optional
        Per-vertex scalar overlay.
    annot : str, tuple, or None, optional
        FreeSurfer ``.annot`` file or ``(labels, ctab[, names])`` tuple.
    bg_map : str, array-like, or None, optional
        Per-vertex scalar file or array for background shading.
    roi : str, array-like, or None, optional
        FreeSurfer label file or boolean array masking overlay coloring.
    invert : bool, optional
        Invert the overlay color mapping. Default is ``False``.
    specular : bool, optional
        Enable specular highlights. Default is ``True``.
    view : ViewType, optional
        Initial camera view direction. Default is ``ViewType.LEFT``.
    app : QApplication
        The already-created ``QApplication`` instance.
    config_window : ConfigWindow
        The already-created Qt configuration panel.

    Raises
    ------
    RuntimeError
        If the GLFW window or OpenGL context could not be created.
    """
    global current_fthresh_, current_fmax_, app_window_closed_

    from PyQt6.QtCore import QTimer  # noqa: PLC0415

    wwidth  = 720
    wheight = 600
    window  = init_window(wwidth, wheight, "WhipperSnapPy", visible=True)
    if not window:
        raise RuntimeError(
            "Could not create a GLFW window/context. OpenGL context unavailable."
        )

    view_mats = get_view_matrices()
    viewmat   = view_mats[view]
    rot_y     = pyrr.Matrix44.from_y_rotation(0)
    ypos      = 0.0

    meshdata, triangles, _fthresh, _fmax, _pos, _neg = prepare_geometry(
        mesh, overlay, annot, bg_map, roi,
        current_fthresh_, current_fmax_,
        invert=invert,
    )
    shader = setup_shader(meshdata, triangles, wwidth, wheight, specular=specular)

    logger.info("Keys:  Left/Right arrows → rotate   ESC → quit")

    def _render_frame():
        """Called by QTimer every frame; does one GLFW poll + one GL draw."""
        global current_fthresh_, current_fmax_, app_window_closed_
        nonlocal meshdata, triangles, shader, rot_y, ypos

        # Check GLFW close conditions
        if (
            glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS
            or glfw.window_should_close(window)
            or app_window_closed_
        ):
            timer.stop()
            glfw.terminate()
            app.quit()
            return

        glfw.poll_events()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Re-render if Qt sliders changed the thresholds
        if config_window is not None:
            new_fthresh = config_window.get_fthresh_value()
            new_fmax    = config_window.get_fmax_value()
            if new_fthresh != current_fthresh_ or new_fmax != current_fmax_:
                current_fthresh_ = new_fthresh
                current_fmax_    = new_fmax
                meshdata, triangles, _fthresh, _fmax, _pos, _neg = prepare_geometry(
                    mesh, overlay, annot, bg_map, roi,
                    current_fthresh_, current_fmax_,
                    invert=invert,
                )
                shader = setup_shader(
                    meshdata, triangles, wwidth, wheight, specular=specular
                )

        # Keyboard rotation
        if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
            ypos += 0.004
        if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
            ypos -= 0.004
        rot_y = pyrr.Matrix44.from_y_rotation(ypos)

        transform_loc = gl.glGetUniformLocation(shader, "transform")
        gl.glUniformMatrix4fv(transform_loc, 1, gl.GL_FALSE, rot_y * viewmat)
        gl.glDrawElements(gl.GL_TRIANGLES, triangles.size, gl.GL_UNSIGNED_INT, None)
        glfw.swap_buffers(window)

    # ~60 fps timer — fires every 16 ms on the main thread
    timer = QTimer()
    timer.timeout.connect(_render_frame)
    timer.start(16)

    app.exec()


def config_app_exit_handler():
    """Mark the configuration window as closed.

    Connected to ``QApplication.aboutToQuit`` so the render timer stops
    cleanly when the user closes the Qt panel.
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
    global current_fthresh_, current_fmax_
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

    # Both QApplication/Qt and GLFW/Cocoa require the main thread on macOS.
    # Create Qt objects here on the main thread, then pass them into
    # show_window which drives rendering via a QTimer (no extra threads).
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.aboutToQuit.connect(config_app_exit_handler)

    screen_geometry = app.primaryScreen().availableGeometry()
    config_window = ConfigWindow(
        screen_dims=(screen_geometry.width(), screen_geometry.height()),
        initial_fthresh_value=current_fthresh_,
        initial_fmax_value=current_fmax_,
    )
    config_window.show()

    # show_window creates the GLFW window, sets up a QTimer render loop,
    # then calls app.exec() — returns when either window is closed.
    show_window(
        mesh=mesh_path,
        overlay=overlay,
        annot=args.annot,
        bg_map=bg_map,
        roi=roi,
        invert=args.invert,
        specular=args.specular,
        view=view,
        app=app,
        config_window=config_window,
    )


if __name__ == "__main__":
    run()


