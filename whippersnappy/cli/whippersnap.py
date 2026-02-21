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
from ..gl import (
    ViewState,
    arcball_rotation_matrix,
    arcball_vector,
    compute_view_matrix,
    get_view_matrices,
    init_window,
    setup_shader,
)
from ..utils.types import ViewType

# Module logger
logger = logging.getLogger(__name__)

# Global thresholds shared between the GL render loop and the Qt config panel.
# All access is from the main thread — no locking needed.
current_fthresh_ = None
current_fmax_ = None


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
    creates a GLFW window, registers GLFW input callbacks, then hands control
    to a ``QTimer``-driven render loop so GLFW polling and Qt event processing
    share the main thread.

    Interaction
    -----------
    * **Left-drag** — arcball rotation in world space (no gimbal lock).
    * **Right-drag / Middle-drag** — pan in screen space.
    * **Scroll wheel** — zoom (Z-translation).
    * **Arrow keys** — rotate in 2° increments.
    * **R key / double-click** — reset view to initial preset.
    * **Q key / ESC** — quit.

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
    global current_fthresh_, current_fmax_

    import numpy as np  # noqa: PLC0415
    from PyQt6.QtCore import QTimer  # noqa: PLC0415

    wwidth  = 720
    wheight = 600
    window  = init_window(wwidth, wheight, "WhipperSnapPy", visible=True)
    if not window:
        raise RuntimeError(
            "Could not create a GLFW window/context. OpenGL context unavailable."
        )

    # ------------------------------------------------------------------
    # Initialise view state and base view matrix
    # ------------------------------------------------------------------
    view_mats = get_view_matrices()
    base_view = view_mats[view]        # fixed orientation preset (→ transform uniform)
    vs        = ViewState(zoom=0.0)    # zoom/pan packed into transform

    _last_left_press_time = [0.0]

    def _reset_view():
        vs.rotation = np.eye(4, dtype=np.float32)
        vs.pan      = np.zeros(2, dtype=np.float32)
        vs.zoom     = 0.0

    # ------------------------------------------------------------------
    # Load mesh and compile shader
    # ------------------------------------------------------------------
    meshdata, triangles, _fthresh, _fmax, _pos, _neg = prepare_geometry(
        mesh, overlay, annot, bg_map, roi,
        current_fthresh_, current_fmax_,
        invert=invert,
    )
    shader = setup_shader(meshdata, triangles, wwidth, wheight, specular=specular)

    logger.info(
        "Mouse: left-drag=rotate  right/middle-drag=pan  scroll=zoom  "
        "R/double-click=reset  Q/ESC=quit"
    )

    # ------------------------------------------------------------------
    # GLFW input callbacks
    # ------------------------------------------------------------------

    def _mouse_button_cb(win, button, action, _mods):
        x, y = glfw.get_cursor_pos(win)
        pos  = np.array([x, y], dtype=np.float64)

        if button == glfw.MOUSE_BUTTON_LEFT:
            vs.left_button_down = (action == glfw.PRESS)
            if action == glfw.PRESS:
                # Double-click detection (threshold: 300 ms)
                now = glfw.get_time()
                if now - _last_left_press_time[0] < 0.3:
                    _reset_view()
                _last_left_press_time[0] = now
                vs.last_mouse_pos = pos
            else:
                vs.last_mouse_pos = None

        elif button == glfw.MOUSE_BUTTON_RIGHT:
            vs.right_button_down = (action == glfw.PRESS)
            vs.last_mouse_pos = pos if action == glfw.PRESS else None

        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            vs.middle_button_down = (action == glfw.PRESS)
            vs.last_mouse_pos = pos if action == glfw.PRESS else None

    def _cursor_pos_cb(win, x, y):
        if vs.last_mouse_pos is None:
            return
        dx = x - vs.last_mouse_pos[0]
        dy = y - vs.last_mouse_pos[1]

        if vs.left_button_down:
            # Arcball rotation — amplify drag for snappier feel
            _sensitivity = 2.5
            mx, my = vs.last_mouse_pos
            v1 = arcball_vector(mx, my, wwidth, wheight)
            v2 = arcball_vector(
                mx + (x - mx) * _sensitivity,
                my + (y - my) * _sensitivity,
                wwidth, wheight,
            )
            delta = arcball_rotation_matrix(v2, v1)
            vs.rotation = vs.rotation @ delta

        elif vs.right_button_down or vs.middle_button_down:
            # Pan in camera space — scale to normalised mesh units
            pan_sensitivity = 1.0 / min(wwidth, wheight)
            vs.pan[0] += dx * pan_sensitivity
            vs.pan[1] -= dy * pan_sensitivity   # y is flipped

        vs.last_mouse_pos = np.array([x, y], dtype=np.float64)

    def _scroll_cb(_win, _x_off, y_off):
        # scroll up (y_off > 0) → move camera closer (positive Z in camera space)
        vs.zoom += y_off * 0.05
        # Allow much further zoom out (e.g. -20.0)
        vs.zoom  = float(np.clip(vs.zoom, -20.0, 4.5))

    _arrow_keys = {glfw.KEY_RIGHT, glfw.KEY_LEFT, glfw.KEY_UP, glfw.KEY_DOWN}

    def _key_cb(win, key, _scancode, action, _mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            # On key release, restore normal render rate
            if key in _arrow_keys:
                timer.setInterval(16)
            return
        delta = np.radians(3.0)
        if key == glfw.KEY_RIGHT:
            rot = np.array(pyrr.Matrix44.from_y_rotation(-delta), dtype=np.float32)
        elif key == glfw.KEY_LEFT:
            rot = np.array(pyrr.Matrix44.from_y_rotation(+delta), dtype=np.float32)
        elif key == glfw.KEY_UP:
            rot = np.array(pyrr.Matrix44.from_x_rotation(+delta), dtype=np.float32)
        elif key == glfw.KEY_DOWN:
            rot = np.array(pyrr.Matrix44.from_x_rotation(-delta), dtype=np.float32)
        elif key == glfw.KEY_R:
            _reset_view()
            return
        elif key == glfw.KEY_Q:
            glfw.set_window_should_close(win, True)
            return
        else:
            return
        # Speed up render loop while key is held for smooth rotation
        if key in _arrow_keys:
            timer.setInterval(8)
        vs.rotation = vs.rotation @ rot

    glfw.set_mouse_button_callback(window, _mouse_button_cb)
    glfw.set_cursor_pos_callback(window,   _cursor_pos_cb)
    glfw.set_scroll_callback(window,       _scroll_cb)
    glfw.set_key_callback(window,          _key_cb)

    from PyQt6.QtCore import QEventLoop  # noqa: PLC0415
    loop = QEventLoop()

    _quitting = [False]  # guard so we only shut down once

    def _begin_quit():
        """Stop rendering and GLFW, then exit the event loop."""
        if _quitting[0]:
            return
        _quitting[0] = True
        timer.stop()
        try:
            glfw.terminate()
        except Exception:
            pass
        # Defer loop.quit() so this callback fully unwinds first,
        # avoiding QThreadStorage destruction mid-stack.
        QTimer.singleShot(0, loop.quit)

    def _render_frame():
        """Called by QTimer every 16 ms on the main thread."""
        global current_fthresh_, current_fmax_
        nonlocal meshdata, triangles, shader

        if _quitting[0]:
            return

        if (
            glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS
            or glfw.window_should_close(window)
        ):
            _begin_quit()
            return

        glfw.poll_events()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Re-prepare geometry if Qt sliders changed thresholds
        if config_window is not None:
            new_fthresh = config_window.get_fthresh_value()
            new_fmax    = config_window.get_fmax_value()
            if new_fthresh != current_fthresh_ or new_fmax != current_fmax_:
                current_fthresh_ = new_fthresh
                current_fmax_    = new_fmax
                meshdata, triangles, _ft, _fm, _p, _n = prepare_geometry(
                    mesh, overlay, annot, bg_map, roi,
                    current_fthresh_, current_fmax_,
                    invert=invert,
                )
                shader = setup_shader(
                    meshdata, triangles, wwidth, wheight, specular=specular
                )

        # Identical to snap_rotate: transl * rotation * base_view → transform uniform.
        # model and view uniforms are left as set by setup_shader (identity / camera).
        gl.glUniformMatrix4fv(
            gl.glGetUniformLocation(shader, "transform"), 1, gl.GL_FALSE,
            compute_view_matrix(vs, base_view),
        )
        gl.glDrawElements(gl.GL_TRIANGLES, triangles.size, gl.GL_UNSIGNED_INT, None)
        glfw.swap_buffers(window)

    timer = QTimer()
    timer.timeout.connect(_render_frame)
    timer.start(16)

    # If the Qt config panel is closed, shut down GLFW and exit the loop.
    if config_window is not None:
        config_window.destroyed.connect(lambda: _begin_quit())

    loop.exec()


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


