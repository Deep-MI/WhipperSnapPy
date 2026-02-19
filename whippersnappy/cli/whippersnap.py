#!/usr/bin/python3

"""Interactive GUI viewer for WhipperSnapPy.

Opens a live OpenGL window for a single hemisphere together with a
Qt-based configuration panel that allows adjusting overlay thresholds
at runtime.

Usage::

    whippersnap -lh <lh_overlay> -sd <subject_dir>
    whippersnap --lh_annot <lh.annot> --rh_annot <rh.annot> -sd <subject_dir>

See ``whippersnap --help`` for the full list of options.
For non-interactive four-view batch rendering use ``whippersnap4``.
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
    hemi,
    overlaypath=None,
    annotpath=None,
    sdir=None,
    caption=None,
    invert=False,
    labelname="cortex.label",
    surfname=None,
    curvname="curv",
    specular=True,
):
    """Start a live interactive OpenGL window for viewing a hemisphere.

    The function initializes a GLFW window and renders the requested
    hemisphere with any provided overlay/annotation. It polls for
    configuration updates from the separate configuration GUI and updates
    the rendered scene accordingly.

    Parameters
    ----------
    hemi : {'lh','rh'}
        Hemisphere to display.
    overlaypath : str or None, optional
        Path to a per-vertex overlay file (e.g. thickness).
    annotpath : str or None, optional
        Path to a ``.annot`` file providing categorical labels for vertices.
    sdir : str or None, optional
        Subject directory containing ``surf/`` and ``label/`` subdirectories.
    caption : str or None, optional
        Caption text to display in the viewer window.
    invert : bool, optional, default False
        Invert the overlay color mapping.
    labelname : str, optional, default 'cortex.label'
        Label filename used to mask vertices.
    surfname : str or None, optional
        Surface basename (e.g. ``'white'``); if ``None`` the function will
        auto-detect a suitable surface in ``sdir``.
    curvname : str or None, optional, default 'curv'
        Curvature filename used to texture non-colored regions.
    specular : bool, optional, default True
        Enable specular highlights in the shader.

    Raises
    ------
    RuntimeError
        If the GLFW window or OpenGL context could not be created.
    FileNotFoundError
        If a requested surface file cannot be located in ``sdir``.
    """
    global current_fthresh_, current_fmax_, app_, app_window_, app_window_closed_

    wwidth = 720
    wheight = 600
    window = init_window(wwidth, wheight, "WhipperSnapPy", visible=True)
    if not window:
        logger.error("Could not create any GLFW window/context. OpenGL context unavailable.")
        raise RuntimeError("Could not create any GLFW window/context. OpenGL context unavailable.")

    if surfname is None:
        logger.info("No surf_name provided. Looking for options in surf directory...")
        found_surfname = get_surf_name(sdir, hemi)
        if found_surfname is None:
            msg = f"Could not find a valid surf file in {sdir} for hemi: {hemi}!"
            logger.error(msg)
            raise FileNotFoundError(msg)
        meshpath = os.path.join(sdir, "surf", hemi + "." + found_surfname)
    else:
        meshpath = os.path.join(sdir, "surf", hemi + "." + surfname)

    curvpath = os.path.join(sdir, "surf", hemi + "." + curvname) if curvname else None
    labelpath = os.path.join(sdir, "label", hemi + "." + labelname) if labelname else None

    view_mats = get_view_matrices()
    viewmat = view_mats[ViewType.RIGHT] if hemi == "rh" else view_mats[ViewType.LEFT]
    rot_y = pyrr.Matrix44.from_y_rotation(0)

    meshdata, triangles, fthresh, fmax, neg = prepare_geometry(
        meshpath, overlaypath, annotpath, curvpath, labelpath, current_fthresh_, current_fmax_
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
                    meshpath, overlaypath, annotpath, curvpath, labelpath,
                    current_fthresh_, current_fmax_,
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

    Raises
    ------
    RuntimeError
        If PyQt6 is not installed.
    ValueError
        For invalid or mutually exclusive argument combinations.
    """
    global current_fthresh_, current_fmax_, app_, app_window_
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        prog="whippersnap",
        description=(
            "Interactive GUI viewer for a single hemisphere. "
            "For batch four-view rendering use whippersnap4."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-lh", "--lh_overlay", type=str, default=None,
                        help="Path to the lh overlay file.")
    parser.add_argument("-rh", "--rh_overlay", type=str, default=None,
                        help="Path to the rh overlay file.")
    parser.add_argument("--lh_annot", type=str, default=None,
                        help="Path to the lh annotation file.")
    parser.add_argument("--rh_annot", type=str, default=None,
                        help="Path to the rh annotation file.")
    parser.add_argument("-sd", "--sdir", type=str, required=True,
                        help="Subject directory containing surf/ and label/ subdirectories.")
    parser.add_argument("-s", "--surf_name", type=str, default=None,
                        help="Surface basename to load (e.g. 'white').")
    parser.add_argument("-c", "--caption", type=str, default="",
                        help="Caption text.")
    parser.add_argument("--fmax", type=float, default=4.0,
                        help="Overlay saturation value (default: 4.0).")
    parser.add_argument("--fthresh", type=float, default=2.0,
                        help="Overlay threshold value (default: 2.0).")
    parser.add_argument("--invert", action="store_true",
                        help="Invert the color scale.")
    parser.add_argument("--diffuse", dest="specular", action="store_false", default=True,
                        help="Diffuse-only shading (no specular).")

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
    current_fmax_ = args.fmax

    thread = threading.Thread(
        target=show_window,
        args=(
            "lh",
            args.lh_overlay,
            args.lh_annot,
            args.sdir,
            None,
            False,
            "cortex.label",
            args.surf_name,
            "curv",
            args.specular,
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

