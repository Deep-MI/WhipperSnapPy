#!/usr/bin/python3

"""
Executes the whippersnappy program in an interactive or non-interactive mode.

The non-interactive mode (the default) creates an image that contains four
views of the surface, an optional color bar, and a configurable caption.

The interactive mode (--interactive) opens a simple GUI with a controllable
view of one of the hemispheres. In addition, the view through a separate
configuration app which allows adjusting thresholds, etc. during runtime.

Usage:
    $ python3 whippersnap.py -lh $LH_OVERLAY_FILE -rh $RH_OVERLAY_FILE \
                             -sd $SURF_SUBJECT_DIR -o $OUTPUT_PATH

(See help for full list of arguments.)
"""

import argparse
import logging
import os
import signal
import sys
import tempfile
import threading

import glfw
import OpenGL.GL as gl
import pyrr

try:
    from PyQt6.QtWidgets import QApplication
except Exception:
    # GUI dependency missing; handle at runtime when interactive mode is requested
    QApplication = None

from .. import snap4
from .._version import __version__
from ..geometry import get_surf_name, prepare_geometry
from ..gl import get_view_matrices, init_window, setup_shader
from ..gui import ConfigWindow
from ..utils.types import ViewType

# Module logger
logger = logging.getLogger(__name__)

# Global variables for config app configuration state:
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
        Path to a per-vertex overlay file (e.g. thickness). If ``None`` no
        overlay will be applied.
    annotpath : str or None, optional
        Path to a .annot file providing categorical labels for vertices.
    sdir : str or None, optional
        Subject directory containing `surf/` and `label/` subdirectories.
    caption : str or None, optional
        Caption text to display in the viewer window.
    invert : bool, optional, default False
        Invert the overlay color mapping.
    labelname : str, optional, default 'cortex.label'
        Label filename used to mask vertices.
    surfname : str or None, optional
        Surface basename (e.g. 'white'); if ``None`` the function will try
        to auto-detect a suitable surface in ``sdir``.
    curvname : str or None, optional, default 'curv'
        Curvature filename used to texture non-colored regions.
    specular : bool, optional, default True
        Enable specular highlights in the shader.

    Returns
    -------
    None
        The function primarily drives an interactive event loop and does not return programmatic geometry objects.

    Raises
    ------
    RuntimeError
        If the window/context could not be created.
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

    curvpath = None
    if curvname:
        curvpath = os.path.join(sdir, "surf", hemi + "." + curvname)
    labelpath = None
    if labelname:
        labelpath = os.path.join(sdir, "label", hemi + "." + labelname)

    # set up canonical view matrix for the selected hemisphere
    view_mats = get_view_matrices()
    viewmat = view_mats[ViewType.LEFT]  # fallback
    if hemi == "rh":
        viewmat = view_mats[ViewType.RIGHT]
    rot_y = pyrr.Matrix44.from_y_rotation(0)

    meshdata, triangles, fthresh, fmax, neg = prepare_geometry(
        meshpath, overlaypath, annotpath, curvpath, labelpath, current_fthresh_, current_fmax_
    )
    shader = setup_shader(meshdata, triangles, wwidth, wheight, specular=specular)

    logger.info("\nKeys:\nLeft - Right : Rotate Geometry\nESC          : Quit\n")

    ypos = 0
    while glfw.get_key(window, glfw.KEY_ESCAPE) != glfw.PRESS and not glfw.window_should_close(window):
        # Terminate if config app window was closed:
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
                    meshpath,
                    overlaypath,
                    annotpath,
                    curvpath,
                    labelpath,
                    current_fthresh_,
                    current_fmax_,
                )
                shader = setup_shader(meshdata, triangles, wwidth, wheight, specular=specular)

        transformLoc = gl.glGetUniformLocation(shader, "transform")
        gl.glUniformMatrix4fv(transformLoc, 1, gl.GL_FALSE, rot_y * viewmat)

        if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
            ypos = ypos + 0.0004
        if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
            ypos = ypos - 0.0004
        rot_y = pyrr.Matrix44.from_y_rotation(ypos)

        # Draw
        gl.glDrawElements(gl.GL_TRIANGLES, triangles.size, gl.GL_UNSIGNED_INT, None)
        glfw.swap_buffers(window)

    glfw.terminate()
    # Do NOT call app_.quit() here; QApplication teardown must be handled in the main thread.
    # Only set app_window_closed_ = True in this thread.
    app_window_closed_ = True


def config_app_exit_handler():
    """Mark the configuration application as closed.

    This handler is connected to the configuration app's about-to-quit
    signal and sets a module-level flag that the main OpenGL loop polls to
    terminate cleanly.
    """
    global app_window_closed_
    app_window_closed_ = True


def run():
    """Command-line entry point for the WhipperSnapPy snapshot/interactive tool.

    Parses command-line arguments, validates argument combinations, and
    either launches a non-interactive snapshot generation (``snap4``) or
    starts the interactive viewer and configuration GUI.

    Raises
    ------
    ValueError
        For invalid or mutually exclusive argument combinations.
    ImportError
        If interactive mode is requested but PyQt6 is not available.

    Notes
    -----
    The function validates that either overlay or annotation inputs are
    provided for both hemispheres; it raises ``ValueError`` for invalid
    combinations.

    In non-interactive mode the function calls :func:`whippersnappy.snap4`
    to produce and optionally save a composed image.

    In interactive mode it spawns the OpenGL viewer thread and launches
    the PyQt6-based configuration window in the main thread.
    """
    global current_fthresh_, current_fmax_, app_, app_window_
    # Configure basic logging for CLI invocation so messages from module loggers
    # are visible to end users. Avoid configuring on import by doing this here.
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "-lh",
        "--lh_overlay",
        type=str,
        default=None,
        required=False,
        help="Absolute path to the lh overlay file.",
    )
    parser.add_argument(
        "-rh",
        "--rh_overlay",
        type=str,
        default=None,
        required=False,
        help="Absolute path to the rh overlay file.",
    )
    parser.add_argument(
        "--lh_annot",
        type=str,
        default=None,
        required=False,
        help="Absolute path to the lh annotation file.",
    )
    parser.add_argument(
        "--rh_annot",
        type=str,
        default=None,
        required=False,
        help="Absolute path to the rh annotation file.",
    )
    parser.add_argument(
        "-sd",
        "--sdir",
        type=str,
        required=True,
        help="Absolute path to subject directory from which surfaces will be loaded. "
        "This is assumed to contain the surface files in a surf/ sub-directory.",
    )
    parser.add_argument(
        "-s",
        "--surf_name",
        type=str,
        default=None,
        help="Name of the surface file to load.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default=os.path.join(tempfile.gettempdir(), "whippersnappy_snap.png"),
        help="Absolute path to the output file (snapshot image), "
        "if not running interactive mode.",
    )
    parser.add_argument(
        "-c", "--caption", type=str, default="", help="Caption to place on the figure"
    )
    parser.add_argument(
        "--no-colorbar",
        dest="no_colorbar",
        action="store_true",
        default=False,
        help="Switch off colorbar.")
    parser.add_argument("--fmax",
        type=float,
        default=4.0,
        help="Overlay saturation value (default: 4.0)")
    parser.add_argument("--fthresh",
        type=float,
        default=2.0,
        help="Overlay threshold value (default: 2.0)")
    parser.add_argument(
        "-i",
        "--interactive",
        dest="interactive",
        action="store_true",
        help="Start an interactive GUI session.",
    )
    parser.add_argument(
        "--invert", dest="invert", action="store_true", help="Invert the color scale."
    )
    parser.add_argument(
        "--diffuse",
        dest="specular",
        action="store_false",
        default=True,
        help="Diffuse surface reflection (switch-off specular).",
    )

    args = parser.parse_args()

    try:
        # check for mutually exclusive arguments
        if (args.lh_overlay or args.rh_overlay) and (args.lh_annot or args.rh_annot):
            msg = "Cannot use lh_overlay/rh_overlay and lh_annot/rh_annot arguments at the same time."
            logger.error(msg)
            raise ValueError(msg)
        # check if at least one variant is present
        if args.lh_overlay is None and args.rh_overlay is None and args.lh_annot is None and args.rh_annot is None:
            msg = "Either lh_overlay/rh_overlay or lh_annot/rh_annot must be present."
            logger.error(msg)
            raise ValueError(msg)
        # check if both hemis are present
        if (args.lh_overlay is None and args.rh_overlay is not None) or \
             (args.lh_overlay is not None and args.rh_overlay is None) or \
             (args.lh_annot is None and args.rh_annot is not None) or \
             (args.lh_annot is not None and args.rh_annot is None):
            msg = "If lh_overlay or lh_annot is present, rh_overlay or rh_annot must also be present (and vice versa)."
            logger.error(msg)
            raise ValueError(msg)
    except ValueError as e:
        parser.error(str(e))

    logger.debug("Parsed args: %s", vars(args))

    #
    if not args.interactive:
        snap4(
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
            colorbar=not(args.no_colorbar),
            outpath=args.output_path,
            specular=args.specular,
        )
    else:
        current_fthresh_ = args.fthresh
        current_fmax_ = args.fmax

        # Ensure GUI toolkit is available
        if QApplication is None:
            print("ERROR: Interactive mode requires PyQt6. Install it (pip install PyQt6)"
                  " or run without --interactive.", file=sys.stderr)
            sys.exit(1)

        # Starting interactive OpenGL window in a separate thread:
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

        # Setting up and running config app window (must be main thread):
        app_ = QApplication([])
        app_.setStyle("Fusion")  # the default
        app_.aboutToQuit.connect(config_app_exit_handler)

        screen_geometry = app_.primaryScreen().availableGeometry()
        app_window_ = ConfigWindow(
            screen_dims=(screen_geometry.width(), screen_geometry.height()),
            initial_fthresh_value=current_fthresh_,
            initial_fmax_value=current_fmax_,
        )

        # The following is a way to allow CTRL+C termination of the app:
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        app_window_.show()
        app_.exec()


# headless docker test using xvfb:
# Note, xvfb is a display server implementing the X11 protocol, and performing
# all graphics on memory.
# glfw needs a windows to render even if that is invisible, so above code
# will not work via ssh or on a headless server. xvfb can solve this by wrapping:
# docker run --name headless_test -ti -v$(pwd):/test ubuntu /bin/bash
# apt update && apt install -y python3 python3-pip xvfb
# pip3 install pyopengl glfw pillow numpy pyrr
# xvfb-run python3 test4.py

# instead of the above one could really do headless off-screen rendering via
# EGL (preferred) or OSMesa. The latter looks doable. EGL looks tricky.
# EGL is part of any modern NVIDIA driver
# OSMesa needs to be installed, but should work almost everywhere

# using EGL maybe like this:
# https://github.com/eduble/gl
# or via these bindings:
# https://github.com/perey/pegl

# or OSMesa
# https://github.com/AntonOvsyannikov/DockerGL
