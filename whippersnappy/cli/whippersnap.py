#!/usr/bin/python3

"""
Executes the whippersnappy program in an interactive or non-interactive mode.

The non-interactive mode (the default) creates an image that contains four
views of the surface, a color bar, and a configurable caption.

The interactive mode (--interactive) opens a simple GUI with a controllable
view of one of the hemispheres. In addition, the view through a separate
configuration app which allows adjusting thresholds, etc. during runtime.

Usage:
    $ python3 whippersnap.py -lh $LH_OVERLAY_FILE -rh $RH_OVERLAY_FILE \
                             -sd $SURF_SUBJECT_DIR -o $OUTPUT_PATH

(See help for full list of arguments.)

@Author1    : Martin Reuter
@Author2    : Ahmed Faisal Abdelrahman
@Created    : 16.03.2022
"""

import argparse
import math
import os
import signal
import sys
import threading

import glfw
import OpenGL.GL as gl
import pyrr
from PyQt6.QtWidgets import QApplication

from whippersnappy.config_app import ConfigWindow
from whippersnappy.core import (
    get_surf_name,
    init_window,
    prepare_geometry,
    setup_shader,
    snap4,
)

# Global variables for config app configuration state:
current_fthresh_ = None
current_fmax_ = None
app_ = None
app_window_ = None
app_window_closed_ = False


def show_window(
    hemi,
    overlaypath,
    sdir=None,
    caption=None,
    invert=False,
    labelname="cortex.label",
    surfname=None,
    curvname="curv",
    specular=True,
):
    """
    Start an interactive window in which an overlay can be viewed.

    Parameters
    ----------
    hemi : str
        Hemisphere; one of: ['lh', 'rh'].
    overlaypath : str
        Path to the overlay file for the specified hemi (FreeSurfer format).
    sdir : str
       Subject dir containing surf files.
    caption : str
       Caption text to be placed on the image.
    invert : bool
       Invert color (blue positive, red negative).
    labelname : str
       Label for masking, usually cortex.label.
    surfname : str
       Surface to display values on, usually pial_semi_inflated from fsaverage.
    curvname : str
       Curvature file for texture in non-colored regions (default curv).
    specular : bool, optional
       If True, enable specular.

    Returns
    -------
    None
        This function does not return any value.
    """
    global current_fthresh_, current_fmax_, app_, app_window_, app_window_closed_

    wwidth = 720
    weight = 600
    window = init_window(wwidth, weight, "WhipperSnapPy", visible=True)
    if not window:
        return False

    if surfname is None:
        print("[INFO] No surf_name provided. Looking for options in surf directory...")
        found_surfname = get_surf_name(sdir, hemi)
        if found_surfname is None:
            print(
                f"[ERROR] Could not find a valid surf file in {sdir} for hemi: {hemi}!"
            )
            sys.exit(0)
        meshpath = os.path.join(sdir, "surf", hemi + "." + found_surfname)
    else:
        meshpath = os.path.join(sdir, "surf", hemi + "." + surfname)

    curvpath = None
    if curvname:
        curvpath = os.path.join(sdir, "surf", hemi + "." + curvname)
    labelpath = None
    if labelname:
        labelpath = os.path.join(sdir, "label", hemi + "." + labelname)

    # set up matrices to show object left and right side:
    rot_z = pyrr.Matrix44.from_z_rotation(-0.5 * math.pi)
    rot_x = pyrr.Matrix44.from_x_rotation(0.5 * math.pi)
    viewLeft = rot_x * rot_z
    # rot_y = pyrr.Matrix44.from_y_rotation(math.pi)
    # viewRight = rot_y * viewLeft
    rot_y = pyrr.Matrix44.from_y_rotation(0)

    meshdata, triangles, fthresh, fmax, neg = prepare_geometry(
        meshpath, overlaypath, curvpath, labelpath, current_fthresh_, current_fmax_
    )
    shader = setup_shader(meshdata, triangles, wwidth, weight, specular=specular)

    print()
    print("Keys:")
    print("Left - Right : Rotate Geometry")
    print("ESC          : Quit")
    print()

    ypos = 0
    while glfw.get_key(
        window, glfw.KEY_ESCAPE
    ) != glfw.PRESS and not glfw.window_should_close(window):
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
                    curvpath,
                    labelpath,
                    current_fthresh_,
                    current_fmax_,
                )
                shader = setup_shader(
                    meshdata, triangles, wwidth, weight, specular=specular
                )

        transformLoc = gl.glGetUniformLocation(shader, "transform")
        gl.glUniformMatrix4fv(transformLoc, 1, gl.GL_FALSE, rot_y * viewLeft)

        if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
            ypos = ypos + 0.0004
        if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
            ypos = ypos - 0.0004
        rot_y = pyrr.Matrix44.from_y_rotation(ypos)

        # Draw
        gl.glDrawElements(gl.GL_TRIANGLES, triangles.size, gl.GL_UNSIGNED_INT, None)

        glfw.swap_buffers(window)

    glfw.terminate()
    app_.quit()


def config_app_exit_handler():
    global app_window_closed_
    app_window_closed_ = True


def run():
    global current_fthresh_, current_fmax_, app_, app_window_

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-lh",
        "--lh_overlay",
        type=str,
        required=True,
        help="Absolute path to the lh overlay file.",
    )
    parser.add_argument(
        "-rh",
        "--rh_overlay",
        type=str,
        required=True,
        help="Absolute path to the rh overlay file.",
    )
    parser.add_argument(
        "-sd",
        "--sdir",
        type=str,
        required=True,
        help="Absolute path to subject directory from which surfaces will be loaded."
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
        default="/tmp/whippersnappy_snap.png",
        help="Absolute path to the output file (snapshot image), "
        "if not running interactive mode.",
    )
    parser.add_argument(
        "-c", "--caption", type=str, default="", help="Caption to place on the figure"
    )
    parser.add_argument("--fmax", type=float, default=4.0)
    parser.add_argument("--fthresh", type=float, default=2.0)
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

    if not args.interactive:
        snap4(
            args.lh_overlay,
            args.rh_overlay,
            sdir=args.sdir,
            caption=args.caption,
            surfname=args.surf_name,
            fthresh=args.fthresh,
            fmax=args.fmax,
            invert=args.invert,
            colorbar=True,
            outpath=args.output_path,
            specular=args.specular,
        )
    else:
        current_fthresh_ = args.fthresh
        current_fmax_ = args.fmax

        # Starting interactive OpenGL window in a separate thread:
        thread = threading.Thread(
            target=show_window,
            args=(
                "lh",
                args.lh_overlay,
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
        app_.signals.aboutToQuit.connect(config_app_exit_handler)

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

# instead of the above one could really do headless off screen rendering via
# EGL (preferred) or OSMesa. The latter looks doable. EGL looks tricky.
# EGL is part of any modern NVIDIA driver
# OSMesa needs to be installed, but should work almost everywhere

# using EGL maybe like this:
# https://github.com/eduble/gl
# or via these bindings:
# https://github.com/perey/pegl

# or OSMesa
# https://github.com/AntonOvsyannikov/DockerGL
