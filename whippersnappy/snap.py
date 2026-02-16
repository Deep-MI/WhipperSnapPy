"""Snapshot (static rendering) API for WhipperSnapPy."""

import logging
import os

import glfw
import numpy as np
import OpenGL.GL as gl
import pyrr
from PIL import Image, ImageDraw, ImageFont

from whippersnappy.geometry import get_surf_name, prepare_geometry
from whippersnappy.utils.image import create_colorbar, load_roboto_font, text_size
from whippersnappy.utils.types import ColorSelection, OrientationType, ViewType

from . import gl as _gl
from .gl import get_view_matrices

# Module logger
logger = logging.getLogger(__name__)


def snap1(
    meshpath,
    outpath=None,
    overlaypath=None,
    annotpath=None,
    labelpath=None,
    curvpath=None,
    view=ViewType.LEFT,
    viewmat=None,
    width=None,
    height=None,
    fthresh=None,
    fmax=None,
    caption=None,
    caption_x=None,
    caption_y=None,
    caption_scale=1,
    invert=False,
    colorbar=True,
    colorbar_x=None,
    colorbar_y=None,
    colorbar_scale=1,
    orientation=OrientationType.HORIZONTAL,
    color_mode=ColorSelection.BOTH,
    font_file=None,
    specular=True,
    brain_scale=1.5,
    ambient=0.0,
):
    """Render a single static snapshot of a surface view.

    This function opens an (offscreen) OpenGL context, uploads the provided
    surface geometry and colors (overlay or annotation), renders the scene
    for a single view, captures the framebuffer, and returns a PIL Image
    containing the rendered brain view. When ``outpath`` is provided the
    image is also written to disk.

    Parameters
    ----------
    meshpath : str
        Path to the surface file (FreeSurfer-format, e.g. "lh.white").
    outpath : str or None, optional
        When provided, the resulting image is saved to this path. If ``None``
        the PIL Image object is returned.
    overlaypath : str or None, optional
        Path to overlay/mgh file providing per-vertex values to color the
        surface. If ``None``, coloring falls back to curvature/annotation.
    annotpath : str or None, optional
        Path to a FreeSurfer .annot file with per-vertex labels.
    labelpath : str or None, optional
        Path to a label file (cortex.label) used to mask overlay values.
    curvpath : str or None, optional
        Path to curvature file used to texture non-colored regions.
    view : ViewType, optional
        Which pre-defined view to render (left, right, front, ...). Default is ``ViewType.LEFT``.
    viewmat : 4x4 matrix-like, optional
        Optional view matrix to override the pre-defined view.
    width, height : int or None, optional
        Requested overall canvas width/height in pixels. If ``None`` defaults
        are used (700x500 reference).
    fthresh, fmax : float or None, optional
        Threshold and saturation values for overlay coloring.
    caption, caption_x, caption_y, caption_scale : str/float, optional
        Caption text and layout parameters. Caption defaults to ``None`` and caption_scale defaults to 1.
    invert : bool, optional
        Invert the color scale. Default is ``False``.
    colorbar : bool, optional
        If True, render a colorbar when an overlay is present. Default is ``True``.
    colorbar_x, colorbar_y, colorbar_scale : float, optional
        Colorbar positioning and scale flags. Scale defaults to 1.
    orientation : OrientationType, optional
        Orientation of the colorbar (HORIZONTAL/VERTICAL). Default is ``OrientationType.HORIZONTAL``.
    color_mode : ColorSelection, optional
        Which sign of overlay to color (POSITIVE/NEGATIVE/BOTH). Default is ``ColorSelection.BOTH``.
    font_file : str or None, optional
        Path to a TTF font for captions; fallback to bundled font if None.
    specular : bool, optional
        Enable specular highlights. Default is ``True``.
    brain_scale : float, optional
        Scale factor applied when preparing the geometry. Default is ``1.5``.
    ambient : float, optional
        Ambient lighting strength for shader. Default is ``0.0``.

    Returns
    -------
    PIL.Image.Image or None
        If ``outpath`` is ``None`` the function returns a PIL Image object
        containing the rendered snapshot. If ``outpath`` is provided the
        image is saved to disk and ``None`` is returned.

    Raises
    ------
    RuntimeError
        If the OpenGL/GLFW context cannot be initialized.
    ValueError
        If the overlay contains no values to display for the chosen
        color_mode.
    FileNotFoundError
        If required surface files cannot be found when deriving from
        SUBJECTS_DIR in multi-view helpers.

    Examples
    --------
    >>> from whippersnappy import snap1
    >>> img = snap1('fsaverage/surf/lh.white', overlaypath='fsaverage/surf/lh.thickness')
    >>> img.save('/tmp/lh.png')
    """
    ref_width = 700
    ref_height = 500
    wwidth = ref_width if width is None else width
    wheight = ref_height if height is None else height
    ui_scale = min(wwidth / ref_width, wheight / ref_height)

    if not glfw.init():
        logger.error("Could not init glfw!")
        raise RuntimeError("Could not initialize GLFW; OpenGL context unavailable")
    primary_monitor = glfw.get_primary_monitor()
    mode = glfw.get_video_mode(primary_monitor)
    screen_width = mode.size.width
    screen_height = mode.size.height
    if wwidth > screen_width:
        logger.info("Requested width %d exceeds screen width %d, expect black bars", wwidth, screen_width)
    elif wheight > screen_height:
        logger.info("Requested height %d exceeds screen height %d, expect black bars", wheight, screen_height)

    image = Image.new("RGB", (wwidth, wheight))

    bwidth = int(540 * brain_scale * ui_scale)
    bheight = int(450 * brain_scale * ui_scale)
    brain_display_width = min(bwidth, wwidth)
    brain_display_height = min(bheight, wheight)

    window = _gl.init_window(brain_display_width, brain_display_height, "WhipperSnapPy 2.0", visible=True)
    if not window:
        return False

    transl = pyrr.Matrix44.from_translation((0, 0, 0.4))

    meshdata, triangles, fthresh, fmax, pos, neg = prepare_geometry(
        meshpath,
        overlaypath,
        annotpath,
        curvpath,
        labelpath,
        fthresh,
        fmax,
        invert,
        scale=brain_scale,
        color_mode=color_mode,
    )

    if overlaypath is not None:
        if color_mode == ColorSelection.POSITIVE:
            if not pos and neg:
                logger.error("Overlay has no values to display with positive color_mode")
                raise ValueError("Overlay has no values to display with positive color_mode")
            neg = False
        elif color_mode == ColorSelection.NEGATIVE:
            if pos and not neg:
                logger.error("Overlay has no values to display with negative color_mode")
                raise ValueError("Overlay has no values to display with negative color_mode")
            pos = False
        if not pos and not neg:
            logger.error("Overlay has no values to display")
            raise ValueError("Overlay has no values to display")

    shader = _gl.setup_shader(meshdata, triangles, brain_display_width, brain_display_height,
                          specular=specular, ambient=ambient)

    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    transform_loc = gl.glGetUniformLocation(shader, "transform")
    view_mats = get_view_matrices()
    viewmat = transl * (view_mats[view] if viewmat is None else viewmat)
    gl.glUniformMatrix4fv(transform_loc, 1, gl.GL_FALSE, viewmat)
    gl.glDrawElements(gl.GL_TRIANGLES, triangles.size, gl.GL_UNSIGNED_INT, None)

    im1 = _gl.capture_window(brain_display_width, brain_display_height)

    brain_x = 0 if wwidth < bwidth else (wwidth - bwidth) // 2
    brain_y = 0 if wheight < bheight else (wheight - bheight) // 2
    image.paste(im1, (brain_x, brain_y))

    bar = None
    bar_w = bar_h = 0
    if overlaypath is not None and colorbar:
        bar = create_colorbar(fthresh, fmax, invert, orientation, colorbar_scale * ui_scale, pos, neg,
                              font_file=font_file)
        bar_w, bar_h = bar.size

    font = None
    text_w = text_h = 0
    if caption:
        if font_file is None:
            font = load_roboto_font(int(20 * caption_scale * ui_scale))
        else:
            try:
                font = ImageFont.truetype(font_file, int(20 * caption_scale * ui_scale))
            except Exception:
                font = load_roboto_font(int(20 * caption_scale * ui_scale))
        text_w, text_h = text_size(caption, font)
        text_w = int(text_w)
        text_h = int(text_h)

    bottom_pad = int(20 * ui_scale)
    right_pad = int(20 * ui_scale)
    gap = int(4 * ui_scale)

    if orientation == OrientationType.HORIZONTAL:
        if bar is not None:
            bx = int(0.5 * (image.width - bar_w)) if colorbar_x is None else int(colorbar_x * wwidth)
            if colorbar_y is None:
                gap_and_caption = (gap + text_h) if caption and caption_y is None else 0
                by = image.height - bottom_pad - gap_and_caption - bar_h
            else:
                by = int(colorbar_y * wheight)
            image.paste(bar, (bx, by))

        if caption:
            cx = int(0.5 * (image.width - text_w)) if caption_x is None else int(caption_x * wwidth)
            cy = image.height - bottom_pad - text_h if caption_y is None else int(caption_y * wheight)
            ImageDraw.Draw(image).text((cx, cy), caption, (220, 220, 220), font=font, anchor="lt")
    else:
        if bar is not None:
            if colorbar_x is None:
                gap_and_caption = (gap + text_h) if caption and caption_x is None else 0
                bx = image.width - right_pad - gap_and_caption - bar_w
            else:
                bx = int(colorbar_x * wwidth)
            by = int(0.5 * (image.height - bar_h)) if colorbar_y is None else int(colorbar_y * wheight)
            image.paste(bar, (bx, by))

        if caption:
            temp_caption_img = Image.new("RGBA", (text_w, text_h), (0, 0, 0, 0))
            ImageDraw.Draw(temp_caption_img).text((0, 0), caption, font=font, anchor="lt")
            rotated_caption = temp_caption_img.rotate(90, expand=True, fillcolor=(0, 0, 0, 0))
            rotated_w, rotated_h = rotated_caption.size

            cx = image.width - right_pad - rotated_w if caption_x is None else int(caption_x * wwidth)
            cy = int(0.5 * (image.height - rotated_h)) if caption_y is None else int(caption_y * wheight)
            image.paste(rotated_caption, (cx, cy), rotated_caption)

    if outpath is None:
        glfw.terminate()
        return image

    logger.info("Saving snapshot to %s", outpath)
    image.save(outpath)
    glfw.terminate()
    return None


def snap4(
    lhoverlaypath=None,
    rhoverlaypath=None,
    lhannotpath=None,
    rhannotpath=None,
    fthresh=None,
    fmax=None,
    sdir=None,
    caption=None,
    invert=False,
    labelname="cortex.label",
    surfname=None,
    curvname="curv",
    colorbar=True,
    outpath=None,
    font_file=None,
    specular=True,
    ambient=0.0,
    brain_scale=1.85,
):
    """Render four snapshot views (left/right hemispheres, front/back).

    This convenience function renders four views (top/bottom for each
    hemisphere), stitches them together into a single PIL Image and returns
    it (or saves it to ``outpath`` when provided). It is typically used to
    produce publication-ready overview figures composed from both
    hemispheres.

    Parameters
    ----------
    lhoverlaypath, rhoverlaypath : str or None
        Paths to left/right hemisphere overlay files (mutually required if
        either is provided).
    lhannotpath, rhannotpath : str or None
        Paths to left/right hemisphere annotation (.annot) files.
    fthresh, fmax : float or None
        Threshold and saturation for overlay coloring.
    sdir : str or None
        Subject directory (used when surfname is not provided). If not
        supplied the environment variable ``SUBJECTS_DIR`` is consulted.
    caption : str or None
        Caption string to place on the final image.
    invert : bool, optional
        Invert color scale. Default is ``False``.
    labelname : str, optional
        Name of the label file (default 'cortex.label').
    surfname : str or None, optional
        Surface basename to load (if None the function will auto-discover a
        suitable surface).
    curvname : str or None, optional
        Curvature file basename to load for texturing non-colored regions. Default is ``curv``.
    colorbar : bool, optional
        Whether to draw a colorbar on the composed image. Default is ``True``.
    outpath : str or None, optional
        If provided, save composed image to this path and return ``None``.
    font_file : str or None, optional
        Path to a font to use for captions.
    specular : bool, optional
        Enable/disable specular highlights in the renderer. Default is ``True``.
    ambient : float, optional
        Ambient lighting strength. Default is ``0``.
    brain_scale : float, optional
        Scaling factor passed to geometry preparation. Default is ``1.85``.

    Returns
    -------
    PIL.Image.Image or None
        Composed image of the four views, or ``None`` if ``outpath`` was
        provided and the image was written to disk.

    Raises
    ------
    ValueError
        For invalid argument combinations or when required overlay values
        are absent.
    FileNotFoundError
        When required surface files are not found.

    Examples
    --------
    >>> from whippersnappy import snap4
    >>> img = snap4(
    >>>          lhoverlaypath='fsaverage/surf/lh.thickness',
    >>>          rhoverlaypath='fsaverage/surf/rh.thickness',
    >>>         sdir='./fsaverage'
    >>>       )
    >>> img.save('/tmp/whippersnappy_overview.png')
    """
    wwidth = 540
    wheight = 450
    # Try to create a visible window first (better for debugging),
    # but fall back to an invisible/offscreen window if that fails.
    window = _gl.init_window(wwidth, wheight, "WhipperSnapPy", visible=True)
    if not window:
        logger.warning("Could not create visible GLFW window; retrying with invisible window (offscreen).")
        window = _gl.init_window(wwidth, wheight, "WhipperSnapPy", visible=False)
        if not window:
            logger.error("Could not create any GLFW window/context. OpenGL context unavailable.")
            return None

    rot_z = pyrr.Matrix44.from_z_rotation(-0.5 * np.pi)
    rot_x = pyrr.Matrix44.from_x_rotation(0.5 * np.pi)
    view_left = rot_x * rot_z
    rot_y = pyrr.Matrix44.from_y_rotation(np.pi)
    view_right = rot_y * view_left
    transl = pyrr.Matrix44.from_translation((0, 0, 0.4))

    # Predefine hemisphere images so static analysis knows they exist even if
    # an earlier step raises an exception (we still will fail at runtime).
    lhimg = None
    rhimg = None

    for hemi in ("lh", "rh"):
        if surfname is None:
            if sdir is None:
                sdir = os.environ.get("SUBJECTS_DIR")
                if not sdir:
                    logger.error("No surf_name or subjects directory (sdir) provided")
                    raise ValueError("No surf_name or SUBJECTS_DIR provided")
            found_surfname = get_surf_name(sdir, hemi)
            if found_surfname is None:
                logger.error("Could not find valid surface in %s for hemi: %s!", sdir, hemi)
                raise FileNotFoundError(f"Could not find valid surface in {sdir} for hemi: {hemi}")
            meshpath = os.path.join(sdir, "surf", hemi + "." + found_surfname)
        else:
            meshpath = os.path.join(sdir, "surf", hemi + "." + surfname)

        # Assign derived paths
        curvpath = os.path.join(sdir, "surf", hemi + "." + curvname) if curvname else None
        labelpath = os.path.join(sdir, "label", hemi + "." + labelname) if labelname else None
        overlaypath = lhoverlaypath if hemi == "lh" else rhoverlaypath
        annotpath = lhannotpath if hemi == "lh" else rhannotpath

        # Diagnostic: report mesh and overlay paths and whether they exist
        logger.debug("hemisphere=%s", hemi)
        logger.debug("meshpath=%s exists=%s", meshpath, os.path.exists(meshpath))
        if overlaypath is not None:
            logger.debug("overlaypath=%s exists=%s", overlaypath, os.path.exists(overlaypath))
        if annotpath is not None:
            logger.debug("annotpath=%s exists=%s", annotpath, os.path.exists(annotpath))
        if curvpath is not None:
            logger.debug("curvpath=%s exists=%s", curvpath, os.path.exists(curvpath))

        try:
            meshdata, triangles, fthresh, fmax, pos, neg = prepare_geometry(
                meshpath, overlaypath, annotpath, curvpath, labelpath, fthresh, fmax, invert, scale=brain_scale
            )
        except Exception as e:
            logger.error("prepare_geometry failed for %s: %s", meshpath, e)
            glfw.terminate()
            return None

        # Diagnostics about mesh data
        try:
            logger.debug("meshdata shape: %s; triangles count: %s", getattr(meshdata, 'shape', None),
                         getattr(triangles, 'size', None))
        except Exception:
            pass

        if pos == 0 and neg == 0:
            logger.error("Overlay has no values to display")
            raise ValueError("Overlay has no values to display")

        try:
            shader = _gl.setup_shader(meshdata, triangles, wwidth, wheight, specular=specular, ambient=ambient)
            logger.debug("Shader setup complete")
        except Exception as e:
            logger.error("setup_shader failed: %s", e)
            glfw.terminate()
            return None

        try:
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        except Exception as e:
            logger.error("glClear failed: %s", e)
            logger.error("glError: %s", gl.glGetError())
            glfw.terminate()
            return None
        transform_loc = gl.glGetUniformLocation(shader, "transform")
        viewmat = view_left if hemi == "lh" else view_right
        gl.glUniformMatrix4fv(transform_loc, 1, gl.GL_FALSE, transl * viewmat)
        gl.glDrawElements(gl.GL_TRIANGLES, triangles.size, gl.GL_UNSIGNED_INT, None)
        try:
            im1 = _gl.capture_window(wwidth, wheight)
            logger.debug("Captured image 1 size: %s", im1.size)
        except Exception as e:
            logger.error("capture_window failed: %s", e)
            glfw.terminate()
            return None

        glfw.swap_buffers(window)
        try:
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        except Exception as e:
            logger.error("glClear failed: %s", e)
            logger.error("glError: %s", gl.glGetError())
            glfw.terminate()
            return None
        viewmat = view_right if hemi == "lh" else view_left
        gl.glUniformMatrix4fv(transform_loc, 1, gl.GL_FALSE, transl * viewmat)
        gl.glDrawElements(gl.GL_TRIANGLES, triangles.size, gl.GL_UNSIGNED_INT, None)
        try:
            im2 = _gl.capture_window(wwidth, wheight)
            logger.debug("Captured image 2 size: %s", im2.size)
        except Exception as e:
            logger.error("capture_window failed: %s", e)
            glfw.terminate()
            return None

        if hemi == "lh":
            lhimg = Image.new("RGB", (im1.width, im1.height + im2.height))
            lhimg.paste(im1, (0, 0))
            lhimg.paste(im2, (0, im1.height))
        else:
            rhimg = Image.new("RGB", (im1.width, im1.height + im2.height))
            # Keep same top/bottom ordering as left hemisphere: top=im1, bottom=im2
            rhimg.paste(im1, (0, 0))
            rhimg.paste(im2, (0, im1.height))

    # Add small padding around each hemisphere to avoid cropping at edges
    pad = max(4, int(0.03 * wwidth))
    padded_lh = Image.new("RGB", (lhimg.width + 2 * pad, lhimg.height + 2 * pad), (0, 0, 0))
    padded_lh.paste(lhimg, (pad, pad))
    padded_rh = Image.new("RGB", (rhimg.width + 2 * pad, rhimg.height + 2 * pad), (0, 0, 0))
    padded_rh.paste(rhimg, (pad, pad))

    image = Image.new("RGB", (padded_lh.width + padded_rh.width, padded_lh.height))
    image.paste(padded_lh, (0, 0))
    image.paste(padded_rh, (padded_lh.width, 0))

    if caption:
        if font_file is None:
            font = load_roboto_font(20)
        else:
            try:
                font = ImageFont.truetype(font_file, 20)
            except Exception:
                font = load_roboto_font(20)
        if font is not None:
            xpos = 0.5 * (image.width - getattr(font, 'getlength', lambda s: 0)(caption))
            ImageDraw.Draw(image).text((xpos, image.height - 40), caption, (220, 220, 220), font=font)
        else:
            ImageDraw.Draw(image).text((10, image.height - 40), caption, (220, 220, 220))

    if lhannotpath is None and rhannotpath is None and colorbar:
        bar = create_colorbar(fthresh, fmax, invert, pos=pos, neg=neg)
        if bar is not None:
            xpos = int(0.5 * (image.width - bar.width))
            ypos = int(0.5 * (image.height - bar.height))
            image.paste(bar, (xpos, ypos))

    # If outpath is None, return the PIL Image object directly (no disk I/O)
    if outpath is None:
        glfw.terminate()
        return image

    # Otherwise save to disk
    if outpath:
        logger.info("Saving snapshot to %s", outpath)
        image.save(outpath)

    glfw.terminate()
    return None

