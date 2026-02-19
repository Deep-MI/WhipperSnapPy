"""Snapshot (static rendering) API for WhipperSnapPy."""

import logging
import os

import glfw
import pyrr
from PIL import Image, ImageFont

from .geometry import estimate_overlay_thresholds, get_surf_name
from .geometry.prepare import prepare_and_validate_geometry
from .gl.utils import capture_window, create_window_with_fallback, render_scene, setup_shader, terminate_context
from .gl.views import get_view_matrices
from .utils.image import create_colorbar, draw_caption, draw_colorbar, load_roboto_font, text_size
from .utils.types import ColorSelection, OrientationType, ViewType

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
    width=700,
    height=500,
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

    This function opens an OpenGL context, uploads the provided
    surface geometry and colors (overlay or annotation), renders the scene
    for a single view, captures the framebuffer, and returns a PIL Image
    containing the rendered brain view. When ``outpath`` is provided the
    image is also written to disk.

    Parameters
    ----------
    meshpath : str
        Path to the surface file (FreeSurfer-format, e.g. "lh.white").
    outpath : str or None, optional
        When provided, the resulting image is saved to this path.
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
    width, height : int, optional
        Requested overall canvas width/height in pixels. Defaults to (700x500).
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
    PIL.Image.Image
        Returns a PIL Image object containing the rendered snapshot.

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
    ui_scale = min(width / ref_width, height / ref_height)
    try:
        if glfw.init():
            primary_monitor = glfw.get_primary_monitor()
            if primary_monitor:
                mode = glfw.get_video_mode(primary_monitor)
                if width > mode.size.width:
                    logger.info("Requested width %d exceeds screen width %d, expect black bars",
                                width, mode.size.width)
                elif height > mode.size.height:
                    logger.info("Requested height %d exceeds screen height %d, expect black bars",
                                height, mode.size.height)
    except Exception:
        pass  # headless — no monitor info available, that's fine

    image = Image.new("RGB", (width, height))

    bwidth = int(540 * brain_scale * ui_scale)
    bheight = int(450 * brain_scale * ui_scale)
    brain_display_width = min(bwidth, width)
    brain_display_height = min(bheight, height)
    logger.debug("Requested (width,height) = (%s,%s)", width, height)
    logger.debug("Brain (width,height)     = (%s,%s)", bwidth, bheight)
    logger.debug("B-Display (width,height) = (%s,%s)", brain_display_width, brain_display_height)

    # will raise exception if it cannot be created
    window = create_window_with_fallback(brain_display_width, brain_display_height, "WhipperSnapPy", visible=True)
    try:
        meshdata, triangles, fthresh, fmax, pos, neg = prepare_and_validate_geometry(
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

        shader = setup_shader(meshdata, triangles, brain_display_width, brain_display_height,
                              specular=specular, ambient=ambient)

        transl = pyrr.Matrix44.from_translation((0, 0, 0.4))
        view_mats = get_view_matrices()
        viewmat = transl * (view_mats[view] if viewmat is None else viewmat)
        render_scene(shader, triangles, viewmat)

        # Center the brain rendering in the output image, clamp to zero
        brain_x = max(0, (width - brain_display_width) // 2)
        brain_y = max(0, (height - brain_display_height) // 2)
        image.paste(capture_window(window), (brain_x, brain_y))

        bar = (
            create_colorbar(
                fthresh, fmax, invert, orientation, colorbar_scale * ui_scale, pos, neg, font_file=font_file
            )
            if overlaypath is not None and colorbar
            else None
        )
        font = (
            load_roboto_font(int(20 * caption_scale * ui_scale))
            if font_file is None
            else ImageFont.truetype(font_file, int(20 * caption_scale * ui_scale))
            if caption
            else None
        )

        # Compute positions to avoid overlap, unless explicit positions are given
        text_w, text_h = text_size(caption, font) if caption and font else (0, 0)
        bar_h = bar.height if bar is not None else 0
        gap = int(4 * ui_scale)
        bottom_pad = int(20 * ui_scale)

        if orientation == OrientationType.HORIZONTAL:
            # If explicit positions are given, use them
            if colorbar_x is not None or colorbar_y is not None or caption_x is not None or caption_y is not None:
                bx = int(colorbar_x * width) if colorbar_x is not None else None
                by = int(colorbar_y * height) if colorbar_y is not None else None
                cx = int(caption_x * width) if caption_x is not None else None
                cy = int(caption_y * height) if caption_y is not None else None
                draw_colorbar(image, bar, orientation, x=bx, y=by)
                draw_caption(image, caption, font, orientation, x=cx, y=cy)
            else:
                # Place colorbar above caption if both present
                if bar is not None and caption:
                    bar_y = image.height - bottom_pad - text_h - gap - bar_h
                    caption_y = image.height - bottom_pad - text_h
                elif bar is not None:
                    bar_y = image.height - bottom_pad - bar_h
                    caption_y = None
                elif caption:
                    bar_y = None
                    caption_y = image.height - bottom_pad - text_h
                else:
                    bar_y = caption_y = None
                draw_colorbar(image, bar, orientation, y=bar_y)
                draw_caption(image, caption, font, orientation, y=caption_y)
        else:
            # For vertical, allow explicit x/y for both, else use default
            bx = int(colorbar_x * width) if colorbar_x is not None else None
            by = int(colorbar_y * height) if colorbar_y is not None else None
            cx = int(caption_x * width) if caption_x is not None else None
            cy = int(caption_y * height) if caption_y is not None else None
            draw_colorbar(image, bar, orientation, x=bx, y=by)
            draw_caption(image, caption, font, orientation, x=cx, y=cy)

        if outpath:
            logger.info("Saving snapshot to %s", outpath)
            image.save(outpath)
        return image
    finally:
        terminate_context(window)


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
    color_mode=ColorSelection.BOTH,
):
    """Render four snapshot views (left/right hemispheres, front/back).

    This convenience function renders four views (top/bottom for each
    hemisphere), stitches them together into a single PIL Image and returns
    it (and saves it to ``outpath`` when provided). It is typically used to
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
        If provided, save composed image to this path.
    font_file : str or None, optional
        Path to a font to use for captions.
    specular : bool, optional
        Enable/disable specular highlights in the renderer. Default is ``True``.
    ambient : float, optional
        Ambient lighting strength. Default is ``0``.
    brain_scale : float, optional
        Scaling factor passed to geometry preparation. Default is ``1.85``.
    color_mode : ColorSelection, optional
        Which sign of overlay to color (POSITIVE/NEGATIVE/BOTH). Default is ``ColorSelection.BOTH``.

    Returns
    -------
    PIL.Image.Image
        Composed image of the four views.

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
    >>>          sdir='./fsaverage'
    >>>       )
    >>> img.save('/tmp/whippersnappy_overview.png')
    """
    wwidth = 540
    wheight = 450

    # Resolve sdir early so path-building works for both the pre-pass and
    # the rendering loop.
    if sdir is None:
        sdir = os.environ.get("SUBJECTS_DIR")
        if not sdir and surfname is None:
            logger.error("No sdir or SUBJECTS_DIR provided")
            raise ValueError("No sdir or SUBJECTS_DIR provided")
        if not sdir and surfname is not None:
            logger.error("surfname provided but sdir is None")
            raise ValueError("surfname provided but sdir is None; cannot construct meshpath.")

    # Pre-pass: estimate missing fthresh/fmax from overlays for global color scale
    has_overlay = lhoverlaypath is not None or rhoverlaypath is not None
    if has_overlay and (fthresh is None or fmax is None):
        est_fthreshs = []
        est_fmaxs = []
        for overlaypath in filter(None, (lhoverlaypath, rhoverlaypath)):
            h_fthresh, h_fmax = estimate_overlay_thresholds(overlaypath, fthresh, fmax)
            est_fthreshs.append(h_fthresh)
            est_fmaxs.append(h_fmax)
        if fthresh is None and est_fthreshs:
            fthresh = min(est_fthreshs)
        if fmax is None and est_fmaxs:
            fmax = max(est_fmaxs)
        logger.debug("Global color range: fthresh=%s  fmax=%s", fthresh, fmax)

    # will raise exception if it cannot be created
    window = create_window_with_fallback(wwidth, wheight, "WhipperSnapPy", visible=True)
    try:
        # Use standard view matrices from get_view_matrices and ViewType
        view_mats = get_view_matrices()
        view_left = view_mats[ViewType.LEFT]
        view_right = view_mats[ViewType.RIGHT]
        transl = pyrr.Matrix44.from_translation((0, 0, 0.4))

        # Predefine hemisphere images so static analysis knows they exist even if
        # an earlier step raises an exception (we still will fail at runtime).
        lhimg = None
        rhimg = None

        for hemi in ("lh", "rh"):
            if surfname is None:
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
                meshdata, triangles, fthresh, fmax, pos, neg = prepare_and_validate_geometry(
                    meshpath, overlaypath, annotpath, curvpath, labelpath, fthresh, fmax, invert,
                    scale=brain_scale, color_mode=color_mode
                )
            except Exception as e:
                logger.error("prepare_geometry failed for %s: %s", meshpath, e)
                raise

            # Diagnostics about mesh data
            try:
                logger.debug("meshdata shape: %s; triangles count: %s", getattr(meshdata, 'shape', None),
                             getattr(triangles, 'size', None))
            except Exception:
                pass

            try:
                shader = setup_shader(meshdata, triangles, wwidth, wheight, specular=specular, ambient=ambient)
                logger.debug("Shader setup complete")
            except Exception as e:
                logger.error("setup_shader failed: %s", e)
                raise

            render_scene(shader, triangles, transl * view_left)
            im1 = capture_window(window)
            render_scene(shader, triangles, transl * view_right)
            im2 = capture_window(window)

            if hemi == "lh":
                lhimg = Image.new("RGB", (im1.width, im1.height + im2.height))
                lhimg.paste(im1, (0, 0))
                lhimg.paste(im2, (0, im1.height))
            else:
                rhimg = Image.new("RGB", (im1.width, im1.height + im2.height))
                # For right hemisphere, reverse the order: top=im2, bottom=im1
                rhimg.paste(im2, (0, 0))
                rhimg.paste(im1, (0, im2.height))

        # Add small padding around each hemisphere to avoid cropping at edges
        pad = max(4, int(0.03 * wwidth))
        padded_lh = Image.new("RGB", (lhimg.width + 2 * pad, lhimg.height + 2 * pad), (0, 0, 0))
        padded_lh.paste(lhimg, (pad, pad))
        padded_rh = Image.new("RGB", (rhimg.width + 2 * pad, rhimg.height + 2 * pad), (0, 0, 0))
        padded_rh.paste(rhimg, (pad, pad))

        image = Image.new("RGB", (padded_lh.width + padded_rh.width, padded_lh.height))
        image.paste(padded_lh, (0, 0))
        image.paste(padded_rh, (padded_lh.width, 0))

        font = load_roboto_font(20) if font_file is None else ImageFont.truetype(font_file, 20) if caption else None
        # Place caption at bottom, colorbar above if both present
        text_w, text_h = text_size(caption, font) if caption and font else (0, 0)
        bottom_pad = 20
        gap = 4
        caption_y = image.height - bottom_pad - text_h
        bar = (
            create_colorbar(fthresh, fmax, invert, pos=pos, neg=neg)
            if lhannotpath is None and rhannotpath is None and colorbar
            else None
        )
        bar_h = bar.height if bar is not None else 0
        if bar is not None and caption:
            bar_y = image.height - bottom_pad - text_h - gap - bar_h
            draw_colorbar(image, bar, OrientationType.HORIZONTAL, y=bar_y)
            draw_caption(image, caption, font, OrientationType.HORIZONTAL, y=caption_y)
        elif bar is not None:
            bar_y = image.height - bottom_pad - bar_h
            draw_colorbar(image, bar, OrientationType.HORIZONTAL, y=bar_y)
        elif caption:
            draw_caption(image, caption, font, OrientationType.HORIZONTAL, y=caption_y)

        # If outpath is specified, save to disk
        if outpath:
            logger.info("Saving snapshot to %s", outpath)
            image.save(outpath)

        return image
    finally:
        terminate_context(window)
